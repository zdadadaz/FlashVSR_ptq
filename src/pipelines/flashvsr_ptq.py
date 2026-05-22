"""
FlashVSR PTQ Pipeline - Post-Training Quantized inference pipeline.

Supports:
- W8A16: Symmetric weight quantization (int8 weights, bf16 activations)
- W8A8: Symmetric weight + asymmetric activation quantization (int8 weights + int8 activations)
"""

import types
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from ..models import ModelManager
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from .flashvsr_full import TorchColorCorrectorWavelet


class FlashVSRPTQPipeline(BasePipeline):
    """
    FlashVSR PTQ Pipeline for quantized inference.

    Uses the same architecture as FlashVSRFullPipeline but with quantized DiT weights.
    Supports W8A16 (symmetric weight-only) and W8A8 (symmetric weight + asymmetric activation).
    """

    def __init__(self, device="cuda", torch_dtype=torch.float16, quant_mode="w8a8"):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.TCDecoder = None
        self.model_names = ["dit", "vae"]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False
        self.prompt_emb_posi = None
        self.ColorCorrector = TorchColorCorrectorWavelet(levels=5)
        self.quant_mode = quant_mode

        print(f"""
 ╔══════════════════════════════════════════════════════════════╗
 ║         FlashVSR PTQ Pipeline ({quant_mode.upper()})                  ║
 ║         Symmetric Weight + Asymmetric Activation             ║
 ╚══════════════════════════════════════════════════════════════╝
""")

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        """Enable VRAM management for PTQ pipeline (same as full pipeline)."""
        dtype = next(iter(self.dit.parameters())).dtype
        from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear

        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()

    def fetch_models(self, model_manager: ModelManager):
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, quant_mode="w8a8"):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = FlashVSRPTQPipeline(device=device, torch_dtype=torch_dtype, quant_mode=quant_mode)
        pipe.fetch_models(model_manager)
        pipe.use_unified_sequence_parallel = False
        return pipe

    def denoising_model(self):
        return self.dit

    def init_cross_kv(
        self,
        context_tensor: Optional[torch.Tensor] = None,
        prompt_path=None,
    ):
        self.load_models_to_device(["dit"])

        if self.dit is None:
            raise RuntimeError("Please initialize self.dit via fetch_models / from_model_manager")

        if context_tensor is None:
            if prompt_path is None:
                raise ValueError("init_cross_kv: need prompt_path or context_tensor")
            ctx = torch.load(prompt_path, map_location=self.device)
        else:
            ctx = context_tensor

        ctx = ctx.to(dtype=self.torch_dtype, device=self.device)

        if self.prompt_emb_posi is None:
            self.prompt_emb_posi = {}
        self.prompt_emb_posi["context"] = ctx
        self.prompt_emb_posi["stats"] = "load"

        if hasattr(self.dit, "reinit_cross_kv"):
            self.dit.reinit_cross_kv(ctx)
        else:
            raise AttributeError("WanModel missing reinit_cross_kv(ctx) method")
        self.timestep = torch.tensor([1000.0], device=self.device, dtype=self.torch_dtype)
        self.t = self.dit.time_embedding(sinusoidal_embedding_1d(self.dit.freq_dim, self.timestep))
        self.t_mod = self.dit.time_projection(self.t).unflatten(1, (6, self.dit.dim))
        self.scheduler.set_timesteps(1, denoising_strength=1.0, shift=5.0)
        self.load_models_to_device([])

    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}

    def prepare_extra_input(self, latents=None):
        return {}

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(
            input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        return latents

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(
            latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        return frames

    def offload_model(self, keep_vae=False):
        self.dit.clear_cross_kv()
        self.prompt_emb_posi["stats"] = "offload"
        if hasattr(self.dit, "LQ_proj_in"):
            self.dit.LQ_proj_in.to("cpu")
        if keep_vae:
            self.load_models_to_device(["vae"])
        else:
            self.load_models_to_device([])

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        negative_prompt="",
        denoising_strength=1.0,
        seed=None,
        rand_device="gpu",
        num_frames=None,
        latent_callback=None,
        cfg_scale=1.0,
        **kwargs,
    ):
        raise NotImplementedError("PTQ pipeline __call__ not yet implemented - use full pipeline for inference with TRT engine")