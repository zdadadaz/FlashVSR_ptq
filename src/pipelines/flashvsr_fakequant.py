"""
FlashVSR FakeQuant PTQ Pipeline.

Handles quantized inference using FakeQuantLinear layers (integer weights + optional
integer activation quantization). Runs the full video super-resolution pipeline —
VAE encode → DiT denoise (fakequant) → VAE decode — without TensorRT.

Supports modes:
  - a16w8: no activation quant,      int8 weights
  - a8w8:  int8 activation quant,     int8 weights
  - a16w4: no activation quant,      packed-int4 weights
  - a8w4:  int8 activation quant,     packed-int4 weights
  - a4w4:  int4 activation QDQ,        packed-int4 weights

All computation inside FakeQuantLinear is performed in float32 after dequantizing
integer tensors back to float, then the output dtype is restored to match the input.
"""

import math
import gc
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from ..models import ModelManager
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import WanVideoVAE, Wan22VideoVAE, LightX2VVAE, RMS_norm, CausalConv3d, Upsample
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from .flashvsr_full import TorchColorCorrectorWavelet


class FlashVSRFakeQuantPipeline(BasePipeline):
    """
    Quantized inference pipeline using FakeQuantLinear.

    The DiT weights are stored as true integer tensors (int4/int8), with an
    optional integer activation quantization path (a8). The float matmul is
    performed in float32 for numerical stability.
    """

    def __init__(
        self,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        quant_mode: str = "a8w8",
    ):
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
        self.quant_mode = quant_mode  # e.g. "a8w8"

        print(f"""
 ╔══════════════════════════════════════════════════════════════╗
 ║        FlashVSR FakeQuant PTQ Pipeline ({quant_mode})              ║
 ║        Integer Weights + Optional Integer Activations    ║
 ╚══════════════════════════════════════════════════════════════╝
""")

    # ------------------------------------------------------------------
    # VRAM management (same pattern as full/tiny pipelines)
    # ------------------------------------------------------------------

    def enable_vram_management(self, num_persistent_param_in_dit=None):
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

    # ------------------------------------------------------------------
    # Model fetching
    # ------------------------------------------------------------------

    def fetch_models(self, model_manager: ModelManager):
        self.dit  = model_manager.fetch_model("wan_video_dit")
        self.vae  = model_manager.fetch_model("wan_video_vae")

    @staticmethod
    def from_model_manager(
        model_manager: ModelManager,
        torch_dtype=None,
        device=None,
        quant_mode: str = "a8w8",
    ):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = FlashVSRFakeQuantPipeline(
            device=device, torch_dtype=torch_dtype, quant_mode=quant_mode
        )
        pipe.fetch_models(model_manager)
        return pipe

    # ------------------------------------------------------------------
    # Standard pipeline interface
    # ------------------------------------------------------------------

    def denoising_model(self):
        return self.dit

    def init_cross_kv(
        self,
        context_tensor: Optional[torch.Tensor] = None,
        prompt_path: Optional[str] = None,
    ):
        self.load_models_to_device(["dit"])
        if self.dit is None:
            raise RuntimeError("DiT not loaded; call fetch_models() first")

        if context_tensor is None:
            if prompt_path is None:
                raise ValueError("init_cross_kv requires prompt_path or context_tensor")
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

    def encode_video(
        self,
        input_video,
        tiled: bool = True,
        tile_size: Tuple[int, int] = (34, 34),
        tile_stride: Tuple[int, int] = (18, 16),
    ):
        return self.vae.encode(
            input_video,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    def decode_video(
        self,
        latents,
        tiled: bool = True,
        tile_size: Tuple[int, int] = (34, 34),
        tile_stride: Tuple[int, int] = (18, 16),
    ):
        return self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    def offload_model(self, keep_vae: bool = False):
        self.dit.clear_cross_kv()
        self.prompt_emb_posi["stats"] = "offload"
        if hasattr(self.dit, "LQ_proj_in"):
            self.dit.LQ_proj_in.to("cpu")
        if keep_vae:
            self.load_models_to_device(["vae"])
        else:
            self.load_models_to_device([])

    # ------------------------------------------------------------------
    # Main denoising call
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        negative_prompt: str = "",
        denoising_strength: float = 1.0,
        seed: Optional[int] = None,
        rand_device: str = "gpu",
        num_frames: Optional[int] = None,
        latent_callback=None,
        cfg_scale: float = 1.0,
        tiled: bool = True,
        tile_size: Tuple[int, int] = (34, 34),
        tile_stride: Tuple[int, int] = (18, 16),
        height: int = 480,
        width: int = 832,
        topk_ratio: float = 2.0,
        kv_ratio: float = 3.0,
        local_range: int = 9,
        LQ_video: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Denoise latents through the quantized DiT.

        All FakeQuantLinear layers perform true integer quantization:
          weight:  int4/int8 → float32 via scale
          act  (a8 modes): float → int8 → float32 via per-channel scale
          matmul:  float32 linear (no TensorRT integer math needed)
        """
        from .flashvsr_full import TorchColorCorrectorWavelet, model_fn_wan_video

        assert cfg_scale == 1.0, "cfg_scale must be 1.0"

        if self.prompt_emb_posi is None or 'context' not in self.prompt_emb_posi:
            raise RuntimeError(
                "Cross-Attn KV 未初始化。請先執行 pipe.init_cross_kv() 或傳入 context_tensor"
            )

        if num_frames is not None and num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")

        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Initialize noise latent
        if_buffer = False
        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + (1 if if_buffer else 0), height // 8, width // 8),
            seed=seed, device=self.device, dtype=self.torch_dtype
        )
        latents = noise

        process_total_num = (num_frames - 1) // 8 - 2
        is_stream = True

        if self.prompt_emb_posi['stats'] == "offload":
            self.init_cross_kv(context_tensor=self.prompt_emb_posi['context'])

        # Load DiT and TCDecoder
        self.load_models_to_device(["dit"])
        self.dit.LQ_proj_in.to(self.device)
        if self.TCDecoder is not None:
            self.TCDecoder.to(self.device)

        if hasattr(self.dit, "LQ_proj_in"):
            self.dit.LQ_proj_in.clear_cache()

        LQ_pre_idx = 0
        LQ_cur_idx = 0
        pre_cache_k = None
        pre_cache_v = None
        LQ_latents = None
        final_frames_list = []

        if self.TCDecoder is not None:
            self.TCDecoder.clean_mem()
        else:
            self.vae.clear_cache()

        if not hasattr(self, 'ColorCorrector') or self.ColorCorrector is None:
            self.ColorCorrector = TorchColorCorrectorWavelet(levels=5)

        with torch.no_grad():
            for cur_process_idx in range(process_total_num):
                if cur_process_idx == 0:
                    pre_cache_k = [None] * len(self.dit.blocks)
                    pre_cache_v = [None] * len(self.dit.blocks)
                    LQ_latents = None
                    inner_loop_num = 7
                    for inner_idx in range(inner_loop_num):
                        lq_slice = LQ_video[:, :, max(0, inner_idx * 4 - 3):(inner_idx + 1) * 4 - 3, :, :].to(self.device) if LQ_video is not None else None
                        cur = self.denoising_model().LQ_proj_in.stream_forward(
                            lq_slice
                        ) if lq_slice is not None else None
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat([LQ_latents[layer_idx], cur[layer_idx]], dim=1)
                    LQ_cur_idx = (inner_loop_num - 1) * 4 - 3  # = 21, matching full pipeline
                    cur_latents = latents[:, :, :6, :, :]  # 6 frames, matching full pipeline
                elif cur_process_idx == 1:
                    LQ_latents = None
                    inner_loop_num = 2  # matching full pipeline
                    for inner_idx in range(inner_loop_num):
                        lq_slice = LQ_video[:, :, cur_process_idx * 8 + 17 + inner_idx * 4:cur_process_idx * 8 + 21 + inner_idx * 4, :, :].to(self.device) if LQ_video is not None else None
                        cur = self.denoising_model().LQ_proj_in.stream_forward(
                            lq_slice
                        ) if lq_slice is not None else None
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat([LQ_latents[layer_idx], cur[layer_idx]], dim=1)
                    LQ_pre_idx = 21 + (inner_loop_num - 2) * 4
                    LQ_cur_idx = cur_process_idx * 8 + 21 + (inner_loop_num - 2) * 4
                    cur_latents = latents[:, :, 4 + cur_process_idx * 2:6 + cur_process_idx * 2, :, :]
                else:
                    inner_loop_num = 5
                    if cur_process_idx == process_total_num - 1:
                        LQ_pre_idx = LQ_cur_idx
                        LQ_cur_idx = LQ_cur_idx + 8 + (inner_loop_num - 2) * 4
                    else:
                        LQ_pre_idx = LQ_cur_idx
                        LQ_cur_idx = cur_process_idx * 8 + 21 + (inner_loop_num - 2) * 4
                    cur = []
                    if LQ_latents is not None:
                        for layer_idx in range(len(LQ_latents)):
                            cur.append(LQ_latents[layer_idx][:, 4:, :, :, :].contiguous())
                    if cur:
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat([LQ_latents[layer_idx], cur[layer_idx]], dim=1)
                    cur_latents = latents[:, :, 4 + cur_process_idx * 2:6 + cur_process_idx * 2, :, :]

                # === DiT forward via model_fn_wan_video (works with FakeQuantLinear) ===
                noise_pred_posi, pre_cache_k, pre_cache_v = model_fn_wan_video(
                    self.dit,
                    x=cur_latents,
                    timestep=self.timestep,
                    context=None,
                    tea_cache=None,
                    use_unified_sequence_parallel=False,
                    LQ_latents=LQ_latents,
                    is_full_block=False,
                    is_stream=is_stream,
                    pre_cache_k=pre_cache_k,
                    pre_cache_v=pre_cache_v,
                    topk_ratio=topk_ratio,
                    kv_ratio=kv_ratio,
                    cur_process_idx=cur_process_idx,
                    t_mod=self.t_mod,
                    t=self.t,
                    local_range=local_range,
                )

                # === Update latent ===
                cur_latents = cur_latents - noise_pred_posi

                # === Decode with TCDecoder ===
                cur_LQ_frame = LQ_video[:, :, LQ_pre_idx:LQ_cur_idx, :, :].to(self.device) if LQ_video is not None else None

                if self.TCDecoder is not None and cur_LQ_frame is not None:
                    cur_frames = self.TCDecoder.decode_video(
                        cur_latents.transpose(1, 2),
                        parallel=False,
                        show_progress_bar=False,
                        cond=cur_LQ_frame
                    ).transpose(1, 2).mul_(2).sub_(1)
                elif self.TCDecoder is None:
                    cur_frames = self.vae.decode(
                        cur_latents, **tiler_kwargs
                    ).mul_(2).sub_(1)
                else:
                    cur_frames = torch.zeros_like(cur_latents).transpose(1, 2).expand(-1, 3, -1, -1, -1)

                # === Color correct ===
                if cur_process_idx == 0:
                    ref_frames = cur_frames
                cur_frames = cur_frames.float()
                cur_frames = self.ColorCorrector(cur_frames, ref_frames)

                final_frames_list.append(cur_frames)

                if latent_callback is not None:
                    latent_callback(cur_latents, cur_process_idx)

            # === Merge frames ===
            final_output = torch.cat(final_frames_list, dim=1)

        self.offload_model(keep_vae=False)
        return final_output
