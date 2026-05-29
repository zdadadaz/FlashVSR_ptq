"""
FakeQuant PTQ layers for FlashVSR DiT.

Implements true integer quantization: activations (int8) and weights (int4/int8)
are quantized to actual integer types, then immediately dequantized back to float32
for computation. No bf16/fp16 passthrough — real int8/int4 throughout.

Supports: a16w8, a8w8, a16w4, a8w4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..wan_video_dit import sinusoidal_embedding_1d


# ------------------------------------------------------------------
# FakeQuantLinear — the main quantized linear layer
# ------------------------------------------------------------------

class FakeQuantLinear(nn.Module):
    """
    FakeQuantized Linear supporting a{16,8}w{8,4}.

    Quantization semantics (true integer round-trip):
        1. x_int = round_cpu(x / x_scale)          → int8
        2. x_fp  = x_int.to(float32) * x_scale     → float32 (dequantized)
        3. w_int = stored int8/packed-int4 weight
        4. w_fp  = dequantize(w_int)               → float32
        5. y     = F.linear(x_fp, w_fp)             → float32
        6. y     = y.to(original_dtype)

    Activation modes:
      a16: no activation quantization — pass through in original dtype.
           x_fp = x.float()  # no int conversion
      a8:  activation quantized to int8 via per-channel scale (ch_axis=-1).

    Weight modes:
      w8:  symmetric int8, one scale per output channel.
      w4:  symmetric int4 (packed 2/byte), one scale per output channel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_mode: str = "a16",   # "a16" or "a8"
        weight_mode: str = "w8",         # "w8" or "w4"
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_mode = activation_mode   # "a16" or "a8"
        self.weight_mode = weight_mode         # "w8" or "w4"

        # ---- Weight buffers ----
        if weight_mode == "w4":
            packed_cols = (in_features + 1) // 2
        else:
            packed_cols = in_features

        self.register_buffer(
            "weight_int",
            torch.zeros((out_features, packed_cols), dtype=torch.int8, device=device),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones((out_features, 1), dtype=torch.float32, device=device),
        )

        # ---- Activation per-channel scale / zero-point (a8 mode only) ----
        if activation_mode == "a8":
            # Per-channel scale along the feature dim (ch_axis=-1 for [B,Seq,Cin]):
            #   scale shape:  [1, 1, in_features]
            #   zero_pt shape: [1, 1, in_features]
            self.register_buffer(
                "act_scale",
                torch.ones(1, 1, in_features, dtype=torch.float32, device=device),
            )
            self.register_buffer(
                "act_zero_point",
                torch.zeros(1, 1, in_features, dtype=torch.int32, device=device),
            )
        else:
            self.register_buffer("act_scale", None)
            self.register_buffer("act_zero_point", None)

        # ---- Bias ----
        if bias:
            self.register_buffer(
                "bias", torch.zeros(out_features, dtype=torch.float32, device=device)
            )
        else:
            self.register_buffer("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ---- (1) Activation quantization → int8 → float32 ----
        if self.activation_mode == "a8":
            # Dynamic per-token symmetric int8 activation QDQ is much more stable for
            # DiT activations than stale/global min-max caches. Reduce over the
            # feature dimension only, preserving token/batch/time variation.
            x_float = x.detach().to(torch.float32)
            x_scale = torch.amax(torch.abs(x_float), dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
            x_q = torch.clamp(torch.round(x_float / x_scale), -127, 127).to(torch.int8)
            x_fp = x_q.to(torch.float32) * x_scale
        else:
            # a16: no-op, just promote to float for matmul
            x_fp = x.float()

        # ---- (2) Weight dequantization → float32 ----
        w_fp = self._dequantize_weight(x.device)

        # ---- (3) Float matmul ----
        # Explicitly cast to float32 to guarantee matching dtype for F.linear.
        # w_fp is produced by _dequantize_weight (float32) and x_fp is already float32
        # from the activation dequant path, but _dequantize_weight may return bf16
        # if weight_scale was not properly initialized (e.g., act_stats=None path).
        y = F.linear(x_fp.to(torch.float32), w_fp.to(torch.float32), self.bias.to(torch.float32) if self.bias is not None else None)

        # ---- (4) Restore original dtype ----
        return y.to(orig_dtype)

    def _dequantize_weight(self, device) -> torch.Tensor:
        """
# Dequantize stored int weight to float32.
        w4: unpack 2 int4 values per byte → float32
        w8: multiply int8 by scale → float32
        """
        if self.weight_mode == "w4":
            w_int = self.weight_int.to(device=device)      # [out, packed]
            scale = self.weight_scale.to(device=device, dtype=torch.float32)  # [out, 1]

            # Unpack nibbles: lower nibble → even cols, upper nibble >> 4 → odd cols
            w_lo = (w_int & 0x0F).to(torch.float32)                        # [out, packed]
            w_hi = ((w_int >> 4) & 0x0F).to(torch.float32)                # [out, packed]
            # w_lo/w_hi nibbles are unsigned [0,15] but represent signed [-8,7].
            # Convert: if value > 7, it represents a negative → subtract 16.
            w_lo = torch.where(w_lo > 7, w_lo - 16, w_lo)
            w_hi = torch.where(w_hi > 7, w_hi - 16, w_hi)

            w_fp = torch.zeros(
                self.out_features, self.in_features,
                dtype=torch.float32, device=device,
            )
            # Only write up to the actual in_features columns (ignore any padding column)
            w_fp[:, 0::2] = w_lo[:, :self.in_features // 2 + self.in_features % 2] * scale
            if self.in_features > 1:
                w_fp[:, 1::2] = w_hi[:, :self.in_features // 2] * scale
        else:
            w_fp = self.weight_int.to(device=device, dtype=torch.float32) * self.weight_scale.to(
                device=device, dtype=torch.float32
            )
        return w_fp

    @classmethod
    def from_float(
        cls,
        linear_module: nn.Linear,
        activation_mode: str = "a16",
        weight_mode: str = "w8",
        act_scale: torch.Tensor = None,
        act_zero_point: torch.Tensor = None,
        ch_axis: int = -1,   # kept for API compat, unused
    ):
        """
        Convert nn.Linear → FakeQuantLinear.

        Args:
            linear_module: source nn.Linear
            activation_mode: "a16" or "a8"
            weight_mode: "w8" or "w4"
            act_scale: per-channel activation scale [1, 1, Cin] or [Cin]
            act_zero_point: per-channel zero-point [1, 1, Cin] or [Cin]
            ch_axis: (unused, kept for API compat)
        """
        assert isinstance(linear_module, nn.Linear)
        device = linear_module.weight.device

        new_module = cls(
            linear_module.in_features,
            linear_module.out_features,
            activation_mode=activation_mode,
            weight_mode=weight_mode,
            bias=linear_module.bias is not None,
            device=device,
            dtype=linear_module.weight.dtype,
        )

        w = linear_module.weight.data  # [out, in]

        # ---- Weight quantization ----
        if weight_mode == "w4":
            # Symmetric int4: range [-7, 7] per output channel
            w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)
            w_scale = (w_max / 7.0).clamp(min=1e-6)
            w_int4 = torch.round(w / w_scale).to(torch.int8)

            in_f = linear_module.in_features
            out_f = linear_module.out_features
            packed_cols = (in_f + 1) // 2

            # Pack two int4 per byte: even cols → lo nibble, odd cols → hi nibble
            lo = w_int4[:, 0::2].contiguous()   # [out, packed=ceil(in_f/2)]
            hi = w_int4[:, 1::2].contiguous()   # [out, floor(in_f/2)]
            # Pad hi with zeros when in_f is odd so shapes match for packing
            if hi.shape[1] < lo.shape[1]:
                hi = torch.nn.functional.pad(hi, (0, lo.shape[1] - hi.shape[1]))
            w_packed = (lo & 0x0F) | ((hi & 0x0F) << 4)
            new_module.weight_int.copy_(w_packed)
            new_module.weight_scale.copy_(w_scale)
        else:
            # Symmetric int8: range [-127, 127] per output channel
            w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)
            w_scale = (w_max / 127.0).clamp(min=1e-6)
            w_int8 = torch.round(w / w_scale).to(torch.int8)
            new_module.weight_int.copy_(w_int8)
            new_module.weight_scale.copy_(w_scale)

        # ---- Activation calibration ----
        if activation_mode == "a8" and act_scale is not None:
            # Broadcast into [1, 1, Cin] shape expected by forward
            if act_scale.dim() == 1:
                new_module.act_scale.copy_(act_scale.view(1, 1, -1))
            else:
                new_module.act_scale.copy_(act_scale)

            if act_zero_point is not None:
                if act_zero_point.dim() == 1:
                    new_module.act_zero_point.copy_(act_zero_point.view(1, 1, -1))
                else:
                    new_module.act_zero_point.copy_(act_zero_point)

        if linear_module.bias is not None:
            new_module.bias.copy_(linear_module.bias.data.float())

        return new_module

    def set_activation_scales(self, scale: torch.Tensor, zero_point: torch.Tensor = None):
        """
        Set activation scales from calibration stats.

        scale: per-channel scale  [Cin] or [1,1,Cin]
        zero_point: per-channel zero_point  [Cin] or [1,1,Cin]
        """
        if self.act_scale is None:
            return
        if scale is not None:
            scale = scale.to(device=self.act_scale.device)
            if scale.dim() == 1:
                self.act_scale.copy_(scale.view(1, 1, -1))
            else:
                self.act_scale.copy_(scale)
        if zero_point is not None:
            zero_point = zero_point.to(device=self.act_zero_point.device)
            if zero_point.dim() == 1:
                self.act_zero_point.copy_(zero_point.view(1, 1, -1))
            else:
                self.act_zero_point.copy_(zero_point)


# ------------------------------------------------------------------
# Model conversion
# ------------------------------------------------------------------

def convert_model_to_fakequant(
    model,
    mode: str = "a16w8",
    act_stats: dict = None,
    ch_axis: int = -1,
    method: str = "max",
):
    """
    Recursively replace nn.Linear → FakeQuantLinear.

    Args:
        model: nn.Module to convert
        mode: "a16w8", "a8w8", "a16w4", "a8w4"
        act_stats: calibration dict {name: {'act_scale': tensor, 'zero_point': tensor}}
        ch_axis: (unused, kept for API compat)
        method: weight quantization method (only "max" currently)
    """
    if mode.startswith("a16"):
        activation_mode = "a16"
        weight_mode = mode[3:]
    elif mode.startswith("a8"):
        activation_mode = "a8"
        weight_mode = mode[2:]
    else:
        raise ValueError(f"Unsupported fakequant mode: {mode}")
    if weight_mode not in ("w8", "w4"):
        raise ValueError(f"Unsupported fakequant weight mode from {mode}: {weight_mode}")

    converted = 0
    fallback = 0

    def get_parent_and_name(mod, full_name):
        parts = full_name.rsplit(".", 1)
        if len(parts) == 1:
            return mod, parts[0]
        parent_name, leaf_name = parts
        parent = mod
        for p in parent_name.split("."):
            parent = getattr(parent, p)
        return parent, leaf_name

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Retrieve activation calibration if available
        act_scale, act_zp = None, None
        if act_stats:
            s = None
            if full_name in act_stats:
                s = act_stats[full_name]
            else:
                leaf = full_name.rsplit(".", 1)[-1] if "." in full_name else full_name
                if leaf in act_stats:
                    s = act_stats[leaf]
            if s is not None:
                act_scale = s.get("scale") or s.get("act_scale")
                act_zp    = s.get("zero_point")

        try:
            new_mod = FakeQuantLinear.from_float(
                module,
                activation_mode=activation_mode,
                weight_mode=weight_mode,
                act_scale=act_scale,
                act_zero_point=act_zp,
                ch_axis=ch_axis,
            )
            parent, leaf_name = get_parent_and_name(model, full_name)
            setattr(parent, leaf_name, new_mod)
            converted += 1
        except Exception as e:
            print(f"  [FakeQuant] Failed to convert {full_name}: {e}")
            fallback += 1

    print(f"[FakeQuant] {mode}: {converted} converted, {fallback} fallback (unchanged)")
    return model


# ------------------------------------------------------------------
# Calibration helper
# ------------------------------------------------------------------

class CalibrationObserverForFakeQuant(nn.Module):
    """
    Standalone observer module that records min/max of inputs.
    Used to collect activation scales before converting to FakeQuantLinear.
    """

    def __init__(self, ch_axis: int = -1):
        super().__init__()
        self.ch_axis = ch_axis

    def forward(self, x):
        return x  # passthrough


def collect_activation_stats_fakequant(
    model,
    latents,
    contexts,
    num_samples: int = 320,
) -> dict:
    """
    Run calibration forward passes and collect per-layer activation statistics.

    Returns:
        dict: {layer_name: {'act_scale': tensor [Cin], 'zero_point': tensor [Cin]}}

    Uses register_forward_hook on every nn.Linear — avoids time_embedding
    dtype issues by bypassing it entirely (hooks fire on sub-module forwards).
    """
    model.eval()

    act_stats = {}
    hooks = []

    # ---- Pre-compute t_mod once to satisfy model internals ----
    model.cuda()
    t_big = torch.randint(0, 1000, (1,), device="cuda")
    t_emb = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, t_big.float()))  # float!
    t_mod_for_fwd = model.time_projection(t_emb).unflatten(1, (6, model.dim))  # (B, 6, dim)
    t_cpu = t_big.cpu()  # keep on CPU for reuse

    # Fixed freqs tensor
    f, h, w = 4, 15, 20
    freqs = torch.cat([
        model.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        model.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        model.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).cuda().float()

    # ---- Hook factory ----
    def make_hook(name):
        def hook_fn(module, input, output):
            act = input[0] if isinstance(input, tuple) else input
            act = act.detach().float()
            if name not in act_stats:
                act_stats[name] = {"min": [], "max": []}
            # Per-channel: amin/amax over all dims except last (feature dim)
            dims_to_reduce = list(range(act.dim() - 1))
            act_min = act.amin(dim=dims_to_reduce, keepdim=True)
            act_max = act.amax(dim=dims_to_reduce, keepdim=True)
            act_stats[name]["min"].append(act_min.cpu())
            act_stats[name]["max"].append(act_max.cpu())
        return hook_fn

    # ---- Register hooks on every Linear ----
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    # ---- Run block-level forward (no time_embedding call per iteration) ----
    # WanModel expects input in (B, H, W, C) format — latents are (B, C, H, W)
    # so we need to rearrange to channels-last before patchify
    with torch.no_grad():
        for i in range(min(num_samples, len(latents))):
            # WanModel.patchify does: x = Conv3d(x) → rearrange(x, 'b c f h w -> b (f h w) c')
            # Our latents are (B, C, H, W) — Conv3d expects (B, C, F, H, W)
            # So rearrange (B, H, W, C) → (B, C, H, W) before calling patchify
            # Support both list-of-tensors (from proxy latent extraction) and single tensor
            if isinstance(latents, (list, tuple)):
                x_4d = latents[i:i+1][0].float() if len(latents) > i else None
                if x_4d is None:
                    continue
            else:
                x_4d = latents[i:i+1].float()
            # Latents from VAE encode are (B, C, D=1, H, W) — squeeze out the D=1 dim
            x_4d = x_4d.squeeze(2)  # (1, C, 1, H, W) -> (1, C, H, W)
            x_4d = x_4d.permute(0, 2, 3, 1).contiguous()                        # (1, H, W, C)
            x_chw = x_4d.permute(0, 3, 1, 2).contiguous()                  # (1, C, H, W)
            x_5d = x_chw.unsqueeze(2).cuda()                                # (1, C, D=1, H, W) on CUDA
            # Pad input so the *patchified* grid divides the attention window (2,8,8).
            # patch_size=(1,2,2), so spatial input must be divisible by 16, not just 8.
            _, _, D_pad, H_pad, W_pad = x_5d.shape
            pf, ph, pw = model.patch_size
            win_f, win_h, win_w = 2, 8, 8
            req_f = pf * win_f
            req_h = ph * win_h
            req_w = pw * win_w
            pad_f = (req_f - D_pad % req_f) % req_f
            pad_h = (req_h - H_pad % req_h) % req_h
            pad_w = (req_w - W_pad % req_w) % req_w
            if pad_f or pad_h or pad_w:
                x_5d = torch.nn.functional.pad(x_5d, (0, pad_w, 0, pad_h, 0, pad_f))
            ctx = contexts[i % len(contexts)].cuda() if i < len(contexts) else torch.randn(1, 10, 4096, device="cuda")
            ctx = model.text_embedding(ctx.float())  # (B, 10, 4096) -> (B, 10, dim)
            x_patched, (f_, h_, w_) = model.patchify(x_5d)
            # Recompute freqs for actual grid size (freqs shape must match L = f_ * h_ * w_)
            freqs_i = torch.cat([
                model.freqs[0][:f_].view(f_, 1, 1, -1).expand(f_, h_, w_, -1),
                model.freqs[1][:h_].view(1, h_, 1, -1).expand(f_, h_, w_, -1),
                model.freqs[2][:w_].view(1, 1, w_, -1).expand(f_, h_, w_, -1),
            ], dim=-1).reshape(f_ * h_ * w_, 1, -1).cuda().float()
            x = x_patched
            for block in model.blocks:
                x = block(x, ctx, t_mod_for_fwd, freqs_i,
                          f_, h_, w_, f_ * h_ * w_, f_ * h_ * w_,
                          False, i, 1, False, False, None, None)
    for h in hooks:
        h.remove()

    # ---- Compute per-channel scales from collected stats ----
    result = {}
    for name, stats in act_stats.items():
        if not stats["min"]:
            continue
        all_min = torch.cat(stats["min"], dim=0)
        all_max = torch.cat(stats["max"], dim=0)
        act_min = torch.amin(all_min, dim=0)
        act_max = torch.amax(all_max, dim=0)
        # Signed int8 asymmetric quantization uses qmin=-128, qmax=127.
        # For q = round(x / scale + zero_point), zero_point must include qmin;
        # using uint8-style zero_point = -min/scale with signed int8 shifts/clips
        # the entire activation distribution and produces severe noise.
        qmin, qmax = -128.0, 127.0
        act_range = act_max - act_min
        act_scale = act_range / (qmax - qmin)
        act_scale = torch.clamp(act_scale, min=1e-6)
        zero_pt = torch.round(qmin - act_min / act_scale).clamp(qmin, qmax)
        result[name] = {
            "act_scale": act_scale.squeeze().float(),
            "zero_point": zero_pt.squeeze().long(),
        }
    return result


def get_all_linear_layers(model) -> list:
    """Return list of (name, module) for all nn.Linear in model."""
    return [
        (name, m) for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
    ]
