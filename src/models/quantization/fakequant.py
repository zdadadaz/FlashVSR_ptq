"""
FakeQuant PTQ layers for FlashVSR DiT.

Implements true integer quantization: activations (int8) and weights (int4/int8)
are quantized to actual integer types, then immediately dequantized back to float32
for computation. No bf16/fp16 passthrough — real int8/int4 throughout.

Supports: a16w8, a8w8, a16w4, a8w4, a4w4
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
      a8:  activation quantized to signed int8 with calibrated asymmetric
           per-channel scale and zero-point (ch_axis=-1).

    Weight modes:
      w8:  symmetric int8, one scale per output channel.
      w4:  symmetric int4 (packed 2/byte), one scale per output channel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_mode: str = "a16",   # "a16", "a8", or "a4"
        weight_mode: str = "w8",         # "w8" or "w4"
        act_quant_enabled: bool = True,
        activation_qdq_mode: str = "static_asymmetric",
        draq_qrange: str = "signed_symmetric",
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_mode = activation_mode   # "a16", "a8", or "a4"
        self.weight_mode = weight_mode         # "w8" or "w4"
        activation_mode_to_id = {"a16": 1, "a8": 2, "a4": 3}
        if activation_mode not in activation_mode_to_id:
            raise ValueError(f"Unsupported activation_mode: {activation_mode}")
        self.register_buffer(
            "activation_mode_code",
            torch.tensor(activation_mode_to_id[activation_mode], dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "act_quant_enabled",
            torch.tensor(bool(act_quant_enabled), dtype=torch.bool, device=device),
        )
        qdq_mode_to_id = {
            "static_asymmetric": 0,
            "dynamic_symmetric": 1,
            "dynamic_asymmetric": 2,
            "draq_symmetric": 3,
        }
        if activation_qdq_mode not in qdq_mode_to_id:
            raise ValueError(f"Unsupported activation_qdq_mode: {activation_qdq_mode}")
        self.register_buffer(
            "activation_qdq_mode",
            torch.tensor(qdq_mode_to_id[activation_qdq_mode], dtype=torch.int32, device=device),
        )
        draq_qrange_to_id = {"signed_symmetric": 0, "signed_full": 1}
        if draq_qrange not in draq_qrange_to_id:
            raise ValueError(f"Unsupported draq_qrange: {draq_qrange}")
        self.register_buffer(
            "draq_qrange",
            torch.tensor(draq_qrange_to_id[draq_qrange], dtype=torch.int32, device=device),
        )

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

        # ---- Activation per-channel scale / zero-point (static a8 mode only) ----
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

        # ---- (1) Activation quantization → integer → float32 ----
        activation_mode_code = int(self.activation_mode_code.item())
        activation_is_a8 = activation_mode_code == 2
        activation_is_a4 = activation_mode_code == 3
        if (activation_is_a8 or activation_is_a4) and bool(self.act_quant_enabled.item()):
            x_float = x.detach().to(torch.float32)
            qdq_mode = int(self.activation_qdq_mode.item())
            if activation_is_a4:
                if qdq_mode not in (1, 3):
                    raise ValueError("A4 activation QDQ supports dynamic_symmetric or draq_symmetric modes only")
                if qdq_mode == 3:
                    # LSGQuant-style DRAQ with signed int4 qrange [-7, 7].
                    qmin, qmax = -7.0, 7.0
                    reduce_channel = tuple(range(x_float.dim() - 1))
                    s = torch.amax(torch.abs(x_float), dim=reduce_channel, keepdim=True).clamp(min=1e-6)
                    x_norm = x_float / s
                    d = torch.amax(torch.abs(x_norm), dim=-1, keepdim=True).clamp(min=1e-6)
                    x_q = torch.clamp(torch.round(qmax * x_norm / d), qmin, qmax).to(torch.int8)
                    x_fp = (x_q.to(torch.float32) / qmax) * d * s
                else:
                    # Dynamic per-token symmetric signed-int4 activation QDQ.
                    x_scale = torch.amax(torch.abs(x_float), dim=-1, keepdim=True).clamp(min=1e-6) / 7.0
                    x_q = torch.clamp(torch.round(x_float / x_scale), -7, 7).to(torch.int8)
                    x_fp = x_q.to(torch.float32) * x_scale
            elif qdq_mode == 1:
                # Dynamic per-token symmetric signed-int8 activation QDQ.
                x_scale = torch.amax(torch.abs(x_float), dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
                x_q = torch.clamp(torch.round(x_float / x_scale), -127, 127).to(torch.int8)
                x_fp = x_q.to(torch.float32) * x_scale
            elif qdq_mode == 2:
                # Dynamic per-token asymmetric signed-int8 activation QDQ.
                qmin, qmax = -128.0, 127.0
                x_min = torch.amin(x_float, dim=-1, keepdim=True)
                x_max = torch.amax(x_float, dim=-1, keepdim=True)
                x_scale = ((x_max - x_min) / (qmax - qmin)).clamp(min=1e-6)
                x_zero_point = torch.round(qmin - x_min / x_scale).clamp(qmin, qmax)
                x_q = torch.clamp(torch.round(x_float / x_scale + x_zero_point), qmin, qmax).to(torch.int8)
                x_fp = (x_q.to(torch.float32) - x_zero_point) * x_scale
            elif qdq_mode == 3:
                # LSGQuant DRAQ: per-channel normalization over token-like dims,
                # followed by per-token symmetric signed-int8 QDQ. Scales are
                # online and do not require calibration cache entries.
                qmin, qmax = (-128.0, 127.0) if int(self.draq_qrange.item()) == 1 else (-127.0, 127.0)
                reduce_channel = tuple(range(x_float.dim() - 1))
                s = torch.amax(torch.abs(x_float), dim=reduce_channel, keepdim=True).clamp(min=1e-6)
                x_norm = x_float / s
                d = torch.amax(torch.abs(x_norm), dim=-1, keepdim=True).clamp(min=1e-6)
                x_q = torch.clamp(torch.round(qmax * x_norm / d), qmin, qmax).to(torch.int8)
                x_fp = (x_q.to(torch.float32) / qmax) * d * s
            else:
                # Static calibrated signed-int8 activation QDQ.
                # act_scale / act_zero_point are collected by fakequant_calibrate.py
                # and loaded by fakequant_convert.py.  Reshape dynamically so both
                # 2D inputs [B,C] and sequence inputs [B,L,C] broadcast correctly.
                x_scale = self.act_scale.to(device=x.device, dtype=torch.float32).reshape(
                    *([1] * (x.dim() - 1)), self.in_features
                ).clamp(min=1e-6)
                x_zero_point = self.act_zero_point.to(device=x.device, dtype=torch.float32).reshape(
                    *([1] * (x.dim() - 1)), self.in_features
                )
                x_q = torch.clamp(torch.round(x_float / x_scale + x_zero_point), -128, 127).to(torch.int8)
                x_fp = (x_q.to(torch.float32) - x_zero_point) * x_scale
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
        act_mean: torch.Tensor = None,
        act_quant_enabled: bool = True,
        activation_qdq_mode: str = "static_asymmetric",
        draq_qrange: str = "signed_symmetric",
        ch_axis: int = -1,   # kept for API compat, unused
    ):
        """
        Convert nn.Linear → FakeQuantLinear.

        Args:
            linear_module: source nn.Linear
            activation_mode: "a16", "a8", or "a4"
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
            act_quant_enabled=act_quant_enabled,
            activation_qdq_mode=activation_qdq_mode,
            draq_qrange=draq_qrange,
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

        # ---- Bias correction ----
        # Lightweight PTQ recovery: if calibration provides per-input-channel
        # mean activation, compensate expected rounding error in the output bias:
        # E[x] @ (W_fp - W_qdq)^T. This is deterministic and training-free.
        bias_correction = None
        if act_mean is not None:
            x_mean = act_mean.detach().to(device=device, dtype=torch.float32).reshape(-1)
            if x_mean.numel() == linear_module.in_features:
                w_deq = new_module._dequantize_weight(device).to(torch.float32)
                bias_correction = torch.matmul(x_mean, (w.to(torch.float32) - w_deq).t())

        # ---- Activation calibration ----
        if activation_mode == "a8" and act_scale is not None:
            # Broadcast into [1, 1, Cin] shape expected by forward.
            # Per-tensor static caches store a single scalar/list entry; expand
            # that scalar to all input channels so the runtime QDQ path remains
            # unchanged and checkpoint loading stays compatible.
            if act_scale.numel() == 1:
                new_module.act_scale.copy_(act_scale.reshape(1, 1, 1).expand_as(new_module.act_scale))
            elif act_scale.dim() == 1:
                new_module.act_scale.copy_(act_scale.view(1, 1, -1))
            else:
                new_module.act_scale.copy_(act_scale)

            if act_zero_point is not None:
                if act_zero_point.numel() == 1:
                    new_module.act_zero_point.copy_(act_zero_point.reshape(1, 1, 1).expand_as(new_module.act_zero_point))
                elif act_zero_point.dim() == 1:
                    new_module.act_zero_point.copy_(act_zero_point.view(1, 1, -1))
                else:
                    new_module.act_zero_point.copy_(act_zero_point)

        if linear_module.bias is not None:
            new_module.bias.copy_(linear_module.bias.data.float())
            if bias_correction is not None:
                new_module.bias.add_(bias_correction)

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
            if scale.numel() == 1:
                self.act_scale.copy_(scale.reshape(1, 1, 1).expand_as(self.act_scale))
            elif scale.dim() == 1:
                self.act_scale.copy_(scale.view(1, 1, -1))
            else:
                self.act_scale.copy_(scale)
        if zero_point is not None:
            zero_point = zero_point.to(device=self.act_zero_point.device)
            if zero_point.numel() == 1:
                self.act_zero_point.copy_(zero_point.reshape(1, 1, 1).expand_as(self.act_zero_point))
            elif zero_point.dim() == 1:
                self.act_zero_point.copy_(zero_point.view(1, 1, -1))
            else:
                self.act_zero_point.copy_(zero_point)



# ------------------------------------------------------------------
# FakeQuant ConvNd — optional quantized convolution layers
# ------------------------------------------------------------------

class _FakeQuantConvNd(nn.Module):
    """Common fake-quant QDQ implementation for Conv2d/Conv3d.

    A8W8 semantics match FakeQuantLinear: activations and weights are rounded to
    true int8 tensors, immediately dequantized to float32, then computed with the
    normal PyTorch convolution kernel. This is intended for sensitivity analysis
    (quality/PSNR impact), not accelerated inference.
    """

    conv_dim = None

    def __init__(self, conv_module: nn.Module, activation_mode: str = "a8", weight_mode: str = "w8"):
        super().__init__()
        if weight_mode != "w8":
            raise ValueError("FakeQuantConv currently supports W8 only")
        if activation_mode not in ("a8", "a16"):
            raise ValueError(f"Unsupported activation mode: {activation_mode}")

        self.activation_mode = activation_mode
        self.weight_mode = weight_mode
        self.in_channels = conv_module.in_channels
        self.out_channels = conv_module.out_channels
        self.kernel_size = conv_module.kernel_size
        self.stride = conv_module.stride
        self.padding = conv_module.padding
        self.dilation = conv_module.dilation
        self.groups = conv_module.groups
        self.padding_mode = conv_module.padding_mode
        # Preserve custom causal padding used by src.models.utils.CausalConv3d.
        self._causal_padding = tuple(getattr(conv_module, "_padding", ()))
        # Same layout PyTorch ConvNd uses for non-zero padding modes.
        self._reversed_padding_repeated_twice = tuple(x for p in reversed(self.padding) for x in (p, p))

        w = conv_module.weight.detach().to(torch.float32)
        reduce_dims = tuple(range(1, w.dim()))
        w_max = torch.amax(torch.abs(w), dim=reduce_dims, keepdim=True)
        w_scale = (w_max / 127.0).clamp(min=1e-6)
        w_int8 = torch.clamp(torch.round(w / w_scale), -127, 127).to(torch.int8)
        self.register_buffer("weight_int", w_int8)
        self.register_buffer("weight_scale", w_scale.to(torch.float32))
        if conv_module.bias is not None:
            self.register_buffer("bias", conv_module.bias.detach().to(torch.float32).clone())
        else:
            self.register_buffer("bias", None)

    def _qdq_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_mode != "a8":
            return x.to(torch.float32)
        x_float = x.detach().to(torch.float32)
        # Per-sample, per-input-channel dynamic scale; reduce spatial/temporal dims.
        reduce_dims = tuple(d for d in range(x_float.dim()) if d != 1)
        x_scale = torch.amax(torch.abs(x_float), dim=reduce_dims, keepdim=True).clamp(min=1e-6) / 127.0
        x_q = torch.clamp(torch.round(x_float / x_scale), -127, 127).to(torch.int8)
        return x_q.to(torch.float32) * x_scale

    def _dequantize_weight(self, device) -> torch.Tensor:
        return self.weight_int.to(device=device, dtype=torch.float32) * self.weight_scale.to(device=device, dtype=torch.float32)


class FakeQuantConv2d(_FakeQuantConvNd):
    conv_dim = 2

    @classmethod
    def from_float(cls, conv_module: nn.Conv2d, activation_mode: str = "a8", weight_mode: str = "w8"):
        if not isinstance(conv_module, nn.Conv2d):
            raise TypeError(f"Expected nn.Conv2d, got {type(conv_module)}")
        return cls(conv_module, activation_mode=activation_mode, weight_mode=weight_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp = self._qdq_activation(x)
        w_fp = self._dequantize_weight(x.device)
        if self.padding_mode != "zeros":
            x_fp = F.pad(x_fp, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        y = F.conv2d(
            x_fp.to(torch.float32), w_fp.to(torch.float32),
            self.bias.to(torch.float32) if self.bias is not None else None,
            self.stride, padding, self.dilation, self.groups,
        )
        return y.to(orig_dtype)


class FakeQuantConv3d(_FakeQuantConvNd):
    conv_dim = 3

    @classmethod
    def from_float(cls, conv_module: nn.Conv3d, activation_mode: str = "a8", weight_mode: str = "w8"):
        if not isinstance(conv_module, nn.Conv3d):
            raise TypeError(f"Expected nn.Conv3d, got {type(conv_module)}")
        return cls(conv_module, activation_mode=activation_mode, weight_mode=weight_mode)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor = None) -> torch.Tensor:
        orig_dtype = x.dtype
        if self._causal_padding:
            padding = list(self._causal_padding)
            if cache_x is not None and padding[4] > 0:
                cache_x = cache_x.to(x.device)
                x = torch.cat([cache_x, x], dim=2)
                padding[4] -= cache_x.shape[2]
            x = F.pad(x, padding, mode='replicate')
            conv_padding = (0, 0, 0)
        else:
            conv_padding = self.padding
        x_fp = self._qdq_activation(x)
        w_fp = self._dequantize_weight(x.device)
        if self.padding_mode != "zeros" and not self._causal_padding:
            x_fp = F.pad(x_fp, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            conv_padding = (0, 0, 0)
        y = F.conv3d(
            x_fp.to(torch.float32), w_fp.to(torch.float32),
            self.bias.to(torch.float32) if self.bias is not None else None,
            self.stride, conv_padding, self.dilation, self.groups,
        )
        return y.to(orig_dtype)

# ------------------------------------------------------------------
# Model conversion
# ------------------------------------------------------------------

def convert_model_to_fakequant(
    model,
    mode: str = "a16w8",
    act_stats: dict = None,
    ch_axis: int = -1,
    method: str = "max",
    static_quality_policy: str = "none",
    activation_qdq_mode: str = "static_asymmetric",
    draq_qrange: str = "signed_symmetric",
    layer_policy: dict | None = None,
    enable_bias_correction: bool = False,
):
    """Recursively replace nn.Linear → FakeQuantLinear.

    `layer_policy` is an optional mapping `{layer_name: {mode, activation_qdq_mode}}`
    used by Person A's August PTQ recovery flow. It can mix `a8w8`, `a16w8`,
    and `fp16_skip` at per-Linear granularity while preserving the existing
    global-mode behavior when omitted.
    """
    supported_qdq_modes = ("static_asymmetric", "dynamic_symmetric", "dynamic_asymmetric", "draq_symmetric")
    if activation_qdq_mode not in supported_qdq_modes:
        raise ValueError(f"Unsupported activation_qdq_mode: {activation_qdq_mode}")
    if draq_qrange not in ("signed_symmetric", "signed_full"):
        raise ValueError(f"Unsupported draq_qrange: {draq_qrange}")
    if not (mode.startswith("a16") or mode.startswith("a8") or mode.startswith("a4")):
        raise ValueError(f"Unsupported fakequant mode: {mode}")
    default_weight_mode = mode[3:] if mode.startswith("a16") else mode[2:]
    if default_weight_mode not in ("w8", "w4"):
        raise ValueError(f"Unsupported fakequant weight mode from {mode}: {default_weight_mode}")

    converted = 0
    fallback = 0
    missing_act_stats = []
    act_disabled = 0
    skipped_fp16 = 0
    mode_counts = {}

    def should_disable_activation_quant(full_name: str) -> bool:
        if static_quality_policy in (None, "", "none"):
            return False
        if static_quality_policy not in ("sensitive_a16", "self_attn_only_a8"):
            raise ValueError(f"Unsupported static_quality_policy: {static_quality_policy}")
        if static_quality_policy == "self_attn_only_a8":
            return ".self_attn." not in full_name
        sensitive_prefixes = (
            "text_embedding.",
            "time_embedding.",
            "time_projection.",
            "head.head",
        )
        if full_name.startswith(sensitive_prefixes):
            return True
        if ".ffn." in full_name:
            return True
        return False

    def get_parent_and_name(mod, full_name):
        parts = full_name.rsplit(".", 1)
        if len(parts) == 1:
            return mod, parts[0]
        parent_name, leaf_name = parts
        parent = mod
        for p in parent_name.split("."):
            parent = getattr(parent, p)
        return parent, leaf_name

    policy = layer_policy or {}
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        entry = policy.get(full_name, {})
        layer_mode = entry.get("mode", mode) if isinstance(entry, dict) else (entry or mode)
        layer_qdq_mode = (
            entry.get("activation_qdq_mode", activation_qdq_mode)
            if isinstance(entry, dict) else activation_qdq_mode
        )
        if layer_qdq_mode not in supported_qdq_modes:
            raise ValueError(f"Unsupported activation_qdq_mode for {full_name}: {layer_qdq_mode}")
        if layer_mode == "fp16_skip":
            skipped_fp16 += 1
            mode_counts[layer_mode] = mode_counts.get(layer_mode, 0) + 1
            continue
        if layer_mode.startswith("a16"):
            layer_activation_mode = "a16"
            layer_weight_mode = layer_mode[3:]
        elif layer_mode.startswith("a8"):
            layer_activation_mode = "a8"
            layer_weight_mode = layer_mode[2:]
        elif layer_mode.startswith("a4"):
            layer_activation_mode = "a4"
            layer_weight_mode = layer_mode[2:]
        else:
            raise ValueError(f"Unsupported layer mode for {full_name}: {layer_mode}")
        if layer_weight_mode not in ("w8", "w4"):
            raise ValueError(f"Unsupported layer weight mode for {full_name}: {layer_weight_mode}")

        act_scale, act_zp, act_mean = None, None, None
        if act_stats:
            s = None
            if full_name in act_stats:
                s = act_stats[full_name]
            else:
                leaf = full_name.rsplit(".", 1)[-1] if "." in full_name else full_name
                if leaf in act_stats:
                    s = act_stats[leaf]
            if s is not None:
                act_scale = s.get("scale", None)
                if act_scale is None:
                    act_scale = s.get("act_scale", None)
                act_zp = s.get("zero_point", None)
                act_mean = s.get("act_mean", None)
        if (
            layer_activation_mode == "a8"
            and layer_qdq_mode == "static_asymmetric"
            and act_stats is not None
            and act_scale is None
        ):
            missing_act_stats.append(full_name)
            continue

        try:
            disable_act_q = layer_activation_mode in ("a8", "a4") and should_disable_activation_quant(full_name)
            new_mod = FakeQuantLinear.from_float(
                module,
                activation_mode=layer_activation_mode,
                weight_mode=layer_weight_mode,
                act_scale=act_scale,
                act_zero_point=act_zp,
                act_mean=act_mean if enable_bias_correction else None,
                act_quant_enabled=not disable_act_q,
                activation_qdq_mode=layer_qdq_mode,
                draq_qrange=draq_qrange,
                ch_axis=ch_axis,
            )
            parent, leaf_name = get_parent_and_name(model, full_name)
            setattr(parent, leaf_name, new_mod)
            converted += 1
            mode_counts[layer_mode] = mode_counts.get(layer_mode, 0) + 1
            if disable_act_q:
                act_disabled += 1
        except Exception as e:
            print(f"  [FakeQuant] Failed to convert {full_name}: {e}")
            fallback += 1

    if missing_act_stats:
        preview = ", ".join(missing_act_stats[:8])
        suffix = "..." if len(missing_act_stats) > 8 else ""
        raise RuntimeError(
            f"A8 FakeQuant requires calibrated asymmetric activation stats for every "
            f"Linear layer; missing {len(missing_act_stats)} layer(s): {preview}{suffix}"
        )

    summary = {
        "converted": converted,
        "fallback": fallback,
        "fp16_skip": skipped_fp16,
        "act_q_disabled": act_disabled,
        "mode_counts": mode_counts,
        "static_quality_policy": static_quality_policy,
        "activation_qdq_mode": activation_qdq_mode,
        "draq_qrange": draq_qrange,
        "enable_bias_correction": enable_bias_correction,
    }
    print(
        f"[FakeQuant] {mode}: {converted} converted, {fallback} fallback (unchanged), "
        f"fp16_skip={skipped_fp16}, act_q_disabled={act_disabled}, "
        f"static_quality_policy={static_quality_policy}, activation_qdq_mode={activation_qdq_mode}, "
        f"mode_counts={mode_counts}"
    )
    model._fakequant_conversion_summary = summary
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
        Activation scales use signed-int8 asymmetric quantization:
            q = clamp(round(x / scale + zero_point), -128, 127)
            x_fp = (q - zero_point) * scale

    Uses register_forward_hook on every nn.Linear — avoids time_embedding
    dtype issues by bypassing it entirely (hooks fire on sub-module forwards).
    """
    model.eval()

    act_stats = {}
    hooks = []

    # ---- Hook factory ----
    def make_hook(name):
        def hook_fn(module, input, output):
            act = input[0] if isinstance(input, tuple) else input
            act = act.detach().float()
            if name not in act_stats:
                act_stats[name] = {"min": [], "max": [], "sum": [], "count": [], "mu": []}
            # Per-channel: amin/amax over all dims except last (feature dim)
            dims_to_reduce = list(range(act.dim() - 1))
            act_min = act.amin(dim=dims_to_reduce, keepdim=True)
            act_max = act.amax(dim=dims_to_reduce, keepdim=True)
            act_stats[name]["min"].append(act_min.cpu())
            act_stats[name]["max"].append(act_max.cpu())
            act_sum = act.sum(dim=dims_to_reduce, keepdim=True)
            reduce_count = 1
            for dim in dims_to_reduce:
                reduce_count *= act.shape[dim]
            act_stats[name]["sum"].append(act_sum.cpu())
            act_stats[name]["count"].append(reduce_count)
            act_stats[name]["mu"].append((act_sum / max(reduce_count, 1)).squeeze().cpu())
        return hook_fn

    # ---- Register hooks on every Linear ----
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    # ---- Pre-compute t_mod once to satisfy model internals ----
    # Hooks must already be registered here so time_embedding/time_projection
    # receive calibrated asymmetric A8 stats too.
    model.cuda()
    t_big = torch.randint(0, 1000, (1,), device="cuda")
    t_emb = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, t_big.float()))  # float!
    t_mod_for_fwd = model.time_projection(t_emb).unflatten(1, (6, model.dim))  # (B, 6, dim)

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
            # Run the output projection as well so head.head receives calibrated
            # asymmetric A8 activation stats instead of falling back to default
            # scale=1 / zero_point=0 during conversion.
            _ = model.head(x, t_emb)
    for h in hooks:
        h.remove()

    # ---- Compute per-channel scales from collected stats ----
    result = {}
    for name, stats in act_stats.items():
        if not stats["min"]:
            continue
        all_min = torch.cat(stats["min"], dim=0)
        all_max = torch.cat(stats["max"], dim=0)
        all_sum = torch.stack(stats["sum"], dim=0).sum(dim=0)
        all_count = float(sum(stats["count"]))
        act_min = torch.amin(all_min, dim=0)
        act_max = torch.amax(all_max, dim=0)
        act_mean = all_sum / max(all_count, 1.0)
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
            "act_min": act_min.squeeze().float(),
            "act_max": act_max.squeeze().float(),
            "act_mean": act_mean.squeeze().float(),
            "mu_samples_mean": torch.stack(stats["mu"], dim=0).float(),
        }
    return result


def get_all_linear_layers(model) -> list:
    """Return list of (name, module) for all nn.Linear in model."""
    return [
        (name, m) for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
    ]


def convert_ops_to_fakequant(
    model,
    mode: str = "a8w8",
    op_types=("linear",),
    prefix: str = "",
):
    """Recursively replace selected op types with FakeQuant QDQ modules.

    Args:
        model: module to mutate in-place.
        mode: a8w8/a16w8. Conv ops currently support W8 only.
        op_types: iterable containing any of: linear, conv2d, conv3d.
        prefix: optional name prefix for logging only.
    """
    if mode.startswith("a16"):
        activation_mode, weight_mode = "a16", mode[3:]
    elif mode.startswith("a8"):
        activation_mode, weight_mode = "a8", mode[2:]
    elif mode.startswith("a4"):
        activation_mode, weight_mode = "a4", mode[2:]
    else:
        raise ValueError(f"Unsupported fakequant mode: {mode}")
    if weight_mode != "w8":
        raise ValueError("Conv/LQ/VAE/TCDecoder op fakequant currently supports W8 modes only")

    op_types = set(op_types or ())
    converted = {"linear": 0, "conv2d": 0, "conv3d": 0}
    fallback = 0

    def get_parent_and_name(mod, full_name):
        parts = full_name.rsplit(".", 1)
        if len(parts) == 1:
            return mod, parts[0]
        parent = mod
        for p in parts[0].split("."):
            parent = getattr(parent, p)
        return parent, parts[1]

    for full_name, module in list(model.named_modules()):
        if full_name == "":
            continue
        kind = None
        factory = None
        # Conv3d before Conv2d is not necessary but keeps subclass intent explicit.
        if "conv3d" in op_types and isinstance(module, nn.Conv3d):
            kind, factory = "conv3d", FakeQuantConv3d.from_float
        elif "conv2d" in op_types and isinstance(module, nn.Conv2d):
            kind, factory = "conv2d", FakeQuantConv2d.from_float
        elif "linear" in op_types and isinstance(module, nn.Linear):
            kind = "linear"
            factory = lambda m, activation_mode, weight_mode: FakeQuantLinear.from_float(
                m, activation_mode=activation_mode, weight_mode=weight_mode
            )
        if kind is None:
            continue
        try:
            new_mod = factory(module, activation_mode=activation_mode, weight_mode=weight_mode)
            parent, leaf_name = get_parent_and_name(model, full_name)
            setattr(parent, leaf_name, new_mod)
            converted[kind] += 1
        except Exception as e:
            print(f"  [FakeQuantOps] Failed to convert {prefix + '.' if prefix else ''}{full_name}: {e}")
            fallback += 1
    print(f"[FakeQuantOps] {prefix or type(model).__name__}: mode={mode} converted={converted} fallback={fallback}")
    return model
