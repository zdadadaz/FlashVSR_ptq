"""
PTQ (Post-Training Quantization) layers for FlashVSR.

Weight: symmetric quantization — scale = absmax(w) / 127, zero_point = 0
Activation: asymmetric quantization — scale = (max - min) / 255, zero_point = round(-min / scale)

Supports:
- W8A16: int8 weight + bf16 activation passthrough (SymmetricWeightLinear)
- W8A8: int8 weight + int8 activation asymmetric (AsymmetricActLinear)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricWeightLinear(nn.Module):
    """
    W8A16: int8 symmetric weight quantization with bf16/fp16 activation passthrough.

    Weight: int8, symmetric (zero_point = 0)
    Activation: bf16/fp16 passthrough (no quantization)
    Forward: y = F.linear(x, weight_int8 * weight_scale, bias)
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight", torch.zeros((out_features, in_features), dtype=torch.int8, device=device)
        )
        self.register_buffer(
            "weight_scale", torch.ones((out_features, 1), dtype=torch.float32, device=device)
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32, device=device))
        else:
            self.register_buffer("bias", None)

    def forward(self, x):
        w_deq = self.weight.to(x.device, dtype=x.dtype) * self.weight_scale.to(x.device, dtype=x.dtype)
        bias = self.bias.to(x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w_deq, bias)

    @classmethod
    def from_float(cls, linear_module, method="max"):
        """Convert nn.Linear to SymmetricWeightLinear with symmetric weight quantization."""
        assert isinstance(linear_module, nn.Linear)
        device = linear_module.weight.device
        dtype = linear_module.weight.dtype

        new_module = cls(
            linear_module.in_features,
            linear_module.out_features,
            bias=linear_module.bias is not None,
            device=device,
            dtype=dtype,
        )

        w = linear_module.weight.data

        if method == "max":
            w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)
        elif method == "percentile99":
            flat_w = w.flatten()
            k = int(flat_w.numel() * 0.99)
            sorted_w, _ = torch.sort(torch.abs(flat_w))
            w_max = sorted_w[k].view(1, 1)
        elif method == "std":
            std = torch.std(w, dim=1, keepdim=True)
            w_max = std * 3.0
        else:
            w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)

        scale_vals = w_max / 127.0
        scale_vals = torch.clamp(scale_vals, min=1e-6)

        w_int8 = torch.round(w / scale_vals).to(torch.int8)

        new_module.weight.copy_(w_int8)
        new_module.weight_scale.copy_(scale_vals)
        if linear_module.bias is not None:
            new_module.bias.copy_(linear_module.bias.data)

        return new_module


class AsymmetricActLinear(nn.Module):
    """
    W8A8: int8 symmetric weight + int8 asymmetric activation quantization.

    Weight: int8, symmetric (zero_point = 0)
    Activation: int8, asymmetric (zero_point != 0)
    Forward:
        x_int8 = round((x - zero_point) / act_scale).to(torch.int8)
        y = F.linear(x_int8.float(), weight_int8.float() * weight_scale, bias.float())
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight", torch.zeros((out_features, in_features), dtype=torch.int8, device=device)
        )
        self.register_buffer(
            "weight_scale", torch.ones((out_features, 1), dtype=torch.float32, device=device)
        )
        # Asymmetric activation scale and zero point (per input channel)
        self.register_buffer(
            "act_scale", torch.ones((1, in_features), dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "zero_point", torch.zeros((1, in_features), dtype=torch.float32, device=device)
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32, device=device))
        else:
            self.register_buffer("bias", None)

    def forward(self, x):
        x_dtype = x.dtype
        # Quantize input to int8 with asymmetric scale and zero point
        # x_int8 = round((x - zero_point) / act_scale)
        x_float = x.float()
        x_int8 = torch.round((x_float - self.zero_point) / self.act_scale).to(torch.int8)

        # Dequantize weight to float for matmul
        w_deq = self.weight.float() * self.weight_scale

        # Matmul in float32, then cast to input dtype
        y = F.linear(x_int8.float(), w_deq, self.bias)
        return y.to(x_dtype)

    @classmethod
    def from_float(cls, linear_module, act_min=None, act_max=None, method="max"):
        """
        Convert nn.Linear to AsymmetricActLinear.

        Args:
            linear_module: nn.Linear module to convert
            act_min: Optional per-channel min of activations (for asymmetric scale)
            act_max: Optional per-channel max of activations (for asymmetric scale)
            method: Fallback weight quantization method if act stats not provided
        """
        assert isinstance(linear_module, nn.Linear)
        device = linear_module.weight.device
        dtype = linear_module.weight.dtype

        new_module = cls(
            linear_module.in_features,
            linear_module.out_features,
            bias=linear_module.bias is not None,
            device=device,
            dtype=dtype,
        )

        w = linear_module.weight.data

        # Weight quantization (symmetric)
        if method == "max":
            w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)
        elif method == "percentile99":
            flat_w = w.flatten()
            k = int(flat_w.numel() * 0.99)
            sorted_w, _ = torch.sort(torch.abs(flat_w))
            w_max = sorted_w[k].view(1, 1)
        elif method == "std":
            std = torch.std(w, dim=1, keepdim=True)
            w_max = std * 3.0
        else:
            w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)

        w_scale = w_max / 127.0
        w_scale = torch.clamp(w_scale, min=1e-6)
        w_int8 = torch.round(w / w_scale).to(torch.int8)

        new_module.weight.copy_(w_int8)
        new_module.weight_scale.copy_(w_scale)

        # Activation quantization (asymmetric)
        if act_min is not None and act_max is not None:
            act_range = act_max - act_min
            act_scale = act_range / 255.0
            act_scale = torch.clamp(act_scale, min=1e-6)
            zero_pt = torch.round(-act_min / act_scale)

            new_module.act_scale.copy_(act_scale.view(1, -1))
            new_module.zero_point.copy_(zero_pt.view(1, -1))

        if linear_module.bias is not None:
            new_module.bias.copy_(linear_module.bias.data)

        return new_module


def convert_model_to_w8a16(model, method="max"):
    """
    Recursively convert all nn.Linear to SymmetricWeightLinear (W8A16).
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, SymmetricWeightLinear.from_float(module, method=method))
        else:
            convert_model_to_w8a16(module, method=method)
    return model


def convert_model_to_w8a8(model, act_stats=None, method="max"):
    """
    Recursively convert all nn.Linear to AsymmetricActLinear (W8A8).

    Args:
        model: nn.Module to convert
        act_stats: Optional dict of {name: (act_min, act_max)} for asymmetric activation scales
        method: Fallback weight quantization method if act_stats not available for a layer
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            act_min, act_max = None, None
            if act_stats and name in act_stats:
                act_min, act_max = act_stats[name]
            setattr(model, name, AsymmetricActLinear.from_float(module, act_min, act_max, method=method))
        else:
            convert_model_to_w8a8(module, act_stats, method=method)
    return model