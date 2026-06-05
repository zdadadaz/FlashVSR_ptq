"""QAT helpers for FlashVSR DiT fake-quant fine-tuning.

This module is intentionally DiT-side only.  It converts trainable nn.Linear
layers to STE fake-quant Linear layers for Person A's September QAT pipeline,
then exports the trained FP weights back to the existing PTQ FakeQuantLinear
integer-buffer format for inference/evaluation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fakequant import FakeQuantLinear
from .policy import layer_policy_entries


@dataclass(frozen=True)
class QATConversionSummary:
    converted: int
    skipped_fp16: int
    mode_counts: dict[str, int]
    activation_qdq_mode: str


def _ste_identity(qdq: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return QDQ value in forward and identity gradient w.r.t. x."""

    return x + (qdq - x).detach()


def quant_dequant_weight_ste(weight: torch.Tensor, weight_mode: str = "w8") -> torch.Tensor:
    """Symmetric per-output-channel weight QDQ with straight-through gradients."""

    if weight_mode == "w8":
        qmax = 127.0
    elif weight_mode == "w4":
        qmax = 7.0
    else:
        raise ValueError(f"Unsupported QAT weight_mode: {weight_mode}")
    w_float = weight.to(torch.float32)
    scale = (torch.amax(torch.abs(w_float), dim=1, keepdim=True) / qmax).clamp(min=1e-6)
    q = torch.clamp(torch.round(w_float / scale), -qmax, qmax)
    qdq = q * scale
    return _ste_identity(qdq, w_float).to(dtype=weight.dtype)


def quant_dequant_activation_ste(
    x: torch.Tensor,
    activation_mode: str = "a8",
    activation_qdq_mode: str = "dynamic_asymmetric",
    draq_qrange: str = "signed_symmetric",
    act_scale: torch.Tensor | None = None,
    act_zero_point: torch.Tensor | None = None,
    in_features: int | None = None,
) -> torch.Tensor:
    """Activation QDQ with STE, matching FakeQuantLinear runtime semantics."""

    if activation_mode == "a16":
        return x
    if activation_mode != "a8":
        raise ValueError(f"Unsupported QAT activation_mode: {activation_mode}")

    x_float = x.to(torch.float32)
    if activation_qdq_mode == "dynamic_symmetric":
        scale = torch.amax(torch.abs(x_float), dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
        qdq = torch.clamp(torch.round(x_float / scale), -127, 127) * scale
    elif activation_qdq_mode == "draq_symmetric":
        if draq_qrange == "signed_full":
            qmin, qmax = -128.0, 127.0
        elif draq_qrange == "signed_symmetric":
            qmin, qmax = -127.0, 127.0
        else:
            raise ValueError(f"Unsupported draq_qrange: {draq_qrange}")
        reduce_channel = tuple(range(x_float.dim() - 1))
        s = torch.amax(torch.abs(x_float), dim=reduce_channel, keepdim=True).clamp(min=1e-6)
        x_norm = x_float / s
        d = torch.amax(torch.abs(x_norm), dim=-1, keepdim=True).clamp(min=1e-6)
        q = torch.clamp(torch.round(qmax * x_norm / d), qmin, qmax)
        qdq = (q / qmax) * d * s
    elif activation_qdq_mode == "dynamic_asymmetric":
        qmin, qmax = -128.0, 127.0
        x_min = torch.amin(x_float, dim=-1, keepdim=True)
        x_max = torch.amax(x_float, dim=-1, keepdim=True)
        scale = ((x_max - x_min) / (qmax - qmin)).clamp(min=1e-6)
        zero_point = torch.round(qmin - x_min / scale).clamp(qmin, qmax)
        q = torch.clamp(torch.round(x_float / scale + zero_point), qmin, qmax)
        qdq = (q - zero_point) * scale
    elif activation_qdq_mode == "static_asymmetric":
        if act_scale is None or act_zero_point is None or in_features is None:
            raise RuntimeError("static_asymmetric QAT requires act_scale/act_zero_point and in_features")
        scale = act_scale.to(device=x.device, dtype=torch.float32).reshape(
            *([1] * (x.dim() - 1)), in_features
        ).clamp(min=1e-6)
        zero_point = act_zero_point.to(device=x.device, dtype=torch.float32).reshape(
            *([1] * (x.dim() - 1)), in_features
        )
        q = torch.clamp(torch.round(x_float / scale + zero_point), -128, 127)
        qdq = (q - zero_point) * scale
    else:
        raise ValueError(f"Unsupported activation_qdq_mode: {activation_qdq_mode}")
    return _ste_identity(qdq, x_float).to(dtype=x.dtype)


class QuantAwareLinear(nn.Module):
    """Trainable Linear with STE fake-quant activations/weights.

    Unlike `FakeQuantLinear`, this module keeps FP trainable `weight`/`bias`
    parameters during QAT.  Use `to_fakequant_linear()` after fine-tuning to
    create an inference-format `FakeQuantLinear` with integer weight buffers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_mode: str = "a8",
        weight_mode: str = "w8",
        activation_qdq_mode: str = "dynamic_asymmetric",
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if activation_mode not in ("a8", "a16"):
            raise ValueError(f"Unsupported activation_mode: {activation_mode}")
        if weight_mode not in ("w8", "w4"):
            raise ValueError(f"Unsupported weight_mode: {weight_mode}")
        if activation_qdq_mode not in ("static_asymmetric", "dynamic_symmetric", "dynamic_asymmetric", "draq_symmetric"):
            raise ValueError(f"Unsupported activation_qdq_mode: {activation_qdq_mode}")

        self.in_features = in_features
        self.out_features = out_features
        self.activation_mode = activation_mode
        self.weight_mode = weight_mode
        self.activation_qdq_mode_name = activation_qdq_mode
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        self.register_buffer("act_quant_enabled", torch.tensor(True, dtype=torch.bool, device=device))
        if activation_mode == "a8":
            self.register_buffer("act_scale", torch.ones(1, 1, in_features, dtype=torch.float32, device=device))
            self.register_buffer("act_zero_point", torch.zeros(1, 1, in_features, dtype=torch.int32, device=device))
            self.register_buffer("observer_min", torch.zeros(1, 1, in_features, dtype=torch.float32, device=device))
            self.register_buffer("observer_max", torch.zeros(1, 1, in_features, dtype=torch.float32, device=device))
        else:
            self.register_buffer("act_scale", None)
            self.register_buffer("act_zero_point", None)
            self.register_buffer("observer_min", None)
            self.register_buffer("observer_max", None)
        self.register_buffer("observer_enabled", torch.tensor(False, dtype=torch.bool, device=device))
        self.register_buffer("observer_initialized", torch.tensor(False, dtype=torch.bool, device=device))
        self.register_buffer("static_qparams_frozen", torch.tensor(False, dtype=torch.bool, device=device))
        self.register_buffer("observer_ema_decay", torch.tensor(0.95, dtype=torch.float32, device=device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = fan_in ** -0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_float(
        cls,
        linear_module: nn.Linear,
        activation_mode: str = "a8",
        weight_mode: str = "w8",
        activation_qdq_mode: str = "dynamic_asymmetric",
        act_scale: torch.Tensor | None = None,
        act_zero_point: torch.Tensor | None = None,
    ) -> "QuantAwareLinear":
        if not isinstance(linear_module, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(linear_module)}")
        qat = cls(
            linear_module.in_features,
            linear_module.out_features,
            activation_mode=activation_mode,
            weight_mode=weight_mode,
            activation_qdq_mode=activation_qdq_mode,
            bias=linear_module.bias is not None,
            device=linear_module.weight.device,
            dtype=linear_module.weight.dtype,
        )
        qat.weight.data.copy_(linear_module.weight.data)
        if linear_module.bias is not None:
            assert qat.bias is not None
            qat.bias.data.copy_(linear_module.bias.data)
        qat.set_activation_scales(act_scale, act_zero_point)
        return qat

    def set_activation_scales(self, scale: torch.Tensor | None, zero_point: torch.Tensor | None = None) -> None:
        if self.act_scale is None or scale is None:
            return
        scale = scale.detach().to(device=self.act_scale.device, dtype=torch.float32)
        if scale.numel() == 1:
            self.act_scale.copy_(scale.reshape(1, 1, 1).expand_as(self.act_scale))
        elif scale.dim() == 1:
            self.act_scale.copy_(scale.view(1, 1, -1))
        else:
            self.act_scale.copy_(scale)
        if zero_point is not None:
            zp = zero_point.detach().to(device=self.act_zero_point.device, dtype=torch.int32)
            if zp.numel() == 1:
                self.act_zero_point.copy_(zp.reshape(1, 1, 1).expand_as(self.act_zero_point))
            elif zp.dim() == 1:
                self.act_zero_point.copy_(zp.view(1, 1, -1))
            else:
                self.act_zero_point.copy_(zp)

    def enable_observer(self, enabled: bool = True, ema_decay: float = 0.95) -> None:
        """Enable/disable EMA activation min/max collection for static QAT."""

        self.observer_enabled.fill_(bool(enabled))
        self.observer_ema_decay.fill_(float(ema_decay))

    def update_activation_observer(self, x: torch.Tensor) -> None:
        """Update per-input-channel EMA min/max over all dimensions except feature."""

        if self.activation_mode != "a8" or self.observer_min is None or self.observer_max is None:
            return
        with torch.no_grad():
            x_float = x.detach().to(torch.float32)
            reduce_dims = tuple(range(x_float.dim() - 1))
            cur_min = torch.amin(x_float, dim=reduce_dims).reshape(1, 1, self.in_features)
            cur_max = torch.amax(x_float, dim=reduce_dims).reshape(1, 1, self.in_features)
            if not bool(self.observer_initialized.item()):
                self.observer_min.copy_(cur_min.to(device=self.observer_min.device))
                self.observer_max.copy_(cur_max.to(device=self.observer_max.device))
                self.observer_initialized.fill_(True)
                return
            decay = float(self.observer_ema_decay.item())
            self.observer_min.mul_(decay).add_(cur_min.to(device=self.observer_min.device), alpha=1.0 - decay)
            self.observer_max.mul_(decay).add_(cur_max.to(device=self.observer_max.device), alpha=1.0 - decay)

    def freeze_activation_qparams(self) -> None:
        """Freeze static asymmetric activation scale/zero-point from observer stats."""

        if self.activation_mode != "a8" or self.act_scale is None or self.act_zero_point is None:
            self.observer_enabled.fill_(False)
            self.static_qparams_frozen.fill_(True)
            return
        if not bool(self.observer_initialized.item()):
            raise RuntimeError("Cannot freeze activation qparams before observer has collected stats")
        qmin, qmax = -128.0, 127.0
        obs_min = self.observer_min.to(device=self.act_scale.device, dtype=torch.float32)
        obs_max = self.observer_max.to(device=self.act_scale.device, dtype=torch.float32)
        scale = ((obs_max - obs_min) / (qmax - qmin)).clamp(min=1e-6)
        zero_point = torch.round(qmin - obs_min / scale).clamp(qmin, qmax).to(torch.int32)
        self.act_scale.copy_(scale)
        self.act_zero_point.copy_(zero_point)
        self.observer_enabled.fill_(False)
        self.static_qparams_frozen.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if bool(self.observer_enabled.item()) and self.activation_qdq_mode_name == "static_asymmetric":
            self.update_activation_observer(x)
        if bool(self.act_quant_enabled.item()):
            x_qdq = quant_dequant_activation_ste(
                x,
                activation_mode=self.activation_mode,
                activation_qdq_mode=self.activation_qdq_mode_name,
                act_scale=self.act_scale,
                act_zero_point=self.act_zero_point,
                in_features=self.in_features,
            )
        else:
            x_qdq = x
        w_qdq = quant_dequant_weight_ste(self.weight, self.weight_mode)
        return F.linear(x_qdq, w_qdq.to(dtype=x_qdq.dtype), self.bias.to(dtype=x_qdq.dtype) if self.bias is not None else None)

    def to_float_linear(self) -> nn.Linear:
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        linear.weight.data.copy_(self.weight.data)
        if self.bias is not None:
            linear.bias.data.copy_(self.bias.data)
        return linear

    def to_fakequant_linear(self) -> FakeQuantLinear:
        return FakeQuantLinear.from_float(
            self.to_float_linear(),
            activation_mode=self.activation_mode,
            weight_mode=self.weight_mode,
            act_scale=self.act_scale.detach() if self.act_scale is not None else None,
            act_zero_point=self.act_zero_point.detach() if self.act_zero_point is not None else None,
            act_quant_enabled=bool(self.act_quant_enabled.item()),
            activation_qdq_mode=self.activation_qdq_mode_name,
        )


def _get_parent_and_name(model: nn.Module, full_name: str) -> tuple[nn.Module, str]:
    parts = full_name.rsplit(".", 1)
    if len(parts) == 1:
        return model, parts[0]
    parent = model
    for part in parts[0].split("."):
        parent = getattr(parent, part)
    return parent, parts[1]


def _split_mode(mode: str) -> tuple[str, str]:
    if mode.startswith("a16"):
        return "a16", mode[3:]
    if mode.startswith("a8"):
        return "a8", mode[2:]
    raise ValueError(f"Unsupported QAT mode: {mode}")


def convert_model_to_qat(
    model: nn.Module,
    mode: str = "a8w8",
    act_stats: dict[str, Any] | None = None,
    activation_qdq_mode: str = "dynamic_asymmetric",
    layer_policy: dict[str, Any] | None = None,
) -> nn.Module:
    """Recursively replace nn.Linear with trainable QuantAwareLinear."""

    policy = layer_policy_entries(layer_policy) if layer_policy else {}
    converted = 0
    skipped_fp16 = 0
    mode_counts: dict[str, int] = {}
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        entry = policy.get(full_name, {})
        layer_mode = entry.get("mode", mode) if isinstance(entry, dict) else (entry or mode)
        layer_qdq_mode = entry.get("activation_qdq_mode", activation_qdq_mode) if isinstance(entry, dict) else activation_qdq_mode
        if layer_mode == "fp16_skip":
            skipped_fp16 += 1
            mode_counts[layer_mode] = mode_counts.get(layer_mode, 0) + 1
            continue
        activation_mode, weight_mode = _split_mode(layer_mode)
        stats = (act_stats or {}).get(full_name, {})
        act_scale = stats.get("act_scale", stats.get("scale")) if isinstance(stats, dict) else None
        act_zp = stats.get("zero_point") if isinstance(stats, dict) else None
        new_mod = QuantAwareLinear.from_float(
            module,
            activation_mode=activation_mode,
            weight_mode=weight_mode,
            activation_qdq_mode=layer_qdq_mode,
            act_scale=act_scale,
            act_zero_point=act_zp,
        )
        parent, leaf = _get_parent_and_name(model, full_name)
        setattr(parent, leaf, new_mod)
        converted += 1
        mode_counts[layer_mode] = mode_counts.get(layer_mode, 0) + 1
    model._qat_conversion_summary = QATConversionSummary(
        converted=converted,
        skipped_fp16=skipped_fp16,
        mode_counts=mode_counts,
        activation_qdq_mode=activation_qdq_mode,
    ).__dict__
    return model


def export_qat_model_to_fakequant(model: nn.Module, inplace: bool = False) -> nn.Module:
    """Replace every QuantAwareLinear with inference-format FakeQuantLinear."""

    out = model if inplace else deepcopy(model)
    exported = 0
    for full_name, module in list(out.named_modules()):
        if not isinstance(module, QuantAwareLinear):
            continue
        parent, leaf = _get_parent_and_name(out, full_name)
        setattr(parent, leaf, module.to_fakequant_linear())
        exported += 1
    out._qat_export_summary = {"exported": exported, "format": "FakeQuantLinear"}
    return out


def set_qat_activation_quant(model: nn.Module, enabled: bool) -> None:
    """Enable/disable activation fake quant for all QAT Linear layers."""

    for module in model.modules():
        if isinstance(module, QuantAwareLinear):
            module.act_quant_enabled.fill_(bool(enabled))


def set_qat_observer(model: nn.Module, enabled: bool, ema_decay: float = 0.95) -> None:
    """Enable/disable static activation observers for all QAT Linear layers."""

    for module in model.modules():
        if isinstance(module, QuantAwareLinear):
            module.enable_observer(enabled, ema_decay=ema_decay)


def freeze_qat_observers(model: nn.Module) -> dict[str, int]:
    """Freeze all initialized static QAT observers and return a summary."""

    summary = {"frozen": 0, "skipped_uninitialized": 0, "non_qat": 0}
    for module in model.modules():
        if not isinstance(module, QuantAwareLinear):
            continue
        if module.activation_mode != "a8":
            summary["non_qat"] += 1
            module.enable_observer(False)
            continue
        if not bool(module.observer_initialized.item()):
            summary["skipped_uninitialized"] += 1
            module.enable_observer(False)
            continue
        module.freeze_activation_qparams()
        summary["frozen"] += 1
    return summary


def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """In-place EMA update for QAT student weights/buffers."""

    with torch.no_grad():
        ema_state = ema_model.state_dict()
        src_state = model.state_dict()
        for key, ema_value in ema_state.items():
            src_value = src_state[key].detach().to(device=ema_value.device)
            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(src_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(src_value)


def tensor_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """PSNR helper for comparing FP16 teacher and QAT outputs."""

    mse = F.mse_loss(pred.float(), target.float())
    return 20.0 * torch.log10(torch.tensor(float(data_range), device=pred.device)) - 10.0 * torch.log10(mse.clamp(min=eps))


def temporal_consistency_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Match teacher frame-to-frame deltas to reduce flicker."""

    if student.dim() < 3 or student.shape[2] < 2:
        return student.new_zeros(())
    s_delta = student[:, :, 1:] - student[:, :, :-1]
    t_delta = teacher[:, :, 1:] - teacher[:, :, :-1]
    return F.mse_loss(s_delta.float(), t_delta.float())
