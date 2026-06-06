"""
LSGQuant low-rank residual linear layer for FlashVSR DiT PTQ.

The module represents a linear projection as a quantized residual branch plus an
optional floating-point low-rank branch:

    y = linear(qdq(x), W_residual_qdq, bias) + linear(linear(x, L2), L1)

Hadamard rotation and QAO decomposition are intentionally left to later PRs; this
PR only provides the runtime/state_dict container used by those conversion paths.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fakequant import FakeQuantLinear


class LSGQuantLinear(nn.Module):
    """Quantized residual + floating low-rank linear module.

    Args mirror :class:`FakeQuantLinear` for the quantized residual branch.
    ``l2_weight`` has shape ``[rank, in_features]`` and ``l1_weight`` has shape
    ``[out_features, rank]``, matching PyTorch ``F.linear`` weight layout.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 0,
        activation_mode: str = "a16",
        weight_mode: str = "w8",
        act_quant_enabled: bool = True,
        activation_qdq_mode: str = "static_asymmetric",
        draq_qrange: str = "signed_symmetric",
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if rank < 0:
            raise ValueError(f"rank must be >= 0, got {rank}")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = int(rank)
        low_rank_dtype = dtype or torch.float32

        self.residual = FakeQuantLinear(
            in_features,
            out_features,
            activation_mode=activation_mode,
            weight_mode=weight_mode,
            act_quant_enabled=act_quant_enabled,
            activation_qdq_mode=activation_qdq_mode,
            draq_qrange=draq_qrange,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.register_buffer(
            "l1_weight",
            torch.zeros(out_features, self.rank, dtype=low_rank_dtype, device=device),
        )
        self.register_buffer(
            "l2_weight",
            torch.zeros(self.rank, in_features, dtype=low_rank_dtype, device=device),
        )

    @classmethod
    def from_float(
        cls,
        linear_module: nn.Linear,
        rank: int = 0,
        activation_mode: str = "a16",
        weight_mode: str = "w8",
        act_scale: torch.Tensor = None,
        act_zero_point: torch.Tensor = None,
        act_mean: torch.Tensor = None,
        act_quant_enabled: bool = True,
        activation_qdq_mode: str = "static_asymmetric",
        draq_qrange: str = "signed_symmetric",
        ch_axis: int = -1,
        low_rank_l1: torch.Tensor = None,
        low_rank_l2: torch.Tensor = None,
    ):
        """Create an ``LSGQuantLinear`` from ``nn.Linear``.

        Initial PR5 semantics quantize the full source weight into the residual
        branch.  Later QAO conversion can pass precomputed ``low_rank_l1`` and
        ``low_rank_l2`` buffers after subtracting that low-rank component from
        the residual weight.
        """
        if not isinstance(linear_module, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(linear_module)}")
        if low_rank_l1 is not None or low_rank_l2 is not None:
            if low_rank_l1 is None or low_rank_l2 is None:
                raise ValueError("low_rank_l1 and low_rank_l2 must be provided together")
            inferred_rank = int(low_rank_l2.shape[0])
            if int(low_rank_l1.shape[1]) != inferred_rank:
                raise ValueError("low_rank_l1/low_rank_l2 rank dimensions do not match")
            rank = inferred_rank

        module = cls(
            linear_module.in_features,
            linear_module.out_features,
            rank=rank,
            activation_mode=activation_mode,
            weight_mode=weight_mode,
            act_quant_enabled=act_quant_enabled,
            activation_qdq_mode=activation_qdq_mode,
            draq_qrange=draq_qrange,
            bias=linear_module.bias is not None,
            device=linear_module.weight.device,
            dtype=linear_module.weight.dtype,
        )
        module.residual = FakeQuantLinear.from_float(
            linear_module,
            activation_mode=activation_mode,
            weight_mode=weight_mode,
            act_scale=act_scale,
            act_zero_point=act_zero_point,
            act_mean=act_mean,
            act_quant_enabled=act_quant_enabled,
            activation_qdq_mode=activation_qdq_mode,
            draq_qrange=draq_qrange,
            ch_axis=ch_axis,
        )
        if rank > 0 and low_rank_l1 is not None:
            module.set_low_rank(low_rank_l1, low_rank_l2)
        return module

    def set_low_rank(self, l1_weight: torch.Tensor, l2_weight: torch.Tensor) -> None:
        """Copy low-rank branch weights into the module buffers."""
        if self.rank == 0:
            raise ValueError("Cannot set low-rank weights on rank=0 LSGQuantLinear")
        if tuple(l1_weight.shape) != (self.out_features, self.rank):
            raise ValueError(
                f"l1_weight must have shape {(self.out_features, self.rank)}, got {tuple(l1_weight.shape)}"
            )
        if tuple(l2_weight.shape) != (self.rank, self.in_features):
            raise ValueError(
                f"l2_weight must have shape {(self.rank, self.in_features)}, got {tuple(l2_weight.shape)}"
            )
        self.l1_weight.copy_(l1_weight.to(device=self.l1_weight.device, dtype=self.l1_weight.dtype))
        self.l2_weight.copy_(l2_weight.to(device=self.l2_weight.device, dtype=self.l2_weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        y = self.residual(x).to(torch.float32)
        if self.rank > 0:
            x_fp = x.to(torch.float32)
            l2 = self.l2_weight.to(device=x.device, dtype=torch.float32)
            l1 = self.l1_weight.to(device=x.device, dtype=torch.float32)
            y = y + F.linear(F.linear(x_fp, l2), l1)
        return y.to(orig_dtype)
