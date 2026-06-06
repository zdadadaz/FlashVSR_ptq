"""QAO helpers for LSGQuant low-rank residual PTQ.

This module implements the CPU, identity-rotation version of LSGQuant's QAO
(Quantization-Aware Optimization) decomposition for a single Linear weight
matrix.  It is intentionally layer-local and deterministic so conversion can run
layer-by-layer without requiring a GPU-resident DiT.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .lsgquant import LSGQuantLinear


def _next_power_of_two(n: int) -> int:
    if n <= 0:
        raise ValueError(f"feature dimension must be positive, got {n}")
    return 1 << (n - 1).bit_length()


def normalized_hadamard_matrix(size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return an orthonormal Hadamard matrix for power-of-two ``size``."""

    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a positive power of two, got {size}")
    h = torch.ones(1, 1, device=device, dtype=torch.float32)
    while h.shape[0] < size:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)],
            dim=0,
        )
    h = h / (float(size) ** 0.5)
    return h.to(dtype=dtype)


def apply_hadamard_rotation(x: torch.Tensor) -> torch.Tensor:
    """Apply normalized Hadamard rotation on the last dimension.

    Non-power-of-two feature dimensions are zero-padded to the next power of two,
    rotated, then cropped back. This keeps tensor shapes stable for debug/runtime
    correctness even though exact orthonormality only holds in the padded space.
    """

    features = x.shape[-1]
    padded_features = _next_power_of_two(features)
    x_float = x.to(torch.float32)
    if padded_features != features:
        x_float = torch.nn.functional.pad(x_float, (0, padded_features - features))
    h = normalized_hadamard_matrix(padded_features, device=x.device, dtype=torch.float32)
    rotated = torch.matmul(x_float, h.t())
    return rotated[..., :features].to(dtype=x.dtype)


def rotate_linear_weight(weight: torch.Tensor, rotation: str = "identity") -> torch.Tensor:
    """Rotate a Linear weight matrix on its input-feature dimension."""

    if rotation == "identity":
        return weight.detach().to(torch.float32)
    if rotation != "hadamard":
        raise ValueError("rotation must be 'identity' or 'hadamard'")
    out_features, in_features = weight.shape
    padded_features = _next_power_of_two(in_features)
    w = weight.detach().to(torch.float32)
    if padded_features != in_features:
        w = torch.nn.functional.pad(w, (0, padded_features - in_features))
    h = normalized_hadamard_matrix(padded_features, device=w.device, dtype=torch.float32)
    return (w @ h)[..., :in_features].contiguous()


@dataclass(frozen=True)
class QAOResult:
    """Result of decomposing ``weight`` into quantized residual + low-rank FP."""

    residual_int: torch.Tensor
    residual_scale: torch.Tensor
    l1: torch.Tensor
    l2: torch.Tensor
    error_before: float
    error_after: float
    weight_bits: int
    rank: int
    rounds: int
    rotation: str = "identity"


def _qmax(bits: int) -> int:
    if bits == 8:
        return 127
    if bits == 4:
        return 7
    raise ValueError(f"Unsupported weight_bits={bits}; expected 8 or 4")


def quantize_weight_symmetric(weight: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric quantization for Linear weights.

    Returns an int8 tensor containing either signed int8 values (W8) or unpacked
    signed int4 values in the range [-7, 7] (W4), plus a per-row scale [out, 1].
    Packing remains the responsibility of FakeQuantLinear conversion/runtime.
    """

    if weight.dim() != 2:
        raise ValueError(f"Expected 2D Linear weight [out, in], got shape {tuple(weight.shape)}")
    qmax = _qmax(bits)
    w = weight.detach().to(torch.float32)
    scale = (torch.amax(torch.abs(w), dim=1, keepdim=True) / float(qmax)).clamp(min=1e-6)
    q = torch.clamp(torch.round(w / scale), -qmax, qmax).to(torch.int8)
    return q, scale.to(torch.float32)


def dequantize_weight(weight_int: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a symmetric per-row quantized weight tensor."""

    return weight_int.to(torch.float32) * scale.to(torch.float32)


def _truncated_svd(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return L1 [out, rank], L2 [rank, in] for best rank-k approximation."""

    out_features, in_features = matrix.shape
    if rank <= 0:
        return (
            torch.zeros(out_features, 0, dtype=torch.float32),
            torch.zeros(0, in_features, dtype=torch.float32),
        )
    k = min(rank, out_features, in_features)
    u, s, vh = torch.linalg.svd(matrix.to(torch.float32), full_matrices=False)
    sqrt_s = torch.sqrt(s[:k].clamp(min=0))
    l1 = u[:, :k] * sqrt_s.unsqueeze(0)
    l2 = sqrt_s.unsqueeze(1) * vh[:k, :]
    if k == rank:
        return l1.contiguous(), l2.contiguous()
    # Preserve requested buffer shapes when rank exceeds matrix rank.
    l1_pad = torch.zeros(out_features, rank, dtype=torch.float32)
    l2_pad = torch.zeros(rank, in_features, dtype=torch.float32)
    l1_pad[:, :k] = l1
    l2_pad[:k, :] = l2
    return l1_pad, l2_pad


def _reconstruct(residual_int: torch.Tensor, residual_scale: torch.Tensor, l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
    return dequantize_weight(residual_int, residual_scale) + l1.to(torch.float32) @ l2.to(torch.float32)


def qao_decompose_weight(
    weight: torch.Tensor,
    weight_bits: int,
    rank: int = 32,
    rounds: int = 4,
    rotation: str = "identity",
) -> QAOResult:
    """Decompose a Linear weight into quantized residual and FP low-rank branch.

    Algorithm: start from a truncated SVD of the full weight, quantize the
    residual, then iteratively recompute the low-rank approximation of the
    remaining error after dequantizing the residual. The best Frobenius-error
    candidate across iterations is returned.
    """

    if rotation not in ("identity", "hadamard"):
        raise ValueError("rotation must be 'identity' or 'hadamard'")
    if rank < 0:
        raise ValueError(f"rank must be >= 0, got {rank}")
    if rounds < 0:
        raise ValueError(f"rounds must be >= 0, got {rounds}")
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D Linear weight [out, in], got shape {tuple(weight.shape)}")

    w = rotate_linear_weight(weight.detach(), rotation=rotation).to(device="cpu", dtype=torch.float32)
    pure_int, pure_scale = quantize_weight_symmetric(w, bits=weight_bits)
    pure_error = torch.linalg.vector_norm(w - dequantize_weight(pure_int, pure_scale)).item()

    l1, l2 = _truncated_svd(w, rank)
    residual_int, residual_scale = quantize_weight_symmetric(w - l1 @ l2, bits=weight_bits)
    best_error = torch.linalg.vector_norm(w - _reconstruct(residual_int, residual_scale, l1, l2)).item()
    best = (best_error, residual_int, residual_scale, l1, l2)

    for _ in range(rounds):
        residual_deq = dequantize_weight(residual_int, residual_scale)
        l1, l2 = _truncated_svd(w - residual_deq, rank)
        residual_int, residual_scale = quantize_weight_symmetric(w - l1 @ l2, bits=weight_bits)
        current_error = torch.linalg.vector_norm(w - _reconstruct(residual_int, residual_scale, l1, l2)).item()
        if current_error < best[0]:
            best = (current_error, residual_int, residual_scale, l1, l2)

    error_after, residual_int, residual_scale, l1, l2 = best
    return QAOResult(
        residual_int=residual_int.contiguous(),
        residual_scale=residual_scale.contiguous(),
        l1=l1.contiguous(),
        l2=l2.contiguous(),
        error_before=float(pure_error),
        error_after=float(error_after),
        weight_bits=int(weight_bits),
        rank=int(rank),
        rounds=int(rounds),
        rotation=rotation,
    )


def _copy_qao_residual_to_lsgquant(module: LSGQuantLinear, result: QAOResult) -> None:
    """Copy QAO residual buffers into an ``LSGQuantLinear`` residual branch."""

    residual = module.residual
    residual.weight_scale.copy_(result.residual_scale.to(device=residual.weight_scale.device))
    if residual.weight_mode == "w4":
        unpacked = result.residual_int.to(device=residual.weight_int.device, dtype=torch.int8)
        lo = unpacked[:, 0::2].contiguous()
        hi = unpacked[:, 1::2].contiguous()
        if hi.shape[1] < lo.shape[1]:
            hi = torch.nn.functional.pad(hi, (0, lo.shape[1] - hi.shape[1]))
        packed = (lo & 0x0F) | ((hi & 0x0F) << 4)
        residual.weight_int.copy_(packed)
    else:
        residual.weight_int.copy_(result.residual_int.to(device=residual.weight_int.device, dtype=torch.int8))


def qao_linear_from_float(
    linear_module: nn.Linear,
    weight_bits: int,
    rank: int = 32,
    rounds: int = 4,
    rotation: str = "identity",
    activation_mode: str = "a16",
    act_scale: torch.Tensor | None = None,
    act_zero_point: torch.Tensor | None = None,
    act_quant_enabled: bool = True,
    activation_qdq_mode: str = "static_asymmetric",
    draq_qrange: str = "signed_symmetric",
) -> tuple[LSGQuantLinear, QAOResult]:
    """Convert one ``nn.Linear`` into ``LSGQuantLinear`` using QAO residuals."""

    if not isinstance(linear_module, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(linear_module)}")
    weight_mode = {8: "w8", 4: "w4"}.get(weight_bits)
    if weight_mode is None:
        raise ValueError(f"Unsupported weight_bits={weight_bits}; expected 8 or 4")

    result = qao_decompose_weight(
        linear_module.weight.detach(),
        weight_bits=weight_bits,
        rank=rank,
        rounds=rounds,
        rotation=rotation,
    )
    module = LSGQuantLinear.from_float(
        linear_module,
        rank=result.rank,
        activation_mode=activation_mode,
        weight_mode=weight_mode,
        act_scale=act_scale,
        act_zero_point=act_zero_point,
        act_quant_enabled=act_quant_enabled,
        activation_qdq_mode=activation_qdq_mode,
        draq_qrange=draq_qrange,
        rotation=rotation,
        low_rank_l1=result.l1.to(device=linear_module.weight.device, dtype=linear_module.weight.dtype),
        low_rank_l2=result.l2.to(device=linear_module.weight.device, dtype=linear_module.weight.dtype),
    )
    _copy_qao_residual_to_lsgquant(module, result)
    return module, result


def _mode_to_activation_and_weight_bits(mode: str) -> tuple[str, int]:
    mapping = {
        "a8w8": ("a8", 8),
        "a16w8": ("a16", 8),
        "a8w4": ("a8", 4),
        "a16w4": ("a16", 4),
        "a4w4": ("a4", 4),
    }
    try:
        return mapping[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode={mode}; expected one of {sorted(mapping)}") from exc


def convert_model_to_lsgquant_qao(
    model: nn.Module,
    mode: str = "a8w8",
    rank: int = 32,
    rounds: int = 4,
    rotation: str = "identity",
    act_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    layer_policy: dict[str, dict] | None = None,
    activation_qdq_mode: str = "static_asymmetric",
    draq_qrange: str = "signed_symmetric",
    max_layers: int | None = None,
) -> list[dict]:
    """Replace eligible ``nn.Linear`` modules with QAO-backed LSGQuantLinear.

    Returns per-layer manifest entries with Frobenius errors and conversion time.
    """

    import time

    default_activation_mode, default_weight_bits = _mode_to_activation_and_weight_bits(mode)
    layer_errors: list[dict] = []

    def convert_children(parent: nn.Module, prefix: str = "") -> None:
        nonlocal layer_errors
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if max_layers is not None and len(layer_errors) >= max_layers:
                return
            if isinstance(child, nn.Linear):
                policy = (layer_policy or {}).get(full_name, {})
                layer_mode = policy.get("mode", mode)
                if layer_mode == "fp16_skip":
                    continue
                activation_mode, weight_bits = _mode_to_activation_and_weight_bits(layer_mode)
                stats = (act_stats or {}).get(full_name, {})
                start = time.perf_counter()
                layer_rank = int(policy.get("rank", rank))
                replacement, result = qao_linear_from_float(
                    child,
                    weight_bits=weight_bits,
                    rank=layer_rank,
                    rounds=rounds,
                    rotation=rotation,
                    activation_mode=activation_mode,
                    act_scale=stats.get("act_scale"),
                    act_zero_point=stats.get("zero_point"),
                    act_quant_enabled=bool(policy.get("act_quant_enabled", True)),
                    activation_qdq_mode=policy.get("activation_qdq_mode", activation_qdq_mode),
                    draq_qrange=policy.get("draq_qrange", draq_qrange),
                )
                setattr(parent, child_name, replacement)
                layer_errors.append(
                    {
                        "name": full_name,
                        "rank": int(result.rank),
                        "mode": layer_mode,
                        "activation_qdq_mode": policy.get("activation_qdq_mode", activation_qdq_mode),
                        "weight_bits": int(result.weight_bits),
                        "error_before": float(result.error_before),
                        "error_after": float(result.error_after),
                        "time_sec": time.perf_counter() - start,
                    }
                )
            else:
                convert_children(child, full_name)

    convert_children(model)
    model._lsgquant_qao_conversion_summary = {
        "mode": mode,
        "rank": rank,
        "rounds": rounds,
        "rotation": rotation,
        "layers": len(layer_errors),
    }
    return layer_errors
