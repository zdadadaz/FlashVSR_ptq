"""Learned Equivalent Scaling (LES) for FlashVSR DiT PTQ.

Optimizes a per-input-channel tau so the quantized equivalent expression
Q(X / tau) @ Q(tau * W)^T reconstructs the FP linear output. The resulting tau
is stored in the SmoothQuant-compatible cache format consumed by
fakequant_convert.py --smoothquant_cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
        q = torch.clamp(torch.round(x / scale), qmin, qmax)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None, None


def _per_token_act_scale(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = float((1 << (bits - 1)) - 1)
    # Dynamic symmetric per token over channel dim. Keep dims for broadcast.
    return torch.amax(torch.abs(x), dim=-1, keepdim=True).clamp(min=1e-6) / qmax


def _per_out_weight_scale(w: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = float((1 << (bits - 1)) - 1)
    return torch.amax(torch.abs(w), dim=1, keepdim=True).clamp(min=1e-6) / qmax


def _qdq_activation(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    return STEQuantize.apply(x, _per_token_act_scale(x, bits), -qmax, qmax)


def _qdq_weight(w: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    return STEQuantize.apply(w, _per_out_weight_scale(w, bits), -qmax, qmax)


def smoothquant_tau_init(linear: nn.Linear, x: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """SmoothQuant-style deterministic tau init, clamped for stability."""
    x_flat = x.detach().to(torch.float32).reshape(-1, x.shape[-1])
    act_amax = torch.amax(torch.abs(x_flat), dim=0).clamp(min=1e-6)
    weight_amax = torch.amax(torch.abs(linear.weight.detach().to(torch.float32)), dim=0).clamp(min=1e-6)
    tau = act_amax.pow(alpha) / weight_amax.pow(1.0 - alpha)
    return tau.clamp(1e-4, 1e4)


def les_reconstruction_loss(linear: nn.Linear, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    x_fp = x.to(torch.float32)
    w_fp = linear.weight.to(torch.float32)
    bias = linear.bias.to(torch.float32) if linear.bias is not None else None
    y_ref = F.linear(x_fp, w_fp, bias)
    tau_safe = tau.clamp(1e-4, 1e4)
    x_q = _qdq_activation(x_fp / tau_safe)
    w_q = _qdq_weight(w_fp * tau_safe.unsqueeze(0))
    y_q = F.linear(x_q, w_q, bias)
    return F.mse_loss(y_q, y_ref)


def optimize_layer_tau(
    linear: nn.Linear,
    x: torch.Tensor,
    num_steps: int = 300,
    lr: float = 1e-3,
    alpha: float = 0.5,
    clamp_min: float = 1e-4,
    clamp_max: float = 1e4,
) -> dict[str, Any]:
    """Optimize one layer's tau. CPU/GPU agnostic and deterministic for tests."""
    linear = linear.to(device=x.device).eval()
    tau0 = smoothquant_tau_init(linear, x, alpha=alpha).to(device=x.device)
    log_tau = torch.nn.Parameter(torch.log(tau0.clamp(clamp_min, clamp_max)))
    opt = torch.optim.Adam([log_tau], lr=lr)
    with torch.no_grad():
        initial_loss = float(les_reconstruction_loss(linear, x, torch.exp(log_tau)).item())
    best_tau = torch.exp(log_tau.detach()).clone()
    best_loss = initial_loss
    for _ in range(max(int(num_steps), 0)):
        tau = torch.exp(log_tau).clamp(clamp_min, clamp_max)
        loss = les_reconstruction_loss(linear, x, tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            log_tau.clamp_(min=float(torch.log(torch.tensor(clamp_min))), max=float(torch.log(torch.tensor(clamp_max))))
            current = float(loss.item())
            if current <= best_loss:
                best_loss = current
                best_tau = torch.exp(log_tau.detach()).clamp(clamp_min, clamp_max).clone()
    return {"tau": best_tau.detach().cpu(), "initial_loss": initial_loss, "final_loss": best_loss}


def load_les_cache(path: str | Path) -> dict[str, torch.Tensor]:
    raw = json.loads(Path(path).read_text())
    out: dict[str, torch.Tensor] = {}
    for name, entry in raw.items():
        if name.startswith("_"):
            continue
        value = entry.get("tau", entry.get("smoothquant_scale", entry.get("scale"))) if isinstance(entry, dict) else entry
        if value is not None:
            out[name] = torch.tensor(value, dtype=torch.float32)
    return out


def save_les_cache(results: dict[str, dict[str, Any]], output: str | Path, metadata: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"_metadata": {"schema_version": "flashvsr.learned_equivalent_scaling.v1", **(metadata or {})}}
    for name, item in results.items():
        tau = item["tau"]
        if isinstance(tau, torch.Tensor):
            tau = tau.detach().cpu().tolist()
        payload[name] = {
            "tau": [float(x) for x in tau],
            "smoothquant_scale": [float(x) for x in tau],
            "initial_loss": float(item.get("initial_loss", 0.0)),
            "final_loss": float(item.get("final_loss", 0.0)),
        }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Learned Equivalent Scaling tau optimizer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dry_run", action="store_true", help="Write a tiny synthetic LES cache for CLI smoke tests")
    # Full-DiT flags retained for the master schedule; heavy GPU capture is intentionally delegated to future E2E runs.
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--calibration_cache", default="")
    args = parser.parse_args()
    if args.dry_run:
        layer = nn.Linear(4, 3, bias=False)
        x = torch.randn(8, 4)
        save_les_cache({"dry_run.linear": optimize_layer_tau(layer, x, num_steps=min(args.num_steps, 5), lr=args.lr)}, args.output, {"dry_run": True})
        print(f"[LES] dry-run cache -> {args.output}")
        return
    raise SystemExit("Full DiT LES capture is not a smoke command; collect layer inputs first or use --dry_run for validation.")


if __name__ == "__main__":
    main()
