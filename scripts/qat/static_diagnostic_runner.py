"""Static QAT diagnostic runner for FlashVSR DiT Linear layers.

Collects per-layer activation ranges and static-QDQ output error/SQNR by
attaching hooks to Linear / QuantAwareLinear modules during a normal DiT forward.
The runner is intended for Phase A of the dynamic-vs-static QAT analysis:
identify the worst static activation-QDQ layers before changing observers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.ptq.fakequant_convert import build_dit, load_calibration_cache, load_checkpoint
from scripts.qat.finetune_fakequant_dit import LatentManifestDataset, collate_batch, move_sample
from src.models.quantization.policy import layer_policy_entries, load_layer_policy
from src.models.quantization.qat import QuantAwareLinear, convert_model_to_qat


@dataclass
class LayerDiagnosticAccumulator:
    name: str
    module_type: str
    activation_mode: str
    weight_mode: str
    activation_qdq_mode: str
    calls: int = 0
    samples: int = 0
    output_mse_sum: float = 0.0
    output_mae_sum: float = 0.0
    signal_power_sum: float = 0.0
    noise_power_sum: float = 0.0
    activation_sum: float = 0.0
    activation_sumsq: float = 0.0
    activation_count: int = 0
    activation_min: float = math.inf
    activation_max: float = -math.inf
    activation_absmax: float = 0.0
    activation_percentiles: dict[str, float] = field(default_factory=dict)

    def update_activation(self, x: torch.Tensor, percentile_keys: tuple[float, ...]) -> None:
        x_float = x.detach().to(torch.float32)
        flat = x_float.reshape(-1)
        if flat.numel() == 0:
            return
        self.activation_count += int(flat.numel())
        self.activation_sum += float(flat.sum().cpu())
        self.activation_sumsq += float(flat.square().sum().cpu())
        self.activation_min = min(self.activation_min, float(flat.min().cpu()))
        self.activation_max = max(self.activation_max, float(flat.max().cpu()))
        self.activation_absmax = max(self.activation_absmax, float(flat.abs().max().cpu()))
        # Percentiles are diagnostic only; computing per-call and taking max keeps
        # memory bounded for full FlashVSR runs.
        abs_flat = flat.abs()
        for p in percentile_keys:
            key = f"p{p:g}_abs"
            value = float(torch.quantile(abs_flat, p / 100.0).cpu())
            self.activation_percentiles[key] = max(self.activation_percentiles.get(key, 0.0), value)

    def update_error(self, ref: torch.Tensor, qdq: torch.Tensor) -> None:
        ref_float = ref.detach().to(torch.float32)
        err = (qdq.detach().to(torch.float32) - ref_float)
        count = int(ref_float.numel())
        if count == 0:
            return
        self.calls += 1
        self.samples += count
        self.output_mse_sum += float(err.square().sum().cpu())
        self.output_mae_sum += float(err.abs().sum().cpu())
        self.signal_power_sum += float(ref_float.square().sum().cpu())
        self.noise_power_sum += float(err.square().sum().cpu())

    def as_row(self) -> dict[str, Any]:
        output_mse = self.output_mse_sum / max(self.samples, 1)
        output_mae = self.output_mae_sum / max(self.samples, 1)
        activation_mean = self.activation_sum / max(self.activation_count, 1)
        activation_var = self.activation_sumsq / max(self.activation_count, 1) - activation_mean * activation_mean
        activation_std = math.sqrt(max(activation_var, 0.0))
        sqnr_db = 10.0 * math.log10((self.signal_power_sum + 1e-12) / (self.noise_power_sum + 1e-12))
        row = {
            "name": self.name,
            "module_type": self.module_type,
            "activation_mode": self.activation_mode,
            "weight_mode": self.weight_mode,
            "activation_qdq_mode": self.activation_qdq_mode,
            "calls": self.calls,
            "samples": self.samples,
            "output_mse": output_mse,
            "output_mae": output_mae,
            "sqnr_db": sqnr_db,
            "activation_min": self.activation_min if self.activation_count else None,
            "activation_max": self.activation_max if self.activation_count else None,
            "activation_absmax": self.activation_absmax,
            "activation_mean": activation_mean,
            "activation_std": activation_std,
        }
        row.update(self.activation_percentiles)
        return row


def _static_activation_qdq(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    if isinstance(module, QuantAwareLinear):
        if module.activation_mode == "a16" or not bool(module.act_quant_enabled.item()):
            return x.detach().to(torch.float32)
        if module.act_scale is None or module.act_zero_point is None:
            return x.detach().to(torch.float32)
        scale = module.act_scale.to(device=x.device, dtype=torch.float32).reshape(
            *([1] * (x.dim() - 1)), module.in_features
        ).clamp(min=1e-6)
        zero_point = module.act_zero_point.to(device=x.device, dtype=torch.float32).reshape(
            *([1] * (x.dim() - 1)), module.in_features
        )
    else:
        return x.detach().to(torch.float32)
    x_float = x.detach().to(torch.float32)
    q = torch.clamp(torch.round(x_float / scale + zero_point), -128.0, 127.0)
    return (q - zero_point) * scale


def _weight_qdq(weight: torch.Tensor, weight_mode: str) -> torch.Tensor:
    qmax = 127.0 if weight_mode == "w8" else 7.0 if weight_mode == "w4" else None
    if qmax is None:
        return weight.detach().to(torch.float32)
    w_float = weight.detach().to(torch.float32)
    scale = (torch.amax(torch.abs(w_float), dim=1, keepdim=True) / qmax).clamp(min=1e-6)
    q = torch.clamp(torch.round(w_float / scale), -qmax, qmax)
    return q * scale


def _module_modes(module: nn.Module) -> tuple[str, str, str]:
    if isinstance(module, QuantAwareLinear):
        return module.activation_mode, module.weight_mode, module.activation_qdq_mode_name
    return "fp", "fp", "none"


def collect_static_linear_diagnostics(
    model: nn.Module,
    samples: list[dict[str, torch.Tensor]],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    local_num: int | None = None,
    percentile_keys: tuple[float, ...] = (99.0, 99.5, 99.9, 99.99),
) -> list[dict[str, Any]]:
    """Run samples through model and return one diagnostic row per Linear layer.

    The model is not modified except for temporary hooks.  For QuantAwareLinear,
    the hook independently compares FP Linear output against static activation QDQ
    plus quantized weight output.  For plain nn.Linear, error is zero unless the
    caller has converted the module to QAT first.
    """

    device = torch.device(device)
    model = model.to(device=device, dtype=dtype).eval()
    accumulators: dict[str, LayerDiagnosticAccumulator] = {}
    handles: list[Any] = []

    def make_hook(name: str, module: nn.Module):
        activation_mode, weight_mode, activation_qdq_mode = _module_modes(module)
        acc = LayerDiagnosticAccumulator(
            name=name,
            module_type=type(module).__name__,
            activation_mode=activation_mode,
            weight_mode=weight_mode,
            activation_qdq_mode=activation_qdq_mode,
        )
        accumulators[name] = acc

        def hook(mod: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            x = inputs[0].detach()
            acc.update_activation(x, percentile_keys)
            if not hasattr(mod, "weight"):
                return
            weight = mod.weight.detach()
            bias = mod.bias.detach() if getattr(mod, "bias", None) is not None else None
            x_float = x.to(torch.float32)
            ref = F.linear(x_float, weight.to(torch.float32), bias.to(torch.float32) if bias is not None else None)
            x_qdq = _static_activation_qdq(x, mod)
            w_qdq = _weight_qdq(weight, weight_mode)
            qdq = F.linear(x_qdq, w_qdq.to(device=x_qdq.device), bias.to(torch.float32) if bias is not None else None)
            acc.update_error(ref, qdq)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, QuantAwareLinear)):
            handles.append(module.register_forward_hook(make_hook(name, module)))

    try:
        with torch.no_grad():
            for sample in samples:
                sample = move_sample(sample, device, dtype)
                if {"x", "timestep", "context"}.issubset(sample):
                    model(
                        sample["x"], sample["timestep"], sample["context"],
                        use_gradient_checkpointing=False,
                        local_num=local_num,
                    )
                elif "input" in sample:
                    model(sample["input"])
                else:
                    raise KeyError("Diagnostic sample must contain x/timestep/context or input")
    finally:
        for handle in handles:
            handle.remove()

    rows = [acc.as_row() for acc in accumulators.values()]
    rows.sort(key=lambda row: (-float(row["output_mse"]), float(row["sqnr_db"])))
    return rows


def write_diagnostic_outputs(rows: list[dict[str, Any]], output_dir: str | Path, top_k: int = 20) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_rows = rows[:top_k]
    json_path = output_dir / "static_qat_linear_diagnostics.json"
    csv_path = output_dir / "static_qat_linear_diagnostics.csv"
    md_path = output_dir / "static_qat_linear_top20.md"

    payload = {
        "schema": "flashvsr.static_qat_linear_diagnostics.v1",
        "linear_count": len(rows),
        "top_k": top_k,
        "top_layers": top_rows,
        "layers": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Static QAT Linear Diagnostics — Top bad layers",
        "",
        f"Linear layers observed: `{len(rows)}`",
        f"Ranking: highest `output_mse`, tie-breaker lowest `sqnr_db`.",
        "",
    ]
    for idx, row in enumerate(top_rows, 1):
        lines.extend([
            f"## {idx}. `{row['name']}`",
            f"- module: `{row['module_type']}`; mode: `{row['activation_mode']}{row['weight_mode']}`; qdq: `{row['activation_qdq_mode']}`",
            f"- output_mse: `{row['output_mse']:.8g}`; output_mae: `{row['output_mae']:.8g}`; SQNR: `{row['sqnr_db']:.4f} dB`",
            f"- activation range: min `{row['activation_min']}`, max `{row['activation_max']}`, absmax `{row['activation_absmax']:.8g}`",
            f"- activation mean/std: `{row['activation_mean']:.8g}` / `{row['activation_std']:.8g}`",
            "",
        ])
    md_path.write_text("\n".join(lines))
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def _load_manifest_samples(manifest: str | Path, max_samples: int) -> list[dict[str, torch.Tensor]]:
    dataset = LatentManifestDataset(manifest)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
    out: list[dict[str, torch.Tensor]] = []
    for sample in loader:
        out.append(sample)
        if len(out) >= max_samples:
            break
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect static QAT per-Linear QDQ error/SQNR/range diagnostics")
    parser.add_argument("--checkpoint", required=True, help="FP FlashVSR DiT checkpoint")
    parser.add_argument("--qat_checkpoint", default="", help="Optional trainable QAT state_dict to load after QAT conversion")
    parser.add_argument("--manifest", required=True, help="JSONL DiT-ready latent/context manifest")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration_cache", default="", help="Optional static calibration cache")
    parser.add_argument("--policy_json", default="", help="Optional mixed A8/A16 policy")
    parser.add_argument("--mode", default="a8w8", choices=["a8w8", "a16w8", "a8w4", "a16w4"])
    parser.add_argument("--activation_qdq_mode", default="static_asymmetric", choices=["static_asymmetric", "dynamic_symmetric", "dynamic_asymmetric"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--local_num", type=int, default=None)
    parser.add_argument("--expected_linear_count", type=int, default=306)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device(args.device)

    print(f"[Diag] Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, build_dit()).eval()
    act_stats = load_calibration_cache(args.calibration_cache, device=device) if args.calibration_cache else {}
    layer_policy = layer_policy_entries(load_layer_policy(args.policy_json)) if args.policy_json else None
    convert_model_to_qat(
        model,
        mode=args.mode,
        act_stats=act_stats,
        activation_qdq_mode=args.activation_qdq_mode,
        layer_policy=layer_policy,
    )
    if args.qat_checkpoint:
        print(f"[Diag] Loading QAT trainable state: {args.qat_checkpoint}")
        state = torch.load(args.qat_checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[Diag][WARN] missing QAT keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[Diag][WARN] unexpected QAT keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"[Diag] QAT conversion summary: {getattr(model, '_qat_conversion_summary', {})}")

    samples = _load_manifest_samples(args.manifest, args.max_samples)
    rows = collect_static_linear_diagnostics(
        model,
        samples,
        device=device,
        dtype=dtype,
        local_num=args.local_num,
    )
    paths = write_diagnostic_outputs(rows, args.output_dir, top_k=args.top_k)
    summary = {
        "linear_count": len(rows),
        "expected_linear_count": args.expected_linear_count,
        "top_layer": rows[0] if rows else None,
        "outputs": paths,
    }
    summary_path = Path(args.output_dir) / "static_qat_linear_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if args.expected_linear_count > 0 and len(rows) != args.expected_linear_count:
        print(f"[Diag][WARN] observed {len(rows)} Linear layers, expected {args.expected_linear_count}")
    print(f"[Diag] Wrote diagnostics: {paths}")
    return summary


if __name__ == "__main__":
    main()
