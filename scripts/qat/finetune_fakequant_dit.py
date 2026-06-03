"""FlashVSR v1.1 DiT QAT fine-tuning CLI.

Person A / 2026-09 scope:
- Use FlashVSR v1.1 DiT as both FP teacher and QAT student when no student v0 exists.
- Train only DiT Linear fake-quant weights; Wan VAE remains out of scope.
- Export the trained student to existing FakeQuantLinear checkpoint format.

Manifest format (JSONL): each row is either
  {"x": "sample_x.pt", "timestep": "sample_t.pt", "context": "sample_ctx.pt"}
or a .pt file containing a dict with keys x, timestep, context.  `context` must
already match WanModel.forward expectations (text-embedded DiT context), because
this repository's WanModel.forward does not call text_embedding on context.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.ptq.fakequant_convert import build_dit, load_calibration_cache, load_checkpoint
from src.models.quantization.policy import load_layer_policy, layer_policy_entries
from src.models.quantization.qat import (
    convert_model_to_qat,
    export_qat_model_to_fakequant,
    temporal_consistency_loss,
    tensor_psnr,
    update_ema_model,
)


class LatentManifestDataset(Dataset):
    def __init__(self, manifest: str | Path):
        self.manifest = Path(manifest)
        self.rows: list[dict[str, Any]] = []
        for line in self.manifest.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            if isinstance(row, str):
                row = {"sample": row}
            self.rows.append(row)
        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        path = Path(value)
        if not path.is_absolute():
            path = self.manifest.parent / path
        return torch.load(path, map_location="cpu", weights_only=False)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        if "sample" in row:
            path = Path(row["sample"])
            if not path.is_absolute() and not path.exists():
                path = self.manifest.parent / path
            sample = torch.load(path, map_location="cpu", weights_only=False)
        else:
            sample = {k: self._load_tensor(v) for k, v in row.items() if k in {"x", "timestep", "context", "target"}}
        required = {"x", "timestep", "context"}
        missing = required.difference(sample)
        if missing:
            raise KeyError(f"Sample {idx} missing required keys: {sorted(missing)}")
        return sample


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError("FlashVSR DiT QAT currently expects --batch_size 1")
    return batch[0]


def move_sample(sample: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if not isinstance(value, torch.Tensor):
            # Manifest samples may carry metadata such as source_video; keep only
            # tensors needed by WanModel.forward / optional target loss.
            continue
        if torch.is_floating_point(value):
            out[key] = value.to(device=device, dtype=dtype)
        else:
            out[key] = value.to(device=device)
    return out


def save_state_dict(model: torch.nn.Module, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    if output.suffix == ".safetensors":
        from safetensors.torch import save_file
        save_file(state, str(output))
    else:
        torch.save(state, output)


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16 if args.dtype == "bf16" else torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[QAT] Loading FlashVSR v1.1 DiT checkpoint: {args.checkpoint}")
    teacher = load_checkpoint(args.checkpoint, build_dit()).to(device=device, dtype=dtype).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = load_checkpoint(args.checkpoint, build_dit()).to(device=device, dtype=dtype).train()
    act_stats = load_calibration_cache(args.calibration_cache, device=device) if args.calibration_cache else {}
    layer_policy = None
    if args.policy_json:
        layer_policy = layer_policy_entries(load_layer_policy(args.policy_json))

    convert_model_to_qat(
        student,
        mode=args.mode,
        act_stats=act_stats,
        activation_qdq_mode=args.activation_qdq_mode,
        layer_policy=layer_policy,
    )
    print(f"[QAT] Conversion summary: {getattr(student, '_qat_conversion_summary', {})}")

    ema_student = deepcopy(student).eval() if args.ema_decay > 0 else None
    optim = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    dataset = LatentManifestDataset(args.manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    step = 0
    last_metrics: dict[str, float] = {}
    while step < args.steps:
        for sample in loader:
            if step >= args.steps:
                break
            sample = move_sample(sample, device, dtype)
            with torch.no_grad():
                teacher_out = teacher(
                    sample["x"], sample["timestep"], sample["context"],
                    use_gradient_checkpointing=False,
                    local_num=args.local_num,
                )
            student_out = student(
                sample["x"], sample["timestep"], sample["context"],
                use_gradient_checkpointing=args.gradient_checkpointing,
                local_num=args.local_num,
            )
            distill_loss = F.mse_loss(student_out.float(), teacher_out.float())
            temporal_loss = temporal_consistency_loss(student_out, teacher_out)
            target_loss = student_out.new_zeros(())
            if "target" in sample and args.target_loss_weight > 0:
                target_loss = F.mse_loss(student_out.float(), sample["target"].float())
            loss = distill_loss + args.temporal_loss_weight * temporal_loss + args.target_loss_weight * target_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optim.step()
            if ema_student is not None:
                update_ema_model(ema_student, student, args.ema_decay)

            with torch.no_grad():
                psnr = tensor_psnr(student_out, teacher_out, data_range=args.data_range)
            last_metrics = {
                "step": float(step + 1),
                "loss": float(loss.detach().cpu()),
                "distill_loss": float(distill_loss.detach().cpu()),
                "temporal_loss": float(temporal_loss.detach().cpu()),
                "teacher_psnr_db": float(psnr.detach().cpu()),
                "target_psnr_drop_db": float(args.target_psnr_drop_db),
            }
            if (step + 1) % args.log_every == 0:
                print(f"[QAT] step={step+1} metrics={last_metrics}")
            step += 1

    export_source = ema_student if ema_student is not None else student
    fakequant_model = export_qat_model_to_fakequant(export_source, inplace=False)
    qat_path = output_dir / "flashvsr_v1.1_qat_trainable.pt"
    fq_path = output_dir / "flashvsr_v1.1_qat_fakequant.pt"
    save_state_dict(export_source, qat_path)
    save_state_dict(fakequant_model, fq_path)

    summary = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "qat_trainable_checkpoint": str(qat_path),
        "fakequant_checkpoint": str(fq_path),
        "mode": args.mode,
        "activation_qdq_mode": args.activation_qdq_mode,
        "steps": args.steps,
        "lr": args.lr,
        "ema_decay": args.ema_decay,
        "last_metrics": last_metrics,
        "psnr_gate": "Run fixed eval set and compare against FP16 teacher; target drop <= %.2f dB" % args.target_psnr_drop_db,
    }
    (output_dir / "qat_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[QAT] Exported trainable checkpoint: {qat_path}")
    print(f"[QAT] Exported FakeQuant checkpoint: {fq_path}")
    print(f"[QAT] Summary: {output_dir / 'qat_summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlashVSR v1.1 DiT QAT/fake-quant fine-tuning")
    parser.add_argument("--checkpoint", required=True, help="FlashVSR v1.1 DiT FP checkpoint")
    parser.add_argument("--manifest", required=True, help="JSONL manifest of DiT-ready latent/context samples")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration_cache", default="", help="Optional PTQ calibration cache for static activation scales")
    parser.add_argument("--policy_json", default="", help="Optional per-layer policy JSON")
    parser.add_argument("--mode", default="a8w8", choices=["a8w8", "a16w8", "a8w4", "a16w4"])
    parser.add_argument("--activation_qdq_mode", default="dynamic_asymmetric", choices=["static_asymmetric", "dynamic_symmetric", "dynamic_asymmetric"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--temporal_loss_weight", type=float, default=0.05)
    parser.add_argument("--target_loss_weight", type=float, default=0.0)
    parser.add_argument("--target_psnr_drop_db", type=float, default=0.4)
    parser.add_argument("--data_range", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--local_num", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--log_every", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size != 1:
        raise ValueError("Only --batch_size 1 is currently supported for FlashVSR DiT QAT")
    train(args)


if __name__ == "__main__":
    main()
