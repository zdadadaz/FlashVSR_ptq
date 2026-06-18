#!/usr/bin/env python3
"""Append structured FlashVSR static-mixed experiment rows to leaderboard.jsonl."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_psnr_mean(path: str | Path | None) -> float | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    for key in ("average_psnr", "mean_psnr", "psnr_mean", "avg_psnr"):
        if key in data:
            return float(data[key])
    if isinstance(data.get("summary"), dict):
        for key in ("average_psnr", "mean_psnr", "psnr_mean"):
            if key in data["summary"]:
                return float(data["summary"][key])
    return None


def build_leaderboard_row(
    *,
    run_id: str,
    policy: str | Path | None = None,
    reproduce_script: str | Path | None = None,
    psnr_json: str | Path | None = None,
    checkpoint: str | Path | None = None,
    manifest: str | Path | None = None,
    a8_layers: int | None = None,
    a16_layers: int | None = None,
    activation_qdq_mode: str = "static_tensor_symmetric",
    clipping: str = "minmax",
    bias_correction: bool = False,
    lba: bool = False,
    qat: bool = False,
    observer: str | None = None,
    observer_steps: int | None = None,
    freeze_step: int | None = None,
    total_steps: int | None = None,
    eval_set: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": "flashvsr.static_mixed_leaderboard_row.v1",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy": str(policy) if policy else None,
        "policy_sha256": sha256_file(policy),
        "reproduce_script": str(reproduce_script) if reproduce_script else None,
        "reproduce_sha256": sha256_file(reproduce_script),
        "manifest": str(manifest) if manifest else None,
        "manifest_sha256": sha256_file(manifest),
        "a8_layers": a8_layers,
        "a16_layers": a16_layers,
        "activation_qdq_mode": activation_qdq_mode,
        "clipping": clipping,
        "bias_correction": bool(bias_correction),
        "lba": bool(lba),
        "qat": bool(qat),
        "observer": observer,
        "observer_steps": observer_steps,
        "freeze_step": freeze_step,
        "total_steps": total_steps,
        "eval_set": eval_set,
        "psnr_json": str(psnr_json) if psnr_json else None,
        "psnr_vs_fp16_mean": _read_psnr_mean(psnr_json),
        "psnr_vs_gt_fp16": None,
        "psnr_vs_gt_quant": None,
        "drop_vs_gt_db": None,
        "notes": notes,
    }


def write_jsonl_row(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a FlashVSR static-mixed leaderboard row")
    parser.add_argument("--leaderboard", default="outputs/static_mixed_qat/leaderboard.jsonl")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--policy", default="")
    parser.add_argument("--reproduce_script", default="")
    parser.add_argument("--psnr_json", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--a8_layers", type=int, default=None)
    parser.add_argument("--a16_layers", type=int, default=None)
    parser.add_argument("--activation_qdq_mode", default="static_tensor_symmetric")
    parser.add_argument("--clipping", default="minmax")
    parser.add_argument("--bias_correction", action="store_true")
    parser.add_argument("--lba", action="store_true")
    parser.add_argument("--qat", action="store_true")
    parser.add_argument("--observer", default=None)
    parser.add_argument("--observer_steps", type=int, default=None)
    parser.add_argument("--freeze_step", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--eval_set", default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    row = build_leaderboard_row(
        run_id=args.run_id,
        policy=args.policy or None,
        reproduce_script=args.reproduce_script or None,
        psnr_json=args.psnr_json or None,
        checkpoint=args.checkpoint or None,
        manifest=args.manifest or None,
        a8_layers=args.a8_layers,
        a16_layers=args.a16_layers,
        activation_qdq_mode=args.activation_qdq_mode,
        clipping=args.clipping,
        bias_correction=args.bias_correction,
        lba=args.lba,
        qat=args.qat,
        observer=args.observer,
        observer_steps=args.observer_steps,
        freeze_step=args.freeze_step,
        total_steps=args.total_steps,
        eval_set=args.eval_set,
        notes=args.notes,
    )
    write_jsonl_row(args.leaderboard, row)
    print(f"[leaderboard] appended {args.run_id} → {args.leaderboard}")


if __name__ == "__main__":
    main()
