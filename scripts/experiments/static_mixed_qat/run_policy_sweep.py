#!/usr/bin/env python3
"""Run a static mixed A8W8/A16W8 PTQ policy conversion/eval sweep.

This wrapper intentionally records commands and manifest paths so each sweep row
is one-click reproducible.  It can run convert-only (for quick PR validation) or
invoke caller-provided inference/PSNR commands in a later production run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.static_mixed_qat.render_leaderboard import render_html
from scripts.experiments.static_mixed_qat.update_leaderboard import build_leaderboard_row, write_jsonl_row
from scripts.ptq.build_static_mixed_policy import build_static_mixed_policy_from_rows, load_sensitivity_rows, parse_percent_list


def summarize_policy_counts(policy: str | Path) -> dict[str, int]:
    data = json.loads(Path(policy).read_text())
    counts: dict[str, int] = {}
    for entry in data.get("layers", {}).values():
        mode = entry.get("mode", "unknown")
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def build_convert_command(
    *,
    python: str,
    checkpoint: str,
    calibration_cache: str,
    policy: str | Path,
    output: str | Path,
    activation_qdq_mode: str,
    clipping: str = "minmax",
    enable_bias_correction: bool = False,
    output_scale_multiplier: float = 1.5,
) -> list[str]:
    cmd = [
        python,
        "scripts/ptq/fakequant_convert.py",
        "--checkpoint", checkpoint,
        "--calibration_cache", calibration_cache,
        "--output", str(output),
        "--mode", "a8w8",
        "--activation_qdq_mode", activation_qdq_mode,
        "--policy", str(policy),
        "--output_scale_multiplier", str(output_scale_multiplier),
    ]
    if enable_bias_correction:
        cmd.append("--enable_bias_correction")
    return cmd


def run_cmd(cmd: list[str], *, cwd: Path, dry_run: bool, log_path: Path) -> int:
    printable = " ".join(str(x) for x in cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"$ {printable}\n")
    print(f"$ {printable}")
    if dry_run:
        return 0
    with log_path.open("a") as log:
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run static mixed A8W8/A16W8 policy conversion sweep")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration_cache", required=True)
    parser.add_argument("--sensitivity_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--a16_percent", default="10,20,40,60")
    parser.add_argument("--activation_qdq_mode", default="static_tensor_symmetric")
    parser.add_argument("--clipping", default="minmax")
    parser.add_argument("--enable_bias_correction", action="store_true")
    parser.add_argument("--output_scale_multiplier", type=float, default=1.5)
    parser.add_argument("--leaderboard", default="outputs/static_mixed_qat/leaderboard.jsonl")
    parser.add_argument("--eval_set", default="convert_only")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    out_dir = Path(args.out_dir)
    policies_dir = out_dir / "policies"
    ckpt_dir = out_dir / "checkpoints"
    logs_dir = out_dir / "logs"
    policies_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows = load_sensitivity_rows(args.sensitivity_json)
    manifest: dict[str, Any] = {
        "schema_version": "flashvsr.static_mixed_sweep.v1",
        "checkpoint": args.checkpoint,
        "calibration_cache": args.calibration_cache,
        "sensitivity_json": args.sensitivity_json,
        "activation_qdq_mode": args.activation_qdq_mode,
        "clipping": args.clipping,
        "enable_bias_correction": args.enable_bias_correction,
        "output_scale_multiplier": args.output_scale_multiplier,
        "policies": [],
    }

    for pct in parse_percent_list(args.a16_percent):
        policy = build_static_mixed_policy_from_rows(
            rows,
            a16_percent=pct,
            default_activation_qdq_mode=args.activation_qdq_mode,
        )
        pct_label = str(int(pct)) if float(pct).is_integer() else str(pct).replace(".", "p")
        policy_path = policies_dir / f"mixed_top{pct_label}_a16.json"
        policy_path.write_text(json.dumps(policy, indent=2))
        output = ckpt_dir / f"static_mixed_top{pct_label}_{args.clipping}.safetensors"
        cmd = build_convert_command(
            python=sys.executable,
            checkpoint=args.checkpoint,
            calibration_cache=args.calibration_cache,
            policy=policy_path,
            output=output,
            activation_qdq_mode=args.activation_qdq_mode,
            clipping=args.clipping,
            enable_bias_correction=args.enable_bias_correction,
            output_scale_multiplier=args.output_scale_multiplier,
        )
        rc = run_cmd(cmd, cwd=root, dry_run=args.dry_run, log_path=logs_dir / f"convert_top{pct_label}.log")
        counts = summarize_policy_counts(policy_path)
        manifest["policies"].append({"percent": pct, "policy": str(policy_path), "checkpoint": str(output), "returncode": rc, "counts": counts, "convert_cmd": cmd})
        write_jsonl_row(
            args.leaderboard,
            build_leaderboard_row(
                run_id=f"{out_dir.name}_top{pct_label}_{args.clipping}{'_bias' if args.enable_bias_correction else ''}",
                policy=policy_path,
                checkpoint=output if output.exists() else None,
                manifest=out_dir / "manifest.json",
                a8_layers=counts.get("a8w8", 0),
                a16_layers=counts.get("a16w8", 0),
                activation_qdq_mode=args.activation_qdq_mode,
                clipping=args.clipping,
                bias_correction=args.enable_bias_correction,
                qat=False,
                eval_set=args.eval_set,
                notes="conversion sweep row; PSNR populated by inference/eval phase",
            ),
        )
        if rc != 0:
            raise RuntimeError(f"Conversion failed for top{pct_label}; see {logs_dir / f'convert_top{pct_label}.log'}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (out_dir / "reproduce.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join([sys.executable, __file__, *sys.argv[1:]]) + "\n")
    os.chmod(out_dir / "reproduce.sh", 0o755)
    html_path = Path(args.leaderboard).with_suffix(".html")
    html_path.write_text(render_html(args.leaderboard))
    print(f"[sweep] manifest → {manifest_path}")
    print(f"[sweep] leaderboard → {args.leaderboard}")


if __name__ == "__main__":
    main()
