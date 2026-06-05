#!/usr/bin/env python3
"""PR-0 LSGQuant paper-grounded evaluation harness for FlashVSR DiT PTQ.

This script does not introduce a new quantizer.  It records the experiment
setting from LSGQuant §4.1, samples the HQ-VSR calibration set exactly like the
paper (50 videos, deterministic seed), enumerates UDM10/REDS30/MVSR4x eval
clips, and emits reproducible FP16/PTQ render commands plus the metric plan.

Scope is intentionally DiT-only: Wan VAE remains unquantized.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


DEFAULT_EXPERIMENT_SETTINGS: dict[str, Any] = {
    "paper": "arXiv:2602.03182v1",
    "title": "LSGQuant: Layer-Sensitivity Guided Quantization for One-Step Diffusion Real-World Video Super-Resolution",
    "backbone": "one-step WAN2.1 full-precision DiT following DOVE settings",
    "calibration": {"dataset": "HQ-VSR", "num_videos": 50, "procedure": "random sample videos and run full FP DiT inference"},
    "evaluation_datasets": {"UDM10": "synthetic", "REDS30": "synthetic", "MVSR4x": "real_world"},
    "metrics": {
        "reference_iqa": ["DISTS", "PSNR", "SSIM", "LPIPS"],
        "no_reference_iqa": ["MANIQA", "CLIP-IQA", "MUSIQ"],
        "vqa": ["Ewarp*", "DOVER"],
    },
    "volts": {
        "delta1": 0.001,
        "delta2": 0.075,
        "frozen_iterations": 1,
        "light_adaptation_iterations": 30,
        "fully_optimized_iterations": "until_convergence",
    },
    "weight_quantization": "static_asymmetric_channel_wise",
    "qao": {"svd_rank": 32},
    "implementation": {"framework": "PyTorch", "paper_gpu": "NVIDIA RTX A6000"},
    "scope": "WanVideoDiT Linear layers only; Wan VAE remains unquantized",
}


@dataclass(frozen=True)
class EvalVideo:
    dataset: str
    dataset_type: str
    input_video: Path
    reference_video: Path | None = None


def discover_videos(root: Path | str, limit: int | None = None, seed: int = 0) -> list[str]:
    """Discover videos recursively and return deterministic sampled string paths."""

    root = Path(root)
    videos = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if limit is not None and len(videos) > limit:
        rng = random.Random(seed)
        videos = sorted(rng.sample(videos, limit))
    return [str(p) for p in videos]


def parse_dataset_args(values: Iterable[str]) -> dict[str, Path]:
    """Parse repeated NAME=PATH CLI args."""

    datasets: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Dataset must be NAME=PATH, got: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Dataset name is empty in: {value}")
        datasets[name] = Path(raw_path).expanduser()
    return datasets


def dataset_type(name: str) -> str:
    return DEFAULT_EXPERIMENT_SETTINGS["evaluation_datasets"].get(name, "unknown")


def discover_eval_videos(eval_datasets: dict[str, Path], refs_root: Path | None = None) -> list[EvalVideo]:
    """Discover eval videos for all configured datasets.

    If `refs_root/<dataset>/<same filename>` exists, it is recorded as the
    reference target for full-reference metrics.  Missing references do not fail
    PR-0 because this harness must also support dry-run planning before datasets
    are staged.
    """

    items: list[EvalVideo] = []
    for name, root in sorted(eval_datasets.items()):
        for video in discover_videos(root):
            input_video = Path(video)
            ref = None
            if refs_root is not None:
                candidate = refs_root / name / input_video.name
                if candidate.exists():
                    ref = candidate
            items.append(EvalVideo(dataset=name, dataset_type=dataset_type(name), input_video=input_video, reference_video=ref))
    return items


def build_cli_command(
    input_video: Path,
    output_video: Path,
    frames: int,
    seed: int,
    fp_checkpoint: Path | None = None,
    quant_checkpoint: Path | None = None,
    quantize_mode: str | None = None,
) -> list[str]:
    """Build the FlashVSR CLI command used by the PR-0 manifest."""

    cmd = [
        ".venv/bin/python",
        "cli_main.py",
        "--input", str(input_video),
        "--output", str(output_video),
        "--model", "FlashVSR-v1.1",
        "--vae_model", "Wan2.1",
        "--scale", "4",
        "--mode", "full",
        "--precision", "fp16",
        "--device", "cuda:0",
        "--attention_mode", "sdpa",
        "--start_frame", "0",
        "--end_frame", str(frames),
        "--seed", str(seed),
    ]
    if fp_checkpoint is not None and quant_checkpoint is None:
        cmd.extend(["--ckpt_path", str(fp_checkpoint)])
    if quantize_mode is not None:
        cmd.extend(["--quantize_mode", quantize_mode])
    if quant_checkpoint is not None:
        cmd.extend(["--ckpt_path", str(quant_checkpoint)])
    return cmd


def _run_command(cmd: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return int(proc.returncode)


def build_pr0_manifest(
    calibration_dataset: Path,
    eval_datasets: dict[str, Path],
    out_dir: Path,
    fp_checkpoint: Path | None = None,
    quant_checkpoint: Path | None = None,
    refs_root: Path | None = None,
    frames: int = 16,
    seed: int = 0,
    execute: bool = False,
    quantize_mode: str = "FakeQuant_A8W8",
) -> dict[str, Any]:
    """Create the PR-0 manifest from paper settings and local dataset paths."""

    calib_all = discover_videos(calibration_dataset)
    calib_sample = discover_videos(calibration_dataset, limit=DEFAULT_EXPERIMENT_SETTINGS["calibration"]["num_videos"], seed=seed)
    eval_items = discover_eval_videos(eval_datasets, refs_root=refs_root)

    runs: list[dict[str, Any]] = []
    root = Path.cwd()
    for item in eval_items:
        stem = item.input_video.stem
        fp16_output = out_dir / "videos" / item.dataset / f"{stem}_fp16_first{frames}.mp4"
        fp16_log = out_dir / "reports" / item.dataset / f"{stem}_fp16.log"
        fp16_cmd = build_cli_command(item.input_video, fp16_output, frames=frames, seed=seed, fp_checkpoint=fp_checkpoint)
        runs.append({
            "dataset": item.dataset,
            "dataset_type": item.dataset_type,
            "input_video": str(item.input_video),
            "reference_video": str(item.reference_video) if item.reference_video else None,
            "precision": "fp16",
            "output_video": str(fp16_output),
            "log": str(fp16_log),
            "command": fp16_cmd,
            "returncode": _run_command(fp16_cmd, fp16_log, root) if execute else None,
        })
        if quant_checkpoint is not None:
            quant_output = out_dir / "videos" / item.dataset / f"{stem}_{quantize_mode.lower()}_first{frames}.mp4"
            quant_log = out_dir / "reports" / item.dataset / f"{stem}_{quantize_mode.lower()}.log"
            quant_cmd = build_cli_command(
                item.input_video,
                quant_output,
                frames=frames,
                seed=seed,
                quant_checkpoint=quant_checkpoint,
                quantize_mode=quantize_mode,
            )
            runs.append({
                "dataset": item.dataset,
                "dataset_type": item.dataset_type,
                "input_video": str(item.input_video),
                "reference_video": str(item.reference_video) if item.reference_video else None,
                "precision": "fp16",
                "quantize_mode": quantize_mode,
                "output_video": str(quant_output),
                "log": str(quant_log),
                "command": quant_cmd,
                "returncode": _run_command(quant_cmd, quant_log, root) if execute else None,
            })

    evaluation: dict[str, Any] = {}
    for name, path in sorted(eval_datasets.items()):
        videos = [item for item in eval_items if item.dataset == name]
        evaluation[name] = {
            "dataset_type": dataset_type(name),
            "root": str(path),
            "num_videos": len(videos),
            "videos": [str(item.input_video) for item in videos],
        }

    manifest = {
        "schema_version": "flashvsr.lsgquant_pr0_eval.v1",
        "experiment_settings": DEFAULT_EXPERIMENT_SETTINGS,
        "calibration": {
            "dataset": "HQ-VSR",
            "root": str(calibration_dataset),
            "videos": calib_sample,
            "sampling": {
                "seed": seed,
                "requested_videos": DEFAULT_EXPERIMENT_SETTINGS["calibration"]["num_videos"],
                "available_videos": len(calib_all),
            },
            "procedure": "Run full FP DiT inference on these videos to collect calibration activations/stats in later PRs.",
        },
        "evaluation": evaluation,
        "metric_plan": {
            "implemented_now": ["PSNR"],
            "planned_reference_iqa": DEFAULT_EXPERIMENT_SETTINGS["metrics"]["reference_iqa"],
            "planned_no_reference_iqa": DEFAULT_EXPERIMENT_SETTINGS["metrics"]["no_reference_iqa"],
            "planned_vqa": DEFAULT_EXPERIMENT_SETTINGS["metrics"]["vqa"],
            "notes": "PR-0 records the paper metric contract and reproducible outputs; later PRs wire DISTS/SSIM/LPIPS/MANIQA/CLIP-IQA/MUSIQ/Ewarp*/DOVER implementations.",
        },
        "execution": {
            "execute": execute,
            "frames": frames,
            "seed": seed,
            "out_dir": str(out_dir),
            "fp_checkpoint": str(fp_checkpoint) if fp_checkpoint else None,
            "quant_checkpoint": str(quant_checkpoint) if quant_checkpoint else None,
        },
        "runs": runs,
    }
    return manifest


def write_manifest(manifest: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lsgquant_pr0_eval_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/run PR-0 LSGQuant experiment-setting manifest for FlashVSR")
    parser.add_argument("--calibration_dataset", required=True, help="HQ-VSR calibration dataset root")
    parser.add_argument("--eval_dataset", action="append", default=[], help="Evaluation dataset as NAME=PATH; use UDM10/REDS30/MVSR4x")
    parser.add_argument("--refs_root", default=None, help="Optional reference videos root laid out as refs_root/NAME/<filename>")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fp_checkpoint", default=None)
    parser.add_argument("--quant_checkpoint", default=None)
    parser.add_argument("--quantize_mode", default="FakeQuant_A8W8")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute", action="store_true", help="Actually execute rendered FP16/PTQ commands; default only writes manifest")
    args = parser.parse_args()

    eval_datasets = parse_dataset_args(args.eval_dataset)
    if not eval_datasets:
        raise SystemExit("At least one --eval_dataset NAME=PATH is required")
    manifest = build_pr0_manifest(
        calibration_dataset=Path(args.calibration_dataset).expanduser(),
        eval_datasets=eval_datasets,
        out_dir=Path(args.out_dir).expanduser(),
        fp_checkpoint=Path(args.fp_checkpoint).expanduser() if args.fp_checkpoint else None,
        quant_checkpoint=Path(args.quant_checkpoint).expanduser() if args.quant_checkpoint else None,
        refs_root=Path(args.refs_root).expanduser() if args.refs_root else None,
        frames=args.frames,
        seed=args.seed,
        execute=args.execute,
        quantize_mode=args.quantize_mode,
    )
    manifest_path = write_manifest(manifest, Path(args.out_dir).expanduser())
    print(json.dumps({"manifest": str(manifest_path), "runs": len(manifest["runs"]), "execute": args.execute}, indent=2))


if __name__ == "__main__":
    main()
