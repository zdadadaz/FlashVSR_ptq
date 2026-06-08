#!/usr/bin/env python3
"""Standardized LSGQuant-style PTQ evaluation manifest and PSNR helper.

This script ties together the PR-0 paper dataset contract with PR-1 DRAQ/static
A8W8 checkpoints.  It records the HQ-VSR calibration sample, verifies the static
calibration cache contains act_scale/zero_point, discovers LSGQuant eval-set
videos, builds deterministic FP16/PTQ render commands, and can compute FP16-vs-
PTQ PSNR once outputs exist.

Scope remains DiT-only: Wan VAE is not quantized.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
LSGQUANT_EVAL_DATASETS = {"UDM10": "synthetic", "REDS30": "synthetic", "MVSR4x": "real_world"}


def discover_videos(root: Path, limit: int | None = None, seed: int = 0) -> list[str]:
    if not root.exists():
        return []
    videos = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if limit is not None and len(videos) > limit:
        videos = sorted(random.Random(seed).sample(videos, limit))
    return [str(p) for p in videos]


def _image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def discover_image_sequence_dirs(root: Path, limit: int | None = None, seed: int = 0) -> list[Path]:
    if not root.exists():
        return []
    seq_dirs = sorted(d for d in root.rglob("*") if d.is_dir() and _image_files(d))
    if limit is not None and len(seq_dirs) > limit:
        seq_dirs = sorted(random.Random(seed).sample(seq_dirs, limit))
    return seq_dirs


def materialize_downsampled_sequence_video(
    sequence_dir: Path,
    output_video: Path,
    frames: int,
    downsample_scale: int = 4,
    fps: float = 8.0,
) -> dict[str, Any]:
    """Convert an image sequence to an MP4 after downsampling LQ by scale.

    Test-set LQ folders are already at the target comparison resolution for this
    FlashVSR workflow.  Downsampling by 4 before inference makes FlashVSR's x4
    output align with the original LQ/GT dimensions.
    """

    images = _image_files(sequence_dir)[:frames]
    if not images:
        raise RuntimeError(f"No image frames found in {sequence_dir}")
    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {images[0]}")
    src_h, src_w = first.shape[:2]
    raw_w = max(1, int(round(src_w / downsample_scale)))
    raw_h = max(1, int(round(src_h / downsample_scale)))
    # yuv420p/libx264 needs even dimensions.  Padding the downsampled input by at
    # most one pixel keeps the resulting FlashVSR x4 output close to the original
    # sequence size while avoiding ffmpeg encoder failures on odd heights.
    video_w = raw_w + (raw_w % 2)
    video_h = raw_h + (raw_h % 2)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (video_w, video_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_video}")
    written = 0
    for image_path in images:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Cannot read frame: {image_path}")
        small = cv2.resize(frame, (raw_w, raw_h), interpolation=cv2.INTER_AREA)
        if (raw_w, raw_h) != (video_w, video_h):
            small = cv2.copyMakeBorder(small, 0, video_h - raw_h, 0, video_w - raw_w, cv2.BORDER_REPLICATE)
        writer.write(small)
        written += 1
    writer.release()
    return {
        "source_sequence": str(sequence_dir),
        "output_video": str(output_video),
        "frames": written,
        "source_size": [src_w, src_h],
        "downsample_scale": downsample_scale,
        "video_size": [video_w, video_h],
        "fps": fps,
    }


def calibration_cache_summary(cache_path: Path) -> dict[str, Any]:
    raw = json.loads(cache_path.read_text())
    layer_items = {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, dict)}
    with_scale = sum(1 for v in layer_items.values() if "act_scale" in v)
    with_zp = sum(1 for v in layer_items.values() if "zero_point" in v)
    with_minmax = sum(1 for v in layer_items.values() if "act_min" in v and "act_max" in v)
    return {
        "path": str(cache_path),
        "layers": len(layer_items),
        "layers_with_act_scale": with_scale,
        "layers_with_zero_point": with_zp,
        "layers_with_minmax": with_minmax,
        "ready_for_static_a8w8": bool(layer_items) and with_scale == len(layer_items) and with_zp == len(layer_items),
        "metadata": raw.get("_metadata", {}),
    }


def discover_sequence_inputs(root: Path, dataset: str, limit: int | None = None, prefer_image_sequences: bool = False) -> list[dict[str, str | None]]:
    """Discover eval inputs.

    Supported layouts:
    - LQ-Video/*.mp4 paired with GT-Video/*.mp4
    - LQ/<seq>/*.png paired with GT/<seq>/*.png (native image sequence)
    - raw videos anywhere under root
    """

    lq_video = root / "LQ-Video"
    gt_video = root / "GT-Video"
    lq_root = root / "LQ"
    gt_root = root / "GT"
    if prefer_image_sequences and lq_root.exists():
        seq_dirs = discover_image_sequence_dirs(lq_root, limit=limit)
        if seq_dirs:
            items = []
            for seq_dir in seq_dirs:
                rel = seq_dir.relative_to(lq_root)
                ref = gt_root / rel
                items.append({
                    "dataset": dataset,
                    "dataset_type": LSGQUANT_EVAL_DATASETS.get(dataset, "unknown"),
                    "input_sequence": str(seq_dir),
                    "reference_sequence": str(ref) if ref.exists() and _image_files(ref) else None,
                    "sequence_name": rel.as_posix().replace("/", "_"),
                })
            return items
    if lq_video.exists():
        videos = [Path(p) for p in discover_videos(lq_video, limit=limit)]
        items = []
        for video in videos:
            ref = gt_video / video.name
            items.append({
                "dataset": dataset,
                "dataset_type": LSGQUANT_EVAL_DATASETS.get(dataset, "unknown"),
                "input_video": str(video),
                "reference_video": str(ref) if ref.exists() else None,
            })
        return items

    lq_root = root / "LQ"
    gt_root = root / "GT"
    if lq_root.exists():
        seq_dirs = discover_image_sequence_dirs(lq_root, limit=limit)
        items = []
        for seq_dir in seq_dirs:
            rel = seq_dir.relative_to(lq_root)
            ref = gt_root / rel
            items.append({
                "dataset": dataset,
                "dataset_type": LSGQUANT_EVAL_DATASETS.get(dataset, "unknown"),
                "input_sequence": str(seq_dir),
                "reference_sequence": str(ref) if ref.exists() and _image_files(ref) else None,
                "sequence_name": rel.as_posix().replace("/", "_"),
            })
        return items

    videos = [Path(p) for p in discover_videos(root, limit=limit)]
    return [
        {
            "dataset": dataset,
            "dataset_type": LSGQUANT_EVAL_DATASETS.get(dataset, "unknown"),
            "input_video": str(video),
            "reference_video": None,
        }
        for video in videos
    ]


def build_cli_command(
    input_video: Path,
    output_video: Path,
    frames: int,
    seed: int,
    checkpoint: Path | None,
    quantize_mode: str,
) -> list[str]:
    cmd = [
        ".venv/bin/python", "cli_main.py",
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
        "--quantize_mode", quantize_mode,
    ]
    if checkpoint is not None:
        cmd.extend(["--ckpt_path", str(checkpoint)])
    return cmd


def compute_video_psnr(ref: Path, dist: Path) -> dict[str, Any]:
    cap_ref = cv2.VideoCapture(str(ref))
    cap_dist = cv2.VideoCapture(str(dist))
    if not cap_ref.isOpened():
        raise RuntimeError(f"Cannot open ref video: {ref}")
    if not cap_dist.isOpened():
        raise RuntimeError(f"Cannot open dist video: {dist}")
    psnrs: list[float] = []
    frame_idx = 0
    while True:
        ok_r, ref_frame = cap_ref.read()
        ok_d, dist_frame = cap_dist.read()
        if not ok_r or not ok_d:
            break
        if ref_frame.shape != dist_frame.shape:
            raise RuntimeError(f"Frame {frame_idx} shape mismatch: {ref_frame.shape} vs {dist_frame.shape}")
        diff = ref_frame.astype(np.float32) - dist_frame.astype(np.float32)
        mse = float(np.mean(diff * diff))
        psnrs.append(float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse)))
        frame_idx += 1
    cap_ref.release()
    cap_dist.release()
    if not psnrs:
        raise RuntimeError(f"No comparable frames: {ref} vs {dist}")
    return {
        "ref": str(ref),
        "dist": str(dist),
        "frames": len(psnrs),
        "psnr_avg_db": float(np.mean(psnrs)),
        "psnr_min_db": float(np.min(psnrs)),
        "psnr_max_db": float(np.max(psnrs)),
        "psnr_per_frame_db": psnrs,
    }


def build_standard_eval_manifest(
    calibration_dataset: Path,
    calibration_cache: Path,
    eval_datasets: dict[str, Path],
    out_dir: Path,
    fp_checkpoint: Path | None,
    quant_checkpoint: Path,
    quantize_mode: str = "FakeQuant_A8W8",
    frames: int = 16,
    seed: int = 0,
    limit_per_dataset: int | None = None,
    prepare_image_sequences: bool = False,
    downsample_lq_scale: int = 4,
    sequence_fps: float = 8.0,
) -> dict[str, Any]:
    calibration_videos = discover_videos(calibration_dataset, limit=50, seed=seed)
    eval_items: list[dict[str, str | None]] = []
    for name, root in sorted(eval_datasets.items()):
        eval_items.extend(discover_sequence_inputs(root, name, limit=limit_per_dataset, prefer_image_sequences=prepare_image_sequences))

    runs: list[dict[str, Any]] = []
    preparation: list[dict[str, Any]] = []
    for item in eval_items:
        dataset = str(item["dataset"])
        if "input_sequence" in item:
            stem = str(item.get("sequence_name") or Path(str(item["input_sequence"])).name)
            prepared_video = out_dir / "prepared_inputs" / dataset / f"{stem}_lq_downx{downsample_lq_scale}_first{frames}.mp4"
            if prepare_image_sequences:
                prep = materialize_downsampled_sequence_video(
                    Path(str(item["input_sequence"])),
                    prepared_video,
                    frames=frames,
                    downsample_scale=downsample_lq_scale,
                    fps=sequence_fps,
                )
                prep["dataset"] = dataset
                prep["sequence_name"] = stem
                preparation.append(prep)
            inp = prepared_video
        else:
            inp = Path(str(item["input_video"]))
            stem = inp.stem
        fp_out = out_dir / "videos" / dataset / f"{stem}_fp16_first{frames}.mp4"
        ptq_out = out_dir / "videos" / dataset / f"{stem}_{quantize_mode.lower()}_first{frames}.mp4"
        common = {
            "input_source_sequence": item.get("input_sequence"),
            "reference_sequence": item.get("reference_sequence"),
            "reference_video": item.get("reference_video"),
            "prepared_from_sequence": "input_sequence" in item,
        }
        runs.append({
            "dataset": dataset,
            "kind": "fp16",
            "input_video": str(inp),
            "output_video": str(fp_out),
            "command": build_cli_command(inp, fp_out, frames, seed, fp_checkpoint, "None"),
            **common,
        })
        runs.append({
            "dataset": dataset,
            "kind": "ptq_a8w8",
            "input_video": str(inp),
            "output_video": str(ptq_out),
            "command": build_cli_command(inp, ptq_out, frames, seed, quant_checkpoint, quantize_mode),
            **common,
        })

    return {
        "schema_version": "flashvsr.lsgquant_standard_eval.v1",
        "scope": "WanVideoDiT Linear layers only; Wan VAE remains unquantized",
        "paper": "arXiv:2602.03182v1",
        "quantize_mode": quantize_mode,
        "calibration": {
            "dataset": "HQ-VSR",
            "root": str(calibration_dataset),
            "videos": calibration_videos,
            "sampling": {"seed": seed, "requested_videos": 50, "available_videos": len(discover_videos(calibration_dataset))},
            "cache": calibration_cache_summary(calibration_cache),
            "static_a8w8_contract": "Use act_scale and zero_point from this HQ-VSR cache for static activation QDQ; DRAQ checkpoints record the same calibration set for standardized comparison but compute activation scales online.",
        },
        "evaluation": eval_items,
        "preparation": {
            "prepare_image_sequences": prepare_image_sequences,
            "downsample_lq_scale": downsample_lq_scale,
            "sequence_fps": sequence_fps,
            "prepared_inputs": preparation,
            "contract": "For datasets/test image-sequence LQ inputs, downsample LQ by this scale before FlashVSR so x4 output aligns with the original LQ/GT resolution.",
        },
        "execution": {"frames": frames, "seed": seed, "out_dir": str(out_dir), "fp_checkpoint": str(fp_checkpoint) if fp_checkpoint else None, "quant_checkpoint": str(quant_checkpoint)},
        "runs": runs,
    }


def add_existing_psnr(manifest: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fp_by_input = {r["input_video"]: r for r in manifest["runs"] if r["kind"] == "fp16"}
    metrics = []
    for ptq_run in [r for r in manifest["runs"] if r["kind"] == "ptq_a8w8"]:
        fp_run = fp_by_input.get(ptq_run["input_video"])
        if not fp_run:
            continue
        fp_video = Path(fp_run["output_video"])
        ptq_video = Path(ptq_run["output_video"])
        if fp_video.exists() and ptq_video.exists():
            metric = compute_video_psnr(fp_video, ptq_video)
            metric_path = metrics_dir / f"{Path(ptq_run['input_video']).stem}_fp16_vs_ptq_psnr.json"
            metric_path.write_text(json.dumps(metric, indent=2))
            metric["metric_json"] = str(metric_path)
            metric["dataset"] = ptq_run["dataset"]
            metrics.append(metric)
    manifest["metrics"] = metrics
    if metrics:
        manifest["summary"] = {
            "mean_psnr_avg_db": float(np.mean([m["psnr_avg_db"] for m in metrics])),
            "min_clip_avg_psnr_db": float(np.min([m["psnr_avg_db"] for m in metrics])),
            "max_clip_avg_psnr_db": float(np.max([m["psnr_avg_db"] for m in metrics])),
            "num_clips": len(metrics),
        }
    return manifest


def parse_dataset_args(values: list[str]) -> dict[str, Path]:
    datasets: dict[str, Path] = {}
    for value in values:
        name, raw = value.split("=", 1)
        datasets[name] = Path(raw)
    return datasets


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/run standardized LSGQuant-style FP16 vs PTQ A8W8 eval manifest")
    ap.add_argument("--calibration_dataset", required=True)
    ap.add_argument("--calibration_cache", required=True)
    ap.add_argument("--eval_dataset", action="append", default=[], help="NAME=PATH, e.g. UDM10=datasets/test/UDM10")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fp_checkpoint", default=None)
    ap.add_argument("--quant_checkpoint", required=True)
    ap.add_argument("--quantize_mode", default="FakeQuant_A8W8")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_per_dataset", type=int, default=None)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--compute_existing_psnr", action="store_true")
    ap.add_argument("--prepare_image_sequences", action="store_true", help="Materialize datasets/test LQ image-sequence inputs as downsampled MP4s before building run commands")
    ap.add_argument("--downsample_lq_scale", type=int, default=4, help="Downsample factor applied to LQ image sequences before FlashVSR x4 inference")
    ap.add_argument("--sequence_fps", type=float, default=8.0, help="FPS for materialized image-sequence MP4 inputs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_standard_eval_manifest(
        calibration_dataset=Path(args.calibration_dataset),
        calibration_cache=Path(args.calibration_cache),
        eval_datasets=parse_dataset_args(args.eval_dataset),
        out_dir=out_dir,
        fp_checkpoint=Path(args.fp_checkpoint) if args.fp_checkpoint else None,
        quant_checkpoint=Path(args.quant_checkpoint),
        quantize_mode=args.quantize_mode,
        frames=args.frames,
        seed=args.seed,
        limit_per_dataset=args.limit_per_dataset,
        prepare_image_sequences=args.prepare_image_sequences,
        downsample_lq_scale=args.downsample_lq_scale,
        sequence_fps=args.sequence_fps,
    )
    if args.execute:
        for run in manifest["runs"]:
            log_path = out_dir / "reports" / run["dataset"] / f"{Path(run['output_video']).stem}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as log:
                proc = subprocess.run(run["command"], cwd=Path(__file__).resolve().parents[2], stdout=log, stderr=subprocess.STDOUT, text=True)
            output_exists = Path(run["output_video"]).exists()
            run["returncode"] = proc.returncode if output_exists else (proc.returncode or 1)
            run["output_exists"] = output_exists
            if not output_exists:
                run["error"] = "command completed without producing output_video"
            run["log"] = str(log_path)
    if args.compute_existing_psnr or args.execute:
        manifest = add_existing_psnr(manifest, out_dir)
    manifest_path = out_dir / "lsgquant_standard_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"manifest": str(manifest_path), "runs": len(manifest["runs"]), "metrics": len(manifest.get("metrics", []))}, indent=2))


if __name__ == "__main__":
    main()
