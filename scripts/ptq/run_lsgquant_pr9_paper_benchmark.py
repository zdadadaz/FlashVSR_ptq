#!/usr/bin/env python3
"""PR-9 paper-comparable benchmark harness for FlashVSR LSGQuant A4W4 QAO.

Builds a reproducible UDM10 / REDS30 / MVSR4x benchmark manifest, executes
FP16 and A4W4-QAO renders when requested, and records full-reference PSNR
against GT plus FP16-vs-PTQ delta.  Scope is DiT-only: Wan VAE remains full
precision/unquantized.
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
PAPER_DATASETS = {"UDM10": "synthetic", "REDS30": "synthetic", "MVSR4x": "real_world"}
DEFAULT_VENV = "/home/user/apps/FlashVSRptq/FlashVSR_Integrated/.venv/bin/python"


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def sequence_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and image_files(d))


def video_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def first_frame_size(video: Path) -> tuple[int, int] | None:
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return int(frame.shape[1]), int(frame.shape[0])


def sequence_size(seq: Path) -> tuple[int, int]:
    imgs = image_files(seq)
    if not imgs:
        raise RuntimeError(f"No frames in sequence: {seq}")
    frame = cv2.imread(str(imgs[0]), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read first frame: {imgs[0]}")
    return int(frame.shape[1]), int(frame.shape[0])


def write_sequence_video(seq: Path, out: Path, frames: int, fps: float = 8.0, downsample_scale: int = 1) -> dict[str, Any]:
    imgs = image_files(seq)[:frames]
    if not imgs:
        raise RuntimeError(f"No images found in {seq}")
    first = cv2.imread(str(imgs[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read {imgs[0]}")
    src_h, src_w = first.shape[:2]
    dst_w = max(1, int(round(src_w / downsample_scale)))
    dst_h = max(1, int(round(src_h / downsample_scale)))
    enc_w = dst_w + (dst_w % 2)
    enc_h = dst_h + (dst_h % 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (enc_w, enc_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {out}")
    written = 0
    for img_path in imgs:
        frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Cannot read {img_path}")
        if downsample_scale != 1:
            frame = cv2.resize(frame, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        elif (frame.shape[1], frame.shape[0]) != (dst_w, dst_h):
            frame = cv2.resize(frame, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        if (dst_w, dst_h) != (enc_w, enc_h):
            frame = cv2.copyMakeBorder(frame, 0, enc_h - dst_h, 0, enc_w - dst_w, cv2.BORDER_REPLICATE)
        writer.write(frame)
        written += 1
    writer.release()
    return {
        "source_sequence": str(seq),
        "output_video": str(out),
        "frames": written,
        "source_size": [src_w, src_h],
        "video_size": [enc_w, enc_h],
        "downsample_scale": downsample_scale,
        "fps": fps,
    }


def discover_named_pairs(dataset: str, root: Path, out_dir: Path, frames: int, limit: int | None, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return benchmark items and preparation records.

    For UDM10 LQ already has 1/4 GT resolution, so use LQ-Video directly.
    For MVSR4x the staged LQ/GT are same resolution; downsample LQ by x4 before
    inference to make FlashVSR x4 output align with GT.  REDS30 is materialized
    from REDS val_sharp GT image sequences by generating a x4 LQ input video.
    """
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    prep: list[dict[str, Any]] = []

    if dataset in {"UDM10", "MVSR4x"}:
        lq_videos = video_files(root / "LQ-Video")
        gt_videos = {p.stem: p for p in video_files(root / "GT-Video")}
        lq_seq_root = root / "LQ"
        candidates = lq_videos
        if limit is not None and len(candidates) > limit:
            candidates = sorted(rng.sample(candidates, limit))
        for lq_video in candidates:
            gt_video = gt_videos.get(lq_video.stem)
            if gt_video is None:
                continue
            lq_size = first_frame_size(lq_video)
            gt_size = first_frame_size(gt_video)
            input_video = lq_video
            input_contract = "native_lq_video"
            if lq_size and gt_size and abs(lq_size[0] * 4 - gt_size[0]) <= 4 and abs(lq_size[1] * 4 - gt_size[1]) <= 4:
                input_contract = "native_lq_video_x4_matches_gt"
            elif lq_size and gt_size and abs(lq_size[0] - gt_size[0]) <= 4 and abs(lq_size[1] - gt_size[1]) <= 4:
                seq = lq_seq_root / lq_video.stem
                input_video = out_dir / "prepared_inputs" / dataset / f"{lq_video.stem}_lq_downx4_first{frames}.mp4"
                rec = write_sequence_video(seq, input_video, frames=frames, downsample_scale=4)
                rec.update({"dataset": dataset, "sequence_name": lq_video.stem, "reason": "LQ and GT staged at same resolution; downsample input for x4 model alignment"})
                prep.append(rec)
                input_contract = "prepared_lq_downx4_from_same_resolution_lq"
            items.append({
                "dataset": dataset,
                "dataset_type": PAPER_DATASETS[dataset],
                "clip": lq_video.stem,
                "input_video": str(input_video),
                "reference_video": str(gt_video),
                "input_contract": input_contract,
                "source_lq_video": str(lq_video),
                "source_gt_video": str(gt_video),
                "lq_size": list(lq_size) if lq_size else None,
                "gt_size": list(gt_size) if gt_size else None,
            })
        return items, prep

    if dataset == "REDS30":
        seqs = sequence_dirs(root)
        if limit is not None and len(seqs) > limit:
            seqs = sorted(rng.sample(seqs, limit))
        else:
            seqs = seqs[:30]
        for seq in seqs:
            clip = seq.name
            lq_out = out_dir / "prepared_inputs" / dataset / f"{clip}_lq_downx4_first{frames}.mp4"
            gt_out = out_dir / "prepared_refs" / dataset / f"{clip}_gt_first{frames}.mp4"
            lq_rec = write_sequence_video(seq, lq_out, frames=frames, downsample_scale=4)
            gt_rec = write_sequence_video(seq, gt_out, frames=frames, downsample_scale=1)
            lq_rec.update({"dataset": dataset, "sequence_name": clip, "reason": "REDS val_sharp GT materialized as x4 LQ input"})
            gt_rec.update({"dataset": dataset, "sequence_name": clip, "reason": "REDS val_sharp GT reference video"})
            prep.extend([lq_rec, gt_rec])
            items.append({
                "dataset": dataset,
                "dataset_type": PAPER_DATASETS[dataset],
                "clip": clip,
                "input_video": str(lq_out),
                "reference_video": str(gt_out),
                "input_contract": "prepared_lq_downx4_from_reds_val_sharp_gt",
                "source_gt_sequence": str(seq),
                "gt_size": list(sequence_size(seq)),
            })
        return items, prep

    raise ValueError(f"Unsupported dataset: {dataset}")


def build_cli_command(python_bin: str, input_video: Path, output_video: Path, frames: int, seed: int, quantize_mode: str, checkpoint: Path | None) -> list[str]:
    cmd = [
        python_bin, "cli_main.py",
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


def compute_video_psnr(ref: Path, dist: Path, crop_to_common: bool = True) -> dict[str, Any]:
    cap_ref = cv2.VideoCapture(str(ref))
    cap_dist = cv2.VideoCapture(str(dist))
    if not cap_ref.isOpened():
        raise RuntimeError(f"Cannot open ref video: {ref}")
    if not cap_dist.isOpened():
        raise RuntimeError(f"Cannot open dist video: {dist}")
    psnrs: list[float] = []
    shapes: list[dict[str, Any]] = []
    idx = 0
    while True:
        ok_r, r = cap_ref.read()
        ok_d, d = cap_dist.read()
        if not ok_r or not ok_d:
            break
        if r.shape != d.shape:
            if not crop_to_common:
                raise RuntimeError(f"Frame {idx} shape mismatch: {r.shape} vs {d.shape}")
            h = min(r.shape[0], d.shape[0])
            w = min(r.shape[1], d.shape[1])
            shapes.append({"frame": idx, "ref_shape": list(r.shape), "dist_shape": list(d.shape), "cropped_to": [h, w, 3]})
            r = r[:h, :w]
            d = d[:h, :w]
        diff = r.astype(np.float32) - d.astype(np.float32)
        mse = float(np.mean(diff * diff))
        psnrs.append(float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse)))
        idx += 1
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
        "shape_adjustments": shapes,
    }


def summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for m in metrics:
        by_dataset.setdefault(m["dataset"], []).append(m)
    ds_summary: dict[str, Any] = {}
    for ds, vals in sorted(by_dataset.items()):
        fp = [m["fp16_vs_gt_psnr_db"] for m in vals if m.get("fp16_vs_gt_psnr_db") is not None]
        ptq = [m["a4w4_qao_vs_gt_psnr_db"] for m in vals if m.get("a4w4_qao_vs_gt_psnr_db") is not None]
        delta = [m["a4w4_qao_minus_fp16_psnr_db"] for m in vals if m.get("a4w4_qao_minus_fp16_psnr_db") is not None]
        ds_summary[ds] = {
            "clips": len(vals),
            "fp16_vs_gt_mean_psnr_db": float(np.mean(fp)) if fp else None,
            "a4w4_qao_vs_gt_mean_psnr_db": float(np.mean(ptq)) if ptq else None,
            "a4w4_qao_minus_fp16_mean_psnr_db": float(np.mean(delta)) if delta else None,
        }
    return {
        "datasets": ds_summary,
        "clips": len(metrics),
        "overall_a4w4_qao_minus_fp16_mean_psnr_db": float(np.mean([m["a4w4_qao_minus_fp16_psnr_db"] for m in metrics if m.get("a4w4_qao_minus_fp16_psnr_db") is not None])) if metrics else None,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    datasets = {
        "UDM10": Path(args.udm10_root),
        "REDS30": Path(args.reds30_root),
        "MVSR4x": Path(args.mvsr4x_root),
    }
    eval_items: list[dict[str, Any]] = []
    prep: list[dict[str, Any]] = []
    for name, root in datasets.items():
        items, records = discover_named_pairs(name, root, out_dir, args.frames, args.limit_per_dataset, args.seed)
        eval_items.extend(items)
        prep.extend(records)

    runs: list[dict[str, Any]] = []
    qckpt = Path(args.quant_checkpoint)
    for item in eval_items:
        ds = item["dataset"]
        clip = item["clip"]
        fp_out = out_dir / "videos" / ds / f"{clip}_fp16_first{args.frames}.mp4"
        q_out = out_dir / "videos" / ds / f"{clip}_a4w4_qao_first{args.frames}.mp4"
        runs.append({
            "dataset": ds,
            "clip": clip,
            "kind": "fp16",
            "input_video": item["input_video"],
            "reference_video": item["reference_video"],
            "output_video": str(fp_out),
            "command": build_cli_command(args.python_bin, Path(item["input_video"]), fp_out, args.frames, args.seed, "None", None),
        })
        runs.append({
            "dataset": ds,
            "clip": clip,
            "kind": "a4w4_qao",
            "input_video": item["input_video"],
            "reference_video": item["reference_video"],
            "output_video": str(q_out),
            "quantize_mode": "FakeQuant_A4W4",
            "quant_checkpoint": str(qckpt),
            "command": build_cli_command(args.python_bin, Path(item["input_video"]), q_out, args.frames, args.seed, "FakeQuant_A4W4", qckpt),
        })
    return {
        "schema_version": "flashvsr.lsgquant.pr9_paper_comparable_benchmark.v1",
        "paper": "LSGQuant arXiv:2602.03182v1",
        "scope": "WanVideoDiT Linear layers only; Wan VAE / decoder remain unquantized.",
        "benchmark_contract": {
            "datasets": PAPER_DATASETS,
            "metrics_implemented": ["PSNR vs GT", "A4W4-QAO minus FP16 PSNR delta"],
            "not_yet_implemented": ["SSIM", "LPIPS", "DISTS", "MANIQA", "CLIP-IQA", "MUSIQ", "Ewarp*", "DOVER"],
            "frames_per_clip": args.frames,
            "limit_per_dataset": args.limit_per_dataset,
            "note": "Paper-comparable dataset split and FP16/PTQ pairing; metric surface currently PSNR only unless optional IQA/VQA packages are later wired.",
        },
        "execution": {"out_dir": str(out_dir), "seed": args.seed, "execute": args.execute, "python_bin": args.python_bin},
        "checkpoints": {"a4w4_qao": str(qckpt)},
        "datasets": {k: str(v) for k, v in datasets.items()},
        "evaluation": eval_items,
        "preparation": prep,
        "runs": runs,
    }


def execute_runs(manifest: dict[str, Any], repo_root: Path, out_dir: Path) -> None:
    for run in manifest["runs"]:
        log_path = out_dir / "reports" / run["dataset"] / f"{Path(run['output_video']).stem}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        Path(run["output_video"]).parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            proc = subprocess.run(run["command"], cwd=repo_root, stdout=log, stderr=subprocess.STDOUT, text=True)
        exists = Path(run["output_video"]).exists()
        run["returncode"] = proc.returncode if exists else (proc.returncode or 1)
        run["output_exists"] = exists
        run["log"] = str(log_path)
        if not exists:
            run["error"] = "command completed without producing output_video"


def add_metrics(manifest: dict[str, Any], out_dir: Path) -> None:
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for run in manifest["runs"]:
        by_key.setdefault((run["dataset"], run["clip"]), {})[run["kind"]] = run
    metrics: list[dict[str, Any]] = []
    for (dataset, clip), kinds in sorted(by_key.items()):
        fp = kinds.get("fp16")
        q = kinds.get("a4w4_qao")
        if not fp or not q:
            continue
        ref = Path(fp["reference_video"])
        fp_out = Path(fp["output_video"])
        q_out = Path(q["output_video"])
        row: dict[str, Any] = {"dataset": dataset, "clip": clip, "reference_video": str(ref), "fp16_video": str(fp_out), "a4w4_qao_video": str(q_out)}
        if ref.exists() and fp_out.exists():
            m = compute_video_psnr(ref, fp_out)
            row["fp16_vs_gt_psnr_db"] = m["psnr_avg_db"]
            row["fp16_vs_gt"] = m
        if ref.exists() and q_out.exists():
            m = compute_video_psnr(ref, q_out)
            row["a4w4_qao_vs_gt_psnr_db"] = m["psnr_avg_db"]
            row["a4w4_qao_vs_gt"] = m
        if fp_out.exists() and q_out.exists():
            m = compute_video_psnr(fp_out, q_out)
            row["a4w4_qao_vs_fp16_psnr_db"] = m["psnr_avg_db"]
            row["a4w4_qao_vs_fp16"] = m
        if row.get("fp16_vs_gt_psnr_db") is not None and row.get("a4w4_qao_vs_gt_psnr_db") is not None:
            row["a4w4_qao_minus_fp16_psnr_db"] = row["a4w4_qao_vs_gt_psnr_db"] - row["fp16_vs_gt_psnr_db"]
        metrics.append(row)
        (metrics_dir / f"{dataset}_{clip}_metrics.json").write_text(json.dumps(row, indent=2))
    manifest["metrics"] = metrics
    manifest["summary"] = summarize_metrics(metrics)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run PR9 LSGQuant A4W4-QAO paper-comparable benchmark")
    ap.add_argument("--udm10_root", default="/home/user/apps/FlashVSRptq/FlashVSR_Integrated/datasets/test/UDM10")
    ap.add_argument("--reds30_root", default="/home/user/data/REDs/val/val_sharp")
    ap.add_argument("--mvsr4x_root", default="/home/user/apps/FlashVSRptq/FlashVSR_Integrated/datasets/test/MVSR4x")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--quant_checkpoint", required=True)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--limit_per_dataset", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--python_bin", default=DEFAULT_VENV)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--compute_existing_metrics", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    repo_root = Path(__file__).resolve().parents[2]
    if args.execute:
        execute_runs(manifest, repo_root, out_dir)
    if args.execute or args.compute_existing_metrics:
        add_metrics(manifest, out_dir)
    manifest_path = out_dir / "lsgquant_pr9_paper_benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"manifest": str(manifest_path), "runs": len(manifest["runs"]), "metrics": len(manifest.get("metrics", [])), "summary": manifest.get("summary")}, indent=2))


if __name__ == "__main__":
    main()
