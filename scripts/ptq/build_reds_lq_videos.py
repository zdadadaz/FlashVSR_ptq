#!/usr/bin/env python3
"""
Generate LQ videos from REDS train_sharp (PNG sequences) via OpenCV bicubic ×4 downsample.

Output: mp4 (libx264, crf=20) at native fps (or 30 fps if unknown).

Usage:
    python build_reds_lq_videos.py --hq_root /home/user/data/REDs/train/train_sharp \
        --lq_root /home/user/data/REDs/train/LQ --scale 4 --fps 30
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import cv2
from tqdm import tqdm


def process_one_sequence(args):
    seq_dir, lq_root, scale, fps, crf, ext = args
    seq_name = os.path.basename(seq_dir.rstrip("/"))
    pngs = sorted(glob.glob(os.path.join(seq_dir, f"*.{ext}")))
    if not pngs:
        return seq_name, "no_pngs"
    out_path = os.path.join(lq_root, f"{seq_name}.mp4")
    if os.path.exists(out_path):
        return seq_name, "exists"

    # Read first frame to derive LQ size
    first = cv2.imread(pngs[0])
    if first is None:
        return seq_name, "read_fail"
    H, W = first.shape[:2]
    lq_h, lq_w = H // scale, W // scale
    if lq_h < 16 or lq_w < 16:
        return seq_name, f"too_small_{lq_w}x{lq_h}"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely supported, no encoder deps
    writer = cv2.VideoWriter(out_path, fourcc, fps, (lq_w, lq_h))
    if not writer.isOpened():
        return seq_name, "writer_open_fail"
    for p in pngs:
        img = cv2.imread(p)
        if img is None:
            continue
        lq = cv2.resize(img, (lq_w, lq_h), interpolation=cv2.INTER_CUBIC)
        writer.write(lq)
    writer.release()
    return seq_name, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hq_root", required=True, help="Path to REDS train_sharp (with 000/, 001/, ... subdirs)")
    parser.add_argument("--lq_root", required=True, help="Output dir for LQ mp4 files")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=20, help="Ignored (using mp4v fourcc); included for future use")
    parser.add_argument("--ext", type=str, default="png", help="Input frame extension")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    args = parser.parse_args()

    os.makedirs(args.lq_root, exist_ok=True)

    seq_dirs = sorted(glob.glob(os.path.join(args.hq_root, "*")))
    seq_dirs = [s for s in seq_dirs if os.path.isdir(s)]
    print(f"Found {len(seq_dirs)} sequences under {args.hq_root}")
    if not seq_dirs:
        print("No sequences found. Abort.")
        sys.exit(1)

    jobs = [(s, args.lq_root, args.scale, args.fps, args.crf, args.ext) for s in seq_dirs]
    print(f"Processing {len(jobs)} sequences with {args.workers} workers…")
    results = {"ok": 0, "exists": 0, "fail": 0}
    fails = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for seq_name, status in tqdm(ex.map(process_one_sequence, jobs, chunksize=4), total=len(jobs)):
            if status in results:
                results[status] += 1
            else:
                results["fail"] += 1
                fails.append((seq_name, status))
    print(f"\nResults: {results}")
    if fails:
        print(f"Failures (first 10): {fails[:10]}")


if __name__ == "__main__":
    main()
