#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref')
    ap.add_argument('dist')
    ap.add_argument('--out-json', default=None)
    args = ap.parse_args()

    cap_ref = cv2.VideoCapture(args.ref)
    cap_dist = cv2.VideoCapture(args.dist)
    if not cap_ref.isOpened():
        raise RuntimeError(f'Cannot open ref video: {args.ref}')
    if not cap_dist.isOpened():
        raise RuntimeError(f'Cannot open dist video: {args.dist}')

    psnrs = []
    frame_idx = 0
    while True:
        ok_r, ref = cap_ref.read()
        ok_d, dist = cap_dist.read()
        if not ok_r or not ok_d:
            break
        if ref.shape != dist.shape:
            raise RuntimeError(f'Frame {frame_idx} shape mismatch: {ref.shape} vs {dist.shape}')
        diff = ref.astype(np.float32) - dist.astype(np.float32)
        mse = float(np.mean(diff * diff))
        psnr = float('inf') if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
        psnrs.append(psnr)
        frame_idx += 1

    cap_ref.release()
    cap_dist.release()
    if not psnrs:
        raise RuntimeError('No comparable frames')

    result = {
        'ref': args.ref,
        'dist': args.dist,
        'frames': len(psnrs),
        'psnr_avg_db': float(np.mean(psnrs)),
        'psnr_min_db': float(np.min(psnrs)),
        'psnr_max_db': float(np.max(psnrs)),
        'psnr_per_frame_db': psnrs,
    }
    print(json.dumps(result, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
