#!/usr/bin/env python3
"""QBasicVSR-inspired video complexity metric for FlashVSR PTQ.

First implementation uses a labelled proxy backend: spatial finite differences
plus frame-difference temporal motion.  It intentionally does not claim SPyNet /
optical-flow parity with the paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

SCHEMA_VERSION = "flashvsr.qbasicvsr.video_complexity.v1"


def _to_gray(frame: np.ndarray) -> np.ndarray:
    arr = frame.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    if arr.ndim == 3:
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return arr.astype(np.float32)


def compute_frame_spatial_score(frame: np.ndarray) -> float:
    gray = _to_gray(frame)
    dx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    dy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    return float((dx + dy) * 1e3)


def compute_proxy_temporal_score(prev: np.ndarray, cur: np.ndarray, *, gamma: float = 200.0) -> float:
    a = _to_gray(prev)
    b = _to_gray(cur)
    diff = np.abs(b - a)
    mag = float(diff.mean())
    if diff.shape[1] > 1:
        gx = np.abs(np.diff(diff, axis=1)).mean()
    else:
        gx = 0.0
    if diff.shape[0] > 1:
        gy = np.abs(np.diff(diff, axis=0)).mean()
    else:
        gy = 0.0
    return float(mag + gamma * (gx + gy))


def score_frames(frames: Iterable[np.ndarray], *, gamma: float = 200.0, lambda_spatiotemporal: float = 10.0) -> dict:
    seq = list(frames)
    if not seq:
        raise ValueError("At least one frame is required")
    spatial = [compute_frame_spatial_score(f) for f in seq]
    temporal = [compute_proxy_temporal_score(a, b, gamma=gamma) for a, b in zip(seq[:-1], seq[1:])]
    spatial_mean = float(np.mean(spatial))
    temporal_mean = float(np.mean(temporal)) if temporal else 0.0
    return {
        "spatial_mean": spatial_mean,
        "temporal_mean": temporal_mean,
        "c_video": float(spatial_mean + lambda_spatiotemporal * temporal_mean),
        "num_frames": len(seq),
    }


def read_video_frames(path: str | Path, *, max_frames: int = 16) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("Cannot percentile empty values")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def assign_video_bit_factor(c_video: float, *, lower: float, upper: float) -> int:
    if c_video <= lower:
        return -1
    if c_video >= upper:
        return 1
    return 0


def build_complexity_report(inputs: list[str], *, p_v: float = 10.0, gamma: float = 200.0, lambda_spatiotemporal: float = 10.0, max_frames: int = 16, flow_backend: str = "proxy") -> dict:
    if flow_backend != "proxy":
        raise ValueError("Only flow_backend='proxy' is implemented in this first FlashVSR mapping")
    videos = []
    for inp in inputs:
        entry = score_frames(read_video_frames(inp, max_frames=max_frames), gamma=gamma, lambda_spatiotemporal=lambda_spatiotemporal)
        entry["path"] = inp
        videos.append(entry)
    c_values = [float(v["c_video"]) for v in videos]
    lower = percentile(c_values, p_v)
    upper = percentile(c_values, 100.0 - p_v)
    for v in videos:
        v["video_bit_factor"] = assign_video_bit_factor(float(v["c_video"]), lower=lower, upper=upper)
    return {
        "schema_version": SCHEMA_VERSION,
        "flow_backend": flow_backend,
        "gamma": float(gamma),
        "lambda": float(lambda_spatiotemporal),
        "videos": videos,
        "thresholds": {"p_v": float(p_v), "l_v2b": lower, "u_v2b": upper},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute QBasicVSR-style FlashVSR video complexity")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--p_v", type=float, default=10.0)
    ap.add_argument("--gamma", type=float, default=200.0)
    ap.add_argument("--lambda_spatiotemporal", type=float, default=10.0)
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--flow_backend", default="proxy", choices=["proxy"])
    args = ap.parse_args()
    report = build_complexity_report(args.inputs, p_v=args.p_v, gamma=args.gamma, lambda_spatiotemporal=args.lambda_spatiotemporal, max_frames=args.max_frames, flow_backend=args.flow_backend)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[qbasicvsr] video complexity → {out}")


if __name__ == "__main__":
    main()
