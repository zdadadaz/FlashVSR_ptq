"""Prepare DiT-ready QAT samples from training videos.

This is a lightweight Person A September data-prep step for environments where
student v0 is not available yet.  It samples short clips from videos, converts
RGB frames into deterministic DiT-shaped pseudo-latents, and writes a JSONL
manifest consumable by `finetune_fakequant_dit.py`.

The generated tensors are DiT-side only; Wan VAE stays unquantized and is not
used by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def find_videos(root: str | Path, limit: int | None = None) -> list[Path]:
    root = Path(root)
    videos = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    return videos[:limit] if limit is not None and limit > 0 else videos


def _read_evenly_spaced_frames(video_path: str | Path, num_frames: int, image_size: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")
    indices = np.linspace(0, max(total - 1, 0), num=num_frames, dtype=np.int64)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()
    if len(frames) != num_frames:
        raise ValueError(f"Only decoded {len(frames)}/{num_frames} frames from {video_path}")
    return np.stack(frames, axis=0)


def frames_to_pseudo_latent(frames: np.ndarray, latent_channels: int = 16) -> torch.Tensor:
    """Convert RGB frames [F,H,W,3] to DiT-shaped latent [1,C,F,H,W]."""

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames [F,H,W,3], got {frames.shape}")
    chw = frames.transpose(3, 0, 1, 2)  # [3,F,H,W]
    latent = np.zeros((latent_channels, frames.shape[0], frames.shape[1], frames.shape[2]), dtype=np.float32)
    for c in range(latent_channels):
        base = chw[c % 3]
        # Center around 0 and give repeated RGB channels slightly different
        # scales so fake-quant sees non-identical channel ranges.
        latent[c] = (base - 0.5) * (1.0 + 0.03 * (c // 3))
    return torch.from_numpy(latent).unsqueeze(0).contiguous()


def deterministic_context(seed: int, tokens: int = 10, dim: int = 1536) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(1, tokens, dim, generator=gen, dtype=torch.float32) * 0.02


def write_manifest_from_videos(
    video_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    max_videos: int = 16,
    frames: int = 4,
    latent_size: tuple[int, int] = (16, 16),
    context_tokens: int = 10,
    context_dim: int = 1536,
) -> dict:
    videos = find_videos(video_dir, max_videos)
    if not videos:
        raise FileNotFoundError(f"No videos found under {video_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, video in enumerate(videos):
        frame_arr = _read_evenly_spaced_frames(video, frames, latent_size)
        sample = {
            "x": frames_to_pseudo_latent(frame_arr),
            "timestep": torch.tensor([1000.0], dtype=torch.float32),
            "context": deterministic_context(idx, tokens=context_tokens, dim=context_dim),
            "source_video": str(video),
        }
        sample_path = output_dir / f"qat_sample_{idx:04d}.pt"
        torch.save(sample, sample_path)
        rows.append({"sample": str(sample_path), "source_video": str(video)})

    with manifest_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = {
        "video_dir": str(video_dir),
        "manifest": str(manifest_path),
        "sample_dir": str(output_dir),
        "samples": len(rows),
        "frames": frames,
        "latent_size": list(latent_size),
        "context_tokens": context_tokens,
        "context_dim": context_dim,
    }
    (manifest_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
    return summary


def parse_hw(value: str) -> tuple[int, int]:
    h, w = value.lower().split("x", 1)
    return int(h), int(w)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FlashVSR DiT QAT manifest from videos")
    parser.add_argument("--video_dir", default="datasets/train")
    parser.add_argument("--output_dir", default="outputs/qat/video_samples")
    parser.add_argument("--manifest", default="outputs/qat/video_samples/manifest.jsonl")
    parser.add_argument("--max_videos", type=int, default=16)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--latent_size", default="16x16")
    parser.add_argument("--context_tokens", type=int, default=10)
    parser.add_argument("--context_dim", type=int, default=1536)
    args = parser.parse_args()

    summary = write_manifest_from_videos(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        max_videos=args.max_videos,
        frames=args.frames,
        latent_size=parse_hw(args.latent_size),
        context_tokens=args.context_tokens,
        context_dim=args.context_dim,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
