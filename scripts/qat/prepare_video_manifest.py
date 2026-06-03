"""Prepare DiT-ready QAT samples from training videos.

This is a lightweight Person A September data-prep step for environments where
student v0 is not available yet.  It samples short clips from videos, converts
RGB frames into deterministic DiT-shaped pseudo-latents, and writes a JSONL
manifest consumable by `finetune_fakequant_dit.py`.

The generated tensors are DiT-side only; Wan VAE stays unquantized.  By
default the script emits lightweight pseudo-latents, with an opt-in real VAE
latent path for distribution checks/full QAT data prep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PAIR_DIRS = (("LQ", "GT"), ("LQ-Video", "GT-Video"), ("LR", "HR"), ("lowres", "GT"), ("LQ", "HR"))


def _is_image_sequence_dir(path: Path) -> bool:
    return path.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in path.iterdir() if p.is_file())


def _paired_sources_by_stem(root: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in VIDEO_EXTS:
            sources[p.stem] = p
        elif _is_image_sequence_dir(p):
            sources[p.name] = p
    return sources


def discover_paired_lq_gt(root: str | Path, limit: int | None = None) -> list[dict]:
    """Discover paired LQ/GT clips in common video or image-sequence layouts."""

    root = Path(root)
    pairs: list[dict] = []
    for lq_name, gt_name in PAIR_DIRS:
        lq_dir = root / lq_name
        gt_dir = root / gt_name
        if not lq_dir.exists() or not gt_dir.exists():
            continue
        gt_by_stem = _paired_sources_by_stem(gt_dir)
        for stem, lq in sorted(_paired_sources_by_stem(lq_dir).items()):
            gt = gt_by_stem.get(stem)
            if gt is None:
                continue
            pairs.append({"name": stem, "lq": lq, "gt": gt})
            if limit is not None and limit > 0 and len(pairs) >= limit:
                return pairs
    return pairs


def downsample_frames(frames: np.ndarray, scale: int = 4) -> np.ndarray:
    """Downsample video frames [F,H,W,C] by integer scale using area filtering."""

    if scale <= 1:
        return frames
    if frames.ndim != 4:
        raise ValueError(f"Expected frames [F,H,W,C], got {frames.shape}")
    h, w = frames.shape[1:3]
    out_h = max(1, h // scale)
    out_w = max(1, w // scale)
    return np.stack([
        cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(frames.dtype)
        for frame in frames
    ], axis=0)


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


def _read_evenly_spaced_image_sequence(seq_dir: str | Path, num_frames: int, image_size: tuple[int, int]) -> np.ndarray:
    seq_dir = Path(seq_dir)
    images = sorted(p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise ValueError(f"Image sequence has no frames: {seq_dir}")
    indices = np.linspace(0, len(images) - 1, num=num_frames, dtype=np.int64)
    frames = []
    for idx in indices:
        image_path = images[int(idx)]
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot read image frame: {image_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.float32) / 255.0)
    return np.stack(frames, axis=0)


def read_evenly_spaced_frames(source_path: str | Path, num_frames: int, image_size: tuple[int, int]) -> np.ndarray:
    source_path = Path(source_path)
    if source_path.is_dir():
        return _read_evenly_spaced_image_sequence(source_path, num_frames, image_size)
    return _read_evenly_spaced_frames(source_path, num_frames, image_size)


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


def load_teacher_text_context(
    prompt_context_path: str | Path = "posi_prompt.pth",
    checkpoint: str | Path = "",
    device: str = "cuda",
) -> torch.Tensor:
    """Load real FlashVSR prompt/content context and embed it into DiT context space.

    `posi_prompt.pth` stores the raw T5-style context ([B,S,4096]). WanModel.forward
    expects already text-embedded context ([B,S,dim]), so when a checkpoint is
    provided we instantiate the v1.1 DiT and run its `text_embedding` once.
    If the tensor is already embedded (last dim != 4096), it is returned as-is.
    """

    ctx = torch.load(str(prompt_context_path), map_location="cpu", weights_only=False)
    if isinstance(ctx, dict):
        ctx = ctx.get("context", ctx.get("prompt_emb_posi", ctx.get("positive", ctx)))
    if not isinstance(ctx, torch.Tensor):
        raise TypeError(f"Prompt context file must contain a Tensor, got {type(ctx)}")
    ctx = ctx.to(torch.float32)
    if ctx.shape[-1] != 4096:
        return ctx.cpu().contiguous()
    if not checkpoint:
        raise ValueError("teacher_text_embedding context_source with raw 4096-dim prompt requires --context_checkpoint")

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from scripts.ptq.fakequant_convert import build_dit, load_checkpoint

    dit = load_checkpoint(str(checkpoint), build_dit()).to(device=device, dtype=torch.float32).eval()
    with torch.no_grad():
        embedded = dit.text_embedding(ctx.to(device=device))
    return embedded.detach().cpu().contiguous()


def frames_to_vae_latent(
    frames: np.ndarray,
    vae_path: str | Path,
    vae_model: str = "Wan2.1",
    device: str = "cuda",
    tiled: bool = False,
) -> torch.Tensor:
    """Encode RGB frames [F,H,W,3] into DiT latent [1,16,F,H,W] with the real Wan VAE."""

    from src.models.wan_video_vae import create_video_vae

    vae_type = {"Wan2.1": "wan2.1", "Wan2.2": "wan2.2", "LightVAE_W2.1": "lightx2v"}.get(vae_model, vae_model)
    vae = create_video_vae(vae_type).eval().to(device)
    state = torch.load(str(vae_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        vae.load_state_dict(state, strict=False)
    except RuntimeError:
        vae.load_state_dict(vae.state_dict_converter().from_civitai(state), strict=False)
    # Wan VAE expects a list of videos, where each video is [C,T,H,W].
    # Keep the temporal axis intact instead of encoding frames independently, so
    # the manifest reflects the true VAE latent distribution seen by FlashVSR.
    video_tensor = torch.from_numpy(frames.transpose(3, 0, 1, 2)).to(torch.float32).mul(2.0).sub(1.0)
    with torch.no_grad():
        encoded = vae.encode([video_tensor], device=device, tiled=tiled)
    # Expected output is [B=1,C,T,H,W].  Retain a compatibility fallback for
    # older wrappers that returned per-frame [F,C,H,W] tensors.
    if encoded.dim() == 5 and encoded.shape[0] == 1:
        latent = encoded
    elif encoded.dim() == 4:
        latent = encoded.permute(1, 0, 2, 3).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected VAE latent shape: {tuple(encoded.shape)}")
    return latent.detach().cpu().contiguous()


def write_manifest_from_videos(
    video_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    max_videos: int = 16,
    frames: int = 4,
    latent_size: tuple[int, int] = (16, 16),
    context_tokens: int = 10,
    context_dim: int = 1536,
    downsample_lq_scale: int = 1,
    prefer_paired_lq_gt: bool = True,
    latent_source: str = "pseudo",
    context_source: str = "deterministic",
    prompt_context_path: str = "posi_prompt.pth",
    context_checkpoint: str = "",
    vae_path: str = "",
    vae_model: str = "Wan2.1",
    device: str = "cuda",
    tiled_vae: bool = False,
) -> dict:
    if latent_source == "vae" and not vae_path:
        raise ValueError("--latent_source vae requires --vae_path")
    if latent_source not in {"pseudo", "vae"}:
        raise ValueError(f"Unsupported latent_source: {latent_source}")
    teacher_context = None
    if context_source == "teacher_text_embedding":
        teacher_context = load_teacher_text_context(prompt_context_path, checkpoint=context_checkpoint, device=device)
        context_tokens = int(teacher_context.shape[1])
        context_dim = int(teacher_context.shape[2])
    elif context_source != "deterministic":
        raise ValueError(f"Unsupported context_source: {context_source}")
    paired = discover_paired_lq_gt(video_dir, max_videos) if prefer_paired_lq_gt else []
    if paired:
        sources = paired
    else:
        videos = find_videos(video_dir, max_videos)
        sources = [{"name": video.stem, "lq": video, "gt": None} for video in videos]
    if not sources:
        raise FileNotFoundError(f"No videos found under {video_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, source in enumerate(sources):
        lq_video = Path(source["lq"])
        frame_arr = read_evenly_spaced_frames(lq_video, frames, latent_size)
        if source.get("gt") is not None and downsample_lq_scale > 1:
            frame_arr = downsample_frames(frame_arr, downsample_lq_scale)
        sample = {
            "x": frames_to_vae_latent(frame_arr, vae_path=vae_path, vae_model=vae_model, device=device, tiled=tiled_vae)
            if latent_source == "vae"
            else frames_to_pseudo_latent(frame_arr),
            "timestep": torch.tensor([1000.0], dtype=torch.float32),
            "context": teacher_context.clone() if teacher_context is not None else deterministic_context(idx, tokens=context_tokens, dim=context_dim),
            "source_video": str(lq_video),
            "lq_source": str(lq_video),
            "gt_source": str(source["gt"]) if source.get("gt") is not None else "",
            "downsample_lq_scale": int(downsample_lq_scale if source.get("gt") is not None else 1),
            "latent_source": latent_source,
            "context_source": context_source,
        }
        sample_path = output_dir / f"qat_sample_{idx:04d}.pt"
        torch.save(sample, sample_path)
        row = {"sample": str(sample_path), "source_video": str(lq_video), "lq_source": str(lq_video)}
        if source.get("gt") is not None:
            row["gt_source"] = str(source["gt"])
        rows.append(row)

    with manifest_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = {
        "video_dir": str(video_dir),
        "manifest": str(manifest_path),
        "sample_dir": str(output_dir),
        "samples": len(rows),
        "paired_samples": sum(1 for row in rows if "gt_source" in row),
        "frames": frames,
        "latent_size": list(latent_size),
        "context_tokens": context_tokens,
        "context_dim": context_dim,
        "downsample_lq_scale": downsample_lq_scale,
        "latent_source": latent_source,
        "context_source": context_source,
        "prompt_context_path": str(prompt_context_path) if context_source == "teacher_text_embedding" else "",
        "context_checkpoint": str(context_checkpoint) if context_source == "teacher_text_embedding" else "",
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
    parser.add_argument("--downsample_lq_scale", type=int, default=4, help="When paired LQ/GT is found, downsample LQ frames by this scale before latent prep")
    parser.add_argument("--no_paired_lq_gt", action="store_true", help="Disable paired LQ/GT discovery and use flat video fallback")
    parser.add_argument("--latent_source", default="pseudo", choices=["pseudo", "vae"])
    parser.add_argument("--context_source", default="deterministic", choices=["deterministic", "teacher_text_embedding"])
    parser.add_argument("--prompt_context_path", default="posi_prompt.pth", help="Real FlashVSR prompt/content tensor for --context_source teacher_text_embedding")
    parser.add_argument("--context_checkpoint", default="", help="DiT checkpoint used to text-embed raw 4096-dim prompt context")
    parser.add_argument("--vae_path", default="", help="Required when --latent_source vae")
    parser.add_argument("--vae_model", default="Wan2.1", choices=["Wan2.1", "Wan2.2", "LightVAE_W2.1"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tiled_vae", action="store_true")
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
        downsample_lq_scale=args.downsample_lq_scale,
        prefer_paired_lq_gt=not args.no_paired_lq_gt,
        latent_source=args.latent_source,
        context_source=args.context_source,
        prompt_context_path=args.prompt_context_path,
        context_checkpoint=args.context_checkpoint,
        vae_path=args.vae_path,
        vae_model=args.vae_model,
        device=args.device,
        tiled_vae=args.tiled_vae,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
