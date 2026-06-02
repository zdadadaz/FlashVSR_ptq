"""
PTQ Calibration for FakeQuant — FlashVSR DiT.

Collects per-layer activation statistics (min/max per channel) via forward hooks,
computes asymmetric per-channel scales, and saves the calibration cache to JSON.
The cache is consumed by fakequant_convert.py to configure FakeQuantLinear layers.

Supports modes: a16w8, a8w8, a16w4, a8w4
  - a16: no activation quantization (acts pass through in float)
  - a8:  activation quantized to int8 via per-channel scale
  - w8/w4: weight quantization depth (int8 / packed-int4)
"""

import argparse
import glob
import json
import math
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.wan_video_dit import WanModel
from src.models.quantization.fakequant import collect_activation_stats_fakequant


VIDEO_EXTENSIONS = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm")


def discover_calibration_videos(dataset_train: str, num_videos: int, seed: int) -> list[str]:
    """Recursively sample calibration videos from datasets/train."""
    candidates = []
    for ext in VIDEO_EXTENSIONS:
        candidates.extend(glob.glob(os.path.join(dataset_train, "**", ext), recursive=True))
    candidates = sorted(set(candidates))
    if not candidates:
        raise RuntimeError(f"No calibration videos found under: {dataset_train}")
    rng = random.Random(seed)
    if len(candidates) <= num_videos:
        return candidates
    return sorted(rng.sample(candidates, num_videos))


# =============================================================================
# Calibration dataset — samples latent-space frames from a video file
# =============================================================================

def sample_latents_from_video(
    video_path: str,
    num_frames: int = 32,
    latent_channels: int = 16,
    frame_size: tuple = (60, 80),  # (H, W) — latent H = video_H // 4
    scale: int = 4,
    vae_path: str = None,
    vae_model: str = "Wan2.1",
):
    """
    Read a video, decode frames, downscale by `scale`, and produce real latent
    tensors by encoding frames through the VAE encoder.

    This replaces the prior RGB-replication approach which produced unrealistic
    activation distributions. Using real VAE encoding preserves the actual
    distribution of latent activations seen during inference.

    Args:
        video_path:    Path to video file
        num_frames:    Number of frames to sample
        latent_channels: Expected channel count (16 for FlashVSR)
        frame_size:    (H, W) of latent spatial dimensions
        scale:         VAE downscale factor (4 = 4x SR)
        vae_path:      Path to VAE checkpoint (optional — uses default if None)
        vae_model:     VAE model variant ("Wan2.1", "Wan2.2", "LightVAE_W2.1", etc.)

    Returns:
        latents: torch.Tensor of shape (F, C, H, W) in [−1, 1] range
        fps:    float, video FPS (for metadata only)
    """
    import cv2
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    # Lazy import to avoid pulling VAE dependencies when not needed
    from src.models.wan_video_vae import WanVideoVAE, Wan22VideoVAE, LightX2VVAE, VAE_FULL_DIM, VAE_LIGHT_DIM, VAE_Z_DIM, create_video_vae

    VAE_MODEL_MAP = {
        "Wan2.1":        {"class": WanVideoVAE,  "file": "Wan2.1_VAE.pth",   "dim": VAE_FULL_DIM,  "z_dim": VAE_Z_DIM, "use_full_arch": False},
        "Wan2.2":        {"class": Wan22VideoVAE, "file": "Wan2.2_VAE.pth",   "dim": VAE_FULL_DIM,  "z_dim": VAE_Z_DIM, "use_full_arch": False},
        "LightVAE_W2.1": {"class": LightX2VVAE,  "file": "lightvaew2_1.pth", "dim": VAE_LIGHT_DIM, "z_dim": VAE_Z_DIM, "use_full_arch": True},
        "TAE_W2.2":      {"class": Wan22VideoVAE, "file": "taew2_2.safetensors", "dim": VAE_FULL_DIM, "z_dim": VAE_Z_DIM, "use_full_arch": False},
        "LightTAE_HY1.5": {"class": LightX2VVAE,  "file": "lighttaehy1_5.pth", "dim": VAE_LIGHT_DIM, "z_dim": VAE_Z_DIM, "use_full_arch": True},
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    target_h, target_w = frame_size

    while len(frames) < num_frames and len(frames) < total:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
            continue
        # resize → H/scale, W/scale (latent spatial dims)
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        # BGR → RGB, [0,255] → [−1, 1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = rgb * 2.0 - 1.0  # normalise to [-1, 1]
        # (H,W,3) → (3,H,W)
        chw = rgb.transpose(2, 0, 1)
        frames.append(chw)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from {video_path}")

    # Pad/trim to num_frames
    batch = torch.from_numpy(np.stack(frames, axis=0))  # (F, 3, H, W)
    if batch.shape[0] < num_frames:
        pad = num_frames - batch.shape[0]
        batch = torch.cat([batch, batch[-1:].repeat(pad, 1, 1, 1)], dim=0)
    batch = batch[:num_frames]

    # ---- Encode with VAE to get real latents ----
    if vae_path and os.path.exists(vae_path):
        vae_cfg = VAE_MODEL_MAP.get(vae_model, VAE_MODEL_MAP["Wan2.1"])
        vae_class = vae_cfg["class"]
        vae_z_dim = vae_cfg["z_dim"]
        vae_dim = vae_cfg["dim"]
        use_full_arch = vae_cfg.get("use_full_arch", False)

        if vae_class == LightX2VVAE:
            vae = LightX2VVAE(z_dim=vae_z_dim, dim=vae_dim, use_full_arch=use_full_arch)
        elif vae_class == Wan22VideoVAE:
            vae = Wan22VideoVAE(z_dim=vae_z_dim, dim=vae_dim)
        else:
            vae = WanVideoVAE(z_dim=vae_z_dim, dim=vae_dim)

        if vae_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            vae_sd = load_file(vae_path)
        else:
            vae_sd = torch.load(vae_path, map_location="cpu", weights_only=False)

        load_result = vae.load_state_dict(vae_sd, strict=False)
        if load_result.missing_keys:
            print(f"  VAE missing keys: {load_result.missing_keys[:3]}...")
        if load_result.unexpected_keys:
            print(f"  VAE unexpected keys: {load_result.unexpected_keys[:3]}...")

        vae.eval()
        vae.cuda()

        # Encode each frame group through VAE
        # VAE.encode expects: list of (3,H,W) tensors or batched (B,3,H,W) per video.
        # We pass a list of individual (3,H,W) frames — one "video" = one frame.
        # Process in chunks to avoid OOM.
        latent_list = []
        B_encode = 4  # encode 4 frames at a time
        for start in range(0, num_frames, B_encode):
            end = min(start + B_encode, num_frames)
            chunk = batch[start:end]             # (B, 3, H, W) with B ≤ 4
            # vae.encode expects list of (3,H,W) tensors — split batch into list
            frame_list = [chunk[i] for i in range(chunk.shape[0])]  # list of (3,H,W)
            vae.cuda()
            hidden_states = vae.encode(frame_list, device="cuda", tiled=True)
            vae.cpu()
            # hidden_states is (B, C, H, W) — stacked result
            if isinstance(hidden_states, (list, tuple)):
                # VAE may return a list of scales; take the finest (first)
                latent = hidden_states[0] if len(hidden_states) > 0 else hidden_states[-1]
            else:
                latent = hidden_states
            latent_list.append(latent.cpu())

        # Concatenate all chunks: (num_chunks, C, H, W) → (F', C, H, W)
        all_latents = torch.cat(latent_list, dim=0)  # (num_chunks * B, C, H, W)

        # If total frames encoded doesn't match requested, trim/pad
        if all_latents.shape[0] > num_frames:
            all_latents = all_latents[:num_frames]
        elif all_latents.shape[0] < num_frames:
            pad = num_frames - all_latents.shape[0]
            all_latents = torch.cat([all_latents, all_latents[-1:].repeat(pad, 1, 1, 1)], dim=0)

        latents = all_latents[:num_frames]
        vae.cpu()
        del vae
        torch.cuda.empty_cache()

        print(f"  VAE-encoded {num_frames} real latents  shape={latents.shape}")
    else:
        # Fallback: replicate 3 RGB channels into latent_channels by tiling + cropping.
        # This preserves the approximate dynamic range but NOT the real distribution.
        # Only used when no VAE is available — prefer providing a VAE path.
        print("  Warning: VAE path not provided or not found. Using RGB replication fallback.")
        print("  This does NOT reflect real latent distribution — provide vae_path for accurate calibration.")
        rgb3 = batch[:, :3, :, :]                         # (F, 3, H, W)
        reps = (latent_channels + 2) // 3                  # ceiling division
        latents = rgb3.repeat(1, reps, 1, 1)[:, :latent_channels, :, :]  # (F, 16, H, W)
        print(f"  Replicated {num_frames} latent tensors  shape={latents.shape}")

    print(f"  Collected {num_frames} latent tensors  shape={latents.shape}")
    return latents, fps


# =============================================================================
# Model loader
# =============================================================================

def build_dit(model_name: str = "FlashVSR-v1.1") -> WanModel:
    """
    Build WanModel architecture with FlashVSR-v1.1 DiT dimensions.

    FlashVSR-v1.1 uses ffn_dim=8960 (note: different from v1's ffn_dim=6144).
    """
    return WanModel(
        dim=1536,
        eps=1e-5,
        ffn_dim=8960,       # v1.1 specific
        freq_dim=256,
        in_dim=16,
        num_heads=12,
        num_layers=30,
        out_dim=16,
        patch_size=(1, 2, 2),
        text_dim=4096,
    )


def load_checkpoint(path: str, model: WanModel):
    """Load state_dict from safetensors or pth, stripping 'model.' prefix."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            new_sd[k[6:]] = v
        else:
            new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"  Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    return model


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PTQ Calibration for FakeQuant FlashVSR")
    parser.add_argument("--video",          type=str, default=None,
                       help="Single video file for calibration (legacy path)")
    parser.add_argument("--dataset_train",  type=str, default="datasets/train",
                       help="Training video root used for calibration when --video is not set")
    parser.add_argument("--num_videos",     type=int, default=10,
                       help="Number of random videos to sample from --dataset_train")
    parser.add_argument("--seed",           type=int, default=42,
                       help="Random seed for selecting calibration videos")
    parser.add_argument("--checkpoint",    type=str, required=True,
                       help="Path to DiT .safetensors checkpoint")
    parser.add_argument("--output_cache",  type=str, required=True,
                       help="Output path for calibration JSON cache")
    parser.add_argument("--num_samples",   type=int, default=64,
                       help="Number of calibration forward passes")
    parser.add_argument("--calib_frames",  type=int, default=32,
                       help="Number of video frames to sample")
    parser.add_argument("--latent_size",   type=str, default="60x80",
                       help="Latent spatial size HxW (e.g. 60x80)")
    parser.add_argument(
        "--mode", type=str, default="a8w8",
        choices=["a16w8", "a8w8", "a16w4", "a8w4"],
        help="Quantization mode (a16=no act quant, a8=act quant int8; w8/w4=weight bits)"
    )
    parser.add_argument(
        "--vae_path", type=str, default=None,
        help="Path to VAE checkpoint for real latent encoding. "
             "If not provided, falls back to RGB replication (inaccurate activation stats)."
    )
    parser.add_argument(
        "--vae_model", type=str, default="Wan2.1",
        choices=["Wan2.1", "Wan2.2", "LightVAE_W2.1", "TAE_W2.2", "LightTAE_HY1.5"],
        help="VAE model variant (must match vae_path if provided)."
    )
    args = parser.parse_args()

    # Parse latent size
    h_str, w_str = args.latent_size.lower().split("x")
    frame_size = (int(h_str), int(w_str))

    # ------------------------------------------------------------------
    # 1. Build DiT model
    # ------------------------------------------------------------------
    print(f"\n[Calibrate] Building WanModel …")
    model = build_dit(args.checkpoint.split("/")[-2] if "/" in args.checkpoint else "FlashVSR-v1.1")
    model = load_checkpoint(args.checkpoint, model)
    model.cuda().eval()

    # ------------------------------------------------------------------
    # 2. Collect latent samples from one video or random datasets/train videos
    # ------------------------------------------------------------------
    if args.video:
        selected_videos = [args.video]
    else:
        selected_videos = discover_calibration_videos(args.dataset_train, args.num_videos, args.seed)

    print(f"\n[Calibrate] Selected {len(selected_videos)} calibration video(s):")
    for idx, video_path in enumerate(selected_videos, 1):
        print(f"  {idx:02d}. {video_path}")

    latent_chunks = []
    fps_values = []
    for video_path in selected_videos:
        print(f"\n[Calibrate] Sampling latents from video: {video_path}")
        latents_i, fps = sample_latents_from_video(
            video_path=video_path,
            num_frames=args.calib_frames,
            latent_channels=16,
            frame_size=frame_size,
            vae_path=args.vae_path,
            vae_model=args.vae_model,
        )
        latent_chunks.append(latents_i)
        fps_values.append(fps)

    latents = torch.cat(latent_chunks, dim=0)
    # Shuffle frame-level calibration samples so --num_samples covers all videos
    # instead of only the first selected video when num_samples is small.
    generator = torch.Generator().manual_seed(args.seed)
    latents = latents[torch.randperm(latents.shape[0], generator=generator)]
    latents = latents.cuda().to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    # ------------------------------------------------------------------
    # 3. Generate dummy timesteps and text contexts
    # ------------------------------------------------------------------
    # Real text embeddings are expensive; use Gaussian noise as proxy.
    num_ctx = min(args.num_samples, 10)
    contexts = [
        torch.randn(1, 10, 4096, dtype=latents.dtype, device="cuda")
        for _ in range(num_ctx)
    ]

    # ------------------------------------------------------------------
    # 4. Collect activation statistics (only meaningful when a8 is used)
    # ------------------------------------------------------------------
    act_stats = None
    if args.mode.startswith("a8"):
        print(f"\n[Calibrate] Running {args.num_samples} forward passes for activation stats …")
        act_stats = collect_activation_stats_fakequant(
            model,
            latents,
            contexts,
            num_samples=args.num_samples,
        )
        print(f"[Calibrate] Collected stats for {len(act_stats)} layers")
    else:
        print(f"\n[Calibrate] Mode={args.mode} — activation quant disabled (a16), skipping hook collection")

    # ------------------------------------------------------------------
    # 5. Save calibration cache
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output_cache) or ".", exist_ok=True)

    cache = {"_metadata": {
        "mode": args.mode,
        "num_samples": args.num_samples,
        "video": args.video,
        "dataset_train": args.dataset_train if not args.video else None,
        "num_videos": len(selected_videos),
        "seed": args.seed,
        "selected_videos": selected_videos,
        "fps_values": fps_values,
        "latent_size": args.latent_size,
        "checkpoint": args.checkpoint,
        "vae_path": args.vae_path,
        "vae_model": args.vae_model,
        "calib_frames": args.calib_frames,
    }}

    if act_stats:
        for name, s in act_stats.items():
            entry = {
                "act_scale": s["act_scale"].cpu().numpy().tolist(),
                "zero_point": s["zero_point"].cpu().numpy().tolist(),
            }
            # Preserve min/max so static per-tensor caches can be derived from
            # the same calibration run without re-running the DiT forward pass.
            if "act_min" in s:
                entry["act_min"] = s["act_min"].cpu().numpy().tolist()
            if "act_max" in s:
                entry["act_max"] = s["act_max"].cpu().numpy().tolist()
            if "act_mean" in s:
                entry["act_mean"] = s["act_mean"].cpu().numpy().tolist()
            cache[name] = entry

    with open(args.output_cache, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\n[Calibrate] Cache saved → {args.output_cache}")
    print(f"[Calibrate] Entries: {len(cache) - 1} layers")


if __name__ == "__main__":
    main()
