#!/usr/bin/env python3
"""
Preprocess video to meet FlashVSR input requirements.

FlashVSR requires:
1. Frame count: input_frame_count + 4 must be 8n+1 (i.e., input % 8 in [1, 4] after accounting for +4 padding)
2. Spatial resolution: (height * scale) and (width * scale) must be multiples of 16 (ideally 128 for best results)

This script ensures your video meets these requirements by:
- Padding frames to reach the next valid 8n+1 count after internal +4 padding
- Aligning spatial dimensions to multiples of 16 (scaled)

Usage:
    python preprocess_for_flashvsr.py --input video.mp4 --output preprocessed.mp4 --scale 4
    python preprocess_for_flashvsr.py --input video.mp4 --output preprocessed.mp4 --scale 4 --align_to_128
"""

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm


def largest_8n1_leq(n):
    """Largest 8n+1 <= n"""
    if n < 1:
        return 0
    return ((n - 1) // 8) * 8 + 1


def next_8n5(n):
    """Next 8n+5 >= n"""
    if n < 21:
        return 21
    return ((n - 5 + 7) // 8) * 8 + 5


def align_to_multiple(value, multiple):
    """Align value UP to nearest multiple"""
    return ((value + multiple - 1) // multiple) * multiple


def compute_target_frames(current_frames):
    """
    Compute target frame count for FlashVSR.

    FlashVSR internally adds 4 padding frames, so we need:
        (current_frames + 4) to be 8n+1, i.e., largest_8n1_leq(current_frames + 4)

    Returns the frame count that FlashVSR will actually process.
    """
    return largest_8n1_leq(current_frames + 4)


def compute_target_dims(width, height, scale, align_multiple=16, force_scale=None):
    """
    Compute target dimensions that meet FlashVSR requirements.

    Args:
        width, height: original dimensions
        scale: upscaling factor
        align_multiple: alignment requirement (16 or 128 recommended)
        force_scale: if provided, resize to this scale instead of automatic

    Returns:
        (target_width, target_height, scaled_width, scaled_height,
         aligned_width, aligned_height, pad_w, pad_h)
    """
    # Scale first
    scaled_w = int(width * scale)
    scaled_h = int(height * scale)

    # Then align (FlashVSR internally pads to 128 for processing, crops back)
    aligned_w = align_to_multiple(scaled_w, align_multiple)
    aligned_h = align_to_multiple(scaled_h, align_multiple)

    # Padding needed
    pad_w = aligned_w - scaled_w
    pad_h = aligned_h - scaled_h

    return scaled_w, scaled_h, aligned_w, aligned_h, pad_w, pad_h


def pad_frame(frame, pad_left, pad_top, pad_right, pad_bottom):
    """Pad a single frame with edge replication."""
    return cv2.copyMakeBorder(
        frame,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_REPLICATE
    )


def resize_frame(frame, target_h, target_w, method='lanczos'):
    """Resize frame to target dimensions."""
    methods = {
        'lanczos': cv2.INTER_LANCZOS4,
        'cubic': cv2.INTER_CUBIC,
        'linear': cv2.INTER_LINEAR,
        'nearest': cv2.INTER_NEAREST,
    }
    return cv2.resize(frame, (target_w, target_h), interpolation=methods.get(method, cv2.INTER_LANCZOS4))


def preprocess_video(input_path, output_path, scale, align_multiple=16, target_fps=None,
                     resize_method='lanczos', dry_run=False):
    """
    Preprocess video to meet FlashVSR requirements.

    Args:
        input_path: Input video path
        output_path: Output video path
        scale: Upscaling factor (2 or 4)
        align_multiple: Spatial alignment (16 or 128)
        target_fps: Force FPS (None = use input fps)
        resize_method: Interpolation method for resizing
        dry_run: If True, only print info without writing
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if target_fps is None:
        target_fps = fps

    # Compute target dimensions
    scaled_w, scaled_h, aligned_w, aligned_h, pad_w, pad_h = compute_target_dims(
        width, height, scale, align_multiple
    )

    # Compute target frame count
    target_frames = compute_target_frames(total_frames)
    frames_to_add = target_frames - total_frames

    print("=" * 60)
    print("FlashVSR Video Preprocessor")
    print("=" * 60)
    print(f"Input:  {width}x{height}, {total_frames} frames, {fps:.2f} FPS")
    print(f"Scale:  {scale}x")
    print(f"Scaled: {scaled_w}x{scaled_h}")
    print(f"Aligned to {align_multiple}: {aligned_w}x{aligned_h} (padding: {pad_w}x{pad_h})")
    print(f"Target frames: {target_frames} (+{frames_to_add} padding frames)")
    print(f"Output FPS: {target_fps:.2f}")
    print("=" * 60)

    if frames_to_add < 0:
        print(f"WARNING: Video has {total_frames} frames but FlashVSR can only process up to {target_frames}.")
        print(f"         The last {-frames_to_add} frames will be DISCARDED!")
        frames_to_add = 0

    if dry_run:
        print("\n[DRY RUN] No file written.")
        cap.release()
        return

    # Setup video writer
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, target_fps, (aligned_w, aligned_h))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")

    # Read and process frames
    processed = 0
    last_frame = None

    pbar = tqdm(total=target_frames, desc="Processing frames")

    while processed < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_qq = frame.copy()
        # Resize to scaled dimensions first
        if scaled_w != width or scaled_h != height:
            frame_qq = resize_frame(frame, scaled_h, scaled_w, resize_method)

        # Pad to aligned dimensions (centered)
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top

        if pad_w > 0 or pad_h > 0:
            frame_qq = pad_frame(frame_qq, pad_left, pad_top, pad_right, pad_bottom)

        # Write frame at scaled+aligned dimensions (aligned_w x aligned_h)
        # After FlashVSR 4x upscale + x4 downsample = original input size
        writer.write(frame_qq)
        last_frame = frame
        processed += 1
        pbar.update(1)

    cap.release()

    # Add padding frames if needed (repeat last frame)
    if frames_to_add > 0 and last_frame is not None:
        print(f"\nAdding {frames_to_add} padding frames (repeating last frame)...")
        for _ in range(frames_to_add):
            writer.write(last_frame)
            pbar.update(1)

    writer.release()
    pbar.close()

    # Verify output
    cap_out = cv2.VideoCapture(output_path)
    out_frames = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
    out_width = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_height = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_out.release()

    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"Output: {out_width}x{out_height}, {out_frames} frames")
    print(f"File: {output_path}")
    print("\nFlashVSR will produce approximately:")
    actual_target = largest_8n1_leq(out_frames + 4)
    approx_output = ((actual_target - 1) // 8 - 2) * 6
    print(f"  ~{approx_output} output frames (after architectural processing)")
    print("=" * 60)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess video for FlashVSR - ensures 8n+1 frame count and 16-aligned dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preprocess for 4x upscaling with 16-pixel alignment
    python preprocess_for_flashvsr.py --input video.mp4 --output preprocessed.mp4 --scale 4

    # Preprocess for 4x upscaling with 128-pixel alignment (recommended)
    python preprocess_for_flashvsr.py --input video.mp4 --output preprocessed.mp4 --scale 4 --align_to_128

    # Preprocess for 2x upscaling
    python preprocess_for_flashvsr.py --input video.mp4 --output preprocessed.mp4 --scale 2

    # Dry run to see what would happen without writing
    python preprocess_for_flashvsr.py --input video.mp4 --output preprocessed.mp4 --scale 4 --dry_run
"""
    )

    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', required=True, help='Output video path')
    parser.add_argument('--scale', type=int, default=4, choices=[2, 4],
                        help='Upscaling factor (default: 4)')
    parser.add_argument('--align_multiple', type=int, default=16,
                        help='Spatial alignment multiple (default: 16, recommended: 128)')
    parser.add_argument('--align_to_128', action='store_true',
                        help='Shortcut for --align_multiple 128 (recommended for best quality)')
    parser.add_argument('--fps', type=float, default=None,
                        help='Output FPS (default: same as input)')
    parser.add_argument('--resize_method', default='lanczos',
                        choices=['lanczos', 'cubic', 'linear', 'nearest'],
                        help='Resize interpolation method (default: lanczos)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print info without writing output file')

    args = parser.parse_args()

    # Handle --align_to_128 shortcut
    if args.align_to_128:
        args.align_multiple = 128

    preprocess_video(
        input_path=args.input,
        output_path=args.output,
        scale=args.scale,
        align_multiple=args.align_multiple,
        target_fps=args.fps,
        resize_method=args.resize_method,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()