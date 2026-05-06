#!/usr/bin/env python3
"""Compare FPS and PSNR between FP16 baseline and W8A16 quantized model outputs."""
import argparse
import cv2
import math
import numpy as np
import re
import sys


def calculate_psnr(img1, img2):
    """Calculate PSNR between two frames (0-255 range)."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def extract_fps_from_log(log_path):
    """Extract FPS from cli_main.py log output."""
    with open(log_path, 'r') as f:
        content = f.read()
    # Match pattern like "Total Time: ... (12.34 FPS)"
    match = re.search(r'\((\d+\.?\d*)\s+FPS\)', content)
    if match:
        return float(match.group(1))
    return None


def compare_videos(fp16_path, w8a16_path):
    """Compare two videos frame by frame for PSNR."""
    cap1 = cv2.VideoCapture(fp16_path)
    cap2 = cv2.VideoCapture(w8a16_path)

    if not cap1.isOpened():
        raise FileNotFoundError(f"Cannot open FP16 video: {fp16_path}")
    if not cap2.isOpened():
        raise FileNotFoundError(f"Cannot open W8A16 video: {w8a16_path}")

    fps1, fps2 = cap1.get(cv2.CAP_PROP_FPS), cap2.get(cv2.CAP_PROP_FPS)
    frame_count1, frame_count2 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    w1, h1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FP16  video: {w1}x{h1}, {fps1:.2f} fps, {frame_count1} frames")
    print(f"W8A16 video: {w2}x{h2}, {fps2:.2f} fps, {frame_count2} frames")

    min_frames = min(frame_count1, frame_count2)
    psnr_values = []

    for i in range(min_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        # Resize if needed
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        psnr = calculate_psnr(frame1, frame2)
        psnr_values.append(psnr)

    cap1.release()
    cap2.release()

    if psnr_values:
        avg_psnr = sum(psnr_values) / len(psnr_values)
        min_psnr = min(psnr_values)
        max_psnr = max(psnr_values)
        print(f"\nPSNR Comparison ({len(psnr_values)} frames):")
        print(f"  Average PSNR: {avg_psnr:.2f} dB")
        print(f"  Min PSNR:     {min_psnr:.2f} dB")
        print(f"  Max PSNR:     {max_psnr:.2f} dB")
        return avg_psnr
    return None


def main():
    parser = argparse.ArgumentParser(description="Compare FP16 vs W8A16 FlashVSR outputs")
    parser.add_argument("--fp16_video", type=str, required=True, help="Path to FP16 baseline video")
    parser.add_argument("--w8a16_video", type=str, required=True, help="Path to W8A16 quantized video")
    parser.add_argument("--fp16_log", type=str, default=None, help="Optional: Path to FP16 run log with FPS")
    parser.add_argument("--w8a16_log", type=str, default=None, help="Optional: Path to W8A16 run log with FPS")
    args = parser.parse_args()

    print("=" * 60)
    print("FlashVSR FP16 vs W8A16 Comparison")
    print("=" * 60)

    # Extract FPS from logs if provided
    fp16_fps, w8a16_fps = None, None
    if args.fp16_log:
        fp16_fps = extract_fps_from_log(args.fp16_log)
    if args.w8a16_log:
        w8a16_fps = extract_fps_from_log(args.w8a16_log)

    if fp16_fps and w8a16_fps:
        speedup = w8a16_fps / fp16_fps
        print(f"\nFPS Comparison:")
        print(f"  FP16  FPS: {fp16_fps:.2f}")
        print(f"  W8A16 FPS: {w8a16_fps:.2f}")
        print(f"  Speedup:   {speedup:.2f}x")

    # Compare video quality
    compare_videos(args.fp16_video, args.w8a16_video)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
