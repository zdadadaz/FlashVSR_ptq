from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts.ptq.run_lsgquant_pr9_paper_benchmark import (
    build_cli_command,
    compute_video_psnr,
    summarize_metrics,
)


def _write_video(path: Path, value: int, frames: int = 2, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, size)
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.full((size[1], size[0], 3), value, dtype=np.uint8))
    writer.release()


def test_pr9_paper_benchmark_command_uses_a4w4_checkpoint() -> None:
    cmd = build_cli_command(
        "/venv/bin/python",
        Path("in.mp4"),
        Path("out.mp4"),
        frames=16,
        seed=7,
        quantize_mode="FakeQuant_A4W4",
        checkpoint=Path("qao.safetensors"),
    )

    assert cmd[:2] == ["/venv/bin/python", "cli_main.py"]
    assert cmd[cmd.index("--quantize_mode") + 1] == "FakeQuant_A4W4"
    assert cmd[cmd.index("--ckpt_path") + 1] == "qao.safetensors"
    assert cmd[cmd.index("--end_frame") + 1] == "16"


def test_pr9_paper_benchmark_psnr_and_dataset_delta_summary(tmp_path: Path) -> None:
    ref = tmp_path / "ref.mp4"
    same = tmp_path / "same.mp4"
    worse = tmp_path / "worse.mp4"
    _write_video(ref, 128)
    _write_video(same, 128)
    _write_video(worse, 118)

    fp = compute_video_psnr(ref, same)
    q = compute_video_psnr(ref, worse)
    assert fp["frames"] == 2
    assert fp["psnr_avg_db"] == float("inf")
    assert q["psnr_avg_db"] < 40

    summary = summarize_metrics([
        {
            "dataset": "UDM10",
            "clip": "000",
            "fp16_vs_gt_psnr_db": 30.0,
            "a4w4_qao_vs_gt_psnr_db": 29.25,
            "a4w4_qao_minus_fp16_psnr_db": -0.75,
        },
        {
            "dataset": "UDM10",
            "clip": "001",
            "fp16_vs_gt_psnr_db": 32.0,
            "a4w4_qao_vs_gt_psnr_db": 31.75,
            "a4w4_qao_minus_fp16_psnr_db": -0.25,
        },
    ])

    assert summary["clips"] == 2
    assert summary["datasets"]["UDM10"]["fp16_vs_gt_mean_psnr_db"] == 31.0
    assert summary["datasets"]["UDM10"]["a4w4_qao_minus_fp16_mean_psnr_db"] == -0.5
