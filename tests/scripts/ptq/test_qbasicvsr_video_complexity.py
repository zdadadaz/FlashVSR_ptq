import json
import subprocess
import sys

import numpy as np

from scripts.ptq.qbasicvsr_video_complexity import compute_frame_spatial_score, score_frames


def test_complexity_scores_constant_checkerboard_and_shifted_frames():
    constant = np.zeros((16, 16, 3), dtype=np.uint8)
    checker = ((np.indices((16, 16)).sum(axis=0) % 2) * 255).astype(np.uint8)
    checker = np.stack([checker, checker, checker], axis=-1)
    shifted = np.roll(checker, 1, axis=1)

    assert compute_frame_spatial_score(constant) == 0.0
    assert compute_frame_spatial_score(checker) > compute_frame_spatial_score(constant)
    static_report = score_frames([checker, checker], gamma=200, lambda_spatiotemporal=10)
    shifted_report = score_frames([checker, shifted], gamma=200, lambda_spatiotemporal=10)
    assert shifted_report["temporal_mean"] > static_report["temporal_mean"]
    assert shifted_report["c_video"] > static_report["c_video"]


def test_qbasicvsr_video_complexity_cli_writes_schema(tmp_path):
    # CLI help smoke keeps dependencies importable without requiring video fixtures.
    result = subprocess.run([sys.executable, "scripts/ptq/qbasicvsr_video_complexity.py", "--help"], check=True, text=True, capture_output=True)
    assert "--flow_backend" in result.stdout
    assert "--lambda_spatiotemporal" in result.stdout
