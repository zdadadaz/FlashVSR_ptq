import json
from pathlib import Path

import cv2
import numpy as np

from scripts.ptq.lsgquant_standard_eval import (
    build_standard_eval_manifest,
    calibration_cache_summary,
    compute_video_psnr,
    discover_sequence_inputs,
    materialize_downsampled_sequence_video,
)


def _touch_video(path: Path, value: int = 0, frames: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (8, 8))
    assert writer.isOpened()
    for _ in range(frames):
        frame = np.full((8, 8, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_calibration_cache_summary_requires_scale_and_zero_point(tmp_path):
    cache = tmp_path / "calib.json"
    cache.write_text(json.dumps({
        "linear": {"act_scale": [0.1, 0.2], "zero_point": [1, 2]},
        "_metadata": {"dataset": "HQ-VSR"},
    }))

    summary = calibration_cache_summary(cache)

    assert summary["path"] == str(cache)
    assert summary["layers"] == 1
    assert summary["layers_with_act_scale"] == 1
    assert summary["layers_with_zero_point"] == 1
    assert summary["ready_for_static_a8w8"] is True


def test_discover_sequence_inputs_handles_lq_gt_video_layout(tmp_path):
    root = tmp_path / "UDM10"
    _touch_video(root / "LQ-Video" / "clip001.mp4")
    _touch_video(root / "GT-Video" / "clip001.mp4")

    items = discover_sequence_inputs(root, "UDM10", limit=10)

    assert len(items) == 1
    assert items[0]["dataset"] == "UDM10"
    assert items[0]["input_video"].endswith("LQ-Video/clip001.mp4")
    assert items[0]["reference_video"].endswith("GT-Video/clip001.mp4")


def _touch_image(path: Path, width: int = 16, height: int = 12, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_discover_sequence_inputs_handles_lq_gt_image_sequence_layout(tmp_path):
    root = tmp_path / "UDM10"
    for idx in range(3):
        _touch_image(root / "LQ" / "000" / f"{idx:04d}.png", width=16, height=12, value=idx)
        _touch_image(root / "GT" / "000" / f"{idx:04d}.png", width=64, height=48, value=idx)

    items = discover_sequence_inputs(root, "UDM10", limit=10)

    assert len(items) == 1
    assert items[0]["input_sequence"] == str(root / "LQ" / "000")
    assert items[0]["reference_sequence"] == str(root / "GT" / "000")
    assert items[0]["sequence_name"] == "000"


def test_materialize_downsampled_sequence_video_reduces_lq_before_flashvsr(tmp_path):
    seq = tmp_path / "LQ" / "000"
    for idx in range(2):
        _touch_image(seq / f"{idx:04d}.png", width=16, height=12, value=20 + idx)
    out = tmp_path / "prepared" / "000_downx4.mp4"

    result = materialize_downsampled_sequence_video(seq, out, frames=2, downsample_scale=4, fps=8.0)

    assert result["frames"] == 2
    assert result["source_size"] == [16, 12]
    assert result["video_size"] == [4, 4]  # 16x12 / 4, padded to even height for yuv420p
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    ok, frame = cap.read()
    cap.release()
    assert ok
    assert frame.shape[:2] == (4, 4)


def test_manifest_prefers_downsampled_lq_sequences_for_test_set_when_preparing(tmp_path):
    dataset = tmp_path / "UDM10"
    _touch_video(dataset / "LQ-Video" / "clip001.mp4", value=0)
    _touch_video(dataset / "GT-Video" / "clip001.mp4", value=0)
    for idx in range(2):
        _touch_image(dataset / "LQ" / "000" / f"{idx:04d}.png", width=16, height=12, value=idx)
        _touch_image(dataset / "GT" / "000" / f"{idx:04d}.png", width=16, height=12, value=idx)
    cache = tmp_path / "calib.json"
    cache.write_text(json.dumps({"linear": {"act_scale": [0.1], "zero_point": [0]}}))

    manifest = build_standard_eval_manifest(
        calibration_dataset=tmp_path / "HQ-VSR",
        calibration_cache=cache,
        eval_datasets={"UDM10": dataset},
        out_dir=tmp_path / "out",
        fp_checkpoint=Path("models/fp.safetensors"),
        quant_checkpoint=Path("outputs/ptq/quant.safetensors"),
        frames=2,
        limit_per_dataset=1,
        prepare_image_sequences=True,
        downsample_lq_scale=4,
    )

    assert manifest["evaluation"][0]["input_sequence"].endswith("LQ/000")
    assert manifest["preparation"]["prepared_inputs"][0]["video_size"] == [4, 4]
    assert manifest["runs"][0]["prepared_from_sequence"] is True
    assert "prepared_inputs/UDM10/000_lq_downx4_first2.mp4" in manifest["runs"][0]["input_video"]


def test_compute_video_psnr_and_standard_manifest(tmp_path):
    fp16 = tmp_path / "fp16.mp4"
    ptq = tmp_path / "ptq.mp4"
    _touch_video(fp16, value=20)
    _touch_video(ptq, value=22)

    metric = compute_video_psnr(fp16, ptq)
    assert metric["frames"] == 2
    assert metric["psnr_avg_db"] > 40.0

    cache = tmp_path / "calib.json"
    cache.write_text(json.dumps({"linear": {"act_scale": [0.1], "zero_point": [0]}}))
    dataset = tmp_path / "UDM10"
    _touch_video(dataset / "LQ-Video" / "clip001.mp4", value=0)
    _touch_video(dataset / "GT-Video" / "clip001.mp4", value=0)

    manifest = build_standard_eval_manifest(
        calibration_dataset=tmp_path / "HQ-VSR",
        calibration_cache=cache,
        eval_datasets={"UDM10": dataset},
        out_dir=tmp_path / "out",
        fp_checkpoint=Path("models/fp.safetensors"),
        quant_checkpoint=Path("outputs/ptq/quant.safetensors"),
        quantize_mode="FakeQuant_A8W8_DRAQ",
        frames=16,
        seed=0,
        limit_per_dataset=1,
    )

    assert manifest["schema_version"] == "flashvsr.lsgquant_standard_eval.v1"
    assert manifest["calibration"]["cache"]["ready_for_static_a8w8"] is True
    assert manifest["quantize_mode"] == "FakeQuant_A8W8_DRAQ"
    assert manifest["evaluation"][0]["dataset"] == "UDM10"
    assert "--quantize_mode" in manifest["runs"][1]["command"]
    assert "FakeQuant_A8W8_DRAQ" in manifest["runs"][1]["command"]
