import json
from pathlib import Path

from scripts.ptq.lsgquant_pr0_eval import (
    DEFAULT_EXPERIMENT_SETTINGS,
    build_cli_command,
    build_pr0_manifest,
    discover_videos,
    parse_dataset_args,
)


def _touch_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-video")


def test_paper_experiment_settings_are_encoded():
    settings = DEFAULT_EXPERIMENT_SETTINGS

    assert settings["paper"] == "arXiv:2602.03182v1"
    assert settings["calibration"]["dataset"] == "HQ-VSR"
    assert settings["calibration"]["num_videos"] == 50
    assert settings["evaluation_datasets"] == {
        "UDM10": "synthetic",
        "REDS30": "synthetic",
        "MVSR4x": "real_world",
    }
    assert settings["metrics"]["reference_iqa"] == ["DISTS", "PSNR", "SSIM", "LPIPS"]
    assert settings["metrics"]["no_reference_iqa"] == ["MANIQA", "CLIP-IQA", "MUSIQ"]
    assert settings["metrics"]["vqa"] == ["Ewarp*", "DOVER"]
    assert settings["volts"] == {
        "delta1": 0.001,
        "delta2": 0.075,
        "frozen_iterations": 1,
        "light_adaptation_iterations": 30,
        "fully_optimized_iterations": "until_convergence",
    }
    assert settings["qao"]["svd_rank"] == 32
    assert settings["weight_quantization"] == "static_asymmetric_channel_wise"
    assert settings["scope"] == "WanVideoDiT Linear layers only; Wan VAE remains unquantized"


def test_discover_videos_deterministically_samples_50(tmp_path):
    root = tmp_path / "hq_vsr"
    for idx in range(60):
        _touch_video(root / f"clip_{idx:03d}.mp4")

    first = discover_videos(root, limit=50, seed=123)
    second = discover_videos(root, limit=50, seed=123)
    different_seed = discover_videos(root, limit=50, seed=124)

    assert len(first) == 50
    assert first == second
    assert first != different_seed
    assert all(path.endswith(".mp4") for path in first)


def test_manifest_records_calibration_eval_datasets_commands_and_metric_plan(tmp_path):
    calib = tmp_path / "HQ-VSR"
    for idx in range(55):
        _touch_video(calib / f"train_{idx:03d}.mp4")
    udm = tmp_path / "UDM10"
    reds = tmp_path / "REDS30"
    mvsr = tmp_path / "MVSR4x"
    _touch_video(udm / "udm_000.mp4")
    _touch_video(reds / "reds_000.mp4")
    _touch_video(mvsr / "mvsr_000.mp4")

    manifest = build_pr0_manifest(
        calibration_dataset=calib,
        eval_datasets={"UDM10": udm, "REDS30": reds, "MVSR4x": mvsr},
        out_dir=tmp_path / "out",
        fp_checkpoint=Path("models/fp_dit.safetensors"),
        quant_checkpoint=Path("outputs/ptq/a8w8.safetensors"),
        frames=16,
        seed=7,
        execute=False,
    )

    assert manifest["schema_version"] == "flashvsr.lsgquant_pr0_eval.v1"
    assert manifest["experiment_settings"] == DEFAULT_EXPERIMENT_SETTINGS
    assert len(manifest["calibration"]["videos"])
    assert len(manifest["calibration"]["videos"]) == 50
    assert manifest["calibration"]["sampling"] == {"seed": 7, "requested_videos": 50, "available_videos": 55}
    assert set(manifest["evaluation"].keys()) == {"UDM10", "REDS30", "MVSR4x"}
    assert manifest["evaluation"]["UDM10"]["dataset_type"] == "synthetic"
    assert manifest["evaluation"]["MVSR4x"]["dataset_type"] == "real_world"
    assert manifest["runs"][0]["precision"] == "fp16"
    assert manifest["runs"][1]["quantize_mode"] == "FakeQuant_A8W8"
    assert "--quantize_mode" in manifest["runs"][1]["command"]
    assert manifest["metric_plan"]["implemented_now"] == ["PSNR"]
    assert "DISTS" in manifest["metric_plan"]["planned_reference_iqa"]
    assert manifest["execution"]["execute"] is False


def test_parse_dataset_args_and_command_builder(tmp_path):
    datasets = parse_dataset_args([f"UDM10={tmp_path / 'u'}", f"MVSR4x={tmp_path / 'm'}"])
    assert datasets == {"UDM10": tmp_path / "u", "MVSR4x": tmp_path / "m"}

    cmd = build_cli_command(
        input_video=Path("input.mp4"),
        output_video=Path("out.mp4"),
        frames=16,
        seed=0,
        quant_checkpoint=Path("q.safetensors"),
        quantize_mode="FakeQuant_A8W8",
    )
    assert cmd[:2] == [".venv/bin/python", "cli_main.py"]
    assert "--vae_model" in cmd and "Wan2.1" in cmd
    assert "--scale" in cmd and "4" in cmd
    assert "--end_frame" in cmd and "16" in cmd
    assert "--quantize_mode" in cmd and "FakeQuant_A8W8" in cmd
    assert "--ckpt_path" in cmd and "q.safetensors" in cmd
