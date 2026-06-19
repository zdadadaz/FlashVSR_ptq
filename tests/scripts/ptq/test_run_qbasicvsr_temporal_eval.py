import json
import subprocess
import sys


def test_run_qbasicvsr_temporal_eval_dry_run_updates_leaderboard(tmp_path):
    policy = tmp_path / "policy.json"
    ckpt = tmp_path / "ckpt.safetensors"
    reproduce = tmp_path / "reproduce.sh"
    input_video = tmp_path / "in.mp4"
    leaderboard = tmp_path / "leaderboard.jsonl"
    daily = tmp_path / "daily"
    ckpt.write_text("placeholder")
    reproduce.write_text("#!/usr/bin/env bash\n")
    input_video.write_text("placeholder")
    policy.write_text(json.dumps({
        "schema_version": "flashvsr.qbasicvsr.temporal_policy.v1",
        "quant_scope": "dit_linear_only",
        "wan_vae_quantized": False,
        "base_bits": 4,
        "video_bit_factor": 0,
        "activation_qdq_mode": "draq_symmetric",
        "flow_backend": "proxy",
        "fab": 8.0,
        "counts": {"a8w8": 2, "a16w8": 1},
        "layers": {},
    }))
    subprocess.run([
        sys.executable, "scripts/ptq/run_qbasicvsr_temporal_eval.py",
        "--run_id", "dry", "--policy", str(policy), "--checkpoint", str(ckpt),
        "--input_video", str(input_video), "--frames", "2", "--eval_set", "dry_eval",
        "--leaderboard", str(leaderboard), "--daily_dir", str(daily), "--dry_run",
        "--reproduce_script", str(reproduce),
    ], check=True)
    row = json.loads(leaderboard.read_text().splitlines()[0])
    assert row["run_id"] == "dry"
    assert row["a8_layers"] == 2
    assert row["a16_layers"] == 1
    assert row["clipping"] == "qbasicvsr_temporal"
    assert row["teacher_ft_steps"] == 0
    assert row["static_ablation_label"] == ""
    assert row["reproduce_script"] == str(reproduce)
    assert row["reproduce_sha256"] is not None
    assert "wan_vae_quantized=false" in row["notes"]
    assert (daily / "20260619_flashvsr_static_mixed_leaderboard.jsonl").exists() or list(daily.glob("*_flashvsr_static_mixed_leaderboard.jsonl"))


def test_run_qbasicvsr_temporal_eval_static_ablation_metadata(tmp_path):
    policy = tmp_path / "policy.json"
    ckpt = tmp_path / "ckpt.safetensors"
    reproduce = tmp_path / "reproduce.sh"
    input_video = tmp_path / "in.mp4"
    leaderboard = tmp_path / "leaderboard.jsonl"
    daily = tmp_path / "daily"
    ckpt.write_text("placeholder")
    reproduce.write_text("#!/usr/bin/env bash\n")
    input_video.write_text("placeholder")
    policy.write_text(json.dumps({
        "schema_version": "flashvsr.qbasicvsr.temporal_policy.v1",
        "quant_scope": "dit_linear_only",
        "wan_vae_quantized": False,
        "base_bits": 4,
        "video_bit_factor": 0,
        "activation_qdq_mode": "static_asymmetric",
        "flow_backend": "proxy",
        "fab": 8.156862745098039,
        "counts": {"a8w8": 300, "a16w8": 6},
        "layers": {},
    }))
    subprocess.run([
        sys.executable, "scripts/ptq/run_qbasicvsr_temporal_eval.py",
        "--run_id", "static", "--policy", str(policy), "--checkpoint", str(ckpt),
        "--input_video", str(input_video), "--frames", "2", "--eval_set", "dry_eval",
        "--leaderboard", str(leaderboard), "--daily_dir", str(daily), "--dry_run",
        "--clipping_method", "omse_teacher_clipft", "--teacher_ft_steps", "7",
        "--static_ablation_label", "static_asym_omse_clipft",
        "--reproduce_script", str(reproduce),
    ], check=True)
    row = json.loads(leaderboard.read_text().splitlines()[0])
    manifest = json.loads((tmp_path / "daily").glob("*_qbasicvsr_flashvsr_report.md").__next__().read_text().split("```json\n", 1)[1].split("\n```", 1)[0])
    assert row["activation_qdq_mode"] == "static_asymmetric"
    assert row["clipping"] == "omse_teacher_clipft"
    assert row["teacher_ft_steps"] == 7
    assert row["static_ablation_label"] == "static_asym_omse_clipft"
    assert "clipping_method=omse_teacher_clipft" in row["notes"]
    assert manifest["clipping_method"] == "omse_teacher_clipft"
    assert manifest["teacher_ft_steps"] == 7
    assert manifest["reproduce_script"] == str(reproduce)
