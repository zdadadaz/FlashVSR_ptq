import json
import subprocess
import sys

import torch
import torch.nn as nn

from scripts.ptq.fakequant_convert import load_lsgquant_layer_policy
from scripts.ptq.run_lsgquant_eval_set import build_lsgquant_eval_manifest, build_lsgquant_convert_command
from src.models.quantization.fakequant import FakeQuantLinear, convert_model_to_fakequant


class TinyWanLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.ModuleDict({"q": nn.Linear(4, 4), "k": nn.Linear(4, 4)}),
                "ffn": nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4)),
            })
        ])
        self.head = nn.ModuleDict({"head": nn.Linear(4, 4)})


def _lsg_policy():
    return {
        "schema_version": "flashvsr.lsgquant.policy.v1",
        "default": {"mode": "a8w8", "activation_qdq_mode": "draq_symmetric"},
        "counts": {"frozen": 1, "light": 1, "full": 1},
        "layers": {
            "blocks.0.self_attn.q": {"mode": "a8w8", "activation_qdq_mode": "draq_symmetric", "tier": "frozen", "mu_var": 0.0005},
            "blocks.0.self_attn.k": {"mode": "a8w8", "activation_qdq_mode": "draq_symmetric", "tier": "light", "mu_var": 0.01},
            "blocks.0.ffn.0": {"mode": "a8w8", "activation_qdq_mode": "draq_symmetric", "tier": "full", "mu_var": 0.1},
            "blocks.0.ffn.2": {"mode": "fp16_skip", "tier": "full", "mu_var": 0.2},
            "head.head": {"mode": "a8w8", "activation_qdq_mode": "draq_symmetric", "tier": "frozen", "mu_var": 0.0002},
        },
    }


def test_load_lsgquant_layer_policy_returns_entries_and_tier_counts(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(_lsg_policy()))

    entries, summary = load_lsgquant_layer_policy(policy_path)

    assert entries["blocks.0.self_attn.q"]["activation_qdq_mode"] == "draq_symmetric"
    assert entries["blocks.0.ffn.2"]["mode"] == "fp16_skip"
    assert summary["schema_version"] == "flashvsr.lsgquant.policy.v1"
    assert summary["tier_counts"] == {"frozen": 2, "light": 1, "full": 2}
    assert summary["mode_counts"] == {"a8w8": 4, "fp16_skip": 1}


def test_convert_model_to_fakequant_applies_lsgquant_draq_policy_and_fp16_skip():
    model = TinyWanLike()
    entries = _lsg_policy()["layers"]

    convert_model_to_fakequant(
        model,
        mode="a8w8",
        act_stats=None,
        activation_qdq_mode="draq_symmetric",
        layer_policy=entries,
        enable_bias_correction=True,
    )

    assert isinstance(model.blocks[0]["self_attn"]["q"], FakeQuantLinear)
    assert int(model.blocks[0]["self_attn"]["q"].activation_qdq_mode.item()) == 3
    assert isinstance(model.blocks[0]["ffn"][0], FakeQuantLinear)
    assert isinstance(model.blocks[0]["ffn"][2], nn.Linear)  # fp16_skip remains unconverted
    assert model._fakequant_conversion_summary["mode_counts"] == {"a8w8": 4, "fp16_skip": 1}


def test_fakequant_convert_cli_exposes_pr3_policy_alias_and_bias_correction():
    result = subprocess.run(
        [sys.executable, "scripts/ptq/fakequant_convert.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--policy" in result.stdout
    assert "--policy_json" in result.stdout
    assert "--enable_bias_correction" in result.stdout
    assert "draq_symmetric" in result.stdout


def test_lsgquant_eval_manifest_builds_convert_command_and_records_quality_gate(tmp_path):
    checkpoint = tmp_path / "dit.safetensors"
    calib = tmp_path / "calib.json"
    policy = tmp_path / "policy.json"
    out_dir = tmp_path / "eval"
    checkpoint.write_text("placeholder")
    calib.write_text(json.dumps({"_metadata": {"schema_version": "flashvsr.lsgquant.calibration.v1"}}))
    policy.write_text(json.dumps(_lsg_policy()))

    cmd = build_lsgquant_convert_command(
        checkpoint=checkpoint,
        calibration_cache=calib,
        policy=policy,
        mode="a8w8",
        out_dir=out_dir,
    )
    manifest = build_lsgquant_eval_manifest(
        checkpoint=checkpoint,
        calibration_cache=calib,
        policy=policy,
        mode="a8w8",
        out_dir=out_dir,
        limit=2,
    )

    assert cmd[:3] == [sys.executable, "scripts/ptq/fakequant_convert.py", "--checkpoint"]
    assert "--policy" in cmd
    assert "--activation_qdq_mode" in cmd and "draq_symmetric" in cmd
    assert "--enable_bias_correction" in cmd
    assert manifest["quality_gate"]["baseline"] == "static_a8w8"
    assert manifest["quality_gate"]["metrics"] == ["psnr", "temporal_drift"]
    assert manifest["convert_command"] == cmd


def test_lsgquant_eval_cli_writes_manifest_without_running_conversion(tmp_path):
    checkpoint = tmp_path / "dit.safetensors"
    calib = tmp_path / "calib.json"
    policy = tmp_path / "policy.json"
    out_dir = tmp_path / "eval"
    checkpoint.write_text("placeholder")
    calib.write_text(json.dumps({"_metadata": {"schema_version": "flashvsr.lsgquant.calibration.v1"}}))
    policy.write_text(json.dumps(_lsg_policy()))

    subprocess.run(
        [
            sys.executable,
            "scripts/ptq/run_lsgquant_eval_set.py",
            "--checkpoint",
            str(checkpoint),
            "--calibration_cache",
            str(calib),
            "--policy",
            str(policy),
            "--mode",
            "a8w8",
            "--out_dir",
            str(out_dir),
            "--limit",
            "1",
            "--dry_run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads((out_dir / "lsgquant_eval_manifest.json").read_text())
    assert manifest["policy_summary"]["tier_counts"] == {"frozen": 2, "light": 1, "full": 2}
    assert manifest["converted_checkpoint"].endswith("dit_lsgquant_a8w8_draq_volts.safetensors")
