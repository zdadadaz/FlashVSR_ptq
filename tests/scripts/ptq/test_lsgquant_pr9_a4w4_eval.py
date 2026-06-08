import json
import subprocess
import sys

import torch
import torch.nn as nn

from scripts.ptq.run_lsgquant_eval_set import build_lsgquant_convert_command, build_lsgquant_eval_manifest
from src.models.quantization.lsgquant import LSGQuantLinear
from src.models.quantization.policy import build_lsgquant_a4w4_fallback_policy, load_layer_policy
from src.models.quantization.qao import convert_model_to_lsgquant_qao


def _cache():
    return {
        "_metadata": {"schema_version": "flashvsr.lsgquant.calibration.v1", "num_videos": 3},
        "layer.low": {"mu_var": 0.0002},
        "layer.mid": {"mu_var": 0.01},
        "layer.high": {"mu_var": 0.2},
        "layer.catastrophic": {"mu_var": 0.9},
    }


def test_a4w4_fallback_policy_assigns_rank_and_high_sensitivity_fallbacks():
    policy = build_lsgquant_a4w4_fallback_policy(
        _cache(),
        delta1=0.001,
        delta2=0.075,
        fp16_topk=1,
    )

    assert policy["schema_version"] == "flashvsr.lsgquant.a4w4_policy.v1"
    assert policy["default"] == {"mode": "a4w4", "activation_qdq_mode": "draq_symmetric", "rank": 16}
    assert policy["layers"]["layer.low"]["mode"] == "a4w4"
    assert policy["layers"]["layer.low"]["rank"] == 16
    assert policy["layers"]["layer.mid"]["mode"] == "a4w4"
    assert policy["layers"]["layer.mid"]["rank"] == 32
    assert policy["layers"]["layer.mid"]["adaptation"] == "light"
    assert policy["layers"]["layer.high"]["mode"] == "a8w8"
    assert policy["layers"]["layer.high"]["rank"] == 32
    assert policy["layers"]["layer.catastrophic"]["mode"] == "fp16_skip"
    assert policy["fallback_counts"] == {"a4w4_rank16": 1, "a4w4_rank32_light": 1, "a8w8_rank32": 1, "fp16_skip": 1}
    assert "Wan VAE remains unquantized" in policy["scope"]


def test_a4w4_fallback_policy_validates_with_existing_policy_loader(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(build_lsgquant_a4w4_fallback_policy(_cache(), fp16_topk=1)))

    loaded = load_layer_policy(path)

    assert loaded["layers"]["layer.high"]["activation_qdq_mode"] == "draq_symmetric"
    assert loaded["layers"]["layer.catastrophic"]["mode"] == "fp16_skip"


def test_lsgquant_policy_cli_writes_pr9_a4w4_policy(tmp_path):
    cache_path = tmp_path / "calib.json"
    out_path = tmp_path / "policy_a4w4.json"
    cache_path.write_text(json.dumps(_cache()))

    subprocess.run(
        [
            sys.executable,
            "scripts/ptq/lsgquant_policy.py",
            "--calibration_cache",
            str(cache_path),
            "--out",
            str(out_path),
            "--policy_type",
            "a4w4_fallback",
            "--fp16_topk",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    written = json.loads(out_path.read_text())
    assert written["schema_version"] == "flashvsr.lsgquant.a4w4_policy.v1"
    assert written["fallback_counts"]["fp16_skip"] == 1


def test_qao_conversion_uses_a4w4_policy_rank_and_a8_high_sensitivity_fallback():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    policy = {
        "0": {"mode": "a4w4", "activation_qdq_mode": "draq_symmetric", "rank": 1},
        "1": {"mode": "a8w8", "activation_qdq_mode": "draq_symmetric", "rank": 2},
    }

    manifest_layers = convert_model_to_lsgquant_qao(
        model,
        mode="a4w4",
        rank=32,
        rounds=1,
        layer_policy=policy,
        activation_qdq_mode="draq_symmetric",
    )

    assert isinstance(model[0], LSGQuantLinear)
    assert isinstance(model[1], LSGQuantLinear)
    assert int(model[0].residual.activation_mode_code.item()) == 3
    assert model[0].rank == 1
    assert int(model[1].residual.activation_mode_code.item()) == 2
    assert model[1].rank == 2
    assert [layer["mode"] for layer in manifest_layers] == ["a4w4", "a8w8"]
    assert torch.isfinite(model(torch.randn(2, 4))).all()


def test_pr9_eval_manifest_uses_qao_converter_and_records_quality_report_contract(tmp_path):
    checkpoint = tmp_path / "dit.safetensors"
    calib = tmp_path / "calib.json"
    policy = tmp_path / "policy.json"
    out_dir = tmp_path / "eval"
    checkpoint.write_text("placeholder")
    calib.write_text(json.dumps(_cache()))
    policy.write_text(json.dumps(build_lsgquant_a4w4_fallback_policy(_cache(), fp16_topk=1)))

    cmd = build_lsgquant_convert_command(
        checkpoint=checkpoint,
        calibration_cache=calib,
        policy=policy,
        mode="a4w4",
        out_dir=out_dir,
        rank=32,
        qao_rounds=4,
        rotation="identity",
    )
    manifest = build_lsgquant_eval_manifest(
        checkpoint=checkpoint,
        calibration_cache=calib,
        policy=policy,
        mode="a4w4",
        out_dir=out_dir,
        rank=32,
        qao_rounds=4,
        limit=2,
    )

    assert cmd[:3] == [sys.executable, "scripts/ptq/lsgquant_convert.py", "--checkpoint"]
    assert "--rank" in cmd and "32" in cmd
    assert "--qao_rounds" in cmd and "4" in cmd
    assert "--activation_qdq_mode" in cmd and "draq_symmetric" in cmd
    assert manifest["schema_version"] == "flashvsr.lsgquant.a4w4_eval_manifest.v1"
    assert manifest["converted_checkpoint"].endswith("dit_lsgquant_a4w4_rank32_qao.safetensors")
    assert manifest["fallback_counts"] == {"a4w4_rank16": 1, "a4w4_rank32_light": 1, "a8w8_rank32": 1, "fp16_skip": 1}
    assert manifest["compression_report"]["fakequant_quality_only"] is True
    assert manifest["quality_delta"]["status"] == "pending_eval"


def test_pr9_eval_cli_dry_run_accepts_a4w4_rank_rounds(tmp_path):
    checkpoint = tmp_path / "dit.safetensors"
    calib = tmp_path / "calib.json"
    policy = tmp_path / "policy.json"
    out_dir = tmp_path / "eval"
    checkpoint.write_text("placeholder")
    calib.write_text(json.dumps(_cache()))
    policy.write_text(json.dumps(build_lsgquant_a4w4_fallback_policy(_cache(), fp16_topk=1)))

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
            "a4w4",
            "--rank",
            "32",
            "--qao_rounds",
            "4",
            "--out_dir",
            str(out_dir),
            "--dry_run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads((out_dir / "lsgquant_eval_manifest.json").read_text())
    assert manifest["mode"] == "a4w4"
    assert manifest["qao"] == {"rank": 32, "rounds": 4, "rotation": "identity"}
