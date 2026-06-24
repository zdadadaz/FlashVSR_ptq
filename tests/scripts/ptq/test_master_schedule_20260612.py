import json
import subprocess
import sys

import torch
import torch.nn as nn

from scripts.ptq.build_true_sensitive_policy import build_true_sensitive_policy
from scripts.ptq.static_ptq_baseline import build_smoothquant_cache, select_ptq_policy_and_scales
from src.models.quantization.lsgquant import LSGQuantLinear
from src.models.quantization.qao import qao_decompose_weight


def test_true_sensitive_policy_uses_top_mse_and_dynamic_a8w8():
    sensitivity = {
        "blocks.0.self_attn.q": {"mse": 0.9, "rel_l1": 0.1},
        "blocks.0.self_attn.k": {"mse": 0.8, "rel_l1": 0.1},
        "blocks.0.ffn.0": {"mse": 0.1, "rel_l1": 0.01},
        "head.head": {"mse": 0.05, "rel_l1": 0.01},
    }

    policy = build_true_sensitive_policy(sensitivity, fp16_skip_ratio=0.5)

    assert policy["counts"]["fp16_skip"] == 2
    assert policy["layers"]["blocks.0.self_attn.q"]["mode"] == "fp16_skip"
    assert policy["layers"]["blocks.0.ffn.0"]["activation_qdq_mode"] == "dynamic_asymmetric"


def test_static_baseline_can_select_true_sensitivity_and_les_caches(tmp_path):
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    layer_names = ["0", "1"]
    calibration = {
        "0": {"act_min": [-1] * 4, "act_max": [1] * 4},
        "1": {"act_min": [-1] * 4, "act_max": [1] * 4},
    }
    sens_path = tmp_path / "sens.json"
    sens_path.write_text(json.dumps({"0": {"mse": 10.0, "rel_l1": 1.0}, "1": {"mse": 0.1, "rel_l1": 0.01}}))
    les_path = tmp_path / "les.json"
    les_path.write_text(json.dumps({"_metadata": {"schema_version": "flashvsr.learned_equivalent_scaling.v1"}, "0": {"tau": [1, 2, 3, 4]}, "1": {"smoothquant_scale": [1, 1, 1, 1]}}))

    policy, scales = select_ptq_policy_and_scales(
        model=model,
        layer_names=layer_names,
        calibration=calibration,
        fallback_ratio=0.5,
        smoothquant_alpha=0.5,
        sensitivity_cache=str(sens_path),
        les_cache=str(les_path),
    )

    assert policy["layers"]["0"]["mode"] == "fp16_skip"
    assert policy["layers"]["1"]["activation_qdq_mode"] == "dynamic_asymmetric"
    assert scales["0"]["smoothquant_scale"] == [1.0, 2.0, 3.0, 4.0]


def test_true_sensitivity_casts_latents_to_model_dtype_before_patchify():
    source = open("scripts/ptq/true_sensitivity.py", encoding="utf-8").read()

    assert "t_big_int = torch.randint" in source
    assert "x_5d = x_5d.to(device=device, dtype=model_dtype)" in source


def test_les_optimizer_reduces_layer_reconstruction_loss():
    from scripts.ptq.learned_equivalent_scaling import optimize_layer_tau

    torch.manual_seed(12)
    linear = nn.Linear(4, 3, bias=False)
    x = torch.randn(16, 4)

    result = optimize_layer_tau(linear, x, num_steps=20, lr=5e-2)

    assert result["tau"].shape == (4,)
    assert torch.isfinite(result["tau"]).all()
    assert result["final_loss"] <= result["initial_loss"]


def test_build_hadamard_cache_from_calibration_variance(tmp_path):
    calib = {
        "high": {"act_max": [1.0, 10.0, 1.0, 10.0], "act_min": [-1.0, -10.0, -1.0, -10.0]},
        "low": {"act_max": [1.0, 1.1, 0.9, 1.0], "act_min": [-1.0, -1.1, -0.9, -1.0]},
    }
    calib_path = tmp_path / "calib.json"
    calib_path.write_text(json.dumps(calib))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/ptq/build_hadamard_cache.py",
            "--calibration_cache",
            str(calib_path),
            "--output",
            str(tmp_path / "hadamard.json"),
            "--variance_threshold",
            "0.5",
            "--seed",
            "42",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    cache = json.loads((tmp_path / "hadamard.json").read_text())

    assert cache["layers"]["high"]["enabled"] is True
    assert cache["layers"]["low"]["enabled"] is False
    assert "enabled_layers" in result.stdout


def test_lsgquant_forward_and_qao_svd_contract():
    torch.manual_seed(13)
    linear = nn.Linear(6, 4)
    module = LSGQuantLinear.from_float(linear, rank=2, activation_mode="a16", weight_mode="w8")
    qao = qao_decompose_weight(linear.weight, weight_bits=8, rank=2, rounds=1)
    module.set_low_rank(qao.l1.to(linear.weight.dtype), qao.l2.to(linear.weight.dtype))
    x = torch.randn(3, 6)

    y = module(x)

    assert y.shape == (3, 4)
    assert qao.l1.shape == (4, 2)
    assert qao.l2.shape == (2, 6)
    assert torch.isfinite(y).all()


def test_lsgquant_full_convert_cli_exposes_master_schedule_flags():
    result = subprocess.run(
        [sys.executable, "scripts/ptq/lsgquant_full_convert.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--rank_policy" in result.stdout
    assert "--finetune_epochs" in result.stdout
    assert "--weight_mode" in result.stdout
