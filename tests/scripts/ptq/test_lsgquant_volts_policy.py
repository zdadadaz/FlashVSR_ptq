import json
import subprocess
import sys

import torch

from scripts.ptq.fakequant_calibrate import build_lsgquant_calibration_cache
from src.models.quantization.policy import build_lsgquant_volts_policy, load_layer_policy


def _stats(act_min, act_max, act_mean, mu_samples_mean):
    return {
        "act_scale": torch.tensor([0.01, 0.02]),
        "zero_point": torch.tensor([0, -1]),
        "act_min": torch.tensor(act_min, dtype=torch.float32),
        "act_max": torch.tensor(act_max, dtype=torch.float32),
        "act_mean": torch.tensor(act_mean, dtype=torch.float32),
        "mu_samples_mean": torch.tensor(mu_samples_mean, dtype=torch.float32),
    }


def test_calibration_cache_serializes_lsgquant_volts_statistics():
    cache = build_lsgquant_calibration_cache(
        act_stats={
            "blocks.0.self_attn.q": _stats([-1.0, -0.5], [1.0, 0.5], [0.1, -0.1], [[0.1, 0.2], [0.2, 0.4]]),
        },
        metadata={"mode": "a8w8", "num_samples": 2, "num_videos": 50},
    )

    assert cache["_metadata"]["schema_version"] == "flashvsr.lsgquant.calibration.v1"
    assert "mu_mean" in cache["_metadata"]["stats"]
    assert cache["blocks.0.self_attn.q"]["act_min"] == [-1.0, -0.5]
    assert cache["blocks.0.self_attn.q"]["mu_samples_mean"] == [[0.10000000149011612, 0.20000000298023224], [0.20000000298023224, 0.4000000059604645]]
    assert abs(cache["blocks.0.self_attn.q"]["mu_var"] - 0.0125) < 1e-7


def test_lsgquant_volts_policy_assigns_three_tiers_from_absolute_thresholds():
    cache = {
        "_metadata": {"schema_version": "flashvsr.lsgquant.calibration.v1"},
        "blocks.0.self_attn.q": {"mu_var": 0.0005},
        "blocks.0.self_attn.k": {"mu_var": 0.0100},
        "blocks.0.ffn.0": {"mu_var": 0.1000},
    }

    policy = build_lsgquant_volts_policy(cache, delta1=0.001, delta2=0.075)

    assert policy["schema_version"] == "flashvsr.lsgquant.policy.v1"
    assert policy["default"] == {"mode": "a8w8", "activation_qdq_mode": "draq_symmetric"}
    assert policy["layers"]["blocks.0.self_attn.q"]["tier"] == "frozen"
    assert policy["layers"]["blocks.0.self_attn.k"]["tier"] == "light"
    assert policy["layers"]["blocks.0.ffn.0"]["tier"] == "full"
    assert policy["counts"] == {"frozen": 1, "light": 1, "full": 1}


def test_lsgquant_volts_policy_percentile_thresholds_avoid_degenerate_counts():
    cache = {
        "_metadata": {"schema_version": "flashvsr.lsgquant.calibration.v1"},
        "layer.low": {"mu_var": 1.0},
        "layer.mid": {"mu_var": 2.0},
        "layer.high": {"mu_var": 3.0},
    }

    policy = build_lsgquant_volts_policy(cache, threshold_mode="percentile", delta1=34.0, delta2=67.0)

    assert policy["thresholds"]["mode"] == "percentile"
    assert policy["counts"] == {"frozen": 1, "light": 1, "full": 1}


def test_lsgquant_policy_json_validation_accepts_draq_qdq(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"layers": {"x": {"mode": "a8w8", "activation_qdq_mode": "draq_symmetric"}}}))

    loaded = load_layer_policy(path)

    assert loaded["layers"]["x"]["activation_qdq_mode"] == "draq_symmetric"


def test_lsgquant_policy_cli_writes_json_from_cache(tmp_path):
    cache_path = tmp_path / "calib.json"
    out_path = tmp_path / "policy.json"
    cache_path.write_text(json.dumps({
        "_metadata": {"schema_version": "flashvsr.lsgquant.calibration.v1"},
        "layer.low": {"mu_var": 0.0005},
        "layer.high": {"mu_var": 0.1},
    }))

    subprocess.run(
        [
            sys.executable,
            "scripts/ptq/lsgquant_policy.py",
            "--calibration_cache",
            str(cache_path),
            "--out",
            str(out_path),
            "--delta1",
            "0.001",
            "--delta2",
            "0.075",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    written = json.loads(out_path.read_text())
    assert written["layers"]["layer.low"]["tier"] == "frozen"
    assert written["layers"]["layer.high"]["tier"] == "full"
