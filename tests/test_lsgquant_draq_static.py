import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

from src.models.quantization.fakequant import (
    FakeQuantLinear,
    build_draq_static_cache_entry,
    build_volts_draq_policy,
    convert_model_to_fakequant,
)
from scripts.ptq.fakequant_convert import load_calibration_cache


def _identity_fq(mode: str, in_features: int = 4) -> FakeQuantLinear:
    linear = nn.Linear(in_features, in_features, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(in_features))
    fq = FakeQuantLinear.from_float(
        linear,
        activation_mode="a8",
        weight_mode="w8",
        activation_qdq_mode=mode,
        draq_s=torch.ones(in_features),
        draq_d=torch.tensor(1.0),
        draq_d_buckets=torch.tensor([0.5, 1.0, 2.0]),
    )
    # Remove weight quantization error from activation-QDQ-focused tests.
    fq.weight_int.copy_(torch.eye(in_features, dtype=torch.float32).to(torch.int8))
    fq.weight_scale.fill_(1.0)
    return fq


def _reference_draq_dynamic(x: torch.Tensor, qmin=-128.0, qmax=127.0):
    x_float = x.to(torch.float32)
    reduce_channel = tuple(range(x_float.dim() - 1))
    s = torch.amax(torch.abs(x_float), dim=reduce_channel, keepdim=True).clamp(min=1e-6)
    x_norm = x_float / s
    d = torch.amax(torch.abs(x_norm), dim=-1, keepdim=True).clamp(min=1e-6)
    x_q = torch.clamp(torch.round(qmax * x_norm / d), qmin, qmax).to(torch.int8)
    return (x_q.to(torch.float32) / qmax) * d * s


def _reference_draq_static_s(x: torch.Tensor, s: torch.Tensor, qmin=-128.0, qmax=127.0):
    x_float = x.to(torch.float32)
    s = s.to(torch.float32).reshape(*([1] * (x_float.dim() - 1)), x_float.shape[-1]).clamp(min=1e-6)
    x_norm = x_float / s
    d = torch.amax(torch.abs(x_norm), dim=-1, keepdim=True).clamp(min=1e-6)
    x_q = torch.clamp(torch.round(qmax * x_norm / d), qmin, qmax).to(torch.int8)
    return (x_q.to(torch.float32) / qmax) * d * s


def _reference_draq_static_sd(x: torch.Tensor, s: torch.Tensor, d: torch.Tensor, qmin=-128.0, qmax=127.0):
    x_float = x.to(torch.float32)
    s = s.to(torch.float32).reshape(*([1] * (x_float.dim() - 1)), x_float.shape[-1]).clamp(min=1e-6)
    d = d.to(torch.float32).reshape(*([1] * x_float.dim())).clamp(min=1e-6)
    x_norm = x_float / s
    x_q = torch.clamp(torch.round(qmax * x_norm / d), qmin, qmax).to(torch.int8)
    return (x_q.to(torch.float32) / qmax) * d * s


def test_draq_symmetric_matches_reference_for_2d_and_3d():
    fq = _identity_fq("draq_symmetric")
    for x in (torch.tensor([[0.2, -1.0, 0.5, 2.0], [1.2, -0.3, 0.1, -0.7]]), torch.randn(2, 3, 4)):
        assert torch.allclose(fq(x), _reference_draq_dynamic(x), atol=1e-6)


def test_draq_static_s_uses_calibrated_channel_scale_and_runtime_token_scale():
    x = torch.tensor([[[0.2, -1.0, 0.5, 2.0], [1.2, -0.3, 0.1, -0.7]]])
    s = torch.tensor([0.5, 2.0, 1.0, 4.0])
    fq = _identity_fq("draq_static_s")
    fq.set_draq_static_params(s=s)
    assert torch.allclose(fq(x), _reference_draq_static_s(x, s), atol=1e-6)


def test_draq_static_sd_layer_uses_no_runtime_dynamic_range():
    x = torch.tensor([[0.2, -1.0, 0.5, 2.0], [1.2, -0.3, 0.1, -0.7]])
    s = torch.tensor([0.5, 2.0, 1.0, 4.0])
    d = torch.tensor(0.75)
    fq = _identity_fq("draq_static_sd_layer")
    fq.set_draq_static_params(s=s, d=d)
    assert torch.allclose(fq(x), _reference_draq_static_sd(x, s, d), atol=1e-6)


def test_draq_static_sd_bucket_uses_selected_static_bucket():
    x = torch.tensor([[0.2, -1.0, 0.5, 2.0], [1.2, -0.3, 0.1, -0.7]])
    s = torch.tensor([0.5, 2.0, 1.0, 4.0])
    buckets = torch.tensor([0.25, 0.75, 1.5])
    fq = _identity_fq("draq_static_sd_bucket")
    fq.set_draq_static_params(s=s, d_buckets=buckets, bucket_index=1)
    assert torch.allclose(fq(x), _reference_draq_static_sd(x, s, torch.tensor(0.75)), atol=1e-6)


def test_static_draq_buffers_round_trip_state_dict():
    fq = _identity_fq("draq_static_sd_bucket")
    fq.set_draq_static_params(s=torch.tensor([1.0, 2.0, 3.0, 4.0]), d=torch.tensor(0.5), d_buckets=torch.tensor([0.25, 0.5]), bucket_index=1)
    clone = _identity_fq("draq_static_sd_bucket")
    clone.load_state_dict(fq.state_dict())
    assert int(clone.activation_qdq_mode.item()) == 6
    assert torch.allclose(clone.draq_s.reshape(-1), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(clone.draq_d.reshape(-1), torch.tensor([0.5]))
    assert torch.allclose(clone.draq_d_buckets[:2], torch.tensor([0.25, 0.5]))
    assert int(clone.draq_bucket_index.item()) == 1


def test_build_draq_static_cache_entry_collects_percentiles_and_volts_tier():
    samples = [
        torch.tensor([[1.0, -2.0, 0.5], [2.0, 1.0, -1.0]]),
        torch.tensor([[0.5, -4.0, 1.5], [3.0, 2.0, -0.25]]),
    ]
    entry = build_draq_static_cache_entry(samples, bucket_ids=["first", "last"], delta1=0.001, delta2=0.075)
    for key in [
        "draq_s_absmax",
        "draq_s_percentile_99",
        "draq_s_percentile_999",
        "draq_d_absmax",
        "draq_d_percentile_99",
        "draq_d_percentile_999",
        "mu_samples_mean",
        "mu_mean",
        "mu_var",
        "volts_tier",
    ]:
        assert key in entry
    assert len(entry["draq_s_absmax"]) == 3
    assert "draq_d_by_bucket" in entry
    assert set(entry["draq_d_by_bucket"]) == {"first", "last"}


def test_load_calibration_cache_preserves_static_draq_fields(tmp_path):
    cache = tmp_path / "calib.json"
    cache.write_text(json.dumps({
        "layer": {
            "act_scale": [1.0, 1.0],
            "zero_point": [0, 0],
            "draq_s_percentile_999": [0.5, 2.0],
            "draq_d_percentile_999": 0.75,
            "draq_d_by_bucket": {"0": 0.5, "1": 1.0},
            "mu_var": 0.01,
            "volts_tier": "light",
        }
    }))
    loaded = load_calibration_cache(str(cache), device="cpu")
    assert torch.allclose(loaded["layer"]["draq_s_percentile_999"], torch.tensor([0.5, 2.0]))
    assert torch.allclose(loaded["layer"]["draq_d_percentile_999"], torch.tensor(0.75))
    assert torch.allclose(loaded["layer"]["draq_d_by_bucket"], torch.tensor([0.5, 1.0]))
    assert loaded["layer"]["volts_tier"] == "light"


def test_convert_model_to_fakequant_fails_static_draq_when_cache_missing():
    model = nn.Sequential(nn.Linear(4, 4))
    try:
        convert_model_to_fakequant(model, mode="a8w8", activation_qdq_mode="draq_static_s", act_stats={})
    except RuntimeError as exc:
        assert "requires DRAQ static calibration" in str(exc)
    else:
        raise AssertionError("expected missing static DRAQ calibration to fail")


def test_convert_model_to_fakequant_applies_static_draq_cache_and_policy():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    act_stats = {
        "0": {"draq_s_percentile_999": [1, 2, 3, 4], "draq_d_percentile_999": 0.5},
        "1": {"draq_s_percentile_999": [2, 2, 2, 2], "draq_d_percentile_999": 1.0},
    }
    policy = {"1": {"mode": "a8w8", "activation_qdq_mode": "draq_static_sd_layer"}}
    convert_model_to_fakequant(
        model,
        mode="a8w8",
        act_stats=act_stats,
        activation_qdq_mode="draq_static_s",
        layer_policy=policy,
    )
    assert isinstance(model[0], FakeQuantLinear)
    assert int(model[0].activation_qdq_mode.item()) == 4
    assert int(model[1].activation_qdq_mode.item()) == 5
    assert torch.allclose(model[1].draq_d.reshape(-1), torch.tensor([1.0]))


def test_build_volts_draq_policy_maps_tiers_to_static_dynamic_modes():
    cache = {
        "low": {"volts_tier": "frozen"},
        "mid": {"volts_tier": "light"},
        "high": {"volts_tier": "full"},
        "bad": {"volts_tier": "catastrophic"},
    }
    policy = build_volts_draq_policy(cache)
    assert policy["layers"]["low"]["activation_qdq_mode"] == "draq_static_sd_layer"
    assert policy["layers"]["mid"]["activation_qdq_mode"] == "draq_static_s"
    assert policy["layers"]["high"]["activation_qdq_mode"] == "draq_symmetric"
    assert policy["layers"]["bad"]["mode"] == "a16w8"
    assert policy["summary"]["mode_counts"]["a16w8"] == 1


def test_fakequant_convert_cli_exposes_static_draq_modes():
    script = Path("scripts/ptq/fakequant_convert.py")
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    assert "draq_static_s" in result.stdout
    assert "draq_static_sd_layer" in result.stdout
    assert "draq_static_sd_bucket" in result.stdout
