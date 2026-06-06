import json
import subprocess
import sys

import torch
import torch.nn as nn

from scripts.ptq.generate_layer_policy import build_policy_from_layer_names
from src.models.quantization.fakequant import FakeQuantLinear, convert_model_to_fakequant
from src.models.quantization.policy import load_layer_policy


def _reference_dynamic_symmetric_a4(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_float = x.to(torch.float32)
    x_scale = torch.amax(torch.abs(x_float), dim=-1, keepdim=True).clamp(min=eps) / 7.0
    x_q = torch.clamp(torch.round(x_float / x_scale), -7, 7)
    return x_q * x_scale


def _identity_a4w4(in_features: int) -> FakeQuantLinear:
    linear = nn.Linear(in_features, in_features, bias=False)
    linear.weight.data.copy_(torch.eye(in_features))
    return FakeQuantLinear.from_float(
        linear,
        activation_mode="a4",
        weight_mode="w4",
        activation_qdq_mode="dynamic_symmetric",
    )


def test_a4_activation_qdq_matches_reference_for_2d_inputs():
    fq = _identity_a4w4(4)
    x = torch.tensor(
        [
            [0.25, -1.50, 3.00, -0.75],
            [2.00, -0.50, 0.125, 1.25],
        ],
        dtype=torch.float32,
    )

    out = fq(x)

    assert int(fq.activation_mode_code.item()) == 3
    assert out.shape == x.shape
    assert torch.allclose(out, _reference_dynamic_symmetric_a4(x), atol=1e-6)


def test_a4_activation_qdq_matches_reference_for_3d_token_inputs():
    fq = _identity_a4w4(4)
    x = torch.tensor(
        [
            [[0.25, -1.50, 3.00, -0.75], [1.00, -0.25, 0.50, -2.25]],
            [[2.00, -0.50, 0.125, 1.25], [-3.00, 0.75, -0.50, 0.25]],
        ],
        dtype=torch.float32,
    )

    out = fq(x)

    assert out.shape == x.shape
    assert torch.allclose(out, _reference_dynamic_symmetric_a4(x), atol=1e-6)


def test_convert_model_to_fakequant_accepts_a4w4_without_calibration_cache():
    model = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 2))

    converted = convert_model_to_fakequant(model, mode="a4w4", act_stats=None, activation_qdq_mode="dynamic_symmetric")

    assert isinstance(converted[0], FakeQuantLinear)
    assert int(converted[0].activation_mode_code.item()) == 3
    assert converted._fakequant_conversion_summary["mode_counts"] == {"a4w4": 2}
    assert converted(torch.randn(2, 5, 4)).shape == (2, 5, 2)


def test_policy_and_generator_allow_a4w4(tmp_path):
    policy = build_policy_from_layer_names(
        ["blocks.0.self_attn.q", "blocks.0.ffn.0"],
        default_mode="a4w4",
        sensitive_mode="a4w4",
        activation_qdq_mode="dynamic_symmetric",
    )
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))

    loaded = load_layer_policy(path)

    assert loaded["default"] == {"mode": "a4w4", "activation_qdq_mode": "dynamic_symmetric"}
    assert loaded["layers"]["blocks.0.ffn.0"]["mode"] == "a4w4"


def test_fakequant_convert_and_cli_expose_a4w4():
    convert_help = subprocess.run(
        [sys.executable, "scripts/ptq/fakequant_convert.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    cli_help = subprocess.run(
        [sys.executable, "cli_main.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "a4w4" in convert_help.stdout
    assert "FakeQuant_A4W4" in cli_help.stdout
