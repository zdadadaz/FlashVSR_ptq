import json
import subprocess
import sys

import torch
import torch.nn as nn

from src.models.quantization.lsgquant import LSGQuantLinear
from src.models.quantization.qao import (
    QAOResult,
    dequantize_weight,
    qao_decompose_weight,
    qao_linear_from_float,
    quantize_weight_symmetric,
)
from scripts.ptq.lsgquant_convert import build_lsgquant_qao_manifest


def _structured_weight():
    torch.manual_seed(2026)
    base = torch.linspace(-1.3, 1.1, steps=48, dtype=torch.float32).view(6, 8)
    low_rank = torch.outer(
        torch.tensor([1.0, -0.8, 0.5, -0.3, 0.2, -0.1]),
        torch.tensor([0.6, -0.4, 0.2, 0.1, -0.2, 0.3, -0.5, 0.7]),
    )
    return base + 0.35 * low_rank


def _fro_error(weight, result: QAOResult):
    reconstructed = dequantize_weight(result.residual_int, result.residual_scale) + result.l1 @ result.l2
    return torch.linalg.vector_norm(weight - reconstructed).item()


def test_qao_decompose_weight_improves_residual_error_over_pure_w8_quantization():
    weight = _structured_weight()
    pure_int, pure_scale = quantize_weight_symmetric(weight, bits=8)
    pure = dequantize_weight(pure_int, pure_scale)
    pure_error = torch.linalg.vector_norm(weight - pure).item()

    result = qao_decompose_weight(weight, weight_bits=8, rank=2, rounds=3)

    assert result.weight_bits == 8
    assert result.rank == 2
    assert result.rounds == 3
    assert result.l1.shape == (6, 2)
    assert result.l2.shape == (2, 8)
    assert result.residual_int.dtype == torch.int8
    assert result.residual_scale.shape == (6, 1)
    assert _fro_error(weight, result) < pure_error
    assert result.error_after < result.error_before


def test_qao_decompose_weight_supports_w4_and_roundtrip_shapes():
    weight = _structured_weight()

    result = qao_decompose_weight(weight, weight_bits=4, rank=3, rounds=2)
    residual = dequantize_weight(result.residual_int, result.residual_scale)
    reconstructed = residual + result.l1 @ result.l2

    assert result.residual_int.dtype == torch.int8
    assert int(result.residual_int.min()) >= -7
    assert int(result.residual_int.max()) <= 7
    assert reconstructed.shape == weight.shape
    assert torch.isfinite(reconstructed).all()


def test_qao_rejects_unknown_rotation():
    try:
        qao_decompose_weight(_structured_weight(), weight_bits=8, rank=2, rotation="fft")
    except ValueError as exc:
        assert "identity" in str(exc) and "hadamard" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported rotation")


def test_qao_linear_from_float_materializes_lsgquant_residual_and_low_rank_buffers():
    linear = nn.Linear(8, 6, bias=True)
    with torch.no_grad():
        linear.weight.copy_(_structured_weight())
        linear.bias.copy_(torch.linspace(-0.2, 0.2, steps=6))

    module, result = qao_linear_from_float(
        linear,
        weight_bits=8,
        rank=2,
        rounds=3,
        activation_mode="a16",
        activation_qdq_mode="draq_symmetric",
    )

    assert isinstance(module, LSGQuantLinear)
    assert module.rank == 2
    assert torch.equal(module.residual.weight_int, result.residual_int)
    assert torch.allclose(module.residual.weight_scale, result.residual_scale)
    assert torch.allclose(module.l1_weight, result.l1)
    assert torch.allclose(module.l2_weight, result.l2)
    x = torch.randn(3, 8)
    reconstructed = dequantize_weight(result.residual_int, result.residual_scale) + result.l1 @ result.l2
    expected = torch.nn.functional.linear(x, reconstructed, linear.bias)
    assert torch.allclose(module(x), expected, atol=1e-5, rtol=1e-5)


def test_convert_model_to_lsgquant_qao_replaces_linear_layers_and_honors_max_layers():
    from src.models.quantization.qao import convert_model_to_lsgquant_qao

    model = nn.Sequential(nn.Linear(8, 6), nn.GELU(), nn.Linear(6, 4))

    layer_errors = convert_model_to_lsgquant_qao(model, mode="a8w8", rank=2, rounds=1, max_layers=1)

    assert isinstance(model[0], LSGQuantLinear)
    assert isinstance(model[2], nn.Linear)
    assert len(layer_errors) == 1
    assert layer_errors[0]["name"] == "0"
    assert layer_errors[0]["error_after"] < layer_errors[0]["error_before"]


def test_lsgquant_convert_manifest_records_qao_settings_and_layer_errors(tmp_path):
    layer_errors = [
        {"name": "blocks.0.self_attn.q", "rank": 2, "weight_bits": 8, "error_before": 0.2, "error_after": 0.1, "time_sec": 0.01},
        {"name": "blocks.0.ffn.0", "rank": 2, "weight_bits": 8, "error_before": 0.3, "error_after": 0.12, "time_sec": 0.02},
    ]

    manifest = build_lsgquant_qao_manifest(
        checkpoint=tmp_path / "dit.safetensors",
        calibration_cache=tmp_path / "calib.json",
        policy=tmp_path / "policy.json",
        mode="a8w8",
        output=tmp_path / "dit_lsg_rank2.safetensors",
        rank=2,
        qao_rounds=4,
        max_layers=2,
        layer_errors=layer_errors,
    )

    assert manifest["schema_version"] == "flashvsr.lsgquant.qao_manifest.v1"
    assert manifest["qao"] == {"rank": 2, "rounds": 4, "rotation": "identity"}
    assert manifest["layers"][0]["error_after"] < manifest["layers"][0]["error_before"]
    assert manifest["summary"]["layers"] == 2
    assert manifest["summary"]["improved_layers"] == 2


def test_lsgquant_convert_cli_exposes_qao_flags():
    result = subprocess.run(
        [sys.executable, "scripts/ptq/lsgquant_convert.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--rank" in result.stdout
    assert "--qao_rounds" in result.stdout
    assert "--max_layers" in result.stdout
    assert "--manifest" in result.stdout
