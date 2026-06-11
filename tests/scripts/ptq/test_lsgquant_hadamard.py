import subprocess
import sys

import torch
import torch.nn as nn

from src.models.quantization.qao import (
    apply_hadamard_rotation,
    normalized_hadamard_matrix,
    qao_decompose_weight,
    qao_linear_from_float,
)


def test_normalized_hadamard_matrix_is_orthonormal_for_power_of_two():
    h = normalized_hadamard_matrix(8)

    assert h.shape == (8, 8)
    assert torch.allclose(h @ h.t(), torch.eye(8), atol=1e-6, rtol=1e-6)
    assert torch.allclose(h.t() @ h, torch.eye(8), atol=1e-6, rtol=1e-6)


def test_apply_hadamard_rotation_preserves_shape_for_2d_and_3d_inputs():
    torch.manual_seed(7)
    x2 = torch.randn(4, 8)
    x3 = torch.randn(2, 3, 8)

    y2 = apply_hadamard_rotation(x2)
    y3 = apply_hadamard_rotation(x3)

    assert y2.shape == x2.shape
    assert y3.shape == x3.shape
    assert torch.allclose(torch.linalg.vector_norm(y2, dim=-1), torch.linalg.vector_norm(x2, dim=-1), atol=1e-5)
    assert torch.allclose(torch.linalg.vector_norm(y3, dim=-1), torch.linalg.vector_norm(x3, dim=-1), atol=1e-5)


def test_apply_hadamard_rotation_pads_non_power_of_two_without_shape_change():
    torch.manual_seed(8)
    x = torch.randn(3, 6)

    y = apply_hadamard_rotation(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_qao_decompose_weight_accepts_hadamard_rotation_and_records_it():
    torch.manual_seed(9)
    weight = torch.randn(5, 8)

    result = qao_decompose_weight(weight, weight_bits=8, rank=2, rounds=1, rotation="hadamard")

    assert result.rotation == "hadamard"
    assert result.l1.shape == (5, 2)
    assert result.l2.shape == (2, 8)
    assert result.error_after < result.error_before


def test_qao_linear_from_float_hadamard_runtime_preserves_output_shape_for_power_and_non_power_two():
    torch.manual_seed(10)
    linear_power = nn.Linear(8, 5)
    linear_non_power = nn.Linear(6, 4)
    x_power = torch.randn(2, 3, 8)
    x_non_power = torch.randn(2, 6)

    module_power, result_power = qao_linear_from_float(
        linear_power,
        weight_bits=8,
        rank=2,
        rounds=1,
        rotation="hadamard",
        activation_mode="a16",
    )
    module_non_power, result_non_power = qao_linear_from_float(
        linear_non_power,
        weight_bits=8,
        rank=2,
        rounds=1,
        rotation="hadamard",
        activation_mode="a16",
    )

    assert result_power.rotation == "hadamard"
    assert result_non_power.rotation == "hadamard"
    assert module_power(x_power).shape == (2, 3, 5)
    assert module_non_power(x_non_power).shape == (2, 4)


def test_lsgquant_convert_cli_exposes_hadamard_rotation_choice():
    result = subprocess.run(
        [sys.executable, "scripts/ptq/lsgquant_convert.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--rotation" in result.stdout
    assert "hadamard" in result.stdout
