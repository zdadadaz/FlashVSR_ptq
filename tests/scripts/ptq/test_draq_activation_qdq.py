import subprocess
import sys

import torch
import torch.nn as nn

from src.models.quantization.fakequant import FakeQuantLinear, convert_model_to_fakequant
from src.models.quantization.qat import QuantAwareLinear, quant_dequant_activation_ste


def _reference_draq(x: torch.Tensor, *, qrange: str = "signed_symmetric", eps: float = 1e-6) -> torch.Tensor:
    x_float = x.to(torch.float32)
    if qrange == "signed_full":
        qmin, qmax = -128.0, 127.0
    elif qrange == "signed_symmetric":
        qmin, qmax = -127.0, 127.0
    else:
        raise ValueError(qrange)
    reduce_channel = tuple(range(x_float.dim() - 1))
    s = torch.amax(torch.abs(x_float), dim=reduce_channel, keepdim=True).clamp(min=eps)
    x_norm = x_float / s
    d = torch.amax(torch.abs(x_norm), dim=-1, keepdim=True).clamp(min=eps)
    x_q = torch.clamp(torch.round(qmax * x_norm / d), qmin, qmax)
    return (x_q / qmax) * d * s


def _identity_fakequant_linear(in_features: int, *, qrange: str = "signed_symmetric") -> FakeQuantLinear:
    linear = nn.Linear(in_features, in_features, bias=False)
    linear.weight.data.copy_(torch.eye(in_features))
    return FakeQuantLinear.from_float(
        linear,
        activation_mode="a8",
        weight_mode="w8",
        activation_qdq_mode="draq_symmetric",
        draq_qrange=qrange,
    )


def test_draq_activation_qdq_matches_lsgquant_formula_for_2d_inputs():
    fq = _identity_fakequant_linear(4)
    x = torch.tensor(
        [
            [0.25, -1.50, 3.00, -0.75],
            [2.00, -0.50, 0.125, 1.25],
        ],
        dtype=torch.float32,
    )

    out = fq(x)

    assert int(fq.activation_qdq_mode.item()) == 3
    assert torch.allclose(out, _reference_draq(x), atol=1e-6)


def test_draq_activation_qdq_matches_lsgquant_formula_for_3d_token_inputs():
    fq = _identity_fakequant_linear(4)
    x = torch.tensor(
        [
            [[0.25, -1.50, 3.00, -0.75], [1.00, -0.25, 0.50, -2.25]],
            [[2.00, -0.50, 0.125, 1.25], [-3.00, 0.75, -0.50, 0.25]],
        ],
        dtype=torch.float32,
    )

    out = fq(x)

    assert out.shape == x.shape
    assert torch.allclose(out, _reference_draq(x), atol=1e-6)


def test_draq_signed_full_qrange_is_configurable():
    fq = _identity_fakequant_linear(4, qrange="signed_full")
    x = torch.tensor([[-1.0, 0.5, 0.25, 0.125]], dtype=torch.float32)

    out = fq(x)

    assert int(fq.draq_qrange.item()) == 1
    assert torch.allclose(out, _reference_draq(x, qrange="signed_full"), atol=1e-6)


def test_draq_qat_ste_matches_runtime_qdq_and_exports_mode():
    x = torch.randn(2, 3, 4, requires_grad=True)

    y = quant_dequant_activation_ste(x, activation_mode="a8", activation_qdq_mode="draq_symmetric")
    y.sum().backward()

    assert torch.allclose(y.detach(), _reference_draq(x.detach()), atol=1e-6)
    assert x.grad is not None
    assert torch.count_nonzero(x.grad).item() > 0

    ref = nn.Linear(4, 3)
    qat = QuantAwareLinear.from_float(ref, activation_mode="a8", weight_mode="w8", activation_qdq_mode="draq_symmetric")
    exported = qat.to_fakequant_linear()

    assert isinstance(exported, FakeQuantLinear)
    assert int(exported.activation_qdq_mode.item()) == 3


def test_convert_model_to_fakequant_accepts_draq_without_calibration_cache():
    model = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 2))

    converted = convert_model_to_fakequant(model, mode="a8w8", act_stats=None, activation_qdq_mode="draq_symmetric")

    assert isinstance(converted[0], FakeQuantLinear)
    assert isinstance(converted[2], FakeQuantLinear)
    assert converted._fakequant_conversion_summary["activation_qdq_mode"] == "draq_symmetric"
    assert converted(torch.randn(2, 5, 4)).shape == (2, 5, 2)


def test_fakequant_convert_cli_exposes_draq_activation_mode():
    result = subprocess.run(
        [sys.executable, "scripts/ptq/fakequant_convert.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "draq_symmetric" in result.stdout
    assert "--draq_qrange" in result.stdout
