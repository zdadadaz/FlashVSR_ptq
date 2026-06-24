import torch
import torch.nn as nn

from src.models.quantization.fakequant import FakeQuantLinear


def _identity_linear(features: int) -> nn.Linear:
    linear = nn.Linear(features, features, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(features))
    return linear


def test_static_token_asymmetric_uses_token_qparams():
    x = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [10.0, 12.0, 14.0, 16.0]]], dtype=torch.float32)
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = ((x_max - x_min) / 255.0).clamp(min=1e-6)
    zp = torch.round(-128.0 - x_min / scale).clamp(-128, 127).to(torch.int32)

    fq = FakeQuantLinear.from_float(
        _identity_linear(4),
        activation_mode="a8",
        weight_mode="w8",
        activation_qdq_mode="static_token_asymmetric",
        act_scale=scale,
        act_zero_point=zp,
    )

    assert int(fq.activation_qdq_mode.item()) == 8
    assert tuple(fq.act_scale.shape) == (1, 2, 1)
    y = fq(x)

    x_q = torch.clamp(torch.round(x / scale + zp.float()), -128, 127).to(torch.int8)
    expected = (x_q.float() - zp.float()) * scale
    assert torch.allclose(y, expected, atol=1e-5)


def test_static_token_asymmetric_state_dict_resizes_qparam_buffers():
    fq = FakeQuantLinear.from_float(
        _identity_linear(4),
        activation_mode="a8",
        weight_mode="w8",
        activation_qdq_mode="static_token_asymmetric",
        act_scale=torch.ones(1, 3, 1) * 0.1,
        act_zero_point=torch.zeros(1, 3, 1, dtype=torch.int32),
    )
    clone = FakeQuantLinear(4, 4, activation_mode="a8", weight_mode="w8", activation_qdq_mode="static_token_asymmetric", bias=False)

    clone.load_state_dict(fq.state_dict())

    assert tuple(clone.act_scale.shape) == (1, 3, 1)
    assert tuple(clone.act_zero_point.shape) == (1, 3, 1)
    assert int(clone.activation_qdq_mode.item()) == 8
