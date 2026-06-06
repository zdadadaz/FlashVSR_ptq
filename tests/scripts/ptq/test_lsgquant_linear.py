import torch
import torch.nn as nn

from src.models.quantization.fakequant import FakeQuantLinear
from src.models.quantization.lsgquant import LSGQuantLinear
from src.models.quantization import LSGQuantLinear as ExportedLSGQuantLinear


def _seeded_linear(in_features=5, out_features=3, bias=True):
    torch.manual_seed(123)
    linear = nn.Linear(in_features, out_features, bias=bias)
    with torch.no_grad():
        linear.weight.copy_(torch.linspace(-0.7, 0.8, steps=out_features * in_features).view(out_features, in_features))
        if bias:
            linear.bias.copy_(torch.linspace(-0.2, 0.3, steps=out_features))
    return linear


def test_lsgquant_rank0_matches_fakequant_linear_residual_only():
    linear = _seeded_linear()
    fq = FakeQuantLinear.from_float(linear, activation_mode="a16", weight_mode="w8")
    lsg = LSGQuantLinear.from_float(linear, rank=0, activation_mode="a16", weight_mode="w8")
    x = torch.linspace(-1.0, 1.0, steps=20, dtype=torch.float32).view(4, 5)

    torch.testing.assert_close(lsg(x), fq(x), rtol=0, atol=0)


def test_lsgquant_rank_branch_adds_low_rank_path_and_preserves_dtype_shape_bias():
    linear = _seeded_linear(in_features=4, out_features=3, bias=True)
    l1 = torch.tensor([[0.5, -0.25], [0.1, 0.2], [-0.4, 0.3]], dtype=torch.float32)
    l2 = torch.tensor([[0.25, -0.5, 0.75, 0.1], [-0.2, 0.4, 0.3, -0.6]], dtype=torch.float32)
    lsg = LSGQuantLinear.from_float(
        linear,
        rank=2,
        activation_mode="a16",
        weight_mode="w8",
        low_rank_l1=l1,
        low_rank_l2=l2,
    )
    x = torch.randn(2, 6, 4, dtype=torch.float16)

    residual = lsg.residual(x).to(torch.float32)
    low_rank = torch.nn.functional.linear(
        torch.nn.functional.linear(x.to(torch.float32), l2), l1
    )
    expected = (residual + low_rank).to(torch.float16)

    out = lsg(x)
    assert out.shape == (2, 6, 3)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_lsgquant_state_dict_roundtrip_preserves_residual_and_low_rank_buffers():
    linear = _seeded_linear(in_features=4, out_features=3, bias=True)
    lsg = LSGQuantLinear.from_float(linear, rank=2, activation_mode="a16", weight_mode="w4")
    with torch.no_grad():
        lsg.l1_weight.copy_(torch.randn_like(lsg.l1_weight))
        lsg.l2_weight.copy_(torch.randn_like(lsg.l2_weight))
    clone = LSGQuantLinear(4, 3, rank=2, activation_mode="a16", weight_mode="w4", bias=True)

    clone.load_state_dict(lsg.state_dict())
    x = torch.randn(3, 4)

    torch.testing.assert_close(clone(x), lsg(x), rtol=0, atol=0)
    assert clone.state_dict()["l1_weight"].shape == (3, 2)
    assert clone.state_dict()["l2_weight"].shape == (2, 4)


def test_lsgquant_linear_is_exported_from_quantization_package():
    assert ExportedLSGQuantLinear is LSGQuantLinear
