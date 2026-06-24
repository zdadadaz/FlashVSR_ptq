import torch
import torch.nn as nn

from src.models.quantization.fakequant import (
    ACTIVATION_QDQ_MODE_TO_ID,
    FakeQuantConv2d,
    FakeQuantConv3d,
    FakeQuantLinear,
    attach_fakequant_conv_calibration_hooks,
    convert_ops_to_fakequant,
    export_fakequant_conv_calibration_cache,
)


def test_conv2d_static_from_float_injects_cache_and_forward_runs():
    conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    cache = {
        "act_scale": [0.01, 0.02, 0.03],
        "act_zero_point": [0, 0, 0],
    }
    fq = FakeQuantConv2d.from_float(
        conv,
        activation_mode="a8",
        weight_mode="w8",
        activation_qdq_mode="static_tensor_symmetric",
        cache=cache,
    )
    assert int(fq.activation_qdq_mode.item()) == ACTIVATION_QDQ_MODE_TO_ID["static_tensor_symmetric"]
    assert tuple(fq.act_scale.shape) == (1, 3, 1, 1)
    assert torch.allclose(fq.act_scale.reshape(-1), torch.tensor(cache["act_scale"]))
    y = fq(torch.randn(2, 3, 8, 8))
    assert y.shape == (2, 4, 8, 8)


def test_conv3d_dynamic_and_static_cache_forward_runs():
    conv = nn.Conv3d(2, 3, kernel_size=1)
    dyn = FakeQuantConv3d.from_float(conv, activation_mode="a8", weight_mode="w8")
    sta = FakeQuantConv3d.from_float(
        conv,
        activation_mode="a8",
        weight_mode="w8",
        activation_qdq_mode="static_tensor_symmetric",
        cache={"act_scale": [0.05, 0.06], "act_zero_point": [0, 0]},
    )
    x = torch.randn(1, 2, 4, 5, 6)
    assert dyn(x).shape == sta(x).shape == (1, 3, 4, 5, 6)


def test_hook_cache_exports_and_convert_ops_consumes_prefixed_tcdecoder_keys():
    model = nn.Sequential(nn.Conv2d(3, 5, 1), nn.ReLU(), nn.Conv2d(5, 7, 1))
    hooks, stats = attach_fakequant_conv_calibration_hooks(model, prefix="tcdecoder", op_types=("conv2d",))
    try:
        _ = model(torch.randn(2, 3, 8, 8))
        _ = model(torch.randn(2, 3, 8, 8) * 2)
    finally:
        for h in hooks:
            h.remove()
    cache = export_fakequant_conv_calibration_cache(stats)
    assert cache["schema_version"] == "flashvsr.fakequant.extra_op_calibration.v2"
    assert cache["summary"]["num_layers"] == 2
    assert "tcdecoder.0" in cache["layers"]

    convert_ops_to_fakequant(
        model,
        mode="a8w8",
        op_types=("conv2d",),
        prefix="tcdecoder",
        activation_qdq_mode="static_tensor_symmetric",
        calibration_cache=cache,
    )
    assert isinstance(model[0], FakeQuantConv2d)
    assert isinstance(model[2], FakeQuantConv2d)
    assert int(model[0].activation_qdq_mode.item()) == ACTIVATION_QDQ_MODE_TO_ID["static_tensor_symmetric"]
    assert tuple(model[0].act_scale.shape) == (1, 3, 1, 1)
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 7, 8, 8)


def test_hook_cache_exports_and_convert_ops_consumes_prefixed_linear_keys():
    model = nn.Sequential(nn.Linear(4, 6))
    hooks, stats = attach_fakequant_conv_calibration_hooks(model, prefix="lq_proj_in", op_types=("linear",))
    try:
        _ = model(torch.randn(2, 5, 4))
        _ = model(torch.randn(2, 5, 4) * 3)
    finally:
        for h in hooks:
            h.remove()

    cache = export_fakequant_conv_calibration_cache(stats)
    assert cache["summary"]["num_layers"] == 1
    assert "lq_proj_in.0" in cache["layers"]
    assert cache["layers"]["lq_proj_in.0"]["activation_qdq_mode"] == "static_token_asymmetric"
    assert len(cache["layers"]["lq_proj_in.0"]["act_scale"]) == 5

    convert_ops_to_fakequant(
        model,
        mode="a8w8",
        op_types=("linear",),
        prefix="lq_proj_in",
        activation_qdq_mode="static_tensor_symmetric",
        calibration_cache=cache,
    )
    assert isinstance(model[0], FakeQuantLinear)
    assert int(model[0].activation_qdq_mode.item()) == ACTIVATION_QDQ_MODE_TO_ID["static_token_asymmetric"]
    assert tuple(model[0].act_scale.shape) == (5,)
    assert model(torch.randn(2, 5, 4)).shape == (2, 5, 6)
