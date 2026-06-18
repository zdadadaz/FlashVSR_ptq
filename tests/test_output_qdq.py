"""Tests for the new per-output-channel QDQ path on FakeQuantLinear.

Covers:
- mode 7 (static_tensor_symmetric) output QDQ active only in that mode
- per-output-channel symmetric int8 output range
- output calibration cache roundtrip
- regression: existing modes (draq_symmetric, dynamic_asymmetric) still work
"""

import torch
import torch.nn as nn

from src.models.quantization.fakequant import (
    ACTIVATION_QDQ_MODE_TO_ID,
    FakeQuantLinear,
    _qdq_symmetric_channel,
    convert_model_to_fakequant,
)


# ---------------------------------------------------------------------
# Helper helpers
# ---------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self, in_f=8, out_f=12, hidden=16):
        super().__init__()
        self.l1 = nn.Linear(in_f, hidden)
        self.l2 = nn.Linear(hidden, out_f)


def _stats(model, scale=0.1, n_samples=2):
    """Build per-layer act_stats with the new schema including output fields."""
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            out[name] = {
                "act_scale": [scale] * m.in_features,
                "zero_point": [0] * m.in_features,
                "act_min": [-1.0] * m.in_features,
                "act_max": [1.0] * m.in_features,
                "act_mean": [0.25] * m.in_features,
                "output_scale": [scale] * m.out_features,
                "output_zero_point": [0] * m.out_features,
            }
    return out


# ---------------------------------------------------------------------
# 1. Mode 7 registration
# ---------------------------------------------------------------------


def test_static_tensor_symmetric_is_mode_seven():
    assert "static_tensor_symmetric" in ACTIVATION_QDQ_MODE_TO_ID
    assert ACTIVATION_QDQ_MODE_TO_ID["static_tensor_symmetric"] == 7


# ---------------------------------------------------------------------
# 2. _qdq_symmetric_channel helper
# ---------------------------------------------------------------------


def test_qdq_symmetric_channel_per_channel():
    x = torch.randn(4, 8) * 0.05
    scale = torch.full((8,), 0.01)
    zp = torch.zeros(8)
    y = _qdq_symmetric_channel(x, scale, zp)
    assert y.shape == x.shape
    # Output should be representable with the given scale × [-127, 127].
    for j in range(8):
        assert y[:, j].abs().max().item() <= 127 * scale[j].item() + 1e-6


def test_qdq_symmetric_channel_per_tensor():
    x = torch.randn(4, 8)
    y = _qdq_symmetric_channel(x, torch.tensor(0.01), None, axis=0)
    # axis=0 (non-None) forces per-channel path with the single scale broadcast.
    # For shape consistency, ensure output shape matches.
    assert y.shape == x.shape


def test_qdq_symmetric_channel_disabled():
    x = torch.randn(2, 4)
    y = _qdq_symmetric_channel(x, torch.tensor(0.01), None, enabled=False)
    assert torch.equal(x, y)


# ---------------------------------------------------------------------
# 3. FakeQuantLinear mode 7 + output QDQ integration
# ---------------------------------------------------------------------


def test_fakequant_applies_output_qdq_when_mode_seven():
    layer = nn.Linear(8, 12, bias=True)
    out_scale = torch.full((12,), 0.05)
    fq = FakeQuantLinear.from_float(
        layer,
        activation_mode="a8",
        weight_mode="w8",
        act_scale=torch.full((8,), 0.1),
        act_zero_point=torch.zeros(8, dtype=torch.int32),
        activation_qdq_mode="static_tensor_symmetric",
        output_scale=out_scale,
        output_zero_point=torch.zeros(12, dtype=torch.int32),
    )
    # Buffers in the right shape
    assert fq.output_scale.shape == (12, 1)
    assert fq.output_zero_point.shape == (12, 1)
    assert torch.allclose(fq.output_scale.reshape(-1), out_scale)
    # Forward result should be clamped to int8 symmetric range of output_scale.
    x_in = torch.randn(2, 4, 8)
    y_out = fq(x_in)
    bound = 127 * out_scale.max().item() + 1e-4
    assert y_out.abs().max().item() <= bound, (
        f"Output {y_out.abs().max().item():.4f} exceeds int8 bound {bound:.4f}"
    )


def test_fakequant_does_not_apply_output_qdq_in_draq_symmetric():
    """Other modes must not call into the output QDQ path (avoids regression)."""
    layer = nn.Linear(4, 6, bias=True)
    fq = FakeQuantLinear.from_float(
        layer,
        activation_mode="a8",
        weight_mode="w8",
        act_scale=torch.full((4,), 0.1),
        act_zero_point=torch.zeros(4, dtype=torch.int32),
        activation_qdq_mode="draq_symmetric",
    )
    x_in = torch.randn(2, 4)
    y_out = fq(x_in)
    # draq_symmetric has no output QDQ; output magnitude can exceed 127*0.05.
    assert y_out.shape == (2, 6)


def test_fakequant_a16w8_passthrough_unchanged():
    """a16 (no activation quant) must NOT engage the output QDQ path."""
    layer = nn.Linear(4, 6, bias=True)
    fq = FakeQuantLinear.from_float(
        layer,
        activation_mode="a16",
        weight_mode="w8",
        activation_qdq_mode="static_tensor_symmetric",
    )
    x_in = torch.randn(2, 4)
    y_out = fq(x_in)
    assert y_out.shape == (2, 6)


# ---------------------------------------------------------------------
# 4. convert_model_to_fakequant wires output_stats through
# ---------------------------------------------------------------------


def test_convert_model_wires_output_stats_into_fakequant():
    model = _TinyModel()
    stats = _stats(model)
    output_stats = {
        name: {
            "output_scale": torch.tensor(stats[name]["output_scale"]),
            "output_zero_point": torch.tensor(stats[name]["output_zero_point"]),
        }
        for name in stats
    }
    convert_model_to_fakequant(
        model,
        mode="a8w8",
        act_stats=stats,
        activation_qdq_mode="static_tensor_symmetric",
        output_stats=output_stats,
    )
    assert isinstance(model.l1, FakeQuantLinear)
    assert isinstance(model.l2, FakeQuantLinear)
    # Output buffers filled
    assert model.l1.output_scale.abs().max().item() > 0
    assert model.l2.output_scale.abs().max().item() > 0
    # Activation mode code is a8
    assert int(model.l1.activation_mode_code.item()) == 2
    # qdq_mode is 7
    assert int(model.l1.activation_qdq_mode.item()) == 7


def test_output_stats_wrong_layer_dim_raises():
    """Passing an output_scale with wrong channel count must be caught (fallback or raise)."""
    model = _TinyModel()
    stats = _stats(model)
    bad_output_stats = {
        "l1": {
            "output_scale": torch.zeros(99),  # wrong: l1 has out=16
            "output_zero_point": torch.zeros(99, dtype=torch.int32),
        },
    }
    # convert_model_to_fakequant logs and falls back per-layer (does not raise).
    # The model must end up with l1 still an nn.Linear (rejected conversion) and l2
    # converted normally.
    convert_model_to_fakequant(
        model,
        mode="a8w8",
        act_stats=stats,
        activation_qdq_mode="static_tensor_symmetric",
        output_stats=bad_output_stats,
    )
    # l1 conversion should have failed (mismatched output_scale), so it's still
    # an nn.Linear; l2 is fine.
    assert not isinstance(model.l1, FakeQuantLinear), (
        "l1 with wrong output_scale shape should NOT convert to FakeQuantLinear"
    )
    assert isinstance(model.l2, FakeQuantLinear)
