import torch

from scripts.ptq.fakequant_convert import recompute_symmetric_act_scales


def _entry():
    return {
        "act_scale": torch.ones(4),
        "zero_point": torch.zeros(4, dtype=torch.int32),
        "act_min": torch.tensor([-10.0, -2.0, -3.0, -4.0]),
        "act_max": torch.tensor([10.0, 2.0, 3.0, 4.0]),
        "draq_s_percentile_99": torch.tensor([1.0, 1.5, 2.0, 2.5]),
        "draq_s_percentile_999": torch.tensor([4.0, 2.0, 3.0, 4.0]),
    }


def test_static_clipping_minmax_uses_absmax_per_channel():
    out = recompute_symmetric_act_scales({"layer": _entry()}, clipping="minmax")

    assert torch.allclose(out["layer"]["act_scale"], torch.tensor([10.0, 2.0, 3.0, 4.0]) / 127.0)
    assert torch.equal(out["layer"]["zero_point"], torch.zeros(4, dtype=torch.int32))


def test_static_clipping_percentile_999_uses_cached_draq_s_percentile():
    out = recompute_symmetric_act_scales({"layer": _entry()}, clipping="percentile_999")

    assert torch.allclose(out["layer"]["act_scale"], torch.tensor([4.0, 2.0, 3.0, 4.0]) / 127.0)


def test_static_clipping_mse_selects_lower_error_candidate_per_channel():
    out = recompute_symmetric_act_scales({"layer": _entry()}, clipping="mse")

    # Channel 0 has a large outlier, so the MSE proxy should prefer p99.9.
    # Other channels match minmax/p99.9 and remain unchanged.
    assert torch.allclose(out["layer"]["act_scale"], torch.tensor([4.0, 2.0, 3.0, 4.0]) / 127.0)
