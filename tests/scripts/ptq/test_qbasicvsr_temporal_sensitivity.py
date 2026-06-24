import torch

from scripts.ptq.qbasicvsr_temporal_sensitivity import build_smoke_sensitivity, compute_spatial_temporal_sensitivity


def test_compute_spatial_temporal_sensitivity_for_token_tensor():
    x = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(2, 8, 4)
    out = compute_spatial_temporal_sensitivity(x, frames=4)
    assert out["shape_quality"] == "approx_token"
    assert out["spatial_sensitivity"] > 0
    assert out["temporal_sensitivity"] > 0


def test_compute_spatial_temporal_sensitivity_for_embedding_tensor():
    x = torch.randn(2, 4)
    out = compute_spatial_temporal_sensitivity(x, frames=1)
    assert out["shape_quality"] == "shape_special"
    assert out["temporal_sensitivity"] == 0.0


def test_build_smoke_sensitivity_has_metadata_and_layer_entries():
    report = build_smoke_sensitivity(["blocks.0.self_attn.q", "time_embedding.0"], frames=8)
    assert report["_metadata"]["schema_version"] == "flashvsr.qbasicvsr.temporal_sensitivity.v1"
    assert report["_metadata"]["wan_vae_quantized"] is False
    assert report["blocks.0.self_attn.q"]["group"] == "self_attn"
