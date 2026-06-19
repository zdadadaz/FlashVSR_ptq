from src.models.quantization.policy import build_qbasicvsr_temporal_policy, qbasicvsr_bit_to_mode


def test_qbasicvsr_bit_to_mode_mapping():
    assert qbasicvsr_bit_to_mode(4) == "a4w4"
    assert qbasicvsr_bit_to_mode(5) == "a8w8"
    assert qbasicvsr_bit_to_mode(8) == "a8w8"
    assert qbasicvsr_bit_to_mode(9) == "a16w8"
    assert qbasicvsr_bit_to_mode(4, a4w4_enabled=False) == "a8w8"


def test_qbasicvsr_policy_metadata_has_unquantized_vae_scope():
    policy = build_qbasicvsr_temporal_policy(
        ["blocks.0.self_attn.q", "time_embedding.0", "head.head"],
        b_base=4,
        video_bit_factor=0,
        layer_bit_factors={"blocks.0.self_attn.q": 1, "time_embedding.0": -1, "head.head": 0},
        spatial_sensitivity={"blocks.0.self_attn.q": 0.9},
        temporal_sensitivity={"blocks.0.self_attn.q": 0.8},
    )
    assert policy["schema_version"] == "flashvsr.qbasicvsr.temporal_policy.v1"
    assert policy["quant_scope"] == "dit_linear_only"
    assert policy["wan_vae_quantized"] is False
    assert policy["layers"]["time_embedding.0"]["mode"] == "a16w8"
    assert policy["layers"]["head.head"]["mode"] == "a16w8"
    assert policy["layers"]["blocks.0.self_attn.q"]["mode"] == "a8w8"
    assert policy["fab"] > 0
