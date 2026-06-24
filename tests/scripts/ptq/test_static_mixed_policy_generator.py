import json
from pathlib import Path

from scripts.ptq.build_static_mixed_policy import (
    build_static_mixed_policy_from_rows,
    load_sensitivity_rows,
    parse_percent_list,
)
from src.models.quantization.policy import layer_policy_entries


def test_parse_percent_list_accepts_comma_separated_values():
    assert parse_percent_list("10,20,40") == [10.0, 20.0, 40.0]


def test_static_mixed_policy_uses_top_mse_layers_as_a16w8():
    rows = [
        {"name": "blocks.0.self_attn.q", "output_mse": 0.1, "sqnr_db": 10.0},
        {"name": "blocks.0.self_attn.k", "output_mse": 0.4, "sqnr_db": 3.0},
        {"name": "blocks.0.ffn.0", "output_mse": 0.3, "sqnr_db": 5.0},
        {"name": "head.head", "output_mse": 0.2, "sqnr_db": 8.0},
    ]

    policy = build_static_mixed_policy_from_rows(rows, a16_percent=50.0)

    assert policy["schema_version"] == "flashvsr.static_mixed_policy.v1"
    assert policy["quant_scope"] == "dit_linear_only"
    assert policy["wan_vae_quantized"] is False
    assert policy["summary"]["total_linear_layers"] == 4
    assert policy["summary"]["a16w8_layers"] == 2
    assert policy["summary"]["a8w8_layers"] == 2
    assert policy["layers"]["blocks.0.self_attn.k"]["mode"] == "a16w8"
    assert policy["layers"]["blocks.0.ffn.0"]["mode"] == "a16w8"
    assert policy["layers"]["blocks.0.self_attn.q"]["activation_qdq_mode"] == "static_token_asymmetric"
    assert policy["default"]["activation_qdq_mode"] == "static_token_asymmetric"

    entries = layer_policy_entries(policy)
    assert entries["blocks.0.self_attn.k"]["mode"] == "a16w8"


def test_load_sensitivity_rows_accepts_static_diagnostic_schema(tmp_path):
    path = tmp_path / "diag.json"
    path.write_text(json.dumps({
        "schema": "flashvsr.static_qat_linear_diagnostics.v1",
        "layers": [
            {"name": "a", "output_mse": 1.0},
            {"name": "b", "output_mse": 0.5},
        ],
    }))

    rows = load_sensitivity_rows(path)

    assert [row["name"] for row in rows] == ["a", "b"]
