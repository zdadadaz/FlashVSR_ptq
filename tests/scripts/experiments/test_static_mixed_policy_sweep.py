import json

from scripts.experiments.static_mixed_qat.run_policy_sweep import build_convert_command, summarize_policy_counts


def test_build_convert_command_uses_static_tensor_policy_and_bias_flag(tmp_path):
    cmd = build_convert_command(
        python="python",
        checkpoint="fp.safetensors",
        calibration_cache="calib.json",
        policy="policy.json",
        output="out.safetensors",
        activation_qdq_mode="static_tensor_symmetric",
        clipping="minmax",
        enable_bias_correction=True,
    )

    assert cmd[:3] == ["python", "scripts/ptq/fakequant_convert.py", "--checkpoint"]
    assert "--policy" in cmd
    assert "--activation_qdq_mode" in cmd
    assert "static_tensor_symmetric" in cmd
    assert "--enable_bias_correction" in cmd


def test_summarize_policy_counts_reads_static_mixed_schema(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "layers": {
            "a": {"mode": "a8w8"},
            "b": {"mode": "a16w8"},
            "c": {"mode": "a16w8"},
        }
    }))

    assert summarize_policy_counts(policy) == {"a8w8": 1, "a16w8": 2}
