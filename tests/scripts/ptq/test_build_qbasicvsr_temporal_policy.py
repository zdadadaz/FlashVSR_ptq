import json
import subprocess
import sys

from scripts.ptq.build_qbasicvsr_temporal_policy import decide_layer_bit_factors


def test_decide_layer_bit_factors_thresholds():
    sens = {
        "low": {"spatial_sensitivity": 0.0, "temporal_sensitivity": 0.0},
        "mid": {"spatial_sensitivity": 0.5, "temporal_sensitivity": 0.5},
        "high": {"spatial_sensitivity": 1.0, "temporal_sensitivity": 1.0},
    }
    factors, _, _, thresholds = decide_layer_bit_factors(sens, p_space=34, p_temp=34)
    assert factors["low"] == -1
    assert factors["mid"] == 0
    assert factors["high"] == 1
    assert thresholds["space_high"] >= thresholds["space_low"]


def test_build_qbasicvsr_temporal_policy_cli(tmp_path):
    sens = tmp_path / "sens.json"
    comp = tmp_path / "comp.json"
    out = tmp_path / "policy.json"
    sens.write_text(json.dumps({
        "_metadata": {"schema_version": "flashvsr.qbasicvsr.temporal_sensitivity.v1"},
        "blocks.0.self_attn.q": {"spatial_sensitivity": 1.0, "temporal_sensitivity": 1.0},
        "blocks.0.self_attn.k": {"spatial_sensitivity": 0.0, "temporal_sensitivity": 0.0},
        "time_embedding.0": {"spatial_sensitivity": 0.0, "temporal_sensitivity": 0.0},
    }))
    comp.write_text(json.dumps({"schema_version": "flashvsr.qbasicvsr.video_complexity.v1", "flow_backend": "proxy", "videos": [{"path": "v.mp4", "video_bit_factor": 0}], "thresholds": {}}))
    subprocess.run([
        sys.executable, "scripts/ptq/build_qbasicvsr_temporal_policy.py",
        "--temporal_sensitivity", str(sens), "--video_complexity", str(comp),
        "--video_path", "v.mp4", "--output", str(out), "--base_bits", "4",
    ], check=True)
    policy = json.loads(out.read_text())
    assert policy["schema_version"] == "flashvsr.qbasicvsr.temporal_policy.v1"
    assert sum(policy["counts"].values()) == 3
    assert policy["layers"]["time_embedding.0"]["mode"] == "a16w8"
