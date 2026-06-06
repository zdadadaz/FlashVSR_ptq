import json
import subprocess
import sys

from scripts.ptq.generate_layer_policy import (
    build_policy_from_layer_names,
    classify_layer_name,
    load_sensitivity_scores,
)
from scripts.ptq.run_lsgquant_eval_set import build_lsgquant_eval_manifest


def test_classify_pr4_sensitive_heuristics():
    assert classify_layer_name("blocks.0.ffn.0") == (True, "ffn layers are sensitive in FlashVSR DiT PTQ")
    assert classify_layer_name("time_embedding.0") == (True, "embedding/projection layers are sensitive")
    assert classify_layer_name("blocks.0.text_embedding.linear") == (True, "embedding/projection layers are sensitive")
    assert classify_layer_name("time_projection.0") == (True, "embedding/projection layers are sensitive")
    assert classify_layer_name("head") == (True, "output head preserved for detail fidelity")
    assert classify_layer_name("LQ_proj_in") == (True, "LQ projection is special conditioning path")
    assert classify_layer_name("blocks.0.self_attn.q") == (False, "default robust layer")


def test_build_policy_marks_heuristic_sensitive_layers_only():
    policy = build_policy_from_layer_names(
        [
            "blocks.0.self_attn.q",
            "blocks.0.ffn.0",
            "time_embedding.0",
            "head",
        ],
        default_mode="a8w8",
        sensitive_mode="a16w8",
        activation_qdq_mode="draq_symmetric",
    )

    assert policy["schema_version"] == "flashvsr.pr4.layer_policy.v1"
    assert policy["default"] == {"mode": "a8w8", "activation_qdq_mode": "draq_symmetric"}
    assert policy["counts"] == {"a8w8": 1, "a16w8": 3}
    assert policy["layers"]["blocks.0.ffn.0"]["mode"] == "a16w8"
    assert policy["layers"]["time_embedding.0"]["mode"] == "a16w8"
    assert policy["layers"]["head"]["mode"] == "a16w8"
    assert "blocks.0.self_attn.q" not in policy["layers"]


def test_build_policy_applies_topk_sensitivity_json_over_heuristics(tmp_path):
    sensitivity = tmp_path / "sensitivity.json"
    sensitivity.write_text(json.dumps({
        "layers": {
            "blocks.0.self_attn.q": {"sensitivity": 0.9},
            "blocks.0.self_attn.k": {"score": 0.7},
            "blocks.0.self_attn.v": 0.1,
        }
    }))
    scores = load_sensitivity_scores(sensitivity)

    policy = build_policy_from_layer_names(
        ["blocks.0.self_attn.q", "blocks.0.self_attn.k", "blocks.0.self_attn.v"],
        default_mode="a8w8",
        sensitive_mode="a16w8",
        topk_sensitive=2,
        sensitivity_scores=scores,
    )

    assert policy["layers"]["blocks.0.self_attn.q"]["mode"] == "a16w8"
    assert policy["layers"]["blocks.0.self_attn.q"]["sensitivity_score"] == 0.9
    assert policy["layers"]["blocks.0.self_attn.k"]["mode"] == "a16w8"
    assert "blocks.0.self_attn.v" not in policy["layers"]


def test_generate_layer_policy_cli_from_layer_names_json(tmp_path):
    names = tmp_path / "linear_layers.json"
    out = tmp_path / "policy.json"
    names.write_text(json.dumps([
        "blocks.0.self_attn.q",
        "blocks.0.ffn.0",
        "time_embedding.0",
    ]))

    subprocess.run(
        [
            sys.executable,
            "scripts/ptq/generate_layer_policy.py",
            "--layer_names_json",
            str(names),
            "--output_policy",
            str(out),
            "--default_mode",
            "a8w8",
            "--sensitive_mode",
            "a16w8",
            "--activation_qdq_mode",
            "draq_symmetric",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    policy = json.loads(out.read_text())
    assert policy["metadata"]["layer_name_source"] == str(names)
    assert policy["layers"]["blocks.0.ffn.0"]["mode"] == "a16w8"
    assert "blocks.0.self_attn.q" not in policy["layers"]


def test_lsgquant_manifest_marks_bias_correction_as_experimental_opt_in(tmp_path):
    checkpoint = tmp_path / "dit.safetensors"
    calib = tmp_path / "calib.json"
    policy = tmp_path / "policy.json"
    out_dir = tmp_path / "eval"
    checkpoint.write_text("placeholder")
    calib.write_text(json.dumps({"_metadata": {"schema_version": "flashvsr.lsgquant.calibration.v1"}}))
    policy.write_text(json.dumps({
        "schema_version": "flashvsr.lsgquant.policy.v1",
        "default": {"mode": "a8w8", "activation_qdq_mode": "draq_symmetric"},
        "layers": {},
    }))

    manifest = build_lsgquant_eval_manifest(
        checkpoint=checkpoint,
        calibration_cache=calib,
        policy=policy,
        mode="a8w8",
        out_dir=out_dir,
        enable_bias_correction=False,
    )

    assert manifest["bias_correction"] is False
    assert manifest["experimental_options"]["bias_correction"] == {
        "enabled": False,
        "opt_in_flag": "--enable_bias_correction",
        "status": "experimental_opt_in",
        "default": False,
        "rationale": "Disabled by default after PR3 smoke ablation showed PSNR regression.",
    }
