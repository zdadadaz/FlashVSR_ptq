import json
import subprocess
import sys


def test_omse_cache_builder_prefers_clipping_for_outlier_samples(tmp_path):
    cache = tmp_path / "cache.json"
    out = tmp_path / "omse.json"
    normal_samples = [-1.0, -0.5, 0.0, 0.5, 1.0] * 10000
    cache.write_text(json.dumps({
        "layer": {
            "act_min": [-1.0],
            "act_max": [10.0],
            "act_scale": [11.0 / 255.0],
            "zero_point": [-105],
            "act_samples": [normal_samples + [10.0]],
        },
        "_metadata": {"source": "test"},
    }))

    subprocess.run([
        sys.executable,
        "scripts/ptq/qbasicvsr_static_omse_clip.py",
        "--input", str(cache),
        "--output", str(out),
        "--factors", "1.0,0.9,0.8",
    ], check=True)

    data = json.loads(out.read_text())
    assert data["_metadata"]["clipping_method"] == "omse"
    assert data["_metadata"]["layers"] == 1
    assert data["layer"]["clipping_method"] == "omse"
    assert data["layer"]["omse_best_factor"][0] < 1.0
    assert "act_scale" in data["layer"]
    assert "zero_point" in data["layer"]
    assert data["layer"]["act_max"][0] < 10.0


def test_omse_cache_builder_preserves_layer_names_and_metadata(tmp_path):
    cache = tmp_path / "cache.json"
    out = tmp_path / "omse.json"
    cache.write_text(json.dumps({
        "blocks.0.ffn.0": {"act_min": [-2.0, -1.0], "act_max": [2.0, 1.0], "act_scale": [4/255, 2/255], "zero_point": [0, 0]},
        "blocks.0.ffn.2": {"act_min": [-3.0], "act_max": [3.0], "act_scale": [6/255], "zero_point": [0]},
        "_metadata": {"schema": "legacy"},
    }))

    subprocess.run([
        sys.executable,
        "scripts/ptq/qbasicvsr_static_omse_clip.py",
        "--input", str(cache),
        "--output", str(out),
    ], check=True)

    data = json.loads(out.read_text())
    assert {k for k in data if not k.startswith("_")} == {"blocks.0.ffn.0", "blocks.0.ffn.2"}
    assert data["_metadata"]["source_metadata"] == {"schema": "legacy"}
    for entry in [data["blocks.0.ffn.0"], data["blocks.0.ffn.2"]]:
        assert len(entry["act_scale"]) == len(entry["zero_point"])
        assert entry["clipping_method"] == "omse"
