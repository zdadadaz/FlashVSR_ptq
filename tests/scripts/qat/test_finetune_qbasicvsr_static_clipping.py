import json
import subprocess
import sys


def test_clipft_dry_run_updates_only_cache_ranges(tmp_path):
    cache = tmp_path / "omse.json"
    out = tmp_path / "clipft.json"
    metrics = tmp_path / "metrics.jsonl"
    cache.write_text(json.dumps({
        "layer": {
            "act_min": [-1.0, -2.0],
            "act_max": [1.0, 2.0],
            "act_scale": [2/255, 4/255],
            "zero_point": [0, 0],
            "clipping_method": "omse",
        },
        "_metadata": {"clipping_method": "omse"},
    }))

    subprocess.run([
        sys.executable,
        "scripts/qat/finetune_qbasicvsr_static_clipping.py",
        "--input_cache", str(cache),
        "--output_cache", str(out),
        "--metrics_jsonl", str(metrics),
        "--steps", "3",
        "--lr", "0.05",
        "--dry_run",
    ], check=True)

    data = json.loads(out.read_text())
    assert data["_metadata"]["clipping_method"] == "omse_teacher_clipft"
    assert data["_metadata"]["teacher_ft_steps"] == 3
    assert data["layer"]["clipping_method"] == "omse_teacher_clipft"
    assert data["layer"]["teacher_ft_steps"] == 3
    assert data["layer"]["act_min"][0] < data["layer"]["act_max"][0]
    lines = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert [line["step"] for line in lines] == [1, 2, 3]
    assert all(line["updated_tensors"] == ["act_min", "act_max", "act_scale", "zero_point"] for line in lines)
