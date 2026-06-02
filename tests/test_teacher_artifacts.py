import json
from pathlib import Path

import torch
import torch.nn as nn

from scripts.teacher.feature_dump import FeatureDumpWriter, register_feature_hooks, remove_hooks


def test_feature_dump_writer_schema_and_hook_capture(tmp_path):
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    writer = FeatureDumpWriter(tmp_path, run_id="test", max_calls_per_module=1)
    handles = register_feature_hooks(model, writer, include=["0", "2"], capture="output")

    with torch.no_grad():
        model(torch.randn(3, 4))
    remove_hooks(handles)
    manifest_path = writer.write_manifest()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == "flashvsr.feature_dump.v1"
    assert len(manifest["features"]) == 2
    for record in manifest["features"]:
        payload = torch.load(Path(tmp_path) / record["path"], map_location="cpu", weights_only=False)
        assert payload["schema_version"] == "flashvsr.feature_tensor.v1"
        assert payload["capture"] == "output"
        assert torch.is_tensor(payload["tensor"])


def test_ptq_qat_insertion_points_config_schema():
    cfg = json.loads(Path("configs/ptq_qat_insertion_points.json").read_text())
    assert cfg["schema_version"] == "flashvsr.ptq_qat_insertion_points.v1"
    assert cfg["scope"]["quantize"] == ["WanModel / WanVideoDiT DiT path"]
    linear = next(x for x in cfg["insertion_points"] if x["id"] == "dit.linear.all")
    assert linear["coverage_expected"] == 306
    assert linear["groups"]["self_attn_qkvo"] == 120
    assert any(x["id"] == "wan_vae" and x["status"] == "excluded" for x in cfg["insertion_points"])
