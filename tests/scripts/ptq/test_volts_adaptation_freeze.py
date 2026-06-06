import json
import subprocess
import sys

import torch
import torch.nn as nn

from src.models.quantization.qat import (
    QuantAwareLinear,
    apply_volts_adaptation_trainability,
    build_volts_adaptation_plan,
    lsgquant_qat_lite_step,
)


class TinyAdaptiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = QuantAwareLinear.from_float(nn.Linear(4, 4), activation_mode="a8", weight_mode="w8")
        self.light = QuantAwareLinear.from_float(nn.Linear(4, 4), activation_mode="a8", weight_mode="w8")
        self.full = QuantAwareLinear.from_float(nn.Linear(4, 4), activation_mode="a8", weight_mode="w8")

    def forward(self, x):
        return self.full(self.light(self.frozen(x)))


def _policy():
    return {
        "schema_version": "flashvsr.lsgquant.policy.v1",
        "layers": {
            "frozen": {"tier": "frozen", "mode": "a8w8"},
            "light": {"tier": "light", "mode": "a8w8"},
            "full": {"tier": "full", "mode": "a8w8"},
        },
    }


def test_build_volts_adaptation_plan_assigns_steps_and_counts_by_tier():
    plan = build_volts_adaptation_plan(_policy(), light_steps=30, full_steps=300)

    assert plan["schema_version"] == "flashvsr.lsgquant.volts_adaptation_plan.v1"
    assert plan["layers"]["frozen"]["steps"] == 0
    assert plan["layers"]["light"]["steps"] == 30
    assert plan["layers"]["full"]["steps"] == 300
    assert plan["counts"] == {"frozen": 1, "light": 1, "full": 1}


def test_apply_volts_adaptation_trainability_freezes_frozen_and_trains_light_full():
    model = TinyAdaptiveModel()
    summary = apply_volts_adaptation_trainability(model, _policy())

    assert summary["frozen_layers"] == 1
    assert summary["trainable_layers"] == 2
    assert all(not p.requires_grad for p in model.frozen.parameters())
    assert all(p.requires_grad for p in model.light.parameters())
    assert all(p.requires_grad for p in model.full.parameters())


def test_lsgquant_qat_lite_step_updates_only_light_and_full_layers():
    torch.manual_seed(11)
    model = TinyAdaptiveModel()
    teacher = TinyAdaptiveModel().eval()
    teacher.load_state_dict(model.state_dict())
    with torch.no_grad():
        teacher.full.weight.add_(0.25)

    apply_volts_adaptation_trainability(model, _policy())
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    optim = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.05)
    batch = {"input": torch.randn(3, 4)}

    metrics = lsgquant_qat_lite_step(model, teacher, batch, optim)

    assert metrics["loss"] > 0
    assert torch.allclose(model.state_dict()["frozen.weight"], before["frozen.weight"])
    assert not torch.allclose(model.state_dict()["light.weight"], before["light.weight"])
    assert not torch.allclose(model.state_dict()["full.weight"], before["full.weight"])


def test_run_lsgquant_volts_adapt_cli_dry_run_writes_manifest(tmp_path):
    policy = tmp_path / "policy.json"
    manifest = tmp_path / "videos.jsonl"
    out = tmp_path / "adapted.safetensors"
    adapt_manifest = tmp_path / "adapt_manifest.json"
    policy.write_text(json.dumps(_policy()))
    manifest.write_text(json.dumps({"sample": "placeholder.pt"}) + "\n")

    subprocess.run(
        [
            sys.executable,
            "scripts/qat/run_lsgquant_volts_adapt.py",
            "--teacher_checkpoint",
            "teacher.safetensors",
            "--student_checkpoint",
            "student.safetensors",
            "--policy",
            str(policy),
            "--calibration_manifest",
            str(manifest),
            "--light_steps",
            "3",
            "--full_steps",
            "7",
            "--out",
            str(out),
            "--manifest",
            str(adapt_manifest),
            "--dry_run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    written = json.loads(adapt_manifest.read_text())
    assert written["schema_version"] == "flashvsr.lsgquant.volts_adaptation_manifest.v1"
    assert written["adaptation_plan"]["layers"]["light"]["steps"] == 3
    assert written["adaptation_plan"]["layers"]["full"]["steps"] == 7
    assert written["status"] == "dry_run"
