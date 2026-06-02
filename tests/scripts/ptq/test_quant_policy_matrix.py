import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ptq.quant_policy_matrix import POLICIES, policy_by_name, reduce_calibration_cache
from src.models.quantization.fakequant import FakeQuantLinear


def test_policy_matrix_covers_july_person_a_granularities():
    names = {policy.name for policy in POLICIES}
    assert "per_tensor_static_asym" in names
    assert "per_channel_static_asym" in names
    assert "per_token_dynamic_asym" in names
    assert policy_by_name("per_token_dynamic_asym").requires_calibration is False


def test_reduce_calibration_cache_to_per_tensor_from_minmax(tmp_path):
    src = tmp_path / "per_channel.json"
    dst = tmp_path / "per_tensor.json"
    src.write_text(
        json.dumps(
            {
                "_metadata": {"mode": "a8w8"},
                "blocks.0.self_attn.q": {
                    "act_min": [-1.0, -2.0, 0.0],
                    "act_max": [1.0, 2.0, 3.0],
                    "act_scale": [0.1, 0.2, 0.3],
                    "zero_point": [0, 1, 2],
                },
            }
        )
    )

    reduced = reduce_calibration_cache(src, dst, "per_tensor")

    entry = reduced["blocks.0.self_attn.q"]
    assert len(entry["act_scale"]) == 1
    assert len(entry["zero_point"]) == 1
    assert entry["act_min"] == [-2.0]
    assert entry["act_max"] == [3.0]
    assert reduced["_metadata"]["calibration_granularity"] == "per_tensor"


def test_fakequant_linear_accepts_scalar_static_activation_cache():
    linear = nn.Linear(4, 3)
    fq = FakeQuantLinear.from_float(
        linear,
        activation_mode="a8",
        weight_mode="w8",
        act_scale=torch.tensor([0.25]),
        act_zero_point=torch.tensor([-3]),
    )

    assert fq.act_scale.shape == (1, 1, 4)
    assert torch.allclose(fq.act_scale, torch.full((1, 1, 4), 0.25))
    assert torch.equal(fq.act_zero_point, torch.full((1, 1, 4), -3, dtype=torch.int32))
