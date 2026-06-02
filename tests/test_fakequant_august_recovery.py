import json

import torch
import torch.nn as nn

from src.models.quantization.fakequant import FakeQuantLinear, convert_model_to_fakequant
from src.models.quantization.policy import build_august_mixed_policy, classify_layer_name, load_layer_policy


class TinyWanLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_embedding = nn.Sequential(nn.Linear(4, 4))
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.ModuleDict({"q": nn.Linear(4, 4)}),
                "cross_attn": nn.ModuleDict({"k": nn.Linear(4, 4)}),
                "ffn": nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4)),
            })
        ])
        self.head = nn.ModuleDict({"head": nn.Linear(4, 4)})


def _stats_for(model):
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            stats[name] = {
                "act_scale": torch.ones(module.in_features),
                "zero_point": torch.zeros(module.in_features, dtype=torch.int32),
                "act_mean": torch.full((module.in_features,), 0.25),
            }
    return stats


def test_august_policy_assigns_sensitive_a16_and_attention_a8():
    names = [
        "text_embedding.0",
        "blocks.0.self_attn.q",
        "blocks.0.cross_attn.k",
        "blocks.0.ffn.0",
        "head.head",
    ]

    policy = build_august_mixed_policy(names)

    assert policy["layers"]["text_embedding.0"]["mode"] == "a16w8"
    assert policy["layers"]["blocks.0.ffn.0"]["mode"] == "a16w8"
    assert policy["layers"]["blocks.0.self_attn.q"]["mode"] == "a8w8"
    assert policy["layers"]["blocks.0.self_attn.q"]["activation_qdq_mode"] == "dynamic_asymmetric"
    assert policy["counts"] == {"a16w8": 2, "a8w8": 3}
    assert classify_layer_name("blocks.0.cross_attn.k") == "cross_attn"


def test_policy_json_validation_roundtrip(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"layers": {"x": {"mode": "a8w8", "activation_qdq_mode": "dynamic_asymmetric"}}}))

    loaded = load_layer_policy(path)

    assert loaded["layers"]["x"]["mode"] == "a8w8"


def test_convert_model_to_fakequant_uses_per_layer_policy_modes():
    model = TinyWanLike()
    stats = _stats_for(model)
    policy = build_august_mixed_policy(sorted(stats))

    convert_model_to_fakequant(
        model,
        mode="a8w8",
        act_stats=stats,
        layer_policy=policy["layers"],
        enable_bias_correction=True,
    )

    assert isinstance(model.text_embedding[0], FakeQuantLinear)
    assert model.text_embedding[0].activation_mode == "a16"
    assert model.blocks[0]["ffn"][0].activation_mode == "a16"
    assert model.blocks[0]["self_attn"]["q"].activation_mode == "a8"
    assert int(model.blocks[0]["self_attn"]["q"].activation_qdq_mode.item()) == 2
    assert model._fakequant_conversion_summary["mode_counts"] == {"a16w8": 3, "a8w8": 3}


def test_bias_correction_changes_bias_when_act_mean_provided():
    layer = nn.Linear(4, 3)
    original_bias = layer.bias.detach().clone()
    fq = FakeQuantLinear.from_float(
        layer,
        activation_mode="a16",
        weight_mode="w8",
        act_mean=torch.tensor([0.5, -0.25, 0.125, 1.0]),
    )

    assert not torch.allclose(fq.bias, original_bias)
