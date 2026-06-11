import json

import torch
import torch.nn as nn

from scripts.ptq.fakequant_convert import load_smoothquant_cache
from scripts.ptq.static_ptq_baseline import (
    build_smoothquant_cache,
    build_static_mixed_policy,
    sensitivity_score,
)
from src.models.quantization.fakequant import FakeQuantLinear, convert_model_to_fakequant


class TinySQModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_embedding = nn.Sequential(nn.Linear(4, 4))
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.ModuleDict({"q": nn.Linear(4, 4), "k": nn.Linear(4, 4)}),
                "ffn": nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4)),
            })
        ])
        self.head = nn.ModuleDict({"head": nn.Linear(4, 4)})


def _stats_for(model):
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            stats[name] = {
                "act_scale": [0.1] * module.in_features,
                "zero_point": [0] * module.in_features,
                "act_min": [-1.0] * module.in_features,
                "act_max": [1.0] * module.in_features,
                "act_mean": [0.25] * module.in_features,
            }
    return stats


def test_fakequant_linear_smoothquant_buffers_and_forward_run():
    layer = nn.Linear(4, 3)
    sq = torch.tensor([0.5, 1.0, 2.0, 4.0])
    fq = FakeQuantLinear.from_float(
        layer,
        activation_mode="a8",
        weight_mode="w8",
        act_scale=torch.ones(4) * 0.1,
        act_zero_point=torch.zeros(4, dtype=torch.int32),
        smoothquant_scale=sq,
        weight_rounding="adaround",
        act_mean=torch.ones(4),
    )

    assert bool(fq.smoothquant_enabled.item())
    assert torch.allclose(fq.smoothquant_scale.reshape(-1), sq)
    assert fq.weight_int.dtype == torch.int8
    y = fq(torch.randn(2, 5, 4))
    assert y.shape == (2, 5, 3)


def test_convert_model_accepts_smoothquant_scales_and_adaround():
    model = TinySQModel()
    stats = _stats_for(model)
    sq = {name: torch.ones(module.in_features) for name, module in model.named_modules() if isinstance(module, nn.Linear)}

    convert_model_to_fakequant(
        model,
        mode="a8w8",
        act_stats=stats,
        smoothquant_scales=sq,
        weight_rounding="adaround",
    )

    assert isinstance(model.text_embedding[0], FakeQuantLinear)
    assert bool(model.text_embedding[0].smoothquant_enabled.item())
    assert model._fakequant_conversion_summary["smoothquant_applied"] == len(sq)
    assert model._fakequant_conversion_summary["weight_rounding"] == "adaround"


def test_mixed_checkpoint_fp16_skip_policy_reloads_without_quantizing_skipped_layers():
    src = TinySQModel()
    stats = _stats_for(src)
    convert_model_to_fakequant(
        src,
        mode="a8w8",
        act_stats=stats,
        layer_policy={"text_embedding.0": {"mode": "fp16_skip"}},
    )
    state = src.state_dict()
    assert "text_embedding.0.weight" in state
    assert "text_embedding.0.weight_int" not in state

    dst = TinySQModel()
    inferred_policy = {}
    for name, module in dst.named_modules():
        if isinstance(module, nn.Linear) and f"{name}.weight" in state and f"{name}.weight_int" not in state:
            inferred_policy[name] = {"mode": "fp16_skip"}
    convert_model_to_fakequant(dst, mode="a8w8", act_stats=None, layer_policy=inferred_policy)
    missing, unexpected = dst.load_state_dict(state, strict=False)

    assert isinstance(dst.text_embedding[0], nn.Linear)
    assert not isinstance(dst.text_embedding[0], FakeQuantLinear)
    assert isinstance(dst.blocks[0]["self_attn"]["q"], FakeQuantLinear)
    assert not unexpected
    assert not [key for key in missing if key.startswith("text_embedding.0.")]


def test_static_mixed_policy_fallback_ratio_uses_fp16_skip():
    layer_names = [f"blocks.0.self_attn.{x}" for x in ("q", "k", "v", "o")] + [
        "text_embedding.0",
        "time_embedding.0",
        "blocks.0.ffn.0",
        "blocks.0.ffn.2",
        "head.head",
        "blocks.0.cross_attn.q",
    ]
    calibration = {name: {"act_min": [-1.0], "act_max": [1.0]} for name in layer_names}
    calibration["time_embedding.0"] = {"act_min": [-10.0], "act_max": [10.0]}

    policy = build_static_mixed_policy(layer_names, calibration, fallback_ratio=0.2)

    assert policy["counts"]["fp16_skip"] == 2
    assert policy["layers"]["time_embedding.0"]["mode"] == "fp16_skip"
    assert sum(1 for entry in policy["layers"].values() if entry["mode"] == "a8w8") == 8


def test_build_smoothquant_cache_from_calibration_and_weights():
    model = TinySQModel()
    stats = _stats_for(model)

    cache = build_smoothquant_cache(model, stats, alpha=0.5)

    assert cache["_metadata"]["alpha"] == 0.5
    assert "text_embedding.0" in cache
    assert len(cache["text_embedding.0"]["smoothquant_scale"]) == 4


def test_load_smoothquant_cache_accepts_multiple_entry_shapes(tmp_path):
    path = tmp_path / "sq.json"
    path.write_text(json.dumps({
        "a": {"smoothquant_scale": [1, 2]},
        "b": {"scale": [3, 4]},
        "c": [5, 6],
        "_metadata": {"ignored": True},
    }))

    loaded = load_smoothquant_cache(str(path), device="cpu")

    assert set(loaded) == {"a", "b", "c"}
    assert torch.equal(loaded["b"], torch.tensor([3.0, 4.0]))


def test_sensitivity_score_prefers_large_dynamic_range():
    small = sensitivity_score("blocks.0.self_attn.q", {"act_min": [-1], "act_max": [1]})
    large = sensitivity_score("blocks.0.self_attn.k", {"act_min": [-5], "act_max": [5]})
    assert large > small
