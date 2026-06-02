import torch
import torch.nn as nn

from src.models.quantization.fakequant import FakeQuantLinear, convert_model_to_fakequant


class TinyPolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_embedding = nn.Sequential(nn.Linear(4, 4))
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.ModuleDict({"q": nn.Linear(4, 4)}),
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
            }
    return stats


def test_sensitive_a16_policy_disables_static_activation_qdq_only_for_sensitive_layers():
    model = TinyPolicyModel()
    stats = _stats_for(model)

    convert_model_to_fakequant(
        model,
        mode="a8w8",
        act_stats=stats,
        static_quality_policy="sensitive_a16",
    )

    assert isinstance(model.text_embedding[0], FakeQuantLinear)
    assert not bool(model.text_embedding[0].act_quant_enabled.item())
    assert not bool(model.blocks[0]["ffn"][0].act_quant_enabled.item())
    assert not bool(model.blocks[0]["ffn"][2].act_quant_enabled.item())
    assert not bool(model.head["head"].act_quant_enabled.item())

    assert bool(model.blocks[0]["self_attn"]["q"].act_quant_enabled.item())
    assert model.blocks[0]["self_attn"]["q"].weight_int.dtype == torch.int8


def test_self_attn_only_policy_keeps_static_a8_only_on_self_attention():
    model = TinyPolicyModel()
    model.blocks[0]["cross_attn"] = nn.ModuleDict({"q": nn.Linear(4, 4)})
    stats = _stats_for(model)

    convert_model_to_fakequant(
        model,
        mode="a8w8",
        act_stats=stats,
        static_quality_policy="self_attn_only_a8",
    )

    assert bool(model.blocks[0]["self_attn"]["q"].act_quant_enabled.item())
    assert not bool(model.blocks[0]["cross_attn"]["q"].act_quant_enabled.item())
    assert not bool(model.text_embedding[0].act_quant_enabled.item())
    assert not bool(model.blocks[0]["ffn"][0].act_quant_enabled.item())


def test_act_quant_enabled_buffer_survives_state_dict_reload_shape():
    src = FakeQuantLinear(4, 4, activation_mode="a8", weight_mode="w8", act_quant_enabled=False)
    dst = FakeQuantLinear(4, 4, activation_mode="a8", weight_mode="w8", act_quant_enabled=True)

    dst.load_state_dict(src.state_dict(), strict=False)

    assert not bool(dst.act_quant_enabled.item())


def test_dynamic_asymmetric_activation_qdq_mode_is_saved_and_forward_runs():
    layer = nn.Linear(4, 3)
    fq = FakeQuantLinear.from_float(
        layer,
        activation_mode="a8",
        weight_mode="w8",
        activation_qdq_mode="dynamic_asymmetric",
    )

    assert int(fq.activation_qdq_mode.item()) == 2
    x = torch.tensor([[[0.0, 1.0, 2.0, 100.0], [-5.0, -1.0, 0.5, 2.0]]])
    y = fq(x)

    assert y.shape == (1, 2, 3)
    assert y.dtype == x.dtype
    assert fq.weight_int.dtype == torch.int8


def test_dynamic_a8_convert_does_not_require_static_act_stats():
    model = TinyPolicyModel()

    convert_model_to_fakequant(
        model,
        mode="a8w8",
        act_stats={},
        activation_qdq_mode="dynamic_asymmetric",
    )

    assert isinstance(model.text_embedding[0], FakeQuantLinear)
    assert int(model.text_embedding[0].activation_qdq_mode.item()) == 2
    assert bool(model.text_embedding[0].act_quant_enabled.item())
