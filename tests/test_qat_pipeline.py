import copy

import torch
import torch.nn as nn

from src.models.quantization.fakequant import FakeQuantLinear
from src.models.quantization.qat import (
    QuantAwareLinear,
    convert_model_to_qat,
    export_qat_model_to_fakequant,
    quant_dequant_activation_ste,
    quant_dequant_weight_ste,
    temporal_consistency_loss,
    tensor_psnr,
    update_ema_model,
)


class TinyWanLike(nn.Module):
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

    def forward(self, x):
        x = self.text_embedding(x)
        x = self.blocks[0]["self_attn"]["q"](x)
        x = self.blocks[0]["ffn"](x)
        return self.head["head"](x)


def test_qat_activation_and_weight_qdq_use_ste_gradients():
    x = torch.randn(2, 3, 4, requires_grad=True)
    w = torch.randn(5, 4, requires_grad=True)

    y = quant_dequant_activation_ste(x, activation_mode="a8", activation_qdq_mode="dynamic_asymmetric")
    z = quant_dequant_weight_ste(w, weight_mode="w8")
    loss = y.sum() + z.sum()
    loss.backward()

    assert x.grad is not None
    assert w.grad is not None
    assert torch.count_nonzero(x.grad).item() > 0
    assert torch.count_nonzero(w.grad).item() > 0


def test_quant_aware_linear_trains_fp_params_and_preserves_shape():
    ref = nn.Linear(4, 3)
    qat = QuantAwareLinear.from_float(ref, activation_mode="a8", weight_mode="w8")
    x = torch.randn(2, 5, 4)

    out = qat(x)
    out.square().mean().backward()

    assert out.shape == (2, 5, 3)
    assert qat.weight.grad is not None
    assert qat.weight.dtype == ref.weight.dtype


def test_convert_model_to_qat_respects_mixed_layer_policy():
    model = TinyWanLike()
    policy = {
        "layers": {
            "text_embedding.0": {"mode": "a16w8"},
            "blocks.0.self_attn.q": {"mode": "a8w8", "activation_qdq_mode": "dynamic_asymmetric"},
            "blocks.0.ffn.0": {"mode": "a16w8"},
            "blocks.0.ffn.2": {"mode": "a16w8"},
            "head.head": {"mode": "fp16_skip"},
        }
    }

    convert_model_to_qat(model, mode="a8w8", layer_policy=policy)

    assert isinstance(model.text_embedding[0], QuantAwareLinear)
    assert model.text_embedding[0].activation_mode == "a16"
    assert isinstance(model.blocks[0]["self_attn"]["q"], QuantAwareLinear)
    assert model.blocks[0]["self_attn"]["q"].activation_mode == "a8"
    assert isinstance(model.head["head"], nn.Linear)
    assert model._qat_conversion_summary["skipped_fp16"] == 1


def test_export_qat_model_to_fakequant_replaces_qat_layers_with_inference_layers():
    model = TinyWanLike()
    convert_model_to_qat(model, mode="a8w8", activation_qdq_mode="dynamic_asymmetric")

    exported = export_qat_model_to_fakequant(model, inplace=False)

    assert isinstance(model.text_embedding[0], QuantAwareLinear)
    assert isinstance(exported.text_embedding[0], FakeQuantLinear)
    assert exported._qat_export_summary == {"exported": 5, "format": "FakeQuantLinear"}
    assert exported(torch.randn(1, 2, 4)).shape == (1, 2, 4)


def test_ema_update_tracks_float_params_and_copies_integer_buffers():
    model = TinyWanLike()
    convert_model_to_qat(model, mode="a8w8")
    ema = copy.deepcopy(model)
    first = model.text_embedding[0]
    ema_first = ema.text_embedding[0]
    with torch.no_grad():
        first.weight.add_(1.0)
        first.act_quant_enabled.fill_(False)

    update_ema_model(ema, model, decay=0.5)

    assert torch.allclose(ema_first.weight, first.weight - 0.5)
    assert bool(ema_first.act_quant_enabled.item()) is False


def test_psnr_and_temporal_consistency_helpers():
    teacher = torch.zeros(1, 1, 3, 2, 2)
    student = teacher.clone()
    student[:, :, 1] = 0.1

    psnr = tensor_psnr(student, teacher, data_range=1.0)
    temporal = temporal_consistency_loss(student, teacher)

    assert psnr.item() > 0
    assert temporal.item() > 0
    assert temporal_consistency_loss(torch.zeros(1, 4), torch.zeros(1, 4)).item() == 0.0
