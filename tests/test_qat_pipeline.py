import copy

import torch
import torch.nn as nn

from scripts.qat.finetune_fakequant_dit import LatentManifestDataset, move_sample
from scripts.qat.prepare_video_manifest import deterministic_context, frames_to_pseudo_latent
from scripts.qat.run_september_video_qat_eval import compute_gt_drop_row, resolve_gt_clip
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
from src.models.wan_video_dit import CrossAttention


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


def test_video_frames_convert_to_dit_ready_pseudo_latent_and_context():
    frames = torch.linspace(0, 1, steps=4 * 16 * 16 * 3).reshape(4, 16, 16, 3).numpy()

    latent = frames_to_pseudo_latent(frames)
    ctx_a = deterministic_context(0)
    ctx_b = deterministic_context(0)

    assert latent.shape == (1, 16, 4, 16, 16)
    assert latent.dtype == torch.float32
    assert latent.min().item() < 0
    assert latent.max().item() > 0
    assert ctx_a.shape == (1, 10, 1536)
    assert torch.allclose(ctx_a, ctx_b)


def test_latent_manifest_dataset_accepts_repo_relative_sample_paths(tmp_path):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    sample_path = sample_dir / "sample.pt"
    torch.save({
        "x": torch.zeros(1, 16, 4, 16, 16),
        "timestep": torch.tensor([1000.0]),
        "context": torch.zeros(1, 10, 1536),
    }, sample_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"sample": "' + str(sample_path) + '"}\n')

    dataset = LatentManifestDataset(manifest)

    assert dataset[0]["x"].shape == (1, 16, 4, 16, 16)


def test_move_sample_skips_manifest_metadata_strings():
    sample = {
        "x": torch.zeros(1, 16, 4, 16, 16),
        "timestep": torch.tensor([1000.0]),
        "context": torch.zeros(1, 10, 1536),
        "source_video": "datasets/train/example.mp4",
    }

    moved = move_sample(sample, torch.device("cpu"), torch.bfloat16)

    assert set(moved) == {"x", "timestep", "context"}
    assert moved["x"].dtype == torch.bfloat16
    assert moved["context"].dtype == torch.bfloat16
    assert moved["timestep"].dtype == torch.bfloat16


def test_cross_attention_offline_context_matches_module_dtype():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn = CrossAttention(dim=128, num_heads=2).to(device=device, dtype=torch.bfloat16)
    x = torch.zeros(1, 4, 128, device=device, dtype=torch.bfloat16)
    context = torch.zeros(1, 3, 128, device=device, dtype=torch.bfloat16)

    out = attn(x, context)

    assert out.shape == x.shape
    assert out.dtype == torch.bfloat16


def test_resolve_gt_clip_accepts_named_and_original_filenames(tmp_path):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    original = gt_dir / "bowing_cif.mp4"
    named = gt_dir / "bowing.mp4"
    original.write_text("original")
    named.write_text("named")

    assert resolve_gt_clip(gt_dir, "bowing", "bowing_cif.mp4") == named
    named.unlink()
    assert resolve_gt_clip(gt_dir, "bowing", "bowing_cif.mp4") == original


def test_compute_gt_drop_row_reports_fp16_minus_qat_drop():
    fp16_metric = {"psnr_avg_db": 31.25, "frames": 16}
    qat_metric = {"psnr_avg_db": 30.9, "frames": 16}

    row = compute_gt_drop_row("bowing", fp16_metric, qat_metric, threshold=0.4)

    assert row["clip"] == "bowing"
    assert abs(row["psnr_drop_db"] - 0.35) < 1e-6
    assert row["passes_threshold"] is True
