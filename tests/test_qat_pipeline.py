import copy

import cv2
import numpy as np
import torch
import torch.nn as nn

from scripts.qat.finetune_fakequant_dit import LatentManifestDataset, move_sample
from scripts.qat.make_static_mixed_policy import build_policy
from scripts.qat.prepare_video_manifest import (
    deterministic_context,
    discover_paired_lq_gt,
    downsample_frames,
    frames_to_pseudo_latent,
    load_teacher_text_context,
    write_manifest_from_videos,
)
from scripts.qat.run_september_video_qat_eval import compute_gt_drop_row, resolve_gt_clip
from scripts.qat.static_diagnostic_runner import collect_static_linear_diagnostics, write_diagnostic_outputs
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
    freeze_qat_observers,
    set_qat_observer,
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


def test_quant_aware_linear_observer_collects_ema_minmax_and_freezes_static_qparams():
    ref = nn.Linear(4, 3)
    qat = QuantAwareLinear.from_float(ref, activation_mode="a8", weight_mode="w8", activation_qdq_mode="static_asymmetric")

    qat.enable_observer(True, ema_decay=0.5)
    _ = qat(torch.tensor([[[-1.0, -0.5, 0.0, 0.5]]]))
    _ = qat(torch.tensor([[[0.0, 1.0, 2.0, 3.0]]]))
    qat.freeze_activation_qparams()

    assert bool(qat.observer_enabled.item()) is False
    assert bool(qat.static_qparams_frozen.item()) is True
    assert bool(qat.observer_initialized.item()) is True
    assert not torch.allclose(qat.act_scale, torch.ones_like(qat.act_scale))
    assert torch.count_nonzero(qat.act_zero_point).item() > 0


def test_freeze_qat_observers_freezes_static_layers_and_export_preserves_static_mode():
    model = TinyWanLike()
    convert_model_to_qat(model, mode="a8w8", activation_qdq_mode="static_asymmetric")
    set_qat_observer(model, True, ema_decay=0.5)
    _ = model(torch.randn(2, 3, 4))

    summary = freeze_qat_observers(model)
    exported = export_qat_model_to_fakequant(model, inplace=False)

    assert summary["frozen"] == 5
    assert isinstance(exported.text_embedding[0], FakeQuantLinear)
    assert int(exported.text_embedding[0].activation_qdq_mode.item()) == 0
    assert torch.count_nonzero(exported.text_embedding[0].act_zero_point).item() > 0


def test_static_diagnostic_runner_ranks_bad_layers_and_writes_outputs(tmp_path):
    model = TinyWanLike()
    convert_model_to_qat(model, mode="a8w8", activation_qdq_mode="static_asymmetric")
    # Deliberately make one layer's static activation qparams poor so the
    # diagnostic ranking has a deterministic worst layer.
    model.blocks[0]["self_attn"]["q"].act_scale.fill_(10.0)
    model.blocks[0]["self_attn"]["q"].act_zero_point.zero_()

    rows = collect_static_linear_diagnostics(
        model,
        [{"input": torch.randn(2, 3, 4)}],
        device="cpu",
        dtype=torch.float32,
    )
    paths = write_diagnostic_outputs(rows, tmp_path, top_k=2)

    assert len(rows) == 5
    assert rows[0]["name"] == "blocks.0.self_attn.q"
    assert rows[0]["output_mse"] > 0
    assert "sqnr_db" in rows[0]
    assert "activation_min" in rows[0]
    assert (tmp_path / "static_qat_linear_diagnostics.json").exists()
    assert (tmp_path / "static_qat_linear_diagnostics.csv").exists()
    assert "blocks.0.self_attn.q" in (tmp_path / "static_qat_linear_top20.md").read_text()
    assert set(paths) == {"json", "csv", "markdown"}


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


def test_downsample_frames_reduces_lq_by_requested_scale():
    frames = torch.arange(2 * 16 * 20 * 3, dtype=torch.float32).reshape(2, 16, 20, 3).numpy()

    down = downsample_frames(frames, scale=4)

    assert down.shape == (2, 4, 5, 3)
    assert down.dtype == frames.dtype


def test_discover_paired_lq_gt_matches_common_dataset_layouts(tmp_path):
    lq_dir = tmp_path / "LQ"
    gt_dir = tmp_path / "GT"
    lq_dir.mkdir()
    gt_dir.mkdir()
    (lq_dir / "clip001.mp4").write_text("lq")
    (gt_dir / "clip001.mp4").write_text("gt")
    (gt_dir / "unmatched.mp4").write_text("gt")

    pairs = discover_paired_lq_gt(tmp_path)

    assert len(pairs) == 1
    assert pairs[0]["name"] == "clip001"
    assert pairs[0]["lq"] == lq_dir / "clip001.mp4"
    assert pairs[0]["gt"] == gt_dir / "clip001.mp4"


def test_discover_paired_lq_gt_matches_image_sequence_dirs_and_manifest_downsamples_lq(tmp_path):
    lq_seq = tmp_path / "LQ" / "003"
    gt_seq = tmp_path / "GT" / "003"
    lq_seq.mkdir(parents=True)
    gt_seq.mkdir(parents=True)
    for i in range(4):
        lq_frame = np.full((32, 40, 3), 32 + i, dtype=np.uint8)
        gt_frame = np.full((128, 160, 3), 64 + i, dtype=np.uint8)
        cv2.imwrite(str(lq_seq / f"{i:05d}.png"), cv2.cvtColor(lq_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(gt_seq / f"{i:05d}.png"), cv2.cvtColor(gt_frame, cv2.COLOR_RGB2BGR))

    pairs = discover_paired_lq_gt(tmp_path)
    summary = write_manifest_from_videos(
        video_dir=tmp_path,
        output_dir=tmp_path / "samples",
        manifest_path=tmp_path / "manifest.jsonl",
        max_videos=1,
        frames=2,
        latent_size=(16, 16),
        downsample_lq_scale=4,
    )
    sample_path = tmp_path / "samples" / "qat_sample_0000.pt"
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)

    assert pairs == [{"name": "003", "lq": lq_seq, "gt": gt_seq}]
    assert summary["paired_samples"] == 1
    assert sample["lq_source"] == str(lq_seq)
    assert sample["gt_source"] == str(gt_seq)
    assert sample["downsample_lq_scale"] == 4
    assert sample["x"].shape == (1, 16, 2, 4, 4)


def test_load_teacher_text_context_accepts_preembedded_context_file(tmp_path):
    context_path = tmp_path / "embedded_context.pt"
    expected = torch.randn(1, 7, 1536)
    torch.save(expected, context_path)

    loaded = load_teacher_text_context(context_path, checkpoint="", device="cpu")

    assert torch.allclose(loaded, expected)


def test_static_mixed_policy_keeps_sensitive_layers_a16_and_attention_static_a8():
    policy = build_policy([
        "time_embedding.0",
        "blocks.0.self_attn.q",
        "blocks.0.cross_attn.o",
        "blocks.0.ffn.0",
        "head.head",
    ])

    assert policy["layers"]["time_embedding.0"]["mode"] == "a16w8"
    assert policy["layers"]["head.head"]["mode"] == "a16w8"
    assert policy["layers"]["blocks.0.ffn.0"]["mode"] == "a16w8"
    assert policy["layers"]["blocks.0.self_attn.q"] == {
        "mode": "a8w8",
        "activation_qdq_mode": "static_asymmetric",
        "reason": "attention_static_a8",
    }
    assert policy["layers"]["blocks.0.cross_attn.o"]["activation_qdq_mode"] == "static_asymmetric"


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
