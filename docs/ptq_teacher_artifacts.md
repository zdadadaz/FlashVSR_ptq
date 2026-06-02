# FlashVSR Teacher Artifacts and PTQ/QAT Insertion Points

## FP16 teacher inference script

Use `scripts/teacher/run_fp16_teacher_inference.sh` to produce deterministic FP16 teacher outputs for PTQ/QAT runs.

Default baseline command:

```bash
RUN_ID=20260602_144432 \
  scripts/teacher/run_fp16_teacher_inference.sh
```

Default settings:

- Input: `data/lowres/bowing_cif.mp4`
- Frames: `0..16`
- Output: `outputs/teacher/fp16/<RUN_ID>/teacher_fp16_first16.mp4`
- Manifest: `outputs/teacher/fp16/<RUN_ID>/teacher_manifest.json`
- Quantization: `--quantize_mode None`
- Precision: `fp16`
- Mode: `full`
- Attention: `sdpa`
- VAE: `Wan2.1` (unquantized)

Override via env vars: `INPUT`, `OUT_DIR`, `OUTPUT`, `START_FRAME`, `END_FRAME`, `MODE`, `PRECISION`, `DEVICE`, `ATTENTION_MODE`, `SCALE`, `SEED`, `REF_VIDEO`.

## Teacher output format

`teacher_manifest.json` uses schema `flashvsr.teacher.v1`:

```json
{
  "schema_version": "flashvsr.teacher.v1",
  "role": "fp16_teacher_output",
  "run_id": "YYYYMMDD_HHMMSS",
  "input": "data/lowres/bowing_cif.mp4",
  "output": {"path": "...mp4", "exists": true, "bytes": 123},
  "model": "FlashVSR-v1.1",
  "vae_model": "Wan2.1",
  "scope": {
    "teacher_precision": "fp16",
    "quantize_mode": "None",
    "dit_quantized": false,
    "wan_vae_quantized": false
  },
  "runtime": {
    "device": "cuda:0",
    "mode": "full",
    "attention_mode": "sdpa",
    "scale": 4,
    "start_frame": 0,
    "end_frame": 16,
    "seed": 0
  },
  "artifacts": {
    "teacher_video": "...mp4",
    "feature_dump_manifest": null,
    "psnr_json": null
  }
}
```

If `REF_VIDEO` is set, the wrapper also writes PSNR JSON via `scripts/compare_video_psnr.py` and embeds the quality block in the manifest.

## Intermediate feature dump format

Use `scripts/teacher/feature_dump.py` helpers in future teacher/student runs. It defines:

- `FeatureDumpWriter(root, run_id, model_role, save_dtype, max_calls_per_module)`
- `register_feature_hooks(model, writer, include=[...], capture="output")`
- `remove_hooks(handles)`

Feature dump root:

```text
<root>/
  manifest.json
  features/
    <safe_module_name>__<capture>__step000000.pt
```

Feature dump manifest schema: `flashvsr.feature_dump.v1`.

Each `.pt` payload schema: `flashvsr.feature_tensor.v1`:

```python
{
  "schema_version": "flashvsr.feature_tensor.v1",
  "name": "blocks.0.self_attn.q",
  "capture": "output",
  "step": 0,
  "dtype": "torch.float16",
  "shape": [1, 6336, 1536],
  "tensor": cpu_tensor_saved_as_float16,
}
```

Keep dumps small: priority-1 points only for first pass unless debugging a layer-specific failure.

## Initial PTQ/QAT insertion points

Machine-readable definition: `configs/ptq_qat_insertion_points.json` (`flashvsr.ptq_qat_insertion_points.v1`).

Initial policy:

1. **Default quantization scope:** WanVideoDiT / DiT only.
2. **Primary insertion point:** all 306 `nn.Linear` modules in DiT.
3. **Weight PTQ:** per-output-channel symmetric int8/int4 QDQ.
4. **Activation A8 options:**
   - dynamic per-token symmetric int8 — current best-quality path from previous runs
   - dynamic per-token asymmetric signed int8 — current asymmetric experiment
   - static calibrated asymmetric signed int8 — useful for sensitivity/mixed policy, lower quality alone
5. **QAT starting point:** reuse `FakeQuantLinear`; freeze Wan VAE; distill final HR output plus a small set of DiT intermediate features.
6. **Excluded initially:** Wan VAE, TCDecoder, LQ projection, attention softmax/matmul kernels, DiT patch Conv3d.

Priority feature targets for distillation/QAT:

- `text_embedding`
- `time_projection`
- `blocks.0.self_attn.q`
- `blocks.15.self_attn.q`
- `blocks.29.self_attn.q`
- `blocks.0.ffn.2`
- `blocks.15.ffn.2`
- `head.head`
