# Deep Neural Network Quantization Agent Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a reproducible quantization experiment agent that can generate quantization candidates, run FlashVSR PTQ/QAT/eval commands, collect metrics, maintain a leaderboard, and choose the next experiment automatically.

**Architecture:** Start with a deterministic harness, not an LLM-first autonomous coder. The agent manipulates YAML/JSON experiment specs, executes existing FlashVSR scripts, records every artifact in SQLite/JSONL, and later plugs in Optuna/Bayesian/LLM planners. Scope is FlashVSR DiT Linear quantization only; Wan VAE remains unquantized.

**Tech Stack:** Python 3.10, FlashVSR repo `.venv`, PyYAML, SQLite, pandas, optional Optuna/W&B later, existing `scripts/ptq/*`, `cli_main.py`, `scripts/compare_video_psnr.py`.

---

## 0. Existing repo context

Target repo:
- `/home/user/apps/FlashVSRptq/FlashVSR_Integrated`

Existing reusable scripts/files:
- Calibration: `scripts/ptq/fakequant_calibrate.py`
- Conversion: `scripts/ptq/fakequant_convert.py`
- Existing policy runner: `scripts/ptq/quant_policy_matrix.py`
- Existing standardized eval helper: `scripts/ptq/lsgquant_standard_eval.py`
- Inference entrypoint: `cli_main.py`
- FP16-vs-quant PSNR: `scripts/compare_video_psnr.py` with positional args: `ref dist --out-json`
- FakeQuant core: `src/models/quantization/fakequant.py`
- Existing tests: `tests/scripts/ptq/test_quant_policy_matrix.py`, `tests/scripts/ptq/test_lsgquant_standard_eval.py`, `tests/test_output_qdq.py`, etc.

Important FlashVSR constraints:
- Quantization scope: DiT Linear layers only by default.
- Wan VAE is always fp16/fp32 and must not be quantized.
- Use repo venv: `/home/user/apps/FlashVSRptq/FlashVSR_Integrated/.venv/bin/python`.
- For FakeQuant safetensors eval, use `cli_main.py --quantize_mode FakeQuant_A8W8` or `FakeQuant_A8W8_DRAQ`, not `W8A8_PTQ`.
- Static A8W8 has known quality-collapse risk; dynamic/DRAQ modes are the strong first baselines.

---

## 1. Product definition

### What the agent should do

Given a search-space YAML and base experiment config, it should:
1. Generate one or more candidate quantization configs.
2. Materialize each config into a stable experiment directory.
3. Run needed stages:
   - optional calibration
   - checkpoint conversion
   - FP16 reference inference if missing
   - quantized inference
   - quality/runtime/VRAM metric extraction
4. Store results in SQLite + per-experiment JSON files.
5. Update a leaderboard.
6. Select the next candidate via grid/random/Optuna/LLM policy.
7. Produce a daily Markdown report under `/home/user/SynologyDrive/daily`.

### What MVP should not do

Do not implement these in the first PR:
- No LangChain/AutoGPT.
- No arbitrary code-editing agent.
- No W4A4-first search.
- No TensorRT engine compilation in the first harness.
- No automatic deletion/cleanup of large artifacts.

---

## 2. Proposed directory layout

Create a self-contained package under `quant_agent/`:

```text
quant_agent/
├── __init__.py
├── cli.py                  # main CLI: plan/run/status/leaderboard/report
├── config.py               # dataclasses + YAML loading/validation
├── search_space.py          # grid/random candidate expansion
├── planner.py               # chooses next candidates; Optuna later
├── runner.py                # stage orchestration + subprocess wrapper
├── flashvsr_adapter.py      # builds FlashVSR commands from config
├── evaluator.py             # metrics collection: PSNR/runtime/VRAM/log parsing
├── memory.py                # SQLite schema + JSONL mirror
├── report.py                # markdown + CSV leaderboard
└── schemas.py               # shared enums/constants

configs/quant_agent/
├── base_flashvsr.yaml
├── search_space_mvp.yaml
└── eval_sets_smoke.yaml

tests/quant_agent/
├── test_config.py
├── test_search_space.py
├── test_flashvsr_adapter.py
├── test_memory.py
├── test_evaluator.py
└── test_cli_dry_run.py
```

Runtime output should be outside git-tracked source:

```text
outputs/quant_agent/<run_id>/
├── agent_run.yaml
├── quant_agent.sqlite
├── leaderboard.csv
├── experiments.jsonl
└── exp_000001_<slug>/
    ├── config.yaml
    ├── status.json
    ├── commands.json
    ├── calibration/
    ├── checkpoints/
    ├── renders/
    ├── metrics.json
    └── logs/
```

Daily report path:

```text
/home/user/SynologyDrive/daily/YYYY-MM-DD_HHMMSS_quant_agent_<run_id>.md
```

---

## 3. MVP search space

Use a constrained FlashVSR-specific search space first:

```yaml
schema_version: flashvsr.quant_agent.search_space.v1
fixed:
  model_name: FlashVSR-v1.1
  checkpoint: models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors
  quant_scope: dit_linear_only
  wan_vae_quantized: false
  weight_bits: 8
  act_bits: 8
  weight_granularity: per_channel
  device: cuda:0

parameters:
  method:
    - fakequant_ptq
  activation_qdq_mode:
    - dynamic_asymmetric
    - draq_symmetric
    - static_tensor_symmetric
  calibration_source:
    - none
    - reds_lq_240x16
  calibration_num_samples:
    - 64
    - 128
    - 256
  static_scale_multiplier:
    - 1.0
    - 2.0
    - 3.0
  mixed_policy:
    - none
    - self_attn_only_a8
    - sensitivity_top10_a16

gates:
  min_fp16_vs_quant_psnr_db: 28.0
  max_vram_gb: 24
  require_all_outputs_exist: true
```

Initial candidate priority:
1. `FakeQuant_A8W8_DRAQ` / `draq_symmetric`, no static cache dependency.
2. `dynamic_asymmetric`, no static cache dependency.
3. `static_tensor_symmetric` with REDS LQ cache, clearly labeled as static ablation.
4. Mixed activation policies only after baseline candidates are recorded.

---

## 4. Experiment config schema

Each experiment should be fully described by YAML, not Python constants:

```yaml
schema_version: flashvsr.quant_agent.experiment.v1
experiment_id: exp_000001_draq_a8w8_smoke
parent_run_id: run_20260618_134842

model:
  name: FlashVSR-v1.1
  checkpoint: models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors
  quant_scope: dit_linear_only
  wan_vae_quantized: false

quant:
  backend: fakequant
  mode: a8w8
  cli_quantize_mode: FakeQuant_A8W8_DRAQ
  activation_qdq_mode: draq_symmetric
  weight_bits: 8
  act_bits: 8
  weight_granularity: per_channel
  act_granularity: dynamic_per_token_channel
  mixed_policy: null

calibration:
  required: false
  cache_path: null
  dataset_train: datasets/train
  num_videos: 240
  num_samples: 256
  calib_frames: 16
  seed: 42
  vae_model: Wan2.1

eval:
  inputs:
    - data/lowres/bowing_cif.mp4
    - data/lowres/carphone_qcif.mp4
  frames: 16
  mode: full
  scale: 4
  vae_model: Wan2.1
  tiled_vae: true
  tiled_dit: true
  frame_chunk_size: 16
  metrics:
    - fp16_vs_quant_psnr
    - runtime_sec
    - fps
    - peak_vram_gb

artifacts:
  output_dir: outputs/quant_agent/run_20260618_134842/exp_000001_draq_a8w8_smoke
```

---

## 5. SQLite schema

Use SQLite as canonical memory; mirror human-readable JSON in experiment dirs.

Tables:

```sql
CREATE TABLE runs (
  run_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  repo_root TEXT NOT NULL,
  git_commit TEXT,
  search_space_json TEXT NOT NULL,
  status TEXT NOT NULL
);

CREATE TABLE experiments (
  experiment_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  config_json TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  output_dir TEXT NOT NULL,
  error TEXT,
  FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE TABLE stages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  experiment_id TEXT NOT NULL,
  stage_name TEXT NOT NULL,
  command_json TEXT,
  log_path TEXT,
  started_at TEXT,
  finished_at TEXT,
  return_code INTEGER,
  status TEXT NOT NULL
);

CREATE TABLE metrics (
  experiment_id TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  metric_value REAL,
  metric_json TEXT,
  PRIMARY KEY(experiment_id, metric_name)
);

CREATE TABLE artifacts (
  experiment_id TEXT NOT NULL,
  artifact_type TEXT NOT NULL,
  path TEXT NOT NULL,
  exists_ok INTEGER NOT NULL DEFAULT 0
);
```

Key rule: command outputs and logs stay on disk; DB stores path + summary only.

---

## 6. CLI contract

Create `quant_agent/cli.py` with subcommands:

```bash
# Validate configs only
.venv/bin/python -m quant_agent.cli validate \
  --base configs/quant_agent/base_flashvsr.yaml \
  --search-space configs/quant_agent/search_space_mvp.yaml

# Show generated candidates without running GPU jobs
.venv/bin/python -m quant_agent.cli plan \
  --base configs/quant_agent/base_flashvsr.yaml \
  --search-space configs/quant_agent/search_space_mvp.yaml \
  --limit 10 \
  --out outputs/quant_agent/dry_run_plan.json

# Run N experiments sequentially
.venv/bin/python -m quant_agent.cli run \
  --base configs/quant_agent/base_flashvsr.yaml \
  --search-space configs/quant_agent/search_space_mvp.yaml \
  --run-id run_$(date +%Y%m%d_%H%M%S) \
  --max-experiments 3

# Recompute leaderboard from DB/artifacts
.venv/bin/python -m quant_agent.cli leaderboard \
  --run-dir outputs/quant_agent/run_20260618_134842

# Emit daily Markdown report
.venv/bin/python -m quant_agent.cli report \
  --run-dir outputs/quant_agent/run_20260618_134842 \
  --out /home/user/SynologyDrive/daily/2026-06-18_134842_quant_agent_run.md
```

Global flags:
- `--dry-run`: build commands/artifacts but do not execute.
- `--resume`: skip completed experiments with same config hash.
- `--fail-fast`: stop after first failure.
- `--continue-on-error`: record failure and continue.
- `--python`: defaults to current interpreter, but examples use `.venv/bin/python`.

---

## 7. Stage orchestration

For each experiment, `runner.py` builds these stages:

### Stage A: prepare

Actions:
- Create experiment dir.
- Write `config.yaml`.
- Write `commands.json`.
- Record config hash.
- Check checkpoint path exists or is auto-downloadable.

Verification:
- `config.yaml` exists.
- DB row status = `prepared`.

### Stage B: calibration, if required

Use `scripts/ptq/fakequant_calibrate.py` only when activation mode requires static cache.

Command pattern:

```bash
.venv/bin/python -u scripts/ptq/fakequant_calibrate.py \
  --checkpoint models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors \
  --output_cache outputs/quant_agent/<run>/<exp>/calibration/calib.json \
  --mode a8w8 \
  --dataset_train datasets/train \
  --num_videos 240 \
  --num_samples 256 \
  --calib_frames 16 \
  --seed 42 \
  --vae_model Wan2.1
```

Pitfalls:
- Use `-u` for unbuffered logs.
- Do not pipe through `tee | tail`; redirect logs or tee without tail.
- Calibration should run fp32 if standalone forward hits dtype mismatch.

Verification:
- Cache JSON exists.
- Layer coverage is 306 DiT Linear layers when full coverage is expected.
- Static mode 7 cache must include output QDQ fields if `static_tensor_symmetric` is used.

### Stage C: convert

Command pattern:

```bash
.venv/bin/python -u scripts/ptq/fakequant_convert.py \
  --checkpoint models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors \
  --output outputs/quant_agent/<run>/<exp>/checkpoints/model_fakequant.safetensors \
  --mode a8w8 \
  --activation_qdq_mode draq_symmetric
```

For static cache modes, add:

```bash
  --calibration_cache outputs/quant_agent/<run>/<exp>/calibration/calib.json
```

Verification:
- Output `.safetensors` exists and size > 0.
- Optional inspection confirms `activation_qdq_mode` buffers across all 306 layers.

### Stage D: FP16 reference inference

If reference output for the same input/settings already exists in run cache, reuse it.

Command pattern:

```bash
.venv/bin/python -u cli_main.py \
  --input data/lowres/bowing_cif.mp4 \
  --output outputs/quant_agent/<run>/references/bowing_fp16.mp4 \
  --scale 4 \
  --device cuda:0 \
  --quantize_mode None \
  --vae_model Wan2.1 \
  --mode full \
  --end_frame 16 \
  --tiled_vae --tiled_dit \
  --frame_chunk_size 16
```

Verification:
- Output MP4 exists and has nonzero frames via `ffprobe`/OpenCV.

### Stage E: quantized inference

Command pattern:

```bash
.venv/bin/python -u cli_main.py \
  --input data/lowres/bowing_cif.mp4 \
  --output outputs/quant_agent/<run>/<exp>/renders/bowing_quant.mp4 \
  --scale 4 \
  --device cuda:0 \
  --quantize_mode FakeQuant_A8W8_DRAQ \
  --ckpt_path outputs/quant_agent/<run>/<exp>/checkpoints/model_fakequant.safetensors \
  --vae_model Wan2.1 \
  --mode full \
  --end_frame 16 \
  --tiled_vae --tiled_dit \
  --frame_chunk_size 16
```

Verification:
- Output MP4 exists and has nonzero frames.
- Log does not indicate fallback to wrong quantization path.

### Stage F: metrics

Use positional `compare_video_psnr.py`:

```bash
.venv/bin/python scripts/compare_video_psnr.py \
  outputs/quant_agent/<run>/references/bowing_fp16.mp4 \
  outputs/quant_agent/<run>/<exp>/renders/bowing_quant.mp4 \
  --out-json outputs/quant_agent/<run>/<exp>/metrics/bowing_psnr.json
```

Also parse:
- runtime from CLI logs
- FPS if present
- peak VRAM from `nvidia-smi` snapshots or CLI logs if available
- artifact sizes

Verification:
- `metrics.json` exists.
- Leaderboard row includes PSNR, runtime/FPS, status, config hash, checkpoint path.

---

## 8. Planner policy

### MVP planner: deterministic grid + priority ordering

Implement in `search_space.py`:
- Cartesian product with exclusions.
- Stable candidate hash from normalized JSON.
- Skip candidates already completed in DB.
- Prioritize known-good modes:
  1. DRAQ
  2. dynamic asymmetric
  3. static tensor symmetric
  4. mixed static policies

### V1 planner: random search

Add:
- `--strategy random`
- `--seed`
- `--max-experiments`

### V2 planner: Optuna

Add later:
- Objective: maximize `mean_psnr - penalty_runtime - penalty_vram`.
- Pruning: abort low-quality candidates after first eval clip if PSNR is below threshold.
- Storage: Optuna RDB can reuse the same SQLite or separate DB.

### V3 planner: LLM planner, optional

Only after deterministic harness is trusted:
- LLM may propose a YAML candidate.
- Agent must validate candidate against schema and safety constraints.
- LLM cannot edit source code during experiment search.
- LLM output must be saved as `planner_rationale.md` per candidate.

---

## 9. Implementation tasks

### Task 1: Create package skeleton and config models

**Files:**
- Create: `quant_agent/__init__.py`
- Create: `quant_agent/config.py`
- Create: `quant_agent/schemas.py`
- Create: `configs/quant_agent/base_flashvsr.yaml`
- Create: `configs/quant_agent/search_space_mvp.yaml`
- Test: `tests/quant_agent/test_config.py`

**Objective:** Load and validate base/search-space YAML into typed Python objects.

**Verification:**

```bash
cd /home/user/apps/FlashVSRptq/FlashVSR_Integrated
.venv/bin/python -m pytest tests/quant_agent/test_config.py -v
```

Expected: config validation passes, invalid schema raises useful error.

---

### Task 2: Implement search-space candidate generation

**Files:**
- Create: `quant_agent/search_space.py`
- Test: `tests/quant_agent/test_search_space.py`

**Objective:** Generate stable, deduplicated candidates with config hashes.

**Details:**
- Product over `parameters`.
- Apply exclusion rules like `calibration_source: none` incompatible with `static_tensor_symmetric`.
- Stable hash: `sha256(json.dumps(candidate, sort_keys=True))[:12]`.

**Verification:**

```bash
.venv/bin/python -m pytest tests/quant_agent/test_search_space.py -v
```

Expected: deterministic ordering and stable hashes.

---

### Task 3: Implement FlashVSR adapter command builder

**Files:**
- Create: `quant_agent/flashvsr_adapter.py`
- Test: `tests/quant_agent/test_flashvsr_adapter.py`

**Objective:** Convert experiment configs into calibration/convert/inference/PSNR command arrays.

**Critical rules:**
- `FakeQuant_A8W8_DRAQ` for DRAQ eval.
- `FakeQuant_A8W8` for regular FakeQuant A8W8 eval.
- Never use `W8A8_PTQ` for FakeQuant safetensors.
- `compare_video_psnr.py` uses positional args.
- Always record `wan_vae_quantized=false`.

**Verification:**

```bash
.venv/bin/python -m pytest tests/quant_agent/test_flashvsr_adapter.py -v
```

Expected: generated commands match known-good CLI contracts.

---

### Task 4: Implement SQLite memory layer

**Files:**
- Create: `quant_agent/memory.py`
- Test: `tests/quant_agent/test_memory.py`

**Objective:** Persist runs, experiments, stages, metrics, and artifacts.

**Verification:**

```bash
.venv/bin/python -m pytest tests/quant_agent/test_memory.py -v
```

Expected: can create run, add experiment, update stages, query leaderboard.

---

### Task 5: Implement subprocess runner with robust logging

**Files:**
- Create: `quant_agent/runner.py`
- Test: `tests/quant_agent/test_runner.py`

**Objective:** Run command stages with logs, status updates, timeout, and failure capture.

**Requirements:**
- Use `subprocess.Popen` or `subprocess.run` with stdout/stderr to log file.
- Use unbuffered Python commands (`-u`) where applicable.
- Store return code and log path in DB.
- Dry-run mode writes commands without execution.

**Verification:**

```bash
.venv/bin/python -m pytest tests/quant_agent/test_runner.py -v
```

Expected: dry-run records commands; failing command records failure without corrupting DB.

---

### Task 6: Implement evaluator and leaderboard

**Files:**
- Create: `quant_agent/evaluator.py`
- Modify/Create: `quant_agent/report.py`
- Test: `tests/quant_agent/test_evaluator.py`

**Objective:** Aggregate per-clip PSNR JSON, runtime, FPS, VRAM into `metrics.json` and `leaderboard.csv`.

**Metric names:**
- `mean_fp16_vs_quant_psnr_db`
- `min_fp16_vs_quant_psnr_db`
- `mean_runtime_sec`
- `mean_fps`
- `peak_vram_gb`
- `checkpoint_size_mb`
- `status`

**Verification:**

```bash
.venv/bin/python -m pytest tests/quant_agent/test_evaluator.py -v
```

Expected: sorted leaderboard ranks higher PSNR first, then faster runtime.

---

### Task 7: Implement CLI

**Files:**
- Create: `quant_agent/cli.py`
- Test: `tests/quant_agent/test_cli_dry_run.py`

**Objective:** Provide `validate`, `plan`, `run`, `leaderboard`, `report` subcommands.

**Verification:**

```bash
.venv/bin/python -m quant_agent.cli validate \
  --base configs/quant_agent/base_flashvsr.yaml \
  --search-space configs/quant_agent/search_space_mvp.yaml

.venv/bin/python -m quant_agent.cli plan \
  --base configs/quant_agent/base_flashvsr.yaml \
  --search-space configs/quant_agent/search_space_mvp.yaml \
  --limit 3 \
  --out /tmp/quant_agent_plan.json

.venv/bin/python -m pytest tests/quant_agent/test_cli_dry_run.py -v
```

Expected: dry-run plan emits 3 candidate configs without GPU execution.

---

### Task 8: Smoke run on one clip

**Files:**
- No source changes unless bugs found.
- Output: `outputs/quant_agent/<run_id>/...`
- Report: `/home/user/SynologyDrive/daily/<timestamp>_quant_agent_smoke.md`

**Objective:** Execute one DRAQ A8W8 experiment on `data/lowres/bowing_cif.mp4`, first 16 frames.

**Command:**

```bash
cd /home/user/apps/FlashVSRptq/FlashVSR_Integrated
.venv/bin/python -m quant_agent.cli run \
  --base configs/quant_agent/base_flashvsr.yaml \
  --search-space configs/quant_agent/search_space_mvp.yaml \
  --run-id run_$(date +%Y%m%d_%H%M%S)_smoke \
  --max-experiments 1 \
  --strategy grid \
  --continue-on-error
```

**Verification:**
- Quant checkpoint exists.
- FP16 and quant MP4 exist and are nonempty.
- PSNR JSON exists.
- Leaderboard has one successful row.
- Daily report exists under Synology daily folder.

---

### Task 9: Add static-cache candidate smoke

**Objective:** Run one `static_tensor_symmetric` candidate with small calibration to verify calibration/convert/eval stage wiring.

**Verification:**
- Cache has expected metadata and layer count.
- Converted checkpoint contains mode-7 buffers.
- Report labels it as static ablation, not default production path.

---

### Task 10: Optional Optuna planner

**Files:**
- Modify: `quant_agent/planner.py`
- Test: `tests/quant_agent/test_planner_optuna.py`

**Objective:** Add `--strategy optuna` after grid runner is stable.

**Do not start this until:**
- Dry-run CLI works.
- One DRAQ smoke run succeeds.
- DB/leaderboard/report are stable.

---

## 10. Testing strategy

Fast CPU tests:

```bash
.venv/bin/python -m pytest tests/quant_agent -v
```

Integration dry-run:

```bash
.venv/bin/python -m quant_agent.cli plan \
  --base configs/quant_agent/base_flashvsr.yaml \
  --search-space configs/quant_agent/search_space_mvp.yaml \
  --limit 5 \
  --out /tmp/quant_agent_plan.json
```

Repo regression subset:

```bash
.venv/bin/python -m pytest \
  tests/scripts/ptq/test_quant_policy_matrix.py \
  tests/scripts/ptq/test_lsgquant_standard_eval.py \
  tests/test_output_qdq.py \
  tests/quant_agent -v
```

GPU smoke:

```bash
.venv/bin/python -m quant_agent.cli run \
  --base configs/quant_agent/base_flashvsr.yaml \
  --search-space configs/quant_agent/search_space_mvp.yaml \
  --max-experiments 1
```

---

## 11. Success criteria

MVP is complete when:
- `validate` rejects invalid YAML and accepts provided configs.
- `plan --limit 3` emits deterministic candidate configs.
- `run --max-experiments 1` can execute a DRAQ A8W8 smoke experiment end-to-end.
- SQLite records run/experiment/stage/metric/artifact rows.
- `leaderboard.csv` is generated and sorted.
- Daily Markdown report is written to `/home/user/SynologyDrive/daily`.
- Tests pass for `tests/quant_agent` and key PTQ adapter tests.

Quality gate for first useful agent:
- DRAQ/dynamic A8W8 should be in the expected ~27–32 dB FP16-vs-quant consistency range on local smoke clips.
- Static modes can be lower but must be labeled as ablations.

---

## 12. Risks and mitigations

Risk: Static A8W8 collapse due to diffusion activation drift.
- Mitigation: prioritize DRAQ/dynamic first; label static as ablation.

Risk: Wrong quantize mode during eval.
- Mitigation: adapter tests assert FakeQuant checkpoints use `FakeQuant_*`, not `W8A8_PTQ`.

Risk: Calibration logs appear hung.
- Mitigation: use `python -u`; avoid `tee | tail` buffering.

Risk: Large artifact sprawl.
- Mitigation: deterministic output dirs, artifact DB, no automatic deletion in MVP.

Risk: Planner explores invalid combinations.
- Mitigation: schema validation + exclusion rules before execution.

Risk: Metrics compare wrong resolution/frame count.
- Mitigation: evaluator verifies output video exists, frame count > 0, and compares same-run FP16 reference.

---

## 13. Recommended implementation order

1. Config/schema + tests.
2. Search-space generation + dry-run plan.
3. FlashVSR command adapter + tests.
4. SQLite memory layer.
5. Runner dry-run and command logging.
6. Evaluator/leaderboard/report.
7. CLI integration.
8. One-clip DRAQ A8W8 smoke.
9. Static-cache smoke.
10. Optuna/LLM planner only after harness is stable.

This keeps the first version deterministic, auditable, and directly useful for FlashVSR quantization experiments.
