# Agent Quant Loop — PSNR oracle as the agent's outer loop

Concept (NVIDIA kernel loop / CodeMender critique loop, applied to FlashVSR PTQ):
the LLM agent never touches the GPU. It only **proposes configs**; the harness
`scripts/ptq/agent_quant_loop.py` executes the deterministic
`calibrate → convert → render → PSNR` pipeline and returns a compact verdict.
PSNR-vs-FP16 is the numeric oracle — the equivalent of NVIDIA's compile+numeric
check that made their kernel agent work.

## Division of labor

| Layer | Owner | Examples |
|---|---|---|
| Hypothesis / experiment design | human (α) | mode 7/8 design, per-token frozen dynamic qparam insight |
| Config proposal + log diagnosis | agent | sweep `activation_qdq_mode`, `draq_qrange`, `static_quality_policy`, calibration `num_videos`/`latent_size`; read failure log tails to localize which stage/layer broke |
| Execution + scoring | harness (deterministic) | fakequant_calibrate → fakequant_convert → cli_main render → compare_video_psnr |
| Gate | harness | `gate_psnr_db` per variant; exit code 0/1 for the loop to branch on |

## Protocol for an agent session

1. Write/edit a config JSON (template: `configs/agent_quant_loop_example.json`).
   Each variant = `quantize_mode` + `calib{}` + `convert{}` kwargs that map 1:1
   onto `fakequant_calibrate.py` / `fakequant_convert.py` CLI flags.
2. `--dry-run` first: the manifest lists every command before any GPU work.
3. Run. Re-runs are incremental — existing calib caches, converted ckpts and
   rendered videos are reused, so a one-field tweak (e.g. change
   `activation_qdq_mode` only) costs just the stages that are missing.
4. Read `metrics/validation_set_psnr_summary.json` (machine-readable) or
   `reports/<ts>_summary.md` (table + failure log tails). On FAIL, the agent
   diagnoses from the log tail, edits the config, goto 2.

## Anti-reward-hacking rules baked in

- PSNR is computed against **fixed, cached FP16 refs** (`fp16_ref_dir`), never
  against a freshly-rendered baseline, so the oracle can't drift between runs.
- `psnr_min_db` (worst frame) is reported alongside the mean — a variant that
  passes on average but blows up on one frame is caught.
- Seeds and frame windows are pinned in `render` so renders are reproducible.
- The harness never lets the agent edit the scoring code path; only configs.

## What the agent should NOT be asked to do

- Invent quantization theory (mode 7/8-style insights stay human).
- Judge whether a failed variant's idea is worth another sweep — that call
  belongs to the human after reading the summary.
