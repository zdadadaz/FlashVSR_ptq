# Legacy PTQ test fixes

Date: 2026-06-09 11:06:10 CST
Branch: `fix/legacy-ptq-tests`
Base: `origin/main` (`5555648`)

## Scope

Fix legacy PTQ/mock test failures found after merging the LSGQuant PR stack into `main`.

## Root causes

1. `tests/scripts/ptq/test_calibrator_w8a8.py`
   - Test still expected a single latent tensor shape `(16, 24, 24)`.
   - Production `FlashVSRTQDataset` now returns a temporal calibration sample `(T, C, H, W)` and `run_calibration` batches it as `(B, T, C, H, W)`.

2. `tests/scripts/ptq/test_cli_w8a8_ptq.py`
   - Test used `../../../..` from `tests/scripts/ptq`, resolving one level above repo root.
   - Subprocess tried `/tmp/cli_main.py` instead of repo `cli_main.py`.

3. `scripts/ptq/compile_trt_w8a8.py`
   - Legacy tests still imported `export_dit_for_trt` and `create_trt_calibrator`.
   - The compile path had moved to `torch_tensorrt.dynamo.compile` and removed the old helper APIs.

4. Stale LSGQuant comments/docstrings
   - `lsgquant_convert.py` and `lsgquant.py` still described QAO/Hadamard as future scaffolding after the PR stack implemented them.

## Changes

- Updated calibrator test and doc comments to the temporal latent contract `(6, 16, H, W)`.
- Fixed CLI test repo-root resolution and uses `sys.executable` for the subprocess.
- Restored lightweight legacy helper APIs:
  - `export_dit_for_trt(model, example_input)` wraps `torch.export.export` for graph inspection.
  - `create_trt_calibrator(dataloader, num_samples=None)` returns a DataLoader-backed descriptor without relying on removed torch_tensorrt calibrator classes.
- Refreshed stale LSGQuant/QAO docstrings.

## Verification

Targeted failing tests:

```text
4 passed, 1 warning in 1.64s
```

Broader PTQ/mock suite:

```text
103 passed, 1 skipped, 1 warning in 15.78s
```

Commands:

```bash
/home/user/apps/FlashVSRptq/FlashVSR_Integrated/.venv/bin/python -m pytest \
  tests/scripts/ptq/test_calibrator_w8a8.py::TestFlashVSRTQDataset::test_dataset_returns_calibration_sample \
  tests/scripts/ptq/test_cli_w8a8_ptq.py::test_cli_has_w8a8_ptq_flags \
  tests/scripts/ptq/test_compile_trt_w8a8.py::TestImports::test_compile_trt_imports \
  tests/scripts/ptq/test_compile_trt_w8a8.py::TestCalibratorCreation::test_create_trt_calibrator_requires_dataloader \
  -vv --tb=short

/home/user/apps/FlashVSRptq/FlashVSR_Integrated/.venv/bin/python -m pytest test_mock.py tests/scripts/ptq -q
```
