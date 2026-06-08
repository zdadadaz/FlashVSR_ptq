# PR7 CI torchvision fix

## Issue
GitHub Actions failed in `python test_mock.py -v` because `src/pipelines/base.py` imports `torchvision.transforms.GaussianBlur`, but CI installed `torch` without `torchvision`.

```text
ModuleNotFoundError: No module named 'torchvision'
```

## Change
Updated `.github/workflows/test.yml` dependency install command to include `torchvision`.

## Validation
Ran locally:

```bash
.venv/bin/python test_mock.py -v
```

Result: 16 tests OK.
