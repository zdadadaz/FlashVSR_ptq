# PTQ W8A8 Bug Fix: Observer Restore Recursive — 2026-05-07

## Problem
W8A8 conversion found 0 layers because `inject_observers()` replaces modules recursively, but the restore logic only iterated direct children (`named_children()`), leaving nested `ObserverLinear` modules in the model tree.

When `convert_model_to_w8a8()` ran, it couldn't find any `nn.Linear` modules because they were still `ObserverLinear` instances.

## Root Cause
- `inject_observers()`: recursively replaces ALL `nn.Linear` → `ObserverLinear` at any depth
- Old restore: only used `model.named_children()` (top-level only) → missed nested modules like `blocks.0.attn.q`
- DiT module hierarchy: `model → blocks → ModuleDict → attn/q/k/v` — 4 levels deep

## Fix
Added `restore_observers()` function (recursive) and used `model.named_modules()` for stat collection:

```python
def restore_observers(model):
    """Recursively restore ObserverLinear back to nn.Linear (matching inject_observers depth)."""
    for name, module in model.named_children():
        if isinstance(module, ObserverLinear):
            new_linear = nn.Linear(...)
            setattr(model, name, new_linear)
        else:
            restore_observers(module)
```

Then in `collect_activation_stats()`:
- Collect stats using `model.named_modules()` (catches nested)
- Restore using `restore_observers(model)` (recursive)

## Verification
Tested with mock model (9 nested Linear modules):
- inject_observers: 9 ObserverLinear created
- restore_observers: 9 Linear restored (0 ObserverLinear remaining)

## Files Modified
- `src/models/quantization/smoothquant.py`: Added `restore_observers()`, fixed collect/restore loops