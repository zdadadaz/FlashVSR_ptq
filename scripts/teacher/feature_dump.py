"""Teacher/intermediate feature dump helpers for FlashVSR PTQ/QAT experiments.

Format (schema_version=flashvsr.feature_dump.v1):
  root/
    manifest.json
    features/
      <safe_module_name>__<capture>__step000000.pt

Each feature file is a torch.save() payload:
  {
    "schema_version": "flashvsr.feature_tensor.v1",
    "name": module name,
    "capture": "input" | "output",
    "step": integer hook call index,
    "dtype": original tensor dtype string,
    "shape": original tensor shape,
    "tensor": CPU tensor, optionally cast to save_dtype,
  }

The helpers are intentionally pipeline-agnostic. They can be attached to
WanModel, FakeQuant WanModel, or small unit-test modules. Keep dumps small:
select a few insertion-point groups and first N calls rather than all tensors.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, cast

import torch
import torch.nn as nn

CaptureKind = Literal["input", "output"]


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).replace(".", "__")


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


@dataclass
class FeatureDumpWriter:
    root: str | Path
    run_id: str
    model_role: str = "fp16_teacher"
    save_dtype: str = "float16"
    max_calls_per_module: int = 1
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        root = cast(Path, self.root)
        self.feature_dir = root / "features"
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.calls: dict[str, int] = {}
        self.records: list[dict] = []

    def _cast_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        cpu = tensor.detach().cpu()
        if self.save_dtype == "float16" and torch.is_floating_point(cpu):
            return cpu.to(torch.float16)
        if self.save_dtype == "bfloat16" and torch.is_floating_point(cpu):
            return cpu.to(torch.bfloat16)
        if self.save_dtype == "float32" and torch.is_floating_point(cpu):
            return cpu.to(torch.float32)
        return cpu

    def write(self, name: str, capture: CaptureKind, value) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            return
        key = f"{name}:{capture}"
        step = self.calls.get(key, 0)
        if step >= self.max_calls_per_module:
            return
        self.calls[key] = step + 1

        rel = Path("features") / f"{_safe_name(name)}__{capture}__step{step:06d}.pt"
        payload = {
            "schema_version": "flashvsr.feature_tensor.v1",
            "name": name,
            "capture": capture,
            "step": step,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "tensor": self._cast_tensor(tensor),
        }
        root = cast(Path, self.root)
        torch.save(payload, root / rel)
        self.records.append(
            {
                "name": name,
                "capture": capture,
                "step": step,
                "path": str(rel),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "saved_dtype": self.save_dtype if torch.is_floating_point(tensor) else str(tensor.dtype),
            }
        )

    def write_manifest(self) -> Path:
        manifest = {
            "schema_version": "flashvsr.feature_dump.v1",
            "run_id": self.run_id,
            "model_role": self.model_role,
            "save_dtype": self.save_dtype,
            "max_calls_per_module": self.max_calls_per_module,
            "metadata": self.metadata,
            "features": self.records,
        }
        root = cast(Path, self.root)
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path


def register_feature_hooks(
    model: nn.Module,
    writer: FeatureDumpWriter,
    include: Iterable[str],
    capture: CaptureKind = "output",
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register hooks for named modules matched by Unix-shell wildcards.

    Example include patterns:
      - "text_embedding.*"
      - "time_projection"
      - "blocks.0.self_attn.q"
      - "blocks.*.ffn.*"
      - "head.head"
    """
    patterns = list(include)
    handles = []

    for name, module in model.named_modules():
        if not name:
            continue
        if not any(fnmatch.fnmatch(name, pat) for pat in patterns):
            continue

        def hook(mod, inputs, output, *, module_name=name):
            writer.write(module_name, capture, inputs if capture == "input" else output)

        handles.append(module.register_forward_hook(hook))
    return handles


def remove_hooks(handles: Iterable[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()
