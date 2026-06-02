"""PTQ policy matrix runner for FlashVSR DiT W8A8.

Person A / 2026-07 deliverable:
- compare static per-tensor, static per-channel, and dynamic per-token A8 policies;
- dump calibration stats with enough min/max metadata to derive per-tensor caches;
- optionally convert one W8A8 checkpoint per policy.

This module is intentionally light on imports so it can be unit-tested without CUDA.
The heavy calibration/conversion steps are delegated to fakequant_calibrate.py and
fakequant_convert.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class QuantPolicy:
    """A single W8A8 activation policy candidate for July PTQ comparison."""

    name: str
    description: str
    mode: str
    activation_qdq_mode: str
    calibration_granularity: str
    requires_calibration: bool = True


POLICIES: tuple[QuantPolicy, ...] = (
    QuantPolicy(
        name="per_tensor_static_asym",
        description="Static asymmetric A8, one activation scale/zero-point per Linear layer.",
        mode="a8w8",
        activation_qdq_mode="static_asymmetric",
        calibration_granularity="per_tensor",
    ),
    QuantPolicy(
        name="per_channel_static_asym",
        description="Static asymmetric A8, one activation scale/zero-point per Linear input channel.",
        mode="a8w8",
        activation_qdq_mode="static_asymmetric",
        calibration_granularity="per_channel",
    ),
    QuantPolicy(
        name="per_token_dynamic_asym",
        description="Dynamic asymmetric A8, one activation scale/zero-point per token at runtime.",
        mode="a8w8",
        activation_qdq_mode="dynamic_asymmetric",
        calibration_granularity="per_token",
        requires_calibration=False,
    ),
)


def policy_by_name(name: str) -> QuantPolicy:
    for policy in POLICIES:
        if policy.name == name:
            return policy
    valid = ", ".join(policy.name for policy in POLICIES)
    raise ValueError(f"Unknown policy {name!r}. Valid policies: {valid}")


def _as_list(value: Any) -> list[float]:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            # Flatten nested [1,1,C] style JSON tensors.
            out: list[float] = []
            stack = list(value)
            while stack:
                item = stack.pop(0)
                if isinstance(item, list):
                    stack = list(item) + stack
                else:
                    out.append(float(item))
            return out
        return [float(x) for x in value]
    return [float(value)]


def reduce_calibration_cache(
    input_cache: str | Path,
    output_cache: str | Path,
    granularity: str,
) -> dict[str, Any]:
    """Write a calibration cache transformed to the requested granularity.

    Args:
        input_cache: JSON produced by fakequant_calibrate.py.
        output_cache: destination JSON.
        granularity: "per_channel" (pass-through) or "per_tensor".

    Per-tensor reduction prefers act_min/act_max when available; this preserves
    asymmetric quantization math. Older caches that only contain scale/zp are
    reduced conservatively with max(scale) and zero_point=0.
    """

    input_cache = Path(input_cache)
    output_cache = Path(output_cache)
    raw = json.loads(input_cache.read_text())

    if granularity not in {"per_channel", "per_tensor"}:
        raise ValueError(f"Unsupported static calibration granularity: {granularity}")

    if granularity == "per_channel":
        raw.setdefault("_metadata", {})["calibration_granularity"] = "per_channel"
        output_cache.parent.mkdir(parents=True, exist_ok=True)
        output_cache.write_text(json.dumps(raw, indent=2))
        return raw

    reduced: dict[str, Any] = {}
    metadata = dict(raw.get("_metadata", {}))
    metadata["calibration_granularity"] = "per_tensor"
    metadata["source_cache"] = str(input_cache)
    reduced["_metadata"] = metadata

    qmin, qmax = -128.0, 127.0
    for name, stats in raw.items():
        if name.startswith("_"):
            continue
        if "act_min" in stats and "act_max" in stats:
            act_min = min(_as_list(stats["act_min"]))
            act_max = max(_as_list(stats["act_max"]))
            scale = max((act_max - act_min) / (qmax - qmin), 1e-6)
            zero_point = round(qmin - act_min / scale)
            zero_point = int(max(qmin, min(qmax, zero_point)))
        else:
            # Backward-compatible fallback for old caches. This is usable for
            # smoke tests but less faithful than min/max-derived per-tensor stats.
            scale = max(_as_list(stats["act_scale"]), default=1.0)
            zero_point = 0
            act_min = None
            act_max = None
        reduced[name] = {
            "act_scale": [float(scale)],
            "zero_point": [int(zero_point)],
        }
        if act_min is not None and act_max is not None:
            reduced[name]["act_min"] = [float(act_min)]
            reduced[name]["act_max"] = [float(act_max)]

    output_cache.parent.mkdir(parents=True, exist_ok=True)
    output_cache.write_text(json.dumps(reduced, indent=2))
    return reduced


def build_calibrate_cmd(args: argparse.Namespace, base_cache: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/ptq/fakequant_calibrate.py",
        "--checkpoint",
        args.checkpoint,
        "--output_cache",
        str(base_cache),
        "--mode",
        "a8w8",
        "--dataset_train",
        args.dataset_train,
        "--num_videos",
        str(args.num_videos),
        "--num_samples",
        str(args.num_samples),
        "--calib_frames",
        str(args.calib_frames),
        "--latent_size",
        args.latent_size,
        "--seed",
        str(args.seed),
    ]
    if args.video:
        cmd.extend(["--video", args.video])
    if args.vae_path:
        cmd.extend(["--vae_path", args.vae_path, "--vae_model", args.vae_model])
    return cmd


def build_convert_cmd(
    args: argparse.Namespace,
    policy: QuantPolicy,
    cache_path: Path | None,
    output_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/ptq/fakequant_convert.py",
        "--checkpoint",
        args.checkpoint,
        "--output",
        str(output_path),
        "--mode",
        policy.mode,
        "--activation_qdq_mode",
        policy.activation_qdq_mode,
        "--static_quality_policy",
        args.static_quality_policy,
    ]
    if cache_path is not None:
        cmd.extend(["--calibration_cache", str(cache_path)])
    return cmd


def run_cmd(cmd: Iterable[str], dry_run: bool) -> None:
    printable = " ".join(str(x) for x in cmd)
    print(f"$ {printable}")
    if not dry_run:
        subprocess.run(list(cmd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/emit FlashVSR DiT W8A8 PTQ policy matrix")
    parser.add_argument("--checkpoint", required=True, help="FP16 DiT checkpoint path")
    parser.add_argument("--out_dir", default="outputs/ptq_policy_matrix", help="Output directory")
    parser.add_argument("--dataset_train", default="datasets/train")
    parser.add_argument("--video", default=None, help="Optional single calibration video")
    parser.add_argument("--num_videos", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--calib_frames", type=int, default=32)
    parser.add_argument("--latent_size", default="60x80")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae_path", default=None)
    parser.add_argument("--vae_model", default="Wan2.1")
    parser.add_argument(
        "--static_quality_policy",
        default="none",
        choices=["none", "sensitive_a16", "self_attn_only_a8"],
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[p.name for p in POLICIES],
        help="Subset of policies to run",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only print commands and write manifest")
    parser.add_argument("--skip_calibration", action="store_true", help="Reuse existing base per-channel cache")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "calibration"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    selected = [policy_by_name(name) for name in args.policies]
    base_cache = cache_dir / "base_per_channel_static_asym.json"

    if any(p.requires_calibration for p in selected) and not args.skip_calibration:
        run_cmd(build_calibrate_cmd(args, base_cache), args.dry_run)

    manifest: dict[str, Any] = {
        "schema_version": "flashvsr.ptq_policy_matrix.v1",
        "checkpoint": args.checkpoint,
        "out_dir": str(out_dir),
        "policies": [],
    }

    for policy in selected:
        cache_path: Path | None = None
        if policy.calibration_granularity == "per_channel":
            cache_path = cache_dir / f"{policy.name}.json"
            if not args.dry_run and base_cache.exists():
                reduce_calibration_cache(base_cache, cache_path, "per_channel")
        elif policy.calibration_granularity == "per_tensor":
            cache_path = cache_dir / f"{policy.name}.json"
            if not args.dry_run and base_cache.exists():
                reduce_calibration_cache(base_cache, cache_path, "per_tensor")

        output_path = ckpt_dir / f"dit_w8a8_{policy.name}.safetensors"
        run_cmd(build_convert_cmd(args, policy, cache_path, output_path), args.dry_run)

        manifest["policies"].append(
            {
                **asdict(policy),
                "calibration_cache": str(cache_path) if cache_path else None,
                "checkpoint": str(output_path),
            }
        )

    manifest_path = out_dir / "policy_matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[PolicyMatrix] manifest → {manifest_path}")


if __name__ == "__main__":
    main()
