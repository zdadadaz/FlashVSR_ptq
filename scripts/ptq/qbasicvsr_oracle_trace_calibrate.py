#!/usr/bin/env python3
"""Dynamic-path oracle static-token calibration for QBasicVSR/FlashVSR.

This is the supported static A8W8 calibration path for bowing-style target
calibration.  It runs the real cli_main/nodes FlashVSR inference path with a
pre-converted dynamic FakeQuant checkpoint, hooks WanVideoDiT FakeQuantLinear
modules, and stores per-token qparams that can be converted with
fakequant_convert.py --activation_qdq_mode static_token_asymmetric.

Do not use this as an FP16/static-asymmetric calibration helper for production
static rows: FP-trace qparams do not match the quantized inference trajectory.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import torch
import torch.nn as nn

# Ensure repository root is importable when run as a script.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cli_main import VideoReader, VideoWriter  # noqa: E402
from scripts.ptq.fakequant_calibrate import build_lsgquant_calibration_cache  # noqa: E402
from src.models.quantization.fakequant import (  # noqa: E402
    FakeQuantLinear,
    attach_fakequant_conv_calibration_hooks,
    export_fakequant_conv_calibration_cache,
)


def install_comfy_mocks(models_dir: str | None = None) -> None:
    folder_paths_mock = MagicMock()
    folder_paths_mock.models_dir = models_dir or str(ROOT / "models")
    folder_paths_mock.get_filename_list = MagicMock(return_value=[])
    sys.modules["folder_paths"] = folder_paths_mock
    comfy_mock = MagicMock()
    comfy_utils_mock = MagicMock()
    comfy_utils_mock.ProgressBar = MagicMock()
    sys.modules["comfy"] = comfy_mock
    sys.modules["comfy.utils"] = comfy_utils_mock


def register_linear_hooks(model: nn.Module, *, calibration_granularity: str = "per_token"):
    act_stats: dict[str, dict[str, Any]] = {}
    output_stats: dict[str, dict[str, torch.Tensor]] = {}
    hooks = []

    if calibration_granularity not in {"per_channel", "per_token"}:
        raise ValueError(f"Unsupported calibration_granularity: {calibration_granularity}")

    def _merge_stat(name: str, act_min: torch.Tensor, act_max: torch.Tensor, act_sum: torch.Tensor, count: int, shape: tuple[int, ...]) -> None:
        # Keep only online aggregates.  REDS30 dynamic traces can invoke each
        # DiT Linear hundreds of times; retaining every per-token tensor until
        # finalize was enough to get killed after the 30th clip.
        act_min = act_min.detach().cpu()
        act_max = act_max.detach().cpu()
        act_sum = act_sum.detach().cpu()
        if name not in act_stats:
            act_stats[name] = {"min": act_min, "max": act_max, "sum": act_sum, "count": float(count), "shape": shape}
            return
        stats = act_stats[name]
        if tuple(stats["min"].shape) != tuple(act_min.shape):
            raise RuntimeError(
                f"Static-token trace shape changed for {name}: {tuple(stats['min'].shape)} -> {tuple(act_min.shape)}. "
                "Use a calibration set with the same CLI token shape or split caches per shape."
            )
        stats["min"] = torch.minimum(stats["min"], act_min)
        stats["max"] = torch.maximum(stats["max"], act_max)
        stats["sum"] = stats["sum"] + act_sum
        stats["count"] = float(stats["count"]) + float(count)

    def make_hook(name: str):
        def hook_fn(module, input, output):
            act = input[0] if isinstance(input, tuple) else input
            if not torch.is_tensor(act):
                return
            act = act.detach().float()
            if calibration_granularity == "per_token":
                # Match dynamic per-token QDQ granularity: one qparam per token,
                # shared across the feature/channel dimension.
                reduce_dims = (-1,)
                reduce_count = act.shape[-1]
            else:
                # Legacy diagnostic path: one qparam per input channel.
                reduce_dims = tuple(range(act.dim() - 1))
                reduce_count = 1
                for dim in reduce_dims:
                    reduce_count *= act.shape[dim]
            _merge_stat(
                name,
                act.amin(dim=reduce_dims, keepdim=True),
                act.amax(dim=reduce_dims, keepdim=True),
                act.sum(dim=reduce_dims, keepdim=True),
                reduce_count,
                tuple(act.shape),
            )

            out = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(out):
                out_amax = out.detach().float().abs().amax(dim=tuple(range(out.dim() - 1)), keepdim=False).cpu()
                if name not in output_stats:
                    output_stats[name] = {"amax": out_amax}
                else:
                    output_stats[name]["amax"] = torch.maximum(output_stats[name]["amax"], out_amax)
        return hook_fn

    for name, module in model.named_modules():
        # WanVideoDiT has exactly 306 target Linear layers.  In FP16 trace mode
        # they are nn.Linear; in quantized trace mode (e.g. dynamic A8W8) they are
        # FakeQuantLinear.  Hook both so calibration can replay the same runtime
        # path used by the target inference checkpoint.
        if (isinstance(module, nn.Linear) or isinstance(module, FakeQuantLinear)) and not name.startswith("LQ_proj_in"):
            hooks.append(module.register_forward_hook(make_hook(name)))
    return hooks, act_stats, output_stats


def finalize_stats(act_stats: dict, output_stats: dict) -> tuple[dict, dict]:
    result = {}
    qmin, qmax = -128.0, 127.0
    for name, stats in act_stats.items():
        act_min = stats["min"]
        act_max = stats["max"]
        act_mean = stats["sum"] / max(float(stats["count"]), 1.0)
        act_scale = ((act_max - act_min) / (qmax - qmin)).clamp(min=1e-6)
        zero_pt = torch.round(qmin - act_min / act_scale).clamp(qmin, qmax)
        result[name] = {
            "act_scale": act_scale.squeeze().float(),
            "zero_point": zero_pt.squeeze().long(),
            "act_min": act_min.squeeze().float(),
            "act_max": act_max.squeeze().float(),
            "act_mean": act_mean.squeeze().float(),
        }

    out_result = {}
    for name, stats in output_stats.items():
        out_scale = (stats["amax"] / 127.0).clamp(min=1e-8)
        out_result[name] = {
            "output_scale": out_scale.float(),
            "output_zero_point": torch.zeros_like(out_scale, dtype=torch.int32),
        }
    return result, out_result


def main() -> None:
    ap = argparse.ArgumentParser(description="Run real FlashVSR inference and collect oracle static activation qparams")
    ap.add_argument("--input", default="", help="Single calibration video. For REDS30 use --input_glob or --input_list.")
    ap.add_argument("--input_glob", default="", help="Glob of calibration videos; sorted lexicographically and replayed through the real CLI inference path.")
    ap.add_argument("--input_list", default="", help="Text file with one calibration video path per line.")
    ap.add_argument("--max_inputs", type=int, default=0, help="Optional cap after sorting/list loading; 0 means all inputs.")
    ap.add_argument("--output_cache", required=True)
    ap.add_argument("--output_video", default="")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--model", default="FlashVSR-v1.1")
    ap.add_argument("--mode", default="tiny", choices=["tiny", "tiny-long", "full"])
    ap.add_argument("--vae_model", default="Wan2.1")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16"])
    ap.add_argument("--checkpoint", required=True, help="Dynamic FakeQuant checkpoint to trace (typically activation_qdq_mode=dynamic_asymmetric).")
    ap.add_argument(
        "--trace_quantize_mode",
        default="FakeQuant_A8W8",
        choices=["FakeQuant_A8W8", "FakeQuant_A8W8_DRAQ", "FakeQuant_A8W4", "FakeQuant_A16W8", "FakeQuant_A16W4", "FakeQuant_A4W4", "None"],
        help="Quantization mode used while collecting trace activations. Production static-token calibration should use FakeQuant_A8W8 with a dynamic checkpoint; None is diagnostic-only and requires --allow_fp_trace_diagnostic.",
    )
    ap.add_argument("--allow_fp_trace_diagnostic", action="store_true", help="Allow trace_quantize_mode=None for diagnostics only. Do not use this cache for production static-token calibration.")
    ap.add_argument("--models_dir", default="")
    ap.add_argument(
        "--calibration_granularity",
        default="per_token",
        choices=["per_token", "per_channel"],
        help="Static qparam granularity. Production static-token calibration uses per_token; per_channel is legacy diagnostic-only and requires --allow_fp_trace_diagnostic.",
    )
    ap.add_argument("--tiled_vae", action="store_true")
    ap.add_argument("--tiled_dit", action="store_true")
    ap.add_argument("--tile_size", type=int, default=256)
    ap.add_argument("--tile_overlap", type=int, default=24)
    ap.add_argument("--resize_factor", type=float, default=1.0)
    ap.add_argument("--attention_mode", default="sparse_sage_attention", choices=["sparse_sage_attention", "block_sparse_attention", "flash_attention_2", "sdpa"])
    ap.add_argument("--sparse_ratio", type=float, default=2.0)
    ap.add_argument("--kv_ratio", type=float, default=3.0)
    ap.add_argument("--local_range", type=int, default=9)
    ap.add_argument(
        "--extra_calibration_cache_out",
        default="",
        help="Optional JSON cache for extra scopes collected from the same dynamic trace path.",
    )
    ap.add_argument(
        "--extra_calibration_scopes",
        default="",
        help="Comma-separated extra scopes to hook while replaying calibration inputs; currently supports tcdecoder.",
    )
    args = ap.parse_args()
    inputs: list[str] = []
    if args.input_list:
        inputs.extend([line.strip() for line in Path(args.input_list).read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")])
    if args.input_glob:
        import glob
        inputs.extend(sorted(glob.glob(args.input_glob)))
    if args.input:
        inputs.append(args.input)
    # Preserve order while dropping duplicates.
    inputs = list(dict.fromkeys(inputs))
    if args.max_inputs > 0:
        inputs = inputs[: args.max_inputs]
    if not inputs:
        raise SystemExit("No calibration inputs supplied; pass --input, --input_glob, or --input_list")

    if args.trace_quantize_mode == "None" and not args.allow_fp_trace_diagnostic:
        raise SystemExit(
            "trace_quantize_mode=None is an old FP-trace diagnostic path and is blocked for production static calibration. "
            "Use --trace_quantize_mode FakeQuant_A8W8 with a dynamic FakeQuant checkpoint, or pass "
            "--allow_fp_trace_diagnostic only for debugging."
        )
    if args.calibration_granularity != "per_token" and not args.allow_fp_trace_diagnostic:
        raise SystemExit(
            "Only per_token calibration is supported for the current static-token method. "
            "per_channel is legacy diagnostic-only; pass --allow_fp_trace_diagnostic to inspect it."
        )

    install_comfy_mocks(args.models_dir or None)
    from nodes import init_pipeline, flashvsr  # noqa: E402
    from src.models import wan_video_dit  # noqa: E402

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    if args.precision == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    wan_video_dit.ATTENTION_MODE = args.attention_mode
    pipe = init_pipeline(
        model=args.model,
        mode=args.mode,
        device=device,
        dtype=dtype,
        vae_model=args.vae_model,
        quantize_mode=args.trace_quantize_mode,
        ckpt_path=args.checkpoint,
    )
    dit = pipe.denoising_model()
    hooks, raw_act, raw_out = register_linear_hooks(dit, calibration_granularity=args.calibration_granularity)
    extra_hooks = []
    extra_stats_list = []
    extra_scopes = {x.strip().lower() for x in (args.extra_calibration_scopes or "").split(",") if x.strip()}
    if args.extra_calibration_cache_out:
        unsupported = sorted(extra_scopes - {"tcdecoder"})
        if unsupported:
            raise SystemExit(f"Unsupported extra calibration scope(s): {unsupported}")
        if "tcdecoder" in extra_scopes:
            if getattr(pipe, "TCDecoder", None) is None:
                raise RuntimeError("Requested tcdecoder extra calibration, but pipeline has no TCDecoder")
            tc_hooks, tc_stats = attach_fakequant_conv_calibration_hooks(
                pipe.TCDecoder,
                prefix="tcdecoder",
                op_types=("linear", "conv2d", "conv3d"),
            )
            extra_hooks.extend(tc_hooks)
            extra_stats_list.append(tc_stats)
            print(f"[oracle-trace] registered TCDecoder extra hooks: {len(tc_hooks)}", flush=True)
    print(f"[oracle-trace] registered Linear hooks: {len(hooks)}", flush=True)
    if len(hooks) != 306:
        raise RuntimeError(f"Expected 306 WanVideoDiT Linear hooks, got {len(hooks)}")

    print(f"[oracle-trace] calibration inputs: {len(inputs)}", flush=True)
    writer = None
    try:
        for input_idx, input_path in enumerate(inputs, start=1):
            print(f"[oracle-trace] ({input_idx}/{len(inputs)}) {input_path}", flush=True)
            reader = VideoReader(input_path, start_frame=0, end_frame=args.frames, chunk_size=args.frames)
            fps, _ = reader.get_info()
            for frames in reader:
                output_frames = flashvsr(
                    pipe=pipe,
                    frames=frames,
                    scale=args.scale,
                    color_fix=True,
                    color_fix_method="wavelet",
                    tiled_vae=args.tiled_vae,
                    tiled_dit=args.tiled_dit,
                    tile_size=args.tile_size,
                    tile_overlap=args.tile_overlap,
                    unload_dit=False,
                    sparse_ratio=args.sparse_ratio,
                    kv_ratio=args.kv_ratio,
                    local_range=args.local_range,
                    seed=0,
                    force_offload=True,
                    enable_debug=False,
                    chunk_size=args.frames,
                    resize_factor=args.resize_factor,
                    mode=args.mode,
                )
                if args.output_video and input_idx == 1:
                    out_path = Path(args.output_video)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if writer is None:
                        h, w = output_frames.shape[1], output_frames.shape[2]
                        writer = VideoWriter(str(out_path), fps=fps, width=w, height=h, codec="mp4v", crf=18)
                    writer.write(output_frames)
                del frames, output_frames
                break
            del reader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()
        for h in extra_hooks:
            h.remove()
        if writer is not None:
            writer.release()

    act_stats, output_stats = finalize_stats(raw_act, raw_out)
    if len(act_stats) != 306:
        raise RuntimeError(f"Expected stats for 306 Linear layers, got {len(act_stats)}")
    cache = build_lsgquant_calibration_cache(
        act_stats,
        metadata={
            "mode": "a8w8",
            "calibration_type": f"oracle_trace_{args.calibration_granularity}_asymmetric",
            "trace_path": "cli_main/nodes.flashvsr real inference",
            "calibration_granularity": args.calibration_granularity,
            "input_video": args.input,
            "input_glob": args.input_glob,
            "input_list": args.input_list,
            "input_count": len(inputs),
            "inputs": inputs,
            "frames": args.frames,
            "model": args.model,
            "pipeline_mode": args.mode,
            "vae_model": args.vae_model,
            "checkpoint": args.checkpoint,
            "trace_quantize_mode": args.trace_quantize_mode,
            "attention_mode": args.attention_mode,
            "sparse_ratio": args.sparse_ratio,
            "kv_ratio": args.kv_ratio,
            "local_range": args.local_range,
            "quant_scope": "dit_linear_only",
            "wan_vae_quantized": False,
            "notes": "Actual inference-path activations from target clip. Use trace_quantize_mode=FakeQuant_A8W8 with a dynamic checkpoint to make static-token qparams replay the quantized dynamic path.",
        },
        output_stats=output_stats,
    )
    out_cache = Path(args.output_cache)
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    out_cache.write_text(json.dumps(cache, indent=2))
    print(f"[oracle-trace] wrote {out_cache} with {len(cache)-1} layers", flush=True)

    if args.extra_calibration_cache_out:
        extra_stats = {}
        for item in extra_stats_list:
            extra_stats.update(item)
        extra_cache = export_fakequant_conv_calibration_cache(extra_stats)
        extra_cache.setdefault("metadata", {})
        extra_cache["metadata"].update({
            "trace_path": "cli_main/nodes.flashvsr real inference",
            "input_glob": args.input_glob,
            "input_list": args.input_list,
            "input_count": len(inputs),
            "inputs": inputs,
            "frames": args.frames,
            "checkpoint": args.checkpoint,
            "trace_quantize_mode": args.trace_quantize_mode,
            "extra_calibration_scopes": sorted(extra_scopes),
            "notes": "Extra scope activations captured during the same dynamic FakeQuant DiT REDS30 trace.",
        })
        extra_out = Path(args.extra_calibration_cache_out)
        extra_out.parent.mkdir(parents=True, exist_ok=True)
        extra_out.write_text(json.dumps(extra_cache, indent=2))
        print(
            f"[oracle-trace] wrote extra calibration cache {extra_out} "
            f"with {extra_cache.get('summary', {}).get('num_layers', 0)} layers",
            flush=True,
        )

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
