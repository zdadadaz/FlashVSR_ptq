#!/usr/bin/env python3
"""Build static mixed A8W8/A16W8 policies from per-layer sensitivity reports.

The generated policy keeps W8 weights everywhere.  Sensitive layers use A16
activation passthrough (`a16w8`); robust layers use static A8 activation QDQ
(`a8w8` + static qparams).  This is the NPU-static route from the
20260618 static mixed QAT plan.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def parse_percent_list(value: str) -> list[float]:
    out: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        pct = float(item)
        if pct < 0.0 or pct > 100.0:
            raise ValueError(f"A16 percent must be in [0, 100], got {pct}")
        out.append(pct)
    if not out:
        raise ValueError("No A16 percentages supplied")
    return out


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("layers"), list):
            rows = payload["layers"]
        elif isinstance(payload.get("top_layers"), list):
            rows = payload["top_layers"]
        elif isinstance(payload.get("layers"), dict):
            rows = []
            for name, entry in payload["layers"].items():
                row = dict(entry)
                row.setdefault("name", name)
                rows.append(row)
        else:
            # Accept simple {layer: {output_mse: ...}} fixtures.
            rows = []
            for name, entry in payload.items():
                if str(name).startswith("_") or not isinstance(entry, dict):
                    continue
                row = dict(entry)
                row.setdefault("name", name)
                rows.append(row)
    else:
        raise ValueError("Sensitivity report must be a JSON object or list")
    clean = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("name"):
            continue
        metric = row.get("output_mse", row.get("mse", row.get("sensitivity", 0.0)))
        clean.append({**row, "output_mse": float(metric)})
    if not clean:
        raise ValueError("Sensitivity report has no named layer rows")
    return clean


def load_sensitivity_rows(path: str | Path) -> list[dict[str, Any]]:
    return _rows_from_payload(json.loads(Path(path).read_text()))


def _layer_group(name: str) -> str:
    if name.startswith("text_embedding"):
        return "text_embedding"
    if name.startswith("time_embedding") or name.startswith("time_projection"):
        return "time"
    if name.startswith("head"):
        return "head"
    if ".self_attn." in name:
        return "self_attn"
    if ".cross_attn." in name:
        return "cross_attn"
    if ".ffn." in name:
        return "ffn"
    return "other"


def build_static_mixed_policy_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    a16_percent: float,
    default_activation_qdq_mode: str = "static_tensor_symmetric",
    metric_key: str = "output_mse",
) -> dict[str, Any]:
    ranked = []
    for row in rows:
        name = str(row["name"])
        metric = float(row.get(metric_key, row.get("output_mse", row.get("mse", 0.0))))
        ranked.append((name, metric, row))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    total = len(ranked)
    a16_count = int(math.ceil(total * float(a16_percent) / 100.0))
    a16_names = {name for name, _, _ in ranked[:a16_count]}

    layers: dict[str, dict[str, Any]] = {}
    for rank, (name, metric, row) in enumerate(ranked, start=1):
        if name in a16_names:
            layers[name] = {
                "mode": "a16w8",
                "reason": "top_sensitivity_static_a16_fallback",
                "sensitivity_rank": rank,
                "mse": metric,
                "group": _layer_group(name),
            }
        else:
            layers[name] = {
                "mode": "a8w8",
                "activation_qdq_mode": default_activation_qdq_mode,
                "reason": "robust_static_a8",
                "sensitivity_rank": rank,
                "mse": metric,
                "group": _layer_group(name),
            }
        if "sqnr_db" in row:
            layers[name]["sqnr_db"] = float(row["sqnr_db"])

    return {
        "schema_version": "flashvsr.static_mixed_policy.v1",
        "quant_scope": "dit_linear_only",
        "wan_vae_quantized": False,
        "activation_qdq_mode": default_activation_qdq_mode,
        "weight_mode": "w8",
        "default": {
            "mode": "a8w8",
            "activation_qdq_mode": default_activation_qdq_mode,
            "weight_mode": "w8",
            "act_quant_enabled": True,
        },
        "fallback": {
            "mode": "a16w8",
            "activation_semantics": "A16 activation passthrough; W8 weights remain quantized",
            "a16_percent": float(a16_percent),
        },
        "summary": {
            "total_linear_layers": total,
            "a8w8_layers": total - a16_count,
            "a16w8_layers": a16_count,
            "a16_percent": float(a16_percent),
        },
        "counts": {"a8w8": total - a16_count, "a16w8": a16_count},
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static mixed A8W8/A16W8 policies from sensitivity JSON")
    parser.add_argument("--sensitivity_json", required=True)
    parser.add_argument("--a16_percent", required=True, help="Comma-separated percentages, e.g. 10,20,40,60")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--activation_qdq_mode", default="static_tensor_symmetric")
    args = parser.parse_args()

    rows = load_sensitivity_rows(args.sensitivity_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pct in parse_percent_list(args.a16_percent):
        policy = build_static_mixed_policy_from_rows(
            rows,
            a16_percent=pct,
            default_activation_qdq_mode=args.activation_qdq_mode,
        )
        pct_label = str(int(pct)) if float(pct).is_integer() else str(pct).replace(".", "p")
        path = out_dir / f"mixed_top{pct_label}_a16.json"
        path.write_text(json.dumps(policy, indent=2))
        print(f"[static-mixed-policy] wrote {path} counts={policy['counts']}")


if __name__ == "__main__":
    main()
