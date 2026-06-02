"""Generate FlashVSR Person A August PTQ recovery policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.quantization.policy import build_august_mixed_policy


def load_layer_names_from_cache(path: str | Path) -> list[str]:
    raw = json.loads(Path(path).read_text())
    return sorted(k for k in raw if not k.startswith("_"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate August PTQ mixed precision recovery policy")
    parser.add_argument("--calibration_cache", required=True, help="JSON cache from July PTQ calibration")
    parser.add_argument("--output", required=True, help="Output policy JSON")
    parser.add_argument("--sensitive_mode", default="a16w8", choices=["a16w8", "fp16_skip"])
    parser.add_argument("--robust_mode", default="a8w8", choices=["a8w8", "a16w8"])
    parser.add_argument(
        "--robust_activation_qdq_mode",
        default="dynamic_asymmetric",
        choices=["static_asymmetric", "dynamic_symmetric", "dynamic_asymmetric"],
    )
    args = parser.parse_args()

    names = load_layer_names_from_cache(args.calibration_cache)
    policy = build_august_mixed_policy(
        names,
        sensitive_mode=args.sensitive_mode,
        robust_mode=args.robust_mode,
        robust_activation_qdq_mode=args.robust_activation_qdq_mode,
    )
    policy["source_calibration_cache"] = args.calibration_cache

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(policy, indent=2))
    print(f"[Policy] wrote {output} layers={len(names)} counts={policy['counts']}")


if __name__ == "__main__":
    main()
