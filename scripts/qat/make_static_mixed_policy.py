"""Generate conservative static QAT mixed A8W8/A16W8 policy for FlashVSR DiT."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.ptq.fakequant_convert import build_dit


def classify_layer(name: str, ffn: str = "a16w8") -> dict:
    """Return policy entry for one DiT Linear layer name."""

    sensitive_prefixes = ("text_embedding.", "time_embedding.", "time_projection.", "head.head")
    if name.startswith(sensitive_prefixes):
        return {"mode": "a16w8", "reason": "sensitive_embedding_or_head"}
    if ".ffn." in name:
        return {"mode": ffn, "reason": "ffn_policy"}
    if ".self_attn." in name or ".cross_attn." in name:
        return {"mode": "a8w8", "activation_qdq_mode": "static_asymmetric", "reason": "attention_static_a8"}
    return {"mode": "a16w8", "reason": "conservative_default"}


def build_policy(layer_names: list[str], ffn: str = "a16w8") -> dict:
    return {
        "schema": "flashvsr.qat.layer_policy.v1",
        "description": "Conservative static mixed QAT: attention A8W8 static, embeddings/head and selected sensitive layers A16W8.",
        "layers": {name: classify_layer(name, ffn=ffn) for name in layer_names},
    }


def dit_linear_names() -> list[str]:
    model = build_dit()
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FlashVSR static mixed A8W8/A16W8 QAT policy")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffn", default="a16w8", choices=["a16w8", "a8w8"], help="FFN policy; a16w8 is conservative default")
    args = parser.parse_args()

    policy = build_policy(dit_linear_names(), ffn=args.ffn)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(policy, indent=2))
    print(json.dumps({"output": str(output), "layers": len(policy["layers"]), "ffn": args.ffn}, indent=2))


if __name__ == "__main__":
    main()
