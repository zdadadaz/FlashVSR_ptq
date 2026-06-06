"""
Quantization package for FlashVSR.

Exports FakeQuantLinear for true integer PTQ quantization.
Supports modes: a16w8, a8w8, a16w4, a8w4
"""

from .fakequant import (
    FakeQuantLinear,
    convert_model_to_fakequant,
    collect_activation_stats_fakequant,
    get_all_linear_layers,
)
from .lsgquant import LSGQuantLinear

from .ptq import (
    SymmetricWeightLinear,
    AsymmetricActLinear,
    convert_model_to_w8a16,
    convert_model_to_w8a8,
)

from .smoothquant import (
    inject_observers,
    calculate_smoothquant_scales,
)

__all__ = [
    # FakeQuant / LSGQuant (true integer residual)
    "FakeQuantLinear",
    "LSGQuantLinear",
    "convert_model_to_fakequant",
    "collect_activation_stats_fakequant",
    "get_all_linear_layers",
    # PTQ
    "SymmetricWeightLinear",
    "AsymmetricActLinear",
    "convert_model_to_w8a16",
    "convert_model_to_w8a8",
    # SmoothQuant
    "inject_observers",
    "calculate_smoothquant_scales",
]