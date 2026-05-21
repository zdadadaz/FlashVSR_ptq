"""RMSNorm folding utilities for TensorRT export.

Folds RMSNorm operations into preceding Linear layers to simplify
the computation graph for quantization and TensorRT export.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


def get_rmsnorm_weight(model: nn.Module, name: str) -> torch.Tensor:
    """Get RMSNorm weight parameter by module name.

    Args:
        model: Parent module containing the RMSNorm.
        name: Full path to RMSNorm module.

    Returns:
        Weight tensor from the RMSNorm module.
    """
    module = model.get_submodule(name)
    if not isinstance(module, nn.Module):
        raise ValueError(f"Module at {name} is not an RMSNorm")
    return module.weight


def fold_rmsnorm_into_linear(linear: nn.Linear, rmsnorm: nn.Module) -> nn.Linear:
    """Fold RMSNorm into a preceding Linear layer.

    RMSNorm: y = x * weight / sqrt(mean(x^2) + eps)
    Absorbed into Linear: weight' = weight * weight_rms / sqrt(mean(x^2) + eps)

    For export/quantization, we fold the RMSNorm scaling into the Linear weights.

    Args:
        linear: The Linear layer to fold RMSNorm into.
        rmsnorm: The RMSNorm module to fold.

    Returns:
        Modified Linear layer with folded weights.
    """
    if not isinstance(linear, nn.Linear):
        raise ValueError(f"Expected nn.Linear, got {type(linear)}")

    # Get RMSNorm weight and compute effective scaling
    # For RMSNorm: y = x * rms_weight / sqrt(mean(x^2) + eps)
    # We fold this into linear weights by scaling weight
    rms_weight = rmsnorm.weight.float()

    # Compute the fold: new_weight = weight * rms_weight
    # Bias is unaffected (RMSNorm doesn't have bias)
    with torch.no_grad():
        linear.weight.data = linear.weight.data.float() * rms_weight

    return linear


def fold_dit_rmsnorms(model: nn.Module) -> nn.Module:
    """Fold all RMSNorm operations in a WanModel DiT into preceding Linear layers.

    WanModel DiTBlock structure:
    - Each DiTBlock has: self.self_attn, self.cross_attn, self.mlp
    - SelfAttention has: .qkv (Linear), .proj (Linear)
    - CrossAttention has: .qkv (Linear), .proj (Linear)
    - MLP has: .fc1 (Linear), .fc2 (Linear)
    - DiTBlock has: self.norm1 (RMSNorm), self.norm2 (RMSNorm), self.norm3 (RMSNorm)

    Folding pattern:
    - block.norm1 -> block.self_attn.qkv (SelfAttention QKV linear)
    - block.norm2 -> block.cross_attn.qkv (CrossAttention QKV linear)
    - block.norm3 -> block.mlp.fc1 (MLP first linear)

    Args:
        model: WanModel to fold RMSNorms in.

    Returns:
        Model with folded RMSNorm operations.
    """
    from src.models.wan_video_dit import RMSNorm

    for block in model.blocks:
        # norm1 -> self_attn.qkv
        if hasattr(block, 'norm1') and isinstance(block.norm1, RMSNorm):
            if hasattr(block.self_attn, 'qkv') and isinstance(block.self_attn.qkv, nn.Linear):
                block.self_attn.qkv = fold_rmsnorm_into_linear(block.self_attn.qkv, block.norm1)

        # norm2 -> cross_attn.qkv
        if hasattr(block, 'norm2') and isinstance(block.norm2, RMSNorm):
            if hasattr(block.cross_attn, 'qkv') and isinstance(block.cross_attn.qkv, nn.Linear):
                block.cross_attn.qkv = fold_rmsnorm_into_linear(block.cross_attn.qkv, block.norm2)

        # norm3 -> mlp.fc1
        if hasattr(block, 'norm3') and isinstance(block.norm3, RMSNorm):
            if hasattr(block.mlp, 'fc1') and isinstance(block.mlp.fc1, nn.Linear):
                block.mlp.fc1 = fold_rmsnorm_into_linear(block.mlp.fc1, block.norm3)

    return model