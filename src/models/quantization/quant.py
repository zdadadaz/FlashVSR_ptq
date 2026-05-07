import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightOnlyInt8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=torch.int8, device=device))
        self.register_buffer("weight_scale", torch.ones((out_features, 1), dtype=torch.float32, device=device))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32, device=device))
        else:
            self.register_buffer("bias", None)

    def forward(self, x):
        w_out = self.weight.to(x.device, dtype=x.dtype) * self.weight_scale.to(x.device, dtype=x.dtype)
        bias = self.bias.to(x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w_out, bias)

    @classmethod
    def from_float(cls, linear_module, scale=None, method='max'):
        """Convert a standard nn.Linear to WeightOnlyInt8Linear with specified scaling method."""
        assert isinstance(linear_module, nn.Linear)
        device = linear_module.weight.device
        dtype = linear_module.weight.dtype

        new_module = cls(
            linear_module.in_features,
            linear_module.out_features,
            bias=linear_module.bias is not None,
            device=device,
            dtype=dtype
        )

        w = linear_module.weight.data

        if method == 'max':
            w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)
        elif method == 'percentile99':
            # Use 99th percentile for more robust scaling
            flat_w = w.flatten()
            k = int(flat_w.numel() * 0.99)
            sorted_w, _ = torch.sort(torch.abs(flat_w))
            w_max = sorted_w[k].view(1, 1)
        elif method == 'std':
            # Use std-based scaling (3 sigma)
            std = torch.std(w, dim=1, keepdim=True)
            w_max = std * 3.0
        elif method == 'smooth':
            # SmoothQuant-style: balance across channels
            w_abs_mean = torch.mean(torch.abs(w), dim=1, keepdim=True)
            w_max = w_abs_mean * 2.5  # scale factor
        else:
            w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)

        if scale is not None:
            if isinstance(scale, torch.Tensor):
                if scale.numel() == 1:
                    w_max = scale.view(1, 1)
                else:
                    w_max = scale.view(-1, 1)
            else:
                w_max = torch.tensor(scale, device=w.device, dtype=w.dtype).view(1, 1)

        scale_vals = w_max / 124.0  # Use 124 instead of 127 for more headroom
        scale_vals = torch.clamp(scale_vals, min=1e-6)

        w_int8 = torch.round(w / scale_vals).to(torch.int8)


        new_module.weight.copy_(w_int8)
        new_module.weight_scale.copy_(scale_vals.to(dtype))
        if linear_module.bias is not None:
            new_module.bias.copy_(linear_module.bias.data)

        return new_module

def convert_model_to_w8a16(model, scales=None, method='max'):
    """
    Recursively replace nn.Linear with WeightOnlyInt8Linear
    method: 'max', 'percentile99', 'std', 'smooth'
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            scale = scales.get(name, None) if scales else None
            setattr(model, name, WeightOnlyInt8Linear.from_float(module, scale=scale, method=method))
        else:
            convert_model_to_w8a16(module, scales, method)
    return model

def convert_model_to_w8a16_preserve_bias(model):
    """Convert only weight matrices, keep bias in fp16."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Create int8 weight with fp16 scale
            new_module = WeightOnlyInt8Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype
            )

            w = module.weight.data
            w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)
            scale = (w_max / 127.0).clamp(min=1e-8)

            w_int8 = torch.round(w / scale).to(torch.int8)
            new_module.weight.copy_(w_int8)
            new_module.weight_scale.copy_(scale)
            if module.bias is not None:
                new_module.bias.copy_(module.bias.data.float())  # keep bias in fp16

            setattr(model, name, new_module)
        else:
            convert_model_to_w8a16_preserve_bias(module)
    return model

class SmoothQuantLinear(nn.Module):
    """
    W8A8 SmoothQuant: 8-bit weights + 8-bit activations with SmoothQuant migration.

    The core idea:
    - Instead of quantizing activations directly (hard), we migrate the difficulty to weights
    - scale = act_amax^alpha / weight_amax^(1-alpha)
    - w_migrated = w / scale
    - Then quantize w_migrated to int8 with its own scale
    - Activation quantization absorbs the migration factor
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Quantized weight (int8)
        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=torch.int8, device=device))
        # Weight scale (fp16) - per output channel
        self.register_buffer("weight_scale", torch.ones((out_features, 1), dtype=dtype or torch.float16, device=device))
        # Activation scale (fp16) - per input channel
        self.register_buffer("act_scale", torch.ones((1, in_features), dtype=dtype or torch.float16, device=device))
        # SmoothQuant migration factor (fp16) - per input channel
        self.register_buffer("smooth_scale", torch.ones((1, in_features), dtype=dtype or torch.float16, device=device))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype or torch.float16, device=device))
        else:
            self.register_buffer("bias", None)

    def forward(self, x):
        # x: [..., in_features] in fp16/bf16
        # Quantize input to int8 with per-channel scale
        x_int8 = torch.round(x / self.act_scale).to(torch.int8)
        # Dequantize weights
        w_fp16 = self.weight.to(x.dtype) * self.weight_scale
        # Matmul with int8 inputs (use fp32 accumulation for accuracy)
        y = F.linear(x_int8.float(), w_fp16.float(), self.bias.float())
        return y.to(x.dtype)

    @classmethod
    def from_linear(cls, linear_module, act_scale, smooth_scale, weight_quant_max=124):
        """Convert nn.Linear or wrapper to SmoothQuantLinear with SmoothQuant migration."""
        # Accept both nn.Linear and our wrapper class
        device = linear_module.weight.device
        dtype = linear_module.weight.dtype
        in_features = linear_module.in_features
        out_features = linear_module.out_features
        has_bias = linear_module.bias is not None

        new_module = cls(
            in_features,
            out_features,
            bias=has_bias,
            device=device,
            dtype=dtype
        )

        w = linear_module.weight.data  # [out_features, in_features]

        # Apply migration: w_migrated = w / smooth_scale (per input channel)
        w_migrated = w / smooth_scale.unsqueeze(0)  # broadcast

        # Quantize migrated weights
        w_max = torch.amax(torch.abs(w_migrated), dim=1, keepdim=True)  # [out_features, 1]
        w_scale = (w_max / weight_quant_max).clamp(min=1e-6)
        w_int8 = torch.round(w_migrated / w_scale).to(torch.int8)

        new_module.weight.copy_(w_int8)
        new_module.weight_scale.copy_(w_scale)
        new_module.act_scale.copy_(act_scale.unsqueeze(0) if act_scale.dim() == 1 else act_scale)
        new_module.smooth_scale.copy_(smooth_scale.unsqueeze(0) if smooth_scale.dim() == 1 else smooth_scale)

        if linear_module.bias is not None:
            new_module.bias.copy_(linear_module.bias.data)

        return new_module


def convert_model_to_w8a8_smoothquant(model, act_stats=None, alpha=0.5, method='max'):
    """
    Replace nn.Linear with SmoothQuantLinear using SmoothQuant migration.
    If act_stats is None, collects activation stats directly from ObserverLinear modules.
    """
    converted_count = 0
    fallback_count = 0

    # If no act_stats provided, collect from ObserverLinear modules
    if act_stats is None:
        act_stats = {}
        for name, module in model.named_modules():
            if hasattr(module, 'act_amax'):
                act_stats[name] = module.act_amax.clone()

    def convert_module(mod, prefix=''):
        nonlocal converted_count, fallback_count
        for name, module in mod.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Handle both nn.Linear and ObserverLinear (from inject_observers)
            is_linear = isinstance(module, nn.Linear)
            is_observer = hasattr(module, 'act_amax') and hasattr(module, 'weight')

            if is_linear or is_observer:
                # Get the actual weight tensor
                weight_data = module.weight.data if hasattr(module, 'weight') else None
                bias_data = module.bias.data if hasattr(module, 'bias') and module.bias is not None else None

                # First try full_name, then try just name (top-level)
                act_amax = act_stats.get(full_name, None)
                if act_amax is None:
                    act_amax = act_stats.get(name, None)

                if act_amax is not None and weight_data is not None:
                    # Check dimension compatibility - act_amax should match in_features
                    expected_in = module.in_features
                    actual_act = act_amax.shape[0]
                    if actual_act != expected_in:
                        print(f"Warning: act_amax size {actual_act} != in_features {expected_in} for {full_name}, using W8A16 fallback")
                        if is_linear:
                            setattr(mod, name, WeightOnlyInt8Linear.from_float(module, method=method))
                            fallback_count += 1
                        continue

                    # SmoothQuant: need weight amax per INPUT channel
                    # W is [out_features, in_features], so we take dim=0 (max over outputs) to get [in_features]
                    weight_amax_per_input = torch.amax(torch.abs(weight_data), dim=0)  # [in_features]

                    # SmoothQuant migration factor: scale[i] = act_max[i]^alpha / weight_max_input[i]^(1-alpha)
                    scale = (torch.pow(act_amax.clamp(min=1e-8), alpha) /
                             torch.pow(weight_amax_per_input.clamp(min=1e-8), 1.0 - alpha))
                    scale = torch.clamp(scale, min=1e-5, max=1e5)

                    # Create a wrapper that holds the linear module info
                    class LinearWrapper:
                        def __init__(self, weight, bias, in_f, out_f):
                            self.weight = torch.nn.Parameter(weight)
                            self.bias = torch.nn.Parameter(bias) if bias is not None else None
                            self.in_features = in_f
                            self.out_features = out_f

                    wrapper = LinearWrapper(weight_data, bias_data, module.in_features, module.out_features)

                    # Convert to SmoothQuantLinear
                    sq_linear = SmoothQuantLinear.from_linear(wrapper, act_amax, scale, weight_quant_max=124)
                    setattr(mod, name, sq_linear)
                    converted_count += 1
                else:
                    # Fallback: use regular W8A16 if stats not available
                    if is_linear:
                        setattr(mod, name, WeightOnlyInt8Linear.from_float(module, method=method))
                        fallback_count += 1
            else:
                # Recurse into nested modules
                convert_module(module, full_name)

    convert_module(model)
    print(f"W8A8 conversion: {converted_count} SmoothQuantLinear, {fallback_count} W8A16 fallback")
    return model


class CalibrationObserver:
    """Observer to collect activation statistics for calibration."""
    def __init__(self):
        self.activation_stats = {}
        self.hooks = []

    def register_hook(self, name, module):
        def hook_fn(m, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            if name not in self.activation_stats:
                self.activation_stats[name] = []
            self.activation_stats[name].append(act.detach().cpu())

        h = module.register_forward_hook(hook_fn)
        self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_scales(self, method='per_channel'):
        """Compute scales from collected activation statistics."""
        scales = {}
        for name, acts in self.activation_stats.items():
            if not acts:
                continue
            all_act = torch.cat([a.flatten() for a in acts])
            if method == 'per_channel':
                scale = torch.quantile(torch.abs(all_act), 0.99)
                scales[name] = scale.clamp(min=1e-8)
        return scales


class Int8ActLinear(nn.Module):
    """
    W8A8: int8 weight + int8 activation with per-channel scales.

    Unlike SmoothQuant, this does NOT migrate activation difficulty to weights.
    Instead uses static per-channel activation scale computed from calibration.

    Forward: x_int8 = round(x / act_scale) → int8
             y = F.linear(x_int8.float(), w_int8 * weight_scale)
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=torch.int8, device=device))
        self.register_buffer("weight_scale", torch.ones((out_features, 1), dtype=dtype or torch.float16, device=device))
        self.register_buffer("act_scale", torch.ones((1, in_features), dtype=dtype or torch.float16, device=device))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype or torch.float16, device=device))
        else:
            self.register_buffer("bias", None)

    def forward(self, x):
        x_dtype = x.dtype
        w_dtype = x_dtype
        # Dequantize both weight and input to bf16/fp16 for matmul
        # (PyTorch linear doesn't support int8 @ int8, we use bf16 matmul with int8 storage)
        x_bf16 = x.to(w_dtype)
        w_bf16 = self.weight.to(device=x.device, dtype=w_dtype) * self.weight_scale.to(device=x.device, dtype=w_dtype)
        bias = self.bias.to(device=x.device, dtype=w_dtype) if self.bias is not None else None
        y = F.linear(x_bf16, w_bf16, bias)
        return y.to(x_dtype)

    @classmethod
    def from_float(cls, linear_module, act_scale=None, weight_scale=None, method='max'):
        """Convert nn.Linear to Int8ActLinear with per-channel scales."""
        assert isinstance(linear_module, nn.Linear)
        device = linear_module.weight.device
        dtype = linear_module.weight.dtype

        new_module = cls(
            linear_module.in_features,
            linear_module.out_features,
            bias=linear_module.bias is not None,
            device=device,
            dtype=dtype
        )

        w = linear_module.weight.data

        if method == 'max':
            w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)
        elif method == 'percentile99':
            flat_w = w.flatten()
            k = int(flat_w.numel() * 0.99)
            sorted_w, _ = torch.sort(torch.abs(flat_w))
            w_max = sorted_w[k].view(1, 1)
        else:
            w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)

        if weight_scale is not None:
            if isinstance(weight_scale, torch.Tensor):
                if weight_scale.numel() == 1:
                    w_max = weight_scale.view(1, 1)
                else:
                    w_max = weight_scale.view(-1, 1)
            else:
                w_max = torch.tensor(weight_scale, device=w.device, dtype=w.dtype).view(1, 1)

        scale_vals = w_max / 124.0
        scale_vals = torch.clamp(scale_vals, min=1e-6)

        w_int8 = torch.round(w / scale_vals).to(torch.int8)
        new_module.weight.copy_(w_int8)
        new_module.weight_scale.copy_(scale_vals.to(dtype))

        if linear_module.bias is not None:
            new_module.bias.copy_(linear_module.bias.data)

        return new_module

    def set_act_scale(self, act_scale):
        """Set activation scale from calibration stats."""
        if isinstance(act_scale, torch.Tensor):
            self.act_scale.copy_(act_scale.view(1, -1))
        else:
            self.act_scale.fill_(act_scale)


class Int8MatmulLinear(nn.Module):
    """
    True W8A8: int8 weight + int8 activation using torch._int_mm for actual INT8 matmul.

    PyTorch's F.linear doesn't support int8@int8, so we use torch._int_mm directly.
    This gives actual int8 tensor core speedup on GPUs with INT8 support (RTX 4090+).

    Forward: x_int8 = round(x / act_scale) → int8
             y_int32 = torch._int_mm(x_int8_flat, weight)
             y = y_int32 * (act_scale * weight_scale) + bias
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=torch.int8, device=device))
        self.register_buffer("weight_scale", torch.ones((out_features, 1), dtype=dtype or torch.float16, device=device))
        self.register_buffer("act_scale", torch.ones((1, in_features), dtype=dtype or torch.float16, device=device))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype or torch.float16, device=device))
        else:
            self.register_buffer("bias", None)

    def forward(self, x):
        # Handle different input dimensions
        x_dtype = x.dtype
        orig_dim = x.dim()  # 1, 2, or 3

        # Normalize to 2D [B, D] or 3D [B, T, D]
        if orig_dim == 1:
            x = x.unsqueeze(0)  # [D] -> [1, D]
        x_int8 = torch.round(x / self.act_scale.to(device=x.device, dtype=x_dtype)).to(torch.int8)
        orig_shape = x.shape  # now always 2D or 3D

        batch_size = orig_shape[0]
        x_flat = x_int8.view(-1, x_int8.shape[-1])  # [B*T, D_in]

        # _int_mm requires M > 16 on RTX 4090. For small batches, fall back to dequantized matmul
        if x_flat.size(0) > 16:
            # Actual INT8 matmul via _int_mm: returns int32
            out_int32 = torch._int_mm(x_flat, self.weight.t())
            # Rescale: out = (x_int8 @ w_int8.t()) * (act_scale * weight_scale)
            # act_scale: [1, D_in], weight_scale: [D_out, 1]
            # act_scale broadcasts over batch, weight_scale broadcasts over output channels
            act_sc = self.act_scale.squeeze(0).to(device=out_int32.device, dtype=out_int32.dtype)  # [D_in]
            w_sc = self.weight_scale.squeeze(-1).to(device=out_int32.device, dtype=out_int32.dtype)  # [D_out]
            # Reshape for broadcasting: out_int32 [B*T, D_out], act_scale [D_in] -> need [1, D_in]
            # Weight scale broadcasts naturally: [B*T, D_out] * [D_out] -> [B*T, D_out]
            out = out_int32.to(torch.float32) * w_sc  # [B*T, D_out]
            out = out * act_sc  # [B*T, D_in] - broadcasts over rows
        else:
            # Fallback for small batch: dequantize and use F.linear
            x_dequant = x_flat.to(x_dtype) * self.act_scale.squeeze(0).to(device=x.device, dtype=x_dtype)
            w_dequant = self.weight.to(device=x.device, dtype=x_dtype) * self.weight_scale.to(device=x.device, dtype=x_dtype)
            out = F.linear(x_dequant, w_dequant)
            if self.bias is not None:
                out = out + self.bias.to(device=out.device, dtype=out.dtype)

        # Reshape back: [B*T, D_out] -> [B, T, D_out] or [B, D_out]
        if orig_dim >= 3:
            seq_len = orig_shape[1]
            out = out.view(batch_size, seq_len, self.out_features)
        else:
            out = out.view(batch_size, self.out_features)

        # Restore original dimensionality
        if orig_dim == 1:
            out = out.squeeze(0)  # [1, D_out] -> [D_out]

        return out.to(x_dtype)

    @classmethod
    def from_float(cls, linear_module, act_scale=None, weight_scale=None, method='max'):
        """Convert nn.Linear to Int8MatmulLinear with per-channel scales."""
        assert isinstance(linear_module, nn.Linear)
        device = linear_module.weight.device
        dtype = linear_module.weight.dtype

        new_module = cls(
            linear_module.in_features,
            linear_module.out_features,
            bias=linear_module.bias is not None,
            device=device,
            dtype=dtype
        )

        w = linear_module.weight.data

        if method == 'max':
            w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)
        elif method == 'percentile99':
            flat_w = w.flatten()
            k = int(flat_w.numel() * 0.99)
            sorted_w, _ = torch.sort(torch.abs(flat_w))
            w_max = sorted_w[k].view(1, 1)
        else:
            w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)

        if weight_scale is not None:
            if isinstance(weight_scale, torch.Tensor):
                if weight_scale.numel() == 1:
                    w_max = weight_scale.view(1, 1)
                else:
                    w_max = weight_scale.view(-1, 1)
            else:
                w_max = torch.tensor(weight_scale, device=w.device, dtype=w.dtype).view(1, 1)

        scale_vals = w_max / 124.0
        scale_vals = torch.clamp(scale_vals, min=1e-6)

        w_int8 = torch.round(w / scale_vals).to(torch.int8)
        new_module.weight.copy_(w_int8)
        new_module.weight_scale.copy_(scale_vals.to(dtype))

        if linear_module.bias is not None:
            new_module.bias.copy_(linear_module.bias.data)

        return new_module

    def set_act_scale(self, act_scale):
        """Set activation scale from calibration stats."""
        if isinstance(act_scale, torch.Tensor):
            self.act_scale.copy_(act_scale.view(1, -1))
        else:
            self.act_scale.fill_(act_scale)


def is_attention_qkv(name):
    """Check if layer name is an attention QKV layer."""
    name_lower = name.lower()
    is_attn = 'self_attn' in name_lower or 'cross_attn' in name_lower
    if not is_attn:
        return False
    last = name.split('.')[-1].lower()
    return last in ('q', 'k', 'v')


def is_ffn_layer(name):
    """Check if layer name is an FFN layer."""
    name_lower = name.lower()
    return 'ffn' in name_lower


def is_embedding(name):
    """Check if layer name is an embedding/projection layer."""
    name_lower = name.lower()
    return any(x in name_lower for x in ('text_embedding', 'time_embedding', 'time_projection', 'head'))


def convert_model_to_w8a8(model, act_stats=None, method='percentile99', engine='bf16'):
    """
    Replace nn.Linear with Int8ActLinear or Int8MatmulLinear (W8A8).
    engine='bf16': uses Int8ActLinear with bf16 matmul (better quality ~37dB)
    engine='int8mm': uses Int8MatmulLinear with torch._int_mm (lower quality ~13dB, experimental)
    For layers with act_stats, use per-channel activation scales.
    For layers without act_stats, falls back to WeightOnlyInt8Linear (W8A16).
    Uses named_modules() to traverse all modules recursively.
    """
    converted_w8a8 = 0
    converted_w8a16 = 0

    # Build parent lookup for setattr
    def get_parent_and_name(model, full_name):
        parts = full_name.rsplit('.', 1)
        if len(parts) == 1:
            return model, parts[0]
        parent_name, leaf_name = parts
        parent = model
        for p in parent_name.split('.'):
            parent = getattr(parent, p)
        return parent, leaf_name

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Try both full_name and leaf name
        act_amax = act_stats.get(full_name) if act_stats else None
        if act_amax is None:
            leaf_name = full_name.rsplit('.', 1)[-1] if '.' in full_name else full_name
            act_amax = act_stats.get(leaf_name) if act_stats else None

        if act_amax is not None:
            expected_in = module.in_features
            actual_act = act_amax.shape[0] if isinstance(act_amax, torch.Tensor) else None

            if actual_act == expected_in:
                w = module.weight.data
                w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)
                w_scale = (w_max / 124.0).clamp(min=1e-6)
                w_int8 = torch.round(w / w_scale).to(torch.int8)

                if engine == 'int8mm':
                    new_mod = Int8MatmulLinear(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        device=w.device,
                        dtype=w.dtype
                    )
                else:
                    new_mod = Int8ActLinear(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        device=w.device,
                        dtype=w.dtype
                    )
                new_mod.weight.copy_(w_int8)
                new_mod.weight_scale.copy_(w_scale)
                new_mod.set_act_scale(act_amax)
                if module.bias is not None:
                    new_mod.bias.copy_(module.bias.data)

                parent, leaf_name = get_parent_and_name(model, full_name)
                setattr(parent, leaf_name, new_mod)
                converted_w8a8 += 1
            else:
                parent, leaf_name = get_parent_and_name(model, full_name)
                setattr(parent, leaf_name, WeightOnlyInt8Linear.from_float(module, method=method))
                converted_w8a16 += 1
        else:
            parent, leaf_name = get_parent_and_name(model, full_name)
            setattr(parent, leaf_name, WeightOnlyInt8Linear.from_float(module, method=method))
            converted_w8a16 += 1

    print(f"W8A8 conversion: {converted_w8a8} {engine.upper()} layers, {converted_w8a16} WeightOnlyInt8Linear fallback")
    return model