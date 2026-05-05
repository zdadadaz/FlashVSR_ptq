import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightOnlyInt8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=torch.int8, device=device))
        self.register_buffer("weight_scale", torch.ones((out_features, 1), dtype=dtype or torch.float16, device=device))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype or torch.float16, device=device))
        else:
            self.register_buffer("bias", None)

    def forward(self, x):
        w_fp16 = self.weight.to(x.dtype) * self.weight_scale
        return F.linear(x, w_fp16, self.bias)

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
        new_module.weight_scale.copy_(scale_vals)
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