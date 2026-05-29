import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from pathlib import Path

class ObserverLinear(nn.Module):
    def __init__(self, linear_module):
        super().__init__()
        self.in_features = linear_module.in_features
        self.out_features = linear_module.out_features
        self.weight = linear_module.weight
        self.bias = linear_module.bias

        # Observers for activation inputs (X)
        self.register_buffer("act_amax", torch.zeros(self.in_features, dtype=torch.float32))
        self.register_buffer("num_batches", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # x is [..., in_features]
        # We want to find the max absolute value per input channel
        x_flat = x.view(-1, x.shape[-1])
        batch_amax = torch.amax(torch.abs(x_flat), dim=0)

        # Simple moving average or max — ensure same device
        self.act_amax = torch.max(self.act_amax.to(x.device), batch_amax.float())
        self.num_batches += 1

        # Ensure weight and bias are on the same device as x
        w = self.weight.to(x.device, dtype=x.dtype)
        b = self.bias.to(x.device) if self.bias is not None else None
        return F.linear(x, w, b)

def inject_observers(model):
    """
    Replace nn.Linear with ObserverLinear for calibration.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, ObserverLinear(module))
        else:
            inject_observers(module)
    return model

def calculate_smoothquant_scales(model, alpha=0.5):
    """
    Calculate the smoothing scales based on observed act_amax.
    """
    scales_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, ObserverLinear):
            act_amax = module.act_amax
            weight_amax = torch.amax(torch.abs(module.weight), dim=0).float()

            # Ensure same device
            act_amax = act_amax.to(weight_amax.device)

            # SmoothQuant formula: s = act_max^alpha / weight_max^(1-alpha)
            scale = torch.pow(act_amax, alpha) / torch.pow(weight_amax, 1.0 - alpha)
            scale = torch.clamp(scale, min=1e-5)
            scales_dict[name] = scale

    return scales_dict

def load_video_frames(video_path, num_frames=8, max_size=256):
    """Load frames from video, resize if needed."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        frames.append(frame)
    cap.release()
    return frames

def collect_activation_stats(model, dataset_path, pipe, num_videos=3, frames_per_video=4):
    """
    Run calibration with ObserverLinear modules to collect activation statistics.
    """
    # Inject observers
    model = inject_observers(model)
    model.cuda()
    model.eval()

    # Initialize cross KV cache with a dummy forward pass if not already done
    try:
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "posi_prompt.pth")
        if os.path.exists(prompt_path):
            model_path = os.path.dirname(prompt_path)
            init_cross_kv_path = os.path.join(model_path, "..", "nodes.py")
        # Try to initialize cross kv
        if hasattr(pipe, 'init_cross_kv'):
            # Do a dummy init
            dummy_ctx = torch.randn(1, 10, 4096, device='cuda')
            pipe.init_cross_kv(context_tensor=dummy_ctx)
    except:
        pass

    # Find calibration videos
    video_dirs = [
        ("SPMCS", "LQ-Video"),
        ("VideoLQ", "LQ-Video"),
        ("RealVSR", "LQ-Video"),
        ("UDM10", "LQ-Video"),
        ("YouHQ40", "LQ-Video"),
    ]

    videos = []
    for dataset, subdir in video_dirs:
        path = Path(dataset_path) / dataset / subdir
        if path.exists():
            for f in sorted(path.glob("*.mkv"))[:1]:
                videos.append(str(f))
            for f in sorted(path.glob("*.mp4"))[:1]:
                videos.append(str(f))
        if len(videos) >= num_videos:
            break

    if not videos:
        print(f"No videos found in {dataset_path}")
        return {}

    print(f"W8A8 Calibration: using {len(videos)} videos for activation stats")

    # Import flashvsr for running inference
    try:
        from nodes import flashvsr
    except ImportError:
        print("Warning: Could not import flashvsr, using manual forward pass")
        return {}

    for video_path in videos[:num_videos]:
        print(f"  Processing: {video_path}")
        frames = load_video_frames(video_path, num_frames=frames_per_video, max_size=128)
        if len(frames) < 2:
            continue

        # Convert to tensor
        frames_np = np.stack(frames).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_np)

        try:
            with torch.no_grad():
                _ = flashvsr(
                    pipe=pipe,
                    frames=frames_tensor,
                    scale=2.0,
                    color_fix=True,
                    color_fix_method="wavelet",
                    tiled_vae=False,
                    tiled_dit=False,
                    tile_size=256,
                    tile_overlap=16,
                    unload_dit=False,
                    sparse_ratio=0.5,
                    kv_ratio=0.5,
                    local_range=128,
                    seed=42,
                    force_offload=False,
                    enable_debug=False,
                    chunk_size=0,
                    resize_factor=1.0,
                    mode="tiny",
                    context_pad=0
                )
        except Exception as e:
            print(f"    Warning: inference failed: {e}")
            continue

    # Collect stats from all ObserverLinear modules (named_modules to catch nested)
    act_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, ObserverLinear):
            act_stats[name] = module.act_amax.clone()

    # Restore all ObserverLinear back to nn.Linear using recursive restore
    restore_observers(model)

    return act_stats


def restore_observers(model):
    """Recursively restore ObserverLinear back to nn.Linear (matching inject_observers depth)."""
    for name, module in model.named_children():
        if isinstance(module, ObserverLinear):
            new_linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
            new_linear.weight = nn.Parameter(module.weight.data.clone())
            if module.bias is not None:
                new_linear.bias = nn.Parameter(module.bias.data.clone())
            setattr(model, name, new_linear)
        else:
            restore_observers(module)
    return model
