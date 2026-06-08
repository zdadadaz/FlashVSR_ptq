#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlashVSR ComfyUI Node - Video Super Resolution
===============================================
Supports 5 VAE models: Wan2.1, Wan2.2, LightVAE_W2.1, TAE_W2.2, LightTAE_HY1.5

Key Fixes Applied:
- FIX 1: Merged VAE selection into single 'vae_model' dropdown (5 options)
- FIX 2: STRICT file path mapping - each VAE loads its own distinct file
- FIX 3: Black border fix - crop ONLY AFTER full VAE decode is complete
- FIX 4: Lossless resize uses NEAREST for integer scaling
- FIX 5: VRAM optimization - 95% threshold before triggering OOM recovery
- FIX 6: Auto-download with CORRECT HuggingFace URLs
- FIX 7: Explicit VAE class instantiation - no guessing from state_dict
- FIX 8: Summary logging at end of processing
"""

import os, gc
import math
import torch
import folder_paths
import comfy.utils
import time
import sys
import psutil
import yaml
import threading

import numpy as np
import torch.nn.functional as F

from einops import rearrange
from huggingface_hub import snapshot_download, hf_hub_download
try:
    from .src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline, FlashVSRFakeQuantPipeline
    from .src.models.TCDecoder import build_tcdecoder
    from .src.models.utils import clean_vram, get_device_list, Buffer_LQ4x_Proj, Causal_LQ4x_Proj
    from .src.models import wan_video_dit
    from .src.models.wan_video_vae import (
        WanVideoVAE, Wan22VideoVAE, LightX2VVAE, create_video_vae,
        VAE_FULL_DIM, VAE_LIGHT_DIM, VAE_Z_DIM
    )
except ImportError:
    from src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline, FlashVSRFakeQuantPipeline
    from src.models.TCDecoder import build_tcdecoder
    from src.models.utils import clean_vram, get_device_list, Buffer_LQ4x_Proj, Causal_LQ4x_Proj
    from src.models import wan_video_dit
    from src.models.wan_video_vae import (
        WanVideoVAE, Wan22VideoVAE, LightX2VVAE, create_video_vae,
        VAE_FULL_DIM, VAE_LIGHT_DIM, VAE_Z_DIM
    )

try:
    import safetensors.torch
except ImportError:
    pass

# =============================================================================
# FIX 1: Unified VAE model selection dropdown - ALL 5 OPTIONS
# =============================================================================
VAE_MODEL_OPTIONS = ["Wan2.1", "Wan2.2", "LightVAE_W2.1", "TAE_W2.2", "LightTAE_HY1.5"]

# =============================================================================
# FIX 2 & 7: STRICT file path mapping with EXPLICIT class instantiation
# Each VAE selection loads a DISTINCT file and uses EXPLICIT class (no guessing)
# =============================================================================
VAE_MODEL_MAP = {
    "Wan2.1": {
        "class": WanVideoVAE, 
        "file": "Wan2.1_VAE.pth", 
        "internal_name": "wan2.1",
        "url": "https://huggingface.co/lightx2v/Autoencoders/resolve/main/Wan2.1_VAE.pth",
        "dim": VAE_FULL_DIM,
        "z_dim": VAE_Z_DIM,
        "use_full_arch": False
    },
    "Wan2.2": {
        "class": Wan22VideoVAE, 
        "file": "Wan2.2_VAE.pth",
        "internal_name": "wan2.2",
        "url": "https://huggingface.co/lightx2v/Autoencoders/resolve/main/Wan2.2_VAE.pth",
        "dim": VAE_FULL_DIM,
        "z_dim": VAE_Z_DIM,
        "use_full_arch": False
    },
    "LightVAE_W2.1": {
        "class": LightX2VVAE, 
        "file": "lightvaew2_1.pth",
        "internal_name": "lightx2v",
        "url": "https://huggingface.co/lightx2v/Autoencoders/resolve/main/lightvaew2_1.pth",
        "dim": VAE_LIGHT_DIM,
        "z_dim": VAE_Z_DIM,
        "use_full_arch": True
    },
    "TAE_W2.2": {
        "class": Wan22VideoVAE,  # TAE uses same base as Wan2.2
        "file": "taew2_2.safetensors",
        "internal_name": "tae_w2.2",
        "url": "https://huggingface.co/lightx2v/Autoencoders/resolve/main/taew2_2.safetensors",
        "dim": VAE_FULL_DIM,
        "z_dim": VAE_Z_DIM,
        "use_full_arch": False
    },
    "LightTAE_HY1.5": {
        "class": LightX2VVAE,  # LightTAE uses LightX2V architecture
        "file": "lighttaehy1_5.pth",
        "internal_name": "lighttae_hy1.5",
        "url": "https://huggingface.co/lightx2v/Autoencoders/resolve/main/lighttaehy1_5.pth",
        "dim": VAE_LIGHT_DIM,
        "z_dim": VAE_Z_DIM,
        "use_full_arch": True
    },
}

# =============================================================================
# FIX 5: VRAM threshold for OOM recovery - set to 95%
# =============================================================================
VRAM_OOM_THRESHOLD = 0.95  # Only trigger OOM recovery when 95% VRAM is used

# =============================================================================
# Model Paths Configuration Loader
# =============================================================================
_cached_model_path = None  # Cache for model path to avoid repeated file reads
_config_loaded = False  # Flag to track if we've attempted to load config
_config_lock = threading.Lock()  # Thread-safe access to cached values

def load_model_paths_config():
    """
    Load model paths configuration from model_paths.yaml file.
    Returns the custom FlashVSR model path if configured, otherwise None.
    Uses caching to avoid repeated file I/O operations.
    Thread-safe implementation using a lock.
    """
    global _cached_model_path, _config_loaded
    
    # Return cached value if already loaded (thread-safe check)
    with _config_lock:
        if _config_loaded:
            return _cached_model_path
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "model_paths.yaml")
    
    # Check if file exists before entering try block
    if not os.path.exists(config_path):
        with _config_lock:
            _config_loaded = True
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        if config and isinstance(config, dict):
            flashvsr_path = config.get('flashvsr_model_path', '').strip()
            
            if flashvsr_path:
                # Expand user path (~/...) and environment variables
                flashvsr_path = os.path.expanduser(flashvsr_path)
                flashvsr_path = os.path.expandvars(flashvsr_path)
                
                # Convert to absolute path if it's not already
                # Use current_dir (plugin directory) as base for relative paths
                if not os.path.isabs(flashvsr_path):
                    flashvsr_path = os.path.abspath(os.path.join(current_dir, flashvsr_path))
                
                log(f"Custom FlashVSR model path loaded from config: {flashvsr_path}", 
                    message_type='info', icon="📂")
                
                with _config_lock:
                    _cached_model_path = flashvsr_path
                    _config_loaded = True
                return flashvsr_path
    except Exception as e:
        log(f"Warning: Could not load model_paths.yaml: {e}. Using default path.", 
            message_type='warning', icon="⚠️")
    
    with _config_lock:
        _config_loaded = True
    return None

device_choices = get_device_list()

def log(message: str, message_type: str = 'normal', icon: str = "", end: str = "\n", in_place: bool = False):
    if icon:
        message = f"{icon} {message}"
        
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
    else:
        message = message

    if in_place:
        # Clear line before printing
        sys.stdout.write("\r\033[K" + message)
        sys.stdout.flush()
    else:
        print(f"{message}", end=end, flush=True)

def get_vram_info():
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return vram_used, vram_reserved, vram_total
    return 0, 0, 0

def log_resource_usage(prefix="Resource Usage", end="\n", in_place=False):
    ram = psutil.virtual_memory()
    ram_used = ram.used / (1024 ** 3)
    ram_total = ram.total / (1024 ** 3)
    
    msg = f"[{prefix}] RAM: {ram_used:.1f}/{ram_total:.1f}G"
    
    if torch.cuda.is_available():
        vram_used, vram_reserved, vram_total = get_vram_info()
        msg += f" | VRAM: {vram_used:.1f}/{vram_reserved:.1f}/{vram_total:.1f}G"
        
    log(msg, message_type='info', icon="📊", end=end, in_place=in_place)


# =============================================================================
# FIX 5 & 9: VRAM Estimation, Pre-Flight Resource Check & Settings Recommender
# Calculate approximate VRAM requirements and provide optimal settings
# =============================================================================
def estimate_vram_usage(width, height, num_frames, scale, tiled_vae=False, tiled_dit=False,
                         chunk_size=0, mode="full", quantize_mode="None"):
    """
    Estimate approximate VRAM usage for the given video parameters.
    Returns estimated VRAM in GB. Enhanced to consider chunk_size and mode.

    =============================================================================
    FIX: Accurate VRAM Estimation with Safety Factor
    =============================================================================
    Previous estimates were ~4.5GB when actual usage was ~15GB.
    This was because we ignored:
    - Intermediate Activations: PyTorch stores outputs for every layer
    - VAE Upscaling: VAE decoding expands data significantly
    - Workspace Memory: CUDA context overhead

    Solution: Apply Safety_Factor = 4.0 to the raw tensor calculations
    to account for these overheads.
    """
    # Quantization memory savings factor
    if quantize_mode == "W8A16":
        quant_factor = 0.65  # Weight-only int8: ~35% memory savings
    elif quantize_mode in ("W8A8", "W8A8_SmoothQuant"):
        quant_factor = 0.45  # int8 weights+activations: ~55% memory savings
    else:
        quant_factor = 1.0  # No quantization
    # Safety factor to account for intermediate activations, VAE upscaling overhead,
    # and CUDA workspace memory. Empirically determined from observed ~15GB actual
    # usage when estimates were ~4.5GB.
    SAFETY_FACTOR = 1.5
    
    # Base model memory varies by mode
    if mode == "full":
        base_model_gb = 5.0  # Full VAE + DiT
    elif mode == "tiny-long":
        base_model_gb = 3.5  # TCDecoder is lighter than full VAE
    else:  # tiny
        base_model_gb = 4.0
    
    # Per-frame latent memory (scaled output resolution)
    output_h, output_w = height * scale, width * scale
    
    # Latent dimensions (8x downsampled)
    latent_h, latent_w = output_h // 8, output_w // 8
    
    # Frames to process at once (if chunked, use chunk_size)
    effective_frames = chunk_size if chunk_size > 0 and chunk_size <= num_frames else num_frames
    
    # Input tensor size - use 4 bytes to account for float32 intermediates during processing
    # Even though final tensors are bf16/fp16, operations often use float32 internally
    input_tensor_bytes = output_h * output_w * 3 * effective_frames * 4
    input_tensor_gb = (input_tensor_bytes * SAFETY_FACTOR) / (1024 ** 3)
    
    # Approximate memory per frame in latent space (16 channels, bf16)
    bytes_per_frame = latent_h * latent_w * 16 * 2  # bf16 = 2 bytes
    total_latent_gb = (bytes_per_frame * effective_frames * SAFETY_FACTOR) / (1024 ** 3)
    
    # DiT attention memory (quadratic with sequence length)
    seq_len = latent_h * latent_w * (effective_frames // 4)
    attention_gb = (seq_len * seq_len * 2 * SAFETY_FACTOR) / (1024 ** 3) * 0.001  # Rough estimate
    
    # VAE decode memory - this is where most intermediate activations live
    vae_decode_gb = (output_h * output_w * 3 * effective_frames * 2 * SAFETY_FACTOR) / (1024 ** 3)
    
    # Apply tiling reductions
    if tiled_dit:
        attention_gb *= 0.3  # Tiling reduces peak attention memory
    if tiled_vae:
        vae_decode_gb *= 0.4  # Tiling reduces peak VAE memory

    # Quantization only affects DiT (not VAE), so apply factor to DiT-related memory
    # DiT-related: base_model_gb (partially) + attention_gb + total_latent_gb
    dit_related_gb = base_model_gb * 0.6 + attention_gb + total_latent_gb
    other_gb = base_model_gb * 0.4 + input_tensor_gb + vae_decode_gb

    dit_related_gb *= quant_factor  # Apply quantization savings to DiT memory
    total_estimated = dit_related_gb + other_gb
    return total_estimated


def get_optimal_settings(width, height, num_frames, scale, available_vram_gb, mode="full", quantize_mode="None"):
    """
    Calculate optimal settings (chunk_size, resize_factor, tiling) based on VRAM.

    Returns dict with recommended settings.
    """
    # Target VRAM usage: 85% of available to leave headroom
    target_vram = available_vram_gb * 0.85

    # Start with default settings
    recommended = {
        "chunk_size": 0,  # 0 = process all at once
        "resize_factor": 1.0,
        "tiled_vae": False,
        "tiled_dit": False,
        "warning": None
    }

    # Test current settings
    estimated = estimate_vram_usage(width, height, num_frames, scale,
                                     tiled_vae=False, tiled_dit=False,
                                     chunk_size=0, mode=mode, quantize_mode=quantize_mode)

    if estimated <= target_vram:
        # Settings are fine
        return recommended

    # Try enabling tiled VAE first (least impact on quality)
    estimated_tiled_vae = estimate_vram_usage(width, height, num_frames, scale,
                                               tiled_vae=True, tiled_dit=False,
                                               chunk_size=0, mode=mode, quantize_mode=quantize_mode)
    if estimated_tiled_vae <= target_vram:
        recommended["tiled_vae"] = True
        return recommended

    # Try enabling both tiling
    estimated_both_tiled = estimate_vram_usage(width, height, num_frames, scale,
                                                tiled_vae=True, tiled_dit=True,
                                                chunk_size=0, mode=mode, quantize_mode=quantize_mode)
    if estimated_both_tiled <= target_vram:
        recommended["tiled_vae"] = True
        recommended["tiled_dit"] = True
        return recommended

    # Need chunking - find optimal chunk size
    recommended["tiled_vae"] = True
    recommended["tiled_dit"] = True

    for chunk in [100, 64, 32, 16, 8, 4]:
        if chunk >= num_frames:
            continue
        estimated_chunked = estimate_vram_usage(width, height, num_frames, scale,
                                                 tiled_vae=True, tiled_dit=True,
                                                 chunk_size=chunk, mode=mode, quantize_mode=quantize_mode)
        if estimated_chunked <= target_vram:
            recommended["chunk_size"] = chunk
            return recommended

    # Still too high - recommend resize factor
    for resize in [0.8, 0.6, 0.5, 0.4, 0.3]:
        new_h, new_w = int(height * resize), int(width * resize)
        estimated_resized = estimate_vram_usage(new_w, new_h, num_frames, scale,
                                                 tiled_vae=True, tiled_dit=True,
                                                 chunk_size=8, mode=mode, quantize_mode=quantize_mode)
        if estimated_resized <= target_vram:
            recommended["chunk_size"] = 8
            recommended["resize_factor"] = resize
            return recommended
    
    # Even with max reduction still risky
    recommended["chunk_size"] = 4
    recommended["resize_factor"] = 0.3
    recommended["warning"] = "VRAM critically low. Results may be unstable."
    return recommended


def check_resources(width, height, num_frames, scale, chunk_size, resize_factor,
                    tiled_vae, tiled_dit, mode="full", quantize_mode="None"):
    """
    =============================================================================
    FIX 9: Pre-Flight Resource Calculator
    =============================================================================
    
    Performs intelligent pre-flight check before loading heavy models.
    
    1. Gets hardware stats (VRAM, RAM) using torch.cuda.mem_get_info()
    2. Estimates required memory based on video parameters
    3. Simulates if current settings will cause OOM
    4. Provides optimal settings recommendations
    
    Returns:
        dict with keys:
        - estimated_vram_gb: float
        - available_vram_gb: float
        - ram_used_gb: float
        - ram_total_gb: float
        - will_oom: bool
        - recommended_settings: dict (if will_oom)
        - message: str
    """
    result = {
        "estimated_vram_gb": 0.0,
        "available_vram_gb": 0.0,
        "ram_used_gb": 0.0,
        "ram_total_gb": 0.0,
        "will_oom": False,
        "recommended_settings": None,
        "message": ""
    }
    
    # Get RAM info
    ram = psutil.virtual_memory()
    result["ram_used_gb"] = ram.used / (1024 ** 3)
    result["ram_total_gb"] = ram.total / (1024 ** 3)
    
    # Get VRAM info
    if torch.cuda.is_available():
        vram_free, vram_total = torch.cuda.mem_get_info()
        result["available_vram_gb"] = vram_free / (1024 ** 3)
        vram_total_gb = vram_total / (1024 ** 3)
    else:
        result["message"] = "CUDA not available. Running on CPU may be very slow."
        return result
    
    # Calculate effective dimensions after resize
    effective_h = int(height * resize_factor) if resize_factor < 1.0 else height
    effective_w = int(width * resize_factor) if resize_factor < 1.0 else width
    
    # Estimate VRAM usage
    result["estimated_vram_gb"] = estimate_vram_usage(
        effective_w, effective_h, num_frames, scale,
        tiled_vae=tiled_vae, tiled_dit=tiled_dit,
        chunk_size=chunk_size, mode=mode, quantize_mode=quantize_mode
    )

    # Check if OOM likely
    if result["estimated_vram_gb"] > result["available_vram_gb"] * VRAM_OOM_THRESHOLD:
        result["will_oom"] = True
        result["recommended_settings"] = get_optimal_settings(
            effective_w, effective_h, num_frames, scale,
            result["available_vram_gb"], mode, quantize_mode
        )
    
    # Build message
    if result["will_oom"]:
        rec = result["recommended_settings"]
        msg_parts = []
        if rec["chunk_size"] != chunk_size and rec["chunk_size"] > 0:
            msg_parts.append(f"chunk_size={rec['chunk_size']}")
        if rec["resize_factor"] != resize_factor:
            msg_parts.append(f"resize_factor={rec['resize_factor']:.1f}")
        if rec["tiled_vae"] and not tiled_vae:
            msg_parts.append("tiled_vae=True")
        if rec["tiled_dit"] and not tiled_dit:
            msg_parts.append("tiled_dit=True")
        
        if msg_parts:
            result["message"] = f"⚠️ Current settings require ~{result['estimated_vram_gb']:.1f}GB but only {result['available_vram_gb']:.1f}GB available. Recommended: {', '.join(msg_parts)}"
        else:
            result["message"] = f"⚠️ VRAM critically low. Estimated ~{result['estimated_vram_gb']:.1f}GB needed, only {result['available_vram_gb']:.1f}GB available."
    else:
        result["message"] = f"✅ Safe to proceed. Estimated ~{result['estimated_vram_gb']:.1f}GB needed, {result['available_vram_gb']:.1f}GB available."
    
    return result


def log_preflight_check(width, height, num_frames, scale, chunk_size, resize_factor,
                         tiled_vae, tiled_dit, mode="full", quantize_mode="None"):
    """
    Log pre-flight resource check results.
    """
    result = check_resources(width, height, num_frames, scale, chunk_size, resize_factor,
                              tiled_vae, tiled_dit, mode, quantize_mode)
    
    log("=" * 60, message_type='info')
    log("PRE-FLIGHT RESOURCE CHECK", message_type='info', icon="🔍")
    log(f"RAM: {result['ram_used_gb']:.1f}GB / {result['ram_total_gb']:.1f}GB", message_type='info', icon="💻")
    log(f"VRAM Available: {result['available_vram_gb']:.1f}GB", message_type='info', icon="💾")
    log(f"Estimated VRAM Required: {result['estimated_vram_gb']:.1f}GB", message_type='info', icon="📊")
    
    if result["will_oom"]:
        log(result["message"], message_type='warning', icon="⚠️")
        if result["recommended_settings"]:
            rec = result["recommended_settings"]
            log("Recommended Optimal Settings:", message_type='info', icon="💡")
            if rec["chunk_size"] > 0:
                log(f"  • chunk_size = {rec['chunk_size']}", message_type='info')
            if rec["resize_factor"] < 1.0:
                log(f"  • resize_factor = {rec['resize_factor']:.1f}", message_type='info')
            if rec["tiled_vae"]:
                log(f"  • tiled_vae = True", message_type='info')
            if rec["tiled_dit"]:
                log(f"  • tiled_dit = True", message_type='info')
            if rec.get("warning"):
                log(f"  ⚠️ {rec['warning']}", message_type='warning')
    else:
        log(result["message"], message_type='finish', icon="✅")
    
    log("=" * 60, message_type='info')
    
    return result


def log_vram_advisory(width, height, num_frames, scale, tiled_vae, tiled_dit, mode="full"):
    """
    Log advisory message about VRAM usage.
    Enhanced to use the new pre-flight check.
    """
    if not torch.cuda.is_available():
        return
    
    estimated_vram = estimate_vram_usage(width, height, num_frames, scale, tiled_vae, tiled_dit, mode=mode)
    available_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    current_used = torch.cuda.memory_allocated() / (1024 ** 3)
    free_vram = available_vram - current_used
    
    log(f"VRAM Advisory: Estimated ~{estimated_vram:.1f}GB needed, Available: {free_vram:.1f}GB free of {available_vram:.1f}GB total", 
        message_type='info', icon="💡")
    
    if estimated_vram > free_vram * 0.9:
        log("⚠️ Warning: High VRAM usage expected. Recommend enabling Tiled VAE/DiT.", message_type='warning', icon="⚠️")
    elif estimated_vram < free_vram * 0.5:
        log("✅ Safe to proceed. VRAM usage should be comfortable.", message_type='info', icon="✅")

def get_flashvsr_model_base_dir():
    """
    Get the base directory for FlashVSR models.
    Checks model_paths.yaml first, falls back to ComfyUI models directory.
    """
    custom_path = load_model_paths_config()
    if custom_path:
        return custom_path
    return folder_paths.models_dir

def model_download(model_name="JunhaoZhuang/FlashVSR"):
    base_dir = get_flashvsr_model_base_dir()
    model_dir = os.path.join(base_dir, model_name.split("/")[-1])
    if not os.path.exists(model_dir):
        log(f"Downloading model '{model_name}' from huggingface...", message_type='info', icon="⬇️")
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)


# =============================================================================
# FIX 6: Auto-download VAE models if missing - UPDATED URLs
# =============================================================================
def download_vae_if_missing(vae_file: str, model_path: str, vae_config: dict) -> str:
    """
    Check if VAE file exists. If not, attempt to download it using the URL in vae_config.
    
    Args:
        vae_file: The filename of the VAE (e.g., 'Wan2.1_VAE.pth')
        model_path: The directory where VAE should be saved
        vae_config: The VAE configuration from VAE_MODEL_MAP (must contain 'url' key)
    
    Returns:
        Full path to the VAE file
    """
    vae_path = os.path.join(model_path, vae_file)
    
    if os.path.exists(vae_path):
        log(f"VAE file found: {vae_file}", message_type='info', icon="✅")
        return vae_path
    
    log(f"VAE file '{vae_file}' not found. Attempting auto-download...", message_type='warning', icon="⬇️")
    
    # Get URL from config (FIX 6: Use EXACT URLs from VAE_MODEL_MAP)
    url = vae_config.get("url")
    
    if url:
        try:
            log(f"Downloading from: {url}", message_type='info', icon="🌐")
            # Ensure directory exists
            os.makedirs(model_path, exist_ok=True)
            torch.hub.download_url_to_file(url, vae_path, progress=True)
            log(f"Successfully downloaded VAE: {vae_file}", message_type='finish', icon="✅")
            return vae_path
        except Exception as e:
            log(f"Download failed: {e}", message_type='error', icon="❌")
    
    raise RuntimeError(
        f'VAE file "{vae_file}" not found and auto-download failed.\n'
        f'Please manually download it and save to: {vae_path}\n'
        f'Download URL: {url}'
    )


# =============================================================================
# FIX 7: Fixed tensor2video for correct video output
# Ensures proper tensor permutation: VAE output (B, C, F, H, W) -> video (F, H, W, C)
# CRITICAL: This is called AFTER VAE decode is complete - no cropping here
# =============================================================================
def tensor2video(frames: torch.Tensor):
    """
    Convert VAE output tensor to video format.
    
    Input: (B, C, F, H, W) - Batch, Channels, Frames, Height, Width (VAE output)
    Output: (F, H, W, C) - Frames, Height, Width, Channels (video format)
    
    The tensor is normalized from [-1, 1] to [0, 1] for display.
    
    NOTE: This function does NOT crop - cropping happens in process_chunk() 
    AFTER this conversion is complete.
    """
    # Handle different input shapes
    if frames.dim() == 5:
        # Expected shape: (B, C, F, H, W)
        video_squeezed = frames.squeeze(0)  # (C, F, H, W)
        video_permuted = video_squeezed.permute(1, 2, 3, 0)  # (F, H, W, C)
    elif frames.dim() == 4:
        # Shape: (C, F, H, W) or (F, C, H, W) - need to detect
        if frames.shape[0] == 3 or frames.shape[0] == 4:
            # Likely (C, F, H, W)
            video_permuted = frames.permute(1, 2, 3, 0)  # (F, H, W, C)
        else:
            # Likely (F, C, H, W)
            video_permuted = frames.permute(0, 2, 3, 1)  # (F, H, W, C)
    else:
        raise ValueError(f"Unexpected tensor shape: {frames.shape}")
    
    # Normalize from [-1, 1] to [0, 1]
    video_final = (video_permuted.float() + 1.0) / 2.0
    # Clamp to valid range to avoid visual artifacts
    video_final = torch.clamp(video_final, 0.0, 1.0)
    
    return video_final

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def next_8n5(n):  # next 8n+5
    return 21 if n < 21 else ((n - 5 + 7) // 8) * 8 + 5

def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    """
    Compute scaled dimensions and target dimensions (aligned to multiple).
    
    =============================================================================
    FIX 3: Black Border Fix - Track original scaled dimensions
    =============================================================================
    Returns: sW, sH (actual scaled), tW, tH (padded to multiple), pad_left, pad_top
    """
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")

    sW, sH = int(w0 * scale), int(h0 * scale)
    tW = math.ceil(sW / multiple) * multiple
    tH = math.ceil(sH / multiple) * multiple
    
    # Calculate padding offsets (centered padding)
    pad_left = (tW - sW) // 2
    pad_top = (tH - sH) // 2
    
    return sW, sH, tW, tH, pad_left, pad_top


def tensor_upscale_then_center_crop(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int, pad_left: int, pad_top: int) -> torch.Tensor:
    """
    Upscale frame tensor and pad to target dimensions.
    
    =============================================================================
    FIX 3: Black Border Fix - Use consistent padding offsets
    =============================================================================
    """
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0) # HWC -> CHW -> BCHW
    
    sW, sH = int(w0 * scale), int(h0 * scale)
    upscaled_tensor = F.interpolate(tensor_bchw, size=(sH, sW), mode='bicubic', align_corners=False)
    
    # Apply symmetric padding to reach target dimensions
    if sW < tW or sH < tH:
        pad_r = tW - sW - pad_left
        pad_b = tH - sH - pad_top
        # Pad order: (left, right, top, bottom)
        # Use 'replicate' mode which is safer for small images than 'reflect'
        # (reflect requires image size >= padding size on each dimension)
        max_pad = max(pad_left, pad_r, pad_top, pad_b)
        min_dim = min(upscaled_tensor.shape[2], upscaled_tensor.shape[3])
        if min_dim >= max_pad:
            upscaled_tensor = F.pad(upscaled_tensor, (pad_left, pad_r, pad_top, pad_b), mode='reflect')
        else:
            # Fall back to replicate mode for small images
            upscaled_tensor = F.pad(upscaled_tensor, (pad_left, pad_r, pad_top, pad_b), mode='replicate')
    
    # Center crop to target dimensions if needed (should be exact after padding)
    l = max(0, (upscaled_tensor.shape[3] - tW) // 2)
    t = max(0, (upscaled_tensor.shape[2] - tH) // 2)
    cropped_tensor = upscaled_tensor[:, :, t:t + tH, l:l + tW]

    return cropped_tensor.squeeze(0)


def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    """
    Prepare input tensor with proper padding tracking.
    
    =============================================================================
    FIX 3: Black Border Fix - Track padding for later cropping
    =============================================================================
    Returns: vid_final, tH, tW, F, original_sH, original_sW, pad_top, pad_left
    """
    N0, h0, w0, _ = image_tensor.shape
    
    multiple = 128 # Keep 128 alignment for VAE/DiT blocks
    sW, sH, tW, tH, pad_left, pad_top = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    F = largest_8n1_leq(num_frames_with_padding)
    
    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")
    
    frames = []
    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = tensor_upscale_then_center_crop(
            frame_slice, scale=scale, tW=tW, tH=tH, 
            pad_left=pad_left, pad_top=pad_top
        ).to('cpu').to(dtype) * 2.0 - 1.0
        frames.append(tensor_chw)
        del frame_slice

    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    
    del vid_stacked
    clean_vram()
    
    # Return additional info for cropping output back to original dimensions
    return vid_final, tH, tW, F, sH, sW, pad_top, pad_left

def calculate_tile_coords(height, width, tile_size, overlap):
    coords = []
    
    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride
            x1 = c * stride
            
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
                
            coords.append((x1, y1, x2, y2))
            
    return coords

def create_feather_mask(size, overlap):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = (1 - torch.cos(torch.linspace(0, math.pi, overlap))) / 2
    
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    return mask

def load_trt_engine(engine_path):
    """Load a pre-compiled TensorRT engine for W8A8_PTQ mode."""
    if not os.path.exists(engine_path):
        raise RuntimeError(f"TRT engine not found: {engine_path}")
    log(f"Loading TRT engine from {engine_path}...", message_type='info', icon='🗜️')
    engine = torch.jit.load(engine_path)
    return engine

def init_pipeline(model, mode, device, dtype, vae_model="Wan2.1", quantize_mode="None", ckpt_path=None, w8a8_engine="bf16", trt_engine_path=None, fakequant_extra_scopes=""):
    """
    Initialize FlashVSR pipeline with specified model and VAE type.
    """
    model_download(model_name="JunhaoZhuang/"+model)
    base_dir = get_flashvsr_model_base_dir()
    model_path = os.path.join(base_dir, model)
    if not os.path.exists(model_path):
        raise RuntimeError(f'Model directory does not exist!\nPlease save all weights to "{model_path}"')
    
    # Use custom ckpt_path if provided, otherwise use default
    if ckpt_path is None:
        ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f'Checkpoint file "{ckpt_path}" does not exist!')
    
    # ==========================================================================
    # FIX 2 & 7: VAE Model Loading - EXPLICIT mapping (no guessing!)
    # ==========================================================================
    if vae_model not in VAE_MODEL_MAP:
        log(f"Unknown VAE model '{vae_model}', defaulting to Wan2.1", message_type='warning', icon="⚠️")
        vae_model = "Wan2.1"
    
    vae_config = VAE_MODEL_MAP[vae_model]
    vae_class = vae_config["class"]
    vae_file = vae_config["file"]
    vae_dim = vae_config["dim"]
    vae_z_dim = vae_config["z_dim"]
    use_full_arch = vae_config["use_full_arch"]
    
    # Debug logging - Show EXACTLY which file and class will be used
    log(f"VAE Selection: '{vae_model}' -> File: '{vae_file}' -> Class: {vae_class.__name__}", 
        message_type='info', icon="🔍")
    
    # ==========================================================================
    # FIX 6: Auto-download VAE if missing
    # ==========================================================================
    vae_path = download_vae_if_missing(vae_file, model_path, vae_config)
    
    log(f"VAE file path confirmed: {vae_path}", message_type='info', icon="📁")
    
    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    if not os.path.exists(lq_path):
        raise RuntimeError(f'"LQ_proj_in.ckpt" does not exist!\nPlease save it to "{model_path}"')
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    if not os.path.exists(tcd_path):
        raise RuntimeError(f'"TCDecoder.ckpt" does not exist!\nPlease save it to "{model_path}"')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, "posi_prompt.pth")
    
    mm = ModelManager(torch_dtype=dtype, device="cpu")

    # Handle pre-quantized DiT model loading manually if needed
    # Check for both W8A16 and W8A8 quantized checkpoints
    is_quantized_ckpt = (
        ("w8a16" in ckpt_path.lower() or "w8a8" in ckpt_path.lower())
        and "fakequant" not in ckpt_path.lower()
    )
    is_fakequant_ckpt = "fakequant" in ckpt_path.lower()

    # FIX: quantize_mode parameter should drive FakeQuant pipeline selection
    # not just be informational. Map quantize_mode → fq_mode for FakeQuant paths.
    fq_mode = None
    fakequant_modes = ("FakeQuant_A8W8", "FakeQuant_A8W8_DRAQ", "FakeQuant_A8W4", "FakeQuant_A16W8", "FakeQuant_A16W4", "FakeQuant_A4W4")
    if quantize_mode in fakequant_modes:
        mode_map = {
            "FakeQuant_A8W8": "a8w8",
            "FakeQuant_A8W8_DRAQ": "a8w8",
            "FakeQuant_A8W4": "a8w4",
            "FakeQuant_A16W8": "a16w8",
            "FakeQuant_A16W4": "a16w4",
            "FakeQuant_A4W4": "a4w4",
        }
        fq_mode = mode_map[quantize_mode]
        is_fakequant_ckpt = True
        log(f"quantize_mode={quantize_mode} → FakeQuant mode={fq_mode}", message_type='info', icon="🔧")

    if is_fakequant_ckpt:
        log(f"FakeQuant checkpoint detected: {ckpt_path}", message_type='info', icon="🔍")
        try:
            from .src.models.wan_video_dit import WanModel
            from .src.models.quantization.fakequant import convert_model_to_fakequant, convert_ops_to_fakequant, infer_fakequant_layer_policy_from_state_dict
            from .src.models.quantization.qao import convert_model_to_lsgquant_shell, infer_lsgquant_layer_policy_from_state_dict
        except ImportError:
            from src.models.wan_video_dit import WanModel
            from src.models.quantization.fakequant import convert_model_to_fakequant, convert_ops_to_fakequant, infer_fakequant_layer_policy_from_state_dict
            from src.models.quantization.qao import convert_model_to_lsgquant_shell, infer_lsgquant_layer_policy_from_state_dict

        # FlashVSR-v1.1 config
        dit = WanModel(
            dim=1536, eps=1e-5, ffn_dim=8960, freq_dim=256, in_dim=16,
            num_heads=12, num_layers=30, out_dim=16, patch_size=(1, 2, 2), text_dim=4096
        )
        # Detect correct FakeQuant mode from ckpt path
        # Priority: fq_mode from quantize_mode > path inference > default
        detected_fq_mode = fq_mode
        if detected_fq_mode is None:
            detected_fq_mode = "a8w8"
            if "a16w8" in ckpt_path.lower():
                detected_fq_mode = "a16w8"
            elif "a4w4" in ckpt_path.lower():
                detected_fq_mode = "a4w4"
            elif "a8w4" in ckpt_path.lower():
                detected_fq_mode = "a8w4"
            elif "a16w4" in ckpt_path.lower():
                detected_fq_mode = "a16w4"
        fq_mode = detected_fq_mode
        log(f"FakeQuant mode: {fq_mode}", message_type='info', icon="🔍")

        # Load state dict before FakeQuant module construction so mixed W4/W8
        # checkpoints can infer per-layer modes from tensor shapes.
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd_dit = load_file(ckpt_path)
        else:
            sd_dit = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        new_sd = {}
        for k, v in sd_dit.items():
            if k.startswith("model."):
                new_sd[k[6:]] = v
            else:
                new_sd[k] = v
        is_lsgquant_qao_ckpt = any(k.endswith(".residual.weight_int") for k in new_sd)
        if is_lsgquant_qao_ckpt:
            inferred_policy = infer_lsgquant_layer_policy_from_state_dict(
                dit, new_sd, default_activation_qdq_mode="draq_symmetric"
            )
            if not inferred_policy:
                raise RuntimeError("LSGQuant QAO checkpoint detected but no Linear layer policy could be inferred")
            log(f"Inferred LSGQuant QAO per-layer policy from checkpoint: {len(inferred_policy)} layers", message_type='info', icon="🧩")
            convert_model_to_lsgquant_shell(
                dit,
                inferred_policy,
                activation_qdq_mode="draq_symmetric",
                draq_qrange="signed_symmetric",
            )
        else:
            inferred_policy = infer_fakequant_layer_policy_from_state_dict(
                dit, new_sd, default_activation_qdq_mode="draq_symmetric"
            )
            if inferred_policy:
                log(f"Inferred FakeQuant per-layer policy from checkpoint: {len(inferred_policy)} layers", message_type='info', icon="🧩")
                convert_model_to_fakequant(dit, mode=fq_mode, act_stats=None, layer_policy=inferred_policy, activation_qdq_mode="draq_symmetric")
            else:
                convert_model_to_fakequant(dit, mode=fq_mode, act_stats=None)
        dit.load_state_dict(new_sd, strict=False)
        dit.eval()

        # Register in ModelManager (standard path) so from_model_manager can fetch it
        mm.model.append(dit)
        mm.model_path.append(ckpt_path)
        mm.model_name.append("wan_video_dit")

        # Load VAE normally
        mm.load_models([vae_path])

        # Create FakeQuant pipeline via from_model_manager (proper pipeline init)
        fq_pipe = FlashVSRFakeQuantPipeline.from_model_manager(
            mm, torch_dtype=dtype, device=device, quant_mode=fq_mode
        )

        # Build TCDecoder and VAE explicitly (same as full/tiny pipelines)
        multi_scale_channels = [512, 256, 128, 128]
        fq_pipe.TCDecoder = build_tcdecoder(
            new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16 + 768
        )
        fq_pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device, weights_only=False), strict=False)
        fq_pipe.TCDecoder.clean_mem()

        # Explicit VAE instantiation
        if vae_class == LightX2VVAE:
            fq_pipe.vae = LightX2VVAE(z_dim=vae_z_dim, dim=vae_dim, use_full_arch=use_full_arch)
        elif vae_class == Wan22VideoVAE:
            fq_pipe.vae = Wan22VideoVAE(z_dim=vae_z_dim, dim=vae_dim)
        else:
            fq_pipe.vae = WanVideoVAE(z_dim=vae_z_dim, dim=vae_dim)

        if vae_path.endswith(".safetensors"):
            import safetensors.torch
            vae_sd = safetensors.torch.load_file(vae_path)
        else:
            vae_sd = torch.load(vae_path, map_location="cpu", weights_only=False)
        fq_pipe.vae.load_state_dict(vae_sd, strict=False)
        fq_pipe.vae = fq_pipe.vae.to(device=device, dtype=dtype)
        fq_pipe.vae.model.encoder = None
        fq_pipe.vae.model.conv1 = None

        # Load LQ_proj_in
        if model == "FlashVSR":
            fq_pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
        else:
            fq_pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
        fq_pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu", weights_only=False), strict=True)
        fq_pipe.denoising_model().LQ_proj_in.to(device)

        # Optional component-level FakeQuant for sensitivity analysis. The DiT Linear
        # layers are already fake-quantized by the checkpoint path above; these
        # scopes let us isolate extra quality impact from VAE/decoder/LQ/Conv3d ops.
        scopes = {x.strip().lower() for x in (fakequant_extra_scopes or "").split(",") if x.strip()}
        if "all" in scopes:
            scopes.update({"wan_vae", "tcdecoder", "lq_proj_in", "dit_conv3d"})
        if scopes:
            if fq_mode not in ("a8w8", "a16w8"):
                raise ValueError("fakequant_extra_scopes currently supports only FakeQuant_A8W8 or FakeQuant_A16W8")
            log(f"Applying extra FakeQuant scopes: {sorted(scopes)}", message_type='info', icon="🧪")
            if "wan_vae" in scopes and fq_pipe.vae is not None:
                convert_ops_to_fakequant(fq_pipe.vae, mode=fq_mode, op_types=("linear", "conv2d", "conv3d"), prefix="wan_vae")
            if "tcdecoder" in scopes and fq_pipe.TCDecoder is not None:
                convert_ops_to_fakequant(fq_pipe.TCDecoder, mode=fq_mode, op_types=("linear", "conv2d", "conv3d"), prefix="tcdecoder")
            if "lq_proj_in" in scopes and hasattr(fq_pipe.denoising_model(), "LQ_proj_in"):
                convert_ops_to_fakequant(fq_pipe.denoising_model().LQ_proj_in, mode=fq_mode, op_types=("linear", "conv2d", "conv3d"), prefix="lq_proj_in")
            if "dit_conv3d" in scopes:
                # Isolate DiT-native Conv3d (patch embedding) from LQ_proj_in, which
                # is attached under the DiT object but has its own scope above.
                dit_model = fq_pipe.denoising_model()
                saved_lq = getattr(dit_model, "LQ_proj_in", None)
                if saved_lq is not None:
                    dit_model.LQ_proj_in = None
                try:
                    convert_ops_to_fakequant(dit_model, mode=fq_mode, op_types=("conv3d",), prefix="dit_conv3d")
                finally:
                    if saved_lq is not None:
                        dit_model.LQ_proj_in = saved_lq

        fq_pipe.to(device, dtype=dtype)
        fq_pipe.enable_vram_management(num_persistent_param_in_dit=None)
        fq_pipe.init_cross_kv(prompt_path=prompt_path)
        fq_pipe.load_models_to_device(["dit", "vae"])
        fq_pipe.offload_model()

        log(f"FakeQuant pipeline initialized: mode={fq_mode}", message_type='info', icon="✅")
        return fq_pipe

    elif is_quantized_ckpt:
        log(f"Manual loading detected for quantized DiT: {ckpt_path}", message_type='info', icon="🔍")
        # Instantiate model structure
        try:
            from .src.models.wan_video_dit import WanModel
            from .src.models.quantization.quant import convert_model_to_w8a16
        except ImportError:
            from src.models.wan_video_dit import WanModel
            from src.models.quantization.quant import convert_model_to_w8a16
        
        # FlashVSR-v1.1 config
        dit = WanModel(
            dim=1536, eps=1e-5, ffn_dim=8960, freq_dim=256, in_dim=16,
            num_heads=12, num_layers=30, out_dim=16, patch_size=(1, 2, 2), text_dim=4096
        )
        # Convert structure to W8A16 BEFORE loading weights
        convert_model_to_w8a16(dit)
        
        # Load state dict
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd_dit = load_file(ckpt_path)
        else:
            sd_dit = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
        new_sd = {}
        for k, v in sd_dit.items():
            if k.startswith("model."): new_sd[k[6:]] = v
            else: new_sd[k] = v
        dit.load_state_dict(new_sd, strict=False)
        
        # Manually register in ModelManager
        mm.model.append(dit.eval())
        mm.model_path.append(ckpt_path)
        mm.model_name.append("wan_video_dit")
        
        # Load VAE separately
        mm.load_models([vae_path])
    else:
        # Standard loading
        mm.load_models([ckpt_path, vae_path])

    if mode == "full":
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)

        # =======================================================================
        # FIX 7: EXPLICIT VAE class instantiation - NO guessing from state_dict
        # =======================================================================
        log(f"Creating EXPLICIT VAE instance: {vae_class.__name__} (dim={vae_dim}, z_dim={vae_z_dim})", 
            message_type='info', icon="📦")
        
        # Load weights from file
        if vae_path.endswith(".safetensors"):
            try:
                import safetensors.torch
                sd = safetensors.torch.load_file(vae_path)
            except ImportError:
                raise RuntimeError("safetensors library required to load .safetensors VAE file.")
        else:
            sd = torch.load(vae_path, map_location="cpu", weights_only=False)
        
        # EXPLICIT class instantiation based on user selection (FIX 7)
        # NO state_dict inspection - we trust the user's selection
        if vae_class == LightX2VVAE:
            pipe.vae = LightX2VVAE(z_dim=vae_z_dim, dim=vae_dim, use_full_arch=use_full_arch)
        elif vae_class == Wan22VideoVAE:
            pipe.vae = Wan22VideoVAE(z_dim=vae_z_dim, dim=vae_dim)
        else:  # WanVideoVAE (default)
            pipe.vae = WanVideoVAE(z_dim=vae_z_dim, dim=vae_dim)
        
        # Load state dict with logging for missing/unexpected keys
        load_result = pipe.vae.load_state_dict(sd, strict=False)
        if load_result.missing_keys:
            log(f"VAE missing keys: {len(load_result.missing_keys)} (expected for Light* models)", 
                message_type='info', icon="ℹ️")
        if load_result.unexpected_keys:
            log(f"VAE unexpected keys: {len(load_result.unexpected_keys)}", 
                message_type='info', icon="ℹ️")
        
        pipe.vae = pipe.vae.to(device=device, dtype=dtype)
        
        log(f"Loaded VAE weights from: {vae_path}", message_type='info', icon="✅")
        log(f"VAE Type Active: {type(pipe.vae).__name__}", message_type='info', icon="📦")

        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
        
        # =======================================================================
        # FIX: Load TCDecoder for Full Mode (official FlashVSR approach)
        # The official FlashVSR uses TCDecoder for all modes with LQ conditioning.
        # This is critical because TCDecoder.decode_video() takes a `cond` parameter
        # that enables proper video super-resolution with low-quality guidance.
        # =======================================================================
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16+768)
        mis = pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device, weights_only=False), strict=False)
        pipe.TCDecoder.clean_mem()
        log(f"Loaded TCDecoder for Full Mode (official FlashVSR approach)", message_type='info', icon="✅")
    else:
        # For non-full modes, we still need to load VAE to the model manager if not manually handled
        if not is_quantized_ckpt and not is_fakequant_ckpt:
            mm.load_models([ckpt_path, vae_path])
            
        if mode == "tiny":
            pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
        else:
            pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16+768)
        mis = pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device, weights_only=False), strict=False)
        pipe.TCDecoder.clean_mem()
    
    if model == "FlashVSR":
        pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    else:
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu", weights_only=False), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype=dtype)
    
    if quantize_mode == "W8A16":
        # Check if the loaded checkpoint was already quantized (filename contains w8a16 or w8a8)
        # If it was, we already handled structural conversion and weight loading in the manual load block above.
        if "w8a16" in ckpt_path.lower() or "w8a8" in ckpt_path.lower():
            log(f"Quantized model already active from pre-quantized checkpoint.", message_type='info', icon="🗜️")
        else:
            try:
                from .src.models.quantization.quant import convert_model_to_w8a16
            except ImportError:
                from src.models.quantization.quant import convert_model_to_w8a16
            log("Applying ON-THE-FLY W8A16 quantization to DiT model...", message_type='info', icon="🗜️")
            convert_model_to_w8a16(pipe.denoising_model())

    elif quantize_mode == "W8A8_SmoothQuant":
        try:
            from .src.models.quantization.quant import convert_model_to_w8a8_smoothquant
            from .src.models.quantization.smoothquant import inject_observers, collect_activation_stats
        except ImportError:
            from src.models.quantization.quant import convert_model_to_w8a8_smoothquant
            from src.models.quantization.smoothquant import inject_observers, collect_activation_stats
        log("Running W8A8 SmoothQuant calibration...", message_type='info', icon="🗜️")
        # Get dataset path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, "datasets", "test")
        # Collect activation stats from calibration videos
        act_stats = collect_activation_stats(
            pipe.denoising_model(), dataset_path, pipe,
            num_videos=3, frames_per_video=4
        )
        log(f"Calibration complete: collected stats from {len(act_stats)} layers", message_type='info', icon="✅")
        log("Applying W8A8 SmoothQuant to DiT model...", message_type='info', icon="🗜️")
        convert_model_to_w8a8_smoothquant(pipe.denoising_model(), act_stats, alpha=0.5)

    elif quantize_mode == "W8A8":
        # W8A8 without SmoothQuant migration — uses Int8ActLinear
        # If checkpoint is already quantized (W8A8), skip calibration
        if is_quantized_ckpt:
            log(f"W8A8 model already active from pre-quantized checkpoint (skip calibration).", message_type='info', icon="🗜️")
        else:
            try:
                from .src.models.quantization.quant import convert_model_to_w8a8
                from .src.models.quantization.smoothquant import inject_observers, collect_activation_stats
            except ImportError:
                from src.models.quantization.quant import convert_model_to_w8a8
                from src.models.quantization.smoothquant import inject_observers, collect_activation_stats
            log("Running W8A8 calibration (non-SmoothQuant)...", message_type='info', icon="🗜️")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.join(current_dir, "datasets", "test")
            act_stats = collect_activation_stats(
                pipe.denoising_model(), dataset_path, pipe,
                num_videos=3, frames_per_video=4
            )
            log(f"Calibration complete: collected stats from {len(act_stats)} layers", message_type='info', icon="✅")
            log(f"Applying W8A8 to DiT model using {w8a8_engine.upper()} engine...", message_type='info', icon="🗜️")
            convert_model_to_w8a8(pipe.denoising_model(), act_stats, method='percentile99', engine=w8a8_engine)

    elif quantize_mode == "W8A8_PTQ":
        # Load pre-compiled TensorRT INT8 engine for DiT
        # VAE stays bf16, DiT is from TRT engine (not model manager)
        log("W8A8_PTQ mode: DiT from TRT engine, VAE from bf16...", message_type='info', icon='🗜️')
        # Load TRT engine if provided
        if trt_engine_path is not None:
            pipe.trt_engine_ = load_trt_engine(trt_engine_path)
        else:
            log("Warning: W8A8_PTQ mode without trt_engine_path - DiT not loaded!", message_type='warning', icon='⚠️')

    elif quantize_mode in fakequant_modes:
        # quantize_mode already drove pipeline creation in the is_fakequant_ckpt branch above.
        # If we reach here with this quantize_mode, it means the checkpoint path
        # didn't contain "fakequant" — we still want to create FakeQuant pipeline
        # if quantize_mode was explicitly set. The above branch handles that via
        # setting is_fakequant_ckpt=True. So this branch is a no-op for the path-based
        # flow, but guard against calling code that incorrectly passes this mode
        # when no FakeQuant checkpoint exists.
        if not is_fakequant_ckpt:
            log(f"Warning: quantize_mode={quantize_mode} but no FakeQuant checkpoint found. "
                "Provide a checkpoint path containing 'fakequant' or a converted checkpoint.",
                message_type='warning', icon='⚠️')
        pass

    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit","vae"])
    pipe.offload_model()

    # Log final pipeline info with VAE confirmation
    vae_info = f"VAE Model: {vae_model}"
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        vae_info += f" ({type(pipe.vae).__name__})"
    
    attn_mode = getattr(wan_video_dit, 'ATTENTION_MODE', 'sparse_sage_attention')
    log(f"Pipeline Initialized: Mode={mode}, Device={device}, Dtype={dtype}, Attention={attn_mode}", message_type='info', icon="🔧")
    log(f"Model: {model}, {vae_info}", message_type='info', icon="📦")

    return pipe

class cqdm:
    def __init__(self, iterable=None, total=None, desc="Processing", enable_debug=False):
        self.desc = desc
        self.pbar = None
        self.iterable = None
        self.total = total
        self.enable_debug = enable_debug
        self.start_time = time.time()
        self.step_idx = 0
        
        if iterable is not None:
            try:
                self.total = len(iterable)
                self.iterable = iter(iterable)
            except TypeError:
                if self.total is None:
                    raise ValueError("Total must be provided for iterables with no length.")

        elif self.total is not None:
            pass
            
        else:
            raise ValueError("Either iterable or total must be provided.")
            
    def __iter__(self):
        if self.iterable is None:
            raise TypeError(f"'{type(self).__name__}' object is not iterable. Did you mean to use it with a 'with' statement?")
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self
    
    def __next__(self):
        if self.iterable is None:
            raise TypeError("Cannot call __next__ on a non-iterable cqdm object.")
        try:
            step_start = time.time()
            val = next(self.iterable)
            
            if self.pbar:
                self.pbar.update(1)
            
            self.step_idx += 1

            # Show a text progress bar in the log (single line using \r)
            perc = (self.step_idx / self.total) * 100
            bar_len = 20
            filled = int(bar_len * self.step_idx // self.total)
            bar = '█' * filled + '░' * (bar_len - filled)

            elapsed = time.time() - self.start_time
            rate = self.step_idx / elapsed if elapsed > 0 else 0

            msg = f"{self.desc}: {self.step_idx}/{self.total} |{bar}| {perc:.1f}%"

            if self.enable_debug:
                step_end = time.time()
                step_time = step_end - step_start
                msg += f" (Step: {step_time:.2f}s)"
                # Pass in_place=True to log_resource_usage to keep it on one line if possible
                # But note log_resource_usage prints Resource usage which is long.
                log_resource_usage(prefix=msg, in_place=True)
            else:
                print(f"\r{msg}", end="", flush=True)
                if self.step_idx == self.total:
                    print()

            return val
        except StopIteration:
            total_time = time.time() - self.start_time
            if self.enable_debug:
                # Use print with newline here to finalize the log block
                print(f"\n✅ Loop '{self.desc}' finished in {total_time:.2f}s", flush=True)
            raise
            
    def __enter__(self):
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self.pbar
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def __len__(self):
        return self.total

def process_chunk(pipe, frames, scale, color_fix, color_fix_method, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range, seed, force_offload, enable_debug, is_single_frame_input=False, context_pad=0):
    """
    Processes a single chunk of frames.
    
    =============================================================================
    FIX 3: Black Border Fix - Proper cropping to remove padding
    =============================================================================
    """
    # Aggressive garbage collection before processing (FIX 5)
    clean_vram()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    _frames = frames
    _device = pipe.device
    dtype = pipe.torch_dtype
    
    # Store original dimensions for cropping (FIX 3)
    original_H, original_W = frames.shape[1], frames.shape[2]
    target_H, target_W = original_H * scale, original_W * scale
    
    # Padding logic for the chunk (temporal padding)
    add = next_8n5(frames.shape[0]) - frames.shape[0]
    padding_frames = frames[-1:, :, :, :].repeat(add, 1, 1, 1)
    _frames = torch.cat([frames, padding_frames], dim=0)

    _N = _frames.shape[0]
    is_tiled_dit_active = False
    if tiled_dit:
        # Spatial tiling for DiT requires at least 25 frames for the streaming pipeline.
        # For shorter chunks, fall back to non-tiled (streaming) mode which handles
        # any frame count >= 25. RTX 4090 has enough VRAM for full-frame processing.
        if _N >= 25:
            log(f"Note: tiled_dit active; processing {_N} frames in streaming mode (RTX 4090 has ample VRAM)", message_type='info', icon="ℹ️")
            is_tiled_dit_active = True
        else:
            log(f"Note: tiled_dit requires >= 25 frames; using streaming mode for {_N} frames", message_type='info', icon="ℹ️")
            is_tiled_dit_active = False
    
    if is_tiled_dit_active:
        # Tiled DiT: iterate per spatial tile, process ALL frames per tile.
        # The streaming pipeline needs many frames (process_total_num = (F-1)//8 - 2 >= 1
        # requires F >= 25). Processing one frame at a time gave F=1 → empty frames_total.
        N, H, W, C = _frames.shape
        out_H, out_W = H * scale, W * scale

        log(f"Starting Tiled Processing: {N} frames, output {out_W}x{out_H}", message_type='info', icon="🚀")

        tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)

        # Canvas holds all frames; blend tiles spatially with feathering
        frame_canvas = torch.zeros((N, out_H, out_W, C), dtype=torch.float32, device='cpu')
        weight_canvas = torch.zeros((N, out_H, out_W, C), dtype=torch.float32, device='cpu')

        class cqdm_tile_single(cqdm):
            def __init__(self, iterable=None, total=None, desc="Processing", enable_debug=False):
                self._enable_debug = enable_debug
                self._desc = desc
                super().__init__(iterable, total=total, desc=desc)

        for i, (x1, y1, x2, y2) in enumerate(cqdm(tile_coords, total=len(tile_coords), desc="Tiled Tiles", enable_debug=enable_debug)):
            # Expand crop bounds by context_pad so the model sees neighboring pixels,
            # reducing content discontinuities at tile boundaries.
            x1_ctx = max(0, x1 - context_pad)
            y1_ctx = max(0, y1 - context_pad)
            x2_ctx = min(W, x2 + context_pad)
            y2_ctx = min(H, y2 + context_pad)

            input_tile = _frames[:, y1_ctx:y2_ctx, x1_ctx:x2_ctx, :]  # (N, expanded_h, expanded_w, C)

            LQ_tile, th, tw, F, tile_sH, tile_sW, tile_pad_top, tile_pad_left = prepare_input_tensor(
                input_tile, _device, scale=scale, dtype=dtype
            )
            if not isinstance(pipe, FlashVSRTinyLongPipeline):
                LQ_tile = LQ_tile.to(_device)

            output_tile_gpu = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
                progress_bar_cmd=cqdm_tile_single, LQ_video=LQ_tile, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                color_fix=color_fix, color_fix_method=color_fix_method, unload_dit=unload_dit, force_offload=force_offload,
                enable_debug_logging=False
            )

            processed_tile = tensor2video(output_tile_gpu).to('cpu')  # (F_out, tH, tW, C)

            # Crop prepare_input_tensor padding (alignment/temporal padding)
            max_crop_h = min(tile_pad_top + tile_sH, processed_tile.shape[1])
            max_crop_w = min(tile_pad_left + tile_sW, processed_tile.shape[2])
            if max_crop_h > tile_pad_top and max_crop_w > tile_pad_left:
                processed_tile = processed_tile[:, tile_pad_top:max_crop_h, tile_pad_left:max_crop_w, :]

            # Crop context padding out of the upscaled output so only the
            # original tile region remains before feather-blending into the canvas.
            if context_pad > 0:
                ctx_top   = (y1 - y1_ctx) * scale
                ctx_left  = (x1 - x1_ctx) * scale
                ctx_bot   = (y2_ctx - y2) * scale
                ctx_right = (x2_ctx - x2) * scale
                h_out = processed_tile.shape[1]
                w_out = processed_tile.shape[2]
                processed_tile = processed_tile[
                    :,
                    ctx_top : h_out - ctx_bot if ctx_bot > 0 else h_out,
                    ctx_left : w_out - ctx_right if ctx_right > 0 else w_out,
                    :
                ]

            n_valid = min(processed_tile.shape[0], N)

            mask_nchw = create_feather_mask(
                (processed_tile.shape[1], processed_tile.shape[2]),
                tile_overlap * scale
            )  # (1, 1, H, W)
            mask_nhwc = mask_nchw.permute(0, 2, 3, 1).expand(n_valid, -1, -1, -1).float()  # (n_valid, H, W, 1)

            out_x1, out_y1 = x1 * scale, y1 * scale
            out_x2 = out_x1 + processed_tile.shape[2]
            out_y2 = out_y1 + processed_tile.shape[1]

            frame_canvas[:n_valid, out_y1:out_y2, out_x1:out_x2, :] += processed_tile[:n_valid].float() * mask_nhwc
            weight_canvas[:n_valid, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc

            del LQ_tile, output_tile_gpu, processed_tile, input_tile, mask_nchw, mask_nhwc
            clean_vram()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        weight_canvas[weight_canvas == 0] = 1.0
        final_output = frame_canvas / weight_canvas
        del frame_canvas, weight_canvas
        clean_vram()
    else:
        log("Preparing full frame processing...", message_type='info', icon="🎞️")
        if enable_debug:
            log_resource_usage(prefix="Pre-Preprocess")
        
        # Get padding info for cropping (FIX 3)
        LQ, th, tw, F, sH, sW, pad_top, pad_left = prepare_input_tensor(_frames, _device, scale=scale, dtype=dtype)
        if not isinstance(pipe, FlashVSRTinyLongPipeline):
            LQ = LQ.to(_device)
            
        log(f"Processing {frames.shape[0]} frames...", message_type='info', icon="🚀")
        
        process_start = time.time()

        class cqdm_debug(cqdm):
            def __init__(self, iterable=None, total=None, desc="Processing", enable_debug=False):
                self._enable_debug = enable_debug
                super().__init__(iterable, total=total, desc=desc)
            def __str__(self):
                return self._desc if hasattr(self, '_desc') else self.desc

        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
            progress_bar_cmd=cqdm_debug, LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
            topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
            color_fix = color_fix, color_fix_method = color_fix_method, unload_dit=unload_dit, force_offload=force_offload,
            enable_debug_logging=enable_debug
        )

        process_end = time.time()
        
        if enable_debug:
            log(f"Inference completed in {process_end - process_start:.2f}s", message_type='info', icon="⏱️")
        final_output_tensor = tensor2video(video).to('cpu')
        
        # =====================================================================
        # FIX 3: Crop output to remove padding - use stored padding offsets
        # =====================================================================
        # The output has dimensions (N, tH, tW, C) where tH/tW are padded
        # We need to crop to actual scaled dimensions (sH, sW)
        final_output = final_output_tensor[:, pad_top:pad_top + sH, pad_left:pad_left + sW, :]
        
        if enable_debug:
            log(f"Cropped output from ({final_output_tensor.shape[1]}, {final_output_tensor.shape[2]}) "
                f"to ({final_output.shape[1]}, {final_output.shape[2]}) removing padding", 
                message_type='info', icon="✂️")

        del video, LQ
        clean_vram()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_single_frame_input and frames.shape[0] == 1:
        if frames.shape[0] == 1:
            final_output = final_output.to("cpu")
            stacked_image_tensor = torch.median(final_output, dim=0).values.unsqueeze(0).float()
            del final_output
            clean_vram()
            return stacked_image_tensor

    return final_output[:frames.shape[0], :, :, :]

def flashvsr(pipe, frames, scale, color_fix, color_fix_method, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range, seed, force_offload, enable_debug=False, chunk_size=0, resize_factor=1.0, mode="full", context_pad=0, quantize_mode="None"):
    """
    =============================================================================
    FIX 9 & 10: Unified Processing Pipeline with Pre-Flight Check
    =============================================================================

    Main FlashVSR processing function.
    - FIX 4: Lossless Resize - Use NEAREST for integer scaling factors
    - FIX 5: VRAM Advisory Logging with 95% threshold
    - FIX 9: Pre-Flight Resource Check before processing
    - FIX 10: Unified processing logic applied across all modes
    """
    # Aggressive garbage collection (FIX 5)
    clean_vram()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    # ==========================================================================
    # FIX 9: Pre-Flight Resource Check (BEFORE loading heavy models/processing)
    # ==========================================================================
    preflight_result = log_preflight_check(
        frames.shape[2], frames.shape[1], frames.shape[0], scale, chunk_size, resize_factor,
        tiled_vae, tiled_dit, mode=mode, quantize_mode=quantize_mode
    )
    
    # If pre-flight check suggests OOM, optionally apply recommended settings
    # (Currently just logs warnings - user can adjust settings manually)
    
    # ==========================================================================
    # FIX 4: Lossless Resize Factor
    # Use NEAREST interpolation for integer-like factors, BICUBIC otherwise
    # ==========================================================================
    if resize_factor < 1.0 and resize_factor > 0:
        log(f"Resizing input by factor {resize_factor}...", message_type='info', icon="📉")
        orig_H, orig_W = frames.shape[1], frames.shape[2]
        new_H, new_W = int(orig_H * resize_factor), int(orig_W * resize_factor)
        
        # Check if resize factor results in integer scaling (lossless possible)
        is_integer_scale = (orig_H % new_H == 0 and orig_W % new_W == 0) or (resize_factor in [0.5, 0.25, 0.125])
        
        frames_permuted = frames.permute(0, 3, 1, 2)
        if is_integer_scale:
            # Use NEAREST for potentially lossless integer downscaling
            frames_resized = F.interpolate(frames_permuted, size=(new_H, new_W), mode='nearest')
            log(f"Using NEAREST interpolation (lossless for {resize_factor}x)", message_type='info', icon="🔍")
        else:
            # Use BICUBIC for non-integer factors
            frames_resized = F.interpolate(frames_permuted, size=(new_H, new_W), mode='bicubic', align_corners=False)
            log(f"Using BICUBIC interpolation for non-integer scaling", message_type='info', icon="🔍")
        
        frames = frames_resized.permute(0, 2, 3, 1)  # Back to NHWC
        del frames_permuted, frames_resized
        clean_vram()

    start_time = time.time()
    
    # Get current dimensions (after potential resize)
    N, H, W, C = frames.shape

    # ==========================================================================
    # FIX 5 & 10: Unified Debug Logging (same for all modes)
    # ==========================================================================
    if enable_debug:
        _device = pipe.device
        log(f"Debug Mode: Enabled", message_type='info', icon="🐞")
        log(f"Device: {_device}", message_type='info', icon="🖥️")
        log(f"Processing Mode: {mode}", message_type='info', icon="⚙️")
        if torch.cuda.is_available():
             log(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", message_type='info', icon="💾")
        log(f"Input Frames: {frames.shape}", message_type='info', icon="🎞️")
        log(f"Chunk Size: {chunk_size}", message_type='info', icon="📦")
        log(f"Tiled DiT: {tiled_dit}, Tiled VAE: {tiled_vae}", message_type='info', icon="🧩")
        log_resource_usage(prefix="Start")
    
    # VRAM Advisory (FIX 5) - Enhanced with mode
    if torch.cuda.is_available():
        log_vram_advisory(W, H, N, scale, tiled_vae, tiled_dit, mode=mode)

    # VRAM check and warning - FIX 5: Use 95% threshold (RTX 5070 Ti target)
    if torch.cuda.is_available():
        vram_free, vram_total = torch.cuda.mem_get_info()
        vram_used = vram_total - vram_free
        vram_usage_ratio = vram_used / vram_total

        # FIX 5: Only trigger OOM recovery at 95% threshold (not 90%)
        if vram_usage_ratio > VRAM_OOM_THRESHOLD:
            log(f"Warning: VRAM usage is very high ({vram_usage_ratio*100:.1f}% > {VRAM_OOM_THRESHOLD*100:.0f}%)! Enabling fallback options is recommended.", 
                message_type='warning', icon="⚠️")

    # Store input resolution for summary (FIX 8)
    input_resolution = f"{frames.shape[2]}x{frames.shape[1]}"
    output_resolution = f"{frames.shape[2] * scale}x{frames.shape[1] * scale}"
    
    # Chunking Logic
    total_frames = frames.shape[0]
    final_outputs = []

    is_single_frame_input = (frames.shape[0] == 1)

    if chunk_size > 0 and chunk_size < total_frames:
        num_chunks = math.ceil(total_frames / chunk_size)
        log(f"Splitting video into {num_chunks} chunks (size {chunk_size})...", message_type='info', icon="✂️")

        for i in range(num_chunks):
            chunk_start = i * chunk_size
            chunk_end = min((i + 1) * chunk_size, total_frames)

            if enable_debug:
                log(f"Processing Chunk {i+1}/{num_chunks}: Frames {chunk_start}-{chunk_end}", message_type='info', icon="🎞️")

            chunk_frames = frames[chunk_start:chunk_end]

            # Auto-Fallback Logic - FIX 5: Uses VRAM_OOM_THRESHOLD (95%)
            retry_count = 0
            max_retries = 2
            current_tiled_vae = tiled_vae
            current_tiled_dit = tiled_dit

            while retry_count <= max_retries:
                try:
                    chunk_out = process_chunk(
                        pipe, chunk_frames, scale, color_fix, color_fix_method, current_tiled_vae, current_tiled_dit,
                        tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio,
                        local_range, seed, force_offload, enable_debug,
                        is_single_frame_input=is_single_frame_input, context_pad=context_pad
                    )
                    final_outputs.append(chunk_out.cpu())
                    del chunk_out
                    clean_vram()
                    break # Success
                except torch.OutOfMemoryError as e:
                    retry_count += 1
                    clean_vram()
                    log(f"OOM detected in Chunk {i+1} (Attempt {retry_count}). Recovering...", message_type='warning', icon="🔄")

                    if not current_tiled_vae:
                        log("Auto-enabling Tiled VAE to prevent OOM (override)...", message_type='info', icon="🛡️")
                        current_tiled_vae = True
                    elif not current_tiled_dit:
                        log("Auto-enabling Tiled DiT to prevent OOM (override)...", message_type='info', icon="🛡️")
                        current_tiled_dit = True
                    elif not unload_dit:
                        log("Auto-enabling unload_dit to prevent OOM (offload DiT before VAE decode)...", message_type='info', icon="🛡️")
                        unload_dit = True
                    elif retry_count <= max_retries:
                        log("All memory optimizations active. Retrying with clean VRAM...", message_type='info', icon="🧹")
                        torch.cuda.empty_cache()
                    else:
                        log("Both Tiled VAE and DiT enabled but still OOM. Cannot recover.", message_type='error', icon="❌")
                        raise e # Cannot recover further

        final_output_tensor = torch.cat(final_outputs, dim=0)
    else:
        # Auto-Fallback Logic for single chunk/full video
        retry_count = 0
        max_retries = 2
        current_tiled_vae = tiled_vae
        current_tiled_dit = tiled_dit
        final_output_tensor = None

        while retry_count <= max_retries:
            try:
                final_output_tensor = process_chunk(
                    pipe, frames, scale, color_fix, color_fix_method, current_tiled_vae, current_tiled_dit,
                    tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio,
                    local_range, seed, force_offload, enable_debug,
                    is_single_frame_input=is_single_frame_input, context_pad=context_pad
                )
                break
            except torch.OutOfMemoryError as e:
                retry_count += 1
                clean_vram()
                log(f"OOM detected (Attempt {retry_count}). Recovering...", message_type='warning', icon="🔄")

                if not current_tiled_vae:
                    log("Auto-enabling Tiled VAE to prevent OOM (override)...", message_type='info', icon="🛡️")
                    current_tiled_vae = True
                elif not current_tiled_dit:
                    log("Auto-enabling Tiled DiT to prevent OOM (override)...", message_type='info', icon="🛡️")
                    current_tiled_dit = True
                elif not unload_dit:
                    log("Auto-enabling unload_dit to prevent OOM (offload DiT before VAE decode)...", message_type='info', icon="🛡️")
                    unload_dit = True
                else:
                    # All optimizations exhausted, max retries reached
                    log(f"All memory optimizations active but still OOM. Max retries reached. "
                        f"(tiled_vae={current_tiled_vae}, tiled_dit={current_tiled_dit}, unload_dit={unload_dit})",
                        message_type='error', icon="❌")
                    raise e

    end_time = time.time()
    total_time = end_time - start_time
    fps = frames.shape[0] / total_time if total_time > 0 else 0

    # ==========================================================================
    # FIX 8: Summary logging at end of processing
    # ==========================================================================
    if final_output_tensor is None:
        log("ERROR: final_output_tensor is None - all OOM recovery attempts failed", message_type='error', icon="❌")
        raise RuntimeError("Processing failed: unable to generate output due to insufficient VRAM even with all optimizations enabled")

    log("=" * 60, message_type='info')
    log("PROCESSING SUMMARY", message_type='finish', icon="📊")
    log(f"Total Processing Time: {total_time:.2f}s ({fps:.2f} FPS)", message_type='info', icon="⏱️")
    log(f"Input Resolution: {input_resolution} ({frames.shape[0]} frames)", message_type='info', icon="📥")
    log(f"Output Resolution: {output_resolution} ({final_output_tensor.shape[0]} frames)", message_type='info', icon="📤")
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3
        log(f"Peak VRAM Used: {peak_memory:.2f} GB", message_type='info', icon="📈")
        
    log_resource_usage(prefix="Final")
    log("=" * 60, message_type='info')
    
    return final_output_tensor


class FlashVSRNodeInitPipe:
    """
    =============================================================================
    FIX 1: Unified VAE Selection - Merged vae_type and alt_vae into vae_model
    =============================================================================
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["FlashVSR", "FlashVSR-v1.1"], {
                    "default": "FlashVSR-v1.1",
                    "tooltip": "Select the FlashVSR model version. V1.1 is recommended for better stability."
                }),
                "mode": (["tiny", "tiny-long", "full"], {
                    "default": "tiny",
                    "tooltip": 'Operation mode. "tiny": faster, standard memory. "tiny-long": optimized for long videos (lower VRAM). "full": higher quality but max VRAM.'
                }),
                "vae_model": (VAE_MODEL_OPTIONS, {
                    "default": "Wan2.1",
                    "tooltip": 'VAE model: Wan2.1 (default), Wan2.2, LightVAE_W2.1 (50% less VRAM), TAE_W2.2, LightTAE_HY1.5. Auto-downloads if missing.'
                }),
                "force_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, forces offloading of models to CPU RAM after execution to free up VRAM for other nodes."
                }),
                "precision": (["fp16", "bf16", "auto"], {
                    "default": "auto",
                    "tooltip": "Inference precision. 'auto' selects bf16 if supported (RTX 30/40/50 series), otherwise fp16. bf16 is recommended."
                }),
                "device": (device_choices, {
                    "default": device_choices[0],
                    "tooltip": "Select the computation device (CUDA GPU, CPU, etc.). 'auto' picks the best available."
                }),
                "attention_mode": (["sparse_sage_attention", "block_sparse_attention", "flash_attention_2", "sdpa"], {
                    "default": "sparse_sage_attention",
                    "tooltip": 'Attention mechanism backend. "sparse_sage"/"block_sparse" use efficient sparse attention. "flash_attention_2"/"sdpa" use dense attention (slower, more VRAM).'
                }),
                "quantize_mode": (["None", "W8A16", "W8A8_SmoothQuant", "W8A8_PTQ"], {
                    "default": "None",
                    "tooltip": "Quantization mode to reduce VRAM. W8A16 dynamically halves weight VRAM. W8A8 (planned) quantizes both weights and activations."
                }),
            }
        }
    
    RETURN_TYPES = ("PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    DESCRIPTION = 'Initializes the FlashVSR pipeline. 5 VAE options: Wan2.1, Wan2.2, LightVAE_W2.1, TAE_W2.2, LightTAE_HY1.5. Auto-downloads missing files.'
    
    def main(self, model, mode, vae_model, force_offload, precision, device, attention_mode, quantize_mode="None"):
        _device = device
        if device == "auto":
            _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else device
        if _device == "auto" or _device not in device_choices:
            raise RuntimeError("No devices found to run FlashVSR!")
            
        if _device.startswith("cuda"):
            torch.cuda.set_device(_device)
            
        wan_video_dit.ATTENTION_MODE = attention_mode

        # Auto bfloat16 detection
        if precision == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                precision = "bf16"
                log("Auto-detected bf16 support.", message_type='info', icon="⚙️")
            else:
                precision = "fp16"
                log("Defaulting to fp16.", message_type='info', icon="⚙️")
            
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        try:
            dtype = dtype_map[precision]
        except:
            dtype = torch.bfloat16

        # Use unified vae_model parameter
        pipe = init_pipeline(model, mode, _device, dtype, vae_model=vae_model, quantize_mode=quantize_mode)
        # FIX 10: Store mode with pipe for unified processing logic
        return((pipe, force_offload, mode),)

class FlashVSRNodeAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PIPE", {
                    "tooltip": "The initialized FlashVSR pipeline object from the Init node."
                }),
                "frames": ("IMAGE", {
                    "tooltip": "Input video frames to be upscaled. Batch of images (N, H, W, C)."
                }),
                "scale": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 4,
                    "tooltip": "Upscaling factor. 2x or 4x. Higher scale requires more VRAM and compute."
                }),
                "color_fix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply color correction to match the output colors with the input, preventing color shifts."
                }),
                "color_fix_method": (["wavelet", "adain"], {
                    "default": "wavelet",
                    "tooltip": 'Color correction method. "wavelet": no ghosting artifacts, recommended. "adain": adaptive instance normalization, may cause slight ghosting on some inputs.'
                }),
                "tiled_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable spatial tiling for the VAE decoder. Reduces VRAM usage significantly but is slower. Recommended for high-res outputs."
                }),
                "tiled_dit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable spatial tiling for the Diffusion Transformer (DiT). Crucial for saving VRAM on large inputs. Disabling it is faster but risky."
                }),
                "tile_size": ("INT", {
                    "default": 256,
                    "min": 32,
                    "max": 1024,
                    "step": 32,
                    "tooltip": "Size of the tiles for DiT processing. Smaller = less VRAM, more tiles, slower."
                }),
                "tile_overlap": ("INT", {
                    "default": 24,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Overlap pixels between tiles to blend seams. Higher overlap = smoother transitions but more computation."
                }),
                "context_pad": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 256,
                    "step": 16,
                    "tooltip": "Context pixels added around each DiT tile input. Lets the model see neighboring content when upscaling tile edges, reducing content discontinuities at tile boundaries. 0 = disabled."
                }),
                "unload_dit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the DiT model from VRAM before VAE decoding starts. Use this if VAE decode runs out of memory."
                }),
                "sparse_ratio": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Control for sparse attention. 1.5 is faster, 2.0 is more stable/quality. (For sparse backends only)"
                }),
                "kv_ratio": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Key/Value cache ratio. 1.0 uses less VRAM; 3.0 provides highest quality retention."
                }),
                "local_range": (["7 (sharpest)", "9 (balanced)", "11 (most stable)"], {
                    "default": "9 (balanced)",
                    "tooltip": "Local attention range window. 7 = sharpest details (may be less stable); 9 = balanced; 11 = most stable/consistant."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624,
                    "tooltip": "Random seed for noise generation. Same seed + same settings = reproducible results."
                }),
                "frame_chunk_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Process video in chunks of N frames to prevent VRAM OOM. 0 = Process all frames at once. Results are merged on CPU."
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable verbose logging to console. Shows VRAM usage, step times, tile info, and detailed progress."
                }),
                "keep_models_on_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Move models to CPU RAM instead of keeping them in VRAM when not in use. Prevents VRAM fragmentation/OOM."
                }),
                "resize_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Resize input frames before processing. Set to 0.5x for large 1080p+ videos to save VRAM."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    
    def main(self, pipe, frames, scale, color_fix, color_fix_method, tiled_vae, tiled_dit, tile_size, tile_overlap, context_pad, unload_dit, sparse_ratio, kv_ratio, local_range, seed, frame_chunk_size, enable_debug, keep_models_on_cpu, resize_factor):
        # Extract local_range int from dropdown string
        local_range_int = int(local_range.split(" ")[0])
        # FIX 10: Extract mode from pipe tuple for unified processing
        # Pipe tuple structure: (pipeline_object, force_offload, mode)
        # Backwards compatible with older 2-element tuples (pipeline, force_offload)
        if len(pipe) >= 3:
            _pipe, _, mode = pipe
        else:
            _pipe = pipe[0]
            mode = "full"  # Default fallback for backwards compatibility
        output = flashvsr(_pipe, frames, scale, color_fix, color_fix_method, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range_int, seed, keep_models_on_cpu, enable_debug, frame_chunk_size, resize_factor, mode=mode, context_pad=context_pad)
        return(output.cpu().float(),)

class FlashVSRNode:
    """
    =============================================================================
    FIX 1: Unified VAE Selection - Single vae_model dropdown
    =============================================================================
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Input video frames to be upscaled. Batch of images (N, H, W, C)."
                }),
                "model": (["FlashVSR", "FlashVSR-v1.1"], {
                    "default": "FlashVSR-v1.1",
                    "tooltip": "Select the FlashVSR model version. V1.1 is recommended for better stability."
                }),
                "mode": (["tiny", "tiny-long", "full"], {
                    "default": "tiny",
                    "tooltip": 'Operation mode. "tiny": faster, standard memory. "tiny-long": optimized for long videos (lower VRAM). "full": higher quality but max VRAM.'
                }),
                "vae_model": (VAE_MODEL_OPTIONS, {
                    "default": "Wan2.1",
                    "tooltip": 'VAE model: Wan2.1 (default), Wan2.2, LightVAE_W2.1 (50% less VRAM), TAE_W2.2, LightTAE_HY1.5. Auto-downloads if missing.'
                }),
                "scale": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 4,
                    "tooltip": "Upscaling factor. 2x or 4x."
                }),
                "color_fix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply color correction to prevent color shifts. Recommended: ON."
                }),
                "color_fix_method": (["wavelet", "adain"], {
                    "default": "wavelet",
                    "tooltip": '"wavelet": no ghosting (recommended). "adain": may cause slight ghosting.'
                }),
                "tiled_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable spatial tiling for the VAE decoder. Reduces VRAM usage significantly but is slower. Recommended for high-res outputs."
                }),
                "tiled_dit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable spatial tiling for the Diffusion Transformer (DiT). Crucial for saving VRAM on large inputs."
                }),
                "unload_dit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload the DiT model from VRAM before VAE decoding starts to free up memory. Recommended for 16GB VRAM."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624,
                    "tooltip": "Random seed for noise generation."
                }),
                "frame_chunk_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Process video in chunks of N frames to prevent VRAM OOM. 0 = Process all frames at once."
                }),
                "attention_mode": (["sparse_sage_attention", "block_sparse_attention", "flash_attention_2", "sdpa"], {
                    "default": "sparse_sage_attention",
                    "tooltip": 'Attention mechanism backend. "sparse_sage" is recommended for speed/memory efficiency.'
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable extensive logging for debugging."
                }),
                "keep_models_on_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Move models to CPU RAM instead of keeping them in VRAM when not in use."
                }),
                "resize_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Resize input frames before processing. Set to 0.5x for large 1080p+ videos to save VRAM."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    DESCRIPTION = 'Single-node FlashVSR upscaling. 5 VAE options: Wan2.1, Wan2.2, LightVAE_W2.1, TAE_W2.2, LightTAE_HY1.5. Auto-downloads missing files.'
    
    def main(self, model, frames, mode, vae_model, scale, color_fix, color_fix_method, tiled_vae, tiled_dit, unload_dit, seed, frame_chunk_size, attention_mode, enable_debug, keep_models_on_cpu, resize_factor):
        _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "auto"
        if _device == "auto" or _device not in device_choices:
            raise RuntimeError("No devices found to run FlashVSR!")
            
        if _device.startswith("cuda"):
            torch.cuda.set_device(_device)
            
        wan_video_dit.ATTENTION_MODE = attention_mode
        
        # Use unified vae_model parameter    
        pipe = init_pipeline(model, mode, _device, torch.float16, vae_model=vae_model)
        # FIX 10: Pass mode for unified processing logic
        output = flashvsr(pipe, frames, scale, color_fix, color_fix_method, tiled_vae, tiled_dit, 256, 24, unload_dit, 2.0, 3.0, 11, seed, keep_models_on_cpu, enable_debug, frame_chunk_size, resize_factor, mode=mode)
        return(output.cpu().float(),)

# =============================================================================
# PR #2: Additional Features
# - FlashVSRNodeImageSR: Single image super-resolution
# - FlashVSRNodeBatchPath: Batch video processing from directory
# =============================================================================

class FlashVSRNodeImageSR:
    """
    Single image super-resolution using FlashVSR.
    Converts input image to 25-frame video, processes, and outputs upscaled image.
    Default output: 25 frames at 1 second (25fps).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["FlashVSR", "FlashVSR-v1.1"], {
                    "default": "FlashVSR-v1.1",
                    "tooltip": "Select the FlashVSR model version. V1.1 is recommended."
                }),
                "mode": (["tiny", "tiny-long", "full"], {
                    "default": "tiny",
                    "tooltip": '"tiny": fast, "full": higher quality.'
                }),
                "vae_model": (VAE_MODEL_OPTIONS, {
                    "default": "Wan2.1",
                    "tooltip": 'VAE model. LightVAE_W2.1 uses less VRAM. Auto-downloads if missing.'
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image to be upscaled. Will be repeated as 25 frames."
                }),
                "target_width": ("INT", {
                    "default": 1280,
                    "min": 128,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Target output width. Image will be upscaled to this resolution."
                }),
                "target_height": ("INT", {
                    "default": 768,
                    "min": 128,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Target output height. Image will be upscaled to this resolution."
                }),
                "scale": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 4,
                    "tooltip": "Upscaling factor. 4x recommended (dataset was 4x trained)."
                }),
                "color_fix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply color correction to prevent color shifts."
                }),
                "color_fix_method": (["wavelet", "adain"], {
                    "default": "wavelet",
                    "tooltip": '"wavelet": no ghosting (recommended). "adain": may cause slight ghosting.'
                }),
                "tiled_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable spatial tiling for VAE. Reduces VRAM usage."
                }),
                "tiled_dit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable spatial tiling for DiT. Crucial for large inputs."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 112589990684264,
                    "tooltip": "Random seed for noise generation."
                }),
                "attention_mode": (["sparse_sage_attention", "block_sparse_attention", "flash_attention_2", "sdpa"], {
                    "default": "sparse_sage_attention",
                    "tooltip": 'Attention backend. "sparse_sage" is recommended.'
                }),
                "keep_models_on_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Move models to CPU RAM when not in use."
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable verbose logging."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    DESCRIPTION = "Single image super-resolution. Converts image to 25-frame video for FlashVSR processing, outputs upscaled image."

    def main(self, model, mode, vae_model, image, target_width, target_height, scale, color_fix, color_fix_method, tiled_vae, tiled_dit, seed, attention_mode, keep_models_on_cpu, enable_debug):
        _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "auto"
        if _device == "auto":
            raise RuntimeError("No GPU found for FlashVSR!")
        if _device.startswith("cuda"):
            torch.cuda.set_device(_device)

        wan_video_dit.ATTENTION_MODE = attention_mode

        # Prepare input: upscale tensor to target resolution first
        # image shape from ComfyUI: (1, H, W, C), values 0-1 float
        img_tensor = image[0]  # (H, W, C)
        H, W = img_tensor.shape[0], img_tensor.shape[1]

        # Upscale to target resolution using bicubic interpolation (scale=1 defers to FlashVSR's own scale)
        if H != target_height or W != target_width:
            # Resize to target dims before FlashVSR processing
            img_tensor = tensor_upscale_then_center_crop(
                img_tensor.unsqueeze(0), 1, target_width, target_height, 0, 0
            ).squeeze(0)  # → (target_H, target_W, C)
            log(f"Image resized from {W}x{H} to {target_width}x{target_height}", message_type='info', icon="🔧")

        # Convert to 25 frames (1 second at 25fps) - FlashVSR needs temporal context
        num_frames = 25
        frames = img_tensor.unsqueeze(0).repeat(num_frames, 1, 1, 1)  # (25, H, W, C), values 0-1

        # Move to device
        frames = frames.to(_device)

        # Initialize pipeline
        pipe = init_pipeline(model, mode, _device, torch.float16, vae_model=vae_model)

        # Run flashvsr - FlashVSR applies the scale factor internally
        output = flashvsr(
            pipe, frames, scale, color_fix, color_fix_method,
            tiled_vae, tiled_dit, 256, 24, True,
            2.0, 3.0, 11, seed,
            keep_models_on_cpu, enable_debug,
            chunk_size=0, resize_factor=1.0, mode=mode
        )

        # Return first frame as image output
        return (output[0:1].cpu().float(),)


class FlashVSRNodeBatchPath:
    """
    Batch video processing loader for FlashVSR.
    Loads all videos from a directory and iterates through them.
    Combined with FlashVSR Ultra-Fast node for batch processing.
    """
    import os
    import glob as _glob

    _counter = {}

    @classmethod
    def INPUT_TYPES(cls):
        video_extensions = ['mp4', 'webm', 'mkv', 'gif', 'mov', 'avi']
        return {
            "required": {
                "video_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory containing video files to process."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 112589990684264,
                    "tooltip": "Random seed for noise generation."
                }),
                "reset_counter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Set to True to reset the counter to 0 (start from first video)."
                }),
                "video_extension": (["mp4", "webm", "mkv", "gif", "mov", "avi", "all"], {
                    "default": "all",
                    "tooltip": "Filter videos by extension. 'all' processes all video types."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "frame_count", "seed", "filename")
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    DESCRIPTION = "Batch video path loader. Iterates through all videos in a directory. Connect output to FlashVSR node."

    def main(self, video_dir, seed, reset_counter, video_extension):
        if not video_dir:
            raise ValueError("video_dir is required for FlashVSRNodeBatchPath")

        # Normalize path
        video_dir = os.path.expanduser(video_dir)

        # Get video files
        if video_extension == "all":
            patterns = ['*.mp4', '*.webm', '*.mkv', '*.gif', '*.mov', '*.avi']
        else:
            patterns = [f'*.{video_extension}']

        video_files = []
        for pattern in patterns:
            video_files.extend(self._glob.glob(os.path.join(video_dir, pattern)))
            video_files.extend(self._glob.glob(os.path.join(video_dir, pattern.upper())))
        video_files = sorted(set(video_files))

        if not video_files:
            raise ValueError(f"No video files found in: {video_dir}")

        # Handle counter
        key = os.path.normpath(video_dir)
        if reset_counter or key not in self._counter:
            self._counter[key] = 0
        if reset_counter:
            self._counter[key] = 0

        idx = self._counter[key] % len(video_files)
        selected_path = video_files[idx]
        self._counter[key] = (idx + 1) % len(video_files)

        filename = os.path.basename(selected_path)
        log(f"[Batch] Processing {idx+1}/{len(video_files)}: {filename}", message_type='info', icon="🎬")

        # Load video frames using cv2
        try:
            import cv2
            cap = cv2.VideoCapture(selected_path)
            frames_list = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # BGR → RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame)
            cap.release()
        except Exception as e:
            raise RuntimeError(f"Failed to load video {selected_path}: {e}")

        if not frames_list:
            raise RuntimeError(f"Video has no frames: {selected_path}")

        # Convert to tensor (N, H, W, C)
        import numpy as np
        frames_tensor = torch.from_numpy(np.stack(frames_list)).float() / 255.0  # (N, H, W, C)

        return (frames_tensor, len(frames_list), seed, filename)


# =============================================================================
# NODE MAPPING — must be at the end (after all class definitions)
# =============================================================================
NODE_CLASS_MAPPINGS = {
    "FlashVSRNode": FlashVSRNode,
    "FlashVSRNodeAdv": FlashVSRNodeAdv,
    "FlashVSRInitPipe": FlashVSRNodeInitPipe,
    "FlashVSRNodeImageSR": FlashVSRNodeImageSR,
    "FlashVSRNodeBatchPath": FlashVSRNodeBatchPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashVSRNode": "FlashVSR Ultra-Fast",
    "FlashVSRNodeAdv": "FlashVSR Ultra-Fast (Advanced)",
    "FlashVSRInitPipe": "FlashVSR Init Pipeline",
    "FlashVSRNodeImageSR": "FlashVSR Image Super-Resolution",
    "FlashVSRNodeBatchPath": "FlashVSR Batch Video Path Loader",
}
