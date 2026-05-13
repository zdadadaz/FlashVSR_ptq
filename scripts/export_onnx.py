import torch
import torch.nn as nn
import os
import sys
import argparse
import numpy as np
import types
import onnx

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.wan_video_dit import WanModel

class WanModelONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x, timestep, context):
        # 1. Sinusoidal embedding and projection (Manual replication of top-level)
        from src.models.wan_video_dit import sinusoidal_embedding_1d
        t = self.model.time_embedding(sinusoidal_embedding_1d(self.model.freq_dim, timestep))
        t_mod = self.model.time_projection(t).unflatten(1, (6, self.model.dim))
        
        # 2. Context embedding
        context = self.model.text_embedding(context)
        
        # 3. Patchify
        x, (f, h, w) = self.model.patchify(x)
        
        # 4. RoPE (Concatenate along dim=-2 because last dim is real/imag [2])
        freqs = torch.cat([
            self.model.freqs[0][:f].view(f, 1, 1, -1, 2).expand(f, h, w, -1, 2),
            self.model.freqs[1][:h].view(1, h, 1, -1, 2).expand(f, h, w, -1, 2),
            self.model.freqs[2][:w].view(1, 1, w, -1, 2).expand(f, h, w, -1, 2)
        ], dim=-2).reshape(f * h * w, 1, -1, 2).to(x.device)
        
        # 5. Process blocks
        for block in self.model.blocks:
            x, _, _ = block(
                x, context, t_mod, freqs, f, h, w, 
                local_num=f//2, topk=64, is_full_block=True,
                kv_len=512 
            )

        # 6. Final Projection
        x = self.model.head(x, t)
        
        # 7. Unpatchify
        x = self.model.unpatchify(x, (f, h, w))
        return x

def export_flashvsr_onnx(ckpt_path, output_path, device="cpu", dummy_weights=False, opset_version=17):
    print(f"Loading model from {ckpt_path}...")

    import src.models.wan_video_dit as wan_video_dit

    # --- Monkey-patches for ONNX Compatibility ---

    # 1. Sinusoidal embedding (Avoid float64)
    def onnx_compatible_sinusoidal_embedding_1d(dim, position):
        # Use float32
        half_dim = dim // 2
        exponent = torch.arange(half_dim, dtype=torch.float32, device=position.device) / half_dim
        inv_freq = 1.0 / (10000 ** exponent)
        sinusoid = torch.outer(position.float(), inv_freq)
        x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        return x.to(position.dtype)

    wan_video_dit.sinusoidal_embedding_1d = onnx_compatible_sinusoidal_embedding_1d

    # 2. Precompute freqs (Avoid complex tensors)
    def onnx_compatible_precompute_freqs_cis(dim, end=1024, theta=10000.0):
        # Returns [end, dim//2, 2] representing cos and sin
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))
        freqs = torch.outer(torch.arange(end, dtype=torch.float32), freqs)
        return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)

    wan_video_dit.precompute_freqs_cis = onnx_compatible_precompute_freqs_cis

    # 3. RoPE Apply (Manual real-number arithmetic)
    def onnx_compatible_rope_apply(x, freqs, num_heads):
        # x: [B, S, N*D]
        # freqs: [S, 1, D/2, 2] (concatenated from 3D freqs)
        B, S, ND = x.shape
        D = ND // num_heads

        x = x.view(B, S, num_heads, D // 2, 2)
        x_real = x[..., 0]
        x_imag = x[..., 1]

        cos = freqs[..., 0]
        sin = freqs[..., 1]

        # (x_real + i*x_imag) * (cos + i*sin) = (x_real*cos - x_imag*sin) + i*(x_real*sin + x_imag*cos)
        res_real = x_real * cos - x_imag * sin
        res_imag = x_real * sin + x_imag * cos

        res = torch.stack([res_real, res_imag], dim=-1)
        return res.view(B, S, ND)

    wan_video_dit.rope_apply = onnx_compatible_rope_apply

    # 4. Flash Attention (Standard attention fallback)
    def onnx_compatible_flash_attention(q, k, v, num_heads, **kwargs):
        from einops import rearrange
        import torch.nn.functional as F
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        if kwargs.get("return_KV", False):
            return out, None, None
        return out

    wan_video_dit.flash_attention = onnx_compatible_flash_attention

    # 5. SelfAttention/CrossAttention forward fixes
    original_self_attention_forward = wan_video_dit.SelfAttention.forward
    def onnx_compatible_self_attention_forward(self, *args, **kwargs):
        result = original_self_attention_forward(self, *args, **kwargs)
        if isinstance(result, torch.Tensor):
            return result, None, None
        return result
    wan_video_dit.SelfAttention.forward = onnx_compatible_self_attention_forward

    def onnx_compatible_cross_attention_forward(self, x, context, **kwargs):
        from einops import rearrange
        import torch.nn.functional as F
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        return self.o(x)
    wan_video_dit.CrossAttention.forward = onnx_compatible_cross_attention_forward

    # --- Load Model ---

    model = WanModel(
        dim=1536, eps=1e-5, ffn_dim=8960, freq_dim=256, in_dim=16,
        num_heads=12, num_layers=30, out_dim=16,
        patch_size=(1, 2, 2), text_dim=4096
    )

    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    if dummy_weights:
        print("  [dummy_weights] Replacing all parameters with zeros...")
        for k in new_state_dict:
            new_state_dict[k] = torch.zeros_like(new_state_dict[k])

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device).eval()
    
    # Surgically patch every block's forward
    for block in model.blocks:
        def dit_block_forward_onnx(self, x, context, t_mod, freqs, f, h, w, **kwargs):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
            
            from src.models.wan_video_dit import modulate
            input_x = modulate(self.norm1(x), shift_msa, scale_msa)
            
            attn_res = self.self_attn(
                input_x, freqs, f, h, w, 
                is_full_block=True, is_stream=False, kv_len=512,
                topk=64, local_num=f//2
            )
            if isinstance(attn_res, tuple):
                self_attn_output = attn_res[0]
            else:
                self_attn_output = attn_res
                
            x = self.gate(x, gate_msa, self_attn_output)
            x = x + self.cross_attn(self.norm3(x), context)
            input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
            x = self.gate(x, gate_mlp, self.ffn(input_x))
            return x, None, None
            
        block.forward = types.MethodType(dit_block_forward_onnx, block)
    
    wrapper = WanModelONNXWrapper(model)
    
    # Dummy inputs
    dummy_x = torch.randn(1, 16, 4, 32, 32).to(device)
    dummy_t = torch.tensor([500.0]).to(device)
    dummy_context = torch.randn(1, 10, 4096).to(device)
    
    print(f"Exporting to {output_path}...")
    
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_t, dummy_context),
        output_path,
        input_names=["x", "timestep", "context"],
        output_names=["output"],
        dynamic_axes={
            "x": {0: "batch", 2: "frames", 3: "height", 4: "width"},
            "context": {0: "batch", 1: "seq_len"},
            "output": {0: "batch"}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )
    print("Export complete!")

    # Post-process: remove leading "/" from node names and tensor references (causes Netron display issues)
    onnx_model = onnx.load(output_path)
    for node in onnx_model.graph.node:
        if node.name.startswith("/"):
            node.name = node.name[1:]
        for i, inp in enumerate(node.input):
            if inp.startswith("/"):
                node.input[i] = inp[1:]
        for i, out in enumerate(node.output):
            if out.startswith("/"):
                node.output[i] = out[1:]
    onnx.save(onnx_model, output_path)

def test_onnx(onnx_path, device="cpu"):
    import onnxruntime as ort
    print(f"Testing ONNX model at {onnx_path}...")
    providers = ['CPUExecutionProvider']
    if device == "cuda" and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    x = np.random.randn(1, 16, 4, 32, 32).astype(np.float32)  # Match export
    t = np.array([500.0], dtype=np.float32)  # Match export
    c = np.random.randn(1, 10, 4096).astype(np.float32)  # Match export
    
    print("Running inference...")
    outputs = session.run(None, {"x": x, "timestep": t, "context": c})
    print(f"Inference successful! Output shape: {outputs[0].shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors")
    parser.add_argument("--output", type=str, default="models/FlashVSR-v1.1/flashvsr_v1.1.onnx")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--dummy_weights", action="store_true", help="Replace all model weights with zeros (produces single .onnx file without external data)")
    parser.add_argument("--opset_version", type=int, default=14, help="ONNX opset version (default: 14)")
    args = parser.parse_args()

    if not args.test_only:
        export_flashvsr_onnx(args.ckpt, args.output, dummy_weights=args.dummy_weights, opset_version=args.opset_version)

    if os.path.exists(args.output):
        test_onnx(args.output)
        if not args.dummy_weights:
            output_dir = os.path.dirname(args.output) or "."
            for f in os.listdir(output_dir):
                if f.startswith("_blocks.") or f.startswith("_Constant_") or f.startswith("_head_"):
                    try:
                        os.remove(os.path.join(output_dir, f))
                    except OSError:
                        pass
    else:
        print(f"Error: ONNX file {args.output} not found.")
