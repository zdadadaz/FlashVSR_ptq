"""
Split FlashVSR ONNX into multiple parts for Netron compatibility.
Each part is a complete subgraph that can be opened independently.
"""
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


class WanModelPart1(nn.Module):
    """Part 1: Patch embedding + Time embedding + Text embedding + First N blocks"""
    def __init__(self, model, num_blocks=10):
        super().__init__()
        self.model = model
        self.num_blocks = num_blocks

    def forward(self, x, timestep, context):
        from src.models.wan_video_dit import sinusoidal_embedding_1d
        t = self.model.time_embedding(sinusoidal_embedding_1d(self.model.freq_dim, timestep))
        t_mod = self.model.time_projection(t).unflatten(1, (6, self.model.dim))

        context = self.model.text_embedding(context)

        x, (f, h, w) = self.model.patchify(x)

        freqs = torch.cat([
            self.model.freqs[0][:f].view(f, 1, 1, -1, 2).expand(f, h, w, -1, 2),
            self.model.freqs[1][:h].view(1, h, 1, -1, 2).expand(f, h, w, -1, 2),
            self.model.freqs[2][:w].view(1, 1, w, -1, 2).expand(f, h, w, -1, 2)
        ], dim=-2).reshape(f * h * w, 1, -1, 2).to(x.device)

        for i, block in enumerate(self.model.blocks[:self.num_blocks]):
            x, _, _ = block(
                x, context, t_mod, freqs, f, h, w,
                local_num=f//2, topk=64, is_full_block=True,
                kv_len=512
            )

        return x, (f, h, w), t_mod, freqs, context


class WanModelPart2(nn.Module):
    """Part 2: Middle blocks (Part 1 output -> Part 3 output)"""
    def __init__(self, model, start_block, num_blocks):
        super().__init__()
        self.model = model
        self.start_block = start_block
        self.num_blocks = num_blocks

    def forward(self, x, f, h, w, t_mod, freqs, context):
        for i, block in enumerate(self.model.blocks[self.start_block:self.start_block + self.num_blocks]):
            x, _, _ = block(
                x, context, t_mod, freqs, f, h, w,
                local_num=f//2, topk=64, is_full_block=True,
                kv_len=512
            )
        return x


class WanModelPart3(nn.Module):
    """Part 3: Final blocks + Head + Unpatchify"""
    def __init__(self, model, start_block):
        super().__init__()
        self.model = model
        self.start_block = start_block

    def forward(self, x, f, h, w, t_mod, freqs, context):
        for block in self.model.blocks[self.start_block:]:
            x, _, _ = block(
                x, context, t_mod, freqs, f, h, w,
                local_num=f//2, topk=64, is_full_block=True,
                kv_len=512
            )
        x = self.model.head(x, t_mod.squeeze(0).mean(dim=0, keepdim=True))
        x = self.model.unpatchify(x, (f, h, w))
        return x


def export_flashvsr_part(ckpt_path, output_dir, part_num, device="cpu"):
    """Export a specific part of FlashVSR."""
    print(f"Loading model from {ckpt_path}...")

    import src.models.wan_video_dit as wan_video_dit

    # Monkey-patches (same as main export)
    def onnx_compatible_sinusoidal_embedding_1d(dim, position):
        half_dim = dim // 2
        exponent = torch.arange(half_dim, dtype=torch.float32, device=position.device) / half_dim
        inv_freq = 1.0 / (10000 ** exponent)
        sinusoid = torch.outer(position.float(), inv_freq)
        x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        return x.to(position.dtype)

    wan_video_dit.sinusoidal_embedding_1d = onnx_compatible_sinusoidal_embedding_1d

    def onnx_compatible_precompute_freqs_cis(dim, end=1024, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))
        freqs = torch.outer(torch.arange(end, dtype=torch.float32), freqs)
        return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)

    wan_video_dit.precompute_freqs_cis = onnx_compatible_precompute_freqs_cis

    def onnx_compatible_rope_apply(x, freqs, num_heads):
        B, S, ND = x.shape
        D = ND // num_heads
        x = x.view(B, S, num_heads, D // 2, 2)
        x_real = x[..., 0]
        x_imag = x[..., 1]
        cos = freqs[..., 0]
        sin = freqs[..., 1]
        res_real = x_real * cos - x_imag * sin
        res_imag = x_real * sin + x_imag * cos
        res = torch.stack([res_real, res_imag], dim=-1)
        return res.view(B, S, ND)

    wan_video_dit.rope_apply = onnx_compatible_rope_apply

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

    # Load model
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

    print("  [dummy_weights] Replacing all parameters with zeros...")
    for k in new_state_dict:
        new_state_dict[k] = torch.zeros_like(new_state_dict[k])

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device).eval()

    # Patch every block's forward (same as main export)
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

    # Dummy inputs for tracing
    dummy_x = torch.randn(1, 16, 4, 32, 32).to(device)
    dummy_t = torch.tensor([500.0]).to(device)
    dummy_context = torch.randn(1, 10, 4096).to(device)

    os.makedirs(output_dir, exist_ok=True)

    if part_num == 1:
        # Part 1: patch_embed + time_embed + text_embed + first 10 blocks
        print("Exporting Part 1: patch_embed + time_embed + text_embed + first 10 blocks...")
        wrapper = WanModelPart1(model, num_blocks=10)
        dummy_out = wrapper(dummy_x, dummy_t, dummy_context)
        x_out, (f, h, w), t_mod, freqs, context = dummy_out

        torch.onnx.export(
            wrapper,
            (dummy_x, dummy_t, dummy_context),
            os.path.join(output_dir, "flashvsr_part1.onnx"),
            input_names=["x", "timestep", "context"],
            output_names=["x_out", "fhw", "t_mod", "freqs", "context_out"],
            dynamic_axes={
                "x": {0: "batch", 2: "frames", 3: "height", 4: "width"},
                "context": {0: "batch", 1: "seq_len"},
            },
            opset_version=14,
            do_constant_folding=True
        )

    elif part_num == 2:
        # Part 2: blocks 10-20
        print("Exporting Part 2: blocks 10-20...")
        wrapper = WanModelPart2(model, start_block=10, num_blocks=10)

        # Need to trace to get intermediate tensors
        with torch.no_grad():
            from src.models.wan_video_dit import sinusoidal_embedding_1d
            t = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, dummy_t))
            t_mod = model.time_projection(t).unflatten(1, (6, model.dim))
            context = model.text_embedding(dummy_context)
            x, (f, h, w) = model.patchify(dummy_x)
            freqs = torch.cat([
                model.freqs[0][:f].view(f, 1, 1, -1, 2).expand(f, h, w, -1, 2),
                model.freqs[1][:h].view(1, h, 1, -1, 2).expand(f, h, w, -1, 2),
                model.freqs[2][:w].view(1, 1, w, -1, 2).expand(f, h, w, -1, 2)
            ], dim=-2).reshape(f * h * w, 1, -1, 2).to(x.device)
            for i in range(10):
                x, _, _ = model.blocks[i](x, context, t_mod, freqs, f, h, w, local_num=f//2, topk=64, is_full_block=True, kv_len=512)

        x_out = wrapper(x, f, h, w, t_mod, freqs, context)

        torch.onnx.export(
            wrapper,
            (x, f, h, w, t_mod, freqs, context),
            os.path.join(output_dir, "flashvsr_part2.onnx"),
            input_names=["x_in", "f", "h", "w", "t_mod", "freqs", "context"],
            output_names=["x_out"],
            opset_version=14,
            do_constant_folding=True
        )

    elif part_num == 3:
        # Part 3: blocks 20-30 + head
        print("Exporting Part 3: blocks 20-30 + head...")
        wrapper = WanModelPart3(model, start_block=20)

        # Trace to get intermediate tensors
        with torch.no_grad():
            from src.models.wan_video_dit import sinusoidal_embedding_1d
            t = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, dummy_t))
            t_mod = model.time_projection(t).unflatten(1, (6, model.dim))
            context = model.text_embedding(dummy_context)
            x, (f, h, w) = model.patchify(dummy_x)
            freqs = torch.cat([
                model.freqs[0][:f].view(f, 1, 1, -1, 2).expand(f, h, w, -1, 2),
                model.freqs[1][:h].view(1, h, 1, -1, 2).expand(f, h, w, -1, 2),
                model.freqs[2][:w].view(1, 1, w, -1, 2).expand(f, h, w, -1, 2)
            ], dim=-2).reshape(f * h * w, 1, -1, 2).to(x.device)
            for i in range(20):
                x, _, _ = model.blocks[i](x, context, t_mod, freqs, f, h, w, local_num=f//2, topk=64, is_full_block=True, kv_len=512)

        x_out = wrapper(x, f, h, w, t_mod, freqs, context)

        torch.onnx.export(
            wrapper,
            (x, f, h, w, t_mod, freqs, context),
            os.path.join(output_dir, "flashvsr_part3.onnx"),
            input_names=["x_in", "f", "h", "w", "t_mod", "freqs", "context"],
            output_names=["output"],
            dynamic_axes={"output": {0: "batch"}},
            opset_version=14,
            do_constant_folding=True
        )

    print(f"Part {part_num} exported!")

    # Post-process: remove leading "/" from node names and tensor references
    import google.protobuf
    output_path = os.path.join(output_dir, f"flashvsr_part{part_num}.onnx")
    model_fixed = onnx.load(output_path)
    for node in model_fixed.graph.node:
        if node.name.startswith("/"):
            node.name = node.name[1:]
        for i, inp in enumerate(node.input):
            if inp.startswith("/"):
                node.input[i] = inp[1:]
        for i, out in enumerate(node.output):
            if out.startswith("/"):
                node.output[i] = out[1:]
    onnx.save(model_fixed, output_path)
    print(f"  Cleaned / prefixes in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split FlashVSR ONNX into parts")
    parser.add_argument("--ckpt", type=str, default="models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors")
    parser.add_argument("--output_dir", type=str, default="models/FlashVSR-v1.1/split")
    parser.add_argument("--part", type=int, choices=[1, 2, 3], required=True, help="Part number to export")
    args = parser.parse_args()

    export_flashvsr_part(args.ckpt, args.output_dir, args.part)
