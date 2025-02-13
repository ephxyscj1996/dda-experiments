import torch
import math

from .utils import apply_rope, apply_rope_x


class Rope_DDAK(torch.nn.Module):

    def __init__(self, d_model, n_heads, max_len=1024, rope_theta=10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.q = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))
        self.k = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))
        self.v = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))
        self.wo = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

        # RoPE
        self.max_seq_len = max_len
        self.rope_theta = rope_theta

        # https://github.com/lucidrains/rotary-embedding-torch/tree/main
        # visualize emb later to make sure it looks ok
        freqs = 1.0 / (rope_theta ** (torch.arange(0, self.dh, 2).float() / self.dh))
        emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]

        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        # This is like a parameter but its a constant so we can use register_buffer
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(self, x, y, kv_cache=None, past_length=0):
        # queries, keys, and values
        B, S, D = x.shape
        Q = x @ self.q.T # B, S, D
        K = y            # B, S, D
        V = y @ self.v.T # B, S, D

        K1 = x @ self.k.T # B, S, D
        V1 = x @ self.wo.T # B, S, D

        # split into multiple heads
        q_heads = Q.view(B, S, self.n_heads, self.dh).transpose(1,2)
        k_heads = K.view(B, S, self.n_heads, self.dh).transpose(1,2)
        v_heads = V.view(B, S, self.n_heads, self.dh).transpose(1,2)
        k1_heads = K1.view(B, S, self.n_heads, self.dh).transpose(1,2)
        v1_heads = V1.view(B, S, self.n_heads, self.dh).transpose(1,2)

        ## Apply RoPE
        cos = self.cos_cached[:, :, past_length:past_length+S, :self.dh//2].repeat(1, 1, 1, 2)
        sin = self.sin_cached[:, :, past_length:past_length+S, :self.dh//2].repeat(1, 1, 1, 2)
        q_heads, k_heads = apply_rope(q_heads, k_heads, cos, sin)
        _, k1_heads = apply_rope(q_heads, k_heads, cos, sin)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_heads = torch.cat([k_cache, k_heads], dim=2)
            v_heads = torch.cat([v_cache, v_heads], dim=2)

        k_heads = torch.cat([k_heads, k1_heads], dim=2)
        v_heads = torch.cat([v_heads, v1_heads], dim=2)
        S_full = k_heads.size(2)
        # make attention mask
        mask = torch.ones((S,S_full), device=x.device)
        mask = torch.tril(mask, diagonal=past_length)
        mask[:,-S:] = torch.eye(S, device = x.device)
        mask = mask[None, None, :, :]

        sq_mask = mask == 1
        
        # attention
        x = torch.nn.functional.scaled_dot_product_attention(
            q_heads, k_heads, v_heads,
            attn_mask=sq_mask
        )

        x = x.transpose(1, 2).reshape(B, S, D)

        # apply projection
        #x = x @ self.wo.T

        return x, (k_heads[:,:,:-S,:], None)