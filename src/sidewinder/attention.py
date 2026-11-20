"""EMA running-block grouped-query attention for SideWinder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ema import (
    attention_interval,
    block_ema_running,
    causal_ema,
    causal_ema_state,
)


class TemporalGQA(nn.Module):
    """Causal grouped-query attention over a temporal Mamba signal."""

    def __init__(self, d_model, n_heads, n_kv_heads,
                 query_chunk=4096):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by attention heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.query_chunk = query_chunk
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(
            d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(
            d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, length, d_model = x.shape
        q = self.q_proj(x).view(
            b, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(
            b, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(
            b, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True)
        out = out.transpose(1, 2).reshape(b, length, d_model)
        return self.out_proj(out)

    def _expand_kv(self, x):
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            x = x.repeat_interleave(repeat, dim=1)
        return x

    def forward_running(self, query, key_frozen, value_frozen,
                        key_running, value_running, interval):
        b, length, d_model = query.shape
        n_blocks = key_frozen.shape[1]
        q = self.q_proj(query).view(
            b, length, self.n_heads, self.head_dim).transpose(1, 2)
        kf = self.k_proj(key_frozen).view(
            b, n_blocks, self.n_kv_heads, self.head_dim).transpose(1, 2)
        vf = self.v_proj(value_frozen).view(
            b, n_blocks, self.n_kv_heads, self.head_dim).transpose(1, 2)
        kr = self.k_proj(key_running).view(
            b, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        vr = self.v_proj(value_running).view(
            b, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        kf = self._expand_kv(kf)
        vf = self._expand_kv(vf)
        kr = self._expand_kv(kr)
        vr = self._expand_kv(vr)
        scale = self.head_dim ** -0.5
        scores_f = torch.einsum("bhtd,bhkd->bhtk", q, kf) * scale
        scores_r = torch.einsum("bhtd,bhtd->bht", q, kr) * scale
        token_block = torch.arange(length, device=query.device) // interval
        key_block = torch.arange(n_blocks, device=query.device)
        allowed = key_block.unsqueeze(0) < token_block.unsqueeze(1)
        scores_f = scores_f.masked_fill(
            ~allowed.view(1, 1, length, n_blocks), float("-inf"))
        weights = torch.softmax(
            torch.cat([scores_f, scores_r.unsqueeze(-1)], dim=-1), dim=-1)
        out = torch.einsum(
            "bhtk,bhkd->bhtd", weights[..., :n_blocks], vf)
        out = out + weights[..., n_blocks].unsqueeze(-1) * vr
        out = out.transpose(1, 2).reshape(b, length, d_model)
        return self.out_proj(out)

    def forward_running_chunked(self, query, key_frozen, value_frozen,
                                key_running, value_running, interval,
                                chunk=None, block_offset=0):
        chunk = self.query_chunk if chunk is None else int(chunk)
        b, length, d_model = query.shape
        n_blocks = key_frozen.shape[1]
        kf = self.k_proj(key_frozen).view(
            b, n_blocks, self.n_kv_heads, self.head_dim).transpose(1, 2)
        vf = self.v_proj(value_frozen).view(
            b, n_blocks, self.n_kv_heads, self.head_dim).transpose(1, 2)
        kf = self._expand_kv(kf)
        vf = self._expand_kv(vf)
        scale = self.head_dim ** -0.5
        key_block = torch.arange(n_blocks, device=query.device)
        parts = []
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            q = self.q_proj(query[:, start:end]).view(
                b, end - start, self.n_heads,
                self.head_dim).transpose(1, 2)
            kr = self.k_proj(key_running[:, start:end]).view(
                b, end - start, self.n_kv_heads,
                self.head_dim).transpose(1, 2)
            vr = self.v_proj(value_running[:, start:end]).view(
                b, end - start, self.n_kv_heads,
                self.head_dim).transpose(1, 2)
            kr = self._expand_kv(kr)
            vr = self._expand_kv(vr)
            scores_f = torch.einsum("bhtd,bhkd->bhtk", q, kf) * scale
            scores_r = torch.einsum("bhtd,bhtd->bht", q, kr) * scale
            token_block = (block_offset + torch.arange(
                start, end, device=query.device)) // interval
            allowed = key_block.unsqueeze(0) < token_block.unsqueeze(1)
            scores_f = scores_f.masked_fill(
                ~allowed.view(1, 1, end - start, n_blocks),
                float("-inf"))
            weights = torch.softmax(
                torch.cat([scores_f, scores_r.unsqueeze(-1)], dim=-1),
                dim=-1)
            out = torch.einsum(
                "bhtk,bhkd->bhtd", weights[..., :n_blocks], vf)
            out = out + weights[..., n_blocks].unsqueeze(-1) * vr
            out = out.transpose(1, 2).reshape(b, end - start, d_model)
            parts.append(self.out_proj(out))
        return torch.cat(parts, dim=1)


class AttentionTransition(nn.Module):
    """EMA running-block attention with a residual and learned gate."""

    def __init__(self, d=64, attn_heads=4, attn_kv_heads=1,
                 attn_ema_beta=0.9, query_chunk=4096, ema_chunk=1024):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d)
        self.attn = TemporalGQA(
            d, attn_heads, attn_kv_heads, query_chunk=query_chunk)
        self.attn_ema_beta = float(attn_ema_beta)
        self.attn_interval = attention_interval(self.attn_ema_beta)
        self.ema_chunk = ema_chunk
        self.context_gate = nn.Linear(2 * d, d)
        with torch.no_grad():
            self.context_gate.weight.zero_()
            self.context_gate.bias.zero_()

    def forward(self, h, chunked=False):
        source = self.norm_attn(h)
        temporal = causal_ema(
            source, self.attn_ema_beta, chunk=self.ema_chunk)
        running, frozen = block_ema_running(
            temporal, self.attn_ema_beta, self.attn_interval)
        if chunked:
            context = self.attn.forward_running_chunked(
                temporal, frozen, frozen, running, running,
                self.attn_interval, self.attn.query_chunk)
        else:
            context = self.attn.forward_running(
                temporal, frozen, frozen, running, running,
                self.attn_interval)
        gate = 1.0 + torch.tanh(
            self.context_gate(torch.cat([h, context], dim=-1)))
        return h + gate * context

    def forward_state(self, h, state=None, chunk=None):
        source = self.norm_attn(h)
        ema_state = None if state is None else state[1]
        temporal, new_ema = causal_ema_state(
            source, self.attn_ema_beta, ema_state, chunk=self.ema_chunk)
        running, frozen_new = block_ema_running(
            temporal, self.attn_ema_beta, self.attn_interval)
        if state is None:
            frozen_all = frozen_new
            block_offset = 0
        else:
            frozen_all = torch.cat([state[0], frozen_new], dim=1)
            block_offset = int(state[2])
        context = self.attn.forward_running_chunked(
            temporal, frozen_all, frozen_all, running, running,
            self.attn_interval,
            self.attn.query_chunk if chunk is None else int(chunk),
            block_offset=block_offset)
        gate = 1.0 + torch.tanh(
            self.context_gate(torch.cat([h, context], dim=-1)))
        return h + gate * context, (
            frozen_all, new_ema, block_offset + h.shape[1])
