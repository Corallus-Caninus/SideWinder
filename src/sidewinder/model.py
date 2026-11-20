"""SideWinder language model and long-context training helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionTransition
from .blocks import TinyLMStage
from .config import SideWinderConfig


class TinyLM(nn.Module):
    """SideWinder byte-level causal language model.

    The default model is four SideWinder stages with three EMA running-block
    attention transitions. The final stage has no outgoing attention and goes
    directly to the output head.
    """

    def __init__(self, d=64, hidden=90, n_layers=4, attn_ema_beta=0.9,
                 vocab=256, attn_heads=4, attn_kv_heads=1,
                 mamba_d_state=32, mamba_conv=3, mamba_scan_chunk=64,
                 ema_chunk=1024, query_chunk=4096,
                 long_context=False, long_context_chunk=2000,
                 stateful=False, debug=False):
        super().__init__()
        self.vocab = int(vocab)
        self.long_context = bool(long_context)
        self.long_context_chunk = int(long_context_chunk)
        self.stateful = bool(stateful)
        self.debug = bool(debug)
        self.attn_heads = int(attn_heads)
        self.attn_kv_heads = int(attn_kv_heads)
        self.attn_ema_beta = float(attn_ema_beta)
        self.embed = nn.Embedding(self.vocab, d)
        self.stages = nn.ModuleList([
            TinyLMStage(
                d, hidden, d_state=mamba_d_state, d_conv=mamba_conv,
                scan_chunk=mamba_scan_chunk)
            for _ in range(n_layers)
        ])
        self.transitions = nn.ModuleList([
            AttentionTransition(
                d, attn_heads=attn_heads, attn_kv_heads=attn_kv_heads,
                attn_ema_beta=attn_ema_beta, query_chunk=query_chunk,
                ema_chunk=ema_chunk)
            for _ in range(max(0, n_layers - 1))
        ])
        self.norm_out = nn.LayerNorm(d)
        self.out = nn.Linear(d, self.vocab)

    @classmethod
    def from_config(cls, config: SideWinderConfig):
        return cls(**config.__dict__)

    def _mem(self, label):
        if not (self.debug and torch.cuda.is_available()):
            return
        print(f"[mem] {label} "
              f"alloc={torch.cuda.memory_allocated()/2**20:.0f}MiB "
              f"reserved={torch.cuda.memory_reserved()/2**20:.0f}MiB",
              flush=True)

    def forward_hidden(self, x, use_checkpoint=False, chunked_attn=False):
        h = self.embed(x)
        self._mem("embed")
        for i, stage in enumerate(self.stages):
            if use_checkpoint and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    stage, h, use_reentrant=True)
            else:
                h = stage(h)
            if i < len(self.transitions):
                if use_checkpoint and torch.is_grad_enabled():
                    h = torch.utils.checkpoint.checkpoint(
                        self.transitions[i], h, chunked_attn,
                        use_reentrant=True)
                else:
                    h = self.transitions[i](h, chunked_attn)
            self._mem(f"after stage {i}")
        return h

    def _head_ce(self, h, y):
        logits = self.out(self.norm_out(h)).reshape(-1, self.vocab)
        return F.cross_entropy(logits, y.reshape(-1), reduction="sum")

    def chunked_grad_loss(self, x, y, chunk=None):
        """Detached-state chunked BPTT: generate grads without holding T graph."""
        chunk = self.long_context_chunk if chunk is None else int(chunk)
        if self.transitions:
            interval = self.transitions[0].attn_interval
            chunk = max(interval, int(round(chunk / interval)) * interval)
        m_states = [None] * len(self.stages)
        a_states = [None] * len(self.transitions)
        self.zero_grad(set_to_none=True)
        total = 0.0
        tokens = 0
        n_chunks = 0
        for start in range(0, x.shape[1], chunk):
            end = min(start + chunk, x.shape[1])
            h = self.embed(x[:, start:end])
            for i, stage in enumerate(self.stages):
                y_hat, m_states[i] = stage.mamba.forward_state(
                    stage.norm_m(h), m_states[i])
                h = h + y_hat
                h = h + stage.mlp1(stage.norm1(h))
                h = h + stage.mlp2(stage.norm2(h))
                h = h + stage.mlp3(stage.norm3(h))
                if i < len(self.transitions):
                    h, a_states[i] = self.transitions[i].forward_state(
                        h, a_states[i], chunk)
            yb = y[:, start:end] if y.ndim > 1 else y[start:end]
            loss_chunk = self._head_ce(h, yb)
            loss_chunk.backward()
            total += float(loss_chunk.detach())
            tokens += end - start
            n_chunks += 1
            for j, st in enumerate(m_states):
                if st is not None:
                    m_states[j] = (st[0].detach(), st[1].detach())
            for j, st in enumerate(a_states):
                if st is not None:
                    a_states[j] = (st[0].detach(), st[1].detach(), st[2])
        return total / max(1, tokens), n_chunks

    def chunked_loss_stateful(self, x, y, chunk=None):
        chunk = self.long_context_chunk if chunk is None else int(chunk)
        if self.transitions:
            interval = self.transitions[0].attn_interval
            chunk = max(interval, int(round(chunk / interval)) * interval)
        m_states = [None] * len(self.stages)
        a_states = [None] * len(self.transitions)
        total = x.new_zeros(())
        tokens = 0
        for start in range(0, x.shape[1], chunk):
            end = min(start + chunk, x.shape[1])
            h = self.embed(x[:, start:end])
            for i, stage in enumerate(self.stages):
                u = stage.norm_m(h)
                y_hat, m_states[i] = stage.mamba.forward_state(
                    u, m_states[i])
                h = h + y_hat
                h = h + stage.mlp1(stage.norm1(h))
                h = h + stage.mlp2(stage.norm2(h))
                h = h + stage.mlp3(stage.norm3(h))
                if i < len(self.transitions):
                    h, a_states[i] = self.transitions[i].forward_state(
                        h, a_states[i], chunk)
            yb = y[:, start:end] if y.ndim > 1 else y[start:end]
            total = total + self._head_ce(h, yb)
            tokens += end - start
        return total / max(1, tokens)

    @torch.no_grad()
    def chunked_eval_stateful(self, x, y, chunk=None):
        """Stateful chunked validation without materializing full logits."""
        chunk = self.long_context_chunk if chunk is None else int(chunk)
        if self.transitions:
            interval = self.transitions[0].attn_interval
            chunk = max(interval, int(round(chunk / interval)) * interval)
        m_states = [None] * len(self.stages)
        a_states = [None] * len(self.transitions)
        total = 0.0
        correct = 0
        tokens = 0
        for start in range(0, x.shape[1], chunk):
            end = min(start + chunk, x.shape[1])
            h = self.embed(x[:, start:end])
            for i, stage in enumerate(self.stages):
                y_hat, m_states[i] = stage.mamba.forward_state(
                    stage.norm_m(h), m_states[i])
                h = h + y_hat
                h = h + stage.mlp1(stage.norm1(h))
                h = h + stage.mlp2(stage.norm2(h))
                h = h + stage.mlp3(stage.norm3(h))
                if i < len(self.transitions):
                    h, a_states[i] = self.transitions[i].forward_state(
                        h, a_states[i], chunk)
            yb = y[:, start:end] if y.ndim > 1 else y[start:end]
            logits = self.out(self.norm_out(h)).reshape(-1, self.vocab)
            target = yb.reshape(-1)
            total += float(F.cross_entropy(
                logits, target, reduction="sum"))
            correct += int((logits.argmax(-1) == target).sum())
            tokens += end - start
        tokens = max(1, tokens)
        return total / tokens, correct / tokens

    def chunked_loss(self, x, y, chunk=None):
        chunk = self.long_context_chunk if chunk is None else int(chunk)
        if self.stateful:
            return self.chunked_loss_stateful(x, y, chunk)
        h = self.forward_hidden(
            x, use_checkpoint=self.long_context,
            chunked_attn=self.long_context)
        if self.long_context and torch.cuda.is_available():
            torch.cuda.empty_cache()
        total = h.new_zeros(())
        tokens = 0
        for start in range(0, x.shape[1], chunk):
            end = min(start + chunk, x.shape[1])
            hb = h[:, start:end]
            yb = y[:, start:end] if y.ndim > 1 else y[start:end]
            total = total + torch.utils.checkpoint.checkpoint(
                self._head_ce, hb, yb, use_reentrant=True)
            tokens += end - start
            if self.debug and start == 0:
                self._mem("head chunk")
        return total / max(1, tokens)

    def forward(self, x):
        h = self.forward_hidden(x)
        return self.out(self.norm_out(h))
