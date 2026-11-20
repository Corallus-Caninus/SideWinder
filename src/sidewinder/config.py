"""Configuration for the SideWinder byte-level language model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SideWinderConfig:
    """Architecture and runtime defaults for :class:`sidewinder.TinyLM`."""

    vocab: int = 256
    d_model: int = 64
    d_hidden: int = 90
    n_layers: int = 4
    attn_heads: int = 4
    attn_kv_heads: int = 1
    attn_ema_beta: float = 0.9
    mamba_d_state: int = 32
    mamba_conv: int = 3
    mamba_scan_chunk: int = 64
    ema_chunk: int = 1024
    query_chunk: int = 4096
    long_context: bool = False
    long_context_chunk: int = 2000
    stateful: bool = False
    debug: bool = False

    def __post_init__(self) -> None:
        if self.vocab < 1:
            raise ValueError("vocab must be >= 1")
        if self.d_model < 1:
            raise ValueError("d_model must be >= 1")
        if self.d_hidden < 1:
            raise ValueError("d_hidden must be >= 1")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.d_model % self.attn_heads != 0:
            raise ValueError("d_model must be divisible by attn_heads")
        if self.attn_heads % self.attn_kv_heads != 0:
            raise ValueError("attn_heads must be divisible by attn_kv_heads")
        if not 0.0 <= self.attn_ema_beta <= 1.0:
            raise ValueError("attn_ema_beta must be in [0, 1]")
        if self.mamba_d_state < 1:
            raise ValueError("mamba_d_state must be >= 1")
        if self.mamba_conv < 1:
            raise ValueError("mamba_conv must be >= 1")
        if self.mamba_scan_chunk < 1:
            raise ValueError("mamba_scan_chunk must be >= 1")
        if self.ema_chunk < 1:
            raise ValueError("ema_chunk must be >= 1")
        if self.query_chunk < 1:
            raise ValueError("query_chunk must be >= 1")
        if self.long_context_chunk < 1:
            raise ValueError("long_context_chunk must be >= 1")
