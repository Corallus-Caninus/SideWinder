"""SideWinder: Mamba/MLP stages with EMA running-block GQA attention."""

from .attention import AttentionTransition, TemporalGQA
from .blocks import MLP, Mamba2Block, TinyLMStage
from .config import SideWinderConfig
from .ema import (
    attention_interval,
    block_ema_running,
    causal_ema,
    causal_ema_state,
)
from .model import TinyLM

__all__ = [
    "AttentionTransition",
    "MLP",
    "Mamba2Block",
    "SideWinderConfig",
    "TemporalGQA",
    "TinyLM",
    "TinyLMStage",
    "attention_interval",
    "block_ema_running",
    "causal_ema",
    "causal_ema_state",
]
