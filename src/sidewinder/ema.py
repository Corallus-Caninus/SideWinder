"""Stable causal EMA and running-block helpers for SideWinder attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_EMA_CHUNK = 1024


def causal_ema(x: torch.Tensor, beta: float, chunk: int = _EMA_CHUNK) -> torch.Tensor:
    """Stable causal exponential moving average over the sequence dimension."""
    beta = min(max(float(beta), 0.0), 1.0)
    if beta <= 0.0:
        return x
    b, length, _ = x.shape
    if length == 0:
        return x
    if beta >= 1.0:
        return torch.cumsum(x, dim=1) / torch.arange(
            1, length + 1, device=x.device, dtype=x.dtype).view(1, length, 1)
    xd = x.to(torch.float64)
    log_beta = math.log(beta)
    one_minus = 1.0 - beta
    state = torch.zeros(b, x.shape[2], device=x.device, dtype=torch.float64)
    parts = []
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        n = end - start
        t = torch.arange(n, device=x.device, dtype=torch.float64)
        logw = t * log_beta
        inv_w = torch.exp(-logw).view(1, n, 1)
        c = torch.cumsum(xd[:, start:end] * inv_w, dim=1)
        block = one_minus * torch.exp(logw).view(1, n, 1) * c
        carry = state.unsqueeze(1) * torch.exp(
            (t + 1).view(1, n, 1) * log_beta)
        block = block + carry
        parts.append(block.to(x.dtype))
        state = block[:, -1, :]
    return torch.cat(parts, dim=1)


def causal_ema_state(x: torch.Tensor, beta: float, state=None,
                     chunk: int = _EMA_CHUNK):
    """Stable causal EMA for one chunk with an optional incoming state."""
    beta = min(max(float(beta), 0.0), 0.999999)
    if beta <= 0.0:
        return x, state
    b, length, d = x.shape
    if length == 0:
        return x, state
    xd = x.to(torch.float64)
    log_beta = math.log(beta)
    one_minus = 1.0 - beta
    if state is None:
        state = torch.zeros(b, d, device=x.device, dtype=torch.float64)
    parts = []
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        n = end - start
        t = torch.arange(n, device=x.device, dtype=torch.float64)
        logw = t * log_beta
        inv_w = torch.exp(-logw).view(1, n, 1)
        c = torch.cumsum(xd[:, start:end] * inv_w, dim=1)
        block = one_minus * torch.exp(logw).view(1, n, 1) * c
        carry = state.unsqueeze(1) * torch.exp(
            (t + 1).view(1, n, 1) * log_beta)
        block = block + carry
        parts.append(block.to(x.dtype))
        state = block[:, -1, :]
    return torch.cat(parts, dim=1), state


def attention_interval(beta: float) -> int:
    """Block interval locked to the EMA time constant."""
    beta = min(max(float(beta), 0.0), 0.999999)
    return max(1, int(round(1.0 / (1.0 - beta))))


def block_ema_running(x: torch.Tensor, beta: float, interval: int):
    """Per-block running EMA states before and after each token."""
    b, length, d = x.shape
    n_blocks = (length + interval - 1) // interval
    padded = n_blocks * interval
    if padded > length:
        x = F.pad(x, (0, 0, 0, padded - length))
    xb = x.view(b, n_blocks, interval, d)
    state = torch.zeros(b, n_blocks, d, device=x.device, dtype=x.dtype)
    pre = torch.empty_like(xb)
    starts = torch.arange(n_blocks, device=x.device) * interval
    for k in range(interval):
        pre[:, :, k, :] = state
        valid = ((starts + k) < length).view(1, n_blocks, 1)
        update = beta * state + (1.0 - beta) * xb[:, :, k, :]
        state = torch.where(valid, update, state)
    pre = pre.reshape(b, padded, d)[:, :length, :]
    return pre, state
