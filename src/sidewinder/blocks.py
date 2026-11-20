"""Core SideWinder sequence and feature-mixing blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba2Block(nn.Module):
    """A small causal Mamba-style SSM block with a chunked scan."""

    def __init__(self, d, d_inner, d_state=32, d_conv=3, scan_chunk=64):
        super().__init__()
        self.d = d
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.scan_chunk = scan_chunk
        n_in = 2 * d_inner + 2 * d_state + 1
        self.in_proj = nn.Linear(d, n_in, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.act = nn.SiLU()
        self.norm = nn.RMSNorm(d_inner)
        self.A_log = nn.Parameter(torch.zeros(1, d_state))
        self.dt_bias = nn.Parameter(torch.full((1,), -4.0))
        self.out_proj = nn.Linear(d_inner, d, bias=False)

    def forward(self, u):
        b, length, _ = u.shape
        xz = self.in_proj(u)
        x, z, bc, dtc = xz.split(
            [self.d_inner, self.d_inner, 2 * self.d_state, 1], dim=-1)
        Bv, c = bc.split([self.d_state, self.d_state], dim=-1)
        xc = self.conv1d(x.permute(0, 2, 1))[:, :, :length]
        x = self.act(self.norm(xc.permute(0, 2, 1)))
        dt = F.softplus(dtc + self.dt_bias).clamp_max(0.1)
        logA = -torch.exp(self.A_log.clamp(-10.0, 0.0))
        m = torch.cumsum(dt, dim=1)
        h_prev = torch.zeros(
            b, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for j in range(0, length, self.scan_chunk):
            hi = min(j + self.scan_chunk, length)
            ml = m[:, j:hi]
            rel_in = (ml - m[:, j:j + 1]).unsqueeze(-1)
            P = torch.exp(logA.unsqueeze(0).unsqueeze(0) * rel_in)
            q = x[:, j:hi].unsqueeze(-1) * Bv[:, j:hi].unsqueeze(2) / P
            local = P * torch.cumsum(q, dim=1)
            mjm1 = m[:, max(j - 1, 0):max(j - 1, 0) + 1]
            carry = torch.exp(
                logA.unsqueeze(0).unsqueeze(0)
                * (ml - mjm1).unsqueeze(-1)) * h_prev.unsqueeze(1)
            h = carry + local
            ys.append((h * c[:, j:hi].unsqueeze(2)).sum(-1))
            h_prev = h[:, -1]
        y = torch.cat(ys, dim=1)
        return self.out_proj(y * F.silu(z))

    def forward_state(self, u, state=None):
        """Process one chunk and return ``(output, (ssm, conv))`` state."""
        b, length, _ = u.shape
        xz = self.in_proj(u)
        x, z, bc, dtc = xz.split(
            [self.d_inner, self.d_inner, 2 * self.d_state, 1], dim=-1)
        Bv, c = bc.split([self.d_state, self.d_state], dim=-1)
        xr = x.transpose(1, 2)
        conv_state = None if state is None else state[1]
        if conv_state is None:
            xin = F.pad(xr, (self.d_conv - 1, 0))
        else:
            xin = torch.cat([conv_state, xr], dim=2)
        xc = F.conv1d(
            xin, self.conv1d.weight, self.conv1d.bias,
            groups=self.d_inner, padding=0)
        new_conv = xin[:, :, -(self.d_conv - 1):]
        x = self.act(self.norm(xc.transpose(1, 2)))
        dt = F.softplus(dtc + self.dt_bias).clamp_max(0.1)
        logA = -torch.exp(self.A_log.clamp(-10.0, 0.0))
        m = torch.cumsum(dt, dim=1)
        h_prev = torch.zeros(
            b, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        if state is not None:
            h_prev = state[0]
        ys = []
        for j in range(0, length, self.scan_chunk):
            hi = min(j + self.scan_chunk, length)
            ml = m[:, j:hi]
            rel_in = (ml - m[:, j:j + 1]).unsqueeze(-1)
            P = torch.exp(logA.unsqueeze(0).unsqueeze(0) * rel_in)
            q = x[:, j:hi].unsqueeze(-1) * Bv[:, j:hi].unsqueeze(2) / P
            local = P * torch.cumsum(q, dim=1)
            if j == 0:
                m_prev = m[:, j:j + 1] - dt[:, j:j + 1]
            else:
                m_prev = m[:, j - 1:j]
            carry = torch.exp(
                logA.unsqueeze(0).unsqueeze(0)
                * (ml - m_prev).unsqueeze(-1)) * h_prev.unsqueeze(1)
            h = carry + local
            ys.append((h * c[:, j:hi].unsqueeze(2)).sum(-1))
            h_prev = h[:, -1]
        y = torch.cat(ys, dim=1)
        return (self.out_proj(y * F.silu(z)),
                (h_prev, new_conv))


class MLP(nn.Module):
    """Residual MLP branch using SiLU."""

    def __init__(self, d, hidden):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TinyLMStage(nn.Module):
    """One Mamba residual followed by three MLP residual branches."""

    def __init__(self, d=64, hidden=90, d_state=32, d_conv=3,
                 scan_chunk=64):
        super().__init__()
        self.norm_m = nn.LayerNorm(d)
        self.mamba = Mamba2Block(
            d, d_inner=2 * d, d_state=d_state, d_conv=d_conv,
            scan_chunk=scan_chunk)
        self.norm1 = nn.LayerNorm(d)
        self.mlp1 = MLP(d, hidden)
        self.norm2 = nn.LayerNorm(d)
        self.mlp2 = MLP(d, hidden)
        self.norm3 = nn.LayerNorm(d)
        self.mlp3 = MLP(d, hidden)

    def forward(self, h):
        h = h + self.mamba(self.norm_m(h))
        h = h + self.mlp1(self.norm1(h))
        h = h + self.mlp2(self.norm2(h))
        h = h + self.mlp3(self.norm3(h))
        return h
