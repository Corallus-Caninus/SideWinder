import math
import unittest

import torch

from sidewinder.ema import causal_ema as _causal_ema


def _reference(x, beta):
    """Original full-sequence float64 inverse-weight implementation."""
    beta = min(max(float(beta), 0.0), 1.0)
    if beta <= 0.0:
        return x
    if beta >= 1.0:
        length = x.shape[1]
        return torch.cumsum(x, dim=1) / torch.arange(
            1, length + 1, dtype=x.dtype).view(1, length, 1)
    length = x.shape[1]
    t = torch.arange(length, dtype=torch.float64)
    logw = t * math.log(beta)
    c = torch.cumsum(
        x.to(torch.float64) * torch.exp(-logw).view(1, length, 1), dim=1)
    return ((1.0 - beta) * torch.exp(logw).view(1, length, 1) * c).to(x.dtype)


class StableEmaTest(unittest.TestCase):
    def test_matches_reference_across_betas(self):
        torch.manual_seed(0)
        x = torch.randn(2, 2048, 3)
        for beta in (0.9, 0.99, 0.999):
            got = _causal_ema(x, beta)
            want = _reference(x, beta)
            self.assertTrue(torch.allclose(got, want, rtol=1e-5, atol=1e-6))

    def test_long_beta_099_is_finite(self):
        x = torch.ones(1, 100_000, 1)
        out = _causal_ema(x, 0.99)
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(float((out[:, -1] - 1.0).abs().max()), 1e-3)

    def test_million_context_beta_0999_is_finite(self):
        x = torch.ones(1, 1_000_000, 1)
        out = _causal_ema(x, 0.999)
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(float((out[:, -1] - 1.0).abs().max()), 1e-3)

    def test_gradient_is_finite(self):
        x = torch.randn(1, 2048, 2, requires_grad=True)
        _causal_ema(x, 0.999).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    unittest.main()
