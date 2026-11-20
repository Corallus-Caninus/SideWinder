import unittest
from pathlib import Path

import torch

from sidewinder import TinyLM


class ModelTest(unittest.TestCase):
    def test_default_parameter_count(self):
        model = TinyLM()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 348_796)

    def test_default_forward_shape(self):
        model = TinyLM().eval()
        x = torch.randint(0, 256, (2, 32))
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 32, 256))

    def test_golden_logits(self):
        torch.manual_seed(0)
        model = TinyLM().eval()
        x = torch.arange(64, dtype=torch.long).remainder(256).unsqueeze(0)
        expected = torch.load(
            Path(__file__).parent / "fixtures" / "default_logits.pt",
            weights_only=True)
        with torch.no_grad():
            got = model(x)
        self.assertTrue(torch.equal(got.cpu(), expected))

    def test_stateful_chunk_parity(self):
        torch.manual_seed(0)
        model = TinyLM(
            vocab=32, d=16, hidden=24, n_layers=2, attn_ema_beta=0.9,
            attn_heads=4, attn_kv_heads=1, long_context_chunk=10).eval()
        x = torch.randint(0, 32, (1, 20))
        y = torch.randint(0, 32, (1, 20))
        with torch.no_grad():
            whole = model.chunked_loss_stateful(x, y, chunk=20)
            split = model.chunked_loss_stateful(x, y, chunk=10)
        self.assertTrue(torch.allclose(whole, split, rtol=1e-5, atol=1e-6))

    def test_chunked_grad_is_finite(self):
        torch.manual_seed(0)
        model = TinyLM(
            vocab=32, d=16, hidden=24, n_layers=2, attn_ema_beta=0.9,
            attn_heads=4, attn_kv_heads=1, long_context_chunk=10)
        x = torch.randint(0, 32, (1, 20))
        y = torch.randint(0, 32, (1, 20))
        loss, chunks = model.chunked_grad_loss(x, y, chunk=10)
        self.assertEqual(chunks, 2)
        self.assertEqual(loss, loss)
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all()
                            for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
