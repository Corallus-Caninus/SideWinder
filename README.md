# SideWinder

SideWinder is a compact byte-level causal language model built from Mamba-style
sequence stages and EMA-compressed grouped-query attention transitions.

This repository is the single source of truth for the SideWinder architecture.
The DBFGS and SVDBFGS experiment harnesses import `TinyLM` from this package.

## Default model

```text
vocab:          256 UTF-8 byte tokens
d_model:        64
d_hidden:       90
stages:         4
transitions:     3
Mamba d_inner:  128  (2 * d_model)
Mamba d_state:  32
attention:      4 query heads / 1 KV head
attention beta: 0.9
attention N:    10 tokens
parameters:     348,796
```

`n_layers` SideWinder stages produce `n_layers - 1` attention transitions.
The last stage has no outgoing attention: its final MLP output goes through the
final LayerNorm and LM head.

## Architecture flow

```text
 token IDs [B, T]
       │
       ▼
 ┌───────────────┐
 │  Embedding    │  vocab → d_model
 └───────┬───────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ STAGE 0                                                     │
 │                                                             │
 │   LayerNorm ──► Mamba2 ──────────────────────────────► (+)  │
 │                                                             │
 │   LayerNorm ──► MLP 1 ───────────────────────────────► (+)  │
 │                                                             │
 │   LayerNorm ──► MLP 2 ───────────────────────────────► (+)  │
 │                                                             │
 │   LayerNorm ──► MLP 3 ───────────────────────────────► (+)  │
 └─────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ ATTENTION TRANSITION 0                                      │
 │                                                             │
 │   h ──► LayerNorm ──► causal EMA ──► running block EMA      │
 │                                      │                      │
 │                                      ├─► frozen block K/V   │
 │                                      └─► running prefix K/V │
 │                                                │            │
 │                                                ▼            │
 │                                      grouped-query attention│
 │                                                │            │
 │                                      context gate:          │
 │                                      1 + tanh(Linear(2d,d)) │
 │                                                │            │
 │   h + gate * context ◄─────────────────────────┘            │
 └─────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ STAGE 1                                                     │
 │   LayerNorm → Mamba2 → (+)                                  │
 │   LayerNorm → MLP 1  → (+)                                  │
 │   LayerNorm → MLP 2  → (+)                                  │
 │   LayerNorm → MLP 3  → (+)                                  │
 └─────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
                              ...
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ ATTENTION TRANSITION 1                                      │
 │   same EMA running-block GQA block                          │
 └─────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ FINAL STAGE                                                 │
 │   LayerNorm → Mamba2 → (+)                                  │
 │   LayerNorm → MLP 1  → (+)                                  │
 │   LayerNorm → MLP 2  → (+)                                  │
 │   LayerNorm → MLP 3  → (+)                                  │
 │                                                             │
 │   NO outgoing attention                                     │
 └─────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │ Final LayerNorm  │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ LM head          │
                     │ d_model → vocab  │
                     └────────┬─────────┘
                              │
                              ▼
                     next-byte logits [B, T, vocab]
```

For the default four-stage model:

```text
stage 0 → attention 0 → stage 1
stage 1 → attention 1 → stage 2
stage 2 → attention 2 → stage 3
stage 3 → final norm → LM head
```

For a smaller one-attention model, use `n_layers=2`:

```text
stage 0 → attention 0 → stage 1 → final norm → LM head
```

## Layer explanations

### Embedding

```text
input:  token IDs [B, T]
output: hidden states [B, T, d_model]
```

There is no learned positional embedding. Mamba supplies sequence ordering
through its recurrent state. The default tokenizer is UTF-8 bytes, so the
vocabulary is 256.

### Mamba2Block

```text
u [B, T, d]
  │
  ├─► in_proj ─► split into x, z, B, C, dt
  │
  ├─► depthwise causal Conv1d over x
  │
  ├─► SiLU + RMSNorm
  │
  ├─► input-dependent dt
  │
  ├─► chunked diagonal SSM scan
  │
  └─► y * SiLU(z) ─► out_proj ─► [B, T, d]
```

The scan carries a recurrent state between internal chunks. The current
default internal scan chunk is 64 tokens. `forward_state()` exposes the SSM
and convolution state for long-context streaming.

### MLP branch

```text
Linear(d → hidden) → SiLU → Linear(hidden → d)
```

Each SideWinder stage contains three of these residual branches after the
Mamba residual branch.

### SideWinder stage

```text
h = h + Mamba(LayerNorm(h))
h = h + MLP1(LayerNorm(h))
h = h + MLP2(LayerNorm(h))
h = h + MLP3(LayerNorm(h))
```

All branches are pre-normalized residual branches.

### Causal EMA

The attention source is smoothed causally:

```text
e_t = (1 - beta) * Σ_{j <= t} beta^(t-j) * x_j
```

The implementation uses bounded chunks and float64 intermediate state to avoid
overflow at long sequence lengths.

### Block size

The attention block interval is derived directly from beta:

```text
N = round(1 / (1 - beta))
```

| Beta | Interval N | Meaning |
|---:|---:|---|
| 0.9 | 10 | 10-token blocks |
| 0.99 | 100 | 100-token blocks |
| 0.999 | 1000 | 1000-token blocks |

At a 400k context, beta `0.999` produces about 400 frozen attention blocks.

### Running-block GQA

```text
query tokens
    │
    ├──────────────► attend to frozen completed blocks
    │
    └──────────────► attend to the live running prefix
```

Each completed block contributes one frozen K/V summary. The current block
keeps one running K/V entry that changes with every token. This keeps attention
memory at `O(context / N)` frozen entries instead of storing every token in the
past.

The default attention uses four query heads and one KV head:
`4Q / 1KV`.

### Context gate

```text
gate = 1 + tanh(Linear(concat(h, context)))
out  = h + gate * context
```

The gate projection is initialized to zero. The initial gate is therefore
exactly one, and the model learns how much attention context to admit.

### Final output

```text
h = LayerNorm(h)
logits = Linear(d_model → vocab)
```

The final stage does not feed another attention transition.

## Long-context API

```python
from sidewinder import TinyLM

model = TinyLM(
    vocab=256,
    d=64,
    hidden=90,
    n_layers=4,
    attn_heads=4,
    attn_kv_heads=1,
    attn_ema_beta=0.999,
    long_context=True,
    long_context_chunk=2000,
    stateful=True,
)
```

Available methods:

```python
model(x)                                  # full-sequence logits
model.forward_hidden(x)                   # hidden states only
model.chunked_loss_stateful(x, y, chunk)  # stateful chunked CE loss
model.chunked_grad_loss(x, y, chunk)      # detached-state chunked BPTT
model.chunked_eval_stateful(x, y, chunk)  # stateful chunked eval loss/acc
```

`chunked_grad_loss` processes chunks sequentially, accumulates parameter
gradients, and detaches the recurrent/attention state between chunks. This keeps
the full 400k-context graph out of memory.

The package itself does not read benchmark environment variables. The harness
chooses checkpointing, offloading, chunk size, and optimizer behavior.

## Importing from SideWinder

Add the source directory to `PYTHONPATH`:

```bash
export PYTHONPATH="$HOME/Code/SideWinder/src:$PYTHONPATH"
```

The consuming projects also default their fallback path to:

```text
~/Code/SideWinder/src
```

or respect:

```bash
SIDEWINDER_ROOT=/path/to/SideWinder/src
```

## Repository layout

```text
SideWinder/
├── README.md
├── pyproject.toml
├── src/
│   └── sidewinder/
│       ├── __init__.py
│       ├── config.py
│       ├── ema.py
│       ├── blocks.py
│       ├── attention.py
│       └── model.py
└── tests/
    ├── fixtures/
    │   └── default_logits.pt
    ├── test_ema.py
    └── test_model.py
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The tests cover the default parameter count, forward shape, golden logits,
stable EMA behavior, stateful chunk parity, and finite detached chunk gradients.
