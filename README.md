# UT-GCDT: Universal Transformer for Goal-Conditioned Decision Transformers

## Research Question

Can Universal Transformer-style recurrence (weight-tied iterations + plan token + auxiliary supervision) improve offline goal-conditioned RL where standard depth scaling failed?

## Motivation

The "1000 Layer Networks for Self-Supervised RL" paper (Wang et al., NeurIPS 2025) showed that:
- Depth scaling dramatically improves **online** goal-conditioned RL (2-50× gains)
- Depth scaling **fails** in **offline** settings (performance degrades with depth)

We hypothesize that recursive architectures (inspired by Universal Transformer, TRM, URM) can provide the benefits of depth via iterative refinement without the optimization difficulties of deep untied networks in offline settings.

## Architecture

```
Input: [PLAN] [GOAL] s_0 a_0 s_1 a_1 ... s_t
       ↓
   Embedding Layer (state, action, goal, plan embeddings)
       ↓
   ┌─────────────────────────────┐
   │  Transformer Block          │ ←── shared weights
   │  (Self-Attention + FFN)     │     × K iterations
   │  + step embedding (iter k)  │
   └─────────────────────────────┘
       ↓
   Plan token → Waypoint head (predict s_{t+H})  [auxiliary loss]
   Final s_t  → Action head (predict a_t)        [main loss]
```

### Key Components

1. **Plan Token**: Learnable token prepended to sequence, iteratively refined
2. **Step Embeddings**: Position-like embeddings for each UT iteration
3. **Weight Tying**: Single transformer block reused K times
4. **Waypoint Auxiliary Loss**: Predict future state s_{t+H} from plan token
   - Provides deep supervision signal
   - Forces plan token to encode trajectory-relevant information

## Experimental Plan

### Dataset
- Primary: `antmaze-medium-stitch-v0` (showed slight depth benefit offline)
- Secondary: `humanoidmaze-medium-navigate-v0`

### Baselines & Ablations

| Model | Layers | Tied | Plan Token | Waypoint Loss |
|-------|--------|------|------------|---------------|
| GCDT-L4 | 4 | No | No | No |
| GCDT-L8 | 8 | No | No | No |
| UT-GCDT-K4 | 4 iters | Yes | No | No |
| UT-GCDT-K4-Plan | 4 iters | Yes | Yes | No |
| UT-GCDT-K4-Plan-WP | 4 iters | Yes | Yes | Yes |

### Test-Time Scaling
- Train with K=4, evaluate with K=4,6,8
- Hypothesis: UT allows beneficial extra computation at test time

### Metrics
- Success rate (reaches goal within threshold)
- Steps to goal (efficiency)
- Performance vs. goal distance (long-horizon generalization)

## Project Structure

```
ut-gcdt/
├── configs/           # Hyperparameter configs
│   └── default.py
├── models/            # Model implementations
│   ├── gcdt.py        # Standard GCDT baseline
│   ├── ut_gcdt.py     # Universal Transformer GCDT
│   └── components.py  # Shared components (embeddings, heads)
├── data/              # Data loading utilities
│   └── ogbench_loader.py
├── scripts/           # Training and evaluation
│   ├── train.py
│   └── eval.py
├── utils/             # Utilities
│   └── metrics.py
└── README.md
```

## Dependencies

- JAX + Flax (to match OGBench reference implementations)
- OGBench (`pip install ogbench`)
- Optax (optimizer)

## References

- [1000 Layer Networks for Self-Supervised RL](https://arxiv.org/abs/...) - Wang et al., NeurIPS 2025
- [Universal Transformers](https://arxiv.org/abs/1807.03819) - Dehghani et al., ICLR 2019
- [OGBench](https://arxiv.org/abs/2410.20092) - Park et al., 2024
- [Decision Transformer](https://arxiv.org/abs/2106.01345) - Chen et al., NeurIPS 2021
- [URM](https://arxiv.org/abs/2512.14693) - Gao et al., 2025
- [TRM](https://arxiv.org/abs/2510.04871) - Jolicoeur-Martineau, 2025
