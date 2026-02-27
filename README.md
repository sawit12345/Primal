# Primal

Primal is a lightweight, fully sub-symbolic learning framework that combines active-inference-inspired control, fast Bayesian belief fusion, and brain-inspired modular processing.

The implementation is built with `numpy`/`scipy` plus small support libraries, and it is organized around a modular `brain/` stack plus an integrated `agent.py` runtime.

## What Is Implemented

Each AGENTS.md requirement is implemented as a dedicated module and wired through the main `PrimalAgent` in `agent.py`.

1. Active inference core (free-energy objective, predictive coding, action generation): `brain/active_inference.py`
2. Log-space fusion + exponential-conjugate growth: `brain/log_fusion.py`
3. Core knowledge + transfer substrate (Spelke channels): `brain/core_knowledge.py`
4. Theory Theory multi-hypothesis selection: `brain/theory_theory.py`
5. Bayesian model reduction merge/prune: `brain/bmr.py`
6. PFC/VLPFC, retina, visual cortex, ATL, hippocampus, homeostasis: `brain/cortical_stack.py`
7. Cerebellar motor smoothing: `brain/cerebellum.py`
8. Weber-Fechner ANS precision scaling: `brain/weber_fechner.py`
9. Superior colliculus orienting/saliency: `brain/superior_colliculus.py`
10. Occipital feature extraction: `brain/occipital_lobe.py`
11. Bilateral hemifield split + pull imbalance: `brain/hemifield.py`
12. Survival urgency precision alpha controller: `brain/survival_precision.py`
13. 18-step lattice-Boltzmann fluid advection prior: `brain/fluid_lattice.py`
14. Proprioception as continuous Gaussian: `brain/proprioception.py`
15. Markov temporal decay (0.7 old + 0.3 new): `brain/temporal_decay.py`
16. Renormalization group multiscale pooling: `brain/renormalization.py`
17. Common-sense gap filling: `brain/common_sense.py`
18. Additional brain utilities (thalamic gating, neuromodulation): `brain/extras.py`

## Project Layout

- `brain/`: modular brain-inspired mechanisms
- `agent.py`: integrated `PrimalAgent` + benchmark helpers
- `scripts/verify_primal.py`: end-to-end verification on physics + image tasks
- `tests/`: unit/integration validation
- `artifacts/verification_report.json`: latest benchmark report
- `pyproject.toml`: dependencies + project config
- `LICENSE`: custom Primeval license

## Install

```bash
python -m pip install -e ".[dev]"
```

## Run Verification

```bash
python scripts/verify_primal.py
pytest
```

## Latest Verification Snapshot

From `artifacts/verification_report.json`:

- Physics (Gymnasium `CartPole-v1`, 6 episodes, two seeds):
  - Seed 0 episode rewards: `[53, 500, 500, 500, 500, 500]`
  - Seed 1 episode rewards: `[51, 500, 500, 500, 500, 500]`
  - Mastery reached by episode 2 in both runs
  - Throughput: about `28 steps/s`
  - RAM footprint: about `90-95 MB`
- Image benchmark (`10` train samples per digit, full `10,000` MNIST test set, no augmentation or synthetic transforms):
  - Dataset source: `torchvision_mnist`
  - Accuracy: `0.7794`

## Minimal Usage Example

```python
import gymnasium as gym
import numpy as np

from agent import PrimalAgent

env = gym.make("CartPole-v1")
obs, _ = env.reset(seed=0)

agent = PrimalAgent(observation_dim=np.asarray(obs).size, action_space=env.action_space)
result = agent.run_episode(env, learn=True)
print(result)
env.close()
```
