# AGENTS.md

## Cursor Cloud specific instructions

**Product:** Primal is a pure-Python brain-inspired AGI research framework built on active inference. No web servers, databases, or Docker — everything runs in-process on CPU.

**Install:** `pip install -e ".[dev]"` (see `pyproject.toml` for the full dependency list).

**Key commands** (all documented in `README.md`):
- **Tests:** `pytest` (7 tests across `tests/test_brain_components.py` and `tests/test_agent_integration.py`, runs in ~3s)
- **Lint:** `python3 -m ruff check .` (ruff is not a project dependency; install separately if needed). There are 2 pre-existing F401 warnings (unused `field` imports in `agent.py` and `brain/cortical_stack.py`).
- **Full verification:** `python scripts/verify_primal.py` (runs CartPole physics + MNIST image benchmarks; takes ~2 min, downloads MNIST on first run)
- **Quick smoke test:** `python3 -c "from agent import PrimalAgent; import gymnasium as gym, numpy as np; env=gym.make('CartPole-v1'); obs,_=env.reset(seed=0); a=PrimalAgent(np.asarray(obs).size, env.action_space); r=a.run_episode(env, learn=True); print(r); env.close()"`

**Caveats:**
- The `verify_primal.py` script auto-downloads MNIST data via torchvision on first run (~50 MB into `data/`). Subsequent runs use the cached dataset.
- Atari ROM environments (e.g. `ALE/Pong-v5`) are optional and gracefully skipped if not installed.
- CartPole episodes are deterministic given the same seed; the agent consistently reaches 500 reward by episode 2.
- PyTorch is CPU-only in this project; no GPU/CUDA is needed.
