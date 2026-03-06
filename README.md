# Cheap Universal AGI Blueprint (NumPy/SciPy)

This repository contains a CPU-only implementation inspired by `BLUEPRINT.md`,
focused on biologically grounded active-inference style modules and two required
benchmarks:

1. **MNIST one-shot**
   - One labeled real sample per class (10 labels total)
   - Full 10k test-set evaluation
   - No deep learning frameworks

2. **Gymnasium Breakout**
   - `gymnasium>=1.0`, `ale-py>=0.9` setup
   - No AutoROM license prompt path
   - First-episode viability logging (lives/score)

## Architecture implemented

The code mirrors the blueprint in modular form:

- Retina + color opponency + DoG (`vision.py`)
- Fixed V1 Gabor bank and sparse thresholding (`vision.py`)
- V5 motion field and superior colliculus salience (`vision.py`)
- Multi-level sparse discrete RGM hierarchy with structure growth (`rgm.py`)
- Dentate gyrus random expansion projection (`memory.py`)
- Hippocampal one-shot tuple memory + retrieval + retention/pruning + clustering (`memory.py`)
- Predictive coding cortical hierarchy + recurrent layer 6 (`cortex.py`)
- Thalamic routing mask (`thalamus.py`)
- Basal ganglia TD(λ) action system (`action.py`)
- Neuromodulator scalars (`neuromod.py`)
- Amygdala valence (`affect.py`)
- End-to-end integrated brain scaffold (`brain.py`)

## Install

```bash
python3 -m pip install scipy scikit-learn gymnasium ale-py tqdm pillow
```

## Run MNIST one-shot

```bash
PYTHONPATH=src python3 scripts/run_mnist_oneshot.py \
  --unsupervised-subset 60000 \
  --aug-per-class 0
```

Result JSON is written to `results/mnist_oneshot_results.json`.

## Run Breakout evaluation

```bash
PYTHONPATH=src python3 scripts/run_breakout_eval.py \
  --episodes 2 \
  --max-steps 12000
```

Result JSON is written to `results/breakout_eval_results.json`.

## Run tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py"
```