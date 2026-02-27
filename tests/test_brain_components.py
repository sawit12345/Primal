from __future__ import annotations

import numpy as np

from brain.bmr import BayesianModelReduction
from brain.differ import DifferCoreKnowledge
from brain.fluid_lattice import LatticeBoltzmannIntuition
from brain.log_fusion import LogSpaceFusion
from brain.temporal_decay import MarkovTemporalDecay
from brain.theory_theory import TheoryTheoryEnsemble


def test_markov_temporal_decay_default_rule() -> None:
    decay = MarkovTemporalDecay()
    old = np.array([10.0, -2.0])
    new = np.array([2.0, 4.0])
    blended = decay.blend(old, new)
    assert np.allclose(blended, np.array([7.6, -0.2]))


def test_lattice_boltzmann_advances_18_steps() -> None:
    lbm = LatticeBoltzmannIntuition(height=8, width=8)
    lbm.inject_velocity(0.2, -0.1)
    velocity = lbm.advance_18_steps()
    assert velocity.shape == (2,)
    assert np.isfinite(velocity).all()


def test_log_fusion_growth_and_bmr_reduction() -> None:
    fusion = LogSpaceFusion(dim=4, initial_components=1, max_components=8, growth_surprise_threshold=1.5)
    rng = np.random.default_rng(0)

    for _ in range(50):
        sample = rng.normal(0.0, 1.0, size=4)
        fusion.update(sample)

    assert len(fusion.components) >= 1
    slot_probs = fusion.slot_attention(np.ones(4, dtype=np.float64))
    assert np.isclose(np.sum(slot_probs), 1.0)
    reducer = BayesianModelReduction(max_components=3)
    reducer.reduce(fusion)
    assert len(fusion.components) <= 3


def test_theory_theory_updates_posterior() -> None:
    ensemble = TheoryTheoryEnsemble(state_dim=6, action_dim=2, num_hypotheses=4, seed=0)
    state = np.linspace(-1.0, 1.0, 6)
    action = np.array([0.5, -0.5])
    target = state * 0.8

    before = ensemble.posterior().copy()
    for _ in range(5):
        ensemble.update(state, action, target)
    after = ensemble.posterior()

    assert np.isclose(np.sum(after), 1.0)
    assert np.linalg.norm(after - before) > 1e-6

    mixed, per_hyp = ensemble.predict_multiple(state, action)
    assert set(mixed.keys()) == {1, 2, 3}
    assert set(per_hyp.keys()) == {1, 2, 3}
    assert mixed[1].shape == (6,)


def test_differ_universal_mode_handles_non_image_vectors() -> None:
    differ = DifferCoreKnowledge(embedding_dim=12, seed=0)
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(64, 17))
    labels = np.arange(64) % 4

    differ.fit(vectors, labels)
    embeddings = differ.encode(vectors)
    probe_embedding, probe_confidence, probe_scores = differ.differentiate(vectors[0], learn=True)

    assert embeddings.shape == (64, 12)
    assert probe_embedding.shape == (12,)
    assert 0.0 <= probe_confidence <= 1.0
    assert probe_scores.shape[0] == differ.online_slots
