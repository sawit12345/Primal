import numpy as np

def test_fe_decreases_after_update():
    from primal.brain.log_space_gmm import LogSpaceGMM
    gmm = LogSpaceGMM(feature_dim=8, max_components=10)
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    fe_before = gmm.compute_fe(obs)
    for _ in range(5):
        gmm.update(obs)
    fe_after = gmm.compute_fe(obs)
    assert fe_after < fe_before, (
        f"FE did not decrease after 5 updates on same observation. "
        f"Before: {fe_before:.4f}, After: {fe_after:.4f}. "
    )

def test_gmm_grows_on_novel_observation():
    from primal.brain.log_space_gmm import LogSpaceGMM
    gmm = LogSpaceGMM(feature_dim=4, max_components=20, novelty_threshold=0.1)
    obs_a = np.array([0.0, 0.0, 0.0, 0.0])
    obs_b = np.array([100.0, 100.0, 100.0, 100.0])
    for _ in range(10):
        gmm.update(obs_a)
    n_before = gmm.n_components
    gmm.update(obs_b)
    n_after = gmm.n_components
    assert n_after > n_before, (
        f"GMM did not grow on novel observation. "
        f"Components before: {n_before}, after: {n_after}. "
    )
