from __future__ import annotations

import unittest

import numpy as np

from cheap_universal_agi.config import BlueprintConfig
from cheap_universal_agi.memory import DentateGyrus, HippocampalBuffer
from cheap_universal_agi.rgm import HierarchicalRGM
from cheap_universal_agi.vision import RetinaV1Pipeline


class BlueprintCoreTests(unittest.TestCase):
    def test_retina_v1_shapes(self):
        cfg = BlueprintConfig()
        vision = RetinaV1Pipeline(cfg.vision, seed=cfg.seed)
        frame = np.zeros((84, 84, 3), dtype=np.uint8)
        ret = vision.retina(frame)
        v1 = vision.v1(ret)
        flow = vision.v5_motion(v1)
        self.assertEqual(ret.channels.shape, (4, cfg.vision.input_size, cfg.vision.input_size))
        self.assertEqual(v1.shape[0], 48)
        self.assertEqual(flow.shape, (cfg.vision.v5_grid, cfg.vision.v5_grid, 2))

    def test_rgm_encode(self):
        cfg = BlueprintConfig()
        rgm = HierarchicalRGM(cfg.rgm_levels, seed=cfg.seed)
        x = np.random.default_rng(0).normal(size=384).astype(np.float32)
        rgm.fit([x], epochs=1)
        top = rgm.top_belief(x)
        self.assertGreater(top.shape[0], 0)
        self.assertAlmostEqual(float(top.sum()), 1.0, places=4)

    def test_dg_and_hippocampus(self):
        dg = DentateGyrus(input_dim=384, output_dim=3840, sparsity=0.02, seed=0)
        x = np.random.default_rng(1).normal(size=384).astype(np.float32)
        z = dg.project(x)
        self.assertEqual(int(z.sum()), int(3840 * 0.02))

        hp = HippocampalBuffer(dim=1152, capacity=128)
        a = np.zeros(1152, dtype=np.float32)
        a[:5] = 1
        hp.write(a, novelty=0.9)
        r = hp.retrieve(a)
        self.assertEqual(r.index, 0)
        self.assertGreater(r.confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
