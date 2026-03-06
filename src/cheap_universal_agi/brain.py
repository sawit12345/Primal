from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action import BasalGangliaTDLambda
from .affect import AmygdalaValence
from .config import BlueprintConfig
from .cortex import CorticalHierarchy
from .memory import DentateGyrus, HippocampalBuffer
from .neuromod import Neuromodulators
from .thalamus import ThalamicRouter
from .vision import RetinaV1Pipeline


@dataclass(slots=True)
class BrainStepOutput:
    action: int
    dopamine: float
    norepinephrine: float
    serotonin_gamma: float
    acetylcholine: float
    threat: float
    reward_anticipation: float
    hippocampal_written: bool
    retrieval_confidence: float


class BlueprintBrain:
    """
    End-to-end integration of major blueprint modules.

    This class is intentionally generic and environment-agnostic.
    """

    def __init__(self, cfg: BlueprintConfig):
        self.cfg = cfg
        self.vision = RetinaV1Pipeline(cfg.vision, seed=cfg.seed)
        self.dg = DentateGyrus(
            input_dim=cfg.memory.dg_input_dim,
            output_dim=cfg.memory.dg_output_dim,
            sparsity=cfg.memory.dg_sparsity,
            seed=cfg.seed,
        )
        self.cortex = CorticalHierarchy(cfg.cortex, seed=cfg.seed)
        self.thalamus = ThalamicRouter(input_dim=cfg.cortex.input_dim, hidden_dim=128, seed=cfg.seed)
        self.buffer = HippocampalBuffer(dim=cfg.memory.hippocampal_dim, capacity=cfg.memory.hippocampal_capacity)
        self.bg = BasalGangliaTDLambda(
            cfg.action,
            state_dim=cfg.cortex.layer_dims[-1],
            goal_dim=64,
            seed=cfg.seed,
        )
        self.neuro = Neuromodulators()
        self.amygdala = AmygdalaValence(input_dim=512, seed=cfg.seed)
        self.goal = np.zeros(64, dtype=np.float32)
        self.goal[0] = 1.0
        self.last_cortical_state = np.zeros(cfg.cortex.layer_dims[-1], dtype=np.float32)

    def set_goal(self, goal_vec: np.ndarray):
        if goal_vec.shape[0] != 64:
            raise ValueError("Goal vector must be 64-dimensional.")
        self.goal = goal_vec.astype(np.float32)

    def step(self, visual_it_512: np.ndarray, auditory_a2_128: np.ndarray, reward: float) -> BrainStepOutput:
        cortical_error_pad = np.zeros(self.cfg.cortex.input_dim, dtype=np.float32)
        mask = self.thalamus.mask(
            norepinephrine=0.0,
            cortical_error=cortical_error_pad,
            goal_vector=self.goal,
        )
        c = self.cortex.step(visual_it=visual_it_512, auditory_a2=auditory_a2_128, thalamic_mask=mask)
        layer5 = c["layer5"]
        layer6 = c["layer6"]
        errors = c["errors"]

        # Hippocampal retrieval query.
        query = np.zeros(self.cfg.memory.hippocampal_dim, dtype=np.float32)
        n_l5 = min(layer5.shape[0], 256)
        n_v = min(visual_it_512.shape[0], 512)
        n_l6 = min(layer6.shape[0], 256)
        query[:n_l5] = layer5[:n_l5]
        query[256 : 256 + n_v] = visual_it_512[:n_v]
        query[768 : 768 + n_l6] = layer6[:n_l6]
        query[1024:1088] = self.goal
        retrieval = self.buffer.retrieve(query, top_k=5)

        amyg_in = np.zeros(512, dtype=np.float32)
        amyg_in[:256] = visual_it_512[:256]
        amyg_in[256:] = retrieval.vector[:256]
        threat, reward_ant = self.amygdala.forward(amyg_in)
        valence_offset = np.full(self.cfg.action.n_actions, reward_ant - threat, dtype=np.float32)
        action, _scores = self.bg.select_action(layer5, self.goal, valence_offset=valence_offset)

        td_error = self.bg.td_update(reward=reward, next_state=layer5, goal=self.goal)
        dopamine = self.neuro.dopamine(td_error)
        norepi = self.neuro.norepinephrine(errors)
        gamma = self.neuro.serotonin_gamma()
        self.bg.cfg.gamma = gamma
        ach = self.neuro.acetylcholine(retrieval.top_scores)
        self.amygdala.update(amyg_in, dopamine=dopamine)

        wrote = False
        if norepi > self.cfg.memory.write_novelty_threshold:
            self.buffer.write(query, novelty=norepi, social=False)
            wrote = True

        self.last_cortical_state = layer5.copy()
        return BrainStepOutput(
            action=action,
            dopamine=float(dopamine),
            norepinephrine=float(norepi),
            serotonin_gamma=float(gamma),
            acetylcholine=float(ach),
            threat=float(threat),
            reward_anticipation=float(reward_ant),
            hippocampal_written=wrote,
            retrieval_confidence=float(retrieval.confidence),
        )
