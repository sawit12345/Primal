import numpy as np

from primal.visual import Retina, V1, V5
from primal.auditory import Auditory, InferiorColliculus
from primal.rgm import VisualRGM, AuditoryRGM
from primal.memory import LMUIndex, DentateGyrus, HippocampalBuffer, CA1Mismatch, SpatialMap
from primal.cortex import CorticalHierarchy
from primal.subcortical import BasalGanglia, PrefrontalCortex, Neuromodulators, Amygdala, Cerebellum
from primal.core_knowledge import CoreKnowledgePriors, GlobalWorkspace, TPJSocialCognition

class PrimalAgent:
    def __init__(self, action_dim=6, is_continuous=False):
        self.retina = Retina()
        self.v1 = V1()
        self.v5 = V5()
        self.auditory = Auditory()
        self.ic = InferiorColliculus()

        self.visual_rgm = VisualRGM(input_dim=128)
        self.aud_rgm = AuditoryRGM()

        self.lmu = LMUIndex()
        self.dg = DentateGyrus()
        self.hippocampus = HippocampalBuffer(max_elements=200000)
        self.ca1 = CA1Mismatch()
        self.spatial = SpatialMap()

        self.cortex = CorticalHierarchy()
        self.bg = BasalGanglia(action_dim=action_dim, is_continuous=is_continuous)
        self.pfc = PrefrontalCortex()
        self.neuromods = Neuromodulators()
        self.amygdala = Amygdala()
        self.cerebellum = Cerebellum()

        self.core = CoreKnowledgePriors()
        self.workspace = GlobalWorkspace()
        self.tpj = TPJSocialCognition()

        self.prev_l5_state = np.zeros(256)
        self.prev_cortical_error = 0.0
        self.prev_action = 0
        self.last_observation_tuple = None

    def act(self, rgb_image, reward=0.0):
        td_error = reward + self.bg.gamma * np.max(self.bg.V) - self.bg.last_state_val
        self.neuromods.update(td_error, self.prev_cortical_error, retrieval_confidence=0.5)

        self.bg.update(reward, self.neuromods.dopamine)

        ret_out = self.retina.process(rgb_image)
        v1_out = self.v1.process(ret_out)
        v5_out = self.v5.process(v1_out)

        # Filter iteration order is theta -> lambda -> phase.
        # This means 3 scales (lambda) per orientation (theta), and 2 phases per scale.
        v1_power = np.zeros((128, 128, 24))
        for j in range(24):
            v1_power[:, :, j] = np.maximum(v1_out[:, :, 2*j], v1_out[:, :, 2*j+1])

        # Now every 3 elements in v1_power are the 3 scales for a given orientation.
        v1_orient = np.zeros((128, 128, 8))
        for j in range(8):
            v1_orient[:, :, j] = (v1_power[:, :, j*3] + v1_power[:, :, j*3+1] + v1_power[:, :, j*3+2]) / 3.0

        v1_grid_8x8 = np.max(v1_orient.reshape(8, 16, 8, 16, 8), axis=(1, 3))
        v1_flat_128 = np.max(v1_grid_8x8.reshape(4, 2, 4, 2, 8), axis=(1, 3)).flatten()

        it_visual, ffa_visual = self.visual_rgm.process(v1_flat_128, learn=True)

        mel_spec = np.zeros(128)
        a2_audio = self.aud_rgm.process(mel_spec, learn=True)

        if len(a2_audio) > 128:
             a2_audio = a2_audio[:128]
        elif len(a2_audio) < 128:
             a2_audio = np.pad(a2_audio, (0, 128 - len(a2_audio)))

        l5_state, l6_state, total_err = self.cortex.process(
            it_visual, a2_audio, self.neuromods.norepinephrine, self.pfc.state
        )
        self.prev_cortical_error = total_err

        lmu_state = self.lmu.process(l5_state)

        it_full = np.concatenate([it_visual, v1_flat_128])[:512]
        if len(it_full) < 512: it_full = np.pad(it_full, (0, 512-len(it_full)))
        if np.sum(it_full) > 0: it_full = it_full / np.linalg.norm(it_full)

        context_tag = np.zeros(64)
        index_vec = np.concatenate([lmu_state, it_full, context_tag])[:704]

        if len(it_visual) < 256: it_v = np.pad(it_visual, (0, 256-len(it_visual)))
        elif len(it_visual) > 256: it_v = it_visual[:256]
        else: it_v = it_visual

        tuple_vec = np.concatenate([l5_state, it_full, l6_state, self.pfc.state, context_tag])[:1152]

        retrieved_idx, retrieved_tuple = self.hippocampus.query(index_vec)

        write_flag = True
        if retrieved_tuple is not None:
            write_flag, mismatch_score = self.ca1.process(retrieved_tuple, tuple_vec)
            if self.neuromods.norepinephrine > 0.6: write_flag = True

        if write_flag:
            self.hippocampus.write(index_vec, tuple_vec, novelty_score=self.neuromods.norepinephrine)

        hc_vec = retrieved_tuple if retrieved_tuple is not None else np.zeros(1152)
        threat, rew_anticipation = self.amygdala.process(it_v, hc_vec)
        self.amygdala.learn(it_v, hc_vec, self.neuromods.dopamine)

        action, conflict = self.bg.select_action(l5_state, self.pfc.state, threat, rew_anticipation)

        if conflict:
            action = self.prev_action

        self.prev_action = action
        return action
