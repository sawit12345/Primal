import numpy as np

class BasalGanglia:
    def __init__(self, action_dim=64, in_dim=256, is_continuous=False):
        self.action_dim = action_dim
        self.in_dim = in_dim
        self.is_continuous = is_continuous

        self.V = np.zeros(action_dim)
        self.suppression = np.full(action_dim, 0.5)

        self.e = np.zeros(action_dim)
        self.lambda_ = 0.9
        self.gamma = 0.95

        self.W_goal = np.random.randn(action_dim, 64) / np.sqrt(64)

        if is_continuous:
            self.W_act1 = np.random.randn(128, in_dim) / np.sqrt(in_dim)
            self.W_act2 = np.random.randn(action_dim, 128) / np.sqrt(128)

        self.last_action = None
        self.last_state_val = 0.0

    def select_action(self, cortical_state, pfc_goal, threat_level, reward_anticipation):
        goal_score = np.dot(self.W_goal, pfc_goal)
        valence_offset = reward_anticipation - threat_level
        eff_V = self.V + goal_score + valence_offset - self.suppression
        variance = np.var(eff_V)
        conflict = variance < 0.15

        if conflict:
            return None, True

        if self.is_continuous:
            h = np.maximum(0, np.dot(self.W_act1, cortical_state))
            action = np.dot(self.W_act2, h)
            self.last_action = action
        else:
            # According to blueprint: Norepinephrine drives exploration natively.
            # If all values are suppressed or close to zero, inject NE-scaled noise to values
            # before argmax to trigger spontaneous action (Active Inference exploration)
            if np.max(eff_V) <= 0:
                # Mock high NE exploration noise since true NE scalar is passed around the network
                eff_V += np.random.randn(self.action_dim) * 0.5

            action = np.argmax(eff_V)
            self.last_action = action

        self.last_state_val = np.max(self.V) if not self.is_continuous else np.mean(self.V)
        return action, False

    def update(self, reward, dopamine):
        if self.last_action is not None:
            if self.is_continuous:
                pass
            else:
                self.e[self.last_action] += 1
                self.V += 0.1 * dopamine * self.e
                self.e *= self.lambda_ * self.gamma

                if dopamine > 0:
                    self.suppression[self.last_action] -= 0.05

                self.suppression = np.clip(self.suppression, 0, 1)

class PrefrontalCortex:
    def __init__(self, slots=3, slot_size=20):
        self.slots = slots
        self.slot_size = slot_size
        self.state = np.zeros(64)
        self.active_slot = 0

    def set_goal(self, goal_vec, slot=0):
        start = slot * self.slot_size
        end = start + self.slot_size
        self.state[start:end] = goal_vec[:self.slot_size]
        self.active_slot = slot
        self.state[-4:] = 0
        self.state[-4+slot] = 1

class Neuromodulators:
    def __init__(self):
        self.dopamine = 0.0
        self.norepinephrine = 0.0
        self.serotonin = 0.0
        self.acetylcholine = 0.0

        self.steps_since_reward = 0
        self.reward_interval_ma = 100.0

    def update(self, td_error, cortical_mean_error, retrieval_confidence):
        self.dopamine = td_error
        self.norepinephrine = cortical_mean_error
        if td_error > 0.1:
            self.reward_interval_ma = 0.9 * self.reward_interval_ma + 0.1 * self.steps_since_reward
            self.steps_since_reward = 0
        else:
            self.steps_since_reward += 1

        self.serotonin = np.clip(self.reward_interval_ma / 1000.0, 0, 1)
        self.acetylcholine = 1.0 - np.clip(retrieval_confidence, 0, 1)

class Amygdala:
    def __init__(self, in_dim=512):
        self.in_dim = in_dim
        self.W = np.random.randn(2, in_dim) / np.sqrt(in_dim)

    def process(self, rgm_belief, hc_vector):
        if len(rgm_belief) < 256: rgm_belief = np.pad(rgm_belief, (0, 256 - len(rgm_belief)))
        elif len(rgm_belief) > 256: rgm_belief = rgm_belief[:256]

        inp = np.concatenate([rgm_belief, hc_vector[:256]])
        out = np.dot(self.W, inp)
        threat = np.maximum(0, out[0])
        reward = np.maximum(0, out[1])
        return threat, reward

    def learn(self, rgm_belief, hc_vector, dopamine):
        if len(rgm_belief) < 256: rgm_belief = np.pad(rgm_belief, (0, 256 - len(rgm_belief)))
        elif len(rgm_belief) > 256: rgm_belief = rgm_belief[:256]

        inp = np.concatenate([rgm_belief, hc_vector[:256]])
        err_threat = -dopamine - np.dot(self.W[0], inp) if dopamine < 0 else 0
        err_reward = dopamine - np.dot(self.W[1], inp) if dopamine > 0 else 0

        self.W[0] += 0.01 * err_threat * inp
        self.W[1] += 0.01 * err_reward * inp

class Cerebellum:
    def __init__(self, prop_dim=64, motor_dim=32):
        self.prop_dim = prop_dim
        self.motor_dim = motor_dim
        self.W1 = np.random.randn(128, prop_dim + motor_dim) / np.sqrt(prop_dim + motor_dim)
        self.W2 = np.random.randn(prop_dim, 128) / np.sqrt(128)

    def predict(self, prop_state, motor_cmd):
        inp = np.concatenate([prop_state, motor_cmd])
        h = np.maximum(0, np.dot(self.W1, inp))
        next_prop = np.dot(self.W2, h)
        return next_prop

    def learn(self, prev_prop, prev_motor, actual_next_prop):
        inp = np.concatenate([prev_prop, prev_motor])
        h = np.maximum(0, np.dot(self.W1, inp))
        pred = np.dot(self.W2, h)
        err = actual_next_prop - pred

        self.W2 += 0.01 * np.outer(err, h)
        dh = np.dot(self.W2.T, err)
        dh[h <= 0] = 0
        self.W1 += 0.01 * np.outer(dh, inp)
