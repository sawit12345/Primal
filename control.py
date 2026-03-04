import numpy as np

class PFC:
    def __init__(self, size=64):
        self.size = size
        self.goal = np.zeros(size)

    def set_goal(self, new_goal):
        self.goal = new_goal

class Amygdala:
    def __init__(self, input_dim=512, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(output_dim, input_dim) * 0.01

    def process(self, rgm_belief, hc_retrieval_confidence):
        # hc_retrieval_confidence might just be a scalar or subset of retrieval tuple, we use a zero-padded vec for now
        inp = np.concatenate([rgm_belief, np.zeros(256)])
        out = self.W @ inp
        threat_level = out[0]
        reward_anticipation = out[1]
        return threat_level, reward_anticipation

class Neuromodulators:
    def __init__(self):
        self.dopamine = 0.0
        self.norepinephrine = 0.0
        self.serotonin = 0.0
        self.acetylcholine = 0.0

        self.avg_reward_interval = 100.0
        self.steps_since_reward = 0

    def update(self, td_error, mean_cortical_error, hc_entropy):
        self.dopamine = td_error
        self.norepinephrine = mean_cortical_error

        if td_error > 0.1:
            self.avg_reward_interval = 0.9 * self.avg_reward_interval + 0.1 * self.steps_since_reward
            self.steps_since_reward = 0
        else:
            self.steps_since_reward += 1

        # Serotonin tracks expected time to reward
        self.serotonin = self.avg_reward_interval / 1000.0

        # Acetylcholine tracks uncertainty (entropy of HC retrieval)
        self.acetylcholine = hc_entropy

class BasalGanglia:
    def __init__(self, action_space_size, state_dim=256, goal_dim=64):
        self.action_space_size = action_space_size
        self.state_dim = state_dim

        # Q-values parameterized by a simple linear layer
        self.W = np.random.randn(action_space_size, state_dim) * 0.01
        self.suppression = np.ones(action_space_size) * 0.5
        self.e = np.zeros_like(self.W) # Eligibility traces

        # Goal congruence matrix
        self.W_goal = np.random.randn(action_space_size, goal_dim) * 0.01

        self.last_state = None
        self.last_action = None

    def select_action(self, state, pfc_goal, threat_level, reward_anticipation):
        V = self.W @ state
        goal_congruence = self.W_goal @ pfc_goal

        # Amygdala modulation
        V += reward_anticipation - threat_level

        # Combine value and goal congruence
        V_total = V + goal_congruence

        # Winner-take-all with suppression
        action_scores = V_total - self.suppression
        action = np.argmax(action_scores)

        self.last_state = state
        self.last_action = action
        return action, np.max(V_total)

    def update(self, reward, next_state, lambda_=0.9, base_gamma=0.95, serotonin=0.0):
        if self.last_state is None:
            return 0.0

        # Gamma modulated by serotonin
        gamma = np.clip(base_gamma + 0.05 * serotonin, 0.9, 0.99)

        V_next = np.max(self.W @ next_state)
        V_curr = self.W[self.last_action] @ self.last_state

        td_error = reward + gamma * V_next - V_curr

        # Update eligibility traces
        self.e *= gamma * lambda_
        self.e[self.last_action] += self.last_state

        # Update weights
        self.W += 0.01 * td_error * self.e

        return td_error

class ControlSystem:
    def __init__(self, action_space_size):
        self.pfc = PFC()
        self.amygdala = Amygdala()
        self.neuromodulators = Neuromodulators()
        self.basal_ganglia = BasalGanglia(action_space_size)

    def process(self, state, rgm_belief, hc_entropy, current_reward):
        # Cortical mean error is passed in from cortex, assuming it's available via state wrapper
        # For simplicity, we split the processing steps in integration.
        pass
