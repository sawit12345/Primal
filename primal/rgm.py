import numpy as np
import scipy.special

class RGMLevel:
    def __init__(self, num_states, num_lower_states, paths=4, sparsity=0.1):
        self.num_states = num_states
        self.num_lower_states = num_lower_states

        self.D = np.zeros((num_states, num_lower_states))
        for i in range(num_states):
            indices = np.random.choice(num_lower_states, size=max(1, int(num_lower_states*sparsity)), replace=False)
            self.D[i, indices] = np.random.dirichlet(np.ones(len(indices)))

        self.B = np.random.dirichlet(np.ones(num_states), size=(paths, num_states)).transpose((1, 2, 0))

        self.s_prior = np.ones(num_states) / num_states
        self.s_posterior = np.copy(self.s_prior)

    def infer(self, lower_state_probs, iterations=5):
        s = np.copy(self.s_prior)
        for _ in range(iterations):
            log_lower = np.log(lower_state_probs + 1e-12)
            log_lik = np.dot(self.D, log_lower)
            log_s = np.log(self.s_prior + 1e-12) + log_lik
            s = scipy.special.softmax(log_s)
        self.s_posterior = s
        return self.s_posterior

    def learn(self, lower_state_probs, learning_rate=0.05):
        coincidence = np.outer(self.s_posterior, lower_state_probs)
        self.D += learning_rate * (coincidence - self.D)
        row_sums = self.D.sum(axis=1, keepdims=True)
        self.D = np.divide(self.D, row_sums, out=np.zeros_like(self.D), where=row_sums!=0)

        if np.max(self.s_posterior) < 0.2 and self.num_states < 2000:
            self.add_state()

    def add_state(self):
        new_row = np.zeros(self.num_lower_states)
        indices = np.random.choice(self.num_lower_states, size=max(1, int(self.num_lower_states*0.1)), replace=False)
        new_row[indices] = np.random.dirichlet(np.ones(len(indices)))
        self.D = np.vstack([self.D, new_row])
        self.num_states += 1
        self.s_prior = np.ones(self.num_states) / self.num_states

class VisualRGM:
    def __init__(self, input_dim):
        self.level1 = RGMLevel(64, input_dim, sparsity=8/input_dim)
        self.level2 = RGMLevel(128, 64)
        self.level3 = RGMLevel(256, 128)
        self.ffa = RGMLevel(32, input_dim)

    def process(self, v1_out_flat, learn=True):
        v1_prob = np.abs(v1_out_flat)
        v1_prob = v1_prob / (np.sum(v1_prob) + 1e-12)

        l1_post = self.level1.infer(v1_prob, iterations=15)
        if len(l1_post) > self.level2.num_lower_states:
            new_cols = np.zeros((self.level2.num_states, len(l1_post) - self.level2.num_lower_states))
            self.level2.D = np.hstack([self.level2.D, new_cols])
            self.level2.num_lower_states = len(l1_post)

        l2_post = self.level2.infer(l1_post, iterations=15)
        if len(l2_post) > self.level3.num_lower_states:
            new_cols = np.zeros((self.level3.num_states, len(l2_post) - self.level3.num_lower_states))
            self.level3.D = np.hstack([self.level3.D, new_cols])
            self.level3.num_lower_states = len(l2_post)

        l3_post = self.level3.infer(l2_post, iterations=15)

        if len(v1_prob) > self.ffa.num_lower_states:
            new_cols = np.zeros((self.ffa.num_states, len(v1_prob) - self.ffa.num_lower_states))
            self.ffa.D = np.hstack([self.ffa.D, new_cols])
            self.ffa.num_lower_states = len(v1_prob)

        ffa_post = self.ffa.infer(v1_prob)

        if learn:
            self.level1.learn(v1_prob)
            self.level2.learn(l1_post)
            self.level3.learn(l2_post)
            self.ffa.learn(v1_prob)

        return l3_post, ffa_post

class AuditoryRGM:
    def __init__(self):
        self.level1 = RGMLevel(64, 128)
        self.level2 = RGMLevel(128, 64)

    def process(self, mel_spec, learn=True):
        mel_prob = mel_spec / (np.sum(mel_spec) + 1e-12)
        l1_post = self.level1.infer(mel_prob)

        if len(l1_post) > self.level2.num_lower_states:
            new_cols = np.zeros((self.level2.num_states, len(l1_post) - self.level2.num_lower_states))
            self.level2.D = np.hstack([self.level2.D, new_cols])
            self.level2.num_lower_states = len(l1_post)

        l2_post = self.level2.infer(l1_post)

        if learn:
            self.level1.learn(mel_prob)
            self.level2.learn(l1_post)

        return l2_post
