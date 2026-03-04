import numpy as np
import scipy.special

class CorticalLayer:
    def __init__(self, in_dim, hidden_dim, learning_rate=0.005, threshold=0.05):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.threshold = threshold

        self.W_gen = np.random.randn(in_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.b_gen = np.zeros(in_dim)

        self.W_rec = np.random.randn(hidden_dim, in_dim) / np.sqrt(in_dim)
        self.b_rec = np.zeros(hidden_dim)

        self.state = np.zeros(hidden_dim)
        self.prediction_error = np.zeros(in_dim)

    def infer(self, bottom_up_input, top_down_prediction=None):
        own_pred = np.dot(self.W_gen, self.state) + self.b_gen
        self.prediction_error = bottom_up_input - own_pred
        mean_abs_error = np.mean(np.abs(self.prediction_error))

        if mean_abs_error < self.threshold:
            return self.state, np.zeros_like(self.state), False

        state_grad = np.dot(self.W_gen.T, self.prediction_error)

        if top_down_prediction is not None:
            state_grad += (top_down_prediction - self.state)

        self.state += self.learning_rate * state_grad

        bu_state = np.dot(self.W_rec, bottom_up_input) + self.b_rec
        self.state += self.learning_rate * (bu_state - self.state)
        self.state = np.maximum(0, self.state)

        return self.state, self.prediction_error, True

    def learn(self, bottom_up_input):
        err_unit = np.maximum(0, self.prediction_error)
        self.W_gen += self.learning_rate * np.outer(err_unit, self.state)
        self.b_gen += self.learning_rate * err_unit

        self.W_rec += self.learning_rate * np.outer(self.state, bottom_up_input)
        self.b_rec += self.learning_rate * self.state

class RecurrentLayer6:
    def __init__(self, in_dim=128, hidden_dim=256, learning_rate=0.005):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        self.W = np.random.randn(hidden_dim, in_dim) / np.sqrt(in_dim)
        self.R = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.b = np.zeros(hidden_dim)

        self.W_out = np.random.randn(512, hidden_dim) / np.sqrt(hidden_dim)
        self.h = np.zeros(hidden_dim)

    def process(self, a2_input, reset=False):
        if reset:
            self.h = np.zeros(self.hidden_dim)

        activation = np.dot(self.W, a2_input) + np.dot(self.R, self.h) + self.b
        self.h = np.maximum(0, activation)

        phoneme_dist = scipy.special.softmax(np.dot(self.W_out, self.h))
        return self.h, phoneme_dist

class Thalamus:
    def __init__(self, in_dim=640, out_layers=5):
        self.in_dim = in_dim
        self.out_layers = out_layers
        net_in_dim = 1 + 256 + 64
        self.W1 = np.random.randn(128, net_in_dim) / np.sqrt(net_in_dim)
        self.W2 = np.random.randn(out_layers, 128) / np.sqrt(128)

    def process(self, ne, error_vec, pfc_goal):
        inp = np.concatenate([[ne], error_vec, pfc_goal])
        h = np.maximum(0, np.dot(self.W1, inp))
        routing_mask = scipy.special.expit(np.dot(self.W2, h))
        routing_mask = np.clip(routing_mask * (ne * 2), 0, 1)
        return routing_mask > 0.5

class CorticalHierarchy:
    def __init__(self):
        self.layer1 = CorticalLayer(640, 512)
        self.layer2 = CorticalLayer(512, 256)
        self.layer3 = CorticalLayer(256, 256)
        self.layer4 = CorticalLayer(256, 256)
        self.layer5 = CorticalLayer(256, 256)
        self.layer6 = RecurrentLayer6(128, 256)
        self.thalamus = Thalamus(640, 5)

    def process(self, it_visual, a2_audio, ne_scalar, pfc_goal):
        if len(it_visual) > 512:
             it_visual = it_visual[:512]
        elif len(it_visual) < 512:
             it_visual = np.pad(it_visual, (0, 512 - len(it_visual)))

        l1_input = np.concatenate([it_visual, a2_audio])

        prev_error = self.layer5.prediction_error if self.layer5.prediction_error is not None else np.zeros(256)
        mask = self.thalamus.process(ne_scalar, prev_error, pfc_goal)

        active_layers = 0
        mean_abs_errors = []

        l6_state, _ = self.layer6.process(a2_audio)

        if mask[4]:
            l5_state, l5_err, act = self.layer5.infer(self.layer4.state, l6_state)
            if act: active_layers += 1; mean_abs_errors.append(np.mean(np.abs(l5_err)))
            self.layer5.learn(self.layer4.state)

        if mask[3]:
            l4_state, l4_err, act = self.layer4.infer(self.layer3.state, self.layer5.state)
            if act: active_layers += 1; mean_abs_errors.append(np.mean(np.abs(l4_err)))
            self.layer4.learn(self.layer3.state)

        if mask[2]:
            l3_state, l3_err, act = self.layer3.infer(self.layer2.state, self.layer4.state)
            if act: active_layers += 1; mean_abs_errors.append(np.mean(np.abs(l3_err)))
            self.layer3.learn(self.layer2.state)

        if mask[1]:
            l2_state, l2_err, act = self.layer2.infer(self.layer1.state, self.layer3.state)
            if act: active_layers += 1; mean_abs_errors.append(np.mean(np.abs(l2_err)))
            self.layer2.learn(self.layer1.state)

        if mask[0]:
            l1_state, l1_err, act = self.layer1.infer(l1_input, self.layer2.state)
            if act: active_layers += 1; mean_abs_errors.append(np.mean(np.abs(l1_err)))
            self.layer1.learn(l1_input)

        total_error = np.mean(mean_abs_errors) if len(mean_abs_errors) > 0 else 0.0

        return self.layer5.state, l6_state, total_error
