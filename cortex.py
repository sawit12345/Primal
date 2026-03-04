import numpy as np

class Thalamus:
    def __init__(self, input_dim=640, num_layers=5):
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.W = np.random.randn(num_layers, input_dim) * 0.01

    def process(self, norepinephrine, current_error_vec, pfc_goal):
        """
        Input to gating network: NE (1) + error (~640) + PFC goal (64) -> let's say we simplify to NE for routing width
        "High norepinephrine opens more thalamic gates. Low norepinephrine closes them."
        """
        # Threshold-based routing mask
        mask = np.zeros((self.num_layers, self.input_dim))
        if norepinephrine > 0.6:
            mask[:] = 1.0 # Open all gates
        elif norepinephrine > 0.3:
            mask[:3] = 1.0 # Open bottom 3 layers
        else:
            mask[0] = 1.0 # Open only layer 1

        return mask

class CorticalLayer:
    def __init__(self, input_dim, hidden_dim, learning_rate=0.005, threshold=0.05):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.threshold = threshold

        # Generative model: Predicts layer below
        self.W_gen = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        # Recognition model: Infers state from layer below (Not strictly required in pure PC, but useful for fast inference)
        self.W_rec = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)

        self.state = np.zeros(hidden_dim)

    def process(self, bottom_up, top_down=None, routing_mask=None):
        if routing_mask is not None:
            bottom_up = bottom_up * routing_mask

        # Prediction
        pred_bottom = self.W_gen @ self.state
        error_bottom = bottom_up - pred_bottom

        mean_abs_error = np.mean(np.abs(error_bottom))

        if mean_abs_error < self.threshold:
            # Gated update: don't change state if error is low
            return self.state, 0.0, pred_bottom

        # State update (gradient descent on prediction error)
        state_grad = self.W_gen.T @ error_bottom
        if top_down is not None:
            # Add top-down prediction error constraint
            pred_top = self.state
            error_top = top_down - pred_top
            state_grad += error_top

        self.state += self.lr * state_grad

        # Weight update (local learning)
        self.W_gen += self.lr * np.outer(error_bottom, self.state)

        return self.state, mean_abs_error, pred_bottom

class CorticalLayer6:
    def __init__(self, input_dim=128, hidden_dim=256, learning_rate=0.005):
        # Recurrent Language Processing (Elman Network)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_in = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W_rec = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b = np.zeros(hidden_dim)
        self.state = np.zeros(hidden_dim)

    def process(self, a2_vector):
        # h(N) = ReLU(W * x(N) + R * h(N-1) + b)
        net_input = self.W_in @ a2_vector + self.W_rec @ self.state + self.b
        self.state = np.maximum(net_input, 0)
        return self.state

class CortexSystem:
    def __init__(self):
        # Layer 1 receives IT (512) + A2 (128) = 640 dim. 512 hidden units.
        self.layer1 = CorticalLayer(input_dim=640, hidden_dim=512)
        # Layer 2-5 receive 512, 256, 256, 256. All have 256 hidden units.
        self.layer2 = CorticalLayer(input_dim=512, hidden_dim=256)
        self.layer3 = CorticalLayer(input_dim=256, hidden_dim=256)
        self.layer4 = CorticalLayer(input_dim=256, hidden_dim=256)
        self.layer5 = CorticalLayer(input_dim=256, hidden_dim=256)
        # Layer 6 is recurrent language layer, feeds layer 5
        self.layer6 = CorticalLayer6(input_dim=128, hidden_dim=256)

        self.thalamus = Thalamus()

    def process(self, it_vector, a2_vector, norepinephrine, pfc_goal):
        l1_input = np.concatenate([it_vector, a2_vector])

        # Thalamus routing
        # Pass a dummy current error vector for now
        routing_masks = self.thalamus.process(norepinephrine, current_error_vec=np.zeros(640), pfc_goal=pfc_goal)

        l6_state = self.layer6.process(a2_vector)

        # Top-down pass for predictions (simplified cascade)
        # In full predictive coding, inference happens concurrently. We simulate via a bottom-up pass followed by top-down adjustment.
        s1, e1, p1 = self.layer1.process(bottom_up=l1_input, routing_mask=routing_masks[0])
        s2, e2, p2 = self.layer2.process(bottom_up=s1, routing_mask=routing_masks[1][:512])
        s3, e3, p3 = self.layer3.process(bottom_up=s2, routing_mask=routing_masks[2][:256])
        s4, e4, p4 = self.layer4.process(bottom_up=s3, routing_mask=routing_masks[3][:256])
        s5, e5, p5 = self.layer5.process(bottom_up=s4, routing_mask=routing_masks[4][:256], top_down=l6_state)

        mean_error = np.mean([e1, e2, e3, e4, e5])

        return {
            'l5_state': s5,
            'l6_state': l6_state,
            'mean_error': mean_error
        }
