import numpy as np

class CoreKnowledgePriors:
    def __init__(self, obj_tracker_limit=16):
        self.obj_tracker_limit = obj_tracker_limit
        self.tracked_objects = {}
        self.agent_classifier = np.random.randn(64, 256) / np.sqrt(256)
        self.W_agent_out = np.random.randn(1, 64) / np.sqrt(64)

        self.grid_cell_bases = self._init_grid_cells()
        self.number_line = np.logspace(0, np.log10(100), 16)
        self.weber_fraction = 0.2

    def _init_grid_cells(self):
        scales = np.logspace(np.log10(0.1), np.log10(10), 6)
        bases = []
        for i in range(64):
            scale = scales[i % 6]
            angle = (i // 6) * (np.pi / 5)
            k = np.array([np.cos(angle), np.sin(angle)]) / scale
            bases.append(k)
        return bases

    def object_permanence(self, current_objects):
        for obj_id, (pos, vel, rep, vis) in list(self.tracked_objects.items()):
            if not vis:
                rep -= 0.02 * rep
                pos += vel

                if np.sum(np.abs(rep)) < 0.1:
                    del self.tracked_objects[obj_id]

    def detect_agents(self, v5_velocity_field):
        v5_flat = v5_velocity_field.flatten()
        if len(v5_flat) > 256: v5_flat = v5_flat[:256]
        else: v5_flat = np.pad(v5_flat, (0, 256 - len(v5_flat)))

        h = np.maximum(0, np.dot(self.agent_classifier, v5_flat))
        is_agent = 1 / (1 + np.exp(-np.dot(self.W_agent_out, h)))[0]
        return is_agent > 0.5

    def approximate_number_system(self, detected_count):
        activations = np.zeros(16)
        for i, q in enumerate(self.number_line):
            std_dev = self.weber_fraction * q
            activations[i] = np.exp(-0.5 * ((detected_count - q) / std_dev)**2)
        return activations

class GlobalWorkspace:
    def __init__(self, layer_hidden_dim=256, broadcast_dim=32):
        self.layer_dim = layer_hidden_dim
        self.broadcast_dim = broadcast_dim
        self.W_proj = np.random.randn(broadcast_dim, layer_hidden_dim) / np.sqrt(layer_hidden_dim)

    def process(self, cortical_states):
        max_act = -1
        max_layer_state = None
        for state in cortical_states:
            act = np.mean(np.abs(state))
            if act > max_act:
                max_act = act
                max_layer_state = state

        broadcast_vec = np.dot(self.W_proj, max_layer_state)
        return broadcast_vec

class TPJSocialCognition:
    def __init__(self, inferred_belief_dim=64, action_dim=32):
        self.belief_dim = inferred_belief_dim
        self.action_dim = action_dim

        self.W1 = np.random.randn(128, 96) / np.sqrt(96)
        self.W2 = np.random.randn(32, 128) / np.sqrt(128)

    def process(self, current_prediction, retrieved_other_belief, observed_action):
        false_belief_signal = current_prediction - retrieved_other_belief
        inp = np.concatenate([false_belief_signal, observed_action])
        h = np.maximum(0, np.dot(self.W1, inp))
        mentalizing_vector = np.dot(self.W2, h)
        return false_belief_signal, mentalizing_vector
