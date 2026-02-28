import numpy as np
import cv2
from primal.brain.log_space_gmm import LogSpaceGMM
from skimage.feature import hog

LEFT = 3
RIGHT = 2
paddle_speed = 2.0

class PrimalAgent:
    def __init__(self, obs_shape: tuple, n_actions: int, is_mnist: bool = False):
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.is_mnist = is_mnist
        
        self.feature_dim = 3528 if not is_mnist else 324
        self.gmm = LogSpaceGMM(feature_dim=self.feature_dim, max_components=10 if is_mnist else 5)
        
        self.prev_gray = None
        self.fe_running_mean = 1.0
        self.base_temp = 1.0
        self.smooth_probs = np.ones(self.n_actions) / self.n_actions
        
    def reset(self):
        self.smooth_probs = np.ones(self.n_actions) / self.n_actions
        self.prev_gray = None

    def extract_features(self, obs: np.ndarray) -> np.ndarray:
        if self.is_mnist:
            # MVP MNIST HOG features
            return hog(obs.reshape(28, 28), orientations=8,
                       pixels_per_cell=(4, 4), cells_per_block=(2, 2),
                       feature_vector=True)
            
        # Breakout feature extraction
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) if len(obs.shape) == 3 else obs
        gray = cv2.resize(gray, (42, 42))
        
        if self.prev_gray is None:
            self.prev_gray = gray
            
        diff = gray.astype(np.float32) - self.prev_gray.astype(np.float32)
        self.prev_gray = gray
        
        features = np.concatenate([gray.flatten(), diff.flatten()])
        features = (features / 255.0) * 2.0 - 1.0
        return features

    def predict_next_state(self, current_features, action):
        """
        Given current GMM state and an action, predict the next feature vector.
        This is the forward model. Without this, action selection is random.
        """
        predicted_slots = []
        for k, slot in enumerate(self.gmm.slots):
            next_mu = slot["mu"] + self.gmm.velocity[k]   # physics: constant velocity
            if slot["is_self"]:                    # this is the paddle
                # action directly moves paddle
                if action == LEFT:
                    next_mu[0] -= paddle_speed
                elif action == RIGHT:
                    next_mu[0] += paddle_speed
            predicted_slots.append(next_mu)
        return predicted_slots

    def get_action_values(self, current_obs):
        """
        For each action, predict next state, compute FE of that prediction vs
        the actual next observation. Lower predicted FE = better action.
        """
        current_features = self.extract_features(current_obs)
        values = []
        for action in range(self.n_actions):
            predicted_next = self.predict_next_state(current_features, action)
            # In MVP, we just take the first slot's prediction to compute FE proxy
            pred_feature = predicted_next[0] if len(predicted_next) > 0 else current_features
            
            # compute_fe_for_prediction proxy
            diff = pred_feature - current_features
            predicted_fe = np.sum(diff**2)
            
            values.append(-predicted_fe)   # higher value = lower FE = better
        return np.array(values)

    def act(self, obs):
        if self.is_mnist:
            features = self.extract_features(obs)
            resp = self.gmm.e_step(features)
            return int(np.argmax(resp))

        features = self.extract_features(obs)
        action_values = self.get_action_values(obs)
        temperature = max(0.5, self.base_temp * np.exp(self.fe_running_mean / 10.0))
        logits = action_values / temperature
        logits -= logits.max()   # numerical stability
        probs = np.exp(logits)
        probs /= (probs.sum() + 1e-8)
        
        # EMA smoothing
        self.smooth_probs = 0.7 * self.smooth_probs + 0.3 * probs
        self.smooth_probs /= (self.smooth_probs.sum() + 1e-8)
        
        return int(np.random.choice(self.n_actions, p=self.smooth_probs))

    def update(self, obs, action, reward, next_obs, done):
        features = self.extract_features(next_obs)
        fe = self.gmm.compute_fe(features)
        
        # Simple self-assignment for MVP
        if len(self.gmm.slots) > 0:
            self.gmm.slots[0]["is_self"] = True
            
        self.gmm.update(features)
        self.fe_running_mean = 0.99 * self.fe_running_mean + 0.01 * fe
        if done:
            self.smooth_probs = np.ones(self.n_actions) / self.n_actions  # reset on episode end
        return fe
