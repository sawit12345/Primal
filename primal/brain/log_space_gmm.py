import numpy as np
import scipy.special

class LogSpaceGMM:
    def __init__(self, feature_dim: int, max_components: int = 10, novelty_threshold: float = 0.1):
        self.feature_dim = feature_dim
        self.max_components = max_components
        self.novelty_threshold = novelty_threshold
        
        self.n_components = 1
        self.slots = [self._new_slot(np.zeros(feature_dim))]
        self.velocity = [np.zeros(feature_dim)]
        
        # Base BMR switch needed later in phase 1, disable for MNIST learning phase
        self.bmr_enabled = True

    def _new_slot(self, mu_init):
        return {
            "mu": mu_init.copy(),
            "Sigma_inv": np.eye(self.feature_dim) / 10.0,
            "log_det_Sigma": self.feature_dim * np.log(10.0),
            "log_pi": 0.0,
            "N": 1.0,
            "is_self": False,
            "is_agent": False
        }

    def compute_fe(self, features: np.ndarray) -> float:
        """Returns the negative log-likelihood of features under the current GMM."""
        log_resp = self._compute_log_responsibilities(features)
        return -scipy.special.logsumexp(log_resp)

    def _compute_log_responsibilities(self, features: np.ndarray) -> np.ndarray:
        log_resp = []
        for slot in self.slots:
            diff = features - slot["mu"]
            mahal = np.dot(diff, np.dot(slot["Sigma_inv"], diff))
            log_N = -0.5 * (mahal + slot["log_det_Sigma"] + self.feature_dim * np.log(2 * np.pi))
            log_resp.append(slot["log_pi"] + log_N)
        return np.array(log_resp)

    def e_step(self, features: np.ndarray) -> np.ndarray:
        log_resp = self._compute_log_responsibilities(features)
        log_resp -= scipy.special.logsumexp(log_resp)
        return np.exp(log_resp)

    def update(self, features: np.ndarray):
        resp = self.e_step(features)
        max_resp = np.max(resp)
        
        # Bayesian Model Expansion (BME)
        if max_resp < self.novelty_threshold and self.n_components < self.max_components:
            self.slots.append(self._new_slot(features))
            self.velocity.append(np.zeros(self.feature_dim))
            self.n_components += 1
            # Recalculate responsibilities with new slot
            resp = self.e_step(features)
            
        # Very simple M-Step for MVP (EMA update of mu)
        for k in range(self.n_components):
            r = resp[k]
            # Tracking velocity
            new_mu = self.slots[k]["mu"] + 0.1 * r * (features - self.slots[k]["mu"])
            self.velocity[k] = new_mu - self.slots[k]["mu"]
            self.slots[k]["mu"] = new_mu
            
            # Simple identity variance for MVP to prevent collapse
            self.slots[k]["Sigma_inv"] = np.eye(self.feature_dim) / 10.0
            self.slots[k]["log_det_Sigma"] = self.feature_dim * np.log(10.0)
            
            self.slots[k]["N"] += r
            
        # Update priors
        total_N = sum(slot["N"] for slot in self.slots)
        for k in range(self.n_components):
            self.slots[k]["log_pi"] = np.log(self.slots[k]["N"] / total_N)
            
    def compute_fe_for_prediction(self, predicted_features: np.ndarray, current_features: np.ndarray) -> float:
        # Simple proxy: FE is negative log likelihood of predicted features
        return self.compute_fe(predicted_features)
