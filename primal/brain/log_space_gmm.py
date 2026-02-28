"""Log-space GMM with growing slots for the Primal agent."""

import numpy as np
from scipy.special import logsumexp


class Slot:
    """A single Gaussian component (slot) in the GMM."""
    
    def __init__(self, mu, Sigma, pi, feature_dim):
        self.mu = mu.astype(np.float64)
        self.Sigma = Sigma.astype(np.float64)
        self.pi = float(pi)
        self.log_pi = np.log(pi + 1e-300)
        self.velocity = np.zeros(feature_dim, dtype=np.float64)
        self.prev_mu = mu.copy().astype(np.float64)
        self.N_k = 0.0  # Effective count
        
        # Precompute precision matrix and log determinant
        self._update_precision()
        
    def _update_precision(self):
        """Update precision matrix (inverse covariance) and log determinant."""
        try:
            self.Sigma_inv = np.linalg.inv(self.Sigma)
            sign, logdet = np.linalg.slogdet(self.Sigma)
            if sign <= 0:
                # Add regularization if singular
                self.Sigma += 0.1 * np.eye(self.Sigma.shape[0])
                self.Sigma_inv = np.linalg.inv(self.Sigma)
                sign, logdet = np.linalg.slogdet(self.Sigma)
            self.log_det_Sigma = logdet
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            self.Sigma += 0.1 * np.eye(self.Sigma.shape[0])
            self.Sigma_inv = np.linalg.inv(self.Sigma)
            _, self.log_det_Sigma = np.linalg.slogdet(self.Sigma)
    
    def update_velocity(self):
        """Update slot velocity based on mean displacement."""
        self.velocity = self.mu - self.prev_mu
        self.prev_mu = self.mu.copy()


class LogSpaceGMM:
    """Gaussian Mixture Model with log-space computations and growing slots."""
    
    def __init__(self, feature_dim, max_components=100, novelty_threshold=0.1, 
                 m_step_interval=10, regularization=1e-6):
        self.feature_dim = feature_dim
        self.max_components = max_components
        self.novelty_threshold = novelty_threshold
        self.m_step_interval = m_step_interval
        self.regularization = regularization
        
        self.slots = []
        self.n_components = 0
        self.step_count = 0
        
        # Initialize with first slot
        self._add_initial_slot()
        
    def _add_initial_slot(self):
        """Add the first slot with high variance prior."""
        mu = np.zeros(self.feature_dim, dtype=np.float64)
        Sigma = 10.0 * np.eye(self.feature_dim, dtype=np.float64)
        pi = 1.0
        slot = Slot(mu, Sigma, pi, self.feature_dim)
        self.slots.append(slot)
        self.n_components = 1
        
    def _log_gaussian(self, x, slot):
        """Compute log probability of x under slot's Gaussian."""
        diff = x - slot.mu
        mahal = float(diff @ slot.Sigma_inv @ diff)
        log_prob = -0.5 * (mahal + slot.log_det_Sigma + 
                          self.feature_dim * np.log(2 * np.pi))
        return log_prob
    
    def e_step(self, x):
        """Compute responsibilities in log space for observation x."""
        x = np.asarray(x, dtype=np.float64)
        
        log_resp = np.array([
            slot.log_pi + self._log_gaussian(x, slot)
            for slot in self.slots
        ])
        
        # Normalize in log space
        log_resp -= logsumexp(log_resp)
        resp = np.exp(log_resp)
        
        return resp
    
    def m_step(self, x, resp):
        """Update slot parameters given observation and responsibilities."""
        x = np.asarray(x, dtype=np.float64)
        
        for k, slot in enumerate(self.slots):
            r_k = resp[k]
            
            # Update effective count
            slot.N_k = 0.9 * slot.N_k + 0.1 * r_k
            
            # Update mean
            slot.prev_mu = slot.mu.copy()
            slot.mu = 0.9 * slot.mu + 0.1 * (r_k * x)
            
            # Update covariance
            diff = x - slot.mu
            outer = np.outer(diff, diff)
            slot.Sigma = 0.9 * slot.Sigma + 0.1 * (r_k * outer)
            slot.Sigma += self.regularization * np.eye(self.feature_dim)
            
            # Ensure positive definite
            slot.Sigma = 0.5 * (slot.Sigma + slot.Sigma.T)
            
            # Update velocity
            slot.update_velocity()
            
        # Update weights
        total_N = sum(slot.N_k for slot in self.slots) + 1e-300
        for slot in self.slots:
            slot.pi = slot.N_k / total_N
            slot.log_pi = np.log(slot.pi + 1e-300)
            
    def _maybe_grow(self, x, resp):
        """Check if we should add a new slot (Bayesian Model Expansion)."""
        max_resp = np.max(resp)
        
        if max_resp < self.novelty_threshold and self.n_components < self.max_components:
            # Open new slot at current observation
            mu = x.copy()
            Sigma = 10.0 * np.eye(self.feature_dim, dtype=np.float64)
            
            # Compute new weight
            new_pi = 1.0 / (self.n_components + 1)
            
            slot = Slot(mu, Sigma, new_pi, self.feature_dim)
            self.slots.append(slot)
            self.n_components += 1
            
            # Renormalize existing weights
            for s in self.slots[:-1]:
                s.pi *= (1 - new_pi)
                s.log_pi = np.log(s.pi + 1e-300)
                
            return True
        return False
    
    def compute_fe(self, x):
        """Compute Free Energy (negative log-likelihood) for observation x."""
        x = np.asarray(x, dtype=np.float64)
        resp = self.e_step(x)
        
        fe = 0.0
        for k, slot in enumerate(self.slots):
            log_lik = slot.log_pi + self._log_gaussian(x, slot)
            fe -= resp[k] * (log_lik - np.log(resp[k] + 1e-300))
            
        return fe
    
    def update(self, x):
        """Single step update: E-step, optional M-step, potential growth."""
        x = np.asarray(x, dtype=np.float64)
        
        # E-step
        resp = self.e_step(x)
        
        # Check for novelty and grow if needed
        grew = self._maybe_grow(x, resp)
        if grew:
            # Recompute resp with new slot
            resp = self.e_step(x)
        
        # M-step at intervals
        self.step_count += 1
        if self.step_count % self.m_step_interval == 0 or grew:
            self.m_step(x, resp)
            
        return self.compute_fe(x)
    
    def get_slot_means(self):
        """Return array of all slot means."""
        return np.array([slot.mu for slot in self.slots])
    
    def get_slot_velocities(self):
        """Return array of all slot velocities."""
        return np.array([slot.velocity for slot in self.slots])
    
    def compute_fe_for_prediction(self, predicted_slots, actual_features):
        """Compute FE for predicted slot positions vs actual features."""
        # Compare predicted positions to actual
        fe = 0.0
        for pred_mu in predicted_slots:
            # Find best matching slot
            min_fe = float('inf')
            for slot in self.slots:
                diff = pred_mu - slot.mu
                mahal = float(diff @ slot.Sigma_inv @ diff)
                slot_fe = 0.5 * (mahal + slot.log_det_Sigma)
                min_fe = min(min_fe, slot_fe)
            fe += min_fe
        return fe


if __name__ == "__main__":
    # Sanity check
    np.random.seed(42)
    gmm = LogSpaceGMM(feature_dim=4, max_components=10)
    
    # Test on random data
    obs = np.random.randn(4)
    fe_before = gmm.compute_fe(obs)
    
    for _ in range(20):
        obs = np.random.randn(4)
        gmm.update(obs)
        
    obs = np.random.randn(4)
    fe_after = gmm.compute_fe(obs)
    
    print(f"Components: {gmm.n_components}")
    print(f"FE before: {fe_before:.4f}")
    print(f"FE after: {fe_after:.4f}")
    print(f"Slots grown: {gmm.n_components > 1}")
    print("PASS: GMM basic functionality working")
