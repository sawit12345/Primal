"""MVP Primal Agent - Minimum Viable Agent for Phase -1."""

import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog

from .brain.log_space_gmm import LogSpaceGMM


class PrimalAgent:
    """Minimum Viable Primal Agent for Phase -1 testing."""
    
    def __init__(self, obs_shape, n_actions, mode='atari'):
        """
        Initialize the agent.
        
        Args:
            obs_shape: Shape of observations (H, W, C) or (H, W)
            n_actions: Number of possible actions
            mode: 'atari' or 'mnist'
        """
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.mode = mode
        
        # Feature extraction parameters
        if mode == 'atari':
            self.downsample_size = (42, 42)
            self.feature_dim = self.downsample_size[0] * self.downsample_size[1] * 2
        else:  # MNIST
            # HOG features for MNIST
            # 28x28 image, 4x4 pixels per cell = 7x7 cells
            # 2x2 cells per block = 6x6 blocks
            # 8 orientations = 1152 features total
            self.feature_dim = 1152  # (7-2+1)^2 * 2*2 * 8 = 1152
        
        # Initialize GMM
        self.gmm = LogSpaceGMM(
            feature_dim=self.feature_dim,
            max_components=50,
            novelty_threshold=0.1,
            m_step_interval=10
        )
        
        # For feature extraction
        self.prev_gray = None
        self.use_hog = (mode == 'mnist')
        
        # For action generation
        self.base_temp = 1.0
        self.fe_running_mean = 1.0
        self.smooth_probs = np.ones(n_actions) / n_actions
        
        # For paddle tracking (Atari only)
        self.paddle_y_range = (180, 200)  # Approximate paddle Y range in original frame
        self.paddle_slot_idx = None
        self.ball_slot_idx = None
        
    def extract_features_atari(self, obs):
        """Extract features for Atari: grayscale, downsample, frame diff."""
        # Convert to grayscale
        if len(obs.shape) == 3:
            gray = rgb2gray(obs)
        else:
            gray = obs.astype(np.float64) / 255.0
            
        # Downsample to 42x42
        small = resize(gray, self.downsample_size, anti_aliasing=True)
        
        # Compute frame difference
        if self.prev_gray is None:
            diff = np.zeros_like(small)
        else:
            prev_small = resize(self.prev_gray, self.downsample_size, anti_aliasing=True)
            diff = small - prev_small
            
        self.prev_gray = gray.copy()
        
        # Flatten and concatenate
        features = np.concatenate([small.flatten(), diff.flatten()])
        
        # Normalize to [-1, 1]
        features = np.clip(features, -1, 1)
        
        return features.astype(np.float64)
    
    def extract_features_mnist(self, image):
        """Extract HOG features for MNIST."""
        if len(image.shape) == 1:
            image = image.reshape(28, 28)
            
        features = hog(
            image,
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            feature_vector=True
        )
        
        return features.astype(np.float64)
    
    def extract_features(self, obs):
        """Extract features based on mode."""
        if self.mode == 'atari':
            return self.extract_features_atari(obs)
        else:
            return self.extract_features_mnist(obs)
    
    def predict_next_state(self, features, action):
        """
        Predict next state given current features and action.
        This is the forward model for action selection.
        """
        predicted_slots = []
        
        for k, slot in enumerate(self.gmm.slots):
            # Predict next position using constant velocity
            next_mu = slot.mu + slot.velocity
            
            # Apply action effect if this is the paddle slot (agent-controlled)
            if self.mode == 'atari' and k == self.paddle_slot_idx:
                # Actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT (ALE/Breakout)
                paddle_speed = 0.1  # Feature space speed
                if action == 2:  # RIGHT
                    next_mu[0] += paddle_speed
                elif action == 3:  # LEFT
                    next_mu[0] -= paddle_speed
                    
            predicted_slots.append(next_mu)
            
        return predicted_slots
    
    def get_action_values(self, obs):
        """
        Get action values for each possible action.
        Returns array of values (higher = better = lower predicted FE).
        """
        current_features = self.extract_features(obs)
        
        values = []
        for action in range(self.n_actions):
            predicted_slots = self.predict_next_state(current_features, action)
            predicted_fe = self.gmm.compute_fe_for_prediction(
                predicted_slots, current_features
            )
            values.append(-predicted_fe)  # Higher value = lower FE = better
            
        return np.array(values)
    
    def act(self, obs):
        """Select action based on current observation."""
        # Get action values
        action_values = self.get_action_values(obs)
        
        # Compute temperature
        temperature = max(0.5, self.base_temp * np.exp(self.fe_running_mean / 10.0))
        
        # Softmax with temperature
        logits = action_values / temperature
        logits -= logits.max()  # Numerical stability
        
        probs = np.exp(logits)
        probs /= probs.sum() + 1e-10
        
        # EMA smoothing
        self.smooth_probs = 0.7 * self.smooth_probs + 0.3 * probs
        self.smooth_probs /= self.smooth_probs.sum() + 1e-10
        
        # Sample action
        action = int(np.random.choice(self.n_actions, p=self.smooth_probs))
        
        return action
    
    def update(self, obs, action, reward, next_obs, done):
        """
        Update the agent with new experience.
        Returns Free Energy of the observation.
        """
        # Extract features
        features = self.extract_features(next_obs)
        
        # Compute FE
        fe = self.gmm.compute_fe(features)
        
        # Update GMM
        self.gmm.update(features)
        
        # Update running mean of FE
        self.fe_running_mean = 0.99 * self.fe_running_mean + 0.01 * fe
        
        # Identify paddle and ball slots (Atari only)
        if self.mode == 'atari':
            self._identify_slots(features)
        
        # Reset on episode end
        if done:
            self.smooth_probs = np.ones(self.n_actions) / self.n_actions
            self.prev_gray = None
            
        return fe
    
    def _identify_slots(self, features):
        """Identify which slots correspond to paddle and ball."""
        # Simple heuristic: slots with highest X velocity are likely the ball
        if len(self.gmm.slots) >= 2:
            velocities = np.array([np.abs(slot.velocity[0]) for slot in self.gmm.slots])
            
            # Slot with highest velocity is likely the ball
            self.ball_slot_idx = int(np.argmax(velocities))
            
            # Another slot with different characteristics might be paddle
            for i, slot in enumerate(self.gmm.slots):
                if i != self.ball_slot_idx:
                    self.paddle_slot_idx = i
                    break
    
    def get_ball_and_paddle_x(self):
        """Get X coordinates of ball and paddle slots."""
        ball_x = None
        paddle_x = None
        
        if self.ball_slot_idx is not None and self.ball_slot_idx < len(self.gmm.slots):
            ball_x = self.gmm.slots[self.ball_slot_idx].mu[0]
            
        if self.paddle_slot_idx is not None and self.paddle_slot_idx < len(self.gmm.slots):
            paddle_x = self.gmm.slots[self.paddle_slot_idx].mu[0]
            
        return ball_x, paddle_x
    
    def reset(self):
        """Reset agent for new episode."""
        self.smooth_probs = np.ones(self.n_actions) / self.n_actions
        self.prev_gray = None
        self.paddle_slot_idx = None
        self.ball_slot_idx = None
        
        # Keep GMM slots but reset velocities
        for slot in self.gmm.slots:
            slot.velocity = np.zeros(self.feature_dim)
            slot.prev_mu = slot.mu.copy()


if __name__ == "__main__":
    # Test the agent
    print("Testing MVP PrimalAgent...")
    
    # Test Atari mode
    agent = PrimalAgent(obs_shape=(210, 160, 3), n_actions=4, mode='atari')
    
    # Simulate some frames
    for i in range(50):
        obs = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        action = agent.act(obs)
        next_obs = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        fe = agent.update(obs, action, 0, next_obs, False)
        
    print(f"Final slots: {agent.gmm.n_components}")
    print(f"Final FE: {fe:.4f}")
    print(f"FE running mean: {agent.fe_running_mean:.4f}")
    
    # Test MNIST mode
    agent_mnist = PrimalAgent(obs_shape=(28, 28), n_actions=10, mode='mnist')
    
    for i in range(20):
        obs = np.random.rand(28, 28).astype(np.float32)
        action = agent_mnist.act(obs)
        next_obs = np.random.rand(28, 28).astype(np.float32)
        fe = agent_mnist.update(obs, action, 0, next_obs, False)
        
    print(f"\nMNIST agent slots: {agent_mnist.gmm.n_components}")
    print(f"MNIST agent FE: {fe:.4f}")
    print("\nPASS: MVP Agent basic functionality working")
