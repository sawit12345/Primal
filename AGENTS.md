# AGENTS.md

Hey! You are the autonomous Agent assigned to build a framework named Primal by Primeval. I am your Checker. This is a massive task, so we need to approach it like a senior engineering team. You will plan thoroughly and implement efficiently. I will verify your output rigorously.

The goal is to unify brain-inspired mechanics into a lightweight, high-performance architecture. We want the engineering benefits of how the brain works without the bloated, expensive biological computations. 

Before writing any code, you must read this entire document, understand the constraints, and follow the exact phases laid out below. Feel free to come back to this AGENTS.md whenever you want or if you lose track of the task. Do not be arrogant, just stick to the process, listen to my feedback as your Checker, and refine as you go.

## The Core Framework (Primal)
Primal must unify the following concepts into a single, cohesive system. 

1. **Active Inference**. Keep it strictly to "Change the world to match yours, or change yours to match the world". Use Free Energy, Bayesian Inference, and predictive coding for action generation. Full Variational Inference planning is too expensive and slow. We are just taking the raw, practical ideas of Active Inference.
2. **Log Space Fusion & Exponential Conjugates**. Use this for cheap Bayesian updating and continual learning. Implement a Gaussian Mixture Model for growing components without fixing the parameters initially. Start minimal and expand, ensuring it does not forget until Bayesian Model Reduction decides to prune or merge it. 
3. **Core Knowledge and Transfer Learning**. Implement Spelke's priors (objects, space and geometry, number, agents, and physics). 
4. **Theory Theory**. Take the main idea of maintaining multiple beliefs or hypotheses based on current context, prediction, and selection, completely avoiding full Variational Inference.
5. **Bayesian Model Reduction (BMR)**. Use this specifically to merge or prune components and keep the model lightweight.
6. **Abstracted Brain Mechanisms**. Implement functional, simplified versions of the PFC, VLPFC, Retinas, Macula, Fovea, Magnocellular and Parvocellular pathways, Basal Ganglia, Amygdala, Thalamus, Homeostasis, Anterior temporal lobe, Hippocampus, Hemisphere split, and Inferior temporal cortex.
7. **Cerebellar Smoothing**. Implement this for fluid motor output.
8. **Weber-Fechner Law**. Implement logarithmic Approximate Number System precision scaling.
9. **Superior Colliculus**. Abstract its orienting, saccadic eye movements, and visual attention functions.
10. **Occipital Lobe**. Abstract its visual processing pipeline from V1 edge detection up to V4 complex shape abstractions.
11. **Hemifield Pull Imbalance**. Map this over coordinates to simulate biological visual attention.
12. **Precision Alpha Scaling**. Model this to simulate survival urgency.
13. **Intuitive Physics Simulation**. Implement an 18-step lattice-boltzmann fluid advection specifically to support Spelke's physics priors.
14. **Proprioception**. Model body awareness as a continuous Gaussian distribution.
15. **Markovian Temporal Decay**. Apply a prior belief decay formula (0.7 old + 0.3 new).
16. **Renormalization Group**. Use this for scaling and smoothing states.
17. **Common Sense Reasoning**. Build a mechanism for filling in the gaps of missing information.
18. **Slot Centric Processing**. Bind this directly to the Gaussian Mixture Model for object representation.
19. **Open Expansion**. Brainstorm and add any other highly beneficial, lightweight sub-symbolic mechanisms you can think of.
20. **Hierarchical Predictive Coding**. Implement top-down prediction signals and bottom-up prediction error signals. Higher cortical layers must pass context down, and lower visual layers must pass surprise up.

## Core Modules: Mathematical Foundations and Professional Code
To ensure we do not over-engineer, you must base your implementations on these typed, vectorized, and professional class structures for every major functional section.

### Section A: Active Inference & Theory Theory (Points 1, 4, 12)
* **Math**: Free Energy is approximated as the difference between Expected Energy and Entropy. $F = E_Q[ \log Q(s) - \log P(o, s) ]$. Action selection minimizes this Free Energy.
* **Code**:
```python
import numpy as np

class ActiveInferenceEngine:
    """Handles core predictive coding and Free Energy minimization."""
    
    def __init__(self, precision_alpha: float = 1.0):
        self.precision_alpha = precision_alpha

    def calculate_free_energy(self, predictions: np.ndarray, observations: np.ndarray) -> float:
        """
        Calculates the variational free energy in log space.
        Surprise is scaled by the survival urgency parameter (precision_alpha).
        """
        predictions = np.clip(predictions, 1e-8, 1.0)
        observations = np.clip(observations, 1e-8, 1.0)
        
        prediction_error = np.log(observations) - np.log(predictions)
        surprise = np.sum(prediction_error ** 2)
        entropy = -np.sum(predictions * np.log(predictions))
        
        free_energy = (surprise * self.precision_alpha) - entropy
        return float(free_energy)
```

### Section B: Log Space Fusion, GMM & BMR (Points 2, 5, 15, 18)
* **Math**: Bayesian updating via Log-Sum-Exp to prevent numerical underflow. $P(x) = \log \sum \exp(\log w_i + \log N(x|\mu_i, \sigma_i))$. Markovian decay applies a 0.7 temporal momentum.
* **Code**:
```python
import numpy as np
from scipy.special import logsumexp
from typing import List

class GaussianComponent:
    """Represents a single slot in the GMM."""
    def __init__(self, mean: np.ndarray, log_var: np.ndarray):
        self.mean = mean
        self.log_var = log_var
        self.n_obs = 0   # tracks maturity; BMR must not merge slots where n_obs < 5

class LogSpaceGMM:
    """Handles Gaussian Mixture Model operations securely in log space."""
    
    def __init__(self, decay_rate: float = 0.7):
        self.components: List[GaussianComponent] = []
        self.decay_rate = decay_rate

    def log_space_bayesian_update(self, log_prior: np.ndarray, log_likelihood: np.ndarray) -> np.ndarray:
        """
        Fuses prior and likelihood securely in log space to prevent numerical underflow.
        """
        unnormalized_posterior = log_prior + log_likelihood
        evidence = logsumexp(unnormalized_posterior)
        log_posterior = unnormalized_posterior - evidence
        return log_posterior

    def markovian_decay(self, old_belief: np.ndarray, new_belief: np.ndarray) -> np.ndarray:
        """Applies temporal momentum to beliefs."""
        return self.decay_rate * old_belief + (1.0 - self.decay_rate) * new_belief

    def m_step_update(self, component: GaussianComponent, new_obs: np.ndarray) -> None:
        """
        Updates a component mean with Markovian decay and increments its observation
        counter. n_obs is required for BMR maturity gating. Without incrementing it,
        n_obs stays 0 forever, BMR skips every slot, and the model never prunes.
        """
        component.mean = self.markovian_decay(component.mean, new_obs)
        component.n_obs += 1
```

### Section C: Visual Mechanics & Saliency (Points 6, 8, 9, 10, 11)
* **Math**: Weber-Fechner scaling for numerical perception $P = k \log(S)$. Hemifield pull calculates an attention vector based on foveal center offsets.
* **Code**:
```python
import numpy as np

class VisualSystem:
    """Abstracts retinal foveation and numerical perception scaling."""
    
    def __init__(self, weber_k: float = 1.0):
        self.weber_k = weber_k

    def apply_foveal_mask(self, image: np.ndarray, focal_y: int, focal_x: int, radius: float) -> np.ndarray:
        """
        Simulates foveal vision with high resolution at the focal point, 
        degrading to low resolution in the periphery (Magnocellular pathway).
        """
        y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
        squared_dist = (x_coords - focal_x)**2 + (y_coords - focal_y)**2
        distance = np.sqrt(squared_dist)
        
        mask = np.clip(1.0 - (distance / radius), 0.1, 1.0)
        return image * mask

    def weber_fechner_scaling(self, stimulus_intensity: np.ndarray) -> np.ndarray:
        """Scales numerical perception logarithmically."""
        return self.weber_k * np.log(stimulus_intensity + 1.0)
```

### Section D: Intuitive Physics & Core Knowledge (Points 3, 7, 13, 14, 16)
* **Math**: Lattice-Boltzmann method for fluid/physics advection. $f_i(x+c_i \Delta t, t+\Delta t) = f_i(x,t) - \frac{1}{\tau}(f_i(x,t) - f_i^{eq}(x,t))$.
* **Code**:
```python
import numpy as np

class IntuitivePhysicsEngine:
    """Implements fluid advection for Spelke physics priors."""
    
    def __init__(self, tau: float = 0.6):
        self.tau = tau
        # D2Q9 lattice velocities (cy, cx)
        self.velocities = np.array([
            [0, 0], [0, 1], [1, 0], [0, -1], [-1, 0],
            [1, 1], [-1, 1], [-1, -1], [1, -1]
        ])

    def lattice_boltzmann_step(self, grid: np.ndarray) -> np.ndarray:
        """
        Executes a vectorized D2Q9 streaming and collision step.

        The outer loop runs exactly 9 times (one per lattice direction). This is
        O(9) and perfectly acceptable. The forbidden pattern is an inner loop over
        grid cells: for x in range(W): for y in range(H). That is O(W*H) Python
        iterations and will destroy performance. Every operation inside this loop
        must act on the full grid array at once via np.roll and numpy arithmetic.
        """
        next_grid = np.zeros_like(grid)
        grid_mean = grid.mean(axis=0)
        
        for i, (cy, cx) in enumerate(self.velocities):
            rolled = np.roll(grid[i], shift=(cy, cx), axis=(0, 1))
            next_grid[i] = rolled - (rolled - grid_mean) / self.tau
            
        return next_grid
```

## Failure Modes to Avoid
Before building the architecture, you must design against these specific failure modes.

1. **Gaussian Component Explosion**: The GMM will try to spawn a new component for every slight variation. You must implement aggressive Bayesian Model Reduction (BMR) to prune overlapping distributions based on Kullback-Leibler divergence.
2. **Variational Inference Bloat**: Do not write full VI graphical unrolling. It will destroy CPU performance. Use greedy, local predictive coding for step-by-step Free Energy minimization.
3. **Log Space Underflow**: When fusing distributions, raw probabilities will multiply to zero. Everything must be passed through `scipy.special.logsumexp`.
4. **Saccadic Thrashing**: The Superior Colliculus logic might get stuck rapidly bouncing between two equally salient visual points. Implement Inhibition of Return (IOR) to force the attention gate to explore new visual coordinates.
5. **Action Signal Collapse**: `get_action_values()` must return values that differ by more than 0.5 across actions. If you compute FE in the raw feature space (thousands of dims), moving the paddle 3 pixels changes almost nothing and every action gets the same score, meaning the agent becomes random. Compute action values in slot-position space instead: find the ball slot (highest velocity magnitude), predict its next x position, score each action by negative squared distance between predicted ball x and predicted paddle x. Signal must be measurable. Add this assertion to your verify step:
   ```python
   signal = max(action_values) - min(action_values)
   assert signal > 5.0, f"Action signal {signal:.4f} too weak, agent will be random"
   ```
6. **Symbolic Cheating**: Never read game RAM (`env.unwrapped.ale.getRAM()`), never hardcode pixel regions, never write `if env_name == "breakout"`. The agent must infer all world state from GMM slots alone. Ball = slot with highest velocity magnitude. Paddle = proprioception Kalman state. Run `grep -r "getRAM\|ram\[" primal/` before the Breakout test to ensure it returns empty.
7. **Smoke Run Passed as Episode**: 150 steps is not an episode. Run Breakout until `terminated or truncated`. No `max_steps` below 10,000.
8. **BMR Destroying Immature Prototypes**: BMR merges slots whose Gaussians overlap significantly. After seeing only 1 to 3 samples, every slot has huge covariance because the distribution is wide due to uncertainty, not because it is similar to another class. Wide Gaussians look similar to BMR even if their means are far apart, so BMR merges digit prototypes and accuracy collapses to roughly 10%. The correct fix is slot maturity gating: every slot tracks `n_obs` (number of observations absorbed). BMR must skip any slot where `n_obs < min_obs` (suggested: 5). Once a slot has seen enough observations, its covariance tightens around the true mean, and genuinely different prototypes will no longer overlap, so BMR will leave them alone naturally. Do NOT disable BMR globally. That breaks the sub-symbolic purity of the system. Instead fix BMR to be maturity-aware:
   ```python
   def should_merge(slot_a: GaussianComponent, slot_b: GaussianComponent, threshold: float) -> bool:
       if slot_a.n_obs < 5 or slot_b.n_obs < 5:
           return False   # never merge immature slots
       kl = kl_divergence(slot_a, slot_b)
       return kl < threshold
   ```
   The GMM must also call `m_step_update()` on every observation so that `n_obs` actually increments. If `n_obs` is never incremented, it stays 0 forever, the maturity guard fires on every slot unconditionally, and BMR silently stops pruning anything. This means MNIST slot count is discovered automatically. The agent opens a slot whenever a novel digit triggers high FE, and BMR only prunes it if it genuinely converges with another after sufficient observations.
9. **Shortcut Features Breaking Sub-Symbolic Purity**: Using HOG, Euler number, or skeletonize for MNIST is symbolic, as these are handcrafted descriptors that encode human knowledge about digit structure. A truly sub-symbolic agent does not know it is looking at a digit any more than it knows it is playing Breakout. It simply receives an observation and runs it through its visual pipeline. The occipital module already does this: DoG on/off channels detect edges, end-stopped cells fire at stroke endpoints and corners, 24 Gabor filters capture oriented edges at multiple scales, V2 captures curvature, V4 responds to closed loops and complex shapes. End-stopped cells naturally encode what Euler number encodes manually. V4 cells naturally encode what skeletonize encodes manually. The 94.79% ceiling happened because HOG was used as a shortcut instead of trusting the visual pipeline. The fix is to feed every observation (whether it is a game frame or a digit image) through the same visual pathway without modification. The architecture must never branch on what type of input it is receiving.
10. **Hippocampus Replay Direction**: If the hippocampus replays miss events by calling `gmm.update(miss_features)`, it teaches the model that miss states are normal, lowers FE on misses, and score drops episode over episode. Replay must only update action values. Call `update_action_value(bad_action, delta=-abs(fe))`. Never call `gmm.update()` inside replay.

## Strict Constraints
* **Zero Tolerance for Laziness or Stubs**. You must write complete, production-ready code. Do not use `pass`, `NotImplementedError`, or placeholders like `# TODO: implement this logic later`. Every function must contain the actual, working mathematical and logical implementation. I am your Checker, not your assistant. I will not finish your code for you.
* **Universal Generalization (100% Sub-symbolic)**. The framework must be entirely universal. By remaining completely sub-symbolic, it will naturally solve any environment without modification. No hardcoded domain knowledge, no hardcoded policies, no hand-tuned heuristics, and no hints. The architecture must be completely blind about the world at initialization. It must never know if it is playing Breakout or classifying MNIST digits.
* **No Seed Hunting**. You are strictly forbidden from using fixed or tuned seeds to achieve the benchmark. The framework must be mathematically robust. Do not pass a seed to `env.reset()` or any random number generator. If the framework only works with a specific seed, the logic is brittle and you must fix the math, not the seed.
* **No Pre-trained Weights**. You cannot download or load any pre-trained models, weights, or heuristic filters. Everything must be generated mathematically at runtime.
* **Numerical Stability and NaN Prevention**. Bayesian updating and log space math are highly prone to NaN values or infinite values if handled poorly. You must use rigorous clipping and `logsumexp`. Add strict assertions checking for NaNs after every major math block to instantly catch numerical collapses before they infect the Gaussian Mixture Model.
   ```python
   assert not np.isnan(computed_array).any(), "Fatal: NaN value detected in math block."
   ```
* **Lightweight**. As long as you implement the items above strictly for their engineering benefits, it will be fast. Use Python, numpy, and scipy. Avoid massive deep learning libraries unless absolutely necessary.
* **Do Not Over-engineer**. Do not overthink it, do not oversimplify it, and do not write bloated math. Keep the logic sharp and functional.

## Execution Phases

### Phase 1: The Plan (Deep Think & Architecture)
Before writing any actual framework code, you must brainstorm and write out a concrete architectural plan.
* Map out how all 20+ components connect.
* Define the data structures. Everything needs to flow logically.
* Verify your math mentally based on the core module examples provided above.
* **Create a TODO.md**: You must explicitly create a `TODO.md` file to track progress. Keep it updated as you move through the phases. Only proceed to coding once you are absolutely certain the theory holds up and fits together smoothly. Submit your plan to me (your Checker) before proceeding.

### The TODO.md Format Requirement
Here is the exact format you must use for your `TODO.md` file. You will update the checkboxes as you complete each task.

```markdown
# TODO: Primal Framework Execution Tracker

## Phase 1: Planning
- [x] Brainstorm architecture and module connections.
- [x] Define data structures and mentally verify math.
- [x] Create this TODO.md file.

## Phase 2: MVP & Critical Thinking
- [ ] Implement Active Inference Free Energy calculator.
- [ ] Implement Log Space GMM with NaN protection and n_obs tracking.
- [ ] Implement basic visual routing and Hierarchical Predictive Coding.
- [ ] Implement Hippocampal Episodic Buffer for one-shot anchoring.
- [ ] Verify action signal > 5.0 (slot-position forward model, not feature-space FE).
- [ ] Run MVP Benchmark (Breakout >5 score, MNIST >70%).
- [ ] Output terminal logs to the conversation for debugging.

## Phase 3: Full Implementation
- [ ] Create `brain/` directory structure.
- [ ] Write `agent.py` API wrapper.
- [ ] Write Primeval `LICENSE`.
- [ ] Write `pyproject.toml`.
- [ ] Implement all remaining cortex and physics mechanisms.

## Phase 4: Full Verification
- [ ] grep -r "getRAM\|ram\[" primal/ returns empty.
- [ ] Run 1-shot MNIST test (10k evaluation) targeting >90% using occipital pipeline features (no handcrafted descriptors).
- [ ] Confirm digit 7 and digit 9 are not confused by checking per-class accuracy and tuning V4 if needed.
- [ ] Run Breakout/Pong physics test with no fixed seed, targeting mastery in <2 episodes.
- [ ] Capture all terminal output, analyze, and debug failures.
- [ ] Profile iterations per second (target >10) and RAM usage (target <2GB).

## Phase 5: Finalization
- [ ] Write detailed `README.md`.

## Stuck
<!-- Only fill this in if you have made 3 genuine attempts at a specific problem and are still failing. -->
<!-- Describe what is failing, what you have already tried, and paste the exact terminal output. -->
<!-- The Checker will read this and send a targeted fix. Do not keep guessing past 3 attempts. -->
```

### Phase 2: MVP & Critical Thinking (Build Small, Verify Early)
Before building the entire 20-component monolith, you must exercise critical thinking by building a Minimum Viable Product.
* Identify and build the smallest functional core. This means implementing only the Active Inference engine, the Log Space GMM, basic visual routing, and a **Hippocampal Episodic Buffer**.
* The Hippocampal Episodic Buffer is the absolute smallest brain component required for one-shot learning. It acts as an instant associative memory cache. When a novel observation triggers a massive prediction error, the Hippocampus instantly anchors that topological snapshot into a new GMM slot with maximum precision. This avoids the slow gradient-based learning problem and allows immediate recall for the next frame. The Hippocampus also relies on Theory Theory: when a high-FE event is stored, it is tagged with the currently active hypothesis from the Theory Theory module. During replay, the agent does not just punish the bad action. It also updates the hypothesis weights, demoting the hypothesis that predicted the missed outcome and promoting any hypothesis that would have predicted it correctly. This means episodic memory and belief revision happen together, which is how biological recall actually works.
* Verify this small core thoroughly before adding the rest of the cortex and physics mechanics.
* **MVP Benchmark**: This simplified version must prove the foundational math actually works. It absolutely must achieve a score of >5 on Breakout and a >70% accuracy on the one-shot MNIST task relying purely on the Hippocampal buffer.
* **Terminal Debugging Loop**: You must run the code, capture the exact terminal output, and print it into our conversation so I can check it. If it fails the MVP benchmark, you will read your own terminal traces, critically analyze the math, debug the core logic, and refine it until it passes.

### Phase 3: Full Implementation (The Complete Code)
Once the MVP proves the foundation is rock solid, build out the rest of the architecture.
1. Create a `brain/` directory to house all modular concepts cleanly.
2. Create an `agent.py` file in the root directory that imports from `brain/` and exposes a single, clean API.
3. Write a custom license file `LICENSE` for our company, Primeval.
4. Write a fully configured `pyproject.toml` to manage dependencies.
5. Create an `__init__.py` setup to make it a proper package.
6. Update the `TODO.md` file.

### Phase 4: Full Verification (The Verifial Loop)
This is the most important step. You must verify that your fully assembled logic and code work flawlessly.

* **Image Test (MNIST)**: Test the framework on MNIST. It must learn using exactly 3 samples per class (30 training samples total, first/middle/last occurrence of each digit). Features must come from the occipital pipeline without HOG or any handcrafted descriptors. After this rapid few-shot learning phase, it absolutely must run inference on all 10,000 test samples and achieve **>90% accuracy**. Track per-digit accuracy, noting that digit 7 and digit 9 are the hardest pairs (confused by pure edge features). If overall accuracy plateaus below 90% and 7/9 confusion is high, the occipital V4 closed-loop response is not strong enough. Tune V4 sensitivity before adding any handcrafted features.
* **Why this is possible**: Standard neural networks fail at one-shot learning because they blindly map flat pixel arrays to arbitrary weight vectors, requiring thousands of examples to learn translation invariance. Primal behaves like a biological visual system. By using foveation and saccadic movements via the Superior Colliculus, it traces the actual edges of a digit (Occipital V1 processing). It builds a relational, topological graph of strokes, loops, and intersections using Spelke's geometry priors. Because a digit "8" is fundamentally just two stacked topological loops, matching this geometric graph requires only one visual example anchored by the Hippocampus. The Log Space Fusion and GMM allow the model to dynamically stretch and fit this topology over the test set without forgetting the core structural rules.
* **Data Fallback Mechanism**: Network requests fail. If fetching MNIST from OpenML throws an error, you must include a fallback function. This fallback should use standard `urllib` to directly download the raw gzip files from a stable mirror. You must then decompress them and parse the byte structures manually into numpy arrays. Do not let a simple network error stop the verification loop.

```python
import urllib.request
import gzip
import os
import numpy as np
from typing import Tuple

def load_mnist_fallback(path: str = "/tmp/mnist") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    os.makedirs(path, exist_ok=True)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    
    def download_file(name: str) -> str:
        filepath = os.path.join(path, name)
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(base_url + name, filepath)
        return filepath
        
    def read_images(filepath: str) -> np.ndarray:
        with gzip.open(filepath) as file_buffer:
            file_buffer.read(16)
            buffer = file_buffer.read()
            return np.frombuffer(buffer, np.uint8).reshape(-1, 784) / 255.0
            
    def read_labels(filepath: str) -> np.ndarray:
        with gzip.open(filepath) as file_buffer:
            file_buffer.read(8)
            buffer = file_buffer.read()
            return np.frombuffer(buffer, np.uint8)
            
    return (
        read_images(download_file("train-images-idx3-ubyte.gz")),
        read_labels(download_file("train-labels-idx1-ubyte.gz")),
        read_images(download_file("t10k-images-idx3-ubyte.gz")),
        read_labels(download_file("t10k-labels-idx1-ubyte.gz"))
    )
```

* **Physics Test**: Test it on a heavy physics Gymnasium ALE environment like Breakout or Pong. Note that you must install dependencies strictly using `pip install gymnasium[atari] ale-py`. Gymnasium versions >= 1.0 and ale-py >= 0.9 come with ROMs natively bundled. You do not need to install AutoROM and you do not need to accept any ROM licenses. It works completely out of the box.

```python
import gymnasium as gym
import ale_py

# Required for Gymnasium >= 1.0; without this line, Atari envs do not exist
gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
obs, _ = env.reset()   # no seed
# Run loop and benchmark
```

* **The Benchmark**: With a correct full implementation, it should master the physics game under 2 episodes or with 3 lives left. Track the outputs closely. 
* **Terminal Debugging Loop**: Just like in Phase 2, you must capture the standard output, stack traces, and performance metrics from the terminal. Output them directly into the conversation so I can check your work. If the benchmarks are not met, read the terminal output, deep think about what went wrong, and fix the code. Do not just output failing code. Iterate until it works. If after 3 genuine attempts at a specific failure you are still stuck, do not keep guessing. Write a `## Stuck` section at the bottom of `TODO.md` describing exactly what is failing, what you have already tried, and what the terminal output says. I will read it and send you a targeted fix. Guessing blindly past 3 attempts wastes time and risks breaking things that already work.
* **Performance Check**: Track the iterations per second and RAM usage in the terminal logs. If the CPU is suffering or RAM usage is huge, you must read the logs, debug, and optimize the numpy/scipy operations. LBM and Gabor filtering are the two operations most likely to be slow. Both must use vectorized numpy without any Python loops over grid cells or pixels. Note that the LBM outer loop over 9 lattice directions is fine: that is O(9) and constant. The forbidden pattern is an inner loop over grid cells: `for x in range(W): for y in range(H)`. That is O(W×H) Python iterations and will destroy performance.

### Phase 5: Final Output
Once the plan is executed, the code is written, and the verification loop is successful, compile all the final files. Document everything clearly in a `README.md` explaining how to run the agent, how the brain modules work together, and how to replicate the massive MNIST and Gymnasium tests. Ensure the `TODO.md` shows all tasks completed. Show me the final package.
