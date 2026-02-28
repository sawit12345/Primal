# AGENTS.md - Primal Framework

Come back here whenever you lose track. This is the source of truth. Read it fully before touching code.

---

## What you're building

A framework called **Primal**, built by **Primeval Company**. It unifies neuroscience and Bayesian ideas into a single lightweight, fully sub-symbolic agent. No hardcoded domain knowledge. No hardcoded policy. No hardcoded predictions. The agent starts completely blind.

**It must be universal.** Breakout, Pong, MNIST, robotic control, text tokens, audio spectrograms, financial time series: the agent handles any observation space and any action space without modifying the architecture. The only thing that changes between tasks is the observation shape and action count passed to `__init__`. If you find yourself writing task-specific logic anywhere except the test scripts, stop and rethink.

The output must:
- Run on Python with numpy, scipy, and small libs only (no PyTorch, no TensorFlow, no JAX)
- Learn Breakout or Pong in under 2 episodes (3 lives)
- Classify the full MNIST test set (10,000 samples) above 90% accuracy, given 1 sample per class, with graceful one-shot handling for any class the agent has not seen at test time (open a new slot on the spot, classify by nearest slot)
- Stay fast (track it/s, target above 10) and stay lean (track RAM, target below 2GB)
- Pass logic and math verification, not just execution

---

## WARNING: READ THIS BEFORE WRITING A SINGLE LINE OF CODE

The previous attempt at this task failed because the agent:
- Wrote thin stub modules (total codebase under 2000 lines for 20 modules)
- Accepted score=0.0 in Breakout as a pass because FE was decreasing
- Accepted entropy=0.000 as "the agent became decisive" instead of recognizing it as action collapse
- Moved to the next phase without meeting the current phase's hard criteria
- Got blocked on a pandas dependency for MNIST and did not use the fallback

Do not repeat these mistakes.

**Score=0.0 in Breakout means the paddle never touched the ball. That is not learning. The target is >= 30, not > 0. Fix it before moving on.**

**Entropy=0.000 means the agent picks the same action every single step. That is a bug, not a sign of learning. Entropy collapse means action generation is broken: either the temperature is near zero, the GMM is producing identical predictions for all actions, or the cerebellar smoothing has locked onto one action. Fix it before moving on.**

**If your total brain/ folder is under 1500 lines, you have not implemented all 19 modules. Count the lines. Audit each file.**

**If MNIST fails due to pandas, use the urllib fallback in the Definition of Done section. Do not skip MNIST.**

The Definition of Done section at the bottom of this file has a checklist. Run through it before claiming the task is complete.

**Known failure modes from previous attempt (do not repeat these):**

1. Smoke run of 150 steps passed off as a Breakout episode. 150 steps is not an episode. Run full episodes until the game ends (lives exhausted). No max_steps below 10,000.

2. it/s = 2.20 (5x below requirement). LBM had Python loops over grid cells. Gabor was applied pixel by pixel. Both must be fully vectorized numpy operations. Run the speed gate before Phase 6 and fix it first.

3. MNIST: 2 slots opened instead of 10. BMR was running during the 10-sample learning phase and merged digit prototypes. Set `agent.bmr.enabled = False` before the learning phase. Set `agent.gmm.novelty_threshold = 0.0` to force new slot creation for each sample. Verify slot count == 10 before running classification.

4. Accuracy = 0.1009 on MNIST. This is random guessing. It is a direct consequence of having only 2 slots. Fix the slot count first. If slot count is 10 and accuracy is still below 0.5, the visual features are collapsing (DoG returning zeros, PCA too low-dimensional, or Gabor applied to wrong input).

---

## ANTI-LAZINESS RULES

These rules exist because the previous attempt produced stubs, skipped tests, and called score=0 a success. Read them now.

### Rule 1: No stub functions

Every function must do the actual work. No exceptions.

```python
# FORBIDDEN - this is a stub:
def update_gmm(self, obs):
    pass

# FORBIDDEN - this is a stub with a return:
def compute_fe(self, obs):
    return 0.0

# FORBIDDEN - this is a TODO:
def e_step(self, obs):
    # TODO: implement log-space E-step
    return np.ones(self.n_components) / self.n_components

# REQUIRED - this is an implementation:
def e_step(self, obs):
    log_resp = np.array([
        self.log_pi[k] + self._log_gaussian(obs, self.mu[k], self.Sigma_inv[k], self.log_det_Sigma[k])
        for k in range(self.n_components)
    ])
    log_resp -= scipy.special.logsumexp(log_resp)
    return np.exp(log_resp)
```

If you write a stub and move on, you are not implementing the framework. You are writing a skeleton that will fail every test and produce score=0 in Breakout.

### Rule 2: Count your lines before moving to the next phase

After implementing each module, run `wc -l primal/brain/<module>.py`. If it is below the minimum for that module, you are not done with it. Go back.

Do not batch-verify at the end. Check each module as you finish it.

### Rule 3: Every brain module must be independently testable

After writing each file, write a quick `if __name__ == "__main__"` block at the bottom that runs a sanity check on that module with synthetic data and prints a PASS or FAIL. This is not the Phase 5 test suite, it is a fast individual module check.

Example for `log_space_gmm.py`:
```python
if __name__ == "__main__":
    import numpy as np
    gmm = LogSpaceGMM(feature_dim=4, max_components=10)
    obs = np.random.randn(4)
    fe_before = gmm.compute_fe(obs)
    gmm.update(obs)
    fe_after = gmm.compute_fe(obs)
    assert fe_after < fe_before, f"FE did not decrease: {fe_before:.4f} -> {fe_after:.4f}"
    print("PASS: GMM update decreases FE on seen observation")
```

If this check fails after you write the module, the module is broken. Fix it before writing the next one.

### Rule 4: Do not move phases when a phase has unresolved failures

The phase gate is a hard wall, not a suggestion. If Phase 5 logic tests have 3 failures, you do not run Breakout. You fix the 3 failures first. If you start running Breakout before all 15 tests pass, you are wasting time: the broken module will cause random behavior in the game and you will debug the wrong thing.

### Rule 5: Breakout score < 30 is not a partial success

Score of 5, 10, or 15 with no upward trend from episode 1 is not "getting closer." It is a broken action pipeline with occasional noise. Do not accept it. Do not write README. Fix the action pipeline.

The specific failure mode to check: if score is in the range 1-15 but not improving, print paddle_alignment. If alignment is below 0.4, the agent is not tracking the ball. The visual streams M-path is not feeding ball position into action selection correctly.

### Rule 6: All printed results must come from actual execution

Do not write print statements that output hardcoded expected values. Every number printed must come from actually running the code.

```python
# FORBIDDEN:
print("PASS: ep2 score >= 30")  # hardcoded, not from actual game run

# REQUIRED:
ep2_score = run_episode(agent, env)
status = "PASS" if ep2_score >= 30 else "FAIL"
print(f"{status}: ep2 score = {ep2_score:.1f} (need >= 30)")
```

### Rule 7: When stuck, diagnose before changing random things

If Breakout score is not improving, do not start randomly tweaking hyperparameters. Diagnose in this order:
1. Print slot count per frame. Is the ball being tracked as a separate slot from the paddle?
2. Print action_values for all 4 actions. Are they meaningfully different?
3. Print paddle_alignment. Is the paddle moving toward the ball at all?
4. Print temperature. Is it sane (between 0.1 and 10.0)?

Each of these narrows down which module is broken. Only change hyperparameters after you know which module is producing wrong outputs.

---

## Installation and environment setup

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "primal"
version = "0.1.0"
description = "A lightweight sub-symbolic cognitive agent framework by Primeval Company"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "ale-py>=0.9",                   # ROMs are bundled inside the package as of 0.9. No AutoROM needed.
    "gymnasium[atari]>=1.0",         # The [atari] extra pulls in ale-py wrappers. ROMs come with ale-py itself.
    "scikit-learn>=1.4",             # Only used for MNIST dataset loading (sklearn.datasets.fetch_openml)
    "psutil>=5.9",                   # RAM and CPU usage tracking
    "opencv-python-headless>=4.9",   # Fast image ops (resize, Gabor). Headless avoids display dependencies.
    "tqdm>=4.66",                    # Progress bars for test runs
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-benchmark",
]

[tool.hatch.build.targets.wheel]
packages = ["primal"]
```

### Why no AutoROM and no accept-rom-license

As of `ale-py >= 0.9`, all Atari ROMs ship inside the pip wheel itself. You no longer need the `AutoROM` package, the `AutoROM.accept-rom-license` package, or the `gymnasium[accept-rom-license]` extra. That extra was removed from Gymnasium entirely when ale-py bundled the ROMs. Installing `ale-py>=0.9` is all that is needed.

The one thing that changed in Gymnasium 1.0: environments are no longer registered automatically behind the scenes. You must explicitly register them at the top of any script that uses Atari:

```python
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)  # registers ALE/Breakout-v5, ALE/Pong-v5, and all others
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
```

If you see `gymnasium.error.NameNotFound: Environment ALE/Breakout-v5 not found`, you forgot `gym.register_envs(ale_py)`.

**References to search for more detail:**
- ALE ROM bundling and registration changes: https://ale.farama.org/release_notes/index.html (search "ROMs are packaged within the PyPI installation")
- Gymnasium 1.0 release notes: https://gymnasium.farama.org/gymnasium_release_notes/index.html
- ale-py PyPI page: https://pypi.org/project/ale-py/
- AutoROM (legacy, for historical reference only): https://github.com/Farama-Foundation/AutoROM
- ALE Gymnasium environment docs: https://gymnasium.farama.org/environments/atari/

---

## Core ideas (read this before writing any code)

These are engineering-useful abstractions, not full neuroscience replications. Take the output and the function, not the full biological computation. The test for each one: does it produce the engineering benefit cheaply?

---

### 1. Active Inference (the only two moves)

The whole framework is built on two operations:

- "Change the world to match your prediction" = **action** (active inference)
- "Change your prediction to match the world" = **perception** (passive inference)

Free Energy (FE) = surprise = prediction error. The agent always minimizes FE.

**Perception update:** Given observation `o` and current belief `q(s)`, update `q(s)` to reduce `FE = E_q[log q(s) - log p(s,o)]`. In practice this is a precision-weighted prediction error signal added to the current mean.

**Action generation:** Select action `a` that minimizes expected FE under the current generative model. Because we skip full VI planning (too slow), action selection uses a softmax over predicted FE reduction for each available action, computed from the current GMM state.

**Predictive coding:** The generative model produces a top-down prediction. The residual between prediction and observation is the prediction error. Positive prediction errors (observation exceeded prediction) and negative ones carry opposite update signs. Precision weights the error before it updates the belief.

**What to skip:** Full Variational Inference planning (POMDP rollouts, belief propagation over future states). This is what Friston's later work expanded into and it is expensive. The core idea is the two moves above. Take that and nothing more from AIF.

**FE formula in log space:** `FE = -log p(o | mu) - log p(mu) + log q(mu)`. For a Gaussian generative model this simplifies to `0.5 * precision * (o - mu)^2 + log_Z`. Compute this per GMM component, then mix by responsibility.

**Search for more:** "Karl Friston free energy principle predictive coding tutorial" and "active inference tutorial Parr Friston 2022"

---

### 2. Log-space GMM with growing components (slot-centric)

All belief updates happen in log space. This is not an optimization choice, it is a correctness choice: probabilities near zero underflow in float32 if you work in linear space.

**E-step (responsibility):**
```python
log_r_k = log_pi_k + log_N(x; mu_k, Sigma_k)
log_r_k -= logsumexp(log_r_k)   # normalize
r_k = exp(log_r_k)
```

**M-step (update):**
```python
N_k = sum(r_k)
mu_k = sum(r_k * x) / N_k
Sigma_k = sum(r_k * outer(x - mu_k, x - mu_k)) / N_k + eps * I
log_pi_k = log(N_k) - log(sum(N_k))
```

Run E-step every timestep. Run M-step every 10 steps or when FE spikes above 2x running mean.

**Growing components:** If `max_k(r_k)` for the current observation is below `novelty_threshold` (start with 0.1), open a new slot initialized at the current observation with high variance (`Sigma = 10 * I`). This is Bayesian Model Expansion. Component count is not fixed.

**Slot-centric:** Each component is a slot that tracks one entity (one object, one digit pattern). Slots compete for each observation via responsibilities. Slots that consistently win on similar observations specialize.

**Never forget:** Components accumulate observations via the running M-step. They are only removed by BMR (see section 5), never by age or timeout.

**Conjugate prior:** Use Normal-Wishart prior for each component so the M-step is a closed-form posterior update. This keeps the fusion exact rather than approximate.

**Search for more:** "Normal-Wishart conjugate prior Bayesian GMM" and "online EM algorithm Gaussian mixture model"

---

### 3. Core Knowledge (Spelke's 5 systems as inductive biases)

Spelke's 5 core knowledge systems are not hardcoded rules. They are inductive biases baked into how the GMM initializes, what precision it assigns to different prediction errors, and how slots grow and compete. None of them name anything. They do not say "this is a ball" or "this is a digit." They shape the statistical geometry of belief space so the agent does not start from a completely flat prior.

The engineering principle for each system: **what output does this system produce in infants, and what is the cheapest function that replicates that output in the GMM?**

---

#### System 1: Objects

**What infants know:** Objects are cohesive (they move as a whole, not as disconnected patches), continuous (they trace a path, they do not teleport), and contact-constrained (they only affect each other by touching). Infants as young as 3 months look surprised when an object appears on the wrong side of an occluder without passing through it.

**Engineering output:** When a new observation arrives, check whether any existing slot can plausibly claim it by continuity. "Plausibly" means: does the observation fall within `k` standard deviations of the slot's predicted next position, where `predicted = mu_prev + velocity_prev`? If yes, assign to that slot. If no slot can claim it, open a new one.

**How it works in code:**

```python
predicted_means = [mu_k + velocity_k for each slot k]
distances = [mahalanobis(x, pred_mu_k, Sigma_k) for each k]
best_k = argmin(distances)
if distances[best_k] < continuity_threshold:   # e.g., 3.0 sigma
    assign x to best_k, run normal E-step
else:
    open new slot at x with Sigma = 10*I
```

This is object permanence without naming objects. The agent naturally treats things that move continuously as the same thing over time.

**Cohesion prior:** Group observations within `cohesion_radius` pixels before slot assignment. Observations close together are treated as one entity. A tall paddle is tracked as one slot, not 80 independent pixels.

**Contact constraint:** If two slot means converge within `contact_threshold` of each other and both have nonzero velocity, apply an elastic collision update to their velocities. This is coordinated with the LBM module (section 10), but the contact trigger originates here in core_knowledge.

---

#### System 2: Agents (goal-directed, self-propelled entities)

**What infants know:** Some things move on their own, toward goals, and by efficient paths. Agents are distinguished from objects by self-propulsion: their velocity cannot be explained by external physics. Infants at 12 months attribute goals to moving agents and expect them to take the shortest available path.

**Engineering output:** A slot is flagged as an agent slot if its velocity over the last 10 steps consistently exceeds what LBM physics would predict. Residual velocity (observed minus LBM-predicted) above `agent_threshold` means self-propulsion.

**How it works in code:**

```python
residual_k = velocity_k_observed - velocity_k_lbm_predicted
is_agent_k = rolling_mean(np.abs(residual_k), window=10) > agent_threshold

# Agent slots get higher weight in action-prediction:
action_relevance_k = base_weight + agent_bonus if is_agent_k else base_weight
```

**Goal inference:** Agent slots maintain a goal estimate: the location the slot appears to be moving toward, inferred by extrapolating its velocity. Updated each step. When the Primal agent selects an action, it estimates the expected change in distance between its own proprioceptive state and each agent slot's goal. This drives purposive action without naming what the goal is.

**Why this matters universally:** In Breakout, the ball becomes an agent slot after paddle contact (its motion is no longer explained by simple physics). In MNIST, nothing moves, so no agent slots open, which is correct. In robotic control, the end-effector is an agent slot. The flagging is automatic and requires no domain knowledge.

---

#### System 3: Number (Approximate Number System, ANS)

**What infants know:** Infants can distinguish "more" from "less" without counting. At 6 months they habituate to arrays of 8 dots and look longer at 16-dot arrays than 8-dot arrays in a new arrangement. The discrimination follows Weber's law: 8 vs 16 is easy (2:1 ratio), 8 vs 9 is hard (9:8 ratio). The resolution is logarithmic, not linear.

**Engineering output:** Track the log-cardinality of active slots (those with `pi_k > weight_floor`). A sudden change in log-cardinality adds a surprise bonus to FE.

**How it works in code:**

```python
active_slots = [k for k in slots if pi_k > weight_floor]
log_card = np.log1p(len(active_slots))   # log1p for numerical stability
delta_log_card = abs(log_card - log_card_prev)

if delta_log_card > cardinality_change_threshold:
    FE_total += cardinality_surprise_weight * delta_log_card
```

The FE bonus triggers homeostasis (potential model expansion) and hippocampus replay (the event is surprising, so remember it).

**Weber scaling of cardinality:** Use `weber_precision(n_active)` to weight the surprise. Going from 1 slot to 2 is very surprising. Going from 20 to 21 is not. This matches infant data where small number discrimination is sharper than large number discrimination.

**Why this matters universally:** In Breakout, bricks disappearing reduces cardinality and triggers surprise. In multi-agent environments, new agents appearing increases cardinality. In financial time series, a sudden new cluster of price behavior triggers cardinality surprise. None of this requires naming what the entities are.

---

#### System 4: Space and Geometry

**What infants know:** Space is Euclidean. Objects have locations. Infants use geometric properties (distance, left/right, in front/behind) to navigate and to find hidden objects. They know "behind the wall" is different from "in front of the wall." By 18 months, they use the shape of a room's walls to reorient themselves after being spun around.

**Engineering output:** Every slot carries (x, y) as mandatory first dimensions of its mean vector, regardless of domain. Spatial dimensions receive higher precision than other feature dimensions by default. Distance and direction between slots are first-class quantities used in slot assignment, contact detection, goal inference, and action selection.

**How it works in code:**

```python
# Slot mean: [x, y, feature_1, ..., feature_d]
precision_weights = np.ones(feature_dim)
precision_weights[0] = spatial_precision_boost   # e.g., 3.0
precision_weights[1] = spatial_precision_boost

# Weighted Mahalanobis for slot assignment:
diff = x - mu_k
fe_k = 0.5 * (diff * precision_weights) @ inv_Sigma_k @ diff
```

**Geometry prior on new slots:** Initial spatial variance for a new slot is `sigma_xy^2 = (frame_width / 10)^2`. This encodes the prior that objects are probably somewhere in the frame, but uncertain within a region roughly 1/10th of the frame wide.

**Containment:** If a slot's mean is within the bounding box of another slot (approximated as mean plus/minus 2 std), flag it as "contained." Contained slots inherit a component of the outer slot's velocity. This implements the intuition that an object inside a container moves with the container.

**Allocentric vs egocentric frames:** The GMM maintains two reference frames. Allocentric uses absolute frame coordinates. Egocentric uses coordinates relative to the agent's own proprioceptive center. Action selection uses the egocentric frame. Object tracking and memory use the allocentric frame. Transform between them using the proprioception module.

**Why this matters universally:** Every domain with spatial or positional structure benefits from treating coordinates as high-precision first-class dimensions. For MNIST, (x, y) of stroke features index into digit topology. For Breakout, (x, y) track game objects. For 1D time series, a single position dimension along the time axis serves the same role.

---

#### System 5: Social Partners (agents that respond contingently to you)

**What infants know:** Some agents are special: they attend to you, respond to your actions, and have intentions directed at you. By 9 months, infants follow gaze, engage in joint attention, and attribute communicative intent to agents that respond contingently to their behavior. The key detection cue is contingency: the other agent's behavior changes reliably in response to mine, with a short lag.

**Engineering output:** A slot is flagged as a social slot if its behavior is statistically contingent on the agent's own actions over the last 20 steps. Contingent means: the slot's velocity or position changes reliably after the agent acts, with a lag of 1-3 steps.

**How it works in code:**

```python
for slot k in agent_slots:
    action_vec = one_hot(action_history[-20:])    # shape: (20, n_actions)
    slot_vel   = slot_velocity_history_k[-20:]    # shape: (20, 2)
    for lag in [1, 2, 3]:
        corr = np.corrcoef(
            action_vec[:-lag].flatten(),
            slot_vel[lag:].flatten()
        )[0, 1]
        if abs(corr) > social_threshold:          # e.g., 0.4
            flag slot k as social
```

**Reciprocity prior:** Social slots get a separate expected-response prediction. When the agent acts, it predicts what the social slot will do next (1-3 steps ahead). Prediction error on this response updates the expectation. Over time, the agent builds a model of how to influence the social slot. This is proto-communication and proto-strategic reasoning, fully sub-symbolic.

**Why this matters universally:** In Pong, the opponent paddle is a social slot: it responds contingently to the ball's position, which is influenced by the agent's own paddle. In multi-agent RL, other agents are social slots. In a dialogue environment where text is the observation, the responding system is a social slot. In Breakout, nothing is a social slot (nothing responds to the agent with a lag). The detection is purely statistical and domain-free.

---

None of these 5 systems name anything. They carry no string labels, no hardcoded domain knowledge, no task-specific logic. They are precision biases, initialization strategies, and slot-flagging rules that emerge automatically from the statistical structure of any sequential observation stream.

**Search for more:** "Elizabeth Spelke core knowledge systems 2007 review", "core knowledge object permanence infant cohesion continuity", "approximate number system ANS Weber law infants", "goal attribution infant agents efficiency 12 months", "geometric reorientation infant spatial cognition room shape", "joint attention 9 month infants social contingency"


---

### 4. Theory Theory (cheap multi-hypothesis selection)

The agent maintains a small set H (start with H=4, max H=12) of hypotheses. Each hypothesis is a configuration of the GMM: which slots are active, what their means and variances are, what causal structure connects them.

Each timestep:
1. Each hypothesis generates a prediction.
2. Each hypothesis scores itself by `log p(o | h_i)`.
3. Hypothesis weights update: `w_i = w_i * p(o | h_i) / sum(w_j * p(o | h_j))`.
4. The MAP hypothesis (argmax w_i) drives the next prediction and action.

This is Bayesian Model Selection over a small set. It is not full VI. The cost is O(H) per step, which is cheap.

New hypotheses are generated by perturbation: take the MAP hypothesis, add small noise to means and variances, and create H-1 variants. Prune hypotheses whose weight falls below `1/H^2`. Generate new variants from the current MAP when the set shrinks below H/2.

**Search for more:** "Bayesian model selection AIC BIC model evidence" and "Alison Gopnik theory theory children as scientists"

---

### 5. Bayesian Model Reduction (BMR)

BMR runs on a schedule (every 100 steps by default). It does two things:

**Merge:** Compare every pair of components by symmetric KL divergence. If `KL(k1 || k2) + KL(k2 || k1) < merge_threshold` (e.g., 0.5), merge them. Merged component mean: `mu = (N_1 * mu_1 + N_2 * mu_2) / (N_1 + N_2)`. Merged covariance: use the parallel covariance formula (involves the outer product of the mean difference).

**Prune:** Remove any component with `pi_k < prune_threshold` (e.g., 0.01). After pruning, renormalize weights.

**KL divergence between two Gaussians (closed form):** `KL(p||q) = 0.5 * (tr(Sigma_q^-1 @ Sigma_p) + (mu_q - mu_p)^T @ Sigma_q^-1 @ (mu_q - mu_p) - d + log(det(Sigma_q) / det(Sigma_p)))`. Use log-det for numerical stability.

**Search for more:** "Karl Friston Bayesian Model Reduction 2016 paper" and "KL divergence Gaussians closed form"

---

### 6. Brain mechanisms (engineering abstractions only)

The rule for every mechanism here: implement the output function, not the biological computation. What does this mechanism produce, and what is the simplest function that produces that output?

#### PFC / vlPFC: Temperature control

PFC modulates the sharpness of distributions based on prediction error. High FE = uncertain, explore. Low FE = confident, exploit.

```python
temperature = base_temp * np.exp(alpha * FE_normalized)
action_probs = softmax(action_values / temperature)
```

`FE_normalized = FE_current / FE_running_mean`. Temperature is relative to what is "normal" for this agent. Start with `base_temp = 1.0`.

#### Retina: Foveal weighting

The fovea has roughly 10x higher cone density than the periphery. Implement as a 2D Gaussian weight map centered at the current fixation point:

```python
foveal_weight = gaussian_2d(H, W, center=fixation_xy, sigma=H/4)
weighted_input = input_frame * foveal_weight
```

Apply per fixation. With 3 fixations per frame (from saccades module), you get 3 differently weighted versions of the input.

#### Hippocampus: Episodic buffer

A circular buffer of `(observation_features, action, FE, timestamp)` tuples. Default size: 1000 entries.

Two uses:
1. **High-FE replay:** Events with FE above the 90th percentile get resampled during M-step. This biases learning toward surprising events.
2. **Common sense gap-filling:** When current FE is too high but the agent must act, retrieve the nearest buffer entry by cosine similarity on V4 features and use its action.

Write every step. Read on high-FE events and during gap-filling.

#### V1/V2/V4 visual cortex hierarchy

V1: oriented edge detection via 24 precomputed Gabor filters (8 orientations x 3 spatial frequencies). Apply via convolution on DoG output.

V2: junction detection. For each spatial location, multiply Gabor responses at orthogonal orientations. High product = junction, T-bar, cross.

V4: curvature and shape fragments. 3x3 max pooling over V2 responses, plus 3x3 average pooling over V1 magnitudes. Concatenate.

Feed V4 output into the GMM, not raw V1. V4 is much more informative for both digit identity and object tracking.

#### Anterior temporal lobe: Category abstraction

Maps V4 features to GMM component indices via E-step responsibility. The "category" is the slot index, not a name.

#### Inferior temporal cortex: Contrastive sharpening

After each E-step, move component means slightly away from each other:

```python
for k in range(n_components):
    for j in range(n_components):
        if j != k:
            diff = mu_k - mu_j
            dist = np.linalg.norm(diff)
            mu_k += contrast_rate * diff * np.exp(-dist)
```

This prevents components from collapsing toward each other. Start with `contrast_rate = 0.01`.

#### Homeostasis

Running exponential mean of FE: `FE_mean = 0.99 * FE_mean + 0.01 * FE_current`.

If `FE_current / FE_mean > 2.0` for more than 10 consecutive steps: trigger GMM expansion (open new slot).

If `FE_current / FE_mean < 0.5` for more than 50 steps: trigger BMR (model is overfit, clean up).

#### Hemisphere / bilateral hemifield

Split input frame into left and right halves. Run full visual pipeline on each half independently. Merge at action selection by concatenating both feature vectors.

Weight each hemifield by its FE: the hemifield with higher FE gets higher weight in the merged representation. Rationale: the more surprising side is more task-relevant.

---

### 7. Weber-Fechner / ANS precision scaling

The brain's precision on a quantity scales logarithmically with magnitude. A difference of 1 near zero feels huge. A difference of 1 near 100 feels tiny.

```python
def weber_precision(x, alpha=1.0):
    return alpha / (np.log1p(np.abs(x)) + 1e-8)
```

Weight prediction errors by `weber_precision(observation_value)` before updating beliefs. Apply this to raw pixel values (0-255), coordinate values, and all scalar features.

**Search for more:** "Weber-Fechner law psychophysics" and "approximate number system ANS logarithmic compression"

---

### 8. Hemifield pull imbalance

Compute a saliency-weighted centroid for each hemifield:

```python
left_pull  = weighted_centroid(saliency_map[:, :W//2])
right_pull = weighted_centroid(saliency_map[:, W//2:])
left_sal   = saliency_map[:, :W//2].sum()
right_sal  = saliency_map[:, W//2:].sum()

fixation_x = (left_sal * left_pull_x + right_sal * right_pull_x) / (left_sal + right_sal)
```

This is why humans look at the more visually interesting side first. It also prevents the agent from ignoring the side of the screen where the ball is.

---

### 9. Precision alpha for survival urgency

`alpha` is the global precision scale. High alpha = sharper, more decisive predictions. Low alpha = softer, more exploratory.

```python
urgency = np.clip(1.0 - reward_running_mean / max_possible_reward, 0.0, 1.0)
alpha   = alpha_min + (alpha_max - alpha_min) * urgency
```

When the agent is losing (lives lost, negative rewards), urgency is high, alpha is high, and the agent is more decisive. When things are going well, alpha drops and the agent explores. Start with `alpha_min=0.5`, `alpha_max=3.0`.

---

### 10. Lattice-Boltzmann fluid advection (D2Q9, 18 steps)

LBM is used as a cheap physics prior, not a learned model. It advects a "presence density field" one frame forward using fluid dynamics.

**D2Q9:** 2D grid, 9 velocity directions (center + 4 cardinal + 4 diagonal).

Velocity vectors: `{(0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1)}`
Weights: `{4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36}`

**One LBM step:**
1. Streaming: shift `f_i` (distribution for direction i) by its velocity vector using `np.roll`.
2. Collision (BGK): `f_i_new = f_i - (f_i - f_i_eq) / tau`, where `tau=0.6` for stability.

Equilibrium distribution: `f_i_eq = w_i * rho * (1 + 3*(e_i . u) + 4.5*(e_i . u)^2 - 1.5*|u|^2)`

Where `rho` is local density (sum of f_i at each cell), `u` is local velocity (momentum / density).

Run 18 steps per frame to get a stable one-frame lookahead.

**Boundary conditions:** Use bounce-back at frame edges. A distribution function heading toward a wall reflects to the opposite direction. This naturally handles ball-wall bounces in Breakout without any hardcoded wall logic.

**Mass conservation:** `sum(f over all directions and all cells)` must be constant before and after every step. Add `assert np.isclose(mass_before, mass_after, rtol=1e-5)` in dev mode.

**Vectorization requirement (non-negotiable):**

The entire LBM step must run in vectorized numpy operations. No Python loops over grid cells. The streaming step uses `np.roll`. The collision step uses array-wise arithmetic on the full `f[9, H, W]` array. A correctly vectorized 18-step LBM on a 210x160 grid runs in under 3ms on a modern CPU. If your LBM step is taking 50ms, you have Python loops inside it. Find them and remove them.

```python
# FORBIDDEN (Python loop over grid):
for x in range(W):
    for y in range(H):
        f[i, y, x] = f[i, y, x] - (f[i, y, x] - feq[i, y, x]) / tau

# REQUIRED (fully vectorized):
f -= (f - feq) / tau   # collision: entire 9xHxW array at once
for i, (dy, dx) in enumerate(velocities):
    f[i] = np.roll(np.roll(f[i], dy, axis=0), dx, axis=1)  # streaming
```

**Search for more:**
- "D2Q9 lattice Boltzmann weights velocities" at Wikipedia or http://wiki.palabos.org
- "lattice Boltzmann BGK collision operator"
- "lattice Boltzmann bounce-back boundary condition"

---

### 11. Proprioception: continuous Gaussian body state

The agent's own state (estimated position, velocity, action history) is tracked as a continuous Gaussian.

```python
# State: [x, y, vx, vy]
# Prediction step (constant velocity model):
F = [[1,0,1,0],[0,1,0,1],[0,0,0.9,0],[0,0,0,0.9]]
mu_prior = F @ mu_prev
P_prior  = F @ P_prev @ F.T + Q  # Q = process noise, start with 0.1*I

# Update step (Kalman):
K      = P_prior @ H.T @ inv(H @ P_prior @ H.T + R)
mu_post = mu_prior + K @ (obs - H @ mu_prior)
P_post  = (I - K @ H) @ P_prior
```

`R` is observation noise covariance (start with `0.1 * I`). `H` projects state to observation.

Proprioception uncertainty `trace(P_post)` feeds into survival alpha: higher uncertainty = higher urgency.

**Search for more:** "Kalman filter equations tutorial" and "extended Kalman filter robotics"

---

### 12. Markovian temporal decay

Prior at time t: `p_t = 0.7 * p_{t-1} + 0.3 * likelihood_t`

Applied per GMM component during M-step: blend new sufficient statistics with old at 0.3/0.7 before updating mu and Sigma. The ratio is a hyperparameter. For fast environments, lower the 0.7. For slow environments, raise it.

The same 0.7/0.3 split is used in cerebellar smoothing (section 16) because the brain uses similar forgetting rates at multiple levels.

---

### 13. Renormalization Group (multi-scale feature extraction)

Run the full visual pipeline at 3 scales: original, 1/2, 1/4. For a 210x160 Atari frame: 210x160, 105x80, 53x40.

At each scale, run DoG + Gabor + V2 + V4 independently.

**Merging:** Weight each scale's features by inverse FE at that scale. The scale with lower FE gets higher weight.

```python
weights = softmax([-FE_scale1, -FE_scale2, -FE_scale3])
merged  = np.concatenate([w1*feat1, w2*feat2, w3*feat3])
```

Coarse scale catches global structure. Fine scale catches local detail. The FE-weighted merge automatically adapts to which scale is more informative for the current observation.

**Search for more:** "renormalization group statistical physics coarse graining" and "multi-scale feature extraction image recognition"

---

### 14. Common Sense Reasoning (gap-filling from episodic buffer)

When `FE_current > 2.0 * FE_mean` AND the current observation is partial or ambiguous:

1. Compute V4 feature vector for current observation.
2. Find top-3 buffer entries by cosine similarity on V4 features.
3. Weighted-average their next-state predictions by similarity score.
4. Use the averaged prediction to fill the gap in the current belief.

If no buffer entry has cosine similarity above 0.5: do not gap-fill. Let the high FE propagate and trigger expansion.

This is principled Bayesian interpolation, not hallucination. The agent uses the most similar past state as evidence when current evidence is weak.

---

### 15. Slot-Centric GMM

Already covered in section 2, but explicitly: each slot has `mu_k`, `Sigma_k`, `pi_k`, `age_k`, `velocity_k`.

Slot velocity: `velocity_k = mu_k_current - mu_k_prev`. Estimated each M-step. Feeds into LBM as initial condition and into Spelke's object continuity prior.

New slot: `pi_k = 1 / (n_slots + 1)`. Existing weights renormalized. Variance: `Sigma_k = 10 * I`.

---

### 16. Cerebellar smoothing

EMA over the action probability vector to reduce jitter:

```python
smooth_probs = 0.7 * prev_smooth_probs + 0.3 * raw_action_probs
action = np.argmax(smooth_probs)  # or sample if exploring
```

Also smooth action logits before softmax. Never apply softmax to unsmoothed values.

---

### 17. Superior Colliculus: Saliency map

Two signals combined:

```python
contrast = np.abs(frame - uniform_filter(frame, size=5))
motion   = np.abs(frame - prev_frame)
saliency = 0.5 * contrast + 0.5 * motion
saliency /= (saliency.max() + 1e-8)
```

Normalize to [0,1]. Use saliency map to drive saccadic fixation selection (top-K peaks, K=2 or 3 per frame). Also feeds into hemifield pull imbalance.

---

### 18. Occipital lobe: Full visual pipeline

This is the most detailed module because it determines whether MNIST works.

**Step 1: DoG center-surround (retinal ganglion cells)**

```python
on_center  =  gaussian(frame, sigma=1.0) - gaussian(frame, sigma=3.0)
off_center = -on_center
```

Without DoG, uniform regions (blank backgrounds) produce strong spurious Gabor responses. DoG kills them. Run both on-center and off-center channels forward.

**Step 2: End-stopped cells (line endings and corners)**

```python
end_stopped = gabor(frame, theta, length=L) - 0.5 * gabor(frame, theta, length=2*L)
```

High response at stroke endings, low in the middle of strokes. Critical for MNIST digit topology (the endings of "1", "7", the loops of "6", "9").

**Step 3: V1 Gabor filters (8 orientations x 3 frequencies = 24 filters)**

Precompute all 24 kernels at init time. Never recompute. Store as a stacked array `kernels: np.ndarray` of shape `(24, kH, kW)`.

**Vectorization requirement:** Apply all 24 filters in a batch, not a loop. Use `scipy.ndimage.convolve` once per filter but stack the results. Or use `cv2.filter2D` in a list comprehension. A batch of 24 Gabor convolutions on a 210x160 image should complete in under 15ms. If it is taking 200ms, you are recomputing the kernels on each frame or applying them in a Python loop with per-pixel operations.

```python
# FORBIDDEN:
responses = []
for kernel in self.kernels:
    for y in range(H):
        for x in range(W):
            responses.append(convolve_at(frame, kernel, y, x))

# REQUIRED:
responses = np.stack([
    scipy.ndimage.convolve(frame, k, mode="reflect")
    for k in self.kernels   # 24 calls, each vectorized internally
], axis=0)   # shape: (24, H, W)
```

Orientations: 0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5 degrees.
Spatial frequencies (sigma): 4, 2, 1 pixels.

Apply to DoG output.

**Step 4: V2 junction detection**

For each spatial location, multiply Gabor responses at orthogonal orientations:

```python
V2_junction = gabor_0deg * gabor_90deg + gabor_45deg * gabor_135deg
```

High values = junctions, T-bars, corners. Useful for digit intersections.

**Step 5: V4 curvature pooling**

3x3 max pooling over V2 responses, plus 3x3 average pooling over V1 magnitudes. Concatenate:

```python
V4 = np.concatenate([max_pool(V2, 3), avg_pool(V1_magnitudes, 3)], axis=-1)
```

**Step 6: Flatten and PCA reduce**

Flatten V4 output. For the first 10 observations, store raw features. After 10 observations, fit PCA to 96 components. From then on, project to 96 dims before feeding to GMM. Never refit PCA after the first fit.

---

### 19. Visual streams (Magnocellular and Parvocellular)

Two parallel pathways from the start, every frame.

**Magnocellular (M, dorsal "Where"):**
- Input: low-pass filtered frame (`gaussian(frame, sigma=2)`, then 2x downsample)
- Sensitive to motion, not fine detail
- Output: motion energy map + coarse object positions
- Used for: ball tracking in Breakout, paddle tracking in Pong, spatial reasoning
- Feeds into: dorsal stream (V5/MT), then proprioception and hemifield modules

**Parvocellular (P, ventral "What"):**
- Input: high-pass = `frame - gaussian(frame, sigma=2)` (original minus M-path input)
- Sensitive to fine detail and edge sharpness
- Output: full occipital pipeline (DoG, end-stopped, V1/V2/V4)
- Used for: MNIST digit identity, object category
- Feeds into: ventral stream, then slot-centric GMM

For Atari, M-path dominates action selection. For MNIST, P-path dominates classification. The GMM learns this implicitly: slots representing fast-moving objects align with M-path features; slots representing static shapes align with P-path features.

**Search for more:** "magnocellular parvocellular visual pathway" and "dorsal ventral stream what where"

---

### 20. Saccades (microsaccades and fixation sequence)

**Microsaccades:** Every 5 frames, add a small random offset to fixation center:

```python
offset   = np.random.normal(0, 1.5, size=2)  # 1.5 pixel std
fixation = np.clip(fixation + offset, margin, frame_size - margin)
```

Prevents adaptation to static stimuli. Critical for MNIST where the image does not move.

**Fixation sequence (3 fixations per frame):**
1. Current fixation (carried over from last frame, with microsaccade offset)
2. Peak saliency location from SC map (top salient point not within 10px of fixation 1)
3. Hemifield pull location from section 8

For each fixation: apply foveal Gaussian weighting centered there, run P-path pipeline, collect V4 features. Concatenate 3 feature vectors before feeding to GMM.

For MNIST: fixation sampling is crucial because digit images are not uniformly centered. The 3 fixations sample different stroke regions and merge the evidence, like a human reading handwriting.

---

### 21. Additional mechanisms (implement these too)

**Contrast gain control:** After each Gabor filter bank application, normalize each filter's response map by the local RMS energy (3x3 neighborhood). This prevents high-contrast regions from dominating low-contrast but informative regions. Formula: `response_normalized = response / (sqrt(mean(response^2 in 3x3)) + epsilon)`.

**Temporal contrast:** In addition to spatial contrast, compute response to frame difference (current frame minus previous frame, filtered through the M-path Gabors). This gives the agent a dedicated "what just changed" signal separate from the "what is currently here" signal.

**Surprise-gated learning:** Scale the M-step learning rate by the current normalized FE. High FE = high learning rate. Low FE = low learning rate. The agent learns faster from surprising events. Formula: `lr = base_lr * clip(FE / FE_mean, 0.1, 5.0)`.

**Lateral inhibition in GMM:** After each E-step, reduce the responsibilities of components that are similar to the winning component. This sharpens competition and prevents two slots from tracking the same object. Formula: for the winning component k_win and all other k, reduce `r_k` by `inhibition_rate * exp(-KL(k, k_win))`.

---

## File structure

```
primal/
  brain/
    active_inference.py       # FE computation, perception update, action generation
    log_space_gmm.py          # Growing slot-centric GMM, E/M steps, BME, slot velocity
    core_knowledge.py         # Spelke priors as init biases and precision weighting
    theory_theory.py          # Hypothesis set, MAP selection, perturbation, Bayesian avg
    bmr.py                    # KL-based merge, weight-based prune, on schedule
    brain_mechanisms.py       # PFC temp, retina, hippocampus, ITC contrast, homeostasis
    weber_fechner.py          # ANS precision scaling per dimension
    hemifield.py              # Bilateral split, per-hemifield pipeline, pull merge
    survival_alpha.py         # Urgency from reward, alpha scaling
    lbm_physics.py            # D2Q9, 18 steps, mass conservation, bounce-back BCs
    proprioception.py         # Kalman-like continuous Gaussian body state
    temporal_decay.py         # Markovian 0.7/0.3 decay on priors and actions
    renormalization.py        # 3-scale feature extraction, FE-weighted merge
    common_sense.py           # Gap-filling from hippocampus via cosine similarity
    cerebellar_smoothing.py   # EMA(0.7) over action probability vector
    superior_colliculus.py    # Saliency from contrast + motion, top-K fixation peaks
    occipital.py              # DoG, end-stopped, V1 Gabor, V2 junctions, V4 pooling, PCA
    visual_streams.py         # M-path (dorsal) and P-path (ventral) separation
    saccades.py               # Microsaccades, 3-fixation sequence per frame
  agent.py                    # PrimalAgent: __init__, act, update, reset
  __init__.py
tests/
  test_logic.py               # Math and logic verification, run before game tests
  test_breakout.py            # ALE Breakout with it/s and RAM tracking
  test_mnist.py               # MNIST 1-shot, full 10k test set
pyproject.toml
LICENSE
README.md
AGENTS.md
```

---

## Implementation plan (phases in order, do not skip steps)

### Phase 0: Verify math before any production code

Write a standalone `verify_math.py`. All 10 checks must pass before Phase 1.

1. **FE is lower for correct predictions.** Generate Gaussian mu=5, sigma=1. Assert `FE(obs=5.0) < FE(obs=15.0)`.

2. **Log-space E-step is numerically stable.** Initialize 3 components. Compute responsibilities for obs 100 std deviations from all means. Assert no NaN or inf.

3. **E-step responsibilities sum to 1.** For any observation, `sum(r_k over k)` must equal 1.0. Assert with `np.testing.assert_allclose`.

4. **BMR reduces component count.** Generate 100 points from 2 Gaussians with K=5 initial components. After BMR, assert `K < 5`.

5. **LBM conserves mass.** Random density field. 18 LBM steps. Assert `abs(mass_after - mass_before) < 1e-5`.

6. **Temporal decay converges.** 100 steps of 0.7/0.3 decay with fixed true value. Assert belief is within 0.1 of true value.

7. **Weber-Fechner is monotonically decreasing.** Assert `wp(1) > wp(10) > wp(100)`.

8. **Hemifield pull is asymmetric.** Left saliency 0.9, right 0.1. Assert merged fixation x is left of center.

9. **RG features differ across scales.** Cosine similarity between scale 1 and scale 3 features must be below 0.99.

10. **Proprioception uncertainty decreases.** Start `P = 10*I`. Apply Kalman update 20 times with consistent observation. Assert `trace(P_final) < trace(P_initial)`.

---

### Phase 1: Core inference stack (build in this order)

1. `temporal_decay.py` - no dependencies, pure numpy
2. `weber_fechner.py` - no dependencies, pure numpy
3. `proprioception.py` - Kalman update, no dependencies
4. `log_space_gmm.py` - growing slots, E/M in log space, slot velocity
5. `active_inference.py` - FE, perception update, action generation
6. `bmr.py` - merge and prune on schedule

After each file: run the corresponding Phase 0 check on the actual implementation.

---

### Phase 2: Brain mechanisms

7. `occipital.py` - DoG, end-stopped, V1/V2/V4, PCA
8. `visual_streams.py` - M-path and P-path separation
9. `saccades.py` - microsaccade jitter, 3-fixation sequence
10. `superior_colliculus.py` - contrast and motion saliency, top-K peaks
11. `hemifield.py` - bilateral split, per-hemifield pipeline, pull merge
12. `brain_mechanisms.py` - PFC temperature, retina, hippocampus, ITC sharpening, homeostasis
13. `survival_alpha.py` - urgency, alpha scaling
14. `renormalization.py` - 3-scale extraction, FE-weighted merge
15. `cerebellar_smoothing.py` - EMA(0.7) over action probs

---

### Phase 3: Higher cognition

16. `core_knowledge.py` - Spelke priors as GMM init and precision biases
17. `theory_theory.py` - H=4 hypothesis set, MAP selection, perturbation, weight update
18. `lbm_physics.py` - D2Q9, 18 steps, mass-conserving, bounce-back BCs
19. `common_sense.py` - gap-filling from buffer, cosine sim threshold 0.5

---

### Phase 4: Wiring

20. `agent.py` - `PrimalAgent` class:

```python
class PrimalAgent:
    def __init__(self, obs_shape: tuple, n_actions: int): ...
    def act(self, obs: np.ndarray) -> int: ...
    def update(self, obs, action, reward, next_obs, done) -> float: ...  # returns FE
    def reset(self) -> None: ...
```

`act` pipeline (in order): saccades (3 fixations from SC), foveal weighting per fixation, visual_streams (M and P), RG merge, theory_theory MAP, active_inference FE + action, cerebellar smoothing, return action.

`update` pipeline (in order): compute FE on next_obs, perception update on GMM, hippocampus write if FE high, homeostasis update, proprioception Kalman update, survival alpha update, BMR if step % 100 == 0, return FE.

---

### Phase 5: Logic verification tests

`tests/test_logic.py` must include all of these as proper `pytest` functions with assertions (not print statements):

```python
def test_fe_decreases_after_update()
def test_gmm_grows_on_novel_observation()
def test_bmr_reduces_components()
def test_lbm_mass_conservation()
def test_temporal_decay_converges()
def test_hemifield_pull_asymmetry()
def test_proprioception_uncertainty_decreases()
def test_saccades_produce_distinct_fixations()
def test_rg_features_are_scale_distinct()
def test_theory_theory_map_improves_over_time()
def test_weber_fechner_monotone()
def test_cerebellar_smoothing_reduces_variance()
def test_survival_alpha_increases_on_negative_reward()
def test_common_sense_retrieves_similar_state()
def test_lateral_inhibition_sharpens_responsibilities()
```

All must pass before Phase 6.

---

### Phase 6: Breakout/Pong test

```python
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)  # REQUIRED, without this Atari envs do not exist
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
```

**Speed gate (run this before anything else in Phase 6):**

Before running full episodes, verify the agent can actually run at speed. Run 200 steps and measure it/s:

```python
import time, psutil, os
obs, _ = env.reset()
t0 = time.perf_counter()
for _ in range(200):
    action = agent.act(obs)
    obs, reward, term, trunc, _ = env.step(action)
    if term or trunc:
        obs, _ = env.reset()
elapsed = time.perf_counter() - t0
its = 200 / elapsed
print(f"Speed check: {its:.2f} it/s")
assert its >= 10.0, f"FAIL: {its:.2f} it/s is below 10. Optimize before running full episodes."
```

If it/s is below 10, stop. Do not run full episodes at 2 it/s: it wastes time and produces garbage results because the agent cannot react fast enough. Profile with `cProfile` and fix the bottleneck first. The two known bottlenecks are:

**LBM:** Must be fully vectorized over the entire spatial grid in one numpy operation per step. No Python loops over grid cells. `np.roll` for streaming, array-wise BGK for collision. If you have any `for x in range(width): for y in range(height):` inside the LBM step, that is the bottleneck.

**Gabor convolution:** Must use `scipy.ndimage.convolve` or `cv2.filter2D` applied to the full image at once. Precompute all 24 kernels at init time. Apply them in a batch using `np.stack` and `scipy.ndimage.convolve`. No per-pixel loops.

**Full episode run (no step limits, no smoke runs):**

A smoke run of 150 steps is not a Breakout test. 150 steps is roughly 2 seconds of gameplay and covers less than one life. The agent never gets enough frames to learn anything meaningful.

Run at least 2 complete episodes where each episode ends only when the game says it is over (all lives exhausted or episode done flag). Do not impose `max_steps` unless it is above 10,000. Print results after each full episode.

```python
for ep in range(1, 3):
    obs, _ = env.reset()
    ep_score = 0
    ep_steps = 0
    aligned = 0
    done = False
    agent.reset()
    while not done:
        action = agent.act(obs)
        # compute paddle alignment before stepping
        # (requires extracting ball_x and paddle_x from slot means)
        next_obs, reward, term, trunc, info = env.step(action)
        agent.update(obs, action, reward, next_obs, term or trunc)
        ep_score += reward
        ep_steps += 1
        obs = next_obs
        done = term or trunc
    print(f"ep={ep} score={ep_score} steps={ep_steps} alignment={aligned/ep_steps:.3f}")
```

Track per step: episode, score, lives, FE value, it/s (rolling 100-step window), RAM via psutil.

Track per episode: total score, episode length, mean FE, FE trend, action distribution entropy, paddle_alignment.

Target by end of episode 2: mastery, not lucky contact. All criteria are defined in the "Breakout mastery criteria" section of the Definition of Done. They must all print as PASS.

The minimum bar is:
- ep2 score >= 30
- ep2 paddle_alignment >= 0.60 (agent actively tracks the ball)
- ep2 entropy in [0.05, 0.80]
- ep2 score strictly greater than ep1 score
- it/s >= 10 (measured during episode, not just warmup)

Score=0.0 for both episodes is a hard failure. Score of 1-5 with no improvement is a hard failure. Entropy=0.000 is a hard failure. FE decreasing while score stays at 0 means perception is learning but action is broken. Do not move on.

**Specific debug for paddle_alignment = 0.000:**

If paddle never moves toward the ball, print the following every 50 steps during the episode:

```python
ball_slot  = agent.get_slot_by_flag("agent")   # highest residual velocity slot
paddle_slot = agent.get_slot_by_flag("self")    # proprioception center
ball_x   = ball_slot.mu[0] if ball_slot else None
paddle_x = paddle_slot.mu[0] if paddle_slot else None
action_values = agent.get_action_values()       # FE reduction per action
print(f"  ball_x={ball_x:.1f} paddle_x={paddle_x:.1f} action_values={action_values}")
```

If `ball_x` is None, the ball is not being tracked as a slot. Spelke object continuity threshold is too loose, or M-path is not feeding position into slots.

If `action_values` are all equal, the generative model does not distinguish action consequences. The action head is returning a constant. Check that action is included in the state prediction and that different actions actually produce different predicted next states.

If `action_values` differ but the agent still picks the same action every time, check that temperature is not near zero. Print `agent.temperature` every step.

**Specific debug for entropy = 0.012 (near-zero but not exactly zero):**

Entropy of 0.012 means the agent is picking one action about 99% of the time with tiny probability on others. This is not "slightly exploratory," it is collapsed. The cerebellar EMA is locking onto one action. Check:

1. Is `prev_smooth_probs` being reset at episode start? If not, it carries the previous episode's collapsed distribution.
2. Is the EMA alpha too aggressive? `0.7 * stale_probs + 0.3 * new_probs` where stale_probs is already [1, 0, 0, 0] will converge back to [1, 0, 0, 0] regardless of new_probs within a few steps.
3. Print the raw (pre-smoothing) action_probs. If those are already collapsed, the problem is upstream in action generation, not in the smoother.

**Debug order if not improving:**

1. FE trajectory. If flat or rising, perception update is broken. Check E-step sign: does moving mu toward observation decrease FE? It must.

2. Action generation. If FE decreases but actions are random, check that action value estimates actually vary by action. If they are all equal, the generative model is not predicting action consequences.

3. Cerebellar smoothing. If EMA alpha is too close to 1.0, the signal washes out. Try 0.7/0.3.

4. Hippocampus replay. Are high-FE events actually being replayed? Add a print to confirm.

5. Performance. If below 10 it/s, profile with `cProfile`. LBM and Gabor convolutions are the likely bottlenecks. Both can be fully vectorized over the spatial grid.

---

### Phase 7: MNIST test

Load via `sklearn.datasets.fetch_openml("mnist_784", version=1)`.

**Critical: disable BMR during the learning phase.**

The single most common MNIST failure is that BMR merges digit prototypes together during the 10-sample learning phase. If you show the agent "0" and "6" and they have similar visual features, BMR will merge their slots before you finish the learning phase. You end up with 2 slots instead of 10, and accuracy drops to 10% (random guessing).

The fix is a one-line flag: `agent.bmr.enabled = False` before learning, `agent.bmr.enabled = True` after. The BMR module must expose this flag.

```python
# MNIST learning phase
agent.reset()
agent.bmr.enabled = False          # CRITICAL: disable BMR during learning
agent.gmm.novelty_threshold = 0.0  # CRITICAL: always open new slot for each new sample

slot_to_class = {}
for label in range(10):
    sample = train_X[train_y == label][0].reshape(28, 28)
    agent.update(obs=sample, action=0, reward=1.0, next_obs=sample, done=False)
    # find which slot just opened (highest responsibility for this sample)
    resp = agent.gmm.e_step(agent.extract_features(sample))
    winning_slot = int(np.argmax(resp))
    slot_to_class[winning_slot] = label

agent.bmr.enabled = True           # re-enable BMR for classification phase
print(f"Slots after learning: {agent.gmm.n_components} (need exactly 10)")
assert agent.gmm.n_components >= 10, "FAIL: BMR destroyed prototypes during learning"
```

If the slot count after learning is less than 10, do not proceed to classification. Debug this first.

**Learning (10 samples):** Show 1 sample per class in order 0-9. After each update, record the mapping `slot_index -> class_label` by noting which slot has highest responsibility for that sample. Store in `slot_to_class = {slot_idx: label}`.

**One-shot handling for unseen classes at test time:** If a test sample's winning slot is not in `slot_to_class` (because BMR opened a new slot during classification, or a class was never in training), open a new slot on the spot, assign provisional label `"unknown_N"`, count as incorrect, but do not crash.

For the accuracy calculation: unknown predictions count as incorrect. Track them separately so you can see how many occurred.

**Classification (10,000 samples):** For each test sample, call `agent.act(obs=image)`. Get winning slot. Look up `slot_to_class`. If missing, apply one-shot handler. Track accuracy over all 10,000 samples.

Target: above 90%.

**Debug order if below 90%:**

1. **Slot count.** If fewer than 10 slots after learning, BMR is the problem. Verify `agent.bmr.enabled = False` was set, and verify `agent.gmm.novelty_threshold = 0.0` forces new slot creation. If novelty_threshold is not 0, the GMM may assign a new sample to an existing slot instead of opening a new one.

2. **Feature distinctiveness.** After the learning phase, print the pairwise cosine similarity between all 10 slot means. If any two slots have cosine similarity above 0.98, their feature vectors are nearly identical. This means the visual pipeline is not distinguishing those two digit classes. Check V4 feature dimensionality: is PCA reducing to enough dimensions? Try increasing to 128 or 192.

3. **Saccades.** Print fixation coordinates for 5 test samples from different classes. They must differ between images. If all images get the same fixation at (14, 14), the SC saliency map is returning uniform output. Check DoG is not returning all-zeros.

4. **DoG.** On a MNIST image: a blank corner must give near-zero response. A stroke edge must give a clearly positive response. If DoG is all-zeros or all-identical, Gabor receives nothing useful and all features collapse.

5. **End-stopped cells.** On a digit "1" image: the tip of the stroke should have high end-stopped response. The middle of the stroke should have near-zero end-stopped response. If end-stopped response is uniform, the subtraction of the long-Gabor from the short-Gabor is wrong (signs may be inverted).

6. **PCA variance.** Print `pca.explained_variance_ratio_.sum()`. If below 0.90, increase n_components from 96 to 192 and refit.

---

### Phase 8: Documentation

`README.md`:
- What Primal is (2-3 sentences, plain English)
- Single install command: `pip install -e .`
- Run tests: `pytest tests/test_logic.py`, `python tests/test_breakout.py`, `python tests/test_mnist.py`
- Table: each brain module, 2-sentence description in plain English
- Expected output for each test

---

## Rules that do not bend

**Sub-symbolic only.** No string labels. No hardcoded "if ball then X". No fixed action policies. No domain-specific feature engineering. The agent discovers everything from raw observations.

**Minimum AND maximum lines.** Every module must be above its minimum line count (see Definition of Done section). No single module should exceed 200 lines. The entire `brain/` folder must be between 1500 and 3000 lines. Under 1500 means you skipped implementations. Over 3000 means you are over-engineering.

**No deep learning libraries.** No PyTorch. No TensorFlow. No JAX. No Keras. If you feel the urge, you are avoiding a math problem. Solve the math problem.

**Verify math, not just execution.** A function that runs and returns NaN is worse than one that fails loudly. Use `np.testing.assert_allclose`. Add assertions in dev mode.

**Fix problems in order.** FE math first. GMM stability second. Brain modules third. Wiring fourth. Performance last.

**No em dashes in code comments or docs.** Use commas or periods.

**Do not be arrogant.** Come back to this file when lost. It is here.

---

## Performance targets

| Test | Hard requirement | What failure means |
|---|---|---|
| Breakout ep2 score | >= 30 bricks | agent not tracking ball; action pipeline broken |
| Breakout paddle alignment | >= 0.60 (moves toward ball 60% of steps) | agent moving randomly, not reacting to ball position |
| Breakout ep2 entropy | in [0.05, 0.80] | below 0.05 = action collapse; above 0.80 = pure random walk |
| Breakout score improvement | ep2 > ep1 strictly | no learning happening between episodes |
| Breakout speed | >= 10 it/s | profile LBM and Gabor; both must be vectorized |
| RAM | <= 2GB steady state | GMM growing unbounded; BMR schedule broken |
| MNIST accuracy | >= 0.90 on all 10,000 samples | visual pipeline broken; debug occipital then PCA dims |
| MNIST learning slots | exactly 10 after learning phase | BMR too aggressive; raise merge threshold for learning |
| Codebase size | >= 3000 lines in brain/, >= 3500 total | stubs not implementations; audit every module |

---

## Dependencies

```toml
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "ale-py>=0.9",                   # ROMs bundled, no AutoROM needed
    "gymnasium[atari]>=1.0",         # Atari wrappers, requires gym.register_envs(ale_py)
    "scikit-learn>=1.4",             # MNIST loading only
    "psutil>=5.9",                   # RAM tracking
    "opencv-python-headless>=4.9",   # Fast image ops, no display
    "tqdm>=4.66",                    # Progress bars
]
```

Do not add anything else.

---

## Quick reference: web searches for implementation detail

When you need more detail, search these:

| Topic | Search query or URL |
|---|---|
| D2Q9 weights and velocities | "D2Q9 lattice Boltzmann weights velocities wiki" |
| LBM BGK collision | "lattice Boltzmann BGK collision operator tau relaxation" |
| LBM bounce-back boundaries | "lattice Boltzmann half-way bounce-back boundary" |
| LBM reference | http://wiki.palabos.org/numerics:lbm_reference |
| Normal-Wishart conjugate | "Normal-Wishart distribution conjugate prior Bayesian GMM" |
| BMR paper | "Karl Friston Bayesian Model Reduction 2016" |
| Merging Gaussian components | "merging Gaussian components mixture model parallel covariance" |
| KL divergence Gaussians | "KL divergence two Gaussians closed form" |
| AIF tutorial | "active inference tutorial Parr Friston 2022 textbook" |
| Spelke core knowledge | "Elizabeth Spelke core knowledge systems review 2007" |
| Gabor filters visual cortex | "Gabor filter V1 orientation selectivity parameters" |
| DoG retinal ganglion | "difference of Gaussians retinal ganglion center surround" |
| Magnocellular vs parvocellular | "magnocellular parvocellular pathway review" |
| Dorsal vs ventral stream | "dorsal ventral visual stream what where pathway" |
| Weber-Fechner | "Weber-Fechner law psychophysics just noticeable difference" |
| Kalman filter | "Kalman filter tutorial equations update predict" |
| ALE ROM bundling | https://ale.farama.org/release_notes/index.html |
| Gymnasium ALE registration | https://gymnasium.farama.org/gymnasium_release_notes/index.html |
| ale-py PyPI | https://pypi.org/project/ale-py/ |

---

## DEFINITION OF DONE (read this before claiming anything is finished)

This section exists because agents tend to interpret partial progress as completion. Do not mark the task as done until every single item below is checked with actual printed proof in the terminal output.

### Codebase minimums (check with `wc -l`)

```
brain/active_inference.py      >= 150 lines   (FE formula, perception update, action generation, expected FE per action)
brain/log_space_gmm.py         >= 250 lines   (E-step, M-step, log-space ops, slot tracking, velocity, Normal-Wishart prior, BME trigger)
brain/core_knowledge.py        >= 220 lines   (all 5 Spelke systems: object continuity, agent flagging, ANS cardinality, spatial geometry, social contingency)
brain/theory_theory.py         >= 130 lines   (hypothesis set, scoring, Bayesian averaging, perturbation generator, pruning)
brain/bmr.py                   >= 120 lines   (KL divergence, symmetric KL, merge logic, parallel covariance formula, prune, schedule)
brain/brain_mechanisms.py      >= 250 lines   (PFC temperature, retina Gaussian weighting, hippocampus circular buffer, ITC contrast sharpening, homeostasis, ATL)
brain/weber_fechner.py         >= 60 lines    (precision function, per-dimension weighting, application helpers)
brain/hemifield.py             >= 120 lines   (bilateral split, per-hemifield pipeline runner, pull imbalance, FE-weighted merge)
brain/survival_alpha.py        >= 80 lines    (urgency signal, reward running mean, alpha scaling, proprioception uncertainty feed-in)
brain/lbm_physics.py           >= 180 lines   (D2Q9 init, streaming step, BGK collision, equilibrium distribution, bounce-back BCs, mass conservation check, 18-step loop)
brain/proprioception.py        >= 120 lines   (state vector, prediction step, Kalman update, noise covariances, uncertainty output)
brain/temporal_decay.py        >= 60 lines    (0.7/0.3 decay, per-slot application, convergence helpers)
brain/renormalization.py       >= 120 lines   (3-scale downsampling, per-scale pipeline runner, FE-weighted merge, feature concatenation)
brain/common_sense.py          >= 120 lines   (gap detection, cosine similarity retrieval, top-3 weighted average, threshold logic)
brain/cerebellar_smoothing.py  >= 70 lines    (EMA on action probs, logit smoothing, episode reset, separate smoothing state per agent)
brain/superior_colliculus.py   >= 100 lines   (contrast computation, motion energy, saliency map, top-K peak extraction, hemifield integration)
brain/occipital.py             >= 280 lines   (DoG on/off, end-stopped cells, 24 Gabor filters precomputed, V1/V2/V4 hierarchy, PCA fit/project, feature assembly)
brain/visual_streams.py        >= 150 lines   (M-path low-pass + downsample, P-path high-pass, parallel processing, ventral/dorsal routing, output merging)
brain/saccades.py              >= 120 lines   (microsaccade jitter, SC-driven fixation selection, 3-fixation sequence, foveal patch extraction, feature merging)
agent.py                       >= 200 lines   (full PrimalAgent class, act pipeline, update pipeline, reset, episode tracking, FE history, all modules wired)
Total brain/ folder            >= 3000 lines
Total project (all .py files)  >= 3500 lines
```

If any module is below its minimum, it is a stub, not an implementation. Do not proceed. Do not try to hide the shortfall by adding blank lines or comments. The minimums count executable, meaningful lines of logic, not whitespace or docstrings.

**What a stub looks like (WRONG):**
```python
def compute_fe(obs, mu, sigma):
    # TODO: implement
    return 0.0
```

**What a real implementation looks like (RIGHT):**
```python
def compute_fe(obs, mu, Sigma_inv, log_det_Sigma, d):
    diff = obs - mu
    mahal = float(diff @ Sigma_inv @ diff)
    return 0.5 * (mahal + log_det_Sigma + d * np.log(2 * np.pi))
```

Every function must have actual math, actual numpy operations, actual logic. No `pass`. No `return None` where a value is expected. No `# TODO`. No placeholder returns.

Run this to check all at once:

```bash
for f in primal/brain/*.py primal/agent.py; do
    count=$(wc -l < "$f")
    echo "$count $f"
done
echo "Total brain/: $(cat primal/brain/*.py | wc -l) lines"
```

### Breakout mastery criteria (all must be printed as PASS)

The bar here is mastery within 2 episodes, not just "hit something by accident." The agent needs to demonstrate that it has learned to track the ball and actively direct the paddle toward it. Lucky collisions and random walks do not count.

```
PASS: ep2 score >= 30         # 30 bricks broken is not luck; it requires sustained ball tracking
PASS: ep2 paddle_alignment >= 0.60   # paddle moves toward ball >= 60% of steps (measured below)
PASS: ep2 entropy in [0.05, 0.80]    # not collapsed, not purely random
PASS: ep1_to_ep2_score_improvement > 0   # ep2 score strictly greater than ep1 score
PASS: it/s >= 10.0
PASS: ram_gb <= 2.0
```

**How to compute paddle_alignment:** On each step, compute `sign(ball_x - paddle_x)`. If the action taken matches that sign (agent moves toward ball) or the paddle is already within 4px of the ball (no movement needed), count it as aligned. `paddle_alignment = aligned_steps / total_steps`.

A score of 1-5 with no improvement from episode 1 to 2 is not mastery. It is a lucky collision followed by a broken agent. Do not accept it.

A score of 0 in episode 2 is a hard failure. The action pipeline is broken. Debug before moving on.

Entropy=0.000 or entropy=-0.000 is a hard failure. That is action collapse (same action every step), not learning. Debug before moving on.

**What distinguishes mastery from luck:**
- Lucky agents get occasional points then die immediately after. Paddle does not track the ball between contacts.
- Mastering agents move the paddle continuously toward the ball after each launch. Paddle alignment is above 0.60 even on steps where there is no immediate reward.
- If you see a score of 3 in episode 1 and 4 in episode 2 with paddle_alignment of 0.3, that is random movement with occasional contact, not mastery. Keep debugging.

**If score >= 30 seems hard to reach in 2 episodes:** it is not. A Breakout episode with Primal's visual pipeline should be straightforward because:
1. The ball and paddle are the two highest-motion-energy slots. The M-path identifies them in episode 1.
2. The agent's action space has only 4 actions (NOOP, FIRE, LEFT, RIGHT). Reducing FE by moving the paddle under the predicted ball position is a near-trivial one-step lookahead.
3. By episode 2, the hippocampus buffer has high-FE events (ball missing paddle) replayed, which trains the model to avoid those states.

If the agent is not reaching 30 by episode 2, the visual feature extraction or action selection is broken, not the target being too high.

**Debug order for score < 30:**

1. Print ball slot position and paddle slot position every 10 steps. Are they being tracked as distinct slots? If ball and paddle are merged into one slot, slot assignment is broken (Spelke object continuity threshold too loose).

2. Print action_values for all 4 actions every 50 steps. Are they different from each other? If all equal, the generative model is not distinguishing consequences of actions. The action prediction head is broken.

3. Print paddle_alignment every episode. If below 0.4, the agent is not moving toward the ball. Check whether (ball_x - paddle_x) is even being computed in the action selection path.

4. Print whether the LBM prediction matches the ball's actual next position (error in pixels). If LBM error is above 10px consistently, the physics prior is not helping. Check D2Q9 initialization and boundary conditions.

5. Check whether survival_alpha is stuck at alpha_max. If the agent always operates at maximum precision, temperature is always near zero, which gives near-deterministic action selection on whatever the GMM happened to initialize to. Add some initial entropy by ensuring temperature starts above 0.5.

### MNIST success criteria (all must be true, printed as PASS)

```
PASS: agent.gmm.n_components >= 10 after learning phase (BMR was disabled during learning)
PASS: slot_to_class has exactly 10 entries (one per class)
PASS: pairwise cosine similarity between all 10 slot means is below 0.98 (slots are distinct)
PASS: accuracy on all 10,000 test samples >= 0.90
PASS: test ran without crashing on unknown-class inputs
```

If `n_components < 10` after learning, the only cause is BMR merging prototypes or GMM not opening new slots. Fix: ensure `agent.bmr.enabled = False` and `agent.gmm.novelty_threshold = 0.0` during the learning phase.

Accuracy = 0.10 means random guessing. This is not "a low score" or "partially working." It means the classifier has exactly 0 useful information. Debug starts at slot count, not at hyperparameter tuning.

### MNIST download fallback

`sklearn.datasets.fetch_openml` requires `pandas`. If pandas is not installed or causes issues, use this fallback instead:

```python
import urllib.request, gzip, numpy as np, os

def load_mnist_fallback(path="/tmp/mnist"):
    os.makedirs(path, exist_ok=True)
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }
    for key, fname in files.items():
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            urllib.request.urlretrieve(base + fname, fpath)

    def read_images(fname):
        with gzip.open(os.path.join(path, fname)) as f:
            f.read(16)
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784).astype(np.float32) / 255.0

    def read_labels(fname):
        with gzip.open(os.path.join(path, fname)) as f:
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)

    return (
        read_images("train-images-idx3-ubyte.gz"),
        read_labels("train-labels-idx1-ubyte.gz"),
        read_images("t10k-images-idx3-ubyte.gz"),
        read_labels("t10k-labels-idx1-ubyte.gz"),
    )
```

Always try fetch_openml first. If it raises any exception (ImportError, pandas missing, network timeout), immediately fall back to `load_mnist_fallback`. Do not let a pandas dependency block the MNIST test.

### Hard stop gates between phases

These are not suggestions. They are required checkpoints.

**Before Phase 2:** Run `python verify_math.py`. All 10 checks must print PASS. If any print FAIL, fix that module before writing any brain mechanism code.

**Before Phase 6:** Run `pytest tests/test_logic.py -v`. All 15 tests must show PASSED. Zero failures, zero errors. If any fail, fix them before running Breakout.

**Before Phase 7:** Breakout must print PASS for all 5 criteria: ep2 score >= 30, paddle_alignment >= 0.60, entropy in [0.05, 0.80], ep2 > ep1 score, it/s >= 10. The speed gate (200-step warmup timing) must have passed first. If any criteria fail, STOP. Do not run MNIST. Fix Breakout first. A score of 1-5 with lucky alignment is not a pass. A smoke run of 150 steps is not Breakout.

**Before Phase 8:** MNIST must show accuracy >= 0.90 on 10,000 samples. If below 0.90, STOP. Do not write README. Debug MNIST first.

The task is not done until Phase 8 is complete with all criteria met. Printing "task complete" or moving to documentation while any test is failing or skipped is not acceptable.

### Self-audit checklist (run this mentally before claiming done)

```
[ ] brain/ folder >= 3000 lines (run: cat primal/brain/*.py | wc -l)
[ ] every module above its individual minimum (run the wc -l loop)
[ ] no module contains "pass", "TODO", or a placeholder return
[ ] verify_math.py all 10 checks PASS
[ ] pytest tests/test_logic.py all 15 tests PASSED
[ ] Speed gate: 200-step timing gives >= 10 it/s (printed before Phase 6)
[ ] Breakout test used full episodes, no max_steps below 10,000
[ ] Breakout ep2 score >= 30 (printed in terminal)
[ ] Breakout ep2 paddle_alignment >= 0.60 (printed in terminal)
[ ] Breakout ep2 entropy in [0.05, 0.80] (printed in terminal)
[ ] Breakout ep2 score > ep1 score (printed in terminal)
[ ] Breakout it/s >= 10 during episode (printed in terminal)
[ ] agent.bmr.enabled = False was set before MNIST learning phase
[ ] agent.gmm.novelty_threshold = 0.0 was set before MNIST learning phase
[ ] MNIST slot count == 10 after learning (printed in terminal)
[ ] MNIST pairwise cosine similarity < 0.98 for all slot pairs (printed)
[ ] MNIST accuracy >= 0.90 on all 10,000 samples (printed in terminal)
[ ] MNIST fallback implemented (no hard pandas dependency)
[ ] README.md exists and documents every module
[ ] LICENSE exists with Primeval Company name
[ ] pyproject.toml is complete and installs cleanly
```

If any box is unchecked, the task is not done.

