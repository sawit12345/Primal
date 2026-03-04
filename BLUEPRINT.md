CHEAP UNIVERSAL AGI: A BIOLOGICALLY-GROUNDED ACTIVE INFERENCE BLUEPRINT
version 0.4


OVERVIEW

This document specifies an artificial general intelligence architecture modeled on the human brain. It does not use variational inference over full posterior distributions. It does not use POMDP belief state matrices. Each module maps to a specific brain region with a known computational role. The full system fits in under 60 megabytes of RAM. On a single modern CPU core, it runs one inference step in under 25 milliseconds.

The design draws on predictive coding theory (Rao and Ballard, 1999), complementary learning systems theory (McClelland, McNaughton, and O'Reilly, 1995), the free energy principle (Friston, 2010), and Spelke's core knowledge theory (Spelke and Kinzler, 2007). No component requires GPU acceleration. No component requires automatic differentiation through the full system.


CORE DESIGN PHILOSOPHY

The brain does not maintain a global probability distribution over all possible world states. It runs dozens of specialized modules in parallel. Each module has a narrow job, a local learning rule, and a low communication budget with other modules.

The system described here follows that same logic. Free energy minimization is implicit. Each module minimizes its own local objective. No single scalar is computed for the whole system. Action selection emerges from the interaction of modules, not from a centralized policy computation.

The biological grounding is not cosmetic. Each module is constrained by the input dimensionality, output dimensionality, latency, and learning rule of its biological counterpart. Where biological parameters are measured, they are used directly.


SENSORY INPUT PIPELINES

retina and early visual processing

The retina performs contrast normalization and center-surround filtering. Implement this as a difference-of-Gaussians convolution applied to each RGB channel. The narrow Gaussian uses sigma equal to 1 pixel. The wide Gaussian uses sigma equal to 3 pixels. Subtract wide from narrow. This produces ON-center and OFF-center response maps matching the two main retinal ganglion cell types. No learned parameters. The filter runs on a 128 by 128 pixel input in under 0.5 milliseconds on a modern CPU.

Color opponency applies before the difference-of-Gaussians step. Compute red-minus-green and blue-minus-yellow channels from the raw RGB input. These match the four cone-opponent channels measured in primate retina. The resulting input to V1 is four channels: luminance ON, luminance OFF, red-green, and blue-yellow.

V1: primary visual cortex

V1 extracts oriented edges and spatial frequencies. Implement this as a fixed Gabor filter bank. Use 8 orientations spaced at 22.5 degrees from 0 to 157.5 degrees. Use 3 spatial frequency scales with peak frequencies at 2, 4, and 8 cycles per degree of visual angle. Total filter count is 8 orientations times 3 scales times 2 phases (sine and cosine), giving 48 filters applied across 4 input channels. No learned parameters. This layer is fixed at initialization. Hubel and Wiesel described this tuning in 1962. Daugman's Gabor model formalized it in 1985.

V1 output is sparse by design. Roughly 5 to 10 percent of V1 units respond strongly to any natural image patch. Apply a hard threshold at 0.2 times the maximum activation per filter to enforce this sparsity. Sparse V1 output reduces downstream computation at every subsequent step.

V2 through inferotemporal cortex: renormalizing generative model visual hierarchy

From V2 upward, this architecture replaces all convolutional neural network layers with a Renormalizing Generative Model (RGM) hierarchy. The justification comes directly from Friston et al. (2024), "From pixels to planning: scale-free active inference," published in Frontiers in Network Physiology. That paper demonstrates RGMs achieving 99.8 percent accuracy on MNIST digit recognition using only 10 percent of standard training data, and applies the same architecture to movie compression, music generation, and Atari game learning. RGMs are described in the paper as discrete homologs of deep convolutional neural networks with fundamentally better sample efficiency and parameter efficiency.

The CNN approach used in version 0.3 and earlier had approximately 2.1 million learned parameters in the ventral visual stream. The RGM replacement achieves equivalent representational depth at approximately 15,000 to 20,000 learned parameters. The reduction factor is roughly 100 to 140 times.

The mathematical basis is the renormalization group from physics. Each RGM level coarse-grains the level below through a grouping and dimension reduction operation called the RG operator. The functional form of belief updating, which is free energy minimization, is identical at every level. Only the parameters change between levels. This scale-invariance is the property that allows the architecture to handle data at any spatial or temporal scale without redesign.

Each RGM level consists of three components. First, a set of discrete latent states drawn from a categorical distribution. Second, a sparse block-diagonal likelihood matrix D mapping each latent state to a group of states at the level below. Third, a transition tensor B encoding how states transition under discrete paths, which are sequences of latent state transitions. States generate paths, which generate states at the level below. This recursion is what Friston et al. call the hierarchical path structure.

The block-diagonal structure of D is the key to parameter efficiency. Each latent state at level k generates only a small local group of states at level k-1, not all states at level k-1. This mirrors the biological fact that neurons in V2 have local receptive fields, not global ones. After structure learning, typical sparsity of D exceeds 90 percent. The non-zero entries in D are the only learned parameters per level.

Structure learning adds new latent states automatically. When the current model at level k cannot explain an incoming level k-1 pattern without an increase in expected free energy, the system adds a new latent state to accommodate it. This is Bayesian model selection implemented online, one of the central contributions of the Friston et al. (2024) paper. The system grows its representational vocabulary only when the evidence demands it, not by pre-specifying layer sizes.

RGM level 1 (V2 functional equivalent): groups V1 Gabor activation patterns into edge combination and texture states. Starting latent state count: 64. D matrix: 64 by 8 non-zero entries per row, block-diagonal. B tensor: 64 by 64 by 4 paths, approximately 10 percent non-zero after structure learning. Total learned parameters: approximately 2,150. Updates via free energy minimization using variational message passing. No backpropagation.

RGM level 2 (V3 and V4 functional equivalent): groups level 1 states into mid-level shape, curvature, and color region states. Starting latent state count: 128. Similar sparsity profile. Total learned parameters: approximately 4,300.

RGM level 3 (inferotemporal cortex functional equivalent): groups level 2 states into object-level identity representations. Starting latent state count: 256. After structure learning across diverse visual experience, this level develops states corresponding to object categories, faces, and scenes. Total learned parameters: approximately 8,600.

The fusiform face area maps to a dedicated branch of RGM level 3 with 32 additional latent states initialized with higher prior probability for face-like V1 patterns. Face patterns share specific orientation and spatial frequency combinations at the Gabor level. The fusiform branch develops face-selective states faster than the general object branch due to this initialization bias. These 32 face-selective states feed the TPJ social cognition module directly, bypassing the general IT output vector. Additional learned parameters: approximately 1,500. Zero fixed parameters.

Total RGM visual hierarchy learned parameters: approximately 16,550 weights. At 4 bytes per float, that is 66 kilobytes. Compare to 8.4 megabytes for the version 0.3 CNN ventral stream.

The output of RGM level 3 is a probability distribution over 256 discrete object states, not a 512-dimensional continuous vector as in prior versions. The MAP (maximum a posteriori) state index feeds downstream as a 256-dimensional one-hot vector for compatibility with the hippocampal tuple format. The belief distribution itself (256 floats) feeds the amygdala and TPJ for uncertainty-sensitive valence and social inference.

V5 / MT: motion detection

V5 computes motion across consecutive frames. Apply a fixed spatiotemporal Gabor bank across two consecutive V1 activation maps. Alternatively, use a Lucas-Kanade optical flow estimator on consecutive frames. No learned parameters. Output is a velocity field: direction and speed at each spatial location. V5 output dimension is 16 by 16 by 2, storing horizontal and vertical velocity per spatial cell. This feeds the superior colliculus salience map as a separate channel.

dentate gyrus: pattern separation before hippocampal storage

The dentate gyrus sits between the entorhinal cortex and CA3. Its function is pattern separation: making similar incoming patterns more orthogonal before storage, which reduces interference between related memories in the hippocampal attractor network.

Implement as a fixed sparse random projection. The input is the 256-dimensional RGM level 3 belief distribution concatenated with the 128-dimensional auditory A2 vector, giving 384 total input dimensions. The projection expands this to 3,840 dimensions (10x expansion) via a fixed random matrix initialized once at startup. Apply a hard threshold keeping the top 2 percent of activations, producing a sparse 3,840-dimensional binary vector. No learned parameters. Cost: one random matrix multiply, approximately 0.15 milliseconds. The output feeds CA3 as a separated pattern.

Pattern separation reduces retrieval interference at the CA3 level. Two input patterns sharing 70 percent of their features produce DG outputs sharing only approximately 30 percent of their active bits, due to the expansion and thresholding. This is the computational role O'Reilly and McClelland described in their 1994 hippocampal model.

CA3: pattern completion attractor

CA3 is the associative memory core of the hippocampal buffer. It receives the DG-separated pattern and performs pattern completion via attractor dynamics. The dense associative memory (Ramsauer et al., 2020) with 3,840 input dimensions supports exponentially many non-interfering patterns. After DG separation, effective pattern orthogonality is high enough that millions of stored patterns can coexist without attractor interference.

CA1: mismatch detection and novelty gating

CA1 compares the CA3-retrieved pattern against the current cortical input. The mismatch between what CA3 retrieved (the expected past) and what cortical layer 5 currently outputs (the present) is the biological source of the norepinephrine novelty signal. In the architecture, this mismatch is the hippocampal write gate. High CA1 mismatch above the 0.6 threshold triggers a buffer write and a norepinephrine spike. Low mismatch means the current experience matches a stored pattern and no write occurs.

This separates pattern completion (CA3) from novelty detection (CA1) into anatomically accurate submodules. The operational cost is one vector subtraction and one norm computation per step. Under 0.05 milliseconds.

dorsal stream: parietal cortex

The dorsal stream runs V1 to V3 to V5 to parietal cortex. Parietal cortex maintains a spatial map of the environment relative to the agent body. Implement this as a 64 by 64 grid of 32-dimensional feature vectors. Updates are sparse: only cells near the currently attended location update per step. The attended location is provided by the superior colliculus. Total parietal spatial map memory: 64 times 64 times 32 times 4 bytes, approximately 524 kilobytes. Updates cost one dot product per active cell per step.

superior colliculus: attentional pointer

The superior colliculus builds a salience map by combining V1 edge density, V5 motion magnitude, and subcortical threat signals from the amygdala. Each input is normalized to 0 to 1 range and summed with learned scalar weights. The argmax of the resulting 16 by 16 salience map determines the foveal crop center for the next visual frame. This drives covert and overt attention. Cost: one weighted sum over 256 cells plus one argmax. Under 0.1 milliseconds per step.

cochlea and auditory pathway

The cochlea performs frequency decomposition. Implement as a mel-filterbank with 128 frequency bands applied to a 25-millisecond audio window with 10-millisecond hop. Output is a 128-dimensional mel-spectrum vector per audio frame. No learned parameters.

A1 and A2 also use the RGM framework rather than fixed 1D convolutional filters. RGM level 1 for audio groups mel-spectrum frequency bands into spectral feature states (formants, noise bursts, harmonic patterns). Starting latent state count: 64. RGM level 2 groups level 1 states into phoneme and sound-event states. Starting count: 128. Total learned parameters for auditory RGM: approximately 6,000 weights. This replaces the approximately 45,000-weight 1D CNN auditory pathway from version 0.3. Output is a probability distribution over 128 discrete auditory states, feeding the thalamus and the recurrent cortical layer 6.


LANGUAGE SYSTEM: RECURRENT CORTICAL LAYER 6 AND LMU ENTORHINAL INDEX

Language is a temporal sensory stream. Phonemes arrive as sequential auditory feature vectors from A2. Words arrive as sequential token embeddings. The cortical hierarchy already processes temporal sequences for vision and motor control. Language requires no separate architectural module. Adding it as a sixth cortical layer is structurally correct and biologically accurate. Broca's and Wernicke's areas are cortical regions, not separate organs.

No transformer is used in this architecture. Transformers impose a hard context window baked into matrix dimensions at design time. That ceiling is incompatible with the infinite context requirement established by the hippocampal buffer and LMU indexing system described below.

cortical layer 6: recurrent language processing

Add layer 6 above the existing five cortical layers. Layer 6 receives the A2 auditory vector (128 dimensions) as its primary bottom-up input, identical to how layer 1 receives the visual IT vector. Apply the same predictive coding update rule as all other cortical layers. Same learning rate of 0.005. Same threshold gating at 0.05. Same local error propagation. Layer 6 feeds layer 5 directly, so language and vision converge at the top of the hierarchy before reaching the hippocampus and PFC. Cross-modal binding happens automatically.

The recurrent connection within layer 6 is the within-episode sequence memory mechanism. The hidden state h at step N updates as: h(N) equals f(W times x(N) plus R times h(N-1) plus b), where x is the A2 bottom-up input, R is the recurrent weight matrix, and f is a rectified linear activation. This is an Elman network. No gates. No cell state.

Gates are absent by design. The PFC and thalamus already perform gating externally. The thalamic forget function closes gates. The thalamic input function opens them. Building duplicate gates inside the recurrent layer wastes parameters without adding capability.

Layer 6 hidden units: 256. Learned parameters: 256 squared equals 65,536 recurrent weights, plus 128 times 256 equals 32,768 input projection weights. Total: approximately 98,000 weights, or 392 kilobytes. The recurrent layer handles within-episode sequential context reliably for approximately 20 to 50 steps. Beyond that range, the hippocampal buffer takes over.

Broca's area maps to the layer 6 output feeding the motor cortex as a speech action channel. The layer 6 hidden state generates a phoneme probability distribution via a small 256 by 512 output projection. The motor cortex maps that distribution to articulatory parameters. Speech is a motor action, treated identically to any other motor output.

Language grounding occurs through co-occurrence inside the hippocampal buffer. When the agent sees a dog and hears the word "dog," both the visual IT vector and the layer 6 language state write to the same tuple. Presenting either as a partial query later retrieves the other through pattern completion. No separate alignment mechanism is needed.

The olivocochlear efference gate applies here. Reduce A2 output gain by 0.4 for 80 milliseconds after any speech motor command. This prevents layer 6 from generating spurious prediction errors in response to the agent's own voice.

LMU entorhinal index: temporally optimal hippocampal addressing

The Legendre Memory Unit (LMU), developed by Voelker, Kajic, and Eliasmith at the University of Waterloo in 2019, compresses a fixed history window into a state using Legendre polynomial basis functions. The compression is mathematically proven to be optimal for representing the recent past for subsequent reconstruction.

Map the LMU to the entorhinal cortex. The entorhinal cortex maintains a compressed index over the full hippocampal memory space using grid-code representations. The LMU polynomial basis is the temporal analog of spatial grid coding.

Implement as follows. At each step, apply the fixed LMU kernel matrix of dimension 128 by 256 to the cortical layer 5 activation history over the last 200 steps. Output is a 128-dimensional vector of Legendre polynomial coefficients encoding the temporal structure of recent experience. The LMU kernel is fixed at initialization using analytically derived Legendre coefficients. Zero learned parameters.

Concatenate this 128-dimensional temporal state with the 512-dimensional content vector (from the IT visual pathway) and the 64-dimensional context tag to form a 704-dimensional hippocampal index entry per tuple. This index entry is what HNSW searches during retrieval. The full 1,104-dimensional tuple content is stored separately and loaded only after the index identifies the top-K candidates.

The LMU index makes two previously stored tuples from different times but during similar behavioral sequences produce similar index vectors. Retrieval becomes sensitive to temporal context as well as content similarity. An agent trying to remember what happened after a specific event type retrieves temporally proximal tuples automatically.

At zero learned parameters and 128 times 256 times 4 bytes equals 131 kilobytes of fixed storage, the LMU is the most favorable cost-to-capability ratio of any component in this architecture.


CORE CORTICAL HIERARCHY

The cortex is the world-model system. It runs predictive coding across six layers. Each layer maintains a forward model that predicts the activity of the layer below. Prediction error propagates upward. Prediction signals propagate downward.

Layer 1 receives the concatenated IT visual vector (512 dimensions) plus auditory A2 vector (128 dimensions) plus thalamic routing mask (see thalamus section). Total input dimension: 640. Layer 1 has 512 hidden units. Layer 2 through 5 each have 256 hidden units. Layer 5 connects to the PFC and hippocampus. Layer 6 is the recurrent language layer described in the language system section above. It feeds layer 5.

Each layer consists of a prediction unit that generates top-down predictions and an error unit that computes the mismatch between the prediction and the actual input. Error units use rectified linear activation. The prediction update rule is a simple gradient step on mean squared error between prediction and actual input, with learning rate 0.005. No backpropagation through multiple layers during online inference. Each layer updates locally.

The threshold gate is the key cost-reduction mechanism. Each layer computes the mean absolute prediction error for the current step. If that error is below 0.05, the layer does not update and does not propagate its error signal upward. In a stable environment, this silences two to three cortical layers per step, reducing per-step computation by 40 to 60 percent. The threshold is a hyperparameter. Set it lower for more precise environments. Set it higher for noisier ones.

Total learned parameters in the cortical hierarchy: 512 squared plus 4 times 256 squared, approximately 524,000 weights. At 4 bytes per float, that is 2.1 megabytes.


THALAMUS: ROUTING AND GATING

The thalamus decides which cortical signals reach which other cortical regions. It is not a full attention mechanism. It selects one active pathway per step. Implement as a learned routing matrix of dimension 640 by 5, one row per cortical layer. Each row is a binary mask generated by a small feedforward network of 128 units. Input to the gating network is the norepinephrine scalar (see neuromodulators), the current cortical error vector, and the PFC goal vector.

High norepinephrine opens more thalamic gates. Low norepinephrine closes them. This implements the biological relationship between locus coeruleus activity and thalamic relay modulation measured by Berridge and Waterhouse (2003).

Computation cost: one 128-unit forward pass per step. Under 0.2 milliseconds.


HIPPOCAMPAL BUFFER: ONE-SHOT STORAGE, INFINITE CONTEXT, AND EPISODIC COMPRESSION

The hippocampal buffer is the fastest learning component in the system. It writes on first contact. One observation produces one stored tuple. No gradient steps required for storage.

Each stored tuple concatenates five vectors: the cortical layer 5 state vector (256 dimensions), the inferotemporal visual vector (512 dimensions), the layer 6 language state vector (256 dimensions), the PFC goal vector (64 dimensions), and a context tag (64 dimensions, widened from 16 for richer context encoding). Total tuple dimension: 1,152. The buffer holds 10,000 such tuples as the default deployment size. Memory cost at default: 10,000 times 1,152 times 4 bytes, approximately 46 megabytes.

Retrieval uses a modern dense associative memory formulation. Present a partial input vector with some dimensions zeroed out. The buffer returns the stored tuple whose non-masked dimensions best match the query. This is pattern completion. In a Kanerva-style sparse distributed memory implementation with 10,000 address cells, it supports several thousand distinct patterns.

The gap-filling function runs automatically during retrieval. Query with a partial visual vector and the buffer returns the predicted auditory, linguistic, and outcome components. Query with a goal vector and the buffer returns past action sequences that satisfied that goal. This is the mechanism of common sense reasoning. No separate module computes causal inference. The attractor dynamics do constraint satisfaction as a byproduct of retrieval.

For multi-step gap-filling, run the buffer iteratively. Feed each retrieved output back as the next partial query. Set a maximum of 7 iterations to match the biological estimate of 4 to 7 sequential retrievals per hippocampal theta cycle (4 to 8 Hz). Seven passes through a 1,152-dimensional buffer costs under 1 millisecond total.

episodic compression and schema formation

The brain does not store verbatim recordings. A full conversation consolidates into something like "had a conversation with a colleague about machine learning last Tuesday." The specific words, exact phrasing, and sequential detail compress into a gist representation during consolidation. Episode-specific residuals remain in the hippocampus. The gist migrates into cortical weights.

This architecture implements that compression explicitly during level 3 replay cycles. Before running the cortical weight update, cluster the replay batch by semantic similarity. Use k-means with k equal to 16 clusters on the IT visual and language semantic dimensions of each tuple. Each cluster centroid is the gist: the shared structure across similar episodes. A conversation about Python coding and a conversation about cooking share most of their "conversation" schema dimensions. Only the language semantic and task-specific dimensions differ.

The cortical weight update trains on the centroids, not the individual tuples. The cortex learns the gist. Unique episode details cancel across the batch gradient. Shared structure accumulates. This is schema formation in the Bartlett (1932) sense.

After consolidation, rewrite each hippocampal tuple as a delta from its cluster centroid. Compute the residual: residual equals tuple minus centroid. In natural experience data, roughly 10 to 20 percent of dimensions differ significantly from the centroid. A tuple with 1,152 dimensions where 85 percent match the centroid needs only 173 non-zero values stored, plus a cluster ID. At 4 bytes per float plus 2 bytes per dimension index, that is 173 times 6 bytes equals 1,038 bytes per compressed tuple, down from 1,152 times 4 bytes equals 4,608 bytes. Compression ratio is approximately 4.4 to 1 on typical episodic data.

Store up to 1,024 active cluster centroids at full 1,152 dimensions each. Memory cost: 1,024 times 1,152 times 4 bytes equals approximately 4.7 megabytes. This is the semantic memory store, sitting between the hippocampal episodic buffer and the cortical weights. When a centroid has been stable (mean shift below 0.01 per replay cycle) for three consecutive level 3 cycles, push it into cortical weights via one targeted replay batch of 64 tuples from that cluster. After that push, free the centroid's storage. The cortex carries that schema implicitly. Hippocampal residuals for that cluster remain as retrievable episodic specifics.

Patients with hippocampal lesions retain semantic knowledge and old gist-level memories but lose recent episodic specifics. Ribot's law (1881) captures this: older memories survive hippocampal damage better than recent ones. This architecture produces the same gradient naturally. The gist migrates into the cortex over level 3 cycles. The residuals stay in the hippocampus until pruned.

infinite context window

The context window of this architecture is unbounded. No architectural constant caps it. A transformer's context limit is a number baked into matrix dimensions at design time. This system has no equivalent number. Buffer depth is a deployment parameter set by available storage, not by model design.

Three storage systems cover three different failure modes, each of which would individually limit context depth.

The recurrent cortical layer 6 handles accurate short-range sequential context within approximately 20 to 50 steps. This covers the current sentence and the current behavioral episode.

The LMU entorhinal index provides temporally-structured retrieval cues that do not decay. The 128-dimensional Legendre coefficient vector recomputes fresh at every step from current input history. It does not store events. It stores a temporal address telling the hippocampal retrieval system when something happened relative to behavioral structure.

The hippocampal buffer guarantees that every written tuple is individually retrievable until the retention score drops it during level 3 pruning. A tuple written 10 million steps ago retrieves identically to one written 10 steps ago, given sufficient query overlap. There is no temporal decay in the storage itself.

At 10,000 stored tuples averaging 20 tokens per linguistic episode, the buffer covers the equivalent of 200,000 tokens of selective episodic context. Scale to 1 million tuples on an NVMe drive with the HNSW RAM index: effective context reaches 20 million token-equivalents, retrieved in under 5 milliseconds. No transformer architecture achieves that at any cost without attention compute scaling as context length squared.

HNSW retrieval scales logarithmically with buffer depth. At 10,000 tuples, retrieval completes in approximately 1 millisecond. At 1 million tuples, under 5 milliseconds. At 10 million tuples, under 10 milliseconds. Doubling transformer context doubles memory and quadruples attention compute. Doubling buffer depth adds storage linearly and retrieval time logarithmically.

tiered replay and consolidation schedule

Level 1 runs continuously during inter-step gaps. Any pause exceeding 50 milliseconds triggers a micro-replay batch of 16 tuples. Cost: approximately 12 milliseconds. This provides continuous trickle consolidation during normal operation.

Level 2 runs during rest periods exceeding 2 seconds. Run 512-tuple batches prioritized by novelty score recorded at write time. High-novelty tuples replay first. A 512-tuple batch takes approximately 400 milliseconds. During a 2-second rest, the system completes 4 to 5 batches.

Level 3 runs once every 10,000 inference steps, roughly every 3 hours at 70 steps per second. Replay the full buffer in random order in batches of 512. Total time: approximately 8 seconds. Suspend active inference during level 3. Raise cortical learning rate to 0.02 during this window, four times the normal online rate. Apply centroid clustering and episodic compression at this stage. After each level 3 cycle, prune the lowest-scoring 10 percent of tuples by retention score (recency weight times 0.5 plus retrieval frequency times 0.5).

Run level 2 and level 3 replay on a separate thread. The main inference loop does not block. Level 3 writes to shadow cortical weights and swaps them atomically after the batch completes.

social memory slot

Social tuples share the main buffer pool with a social flag bit. Each social tuple stores (observed agent ID, inferred goal vector, inferred belief state vector, observed action, outcome). Social tuples consume at most 20 percent of buffer capacity. Social belief inference uses hippocampal pattern completion: query the buffer with another agent's (state, action) as a partial input and read back the predicted outcome. That retrieved prediction is the inferred belief. No separate module required.


BASAL GANGLIA: ACTION SELECTION

The basal ganglia suppress all actions by default. A dopamine-coded prediction error releases suppression for the winning action. This is a winner-take-all mechanism.

Maintain a value vector V of dimension equal to the action space size. For a discrete action space of 64 actions, V is a 64-dimensional vector. Each element stores the expected reward for that action given the current cortical state. Update V using a TD-lambda rule with lambda set to 0.9 and discount factor gamma set to 0.95.

The suppression threshold is a learned scalar per action, initialized at 0.5. An action fires when V[i] minus suppression[i] is greater than all other V[j] minus suppression[j]. No softmax. No policy distribution to normalize. One argmax over 64 values.

Goal congruence modifies action values before selection. The PFC goal vector projects through a learned 64 by 64 matrix to produce a goal-congruence score per action. Add this score to V before the argmax. Switching goals swaps the PFC pattern, recomputes goal-congruence scores in one matrix multiply, and the new goal takes effect immediately. No policy recomputation required.

For continuous action spaces, replace the argmax with a small actor network of two layers at 128 units each, outputting a continuous action vector. The TD-lambda update still applies to a scalar value estimate. The actor updates via the advantage signal, which is the TD error. Total parameter count for the continuous actor: approximately 16,000 weights.


PREFRONTAL CORTEX: GOAL HOLDING

The PFC holds the current goal as an active pattern. It is 64 units. It does not recompute every step. Goals are set by the hippocampal retrieval system or by external instruction through the language pathway. The PFC pattern is static during a trial. This eliminates per-step policy inference entirely.

Working memory in the PFC holds up to three concurrent goal patterns by partitioning the 64 units into three 20-unit slots plus 4 units of slot-selector state. Switching between slots costs one vector copy. Holding a goal across time costs nothing: the pattern persists passively.

The anterior cingulate cortex monitors action selection conflict. Conflict is defined as the variance across action values after suppression. When variance drops below 0.15, two or more actions are competing equally. The ACC sends a hold signal to the PFC, delaying action selection by one step and requesting an additional cortical update cycle. This prevents premature commitment under ambiguity.


NEUROMODULATOR SYSTEM

Four scalar signals modulate the entire architecture. Each costs under 0.1 milliseconds to compute per step.

Dopamine (VTA and substantia nigra) is the reward prediction error. Compute as the TD error from the basal ganglia update: delta equals r plus gamma times V(s') minus V(s). Positive delta means better than expected. Negative delta means worse. Dopamine broadcasts to the basal ganglia value update and to the hippocampal buffer to upweight recently retrieved patterns.

Norepinephrine (locus coeruleus) tracks novelty. Compute as the mean absolute prediction error across all active cortical layers. High norepinephrine widens thalamic gates and increases the hippocampal write probability for new patterns. Low norepinephrine narrows gates and conserves computation. This implements automatic exploration without a separate epsilon parameter.

Serotonin (raphe nuclei) tracks expected time to reward. Maintain a running estimate of average steps between dopamine spikes. When the estimate is high (reward is rare), serotonin is high and gamma rises toward 1.0. When reward is frequent, serotonin is low and gamma drops toward 0.9. One scalar, one running average, one multiply applied to the basal ganglia update.

Acetylcholine (basal forebrain) tracks uncertainty about the current context. Compute as the entropy of the hippocampal retrieval confidence distribution. High acetylcholine increases cortical layer learning rates temporarily, allowing faster adaptation to a new context. Low acetylcholine suppresses learning rates to protect consolidated knowledge.


AMYGDALA: VALENCE TAGGING

The amygdala reads the RGM level 3 belief distribution (256 dimensions) and the hippocampal retrieval output. It outputs two scalars: threat level and reward anticipation. These are computed by a two-output linear layer reading the concatenated RGM belief and hippocampal vectors, dimension 256 plus 256 equals 512 input, 2 output. Learned parameters: 1,024 weights.

Threat level adds a negative offset to all approach action values in the basal ganglia. Reward anticipation adds a positive offset to all approach values. The amygdala trains using the same dopamine signal as the basal ganglia. Negative dopamine (worse than expected) increases the association between the current sensory pattern and high threat. Positive dopamine increases the association with high reward anticipation.

The amygdala also modulates thalamic gating directly. High threat forces all thalamic gates open, overriding norepinephrine. This implements the biological threat-response of full sensory intake before rapid action.


CEREBELLUM: FORWARD MODELS AND MOTOR REFINEMENT

The cerebellum predicts the sensory consequence of a motor command before the movement completes. It operates in parallel with the cortical hierarchy and does not wait for outcome feedback.

Implement as a feedforward network with two layers of 128 units. Input: the current proprioceptive state vector (64 dimensions) concatenated with the motor command vector (32 dimensions). Output: the predicted next proprioceptive state (64 dimensions). Learned parameters: approximately 24,000 weights.

The cerebellum's prediction error is the difference between its predicted next state and the actual next state. This error drives weight updates inside the cerebellum only. It does not propagate into the cortical hierarchy. Motor refinement is computationally isolated.

The motor cortex translates basal ganglia action selections into motor parameter vectors. For discrete actions, it is a 64 by 32 lookup table. For continuous actions, it is a two-layer network of 128 units outputting a 32-dimensional motor vector. The cerebellum then applies a correction to that vector based on current proprioceptive state before the command reaches the actuators.


INSULA: INTEROCEPTIVE STATE

The insula tracks internal resource state. In a biological system, this is hunger, pain, and fatigue. In a computational system, map these to measurable system metrics: memory usage, CPU load per module, cumulative error rate over the last 100 steps, and inference latency.

Implement as a 16-dimensional interoceptive state vector. Update it every 10 steps by reading system metrics and mapping them through a fixed linear transform to the 16-unit vector. Concatenate this vector with the PFC goal representation. High memory pressure adds a negative term to memory-intensive action values in the basal ganglia. This biases action selection toward simpler behaviors when resources are constrained.


ADDITIONAL BRAIN COMPONENTS

orbitofrontal cortex

The orbitofrontal cortex (OFC) sits between the amygdala and the PFC. It encodes expected outcome value at finer resolution than the amygdala's two-scalar threat and reward system. Where the amygdala outputs good or bad, the OFC distinguishes magnitude, probability, and contingency of reward.

Implement as a two-layer network of 32 units. Input: the amygdala valence pair (2 dimensions) concatenated with the RGM level 3 belief distribution (256 dimensions) and the hippocampal retrieval confidence scalar (1 dimension). Output: a 32-dimensional value representation encoding expected outcome magnitude and reliability. Learned parameters: approximately 9,000 weights. The OFC output adds a fine-grained value offset to basal ganglia action values, layered on top of the amygdala signal. The OFC also modulates extinction learning. When a previously rewarding stimulus stops producing reward, OFC prediction error drives faster reversal than the basal ganglia TD rule alone.

supplementary motor area

The supplementary motor area (SMA) plans sequences of motor actions before execution begins. Tanji and Shima (1994) documented SMA neurons firing 500 to 1,000 milliseconds before movement onset, encoding the planned sequence rather than the immediate action.

Implement as a 32-unit recurrent network reading the PFC goal vector and the current basal ganglia value vector. Output is a planned sequence of up to 5 action indices drawn from the action vocabulary. This sequence feeds the motor cortex as a look-ahead buffer. When environmental conditions match the planned sequence, motor cortex draws the next action from the buffer rather than triggering a full basal ganglia selection cycle. Learned parameters: approximately 5,000 weights. Per-step cost when buffer is active: under 0.1 milliseconds versus 0.5 milliseconds for a full basal ganglia cycle.

retrosplenial cortex

The retrosplenial cortex (RSC) translates between egocentric and allocentric spatial representations. The parietal spatial map tracks object positions relative to the agent body (egocentric). The RSC maintains a second map in world-centered coordinates (allocentric). When the agent moves, the egocentric map must update for every stored object. The allocentric map does not.

Implement as a learned 64 by 64 linear transform mapping egocentric parietal grid coordinates to allocentric world-grid coordinates. Update it from proprioceptive and vestibular signals: a 6-dimensional vector of 3 angular velocity axes and 3 linear acceleration axes from an IMU sensor. The RSC weight update rule is a simple outer product between the vestibular input and the parietal position error. Learned parameters: 4,096 weights. The allocentric map adds a second 64 by 64 by 32 runtime grid, approximately 524 kilobytes.

lateral intraparietal area

The lateral intraparietal area (LIP) accumulates sensory evidence over time until a decision threshold is reached. This is the drift-diffusion model of perceptual decision-making, documented in primate LIP recordings by Roitman and Shadlen (2002). LIP neurons ramp firing rate from baseline toward a threshold over hundreds of milliseconds before a perceptual judgment.

Implement as a 64-unit accumulator vector, one unit per discrete action. Each step, add the current cortical prediction error weighted by action relevance to the accumulator. When any unit crosses 1.0, that action is flagged as evidence-sufficient and sent to the ACC as a low-conflict signal. The accumulator resets after action execution. The drift rate is modulated by norepinephrine: high norepinephrine speeds accumulation for faster but less accurate decisions. Zero learned parameters. Cost: one vector add per step.

claustrum and global workspace

The claustrum is a thin sheet of neurons beneath the insular cortex with dense bidirectional connections to nearly all cortical areas. Crick and Koch (2005) proposed it as a key structure for binding distributed cortical representations. Computationally, it implements Baars (1988) global workspace: a broadcasting mechanism making information available across the whole system simultaneously.

Implement as a 32-unit global broadcast vector computed from the highest-active cortical layer at each step. Take the output of whichever of the six cortical layers had the largest mean absolute activation. Project it through a fixed random matrix to 32 dimensions. Broadcast this vector as an additive bias to all other cortical layers hidden states. Cost: six norm computations plus one matrix multiply plus five vector adds. Under 0.2 milliseconds. Zero learned parameters. A threat signal arriving through the amygdala and thalamus immediately biases all cortical layers simultaneously, implementing the biological global workspace effect.

default mode network

The default mode network (DMN) activates during rest, self-referential processing, and mental simulation. During any pause exceeding 500 milliseconds with no active sensory input, the cortical hierarchy runs in top-down generative mode only. The PFC goal vector drives layer 5, which drives layer 4, cascading to layer 1 without bottom-up sensory input. The layer 1 output is a predicted sensory experience: a mental simulation.

Simulated tuples write to the hippocampal buffer tagged with a simulation flag. CA1 mismatch detection treats these differently during replay: they update cortical weights at half the normal learning rate. This prevents imagination from overwriting real experience. The DMN runs on the same background thread as level 2 replay. During a 2-second rest period, the system runs 3 to 5 DMN generative passes at approximately 2 milliseconds each, costing under 10 milliseconds of thread time.


TRANSFER LEARNING THROUGH ANALOGICAL MATCHING

The brain matches new patterns to old ones automatically. When the hippocampus retrieves a pattern during an incomplete query, the retrieved pattern is always the closest stored attractor, not necessarily an exact match. Partial overlap between a new pattern and an old pattern produces a blended output. That blend is analogical transfer.

Formalize this as follows. When a new sensory tuple arrives, compute its cosine similarity to the 20 most recently active hippocampal patterns. If the maximum similarity exceeds 0.7, treat the new pattern as a variant of the matched old one. Write the new tuple to the buffer with a pointer to the matched old tuple. During replay, the cortex trains on both tuples together, learning their shared structure.

This produces automatic domain transfer. A system trained on navigating indoor environments will partially match outdoor navigation patterns via shared geometric and motion features. The hippocampal buffer does the alignment. The cortical replay extracts the shared features into cortical weights. No explicit transfer learning algorithm is required.

Similarity threshold is a hyperparameter. Setting it at 0.7 gives moderate transfer. Setting it at 0.9 limits transfer to near-identical situations. Setting it at 0.5 allows broad analogical reasoning but risks false matches.


SPELKE'S FIVE CORE KNOWLEDGE SYSTEMS AS ARCHITECTURAL PRIORS

Spelke and Kinzler (2007) identified five core knowledge systems that human infants deploy from birth or shortly after. These are not learned from scratch. They are inductive biases built into perceptual and cognitive machinery. A tabula rasa system wastes millions of training steps relearning constraints that biology solved once and inherited. This architecture treats Spelke's five systems as fixed inductive priors, not as learned components and not as simulation engines.

object permanence and cohesion

Infants expect solid objects to persist when occluded and to move as unified wholes. Implement this as a continuity constraint in the parietal spatial map. When an object representation in the map disappears from the attended location, the system does not zero out that cell. It decays the representation at a rate of 0.02 per step and maintains a visibility flag. If the object reappears at a location consistent with its last known trajectory, the system matches it to the stored representation immediately rather than treating it as a new object. This costs one trajectory prediction per tracked object per step. Track a maximum of 16 objects simultaneously. The prior has no learned parameters. It encodes three constraints: objects exist continuously, objects do not pass through each other, and objects move on smooth trajectories.

agent detection and goal attribution

Infants distinguish self-propelled agents from passive objects and attribute goals to agents. Implement this as a binary classifier on V5 motion output. Self-propelled agents produce motion vectors that do not align with external force fields (gravity, wind). Passive objects do. The classifier is a two-layer network of 64 units reading the V5 velocity field around a tracked object. Output is a scalar from 0 (object) to 1 (agent). Learned parameters: approximately 8,000 weights. Train on labeled agent-vs-object examples during initial supervised pretraining. After pretraining, hold these weights fixed. Detected agents feed a separate slot in the parietal spatial map with an additional goal-vector field of 32 dimensions. This goal-vector field initializes the basal ganglia goal-congruence term when the system plans interactions with other agents.

geometry and navigation

Infants use geometric layout of the environment to reorient after disorientation. This prior biases the parietal spatial map toward encoding metric distances, angles, and boundary structure. Implement by initializing the parietal grid cell representations using a set of 64 fixed spatial frequency basis functions analogous to grid cells in entorhinal cortex. These bases tile the environment at six spatial scales from 0.1 meters to 10 meters. They are fixed sinusoidal patterns, not learned. They provide a coordinate system before any experience. Place cell representations in the hippocampal buffer initialize as combinations of these grid cell bases. This matches the biological finding that grid cell representations precede place cell formation during development.

number and quantity

Infants discriminate quantities up to approximately 3 exactly and larger quantities approximately via ratio. Implement the approximate number system as a set of 16 logarithmically spaced magnitude units, sometimes called a number line. Each unit has a tuning curve that peaks at a specific quantity and falls off with a standard deviation proportional to the quantity (Weber fraction approximately 0.2 in humans). When counting visual objects in the scene, activate the appropriate magnitude units based on detected object count. Concatenate the 16-unit number vector with the cortical layer 3 state. This allows the system to learn number-conditioned behaviors without explicitly representing each integer. No learning required for the magnitude units themselves. The Weber fraction is the prior.

intuitive physics

Infants expect objects to obey gravity, solidity, and continuity before any formal physics instruction. Implement this not as a physics engine but as a set of prediction biases in the cerebellum and the parietal spatial map. The cerebellum's forward model initializes with weights that predict downward acceleration for unsupported objects. The initialization uses a parameterized prior: gravity constant 9.8 meters per second squared, bounce coefficient 0.6, friction coefficient 0.4. These are not constraints on the world model. They are starting weights. The cerebellum can update them through experience. In a low-gravity environment, the weights will shift. The prior just eliminates the need to learn gravity from scratch, saving an estimated 10,000 to 100,000 early training samples based on data from infant habituation studies.

The five priors combined add approximately 75,000 fixed parameters and 8,000 learned parameters to the system. The agent does not start with a blank slate. It starts with the same structural constraints that a newborn human brain carries.


FULL MEMORY SYSTEM: FOUR TIERS

Tier 1 is the cortical weights at approximately 16 megabytes, carrying fully consolidated schemas and semantic knowledge. Access cost is one forward pass, approximately 0.4 milliseconds. This is the slowest to update and the most protected against overwriting.

Tier 2 is the active centroid store at 4.7 megabytes, carrying partially consolidated schemas from recent level 3 cycles. Up to 1,024 centroids at 1,152 dimensions each. Access cost is one dot product against 1,024 centroids, approximately 0.1 milliseconds.

Tier 3 is the hippocampal buffer at variable size, carrying compressed residual episodes after consolidation and full-resolution raw tuples before consolidation. Access cost is one HNSW query, approximately 1 millisecond at 10,000 tuples.

Tier 4 is NVMe cold storage for archived low-retention episodes that passed the pruning threshold but were preserved rather than deleted. Access cost is one disk read, approximately 0.1 milliseconds on NVMe.

Working memory: PFC active pattern, 64 units, 256 bytes, refreshed every step.

Episodic memory: hippocampal buffer, 10,000 tuples at 1,152 dimensions (raw) or approximately 1,038 bytes compressed, 46 megabytes raw or roughly 10 megabytes fully compressed.

Semantic memory: centroid store at 4.7 megabytes plus consolidated cortical weights.

Procedural memory: cerebellum and motor cortex weights, 40,000 weights, 160 kilobytes.

Spatial memory: parietal grid, 64 by 64 by 32 dimensions, 524 kilobytes, sparse updates.

Linguistic memory: recurrent layer 6 weights, 98,000 parameters, 392 kilobytes.

Visual memory: ventral stream weights V2 through IT, approximately 2.1 million parameters, 8.4 megabytes.


TOTAL PARAMETER AND MEMORY BUDGET

Fixed (no learning): retinal difference-of-Gaussians filters, V1 Gabor bank, color opponency transform, cochlear mel-filterbank, LMU entorhinal kernel, grid cell spatial bases, number magnitude units, core knowledge priors for object and physics. These contribute 0 bytes of learned weight storage.

Learned weights: cortical hierarchy (layers 1 through 5) 2.1 MB, recurrent language layer 6 0.39 MB, visual ventral stream 8.4 MB, auditory pathway 0.18 MB, amygdala 0.006 MB, cerebellum and motor cortex 0.16 MB, agent classifier 0.032 MB, basal ganglia actor (continuous) 0.064 MB, thalamic gating network 0.05 MB, TPJ social cognition network 0.064 MB, IC elevation layer 0.016 MB. Total learned weights: approximately 11.5 MB.

Fixed storage: LMU kernel 0.13 MB, Gabor bank and DoG filters 0.05 MB, Jeffress ITD filters 0.01 MB. Total fixed: approximately 0.19 MB.

Runtime buffers: hippocampal buffer 46 MB raw (approximately 10 MB fully compressed after level 3), centroid store 4.7 MB, parietal spatial map 0.5 MB, working memory 0.001 MB. Total runtime buffers: approximately 51 MB raw, approximately 15 MB at full compression.

Full system memory footprint: approximately 63 MB at runtime with raw buffer, approximately 27 MB at full compression after sustained operation.


INFERENCE TIMING PER STEP (single CPU core, 3 GHz)

Retina and color opponency: 0.5 ms. V1 Gabor convolution: 1.5 ms. V2 through IT ventral stream: 6 ms. V5 motion and superior colliculus: 1 ms. Auditory mel-filterbank and A2: 1 ms. Inferior colliculus binaural localization: 0.1 ms. Recurrent layer 6 language forward pass: 0.4 ms. LMU entorhinal index update: 0.1 ms. Thalamic gating: 0.2 ms. Cortical hierarchy layers 1 through 5 with threshold gating (2 to 3 active layers): 2 ms. Hippocampal retrieval (single HNSW pass): 1 ms. Basal ganglia action selection: 0.5 ms. Neuromodulator scalar updates: 0.1 ms. Amygdala valence: 0.1 ms. TPJ social cognition (when agent detected): 0.2 ms. Cerebellum motor correction: 0.3 ms. Insula resource check (every 10 steps, amortized): 0.1 ms. Total per-step: approximately 14 to 15 ms, giving roughly 67 to 70 steps per second.

Tiered replay (runs on separate thread): level 1 micro-batch of 16 tuples during any 50 ms gap, approximately 12 ms. Level 2 batch of 512 tuples during rest periods above 2 seconds, approximately 400 ms per batch. Level 3 full buffer sweep every 10,000 steps, approximately 8 seconds including centroid clustering and episodic compression.


UPDATE SCHEDULE

Step N: read sensors, run retina through IT and auditory pipeline, update superior colliculus attention pointer, update parietal spatial map at attended location, run thalamic gating, run cortical layers above threshold, query hippocampal buffer with current cortical state, run amygdala valence, run ACC conflict check, run basal ganglia selection, execute action, read outcome.

Step N+1: compute TD error (dopamine signal), update basal ganglia values, update norepinephrine from cortical error, update serotonin from reward interval, write new tuple to hippocampal buffer if norepinephrine exceeds 0.6 (novelty threshold), update cerebellum on motor outcome error.

Every 10 steps: update insula resource vector.

During any pause exceeding 400 ms: run hippocampal replay cycle, run language replay if new linguistic data was received, update cortical weights from replay gradients.


TRAINING STRATEGY

The system does not require a large pretraining corpus to function. It starts with core knowledge priors and Gabor-fixed visual filters. From the first step, the hippocampal buffer writes every novel observation. From the first step, the basal ganglia update on every outcome.

Supervised pretraining applies only to the agent classifier (8,000 weights) and the language transformer (3.5 million weights). The agent classifier needs labeled agent-vs-object video examples. A dataset of 10,000 labeled clips is sufficient given the fixed visual features feeding it. The language transformer pretrains on a text corpus using standard masked language modeling, then fine-tunes through cross-modal replay.

All other components train online from experience. The cortical hierarchy trains via local prediction error. The basal ganglia train via TD-lambda. The amygdala trains via dopamine. The cerebellum trains via motor outcome error. No global loss function. No global optimizer. No backpropagation through the full architecture.

Catastrophic forgetting is controlled by two mechanisms. The hippocampal buffer isolates new memories from cortical weights until replay consolidates them. The threshold-gated cortical update restricts weight changes to layers where prediction error is genuinely high, leaving settled representations untouched.


COMMUNICATION BETWEEN MODULES

Each module communicates via fixed-format vectors on a per-step bus. The bus carries: cortical layer 5 state (256 dim), IT visual vector (512 dim), auditory A2 vector (128 dim), language semantic vector (256 dim), PFC goal vector (64 dim), hippocampal retrieval output (1,104 dim), dopamine scalar, norepinephrine scalar, serotonin scalar, acetylcholine scalar, amygdala threat scalar, amygdala reward scalar, superior colliculus attended location (2 dim), parietal update flag (1 bit per spatial cell, sparse). Total bus bandwidth: approximately 2,500 floats per step. At 4 bytes each and 55 steps per second, that is 550 kilobytes per second of inter-module communication. A single L2 cache handles this with no memory bandwidth bottleneck.


SCALING

The architecture scales by widening layers, not by adding layers. Doubling the cortical hidden units from 256 to 512 quadruples the cortical parameter count but does not change the number of sequential operations per step. The threshold gating mechanism scales sub-linearly with width because wider layers produce sparser error distributions.

The hippocampal buffer scales by increasing the tuple count. Moving from 10,000 to 100,000 tuples increases memory from 44 MB to 440 MB. Retrieval time scales logarithmically with buffer size when using approximate nearest-neighbor search (for example, HNSW indexing). At 100,000 tuples with HNSW, retrieval completes in under 3 milliseconds.

The language transformer scales by standard transformer scaling laws. Doubling depth and width roughly quadruples parameter count and doubles per-step inference time. For applications not requiring language, remove the transformer block entirely. The remaining architecture sits at 11 MB of learned weights and runs a full inference step in under 8 milliseconds.


COMPARISON TO STANDARD ACTIVE INFERENCE WITH POMDP AND VI

Standard variational inference over a 100-state POMDP requires inverting or approximating a 100 by 100 covariance matrix at each step. Cost scales with state dimension squared. At 1,000 states, matrix operations dominate the compute budget and take 50 to 300 ms per step on a CPU core. Belief state storage requires the full 1,000-dimensional distribution plus the generative model matrices: transition matrices at 1,000 by 1,000 per action, likelihood matrix at 1,000 by observation-dim. At 10 actions and 200 observations, that is 12 million floats or 48 MB for the generative model alone, before any history.

This architecture replaces the full posterior with a cortical point estimate plus a local uncertainty scalar per layer. Per-step cost scales linearly with the number of active cortical units, not quadratically with state dimension. One-shot learning through the hippocampal buffer replaces the slow generative model update of VI, which requires hundreds of gradient steps to consolidate a new observation into the likelihood matrix. Goal switching costs one vector swap in PFC versus a full policy recomputation in POMDP.

The cost of this simplification is accuracy in high-ambiguity environments requiring Bayesian integration over many steps. The point-estimate cortex will make more mistakes than full VI in tasks where evidence must accumulate across 50 or more ambiguous observations. The hippocampal iterative retrieval recovers some of this loss. Seven sequential retrieval passes approximate but do not match Bayesian belief propagation.

For environments with state spaces above 500 dimensions, dense reward, moderate ambiguity, and non-stationary rules, this architecture trains faster, uses less memory, and runs cheaper per step.


KNOWLEDGE REWRITING AND THE ACTIVE INFERENCE ASYMMETRY

Writing new knowledge is faster than rewriting old knowledge. This asymmetry is not a bug. It is the correct computational property for any system that should resist noise while remaining genuinely updatable given persistent evidence. Active inference provides the theoretical grounding.

Under the free energy principle, an agent faces two options when prediction error is high: change the world through action, or change its internal model through perception and learning. Karl Friston's framing of this is explicit: agents minimize surprise either by acting to make the world conform to their predictions, or by updating their predictions to conform to the world. New knowledge arrives in an empty slot. No prior prediction exists for a genuinely novel observation. The system has no model to defend. Free energy is minimized immediately through a single hippocampal write. The agent changes its mind instantly because there is no prior mind to change.

Rewriting old knowledge is different. A stored prediction already exists. The new contradicting observation generates prediction error, which is free energy. But the system has two responses available. It can update its model (change its mind) or it can dismiss the observation as noise and act to seek confirming evidence (change the world). The threshold-gated cortical update and the retention score penalty system implement exactly this balance. One contradicting observation is treated as potential noise. The cortical learning rate stays at 0.005. Repeated contradicting observations accumulate prediction error across multiple steps. Eventually the sustained error drives genuine model revision.

The hippocampal buffer tier handles new knowledge. The cortical weight tier handles old knowledge. The speed difference between them is the speed difference between a single write operation and a multi-cycle consolidation process requiring level 3 replay. Measured in steps: new knowledge consolidates in 1 step at the hippocampal level. Old knowledge revises across a minimum of one full level 3 cycle, approximately 10,000 steps, plus sustained prediction error across those steps.

rewriting the hippocampal buffer

When a new observation contradicts a stored episode, the TPJ network detects the false belief signal: the difference vector between the current cortical prediction and the retrieved stored prediction is non-zero. Two things happen immediately. The old tuple's retention score receives a penalty proportional to the contradiction magnitude. A new tuple writes with the corrected (state, action, outcome) pattern. On the next retrieval, the new tuple wins because its recency weight is higher. After sufficient level 1 micro-replay cycles, cortical weights shift toward the corrected pattern. After level 3, the old contradicted tuple's retention score may drop below the pruning threshold.

rewriting the centroid store

The centroid store rewrites through centroid migration during level 3 replay. Each cycle recomputes cluster centroids from current hippocampal tuples. If contradicted tuples were pruned and corrected tuples were written, the centroid moves. A schema built on 200 tuples where 10 get contradicted and replaced shifts its centroid by roughly 5 percent per level 3 cycle. Measured, controlled rewriting. Not a catastrophic flip.

rewriting cortical weights: memory reconsolidation

Deeply consolidated cortical knowledge rewrites through the reconsolidation mechanism documented by Nader, Schafe, and LeDoux in 2000. A retrieved memory becomes temporarily labile during the act of retrieval. Before reconsolidation completes, the memory is modifiable.

The implementation maps directly. Retrieval of a cortical schema activates the corresponding centroid from the centroid store. That activation temporarily raises the cortical learning rate for weights that contributed most strongly to generating that centroid, specifically those with activation above 0.7 times maximum in the relevant cortical layers. The elevated learning rate is 0.02, four times the normal online rate. It lasts for 500 steps after retrieval. During that window, contradicting observations produce four times the normal cortical weight shift. After 500 steps, the learning rate returns to baseline and the schema reconsolidates in its updated form.

Acetylcholine controls this process automatically. High acetylcholine signals uncertainty about the current context. When a contradicting observation arrives and hippocampal retrieval returns a low-confidence match, acetylcholine spikes. That spike triggers the elevated cortical learning rate. The reconsolidation window opens without requiring an explicit contradiction-detection module.

handling circular contradiction

If observation A contradicts stored pattern B, and later observation C contradicts the newly written A, the system oscillates. The retention score mechanism handles this by penalizing both A and C for unreliable prediction. Both lose retention weight. The cluster centroid drifts toward higher uncertainty. Acetylcholine persists at elevated levels. The system enters appropriate epistemic uncertainty rather than committing to either contradicting pattern.

Dopamine breaks the tie. The pattern whose downstream predictions produce more positive dopamine signals over subsequent steps wins consolidation. Reward-predictive accuracy is the final arbiter. This is Bayesian model selection weighted by predictive utility, implemented through the existing neuromodulator scalar at zero additional compute cost.

The reconsolidation window length of 500 steps is tunable. Setting it as a function of acetylcholine magnitude rather than a fixed constant makes the window duration scale automatically with uncertainty level, matching biological data from Eisenberg and Dudai (2004) showing that stronger prediction errors produce longer reconsolidation windows in rodent fear conditioning.


RESOLVED LIMITATIONS

This section closes the four open problems identified in version 0.1 and documents the transformer elimination from version 0.2.


SOCIAL COGNITION: TEMPOROPARIETAL JUNCTION AND MEDIAL PFC

The first-order agent classifier from the core architecture detects agents and attributes goals to them. That is not theory of mind. Theory of mind requires the system to represent what another agent believes, specifically a belief the system knows to be different from its own.

The biological substrate is the temporoparietal junction (TPJ) and medial prefrontal cortex (mPFC). The TPJ activates when a human reasons about another person's false beliefs. Saxe and Kanwisher established this in 2003 using fMRI. The mPFC handles self-referential processing and the contrast between self-model and other-model.

The computational solution is a second hippocampal slot, not a second hippocampal buffer. The main buffer stores the agent's own experience tuples. The social slot stores tuples structured as (observed agent ID, inferred goal vector, inferred belief state vector, observed action, outcome). This is a second class of tuple written into the same 10,000-entry buffer pool, tagged with a social flag bit. Social tuples consume at most 20 percent of buffer capacity by default. That cap is a hyperparameter.

The inferred belief state vector is 64 dimensions, matching the PFC goal vector size. It is not computed by a separate neural network. It is retrieved. When the agent observes another agent take action A from state S, it queries the hippocampal buffer with (S, A) as a partial input and retrieves the outcome the other agent was likely predicting. The retrieved predicted-outcome vector is the inferred belief. This works because the buffer already stores (state, action, outcome) tuples from the agent's own experience. The agent projects its own learned world model onto the observed agent.

This is exactly the "simulation theory" of mind proposed by Gordon in 1986 and refined by Goldman in 2006. The agent simulates what it would have predicted in the other agent's situation. No separate module is required. The hippocampal pattern completion mechanism already does the simulation.

False belief detection follows directly. If the system's current cortical state predicts outcome X, but the retrieved social tuple says the observed agent was predicting outcome Y, the difference vector (X minus Y) is the false belief signal. Positive values in this difference vector indicate dimensions where the other agent's beliefs diverge from reality as the system currently models it. Send this difference vector as an additional input to the PFC goal slot 2. The system can now plan actions that account for the other agent's mistaken prediction.

The TPJ maps to a small two-layer network reading the difference vector (64 dimensions) and the other agent's observed action (32 dimensions from the motor cortex action vocabulary). Output is a 32-dimensional "mentalizing vector" that biases the ACC conflict check and the basal ganglia goal-congruence term. Learned parameters: 64 plus 32 equals 96 inputs, 128 hidden units, 32 outputs. Total: approximately 16,000 weights.

The mPFC maps to the contrast computation between the agent's own PFC goal vector and the inferred goal vector stored in the social tuple. This contrast is a simple vector subtraction. The resulting 64-dimensional difference feeds the same TPJ network as an additional input channel. No extra parameters needed.

The full social cognition addition costs approximately 16,000 learned weights plus 20 percent of hippocampal buffer capacity for social tuples. At 4 bytes per weight, that is 64 kilobytes of new parameters. Per-step compute cost is one hippocampal query (already budgeted at 1 millisecond) plus one TPJ network forward pass at under 0.2 milliseconds.

The social cognition module also handles coordination. When the agent detects another agent with an aligned goal vector (cosine similarity above 0.8), the basal ganglia goal-congruence term receives a cooperative bonus proportional to that similarity. This biases action selection toward joint-action strategies without requiring a separate multi-agent planning module.


SLEEP REPLAY AND LONG-TERM CONSOLIDATION

The biological brain runs slow-wave sleep replay for 6 to 8 hours per night. Sharp-wave ripples in the hippocampus during NREM sleep fire at 80 to 100 Hz and replay compressed sequences at roughly 20 times real-time speed. This consolidates hippocampal patterns into neocortical weights. The 400-millisecond offline replay cycle in version 0.1 was a placeholder.

The solution is a tiered replay schedule with three levels, not one.

Level 1 runs continuously during inter-step gaps. Any pause between active inference steps exceeding 50 milliseconds triggers a micro-replay batch of 16 tuples. At 55 steps per second, gaps of 50 milliseconds occur whenever the environment is slower than the system. Each micro-replay batch costs approximately 12 milliseconds. This provides continuous trickle consolidation during normal operation.

Level 2 runs during explicit rest periods. Define a rest period as any continuous interval exceeding 2 seconds where no new sensory input arrives. Level 2 replay runs 512-tuple batches at maximum speed, prioritized by recency and novelty. Novelty is the norepinephrine scalar recorded at write time. High-novelty tuples replay first. A 512-tuple batch through the cortical hierarchy takes approximately 400 milliseconds. During a 2-second rest, the system completes 4 to 5 such batches, covering 2,000 to 2,500 tuples.

Level 3 is scheduled deep consolidation. Run this once every 10,000 inference steps, or every 3 hours of continuous operation at 55 steps per second. Level 3 replays the entire hippocampal buffer in random order, 10,000 tuples in batches of 512. Total time: approximately 8 seconds. During level 3, the system suspends active inference. This is the machine equivalent of sleep. The cortical learning rate during level 3 is raised to 0.02, four times the normal online rate, matching the biological observation that synaptic consolidation during sleep proceeds faster than waking plasticity.

After each level 3 cycle, apply buffer pruning. Compute a retention score for each tuple: retention equals recency weight times 0.5 plus retrieval frequency times 0.5. Discard the lowest-scoring 10 percent of tuples. This frees buffer capacity and mirrors biological memory forgetting, which is not random but preferentially eliminates low-relevance memories. The specific 0.5 weighting between recency and frequency is a hyperparameter.

The three-level schedule provides a replay rate that scales with operational intensity. Light usage means mostly level 1 micro-replays. Heavy usage triggers frequent level 2 rests. Every 3 hours, level 3 sweeps and prunes. Long-term forgetting is controlled, not catastrophic.

One practical implementation detail: run level 2 and level 3 replay on a separate thread. The main inference loop does not block. Level 3 consolidation updates shared cortical weights using a copy-on-write scheme: the replay thread writes to a shadow weight set and swaps it atomically after the batch completes. This keeps inference latency stable during consolidation.

The tiered schedule adds no new learned parameters. It changes only the timing and batch size of existing replay operations. Per-step amortized replay cost increases from the version 0.1 estimate of roughly 4 milliseconds per step to approximately 7 milliseconds per step when level 2 rest cycles are included in the average.


TRANSFORMER ELIMINATION

Version 0.2 compressed the language transformer from 3.5 million to 1.65 million parameters using weight tying, linear attention, and a shallower architecture. Version 0.3 eliminates the transformer entirely.

The reason is structural. A transformer imposes a hard context window baked into matrix dimensions at design time. Even with linear attention and cross-attention to the hippocampal buffer, the transformer's self-attention matrix still bounds within-sequence context. The recurrent cortical layer 6 has no such bound. Its within-episode context fades gracefully over 20 to 50 steps and the hippocampal buffer handles everything beyond that range. No hard ceiling exists anywhere in the system.

Parameters removed: 1.65 million transformer weights, 6.6 megabytes. Parameters added: 98,000 recurrent layer 6 weights plus 131 kilobytes of fixed LMU kernel. Net parameter reduction from version 0.2: approximately 1.55 million learned weights. Net reduction from version 0.1 transformer: approximately 3.4 million weights, 13.6 megabytes.

Per-step language inference time drops from 1.3 milliseconds (compressed transformer with cross-attention) to 0.4 milliseconds (recurrent layer 6 forward pass). The full per-step budget drops from 17 to 18 milliseconds to 14 to 15 milliseconds, raising throughput from 55 steps per second to approximately 67 to 70 steps per second.

Context window depth increases from the transformer's fixed sequence window to effectively unbounded depth through the LMU-hippocampal combination. That is a qualitative improvement, not a marginal one.


INFERIOR COLLICULUS: BINAURAL SOUND LOCALIZATION

The inferior colliculus (IC) is a midbrain auditory structure. Its primary job is computing interaural time differences (ITD) and interaural level differences (ILD). These two cues let the brain locate a sound source in three-dimensional space. ITD resolves azimuth (left-right). ILD resolves both azimuth and elevation in conjunction with head-related transfer functions.

The biological IC contains neurons tuned to specific ITD values from roughly minus 700 microseconds to plus 700 microseconds, corresponding to the maximum acoustic delay across a human head at 15 centimeters width. The Jeffress model (1948) remains the standard computational description: a delay line plus coincidence detection across the two cochlear inputs.

Implementation requires stereo audio input. Apply the mel-filterbank separately to the left and right channels, producing two 128-dimensional mel-spectrum vectors per frame. For each frequency band, compute the cross-correlation between the left and right channel signals across a lag range of minus 32 to plus 32 samples at 16 kHz sampling rate. That range covers plus or minus 2 milliseconds, which exceeds the biological maximum of 700 microseconds and provides margin for room reflections.

The peak lag in each frequency band is the ITD estimate for that band. Collect 128 ITD estimates (one per mel band) into a 128-dimensional ITD vector. Apply a second fixed filter bank of 32 Gaussian-tuned units across the ITD range, peaks spaced at 25-microsecond intervals from minus 700 to plus 700 microseconds. This is the Jeffress delay-line structure. Output is a 32-dimensional azimuth probability vector with no learned parameters.

ILD computation is simpler. Compute the log ratio of left-to-right power in each mel band. That gives a 128-dimensional ILD vector. Pass it through a small learned linear layer of 32 output units to produce an elevation estimate. Learned parameters: 128 times 32 equals 4,096 weights.

The full IC module outputs a 64-dimensional spatial audio vector: 32 azimuth units from the ITD Jeffress filter plus 32 elevation units from the learned ILD layer. This vector feeds the superior colliculus salience map as a third input channel alongside V1 edge density and V5 motion. A sound source now competes with visual stimuli for the attentional pointer.

The IC also feeds the parietal spatial map. When the azimuth and elevation outputs peak at a specific location, the corresponding parietal grid cell activates and marks an audio-source tag. This allows the system to track sound sources that are off-axis or occluded from vision.

Total new parameters: 4,096 learned weights for the ILD elevation layer, plus the fixed Jeffress filter bank with no learned weights. Storage: 16 kilobytes. Per-step compute cost: two mel-filterbank passes (already present in the auditory pipeline, so zero additional cost for the filterbank) plus one cross-correlation across 128 bands at 64 lags equals 8,192 multiply-accumulate operations, plus one 32-unit Gaussian filter pass equals 32 operations, plus one linear layer forward pass equals 4,096 operations. Total new operations per step: approximately 12,320 multiply-accumulates. At 3 GHz with vectorized SIMD, this completes in under 0.1 milliseconds.

The IC addition changes the superior colliculus salience map from a two-input weighted sum to a three-input weighted sum. The third weight, audio salience, initializes at 0.3, equal to the visual edge weight and the motion weight. All three salience weights are learnable. They will shift based on which modality proves more reliably predictive in a given environment.


UPDATED TOTAL PARAMETER AND MEMORY BUDGET (version 0.4)

The RGM visual hierarchy and auditory pathway replacements produce the largest single parameter reduction in the blueprint's history. The CNN ventral visual stream at 8.4 megabytes and the CNN auditory pathway at 0.18 megabytes are replaced by RGM hierarchies at approximately 66 kilobytes visual and 24 kilobytes auditory combined. That is a 130-fold reduction in those two components alone.

Fixed (no learning): retinal difference-of-Gaussians filters, V1 Gabor bank, color opponency transform, cochlear mel-filterbank, LMU entorhinal kernel (131 KB), dentate gyrus random projection matrix (fixed), grid cell spatial bases, number magnitude units, Jeffress ITD filters, claustrum random projection matrix, Spelke core knowledge priors. These contribute 0 bytes of learned weight storage.

Learned weights by component:
RGM visual hierarchy (V2 through IT plus fusiform branch): 67 KB
RGM auditory hierarchy (A1 and A2): 24 KB
Cortical hierarchy layers 1 through 5: 2.1 MB
Recurrent language layer 6: 392 KB
Thalamic gating network: 50 KB
Basal ganglia actor (continuous): 64 KB
SMA look-ahead buffer network: 20 KB
PFC goal matrix: 16 KB
Amygdala valence layer: 4 KB
OFC value network: 36 KB
TPJ social cognition network: 64 KB
Cerebellum and motor cortex: 160 KB
RSC spatial transform: 16 KB
Agent classifier (pretrained, fixed after pretraining): 32 KB
IC elevation layer: 16 KB
Superior colliculus salience weights: 1 KB

Total learned weights: approximately 3.0 MB. This is 2.6 percent of GPT-2 small's 117 million parameters.

Runtime buffers:
Hippocampal buffer (10,000 tuples, raw): 46 MB
Hippocampal buffer (fully compressed after level 3): approximately 10 MB
Centroid store (1,024 centroids): 4.7 MB
Parietal egocentric spatial map: 524 KB
Parietal allocentric spatial map (RSC addition): 524 KB
Working memory (PFC): 256 bytes
LIP evidence accumulator: 256 bytes
SMA action buffer: 1 KB

Total runtime buffers: approximately 52 MB raw, approximately 16 MB at full compression.

Full system memory footprint: approximately 55 MB at runtime with raw buffer. Approximately 19 MB at full compression after sustained level 3 cycling. On a Raspberry Pi 5 with 8 GB RAM, the full system occupies under 0.7 percent of available memory.

Revised per-step inference: approximately 13 to 14 milliseconds, yielding roughly 70 to 75 steps per second on a single CPU core at 3 GHz. The RGM visual hierarchy replaces 6 milliseconds of CNN forward passes with approximately 1.5 milliseconds of variational message passing over sparse discrete state spaces.

Effective context window: unbounded. Equivalent token depth at 10,000 tuples: 200,000 tokens. At 1 million tuples on NVMe: 20 million tokens. Retrieval cost at any depth: logarithmic via HNSW, under 10 milliseconds.

Hardware deployment cost: Raspberry Pi 5 at 80 dollars, 128 GB NVMe SSD at 12 dollars recommended (64 GB at 8 dollars minimum), 32 GB SD card at 5 dollars. Total: 97 dollars at under 8 watts. System learned weights fit in 3 megabytes, smaller than a JPEG photograph.