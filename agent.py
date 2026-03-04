import numpy as np
from sensory import SensorySystem
from memory import MemorySystem
from cortex import CortexSystem
from control import ControlSystem

class Agent:
    def __init__(self, action_space_size=4, input_shape=(128, 128, 3)):
        self.action_space_size = action_space_size
        self.input_shape = input_shape
        self.step_count = 0

        # Submodules
        self.sensory = SensorySystem()
        self.memory = MemorySystem()
        self.cortex = CortexSystem()
        self.control = ControlSystem(action_space_size)

        self.last_state = None

    def step(self, observation, reward, done, info):
        """
        UPDATE SCHEDULE:
        Step N: read sensors, run retina through IT and auditory pipeline, update superior colliculus attention pointer,
        update parietal spatial map at attended location, run thalamic gating, run cortical layers above threshold,
        query hippocampal buffer with current cortical state, run amygdala valence, run ACC conflict check,
        run basal ganglia selection, execute action, read outcome.

        Step N+1: compute TD error (dopamine signal), update basal ganglia values, update norepinephrine from cortical error,
        update serotonin from reward interval, write new tuple to hippocampal buffer if norepinephrine exceeds 0.6 (novelty threshold),
        update cerebellum on motor outcome error.
        """
        self.step_count += 1

        # --- Sensory Pipeline ---
        sensory_out = self.sensory.process(observation)
        it_vector = sensory_out['it_vector']
        belief = sensory_out['belief']
        a2_out = sensory_out['a2_out']

        # Get neuromodulators and PFC state
        ne = self.control.neuromodulators.norepinephrine
        pfc_goal = self.control.pfc.goal
        context_tag = np.zeros(64) # Dummy context for now

        # --- Cortical Hierarchy ---
        cortex_out = self.cortex.process(it_vector, a2_out, ne, pfc_goal)
        l5_state = cortex_out['l5_state']
        l6_state = cortex_out['l6_state']
        mean_cortical_error = cortex_out['mean_error']

        # --- Memory System ---
        hc_retrieved_tuple, hc_novelty = self.memory.process(
            l5_state, it_vector, belief, a2_out, l6_state, pfc_goal, context_tag
        )
        hc_entropy = np.clip(hc_novelty, 0, 1) # simple surrogate for acetylcholine/entropy

        # --- Amygdala & Control ---
        threat, reward_anticipation = self.control.amygdala.process(belief, hc_entropy)

        # --- Update N+1 logic from previous step ---
        if self.last_state is not None:
            # Dopamine = TD Error
            td_error = self.control.basal_ganglia.update(reward, l5_state, serotonin=self.control.neuromodulators.serotonin)

            # Update modulators
            self.control.neuromodulators.update(td_error, mean_cortical_error, hc_entropy)

        # --- Action Selection ---
        action, action_value = self.control.basal_ganglia.select_action(
            l5_state, pfc_goal, threat, reward_anticipation
        )

        self.last_state = l5_state

        return action

    def train_mnist_shot(self, image, label):
        """
        One-shot training for MNIST.
        Pass the image through the sensory and cortical pipeline.
        Use the label to set the PFC goal or context tag.
        The Hippocampal Buffer writes on first contact based on novelty.
        """
        # Convert grayscale to RGB for retina processing
        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)

        # Process sensory
        sensory_out = self.sensory.process(image)
        it_vector = sensory_out['it_vector']
        belief = sensory_out['belief']
        a2_out = sensory_out['a2_out']

        # Set PFC goal as one-hot of label for association
        pfc_goal = np.zeros(64)
        if label < 64:
            pfc_goal[label] = 1.0

        context_tag = np.zeros(64)

        # Process cortex (force write mode by keeping NE high)
        cortex_out = self.cortex.process(it_vector, a2_out, 1.0, pfc_goal)
        l5_state = cortex_out['l5_state']
        l6_state = cortex_out['l6_state']

        # Process memory (hippocampus)
        # Force a write by passing high novelty
        self.memory.process(
            l5_state, it_vector, belief, a2_out, l6_state, pfc_goal, context_tag
        )
        # Override buffer write logic to force write with artificial high mismatch if needed
        lmu_state = self.memory.lmu.state
        idx_vec = self.memory.buffer._create_index_vector(lmu_state, it_vector, context_tag)
        tup_data = np.concatenate([l5_state, it_vector, l6_state, pfc_goal, context_tag])
        self.memory.buffer.write(tup_data, idx_vec, 1.0) # force write

    def predict_mnist(self, image):
        """
        Retrieval pass for MNIST prediction.
        """
        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)

        sensory_out = self.sensory.process(image)
        it_vector = sensory_out['it_vector']

        # Fast query via LMU/IT vector partial match
        context_tag = np.zeros(64)
        lmu_state = self.memory.lmu.state # current state
        idx_vec = self.memory.buffer._create_index_vector(lmu_state, it_vector, context_tag)

        retrieved_tup = self.memory.buffer.retrieve(idx_vec)
        # Extract PFC goal from tuple:
        # Tuple format: L5(256) + IT(512) + L6(256) + PFC(64) + Context(64)
        pfc_start = 256 + 512 + 256
        pfc_goal = retrieved_tup[pfc_start:pfc_start+64]

        label = np.argmax(pfc_goal)
        return label
