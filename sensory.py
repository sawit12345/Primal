import numpy as np
import cv2
import scipy.ndimage as ndimage

class Retina:
    def __init__(self, target_shape=(128, 128)):
        self.target_shape = target_shape
        # Narrow gaussian sigma=1, Wide gaussian sigma=3
        self.sigma_narrow = 1.0
        self.sigma_wide = 3.0

    def process(self, image):
        """
        Process an RGB image into 4 channels: luminance ON, luminance OFF, red-green, blue-yellow.
        Returns array of shape (4, target_shape[0], target_shape[1])
        """
        if image.shape[:2] != self.target_shape:
            image = cv2.resize(image, self.target_shape, interpolation=cv2.INTER_AREA)

        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0

        if len(image.shape) == 2: # Grayscale fallback
            image = np.stack((image,)*3, axis=-1)

        R, G, B = image[..., 0], image[..., 1], image[..., 2]

        # Color opponency
        lum = (R + G + B) / 3.0
        red_green = R - G
        blue_yellow = B - (R + G) / 2.0

        # DoG convolution
        lum_narrow = ndimage.gaussian_filter(lum, sigma=self.sigma_narrow)
        lum_wide = ndimage.gaussian_filter(lum, sigma=self.sigma_wide)

        lum_on = np.clip(lum_narrow - lum_wide, 0, None)
        lum_off = np.clip(lum_wide - lum_narrow, 0, None)

        rg_narrow = ndimage.gaussian_filter(red_green, sigma=self.sigma_narrow)
        rg_wide = ndimage.gaussian_filter(red_green, sigma=self.sigma_wide)
        rg_diff = rg_narrow - rg_wide

        by_narrow = ndimage.gaussian_filter(blue_yellow, sigma=self.sigma_narrow)
        by_wide = ndimage.gaussian_filter(blue_yellow, sigma=self.sigma_wide)
        by_diff = by_narrow - by_wide

        return np.stack([lum_on, lum_off, rg_diff, by_diff])

class V1:
    def __init__(self, target_shape=(128, 128)):
        self.target_shape = target_shape
        self.filters = self._build_gabor_bank()

    def _build_gabor_bank(self):
        filters = []
        ksize = 11
        orientations = [theta * np.pi / 8 for theta in range(8)]
        # Frequencies at 2, 4, 8 cycles per degree. Translating to spatial frequencies approximately.
        lambdas = [3, 6, 12]
        phases = [0, np.pi / 2] # sine, cosine

        for theta in orientations:
            for lambd in lambdas:
                for psi in phases:
                    # fixed parameters to approximate biological tuning
                    sigma = lambd * 0.8
                    gamma = 0.5
                    kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
                    filters.append(kern)
        return filters

    def process(self, retina_output):
        """
        Applies 48 Gabor filters across 4 input channels.
        Total output channels = 48 * 4 = 192.
        Applies hard threshold at 0.2 * max activation per filter.
        """
        out_channels = []
        for c in range(retina_output.shape[0]):
            channel_data = retina_output[c]
            for kern in self.filters:
                # Use valid padding or same padding. Let's use same.
                resp = cv2.filter2D(channel_data, cv2.CV_32F, kern)
                # Sparsity threshold
                max_act = np.max(resp)
                if max_act > 0:
                    resp[resp < 0.2 * max_act] = 0
                else:
                    resp[:] = 0
                out_channels.append(resp)
        # downsample or pool if necessary. Blueprint says V1 output feeds RGM.
        return np.stack(out_channels)

class RGMLevel:
    def __init__(self, num_states, prev_level_size):
        self.num_states = num_states
        # D matrix mapping latent state to states below.
        # Starting with random sparse initialization
        self.D = np.random.rand(num_states, prev_level_size) * 0.1
        # Sparse block diagonal in biological setup, but we simulate it functionally
        self.B = np.random.rand(num_states, num_states, 4) * 0.01

    def infer(self, bottom_up_msg):
        """
        Variational message passing / Free energy minimization.
        For blueprint: RGM outputs discrete latent states or probability distribution over them.
        """
        # Simplified Free Energy Minimization (Variational Inference point estimate)
        # e = bottom_up_msg - D.T @ s
        # s_new = s_old - lr * (D @ e + Prior)

        # We approximate MAP inference over categorical states
        # Compute likelihood of each state generating the bottom up message
        # We can treat bottom_up_msg as a flattened vector for simple inference
        msg_flat = bottom_up_msg.reshape(-1)
        if self.D.shape[1] != msg_flat.shape[0]:
            # Adjust D shape dynamically if it's the first pass (structure learning placeholder)
            self.D = np.random.rand(self.num_states, msg_flat.shape[0]) * 0.1

        # Simplistic activation: dot product as a proxy for likelihood
        activation = self.D @ msg_flat
        activation = np.maximum(activation, 0)
        # Normalize to get probability distribution
        if np.sum(activation) > 0:
            prob = activation / np.sum(activation)
        else:
            prob = np.ones(self.num_states) / self.num_states

        # Learn D
        # Simplified Hebbian / predictive coding update
        pred = self.D.T @ prob
        error = msg_flat - pred
        # dD/dt \propto error * prob
        self.D += 0.05 * np.outer(prob, error)
        # Enforce sparsity
        self.D[self.D < 0.01] = 0.0

        return prob

class VisualHierarchy:
    def __init__(self, v1_output_shape):
        # RGM Levels
        self.level1 = RGMLevel(64, np.prod(v1_output_shape))
        self.level2 = RGMLevel(128, 64)
        self.level3 = RGMLevel(256, 128)
        self.fusiform = RGMLevel(32, np.prod(v1_output_shape)) # Fusiform bypasses directly from V1

    def process(self, v1_output):
        l1_prob = self.level1.infer(v1_output)
        l2_prob = self.level2.infer(l1_prob)
        l3_prob = self.level3.infer(l2_prob)

        # Fusiform branch
        ff_prob = self.fusiform.infer(v1_output)

        # Output is a 256-d belief distribution (l3_prob) plus the 512-d MAP state index one-hot vector
        # Wait, blueprint says:
        # "The output of RGM level 3 is a probability distribution over 256 discrete object states...
        # The MAP state index feeds downstream as a 256-dimensional one-hot vector for compatibility
        # Concatenated with something? IT vector is 512 dimensions in some parts of the text,
        # but 256 belief + 256 one-hot = 512 dimensions.
        map_idx = np.argmax(l3_prob)
        one_hot = np.zeros_like(l3_prob)
        one_hot[map_idx] = 1.0

        it_vector = np.concatenate([l3_prob, one_hot])
        return it_vector, l3_prob, ff_prob

class V5:
    def __init__(self, target_shape=(16, 16)):
        self.target_shape = target_shape
        self.prev_frame = None

    def process(self, image):
        # Lucas-Kanade optical flow
        # Expects grayscale 8-bit image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_small = cv2.resize(gray, self.target_shape, interpolation=cv2.INTER_AREA)

        if self.prev_frame is None:
            self.prev_frame = gray_small
            return np.zeros((*self.target_shape, 2), dtype=np.float32)

        flow = cv2.calcOpticalFlowFarneback(self.prev_frame, gray_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev_frame = gray_small

        return flow # (16, 16, 2)

class Auditory:
    def __init__(self):
        # Output auditory A2 vector: 128 dimensions.
        pass
    def process(self, audio=None):
        return np.zeros(128, dtype=np.float32)

class SensorySystem:
    def __init__(self):
        self.retina = Retina()
        self.v1 = V1()
        self.v5 = V5()
        self.hierarchy = None # initialized lazily based on v1 output shape
        self.auditory = Auditory()

    def process(self, observation):
        retina_out = self.retina.process(observation)
        v1_out = self.v1.process(retina_out)

        if self.hierarchy is None:
            self.hierarchy = VisualHierarchy(v1_out.shape)

        it_vector, belief, ff_prob = self.hierarchy.process(v1_out)
        v5_out = self.v5.process(observation)
        a2_out = self.auditory.process()

        return {
            'it_vector': it_vector, # 512 dim
            'belief': belief, # 256 dim
            'ff_prob': ff_prob, # 32 dim
            'v5_out': v5_out, # (16, 16, 2)
            'a2_out': a2_out, # 128 dim
            'v1_edge_density': np.mean(v1_out, axis=0) # spatial map for superior colliculus
        }
