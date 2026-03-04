import numpy as np
import scipy.ndimage as ndimage
import cv2

class Retina:
    def __init__(self):
        pass

    def process(self, image):
        R = image[:, :, 0].astype(np.float32)
        G = image[:, :, 1].astype(np.float32)
        B = image[:, :, 2].astype(np.float32)

        L = (R + G + B) / 3.0
        RG = R - G
        BY = B - (R + G) / 2.0

        channels = [L, L, RG, BY]

        outputs = []
        for i, ch in enumerate(channels):
            narrow = ndimage.gaussian_filter(ch, sigma=1)
            wide = ndimage.gaussian_filter(ch, sigma=3)
            dog = narrow - wide

            if i == 1:
                dog = -dog

            dog = np.clip(dog, 0, None)
            outputs.append(dog)

        return np.stack(outputs, axis=-1)

class V1:
    def __init__(self):
        self.filters = self._create_gabor_filters()

    def _create_gabor_filters(self):
        filters = []
        ksize = 11
        for theta in np.arange(0, np.pi, np.pi/8):
            for lam in [2, 4, 8]:
                for phase in [0, np.pi/2]:
                    sigma = 0.5 * lam
                    gamma = 0.5
                    kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, phase, ktype=cv2.CV_32F)
                    filters.append(kern)
        return filters

    def process(self, retina_out):
        outputs = []
        for f in self.filters:
            ch_out = np.zeros((128, 128))
            for c in range(4):
                ch_out += cv2.filter2D(retina_out[:, :, c], cv2.CV_32F, f)
            outputs.append(ch_out)

        out = np.stack(outputs, axis=-1)
        for i in range(out.shape[-1]):
            m = np.max(out[:, :, i])
            if m > 1e-5:
                out[:, :, i][out[:, :, i] < 0.2 * m] = 0

        return out

class V5:
    def __init__(self):
        self.prev_frame = None

    def process(self, v1_out):
        current = np.sum(v1_out, axis=-1).astype(np.uint8)
        current_resized = cv2.resize(current, (16, 16))

        if self.prev_frame is None:
            self.prev_frame = current_resized
            return np.zeros((16, 16, 2))

        flow = cv2.calcOpticalFlowFarneback(self.prev_frame, current_resized, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev_frame = current_resized
        return flow
