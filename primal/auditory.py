import numpy as np

class Auditory:
    def __init__(self, sample_rate=16000):
        # Cochlea: 128 mel-filterbank applied to 25ms window, 10ms hop.
        self.sample_rate = sample_rate
        self.window_size = int(0.025 * sample_rate)
        self.hop_size = int(0.010 * sample_rate)
        self.num_mel_bands = 128
        self.mel_filters = self._create_mel_filterbank()

    def _hz_to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700)

    def _mel_to_hz(self, mel):
        return 700 * (10**(mel / 2595) - 1)

    def _create_mel_filterbank(self):
        fft_size = int(2 ** np.ceil(np.log2(self.window_size)))
        mel_min = 0
        mel_max = self._hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(mel_min, mel_max, self.num_mel_bands + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((fft_size + 1) * hz_points / self.sample_rate).astype(int)

        filters = np.zeros((self.num_mel_bands, fft_size // 2 + 1))
        for i in range(1, self.num_mel_bands + 1):
            f_m_minus = bin_points[i - 1]
            f_m = bin_points[i]
            f_m_plus = bin_points[i + 1]

            for k in range(f_m_minus, f_m):
                filters[i - 1, k] = (k - bin_points[i - 1]) / (bin_points[i] - bin_points[i - 1])
            for k in range(f_m, f_m_plus):
                filters[i - 1, k] = (bin_points[i + 1] - k) / (bin_points[i + 1] - bin_points[i])

        return filters

    def process(self, audio):
        if len(audio) < self.window_size:
            audio = np.pad(audio, (0, self.window_size - len(audio)))

        window = audio[-self.window_size:] * np.hamming(self.window_size)
        fft_size = int(2 ** np.ceil(np.log2(self.window_size)))
        mag_spec = np.abs(np.fft.rfft(window, n=fft_size))

        mel_spec = np.dot(self.mel_filters, mag_spec)
        return mel_spec

class InferiorColliculus:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.lags = np.arange(-32, 33)
        self.jeffress_peaks = np.linspace(-700e-6, 700e-6, 32)
        self.W_ild_elev = np.random.randn(128, 32) * 0.01

    def process(self, left_mel_spec, right_mel_spec, left_audio, right_audio):
        itd_estimates = np.zeros(128)

        azimuth = np.zeros(32)
        for i, peak in enumerate(self.jeffress_peaks):
            dist = (itd_estimates - peak) ** 2
            azimuth[i] = np.sum(np.exp(-dist / (2 * (50e-6)**2)))
        azimuth /= (np.sum(azimuth) + 1e-8)

        ild = np.log((left_mel_spec + 1e-8) / (right_mel_spec + 1e-8))
        elevation = np.dot(ild, self.W_ild_elev)

        return np.concatenate([azimuth, elevation])
