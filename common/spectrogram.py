"""Log-spectrogram shared by approaches 1.b and 1.c.

Owner: Maulik (1.b). This is the version from the team guide so that 1.c can start; change it
through a common/spectrogram pull request, since 1.c compresses exactly this picture.
"""
import numpy as np
from scipy.signal import stft

from common import config
from common.preprocessing import normalise


def log_spectrogram(event, fs=config.SAMPLE_RATE, n_frames=config.N_FRAMES):
    """Return a (257, n_frames) dB spectrogram of one word, stretched to a fixed number of frames."""
    _, _, z = stft(
        normalise(event),
        fs=fs,
        window="hann",
        nperseg=config.WIN_LENGTH,
        noverlap=config.WIN_LENGTH - config.HOP_LENGTH,
        nfft=config.N_FFT,
    )
    s = 10 * np.log10(np.abs(z) ** 2 + 1e-10)
    s = np.maximum(s, s.max() - config.DB_FLOOR)
    old, new = np.linspace(0, 1, s.shape[1]), np.linspace(0, 1, n_frames)
    return np.array([np.interp(new, old, row) for row in s], dtype=np.float32)
