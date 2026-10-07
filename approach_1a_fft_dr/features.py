"""Approach 1.a (owner: Akshay): the FFT magnitude spectrum of the whole word, averaged into bands.

One FFT over the whole word tells how strongly each frequency is present, without regard to when
it occurs. Neighbouring frequencies are averaged into bands, which smooths out the pitch of the
speaker's voice, and the result is put on a dB scale so quiet and loud parts both count.

The dimensionality reduction (PCA or LDA) is not done here. It has to be fitted on training data
only, so it is a step of the pipeline in train.py.
"""
import numpy as np

from common import config
from common.preprocessing import fix_length, normalise

N_FFT = 32768    # 2.05 s at 16 kHz: longer than any word, so no word is cut off
N_BANDS = 128    # the 16,384 FFT bins (0-8 kHz) are averaged into 128 bands of 62.5 Hz each


def fft_spectrum(event, n_fft=N_FFT, n_bands=N_BANDS):
    """Return the dB magnitude spectrum of one word as a vector of n_bands values.

    The word is scaled to peak 1 and zero-padded to n_fft samples. With n_bands=None every FFT
    bin is kept (n_fft / 2 values).
    """
    x = fix_length(normalise(event), n_fft)
    magnitude = np.abs(np.fft.rfft(x))[: n_fft // 2]
    if n_bands:
        if (n_fft // 2) % n_bands:
            raise ValueError(f"n_bands must divide {n_fft // 2} evenly, got {n_bands}")
        magnitude = magnitude.reshape(n_bands, -1).mean(axis=1)
    return (20 * np.log10(magnitude + 1e-8)).astype(np.float32)


def band_centres(n_fft=N_FFT, n_bands=N_BANDS, fs=config.SAMPLE_RATE):
    """Centre frequency in Hz of each band, for plotting a spectrum."""
    width = fs / 2 / n_bands
    return (np.arange(n_bands) + 0.5) * width


def features(event, n_bands=N_BANDS):
    """The shared interface: one trimmed word in, one feature vector out."""
    return fft_spectrum(event, n_bands=n_bands)


def features_batch(events, n_bands=N_BANDS):
    """Feature matrix for a list of words; used as the first step of the saved pipeline."""
    return np.stack([features(e, n_bands) for e in events])
