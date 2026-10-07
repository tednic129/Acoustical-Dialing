"""Approach 1.c (owner: Tapan): the shared log-spectrogram, compressed with a 2-D DCT.

The 257 x 50 spectrogram from common/spectrogram.py is described as a sum of smooth cosine
patterns, and only the k x k smoothest patterns are kept: the same idea JPEG uses for photos.
"""
import numpy as np
from scipy.fft import dctn, idctn

from common.spectrogram import log_spectrogram

K = 16   # 16 x 16 = 256 values instead of 12,850 (about 50x smaller); sweep.py shows the trade-off


def compress_dct(spectrogram, k=K):
    """Keep the k x k lowest-frequency 2-D DCT coefficients of a spectrogram, as one vector."""
    coefficients = dctn(spectrogram, norm="ortho")
    return coefficients[:k, :k].ravel().astype(np.float32)


def decompress_dct(values, shape, k=K):
    """Rebuild an approximate spectrogram from the kept coefficients (only to look at what survived)."""
    coefficients = np.zeros(shape, dtype=np.float32)
    coefficients[:k, :k] = np.asarray(values).reshape(k, k)
    return idctn(coefficients, norm="ortho")


def features(event, k=K):
    """The shared interface: one trimmed word in, one feature vector out."""
    return compress_dct(log_spectrogram(event), k)


def features_batch(events, k=K):
    """Feature matrix for a list of words; used as the first step of the saved pipeline."""
    return np.stack([features(e, k) for e in events])
