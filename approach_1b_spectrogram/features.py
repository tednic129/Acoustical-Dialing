"""Approach 1.b (owner: Maulik): the whole log-spectrogram of the word as its feature vector.

common/spectrogram.py cuts the word into 25 ms frames every 10 ms, takes the spectrum of each
frame (257 frequencies, 0-8 kHz) in dB and stretches the word to 50 frames. That 257 x 50 picture
of how the word's frequencies change over time is flattened into one vector of 12,850 values,
with nothing removed: it is the uncompressed baseline that approach 1.c compresses.
"""
import numpy as np

from common import config
from common.spectrogram import log_spectrogram

SHAPE = (config.N_FFT // 2 + 1, config.N_FRAMES)   # 257 frequencies x 50 time frames
N_VALUES = SHAPE[0] * SHAPE[1]                      # 12,850 values per word


def features(event):
    """The shared interface: one trimmed word in, one feature vector out."""
    return log_spectrogram(event).ravel()


def features_batch(events):
    """Feature matrix for a list of words; used as the first step of the saved pipeline."""
    return np.stack([features(e) for e in events])
