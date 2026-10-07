"""Find the spoken word (the "event") inside a recording, using short-term energy and zero-crossing rate."""
import numpy as np

from common import config


def frame_features(x, fs=config.SAMPLE_RATE):
    """Return per-frame RMS energy (smoothed over 5 frames) and zero-crossing rate, plus frame length and hop."""
    n = int(fs * config.FRAME_MS / 1000)
    hop = int(fs * config.HOP_MS / 1000)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    energy = np.sqrt(np.mean(frames ** 2, axis=1))
    energy = np.convolve(energy, np.ones(5) / 5, mode="same")   # a single click no longer dominates
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    return energy, zcr, n, hop


def _region_around(peak, mask, max_gap):
    """First and last active frame of the active region containing `peak` (gaps up to max_gap frames allowed)."""
    active = np.flatnonzero(mask)
    regions = np.split(active, np.flatnonzero(np.diff(active) > max_gap) + 1)
    region = next(r for r in regions if r[0] <= peak <= r[-1])
    return int(region[0]), int(region[-1])


def find_event(x, fs=config.SAMPLE_RATE):
    """Return (start, end) sample indices of the word in x, or None if no speech is found.

    1. The loudest moment of the recording is taken as the middle of the word.
    2. The word grows around it over every frame clearly louder than the background noise,
       bridging short pauses. If it grows longer than a digit can be, background noise got in,
       so the threshold is raised step by step (noisy recordings).
    3. Quiet but hissy frames directly next to the word are added (the "chs" in "sechs").
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) < int(fs * config.FRAME_MS / 1000):
        return None
    energy, zcr, n, hop = frame_features(x, fs)
    noise = np.percentile(energy, config.NOISE_PERCENTILE) + 1e-10
    peak = int(np.argmax(energy))
    if energy[peak] <= config.ENERGY_FACTOR * noise:
        return None                                    # nothing stands out from the noise

    max_gap = max(1, int(config.MAX_GAP_MS / config.HOP_MS))
    max_word = int(config.MAX_WORD_MS / config.HOP_MS)
    threshold = max(config.ENERGY_FACTOR * noise, config.PEAK_FRACTION * energy[peak])
    for fraction in (0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        threshold = max(threshold, fraction * energy[peak])
        first, last = _region_around(peak, energy > threshold, max_gap)
        if last - first <= max_word:
            break

    hissy = (zcr > config.ZCR_THRESHOLD) & (energy > config.ZCR_ENERGY_FACTOR * noise)
    reach = int(config.HISS_EXTEND_MS / config.HOP_MS)
    lo, hi = first, last
    while first > 0 and hissy[first - 1] and lo - first < reach:
        first -= 1
    while last < len(energy) - 1 and hissy[last + 1] and last - hi < reach:
        last += 1

    pad = int(fs * config.PAD_MS / 1000)
    start = max(first * hop - pad, 0)
    end = min(last * hop + n + pad, len(x))
    return start, end


def extract_event(x, fs=config.SAMPLE_RATE):
    """Return only the spoken part of x, or None if no speech is found."""
    bounds = find_event(x, fs)
    if bounds is None:
        return None
    start, end = bounds
    return np.asarray(x[start:end], dtype=np.float32)
