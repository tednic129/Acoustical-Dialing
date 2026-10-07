"""Loading recordings and processed arrays, file-name parsing, normalisation and the speaker split."""
import json
import re
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from common import config

# <digit>_<lang>_<speaker>_<take>.wav, e.g. 6_en_maulik_03.wav or 00_de_tapan_01.wav
NAME_PATTERN = re.compile(r"^(\d{1,2})_([a-z]{2})_([a-z0-9]+)_(\d+)$", re.IGNORECASE)


def parse_name(path):
    """Return {digit, language, speaker, take} from a recording's file name, or None if it doesn't match."""
    match = NAME_PATTERN.match(path.stem)
    if match is None:
        return None
    digit, language, speaker, take = match.groups()
    if not 0 <= int(digit) <= 9:
        return None
    return {"digit": int(digit), "language": language.lower(), "speaker": speaker.lower(), "take": int(take)}


def load_audio(path, fs=config.SAMPLE_RATE):
    """Read a WAV file as float32 mono at fs Hz (stereo is averaged, other rates are resampled)."""
    x, rate = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if rate != fs:
        g = gcd(int(rate), int(fs))
        x = resample_poly(x, fs // g, int(rate) // g).astype(np.float32)
    return x, int(rate)


def normalise(x):
    """Scale a signal so its peak is 1: loud and quiet speakers look alike."""
    return np.asarray(x, dtype=np.float32) / (np.max(np.abs(x)) + 1e-9)


def fix_length(x, length):
    """Zero-pad short signals and cut long ones to exactly `length` samples."""
    return np.pad(x, (0, max(0, length - len(x))))[:length]


def load_processed(languages=config.LANGUAGES):
    """Load what common.prepare_data saved, keeping only the requested languages.

    Returns a dict of arrays: signals, events, labels, speakers, languages, paths.
    """
    names = ["signals", "events", "labels", "speakers", "languages", "paths"]
    files = {n: config.PROCESSED_DIR / f"{n}.npy" for n in names}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Run `python -m common.prepare_data` first. Missing: " + ", ".join(missing))
    data = {n: np.load(p, allow_pickle=True) for n, p in files.items()}
    if languages:
        keep = np.isin(data["languages"], list(languages))
        data = {n: a[keep] for n, a in data.items()}
    return data


def load_split():
    """Return the list of test speakers from split.json (decided once by the team)."""
    with open(config.SPLIT_FILE, encoding="utf-8") as f:
        return [s.lower() for s in json.load(f).get("test_speakers", [])]


def split_masks(speakers):
    """Boolean masks (train, test) for a speaker-independent hold-out split."""
    test_speakers = load_split()
    if not test_speakers:
        raise ValueError(
            "split.json has no test speakers yet. Decide as a team which speaker(s) are held out, "
            'e.g. {"test_speakers": ["maulik"]}, and commit it through a common/ pull request.'
        )
    unknown = sorted(set(test_speakers) - set(np.unique(speakers)))
    if unknown:
        raise ValueError(f"split.json names speakers with no recordings: {unknown}")
    test = np.isin(speakers, test_speakers)
    return ~test, test
