"""Every number the three approaches share. Change it only through a common/ pull request."""
from pathlib import Path

# Paths (all relative to the repository root, wherever it is cloned)
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SPLIT_FILE = ROOT / "split.json"

# Audio
SAMPLE_RATE = 16000        # every recording is resampled to 16 kHz mono when loaded
LANGUAGES = ("de",)        # German numerals for the K-town demo; ("de", "en") uses both

# Event extraction: energy + zero-crossing rate (common/events.py)
FRAME_MS = 25              # analysis frame
HOP_MS = 10                # frame step
PAD_MS = 50                # margin kept before and after the detected word
NOISE_PERCENTILE = 10      # the quietest 10 % of frames estimate the background noise
ENERGY_FACTOR = 4.0        # a frame is speech when it is 4x louder than the noise ...
PEAK_FRACTION = 0.05       # ... and at least 5 % of the loudest frame
MAX_GAP_MS = 250           # pauses shorter than this stay inside one word ("ach-t", "sie-ben")
MAX_WORD_MS = 1200         # longer than this means background noise got in: raise the threshold
ZCR_THRESHOLD = 0.3        # quiet but hissy frames right next to the word also count ...
ZCR_ENERGY_FACTOR = 2.0    # ... when they are at least 2x louder than the noise ...
HISS_EXTEND_MS = 200       # ... for at most 200 ms on each side (the "chs" in "sechs")

# Spectrogram, shared by 1.b and 1.c (common/spectrogram.py)
WIN_LENGTH = 400           # 25 ms Hann window at 16 kHz
HOP_LENGTH = 160           # 10 ms hop
N_FFT = 512                # 257 frequency rows, 0-8 kHz
DB_FLOOR = 80.0            # ignore everything 80 dB below the loudest point
N_FRAMES = 50              # every word is stretched to 50 time frames
