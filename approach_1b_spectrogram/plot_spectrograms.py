"""Report figure: the average log-spectrogram of each digit (training speakers only).

    python -m approach_1b_spectrogram.plot_spectrograms
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import config
from common.preprocessing import load_processed, split_masks
from common.spectrogram import log_spectrogram

NAMES = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]


def mean_spectrograms(languages=("de",)):
    """One (257, 50) average spectrogram per digit, from the training speakers' words."""
    data = load_processed(languages=languages)
    train, _ = split_masks(data["speakers"])
    events, labels = data["events"][train], data["labels"][train]
    return np.stack([
        np.mean([log_spectrogram(e) for e in events[labels == digit]], axis=0)
        for digit in range(10)
    ])


def main():
    S = mean_spectrograms()
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.5), sharex=True, sharey=True)
    for digit, ax in enumerate(axes.flat):
        image = ax.imshow(S[digit], origin="lower", aspect="auto", cmap="magma",
                          extent=[0, 1, 0, config.SAMPLE_RATE / 2000],
                          vmin=S.min(), vmax=S.max())
        ax.set_title(f"{digit}  {NAMES[digit]}")
    fig.supxlabel("word time (stretched to 0-1)")
    fig.supylabel("frequency (kHz)")
    fig.colorbar(image, ax=axes, label="level (dB)", shrink=0.8)
    fig.suptitle("1.b: average log-spectrogram per digit (training speakers akshay + maulik)")
    path = config.RESULTS_DIR / "1b_spectrograms_per_digit.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
    