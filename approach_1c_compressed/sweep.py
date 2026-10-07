"""How much accuracy survives compression? The headline experiment of approach 1.c.

    python -m approach_1c_compressed.sweep              # German (config.LANGUAGES)
    python -m approach_1c_compressed.sweep --languages de en

Uses training speakers only (everyone not in split.json) with speaker-wise cross-validation, so
the test speakers stay untouched. Compares the full 257 x 50 spectrogram (what 1.b uses) with
DCT-compressed versions, all with the same RBF-SVM. Writes results/1c_sweep.json and
results/1c_accuracy_vs_compression.png.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from approach_1c_compressed.features import compress_dct  # noqa: E402
from common import config  # noqa: E402
from common.evaluation import speaker_cv  # noqa: E402
from common.preprocessing import load_processed, load_split  # noqa: E402
from common.spectrogram import log_spectrogram  # noqa: E402

K_VALUES = [4, 8, 12, 16, 24, 32]


def rbf_svm():
    """The classifier 1.c shares with 1.b, so the comparison only changes the features."""
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", nargs="+", default=list(config.LANGUAGES))
    args = parser.parse_args()

    data = load_processed(languages=args.languages)
    test_speakers = load_split()
    keep = ~np.isin(data["speakers"], test_speakers)
    if not test_speakers:
        print("Note: split.json has no test speakers yet, so every speaker is used for this sweep.")
    events, labels, speakers = data["events"][keep], data["labels"][keep], data["speakers"][keep]
    print(f"{len(labels)} words from {sorted(set(speakers.tolist()))}, languages {args.languages}")

    spectrograms = np.stack([log_spectrogram(e) for e in events])
    cv, groups = speaker_cv(speakers)
    full = spectrograms.reshape(len(spectrograms), -1)
    acc_full = cross_val_score(rbf_svm(), full, labels, groups=groups, cv=cv).mean()
    rows = [{"k": None, "values": full.shape[1], "ratio": 1.0, "kb_float32": full.shape[1] * 4 / 1000,
             "accuracy": round(float(acc_full), 4)}]
    print(f"\n{'kept':>9} {'values':>7} {'smaller':>8} {'accuracy':>9}")
    print(f"{'full':>9} {full.shape[1]:7d} {'1x':>8} {acc_full:9.3f}")
    for k in K_VALUES:
        x = np.stack([compress_dct(s, k) for s in spectrograms])
        acc = cross_val_score(rbf_svm(), x, labels, groups=groups, cv=cv).mean()
        ratio = full.shape[1] / x.shape[1]
        rows.append({"k": k, "values": x.shape[1], "ratio": round(ratio, 1), "kb_float32": x.shape[1] * 4 / 1000,
                     "accuracy": round(float(acc), 4)})
        print(f"{f'{k} x {k}':>9} {x.shape[1]:7d} {f'{ratio:.0f}x':>8} {acc:9.3f}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "1c_sweep.json", "w", encoding="utf-8") as f:
        json.dump({"languages": args.languages, "speakers": sorted(set(speakers.tolist())),
                   "classifier": "RBF SVM (C=10, gamma=scale)", "rows": rows}, f, indent=2)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    compressed = rows[1:]
    ax.semilogx([r["ratio"] for r in compressed], [r["accuracy"] for r in compressed], "o-",
                label="1.c compressed (RBF SVM)")
    ax.axhline(acc_full, ls="--", color="gray", label="1.b full spectrogram (RBF SVM)")
    for r in compressed:
        ax.annotate(f"k={r['k']}", (r["ratio"], r["accuracy"]), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.set_xlabel("compression ratio (times smaller than the full spectrogram)")
    ax.set_ylabel("speaker-wise validation accuracy")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "1c_accuracy_vs_compression.png", dpi=150)
    print("\nSaved results/1c_sweep.json and results/1c_accuracy_vs_compression.png")


if __name__ == "__main__":
    main()
