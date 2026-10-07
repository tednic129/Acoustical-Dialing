"""How many values does the FFT spectrum need? The dimensionality-reduction experiment of 1.a.

    python -m approach_1a_fft_dr.sweep                  # German (config.LANGUAGES)
    python -m approach_1a_fft_dr.sweep --languages de en

Uses training speakers only (everyone not in split.json) with speaker-wise cross-validation, so
the test speakers stay untouched. The 128-band FFT spectrum is reduced to fewer values with PCA
(unsupervised) and LDA (supervised, at most 9 values for 10 digits) and classified with the three
pipelines of train.py, then compared with no reduction at all. Writes results/1a_sweep.json and
results/1a_accuracy_vs_components.png.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from approach_1a_fft_dr.features import N_BANDS, features_batch  # noqa: E402
from approach_1a_fft_dr.train import build_models  # noqa: E402
from common import config  # noqa: E402
from common.evaluation import speaker_cv  # noqa: E402
from common.preprocessing import load_processed, load_split  # noqa: E402

PCA_VALUES = [2, 3, 5, 10, 20, 30, 50, 80]
LDA_VALUES = [2, 3, 5, 9]
COLOURS = {"kNN": "#2a78d6", "Linear SVM": "#eb6834", "RBF SVM": "#1baf7a"}   # one colour per classifier
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def pipelines(method, values, bands):
    """train.py's three pipelines without their feature step, because the features are computed once here."""
    for name, model in build_models("pca" if method == "none" else method, values, bands).items():
        steps = model.steps[1:]                      # scaling, reduction, classifier
        if method == "none":
            steps = [steps[0], steps[-1]]            # scaling and classifier only
        yield name, Pipeline(steps)


def plot(rows, bands, path):
    """Accuracy against values kept: PCA solid, LDA dashed, no reduction as a diamond at the right."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (name, colour) in enumerate(COLOURS.items()):
        for method, style, marker in (("pca", "-", "o"), ("lda", "--", "s"), ("none", "", "D")):
            points = [(r["values"], r["accuracy"][name]) for r in rows if r["reduction"] == method]
            if points:
                x, y = zip(*points)
                if method == "none":                 # three diamonds side by side, not on top of each other
                    x = [v * (0.93 + 0.07 * i) for v in x]
                ax.plot(x, y, linestyle=style or "none", marker=marker, color=colour, lw=1.5, ms=6,
                        markeredgecolor=SURFACE, markeredgewidth=1, solid_capstyle="round")
    ax.axhline(0.1, color=MUTED, lw=0.8)
    ax.text(2, 0.115, "chance (10 %)", color=MUTED, fontsize=8)

    ax.set_xscale("log")
    ticks = sorted({r["values"] for r in rows if r["reduction"] != "lda"} | {2})   # LDA's 9 would crowd 10
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK, labelsize=8)
    ax.set_xlabel(f"values kept per word (of {bands} FFT bands, log scale)", color=INK)
    ax.set_ylabel("speaker-wise validation accuracy", color=INK)

    keys = [Line2D([], [], color=c, lw=1.5, label=n) for n, c in COLOURS.items()]
    keys += [Line2D([], [], color=MUTED, lw=1.5, ls="-", marker="o", ms=5, label="PCA"),
             Line2D([], [], color=MUTED, lw=1.5, ls="--", marker="s", ms=5, label="LDA"),
             Line2D([], [], color=MUTED, ls="none", marker="D", ms=5, label="no reduction")]
    ax.legend(handles=keys, loc="upper left", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bands", type=int, default=N_BANDS, help="frequency bands of the FFT spectrum")
    parser.add_argument("--languages", nargs="+", default=list(config.LANGUAGES))
    args = parser.parse_args()

    data = load_processed(languages=args.languages)
    test_speakers = load_split()
    keep = ~np.isin(data["speakers"], test_speakers)
    if not test_speakers:
        print("Note: split.json has no test speakers yet, so every speaker is used for this sweep.")
    events, labels, speakers = data["events"][keep], data["labels"][keep], data["speakers"][keep]
    print(f"{len(labels)} words from {sorted(set(speakers.tolist()))}, languages {args.languages}, "
          f"{args.bands} FFT bands")

    x = features_batch(events, n_bands=args.bands)
    cv, groups = speaker_cv(speakers)
    # PCA can keep at most as many values as the smallest training fold has words
    limit = min(args.bands, min(len(train) for train, _ in cv.split(x, labels, groups)))
    settings = [("none", args.bands)] + [("pca", n) for n in PCA_VALUES if n <= limit]
    settings += [("lda", n) for n in LDA_VALUES]

    names = list(COLOURS)
    print(f"\n{'reduction':>9} {'values':>6} " + " ".join(f"{n:>10}" for n in names))
    rows = []
    for method, values in settings:
        accuracy = {name: round(float(cross_val_score(model, x, labels, groups=groups, cv=cv).mean()), 4)
                    for name, model in pipelines(method, values, args.bands)}
        rows.append({"reduction": method, "values": values, "accuracy": accuracy})
        print(f"{method.upper() if method != 'none' else 'none':>9} {values:6d} "
              + " ".join(f"{accuracy[n]:10.3f}" for n in names))

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "1a_sweep.json", "w", encoding="utf-8") as f:
        json.dump({"languages": args.languages, "speakers": sorted(set(speakers.tolist())),
                   "fft_bands": args.bands, "rows": rows}, f, indent=2)
    plot(rows, args.bands, config.RESULTS_DIR / "1a_accuracy_vs_components.png")
    print("\nSaved results/1a_sweep.json and results/1a_accuracy_vs_components.png")


if __name__ == "__main__":
    main()
