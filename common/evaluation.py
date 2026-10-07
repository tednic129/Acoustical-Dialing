"""Speaker-wise cross-validation and one results format for every approach."""
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedKFold  # noqa: E402

from common import config  # noqa: E402


def speaker_cv(speakers, max_splits=3):
    """Cross-validation that never puts one speaker in two folds.

    Returns (cv, groups) for scikit-learn. With fewer than two speakers it falls back to a
    stratified 5-fold split, which flatters the score; say so in the report.
    """
    n_speakers = len(np.unique(speakers))
    if n_speakers >= 2:
        return GroupKFold(n_splits=min(max_splits, n_speakers)), speakers
    print("Warning: only one training speaker, falling back to a stratified (not speaker-wise) split.")
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=0), None


def save_results(approach, classifier, y_true, y_pred, values_per_word, **extra):
    """Save results/<approach>_<classifier>.json and its confusion matrix; return the dict.

    Every approach uses this, so compare.py can build the final table from all files.
    """
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = f"{approach}_{re.sub(r'[^a-z0-9]+', '_', classifier.lower()).strip('_')}"
    labels = list(range(10))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1)
    per_digit = np.divide(cm.diagonal(), support, out=np.zeros(10), where=support > 0)
    result = {
        "approach": approach,
        "classifier": classifier,
        "test_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "per_digit_accuracy": [round(float(a), 3) for a in per_digit],
        "confusion_matrix": cm.tolist(),
        "values_per_word": int(values_per_word),
        **extra,
    }
    with open(config.RESULTS_DIR / f"{slug}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{approach} {classifier}: {result['test_accuracy']:.1%}")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / f"{slug}_confusion.png", dpi=150)
    plt.close(fig)
    return result
