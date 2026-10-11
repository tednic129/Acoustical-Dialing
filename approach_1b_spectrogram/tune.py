"""Tune the 1.b classifiers on the training speakers only (see the team's tuning protocol).

    python -m approach_1b_spectrogram.tune                    # German  -> results/1b_tuning.json
    python -m approach_1b_spectrogram.tune --languages en     # English -> results/1b_en_tuning.json
"""
import argparse
import json

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from approach_1b_spectrogram.features import N_VALUES, features_batch
from common import config
from common.evaluation import speaker_cv
from common.preprocessing import load_processed, split_masks


def prefix(languages):
    """File-name prefix: '1b' for German (unchanged names), '1b_en' for English, '1b_de_en' for both."""
    languages = list(languages)
    return "1b" if languages == ["de"] else "1b_" + "_".join(languages)


def load_training_features(languages=("de",)):
    """Spectrogram features of the training speakers' words, computed once."""
    data = load_processed(languages=languages)
    train, _ = split_masks(data["speakers"])        # the test mask is thrown away on purpose
    X = features_batch(data["events"][train])        # (n_words, 12850)
    return X, data["labels"][train], data["speakers"][train]


def search(classifier, grid, X, y, speakers):
    """Speaker-wise grid search of scaling + one classifier; returns the fitted search."""
    cv, groups = speaker_cv(speakers)
    gs = GridSearchCV(make_pipeline(StandardScaler(), classifier), grid,
                      cv=cv, scoring="accuracy", refit=True, n_jobs=-1)
    gs.fit(X, y, groups=groups)
    return gs


def simplicity(params):
    """Sort key: simpler settings first (smaller C, smaller gamma, larger k, uniform before distance)."""
    return (params.get("svc__C", 0), params.get("svc__gamma", 0),
            -params.get("kneighborsclassifier__n_neighbors", 0),
            params.get("kneighborsclassifier__weights", "uniform") != "uniform")


def tie_rule(rows, tolerance=0.01):
    """The protocol's choice: among settings within 0.01 of the best validation, the simplest one."""
    best = max(row["validation"] for row in rows)
    close = [row for row in rows if row["validation"] >= best - tolerance - 1e-9]
    return min(close, key=lambda row: simplicity(row["params"]))["params"]


# The shared grids from the tuning protocol (gamma = factor / number of features)
GRIDS = {
    "Linear SVM": (SVC(kernel="linear"), {"svc__C": [0.0001, 0.001, 0.01, 0.1, 1]}),
    "RBF SVM": (SVC(kernel="rbf"), {"svc__C": [1, 10, 100],
                                    "svc__gamma": [f / N_VALUES for f in (0.1, 1, 10)]}),
    "kNN": (KNeighborsClassifier(), {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7],
                                     "kneighborsclassifier__weights": ["uniform", "distance"]}),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", nargs="+", default=["de"])
    args = parser.parse_args()

    X, y, speakers = load_training_features(args.languages)
    print(f"{len(y)} training words from {sorted(set(speakers.tolist()))}, languages {args.languages}")
    report = {"languages": args.languages, "train_speakers": sorted(set(speakers.tolist())),
              "gamma": "factor / 12850 features", "tie_rule": "within 0.01 of best -> simplest",
              "classifiers": {}}
    for name, (classifier, grid) in GRIDS.items():
        gs = search(classifier, grid, X, y, speakers)
        r = gs.cv_results_
        rows = [{"params": p, "validation": round(float(m), 4), "std": round(float(s), 4)}
                for p, m, s in zip(r["params"], r["mean_test_score"], r["std_test_score"])]
        chosen = tie_rule(rows)
        report["classifiers"][name] = {"rows": rows, "best_params": gs.best_params_,
                                       "best_validation": round(float(gs.best_score_), 4),
                                       "chosen_params": chosen}
        print(f"\n{name}: best {gs.best_params_} {gs.best_score_:.3f} | chosen by tie rule: {chosen}")
        for row in sorted(rows, key=lambda row: -row["validation"]):
            print(f"    {row['validation']:.3f} +/- {row['std']:.3f}  {row['params']}")

    path = config.RESULTS_DIR / f"{prefix(args.languages)}_tuning.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
    