"""Tune approach 1.a on the training speakers only, following the team's tuning protocol.

    py -m approach_1a_fft_dr.tune
"""
import json

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from approach_1a_fft_dr.features import features_batch
from common import config
from common.evaluation import speaker_cv
from common.preprocessing import load_processed, split_masks

PCA_SIZES = [10, 20, 30, 40]
LDA_SIZE = 9          # the most LDA can give for 10 digits
TIE_MARGIN = 0.01     # settings this close to the best count as a tie (protocol rule 6)
TIE_RULE = (
    "Within 0.01 of the best validation accuracy, take the simplest setting. Protocol order: smaller C, "
    "larger k, then fewer components (LDA counts as 9). Not covered by the protocol, fixed here: "
    "uniform before distance weights, and for the RBF SVM the smaller gamma factor after C."
)


def pca_runs():
    """One (name, pipeline, grid) per classifier, with PCA as the reduction."""
    knn = {
        "pca__n_components": PCA_SIZES,
        "kneighborsclassifier__n_neighbors": [1, 3, 5, 7],
        "kneighborsclassifier__weights": ["uniform", "distance"],
    }
    linear = {
        "pca__n_components": PCA_SIZES,
        "svc__C": [0.0001, 0.001, 0.01, 0.1, 1],
    }
    rbf = [
        {"pca__n_components": [n], "svc__C": [1, 10, 100], "svc__gamma": [f / n for f in (0.1, 1, 10)]}
        for n in PCA_SIZES
    ]
    return [
        ("kNN", make_pipeline(StandardScaler(), PCA(), KNeighborsClassifier()), knn),
        ("Linear SVM", make_pipeline(StandardScaler(), PCA(), SVC(kernel="linear")), linear),
        ("RBF SVM", make_pipeline(StandardScaler(), PCA(), SVC(kernel="rbf")), rbf),
    ]


def lda_runs():
    """The same three classifiers with LDA (9 components) as the reduction: a separate run."""
    lda = LinearDiscriminantAnalysis(n_components=LDA_SIZE)
    knn = {
        "kneighborsclassifier__n_neighbors": [1, 3, 5, 7],
        "kneighborsclassifier__weights": ["uniform", "distance"],
    }
    linear = {"svc__C": [0.0001, 0.001, 0.01, 0.1, 1]}
    rbf = {"svc__C": [1, 10, 100], "svc__gamma": [f / LDA_SIZE for f in (0.1, 1, 10)]}
    return [
        ("kNN", make_pipeline(StandardScaler(), lda, KNeighborsClassifier()), knn),
        ("Linear SVM", make_pipeline(StandardScaler(), lda, SVC(kernel="linear")), linear),
        ("RBF SVM", make_pipeline(StandardScaler(), lda, SVC(kernel="rbf")), rbf),
    ]


def size(row):
    """How many values reach the classifier: the PCA size, or 9 for LDA."""
    return row["params"].get("pca__n_components", LDA_SIZE)


def simplicity(row):
    """Sort key for the tie rule: the smaller the key, the simpler the setting."""
    p = row["params"]
    if row["classifier"] == "kNN":
        return (-p["kneighborsclassifier__n_neighbors"], p["kneighborsclassifier__weights"] != "uniform", size(row))
    if row["classifier"] == "Linear SVM":
        return (p["svc__C"], size(row))
    return (p["svc__C"], round(p["svc__gamma"] * size(row), 3), size(row))


def choose(rows, classifier):
    """The protocol's choice for one classifier: the simplest setting within TIE_MARGIN of the best."""
    candidates = [r for r in rows if r["classifier"] == classifier]
    best = max(r["mean_validation_accuracy"] for r in candidates)
    close = [r for r in candidates if r["mean_validation_accuracy"] >= best - TIE_MARGIN - 1e-9]
    return min(close, key=simplicity)


def main():
    data = load_processed(languages=["de"])
    train, test = split_masks(data["speakers"])

    X = features_batch(data["events"][train], n_bands=128)
    y = data["labels"][train]
    speakers = data["speakers"][train]

    print("X:", X.shape, "  y:", y.shape)
    print("training speakers:", sorted(set(speakers.tolist())))
    assert "tapan" not in speakers, "tapan must never be used for tuning"

    cv, groups = speaker_cv(speakers)
    rows = []
    for reduction, runs in (("PCA", pca_runs()), ("LDA", lda_runs())):
        for name, pipeline, grid in runs:
            search = GridSearchCV(pipeline, grid, cv=cv, scoring="accuracy", refit=True)
            search.fit(X, y, groups=groups)
            results = search.cv_results_
            for params, mean, std in zip(results["params"], results["mean_test_score"], results["std_test_score"]):
                rows.append({
                    "classifier": name,
                    "reduction": reduction,
                    "params": params,
                    "mean_validation_accuracy": round(float(mean), 4),
                    "std_validation_accuracy": round(float(std), 4),
                })
            print(f"{name:<11} {reduction}: {len(search.cv_results_['params']):3d} combinations, "
                  f"best validation {search.best_score_:.3f} with {search.best_params_}")

    chosen = {name: choose(rows, name) for name in ("kNN", "Linear SVM", "RBF SVM")}
    print("\nchosen with the tie rule:")
    for name, row in chosen.items():
        print(f"  {name:<11} {row['reduction']}  validation {row['mean_validation_accuracy']:.3f}  {row['params']}")

    path = config.RESULTS_DIR / "1a_tuning.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "approach": "1a",
            "languages": ["de"],
            "fft_bands": 128,
            "training_speakers": sorted(set(speakers.tolist())),
            "tie_rule": TIE_RULE,
            "chosen": chosen,
            "combinations": rows,
        }, f, indent=1)
    print(f"saved {len(rows)} combinations to {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
