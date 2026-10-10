"""Train and test the three 1.a classifiers once, and save the best one for the live demo.

    python -m approach_1a_fft_dr.train                       # PCA with 30 components, German
    python -m approach_1a_fft_dr.train --dr lda
    python -m approach_1a_fft_dr.train --components 20 --languages de en
    python -m approach_1a_fft_dr.train --tuned --validate-only   # tuned settings, validation only
    python -m approach_1a_fft_dr.train --tuned                   # the one final run, tag 1a_tuned

Every classifier is one pipeline: FFT features -> scaling -> dimensionality reduction -> classifier.
Because scaling and the reduction are steps of the pipeline, they are fitted on training words only.

Needs split.json with at least one test speaker. Validation accuracy comes from speaker-wise
cross-validation on the training speakers; the test speakers are used exactly once, here.
Writes results/1a_<classifier>.json (+ confusion matrix) for each classifier and
models/1a_best.joblib (the classifier with the best validation accuracy).

With --tuned, each classifier uses the settings tune.py chose (results/1a_tuning.json), and
everything is saved under the tag 1a_tuned, so the default results stay untouched.
"""
import argparse
import json
import time

import joblib
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from approach_1a_fft_dr.features import N_BANDS, features_batch
from common import config
from common.evaluation import save_results, speaker_cv
from common.preprocessing import load_processed, split_masks


def reduction(method, components):
    """The dimensionality-reduction step: PCA, or LDA (at most 9 components for 10 digits)."""
    if method == "lda":
        return LinearDiscriminantAnalysis(n_components=min(components, 9))
    return PCA(n_components=components)


def build_models(method="pca", components=30, n_bands=N_BANDS):
    """Words in, digits out: three pipelines that differ only in the classifier."""
    def pipeline(classifier):
        return make_pipeline(
            FunctionTransformer(features_batch, kw_args={"n_bands": n_bands}),
            StandardScaler(),
            reduction(method, components),
            classifier,
        )

    return {
        "kNN": pipeline(KNeighborsClassifier(n_neighbors=3)),
        "Linear SVM": pipeline(SVC(kernel="linear", C=1.0)),
        "RBF SVM": pipeline(SVC(kernel="rbf", C=10.0, gamma="scale")),
    }


TUNING_FILE = config.RESULTS_DIR / "1a_tuning.json"


def load_chosen():
    """The settings tune.py chose for each classifier, on the training speakers only."""
    with open(TUNING_FILE, encoding="utf-8") as f:
        return json.load(f)["chosen"]


def build_tuned_models(chosen, n_bands=N_BANDS):
    """The same three pipelines as build_models, each switched to its chosen settings."""
    models = {}
    for name, row in chosen.items():
        method = "lda" if row["reduction"] == "LDA" else "pca"
        model = build_models(method, row["params"].get("pca__n_components", 9), n_bands)[name]
        model.set_params(**row["params"])
        models[name] = model
    return models


def values_kept(model):
    """How many values per word reach the classifier: the PCA size, or 9 for LDA."""
    if "pca" in model.named_steps:
        return model.named_steps["pca"].n_components
    return 9


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dr", choices=["pca", "lda"], default="pca", help="dimensionality reduction")
    parser.add_argument("--components", type=int, default=30, help="values kept after the reduction")
    parser.add_argument("--bands", type=int, default=N_BANDS, help="frequency bands of the FFT spectrum")
    parser.add_argument("--languages", nargs="+", default=list(config.LANGUAGES))
    parser.add_argument("--tuned", action="store_true", help="use the settings from results/1a_tuning.json")
    parser.add_argument("--validate-only", action="store_true", help="print validation accuracy, don't test")
    args = parser.parse_args()
    tag = "1a_tuned" if args.tuned else "1a"
    if args.tuned and not args.validate_only and list(config.RESULTS_DIR.glob("1a_tuned_*.json")):
        raise SystemExit("results/1a_tuned_*.json already exist: the tuned test run happens only once.")

    data = load_processed(languages=args.languages)
    train, test = split_masks(data["speakers"])
    events, labels, speakers = data["events"], data["labels"], data["speakers"]
    print(f"Train: {train.sum()} words {sorted(set(speakers[train].tolist()))} | Test: {test.sum()} words "
          f"{sorted(set(speakers[test].tolist()))} | languages {args.languages}")
    if args.tuned:
        chosen = load_chosen()
        models = build_tuned_models(chosen, args.bands)
        print(f"Features: {args.bands} FFT bands, settings per classifier from {TUNING_FILE.name}, tag {tag}\n")
    else:
        models = build_models(args.dr, args.components, args.bands)
        kept = min(args.components, 9) if args.dr == "lda" else args.components
        print(f"Features: {args.bands} FFT bands -> {args.dr.upper()} -> {kept} values per word\n")

    cv, groups = speaker_cv(speakers[train])
    best_name, best_validation, best_model = None, -1.0, None
    print(f"{'classifier':<12} {'validation':>10} {'test':>7} {'ms/word':>8}")
    for name, model in models.items():
        validation = cross_val_score(model, events[train], labels[train], groups=groups, cv=cv).mean()
        if args.validate_only:
            print(f"{name:<12} {validation:10.3f}    (not tested)   values per word: {values_kept(model)}")
            continue
        model.fit(events[train], labels[train])
        start = time.perf_counter()
        predicted = model.predict(events[test])
        ms_per_word = (time.perf_counter() - start) / max(1, test.sum()) * 1000

        extra = {"settings": chosen[name]["params"]} if args.tuned else {}
        result = save_results(
            tag, name, labels[test], predicted, values_per_word=values_kept(model),
            reduction="pca" if "pca" in model.named_steps else "lda", fft_bands=args.bands, **extra,
            validation_accuracy=round(float(validation), 4), ms_per_word=round(ms_per_word, 2),
            languages=args.languages, test_speakers=sorted(set(speakers[test].tolist())),
        )
        print(f"{name:<12} {validation:10.3f} {result['test_accuracy']:7.3f} {ms_per_word:8.2f}")
        if validation > best_validation:
            best_name, best_validation, best_model = name, validation, model

    if args.validate_only:
        print("\nValidation only: nothing was tested or saved.")
        return
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, config.MODELS_DIR / f"{tag}_best.joblib")
    print(f"\nBest on validation: {best_name}. Saved models/{tag}_best.joblib and one results/{tag}_*.json per classifier.")


if __name__ == "__main__":
    main()
