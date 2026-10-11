"""Train and test the three 1.b classifiers once, and save the best one for the live demo.

    python -m approach_1b_spectrogram.train            # default settings  -> results/1b_*
    python -m approach_1b_spectrogram.train --tuned    # tuned settings    -> results/1b_tuned_*

Every classifier is one pipeline: spectrogram features -> scaling -> classifier. --tuned applies
the settings tune.py chose on validation (results/1b_tuning.json). Validation accuracy comes from
speaker-wise cross-validation on the training speakers; the test speakers are used once, here.
"""
import argparse
import json
import time

import joblib
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from approach_1b_spectrogram.features import N_VALUES, SHAPE, features_batch
from common import config
from common.evaluation import save_results, speaker_cv
from common.preprocessing import load_processed, split_masks


def build_models(tuned=False):
    """Words in, digits out: three pipelines that differ only in the classifier.

    tuned=True applies the settings tune.py chose on validation (results/1b_tuning.json).
    """
    def pipeline(classifier):
        return make_pipeline(FunctionTransformer(features_batch), StandardScaler(), classifier)

    models = {
        "kNN": pipeline(KNeighborsClassifier(n_neighbors=3)),
        "Linear SVM": pipeline(SVC(kernel="linear", C=1.0)),
        "RBF SVM": pipeline(SVC(kernel="rbf", C=10.0, gamma="scale")),
    }
    if tuned:
        chosen = json.loads((config.RESULTS_DIR / "1b_tuning.json").read_text())["classifiers"]
        for name, model in models.items():
            model.set_params(**chosen[name]["best_params"])
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", nargs="+", default=list(config.LANGUAGES))
    parser.add_argument("--tuned", action="store_true", help="use the settings from results/1b_tuning.json")
    args = parser.parse_args()
    tag = "1b_tuned" if args.tuned else "1b"

    data = load_processed(languages=args.languages)
    train, test = split_masks(data["speakers"])
    events, labels, speakers = data["events"], data["labels"], data["speakers"]
    window_ms = config.WIN_LENGTH * 1000 // config.SAMPLE_RATE
    hop_ms = config.HOP_LENGTH * 1000 // config.SAMPLE_RATE
    print(f"Train: {train.sum()} words {sorted(set(speakers[train].tolist()))} | Test: {test.sum()} words "
          f"{sorted(set(speakers[test].tolist()))} | languages {args.languages} | {tag}")
    print(f"Features: {SHAPE[0]} x {SHAPE[1]} log-spectrogram ({window_ms} ms window, {hop_ms} ms hop) "
          f"-> {N_VALUES:,} values per word\n")

    cv, groups = speaker_cv(speakers[train])
    best_name, best_validation, best_model = None, -1.0, None
    print(f"{'classifier':<12} {'validation':>10} {'test':>7} {'ms/word':>8}")
    for name, model in build_models(args.tuned).items():
        validation = cross_val_score(model, events[train], labels[train], groups=groups, cv=cv).mean()
        model.fit(events[train], labels[train])
        start = time.perf_counter()
        predicted = model.predict(events[test])
        ms_per_word = (time.perf_counter() - start) / max(1, test.sum()) * 1000

        result = save_results(
            tag, name, labels[test], predicted, values_per_word=N_VALUES,
            spectrogram_shape=list(SHAPE), window_ms=window_ms, hop_ms=hop_ms,
            validation_accuracy=round(float(validation), 4), ms_per_word=round(ms_per_word, 2),
            languages=args.languages, test_speakers=sorted(set(speakers[test].tolist())),
        )
        print(f"{name:<12} {validation:10.3f} {result['test_accuracy']:7.3f} {ms_per_word:8.2f}")
        if validation > best_validation:
            best_name, best_validation, best_model = name, validation, model

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, config.MODELS_DIR / f"{tag}_best.joblib")
    print(f"\nBest on validation: {best_name}. Saved models/{tag}_best.joblib and one results/{tag}_*.json per classifier.")


if __name__ == "__main__":
    main()