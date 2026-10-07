"""Train and test the 1.c model once, and save it for the live demo.

    python -m approach_1c_compressed.train            # k from features.K, German
    python -m approach_1c_compressed.train --k 12 --languages de en

Needs split.json with at least one test speaker. Validation accuracy comes from speaker-wise
cross-validation on the training speakers; the test speakers are used exactly once, here.
Writes results/1c_rbf_svm.json (+ confusion matrix) and models/1c_best.joblib.
"""
import argparse
import time

import joblib
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from approach_1c_compressed.features import K, features_batch
from common import config
from common.evaluation import save_results, speaker_cv
from common.preprocessing import load_processed, split_masks


def build_model(k, c=10.0, gamma="scale"):
    """Words in, digits out: DCT features, scaling and an RBF-SVM in one saved pipeline."""
    return make_pipeline(
        FunctionTransformer(features_batch, kw_args={"k": k}),
        StandardScaler(),
        SVC(kernel="rbf", C=c, gamma=gamma),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=K, help="keep k x k DCT coefficients")
    parser.add_argument("--languages", nargs="+", default=list(config.LANGUAGES))
    parser.add_argument("--C", type=float, default=10.0)
    args = parser.parse_args()

    data = load_processed(languages=args.languages)
    train, test = split_masks(data["speakers"])
    events, labels, speakers = data["events"], data["labels"], data["speakers"]
    print(f"Train: {train.sum()} words {sorted(set(speakers[train].tolist()))} | Test: {test.sum()} words "
          f"{sorted(set(speakers[test].tolist()))} | languages {args.languages} | k = {args.k}")

    model = build_model(args.k, c=args.C)
    cv, groups = speaker_cv(speakers[train])
    validation = cross_val_score(model, events[train], labels[train], groups=groups, cv=cv).mean()
    print(f"Validation accuracy (training speakers): {validation:.3f}")

    model.fit(events[train], labels[train])
    start = time.perf_counter()
    predicted = model.predict(events[test])
    ms_per_word = (time.perf_counter() - start) / max(1, test.sum()) * 1000

    result = save_results(
        "1c", "RBF SVM", labels[test], predicted, values_per_word=args.k * args.k,
        k=args.k, compression_ratio=round((config.N_FFT // 2 + 1) * config.N_FRAMES / (args.k * args.k), 1),
        validation_accuracy=round(float(validation), 4), ms_per_word=round(ms_per_word, 2),
        languages=args.languages, test_speakers=sorted(set(speakers[test].tolist())),
    )
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / "1c_best.joblib")
    print(f"Test accuracy: {result['test_accuracy']:.3f}   per digit: {result['per_digit_accuracy']}")
    print("Saved results/1c_rbf_svm.json, results/1c_rbf_svm_confusion.png and models/1c_best.joblib")


if __name__ == "__main__":
    main()
