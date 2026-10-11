"""Build every model of the demo page from the committed settings, in one command.

    py -3.12 -m live_demo.build_models                         # 1.a, 1.b, 1.c, German and English
    py -3.12 -m live_demo.build_models --approaches 1c --languages en

models/ is not in git, and the tuned run of each train.py refuses to run twice (it checks
results/<tag>_*.json). This script runs the same train.py code of every approach, with the
settings that are already committed, but writes its results to a temporary folder.
Nothing in results/ changes. For each model it prints the rebuilt test accuracy next to the
committed one: they must be equal.

Needs data/processed/ (py -3.12 -m common.prepare_data) and the committed results/*.json.
"""
import argparse
import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

from common import config
from live_demo.web import classifier_name, model_tag

MODULES = {"1a": "approach_1a_fft_dr.train", "1b": "approach_1b_spectrogram.train",
           "1c": "approach_1c_compressed.train"}


def committed(tag, name):
    """The committed results/<tag>_<classifier>.json of a model, or None."""
    slug = name.lower().replace(" ", "_")
    path = config.RESULTS_DIR / f"{tag}_{slug}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def arguments(approach, language, tuned):
    """The train.py command line that gives this model (1.c: k and gamma factor come from its committed result)."""
    args = ["--languages", language]
    if approach == "1c":
        result = committed(model_tag("1c", language, tuned), "RBF SVM")
        if result is None:
            raise SystemExit(f"results/{model_tag('1c', language, tuned)}_rbf_svm.json is missing: git pull first.")
        args += ["--k", str(result["k"]), "--C", str(result.get("C", 10.0))]   # older default result has no C: train.py default
        if result.get("gamma_factor") is not None:
            args += ["--gamma-factor", str(result["gamma_factor"])]
    if tuned:
        args.append("--tuned")
    return args


def build(approach, language, tuned, real_results):
    """Run one train.py with a temporary results folder; return (rebuilt test accuracy, committed test accuracy)."""
    import importlib

    tag = model_tag(approach, language, tuned)
    command = arguments(approach, language, tuned)                  # reads the committed results: do it first
    with tempfile.TemporaryDirectory() as folder:
        for tuning in real_results.glob("*tuning*.json"):          # the settings chosen on validation
            shutil.copy(tuning, folder)
        config.RESULTS_DIR = Path(folder)
        sys.argv = ["train"] + command
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                importlib.import_module(MODULES[approach]).main()
        finally:
            config.RESULTS_DIR = real_results
        # the model that was saved is the one the page loads; find its result in the temporary folder
        from live_demo.web import load_model
        model, _ = load_model(tag)
        name = classifier_name(model)
        slug = name.lower().replace(" ", "_")
        rebuilt = json.loads((Path(folder) / f"{tag}_{slug}.json").read_text(encoding="utf-8"))["test_accuracy"]
    old = committed(tag, name)
    return name, rebuilt, None if old is None else old["test_accuracy"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approaches", nargs="+", default=list(MODULES), choices=list(MODULES))
    parser.add_argument("--languages", nargs="+", default=["de", "en"], choices=["de", "en"])
    args = parser.parse_args()

    real_results = config.RESULTS_DIR
    if not (config.PROCESSED_DIR / "labels.npy").exists():
        raise SystemExit("Run `py -3.12 -m common.prepare_data` first.")
    print(f"{'model':<14} {'classifier':<11} {'rebuilt test':>12} {'committed':>10}")
    problems = 0
    for approach in args.approaches:
        for language in args.languages:
            for tuned in (False, True):
                tag = model_tag(approach, language, tuned)
                try:
                    name, rebuilt, old = build(approach, language, tuned, real_results)
                except SystemExit as stop:
                    print(f"{tag:<14} skipped: {stop}")
                    problems += 1
                    continue
                except FileNotFoundError as error:
                    print(f"{tag:<14} skipped: {error}")
                    problems += 1
                    continue
                state = "equal" if old is not None and abs(rebuilt - old) < 1e-9 else "DIFFERENT"
                problems += state != "equal"
                print(f"{tag:<14} {name:<11} {rebuilt:12.4f} {'' if old is None else format(old, '10.4f')}  {state}")
    print("\nAll models equal the committed results." if not problems else f"\n{problems} model(s) need a look.")
    print("Models are in models/. Start the page: py -3.12 -m live_demo.web")


if __name__ == "__main__":
    main()
