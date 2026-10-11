"""WSL side of the live demo: dial one recorded WAV with the tuned 1.a, 1.b and 1.c models side by side.

    cd ~/Acoustical-Dialing
    python /mnt/c/Users/ASUS/Downloads/replay_compare.py /mnt/c/Users/ASUS/Downloads/dial.wav 0631123456
    python /mnt/c/Users/ASUS/Downloads/replay_compare.py dial_en.wav 0631123456 --language en

Run it from the repository root. Each approach uses its tuned classifier with the best validation
accuracy, rebuilt from the committed tuning results and fitted on the training speakers only
(nothing is tested on tapan, nothing in results/ is written). Every model hears exactly the same
audio through the same word detection as live_demo/dial.py.
"""
import argparse
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.getcwd())                      # import the project from the repo root

import numpy as np  # noqa: E402

from common import config  # noqa: E402
from common.preprocessing import load_audio, load_processed, split_masks  # noqa: E402
from live_demo.dial import BLOCK, dial  # noqa: E402


def best_tuned(prefix):
    """The tuned result with the best validation accuracy for one approach, e.g. prefix '1b_en_tuned'."""
    rows = [json.load(open(p)) for p in config.RESULTS_DIR.glob(f"{prefix}_*.json")]
    rows = [r for r in rows if "validation_accuracy" in r]
    if not rows:
        raise SystemExit(f"No results/{prefix}_*.json: that approach has no tuned run for this language.")
    return max(rows, key=lambda r: r["validation_accuracy"])


def build(approach, language):
    """(model, result) for one approach: the tuned pipeline, not yet fitted."""
    sfx = "" if language == "de" else f"_{language}"
    result = best_tuned(f"{approach}{sfx}_tuned")
    name = result["classifier"]
    if approach == "1a":
        from approach_1a_fft_dr.train import build_models
        comps = result["settings"].get("pca__n_components", 9)
        model = build_models(result["reduction"], comps, result["fft_bands"])[name]
        model.set_params(**result["settings"])
    elif approach == "1b":
        from approach_1b_spectrogram.train import build_models
        model = build_models(tuned=True, languages=[language])[name]
    else:
        from approach_1c_compressed.train import build_model
        k = result["k"]
        model = build_model(k, c=result["C"], gamma=result["gamma_factor"] / (k * k))
    return model, result


def fitted_models(language):
    """The three tuned models, fitted on the training speakers of one language."""
    data = load_processed(languages=[language])
    train, _ = split_masks(data["speakers"])
    models = {}
    for approach in ("1a", "1b", "1c"):
        model, result = build(approach, language)
        model.fit(data["events"][train], data["labels"][train])
        models[approach] = (model, result)
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav")
    parser.add_argument("target", help="the number that was read out, digits only")
    parser.add_argument("--language", default="de", choices=["de", "en"])
    args = parser.parse_args()

    x, _ = load_audio(args.wav)
    noise = float(np.sqrt(np.mean(x[: config.SAMPLE_RATE] ** 2))) + 1e-6   # the first second is silence
    print(f"Recording {args.wav}: {len(x) / config.SAMPLE_RATE:.1f} s | target {args.target} | {args.language}\n")

    dialled = {}
    for approach, (model, result) in fitted_models(args.language).items():
        blocks = (x[i:i + BLOCK] for i in range(0, len(x) - BLOCK + 1, BLOCK))
        with contextlib.redirect_stdout(io.StringIO()):                       # hide dial()'s own prints
            dialled[approach] = dial(model, args.target, noise, blocks)
        dialled[approach + "_info"] = result

    target = args.target
    print(f"{'#':>2}  said  " + "  ".join(f"{a:>4}" for a in ("1a", "1b", "1c")))
    for i, digit in enumerate(target):
        cells = []
        for a in ("1a", "1b", "1c"):
            got = dialled[a][i] if i < len(dialled[a]) else "-"
            cells.append(f"{got:>2} {'ok' if got == digit else 'X '}")
        print(f"{i + 1:>2}  {digit:>4}  " + "  ".join(cells))

    print()
    for a in ("1a", "1b", "1c"):
        got, info = dialled[a], dialled[a + "_info"]
        hits = sum(g == t for g, t in zip(got, target))
        p = info["test_accuracy"]
        missed = f", {len(target) - len(got)} word(s) not detected" if len(got) < len(target) else ""
        print(f"{a} ({info['classifier']:<10}) dialled {got:<{len(target)}}  {hits}/{len(target)} digits  "
              f"{'PIZZA!' if got == target else 'wrong number'}{missed}   "
              f"[expected from test: {p:.2f}^{len(target)} = {p ** len(target):.0%}]")


if __name__ == "__main__":
    main()
