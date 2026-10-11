"""Show the team the evidence behind the null problem in tapan's test set (German).

    cd ~/Acoustical-Dialing
    python /mnt/c/Users/ASUS/Downloads/null_issue.py

Rebuilds each approach's tuned model from the committed results, fits it on the training speakers
and re-scores tapan's test words exactly as train.py did (the plain accuracy printed first must
equal the committed test_accuracy). Nothing is written; no setting is chosen from these numbers.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from common.preprocessing import load_processed, split_masks  # noqa: E402
from replay_compare import fitted_models  # noqa: E402

data = load_processed(languages=["de"])
_, test = split_masks(data["speakers"])
y = data["labels"][test]
take = np.array([int(str(p).split("_")[-1].split(".")[0]) for p in data["paths"][test]])
length = np.array([len(e) / 16000 for e in data["events"][test]])
null = y == 0
bad = null & (take >= 21)

print("1) Tapan's German test words per digit:", np.bincount(y, minlength=10).tolist())
print(f"   null = {null.sum()} of {len(y)} words ({null.mean():.0%}); every other digit has 10\n")

print(f"2) Length of tapan's nulls: takes 01-20 median {np.median(length[null & ~bad]):.2f} s, "
      f"takes 21-30 median {np.median(length[bad]):.2f} s (training speakers' nulls: about 0.56 s)\n")

print("3) Per approach (tuned model, best on validation):")
print(f"   {'':4} {'committed':>9} {'re-scored':>9} {'balanced':>8} | {'null 01-10':>10} {'11-20':>6} {'21-30':>6} "
      f"| {'without 21-30':>13}")
for approach, (model, result) in fitted_models("de").items():
    pred = model.predict(data["events"][test])
    ok = pred == y
    per_digit = [ok[y == d].mean() for d in range(10)]
    groups = [ok[null & (take >= lo) & (take <= lo + 9)].mean() for lo in (1, 11, 21)]
    print(f"   {approach:4} {result['test_accuracy']:9.3f} {ok.mean():9.3f} {np.mean(per_digit):8.3f} | "
          f"{groups[0]:10.0%} {groups[1]:6.0%} {groups[2]:6.0%} | {ok[~bad].mean():13.3f}")
    wrong = pred[null & ~ok]
    top = {int(d): int(n) for d, n in zip(*np.unique(wrong, return_counts=True))}
    print(f"        null heard as: {top}")

print("\n'without 21-30' is a sensitivity check only; the reported results use all 120 words.")
