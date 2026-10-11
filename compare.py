"""Build the final comparison table from every results/*.json file.

    python compare.py

Each approach writes one JSON per classifier with common.evaluation.save_results. This script
prints them side by side and saves the table to results/comparison.md for the report.
"""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
COLUMNS = [
    ("Approach", "approach", "{}"),
    ("Classifier", "classifier", "{}"),
    ("Test accuracy", "test_accuracy", "{:.1%}"),
    ("Balanced test accuracy", "balanced_accuracy", "{:.1%}"),   # NEW
    ("Validation accuracy", "validation_accuracy", "{:.1%}"),
    ("Values per word", "values_per_word", "{:,}"),
    ("ms per word", "ms_per_word", "{:.1f}"),
    ("Pizza test", "pizza_test", "{}"),
]


def balanced_accuracy(confusion_matrix):                          # NEW
    """Mean of the per-digit accuracies, so every digit counts equally however many test words it has."""
    per_digit = [row[i] / sum(row) for i, row in enumerate(confusion_matrix) if sum(row) > 0]
    return sum(per_digit) / len(per_digit)


def main():
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
        if "test_accuracy" in result:          # skips other JSON files such as 1c_sweep.json
            result["balanced_accuracy"] = balanced_accuracy(result["confusion_matrix"])   # NEW
            rows.append(result)
    if not rows:
        raise SystemExit("No results yet. Each approach saves them with common.evaluation.save_results.")

    rows.sort(key=lambda r: (r["approach"], -r["test_accuracy"]))
    header = "| " + " | ".join(name for name, _, _ in COLUMNS) + " |"
    lines = [header, "|" + "---|" * len(COLUMNS)]
    for r in rows:
        cells = [fmt.format(r[key]) if r.get(key) is not None else "" for _, key, fmt in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    (RESULTS / "comparison.md").write_text("# Comparison of approaches\n\n" + table + "\n", encoding="utf-8")
    print("\nSaved results/comparison.md")


if __name__ == "__main__":
    main()