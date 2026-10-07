# Results

One JSON file per classifier, written with `common.evaluation.save_results(...)`, plus its
confusion matrix as a PNG. Every person only adds files with their own prefix:

| Prefix | Owner | Branch |
|---|---|---|
| `1a_*` | Akshay | `approach/1a` |
| `1b_*` | Maulik | `approach/1b` |
| `1c_*` | Tapan | `approach/1c` |

`python compare.py` reads every result and writes the final table to `comparison.md`.

Fields every result has: `approach`, `classifier`, `test_accuracy`, `per_digit_accuracy`,
`confusion_matrix`, `values_per_word`. Optional fields used by `compare.py`:
`validation_accuracy`, `ms_per_word`, `pizza_test` (for example `"7/10"`).
