# Acoustical-Dialing
Acoustic pizza-shop dialling via spoken-digit recognition (0–9). Isolated-word classification in Python: WAV→NPY, energy-based event extraction, speaker-independent split, and three pipelines (FFT+DR, spectrograms, compressed spectrograms) across kNN/SVM/RBF, ending in live mic classification. Sensor Signal Processing, RPTU.

**The pizza test:** one wrong digit dials the wrong shop. At 95 % per-digit accuracy a 7-digit
number is right only about 70 % of the time (0.95⁷), so every digit counts.

## Team

| Approach | Owner | Folder | Branch |
|---|---|---|---|
| 1.a FFT + dimensionality reduction | Akshay | `approach_1a_fft_dr/` | `approach/1a` |
| 1.b Spectrograms | Maulik | `approach_1b_spectrogram/` | `approach/1b` |
| 1.c Compressed spectrograms | Tapan | `approach_1c_compressed/` | `approach/1c` |

Everything else is shared and changes through reviewed pull requests.

## Repository structure

```
Acoustical-Dialing/                  CHANGED ON BRANCH          GIT
├── README.md                        common/<topic>             committed
├── requirements.txt                 common/<topic>             committed
├── .gitignore                       common/<topic>             committed
├── split.json                       common/<topic>             committed
├── compare.py                       demo/<topic>               committed
├── data/
│   ├── README.md                    data/<topic>               committed
│   ├── raw/   (all recordings)      data/<topic>               committed
│   └── processed/                   none (generated)           ignored
├── common/                          common/<topic>             committed
│   ├── config.py                    every shared number
│   ├── prepare_data.py              WAV -> NPY -> event extraction
│   ├── events.py                    extract_event (energy + ZCR)
│   ├── preprocessing.py             loading, names, split
│   ├── spectrogram.py               common/spectrogram (Maulik)
│   └── evaluation.py                speaker-wise CV, results JSON
├── approach_1a_fft_dr/              approach/1a (Akshay)       committed
├── approach_1b_spectrogram/         approach/1b (Maulik)       committed
├── approach_1c_compressed/          approach/1c (Tapan)        committed
├── live_demo/dial.py                demo/<topic>               committed
├── models/                          none (generated)           ignored
├── results/                         own prefix, own branch     committed
├── notebooks/                       own prefix, own branch     committed
└── report/                          report/<topic>             committed
```

`approach_1a_fft_dr/` and `approach_1b_spectrogram/` hold Akshay's and Maulik's existing scripts,
moved here unchanged from their `Akshay` and `maulik` branches.

## Quick start

```bash
git clone https://github.com/tednic129/Acoustical-Dialing.git
cd Acoustical-Dialing
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m common.prepare_data          # WAV -> data/processed/*.npy, a few seconds
```

Run every script from the repository root with `python -m`, because all paths are relative to it.

Approach 1.c, for example:

```bash
python -m approach_1c_compressed.sweep                # accuracy vs. compression (training speakers)
python -m approach_1c_compressed.train                # test once, save models/1c_best.joblib
python -m live_demo.dial --approach 1c --target 0631123456 --simulate
python compare.py                                     # final table from results/*.json
```

## Data

660 recordings from three speakers, German and English, at least 31 per digit and language. See
[data/README.md](data/README.md) for the naming convention, the folder layout and how to add more.
The scripts use German by default (`LANGUAGES` in `common/config.py`); pass `--languages de en`
to use both.

## Plugging an approach into the demo and the comparison

Each approach provides two things:

1. **`features(event)`** in its folder's `features.py`: one trimmed word in, one 1-D vector out.
2. **`models/<approach>_best.joblib`**: a scikit-learn pipeline whose first step turns words into
   features (a `FunctionTransformer`), so `model.predict([event])` returns a digit. See
   `approach_1c_compressed/train.py` for the pattern.

Results are saved with `common.evaluation.save_results(...)`, one JSON per classifier, so
`compare.py` can put all approaches in one table.

## Branches and pull requests

- **Nobody pushes to `main`.** Everything arrives through a pull request.
- Work on your own approach branch and edit only your folder and your `results/` / `notebooks/`
  prefix.
- `common/`, `requirements.txt` and `split.json` change only through a `common/<topic>` pull request
  that both others review.
- Merge `main` into your branch weekly and after every `common/` change, then rerun
  `python -m common.prepare_data` if new recordings came in.

```bash
git fetch origin
git switch -c approach/1c origin/main      # once: start your branch from main
git push -u origin approach/1c
git pull origin main                        # weekly: stay in sync
```

## Decisions still open

- **Test speakers** in `split.json` (with three speakers: hold one out). Training scripts refuse
  to run until this is set.
- **Language** for the demo: German only, or German and English.
- **Target number**: the K-town pizza shop the demo dials.
