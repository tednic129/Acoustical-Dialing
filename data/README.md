# Data

## Recordings: `data/raw/`

Every recording is committed here, so the whole team trains on the same files.

```
data/raw/
├── english/
│   ├── Laptop_EN/            akshay + maulik, laptop
│   ├── Phone_EN/             akshay (files) + maulik (En_01, En_02, En_03)
│   ├── Tablet_EN/            akshay
│   └── tapan/                tapan
└── german/
    ├── Laptop_DE/            akshay + maulik
    ├── Phone_DE/             akshay (files) + maulik (De_01, DE_02, De_03)
    ├── Tablet_DE/            akshay
    └── tapan/                tapan
```

The folder layout is free (`common/prepare_data.py` searches every subfolder). What matters is the
**file name**:

```
<digit>_<lang>_<speaker>_<take>.wav        e.g. 6_de_maulik_03.wav  or  06_de_tapan_03.wav
```

| Part | Meaning |
|---|---|
| `digit` | 0 to 9, written as `6` or `06` |
| `lang` | `de` or `en` |
| `speaker` | lower-case first name |
| `take` | two-digit take number |

Files with other names are skipped (with a warning). `.m4a` originals may sit next to the `.wav`
files; only `.wav` files are read.

**Format.** WAV, mono preferred. Any sample rate works: everything is resampled to 16 kHz when it is
loaded. Akshay's and Maulik's recordings are 16 kHz; Tapan's are 48 kHz.

### What is here (October 2026)

| Speaker | German | English |
|---|---|---|
| akshay | 100 (laptop 30, phone 40, tablet 30) | 100 |
| maulik | 110 (laptop 80, phone 30) | 110 |
| tapan | 120 (digit 0: 30, digits 1–9: 10 each) | 120 |
| **Per digit** | **31 (digit 0: 51)** | **31 (digit 0: 51)** |

### Adding recordings

1. Name the files as above and put them in a folder under `data/raw/<language>/`
   (your own name or the device, as you like).
2. Commit them on a `data/<topic>` branch and open a pull request.
3. Everyone runs `python -m common.prepare_data` after pulling.

**This repository is public.** Only commit recordings of people who agreed that their voice can be
published.

## Processed arrays: `data/processed/`

Made by

```bash
python -m common.prepare_data
```

which writes `signals.npy`, `events.npy` (the trimmed words), `labels.npy`, `speakers.npy`,
`languages.npy` and `paths.npy`. These files are git-ignored and rebuilt on every laptop; load them
with `common.preprocessing.load_processed()`.

The `akshay_*` and `maulik_*` NPY files were committed earlier by their owners with their own
`wav_to_npy.py` scripts. They are kept exactly as they were.

`prepare_data` also lists recordings worth a listen: words that touch the start or end of the
recording (possibly cut off) and words less than 10 dB above the background noise.
