"""WAV -> NPY -> event extraction, in one command.

    python -m common.prepare_data

Reads every recording under data/raw/ (any folder depth), resamples it to 16 kHz mono, cuts out
the spoken word and saves these arrays to data/processed/ (git-ignored, rebuilt on every laptop):

    signals.npy    full recordings (object array, 16 kHz float32)
    events.npy     the spoken word only (object array)
    labels.npy     digit 0-9
    speakers.npy   e.g. "maulik"
    languages.npy  "de" or "en"
    paths.npy      file path relative to data/raw/

Rerun it whenever new recordings arrive. All languages are saved; each approach picks its own
with common.preprocessing.load_processed(languages=...).
"""
from collections import Counter

import numpy as np

from common import config
from common.events import find_event, frame_features
from common.preprocessing import load_audio, parse_name


def word_to_noise_db(signal, start, end):
    """How much louder the word is than the rest of the recording, in dB (None if there is no rest)."""
    energy, _, _, hop = frame_features(signal)
    inside = energy[start // hop: max(start // hop + 1, end // hop)]
    outside = np.concatenate([energy[: start // hop], energy[end // hop:]])
    if outside.size < 5:
        return None
    return 20 * np.log10(np.median(inside) / (np.median(outside) + 1e-10))


def main():
    paths = sorted(p for p in config.RAW_DIR.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not paths:
        raise SystemExit(f"No .wav files found under {config.RAW_DIR}")

    rows, skipped, no_event, rates = [], [], [], Counter()
    at_edge, noisy = [], []
    for path in paths:
        info = parse_name(path)
        if info is None:
            skipped.append(path)
            continue
        signal, rate = load_audio(path)
        rates[rate] += 1
        bounds = find_event(signal)
        if bounds is None:
            no_event.append(path)
            continue
        start, end = bounds
        rel = path.relative_to(config.RAW_DIR).as_posix()
        if start == 0 or end >= len(signal) - 1:
            at_edge.append(rel)
        snr = word_to_noise_db(signal, start, end)
        if snr is not None and snr < 10:
            noisy.append((snr, rel))
        rows.append((signal, signal[start:end].copy(), info, rel))

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {
        "signals": np.array([r[0] for r in rows] + [None], dtype=object)[:-1],
        "events": np.array([r[1] for r in rows] + [None], dtype=object)[:-1],
        "labels": np.array([r[2]["digit"] for r in rows], dtype=np.int8),
        "speakers": np.array([r[2]["speaker"] for r in rows]),
        "languages": np.array([r[2]["language"] for r in rows]),
        "paths": np.array([r[3] for r in rows]),
    }
    for name, array in arrays.items():
        np.save(config.PROCESSED_DIR / f"{name}.npy", array)

    print(f"Saved {len(rows)} recordings to {config.PROCESSED_DIR.relative_to(config.ROOT)}/")
    print("Sample rates read (all resampled to 16 kHz): " + ", ".join(f"{r} Hz x{n}" for r, n in sorted(rates.items())))
    counts = Counter(zip(arrays["speakers"], arrays["languages"]))
    print("\nRecordings per speaker and language:")
    for (speaker, language), n in sorted(counts.items()):
        print(f"  {speaker:<10} {language}  {n:4d}")
    print("\nRecordings per digit and language:")
    for language in sorted(set(arrays["languages"])):
        per_digit = Counter(arrays["labels"][arrays["languages"] == language].tolist())
        print(f"  {language}: " + "  ".join(f"{d}:{per_digit.get(d, 0)}" for d in range(10)))
    lengths = np.array([len(e) for e in arrays["events"]]) / config.SAMPLE_RATE
    print(f"\nWord length: median {np.median(lengths):.2f} s, 95th percentile {np.percentile(lengths, 95):.2f} s, "
          f"longest {lengths.max():.2f} s")
    if skipped:
        print(f"\nSkipped {len(skipped)} file(s) whose names don't follow <digit>_<lang>_<speaker>_<take>.wav:")
        for p in skipped[:10]:
            print("  ", p.relative_to(config.ROOT))
    if no_event:
        print(f"\nNo speech found in {len(no_event)} file(s); listen to them:")
        for p in no_event[:10]:
            print("  ", p.relative_to(config.ROOT))
    if at_edge:
        print(f"\nWord reaches the start or end of the recording in {len(at_edge)} file(s); it may be cut off:")
        for rel in at_edge:
            print("   data/raw/" + rel)
    if noisy:
        print(f"\nWord is less than 10 dB louder than the background in {len(noisy)} file(s):")
        for snr, rel in sorted(noisy):
            print(f"   {snr:4.1f} dB  data/raw/{rel}")


if __name__ == "__main__":
    main()
