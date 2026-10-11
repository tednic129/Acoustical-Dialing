"""A log for every demo run: what worked, what failed and why.

live_demo.web writes these files to logs/demo/ (or to the folder of --log-dir) when a run ends:

    <date>_<time>_<language>_<source>_<voice>.md      the report (read this)
    <date>_<time>_<language>_<source>_<voice>.json    all data of the run
    summary.md                                        all runs, and the pizza-test counters

The report checks the run against what the brief expects (SSP_Pizza_Dialling_Team_Brief, June 2026):
the live loop, the dialled number, the pizza test, a new speaker, the 1.c size-versus-accuracy
result. For a failed pizza test it says why: a word that the detector lost (too quiet, merged with
its neighbour, refused by the event extraction), an extra sound, or a wrong digit (with the
confusion rate that the model had on the test set).

The recorder only watches the listening loop. It never changes a decision of the loop.
To join the logs of several computers:  py -3.12 -m live_demo.run_log logs/official
Standard library and numpy only.
"""
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from common import config
from live_demo.dial import BLOCK, LOUDNESS, QUIET_BLOCKS

LOG_VERSION = 1
BLOCK_S = BLOCK / config.SAMPLE_RATE
APPROACH_NAMES = {"1a": "1.a FFT + DR", "1b": "1.b Spectrogram", "1c": "1.c Compressed spectrogram"}
DIGIT_WORDS = {
    "de": ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"],
    "en": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
}
LANGUAGE_NAMES = {"de": "German", "en": "English"}
EVENT_KEYS = ("FRAME_MS", "HOP_MS", "PAD_MS", "NOISE_PERCENTILE", "ENERGY_FACTOR", "PEAK_FRACTION", "MAX_GAP_MS",
              "MAX_WORD_MS", "ZCR_THRESHOLD", "ZCR_ENERGY_FACTOR", "HISS_EXTEND_MS")
SOURCE_NAMES = {"microphone": "microphone", "simulation": "recordings (simulation)"}
VOICE_NAMES = {"new": "new voice", "training": "training voice"}


DEFAULT_LOG_DIR = Path("logs") / "demo"
_log_dir = DEFAULT_LOG_DIR


def set_log_dir(path):
    """Choose the folder for the logs (relative to the repo, or absolute). Start of the server only."""
    global _log_dir
    _log_dir = Path(path)


def log_dir():
    folder = _log_dir if _log_dir.is_absolute() else config.ROOT / _log_dir
    return folder


def shown(path):
    """Path for the status line: relative to the repo when the file is inside it."""
    try:
        return Path(path).relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)


def label(model_id):
    """'1c_tuned' -> '1.c tuned'."""
    approach, settings = model_id.split("_")
    return f"{approach[0]}.{approach[1]} {settings}"


def say(digit, language):
    return f"{digit} ({DIGIT_WORDS[language][digit]})"


def pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f} %"


# ---------------------------------------------------------------- recording

class Recorder:
    """Collects what the listening loop sees: one level per 0.1 s block, every word, every refused sound."""

    def __init__(self, language, target, source, voice, speaker, noise, models):
        self.started = datetime.now().astimezone()
        self.language, self.target, self.source, self.voice, self.speaker = language, target, source, voice, speaker
        self.noise = float(noise)
        self.models = models          # model id -> info dict (from web.load_model), in display order
        self.levels = []              # dB above the room noise, one per block
        self.words = []               # words that the event extraction accepted
        self.rejected = []            # loud sounds in which the event extraction found no word
        self.device = None

    @property
    def now(self):
        """Index of the block that was added last."""
        return len(self.levels) - 1

    def block(self, db):
        self.levels.append(float(db))

    def _span(self, first, last, n_blocks):
        return {"first": int(first), "last": int(last), "start_s": round(first * BLOCK_S, 2),
                "end_s": round((last + 1) * BLOCK_S, 2), "segment_s": round(n_blocks * BLOCK_S, 2),
                "peak_db": round(max(self.levels[first:last + 1]), 1)}

    def word(self, first, last, n_blocks, length_s, answers, predict_ms):
        span = self._span(first, last, n_blocks)
        span.update(index=len(self.words) + 1, length_s=round(float(length_s), 2), answers=dict(answers),
                    predict_ms={k: round(v, 2) for k, v in predict_ms.items()})
        self.words.append(span)

    def refuse(self, first, last, n_blocks):
        self.rejected.append(self._span(first, last, n_blocks))


# ---------------------------------------------------------------- analysis

def align(target, words, ids):
    """Match the detected words to the target digits, in order.

    The cost of a match is the number of models that disagree with the target digit. A missed
    target digit or an extra word costs 0.6 x the number of models: a plain wrong digit is a
    substitution, a shifted sequence is a missed or extra word. Returns a list of
    {"kind": "match" | "missed" | "extra", "target": j or None, "word": i or None}.
    """
    n, m = len(words), len(target)
    gap = 0.6 * len(ids) + 0.01
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    move = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            options = []                                  # on a tie, a missed digit wins over a match: a missing word is placed late
            if j > 0:
                options.append(("missed", cost[i][j - 1] + gap))
            if i > 0 and j > 0:
                options.append(("match", cost[i - 1][j - 1] + sum(words[i - 1]["answers"].get(k) != int(target[j - 1]) for k in ids)))
            if i > 0:
                options.append(("extra", cost[i - 1][j] + gap))
            best = min(c for _, c in options)
            step = next(name for name, c in options if c <= best + 1e-9)
            cost[i][j], move[i][j] = best, step
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        step = move[i][j]
        if step == "match":
            pairs.append({"kind": "match", "target": j - 1, "word": i - 1})
            i, j = i - 1, j - 1
        elif step == "missed":
            pairs.append({"kind": "missed", "target": j - 1, "word": None})
            j -= 1
        else:
            pairs.append({"kind": "extra", "target": None, "word": i - 1})
            i -= 1
    return pairs[::-1]


def missed_groups(pairs):
    """Runs of target digits that no word matched: [(first index, last index)] in the alignment."""
    groups, k = [], 0
    while k < len(pairs):
        if pairs[k]["kind"] == "missed":
            end = k
            while end + 1 < len(pairs) and pairs[end + 1]["kind"] == "missed":
                end += 1
            groups.append((k, end))
            k = end + 1
        else:
            k += 1
    return groups


def explain_missed(rec, pairs, group, typical_gap, usual_extra, threshold_db, floor_db, status, n_extra):
    """Why did the loop not hear these target digits? Only recorded facts are used."""
    first, last = group
    before = next((p["word"] for p in reversed(pairs[:first]) if p["word"] is not None), None)
    after = next((p["word"] for p in pairs[last + 1:] if p["word"] is not None), None)
    # blocks that belong to no word: after the quiet tail of the word before, up to the lead block of the word after
    start = rec.words[before]["last"] + 1 + QUIET_BLOCKS if before is not None else 0
    stop = rec.words[after]["first"] - 2 if after is not None else len(rec.levels) - 1
    refused = [r for r in rec.rejected if r["first"] >= start and r["last"] <= stop]
    covered = {i for r in refused for i in range(r["first"], r["last"] + 1)}
    free = [rec.levels[i] for i in range(start, stop + 1) if i not in covered]
    peak = max(free) if free else None
    if before is not None and after is not None:
        gap_s = (rec.words[after]["first"] - rec.words[before]["last"] - 1) * BLOCK_S
    else:
        gap_s = len(free) * BLOCK_S
    at_edge = before is None or after is None
    room = at_edge or typical_gap is None or gap_s >= 1.5 * typical_gap
    long_neighbours = [i for i in (before, after) if i is not None
                       and rec.words[i]["segment_s"] - rec.words[i]["length_s"] > usual_extra + 0.5]
    bump = None if peak is None or floor_db is None else peak - floor_db
    n_words = len(rec.words)
    if refused:
        cause = "refused"
        biggest = max(refused, key=lambda r: r["peak_db"])
        detail = (f"A loud sound lasted {biggest['segment_s']:.1f} s (peak {biggest['peak_db']:.1f} dB). "
                  "The event extraction found no word in it, so the loop dropped it.")
    elif after is None and status == "error":
        cause, detail = "run ended by an error", "The run stopped with an error before these digits."
    elif after is None and status == "complete" and n_extra:
        cause = "cut off by an extra sound"
        detail = (f"The loop stops when it has heard as many words as the target has digits ({len(rec.target)}). "
                  f"{n_extra} extra sound(s) counted as words, so the loop did not listen to the last digit(s).")
    elif room and bump is not None and bump >= 3 and peak < threshold_db:
        cause = "too quiet"
        detail = (f"A sound is in the gap, but the loudest block reached {peak:.1f} dB. The threshold is "
                  f"{threshold_db:.1f} dB ({threshold_db - peak:.1f} dB more). The word was too quiet for the threshold.")
    elif after is None and status == "stopped":
        cause = "run stopped"
        detail = f"The run was stopped after word {n_words}."
        if peak is not None and peak >= threshold_db:
            detail += f" A word was still in progress (loudest block {peak:.1f} dB)."
    elif room and bump is not None and bump < 3:
        cause = "no sound"
        detail = (f"The gap lasted {gap_s:.1f} s and has no sound above the room level (loudest block {peak:.1f} dB). "
                  "The log cannot tell if the word was spoken.")
    elif not room or long_neighbours:
        cause = "merged"
        parts = []
        if not room:
            parts.append(f"The gap ({gap_s:.1f} s) is not longer than the usual gap ({typical_gap:.1f} s): there was no room "
                         "for a separate word.")
        for i in long_neighbours:
            w = rec.words[i]
            parts.append(f"Word {w['index']} has a segment of {w['segment_s']:.1f} s but its event is only {w['length_s']:.1f} s: "
                         "the missing word is probably inside this segment (pause shorter than "
                         f"{QUIET_BLOCKS * BLOCK_S:.1f} s).")
        detail = " ".join(parts)
    else:
        cause, detail = "unknown", "The recorded levels do not show the reason."
    positions = [pairs[k]["target"] + 1 for k in range(first, last + 1)]
    return {"target_positions": positions, "expected_digits": [int(rec.target[j - 1]) for j in positions],
            "after_word": None if before is None else before + 1, "before_word": None if after is None else after + 1,
            "gap_s": round(gap_s, 2), "typical_gap_s": None if typical_gap is None else round(typical_gap, 2),
            "loudest_free_block_db": None if peak is None else round(peak, 1), "threshold_db": round(threshold_db, 1),
            "refused_sounds": refused, "likely_cause": cause, "detail": detail}


def explain_extra(rec, index, reference):
    w = rec.words[index]
    shortest = reference["min"] if reference else None
    if shortest is not None and w["length_s"] < 0.8 * shortest:
        cause = "short noise"
        detail = (f"A sound of {w['length_s']:.2f} s (peak {w['peak_db']:.1f} dB) was taken as a word. "
                  f"The shortest digit in the data is {shortest:.2f} s: probably a click, a cough or a breath.")
    else:
        cause = "extra sound or word"
        detail = (f"A sound of {w['length_s']:.2f} s (peak {w['peak_db']:.1f} dB) was taken as a word, "
                  "but no target digit fits it: an extra word, or a noise that looks like speech.")
    return {"word": w["index"], "start_s": w["start_s"], "length_s": w["length_s"], "peak_db": w["peak_db"],
            "likely_cause": cause, "detail": detail}


def file_hash(path):
    """First 12 characters of the SHA-256 of a file (cached by size and time)."""
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in file_hash.cache:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        file_hash.cache[key] = digest.hexdigest()[:12]
    return file_hash.cache[key]


file_hash.cache = {}


def _git(*args):
    try:
        done = subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, timeout=5)
        return done.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment():
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = None
    return {"commit": _git("rev-parse", "--short", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "python": sys.version.split()[0], "scikit_learn": sklearn_version, "numpy": np.__version__,
            "platform": platform.platform()}


def model_records(rec):
    """Facts about every model of the run: files, hashes and the test results that the page shows."""
    records = {}
    for model_id, info in rec.models.items():
        record = {"label": label(model_id), "tag": info.get("tag"), "classifier": info.get("classifier"),
                  "k": info.get("k"), "values_per_word": info.get("values"), "compression_ratio": info.get("ratio"),
                  "validation_accuracy": info.get("validation"), "test_accuracy": info.get("test"),
                  "n_test_words": info.get("n_test"), "model_file": info.get("model_file"),
                  "result_file": info.get("result_file")}
        model_path = config.ROOT / info["model_file"]
        record["model_sha256_12"] = file_hash(model_path) if model_path.exists() else None
        result_path = config.ROOT / info["result_file"] if info.get("result_file") else None
        if result_path and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            record["per_digit_test_accuracy"] = result.get("per_digit_accuracy")
            record["test_confusion_matrix"] = result.get("confusion_matrix")
        records[model_id] = record
    return records


def build(rec, status, message, split):
    """The complete log of a run as one dictionary (also the content of the .json file)."""
    ended = datetime.now().astimezone()
    language, target = rec.language, rec.target
    ids = list(rec.models)
    words, models = rec.words, model_records(rec)
    threshold_db = 20 * np.log10(LOUDNESS)
    covered = {i for s in rec.words + rec.rejected for i in range(s["first"], s["last"] + 1)}
    outside = [v for i, v in enumerate(rec.levels) if i not in covered]
    floor_db = round(float(np.median(outside)), 1) if outside else None
    reference = (split or {}).get("word_length_s")

    pairs = align(target, words, ids) if words else [{"kind": "missed", "target": j, "word": None} for j in range(len(target))]
    gaps = []
    for a, b in zip(pairs, pairs[1:]):
        if a["word"] is not None and b["word"] is not None:
            gaps.append((words[b["word"]]["first"] - words[a["word"]]["last"] - 1) * BLOCK_S)
    typical_gap = statistics.median(gaps) if gaps else None
    usual_extra = statistics.median([w["segment_s"] - w["length_s"] for w in words]) if words else 0.0

    extra = [explain_extra(rec, p["word"], reference) for p in pairs if p["kind"] == "extra"]
    missed = [explain_missed(rec, pairs, g, typical_gap, usual_extra, threshold_db, floor_db, status, len(extra))
              for g in missed_groups(pairs)] if words else []

    per_model = {}
    for mid in ids:
        info, dialled = models[mid], "".join(str(w["answers"][mid]) for w in words)
        confusion = info.get("test_confusion_matrix")
        right, subs = 0, []
        for p in pairs:
            if p["kind"] != "match":
                continue
            expected, heard = int(target[p["target"]]), words[p["word"]]["answers"][mid]
            if heard == expected:
                right += 1
                continue
            w = words[p["word"]]
            sub = {"position": p["target"] + 1, "expected": expected, "heard": heard, "word": p["word"] + 1,
                   "length_s": w["length_s"], "peak_db": w["peak_db"], "notes": []}
            if w["peak_db"] < threshold_db + 6:
                sub["notes"].append(f"weak word: peak only {w['peak_db'] - threshold_db:.1f} dB above the threshold, "
                                    "so the loop may have cut the start or the end")
            if reference and w["length_s"] < reference["min"]:
                sub["notes"].append(f"short word: {w['length_s']} s; the shortest digit in the data is {reference['min']} s")
            if reference and w["length_s"] > reference["max"]:
                sub["notes"].append(f"long word: {w['length_s']} s; the longest digit in the data is {reference['max']} s")
            if confusion:
                support = sum(confusion[expected])
                sub["test_confusions"] = confusion[expected][heard]
                sub["test_words_of_digit"] = support
                sub["test_accuracy_of_digit"] = (info["per_digit_test_accuracy"] or [None] * 10)[expected]
            subs.append(sub)
        matched = sum(p["kind"] == "match" for p in pairs)
        accuracy = info.get("test_accuracy")
        per_model[mid] = {
            "dialled": dialled, "pizza_passed": dialled == target, "digits_right": right, "digits_wrong": len(subs),
            "digits_missed": sum(len(m["target_positions"]) for m in missed), "extra_words": len(extra), "right_of_detected": [right, matched],
            "substitutions": subs,
            "chance_of_correct_number": None if accuracy is None else round(accuracy ** len(target), 4)}

    rec_words = []
    by_word = {p["word"]: p for p in pairs if p["word"] is not None}
    for w in words:
        p = by_word.get(w["index"] - 1)
        rec_words.append({**w, "target_position": None if p is None or p["target"] is None else p["target"] + 1,
                          "expected": None if p is None or p["target"] is None else int(target[p["target"]])})
    fixed = {"block_s": BLOCK_S, "quiet_blocks": QUIET_BLOCKS, "loudness_factor": LOUDNESS, "sample_rate": config.SAMPLE_RATE,
             **{k.lower(): getattr(config, k) for k in EVENT_KEYS}}
    settings_id = hashlib.sha1(json.dumps(fixed, sort_keys=True).encode()).hexdigest()[:8]
    models_id = hashlib.sha1(json.dumps({k: v["model_sha256_12"] for k, v in models.items()}, sort_keys=True).encode()).hexdigest()[:8]
    result = {
        "log_version": LOG_VERSION,
        "run": {"id": f"{rec.started:%Y-%m-%d_%H%M%S}_{language}_{rec.source}_{rec.voice}",
                "started": rec.started.isoformat(timespec="seconds"), "ended": ended.isoformat(timespec="seconds"),
                "duration_s": round(len(rec.levels) * BLOCK_S, 1), "status": status, "message": message,
                "language": language, "source": rec.source, "voice": rec.voice, "speaker": rec.speaker,
                "input_device": rec.device, "target": target, "words_expected": len(target), "words_detected": len(words)},
        "environment": environment(),
        "settings": {"settings_id": settings_id, "models_id": models_id, **fixed, "threshold_db": round(threshold_db, 1),
                     "calibration_noise_rms": round(rec.noise, 6), "threshold_rms": round(LOUDNESS * rec.noise, 6),
                     "room_during_run_db": floor_db},
        "models": models, "split": split, "words": rec_words, "refused_sounds": rec.rejected, "alignment": pairs,
        "missed": missed, "extra": extra, "per_model": per_model, "levels_db": [round(v, 1) for v in rec.levels],
        "typical_gap_s": None if typical_gap is None else round(typical_gap, 2),
    }
    result["checks"] = checks(result)
    return result


def checks(r):
    """The expected outcomes of the brief, one row each: PASS, FAIL or INFO, with the reason."""
    run, per_model, models = r["run"], r["per_model"], r["models"]
    n_target, n_words = run["words_expected"], run["words_detected"]
    ids = list(models)
    rows = []

    def add(ref, expected, status, what):
        rows.append({"id": f"C{len(rows) + 1}", "brief": ref, "expected": expected, "status": status, "what": what})

    # C1 the live loop
    peak = max(r["levels_db"]) if r["levels_db"] else None
    if run["status"] == "error":
        add("§2", "The live loop runs from audio to dialled number.", "FAIL", f"The loop stopped with an error: {run['message']}")
    elif n_words == 0:
        add("§2", "The live loop runs from audio to dialled number.", "FAIL",
            f"No word was detected. The loudest block was {peak} dB; the threshold is {r['settings']['threshold_db']} dB.")
    else:
        add("§2", "The live loop runs from audio to dialled number.", "PASS",
            f"Audio → event detection → features → classification → digit shown → number assembled. {n_words} word(s) went through the loop.")
    # C2 event detection
    if r["missed"] or r["extra"] or n_words != n_target:
        causes = [f"digit {', '.join(map(str, m['target_positions']))}: {m['likely_cause']}" for m in r["missed"]] + \
                 [f"word {e['word']}: {e['likely_cause']}" for e in r["extra"]]
        add("§2, §3", "Event detection finds every spoken digit (energy + zero-crossing rate).", "FAIL",
            f"{n_words} word(s) detected for {n_target} digit(s). Run status: {run['status']}. "
            + (("Likely cause: " + "; ".join(causes) + ". See section 3.") if causes else ""))
    else:
        add("§2, §3", "Event detection finds every spoken digit (energy + zero-crossing rate).", "PASS",
            f"{n_words} word(s) detected for {n_target} digit(s).")
    # C3 all approaches
    missing = [f"{label(f'{a}_{s}')}" for a in ("1a", "1b", "1c") for s in ("default", "tuned") if f"{a}_{s}" not in models]
    add("§2", "All three approaches classify each word live (default and tuned).", "FAIL" if missing else "PASS",
        ("No model for: " + ", ".join(missing) + ".") if missing else f"{len(ids)} models answered every word.")
    # C4 real time
    delay = QUIET_BLOCKS * BLOCK_S
    times = [t for w in r["words"] for t in w["predict_ms"].values()]
    add("§2", "The digit is shown in real time.", "INFO",
        f"The loop closes a word after {delay:.1f} s of quiet. Then each model needs at most {max(times):.1f} ms to answer."
        if times else "No word was classified.")
    # C5 pizza test
    passed = [mid for mid in ids if per_model[mid]["pizza_passed"]]
    failed = [mid for mid in ids if not per_model[mid]["pizza_passed"]]
    if run["status"] == "error" or not ids:
        add("§1", "Pizza test: the dialled number equals the target (one wrong digit = wrong shop).", "FAIL", "No number was dialled.")
    else:
        add("§1", "Pizza test: the dialled number equals the target (one wrong digit = wrong shop).",
            "PASS" if not failed else "FAIL",
            f"Target {run['target']}. Passed: {', '.join(label(m) for m in passed) or 'none'}. "
            f"Failed: {', '.join(label(m) for m in failed) or 'none'}.")
    # C6 new speaker
    split = r["split"] or {}
    if run["source"] == "simulation":
        seen = run["speaker"] in split.get("training", [])
        if run["voice"] == "new":
            add("§5.2", "A new speaker is tested (no leakage of voices between training and test).",
                "FAIL" if seen else "PASS",
                f"The recordings of {run['speaker']} are {'in the training set: this is not a new-speaker test' if seen else 'not in the training set'}.")
        else:
            add("§5.2", "A new speaker is tested (no leakage of voices between training and test).", "INFO",
                f"Training voice ({run['speaker']}): the models saw this voice. This run checks the loop, not speaker independence.")
    else:
        add("§5.2", "A new speaker is tested (no leakage of voices between training and test).", "INFO",
            f"Declared as {VOICE_NAMES[run['voice']]}. The system cannot check who speaks. Training voices: "
            f"{', '.join(split.get('training', [])) or 'unknown'}; test voice: {', '.join(split.get('test', [])) or 'unknown'}.")
    # C7 compounding
    parts = [f"{label(m)} {pct(per_model[m]['chance_of_correct_number'])}" for m in ids
             if per_model[m]["chance_of_correct_number"] is not None]
    add("§1, §7", "Per-digit errors compound: a number needs every digit right.", "INFO",
        f"Chance of a correct {n_target}-digit number from the test accuracy (one run cannot confirm it): " + "; ".join(parts) +
        f". Observed: {len(passed)} of {len(ids)} models correct.")
    # C8 1.c versus 1.b
    lines = []
    for s in ("tuned", "default"):
        b, c = models.get(f"1b_{s}"), models.get(f"1c_{s}")
        if b and c and b["classifier"] == c["classifier"] and b["values_per_word"] and c["values_per_word"]:
            lines.append(f"{s}: {b['values_per_word'] / c['values_per_word']:.1f}× fewer values ({c['values_per_word']} instead of "
                         f"{b['values_per_word']}), validation {100 * (c['validation_accuracy'] - b['validation_accuracy']):+.1f} pp, "
                         f"test {100 * (c['test_accuracy'] - b['test_accuracy']):+.1f} pp (both {c['classifier']})")
    for s in ("default",):
        b, c = models.get(f"1b_{s}"), models.get(f"1c_{s}")
        if lines and b and c and b["classifier"] != c["classifier"]:
            lines.append(f"{s}: not comparable, the classifiers differ ({b['classifier']} against {c['classifier']})")
    add("§5.6", "1.c reports accuracy AND data-size reduction against 1.b (same classifier).", "INFO" if lines else "FAIL",
        "; ".join(lines) if lines else "1.b and 1.c do not use the same classifier here, or a model is missing: no direct comparison.")
    # C9 same loop for simulation and microphone
    add("§6", "The same code runs the simulation and the microphone.", "INFO",
        f"Source: {SOURCE_NAMES[run['source']]}. Both sources use the loop in live_demo/web.py (blocks of {BLOCK_S:.1f} s, "
        f"threshold {LOUDNESS:g}× the room noise, word ends after {QUIET_BLOCKS} quiet blocks).")
    # C10 frozen settings
    floor = r["settings"]["room_during_run_db"]
    note = ""
    if floor is not None and floor <= -6:
        note = f" The room was {-floor:.1f} dB quieter during the run than at the calibration, so the threshold was {-floor:.1f} dB higher than planned."
    elif floor is not None and floor >= 4:
        note = f" The room was {floor:.1f} dB louder during the run than at the calibration, so noise was closer to the threshold."
    add("Tuning Protocol", "The loop settings and models stay the same during the pizza test.", "INFO",
        f"Settings ID {r['settings']['settings_id']} (loop and event-extraction settings), models ID {r['settings']['models_id']} "
        f"(model files). Compare both with the other runs in summary.md.{note}")
    return rows


# ---------------------------------------------------------------- report (markdown)

def strip(r, per_line=50):
    """The level of every 0.1 s block as a text bar. '#' marks a word, 'x' a refused sound."""
    chars = "▁▂▃▄▅▆▇█"
    marks = [" "] * len(r["levels_db"])
    for s, mark in ((r["words"], "#"), (r["refused_sounds"], "x")):
        for w in s:
            for i in range(w["first"], w["last"] + 1):
                marks[i] = mark
    lines = []
    for start in range(0, len(r["levels_db"]), per_line):
        bar = "".join(chars[int(min(max(v, 0), 31.99) / 4)] for v in r["levels_db"][start:start + per_line])
        lines.append(f"{start * BLOCK_S:5.1f} s  {bar}")
        lines.append(f"         {''.join(marks[start:start + per_line])}")
    return "\n".join(lines)


def render(r):
    run, st, models, per_model = r["run"], r["settings"], r["models"], r["per_model"]
    language, target = run["language"], run["target"]
    ids = list(models)
    ok = sum(per_model[m]["pizza_passed"] for m in ids)
    out = [f"# Demo run {run['started'].replace('T', ' ')[:19]} · {LANGUAGE_NAMES[language]} · {SOURCE_NAMES[run['source']]} · "
           f"{VOICE_NAMES[run['voice']]}", ""]
    out.append(f"**Result: {ok} of {len(ids)} models dialled the correct number ({target}).** "
               f"Run status: {run['status']}. Words: {run['words_detected']} detected, {run['words_expected']} expected.")
    if run["message"]:
        out.append(f"Message: {run['message']}")
    out += ["", "## 1. Run", "", "| Item | Value |", "|---|---|"]
    rows = [("Run ID", run["id"]), ("Started / ended", f"{run['started']} / {run['ended']}"),
            ("Language", LANGUAGE_NAMES[language]), ("Source", SOURCE_NAMES[run["source"]]),
            ("Voice (declared)", VOICE_NAMES[run["voice"]]),
            ("Speaker", run["speaker"] or "not recorded (microphone)"), ("Input device", run["input_device"] or "–"),
            ("Target number", target), ("Audio length", f"{run['duration_s']} s"),
            ("Training voices / test voice", f"{', '.join((r['split'] or {}).get('training', [])) or '?'} / "
                                             f"{', '.join((r['split'] or {}).get('test', [])) or '?'}"),
            ("Code", f"commit {r['environment']['commit'] or 'unknown'} on {r['environment']['branch'] or 'unknown'}"),
            ("Python / scikit-learn / numpy", f"{r['environment']['python']} / {r['environment']['scikit_learn']} / {r['environment']['numpy']}"),
            ("Platform", r["environment"]["platform"])]
    out += [f"| {a} | {b} |" for a, b in rows]

    out += ["", "## 2. Checklist of the brief", "", "| # | Brief | Expected outcome | Result | What happened |", "|---|---|---|---|---|"]
    out += [f"| {c['id']} | {c['brief']} | {c['expected']} | **{c['status']}** | {c['what']} |" for c in r["checks"]]

    out += ["", "## 3. What worked and what failed", "", "### Worked", ""]
    worked = []
    if run["words_detected"]:
        worked.append(f"The loop heard {run['words_detected']} word(s) and every model answered each of them.")
    for m in ids:
        pm = per_model[m]
        if pm["right_of_detected"][1]:
            worked.append(f"{label(m)} ({models[m]['classifier']}): {pm['right_of_detected'][0]} of {pm['right_of_detected'][1]} detected words right.")
    out += [f"- {line}" for line in worked] or ["- Nothing."]
    out += ["", "### Failed and why", ""]
    failed = False
    if run["words_detected"] == 0 and run["status"] != "error":
        failed = True
        out.append(f"- **No word was detected.** The loudest block was {max(r['levels_db']) if r['levels_db'] else '–'} dB. "
                   f"The threshold is {st['threshold_db']} dB.")
    for m in r["missed"]:
        failed = True
        digits = ", ".join(f"position {pos}: {say(d, language)}" for pos, d in zip(m["target_positions"], m["expected_digits"]))
        many = len(m["target_positions"]) > 1
        out.append(f"- **Not heard: {digits}.** Likely cause: {m['likely_cause']}. {m['detail']} "
                   f"All models are affected: they never received {'these words' if many else 'this word'}. More training does not help here.")
    for e in r["extra"]:
        failed = True
        out.append(f"- **Extra word {e['word']}** at {e['start_s']} s. Likely cause: {e['likely_cause']}. {e['detail']}")
    subs = [(mid, sub) for mid in ids for sub in per_model[mid]["substitutions"]]
    if subs:
        failed = True
        known = sum(1 for _, sub in subs if sub.get("test_confusions"))
        out += ["", f"**Wrong digits (classification), {len(subs)} in total.** {known} of them are confusions that the model also made on the test set; "
                    f"{len(subs) - known} did not occur on the test set. A wrong digit is a property of one model, not of the loop.", "",
                "| Model | Position | Expected | Heard | Same confusion on the test set | Test accuracy of the expected digit | Word length (s) | Peak (dB) | Note |",
                "|---|---|---|---|---|---|---|---|---|"]
        for mid, sub in subs:
            if "test_confusions" in sub:
                seen = f"{sub['test_confusions']} of {sub['test_words_of_digit']} words" if sub["test_confusions"] else f"never (0 of {sub['test_words_of_digit']})"
                accuracy = pct(sub["test_accuracy_of_digit"], 0)
            else:
                seen, accuracy = "no matrix", "–"
            out.append(f"| {label(mid)} | {sub['position']} | {say(sub['expected'], language)} | {sub['heard']} | {seen} | {accuracy} | "
                       f"{sub['length_s']:.2f} | {sub['peak_db']} | {'; '.join(sub['notes']) or '–'} |")
        out.append("")
    if st["room_during_run_db"] is not None and (st["room_during_run_db"] <= -6 or st["room_during_run_db"] >= 4):
        failed = True
        out.append(f"- **Calibration.** The room level during the run was {st['room_during_run_db']} dB against the calibration "
                   "noise. Calibrate again in the room where you speak.")
    if not failed:
        out.append("- Nothing failed.")

    out += ["", "## 4. Word by word", ""]
    head = ["#", "Expected", "Heard as word no.", "Start (s)", "Length (s)", "Peak (dB)"] + [label(m) for m in ids]
    out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    word_at = {w["target_position"]: w for w in r["words"] if w["target_position"]}
    for j, ch in enumerate(target, start=1):
        w = word_at.get(j)
        if w:
            cells = [str(w["index"]), say(int(ch), language), str(w["index"]), str(w["start_s"]), f"{w['length_s']:.2f}", str(w["peak_db"])]
            cells += [f"{w['answers'][m]} {'ok' if w['answers'][m] == int(ch) else 'WRONG'}" for m in ids]
        else:
            cells = ["–", say(int(ch), language), "–", "–", "–", "–"] + ["not heard"] * len(ids)
        out.append(f"| {j} | " + " | ".join(cells[1:]) + " |")
    for w in r["words"]:
        if w["target_position"] is None:
            cells = ["extra", "–", str(w["index"]), str(w["start_s"]), f"{w['length_s']:.2f}", str(w["peak_db"])] + \
                    [str(w["answers"][m]) for m in ids]
            out.append("| – | " + " | ".join(cells[1:]) + " |")
    out.append("| | **Dialled** | | | | | " + " | ".join(f"`{per_model[m]['dialled'] or '–'}`" for m in ids) + " |")

    out += ["", "## 5. Pizza test per model", "",
            "| Model | Classifier | Dialled | Target | Pizza test | Digits right | Right among detected | Chance of a correct number (test accuracy ^ digits) |",
            "|---|---|---|---|---|---|---|---|"]
    for m in ids:
        pm = per_model[m]
        out.append(f"| {label(m)} | {models[m]['classifier']} | `{pm['dialled'] or '–'}` | `{target}` | "
                   f"**{'PASS' if pm['pizza_passed'] else 'FAIL'}** | {pm['digits_right']} of {run['words_expected']} | "
                   f"{pm['right_of_detected'][0]} of {pm['right_of_detected'][1]} | {pct(pm['chance_of_correct_number'])} |")

    out += ["", "## 6. Models (same values as the page)", "",
            "| Model | Classifier | k | Values per word | Validation | Test | Test words | Model file (SHA-256, first 12) | Results file |",
            "|---|---|---|---|---|---|---|---|---|"]
    for m in ids:
        i = models[m]
        out.append(f"| {i['label']} | {i['classifier']} | {i['k'] or '–'} | {i['values_per_word']} | {pct(i['validation_accuracy'])} | "
                   f"{pct(i['test_accuracy'])} | {i['n_test_words']} | {i['model_file']} ({i['model_sha256_12']}) | {i['result_file']} |")

    out += ["", "## 7. Settings of the loop", "", "| Item | Value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in st.items()]
    out += ["", "## 8. Level of every 0.1 s block", "",
            f"Each character is 4 dB above the room noise at the calibration (▁ = 0 to 4 dB, █ = 28 dB or more). The threshold is "
            f"{st['threshold_db']} dB (▄ and higher). `#` = word, `x` = loud sound without a word.", "", "```", strip(r), "```", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- files and summary

def write(result):
    """Write the .md and .json of a run and rebuild summary.md. Returns the path of the .md (relative to the repo)."""
    folder = log_dir()
    folder.mkdir(parents=True, exist_ok=True)
    stem, number = result["run"]["id"], 1
    while (folder / f"{stem}.json").exists():
        number += 1
        stem = f"{result['run']['id']}_{number}"
    result["run"]["id"] = stem
    (folder / f"{stem}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    (folder / f"{stem}.md").write_text(render(result), encoding="utf-8")
    try:
        write_summary(folder)
    except Exception as error:                                   # the run log is saved; the summary can wait
        print(f"summary.md not updated: {error}")
    return shown(folder / f"{stem}.md")


def counted(run):
    """A run counts for the pizza-test counters when at least one word was heard (a stopped or short run is a failed run)."""
    return run["run"]["status"] != "error" and run["run"]["words_detected"] >= 1


def write_summary(folder):
    runs = []
    for path in sorted(folder.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("log_version") == LOG_VERSION:
            data["_link"] = path.relative_to(folder).with_suffix(".md").as_posix()
            runs.append(data)
    runs.sort(key=lambda d: d["run"]["started"])
    ids = []
    for d in runs:
        for m in d["models"]:
            if m not in ids:
                ids.append(m)
    ids.sort(key=lambda m: (m.split("_")[0], m.split("_")[1] != "default"))
    out = ["# Demo logs: summary", "",
           f"Updated {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}. {len(runs)} run(s) in this folder.", "",
           "A run counts when the loop heard at least one word. A run that you stop early, or in which a word is missed, counts as a failed run. "
           "A run that ended with an error does not count. "
           "Move the logs of dry runs to another folder before you take the counters for the report.", "",
           "## Pizza-test counters (correct numbers / counted runs)", ""]
    groups = {}
    for d in runs:
        if counted(d):
            groups.setdefault((d["run"]["language"], d["run"]["source"], d["run"]["voice"]), []).append(d)
    if groups:
        head = ["Language", "Source", "Voice", "Runs"] + [label(m) for m in ids]
        out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for (language, source, voice), items in sorted(groups.items()):
            counts = [sum(d["per_model"].get(m, {}).get("pizza_passed", False) for d in items) for m in ids]
            out.append(f"| {LANGUAGE_NAMES[language]} | {SOURCE_NAMES[source]} | {VOICE_NAMES[voice]} | {len(items)} | " +
                       " | ".join(f"{c}/{len(items)}" for c in counts) + " |")
    else:
        out.append("No counted run yet.")
    out += ["", "## All runs", ""]
    head = ["Run", "Lang.", "Source", "Voice", "Target", "Words", "Status", "Settings ID / models ID"] + [label(m) for m in ids]
    out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    previous = {}
    for d in runs:
        run = d["run"]
        sid = f"{d['settings']['settings_id']} / {d['settings'].get('models_id', '?')}"
        flag = " (changed)" if previous.get(run["language"], sid) != sid else ""
        previous[run["language"]] = sid
        results = [("pass" if d["per_model"][m]["pizza_passed"] else "FAIL") if m in d["per_model"] else "–" for m in ids]
        out.append(f"| [{d['_link'][:-3]}]({d['_link']}) | {run['language']} | {run['source']} | {run['voice']} | `{run['target']}` | "
                   f"{run['words_detected']}/{run['words_expected']} | {run['status']} | {sid}{flag} | " + " | ".join(results) + " |")
    out += ["", "Settings ID: loop settings and event-extraction settings. Models ID: the model files of the language. "
            "A change from the previous run of the same language means that the runs do not test the same system.", ""]
    (folder / "summary.md").write_text("\n".join(out), encoding="utf-8")
    return len(runs)


def main():
    """py -3.12 -m live_demo.run_log [folder]: rebuild summary.md of a folder and of all folders below it."""
    import argparse
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("folder", nargs="?", default=str(DEFAULT_LOG_DIR), help="default: logs/demo")
    folder = Path(parser.parse_args().folder)
    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")
    count = write_summary(folder)
    print(f"{folder / 'summary.md'} written: {count} run(s).")


if __name__ == "__main__":
    main()
