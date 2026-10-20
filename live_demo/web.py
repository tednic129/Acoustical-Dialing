"""Browser demo: select the language, type the number, speak. The page shows every digit live.

    py -3.12 -m live_demo.web                     # opens http://localhost:8765 in the browser
    py -3.12 -m live_demo.web --target 0631123456 --no-browser

The listening loop is the one in live_demo/dial.py: same 0.1 s blocks, same thresholds, same
event extraction. Every spoken word goes to all models that exist for the selected language:

    approach  1.a, 1.b, 1.c
    settings  default  models/<tag>_best.joblib          (train.py without --tuned)
              tuned    models/<tag>_tuned_best.joblib    (train.py --tuned)

<tag> is 1a / 1b / 1c for German and 1b_en / 1c_en for English (the file-name rule of 1.b and
1.c). The page shows the answers side by side with the expected digit, builds the dialled
number digit by digit, counts the pizza test per voice, and lists each model's values per
word and validation -> test accuracy from its results/*.json file.
Every run that ends writes a log (what worked, what failed and why) to logs/demo/, or to the folder of --log-dir: see run_log.py.
Standard library only: no new packages, no internet connection.
"""
import argparse
import json
import queue
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np

from common import config
from common.events import extract_event
from common.preprocessing import load_processed, split_masks
from live_demo.dial import BLOCK, LOUDNESS, QUIET_BLOCKS, block_source_microphone, block_source_simulated
from live_demo.run_log import DEFAULT_LOG_DIR, Recorder, build as build_log, file_hash, log_dir, set_log_dir, write as write_log

PAGE = Path(__file__).with_name("web.html")
APPROACHES = ("1a", "1b", "1c")
SETTINGS = ("default", "tuned")
LANGUAGES = ("de", "en")
VOICES = ("new", "training")
MAX_DIGITS = 15


def model_tag(approach, language, tuned):
    """1c, 1c_tuned, 1c_en, 1c_en_tuned: the file-name rule of 1.b and 1.c."""
    base = approach if language == "de" else f"{approach}_{language}"
    return base + ("_tuned" if tuned else "")


def classifier_name(model):
    """'kNN', 'Linear SVM' or 'RBF SVM': the last step of a saved pipeline."""
    last = model.steps[-1][1] if hasattr(model, "steps") else model
    if type(last).__name__ == "KNeighborsClassifier":
        return "kNN"
    if type(last).__name__ == "SVC":
        return {"linear": "Linear SVM", "rbf": "RBF SVM"}.get(last.kernel, f"{last.kernel} SVM")
    return type(last).__name__


def kept_k(model):
    """k of a 1.c pipeline (k x k DCT values per word), else None."""
    first = model.steps[0][1] if hasattr(model, "steps") else None
    return (getattr(first, "kw_args", None) or {}).get("k")


_cache = {}


def load_model(tag):
    """(model, info) for models/<tag>_best.joblib; model is None if the file is missing."""
    path = config.MODELS_DIR / f"{tag}_best.joblib"
    if not path.exists():
        return None, {"tag": tag, "available": False, "model_file": f"models/{path.name}"}
    stamp = path.stat().st_mtime
    if tag not in _cache or _cache[tag][0] != stamp:
        _cache[tag] = (stamp, joblib.load(path))
    model = _cache[tag][1]
    name = classifier_name(model)
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    result_path = config.RESULTS_DIR / f"{tag}_{slug}.json"
    info = {"tag": tag, "available": True, "classifier": name, "model_file": f"models/{path.name}",
            "k": kept_k(model)}
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        info.update(result_file=f"results/{result_path.name}", values=result.get("values_per_word"),
                    ratio=result.get("compression_ratio"),
                    validation=result.get("validation_accuracy"), test=result.get("test_accuracy"),
                    n_test=int(np.sum(result["confusion_matrix"])) if "confusion_matrix" in result else None)
        if result.get("k") is not None and info["k"] is not None and result["k"] != info["k"]:
            info["mismatch"] = f"models/{path.name}: k = {info['k']}, results/{result_path.name}: k = {result['k']}"
    return model, info


_voices = {}


def voices(language):
    """Training and test speakers of the split, with word counts and word lengths (None if the data is not prepared).

    Cached until data/processed/ changes, so a run can read it without loading the data again.
    """
    labels = config.PROCESSED_DIR / "labels.npy"
    stamp = labels.stat().st_mtime if labels.exists() else None
    if stamp is not None and _voices.get(language, (None,))[0] == stamp:
        return _voices[language][1]
    try:
        data = load_processed(languages=[language])
        train, test = split_masks(data["speakers"])
    except (FileNotFoundError, ValueError):
        return None
    seconds = np.array([len(event) for event in data["events"]]) / config.SAMPLE_RATE
    result = {"training": sorted(set(data["speakers"][train].tolist())), "n_training": int(train.sum()),
              "test": sorted(set(data["speakers"][test].tolist())), "n_test": int(test.sum()),
              "word_length_s": {"min": round(float(seconds.min()), 2), "median": round(float(np.median(seconds)), 2),
                                "p95": round(float(np.percentile(seconds, 95)), 2), "max": round(float(seconds.max()), 2)}}
    _voices[language] = (stamp, result)
    return result


def input_device():
    """Name of the default microphone, for the log (None if it cannot be read)."""
    try:
        import sounddevice as sd
        device = sd.query_devices(kind="input")
        return f"{device['name']} ({int(device['default_samplerate'])} Hz)"
    except Exception:
        return None


class Demo:
    """One listening run at a time; every open page receives the events."""

    def __init__(self, target=""):
        self.target = target
        self.lock = threading.Lock()
        self.clients = []
        self.stop = threading.Event()
        self.thread = None
        self.active = False           # True from Start until the loop ends (the log is written after that)
        self.runs = 0
        self.microphone = True        # False on a server without a microphone (--no-microphone)

    def send(self, **event):
        with self.lock:
            for client in self.clients:
                client.put(event)

    def running(self):
        return self.active and self.thread is not None and self.thread.is_alive()

    def info(self, language):
        models = []
        for approach in APPROACHES:
            for settings in SETTINGS:
                _, info = load_model(model_tag(approach, language, settings == "tuned"))
                models.append({"id": f"{approach}_{settings}", "approach": approach, "settings": settings, **info})
        return {"language": language, "target": self.target, "models": models, "voices": voices(language),
                "running": self.running(), "microphone": self.microphone, "threshold_db": round(20 * np.log10(LOUDNESS), 1)}

    def start(self, language, target, simulate, voice):
        if language not in LANGUAGES or voice not in VOICES:
            return "unknown"
        if not target.isdigit() or len(target) > MAX_DIGITS:
            return "badNumber"
        if not simulate and not self.microphone:
            return "noMicrophone"
        if self.running():
            return "busy"
        models, infos = {}, {}
        for approach in APPROACHES:
            for settings in SETTINGS:
                model, info = load_model(model_tag(approach, language, settings == "tuned"))
                if model is not None:
                    models[f"{approach}_{settings}"] = model
                    infos[f"{approach}_{settings}"] = info
        if not models:
            return "noModel"
        self.stop.clear()
        self.runs += 1
        self.active = True
        self.thread = threading.Thread(target=self.run, args=(language, target, simulate, voice, models, self.runs, infos),
                                       daemon=True)
        self.thread.start()
        return None

    def run(self, language, target, simulate, voice, models, seed, infos):
        """The loop of live_demo.dial.dial(), with several models, events for the page and a log of the run."""
        blocks, recorder, status, message = None, None, "complete", None
        try:
            split = voices(language)
            speaker = None
            if simulate:
                if split is None:
                    raise RuntimeError("Run `py -3.12 -m common.prepare_data` first.")
                speaker = split["test"][0] if voice == "new" else split["training"][0]
                self.send(type="status", key="simulating", speaker=speaker)
                noise, blocks = block_source_simulated(target, speaker, [language], seed=seed)
            else:
                self.send(type="status", key="calibrating")
                noise, blocks = block_source_microphone()
                self.send(type="status", key="listening")
            recorder = Recorder(language, target, "simulation" if simulate else "microphone", voice, speaker, noise, infos)
            recorder.device = None if simulate else input_device()
            heard = {name: "" for name in models}
            word, quiet, previous, position, first = [], 0, np.zeros(BLOCK, dtype=np.float32), 0, 0
            for block in blocks:
                if self.stop.is_set():
                    break
                if simulate:
                    time.sleep(BLOCK / config.SAMPLE_RATE)        # play the recordings in real time
                rms = float(np.sqrt(np.mean(block ** 2)))
                db = 20 * np.log10(rms / noise + 1e-9)
                self.send(type="level", db=round(db, 1))
                recorder.block(db)                                # the log only watches: no decision uses it
                loud = rms > LOUDNESS * noise
                if loud and not word:
                    word = [previous]
                    first = recorder.now
                if word:
                    word.append(block)
                    quiet = 0 if loud else quiet + 1
                    if quiet >= QUIET_BLOCKS:
                        event = extract_event(np.concatenate(word))
                        if event is not None:
                            answers, milliseconds = {}, {}
                            for name, model in models.items():
                                started = time.perf_counter()
                                answers[name] = int(model.predict([event])[0])
                                milliseconds[name] = (time.perf_counter() - started) * 1000
                            for name, digit in answers.items():
                                heard[name] += str(digit)
                            position += 1
                            recorder.word(first, recorder.now - quiet, len(word), len(event) / config.SAMPLE_RATE,
                                          answers, milliseconds)
                            self.send(type="word", position=position, expected=int(target[position - 1]),
                                      seconds=round(len(event) / config.SAMPLE_RATE, 2), answers=answers)
                        else:
                            recorder.refuse(first, recorder.now - quiet, len(word))
                        word, quiet = [], 0
                        if position == len(target):
                            break
                previous = block
            status = "stopped" if self.stop.is_set() else "complete" if position == len(target) else "recording ended"
            self.send(type="done", target=target, voice=voice, heard=heard, complete=position == len(target),
                      counted=position >= 1, status=status)
        except Exception as error:                                # shown on the page, not only here
            status, message = "error", f"{type(error).__name__}: {error}"
            self.send(type="error", message=message)
        finally:
            if blocks is not None and hasattr(blocks, "close"):
                blocks.close()                                    # closes the microphone stream
            self.active = False                                   # the next run may start while the log is written
            self.log(recorder, status, message, language)

    def log(self, recorder, status, message, language):
        """Write the log of the run (logs/demo/) and tell the page where it is. A failed log never changes the result."""
        if recorder is None or not recorder.levels:
            return
        try:
            self.send(type="log", path=write_log(build_log(recorder, status, message, voices(language))))
        except Exception as error:
            self.send(type="log", error=f"{type(error).__name__}: {error}")


DEMO = Demo()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):                                 # keep the console quiet
        pass

    def reply(self, body, content_type="application/json; charset=utf-8", status=200):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.reply(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/info":
            language = parse_qs(url.query).get("language", ["de"])[0]
            if language not in LANGUAGES:
                self.reply({"error": "unknown language"}, status=400)
            else:
                self.reply(DEMO.info(language))
        elif url.path == "/events":
            self.events()
        else:
            self.reply({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        if self.path == "/start":
            error = DEMO.start(body.get("language"), str(body.get("target", "")), bool(body.get("simulate")),
                               body.get("voice", "new"))
            self.reply({"ok": error is None, "error": error}, status=200 if error is None else 409)
        elif self.path == "/stop":
            DEMO.stop.set()
            self.reply({"ok": True})
        else:
            self.reply({"error": "not found"}, status=404)

    def events(self):
        """Server-sent events: one JSON line per event, as long as the page is open."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        client = queue.Queue()
        with DEMO.lock:
            DEMO.clients.append(client)
        try:
            while True:
                try:
                    chunk = "data: " + json.dumps(client.get(timeout=10)) + "\n\n"
                except queue.Empty:
                    chunk = ": still here\n\n"
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
        except OSError:                                           # the page was closed or reloaded
            pass
        finally:
            with DEMO.lock:
                DEMO.clients.remove(client)


def warm_logs():
    """Hash the model files once, in the background, so the first log is written fast."""
    for language in LANGUAGES:
        for model in DEMO.info(language)["models"]:
            if model["available"]:
                try:
                    file_hash(config.ROOT / model["model_file"])
                except OSError:
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on (default 127.0.0.1: this computer only; 0.0.0.0 on a server)")
    parser.add_argument("--no-microphone", action="store_true",
                        help="this computer has no microphone: the page offers the recordings only")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--target", default="", help="number shown in the input field at the start")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR),
                        help="folder for the run logs, relative to the repo (default: logs/demo)")
    args = parser.parse_args()
    set_log_dir(args.log_dir)

    DEMO.target = args.target if args.target.isdigit() else ""
    DEMO.microphone = not args.no_microphone
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Demo page: {url}   (Ctrl+C stops the server)")
    print(f"  logs: {log_dir()}")
    for language in LANGUAGES:
        files = [m["model_file"] for m in DEMO.info(language)["models"] if m["available"]]
        print(f"  {language}: " + (", ".join(files) if files else "no models found"))
    threading.Thread(target=warm_logs, daemon=True).start()
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        DEMO.stop.set()
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
