"""Windows side of the live demo: show the number, then record you saying it in one go.

    python rec.py C:\\Users\\ASUS\\Downloads\\dial.wav 0631123456          # German
    python rec.py C:\\Users\\ASUS\\Downloads\\dial_en.wav 0631123456 en    # English

Needs only numpy + sounddevice (no scikit-learn), so it runs on Windows where the mic works.
Stay quiet for the first second (the room noise is measured from it), then say the whole number
at your normal pace, with a short break between digits, and press Enter when you are done.
"""
import sys
import time
import wave

import numpy as np
import sounddevice as sd

FS = 16000
QUIET = 1.5          # seconds of silence at the start: replay_compare measures the room noise from it
MAX_SECONDS = 60     # safety limit
WORDS = {
    "de": ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"],
    "en": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
}

path = sys.argv[1] if len(sys.argv) > 1 else "dial.wav"
target = sys.argv[2] if len(sys.argv) > 2 else "0631123456"
language = sys.argv[3] if len(sys.argv) > 3 else "de"
if not target.isdigit() or language not in WORDS:
    raise SystemExit("usage: python rec.py <file.wav> <digits> [de|en]")

print(f"\nNumber to dial:  {target}")
print("Say it as:       " + "  ".join(WORDS[language][int(d)] for d in target))
input("\nPress Enter to start (then stay quiet until 'SPEAK NOW') ...")

chunks = []
def keep(block, frames, time_info, status):           # called by sounddevice for every new audio block
    chunks.append(block[:, 0].copy())

with sd.InputStream(samplerate=FS, channels=1, callback=keep):
    print(f"  quiet please: measuring the room ({QUIET:g} s) ...")
    time.sleep(QUIET)
    print(f"\n  SPEAK NOW:  {target}\n")
    input("  ... press Enter when you have said the whole number ")
x = np.concatenate(chunks)[: MAX_SECONDS * FS]

with wave.open(path, "wb") as w:                       # 16-bit PCM WAV, standard library
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(FS)
    w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())

peak = float(np.max(np.abs(x)))
note = "  <- clipping, speak further away" if peak > 0.98 else "  <- very quiet, speak closer" if peak < 0.05 else ""
print(f"\nSaved {path}  ({len(x) / FS:.1f} s, peak {peak:.2f}){note}")
