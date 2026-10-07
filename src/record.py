import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import os

SAMPLE_RATE = 16000
DURATION    = 2       # seconds per digit
SPEAKER     = "maulik"

def record_digit(digit, lang, take, out_dir):
    filename = f"{digit}_{lang}_{SPEAKER}_{take:02d}.wav"
    filepath = os.path.join(out_dir, filename)
    print(f"\n  Digit: {digit}  →  say it in 3...")
    for i in [3, 2, 1]:
        print(f"    {i}...")
        time.sleep(1)
    print("  🎙  SPEAK NOW")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype="float32")
    sd.wait()
    sf.write(filepath, audio, SAMPLE_RATE)
    print(f"  ✅ Saved: {filename}")

def main():
    lang = input("Language? (en / de): ").strip().lower()
    take = int(input("Take number? (1-8): ").strip())
    out_dir = f"data/raw/{('english' if lang == 'en' else 'german')}/{SPEAKER}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nRecording {lang.upper()} Take {take:02d} — 10 digits")
    print("You will have 2 seconds per digit. Speak clearly.\n")

    for digit in range(10):
        input(f"  Press ENTER when ready for digit {digit}...")
        record_digit(digit, lang, take, out_dir)

    print(f"\n✅ Take {take:02d} complete — 10 files saved to {out_dir}")

if __name__ == "__main__":
    main()
