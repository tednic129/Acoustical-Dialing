"""Live dialling demo: say the digits one by one, the chosen approach recognises each word.

    python -m live_demo.dial --approach 1c --target 0631XXXXXXX              # microphone
    python -m live_demo.dial --approach 1c --target 0631123456 --simulate    # no microphone

Every approach saves its demo model as models/<approach>_best.joblib: a scikit-learn pipeline
whose first step turns trimmed words into features, so model.predict([word]) returns the digit.

--simulate plays recorded words (by default from the first test speaker in split.json) with
pauses in between, through exactly the same loop, so the whole chain can be tested without a mic.
"""
import argparse
import sys

import joblib
import numpy as np

from common import config
from common.events import extract_event
from common.preprocessing import load_processed, load_split, normalise

BLOCK = 1600            # listen in 0.1 s blocks
QUIET_BLOCKS = 3        # 0.3 s of quiet ends a word
LOUDNESS = 4.0          # a block is speech when it is 4x louder than the room


def block_source_microphone():
    """Yield 0.1 s blocks from the default microphone, after measuring the room noise."""
    import sounddevice as sd

    print("Calibrating: stay quiet for 1 second ...")
    room = sd.rec(config.SAMPLE_RATE, samplerate=config.SAMPLE_RATE, channels=1, blocking=True)[:, 0]
    noise = float(np.sqrt(np.mean(room ** 2))) + 1e-6
    print("Listening. Say one digit at a time, with a short pause in between.")

    def blocks():
        with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, blocksize=BLOCK) as stream:
            while True:
                yield stream.read(BLOCK)[0][:, 0].astype(np.float32)

    return noise, blocks()


def block_source_simulated(target, speaker, languages, seed=0):
    """Yield 0.1 s blocks of recorded words for the target digits, separated by quiet pauses."""
    data = load_processed(languages=languages)
    rng = np.random.default_rng(seed)
    pause = lambda seconds: rng.normal(0, 0.002, int(seconds * config.SAMPLE_RATE)).astype(np.float32)  # noqa: E731
    parts = [pause(1.0)]
    for digit in target:
        choices = np.flatnonzero((data["speakers"] == speaker) & (data["labels"] == int(digit)))
        if choices.size == 0:
            raise SystemExit(f"No recording of digit {digit} by {speaker} in languages {languages}")
        parts += [0.8 * normalise(data["events"][rng.choice(choices)]), pause(0.6)]
    audio = np.concatenate(parts + [pause(1.0)])
    noise = float(np.sqrt(np.mean(parts[0] ** 2)))
    return noise, (audio[i:i + BLOCK] for i in range(0, len(audio) - BLOCK + 1, BLOCK))


def dial(model, target, noise, blocks):
    """Run the word-detection loop until as many digits as the target has were heard."""
    number, word, quiet, previous = "", [], 0, np.zeros(BLOCK, dtype=np.float32)
    for block in blocks:
        loud = np.sqrt(np.mean(block ** 2)) > LOUDNESS * noise
        if loud and not word:
            word = [previous]                       # keep the block before, so quiet starts survive
        if word:
            word.append(block)
            quiet = 0 if loud else quiet + 1
            if quiet >= QUIET_BLOCKS:
                event = extract_event(np.concatenate(word))
                if event is not None:
                    digit = int(model.predict([event])[0])
                    number += str(digit)
                    print(f"  heard {digit}    dialled so far: {number}")
                word, quiet = [], 0
                if len(number) == len(target):
                    break
        previous = block
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approach", choices=["1a", "1b", "1c"], required=True)
    parser.add_argument("--target", required=True, help="the number to dial, digits only")
    parser.add_argument("--simulate", action="store_true", help="use recordings instead of the microphone")
    parser.add_argument("--speaker", help="simulate with this speaker's recordings (default: first test speaker)")
    parser.add_argument("--languages", nargs="+", default=list(config.LANGUAGES))
    args = parser.parse_args()

    if not args.target.isdigit():
        raise SystemExit("--target must contain digits only, e.g. 0631123456")
    model_path = config.MODELS_DIR / f"{args.approach}_best.joblib"
    if not model_path.exists():
        raise SystemExit(f"{model_path.relative_to(config.ROOT)} not found. Train approach {args.approach} first.")
    model = joblib.load(model_path)

    if args.simulate:
        speaker = args.speaker or (load_split() or [None])[0]
        if speaker is None:
            raise SystemExit("Pass --speaker, or name a test speaker in split.json.")
        print(f"Simulating {args.target} with {speaker}'s recordings ({', '.join(args.languages)})")
        noise, blocks = block_source_simulated(args.target, speaker, args.languages)
    else:
        noise, blocks = block_source_microphone()

    number = dial(model, args.target, noise, blocks)
    if number == args.target:
        print("Pizza is on its way!")
    else:
        print(f"Wrong number dialled: {number or '(nothing heard)'} instead of {args.target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
