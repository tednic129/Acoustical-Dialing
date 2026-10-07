import soundfile as sf
import numpy as np
import glob
import os

# --- SETTINGS ---
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
SR_EXPECTED = 16000  # we verified all files are 16000 Hz

def parse_filename(path):
    """
    Extracts (digit, language, speaker) from a filename like:
    6_en_maulik_03.wav  →  (6, 'en', 'maulik')
    """
    name = os.path.basename(path)          # e.g. '6_en_maulik_03.wav'
    name = name.replace('.wav', '')        # e.g. '6_en_maulik_03'
    parts = name.split('_')               # ['6', 'en', 'maulik', '03']
    digit   = int(parts[0])              # 6
    lang    = parts[1]                   # 'en'
    speaker = parts[2]                   # 'maulik'
    return digit, lang, speaker

# Quick test of the parser
test_path = "data/raw/english/Phone_EN/En_01/6_en_maulik_03.wav"
print(parse_filename(test_path))
# Expected output: (6, 'en', 'maulik')

# Load all WAV files and collect into lists
signals, labels, speakers, languages = [], [], [], []

for wav_path in glob.glob(os.path.join(RAW_DIR, '**', '*.wav'), recursive=True):
    digit, lang, speaker = parse_filename(wav_path)
    signal, sr = sf.read(wav_path)
    
    assert sr == SR_EXPECTED, f"Unexpected sample rate {sr} in {wav_path}"
    assert signal.ndim == 1, f"Expected mono, got shape {signal.shape} in {wav_path}"
    
    signals.append(signal)
    labels.append(digit)
    speakers.append(speaker)
    languages.append(lang)

print(f"Loaded {len(signals)} files")
print(f"Digits found   : {sorted(set(labels))}")
print(f"Speakers found : {sorted(set(speakers))}")
print(f"Languages found: {sorted(set(languages))}")
print(f"Shortest signal: {min(len(s) for s in signals)} samples")
print(f"Longest signal : {max(len(s) for s in signals)} samples")

# Save separately per language
os.makedirs(OUT_DIR, exist_ok=True)

for lang in ['en', 'de']:
    # Filter to this language
    idx = [i for i, l in enumerate(languages) if l == lang]
    
    lang_signals  = [signals[i]  for i in idx]
    lang_labels   = [labels[i]   for i in idx]
    lang_speakers = [speakers[i] for i in idx]
    
    # Save as object array (signals have different lengths — that's OK for now)
    np.save(f"{OUT_DIR}/maulik_{lang}_signals.npy",  np.array(lang_signals,  dtype=object))
    np.save(f"{OUT_DIR}/maulik_{lang}_labels.npy",   np.array(lang_labels,   dtype=np.int8))
    np.save(f"{OUT_DIR}/maulik_{lang}_speakers.npy", np.array(lang_speakers, dtype=str))
    
    print(f"Saved {len(idx)} files for lang='{lang}'")

print("Done.")