import os
import shutil

base = "data/raw"

for lang in ["english", "german"]:
    # Create flat speaker folder
    out_dir = os.path.join(base, lang, "maulik")
    os.makedirs(out_dir, exist_ok=True)

    lang_path = os.path.join(base, lang)
    for session_folder in os.listdir(lang_path):
        session_path = os.path.join(lang_path, session_folder)
        # Skip if it's the output folder or not a directory
        if not os.path.isdir(session_path) or session_folder == "maulik":
            continue
        for fname in os.listdir(session_path):
            if fname.endswith(".wav"):
                src = os.path.join(session_path, fname)
                dst = os.path.join(out_dir, fname)
                shutil.copy2(src, dst)
                print(f"Copied: {fname} → {lang}/maulik/")

