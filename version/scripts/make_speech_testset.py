"""Build the Pipeline 3 test set: 200 Cantonese voice clips with human transcripts.

    python scripts/make_speech_testset.py

Source: Mozilla Common Voice 22, Cantonese (locale "yue"), *test* split, as published on the
Hugging Face Hub at JackyHoCL/common_voice_22_yue (Common Voice data is CC0). Only clips with
at least two up-votes and no down-votes are used. A fixed seed picks 200 of them, at most two
per speaker, so no single voice dominates. Clips are saved as 16 kHz mono FLAC in
data/speech/clips/ with a manifest, data/speech/cv22_yue_test200.jsonl.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from same_table.speech import SAMPLE_RATE, _resample  # noqa: E402

REPO = "JackyHoCL/common_voice_22_yue"
FILE = "data/test-00000-of-00001.parquet"
OUT = ROOT / "data" / "speech"
N_CLIPS, SEED, PER_SPEAKER = 200, 5240, 2


def main() -> None:
    from huggingface_hub import hf_hub_download

    df = pd.read_parquet(hf_hub_download(REPO, FILE, repo_type="dataset"))
    print(f"test split: {len(df)} clips")
    df = df[(df.up_votes >= 2) & (df.down_votes == 0) & (df.sentence.str.len() >= 3)]
    df = df.sample(frac=1.0, random_state=SEED).groupby("client_id").head(PER_SPEAKER).head(N_CLIPS)
    print(f"selected {len(df)} clips from {df.client_id.nunique()} speakers")

    clips = OUT / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in df.iterrows():
        y, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32", always_2d=True)
        y = _resample(y.mean(axis=1), sr)
        name = f"{r.sentence_id[:16]}.flac"
        sf.write(clips / name, y, SAMPLE_RATE, format="FLAC")
        rows.append({"file": f"clips/{name}", "sentence": r.sentence, "seconds": round(len(y) / SAMPLE_RATE, 2),
                     "speaker": r.client_id[:12],
                     "gender": None if pd.isna(r.gender) else r.gender, "age": None if pd.isna(r.age) else r.age})
    manifest = OUT / "cv22_yue_test200.jsonl"
    manifest.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    (OUT / "SOURCE.txt").write_text(
        "Mozilla Common Voice 22, Cantonese (yue), test split, via https://huggingface.co/datasets/" + REPO + "\n"
        "License: CC0 1.0. Selection: up_votes >= 2, down_votes == 0, seed 5240, max 2 clips per speaker.\n",
        encoding="utf-8")
    secs = np.array([x["seconds"] for x in rows])
    print(f"wrote {len(rows)} clips ({secs.sum() / 60:.1f} min, median {np.median(secs):.1f}s) to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
