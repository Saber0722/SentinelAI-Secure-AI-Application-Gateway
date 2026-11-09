"""Assemble the training pool from sources outside the benchmark, then drop near-duplicates of test prompts."""
from __future__ import annotations

import json
import random
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import POOL_DIR as P, POOL_FILE, ROOT, tokens  # noqa: E402

from sentinelai.evaluation.dataset import read_jsonl  # noqa: E402
from sentinelai.evaluation.external import DATASETS, load_dataset  # noqa: E402

rng = random.Random(1337)
rows: list[tuple[str, int, str]] = []


def add(texts, label, source, cap=None):
    t = [str(x).strip() for x in texts if isinstance(x, str) and 8 <= len(x.strip()) <= 8000]
    t = list(dict.fromkeys(t))
    if cap and len(t) > cap:
        t = rng.sample(t, cap)
    rows.extend((x, label, source) for x in t)
    print(f"{source:18} label={label} n={len(t)}")


m = pd.read_parquet(next((P / "mosscap").glob("data/train-*.parquet")))
m = m[m.level.isin([f"Level {i}" for i in range(3, 9)]) & (m.prompt.str.len() >= 25)]
add(m.prompt.tolist(), 1, "mosscap", 5000)
add(pd.read_parquet(P / "inthewild/jailbreak_2023_12_25/train-00000-of-00001.parquet").prompt.tolist(), 1, "inthewild")
add(pd.read_parquet(P / "inthewild/regular_2023_12_25/train-00000-of-00001.parquet").prompt.tolist(), 0, "inthewild", 6000)
strip = lambda s: s.replace("Subject of the email: ", "", 1).replace("   Body: ", " ", 1)
ll = json.load(open(P / "llmail/data/labelled_unique_submissions_phase2.json"))
add([strip(k) for k in ll], 1, "llmail", 6000)
add([strip(k) for k in json.load(open(P / "llmail/data/emails_for_fp_tests.json"))], 0, "llmail")
w = pd.concat([pd.read_parquet(f) for f in (P / "wambosec/data").glob("*.parquet")], ignore_index=True)
add(w[w.label == 1].prompt.tolist(), 1, "wambosec")
add(w[w.label == 0].prompt.tolist(), 0, "wambosec")
add(pd.read_csv(P / "yanismiraoui/prompt_injections.csv").prompt_injections.tolist(), 1, "yanismiraoui")
add([p.split("\n")[0] for p in pd.read_parquet(next((P / "alespalla/data").glob("train-*.parquet"))).prompt], 0, "alespalla", 10000)
add([json.loads(line)["instruction"] for line in open(P / "dolly/databricks-dolly-15k.jsonl")], 0, "dolly", 5000)
own = read_jsonl(ROOT / "data/training/prompts.jsonl")
add([r["text"] for r in own if r["label"] == "benign"], 0, "own_templates")
add([r["text"] for r in own if r["label"] == "malicious"], 1, "own_templates")

df = pd.DataFrame(rows, columns=["text", "label", "source"])
conflict = df.groupby("text").label.nunique()
df = df[~df.text.isin(conflict[conflict > 1].index)].drop_duplicates("text")

# Remove anything that near-duplicates a benchmark prompt (Jaccard >= 0.8 on word sets).
test = [tokens(c.text) for d in DATASETS for c in load_dataset(d, ROOT / "data/external")]
test = [t for t in set(test) if t]
index: dict[str, list[frozenset[str]]] = {}
for t in test:
    for w_ in t:
        index.setdefault(w_, []).append(t)
keep = []
for text in df.text:
    t = tokens(text)
    dup = False
    if t:
        rare = sorted(t, key=lambda x: len(index.get(x, ())))[:2]
        for r in rare:
            if any(len(t & o) / len(t | o) >= 0.8 for o in index.get(r, ())):
                dup = True
                break
    keep.append(not dup)
print("near-duplicates of benchmark removed:", df.assign(k=keep).query("not k").groupby(["source", "label"]).size().to_dict())
df = df[keep].reset_index(drop=True)
df.to_parquet(POOL_FILE)
print(df.groupby(["source", "label"]).size().to_string()); print("total", len(df), "attacks", int(df.label.sum()))
