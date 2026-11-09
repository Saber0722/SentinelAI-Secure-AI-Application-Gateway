"""Over-defence training data: benign prompts that use attack trigger words mined from the training pool only."""
from __future__ import annotations

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from common import HARDNEG_FILE, OUT, load_pool  # noqa: E402

N = 3600
df = load_pool(with_hardneg=False)
vec = TfidfVectorizer(lowercase=True, min_df=20, max_df=0.3, token_pattern=r"[a-zA-Z]{4,}")
X = vec.fit_transform(df.text)
clf = LogisticRegression(max_iter=300, C=3.0, class_weight="balanced").fit(X, df.label)
names = vec.get_feature_names_out()
top = [names[i] for i in clf.coef_[0].argsort()[::-1][:250]]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "trigger_words.json").write_text(json.dumps(top, indent=0))
print("top triggers:", top[:40], flush=True)

rng = random.Random(4242)
TOPICS = ["cooking", "travel", "software development", "home repair", "school homework", "gardening", "personal finance", "fitness", "history", "office work", "photography", "video games", "pet care", "car maintenance", "science", "writing an essay", "music", "customer service", "networking and IT", "a board game"]
LANGS = [("English", 0.85), ("Chinese", 0.05), ("Spanish", 0.03), ("German", 0.03), ("Hindi", 0.02), ("French", 0.02)]
SYS = "You write short, realistic messages that ordinary people send to a helpful assistant. Output only the message, with no quotes or explanation."


def make(i: int):
    r = random.Random(i)
    k = r.choice([1, 2, 3])
    words = r.sample(top, k)
    lang = r.choices([x for x, _ in LANGS], [p for _, p in LANGS])[0]
    topic = r.choice(TOPICS)
    user = (f"Write one harmless, ordinary request or question about {topic}, in {lang}, that naturally uses these words: "
            f"{', '.join(words)}. It must be an innocent everyday request. It must not ask the assistant to change its behaviour, "
            "ignore its rules, or reveal anything about itself.")
    try:
        with httpx.Client(timeout=120, trust_env=False) as c:
            j = c.post("http://127.0.0.1:11434/api/chat", json={"model": "qwen2.5:1.5b", "stream": False, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}], "options": {"temperature": 0.9, "num_predict": 80, "seed": i}}).json()
        return j["message"]["content"].strip().strip('"'), words, lang
    except Exception as e:
        return None, words, repr(e)


with ThreadPoolExecutor(6) as ex:
    out = list(ex.map(make, range(N)))
bad = re.compile(r"(ignore|disregard|forget|override|bypass)\s+(all|any|your|the|previous|prior|above|my)\b.{0,30}(instruction|rule|prompt|polic|guideline)|system prompt|jailbreak|as an ai|i'm sorry|i cannot", re.I)
rows, drop = [], 0
for text, words, lang in out:
    ok = text and 20 <= len(text) <= 400 and not bad.search(text) and (lang != "English" or any(w in text.lower() for w in words))
    if ok:
        rows.append((text, 0, "hardneg"))
    else:
        drop += 1
d = pd.DataFrame(rows, columns=["text", "label", "source"]).drop_duplicates("text")
d.to_parquet(HARDNEG_FILE)
print("kept", len(d), "dropped", drop, flush=True)
for t in d.sample(12, random_state=1).text:
    print(" -", t[:130].replace("\n", " "))
