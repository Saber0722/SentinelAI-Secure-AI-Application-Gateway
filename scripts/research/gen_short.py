"""Short, ordinary chat messages: the pool had almost no benign text under a few words."""
from __future__ import annotations

import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import POOL_DIR, ROOT, tokens  # noqa: E402

from sentinelai.evaluation.external import DATASETS, load_dataset  # noqa: E402

SEEDS = """hiya|greetings|howdy|hey there friend|good afternoon|good evening|morning!|thank you|thank you so much|many thanks|cheers|thx|much appreciated|goodbye|see you later|see ya|take care|have a nice day|good night|sounds good|got it|alright|okay then|sure|sure thing|no thanks|no problem|not really|maybe later|of course|i agree|that makes sense|nice|cool|great|awesome|wow|interesting|really?|how are you|how are you doing|what's up|how's it going|nice to meet you|what is your name|are you a bot|are you there|can you help|can you help me|i need help|help please|one more question|tell me more|go on|please continue|try again|explain that|say that again|what do you mean|why is that|how so|tell me a fun fact|tell me something interesting|tell me a story|sing a song|recommend a movie|what time is it|what day is today|how is the weather|i am bored|i am tired|i am hungry|never mind|just testing|test|testing 123|lol|haha|hmm|oops|sorry|my bad|please|yep|nope|yes|no|maybe""".split("|")
KINDS = ["a greeting", "a thank-you", "a goodbye", "agreeing with something", "disagreeing politely", "casual small talk", "a tiny question about the weather or time", "a tiny request for a joke, story or fun fact", "asking the assistant how it is doing", "a filler reply in a conversation", "asking for a simple recommendation", "a short follow-up asking to continue or clarify"]
SYS = "You write very short, casual chat messages a person sends to an assistant. Output only the message, at most six words, with no quotes."


def make(i: int):
    r = random.Random(i)
    try:
        with httpx.Client(timeout=120, trust_env=False) as c:
            j = c.post("http://127.0.0.1:11434/api/chat", json={"model": "qwen2.5:1.5b", "stream": False, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": f"Write {r.choice(KINDS)}."}], "options": {"temperature": 1.1, "num_predict": 20, "seed": i}}).json()
        return j["message"]["content"].strip().strip('"')
    except Exception:
        return None


with ThreadPoolExecutor(6) as ex:
    gen = list(ex.map(make, range(2500)))
gen = [g for g in gen if g and 2 <= len(g) <= 48 and len(g.split()) <= 7 and "\n" not in g]
rng = random.Random(8)
variants = set()
for s in SEEDS + gen:
    variants.update({s, s.capitalize(), s.rstrip("!?.") + rng.choice(["", "!", ".", "?"])})
test = {tokens(c.text) for d in DATASETS for c in load_dataset(d, ROOT / "data/external")}
bad = re.compile(r"password|secret|ignore|instruction|prompt|system", re.I)
rows = sorted(v for v in variants if tokens(v) and tokens(v) not in test and not bad.search(v))
pd.DataFrame({"text": rows, "label": 0, "source": "short_benign"}).to_parquet(POOL_DIR / "short.parquet")
print("short benign rows", len(rows), "generated", len(gen))
print(random.Random(2).sample(rows, 24))
