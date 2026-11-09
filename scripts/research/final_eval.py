"""ONE-TIME test on the 5,027 benchmark prompts. Every model, weight and threshold is frozen before this runs."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, ROOT  # noqa: E402

from sentinelai.audit import NullAuditLogger  # noqa: E402
from sentinelai.config import SecurityPolicy, get_settings  # noqa: E402
from sentinelai.detectors.ml_classifier import featurize  # noqa: E402
from sentinelai.detectors.secrets import SecretsConfig  # noqa: E402
from sentinelai.evaluation.external import DATASETS, Scores, guard_scorer, load_dataset, metrics, sentinel_scorer  # noqa: E402
from sentinelai.gateway.pipeline import SecurityGateway  # noqa: E402
from sentinelai.models import MockModel  # noqa: E402

LABELLED = ["deepset", "jackhhao", "safeguard", "spml"]
data = {d.key: load_dataset(d, ROOT / "data/external") for d in DATASETS}
s = get_settings()
gw = SecurityGateway(SecurityPolicy(), MockModel(), NullAuditLogger(), SecretsConfig.load(s.protected_secrets_path), max_input_chars=1_000_000)
rule_run = sentinel_scorer(gw, True, False)
protectai_run, _ = guard_scorer(ROOT / "models/guards/protectai_v2", "cuda")
injec_dir = ROOT / "models/guards/__injecguard_excluded"
injec_run = guard_scorer(injec_dir, "cuda")[0] if injec_dir.exists() else None
ddir = ROOT / "models/research/dsmall_final"
tok = AutoTokenizer.from_pretrained(ddir)
model = AutoModelForSequenceClassification.from_pretrained(ddir).to("cuda").eval()
BODY = 254


@torch.inference_mode()
def dsmall(texts):
    ids = tok(texts, add_special_tokens=False, truncation=False)["input_ids"]
    win, own = [], []
    for k, x in enumerate(ids):
        for st in range(0, max(1, len(x) - BODY + 200), 200) if len(x) > BODY else [0]:
            win.append(x[st : st + BODY])
            own.append(k)
    order = np.argsort([len(w) for w in win])
    out = np.zeros(len(texts))
    for i in range(0, len(order), 64):
        b = [[tok.cls_token_id, *win[q], tok.sep_token_id] for q in order[i : i + 64]]
        m = max(map(len, b))
        x = torch.tensor([v + [tok.pad_token_id] * (m - len(v)) for v in b]).cuda()
        p = torch.softmax(model(input_ids=x, attention_mask=(x != tok.pad_token_id).long()).logits.float(), -1)[:, 1].cpu().numpy()
        for q, v in zip(order[i : i + 64], p, strict=True):
            out[own[q]] = max(out[own[q]], float(v))
    return out


logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
fus = {}


def fuse(cfg, f):
    z = cfg["intercept"] + sum(c * f[n] for n, c in zip(cfg["features"], cfg["coef"], strict=True))
    return 1 / (1 + np.exp(-z))


comp = {}
for key, cases in data.items():
    t = [c.text for c in cases]
    r = np.array(rule_run(t).scores)
    tf = np.zeros(len(t))
    d = dsmall(t)
    g = np.array(protectai_run(t).scores)
    comp[key] = {"y": [c.label for c in cases], "rule": r, "tfidf": tf, "dsmall": d, "protectai": g, "injecguard": np.array(injec_run(t).scores) if injec_run else None}
    print("scored", key, len(t), flush=True)


def evaluate(fn, trained_on=()):
    per = {}
    for key, c in comp.items():
        sc, fl = fn(c)
        per[key] = metrics(c["y"], Scores(list(map(float, sc)), list(map(bool, fl)), [0.0], "n/a"))
    fair = [k for k in LABELLED if k not in trained_on]
    macro = {m: statistics.fmean([per[k][m] for k in fair if per[k][m] is not None]) for m in ("recall", "precision", "f1", "fpr", "roc_auc")}
    common = ["safeguard", "spml"]
    comm = {m: statistics.fmean([per[k][m] for k in common if per[k][m] is not None]) for m in ("recall", "precision", "fpr", "roc_auc")}
    return {"macro": macro, "macro_over": fair, "common": comm, "per_dataset": per, "gandalf_recall": per["gandalf"]["recall"], "notinject_fpr": per["notinject"]["fpr"]}


def feats(c):
    return {"rule": c["rule"], "tfidf": logit(c["tfidf"]), "dsmall": logit(c["dsmall"]), "protectai": logit(c["protectai"])}


P = json.loads((ROOT / "models/research/policy.json").read_text())


def cascade(c):
    f = (c["rule"] >= P["t_rule"]) | (c["dsmall"] >= P["t_d"]) | ((c["protectai"] >= P["t_g"]) & (c["dsmall"] >= P["t_low"]))
    # Ranking score for ROC-AUC: the strongest signal, with the guard capped when the veto applies.
    guard = np.where(c["dsmall"] >= P["t_low"], c["protectai"], np.minimum(c["protectai"], 0.49))
    return np.maximum.reduce([c["rule"], c["dsmall"], guard]), f


systems = {
    "CASCADE: rules | DeBERTa | ProtectAI with veto": evaluate(cascade, ("jackhhao",)),
    "OWN: rules | DeBERTa (no third-party model)": evaluate(lambda c: (np.maximum(c["rule"], c["dsmall"]), (c["rule"] >= P["t_rule"]) | (c["dsmall"] >= P["t_d"]))),
    "DeBERTa alone (trained here)": evaluate(lambda c: (c["dsmall"], c["dsmall"] >= P["t_d"])),
    "ProtectAI v2 alone (default 0.5)": evaluate(lambda c: (c["protectai"], c["protectai"] >= 0.5), ("jackhhao",)),
}
if injec_run:
    systems["InjecGuard (default 0.5)"] = evaluate(lambda c: (c["injecguard"], c["injecguard"] >= 0.5))
old = {x["name"]: x for x in json.loads((s.results_dir / "external_benchmark.json").read_text())["systems"]}
print(f"\n{'system':52}{'recall':>8}{'prec':>7}{'FPR':>7}{'AUC':>7}{'gandalf':>9}{'notinj':>8}")
for n, r in systems.items():
    m = r["macro"]
    print(f"{n:52}{m['recall'] * 100:8.1f}{m['precision'] * 100:7.1f}{m['fpr'] * 100:7.1f}{m['roc_auc']:7.3f}{r['gandalf_recall'] * 100:9.1f}{r['notinject_fpr'] * 100:8.1f}")
for n, x in old.items():
    m = x["macro"]
    print(f"{'[v1] ' + n.replace('*', ''):52}{m['recall'] * 100:8.1f}{m['precision'] * 100:7.1f}{m['fpr'] * 100:7.1f}{m['roc_auc']:7.3f}{x['per_dataset']['gandalf']['recall'] * 100:9.1f}{x['per_dataset']['notinject']['fpr'] * 100:8.1f}")
print("\nper dataset recall / FPR")
for n, r in systems.items():
    print(n, {k: (None if v["recall"] is None else round(v["recall"] * 100, 1), None if v["fpr"] is None else round(v["fpr"] * 100, 1)) for k, v in r["per_dataset"].items()})
print("common (safeguard, spml):", {n: {k: round(v, 3) for k, v in r["common"].items()} for n, r in systems.items()})
(OUT / "final_eval_second.json").write_text(json.dumps({"generated_at": time.strftime("%FT%TZ", time.gmtime()), "systems": systems}, indent=1, default=float))
