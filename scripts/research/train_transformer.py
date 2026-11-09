"""Fine-tune a transformer on the training pool. Selection uses leave-source-out folds; benchmark data is never read."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from common import FOLDS, OUT, ROOT, dump, load_pool, split, val_metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--fold", default="A_wambosec", help="fold name, or 'full' to train on everything")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--cap", type=int, default=0, help="subsample training rows (0 = all)")
ap.add_argument("--hardneg-weight", type=float, default=3.0)
ap.add_argument("--save", action="store_true")
args = ap.parse_args()

torch.manual_seed(7)
np.random.seed(7)
device = "cuda"
df = load_pool()
if args.fold == "full":
    tr, va = np.arange(len(df)), np.array([], dtype=int)
else:
    assert args.fold in FOLDS or args.fold == "calib"
    tr, va = split(df, args.fold)
if args.cap and len(tr) > args.cap:
    tr = np.random.default_rng(5).choice(tr, args.cap, replace=False)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, id2label={0: "SAFE", 1: "INJECTION"}, label2id={"SAFE": 0, "INJECTION": 1}, ignore_mismatched_sizes=True, dtype=torch.float32).to(device)
body = args.max_len - 2
ids_all = tok(df.text.tolist(), add_special_tokens=False, truncation=False)["input_ids"]


def head_tail(ids):
    return ids if len(ids) <= body else ids[: body // 2] + ids[-(body - body // 2) :]


def batch(seqs):
    seqs = [[tok.cls_token_id, *s, tok.sep_token_id] for s in seqs]
    m = max(map(len, seqs))
    x = torch.full((len(seqs), m), tok.pad_token_id)
    a = torch.zeros_like(x)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
        a[i, : len(s)] = 1
    return {"input_ids": x.to(device), "attention_mask": a.to(device)}


y = df.label.to_numpy()
pos = y[tr].mean()
class_w = {1: 0.5 / pos, 0: 0.5 / (1 - pos)}
w = np.array([class_w[int(v)] for v in y], dtype=np.float32)
w[df.source.isin(["hardneg", "short_benign"]).to_numpy()] *= args.hardneg_weight
steps = args.epochs * (len(tr) // args.bs)
no_decay = ("bias", "LayerNorm", "layer_norm")
opt = torch.optim.AdamW([{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": 0.01}, {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}], lr=args.lr)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
scaler = torch.amp.GradScaler("cuda")
t0 = time.time()
rng = np.random.default_rng(11)
model.train()
done = 0
for ep in range(args.epochs):
    order = rng.permutation(tr)
    # Bucket by length within chunks so padding (and VRAM) stays small.
    chunks = [sorted(order[i : i + args.bs * 50], key=lambda j: len(ids_all[j])) for i in range(0, len(order), args.bs * 50)]
    batches = [c[i : i + args.bs] for c in chunks for i in range(0, len(c), args.bs)]
    rng.shuffle(batches)
    for b in batches:
        enc = batch([head_tail(ids_all[j]) for j in b])
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits
            loss = (torch.nn.functional.cross_entropy(logits.float(), torch.tensor(y[b]).to(device), reduction="none") * torch.tensor(w[b]).to(device)).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        done += 1
        if done % 400 == 0:
            print(f"step {done}/{steps} loss={loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

model.eval()


@torch.inference_mode()
def score(indices, step=200, use_windows=True):
    """Probability per text; long texts are scanned in overlapping windows (highest wins) or as head + tail."""
    windows, owners = [], []
    for k, j in enumerate(indices):
        s = ids_all[j]
        if not use_windows:
            windows.append(head_tail(s))
            owners.append(k)
        elif len(s) <= body:
            windows.append(s)
            owners.append(k)
        else:
            for start in range(0, max(1, len(s) - body + step), step):
                windows.append(s[start : start + body])
                owners.append(k)
    order = np.argsort([len(x) for x in windows])
    out = np.zeros(len(indices))
    for i in range(0, len(order), 64):
        idx = order[i : i + 64]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            p = torch.softmax(model(**batch([windows[q] for q in idx])).logits.float(), -1)[:, 1].cpu().numpy()
        for q, v in zip(idx, p, strict=True):
            out[owners[q]] = max(out[owners[q]], float(v))
    return out


res = {"args": vars(args), "train_rows": int(len(tr)), "train_seconds": round(time.time() - t0)}
if len(va):
    s = score(list(va))
    d = df.iloc[va]
    res["val"] = val_metrics(d, s)
    s_ht = score(list(va), use_windows=False)
    res["val_headtail"] = val_metrics(d, s_ht)
    print("headtail", json.dumps(res["val_headtail"]), flush=True)
    res["by_source"] = {k: {"n": int((d.source == k).sum()), "mean_score": float(s[(d.source == k).to_numpy()].mean())} for k in d.source.unique()}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"val_{args.name}_{args.fold}.npz", idx=va, score=s, score_ht=s_ht)
    print(json.dumps(res["val"]), flush=True)
dump(f"run_{args.name}_{args.fold}.json", res)
if args.save:
    d = ROOT / "models" / "research" / args.name
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d)
    tok.save_pretrained(d)
    print("saved", d)
print("done", round(time.time() - t0), "s", flush=True)
