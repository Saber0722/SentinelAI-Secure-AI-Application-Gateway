"""Benchmark SentinelAI against public real-world datasets and open-source guard models.

    uv sync --extra guards
    uv run --extra guards python scripts/download_external.py     # once
    uv run --extra guards python scripts/benchmark_external.py

Writes results/external_benchmark.json and refreshes the generated tables in
documents/external_benchmark.md (between the results markers; the analysis around them is kept).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from sentinelai.audit import NullAuditLogger
from sentinelai.config import SecurityPolicy, get_settings
from sentinelai.detectors.ml_classifier import MlClassifier, featurize
from sentinelai.detectors.secrets import SecretsConfig
from sentinelai.evaluation.dataset import read_jsonl
from sentinelai.evaluation.external import DATASETS, GUARD_MODELS, guard_scorer, load_dataset, metrics, sentinel_scorer
from sentinelai.gateway.pipeline import SecurityGateway
from sentinelai.models import MockModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external"
MODEL_DIR = ROOT / "models" / "guards"
DOC = ROOT / "documents" / "external_benchmark.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"
LABELLED = ["deepset", "jackhhao", "safeguard", "spml"]


def tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", featurize(text)))


def overlap_with_sentinel_data(datasets: dict[str, list]) -> dict[str, int]:
    """Count external prompts that (near-)duplicate anything SentinelAI was built or trained with."""
    own = [r["text"] for r in read_jsonl(ROOT / "data" / "training" / "prompts.jsonl")]
    for d in (ROOT / "data" / "evaluation", ROOT / "data" / "evaluation" / "holdout"):
        own += [r["text"] for f in ("benign.jsonl", "malicious.jsonl") for r in read_jsonl(d / f)]
    own_sets = [t for t in {tokens(x) for x in own} if t]
    index: dict[str, list[frozenset[str]]] = {}
    for t in own_sets:
        for w in t:
            index.setdefault(w, []).append(t)
    counts = {}
    for key, cases in datasets.items():
        n = 0
        for c in cases:
            t = tokens(c.text)
            if not t:
                continue
            rare = min(t, key=lambda w: len(index.get(w, ())))
            candidates = index.get(rare, ())
            if any(len(t & o) / len(t | o) >= 0.8 for o in candidates):
                n += 1
        counts[key] = n
    return counts


def fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}" if pct else f"{v:.3f}"


def render(res: dict) -> str:
    systems = res["systems"]
    lines = [
        START,
        "",
        f"_Generated {res['generated_at']} by `scripts/benchmark_external.py`; numbers are copied from "
        "`results/external_benchmark.json`. † = the guard model's card lists this dataset as training data "
        "(in-distribution, not a fair unseen test for that model)._",
        "",
        "### Datasets",
        "",
        "| Dataset | Rows used | Attacks | Benign | Licence | Near-duplicates of SentinelAI data |",
        "|---|---|---|---|---|---|",
    ]
    for d in res["datasets"]:
        lines.append(f"| [{d['repo']}](https://huggingface.co/datasets/{d['repo']}) | {d['n']} | {d['positives']} | "
                     f"{d['negatives']} | {d['license']} | {res['overlap'][d['key']]} |")  # fmt: skip
    lines += ["", "### Summary (macro-average over the four labelled datasets)", "",
              "| System | Recall % | Precision % | F1 % | FPR % | ROC-AUC | Gandalf recall % | NotInject FPR % | Latency p50 / p95 ms | Runs on |",
              "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for s in systems:
        m = s["macro"]
        lat = s["latency"]
        lines.append(
            f"| {s['name']} | {fmt(m['recall'])} | {fmt(m['precision'])} | {fmt(m['f1'])} | {fmt(m['fpr'])} | "
            f"{fmt(m['roc_auc'], False)} | {fmt(s['per_dataset']['gandalf']['recall'])} | "
            f"{fmt(s['per_dataset']['notinject']['fpr'])} | {lat['p50']:.1f} / {lat['p95']:.1f} | {lat['device']} |"
        )
    lines += ["", "Macro averages exclude in-distribution († ) cells for the guard model concerned, so ProtectAI and "
              "deepset are averaged over three datasets and the rest over four. The table below removes that difference.", ""]  # fmt: skip
    common = [k for k in LABELLED if not any(k in s.get("trained_on", []) for s in systems)]
    lines += [f"### Common subset: datasets no system was trained on ({', '.join(common)})", "",
              "| System | Recall % | Precision % | F1 % | FPR % | ROC-AUC |", "|---|---|---|---|---|---|"]  # fmt: skip
    for s in systems:
        vals = {m: [s["per_dataset"][k][m] for k in common if s["per_dataset"][k][m] is not None]
                for m in ("recall", "precision", "f1", "fpr", "roc_auc")}  # fmt: skip
        avg = {m: (sum(v) / len(v) if v else None) for m, v in vals.items()}
        lines.append(f"| {s['name']} | {fmt(avg['recall'])} | {fmt(avg['precision'])} | {fmt(avg['f1'])} | "
                     f"{fmt(avg['fpr'])} | {fmt(avg['roc_auc'], False)} |")  # fmt: skip
    lines.append("")
    for key in [d["key"] for d in res["datasets"]]:
        spec = next(d for d in res["datasets"] if d["key"] == key)
        lines += [f"### {spec['repo']}", "", f"{spec['description']}. {spec['note']}", "",
                  "| System | Recall % | Precision % | F1 % | FPR % | ROC-AUC | TP / FP / TN / FN |",
                  "|---|---|---|---|---|---|---|"]  # fmt: skip
        for s in systems:
            m = s["per_dataset"][key]
            dagger = " †" if key in s.get("trained_on", []) else ""
            lines.append(f"| {s['name']}{dagger} | {fmt(m['recall'])} | {fmt(m['precision'])} | {fmt(m['f1'])} | "
                         f"{fmt(m['fpr'])} | {fmt(m['roc_auc'], False)} | {m['tp']} / {m['fp']} / {m['tn']} / {m['fn']} |")  # fmt: skip
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def write_doc(block: str) -> None:
    DOC.parent.mkdir(exist_ok=True)
    if DOC.exists() and START in (text := DOC.read_text(encoding="utf-8")):
        before, rest = text.split(START, 1)
        after = rest.split(END, 1)[1] if END in rest else ""
        DOC.write_text(before + block + after, encoding="utf-8")
    else:
        DOC.write_text("# External benchmark\n\n" + block + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="device for the guard models")
    ap.add_argument("--skip-guards", action="store_true")
    ap.add_argument("--render-only", action="store_true", help="re-render the document from the stored JSON")
    args = ap.parse_args()
    if args.render_only:
        write_doc(render(json.loads((get_settings().results_dir / "external_benchmark.json").read_text(encoding="utf-8"))))
        print(f"re-rendered {DOC}")
        return
    if not DATA_DIR.exists():
        sys.exit("data/external/ is missing: run `uv run --extra guards python scripts/download_external.py` first")

    s = get_settings()
    datasets = {d.key: load_dataset(d, DATA_DIR) for d in DATASETS}
    print({k: len(v) for k, v in datasets.items()}, file=sys.stderr)
    overlap = overlap_with_sentinel_data(datasets)
    print(f"overlap with SentinelAI data: {overlap}", file=sys.stderr)

    classifier = MlClassifier.load(s.classifier_dir)
    gateway = SecurityGateway(SecurityPolicy(), MockModel(), NullAuditLogger(), SecretsConfig.load(s.protected_secrets_path),
                              classifier=classifier, max_input_chars=1_000_000)  # fmt: skip
    systems = [
        {"key": "sentinel_rules", "name": "SentinelAI rules", "run": sentinel_scorer(gateway, True, False), "params": "34 regex rules"},
        {"key": "sentinel_ml", "name": "SentinelAI classifier only", "run": sentinel_scorer(gateway, False, True), "params": "TF-IDF + LR (~0.3 MB)"},
        {"key": "sentinel_rules_ml", "name": "**SentinelAI rules + classifier**", "run": sentinel_scorer(gateway, True, True), "params": "rules + TF-IDF + LR"},
    ]  # fmt: skip
    if not args.skip_guards:
        import torch

        device = (
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else (args.device if args.device != "auto" else "cpu")
        )
        for g in GUARD_MODELS:
            run, info = guard_scorer(MODEL_DIR / g.key, device)
            systems.append({"key": g.key, "name": f"{g.repo.split('/')[1]} ({g.repo.split('/')[0]})", "run": run,
                            "params": g.params, "trained_on": list(g.trained_on), "label_info": info})  # fmt: skip

    results = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "overlap": overlap,
        "datasets": [],
        "systems": [],
    }
    for d in DATASETS:
        cases = datasets[d.key]
        results["datasets"].append({"key": d.key, "repo": d.repo, "license": d.license, "description": d.description,
                                    "note": d.note, "n": len(cases), "positives": sum(c.label for c in cases),
                                    "negatives": sum(1 - c.label for c in cases)})  # fmt: skip
    for sysdef in systems:
        t0 = time.time()
        per, latencies, misses = {}, [], {}
        for d in DATASETS:
            cases = datasets[d.key]
            sc = sysdef["run"]([c.text for c in cases])
            per[d.key] = metrics([c.label for c in cases], sc)
            latencies += sc.latency_ms
            device = sc.device
            misses[d.key] = {
                "false_negatives": [c.id for c, f in zip(cases, sc.flagged, strict=True) if c.label and not f][:25],
                "false_positives": [c.id for c, f in zip(cases, sc.flagged, strict=True) if not c.label and f][:25],
            }
            print(
                f"  {sysdef['key']:18} {d.key:10} recall={per[d.key]['recall']} fpr={per[d.key]['fpr']} auc={per[d.key]['roc_auc']}",
                file=sys.stderr,
            )
        fair = [k for k in LABELLED if k not in sysdef.get("trained_on", [])]
        macro = {
            m: statistics.fmean([per[k][m] for k in fair if per[k][m] is not None])
            for m in ("recall", "precision", "f1", "fpr", "roc_auc")
        }
        lat_sorted = sorted(latencies)
        results["systems"].append({
            "key": sysdef["key"], "name": sysdef["name"], "params": sysdef["params"],
            "trained_on": sysdef.get("trained_on", []), "label_info": sysdef.get("label_info"),
            "macro": macro, "macro_over": fair, "per_dataset": per, "misses": misses,
            "latency": {"p50": lat_sorted[len(lat_sorted) // 2], "p95": lat_sorted[int(len(lat_sorted) * 0.95)], "device": device},
            "seconds": round(time.time() - t0, 1),
        })  # fmt: skip

    s.results_dir.mkdir(exist_ok=True)
    (s.results_dir / "external_benchmark.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_doc(render(results))
    print(render(results))


if __name__ == "__main__":
    main()
