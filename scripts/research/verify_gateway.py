"""Check that the real gateway reproduces the frozen cascade. Parameters are not changed by anything here."""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, ROOT  # noqa: E402

from sentinelai.audit import NullAuditLogger  # noqa: E402
from sentinelai.config import SecurityPolicy, get_settings  # noqa: E402
from sentinelai.detectors.cascade import load_detector  # noqa: E402
from sentinelai.detectors.pii import PiiDetector  # noqa: E402
from sentinelai.detectors.secrets import SecretsConfig  # noqa: E402
from sentinelai.evaluation.external import DATASETS, load_dataset  # noqa: E402
from sentinelai.evaluation.report import compact, run_suite  # noqa: E402
from sentinelai.gateway.decisions import ATTACK_CATEGORIES, Action, Category  # noqa: E402
from sentinelai.gateway.pipeline import InputValidationError, SecurityGateway  # noqa: E402
from sentinelai.gateway.policy_engine import decide_input  # noqa: E402
from sentinelai.models import MockModel  # noqa: E402

s = get_settings()
assert s.classifier_kind == "cascade"
clf = load_detector(s)
sec = SecretsConfig.load(s.protected_secrets_path)
gw = SecurityGateway(SecurityPolicy(), MockModel(), NullAuditLogger(), sec, classifier=clf, max_input_chars=1_000_000)
pol = SecurityPolicy(input_rules_enabled=True, classifier_enabled=True, pii_input_enabled=False, input_secret_action="ALLOW")
ref = json.loads((OUT / "final_eval_second.json").read_text())["systems"]["CASCADE: rules | DeBERTa | ProtectAI with veto"]["per_dataset"]
rows, t0 = {}, time.time()
for d in DATASETS:
    tp = fp = fn = tn = blocked_atk = blocked_ben = 0
    for c in load_dataset(d, ROOT / "data/external"):
        try:
            a = gw.analyze_input(c.text, pol)
            rel = [f for f in a.decision.findings if f.category in ATTACK_CATEGORIES or f.category is Category.ML_SUSPICIOUS]
            act = decide_input(rel, pol, clf.block_threshold, clf.warn_threshold).action
        except InputValidationError:
            act = Action.ALLOW
        flag, blk = act is not Action.ALLOW, act is Action.BLOCK
        tp += c.label and flag
        fn += c.label and not flag
        fp += (not c.label) and flag
        tn += (not c.label) and not flag
        blocked_atk += c.label and blk
        blocked_ben += (not c.label) and blk
    rows[d.key] = {"recall": tp / (tp + fn) if tp + fn else None, "fpr": fp / (fp + tn) if fp + tn else None, "block_recall": blocked_atk / (tp + fn) if tp + fn else None, "block_fpr": blocked_ben / (fp + tn) if fp + tn else None}
    r = rows[d.key]
    f = lambda v: "-" if v is None else f"{v * 100:.1f}"
    print(f"{d.key:10} recall {f(r['recall'])} (frozen {f(ref[d.key]['recall'])})  fpr {f(r['fpr'])} (frozen {f(ref[d.key]['fpr'])})  block-recall {f(r['block_recall'])}  block-fpr {f(r['block_fpr'])}", flush=True)
for name, keys in {"macro (deepset, safeguard, spml)": ["deepset", "safeguard", "spml"]}.items():
    print(name, {m: round(statistics.fmean(rows[k][m] for k in keys) * 100, 1) for m in ("recall", "fpr", "block_recall", "block_fpr")})
print("seconds", round(time.time() - t0))
print("\nInternal security suite with the cascade:")
suite = run_suite(s.evaluation_dir, sec, pii=PiiDetector(s.spacy_model), classifier=clf)
c = compact(suite)
for split, res in c["splits"].items():
    for label, cfg in res["configurations"].items():
        a = cfg["attack_detection"]
        print(f"{split:8} {label:13} recall={a['recall']} block={a.get('block_recall')} precision={a['precision']} fpr={a['false_positive_rate']} leaks={cfg['secret_leakage']['leaked']}/{cfg['secret_leakage']['n']} pii={cfg['pii_exposure']['reached_model']}/{cfg['pii_exposure']['n']} blocked_reached={cfg.get('blocked_reaching_model', cfg.get('operational'))} edge={cfg.get('edge_cases')} output={cfg.get('output_suite')}")
(OUT / "verify_gateway.json").write_text(json.dumps({"public": rows, "internal": c}, indent=1, default=str))
