"""Choose the cascade parameters from pool data only. The benchmark is not opened here."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import FOLDS, OUT, ROOT, load_pool  # noqa: E402

df = load_pool()
comp = np.load(OUT / "components_pool.npz")
y_all, src_all = df.label.to_numpy(), df.source.to_numpy()
BENIGN = ["inthewild", "wambosec", "dolly", "alespalla", "llmail"]
ATTACK = ["wambosec", "inthewild", "llmail", "yanismiraoui", "mosscap"]


def view(idx, dsmall):
    return {"y": y_all[idx], "src": src_all[idx], "rule": comp["rule"][idx], "guard": comp["protectai"][idx], "d": dsmall}


c = np.load(OUT / "val_dsmall_final_calib.npz")
calib = view(c["idx"], c["score"])
oof_parts = [np.load(OUT / f"val_dsmall_{f}.npz") for f in FOLDS]
oof = view(np.concatenate([p["idx"] for p in oof_parts]), np.concatenate([p["score"] for p in oof_parts]))


def flag(v, t_d, t_low, t_g=0.5):
    return (v["rule"] >= 0.35) | (v["d"] >= t_d) | ((v["guard"] >= t_g) & (v["d"] >= t_low))


def stats(v, f):
    hn = v["src"] == "hardneg"
    return {
        "recall": float(np.mean([f[(v["src"] == a) & (v["y"] == 1)].mean() for a in ATTACK if ((v["src"] == a) & (v["y"] == 1)).any()])),
        "fpr": float(np.mean([f[(v["src"] == b) & (v["y"] == 0)].mean() for b in BENIGN if ((v["src"] == b) & (v["y"] == 0)).any()])),
        "hardneg_fpr": float(f[hn].mean()),
    }


def calib_threshold(v, target):
    """Lowest dsmall threshold keeping the macro benign FPR at or below target on the calibration set."""
    for t in np.quantile(v["d"], np.linspace(0, 1, 4001)):
        f = v["d"] >= t
        if stats(v, f)["fpr"] <= target:
            return float(t)
    return 0.999


t_d = calib_threshold(calib, 0.02)
sb = calib["src"] == "short_benign"
print("short_benign calibration: n", int(sb.sum()), "flagged at t_d:", round(float((calib["d"][sb] >= t_d).mean()), 3))
print("t_d (2% macro benign FPR on calibration) =", round(t_d, 4))
print(f"{'t_low':>7} | calib recall fpr hardneg | out-of-source recall fpr hardneg")
rows = []
for t_low in [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01]:
    a, b = stats(calib, flag(calib, t_d, t_low)), stats(oof, flag(oof, t_d, t_low))
    rows.append((t_low, a, b))
    print(f"{t_low:7.3f} | {a['recall']:.3f} {a['fpr']:.3f} {a['hardneg_fpr']:.3f} | {b['recall']:.3f} {b['fpr']:.3f} {b['hardneg_fpr']:.3f}")
valid = [r for r in rows if r[1]["fpr"] <= 0.06 and r[1]["hardneg_fpr"] <= 0.03 and r[2]["hardneg_fpr"] <= 0.05 and r[0] <= 1]
best = max(valid, key=lambda r: r[2]["recall"]) if valid else None
print("chosen t_low:", best[0] if best else None, "(max out-of-source recall subject to FPR <=6%, hardneg <=3% on calibration, <=5% out-of-source)")
policy = {"t_rule": 0.35, "t_d": t_d, "t_low": best[0], "t_g": 0.5}
(ROOT / "models/research/policy.json").write_text(json.dumps(policy, indent=1))
(OUT / "policy_selection.json").write_text(json.dumps({"policy": policy, "grid": [{"t_low": r[0], "calib": r[1], "out_of_source": r[2]} for r in rows]}, indent=1))
