"""Package the frozen cascade into models/cascade/ with checksums and its evaluation summary.

Inputs come from the research pipeline (scripts/research/). Only the frozen artefacts are used.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from sentinelai.detectors.ml_classifier import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "cascade"
SOURCES = {"own": ROOT / "models/research/dsmall_final", "guard": ROOT / "models/guards/protectai_v2"}
KEEP = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")


def main() -> None:
    policy = json.loads((ROOT / "models/research/policy.json").read_text())
    final = json.loads((ROOT / "results/research/final_eval_second.json").read_text())["systems"]
    benchmark = {}
    for label, key in {"Cascade": "CASCADE: rules | DeBERTa | ProtectAI with veto", "Own model + rules": "OWN: rules | DeBERTa (no third-party model)", "ProtectAI alone": "ProtectAI v2 alone (default 0.5)"}.items():
        s = final[key]
        benchmark[label] = {**s["macro"], "macro_over": s["macro_over"], "notinject_fpr": s["notinject_fpr"], "gandalf_recall": s["gandalf_recall"]}
    hashes = {}
    for sub, src in SOURCES.items():
        (OUT / sub).mkdir(parents=True, exist_ok=True)
        for name in KEEP:
            if (src / name).exists():
                shutil.copy2(src / name, OUT / sub / name)
                (OUT / sub / name).chmod(0o644)  # the container runs as a non-root user
        hashes[sub] = sha256_file(OUT / sub / "model.safetensors")
    guard_cfg = json.loads((SOURCES["guard"] / "config.json").read_text())
    index = next(int(k) for k, v in guard_cfg["id2label"].items() if "inject" in v.lower())
    meta = {
        "kind": "cascade",
        "model_version": f"cascade-{hashes['own'][:8]}",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "algorithm": "Rules, DeBERTa-v3-small trained on public data with trigger-word hard negatives, and ProtectAI v2 with a veto",
        "own_base_model": "microsoft/deberta-v3-small",
        "guard_model": "protectai/deberta-v3-base-prompt-injection-v2",
        "own_max_length": 256,
        "guard_malicious_index": index,
        "thresholds": {"own": policy["t_d"], "guard": policy["t_g"], "veto": policy["t_low"]},
        "weights_sha256": hashes,
        "training_sources": ["Lakera/mosscap_prompt_injection", "TrustAIRLab/in-the-wild-jailbreak-prompts", "microsoft/llmail-inject-challenge", "wambosec/prompt-injections", "yanismiraoui/prompt_injections", "alespalla/chatbot_instruction_prompts", "databricks/databricks-dolly-15k", "SentinelAI templates", "generated trigger-word hard negatives"],
        "public_benchmark": benchmark,
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {OUT}: {meta['model_version']}")


if __name__ == "__main__":
    main()
