# data/

All data is **synthetic**. No real credentials or personal information are used anywhere.

| Path | Purpose |
|---|---|
| `protected_secrets.json` | Synthetic secrets + canary token embedded in the demo app's system prompt. The gateway must never return them. |
| `evaluation/benign.jsonl` | Dev split: 66 benign prompts (25 everyday, 25 hard negatives with attack vocabulary, 16 with synthetic PII). |
| `evaluation/malicious.jsonl` | Dev split: 64 attacks across 9 categories. |
| `evaluation/output_cases.jsonl` | 22 scripted model responses (secrets, credentials, system-prompt leaks, PII, benign) for output-scanning tests. |
| `evaluation/edge_cases.jsonl` | 13 edge cases: empty, whitespace, invisible-only, too long, non-string, null, emoji, RTL, CJK, control chars, buried and Zalgo injections. |
| `evaluation/holdout/` | Holdout split: 40 benign + 40 malicious prompts written **after** the rules were frozen; never used for tuning. |
| `training/prompts.jsonl` | Template-generated classifier corpus (`scripts/generate_training_data.py --seed 42`), 1,217 rows in 29 families. |

## Record formats

Prompt cases: `{"id", "text", "label": "benign"|"malicious", "category", "sensitive_values"?: [...]}`
— `sensitive_values` lists the synthetic PII strings that must not reach the model.

Output cases: `{"id", "category", "prompt", "model_response", "expected": "BLOCK"|"REDACT_AND_ALLOW"|"ALLOW", "sensitive_values"}`

Edge cases: `{"id", "kind", "text" | "repeat": {"chunk", "times"}, "suffix"?, "expect": "reject"|"handled"|"blocked"}`

Dataset versions are content hashes (e.g. `sentinelai-eval-holdout@c992f73dda3d`) computed at load time and
recorded with every benchmark run.

## Honest caveats

The same person wrote the rules, the dev split, the holdout split and the training templates. The holdout was
written after the rules were frozen, but it still reflects one author's idea of what attacks look like;
independent red-team data would be a stronger test. See `docs/limitations.md`.
