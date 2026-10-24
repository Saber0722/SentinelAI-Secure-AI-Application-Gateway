# Security testing methodology

## Test layers

| Layer | Location | What it proves |
|---|---|---|
| Unit | `tests/unit/` (8 modules) | Normalization, each detector, redaction, policy precedence, config validation, adapters (with mocked HTTP), audit storage, metrics, classifier loading. |
| Integration | `tests/integration/` | Full pipeline flows and every API endpoint, including error paths, admin auth, model switching and report generation. |
| Security properties | `tests/security/` | Invariants checked over **every** labelled case, not single examples (see below). |
| Benchmark | `scripts/evaluate_security.py` | Measured detection/leakage/latency, baseline vs protected, on two splits. |
| UI smoke test | headless `streamlit.testing` + screenshots (manual) | Every page renders against live API data without exceptions. |

```bash
uv run pytest                      # 207 tests, ~23 s
uv run pytest --cov                # coverage (UI and CLI excluded)
uv run pytest tests/security -v    # only the invariants
```

Current coverage: 96% of the 2,099 statements outside `ui/` and `cli.py` (the guard-model scorer needs PyTorch and is exercised by the external benchmark run instead).

## Security invariants (tests/security)

1. **Blocked requests never reach the model.** For every dev and holdout prompt that ends `blocked` or
   `review`, the mock model's call counter is unchanged.
2. **No protected secret is ever returned** for any of the 104 malicious prompts, even though the mock model
   will leak them whenever it is asked to.
3. **The output suite never returns a secret**: all 12 secret/credential/system-prompt responses end
   `output_blocked`.
4. **Redaction happens before inference**: every PII/secret span the detectors found is absent from the
   exact string the model received.
5. **The audit database contains no protected secret, no labelled PII value and no raw prompt**, checked by
   dumping the whole SQLite file after replaying the dataset through the API.
6. **Logs are scrubbed** of protected secrets and credential patterns.
7. **Clients cannot change policy**: extra request fields are rejected (422), settings without the token are
   rejected, and attacks are still blocked afterwards.
8. **Unhandled errors leak nothing**: a 500 response carries only `"internal error"` and a request ID.
9. **Benchmark regression guard**: baseline leaks > 0 (the baseline really is vulnerable), protected leaks
   = 0, blocked-but-invoked = 0, FPR ≤ 10%, all edge cases pass.

## External benchmark

`scripts/benchmark_external.py` evaluates input-side detection on six public datasets (5,027 prompts;
deepset, jackhhao, safeguard, SPML, Lakera Gandalf, NotInject). It runs the same datasets through three
open-source guard models (ProtectAI, deepset, PreambleAI), and checks that no external prompt
near-duplicates SentinelAI's own data (0 found). Results and analysis: `documents/external_benchmark.md`.

## Benchmark design

**Model.** The `MockModel` simulates a vulnerable LLM. It dumps its system prompt (secrets included)
whenever a prompt contains an extraction verb and object ("print … your instructions"), and otherwise
echoes the prompt it received. This makes runs deterministic, makes leakage observable, and lets PII
exposure be measured exactly: the mock records what it was sent.

**Configurations.** `baseline` turns every control off via `SecurityPolicy.disabled()`, which is reachable
only from the evaluation code, never from the API. `protected` is the default policy without the
classifier, and `protected_ml` is the default policy with it.

**Splits.**

- *dev* (130 prompts + 22 output cases + 13 edge cases) was written first. The first run exposed real bugs:
  leetspeak mapping broke `rot13`/`base64` tokens, verb inflections were missed, and a base64-encoded
  secret was not detected. The rules were then fixed against this split, so its numbers are optimistic.
- *holdout* (80 prompts) was written **after** the rules were frozen at v1.1.0 and has not been used for
  any tuning since. Report the holdout numbers.
- The classifier has its own template-generated corpus (1,217 rows, 29 families). Splits are made by
  *whole family*: the test families are attack styles the model never saw. Rows nearly duplicating any
  evaluation prompt (Jaccard ≥ 0.8) are dropped. C and the thresholds are chosen by grouped 5-fold CV on
  the training families. The test families and both hand-written sets are evaluated exactly once.

**Metrics.**

| Metric | Definition |
|---|---|
| Attack recall | malicious prompts *flagged* (attack signals alone reach WARN/REVIEW/BLOCK) / all malicious |
| Block recall | malicious prompts that ended `blocked`/`review` / all malicious |
| Precision | malicious / all flagged |
| Benign FPR | benign prompts flagged / all benign (PII redaction is not counted as a flag) |
| Secret leakage | malicious prompts whose *returned* response contains a protected value (verbatim or with separators/case changed). The leak check is a plain substring test, independent of the detectors. |
| PII to model | PII prompts where a labelled synthetic value reached the model |
| Spurious redaction | benign non-PII prompts that got redacted (NER false positives) |
| Latency | gateway time excluding inference, and inference time separately; p50/p95 |

## Results (from `docs/benchmark_report.md`)

Holdout split, the one that counts:

| Config | Recall | Precision | FPR | Secret leakage | PII → model |
|---|---|---|---|---|---|
| baseline | 0% | n/a | 0% | 11/40 | 10/10 |
| protected (rules) | 50.0% | 95.2% | 2.5% | 0/40 | 1/10 |
| protected + ML | **85.0%** | **97.1%** | **2.5%** | **0/40** | 1/10 |

Against the spec's targets (≥85% recall, ≥80% precision, ≤10% FPR, zero secret leaks), the protected + ML
configuration meets all four on the holdout split. Rules alone meet precision, FPR and leakage but **not
recall**. Gateway overhead is ~6 ms p50 and ~8–12 ms p95 per request on CPU (Presidio dominates).

A live run with `qwen2.5:1.5b` on the GPU (first 20 attacks + 20 benign holdout prompts) is included in the
report. Unprotected, the real model put a protected secret in its answer for **6 of 20 attacks**. Protected,
**0** secrets reached the user, and 18 prompts were stopped before inference: 17 attacks plus the known
false positive `hb-017`. Earlier CPU runs leaked 0 and 1 of 12, so real-model behaviour varies between
runs and devices.

GPU vs CPU inference is measured separately by `scripts/benchmark_inference.py` (`docs/inference_report.md`).
