# Dashboard guide: what every number means

The dashboard (`http://127.0.0.1:8501/dashboard`, opened by `./start.sh`) only **reports** what the gateway
has recorded. It never runs a security check itself. All values come from the API, which reads three tables
in the local SQLite database `var/sentinel.db`:

| Table | Written by | Feeds |
|---|---|---|
| `events` | every request to `/api/v1/chat` and `/api/v1/scan`, plus invalid requests and policy changes | header counts, **Live traffic** tab |
| `eval_runs` | `scripts/evaluate_security.py` or the **Run benchmark now** button | **Overview** effectiveness section, **Security benchmark** tab |
| `inference_runs` | `scripts/benchmark_inference.py` and `evaluate_security.py --live N` | **Overview** LLM/GPU cards, **Inference & hardware** tab |

Responses are cached for 5 seconds. Click **Refresh** for fresh data, or turn on **Auto-refresh** in the Live
traffic tab. The external real-world benchmark is not on the dashboard; see `external_benchmark.md`.

---

## Header strip (all tabs)

| Item | Meaning |
|---|---|
| **Gateway** v0.1.0 · online · *N* detectors | API reachable; *N* detectors loaded (rules, secrets, Presidio PII, output policy, ML classifier). |
| **Model** `name` · available · backend · device | The active model. *device* is where Ollama currently holds it: `gpu`, `cpu`, `gpu+cpu (x% in VRAM)`, or **not loaded**. Not loaded is normal: Ollama unloads an idle model after about 5 minutes and reloads it on the next request. |
| **GPU** used/total MiB · util % · °C | Live `nvidia-smi` readings for the **whole GPU**, including other programs such as your browser or other Ollama models. They're re-read every 3–5 seconds. |
| **ML classifier** enabled · version | Whether the trained classifier is loaded and active in the current policy. |

---

## Overview tab

### Live traffic

The traffic KPIs are counted over **all time** from the `events` table.

| Number | Definition |
|---|---|
| **Requests** | Chat requests + scan requests + rejected (invalid) requests. Policy changes are not counted. |
| **Blocked / held** | `blocked` (stopped before the model) + `output_blocked` (answer withheld) + `review` (held). The small badge is that count as a share of all requests. |
| **Redacted** | Requests whose prompt *or* answer had personal data or credentials replaced with placeholders. |
| **High-risk events** | Requests whose risk level is `high` or `critical` (see *Risk level* below). |
| **Gateway latency p50** | Median time the security pipeline itself took, **excluding** model inference. |

### Security effectiveness (holdout split)

These numbers come from the **latest stored benchmark run** on the holdout split: 40 attacks and 40 benign
prompts written after the rules were frozen, run through a deterministic, deliberately vulnerable **mock**
model. They don't come from live traffic. The headline uses the *Rules + ML* configuration.

| Number | Definition | Good value |
|---|---|---|
| **Attack recall** | Attacks *flagged* (warned, held or blocked by attack signals) ÷ all attacks. The badge shows the gain over rules alone. | high |
| **Precision** | Attacks ÷ everything flagged as an attack. | high |
| **Benign FPR** | Benign prompts flagged ÷ all benign prompts. Redacting PII does **not** count as a flag. | low |
| **Secrets leaked** | Attacks whose *returned* answer contained a protected synthetic secret, verbatim or with separators changed. The badge shows the unprotected baseline. | 0 |
| **PII reaching model** | PII prompts where at least one labelled personal value reached the model unredacted. The badge shows the baseline (10/10). | low |

- **Controls disabled vs enabled** shows those five rates side by side for *Baseline (no controls)*, *Rules
  only* and *Rules + ML*. Colours identify the configuration and are the same in every chart.
- **Project targets** compares results with the specification's targets: recall ≥ 85%, precision ≥ 80%, FPR
  ≤ 10%, zero secret leakage. ✅ means met and ❌ means missed. Rules alone miss the recall target.
- **Real local LLM: secrets leaked to the user** comes from the latest live run, where real holdout attacks
  were sent to the real model (e.g. `qwen2.5:1.5b` on GPU). *Secrets returned to user* counts answers that
  contained a protected secret; *Blocked before the model* counts prompts the gateway stopped. With no
  controls the real model leaked 6 times; with controls, 0.
- **Inference throughput: GPU vs CPU** shows tokens generated per second (median over warm requests) on
  each device, and the speed-up ratio.

---

## Live traffic tab

Use the **Time range** filter (all time, last hour, 24 hours, 7 days) and **Auto-refresh** (every 5 s, this
tab only).

| Item | Definition |
|---|---|
| **Requests** | As on the Overview, within the selected range. |
| **Allowed** | `allowed` + `allowed_with_warning`: the model's answer was shown. |
| **Blocked (input)** | `blocked` + `review`: the model was never called. |
| **Blocked (output)** | `output_blocked`: the model answered, but the answer contained something the output policy forbids, so it was withheld. |
| **Redacted** | Answered, with personal data or credentials replaced. |
| **Rejected / errors** | `rejected` (empty, too long or malformed) + `inference_error` (checks passed, model failed). |
| **Outcome by status** | The same counts as a chart; colours are fixed per status. |
| **Requests over time** | Requests per hour (UTC), stacked by status. |

### Risk level

Every request gets a risk score from 0 to 1. It is the **highest** of the combined attack-rule score, the
classifier probability (counted only when it reaches the warn threshold), the secret findings, and a fixed
0.4 when personal data is present. The level is derived from that score:

| Level | Score | Typical cause |
|---|---|---|
| none | no finding at all | ordinary question |
| low | < 0.35 | a very weak signal |
| medium | 0.35 – 0.70 | personal data present (0.4), or a weak attack pattern → warning |
| high | 0.70 – 0.90 | clear attack pattern → blocked |
| critical | ≥ 0.90 | several attack signals, a classifier probability ≥ 0.9, or a protected secret |

Medium is therefore often just "contained an email address", not an attack.

### Other cards

| Card | Definition |
|---|---|
| **Detection categories** | Number of requests in which each finding category occurred. A request can count in several categories. `prompt_injection`, `jailbreak`, `system_prompt_extraction`, `secret_exfiltration` and `obfuscation` come from the rules; `ml_suspicious` from the classifier; `pii` from Presidio; `secret` from credential patterns; `system_prompt_leak` from output scanning. |
| **PII entities detected** | Total personal-data *items* found, per type, in prompts and answers. One prompt with two email addresses counts 2. PERSON detection is probabilistic, so a few names are false positives. |
| **Latency: gateway p50 / p95** | Median and 95th-percentile time spent in the security pipeline, excluding the model. |
| **Latency: model p50 / p95** | The same for model inference, counted only for requests that reached the model. About 0.4 s p50 on the GPU. |
| **Recent high-risk events** | Up to 15 latest `high`/`critical` requests: time (UTC), status, risk score (bar), categories, input and output actions, gateway time. No prompt text is stored, so none is shown. |

---

## Security benchmark tab

Choose the **split** with the toggle: *holdout* (trustworthy) or *dev* (the rules were debugged on it, so its
100% scores are optimistic). The numbers come from the latest benchmark batch with the mock model.

| Item | Definition |
|---|---|
| **Configuration cards** | Per configuration: recall, precision and FPR (as above); **F1** (harmonic mean of precision and recall); **block recall** (attacks actually *blocked*, not just warned); secrets leaked; PII reaching the model. |
| **Recall by attack category** | Share of attacks in each category that were flagged; *n* is the number of prompts in that category. A low bar shows which attack style is weak, e.g. `non english`. |
| **Confusion matrices** | Rows are the true label, columns the gateway's verdict. **TP** = attacks flagged, **FN** = attacks missed, **FP** = benign prompts wrongly flagged, **TN** = benign prompts passed. Darker cells hold larger counts. |
| **Risk-score separation** | One dot per prompt at its risk score. Filled dots are attacks, hollow dots benign. Dots are spread vertically only so identical scores stay visible; the vertical position has no meaning. The dotted lines are the rule *warn* (0.35) and *block* (0.70) thresholds. Ideally attacks sit to the right and benign prompts at 0. Benign dots at 0.40 are PII prompts, redacted but not flagged. Attack dots at 0 are misses. |
| **Output suite** (dev split) | 22 harmless prompts answered with scripted leaky responses. *Expected action matched* = share of cases where the output filter acted as labelled; *secret cases leaked* = secret/credential/system-prompt responses that reached the user. |
| **Edge cases & operations** | *Edge cases*: empty input, too long, invisible-only, emoji, Arabic, Chinese, control characters, buried and Zalgo-text injections, handled as expected. *blocked → model* = blocked requests that still reached the model (must be 0). *Benign answered* = benign prompts that got an answer; the mock model occasionally produces a leak-like answer to an innocent prompt, which is then withheld. |
| **Gateway latency in benchmark** | Pipeline time during the benchmark. The baseline is near 0 because every check is off. |
| **Misclassified and missed cases** | IDs of every false negative, false positive, PII miss and spurious redaction. Look them up in `data/evaluation/`. |

---

## ML classifier tab

| Item | Definition |
|---|---|
| **Block threshold / Warn threshold** | Probabilities at which the classifier alone blocks (0.95) or warns (0.93). They were chosen on validation data for ≤ 2% / ≤ 5% false positives, deliberately conservative. |
| **Test ROC-AUC** | How well the classifier ranks attacks above benign prompts on **held-out template families** it never saw (1.0 is perfect, 0.5 is chance). |
| **Corpus rows** | Size of the template-generated training corpus after removing near-duplicates of evaluation prompts, split into training and held-out test families. |
| **Classifier alone, at the warn threshold** | Precision, recall, F1 and FPR of the classifier *without the rules* on three held-out sets: template test families, the hand-written dev set and the hand-written holdout set. |
| **Table** | The same metrics at both thresholds, plus *Validation (grouped CV)*: out-of-fold predictions on the training families, which were used to choose the thresholds. |
| **Test families confusion matrix** | Results at the block threshold on the held-out families. |
| **Family split** | Which template families were used for training and testing. Whole families are held out, so the test measures unseen attack styles. |

These numbers are measured on SentinelAI's *own* data. On public real-world data the classifier's ROC-AUC is
0.82–0.94; see `external_benchmark.md`.

---

## Inference & hardware tab

| Item | Definition |
|---|---|
| **GPU (live)** | VRAM used/total for the whole GPU, current utilisation and temperature (`nvidia-smi`). |
| **Active model / placement** | Model the gateway is using and where Ollama holds it right now. |
| **GPU / CPU tokens/s** | Median of *output tokens ÷ inference seconds* over the warm requests of the latest inference benchmark on that device. *placement* confirms where the model actually ran. |
| **GPU speed-up** | GPU tokens/s ÷ CPU tokens/s. It was 5.5× on the RTX 4060. |
| **Inference p50 / p95** | Median and 95th-percentile model time per request on warm runs. |
| **Cold first request** | Time of the first request after the model was unloaded. It includes loading the model from disk, so it is similar on GPU and CPU. |
| **Table: gateway p50/p95** | Security-pipeline time in the same run. The detectors always run on CPU, so the device doesn't change it. |
| **Table: VRAM for model** | GPU memory in use after loading, minus before. It is approximate because other programs also use the GPU. |
| **Live security run on the real model** | From `evaluate_security.py --live N`: holdout attacks and benign prompts sent to the real model. *Model output contained a secret* is the raw answer **before** output filtering. *Secret returned to user* is what the user actually received. *Blocked before model* counts prompts stopped at input, including false positives. *Output blocked* counts answers withheld. Real-model output varies between runs and devices, so these are illustrative; the mock benchmark is the reproducible one. |

---

## Refreshing the numbers

| To update | Run |
|---|---|
| Live traffic | use the **Chat** page or `POST /api/v1/chat` |
| Security benchmark | **Run benchmark now** (admin token from `./start.sh`), or `uv run python scripts/evaluate_security.py` |
| Real-model security run | `uv run python scripts/evaluate_security.py --live 20` (Ollama running) |
| GPU vs CPU | `uv run python scripts/benchmark_inference.py` |
| Start with an empty traffic log | stop the app and delete `var/sentinel.db*` (benchmark runs are stored there too, so re-run them) |
