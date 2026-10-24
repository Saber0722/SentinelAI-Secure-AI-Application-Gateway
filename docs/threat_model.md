# Threat model

## System and assets

The protected system is a customer-support assistant ("AcmeAssist") whose system prompt contains
confidential configuration: four synthetic credentials and a canary token. This deliberately models a common
real-world mistake, putting secrets in the prompt, so that leakage can be measured.

| Asset | Why it matters |
|---|---|
| Protected secrets and the system prompt | Disclosure equals credential compromise / loss of proprietary instructions. |
| User PII in prompts | Should not be sent to (or retained by) the model when policy says redact. |
| PII and credentials in model output | The model may surface data from its context or training. |
| Audit log | Must be useful for investigation without becoming a second copy of sensitive data. |
| Security policy | If a client could change it, every other control is moot. |

## Trust boundaries

1. **Client → gateway.** Untrusted. Every byte of the request is attacker-controlled.
2. **Gateway → model.** The model is *not* trusted to enforce policy: it can be manipulated, and its
   refusals are not a control. The output is treated as untrusted too.
3. **Gateway → disk** (SQLite, logs). Local, but treated as lower-trust storage: no raw prompts, no secrets.
4. **Operator → gateway** (admin API, `.env`). Trusted, authenticated by `SENTINEL_ADMIN_TOKEN`.

## Adversaries

- **A1 — Malicious end user.** Crafts direct injections, jailbreaks, extraction and exfiltration prompts, and
  obfuscates them (encoding, homoglyphs, invisible characters, other languages).
- **A2 — Indirect injector.** Plants instructions in content the user asks the model to process (reviews,
  emails, documents).
- **A3 — Curious or careless user.** Pastes PII or credentials into prompts.
- **A4 — Local attacker with UI access but no admin token.** Tries to switch the policy off.

Out of scope: an attacker with shell access to the host, who can read `.env`, the database and the model
directly; supply-chain compromise of dependencies; denial of service beyond basic size limits.

## Threats and controls

| # | Threat (STRIDE) | Adversary | Controls | Verified by |
|---|---|---|---|---|
| T1 | Instruction override / prompt injection (Tampering) | A1, A2 | Rule detector; ML classifier; input wrapped in `<user_input>` delimiters (defence in depth only) | `test_rules.py`, benchmark per-category recall |
| T2 | Jailbreak personas, refusal suppression | A1 | Rule detector (JB-*), classifier, corroboration rule | benchmark |
| T3 | System-prompt extraction (Information disclosure) | A1, A2 | SPE-* rules; output canary check; 6-word-shingle overlap check | `test_system_prompt_reproduction_blocked`, output suite |
| T4 | Secret exfiltration | A1, A2 | SX-* rules; secret-identifier detection; output scanning of protected values in 5 forms; credential patterns | `test_protected_secrets_never_returned_for_any_attack`, output suite (0/12 leaks) |
| T5 | Obfuscated attacks | A1 | NFKC, invisible-character removal, homoglyph/leet skeleton, letter-joining, mark stripping, base64 decoding | `test_normalize.py`, obfuscated/edge cases |
| T6 | PII sent to the model | A3 | Presidio + redaction before inference; configurable ALLOW/REDACT/BLOCK/REVIEW | `test_redaction_happens_before_inference` |
| T7 | PII/credentials in model output | — | Output PII redaction and credential blocking | output suite |
| T8 | Sensitive data in audit logs or errors (Information disclosure) | — | No raw prompts by default; HMAC fingerprints; safe evidence strings; validation errors don't echo input; 500s return no message; log scrub filter | `test_audit_database_contains_no_sensitive_content`, `test_validation_error_does_not_echo_input`, `test_unhandled_errors_do_not_leak` |
| T9 | Policy bypass from the client (Elevation of privilege) | A4 | `extra="forbid"` request schemas; policy only via the token-protected admin API; constant-time token comparison; policy changes are audited | `test_client_cannot_override_policy`, admin tests |
| T10 | Data leaving the machine | — | Model URL must be loopback unless explicitly allowed; Streamlit telemetry disabled; no network calls in the default path | `test_remote_model_urls_rejected` |
| T11 | Oversized or malformed input (DoS-lite) | A1 | 512 KB body cap, schema length cap, configurable 8,000-char limit (413), type validation | `test_validation_errors`, edge cases |
| T12 | Tampered classifier artifact | local | SHA-256 check against metadata (integrity only; pickle still trusted) | `test_tampered_artifact_rejected` |
| T13 | Model failure | — | Explicit `inference_error` status; scan endpoint still works; adapter exceptions fail closed and log only the exception type | `test_settings_switch_model`, `test_unexpected_adapter_exception_fails_closed` |

## Residual risk

- **Novel phrasings evade detection.** Holdout recall is 50% with rules alone and 85% with the classifier;
  15% of unseen attacks still pass. The output pipeline is the safety net for *secret* leakage specifically,
  which is why secret leakage stays at 0, but not for other harms such as persona adoption.
- **Output scanning only knows what it is told.** It catches configured secrets, credential *patterns* and
  verbatim system-prompt reproduction. A paraphrased summary of the system prompt, or a secret
  re-encoded in a novel way (translated, split across messages), would pass.
- **PII detection is probabilistic.** It misses obfuscated e-mail addresses, unusual formats and many names,
  and it is English-only.
- **The admin token is a single shared secret** with no rate limiting or rotation; it is adequate for a
  local single-operator tool, not for multi-user deployment.
- **Pickle-based classifier artifacts** execute code on load; the integrity check does not protect against an
  attacker who can write to `models/`.

See `limitations.md` for the full list.
