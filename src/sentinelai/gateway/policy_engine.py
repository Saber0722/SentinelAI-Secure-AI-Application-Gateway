"""Combines detector findings into one deterministic action.

Precedence (highest wins): BLOCK > REVIEW > REDACT_AND_ALLOW > ALLOW_WITH_WARNING > ALLOW.

Input rules
  1. Attack signals from the rule detector (injection, jailbreak, extraction, exfiltration,
     obfuscation) are combined with a noisy-OR into one rule score.
     score >= injection_block_threshold -> BLOCK; >= injection_warn_threshold -> WARN.
  2. ML classifier probability (if enabled): >= block threshold -> BLOCK; >= warn -> WARN.
  3. Corroboration: if rules AND the classifier each independently reach WARN -> BLOCK.
  4. Credentials / protected secrets in the prompt -> ``input_secret_action``.
  5. PII -> ``pii_input_action``.
Output rules
  6. Protected secret, canary token or system-prompt reproduction -> BLOCK. Credential
     patterns -> ``output_secret_action`` (REDACT falls back to BLOCK when a match has no exact
     span, e.g. an encoded secret that cannot be cut out precisely).
  7. Denylist match -> BLOCK.
  8. PII -> ``pii_output_action``.
"""

from __future__ import annotations

from sentinelai.config import OutputSecretAction, PiiAction, SecurityPolicy
from sentinelai.gateway.decisions import (
    ATTACK_CATEGORIES,
    Action,
    Category,
    Decision,
    Finding,
    combine_scores,
    strongest,
)

_PII_ACTION = {
    PiiAction.ALLOW: Action.ALLOW,
    PiiAction.REDACT: Action.REDACT_AND_ALLOW,
    PiiAction.BLOCK: Action.BLOCK,
    PiiAction.REVIEW: Action.REVIEW,
}

PII_RISK = 0.4


def _threshold_action(score: float, block: float, warn: float) -> Action:
    if score >= block:
        return Action.BLOCK
    if score >= warn:
        return Action.ALLOW_WITH_WARNING
    return Action.ALLOW


def decide_input(
    findings: list[Finding],
    policy: SecurityPolicy,
    ml_block_threshold: float | None = None,
    ml_warn_threshold: float | None = None,
) -> Decision:
    actions: list[Action] = []
    reasons: list[str] = []
    risk = 0.0

    attack = [f for f in findings if f.category in ATTACK_CATEGORIES]
    rule_action = Action.ALLOW
    if attack:
        rule_score = combine_scores([f.score for f in attack])
        risk = max(risk, rule_score)
        rule_action = _threshold_action(rule_score, policy.injection_block_threshold, policy.injection_warn_threshold)
        cats = ", ".join(sorted({f.category.value for f in attack}))
        reasons.append(f"Rule score {rule_score:.2f} ({cats}) -> {rule_action.value}")
        actions.append(rule_action)

    ml = [f for f in findings if f.category is Category.ML_SUSPICIOUS]
    if ml and ml_block_threshold is not None and ml_warn_threshold is not None:
        p = max(f.score for f in ml)
        risk = max(risk, p)
        ml_action = _threshold_action(p, ml_block_threshold, ml_warn_threshold)
        reasons.append(f"Classifier probability {p:.2f} -> {ml_action.value}")
        actions.append(ml_action)
        if (
            rule_action is not Action.ALLOW
            and ml_action is not Action.ALLOW
            and Action.BLOCK
            not in (
                rule_action,
                ml_action,
            )
        ):
            reasons.append("Rules and classifier both flagged the prompt -> BLOCK (corroboration)")
            actions.append(Action.BLOCK)

    secrets = [f for f in findings if f.category in (Category.SECRET, Category.SYSTEM_PROMPT_LEAK)]
    if secrets:
        action = _PII_ACTION[policy.input_secret_action]
        risk = max(risk, max(f.score for f in secrets))
        reasons.append(f"Credential/secret material in prompt -> {action.value}")
        actions.append(action)

    pii = [f for f in findings if f.category is Category.PII]
    if pii:
        action = _PII_ACTION[policy.pii_input_action]
        risk = max(risk, PII_RISK)
        types = ", ".join(sorted({f.entity_type or "?" for f in pii}))
        reasons.append(f"PII detected ({types}) -> {action.value}")
        actions.append(action)

    final = strongest(actions)
    if not reasons:
        reasons.append("No detector raised a signal")
    return Decision(action=final, risk_score=round(risk, 4), findings=findings, reasons=reasons)


def decide_output(findings: list[Finding], policy: SecurityPolicy) -> Decision:
    actions: list[Action] = []
    reasons: list[str] = []
    risk = 0.0

    for f in findings:
        if f.category is Category.SYSTEM_PROMPT_LEAK or (
            f.category is Category.SECRET and f.entity_type == "PROTECTED_SECRET"
        ):
            actions.append(Action.BLOCK)
            reasons.append(f"{f.evidence} -> BLOCK")
            risk = max(risk, f.score)
        elif f.category is Category.SECRET:
            if policy.output_secret_action is OutputSecretAction.REDACT and f.spans:
                actions.append(Action.REDACT_AND_ALLOW)
            else:
                actions.append(Action.BLOCK)
            reasons.append(f"Credential pattern in response ({f.entity_type}) -> {actions[-1].value}")
            risk = max(risk, f.score)
        elif f.category is Category.POLICY_VIOLATION:
            actions.append(Action.BLOCK)
            reasons.append(f"Output policy violation: {f.evidence} -> BLOCK")
            risk = max(risk, f.score)
        elif f.category is Category.PII:
            action = _PII_ACTION[policy.pii_output_action]
            actions.append(action)
            reasons.append(f"PII in response ({f.entity_type}) -> {action.value}")
            risk = max(risk, PII_RISK)

    if not reasons:
        reasons.append("No detector raised a signal")
    return Decision(action=strongest(actions), risk_score=round(risk, 4), findings=findings, reasons=reasons)
