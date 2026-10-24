import pytest

from sentinelai.config import OutputSecretAction, PiiAction, SecurityPolicy
from sentinelai.gateway.decisions import Action, Category, Finding, Span, severity_for, strongest
from sentinelai.gateway.policy_engine import decide_input, decide_output


def f(category, score=0.9, spans=(), entity_type=None):
    return Finding("t", "1", category, score, severity_for(score), "e", spans=spans, entity_type=entity_type)


POLICY = SecurityPolicy()


def test_precedence_order():
    assert strongest([Action.ALLOW, Action.REDACT_AND_ALLOW, Action.ALLOW_WITH_WARNING]) is Action.REDACT_AND_ALLOW
    assert strongest([Action.REVIEW, Action.BLOCK]) is Action.BLOCK
    assert strongest([]) is Action.ALLOW


def test_no_findings_allows():
    d = decide_input([], POLICY)
    assert d.action is Action.ALLOW and d.risk_level == "none"


@pytest.mark.parametrize(("score", "action"), [(0.2, Action.ALLOW), (0.5, Action.ALLOW_WITH_WARNING), (0.8, Action.BLOCK)])
def test_rule_thresholds(score, action):
    assert decide_input([f(Category.PROMPT_INJECTION, score)], POLICY).action is action


def test_weak_signals_combine_to_block():
    d = decide_input([f(Category.JAILBREAK, 0.5), f(Category.PROMPT_INJECTION, 0.5)], POLICY)
    assert d.action is Action.BLOCK and d.risk_score == pytest.approx(0.75)


def test_block_beats_redaction():
    d = decide_input([f(Category.PROMPT_INJECTION, 0.9), f(Category.PII, 0.9)], POLICY)
    assert d.action is Action.BLOCK


def test_pii_action_mapping():
    pii = [f(Category.PII, 0.9)]
    assert decide_input(pii, POLICY).action is Action.REDACT_AND_ALLOW
    for setting, expected in [
        (PiiAction.ALLOW, Action.ALLOW),
        (PiiAction.BLOCK, Action.BLOCK),
        (PiiAction.REVIEW, Action.REVIEW),
    ]:
        assert decide_input(pii, SecurityPolicy(pii_input_action=setting)).action is expected


def test_classifier_thresholds_and_corroboration():
    ml = f(Category.ML_SUSPICIOUS, 0.6)
    assert decide_input([ml], POLICY, ml_block_threshold=0.9, ml_warn_threshold=0.5).action is Action.ALLOW_WITH_WARNING
    rule = f(Category.JAILBREAK, 0.4)
    d = decide_input([ml, rule], POLICY, ml_block_threshold=0.9, ml_warn_threshold=0.5)
    assert d.action is Action.BLOCK
    assert any("corroboration" in r for r in d.reasons)


def test_classifier_ignored_without_thresholds():
    assert decide_input([f(Category.ML_SUSPICIOUS, 0.99)], POLICY).action is Action.ALLOW


def test_input_secret_action():
    d = decide_input([f(Category.SECRET, 0.95)], POLICY)
    assert d.action is Action.REDACT_AND_ALLOW


def test_output_protected_secret_always_blocks():
    finding = f(Category.SECRET, 1.0, spans=(Span(0, 3, "PROTECTED_SECRET"),), entity_type="PROTECTED_SECRET")
    assert decide_output([finding], SecurityPolicy(output_secret_action=OutputSecretAction.REDACT)).action is Action.BLOCK


def test_output_credential_redact_falls_back_to_block_without_span():
    policy = SecurityPolicy(output_secret_action=OutputSecretAction.REDACT)
    with_span = f(Category.SECRET, 0.95, spans=(Span(0, 3, "JWT"),), entity_type="JWT")
    without_span = f(Category.SECRET, 0.95, entity_type="JWT")
    assert decide_output([with_span], policy).action is Action.REDACT_AND_ALLOW
    assert decide_output([without_span], policy).action is Action.BLOCK
    assert decide_output([with_span], POLICY).action is Action.BLOCK


def test_output_leak_violation_and_pii():
    assert decide_output([f(Category.SYSTEM_PROMPT_LEAK)], POLICY).action is Action.BLOCK
    assert decide_output([f(Category.POLICY_VIOLATION)], POLICY).action is Action.BLOCK
    assert decide_output([f(Category.PII)], POLICY).action is Action.REDACT_AND_ALLOW
    assert decide_output([], POLICY).action is Action.ALLOW


def test_policy_validation():
    with pytest.raises(ValueError):
        SecurityPolicy(injection_warn_threshold=0.9, injection_block_threshold=0.5)
    with pytest.raises(ValueError):
        SecurityPolicy(output_denylist=["("])
    with pytest.raises(ValueError):
        SecurityPolicy(pii_entities=["FAVOURITE_COLOUR"])
    with pytest.raises(ValueError):
        SecurityPolicy(unknown_field=True)


def test_disabled_policy_turns_everything_off():
    p = SecurityPolicy.disabled()
    assert not (p.input_rules_enabled or p.pii_input_enabled or p.output_scanning_enabled or p.classifier_enabled)
