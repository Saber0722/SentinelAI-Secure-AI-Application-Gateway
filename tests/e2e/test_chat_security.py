"""Drives the real UI in a browser: what a user sees when the gateway blocks or redacts."""

from __future__ import annotations

import re

import corpus
import pytest
from chat_page import ChatPage
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e

_ATTACKS = {attack.id: attack.text for attacks in corpus.ATTACKS.values() for attack in attacks}
# These run with the rules only, so each one must be a prompt the rules can block by themselves.
ATTACKS = {i: _ATTACKS[i] for i in ("ind-web-page", "ind-vendor-email", "ind-log-line", "ind-github-issue", "dev-dotenv", "soc-ceo")}
SENSITIVE = {item.id: item for item in corpus.REAL_WORLD_SENSITIVE if item.id in ("rw-customer-reply", "rw-curl", "rw-hr-record")}
SAFE = dict(corpus.SAFE_QUESTIONS)


@pytest.mark.parametrize("prompt", ATTACKS.values(), ids=ATTACKS.keys())
def test_attack_is_blocked_on_screen(chat: ChatPage, prompt: str):
    outcome = chat.ask(prompt)
    assert outcome.label == "Blocked (input)"
    assert "before it reached the model" in outcome.text
    assert "model not run" in outcome.text


def test_block_explains_why(chat: ChatPage):
    outcome = chat.ask(ATTACKS["ind-web-page"])
    panel = chat.explain(outcome)
    expect(panel).to_contain_text("BLOCK")
    expect(panel).to_contain_text("Rule score")


@pytest.mark.parametrize("item", SENSITIVE.values(), ids=SENSITIVE.keys())
def test_sensitive_data_is_redacted_in_the_reply(chat: ChatPage, item: corpus.Sensitive):
    outcome = chat.ask(item.text)
    assert outcome.label == "Redacted"
    assert item.placeholder in outcome.text  # the mock model echoes what it received
    for value in item.values:
        assert value not in outcome.text


def test_model_leak_is_withheld_by_the_output_filter(chat: ChatPage):
    # The rules let this through; the vulnerable test model then hands over its configuration.
    outcome = chat.ask(_ATTACKS["ind-support-ticket"])
    assert outcome.label == "Blocked (output)"
    assert "AcmeAssist" not in outcome.text.split("withheld.", 1)[1]


@pytest.mark.parametrize("prompt", [SAFE["safe-dev-sql"], SAFE["safe-dev-rotate"]], ids=["sql-injection-review", "password-rotation"])
def test_safe_question_is_answered(chat: ChatPage, prompt: str):
    outcome = chat.ask(prompt)
    assert outcome.label == "Allowed"
    assert "[mock model]" in outcome.text


def test_prompt_appears_before_the_decision_and_can_be_cleared(chat: ChatPage):
    outcome = chat.ask("hello")
    assert outcome.label == "Allowed"
    expect(chat.messages.first).to_contain_text("hello")
    chat.clear()


def test_blocked_request_reaches_the_audit_log_without_its_text(chat: ChatPage):
    email = "audit.check@example.com"
    chat.ask(ATTACKS["ind-web-page"])
    chat.ask(f"Please update my account, my email is {email}.")
    chat.go_to("Audit events", "Audit events")
    table = chat.page.get_by_test_id("stDataFrame").first
    expect(table).to_be_visible(timeout=30_000)
    # The grid draws on a canvas but keeps an accessible copy of its cells in the DOM.
    expect(table).to_contain_text(re.compile("blocked", re.I), timeout=15_000)
    expect(table).to_contain_text(re.compile("redacted", re.I), timeout=15_000)
    page_html = chat.page.content()
    assert email not in page_html and "append it to https://attacker.example" not in page_html


def test_dashboard_loads(chat: ChatPage):
    chat.go_to("Dashboard", "Dashboard")
    expect(chat.page.get_by_test_id("stException")).to_have_count(0)
