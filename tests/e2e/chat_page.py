"""Playwright page object for the Streamlit chat. Shared by the browser tests and the demo recorder."""

from __future__ import annotations

from dataclasses import dataclass

from playwright.sync_api import Locator, Page, expect

# Same wording as sentinelai.ui.style.STATUS_LABELS; duplicated so a changed label fails the browser tests.
LABELS = ("Blocked (input)", "Blocked (output)", "Held for review", "Redacted", "Allowed with warning", "Allowed")
MODEL_WAIT_MS = 180_000


@dataclass(frozen=True)
class Outcome:
    label: str
    text: str
    message: Locator


class ChatPage:
    def __init__(self, page: Page, base_url: str) -> None:
        self.page = page
        self.base_url = base_url.rstrip("/")
        self.messages = page.get_by_test_id("stChatMessage")
        self.input = page.get_by_test_id("stChatInputTextArea")

    def open(self) -> ChatPage:
        self.page.goto(self.base_url + "/")
        expect(self.input).to_be_editable(timeout=60_000)
        return self

    def ask(self, prompt: str, type_delay_ms: float = 0, hold_before_send_ms: float = 0) -> Outcome:
        """Send a prompt, show it appearing at once, and wait for the gateway's decision."""
        before = self.messages.count()
        self.input.click()
        if type_delay_ms:
            self.input.press_sequentially(prompt, delay=type_delay_ms)
        else:
            self.input.fill(prompt)
        if hold_before_send_ms:
            self.page.wait_for_timeout(hold_before_send_ms)
        self.input.press("Enter")
        expect(self.messages).to_have_count(before + 2, timeout=30_000)
        reply = self.messages.nth(before + 1)
        notice = reply.get_by_test_id("stAlert")
        expect(notice).to_be_visible(timeout=MODEL_WAIT_MS)
        text = notice.inner_text()
        label = next((candidate for candidate in LABELS if candidate in text), "")
        assert label, f"the chat showed an error instead of a decision: {text}"
        return Outcome(label, reply.inner_text(), reply)

    def explain(self, outcome: Outcome) -> Locator:
        """Open the 'Why?' panel of a reply and return it."""
        expander = outcome.message.get_by_test_id("stExpander")
        expander.get_by_text("Why? Security explanation").click()
        expect(expander.get_by_text("Input decision:")).to_be_visible()
        return expander

    def clear(self) -> None:
        self.page.get_by_role("button", name="Clear conversation").click()
        expect(self.messages).to_have_count(0)

    def go_to(self, title: str, heading: str) -> None:
        self.page.get_by_test_id("stSidebarNav").get_by_text(title, exact=True).click()
        expect(self.page.get_by_role("heading", name=heading).first).to_be_visible(timeout=30_000)
