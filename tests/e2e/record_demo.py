"""Record the demo video: a real browser, real prompts, the real gateway and model, nothing staged.

    uv run python tests/e2e/record_demo.py                         # live Azure site
    uv run python tests/e2e/record_demo.py --url http://127.0.0.1:8501 --output demo.mp4

Each prompt adds an event to that deployment's audit log. All personal data and keys are synthetic. The run stops with
an error if any decision differs from what the scene expects, so a video always shows what the system did.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from chat_page import ChatPage, Outcome
from playwright.sync_api import Locator, Page, expect, sync_playwright

sys.path.insert(0, str(Path(__file__).parents[1] / "security"))
from corpus import ATTACKS, REAL_WORLD_SENSITIVE, SAFE_QUESTIONS

LIVE_URL = "https://sentinelai-demo.centralindia.cloudapp.azure.com"
WIDTH, HEIGHT = 1600, 900
PANEL_WIDTH = 400
TYPE_DELAY_MS = 28
PASTE_OVER_CHARS = 150  # longer messages are pasted, as people really do


@dataclass(frozen=True)
class Scene:
    title: str
    prompt: str
    goal: str
    expected: str
    result_note: str
    explain: bool = False


_ATTACK = {attack.id: attack.text for attacks in ATTACKS.values() for attack in attacks}
_DATA = {item.id: item.text for item in REAL_WORLD_SENSITIVE}
_SAFE = dict(SAFE_QUESTIONS)

SCENES = [
    Scene("Support agent drafts a reply", _DATA["rw-customer-reply"],
          "An agent pastes a customer's record into the chat to get help writing a reply.", "Redacted",
          "A reply was still drafted, but the model never saw the name, email, phone or card number.", explain=True),
    Scene("HR writes a payroll note", _DATA["rw-hr-record"],
          "An employee's SSN and bank details are pasted in along with a routine request.", "Redacted",
          "The note was written around placeholders, so the real values stayed out of the model."),
    Scene("Developer pastes a stack trace", _DATA["rw-stack-trace"],
          "The traceback includes the production database password, which is easy to miss.", "Redacted",
          "The developer still got help, and the password was cut out first.", explain=True),
    Scene("Developer asks for a refactor", _DATA["rw-code-review"],
          "Real work on real code, with an API key hardcoded in the snippet.", "Redacted",
          "The refactor came back and the key never reached the model.", explain=True),
    Scene("Developer shares a .env file", _DATA["rw-dotenv"],
          "A whole environment file goes in, with payment, Slack and API keys.", "Blocked (input)",
          "A message full of live-format keys was stopped before it reached the model."),
    Scene("CI job failing with a token", _DATA["rw-curl"],
          "A curl command is pasted for debugging, with a GitHub token inside it.", "Redacted",
          "The token was removed and the question was still answered."),
    Scene("Poisoned web page", _ATTACK["ind-web-page"],
          "An honest task, but the page hides an HTML comment telling the AI to send out its credentials.", "Blocked (input)",
          "The hidden instruction was found inside the pasted content.", explain=True),
    Scene("Poisoned vendor email", _ATTACK["ind-vendor-email"],
          "A normal invoice, with a postscript pretending to be the administrator.", "Blocked (input)",
          "The user did nothing wrong. The attack came from inside the document."),
    Scene("Poisoned log file", _ATTACK["ind-log-line"],
          "An attacker planted an instruction in a browser User-Agent, and it landed in the logs.", "Blocked (input)",
          "Even a log line pasted for debugging is checked."),
    Scene("Fake on-call emergency", _ATTACK["soc-oncall"],
          "Urgency, an incident number and a plausible story, all to get the production password.", "Blocked (input)",
          "The regex rules allow this one. The machine-learning layer caught the intent.", explain=True),
    Scene("Fake auditor", _ATTACK["soc-auditor"],
          "A compliance request used as cover for extracting the bot's instructions and keys.", "Blocked (input)",
          "Sounding official does not get around the gateway."),
    Scene("Attack that passes the input check", _ATTACK["dev-onboarding-doc"],
          "It looks like a routine docs request and passes the input check, so it reaches the model.", "Blocked (output)",
          "The model really did start revealing its configuration. The output filter withheld it.", explain=True),
    Scene("Poisoned support ticket", _ATTACK["ind-support-ticket"],
          "The hidden note is buried in a long, legitimate ticket, so the signal is weaker.", "Allowed with warning",
          "Not blocked, but flagged as critical risk for review. The model ignored the note and nothing leaked."),
    Scene("Developer asks about security", _SAFE["safe-dev-sql"],
          "Full of words like 'SQL injection', but it is a genuine code-review question.", "Allowed",
          "No over-blocking: developers can still talk about security."),
    Scene("Developer asks about passwords", _SAFE["safe-dev-rotate"],
          "Mentions production database passwords without asking for any.", "Allowed",
          "The gateway judged what was being asked for, not just the vocabulary."),
]

OVERLAY_CSS = f"""
[data-testid="stApp"] {{ width: calc(100% - {PANEL_WIDTH}px) !important; }}
#demo-panel, #demo-card, #demo-cursor {{ font-family: Inter, 'Segoe UI', system-ui, sans-serif; }}
#demo-panel {{ position: fixed; top: 0; right: 0; bottom: 0; width: {PANEL_WIDTH}px; z-index: 2147482000; box-sizing: border-box;
  padding: 44px 34px; background: #111722; color: #e8ecf3; border-left: 1px solid #2a3344; }}
#demo-panel .count {{ color: #8fa3bf; font-size: 15px; letter-spacing: .08em; text-transform: uppercase; }}
#demo-panel h2 {{ font-size: 32px; line-height: 1.2; margin: 14px 0 30px; font-weight: 700; color: #fff; }}
#demo-panel .label {{ color: #8fa3bf; font-size: 14px; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 8px; }}
#demo-panel p {{ font-size: 20px; line-height: 1.45; margin: 0 0 30px; }}
#demo-panel .verdict {{ font-size: 24px; font-weight: 700; margin-bottom: 10px; }}
#demo-card {{ position: fixed; inset: 0; z-index: 2147483000; background: #0c111a; color: #e8ecf3; display: flex;
  flex-direction: column; justify-content: center; padding: 0 130px; box-sizing: border-box; }}
#demo-card h1 {{ font-size: 68px; margin: 0 0 18px; color: #fff; }}
#demo-card .sub {{ font-size: 28px; line-height: 1.45; color: #c3cedd; margin: 0 0 34px; }}
#demo-card .note {{ font-size: 21px; line-height: 1.5; color: #8fa3bf; margin: 0; }}
#demo-card table {{ border-collapse: collapse; width: 100%; font-size: 23px; margin-bottom: 26px; }}
#demo-card td, #demo-card th {{ padding: 10px 14px; border-bottom: 1px solid #2a3344; text-align: left; }}
#demo-card table.compact {{ font-size: 20px; margin-bottom: 18px; }}
#demo-card table.compact td, #demo-card table.compact th {{ padding: 6px 14px; }}
#demo-card th {{ color: #8fa3bf; font-size: 17px; text-transform: uppercase; letter-spacing: .06em; }}
#demo-cursor {{ position: fixed; z-index: 2147483600; width: 20px; height: 20px; margin: -10px 0 0 -10px; border-radius: 50%;
  background: rgba(255, 196, 0, .75); border: 2px solid #fff; pointer-events: none; left: -50px; top: -50px; }}
"""

CURSOR_SCRIPT = """
addEventListener('mousemove', e => {
  let c = document.getElementById('demo-cursor');
  if (!c) { c = document.createElement('div'); c.id = 'demo-cursor'; document.documentElement.appendChild(c); }
  c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px';
}, true);
"""

SET_HTML = """([id, markup]) => {
  let el = document.getElementById(id);
  if (!markup) { if (el) el.remove(); return; }
  if (!el) { el = document.createElement('div'); el.id = id; document.documentElement.appendChild(el); }
  el.innerHTML = markup;
}"""

VERDICT_COLORS = {"Allowed": "#47c96b", "Redacted": "#5ba7ff", "Blocked (input)": "#ff6b6b", "Blocked (output)": "#ffa45c", "Allowed with warning": "#ffd166"}


class Director:
    def __init__(self, page: Page, chat: ChatPage) -> None:
        self.page, self.chat = page, chat

    def overlay(self, element: str, markup: str | None) -> None:
        self.page.evaluate(SET_HTML, [element, markup])

    def pause(self, seconds: float) -> None:
        self.page.wait_for_timeout(int(seconds * 1000))

    def click(self, target: Locator) -> None:
        box = target.bounding_box()
        assert box, "element is not on screen"
        self.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=18)
        self.pause(0.25)
        target.click()

    def panel(self, count: str, title: str, *blocks: tuple[str, str]) -> None:
        body = "".join(f'<div class="label">{html.escape(a)}</div><p>{html.escape(b)}</p>' for a, b in blocks)
        self.overlay("demo-panel", f'<div class="count">{html.escape(count)}</div><h2>{html.escape(title)}</h2>{body}')

    def verdict(self, count: str, scene: Scene, outcome: Outcome) -> None:
        color = VERDICT_COLORS[outcome.label]
        self.overlay(
            "demo-panel",
            f'<div class="count">{html.escape(count)}</div><h2>{html.escape(scene.title)}</h2>'
            f'<div class="label">The situation</div><p>{html.escape(scene.goal)}</p>'
            f'<div class="label">What SentinelAI did</div><div class="verdict" style="color:{color}">{html.escape(outcome.label)}</div>'
            f"<p>{html.escape(scene.result_note)}</p>",
        )

    def play(self, number: int, scene: Scene) -> dict:
        count = f"Scene {number} of {len(SCENES)}"
        self.panel(count, scene.title, ("The situation", scene.goal))
        self.pause(2.2)
        pasted = len(scene.prompt) > PASTE_OVER_CHARS
        outcome = self.chat.ask(scene.prompt, 0 if pasted else TYPE_DELAY_MS, hold_before_send_ms=2500 if pasted else 300)
        if outcome.label != scene.expected:
            raise SystemExit(f"{scene.title}: expected {scene.expected}, the gateway returned {outcome.label}\n{outcome.text}")
        outcome.message.get_by_test_id("stAlert").evaluate("e => e.scrollIntoView({block: 'start', behavior: 'smooth'})")
        self.verdict(count, scene, outcome)
        self.pause(3.5)
        if scene.explain:
            header = outcome.message.get_by_test_id("stExpander").get_by_text("Why? Security explanation")
            self.click(header)
            panel = outcome.message.get_by_test_id("stExpander")
            expect(panel.get_by_text("Input decision:")).to_be_visible()
            self.pause(0.7)
            panel.evaluate("e => e.scrollIntoView({block: 'center', behavior: 'smooth'})")
            self.pause(5.5)
        self.click(self.page.get_by_role("button", name="Clear conversation"))
        expect(self.chat.messages).to_have_count(0)
        return {"scene": scene.title, "expected": scene.expected, "actual": outcome.label}

    def title_card(self, url: str) -> None:
        host = url.split("//", 1)[-1]
        self.overlay(
            "demo-card",
            "<h1>SentinelAI</h1>"
            '<div class="sub">A security gateway that checks every message to and from an AI chatbot.<br>'
            "Watch it stop attacks and protect sensitive data.</div>"
            f'<div class="note">Recorded {date.today():%d %B %Y} from {html.escape(host)}<br>'
            "Automated with Playwright. Answers come from the real model. Decisions are not scripted.<br>"
            "All names, cards and keys shown are synthetic.</div>",
        )
        self.pause(5.5)
        self.overlay("demo-card", None)

    def monitoring(self) -> None:
        self.panel("Monitoring", "Audit log",
                   ("What is recorded", "Every decision, its risk and its timing. The text of the messages is not stored."))
        self.chat.page.mouse.move(150, 200, steps=15)
        self.click(self.page.get_by_test_id("stSidebarNav").get_by_text("Audit events", exact=True))
        expect(self.page.get_by_test_id("stDataFrame").first).to_be_visible(timeout=30_000)
        self.pause(6)
        self.panel("Monitoring", "Live dashboard", ("What it shows", "Blocked, redacted and allowed traffic from the audit log, together with the benchmark results."))
        self.click(self.page.get_by_test_id("stSidebarNav").get_by_text("Dashboard", exact=True))
        expect(self.page.get_by_role("heading", name="Security dashboard")).to_be_visible(timeout=30_000)
        self.pause(7)

    def end_card(self, results: list[dict]) -> None:
        rows = "".join(
            f'<tr><td>{html.escape(r["scene"])}</td><td style="color:{VERDICT_COLORS[r["actual"]]};font-weight:700">{html.escape(r["actual"])}</td></tr>'
            for r in results
        )
        self.overlay(
            "demo-card",
            f'<h1 style="font-size:44px">This run</h1><table class="compact"><tr><th>Scene</th><th>Decision</th></tr>{rows}</table>'
            '<div class="note">No detector is perfect. Some attacks pass the input check, which is why every answer is checked too. '
            "Known misses are tracked in the test suite.</div>",
        )
        self.pause(8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=LIVE_URL)
    parser.add_argument("--output", type=Path, default=Path("SentinelAI_Demo.mp4"))
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            record_video_dir=tmp,
            record_video_size={"width": WIDTH, "height": HEIGHT},
            color_scheme="dark",
        )
        page = context.new_page()
        page.add_init_script(CURSOR_SCRIPT)
        chat = ChatPage(page, args.url)
        results: list[dict] = []
        failure: BaseException | None = None
        try:
            chat.open()
            director = Director(page, chat)
            page.add_style_tag(content=OVERLAY_CSS)
            director.pause(1.5)
            director.title_card(args.url)
            for number, scene in enumerate(SCENES, 1):
                results.append(director.play(number, scene))
                print(json.dumps(results[-1]), flush=True)
            director.monitoring()
            director.end_card(results)
        except BaseException as exc:
            failure = exc
        finally:
            video = page.video
            context.close()
            browser.close()
        if failure is not None:
            kept = args.output.with_suffix(".failed.webm")
            shutil.copy(video.path(), kept)
            print(f"recording stopped; partial video at {kept}", file=sys.stderr)
            raise failure
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video.path()), "-c:v", "libx264", "-preset", "slow", "-crf", "18",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)],
            check=True,
        )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
