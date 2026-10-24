"""Output-side policy checks: system-prompt leakage and operator-configured denylist patterns.

These checks filter *content*; they say nothing about whether an answer is factually correct.
"""

from __future__ import annotations

import re

from sentinelai.gateway.decisions import Category, Finding, Severity

DETECTOR_NAME = "output_policy"
DETECTOR_VERSION = "1.0.0"
SHINGLE = 6


def _shingles(text: str, n: int = SHINGLE) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class OutputPolicyDetector:
    name = DETECTOR_NAME
    version = DETECTOR_VERSION

    def __init__(self, system_prompt: str, leak_threshold: float = 0.2) -> None:
        self._prompt_shingles = _shingles(system_prompt)
        self.leak_threshold = leak_threshold

    def analyze(self, output: str, denylist: list[str]) -> list[Finding]:
        findings: list[Finding] = []
        if self._prompt_shingles:
            overlap = len(_shingles(output) & self._prompt_shingles) / len(self._prompt_shingles)
            if overlap >= self.leak_threshold:
                findings.append(
                    Finding(
                        detector=self.name,
                        detector_version=self.version,
                        category=Category.SYSTEM_PROMPT_LEAK,
                        score=min(1.0, 0.7 + overlap),
                        severity=Severity.HIGH,
                        evidence=f"response reproduces {overlap:.0%} of the system prompt's 6-word phrases",
                        rule_ids=("OUT-001",),
                    )
                )
        hits = [i for i, pattern in enumerate(denylist) if re.search(pattern, output, re.IGNORECASE)]
        if hits:
            findings.append(
                Finding(
                    detector=self.name,
                    detector_version=self.version,
                    category=Category.POLICY_VIOLATION,
                    score=1.0,
                    severity=Severity.HIGH,
                    evidence=f"matched denylist pattern index(es) {hits}",
                    rule_ids=tuple(f"DENY:{i}" for i in hits),
                )
            )
        return findings
