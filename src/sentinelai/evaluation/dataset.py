"""Loading and versioning of the labelled evaluation dataset (data/evaluation/*.jsonl)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATASET_NAME = "sentinelai-eval"
FILES = ("benign.jsonl", "malicious.jsonl", "output_cases.jsonl", "edge_cases.jsonl")


@dataclass(frozen=True, slots=True)
class PromptCase:
    id: str
    text: str
    label: str  # benign | malicious
    category: str
    sensitive_values: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True, slots=True)
class OutputCase:
    id: str
    prompt: str
    model_response: str
    category: str
    expected: str  # BLOCK | REDACT_AND_ALLOW | ALLOW
    sensitive_values: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EdgeCase:
    id: str
    kind: str
    payload: Any
    expect: str  # reject | handled | blocked


@dataclass(slots=True)
class EvalDataset:
    version: str
    prompts: list[PromptCase] = field(default_factory=list)
    outputs: list[OutputCase] = field(default_factory=list)
    edges: list[EdgeCase] = field(default_factory=list)

    @property
    def benign(self) -> list[PromptCase]:
        return [c for c in self.prompts if c.label == "benign"]

    @property
    def malicious(self) -> list[PromptCase]:
        return [c for c in self.prompts if c.label == "malicious"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{n}: invalid JSON ({exc.msg})") from None
    return rows


def _optional(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _edge_payload(row: dict[str, Any]) -> Any:
    if "repeat" in row:
        return row["repeat"]["chunk"] * int(row["repeat"]["times"]) + row.get("suffix", "")
    return row.get("text")


def load_dataset(directory: Path, name: str = DATASET_NAME) -> EvalDataset:
    digest = hashlib.sha256()
    for filename in FILES:
        if (directory / filename).exists():
            digest.update(filename.encode() + (directory / filename).read_bytes())
    ds = EvalDataset(version=f"{name}@{digest.hexdigest()[:12]}")

    seen: set[str] = set()
    for name in ("benign.jsonl", "malicious.jsonl"):
        for row in read_jsonl(directory / name):
            case = PromptCase(
                id=row["id"],
                text=row["text"],
                label=row["label"],
                category=row["category"],
                sensitive_values=tuple(row.get("sensitive_values", [])),
                notes=row.get("notes", ""),
            )
            if case.label not in {"benign", "malicious"}:
                raise ValueError(f"{case.id}: unknown label {case.label!r}")
            ds.prompts.append(case)
    # Output and edge cases are optional (the holdout split has prompts only).
    for row in _optional(directory / "output_cases.jsonl"):
        ds.outputs.append(
            OutputCase(
                id=row["id"],
                prompt=row["prompt"],
                model_response=row["model_response"],
                category=row["category"],
                expected=row["expected"],
                sensitive_values=tuple(row.get("sensitive_values", [])),
            )
        )
    for row in _optional(directory / "edge_cases.jsonl"):
        ds.edges.append(EdgeCase(id=row["id"], kind=row["kind"], payload=_edge_payload(row), expect=row["expect"]))

    for cid in [c.id for c in ds.prompts] + [c.id for c in ds.outputs] + [c.id for c in ds.edges]:
        if cid in seen:
            raise ValueError(f"duplicate case id {cid!r}")
        seen.add(cid)
    return ds
