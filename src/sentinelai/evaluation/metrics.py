"""Binary-classification metrics with explicit handling of empty denominators."""

from __future__ import annotations

from dataclasses import asdict, dataclass


def ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


@dataclass(frozen=True, slots=True)
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @classmethod
    def from_pairs(cls, pairs: list[tuple[bool, bool]]) -> Confusion:
        """``pairs`` of (actually_positive, predicted_positive)."""
        tp = sum(1 for a, p in pairs if a and p)
        fp = sum(1 for a, p in pairs if not a and p)
        tn = sum(1 for a, p in pairs if not a and not p)
        fn = sum(1 for a, p in pairs if a and not p)
        return cls(tp, fp, tn, fn)

    @property
    def recall(self) -> float | None:
        return ratio(self.tp, self.tp + self.fn)

    @property
    def precision(self) -> float | None:
        return ratio(self.tp, self.tp + self.fp)

    @property
    def false_positive_rate(self) -> float | None:
        return ratio(self.fp, self.fp + self.tn)

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if not p or not r:
            return 0.0 if p is not None and r is not None else None
        return round(2 * p * r / (p + r), 4)

    @property
    def accuracy(self) -> float | None:
        return ratio(self.tp + self.tn, self.tp + self.fp + self.tn + self.fn)

    def summary(self) -> dict[str, object]:
        return {
            "confusion_matrix": asdict(self),
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "accuracy": self.accuracy,
        }
