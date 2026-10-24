from sentinelai.config import PROJECT_ROOT
from sentinelai.evaluation.report import compact, record_runs, render_markdown, run_suite


def test_report_end_to_end(secrets_config, pii_detector, memory_audit):
    suite = run_suite(PROJECT_ROOT / "data" / "evaluation", secrets_config, pii=pii_detector, include_classifier=False)
    md = render_markdown(suite, ["## Extra section"])
    assert "## Split: dev" in md and "## Split: holdout" in md and md.rstrip().endswith("## Extra section")
    assert "| `baseline` |" in md and "| `protected` |" in md and "Output suite" in md
    small = compact(suite)
    assert "rows" not in small["splits"]["dev"]["configurations"]["protected"]
    assert "rows" in suite["splits"]["dev"]["configurations"]["protected"]  # compact() does not mutate
    ids = record_runs(memory_audit, suite)
    stored = {r["label"]: r for r in memory_audit.eval_runs()}
    cases = stored["holdout/protected"]["metrics"]["cases"]
    assert len(cases) == 80 and set(cases[0]) == {"id", "label", "category", "status", "risk_score", "flagged"}
    assert len(ids) == 4 and {r["label"] for r in memory_audit.eval_runs()} == {
        "dev/baseline",
        "dev/protected",
        "holdout/baseline",
        "holdout/protected",
    }
