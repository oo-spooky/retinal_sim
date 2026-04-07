"""Status report script tests."""
from __future__ import annotations

from scripts import status_report


def test_parse_phase_table_extracts_rows():
    text = """
| Phase | Component | Status | Tests | Notes |
|-------|-----------|--------|-------|-------|
| 1 | A | **COMPLETE** | ok | note |
| 2 | B | **COMPLETE** | ok | note |
"""
    rows = status_report.parse_phase_table(text)
    assert len(rows) == 2
    assert rows[0][0] == "1"
    assert rows[0][2] == "COMPLETE"


def test_parse_open_items_none():
    text = "## Open Items\n\n*(none)*\n"
    assert status_report.parse_open_items(text) == ""


def test_detect_documentation_drift_flags_stale_docs():
    docs = {
        "AGENTS.md": "Phases 1-11 are COMPLETE.",
        "CLAUDE.md": "Phases 1-10 complete.",
        "PROGRESS.md": "Phase 12 complete.",
    }
    warnings = status_report.detect_documentation_drift(docs)
    assert warnings
    assert any("Phase 13" in warning for warning in warnings)


def test_build_html_separates_status_types():
    html = status_report.build_html(
        phase_rows=[["13", "Validation report generator", "COMPLETE", "41/41", "ok"]],
        pytest_summary="10 passed in 1.2s",
        pytest_counts={"passed": 10.0, "failed": 0.0, "skipped": 0.0, "error": 0.0, "duration": 1.2},
        pytest_failures="",
        pytest_command="python -m pytest --tb=short -q --no-header",
        open_items="",
        doc_warnings=["CLAUDE.md does not clearly advertise the current Phase 13-complete state."],
        validation_report_exists=True,
        figure_uris=[],
        git_commit="deadbeef",
    )
    assert "Test Status" in html
    assert "Implementation Phase Status" in html
    assert "Validation and Architecture Audit Status" in html
    assert "Exact pytest command" in html
    assert "documentation drift" in html.lower()


def test_build_html_includes_open_review_findings():
    html = status_report.build_html(
        phase_rows=[["13", "Validation report generator", "COMPLETE", "51/51", "ok"]],
        pytest_summary="10 passed in 1.2s",
        pytest_counts={"passed": 10.0, "failed": 0.0, "skipped": 0.0, "error": 0.0, "duration": 1.2},
        pytest_failures="",
        pytest_command="python -m pytest --tb=short -q --no-header",
        open_items="### CR-23\n- Issue: sample finding",
        doc_warnings=[],
        validation_report_exists=True,
        figure_uris=[],
        git_commit="deadbeef",
    )
    assert "CR-23" in html
    assert "sample finding" in html
    assert "No open code review findings" not in html
