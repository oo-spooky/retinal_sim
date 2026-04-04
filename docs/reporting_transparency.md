# Reporting Transparency Contract

This project produces two different report classes:

- `reports/status_latest.html`
  Purpose: current automated test status, implementation progress snapshot, documentation drift checks, and open review findings.
- `reports/validation_report.html` plus `reports/validation_report.json`
  Purpose: detailed validation and architecture-audit evidence.

## Required transparency for generated report content

Every validation result shown to the user must expose:

- the stage under test,
- the architecture requirement being checked,
- the exact pass criterion or tolerance,
- the observed result,
- the inputs or scenario used,
- the method or metric used,
- the assumptions behind the check,
- the known limitations or deferred simplifications,
- and the code path(s) responsible for the behavior.

## Wording rules

- Do not present inferred conclusions as directly measured facts.
- Mark visual-inspection checks explicitly as visual/manual evidence.
- Do not use summary-only claims like "physically accurate" without a cited criterion.
- If a threshold is relaxed because of proof-of-concept approximations, say so directly in the report.
- Report pass/fail as "implemented validation checks passed/failed" rather than implying full architecture closure.

## Audit expectations

- `PROGRESS.md` is not an audit artifact. It tracks implementation completion.
- `CODEREVIEW.md` is not a validation artifact. It tracks review findings.
- The status report must clearly distinguish those sources instead of collapsing them into one notion of "done".
- The validation report should be readable on its own, but it should still point readers to review findings and status artifacts.
