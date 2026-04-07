# Retinal Sim - Status Snapshot

_Last updated: 2026-04-07 (documentation unification scaffold)_

## Current State

- Phases 1-13 are complete.
- Current work is code review, architecture audit follow-through, reporting transparency, and documentation discipline.
- Latest notable shipped change: the 2026-04-07 perceptual rendering correction in `retinal_sim/output/perceptual.py` plus simulated-patch cropping in `scripts/render_scene.py`.

## Testing And Validation References

- Fast local loop: `pytest -m "not slow"`
- Phase-specific loop: `pytest tests/test_<phase>.py -v`
- Full gate: `pytest`
- `reports/status_latest.html` is the current audit dashboard for automated test status, implementation snapshot, documentation drift, and review state.
- `reports/validation_report.html` and its JSON companion are the detailed validation artifacts.
- `docs/reporting_transparency.md` defines wording and transparency requirements for generated reports.

## Canonical References

- `retinal_sim_architecture.md` is the canonical software architecture.
- `reports/architecture_audit_2026-04-04.md` is the baseline external-methodology audit.
- `reports/remediation_roadmap_2026-04-04.md` explains the R1-R6 remediation decisions already implemented in the repo.
- `CODEREVIEW.md` tracks open review findings.
- `TOUCHLOG.md` and `docs/llm_coordination.md` define the current cross-LLM workflow.

## History

- Detailed phase-by-phase implementation notes are archived in `docs/archive/PROGRESS_legacy_2026-04.md`.
