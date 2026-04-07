# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Start Here

- Read `docs/llm_coordination.md` for the shared Codex/Claude workflow.
- Read `TOUCHLOG.md` on entry and append one end-of-session entry on exit.
- Check `PROGRESS.md` for the current repo status, `CODEREVIEW.md` for open findings, and `SCRATCHPAD.md` for unresolved short-lived notes.
- For structural, scientific-claim, or report-language work, also read `retinal_sim_architecture.md`, `docs/reporting_transparency.md`, `reports/architecture_audit_2026-04-04.md`, and `reports/remediation_roadmap_2026-04-04.md`.
- `AGENTS.md` remains the Codex-facing technical brief. Keep shared workflow changes in `docs/llm_coordination.md` rather than re-duplicating them here.

## Claude-Specific Workflow

- Use `pytest -m "not slow"` for the fast local loop, `pytest tests/test_<phase>.py -v` for phase-specific work, and `pytest` for the full gate.
- Before closing a substantial session, run `python scripts/status_report.py` so `reports/status_latest.html` reflects current automated test status, implementation snapshot, doc drift warnings, and open review items.
- When validation or audit logic changes, also regenerate `reports/validation_report.html` and its JSON companion if available. Report content must expose pass criteria, assumptions, limitations, and code provenance.
- Before declaring a session ready to close, commit all changes and push to the remote with `git push`.

## Codex CLI From Claude

- `codex` CLI (v0.118.0) is installed. Config: `~/.codex/config.toml` with model `gpt-5.4`.
- Project context for Codex lives in `AGENTS.md`; Codex reads it automatically.
- ChatGPT-auth model support verified here: `gpt-5.4` works. `o3`, `o4-mini`, `gpt-4.1`, and `codex-mini-latest` do not.

### Review

```bash
codex review --uncommitted
codex review --base master
codex review --commit HEAD
```

### Background implementation

```bash
codex exec "<task prompt>" --dangerously-bypass-approvals-and-sandbox
```

### Read-only analysis

```bash
codex exec "<analysis prompt>" --full-auto
```

- Windows sandbox note: `read-only` and `workspace-write` modes block most shell commands on Windows. Use `--dangerously-bypass-approvals-and-sandbox` for tasks that need tests or file writes. Use `--full-auto` for read-only analysis.
- Claude handles architecture decisions and orchestration; Codex handles implementation detail and review.

## Review Sessions

When running as Opus in a review session:

- Audit the codebase against `retinal_sim_architecture.md`.
- Run the relevant tests.
- Write findings to `CODEREVIEW.md` using this format:

```text
- **File:** path/to/file.py
- **Function:** function_name
- **Issue:** what's wrong
- **Fix:** what the fix should be
```

- Move resolved findings into `docs/archive/CODEREVIEW_RESOLVED_2026-04.md` or the latest resolved archive, rather than leaving them in `CODEREVIEW.md`.

## Repo Notes

- The pipeline is a linear chain of five stages: RGB image -> `SpectralImage` -> `RetinalIrradiance` -> `PhotoreceptorMosaic` plus `MosaicActivation` -> rendered figures.
- `pipeline.py::RetinalSimulator` orchestrates the stages. `species/config.py::SpeciesConfig` is the single place to load per-species constants from `data/species/{human,dog,cat}.yaml`.
- Phases 1-13 are complete. Current work is code review, architecture audit follow-through, reporting transparency, and documentation discipline.
- The proof-of-concept patch size is 2 degrees centered on area centralis.
- Each stage is independently validatable. See section 11 of `retinal_sim_architecture.md` and `tests/test_validation_report.py` for the reporting artifact contract.
- Peak wavelengths live in `retina/opsin.py::LAMBDA_MAX`. Optical params (pupil, focal length, axial length) live in `optical/stage.py::OpticalParams`.
