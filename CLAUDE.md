# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session workflow

Check `PROGRESS.md` for current implementation status before starting work.
Read `SCRATCHPAD.md` for non-obvious gotchas and architectural decisions before touching any phase.
Check `CODEREVIEW.md` for open review findings and address any open items before starting new phase work.
Read the Codex-authored architecture audit and remediation roadmap before any structural or scientific-claim work — these are the source of truth for the R-phase remediation effort (R1–R6) and were implemented entirely by Codex without Claude involvement, so they will not appear in Claude's git memory:
- `reports/architecture_audit_2026-04-04.md` — baseline audit, the yardstick for remediation progress.
- `reports/remediation_roadmap_2026-04-04.md` — the R1–R6 phase plan, milestones M1/M2/M3, and work packages A–D.
- `AGENTS.md` — Codex's project context; mirrors much of this file but is the authoritative brief for Codex sessions, so keep the two in sync when workflow rules change.
Any newer dated artifacts under `reports/*.md` from later Codex sessions should be read the same way: assume Codex has already implemented what they describe and treat them as required reading, not optional history.
Update `PROGRESS.md` after completing any phase.
Update `SCRATCHPAD.md` whenever you discover something that would waste time if rediscovered.
Use `pytest -m "not slow"` for the fast local loop, `pytest tests/test_<phase>.py -v` for a phase-specific loop, and `pytest` for the full gate.
**Before closing a session, run all tests and generate a status report at `reports/status_latest.html`. The status page must clearly separate automated test status, implementation progress, architecture/validation status, documentation drift warnings, and open `CODEREVIEW.md` items.**
```bash
python scripts/status_report.py
```
**When validation or audit logic changes, also generate the full validation audit artifact (`reports/validation_report.html`, with JSON companion if available). User-facing report content must expose pass criteria, assumptions, limitations, and code provenance rather than summary-only claims.**
**Before confirming a session is ready to close: commit all changes, then push to the remote.**
```bash
git push
```

## Codex CLI workflow

`codex` CLI (v0.118.0) is installed. Config: `~/.codex/config.toml` (model: `gpt-5.4`).
Project context for Codex is in `AGENTS.md` (repo root) — Codex reads this automatically.

**Model:** `gpt-5.4` (works with ChatGPT Plus auth). Models `o3`, `o4-mini`, `gpt-4.1`, and `codex-mini-latest` do NOT work with ChatGPT auth.

**1. Code review** — read-only sandbox is fine, no special flags needed:
```bash
codex review --uncommitted                    # review working tree changes
codex review --base master                    # review branch vs master
codex review --commit HEAD                    # review last commit
```

**2. Background implementation** — needs sandbox bypass on Windows to run tests/write files:
```bash
codex exec "<task prompt>" --dangerously-bypass-approvals-and-sandbox
```

**3. Read-only tasks** (listing files, reading code, analysis) — `--full-auto` is sufficient:
```bash
codex exec "<analysis prompt>" --full-auto
```

**Windows sandbox note:** The `read-only` and `workspace-write` sandbox modes block most shell commands on Windows. For tasks that need to run pytest or write code, use `--dangerously-bypass-approvals-and-sandbox`. For read-only analysis, `--full-auto` works.

Claude handles architecture decisions and orchestration; Codex handles implementation detail and review.

---

## Code review sessions (Opus)

When running as Opus in a review session: audit the codebase against `retinal_sim_architecture.md`, run all tests, and write findings to `CODEREVIEW.md` using this format per item:

- **File:** path/to/file.py
- **Function:** function_name
- **Issue:** what's wrong
- **Fix:** what the fix should be

Move resolved items to the Resolved section rather than deleting them.

## Commands

```bash
pip install -e ".[dev]"          # install (editable + test deps)
pytest -m "not slow"             # fast local loop
pytest tests/test_retina.py -v   # single test file
pytest                           # full gate
```

## Architecture

The pipeline is a linear chain of five stages, each consuming the previous stage's output dataclass:

```
RGB image
  → SpectralImage (H, W, N_λ)          spectral/upsampler.py
  → RetinalIrradiance (H, W, N_λ)      optical/stage.py
  → PhotoreceptorMosaic + MosaicActivation  retina/stage.py
  → rendered figures                    output/
```

`pipeline.py::RetinalSimulator` orchestrates these stages. `species/config.py::SpeciesConfig` loads all species-specific parameters (optical + retinal) from `data/species/{human,dog,cat}.yaml` and is the single place to change per-species constants.

**Implementation status:** Phases 1-13 are complete (see `PROGRESS.md`). Current work is comprehensive code review, architecture audit, and report/documentation transparency improvements.

## Key design decisions

- **PoC patch size**: 2° centered on area centralis. Scale to 5–10° later.
- **Mosaic generation**: jittered grid (not Poisson disk) for speed. Upgrade path noted in `retina/mosaic.py`.
- **PSF**: Gaussian placeholder first (`optical/psf.py::gaussian_psf`), then diffraction-limited FFT (`diffraction_psf`). Chromatic aberration deferred.
- **Spectral upsampling**: Smits (1999) first, Mallett-Yuksel later. Either can be bypassed entirely if a hyperspectral input is provided.
- **Each stage is independently validatable** — see architecture §11 for the full validation matrix and `tests/test_validation_report.py` for the reporting artifact contract. The repo now has dedicated tests through Phase 13; do not describe later test files as stubs.

## Species constants

Peak wavelengths (nm) live in `retina/opsin.py::LAMBDA_MAX`. Optical params (pupil, focal length, axial length) are in `optical/stage.py::OpticalParams`. Cat uses a vertical slit pupil → anisotropic PSF; human and dog use circular pupils.
