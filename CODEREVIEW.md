# CODEREVIEW.md

Code review findings for `retinal_sim`. Open items should be addressed before starting new phase work. Resolved items are moved here rather than deleted for traceability.

---

## Current review (2026-04-02)

**Reviewer:** Opus
**Scope:** Phases 8–9 (architecture validation), structural audit
**Test results:** 329 passed, 1 skipped in 43.62s

---

## Fix Priority Guide

**Before Phase 10 work (recommended):**
1. **CR-18** (acuity docstring) — stale docstring misleads readers about discriminability metric
2. **CR-20** (pyproject.toml) — editable install is broken; blocks any new contributor setup

**Before Phase 12 (species comparison):**
3. **CR-11** (pupil throughput) — cross-species irradiance comparison is meaningless without it
4. **CR-12** (cat slit pupil warning) — at minimum add a warning; anisotropic PSF ideal
5. **CR-15** (in-vitro sensitivities) — add docstring warning or apply media filtering
6. **CR-16** (edge clipping → zero) — affects small-scene simulations
7. **CR-21** (RetinalIrradiance export) — pipeline.py will need this when wired up

**Low priority (fix when touching adjacent code):**
8. **CR-13** (A2 constants) — dormant, no current species uses A2
9. **CR-19** (comparison DPI) — cosmetic, add explicit dpi=100
10. **CR-22** (epsilon placement) — edge-case numerical stability

---

## Open Items

*(none)*

---

_Resolved items, verified-correct audit trail, and review history archived in `CODEREVIEW_RESOLVED.md`._
