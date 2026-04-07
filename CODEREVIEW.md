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

### CR-23 — Per-species LCA span not parameterized

- **File:** retinal_sim/optical/psf.py (and data/species/human.yaml, dog.yaml, cat.yaml)
- **Function:** `PSFGenerator._lca_offset_diopters`
- **Issue:** Human longitudinal chromatic aberration span is hardcoded at 1.0 D as a PoC simplification (noted in SCRATCHPAD) and is reused for all species. Dog and cat LCA characteristics are not separately parameterized, so cross-species wavelength-dependent defocus is not empirically grounded.
- **Fix:** Surface `lca_span_diopters` (or equivalent) in species YAML and read it in `PSFGenerator`. Update docs/provenance to cite the source when per-species values become available. Low priority — flag as future empirical tightening.

### CR-24 — Implicit ndarray serialization contract in diagnostic builders

- **File:** retinal_sim/output/diagnostics.py
- **Function:** `build_retinal_irradiance_diagnostics`, `build_photoreceptor_activation_diagnostics`
- **Issue:** Diagnostic image payloads are embedded as float32 ndarrays inside the artifact dicts and rely on `validation/report.py::_json_safe()` to become JSON-serializable. The contract is implicit and easy to break by a future refactor that touches either side.
- **Fix:** Either add a short comment at each builder documenting that artifact dicts must be `_json_safe`-round-trippable, or add a targeted unit test asserting round-trip through `_json_safe` for each R6 diagnostic family. Low priority — hardening, not a current bug.

### CR-25 — Visual streak extension point silently disabled

- **File:** retinal_sim/retina/stage.py
- **Function:** `RetinalStage` / `VisualStreakConfig`
- **Issue:** `VisualStreakConfig.enabled` defaults to `False`. The architecture doc lists the visual streak as a planned extension; the disabled-state tests cover current behavior, but there is no reminder that enabling it must be accompanied by new validation entries (claim support, external references).
- **Fix:** When the streak is enabled for any species, add a corresponding validation test under `tests/test_validation_report.py` with an appropriate claim-support level and external reference. Until then, no code change — tracking only.


---

_Resolved items, verified-correct audit trail, and review history archived in `CODEREVIEW_RESOLVED.md`._
