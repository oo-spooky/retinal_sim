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

### CR-18: Acuity discriminability docstring describes wrong metric

- **File:** `retinal_sim/validation/acuity.py`
- **Function:** Module docstring (lines 3–14)
- **Issue:** Docstring describes `D = mean|r1 - r2| / (mean(r1 + r2) + eps)` but implementation uses Pearson correlation distance `D = 1 - corr(r1, r2)`. Implementation is correct; docstring is stale.
- **Fix:** Update docstring to describe the actual Pearson-correlation distance metric.

### CR-19: `render_mosaic_map` implicit DPI assumption

- **File:** `retinal_sim/output/comparison.py`
- **Function:** `render_mosaic_map()` (lines 73–74)
- **Issue:** Figure size divides pixel count by 100.0 assuming 100 DPI, but `dpi` is not set on the figure. If matplotlib default DPI differs, output size won't match intent.
- **Fix:** Add `dpi=100` to `plt.subplots()` call.

### CR-20: pyproject.toml missing package discovery — editable install broken

- **File:** `pyproject.toml`
- **Issue:** `reports/` directory causes setuptools flat-layout auto-discovery to fail with "Multiple top-level packages discovered." `pip install -e ".[dev]"` does not work.
- **Fix:** Add `[tool.setuptools.packages.find]` section with `include = ["retinal_sim*"]`.

### CR-21: `RetinalIrradiance` not exported from `optical/__init__.py`

- **File:** `retinal_sim/optical/__init__.py`
- **Issue:** `RetinalIrradiance` is defined in `optical/stage.py` but not re-exported. Tests import it directly. Will break pipeline integration when `RetinalSimulator` is wired up.
- **Fix:** Add `RetinalIrradiance` to `optical/__init__.py` exports.

### CR-22: Epsilon placement in correlation denominator

- **File:** `retinal_sim/validation/acuity.py`
- **Function:** `_discriminability_one_seed()` (~line 185)
- **Issue:** Epsilon `1e-12` added after sqrt. Mathematically cleaner (and avoids sqrt(0)) to add inside: `np.sqrt(dot(r1,r1) * dot(r2,r2) + 1e-12)`.
- **Fix:** Move epsilon inside the sqrt argument.

---

_Resolved items, verified-correct audit trail, and review history archived in `CODEREVIEW_RESOLVED.md`._
