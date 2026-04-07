# CODEREVIEW.md

Open code review findings for `retinal_sim`.

Resolved items, verified audit trail, and review history are archived in `docs/archive/CODEREVIEW_RESOLVED_2026-04.md`.

## Current Review State

- Last consolidated audit: 2026-04-02 (Opus), structural audit plus phases 8-9 validation review.
- Current open findings: CR-23 through CR-25.

## Open Items

### CR-23 - Per-species LCA span not parameterized

- **File:** retinal_sim/optical/psf.py (and data/species/human.yaml, dog.yaml, cat.yaml)
- **Function:** `PSFGenerator._lca_offset_diopters`
- **Issue:** Human longitudinal chromatic aberration span is hardcoded at 1.0 D as a proof-of-concept simplification and reused for all species. Dog and cat LCA characteristics are not separately parameterized, so cross-species wavelength-dependent defocus is not empirically grounded.
- **Fix:** Surface `lca_span_diopters` (or equivalent) in species YAML and read it in `PSFGenerator`. Update docs and provenance when per-species values become available. Track as future empirical tightening.

### CR-24 - Implicit ndarray serialization contract in diagnostic builders

- **File:** retinal_sim/output/diagnostics.py
- **Function:** `build_retinal_irradiance_diagnostics`, `build_photoreceptor_activation_diagnostics`
- **Issue:** Diagnostic image payloads are embedded as `float32` ndarrays inside the artifact dicts and rely on `validation/report.py::_json_safe()` to become JSON-serializable. The contract is implicit and easy to break with a future refactor.
- **Fix:** Either document at each builder that artifact dicts must be `_json_safe`-round-trippable, or add a targeted unit test asserting round-trip through `_json_safe` for each R6 diagnostic family.

### CR-25 - Visual streak extension point silently disabled

- **File:** retinal_sim/retina/stage.py
- **Function:** `RetinalStage` / `VisualStreakConfig`
- **Issue:** `VisualStreakConfig.enabled` defaults to `False`. The architecture doc lists the visual streak as a planned extension, and the disabled-state tests cover current behavior, but there is no reminder that enabling it must be paired with new validation entries and claim support.
- **Fix:** When the streak is enabled for any species, add a corresponding validation test under `tests/test_validation_report.py` with appropriate claim-support level and external reference. Until then, keep this as a tracking item.
