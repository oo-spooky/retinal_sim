# SCRATCHPAD.md

Non-obvious debugging solutions, architectural decisions, and gotchas.
Update this file whenever you discover something that would waste time if rediscovered.

---

## Setup gotchas

- `pip install -e ".[dev]"` can fail with `BackendUnavailable: Cannot import setuptools.build_meta`.
  Fix: `pip install setuptools wheel && pip install --no-build-isolation -e ".[dev]"`

---

## Architecture decisions

### SceneGeometry uses arctan, not small-angle approximation
The formula `angular_width_deg = 2 * arctan(w / (2*d))` is exact and handles close viewing
distances correctly. Do NOT use the small-angle `w/d` form — it diverges >10°.

### Retinal extent computed from angular subtense, not directly from scene dimensions
`retinal_width_mm = 2 * focal_length_mm * tan(angular_width_deg/2 * π/180)`
This is equivalent to `focal_length_mm * tan(arctan(w/(2*d))) * 2`, but going via degrees
makes the intermediate result (angular_width_deg) inspectable.

### Species accommodation: hard-cutoff model (Phase 3 PoC)
Defocus residual = max(0, accommodation_demand - species_max_accommodation).
No accommodation lag/lead curve — deferred. Max accommodation stored in
`SceneGeometry._MAX_ACCOMMODATION` dict, not in `SpeciesConfig`/YAML (it's a per-age param).

### Mosaic cell size must use max total density across the patch, not just center
`MosaicGenerator._peak_total_density()` scans a radial profile from ecc≈0 to patch edge.
The problem: for human, parafoveal cone+rod total (~365K/mm²) exceeds foveal cone density alone
(200K/mm²), so if cells are sized for the fovea the Bernoulli prob > 1.0 in the parafovea and
receptors are silently under-placed. Fix: size cells for the max total density found anywhere.

### PoC patch size is 2° — clipping logic matters
`scene_covers_patch_fraction` is the *fraction of the 2° patch* the scene subtends.
`clipped = True` when scene angular subtense > 2°. Both fields are informational;
no actual pixel cropping happens inside `SceneGeometry.compute()`.

---

## Test patterns

### Phase 1 (nomogram) — 51 tests, ~0.08 s
`pytest tests/test_retina.py -v`

### Phase 2 (species config) — 58 tests, ~0.20 s
`pytest tests/test_species.py -v`

### Phase 3 (scene geometry) — run after implementation
`pytest tests/test_scene.py -v`

### Phase 4 (mosaic generator) — 32 tests, ~9 s
`pytest tests/test_mosaic.py -v`
