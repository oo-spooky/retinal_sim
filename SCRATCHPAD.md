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

## Phase 5 gotchas

### PSF kernels are float64, not float32
The §11b energy criterion is |sum - 1.0| < 1e-6. Storing kernels as float32 (7 decimal digits
of precision) and summing 961 elements (31×31) accumulates ~5e-5 rounding error — exceeds the
threshold. `gaussian_psf` returns float64. The `RetinalIrradiance.data` is still float32 (image
data); only the kernel itself is float64.

### PSFGenerator pixel scale is a constructor arg, not a method arg
`PSFGenerator(optical_params, pixel_scale_mm_per_px=0.001)` — the scale bakes into the object.
`OpticalStage.apply()` constructs a *new* PSFGenerator with the actual scene pixel scale each
time it is called. `OpticalStage.compute_psf()` uses the default scale (1 µm/px); this is
intentional for standalone PSF inspection without a full scene.

### Defocus formula (thin-lens, PoC)
Circle-of-confusion radius: r_coc = focal_length_mm × defocus_D × pupil_diameter_mm / 2000
Gaussian sigma for defocus: sigma_defocus = r_coc / sqrt(2)
For human (f=22.3mm, D_p=3mm, δ=1D): r_coc ≈ 33 µm, sigma ≈ 24 µm.

---

## Phase 6 gotchas

### Smits (1999) 11-knot table values from memory were wrong for MAGENTA and YELLOW
The hand-recalled table had MAGENTA with zeros at 610–700nm and YELLOW with zeros at 520–580nm,
giving catastrophic roundtrip failures (RMSE > 0.18). Rather than chase the exact paper values,
the basis spectra are computed numerically at init via `scipy.optimize.lsq_linear`.

### D65 integration is required — equal-energy + per-channel normalisation does not work
The sRGB colorspace is D65-adapted. Roundtrip integration must use the D65 SPD as illuminant:
`k = 1 / (D65 @ CIE_Y * dlam)`, then `X = k * dlam * (spectrum * D65) @ CIE_X`, etc.
Per-channel white normalisation tricks fail for saturated colours (RMSE > 0.09).

### Smits MAGENTA exploits CIE x̄ secondary lobe at ~450 nm
The magenta basis spectrum has non-zero weight at ~440–460 nm (blue end) AND ~600–700 nm (red end),
exploiting the CIE x̄ secondary blue lobe. Simple box-filter integration cannot reconstruct this.
The numerically-optimised BLUE basis also has small but non-zero values above 530 nm (~0.02–0.04).
The `test_pure_blue_zero_at_long_wavelengths` threshold is 0.05, not 0.01.

### Phase 6 test patterns — 32 tests, ~0.7 s
`pytest tests/test_spectral.py -v`

## Test patterns

### Phase 1 (nomogram) — 51 tests, ~0.08 s
`pytest tests/test_retina.py -v`

### Phase 2 (species config) — 58 tests, ~0.20 s
`pytest tests/test_species.py -v`

### Phase 3 (scene geometry) — run after implementation
`pytest tests/test_scene.py -v`

### Phase 4 (mosaic generator) — 32 tests, ~9 s
`pytest tests/test_mosaic.py -v`

### Phase 5 (Gaussian PSF) — 28 tests, ~0.4 s
`pytest tests/test_optical.py -v`

### Phase 6 (Smits upsampler) — 32 tests, ~0.7 s
`pytest tests/test_spectral.py -v`
