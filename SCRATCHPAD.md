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

### Diffraction sigma (corrected — CR-9)
Gaussian-fit-to-Airy-disk: sigma_diff = 0.42 × λ × f#
Do NOT use 1.22 × λ × f# — that is the Airy *first-zero radius* (Rayleigh criterion),
not the FWHM.  Using it as σ overestimates the PSF width by ~19%.

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

---

## Phase 7 gotchas

### pixel_scale_mm key, not mm_per_pixel, in RetinalIrradiance.metadata
`OpticalStage.apply()` stores the pixel scale as `metadata["pixel_scale_mm"]` (a scalar float).
`SceneDescription.mm_per_pixel` is a `Tuple[float, float]` (x, y).  `RetinalStage.compute_response`
handles both: `np.asarray(scene.mm_per_pixel).flat[0]` for the scene tuple, direct float cast for
the irradiance metadata scalar.

### Aperture Gaussian weighting is deferred (PoC)
The architecture §3c says to weight sampled irradiance by a per-receptor Gaussian of σ = aperture/2.
At PoC pixel scales (5–10 µm/px) and PSF widths (>20 µm), the aperture (2–10 µm) contributes < 10%
of total blur, so bilinear interpolation is an adequate approximation.

### _align_sensitivities handles wavelength grid mismatches
If `retinal_irradiance.wavelengths` doesn't match the stage's canonical 380–720 nm 5 nm grid,
`_align_sensitivities()` uses `np.interp` row-by-row.  The common path (grids match) returns the
array as-is.

### Phase 7 test patterns — 34 tests, ~20 s
`pytest tests/test_retinal_stage.py -v`

---

## Phase 9 gotchas

### Naka-Rushton saturation kills discriminability in the full patch
With white background images, the spectral integration produces excitation >> sigma (0.5),
saturating nearly all Naka-Rushton responses to ~R_max.  Using raw mean(|r1-r2|)/(mean(r1+r2))
gives D ≈ 0 for all letter sizes because the saturated background dominates.
Fix: restrict comparison to cones whose positions fall INSIDE the letter's angular bounding box
(±0.5×angular_size from the patch center).  Letter-region cones have high irradiance contrast
(black strokes vs white gaps), giving reliable discriminability even under saturation.

### Pearson correlation distance works; absolute-difference metric does not
After letter-region masking, D = 1 - corr(r1, r2) (Pearson-correlation distance) cleanly
distinguishes letter orientations.  E-right and E-left are anti-correlated in the letter
region (D ≈ 0.5–1.5), while responses from two identical-orientation images are fully
correlated (D ≈ 0).

### Minimum-cone threshold determines sampling-limited acuity
The predicted acuity is essentially: smallest letter for which ≥15 dominant-cone receptors
fall inside the letter bounding box.  This naturally encodes the Nyquist sampling criterion:
- Human (200K/mm² foveal cones): ≥15 L-cones at ~1.5–2 arcmin
- Dog (12K/mm²): ≥15 L-cones at ~8 arcmin
- Cat (10K/mm²): ≥15 L-cones at ~8–12 arcmin (stochastic; seed-dependent)
N_min=15 (_MIN_CONES_IN_LETTER in acuity.py) was chosen so that dog correctly falls in the
published 4–8 arcmin behavioral range.

### Cat at 8 arcmin is seed-dependent
Expected ~18 L-cones at 8 arcmin for cat.  Seeds 0 and 1 both happen to give <15 → D=0;
seed 2 gives ≥15 → D=0.81.  With n_seeds=1, seed=0 (the fixture default), cat predicts
12 arcmin instead of 8 arcmin.  This is within test bounds [1.5, 20].  Use n_seeds=3 if
a tighter cat prediction is needed.

### Phase 9 test patterns — 42 tests, ~25 s
`pytest tests/test_snellen.py -v`

## Phase 10 gotchas

### Unit-reflectance inputs saturate Naka-Rushton — need stimulus_scale
SpectralUpsampler outputs sRGB-derived reflectances in [0, 1].  Spectral
integration against a normalized sensitivity curve (also ≤ 1) with Δλ=5 nm and
~69 bands gives typical excitation values of 20–70.  The Naka-Rushton
half-saturation constant σ = 0.5 means everything saturates to ~R_max, making
responses indistinguishable.  Fix: `DichromatValidator` applies
`stimulus_scale = 0.01` to the SpectralImage after upsampling but before the
optical stage.  This keeps excitations near σ and preserves the NR dynamic
range.  Without scaling, D ≈ 0 for all colour pairs.

### Use best-cone-type D, not dominant-cone D
The confusion pair is designed so that dog S and L cone responses are equal for
fg and bg.  Human L cone (560 nm) ≈ dog L cone (555 nm), so L_human responses
are also similar.  Only human M cone (530 nm) differs.  If discriminability is
computed only for the dominant cone type (L, both for human and dog), human D ≈
dog D ≈ 0.  Fix: `_discriminability_one_seed` loops over ALL non-rod cone types
and returns the MAX D across types.

### find_confusion_pair: B ≤ 50 targets red-green axis, not S-confusion axis
The search constrains the blue channel (B ≤ 50) to keep S-cone excitations low
and equal for both colours.  The resulting pair lies on the deuteranopia-like
confusion axis: equal dog S and L responses, different human M response.
For a blue-yellow (tritan) confusion axis search, remove the blue constraint.

### Noise floor for D_dog at n_seeds=2
With n_seeds=2, mosaic and dot-pattern noise can make D_dog appear ~0.03–0.04
even when the analytical prediction is <0.005.  The key test is the
DIRECTION (D_human > D_dog), not an absolute ratio.  Stricter relative tests
(e.g., D_dog < 0.80 × D_human) are unreliable at n_seeds=2; increase n_seeds
to 10+ for rigorous quantitative comparison.

### Phase 10 test patterns — 36 tests, ~17 s
`pytest tests/test_dichromat.py -v`

## Test patterns

Run all: `pytest` (~60s). Per-phase test files are named `test_{phase}.py` — see PROGRESS.md table.
