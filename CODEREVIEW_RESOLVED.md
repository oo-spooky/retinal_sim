# CODEREVIEW — Resolved Items & Audit Trail

Archived from `CODEREVIEW.md` to reduce token usage. Items here are resolved and verified.

---

## Verified Correct

The following were audited and found correct against the architecture doc:

- **Phase 1** — Govardovskii Nomogram (`retina/opsin.py`): A1/A2 α-band, β-band, x=λ_max/λ transform, peak normalization, LAMBDA_MAX values, `build_sensitivity_curves()` all verified.
- **Phase 2** — Species Config (`species/config.py`, YAMLs): optical params, retinal params, cone ratios, density functions, rod-free zone all match architecture doc.
- **Phase 3** — Scene Geometry (`scene/geometry.py`): arctan formula, retinal extent, accommodation, defocus, patch clipping, infinity case all correct.
- **Phase 4** — Mosaic Generator (`retina/mosaic.py`): Bernoulli jittered grid, peak density scan, type assignment, cone ratios, PhotoreceptorMosaic dataclass all correct.
- **Phase 5** — Gaussian PSF (`optical/psf.py`, `optical/stage.py`): kernel normalization, sigma quadrature, defocus CoC, convolution mode, media transmission, RetinalIrradiance all correct.
- **Phase 6** — Spectral Upsampler (`spectral/upsampler.py`): CIE CMFs, D65, sRGB matrix, lsq_linear basis, Smits decomposition, roundtrip RMSE all correct.
- **Phase 7** — Retinal Stage (`retina/stage.py`): bilinear interp, spectral integration, Naka-Rushton, per-type dispatch, pixel-scale chain, grid alignment all correct.
- **Phase 5–7** — Transduction (`retina/transduction.py`): formula, negative guard, default params all correct.
- **Phase 1–7** — Cross-phase: LAMBDA_MAX↔YAML consistency, cone ratio sums, accommodation limits all correct.
- **Phase 8** — Voronoi Visualization (`output/`): KD-tree tessellation, type→hue mapping, response clipping, multi-species panels all correct.
- **Phase 9** — Snellen Acuity (`validation/`): E template, angular conversion, pipeline discriminability, species ordering, published data separation all correct.
- **Phase 1–9** — Structural: wavelength grid centralized, no circular imports, type consistency, pipeline.py still stubbed (expected).

---

## Resolved Items

### CR-1: `_species_name` never set on OpticalParams ✓
- **Fix:** Added `species_name` arg to `_build_optical()`, set `params._species_name`.

### CR-2: Naka-Rushton rod sigma mismatch (0.05 vs 0.1) ✓
- **Fix:** Changed default to 0.1 to match YAML.

### CR-3: Mosaic patch extent uses `focal_length_mm` not `axial_length` ✓
- **Fix:** Updated architecture doc to match code (focal length is correct).

### CR-4: No visual streak for dog/cat ✓
- **Fix:** Added docstring noting PoC simplification.

### CR-5: Hardcoded wavelength range duplicated ✓
- **Fix:** Added TODO; later resolved fully by CR-17 (centralized in `constants.py`).

### CR-6: Magic number 1.1 undocumented ✓
- **Fix:** Added explanatory comment.

### CR-7: Cone aperture model lacks rod eccentricity dependence ✓
- **Fix:** Added docstring noting PoC approximation.

### CR-8: `SpeciesConfig` fields typed as `object` ✓
- **Fix:** Changed to forward references (`OpticalParams`, `RetinalParams`).

### CR-9: Gaussian PSF sigma overestimated ~19% ✓
- **Fix:** Changed to `sigma_diff_mm = 0.42 * lam_mm * f_number`.

### CR-10: sRGB gamma linearization missing ✓
- **Fix:** Applied IEC 61966-2-1 piecewise sRGB→linear transfer function.

### CR-11: Pupil throughput scaling missing ✓
- **Fix:** Added NOTE comment; must be implemented before Phase 12.

### CR-12: Cat slit pupil silently ignored ✓
- **Fix:** Added `warnings.warn()` for slit pupil.

### CR-13: A2 nomogram α-band uses A1 constants ✓
- **Fix:** Corrected `_A2_ALPHA` to Table 2 values.

### CR-14: MosaicActivation.mosaic typed as `object` ✓
- **Fix:** Changed to `PhotoreceptorMosaic`.

### CR-15: Sensitivity curves are in-vitro ✓
- **Fix:** Added docstring warning about PoC equivalence assumption.

### CR-16: Edge receptors receive edge-pixel irradiance ✓
- **Fix:** Added `oob` mask zeroing out-of-bounds receptors.

### CR-17: Wavelength grid triplicated ✓
- **Fix:** Created `constants.py` with canonical `WAVELENGTHS`.

### CR-18: Acuity discriminability docstring describes wrong metric ✓
- **Fix:** Updated module docstring to describe `D = 1 - corr(r1, r2)` (Pearson-correlation distance).

### CR-19: `render_mosaic_map` implicit DPI assumption ✓
- **Fix:** Added `dpi=100` to `plt.subplots()` call in `render_mosaic_map()`.

### CR-20: pyproject.toml missing package discovery — editable install broken ✓
- **Fix:** Added `[tool.setuptools.packages.find]` with `include = ["retinal_sim*"]`.

### CR-21: `RetinalIrradiance` not exported from `optical/__init__.py` ✓
- **Fix:** Added `RetinalIrradiance` to imports and `__all__` in `optical/__init__.py`.

### CR-22: Epsilon placement in correlation denominator ✓
- **Fix:** Moved `1e-12` inside `np.sqrt(... + 1e-12)` to guard against `sqrt(0)`.

---

## Review History

- **2026-04-02 (Opus):** Phases 8–9 + structural audit. 329 passed, 1 skipped. CR-18–CR-22.
- **2026-03-31 (Opus):** Phases 5–7 deep audit. 259 passed, 1 skipped. CR-9–CR-17.
- **2026-03-31 (Sonnet):** Resolved CR-9–CR-17. 259 passed, 1 skipped.
- **2026-03-30 (Opus):** Phases 1–4 initial audit. 165 passed, 3 skipped. CR-1–CR-8 (all resolved).
