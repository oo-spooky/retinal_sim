# CODEREVIEW.md

Code review findings for `retinal_sim`. Open items should be addressed before starting new phase work. Resolved items are moved here rather than deleted for traceability.

**Review date:** 2026-03-30
**Reviewer:** Opus (code review session)
**Scope:** Phases 1–4 audited against `retinal_sim_architecture.md`
**Test results:** 165 passed, 3 skipped (stubs) in 9.12s

---

## Open Items

### CR-1: `_species_name` never set on OpticalParams — accommodation defaults to human for all species

- **File:** `retinal_sim/species/config.py`
- **Function:** `_build_optical()`
- **Issue:** `SceneGeometry.compute()` (geometry.py:118) does `getattr(optical_params, "_species_name", None)` to look up per-species max accommodation in `_MAX_ACCOMMODATION`. But `_build_optical()` never sets this attribute, so `getattr` always returns `None`, and `_MAX_ACCOMMODATION.get(None, 10.0)` always returns the 10.0D human default. Dog (2.5D) and cat (3.0D) accommodation limits are silently ignored. This means defocus residual is wrong for non-human species at near viewing distances.
- **Fix:** Add `params._species_name = species_name` in `_build_optical()`, or pass species name as an argument. Alternatively, add `species_name: Optional[str] = None` as a proper field on `OpticalParams`.
- **Severity:** High — produces physically incorrect results for dog/cat near-distance scenes.

### CR-2: Naka-Rushton rod sigma mismatch between defaults and YAML (0.05 vs 0.1)

- **File:** `retinal_sim/retina/transduction.py`
- **Function:** `NAKA_RUSHTON_DEFAULTS` (line 11)
- **Issue:** `NAKA_RUSHTON_DEFAULTS["rod"]["sigma"]` is 0.05, but all three YAML species files specify rod sigma = 0.1. This is a factor-of-2 difference in rod half-saturation. Code that uses the defaults dict (rather than loading from YAML) will get a different rod saturation curve. The YAML value of 0.1 is more standard for photopic-normalized relative units.
- **Fix:** Change `transduction.py` line 11 to `"sigma": 0.1` to match YAML, or remove the defaults dict and always require loading from config.
- **Severity:** Medium — PoC is photopic-only so rods are less important, but this will silently break scotopic/mesopic work in later phases.

### CR-3: Mosaic patch extent uses `focal_length_mm` where architecture doc §3a specifies `axial_length`

- **File:** `retinal_sim/retina/mosaic.py`
- **Function:** `MosaicGenerator.__init__()` (line 106)
- **Issue:** The code maps angular patch extent to mm via `focal_length_mm × tan(half_deg)`. Architecture §3a says: "mapped to mm via `mm = tan(deg) × axial_length`." For human, this is 22.3mm vs 24.0mm — a 7.6% difference in patch area. The code is arguably *more physically correct* than the doc (focal length governs image magnification, not axial length), and is *consistent with scene geometry* (geometry.py:102 also uses `focal_length_mm`). However, the Nyquist validation in §11c explicitly says `mm_per_degree = axial_length × tan(1°)`. This creates an internal inconsistency: validation targets derived from `axial_length` are being tested against a mosaic built with `focal_length`.
- **Fix:** Document this as an intentional deviation from §3a (focal length is physically correct for retinal image magnification). Update the Nyquist validation formula in the architecture doc to use `focal_length_mm` for consistency.
- **Severity:** Low — the code is more correct than the doc, but the inconsistency could cause confusion when implementing Phase 9 (Snellen validation).

### CR-4: Cone density model uses symmetric Gaussian — no visual streak for dog/cat

- **File:** `retinal_sim/species/config.py`
- **Function:** `_make_cone_density_fn()` (line 93)
- **Issue:** Architecture §3a table lists dog as "Weak" visual streak and cat as "Strong (horiz.)" visual streak. The current density function is a radially symmetric Gaussian: `peak × exp(-(ecc/σ)²)`, which has no angular dependence. The `angle` parameter in the signature is accepted but ignored. For cat especially, the strong horizontal visual streak means cone density should be elevated along the horizontal meridian — this is a defining feature of feline vision. Without it, the cat mosaic is missing a major species-specific characteristic.
- **Fix:** This is acknowledged as a PoC simplification (angle parameter is there for the upgrade path). Add a comment in `_make_cone_density_fn` explicitly noting that visual streak modeling is deferred, and add a TODO marker so Phase 5+ doesn't forget.
- **Severity:** Medium — cat simulation is qualitatively wrong without visual streak, but acceptable for PoC if documented.

### CR-5: Hardcoded wavelength range in MosaicGenerator (380–720nm, 5nm) duplicated from spectral stage

- **File:** `retinal_sim/retina/mosaic.py`
- **Function:** `MosaicGenerator.__init__()` (line 99-100)
- **Issue:** The default wavelength range `np.arange(380, 721, 5)` is hardcoded here and will also be hardcoded in `SpectralUpsampler` (upsampler.py:18 already specifies 380–720 at 5nm). If either is changed independently, the sensitivity curves and spectral image will have different wavelength axes, causing silent shape mismatches or incorrect dot products during spectral integration (Phase 7).
- **Fix:** Define the default wavelength range in one canonical location (e.g., a project-level constant or in `SpeciesConfig`) and import it everywhere. Alternatively, require wavelengths to be passed explicitly from the pipeline orchestrator.
- **Severity:** Low — not a bug today, but a latent integration hazard for Phase 7.

### CR-6: Magic number 1.1 in cell-size calculation undocumented

- **File:** `retinal_sim/retina/mosaic.py`
- **Function:** `MosaicGenerator.generate()` (line 132)
- **Issue:** `cell_size = 1.0 / np.sqrt(max(peak_density, 1.0) * 1.1)` — the 1.1 factor is a safety margin to prevent Bernoulli probabilities from hitting 1.0, but this is not documented in the code. Without explanation, a future developer might remove it (causing under-placement) or increase it (wasting grid cells and memory).
- **Fix:** Add a brief comment: `# 1.1 safety margin: ensures cell_area × peak_density < 1.0 everywhere, so Bernoulli acceptance is never capped`.
- **Severity:** Low — cosmetic, but affects maintainability.

### CR-7: Cone aperture linear model lacks rod aperture eccentricity dependence

- **File:** `retinal_sim/retina/mosaic.py`
- **Function:** `_cone_aperture_um()` (line 46), `_ROD_APERTURE_UM` (line 88)
- **Issue:** Cone aperture grows linearly with eccentricity (2–10µm), which is a reasonable first approximation. Rod aperture is a constant 2µm regardless of eccentricity. In reality, rod inner segments also grow with eccentricity (~1µm foveal to ~3µm peripheral in human). More importantly, the cone aperture slope `2.0 + 4.0 × ecc_mm` gives 10µm at 2mm eccentricity — this is realistic for peripheral human cones but for dog/cat cones (which have different inner segment diameters), the same function is applied without species parameterization.
- **Fix:** For PoC, add a comment noting this is a human-centric approximation. For later phases, move aperture parameters into species YAML.
- **Severity:** Low — aperture affects spatial sampling weight (Phase 7 spectral integration), not mosaic layout.

### CR-8: `SpeciesConfig.optical` and `.retinal` typed as `object` — no static type checking

- **File:** `retinal_sim/species/config.py`
- **Function:** `SpeciesConfig` dataclass (lines 18-19)
- **Issue:** `optical: object` and `retinal: object` lose all type information. Downstream code that accesses `config.optical.focal_length_mm` gets no IDE support and no mypy checking. The types are `OpticalParams` and `RetinalParams` respectively.
- **Fix:** Use forward references: `optical: "OpticalParams"` and `retinal: "RetinalParams"` with `from __future__ import annotations` (already imported). Or use `TYPE_CHECKING` guard for the imports.
- **Severity:** Low — doesn't affect runtime, but hurts developer experience and static analysis.

---

## Verified Correct

The following were audited and found correct against the architecture doc:

### Phase 1 — Govardovskii Nomogram (`retina/opsin.py`)
- A1 α-band template: coefficients (A=69.7, a=0.880, B=28.0, b=0.922, C=-14.9, c=1.104, D=0.674) match Govardovskii et al. (2000) Table 1 verbatim.
- A2 α-band: separate coefficient set correctly implemented.
- β-band: Gaussian parameterization with published center/width coefficients for both A1 and A2.
- `x = λ_max / λ` transformation correctly applied.
- Peak normalization to 1.0 is standard.
- `LAMBDA_MAX` values match architecture §3b exactly for all species/receptor combinations.
- `build_sensitivity_curves()` correctly dispatches by species name.

### Phase 2 — Species Config (`species/config.py`, YAML files)
- All optical parameters match architecture §2b table (focal lengths, axial lengths, pupil shapes, corneal radii).
- All retinal parameters match architecture §3a table (cone types, peak densities, rod-free zone).
- Cone ratios match: human 1:16:32, dog 3:97, cat 5:95.
- Density function factories produce correct Gaussian falloff models.
- Rod-free zone correctly implemented (human only, 0.175mm radius = 0.35mm diameter foveal pit).

### Phase 3 — Scene Geometry (`scene/geometry.py`)
- Angular subtense: exact arctan formula, not small-angle approximation.
- Retinal extent: `2 × focal_length × tan(angle/2)` correct.
- Accommodation demand: `1/distance` diopters correct.
- Defocus residual: `max(0, demand - max)` correct.
- Patch clipping logic and coverage fraction correct.
- Infinity (collimated) case handled correctly (zero angular extent, zero retinal extent, zero accommodation).
- All fields in `SceneDescription` match architecture §0 output spec.

### Phase 4 — Mosaic Generator (`retina/mosaic.py`)
- Jittered grid algorithm correctly implements Bernoulli acceptance with probability = `local_density × cell_area`.
- Peak density radial scan prevents probability capping (SCRATCHPAD gotcha addressed).
- Stochastic type assignment respects local cone/rod density ratio.
- Cone type sub-assignment uses `cone_ratio_fn` probabilities correctly.
- `PhotoreceptorMosaic` dataclass matches architecture §3a data structure.
- Sensitivity curves built directly from `RetinalParams.cone_peak_wavelengths` without species-name lookup — clean design.

### Stubs (Phases 5–8, pipeline)
- All stub signatures and dataclass definitions match architecture doc.
- `RetinalIrradiance`, `SpectralImage`, `MosaicActivation`, `SimulationResult` all have correct fields.
- Method signatures on `OpticalStage`, `RetinalStage`, `SpectralUpsampler`, `RetinalSimulator` match architecture.

---

## Resolved Items

_No resolved items yet._
