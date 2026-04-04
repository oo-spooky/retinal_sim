# `retinal_sim` Remediation Roadmap

Date: 2026-04-04

## Goal

Advance `retinal_sim` from a strong comparative proof of concept to a **physically grounded retinal image formation simulator** that can more credibly answer:

> given the same real scene observed by a human, a dog, and a cat, what differs in the spectral irradiance at the retina, the receptor sampling, and the retained spatial/color information?

This roadmap is optimized for your stated intent:
- not a stylized “animal camera filter”
- not a vague color-remapping toy
- but a pipeline whose differences arise from modeled ocular and retinal structure

## Target End State

The project should ultimately support the following claim:

> `retinal_sim` models species-specific retinal image formation for human, dog, and cat from the same scene geometry, using physically parameterized optics, spectral sensitivity functions, receptor mosaics, and validated comparative outputs.

That claim is only credible if the roadmap closes the current gap between:
- a **comparative PoC with a simplified optical front end**
- and a **physically motivated eye-structure simulation**

## Strategic Principle

The next work should prioritize **scientific leverage**, not surface polish.

The highest-value upgrades are the ones that most change what reaches the retina before transduction:
1. optical throughput and aperture geometry
2. wavelength-dependent optics
3. clearer spectral scene assumptions
4. claim-calibrated validation against external evidence

## Roadmap Summary

### Phase R1: Fix the optical-stage validity floor

**Objective:** Make cross-species comparisons optically defensible.

This is the highest-priority phase because the current optical model is the dominant threat to physical accuracy.

Deliverables:
- Add pupil-area throughput scaling in [retinal_sim/optical/stage.py](C:\Users\steve\retinal_sim\retinal_sim\optical\stage.py)
- Separate circular and slit pupil handling explicitly in [retinal_sim/optical/psf.py](C:\Users\steve\retinal_sim\retinal_sim\optical\psf.py)
- Replace the “Gaussian-only, isotropic for all species” assumption with:
  - circular Gaussian/Airy-derived approximation for human and dog
  - anisotropic elliptical approximation for cat under slit-pupil conditions
- Extend optical metadata so every run records:
  - pupil area
  - effective f-number
  - PSF width by axis
  - defocus residual
  - whether anisotropy was active

Success criteria:
- Cross-species retinal irradiance differs when pupil geometry differs
- Cat optical output is directionally anisotropic when slit geometry is active
- Validation report explicitly shows throughput and PSF shape diagnostics

Why this phase comes first:
- It directly affects whether “dog vs cat vs human” differences are caused by modeled eye structure rather than downstream abstractions
- It resolves the strongest audit finding

### Phase R2: Make the optical model wavelength-aware in the way the architecture already promises

**Objective:** Move from “blur placeholder” toward actual species-specific anterior-eye simulation.

Deliverables:
- Implement longitudinal chromatic aberration as wavelength-dependent defocus offset
- Decide and document the operational model:
  - either per-wavelength PSF kernels
  - or a validated reduced approximation with quantified error
- Add media-transmission functions per species rather than leaving them mostly optional
- Route in-vivo prefiltering consistently so receptor catches use retinally delivered spectra, not raw in-vitro pigment curves

Recommended scope for this phase:
- Keep scatter simple
- Prioritize LCA and media transmission before more elaborate higher-order aberration work

Success criteria:
- Blue and long-wavelength bands show physically different blur/focus behavior where expected
- Species config actually changes delivered spectral irradiance before receptor integration
- Validation artifacts expose wavelength-dependent PSF behavior

Why this phase matters:
- Your goal is specifically to compare spectra, acuity, and resolution as they would be shaped by real ocular structures
- That requires wavelength-aware optics, not only wavelength-aware receptors

### Phase R3: Clarify scene/spectral semantics so the pipeline models the right physical thing

**Objective:** Ensure the input image is interpreted in a way that matches the intended physics.

Current issue:
- The pipeline starts from RGB imagery and a D65/CIE reconstruction framework, which is defensible for comparative work
- But it is still easy to overread the result as if the scene spectrum itself were known

Deliverables:
- Split input mode into explicit scene semantics:
  - `display_rgb`
  - `reflectance_under_d65`
  - `measured_spectrum` or `hyperspectral` bypass
- Document what each mode means physically
- Update [retinal_sim/spectral/upsampler.py](C:\Users\steve\retinal_sim\retinal_sim\spectral\upsampler.py) and [retinal_sim/pipeline.py](C:\Users\steve\retinal_sim\retinal_sim\pipeline.py) so the user cannot silently mix assumptions
- Rename docs and report text so “Smits” is described as a Smits-inspired D65-optimized reconstruction variant

Success criteria:
- Every simulation run records which scene assumption is active
- Reports distinguish “physically measured spectrum” from “RGB-inferred spectrum”
- Claim language stops implying unique spectral recovery from RGB

Why this phase matters:
- A physically accurate eye simulation still needs a physically meaningful scene model
- This phase prevents methodological ambiguity from undermining otherwise strong optics work

### Phase R4: Upgrade retinal physiology where it most affects interpretation

**Objective:** Keep the retinal stage as a model of retinal front-end sampling, not an overclaimed physiology block.

Deliverables:
- Keep Govardovskii as-is; it is already strong
- Add explicit species-level provenance notes for lambda-max values and density functions
- Treat Naka-Rushton parameters as configurable model parameters with declared provenance and confidence level
- Add optional receptor aperture weighting in [retinal_sim/retina/stage.py](C:\Users\steve\retinal_sim\retinal_sim\retina\stage.py)
- Add optional visual-streak anisotropy for dog/cat mosaics once optical-stage anisotropy is in place

Recommended non-goal for now:
- Do not try to model post-receptoral cortical appearance yet

Success criteria:
- The retinal stage is more anatomically grounded where it changes sampling or contrast retention
- The project remains honest that it simulates the retinal front end, not full perception

Why this phase matters:
- Your desired deliverable is “what their eye would deliver,” not “what their brain consciously sees”
- That means retinal-front-end realism is the right target, while perception-level rendering remains clearly out of scope

### Phase R5: Redesign validation around external validity, not only model self-consistency

**Objective:** Turn the validation suite into evidence that the upgraded pipeline earns its claims.

Deliverables:
- Split validation labels into:
  - `analytic correctness`
  - `model self-consistency`
  - `external empirical alignment`
- Add explicit external-validation tables for:
  - species acuity ranges
  - receptor density-derived Nyquist limits
  - pupil and focal-length optical consequences
  - wavelength transmission assumptions
- Add “claim support level” per result:
  - strong
  - moderate
  - weak
- Update [retinal_sim/validation/report.py](C:\Users\steve\retinal_sim\retinal_sim\validation\report.py) and the generated artifacts so readers can see which findings are externally grounded

Success criteria:
- A passed report no longer reads like blanket validation of the whole simulator
- External evidence and model-internal checks are visually distinct
- The report can support publication-quality internal review

Why this phase matters:
- Once optics are upgraded, the project’s bottleneck becomes credibility of interpretation, not implementation completeness

### Phase R6: Expand output products so they match the science

**Objective:** Make the outputs show physically meaningful differences, not just attractive differences.

Deliverables:
- Separate output products into three distinct families:
  - `retinal_irradiance diagnostics`
  - `photoreceptor activation diagnostics`
  - `human-readable comparative renderings`
- For each comparison, expose:
  - spectral irradiance slices
  - PSF diagnostics
  - mosaic sampling overlays
  - activation-space summaries
- Keep reconstructed “what they would see” style views, but label them as:
  - retinal-information renderings
  - not direct perceptual/cortical reconstructions

Success criteria:
- Users can inspect where differences entered the pipeline
- The rendering is traceable back to optics and receptor sampling
- The project avoids collapsing into a single misleading “animal vision filter” output

Why this phase matters:
- It keeps the software aligned with your original motivation: seeing the consequence of modeled biology, not of arbitrary image stylization

## Prioritized Implementation Order

### Priority 1: Required before stronger scientific claims

1. Pupil throughput scaling
2. Cat slit-pupil anisotropy
3. Wavelength-dependent optical behavior via LCA/defocus
4. Species media transmission wiring
5. Validation/report distinction between internal and external evidence

### Priority 2: Required before calling the system “physically grounded” without heavy caveats

1. Explicit scene input semantics
2. In-vivo prefiltering consistency for receptor catches
3. Receptor aperture weighting
4. Better provenance and confidence labeling for species parameters

### Priority 3: Valuable upgrades after the validity floor is fixed

1. Visual streak modeling
2. Larger patch sizes
3. Performance optimizations
4. More advanced scatter and higher-order aberration work

## Concrete Milestones

### Milestone M1: Optically defensible species comparison

This is the first major milestone worth targeting.

The project reaches M1 when:
- human, dog, and cat differ in retinal irradiance for reasons traceable to modeled aperture and wavelength-dependent optics
- cat slit-pupil anisotropy is implemented and exposed
- report artifacts show those differences explicitly
- documentation says “physically grounded comparative retinal simulation” only with bounded caveats

### Milestone M2: Physically grounded retinal front-end simulator

The project reaches M2 when:
- optical throughput, LCA, media transmission, receptor prefiltering, and scene semantics are all coherent
- validation includes external empirical alignment rather than mostly internal plausibility
- outputs can show where spectral, acuity, and resolution differences arise mechanistically

### Milestone M3: Publication-grade comparative simulation platform

The project reaches M3 when:
- parameter provenance is complete
- external validations are more comprehensive
- simplifications are quantified, not only described
- results can be defended as a scientific model rather than an engineering demo

## Recommended Work Packages

### Work Package A: Optical Core

Files likely affected:
- [retinal_sim/optical/stage.py](C:\Users\steve\retinal_sim\retinal_sim\optical\stage.py)
- [retinal_sim/optical/psf.py](C:\Users\steve\retinal_sim\retinal_sim\optical\psf.py)
- [retinal_sim/species/config.py](C:\Users\steve\retinal_sim\retinal_sim\species\config.py)
- [retinal_sim/data/species/human.yaml](C:\Users\steve\retinal_sim\retinal_sim\data\species\human.yaml)
- [retinal_sim/data/species/dog.yaml](C:\Users\steve\retinal_sim\retinal_sim\data\species\dog.yaml)
- [retinal_sim/data/species/cat.yaml](C:\Users\steve\retinal_sim\retinal_sim\data\species\cat.yaml)

Deliver:
- throughput scaling
- slit/circular branching
- anisotropic PSF support
- LCA-ready data model
- richer metadata

### Work Package B: Spectral Semantics

Files likely affected:
- [retinal_sim/spectral/upsampler.py](C:\Users\steve\retinal_sim\retinal_sim\spectral\upsampler.py)
- [retinal_sim/pipeline.py](C:\Users\steve\retinal_sim\retinal_sim\pipeline.py)
- [retinal_sim_architecture.md](C:\Users\steve\retinal_sim\retinal_sim_architecture.md)

Deliver:
- explicit input modes
- clarified docs
- claim-safe terminology

### Work Package C: Retinal Front-End Refinement

Files likely affected:
- [retinal_sim/retina/stage.py](C:\Users\steve\retinal_sim\retinal_sim\retina\stage.py)
- [retinal_sim/retina/mosaic.py](C:\Users\steve\retinal_sim\retinal_sim\retina\mosaic.py)
- [retinal_sim/retina/transduction.py](C:\Users\steve\retinal_sim\retinal_sim\retina\transduction.py)

Deliver:
- aperture weighting
- anisotropic density support
- better provenance labeling for transduction choices

### Work Package D: Validation and Reporting

Files likely affected:
- [retinal_sim/validation/report.py](C:\Users\steve\retinal_sim\retinal_sim\validation\report.py)
- [reports/validation_report.html](C:\Users\steve\retinal_sim\reports\validation_report.html)
- [reports/status_latest.html](C:\Users\steve\retinal_sim\reports\status_latest.html)

Deliver:
- external-vs-internal evidence separation
- claim support labeling
- optical-stage diagnostics

## Documentation Changes Needed Alongside Code

These should be updated as each phase lands:

- [retinal_sim_architecture.md](C:\Users\steve\retinal_sim\retinal_sim_architecture.md)
  Keep architecture intent aligned with actual implementation state.

- [PROGRESS.md](C:\Users\steve\retinal_sim\PROGRESS.md)
  Track remediation milestones separately from implementation phases.

- [CODEREVIEW.md](C:\Users\steve\retinal_sim\CODEREVIEW.md)
  Add architecture-remediation findings when they become code-level issues.

- [reports/architecture_audit_2026-04-04.md](C:\Users\steve\retinal_sim\reports\architecture_audit_2026-04-04.md)
  Treat as the baseline audit for measuring progress.

## Non-Goals for the Next Iteration

To keep the roadmap focused, these should remain out of scope until the optical validity floor is fixed:

- neural/cortical appearance modeling
- “what the animal consciously perceives” claims
- stylistic rendering work
- performance-first optimization
- publication-level full-eye aberration models beyond what materially changes species comparisons

## Recommended Near-Term Session Plan

The next coding sessions should follow this order:

1. Implement pupil throughput scaling and add species-level optical-energy tests.
2. Implement cat slit-pupil anisotropy with validation artifacts that show axis-dependent blur.
3. Add wavelength-dependent optical behavior and media transmission wiring.
4. Refactor spectral input semantics and tighten claim language.
5. Upgrade validation/reporting so the new physics is externally interpretable.

## Bottom Line

If the project’s true mission is:

> “show me, through physical simulation of the eye’s structures, how my dog or cat’s retinal input differs from mine in spectrum, acuity, and resolution”

then the roadmap should be read very simply:

- the project is already pointed in the right direction
- the **optical stage** is the bridge between “promising PoC” and “physically grounded simulator”
- once that bridge is strengthened, the existing scene, spectral, mosaic, and reporting architecture becomes much more scientifically valuable

The most important next move is not to make the outputs prettier. It is to make sure the **retina is receiving the right light for the right structural reasons**.
