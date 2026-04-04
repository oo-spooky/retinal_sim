# `retinal_sim` Architecture Audit With External Methodology Validation

Date: 2026-04-04

## Executive Verdict

**Verdict: `fit with important caveats`**

`retinal_sim` is architecturally sound for its stated proof-of-concept goal if that goal is phrased narrowly and honestly: a **species-comparative, physically-parameterized retinal simulation PoC** that preserves the main dependencies of retinal image scale, receptor spectral sensitivity, mosaic sampling, and compressive photoreceptor response.

The pipeline is **not** currently strong enough to support broader claims of physiologically complete ocular image formation or externally validated species-specific visual performance. The biggest constraint is the **optical stage**, where the architecture document describes a richer model than the implementation actually provides. Because the retinal stage includes a nonlinear Naka-Rushton transduction, missing optical throughput and anisotropy are not harmless omissions for cross-species comparison.

## Scope and Evidence Standard

This memo evaluates two questions separately:

1. Is the implemented software architecture coherent and fit for the repo's stated PoC purpose?
2. Are the core scientific methodologies correctly implemented, defensibly simplified, or not yet adequately supported by primary sources?

Repo docs and tests were treated as implementation evidence, not as independent scientific evidence. Scientific authority was taken from the cited papers and standards below.

## Primary Sources Used

- Govardovskii et al. 2000 visual pigment template:
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/11016572/),
  [Cambridge](https://www.cambridge.org/core/journals/visual-neuroscience/article/in-search-of-the-visual-pigment-template/A4738E821720092B7F5A233C4AB4962B)
- Smits 1999 RGB-to-spectrum method:
  [Taylor & Francis DOI page](https://www.tandfonline.com/doi/abs/10.1080/10867651.1999.10487511)
- CIE standard illuminants D65:
  [CIE term entry](https://www.cie.co.at/eilvterm/17-23-021),
  [ISO/CIE 11664-2:2022 summary](https://webstore.ansi.org/standards/iso/isocie116642022-2478143)
- CIE standard observer:
  [ISO/CIE 11664-1 summary](https://standards.globalspec.com/standards/detail?docId=14334118)
- Original Naka-Rushton lineage:
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/5918060/),
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC1395832/)
- Practical ERG-style Naka-Rushton parameter fitting context:
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/8223107/)

## Stage Verdicts

### Scene

**Architecture verdict:** Sound for the stated PoC goal.

**External-methodology verdict:** Faithfully implemented.

The scene stage uses standard geometric optics relations:
- angular subtense `2 * arctan(size / (2 * distance))`
- retinal image size derived from focal length
- species-specific retinal magnification through focal length

This is the right anchor for the entire pipeline, because it ties a generic RGB image to a physical scene scale before any species comparison occurs. The implementation in [retinal_sim/scene/geometry.py](C:\Users\steve\retinal_sim\retinal_sim\scene\geometry.py) matches the standard equations and is well validated by the distance-scaling tests and report artifacts.

The hard-cutoff accommodation model is a simplification, but it is explicitly documented and does not undermine the PoC claim set if kept within scope.

### Spectral

**Architecture verdict:** Sound with important caveats.

**External-methodology verdict:** Implemented as a defensible simplification.

The repo correctly treats RGB-to-spectrum recovery as underdetermined and uses it for comparative simulation rather than absolute spectral truth. That is methodologically appropriate.

The implementation in [retinal_sim/spectral/upsampler.py](C:\Users\steve\retinal_sim\retinal_sim\spectral\upsampler.py) is **not a canonical reproduction of Smits 1999**. Smits describes a simple reflectance-oriented RGB-to-spectrum method using fixed basis spectra and a practical decomposition. This repo instead:
- keeps the same seven-basis decomposition structure,
- but computes the basis numerically with constrained least squares,
- and optimizes them for D65/CIE roundtrip behavior on the chosen wavelength grid.

That makes the current method best described as a **Smits-inspired, D65-optimized implementation variant**. For this repo's comparative use case, that is defensible. It is not appropriate to describe it as a strict implementation of the original paper without qualification.

Use of the CIE 1931 2° observer and D65 is methodologically appropriate for:
- sRGB roundtrip checks,
- relative cone-catch comparisons on standard RGB inputs,
- deterministic regression tests within this pipeline.

It is not enough to claim unique physical reconstruction of scene spectra.

### Optical

**Architecture verdict:** Internally consistent but methodologically weak relative to the architecture spec.

**External-methodology verdict:** Mixed. Energy normalization is faithfully implemented, but the stage as a whole is only partially supported and currently dominated by placeholders.

This is the most important weakness in the current system.

The architecture document describes:
- pupil throughput scaling,
- slit-pupil anisotropy for cat,
- wavelength-dependent PSF behavior including LCA,
- media transmission and scatter,
- and a more physically grounded optical stack.

The implementation in [retinal_sim/optical/psf.py](C:\Users\steve\retinal_sim\retinal_sim\optical\psf.py) and [retinal_sim/optical/stage.py](C:\Users\steve\retinal_sim\retinal_sim\optical\stage.py) actually provides:
- a wavelength-scaled Gaussian PSF,
- optional defocus broadening,
- optional media-transmission attenuation,
- exact kernel normalization in float64,
- but **no pupil-area throughput scaling**,
- **no slit-pupil anisotropy**,
- **no implemented diffraction PSF**,
- **no implemented LCA blur shift across wavelength**,
- and no realistic scatter model.

The PSF tests validate the implemented Gaussian model against itself. They do not externally validate that the Gaussian optical model is sufficient for species comparison. This matters because missing throughput and geometry differences can interact with transduction nonlinearity and change cross-species relative outcomes.

### Retinal

**Architecture verdict:** Sound with important caveats.

**External-methodology verdict:** Mixed.

The retinal stage contains two distinct methodology classes:

1. **Photoreceptor sensitivities**
   These are strong. The implementation in [retinal_sim/retina/opsin.py](C:\Users\steve\retinal_sim\retinal_sim\retina\opsin.py) closely follows Govardovskii et al. 2000: alpha-band template plus lambda-max-dependent beta-band Gaussian, with A1 and A2 constants separated. This is close enough to call the nomogram implementation externally validated.

2. **Transduction and sampling**
   These are acceptable PoC abstractions, not externally species-validated physiology.

The use of Naka-Rushton in [retinal_sim/retina/transduction.py](C:\Users\steve\retinal_sim\retinal_sim\retina\transduction.py) is methodologically reasonable as a generic compressive response model. The literature supports the form of the nonlinearity. It does **not** validate the specific chosen cone and rod parameters here as species-specific truth. Those values should be treated as modeling choices used to keep the PoC in a useful operating regime.

Likewise, the mosaic generation and bilinear sampling approach are sensible PoC engineering choices for a comparative retinal front end. They are not a faithful reproduction of full retinal anatomy, especially where the architecture spec discusses visual streaks, aperture weighting, and in-vivo media filtering.

### Validation and Reporting

**Architecture verdict:** Strong for transparency, weaker for external scientific validation.

**External-methodology verdict:** Implemented correctly for transparency, but often misread if treated as stronger evidence than it is.

The validation/reporting layer is a genuine strength of the repo. The report artifacts explicitly expose assumptions, thresholds, code paths, and limitations. That is excellent audit hygiene.

But the current validation suite mostly proves one of three things:
- analytic correctness of formulas,
- consistency of downstream behavior with the implemented model,
- or plausibility against broad expected ranges.

It rarely proves independent external validity of the whole biological claim. The warning language in `reports/validation_report.json` is honest about this, and that honesty should be preserved.

## High-Priority Findings

### 1. Optical-stage realism gap is the main threat to comparative validity

**Severity:** High  
**Type:** Methodology / implementation gap  
**Affected subsystem:** Optical stage

The architecture spec promises a richer optical model than the code delivers. Missing pupil throughput scaling is especially important because photoreceptor responses are nonlinear downstream. A missing scalar before Naka-Rushton is not equivalent to a harmless omitted metadata term.

This is the single strongest reason the project is only `fit with important caveats` instead of fully fit for its intended comparative claims.

**Disposition:** Fix now or clearly downgrade cross-species optical claims.

### 2. The spectral stage should not be described as a strict Smits implementation

**Severity:** Medium  
**Type:** Claim-discipline issue  
**Affected subsystem:** Spectral stage

The current spectral upsampler preserves the spirit of Smits but changes the basis construction materially by solving a constrained least-squares optimization under D65/CIE assumptions. That is a valid engineering choice, but it should be called what it is.

If described as “Smits (1999)” without qualification, the repo overstates methodological fidelity.

**Disposition:** Document clearly.

### 3. Naka-Rushton is justified as a form, but not yet as species-specific physiology

**Severity:** Medium  
**Type:** External-validity limitation  
**Affected subsystem:** Retinal transduction

The nonlinearity itself is appropriate. The specific parameters in YAML and defaults are not externally validated in this repo as species-correct photoreceptor physiology. They are best understood as PoC operating-point parameters.

This does not break the architecture, but it limits how strongly the outputs can be interpreted biologically.

**Disposition:** Document clearly; defer species-specific parameter grounding unless the claim set expands.

### 4. Validation coverage is broad, but much of it is self-consistency rather than external validation

**Severity:** Medium  
**Type:** Validation interpretation risk  
**Affected subsystem:** Validation/reporting

Examples:
- PSF MTF tests validate the implemented Gaussian model against predicted behavior of that same model.
- Spectral roundtrip checks validate D65/CIE internal consistency, not uniqueness of recovered spectra.
- Behavioral validations use broad ranges or constructed stimuli, which are good PoC checks but not full external confirmation.

This is acceptable if the reports stay explicit about the difference.

**Disposition:** Keep the transparency language; do not equate “coverage” with external proof.

## Methodology Crosswalk

| Method | Source | What the source supports | Repo implementation status | Audit verdict |
|---|---|---|---|---|
| Scene angular subtense and retinal magnification | Standard geometric optics; implemented equations align with accepted formulae | Physical scene size and distance determine angular size; focal length determines retinal image scale | [scene/geometry.py](C:\Users\steve\retinal_sim\retinal_sim\scene\geometry.py) matches the standard relations | **Faithfully implemented** |
| Govardovskii A1/A2 pigment template | Govardovskii et al. 2000 | Modified universal alpha-band plus lambda-max-dependent beta-band template for A1/A2 pigments | [retina/opsin.py](C:\Users\steve\retinal_sim\retinal_sim\retina\opsin.py) closely follows the paper’s structure and constants | **Faithfully implemented** |
| Smits RGB-to-spectrum recovery | Smits 1999 | Practical RGB-to-reflectance recovery using a simple physically plausible basis method | [spectral/upsampler.py](C:\Users\steve\retinal_sim\retinal_sim\spectral\upsampler.py) keeps the decomposition idea but replaces the classic basis with constrained LS D65-optimized bases | **Implemented as defensible simplification** |
| CIE 1931 2° observer and D65 | ISO/CIE 11664-1 and 11664-2 | Standard observer functions and illuminants for colorimetry | Used consistently for basis construction and roundtrip validation in [spectral/upsampler.py](C:\Users\steve\retinal_sim\retinal_sim\spectral\upsampler.py) and [tests/test_spectral.py](C:\Users\steve\retinal_sim\tests\test_spectral.py) | **Faithfully implemented for intended use** |
| Naka-Rushton response form | Naka & Rushton 1966 lineage; later ERG fitting literature | A saturating nonlinear response form is appropriate | [retina/transduction.py](C:\Users\steve\retinal_sim\retinal_sim\retina\transduction.py) uses the correct general form, but parameter values are PoC choices | **Implemented as defensible simplification** |
| Diffraction/PSF optics | Architecture doc implies richer physical optics than currently implemented | Optical blur should depend on aperture, wavelength, and species-specific geometry | [optical/psf.py](C:\Users\steve\retinal_sim\retinal_sim\optical\psf.py) uses a Gaussian placeholder and warns on slit-pupil mismatch | **Not correctly supported for the full architecture claim** |
| Optical throughput across species | Basic optics expectation from pupil area / aperture | Species differences in pupil size affect flux at the retina | Explicitly missing in [optical/stage.py](C:\Users\steve\retinal_sim\retinal_sim\optical\stage.py) | **Not correctly supported** |
| Validation report transparency | Internal repo methodology | Expose assumptions, methods, pass criteria, and provenance | [validation/report.py](C:\Users\steve\retinal_sim\retinal_sim\validation\report.py) does this well | **Faithfully implemented** |

## Validation-Quality Assessment

### What is externally strong

- Scene geometry equations
- Govardovskii nomogram implementation
- CIE/D65 roundtrip machinery as an internal colorimetric reference frame

### What is mostly internal-consistency evidence

- Gaussian PSF energy conservation and model-relative MTF checks
- Stimulus-panel spectral ordering checks
- Many pipeline-level left/right or figure/ground discriminability tests

### What is plausibility evidence rather than direct external confirmation

- Broad-range acuity checks
- Dichromat confusion reproductions on constructed stimuli
- Species ordering on synthetic panels

The current report stack should therefore be read as:
“the implemented model is coherent, reproducible, and transparent”
not
“the biology has been independently validated end-to-end.”

## Claim Boundaries

### Can say now

- The pipeline is a coherent PoC for comparing how species-specific focal length, receptor sensitivities, mosaic density, and a compressive receptor nonlinearity alter retinal sampling of the same scene.
- The scene geometry stage is physically grounded.
- The photopigment nomogram implementation is closely aligned with Govardovskii 2000.
- The spectral stage is suitable for comparative simulation on shared RGB inputs under a D65/CIE framework.
- The validation/reporting layer is unusually transparent about assumptions and provenance.

### Should say only with caveats

- The software simulates species-specific retinal image formation.
  Caveat: currently only at PoC level, with a simplified optical front end.
- The software reproduces known acuity and color-deficit trends.
  Caveat: these are model- and stimulus-dependent plausibility checks, not full external validation.
- The software supports cross-species comparisons.
  Caveat: optical omissions currently limit the strength of those comparisons.

### Should not say yet

- That the optical model is a realistic species-specific model of human, dog, and cat anterior-eye optics
- That the spectral recovery is a unique or physically true reconstruction of scene spectra
- That Naka-Rushton parameters used here are externally validated species-specific receptor physiology
- That passing validation reports demonstrate end-to-end physiological correctness

## Direct Answers to the Audit Questions

### Is the architecture sound for comparative species simulation, even if it is not yet a full physiological eye model?

Yes, **with important caveats**. The stage decomposition is sound and the interfaces are appropriate for the PoC goal.

### Is the spectral stage implemented correctly for the chosen Smits-style methodology?

Yes, **as a defensible Smits-inspired variant**, not as a canonical paper-faithful reproduction.

### Is the Govardovskii implementation faithful enough to the paper to treat receptor sensitivities as externally validated?

Yes. This is one of the strongest externally grounded parts of the repo.

### Is the Naka-Rushton stage a justified abstraction, or is it currently too under-sourced to support strong biological claims?

It is a **justified abstraction** for PoC transduction, but it is under-sourced for strong species-physiology claims.

### Is the optical stage the dominant threat to comparative validity?

Yes. This is the dominant architectural and methodological risk.

### Do the current validation reports mostly demonstrate transparency and internal reproducibility rather than true external validation?

Yes. That is not a failure, but it must remain explicit in how results are framed.

## Bottom Line

`retinal_sim` is currently achieving its intended purpose **only if that purpose is framed narrowly and honestly**:

> a transparent, physically-parameterized, species-comparative retinal simulation PoC

It is **not yet** a fully supported model of species-specific ocular image formation. The main step between those two levels is not more output polish or more self-consistency tests. It is improving or more carefully scoping the **optical methodology**, because that is the part most likely to change cross-species conclusions in scientifically meaningful ways.
