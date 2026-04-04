# Retinal Simulation Pipeline — Software Architecture

## Overview

A physically-parameterized, species-configurable simulation of image formation on the mammalian retina. The pipeline is decomposed into independent processing layers with well-defined interfaces. Each layer operates on the output of the previous layer and is parameterized by species-specific physical constants.

```
Input Image (spectral or RGB) + Scene Geometry
        │
        ▼
┌─────────────────────┐
│   SCENE GEOMETRY    │  Map physical scene dimensions + viewing
│                     │  distance → angular subtense → retinal extent
└────────┬────────────┘
         │  SceneDescription: angular extent, retinal scale
         ▼
┌─────────────────────┐
│   SPECTRAL STAGE    │  Convert RGB input → spectral radiance estimate
│                     │  (spectral upsampling)
└────────┬────────────┘
         │  SpectralImage: (H, W, N_wavelengths)
         ▼
┌─────────────────────┐
│   OPTICAL STAGE     │  Pupil aperture, lens/cornea PSF,
│                     │  chromatic aberration, vitreous scatter
│                     │  + defocus from accommodation (if near)
└────────┬────────────┘
         │  RetinalIrradiance: (H, W, N_wavelengths)
         ▼
┌─────────────────────┐
│   RETINAL STAGE     │  Photoreceptor mosaic generation,
│                     │  spectral integration, transduction
└────────┬────────────┘
         │  MosaicActivation: (N_receptors,) + positions
         ▼
┌─────────────────────┐
│   OUTPUT STAGE      │  Voronoi visualization, comparative
│                     │  rendering, activation maps
└─────────────────────┘
```

---

## 0. Scene Geometry

### Purpose
Anchors the input image to physical reality. Without this, the simulation has no way to determine what spatial frequencies are present at the retina — a 500×500 pixel image could represent a postage stamp at arm's length or a building at 200 meters. Those produce radically different retinal images despite identical pixel data.

### Input
- `scene_width_m`: Physical width of the scene the image represents (meters)
- `scene_height_m`: Physical height (if non-square; default: inferred from image aspect ratio)
- `viewing_distance_m`: Distance from the eye to the scene (meters)

### Processing

1. **Angular subtense**:
   ```
   angular_width_deg  = 2 × arctan(scene_width_m  / (2 × viewing_distance_m)) × (180/π)
   angular_height_deg = 2 × arctan(scene_height_m / (2 × viewing_distance_m)) × (180/π)
   ```

2. **Retinal image extent** (species-dependent, uses focal length from OpticalParams):
   ```
   retinal_width_mm  = 2 × focal_length_mm × tan(angular_width_deg / 2 × π/180)
   retinal_height_mm = 2 × focal_length_mm × tan(angular_height_deg / 2 × π/180)
   ```
   This means the same scene at the same distance produces a *different sized retinal image* per species because their focal lengths differ (human: 22.3mm, dog: 17.0mm, cat: 18.5mm).

3. **Pixels-to-retinal-mm mapping**:
   ```
   mm_per_pixel_x = retinal_width_mm  / image_width_px
   mm_per_pixel_y = retinal_height_mm / image_height_px
   ```
   This scaling factor is passed to the Optical Stage (for PSF kernel sizing) and the Retinal Stage (for mapping receptor positions to image coordinates).

4. **Patch clipping**: If angular subtense exceeds the simulation patch size (2° for PoC), the image is cropped to the central patch. If smaller, the image occupies a subregion of the mosaic and surrounding receptors receive no stimulation (background luminance only). Both cases are physically correct.

5. **Accommodation demand** (passed to Optical Stage):
   ```
   accommodation_diopters = 1.0 / viewing_distance_m
   ```
   At infinity: 0 D (collimated). At 1m: 1 D. At 0.25m (reading distance): 4 D.
   If demand exceeds the species' accommodation range, the residual is passed to the PSF as defocus:
   ```
   defocus_diopters = max(0, accommodation_diopters - species_max_accommodation)
   ```

   Species accommodation ranges:
   | Species | Max accommodation (D) | Near point (m)  |
   |---------|----------------------|------------------|
   | Human (young) | 10–12        | ~0.08–0.10       |
   | Human (40+)   | 4–6          | ~0.17–0.25       |
   | Dog           | 2–3          | ~0.33–0.50       |
   | Cat           | 2–4          | ~0.25–0.50       |

### Output
```
class SceneDescription:
    # Input geometry
    scene_width_m: float
    scene_height_m: float
    viewing_distance_m: float
    
    # Computed angular geometry
    angular_width_deg: float
    angular_height_deg: float
    
    # Per-species retinal mapping (computed when species is known)
    retinal_width_mm: float
    retinal_height_mm: float
    mm_per_pixel: Tuple[float, float]
    
    # Accommodation
    accommodation_demand_diopters: float
    defocus_residual_diopters: float  # 0 if within accommodation range
    
    # Patch interaction
    clipped: bool                     # True if scene exceeds patch size
    scene_covers_patch_fraction: float # What % of the mosaic is stimulated
```

### Key Class
```
class SceneGeometry:
    def __init__(self, scene_width_m: float, viewing_distance_m: float,
                 scene_height_m: Optional[float] = None)
    
    def compute(self, image_shape: Tuple[int, int], 
                optical_params: OpticalParams) -> SceneDescription
```

### Usage Examples

Demonstrates how identical images map to completely different retinal outcomes:

| Scenario                          | scene_width | distance | angular width | retinal width (human) | result                              |
|-----------------------------------|-------------|----------|---------------|-----------------------|-------------------------------------|
| Reading text (book at arm's length)| 0.15 m     | 0.40 m   | 21.2°         | 8.7 mm                | Within accommodation, high detail   |
| Face at conversation distance      | 0.20 m     | 1.5 m    | 7.6°          | 3.0 mm                | Well-resolved, patch clipping likely|
| Person across a field              | 0.50 m     | 50 m     | 0.57°         | 0.22 mm               | Tiny retinal image, few receptors   |
| Distant sign                       | 1.0 m      | 200 m    | 0.29°         | 0.11 mm               | Near resolution limit               |
| Dog seeing a squirrel at 30m       | 0.15 m     | 30 m     | 0.29°         | 0.09 mm (dog focal)   | ~30 cones sampling the squirrel     |

The last row is the kind of question this framework is designed to answer quantitatively.

### PoC Decision
- Default: `viewing_distance_m = inf` (collimated, no accommodation demand, no defocus)
- Finite distance support included from the start since it's cheap — just one extra Zernike term in the PSF when defocus > 0
- Accommodation model: hard cutoff (no partial accommodation lag). Upgrade path: add lead/lag curve from published accommodation response data

---

## 1. Spectral Stage

### Purpose
Real photoreceptor responses depend on spectral content, not RGB triplets. This stage converts standard image input into an estimated spectral radiance field so downstream processing uses physically meaningful units.

### Input
- `InputImage`: (H, W, 3) RGB array, uint8 or float32 [0,1]
- `illuminant`: SPD of assumed illumination (default: D65 daylight)

### Processing
1. **Spectral upsampling**: Convert RGB → approximate spectral power distribution per pixel. Method: Smits (1999) RGB-to-spectrum conversion, or the more accurate Mallett & Yuksel (2019) approach. Wavelength range: 380–720nm sampled at 5nm intervals (69 bands).
2. **Illuminant application**: If input is a reflectance image, multiply by illuminant SPD. If input is an emissive display image (typical case), treat RGB as display emission — apply display spectral primary decomposition (sRGB primaries have known emission spectra).
3. **Units normalization**: Output in relative spectral radiance (W·sr⁻¹·m⁻²·nm⁻¹). Absolute scaling not needed for comparative simulations, but the relative spectral shape must be correct.

### Output
- `SpectralImage`: (H, W, N_λ) float32 array
- `wavelengths`: (N_λ,) array of sampled wavelengths in nm

### Key Classes
```
class SpectralUpsampler:
    """Converts RGB images to spectral radiance estimates."""
    def __init__(self, method: str = "mallett_yuksel", 
                 wavelength_range: tuple = (380, 720), 
                 wavelength_step: int = 5)
    def upsample(self, rgb_image: np.ndarray) -> SpectralImage
```

### Notes
- This is the weakest link in physical accuracy — RGB-to-spectral is an underdetermined inverse problem. Acceptable for comparative simulations (dog vs human on same input). Not suitable for absolute colorimetric work.
- For users who have access to hyperspectral image datasets, this stage can be bypassed entirely.

---

## 2. Optical Stage

### Purpose
Models light transport from the scene through the anterior eye (cornea, aqueous humor, iris/pupil, crystalline lens, vitreous humor) to produce the retinal irradiance distribution. This is everything that happens before photons hit photoreceptors.

### Input
- `SpectralImage`: (H, W, N_λ) from Spectral Stage
- `OpticalParams`: species-specific optical parameters

### Sub-components

#### 2a. Pupil Model
Controls total light throughput and sets the aperture for diffraction.

**Parameters per species:**
| Parameter         | Human       | Dog          | Cat          |
|-------------------|-------------|--------------|--------------|
| Pupil shape       | Circular    | Circular     | Vertical slit|
| Min diameter (mm) | 2.0         | 3.0          | ~0.5 (slit)  |
| Max diameter (mm) | 8.0         | 9.0          | 14.0         |
| Default (photopic)| 3.0         | 4.0          | 2.0 (width)  |

**Processing:**
1. Generate 2D pupil aperture mask (circular or elliptical/slit).
2. Scale input irradiance by pupil area (light gathering).
3. Pass aperture geometry to PSF generator.

**Cat slit pupil note:** The slit pupil produces an anisotropic diffraction pattern — the PSF is wider horizontally than vertically. This must be modeled as a 2D non-separable function. The slit can be approximated as a rectangle with rounded ends.

#### 2b. Lens & Cornea PSF
Models the combined refractive optics as a point spread function.

**Parameters per species:**
| Parameter              | Human  | Dog    | Cat    |
|------------------------|--------|--------|--------|
| Axial length (mm)      | 24.0   | 21.0   | 22.5   |
| Focal length (mm)      | 22.3   | 17.0   | 18.5   |
| Corneal radius (mm)    | 7.8    | 8.5    | 8.5    |
| f-number (focal/pupil) | ~7.4   | ~4.3   | ~9.3*  |

*Cat f-number is deceptive due to slit geometry — effective light gathering is higher than f-number suggests.*

**Processing:**
1. Compute diffraction-limited PSF from pupil aperture: Fraunhofer diffraction → Fourier transform of pupil function, per wavelength (PSF scales with λ).
2. Convolve with aberration model: Use Zernike polynomial decomposition for typical aberrations (spherical, coma, astigmatism). Defocus can be parameterized for simulating different accommodation states.
3. Add longitudinal chromatic aberration (LCA): Different wavelengths focus at different axial distances. Human LCA is ~2 diopters across visible spectrum. Dog LCA is comparable. This means the PSF is wavelength-dependent — short wavelengths are more blurred than long wavelengths (or vice versa depending on accommodation).

**Output:** Wavelength-dependent PSF kernel: `psf_kernel(λ) → (K, K)` 2D array per wavelength band.

#### 2c. Vitreous & Media Transmission
Models absorption and scatter in the ocular media.

**Processing:**
1. Apply wavelength-dependent transmission function. The lens absorbs short-wavelength (UV/blue) light — this varies dramatically with age in humans, and between species. Dogs transmit more UV than adult humans. Cats also transmit more UV.
2. Apply forward scatter model (veiling glare): small-angle scatter from media imperfections. Simple implementation: add a fraction of mean luminance as a uniform offset. Physically: convolve with a wide low-amplitude Gaussian.

**Transmission function:** `T(λ)` is a scalar per wavelength band, applied as element-wise multiplication.

#### 2d. Retinal Image Formation (Optical Stage Integration)

Full optical pipeline:
```
for each wavelength λ:
    retinal_irradiance[:,:,λ] = convolve(
        spectral_image[:,:,λ] * pupil_throughput * media_transmission(λ),
        psf_kernel(λ)
    )
```

### Output
- `RetinalIrradiance`: (H, W, N_λ) float32 — spectral irradiance at the retinal surface
- `optical_metadata`: Dict containing effective PSF widths, pupil area, etc. for diagnostics

### Key Classes
```
class OpticalParams:
    """Species-specific optical parameters."""
    pupil_shape: str          # "circular" | "slit"
    pupil_diameter_mm: float  # or width for slit
    axial_length_mm: float
    focal_length_mm: float
    corneal_radius_mm: float
    lca_diopters: float       # longitudinal chromatic aberration range
    media_transmission: Callable[[np.ndarray], np.ndarray]  # T(λ)
    # Zernike coefficients for aberrations
    zernike_coeffs: Dict[str, float]

class OpticalStage:
    def __init__(self, params: OpticalParams)
    def compute_psf(self, wavelengths: np.ndarray) -> np.ndarray  # (N_λ, K, K)
    def apply(self, spectral_image: SpectralImage) -> RetinalIrradiance
```

### Notes
- Convolution per wavelength band is the computational bottleneck. FFT-based convolution is mandatory at full resolution.
- For a first pass, a single polychromatic PSF (weighted by illuminant × receptor sensitivity) per receptor type is a reasonable approximation that reduces computation by ~10-20x.

---

## 3. Retinal Stage

### Purpose
Generates the photoreceptor mosaic and computes the activation of each receptor based on the retinal irradiance. This is where the species-specific visual sampling actually happens.

### Input
- `RetinalIrradiance`: (H, W, N_λ) from Optical Stage
- `RetinalParams`: species-specific retinal parameters

### Sub-components

#### 3a. Photoreceptor Mosaic Generator

Generates a 2D irregular sampling array of photoreceptors with species-appropriate spatial density, type ratios, and arrangement.

**Parameters per species:**

| Parameter                 | Human            | Dog              | Cat              |
|---------------------------|------------------|------------------|------------------|
| Cone types                | S, M, L          | S, L (dichromat) | S, L (dichromat) |
| Peak cone density (mm⁻²) | ~200,000 (fovea) | ~12,000 (AC)     | ~10,000 (AC)     |
| Peak rod density (mm⁻²)  | ~176,000 (ring)  | ~400,000         | ~460,000         |
| S:M:L ratio               | ~1:16:32 (typ.)  | ~3:97 (S:L)      | ~5:95 (S:L)      |
| Area centralis diameter   | ~1.5mm (fovea)   | ~3mm             | ~0.5mm           |
| Visual streak             | No               | Weak             | Strong (horiz.)  |
| Rod-free zone             | ~0.35mm (foveal) | None             | None             |

*AC = area centralis. Data from Curcio 1990, Mowat 2008, Steinberg 1973.*

**Processing:**
1. Define the retinal patch to simulate (center, angular extent in degrees, mapped to mm via `mm = tan(deg) × focal_length_mm`). Note: focal length governs retinal image magnification (not axial length); the code and scene geometry stage use `focal_length_mm` consistently.
2. Compute local density for each receptor type at a grid of sample points using the density functions from literature (analytical fits or interpolated lookup tables).
3. Generate receptor positions via Poisson disk sampling with spatially varying radius derived from local density. Assign types based on local type ratios.
4. Store as a `PhotoreceptorMosaic` object.

**Data structure:**
```
class Photoreceptor:
    position: (float, float)  # (x_mm, y_mm) on retinal surface
    type: str                 # "S_cone" | "M_cone" | "L_cone" | "rod"
    aperture_um: float        # inner segment diameter
    sensitivity: np.ndarray   # spectral sensitivity curve S(λ), (N_λ,)

class PhotoreceptorMosaic:
    receptors: List[Photoreceptor]      # or structured numpy array for perf
    positions: np.ndarray               # (N, 2) float32
    types: np.ndarray                   # (N,) enum/int
    apertures: np.ndarray               # (N,) float32
    sensitivities: np.ndarray           # (N, N_λ) float32
    voronoi: scipy.spatial.Voronoi      # precomputed for visualization
```

#### 3b. Spectral Sensitivity Functions

Each receptor type has a characteristic absorption spectrum. Generate using the **Govardovskii nomogram** (Govardovskii et al., 2000) — a parametric template that produces the absorption spectrum from just the peak wavelength (λ_max).

**Peak wavelengths:**
| Receptor | Human λ_max | Dog λ_max | Cat λ_max |
|----------|-------------|-----------|-----------|
| S-cone   | 420 nm      | 429 nm    | 450 nm    |
| M-cone   | 530 nm      | —         | —         |
| L-cone   | 560 nm      | 555 nm    | 553 nm    |
| Rod      | 498 nm      | 506 nm    | 501 nm    |

Apply lens/media pre-filtering (from Optical Stage `media_transmission`) to convert from in-vitro absorption to in-vivo spectral sensitivity.

#### 3c. Photoreceptor Response Computation

For each receptor *i*:

1. **Spatial sampling**: Sample retinal irradiance at receptor position. Since the mosaic is irregular and the irradiance is on a regular grid, use bilinear interpolation. Weight by aperture function (Gaussian approximation to inner segment acceptance cone, σ ≈ aperture_diameter/2).

2. **Spectral integration**:
   ```
   excitation_i = ∫ S_i(λ) · E(x_i, y_i, λ) dλ
   ```
   Discrete: `excitation_i = np.dot(sensitivities[i], irradiance_at_position)`

3. **Transduction (Naka-Rushton)**:
   ```
   response_i = R_max * (excitation_i^n) / (excitation_i^n + sigma^n)
   ```
   Parameters:
   - R_max: maximum response (normalized to 1.0)
   - n: Hill coefficient (~0.7 for cones, ~1.0 for rods)
   - σ: half-saturation constant (different for rods vs cones — rods saturate at much lower light levels)

4. **Adaptation** (optional, adds complexity):
   - Weber adaptation: σ shifts proportional to local background luminance
   - This models light/dark adaptation and is important for HDR scenes
   - Can be deferred to v2

### Output
- `MosaicActivation`:
  ```
  class MosaicActivation:
      mosaic: PhotoreceptorMosaic  # geometry reference
      responses: np.ndarray        # (N_receptors,) float32, [0, 1]
      metadata: dict               # integration diagnostics
  ```

### Key Classes
```
class RetinalParams:
    cone_types: List[str]                    # e.g. ["S", "L"] for dog
    cone_peak_wavelengths: Dict[str, float]  # nm
    rod_peak_wavelength: float               # nm
    cone_density_fn: Callable[[float, float], Dict[str, float]]  # (ecc_mm, angle) → densities
    rod_density_fn: Callable[[float, float], float]
    cone_ratio_fn: Callable[[float], Dict[str, float]]  # eccentricity → type ratios
    naka_rushton_params: Dict[str, Dict[str, float]]     # per type: {n, sigma, R_max}
    patch_center_mm: Tuple[float, float]
    patch_extent_deg: float

class RetinalStage:
    def __init__(self, params: RetinalParams, optical_params: OpticalParams)
    def generate_mosaic(self, seed: int = 0) -> PhotoreceptorMosaic
    def compute_response(self, mosaic: PhotoreceptorMosaic, 
                         retinal_irradiance: RetinalIrradiance) -> MosaicActivation
```

---

## 4. Output Stage

### Purpose
Visualization and comparison of mosaic activations.

### Visualization Modes

1. **Voronoi activation map**: Color each Voronoi cell by receptor response intensity. Receptor type encoded by base hue (rods: gray, S-cones: blue, L-cones: green/red). Intensity mapped to brightness.

2. **Reconstructed image**: Inverse-map from mosaic activations back to a regular grid for side-by-side comparison with input. This is explicitly *not* what the animal sees (that requires neural processing), but it shows what information is preserved/lost by the retinal sampling.

3. **Comparative panel**: Input image | Human retinal activation | Dog retinal activation | Cat retinal activation — same patch, same scale.

4. **Density visualization**: Plot the mosaic itself without activation — show receptor positions, types, and density gradients.

5. **MTF (Modulation Transfer Function)**: Compute spatial frequency response of the combined optical + mosaic system. Shows the effective resolution limit as a function of eccentricity.

### Key Classes
```
class OutputRenderer:
    def render_voronoi(self, activation: MosaicActivation, ...) -> np.ndarray
    def render_reconstructed(self, activation: MosaicActivation, ...) -> np.ndarray
    def render_comparison(self, activations: Dict[str, MosaicActivation], ...) -> Figure
    def render_mosaic_map(self, mosaic: PhotoreceptorMosaic, ...) -> Figure
    def compute_mtf(self, optical_stage: OpticalStage, 
                    retinal_stage: RetinalStage, ...) -> np.ndarray
```

---

## 5. Species Configuration

All species parameters consolidated into a single config object per species, loaded from YAML/JSON files.

```
class SpeciesConfig:
    name: str
    optical: OpticalParams
    retinal: RetinalParams
    
    @classmethod
    def load(cls, species_name: str) -> "SpeciesConfig"
    # Loads from data/species/{species_name}.yaml
```

Directory structure for data:
```
data/
  species/
    human.yaml
    dog.yaml
    cat.yaml
  opsin_templates/
    govardovskii.py         # nomogram generator
  density_maps/
    curcio_1990_human.csv   # digitized density data
    mowat_2008_dog.csv
    steinberg_1973_cat.csv
  illuminants/
    D65.csv
    A.csv                   # tungsten
```

---

## 6. Pipeline Orchestrator

```
class RetinalSimulator:
    def __init__(self, species: str | SpeciesConfig, 
                 patch_extent_deg: float = 2.0,
                 light_level: str = "photopic")  # "scotopic" | "mesopic" | "photopic"
    
    def simulate(self, input_image: np.ndarray,
                 scene_width_m: float = None,
                 viewing_distance_m: float = float('inf')) -> SimulationResult:
        # Scene geometry — compute angular/retinal mapping
        scene = self.scene_geometry.compute(
            image_shape=input_image.shape[:2],
            scene_width_m=scene_width_m,        # None → assume image fills patch
            viewing_distance_m=viewing_distance_m,
            optical_params=self.species_config.optical
        )
        
        spectral = self.spectral_stage.upsample(input_image)
        irradiance = self.optical_stage.apply(spectral, scene)  # scene informs PSF scale + defocus
        mosaic = self.retinal_stage.generate_mosaic()
        activation = self.retinal_stage.compute_response(mosaic, irradiance, scene)  # scene informs coordinate mapping
        return SimulationResult(scene, spectral, irradiance, mosaic, activation)
    
    def compare_species(self, input_image: np.ndarray, 
                        species_list: List[str],
                        scene_width_m: float = None,
                        viewing_distance_m: float = float('inf')) -> Dict[str, SimulationResult]:
        """Run same image + scene geometry through multiple species pipelines.
        Note: retinal image size differs per species (different focal lengths)."""
        ...

class SimulationResult:
    scene: SceneDescription
    spectral_image: SpectralImage
    retinal_irradiance: RetinalIrradiance
    mosaic: PhotoreceptorMosaic
    activation: MosaicActivation
```

---

## 7. Project Structure

```
retinal_sim/
├── retinal_sim/
│   ├── __init__.py
│   ├── pipeline.py              # RetinalSimulator orchestrator
│   ├── scene/
│   │   ├── __init__.py
│   │   └── geometry.py          # SceneGeometry, SceneDescription
│   ├── spectral/
│   │   ├── __init__.py
│   │   └── upsampler.py         # SpectralUpsampler
│   ├── optical/
│   │   ├── __init__.py
│   │   ├── pupil.py             # Pupil aperture models
│   │   ├── psf.py               # PSF computation (diffraction + aberration)
│   │   ├── media.py             # Vitreous/lens transmission
│   │   └── stage.py             # OpticalStage integration
│   ├── retina/
│   │   ├── __init__.py
│   │   ├── mosaic.py            # PhotoreceptorMosaic generator
│   │   ├── opsin.py             # Govardovskii nomogram, sensitivity curves
│   │   ├── transduction.py      # Naka-Rushton, adaptation models
│   │   └── stage.py             # RetinalStage integration
│   ├── output/
│   │   ├── __init__.py
│   │   ├── voronoi.py           # Voronoi activation renderer
│   │   ├── reconstruction.py    # Grid reconstruction from mosaic
│   │   └── comparison.py        # Multi-species comparison panels
│   ├── species/
│   │   ├── __init__.py
│   │   └── config.py            # SpeciesConfig loader
│   └── data/
│       ├── species/             # YAML configs per species
│       ├── density_maps/        # Digitized literature data
│       ├── opsin_templates/     # Nomogram implementation
│       └── illuminants/         # Standard illuminant SPDs
├── tests/
│   ├── test_spectral.py
│   ├── test_optical.py
│   ├── test_retina.py
│   └── test_pipeline.py
├── examples/
│   ├── single_species.py
│   └── compare_three.py
└── pyproject.toml
```

---

## 8. Dependencies

- **numpy** — core array operations
- **scipy** — FFT convolution, spatial Voronoi, interpolation, special functions
- **matplotlib** — visualization
- **opencv-python** — image I/O, optional preprocessing
- **pyyaml** — species config loading
- **numba** (optional) — JIT compilation for mosaic generation and per-receptor integration if performance is insufficient with pure numpy

---

## 9. Implementation Priority

| Phase | Component                          | Complexity | Notes                                        |
|-------|------------------------------------|------------|----------------------------------------------|
| 1     | Govardovskii nomogram              | Low        | Pure math, well-documented formula           |
| 2     | Mosaic generator (single species)  | Medium     | Poisson disk sampling with variable density   |
| 3     | Simplified optical PSF (Gaussian)  | Low        | Placeholder before full diffraction model     |
| 4     | Spectral integration + Naka-Rushton| Low        | Matrix multiply + element-wise nonlinearity   |
| 5     | Voronoi visualization              | Low        | scipy.spatial.Voronoi + matplotlib            |
| 6     | Full diffraction PSF               | Medium     | FFT of pupil function, per-wavelength         |
| 7     | Spectral upsampler                 | Medium     | Implement Mallett-Yuksel or Smits             |
| 8     | Cat slit pupil model               | Medium     | Anisotropic aperture → anisotropic PSF        |
| 9     | Species comparison pipeline        | Low        | Orchestration of existing components          |
| 10    | Chromatic aberration model         | High       | Wavelength-dependent defocus in PSF           |
| 11    | Adaptation model                   | High       | Spatially varying gain control                |

---

## 10. Proof-of-Concept Decisions

| Question                  | PoC Decision                              | Rationale                                                    | Full-scale upgrade path                        |
|---------------------------|-------------------------------------------|--------------------------------------------------------------|------------------------------------------------|
| Simulation patch size     | **2° centered on area centralis**         | ~10k-50k receptors depending on species. Fast iteration.     | Scale to 5-10° or full hemiretina              |
| Light level               | **Photopic only**                         | Cone-dominated simplifies transduction — no rod saturation logic needed. | Add scotopic/mesopic with rod saturation gates |
| Spectral upsampling       | **Smits (1999)**                          | Simple, fast, good enough for comparative work.              | Mallett-Yuksel for absolute spectral accuracy  |
| Performance target        | **< 30s per species on a single core**    | Acceptable for PoC. No GPU/numba needed yet.                 | Numba JIT or GPU for interactive rates         |
| Mosaic generation         | **Jittered grid (not full Poisson disk)** | 10x faster generation, visually similar at PoC scale.        | Proper Poisson disk sampling for publication   |
| Chromatic aberration      | **Deferred**                              | Single polychromatic PSF per receptor class is sufficient.   | Per-wavelength PSF with LCA model              |
| Adaptation                | **Deferred**                              | Fixed σ in Naka-Rushton. No spatial gain control.            | Weber adaptation, spatial surround             |

---

## 11. Validation Framework

Validation is not optional — it's how we know the simulation is producing physically meaningful results rather than pretty garbage. Each stage gets independent validation against known ground truth before integration.

### 11a. Spectral Stage Validation

**Test: Metamer preservation**
- Input: Known metameric pairs (stimuli that are different spectrally but identical under human RGB). After upsampling both to spectra, they should produce identical cone excitations for human S/M/L cones (within tolerance).
- Pass criterion: Cone excitation difference < 1% for known metamers.
- Data source: CIE standard metameric pairs, or generate from known reflectance spectra + D65.

**Test: Roundtrip consistency**
- Upsample an RGB image to spectra, then reproject back to RGB via CIE color matching functions.
- Pass criterion: RMSE < 0.02 (on [0,1] scale) vs original RGB.

### 11b. Optical Stage Validation

**Test: Diffraction-limited resolution**
- Input: A Siemens star or sinusoidal grating pattern at known spatial frequencies.
- Compute MTF from optical stage output.
- Compare against theoretical diffraction-limited MTF: `MTF(f) = (2/π)(arccos(f/f_c) - (f/f_c)√(1-(f/f_c)²))` where `f_c = D/(λF)` (cutoff frequency from pupil diameter D, wavelength λ, f-number F).
- Pass criterion: Simulated MTF within 5% of theoretical at frequencies below 0.8 × f_c.

**Test: PSF energy conservation**
- Sum of PSF kernel must equal 1.0 (no light created or destroyed).
- Pass criterion: |sum(PSF) - 1.0| < 1e-6 per wavelength band.

**Test: Cat vs circular pupil anisotropy** (post-PoC, when slit pupil is implemented)
- Cat slit PSF must show measurably wider spread along the horizontal axis than vertical.
- Compare PSF FWHM in x vs y: ratio should be > 2:1 for a fully constricted slit.

### 11c. Retinal Stage Validation

**Test: Snellen acuity prediction (KEY VALIDATION)**
- Generate a synthetic Snellen chart (letters at calibrated angular sizes).
- Run through full pipeline for human with 3mm pupil.
- Measure: At what angular letter size do individual mosaic activations become indistinguishable between different letters?
- Expected: Human model should resolve ~1 arcminute detail (20/20) at the foveal center, degrading peripherally.
- Dog model should resolve ~4-8 arcminute detail (approx 20/75 equivalent). Published behavioral acuity for dogs is ~20/75 (Miller & Murphy, 1995).
- Cat model should resolve ~6-10 arcminute (approx 20/100-20/200). Published: ~20/100 to 20/200 (Blake et al., 1974).
- Pass criterion: Predicted acuity limit within 30% of published behavioral data.

**Test: Dichromat confusion axis**
- Generate Ishihara-style test patterns (pseudoisochromatic plates).
- Run through human (trichromat) and dog (dichromat) pipelines.
- Human model should show clear activation difference between figure and ground.
- Dog model should show figure/ground activations converging (below discrimination threshold) along the known confusion axis (red-green).
- This is the single most visually dramatic validation — it directly demonstrates the functional consequence of missing M-cones.

**Test: Cone density → sampling limit (Nyquist)**
- For each species, compute the Nyquist sampling frequency from the local cone spacing at the area centralis.
- Nyquist frequency = 1 / (2 × mean_cone_spacing_mm) in cycles/mm, convert to cycles/degree via: `cpd = (cycles/mm) × (mm_per_degree)` where `mm_per_degree = focal_length_mm × tan(1°)`. (Focal length, not axial length, determines retinal image magnification; mosaic.py and geometry.py both use `focal_length_mm`.)
- Compare against known values:
  - Human fovea: ~60 cpd
  - Dog area centralis: ~12 cpd
  - Cat area centralis: ~8-10 cpd
- Pass criterion: Within 20% of published values.

**Test: Receptor count validation**
- Generate mosaic for a known retinal area (e.g., 1mm² centered on area centralis).
- Total receptor count should match published histological counts within 25%.
- Cone/rod ratio should match within 15%.

### 11d. End-to-End Validation

**Test: Known visual deficit reproduction**
- Input: A natural scene with prominent red and green objects (e.g., red apple on green leaves).
- Human reconstruction should preserve red/green distinction.
- Dog reconstruction should show red and green objects collapsing toward similar yellow-brown activations.
- Cat reconstruction should show similar color collapse but with better low-frequency contrast preservation (larger receptive fields).

**Test: Resolution gradient**
- Input: A uniform grid of fine dots across the full patch.
- Output should show resolvable dots at center, merging toward periphery, with the rate of merging matching the density falloff curve for each species.

### 11e. Scene Geometry Validation

**Test: Angular subtense correctness**
- Input: Known physical dimensions and distances with analytically computable angular sizes.
- Case 1: 1m object at 57.3m → should subtend exactly 1° (since tan(1°) ≈ 1/57.3).
- Case 2: 0.00873m (8.73mm, roughly a Snellen 20/20 letter) at 6m → should subtend 5 arcminutes.
- Pass criterion: Computed angular subtense within 0.01% of analytical value.

**Test: Retinal image scaling across species**
- Same scene (1m wide, 10m away) processed for human, dog, cat.
- Retinal image widths should scale proportionally to focal lengths: human (22.3mm) > cat (18.5mm) > dog (17.0mm).
- Pass criterion: Ratios match focal length ratios within 1%.

**Test: Accommodation/defocus injection**
- Scene at 0.25m for dog (max accommodation ~2-3D, demand = 4D).
- Defocus residual should be ~1-2D, injected as PSF broadening.
- Scene at 0.25m for young human (max accommodation ~10-12D, demand = 4D).
- Defocus residual should be 0D (within range).
- Pass criterion: Correct defocus residual computed per species.

**Test: Distance-dependent receptor sampling**
- Same physical object (e.g., 20cm × 20cm) simulated at 1m, 10m, 100m.
- Count how many receptors in the mosaic fall within the retinal image of the object at each distance.
- Receptor count should scale as ~1/d² (since retinal image area scales as 1/d²).
- Pass criterion: Receptor count ratios within 15% of theoretical 1/d² scaling (tolerance accounts for mosaic irregularity at small receptor counts).

### 11f. Validation Infrastructure

```
class ValidationSuite:
    """Runs all validation tests and produces a report."""
    
    def __init__(self, simulator: RetinalSimulator)
    
    # Scene geometry
    def test_angular_subtense(self) -> ValidationResult
    def test_retinal_scaling_across_species(self) -> ValidationResult
    def test_accommodation_defocus(self) -> ValidationResult
    def test_distance_receptor_sampling(self) -> ValidationResult
    
    # Spectral
    def test_metamer_preservation(self) -> ValidationResult
    def test_rgb_roundtrip(self) -> ValidationResult
    
    # Optical
    def test_mtf_vs_diffraction_limit(self) -> ValidationResult
    def test_psf_energy_conservation(self) -> ValidationResult
    
    # Retinal
    def test_snellen_acuity(self) -> ValidationResult
    def test_dichromat_confusion(self) -> ValidationResult
    def test_nyquist_sampling(self) -> ValidationResult
    def test_receptor_count(self) -> ValidationResult
    
    # End-to-end
    def test_color_deficit_reproduction(self) -> ValidationResult
    def test_resolution_gradient(self) -> ValidationResult
    
    def run_all(self) -> ValidationReport
    def run_stage(self, stage: str) -> ValidationReport  # "scene"|"spectral"|"optical"|"retinal"|"e2e"

class ValidationResult:
    test_name: str
    passed: bool
    expected: Any
    actual: Any
    tolerance: float
    details: str
    figure: Optional[matplotlib.Figure]  # visual evidence

class ValidationReport:
    results: List[ValidationResult]
    def summary(self) -> str
    def save_html(self, path: str) -> None  # visual report with embedded figures
```

### Validation data files
```
data/
  validation/
    metameric_pairs.csv          # Known RGB metamers
    snellen_chart_angles.json    # Letter sizes in arcminutes
    ishihara_patterns/           # Pseudoisochromatic test images
    published_acuity.yaml        # Literature values per species
    published_density.yaml       # Histological counts per species
```

---

## 12. Implementation Order (Updated for PoC + Validation)

| Phase | Component                          | Validation integrated                          |
|-------|------------------------------------|-------------------------------------------------|
| 1     | Govardovskii nomogram              | Visual check: plot against published curves     |
| 2     | Species config loader (YAML)       | —                                               |
| 3     | Scene geometry module              | Angular subtense + retinal scaling tests        |
| 4     | Mosaic generator (jittered grid)   | Receptor count validation                       |
| 5     | Simplified optical PSF (Gaussian)  | PSF energy conservation                         |
| 6     | Smits spectral upsampler           | RGB roundtrip test                              |
| 7     | Spectral integration + Naka-Rushton| Nyquist sampling test                           |
| 8     | Voronoi visualization              | Resolution gradient test                        |
| 9     | Snellen acuity validation          | KEY: acuity prediction vs published             |
| 10    | Dichromat confusion validation     | Ishihara reproduction test                      |
| 11    | Distance-dependent resolution test | Receptor count vs 1/d² scaling                  |
| 12    | Species comparison pipeline        | End-to-end color deficit test                   |
| 13    | Full validation report generator   | HTML report with all figures                    |

### Reporting transparency requirement

Generated validation and user-facing audit reports must expose:
- the exact validation criterion or threshold used,
- the measured result or observed output,
- the assumptions and known simplifications behind the check,
- whether the conclusion is measured, inferred, or visual/manual,
- and the implementation location(s) that produced the result.

Summary-only output is not sufficient for audit artifacts. The report should make it possible to trace each conclusion back to architecture intent, test logic, and code.
