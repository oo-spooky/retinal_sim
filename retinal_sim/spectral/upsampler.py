"""Scene-spectrum handling for RGB-inferred and measured spectral inputs."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import lsq_linear

from retinal_sim.constants import WAVELENGTHS as _CANONICAL_WAVELENGTHS


@dataclass
class SpectralImage:
    """(H, W, N_λ) spectral radiance array with wavelength axis."""
    data: np.ndarray       # float32, shape (H, W, N_λ)
    wavelengths: np.ndarray  # nm, shape (N_λ,)
    metadata: dict = field(default_factory=dict)


DEFAULT_SCENE_INPUT_MODE = "reflectance_under_d65"
RGB_SCENE_INPUT_MODES = frozenset({"reflectance_under_d65", "display_rgb"})
SCENE_INPUT_MODES = frozenset(set(RGB_SCENE_INPUT_MODES) | {"measured_spectrum"})


# ---------------------------------------------------------------------------
# CIE 1931 2° standard observer CMFs at 5 nm, 380–720 nm (69 bands).
# Source: CIE 015:2018.
# ---------------------------------------------------------------------------
# fmt: off
_CIE_X = np.array([
    0.001368, 0.002236, 0.004243, 0.007650, 0.014310, 0.023190, 0.043510,
    0.077630, 0.134380, 0.214770, 0.283900, 0.328500, 0.348280, 0.348060,
    0.336200, 0.318700, 0.290800, 0.251100, 0.195360, 0.142100, 0.095640,
    0.057950, 0.031960, 0.014700, 0.004900, 0.002400, 0.009300, 0.029100,
    0.063270, 0.109600, 0.165500, 0.225750, 0.290400, 0.359700, 0.433450,
    0.512050, 0.594500, 0.678400, 0.762100, 0.842500, 0.916300, 0.978600,
    1.026300, 1.056700, 1.062200, 1.045600, 1.002600, 0.938400, 0.854450,
    0.751400, 0.642400, 0.541900, 0.447900, 0.360800, 0.283500, 0.218700,
    0.164900, 0.121200, 0.087400, 0.063600, 0.046770, 0.032900, 0.022700,
    0.015840, 0.011359, 0.008111, 0.005790, 0.004109, 0.002899,
])
_CIE_Y = np.array([
    0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640, 0.001210,
    0.002180, 0.004000, 0.007300, 0.011600, 0.016840, 0.023000, 0.029800,
    0.038000, 0.048000, 0.060000, 0.073900, 0.090980, 0.112600, 0.139020,
    0.169300, 0.208020, 0.258600, 0.323000, 0.407300, 0.503000, 0.608200,
    0.710000, 0.793200, 0.862000, 0.914850, 0.954000, 0.980300, 0.994950,
    1.000000, 0.995000, 0.978600, 0.952000, 0.915400, 0.870000, 0.816300,
    0.757000, 0.694900, 0.631000, 0.566800, 0.503000, 0.441200, 0.381000,
    0.321000, 0.265000, 0.217000, 0.175000, 0.138200, 0.107000, 0.081600,
    0.061000, 0.044580, 0.032000, 0.023200, 0.017000, 0.011920, 0.008210,
    0.005723, 0.004102, 0.002929, 0.002091, 0.001484, 0.001047,
])
_CIE_Z = np.array([
    0.006450, 0.010550, 0.020050, 0.036210, 0.067850, 0.110200, 0.207400,
    0.371300, 0.645600, 1.039050, 1.385600, 1.622960, 1.747060, 1.782600,
    1.772110, 1.744100, 1.669200, 1.528100, 1.287640, 1.041900, 0.812950,
    0.616200, 0.465180, 0.353300, 0.272000, 0.212300, 0.158200, 0.111700,
    0.078250, 0.057250, 0.042160, 0.029840, 0.020300, 0.013400, 0.008750,
    0.005750, 0.003900, 0.002750, 0.002100, 0.001800, 0.001650, 0.001400,
    0.001100, 0.001000, 0.000800, 0.000600, 0.000340, 0.000240, 0.000190,
    0.000100, 0.000050, 0.000030, 0.000020, 0.000010, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
])

# CIE standard illuminant D65 relative SPD at 5 nm, 380–720 nm.
# Normalised to S(560 nm) = 100.  Source: CIE 015:2018.
_D65 = np.array([
     50.000,  52.312,  54.622,  68.702,  82.754,  87.120,  91.486,  92.459,
     93.432,  90.057,  86.682,  95.774, 104.865, 110.936, 117.008, 117.410,
    117.812, 116.336, 114.861, 115.392, 115.923, 112.367, 108.811, 109.082,
    109.354, 108.578, 107.802, 106.296, 104.790, 106.239, 107.689, 106.047,
    104.405, 104.225, 104.046, 102.023, 100.000,  98.167,  96.334,  96.061,
     95.788,  92.237,  88.685,  89.346,  90.006,  89.802,  89.599,  88.649,
     87.699,  85.494,  83.289,  83.493,  83.697,  81.863,  80.026,  80.121,
     80.216,  81.246,  82.277,  80.281,  78.284,  74.003,  69.721,  70.665,
     71.609,  72.979,  74.349,  61.604,  48.860,
])
# fmt: on

# sRGB ↔ XYZ matrices (D65, IEC 61966-2-1).
_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


def _gaussian_primary(center_nm: float, sigma_nm: float) -> np.ndarray:
    wl = _CANONICAL_WAVELENGTHS.astype(np.float64)
    primary = np.exp(-0.5 * ((wl - center_nm) / sigma_nm) ** 2)
    peak = float(np.max(primary))
    return primary / peak if peak > 0.0 else primary


_REFERENCE_DISPLAY_PRIMARIES = {
    "red": _gaussian_primary(610.0, 18.0),
    "green": _gaussian_primary(545.0, 18.0),
    "blue": _gaussian_primary(460.0, 16.0),
}


def normalize_scene_input_mode(input_mode: str) -> str:
    """Validate and normalize a scene input mode string."""
    if input_mode not in SCENE_INPUT_MODES:
        valid = ", ".join(sorted(SCENE_INPUT_MODES))
        raise ValueError(f"Unknown input_mode={input_mode!r}; expected one of {valid}")
    return input_mode


def scene_input_metadata(input_mode: str) -> dict[str, object]:
    """Return declared semantics/provenance for a scene input mode."""
    mode = normalize_scene_input_mode(input_mode)
    if mode == "reflectance_under_d65":
        assumptions = [
            "RGB is interpreted as a display-referred color that is reconstructed into a "
            "Smits-inspired D65-optimized reflectance spectrum.",
            "The recovered spectrum is inferred from RGB and is not a unique physical scene spectrum.",
            "This mode models diffuse reflectance under D65 for comparative PoC work.",
        ]
    elif mode == "display_rgb":
        assumptions = [
            "RGB is interpreted as drive values for a fixed built-in reference display model.",
            "Spectra are emitted by repo-local reference RGB primary SPDs rather than by a calibrated device.",
            "This mode is inferred from RGB and does not represent a measured display spectrum.",
        ]
    else:
        assumptions = [
            "Caller provided a measured spectral image directly.",
            "No RGB-to-spectrum inference is applied in this mode.",
        ]
    return {
        "scene_input_mode": mode,
        "scene_input_is_inferred": mode != "measured_spectrum",
        "scene_input_assumptions": assumptions,
    }


class SpectralUpsampler:
    """Converts RGB imagery into explicitly-declared spectral scene models.

    ``reflectance_under_d65`` uses a Smits-inspired greedy-peel decomposition
    with seven basis spectra (white, cyan, magenta, yellow, red, green, blue).
    Rather than using the original paper's hand-tuned 11-knot tables, the
    basis spectra are computed at initialisation by solving a constrained
    least-squares problem: each basis spectrum is the minimum-norm reflectance
    in [0, 1] that, when integrated against the CIE 1931 2° observer under D65
    illumination, reproduces the target sRGB colour.

    ``display_rgb`` uses a fixed built-in reference display model made from
    repo-local RGB primary emission spectra. It is a declared PoC assumption,
    not a calibrated device model.

    Args:
        method: ``'smits'`` (default).  ``'mallett_yuksel'`` raises
            ``NotImplementedError``.
        wavelength_range: ``(min_nm, max_nm)`` inclusive.
        wavelength_step: Sampling interval in nm.
    """

    def __init__(
        self,
        method: str = "smits",
        wavelength_range: tuple = (380, 720),
        wavelength_step: int = 5,
    ) -> None:
        if method != "smits":
            raise NotImplementedError(
                f"method={method!r} not implemented; only 'smits' is available for the PoC"
            )
        self.method = method
        # Use the canonical grid from constants.py when default parameters are
        # used; otherwise build from the supplied range to support custom grids.
        if wavelength_range == (380, 720) and wavelength_step == 5:
            self.wavelengths: np.ndarray = _CANONICAL_WAVELENGTHS.copy()
        else:
            self.wavelengths = np.arange(
                wavelength_range[0],
                wavelength_range[1] + wavelength_step,
                wavelength_step,
                dtype=np.float64,
            )
        self._basis: dict[str, np.ndarray] = self._compute_basis()
        self._display_primaries = self._build_reference_display_primaries()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def upsample(
        self,
        rgb_image: np.ndarray,
        input_mode: str = DEFAULT_SCENE_INPUT_MODE,
    ) -> SpectralImage:
        """Convert ``(H, W, 3)`` RGB into a declared spectral scene model.

        Args:
            rgb_image: ``(H, W, 3)`` array.  uint8 values are divided by 255.
            input_mode: One of ``reflectance_under_d65`` or ``display_rgb``.

        Returns:
            :class:`SpectralImage` with ``float32`` data of shape
            ``(H, W, N_λ)``.
        """
        mode = normalize_scene_input_mode(input_mode)
        if mode not in RGB_SCENE_INPUT_MODES:
            raise ValueError(
                f"SpectralUpsampler.upsample() only accepts RGB input modes {sorted(RGB_SCENE_INPUT_MODES)}, "
                f"got {mode!r}"
            )
        rgb = self._coerce_rgb(rgb_image)
        H, W, _ = rgb.shape
        flat = rgb.reshape(-1, 3)
        if mode == "reflectance_under_d65":
            spec_flat = self._smits_decompose(flat)
        else:
            spec_flat = self._display_decompose(flat)
        data = spec_flat.reshape(H, W, len(self.wavelengths)).astype(np.float32)
        return SpectralImage(
            data=data,
            wavelengths=self.wavelengths.copy(),
            metadata=dict(scene_input_metadata(mode)),
        )

    def spectral_to_xyz(self, spectra: np.ndarray) -> np.ndarray:
        """Project spectra on this instance's wavelength grid to CIE XYZ.

        The returned XYZ values assume the same D65-referenced reflectance
        convention used by the upsampling basis construction.

        Args:
            spectra: ``(N_lambda,)`` or ``(..., N_lambda)`` array on
                ``self.wavelengths``.

        Returns:
            ``(..., 3)`` XYZ array in relative units with Y normalised so that
            a unit-reflectance diffuser under D65 yields Y=1.
        """
        spectra = np.asarray(spectra, dtype=np.float64)
        if spectra.shape[-1] != len(self.wavelengths):
            raise ValueError(
                f"Expected trailing wavelength axis of length {len(self.wavelengths)}, "
                f"got {spectra.shape[-1]}"
            )
        cie_x, cie_y, cie_z, d65 = self._observer_and_illuminant()
        dlam = float(self.wavelengths[1] - self.wavelengths[0])
        k = 1.0 / float(np.sum(d65 * cie_y) * dlam)
        weights = k * dlam * np.stack([d65 * cie_x, d65 * cie_y, d65 * cie_z], axis=-1)
        return np.tensordot(spectra, weights, axes=([-1], [0]))

    def spectral_to_srgb(self, spectra: np.ndarray) -> np.ndarray:
        """Project spectra to display-referred sRGB values in [0, 255]."""
        xyz = self.spectral_to_xyz(spectra)
        rgb_linear = np.tensordot(xyz, np.linalg.inv(_SRGB_TO_XYZ).T, axes=([-1], [0]))
        rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
        rgb = np.where(
            rgb_linear <= 0.0031308,
            12.92 * rgb_linear,
            1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055,
        )
        return np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_basis(self) -> dict[str, np.ndarray]:
        """Compute basis spectra for the D65 reflectance roundtrip.

        For each of the seven Smits-inspired basis colours, finds the
        reflectance spectrum S ∈ [0, 1]^{N_λ} with minimum L² norm such that

            k · (D65 · S · CMF) · dλ  =  XYZ_target

        where k = 1 / (D65 · ȳ · dλ) is the D65 white normalisation and
        XYZ_target = sRGB_to_XYZ @ target_sRGB.

        Solved via ``scipy.optimize.lsq_linear`` with bounds [0, 1].
        """
        wl = self.wavelengths
        # Interpolate module-level 380–720 nm / 5 nm data to the actual grid.
        from scipy.interpolate import interp1d

        ref_wl = np.arange(380, 725, 5, dtype=np.float64)

        def _interp(arr: np.ndarray) -> np.ndarray:
            return interp1d(ref_wl, arr, kind="linear",
                            bounds_error=False, fill_value=(arr[0], arr[-1]))(wl)

        cie_x, cie_y, cie_z, d65 = self._observer_and_illuminant()

        dlam = float(wl[1] - wl[0])
        k = 1.0 / (d65 @ cie_y * dlam)

        # A[i, j] = k * D65[j] * CMF_i[j] * dλ  →  (3, N_λ)
        A = k * dlam * np.stack([d65 * cie_x, d65 * cie_y, d65 * cie_z])

        basis: dict[str, np.ndarray] = {}
        target_srgb = {
            "white":   np.array([1.0, 1.0, 1.0]),
            "cyan":    np.array([0.0, 1.0, 1.0]),
            "magenta": np.array([1.0, 0.0, 1.0]),
            "yellow":  np.array([1.0, 1.0, 0.0]),
            "red":     np.array([1.0, 0.0, 0.0]),
            "green":   np.array([0.0, 1.0, 0.0]),
            "blue":    np.array([0.0, 0.0, 1.0]),
        }
        for name, srgb in target_srgb.items():
            xyz_target = _SRGB_TO_XYZ @ srgb
            result = lsq_linear(A, xyz_target, bounds=(0.0, 1.0))
            basis[name] = result.x
        return basis

    def _build_reference_display_primaries(self) -> np.ndarray:
        """Return repo-local reference display primary SPDs on this grid."""
        if np.array_equal(self.wavelengths, _CANONICAL_WAVELENGTHS):
            return np.stack(
                [
                    _REFERENCE_DISPLAY_PRIMARIES["red"],
                    _REFERENCE_DISPLAY_PRIMARIES["green"],
                    _REFERENCE_DISPLAY_PRIMARIES["blue"],
                ],
                axis=0,
            )

        from scipy.interpolate import interp1d

        ref_wl = _CANONICAL_WAVELENGTHS.astype(np.float64)

        def _interp(arr: np.ndarray) -> np.ndarray:
            return interp1d(
                ref_wl,
                arr,
                kind="linear",
                bounds_error=False,
                fill_value=(arr[0], arr[-1]),
            )(self.wavelengths)

        return np.stack(
            [
                _interp(_REFERENCE_DISPLAY_PRIMARIES["red"]),
                _interp(_REFERENCE_DISPLAY_PRIMARIES["green"]),
                _interp(_REFERENCE_DISPLAY_PRIMARIES["blue"]),
            ],
            axis=0,
        )

    def _observer_and_illuminant(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return CIE 1931 CMFs and D65 SPD on ``self.wavelengths``."""
        wl = self.wavelengths
        ref_wl = np.arange(380, 725, 5, dtype=np.float64)

        if np.array_equal(wl, ref_wl):
            return _CIE_X.copy(), _CIE_Y.copy(), _CIE_Z.copy(), _D65.copy()

        from scipy.interpolate import interp1d

        def _interp(arr: np.ndarray) -> np.ndarray:
            return interp1d(
                ref_wl,
                arr,
                kind="linear",
                bounds_error=False,
                fill_value=(arr[0], arr[-1]),
            )(wl)

        return _interp(_CIE_X), _interp(_CIE_Y), _interp(_CIE_Z), _interp(_D65)

    def _smits_decompose(self, rgb: np.ndarray) -> np.ndarray:
        """Vectorised Smits-inspired RGB-to-reflectance decomposition.

        Subtracts the minimum channel as the achromatic (white) component,
        then routes each pixel to one of three chromatic paths based on
        which channel hit zero.

        Args:
            rgb: ``(N, 3)`` float64 array in [0, 1].

        Returns:
            ``(N, N_λ)`` float64 spectral array.
        """
        b = self._basis
        r = rgb[:, 0].copy()
        g = rgb[:, 1].copy()
        bv = rgb[:, 2].copy()
        N_lam = len(self.wavelengths)
        N_pix = len(r)
        spec = np.zeros((N_pix, N_lam), dtype=np.float64)

        # Step 1: subtract minimum → achromatic (white) component.
        w = np.minimum(np.minimum(r, g), bv)
        spec += w[:, None] * b["white"]
        r -= w
        g -= w
        bv -= w

        # Partition: which channel became zero?
        r_zero = (r <= g) & (r <= bv)    # cyan path
        g_zero = ~r_zero & (g <= bv)     # magenta path
        b_zero = ~r_zero & ~g_zero       # yellow path

        # r ≈ 0: (g, b) → cyan + pure green or blue
        idx = np.where(r_zero)[0]
        if len(idx):
            gi, bi = g[idx], bv[idx]
            m = np.minimum(gi, bi)
            spec[idx] += m[:, None] * b["cyan"]
            gi -= m; bi -= m
            spec[idx] += gi[:, None] * b["green"]
            spec[idx] += bi[:, None] * b["blue"]

        # g ≈ 0: (r, b) → magenta + pure red or blue
        idx = np.where(g_zero)[0]
        if len(idx):
            ri, bi = r[idx], bv[idx]
            m = np.minimum(ri, bi)
            spec[idx] += m[:, None] * b["magenta"]
            ri -= m; bi -= m
            spec[idx] += ri[:, None] * b["red"]
            spec[idx] += bi[:, None] * b["blue"]

        # b ≈ 0: (r, g) → yellow + pure red or green
        idx = np.where(b_zero)[0]
        if len(idx):
            ri, gi = r[idx], g[idx]
            m = np.minimum(ri, gi)
            spec[idx] += m[:, None] * b["yellow"]
            ri -= m; gi -= m
            spec[idx] += ri[:, None] * b["red"]
            spec[idx] += gi[:, None] * b["green"]

        return spec

    def _display_decompose(self, rgb: np.ndarray) -> np.ndarray:
        """Convert linear RGB drive values into reference-display emission."""
        return rgb @ self._display_primaries

    def _coerce_rgb(self, rgb_image: np.ndarray) -> np.ndarray:
        """Validate an RGB image and return linear-light float64 RGB."""
        img = np.asarray(rgb_image)
        if np.issubdtype(img.dtype, np.unsignedinteger):
            rgb = img.astype(np.float64) / 255.0
        else:
            rgb = img.astype(np.float64)
        rgb = np.clip(rgb, 0.0, 1.0)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) array, got shape {rgb.shape}")
        return np.where(
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4,
        )
