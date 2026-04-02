"""Pseudoisochromatic (Ishihara-style) test pattern generation for Phase 10.

Two public functions:

``make_dot_pattern``
    Generate a circular figure on a textured dot background — the figure and
    background regions use different RGB colours to create a colour-only cue.

``find_confusion_pair``
    Search over random RGB candidates to find a colour pair (fg, bg) that lies
    on a species' confusion axis: the species cannot distinguish the two colours
    (similar photoreceptor responses), but a trichromatic human can (different
    M-cone response).
"""
from __future__ import annotations

import numpy as np

from retinal_sim.constants import WAVELENGTHS
from retinal_sim.retina.opsin import LAMBDA_MAX, govardovskii_a1
from retinal_sim.spectral.upsampler import SpectralUpsampler


def make_dot_pattern(
    fg_rgb: np.ndarray,
    bg_rgb: np.ndarray,
    image_size_px: int = 64,
    n_dots: int = 200,
    dot_radius_frac: float = 0.06,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a pseudoisochromatic dot pattern with a circular figure.

    The figure is a disc centred in the image (radius ≈ 30 % of image width).
    Both figure and background are filled with randomly placed circular dots in
    the dominant region colour.  A small fraction of dots (~15 %) use the
    opposite colour to break up the boundary — mimicking real Ishihara plates.

    Args:
        fg_rgb:          Figure colour, shape (3,) uint8.
        bg_rgb:          Background colour, shape (3,) uint8.
        image_size_px:   Image side length in pixels (square).
        n_dots:          Number of dots placed inside each region.
        dot_radius_frac: Dot radius as a fraction of ``image_size_px``.
        seed:            Random seed for reproducibility.

    Returns:
        image:        (image_size_px, image_size_px, 3) uint8 RGB.
        figure_mask:  (image_size_px, image_size_px) bool; True inside the
                      figure disc.
    """
    fg_rgb = np.asarray(fg_rgb, dtype=np.uint8).ravel()[:3]
    bg_rgb = np.asarray(bg_rgb, dtype=np.uint8).ravel()[:3]
    rng = np.random.default_rng(seed)
    H = W = image_size_px

    # --- Figure region: central circle (radius = 30 % of image) -----------
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    fig_radius = 0.30 * H
    yy, xx = np.mgrid[:H, :W]
    figure_mask: np.ndarray = ((yy - cy) ** 2 + (xx - cx) ** 2) <= fig_radius ** 2

    # --- Base image: fill each region with its dominant colour --------------
    image = np.where(figure_mask[:, :, np.newaxis], fg_rgb, bg_rgb).astype(np.uint8)

    # --- Add textured dots (15 % opposite-colour noise per region) ---------
    dot_r_base = max(1, int(dot_radius_frac * image_size_px))

    # Precompute lists of pixel indices for each region.
    fig_ys, fig_xs = np.where(figure_mask)
    bg_ys, bg_xs = np.where(~figure_mask)

    # Dots inside figure region.
    for _ in range(n_dots):
        idx = int(rng.integers(0, len(fig_ys)))
        y, x = int(fig_ys[idx]), int(fig_xs[idx])
        r = int(rng.integers(max(1, dot_r_base - 1), dot_r_base + 2))
        color = bg_rgb if rng.random() < 0.15 else fg_rgb
        image[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1] = color

    # Dots inside background region.
    for _ in range(n_dots):
        idx = int(rng.integers(0, len(bg_ys)))
        y, x = int(bg_ys[idx]), int(bg_xs[idx])
        r = int(rng.integers(max(1, dot_r_base - 1), dot_r_base + 2))
        color = fg_rgb if rng.random() < 0.15 else bg_rgb
        image[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1] = color

    # Clip to valid uint8 (dot placement near edges can wrap; indexing already
    # uses max(0, ...) so values stay within the array bounds).
    return image.clip(0, 255).astype(np.uint8), figure_mask


def find_confusion_pair(
    species: str = "dog",
    n_candidates: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Find an RGB colour pair on the species' dichromatic confusion axis.

    Searches a random grid of RGB candidates (blue channel capped at 50 to
    target the red-green confusion axis) for the pair (A, B) that:

    * minimises the species' photoreceptor-response difference — the species
      cannot tell them apart; and
    * maximises the human M-cone response difference — a human trichromat can.

    The scoring function is::

        score(A, B) = Δ_Mhuman(A, B) − 2 × ‖Δ_species(A, B)‖₂

    where Δ_species is the L2 distance in the normalised (S, L) species cone
    response space.

    Args:
        species:       ``'dog'`` or ``'cat'``.  ``'human'`` raises ValueError.
        n_candidates:  Number of random RGB colours to search.
        seed:          Random seed for candidate generation.

    Returns:
        (fg_rgb, bg_rgb): Two (3,) uint8 arrays.  ``fg_rgb`` is the "warmer"
        colour (higher R channel) and ``bg_rgb`` the "cooler" one.

    Raises:
        ValueError: If *species* is ``'human'`` (trichromats have no single
                    confusion axis) or unknown.
    """
    if species == "human":
        raise ValueError("Human is trichromat; use 'dog' or 'cat'.")
    if species not in LAMBDA_MAX:
        raise ValueError(f"Unknown species {species!r}. Choose from {list(LAMBDA_MAX)}.")

    rng = np.random.default_rng(seed)
    upsampler = SpectralUpsampler()
    wl = WAVELENGTHS.astype(float)
    dlam = float(np.mean(np.diff(wl)))

    # --- Sensitivity curves ------------------------------------------------
    lmax_species = {k: v for k, v in LAMBDA_MAX[species].items() if k != "rod"}
    lmax_human = {k: v for k, v in LAMBDA_MAX["human"].items() if k != "rod"}

    species_curves = {k: govardovskii_a1(v, wl) for k, v in lmax_species.items()}
    human_curves = {k: govardovskii_a1(v, wl) for k, v in lmax_human.items()}

    # --- Generate candidate colours: R, G random; B low (≤ 50) ------------
    R = rng.integers(40, 230, size=n_candidates).astype(np.uint8)
    G = rng.integers(40, 230, size=n_candidates).astype(np.uint8)
    B = rng.integers(0, 51, size=n_candidates).astype(np.uint8)
    candidates = np.stack([R, G, B], axis=-1)  # (N, 3) uint8

    # --- Spectral responses ------------------------------------------------
    # Upsample a (N, 1, 3) pseudo-image; squeeze to (N, N_λ).
    spectral = upsampler.upsample(candidates[:, np.newaxis, :])
    spectra = spectral.data[:, 0, :].astype(float)  # (N, N_λ)

    def _responses(curves: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: spectra @ v * dlam for k, v in curves.items()}

    species_resp = _responses(species_curves)
    human_resp = _responses(human_curves)

    # --- Normalize to [0, 1] -----------------------------------------------
    def _norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / max(hi - lo, 1e-12)

    # Species: stack all cone types into (N, K) matrix.
    species_keys = list(species_resp.keys())
    species_mat = np.stack([_norm(species_resp[k]) for k in species_keys], axis=-1)

    # Human M-cone only (the axis that is missing in the dichromat species).
    human_m = _norm(human_resp.get("M_cone", np.zeros(n_candidates)))

    # --- Pairwise scoring --------------------------------------------------
    # species_dist[i,j]: L2 distance in species response space.
    # human_m_diff[i,j]: absolute M-cone response difference.
    diff_mat = species_mat[:, np.newaxis, :] - species_mat[np.newaxis, :, :]  # (N,N,K)
    species_dist = np.sqrt((diff_mat ** 2).sum(axis=-1))  # (N, N)
    human_m_diff = np.abs(human_m[:, np.newaxis] - human_m[np.newaxis, :])    # (N, N)

    score = human_m_diff - 2.0 * species_dist
    np.fill_diagonal(score, -np.inf)

    best_i, best_j = np.unravel_index(int(np.argmax(score)), score.shape)

    fg_rgb = candidates[best_i]
    bg_rgb = candidates[best_j]

    # Convention: fg is the "warmer" (higher R) colour.
    if int(fg_rgb[0]) < int(bg_rgb[0]):
        fg_rgb, bg_rgb = bg_rgb, fg_rgb

    return fg_rgb, bg_rgb
