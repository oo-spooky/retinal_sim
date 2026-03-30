"""Phase 1 visual validation: plot Govardovskii nomogram curves for all species.

Run:
    python examples/plot_nomogram.py

Produces a figure with three panels (human, dog, cat) showing each receptor
type's sensitivity curve. Compare visually against published figures from
Govardovskii et al. (2000) and species-specific literature.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from retinal_sim.retina.opsin import build_sensitivity_curves, LAMBDA_MAX

WAVELENGTHS = np.arange(380, 721, 1, dtype=float)   # 1 nm resolution for smooth curves

RECEPTOR_COLORS = {
    "S_cone": "#5b5bd6",   # blue-violet
    "M_cone": "#3aac3a",   # green
    "L_cone": "#d94c4c",   # red
    "rod":    "#888888",   # gray
}

RECEPTOR_LABELS = {
    "S_cone": "S-cone",
    "M_cone": "M-cone",
    "L_cone": "L-cone",
    "rod":    "Rod",
}

SPECIES_TITLES = {
    "human": "Human (trichromat)",
    "dog":   "Dog (dichromat)",
    "cat":   "Cat (dichromat)",
}


def plot_species(ax, species: str) -> None:
    curves = build_sensitivity_curves(species, WAVELENGTHS)
    # Draw rods first (behind cones)
    order = ["rod", "S_cone", "M_cone", "L_cone"]
    for receptor in order:
        if receptor not in curves:
            continue
        lam_max = LAMBDA_MAX[species][receptor]
        label = f"{RECEPTOR_LABELS[receptor]} (λ_max={lam_max:.0f} nm)"
        ax.plot(
            WAVELENGTHS,
            curves[receptor],
            color=RECEPTOR_COLORS[receptor],
            lw=1.8,
            label=label,
        )

    ax.set_title(SPECIES_TITLES[species], fontsize=12, fontweight="bold")
    ax.set_xlim(380, 720)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative sensitivity")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="major", lw=0.4, alpha=0.5)
    ax.grid(True, which="minor", lw=0.2, alpha=0.3)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.suptitle(
        "Govardovskii et al. (2000) A1 Nomogram — Phase 1 Validation",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    for ax, species in zip(axes, ["human", "dog", "cat"]):
        plot_species(ax, species)

    plt.tight_layout()
    out = "examples/nomogram_validation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
