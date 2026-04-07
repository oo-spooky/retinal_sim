"""Ocular media transmission models for species-configured optical filtering."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_MEDIA_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "optical_media"


@dataclass(frozen=True)
class TabulatedMediaTransmission:
    """Interpolation-backed ocular media transmission curve.

    The object is callable so existing optical-stage code can treat it like a
    transmission function, while report/validation code can inspect its source
    and tabulated support directly.
    """

    wavelengths_nm: np.ndarray
    transmission: np.ndarray
    source: str

    def __call__(self, wavelengths_nm: np.ndarray) -> np.ndarray:
        query = np.asarray(wavelengths_nm, dtype=float)
        return np.interp(
            query,
            self.wavelengths_nm,
            self.transmission,
            left=float(self.transmission[0]),
            right=float(self.transmission[-1]),
        )

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe provenance metadata."""
        return {
            "kind": "tabulated",
            "source": self.source,
            "wavelengths_nm": self.wavelengths_nm.tolist(),
            "transmission": self.transmission.tolist(),
        }


def load_media_transmission_table(relative_path: str) -> TabulatedMediaTransmission:
    """Load a species ocular-media transmission table from the data directory."""
    path = (_MEDIA_DATA_DIR / relative_path).resolve()
    data = np.loadtxt(path, delimiter=",", comments="#", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"Media transmission table {path} must have exactly two columns"
        )

    wavelengths_nm = np.asarray(data[:, 0], dtype=float)
    transmission = np.asarray(data[:, 1], dtype=float)

    if np.any(np.diff(wavelengths_nm) <= 0):
        raise ValueError(f"Media transmission wavelengths must be strictly increasing: {path}")
    if np.any(transmission < 0.0) or np.any(transmission > 1.0):
        raise ValueError(f"Media transmission values must lie in [0, 1]: {path}")

    return TabulatedMediaTransmission(
        wavelengths_nm=wavelengths_nm,
        transmission=transmission,
        source=str(path),
    )


def sample_media_transmission(
    media_transmission: object | None,
    wavelengths_nm: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample a media-transmission model and return values plus provenance."""
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    if media_transmission is None:
        values = np.ones(len(wavelengths_nm), dtype=float)
        return values, {"kind": "identity", "source": "none"}

    values = np.asarray(media_transmission(wavelengths_nm), dtype=float)
    if values.shape != wavelengths_nm.shape:
        raise ValueError(
            "Media transmission callable must return one value per wavelength band"
        )

    values = np.clip(values, 0.0, 1.0)
    if hasattr(media_transmission, "summary"):
        summary = media_transmission.summary()
    else:
        summary = {
            "kind": "callable",
            "source": getattr(media_transmission, "__name__", media_transmission.__class__.__name__),
        }
    return values, summary
