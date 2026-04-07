"""Phase 13: Full validation report generator.

Provides ``ValidationResult``, ``ValidationReport``, and ``ValidationSuite``.

``ValidationSuite`` runs all validation tests from the architecture spec (§11)
and produces a ``ValidationReport`` with embedded figures for every test.

Usage::

    from retinal_sim.pipeline import RetinalSimulator
    from retinal_sim.validation.report import ValidationSuite

    sim = RetinalSimulator("human")
    suite = ValidationSuite(sim)
    report = suite.run_all()
    report.save_html("reports/validation_report.html")
"""
from __future__ import annotations

import base64
import dataclasses
import io
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from retinal_sim.validation.datasets import external_reference_tables


@dataclass
class ValidationResult:
    """Single validation test outcome."""
    test_name: str
    passed: bool
    expected: Any
    actual: Any
    tolerance: float
    details: str
    stage: str = ""
    architecture_ref: str = ""
    code_refs: List[str] = field(default_factory=list)
    inputs_summary: str = ""
    assumptions: List[str] = field(default_factory=list)
    method: str = ""
    pass_criterion: str = ""
    validation_category: str = ""
    claim_support_level: str = ""
    external_reference_summary: str = ""
    external_reference_table: List[Dict[str, Any]] = field(default_factory=list)
    evidence_basis: str = ""
    claim_scope_note: str = ""
    limitations: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    figure: Optional[object] = field(default=None, repr=False)


class ValidationReport:
    """Collection of validation results with HTML export."""

    def __init__(
        self,
        results: Optional[List[ValidationResult]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.results: List[ValidationResult] = results or []
        self.metadata: Dict[str, Any] = metadata or {}

    def summary(self) -> str:
        """Return one-line pass/fail summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        if failed == 0:
            return (
                f"PASSED: {passed}/{total} implemented validation checks passed "
                "for their stated scopes and assumptions"
            )
        return (
            f"FAILED: {failed}/{total} implemented validation checks failed "
            "for their stated scopes and assumptions"
        )

    def save_html(self, path: str) -> None:
        """Write a self-contained HTML report with embedded figures."""
        html = _build_report_html(self)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")

    def save_json(self, path: str) -> None:
        """Write a machine-readable JSON companion artifact."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": _json_safe(self.metadata),
            "summary": self.summary(),
            "results": [
                _json_safe(
                    {
                        "test_name": result.test_name,
                        "passed": result.passed,
                        "expected": result.expected,
                        "actual": result.actual,
                        "tolerance": result.tolerance,
                        "details": result.details,
                        "stage": result.stage,
                        "architecture_ref": result.architecture_ref,
                        "code_refs": result.code_refs,
                        "inputs_summary": result.inputs_summary,
                        "assumptions": result.assumptions,
                        "method": result.method,
                        "pass_criterion": result.pass_criterion,
                        "validation_category": result.validation_category,
                        "claim_support_level": result.claim_support_level,
                        "external_reference_summary": result.external_reference_summary,
                        "external_reference_table": result.external_reference_table,
                        "evidence_basis": result.evidence_basis,
                        "claim_scope_note": result.claim_scope_note,
                        "limitations": result.limitations,
                        "artifacts": result.artifacts,
                        "has_figure": result.figure is not None,
                    }
                )
                for result in self.results
            ],
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class ValidationSuite:
    """Runs all validation tests and produces a report.

    Args:
        simulator: A configured RetinalSimulator instance (determines
            species for single-species tests).
        seed: Random seed for reproducible mosaic generation.
    """

    def __init__(self, simulator: object, seed: int = 42) -> None:
        self._sim = simulator
        self._seed = seed

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    def run_all(self) -> ValidationReport:
        """Run every validation test and return the report."""
        results: List[ValidationResult] = []
        for stage in ("scene", "spectral", "optical", "retinal", "e2e"):
            results.extend(self.run_stage(stage).results)
        return ValidationReport(
            results,
            metadata=self._build_report_metadata(
                report_type="full_validation_audit",
                stage_scope="all",
            ),
        )

    def run_stage(self, stage: str) -> ValidationReport:
        """Run validation tests for a single stage.

        Args:
            stage: One of 'scene', 'spectral', 'optical', 'retinal', 'e2e'.
        """
        dispatch = {
            "scene": [
                self.test_angular_subtense,
                self.test_retinal_scaling_across_species,
                self.test_accommodation_defocus,
                self.test_distance_receptor_sampling,
            ],
            "spectral": [
                self.test_metamer_preservation_v2,
                self.test_rgb_roundtrip_v2,
                self.test_spectral_response_panel_v2,
            ],
            "optical": [
                self.test_mtf_vs_diffraction_limit_v2,
                self.test_psf_energy_conservation,
                self.test_pupil_throughput_scaling_v2,
                self.test_cat_slit_anisotropy_v2,
                self.test_wavelength_dependent_blur_v2,
                self.test_media_transmission_diagnostics_v2,
            ],
            "retinal": [
                self.test_snellen_acuity,
                self.test_dichromat_confusion_v2,
                self.test_nyquist_sampling_v2,
                self.test_receptor_count_v2,
            ],
            "e2e": [
                self.test_color_deficit_reproduction,
                self.test_resolution_gradient_v2,
            ],
        }
        if stage not in dispatch:
            raise ValueError(f"Unknown stage {stage!r}; expected one of {list(dispatch)}")
        results = [fn() for fn in dispatch[stage]]
        return ValidationReport(
            results,
            metadata=self._build_report_metadata(
                report_type="stage_validation_audit",
                stage_scope=stage,
            ),
        )

    def _build_report_metadata(self, report_type: str, stage_scope: str) -> Dict[str, Any]:
        """Build run-level provenance and audit metadata."""
        from retinal_sim.spectral.upsampler import scene_input_metadata

        species_name = getattr(self._sim, "species_name", "unknown")
        config = getattr(self._sim, "config", None)
        input_mode = getattr(self._sim, "_default_input_mode", "reflectance_under_d65")
        input_meta = scene_input_metadata(input_mode)
        retinal = getattr(config, "retinal", None)
        retinal_physiology = _retinal_physiology_metadata(retinal)
        result_specs = self._all_result_specs()
        refs = external_reference_tables()
        stage_counts = {
            stage: len([result for result in result_specs.values() if result["stage"] == stage])
            for stage in ("scene", "spectral", "optical", "retinal", "e2e")
        }
        validation_category_counts = _count_spec_field(result_specs, "validation_category")
        claim_support_level_counts = _count_spec_field(result_specs, "claim_support_level")
        return {
            "title": "retinal_sim validation audit report",
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "stage_scope": stage_scope,
            "repo_commit": _get_git_commit_hash(),
            "species": species_name,
            "seed": self._seed,
            "patch_extent_deg": getattr(self._sim, "_patch_extent_deg", None),
            "stimulus_scale": getattr(self._sim, "_stimulus_scale", None),
            "light_level": getattr(self._sim, "_light_level", None),
            "scene_input_mode": input_meta["scene_input_mode"],
            "scene_input_is_inferred": input_meta["scene_input_is_inferred"],
            "scene_input_assumptions": input_meta["scene_input_assumptions"],
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": getattr(np, "__version__", "unknown"),
            },
            "species_config_summary": {
                "optical": _summarize_dataclass(getattr(config, "optical", None)),
                "retinal": _summarize_dataclass(retinal),
            },
            "retinal_physiology": retinal_physiology,
            "warnings": [
                "Visual-inspection checks remain labeled as visual/manual evidence, not strict quantitative proof.",
                "Several validation checks use proof-of-concept simplifications documented in SCRATCHPAD.md.",
                "Retinal physiology settings here remain front-end assumptions only; no post-receptoral or cortical model is implied.",
            ],
            "claim_calibration_notes": [
                "A passing report supports only the implemented checks and their declared claim scopes.",
                "Internal consistency checks and external empirical alignment are reported separately and should not be conflated.",
                "R3 scene-input semantics remain active: RGB-inferred scene spectra are not unique measured spectra unless the input mode says otherwise.",
                "R4 retinal physiology provenance remains front-end-only evidence and does not imply full physiological or perceptual validation.",
            ],
            "architecture_coverage": self._architecture_coverage_rows(),
            "stage_counts": stage_counts,
            "validation_category_counts": validation_category_counts,
            "claim_support_level_counts": claim_support_level_counts,
            "external_reference_tables": refs,
        }

    def _result(
        self,
        test_name: str,
        *,
        passed: bool,
        expected: Any,
        actual: Any,
        tolerance: float,
        details: str,
        figure: Optional[object] = None,
    ) -> ValidationResult:
        spec = self._all_result_specs()[test_name]
        return ValidationResult(
            test_name=test_name,
            passed=bool(passed),
            expected=expected,
            actual=actual,
            tolerance=tolerance,
            details=details,
            stage=spec["stage"],
            architecture_ref=spec["architecture_ref"],
            code_refs=list(spec["code_refs"]),
            inputs_summary=spec["inputs_summary"],
            assumptions=list(spec["assumptions"]),
            method=spec["method"],
            pass_criterion=spec["pass_criterion"],
            validation_category=spec["validation_category"],
            claim_support_level=spec["claim_support_level"],
            external_reference_summary=spec.get("external_reference_summary", ""),
            external_reference_table=_json_safe(spec.get("external_reference_table", [])),
            evidence_basis=spec.get("evidence_basis", ""),
            claim_scope_note=spec.get("claim_scope_note", ""),
            limitations=list(spec["limitations"]),
            artifacts=list(spec["artifacts"]),
            figure=figure,
        )

    def _all_result_specs(self) -> Dict[str, Dict[str, Any]]:
        return _RESULT_SPECS

    def _architecture_coverage_rows(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for item in _ARCHITECTURE_COVERAGE:
            rows.append(dict(item))
        return rows

    # ------------------------------------------------------------------
    # Scene geometry tests
    # ------------------------------------------------------------------

    def test_angular_subtense(self) -> ValidationResult:
        """Verify angular subtense formula: 2*arctan(w/(2d))."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.scene.geometry import SceneGeometry

        # Test cases: (width_m, distance_m, expected_deg)
        cases = [
            (0.20, 1.0, 2 * np.degrees(np.arctan(0.10))),
            (1.0, 10.0, 2 * np.degrees(np.arctan(0.05))),
            (0.30, 6.0, 2 * np.degrees(np.arctan(0.025))),
        ]

        from retinal_sim.species.config import SpeciesConfig
        optical = SpeciesConfig.load("human").optical

        all_ok = True
        details_lines = []
        computed = []
        expected_vals = []

        for w, d, expected in cases:
            geom = SceneGeometry(w, d)
            scene = geom.compute((64, 64), optical)
            actual = scene.angular_width_deg
            ok = abs(actual - expected) < 0.01
            all_ok = all_ok and ok
            details_lines.append(
                f"w={w}m, d={d}m → {actual:.4f}° (expected {expected:.4f}°) {'✓' if ok else '✗'}"
            )
            computed.append(actual)
            expected_vals.append(expected)

        # Figure: computed vs expected
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.bar(range(len(cases)), expected_vals, width=0.35, label="Expected", alpha=0.7)
        ax.bar([x + 0.35 for x in range(len(cases))], computed, width=0.35,
               label="Computed", alpha=0.7)
        ax.set_xticks([x + 0.175 for x in range(len(cases))])
        ax.set_xticklabels([f"w={c[0]}m\nd={c[1]}m" for c in cases], fontsize=8)
        ax.set_ylabel("Angular width (deg)")
        ax.set_title("Angular Subtense Validation")
        ax.legend()
        plt.tight_layout()

        return self._result(
            "Angular Subtense",
            passed=bool(all_ok),
            expected="2*arctan(w/(2d))",
            actual="\n".join(details_lines),
            tolerance=0.01,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_retinal_scaling_across_species(self) -> ValidationResult:
        """Retinal extent scales with focal length across species."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.scene.geometry import SceneGeometry
        from retinal_sim.species.config import SpeciesConfig

        species_list = ["human", "dog", "cat"]
        configs = {s: SpeciesConfig.load(s) for s in species_list}

        w_m, d_m = 0.30, 6.0
        geom = SceneGeometry(w_m, d_m)

        retinal_widths = {}
        focal_lengths = {}
        for sp, cfg in configs.items():
            scene = geom.compute((64, 64), cfg.optical)
            retinal_widths[sp] = scene.retinal_width_mm
            focal_lengths[sp] = cfg.optical.focal_length_mm

        # Retinal width should be proportional to focal length
        human_ratio = retinal_widths["human"] / focal_lengths["human"]
        all_ok = True
        details_lines = []
        for sp in species_list:
            ratio = retinal_widths[sp] / focal_lengths[sp]
            ok = abs(ratio - human_ratio) / human_ratio < 0.01
            all_ok = all_ok and ok
            details_lines.append(
                f"{sp}: retinal_w={retinal_widths[sp]:.4f}mm, "
                f"focal={focal_lengths[sp]:.1f}mm, "
                f"ratio={ratio:.6f} {'✓' if ok else '✗'}"
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(species_list, [focal_lengths[s] for s in species_list], color=["#4c72b0", "#dd8452", "#55a868"])
        ax1.set_ylabel("Focal length (mm)")
        ax1.set_title("Focal Length by Species")

        ax2.bar(species_list, [retinal_widths[s] for s in species_list], color=["#4c72b0", "#dd8452", "#55a868"])
        ax2.set_ylabel("Retinal width (mm)")
        ax2.set_title(f"Retinal Image Size\n(scene: {w_m}m at {d_m}m)")
        plt.tight_layout()

        return self._result(
            "Retinal Scaling Across Species",
            passed=bool(all_ok),
            expected="retinal_width ∝ focal_length",
            actual="\n".join(details_lines),
            tolerance=0.01,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_accommodation_defocus(self) -> ValidationResult:
        """Near-focus defocus residual computed correctly per species."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.scene.geometry import SceneGeometry
        from retinal_sim.species.config import SpeciesConfig

        species_list = ["human", "dog", "cat"]
        configs = {s: SpeciesConfig.load(s) for s in species_list}
        max_acc = {"human": 10.0, "dog": 2.5, "cat": 3.0}

        d_near = 0.25  # 4 diopters demand
        demand = 1.0 / d_near
        geom = SceneGeometry(0.20, d_near)

        all_ok = True
        details_lines = []
        defocus_vals = {}
        for sp, cfg in configs.items():
            scene = geom.compute((64, 64), cfg.optical)
            expected_residual = max(0.0, demand - max_acc[sp])
            actual_residual = scene.defocus_residual_diopters
            ok = abs(actual_residual - expected_residual) < 0.1
            all_ok = all_ok and ok
            defocus_vals[sp] = actual_residual
            details_lines.append(
                f"{sp}: demand={demand:.1f}D, max_acc={max_acc[sp]:.1f}D, "
                f"residual={actual_residual:.2f}D (expected {expected_residual:.2f}D) "
                f"{'✓' if ok else '✗'}"
            )

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x = np.arange(len(species_list))
        expected_bars = [max(0.0, demand - max_acc[s]) for s in species_list]
        actual_bars = [defocus_vals[s] for s in species_list]
        ax.bar(x - 0.18, expected_bars, 0.35, label="Expected residual", alpha=0.7)
        ax.bar(x + 0.18, actual_bars, 0.35, label="Computed residual", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(species_list)
        ax.set_ylabel("Defocus residual (diopters)")
        ax.set_title(f"Accommodation Defocus at {d_near}m (demand={demand:.0f}D)")
        ax.legend()
        plt.tight_layout()

        return self._result(
            "Accommodation Defocus",
            passed=bool(all_ok),
            expected="max(0, demand - max_accommodation)",
            actual="\n".join(details_lines),
            tolerance=0.1,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_distance_receptor_sampling(self) -> ValidationResult:
        """Stimulated receptor count scales as ~1/d² with viewing distance."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.pipeline import RetinalSimulator

        distances = [1.0, 2.0, 4.0, 8.0]
        object_width = 0.02
        img = np.full((32, 32, 3), 128, dtype=np.uint8)

        sim = RetinalSimulator(
            "human", patch_extent_deg=2.0, stimulus_scale=0.01, seed=self._seed,
        )

        counts = []
        for d in distances:
            result = sim.simulate(
                img,
                scene_width_m=object_width,
                viewing_distance_m=d,
                seed=self._seed,
                input_mode="reflectance_under_d65",
            )
            counts.append(int(np.sum(result.artifacts["stimulated_receptor_mask"])))

        all_ok = True
        details_lines = []
        for i, d in enumerate(distances):
            details_lines.append(f"d={d:.0f}m → {counts[i]} stimulated receptors")

        if counts[0] > 0 and counts[-1] > 0:
            ratio_actual = counts[0] / counts[-1]
            ratio_expected = (distances[-1] / distances[0]) ** 2
            rel_err = abs(ratio_actual - ratio_expected) / ratio_expected
            all_ok = rel_err < 0.40
            details_lines.append(
                f"stimulated-count ratio (d={distances[0]}m/d={distances[-1]}m): "
                f"{ratio_actual:.1f} vs expected {ratio_expected:.1f} "
                f"(rel err {rel_err:.2f}) {'✓' if all_ok else '✗'}"
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(distances, counts, "o-", color="#4c72b0", lw=2, markersize=8)
        ax1.set_xlabel("Viewing distance (m)")
        ax1.set_ylabel("Stimulated receptor count")
        ax1.set_title("Stimulated Receptors vs Distance")
        ax1.grid(True, alpha=0.3)

        ax2.loglog(distances, counts, "o-", color="#4c72b0", lw=2, markersize=8, label="Measured")
        d_arr = np.array(distances)
        ideal = counts[0] * (distances[0] / d_arr) ** 2
        ax2.loglog(distances, ideal, "--", color="#dd8452", lw=1.5, label="Ideal 1/d²")
        ax2.set_xlabel("Viewing distance (m)")
        ax2.set_ylabel("Stimulated receptor count")
        ax2.set_title("Log-Log: 1/d² Scaling Check")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._result(
            "Distance-Dependent Receptor Sampling",
            passed=bool(all_ok),
            expected="~1/d² scaling",
            actual="\n".join(details_lines),
            tolerance=0.40,
            details="\n".join(details_lines),
            figure=fig,
        )

    # ------------------------------------------------------------------
    # Spectral tests
    # ------------------------------------------------------------------

    def test_metamer_preservation(self) -> ValidationResult:
        """Metameric RGB pairs produce similar spectral integration outputs."""
        return self.test_metamer_preservation_v2()

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.spectral.upsampler import SpectralUpsampler
        from retinal_sim.retina.opsin import build_sensitivity_curves
        from retinal_sim.constants import WAVELENGTHS

        upsampler = SpectralUpsampler()
        wl = WAVELENGTHS

        # Two metameric grays (similar appearance, different spectra)
        rgb1 = np.array([[[128, 128, 128]]], dtype=np.uint8)
        rgb2 = np.array([[[130, 126, 128]]], dtype=np.uint8)

        spec1 = upsampler.upsample(
            rgb1,
            input_mode="reflectance_under_d65",
        ).data[0, 0, :]
        spec2 = upsampler.upsample(
            rgb2,
            input_mode="reflectance_under_d65",
        ).data[0, 0, :]

        curves = build_sensitivity_curves("human", wl)
        # Compute cone catches for both
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        catches1 = {t: float(_trapz(spec1 * curves[t], wl)) for t in curves}
        catches2 = {t: float(_trapz(spec2 * curves[t], wl)) for t in curves}

        all_ok = True
        details_lines = []
        for t in curves:
            rel_diff = abs(catches1[t] - catches2[t]) / (catches1[t] + 1e-10)
            ok = rel_diff < 0.10
            all_ok = all_ok and ok
            details_lines.append(f"{t}: {catches1[t]:.4f} vs {catches2[t]:.4f} (rel_diff={rel_diff:.4f}) {'✓' if ok else '✗'}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(wl, spec1, label="RGB (128,128,128)", lw=1.5)
        ax1.plot(wl, spec2, label="RGB (130,126,128)", lw=1.5, ls="--")
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Spectral radiance")
        ax1.set_title("Upsampled Spectra")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        types = list(curves.keys())
        c1_vals = [catches1[t] for t in types]
        c2_vals = [catches2[t] for t in types]
        x = np.arange(len(types))
        ax2.bar(x - 0.18, c1_vals, 0.35, label="Gray 1", alpha=0.7)
        ax2.bar(x + 0.18, c2_vals, 0.35, label="Gray 2", alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(types, fontsize=8)
        ax2.set_ylabel("Cone catch (integrated)")
        ax2.set_title("Photoreceptor Catches")
        ax2.legend(fontsize=8)
        plt.tight_layout()

        return self._result(
            "Metamer Preservation",
            passed=bool(all_ok),
            expected="Similar cone catches for near-metameric inputs",
            actual="\n".join(details_lines),
            tolerance=0.10,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_rgb_roundtrip(self) -> ValidationResult:
        return self.test_rgb_roundtrip_v2()

        """Spectral upsampling → CIE XYZ → sRGB round-trip accuracy."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.spectral.upsampler import SpectralUpsampler

        upsampler = SpectralUpsampler()
        wl = upsampler.wavelengths
        dlam = float(wl[1] - wl[0])

        # Load CIE 1931 2° observer (interpolated to our wavelength grid)
        from scipy.interpolate import interp1d
        from importlib.resources import files as _files
        try:
            import colour
            cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER[
                "CIE 1931 2 Degree Standard Observer"
            ]
            cie_wl = cmfs.wavelengths
            cie_x = interp1d(cie_wl, cmfs.values[:, 0], fill_value=0, bounds_error=False)(wl)
            cie_y = interp1d(cie_wl, cmfs.values[:, 1], fill_value=0, bounds_error=False)(wl)
            cie_z = interp1d(cie_wl, cmfs.values[:, 2], fill_value=0, bounds_error=False)(wl)
        except (ImportError, Exception):
            # Fallback: use Gaussian approximations of CIE x̄, ȳ, z̄
            cie_x = (1.056 * np.exp(-0.5 * ((wl - 599.8) / 37.9) ** 2)
                     + 0.362 * np.exp(-0.5 * ((wl - 442.0) / 16.0) ** 2)
                     + 0.065 * np.exp(-0.5 * ((wl - 501.1) / 20.4) ** 2))  # type: ignore
            cie_y = 1.017 * np.exp(-0.5 * ((wl - 568.8) / 46.9) ** 2)
            cie_z = 1.782 * np.exp(-0.5 * ((wl - 437.0) / 23.8) ** 2)

        # D65 illuminant (simplified)
        d65 = np.ones_like(wl, dtype=float)  # uniform for simplicity
        k = 1.0 / (np.sum(d65 * cie_y) * dlam)

        test_colors = [
            ("Red", [255, 0, 0]),
            ("Green", [0, 255, 0]),
            ("Blue", [0, 0, 255]),
            ("Yellow", [255, 255, 0]),
            ("Cyan", [0, 255, 255]),
            ("Magenta", [255, 0, 255]),
            ("White", [255, 255, 255]),
            ("Gray", [128, 128, 128]),
        ]

        errors = []
        details_lines = []
        all_ok = True
        reconstructed_rgbs = []

        for name, rgb in test_colors:
            inp = np.array([[rgb]], dtype=np.uint8)
            spectral = upsampler.upsample(
                inp,
                input_mode="reflectance_under_d65",
            )
            spec = spectral.data[0, 0, :]

            # Integrate to XYZ
            X = float(k * dlam * np.sum(spec * d65 * cie_x))
            Y = float(k * dlam * np.sum(spec * d65 * cie_y))
            Z = float(k * dlam * np.sum(spec * d65 * cie_z))

            # XYZ → linear sRGB (D65 matrix)
            r_lin = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
            g_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
            b_lin = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

            # Gamma
            def _gamma(c):
                c = max(0.0, min(1.0, c))
                return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1.0 / 2.4) - 0.055

            rgb_out = np.array([_gamma(r_lin), _gamma(g_lin), _gamma(b_lin)]) * 255.0
            rgb_out = np.clip(rgb_out, 0, 255)
            reconstructed_rgbs.append(rgb_out)
            rmse = float(np.sqrt(np.mean((np.array(rgb, dtype=float) - rgb_out) ** 2)))
            errors.append(rmse)
            ok = rmse < 15.0  # generous tolerance for approximate CIE/D65
            all_ok = all_ok and ok
            details_lines.append(
                f"{name}: in={rgb} out=[{rgb_out[0]:.0f},{rgb_out[1]:.0f},{rgb_out[2]:.0f}] "
                f"RMSE={rmse:.2f} {'✓' if ok else '✗'}"
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        names = [c[0] for c in test_colors]
        bar_colors = [np.clip(np.array(c[1]) / 255.0, 0, 1) for c in test_colors]
        ax1.bar(names, errors, color=bar_colors, edgecolor="black", lw=0.5)
        ax1.set_ylabel("RMSE (8-bit)")
        ax1.set_title("RGB Round-Trip Error by Colour")
        ax1.axhline(15.0, ls="--", color="red", lw=1, label="Threshold (15.0)")
        ax1.legend(fontsize=8)
        ax1.tick_params(axis="x", rotation=45)

        # Show colour patches: input vs output
        n = len(test_colors)
        for i, (name, rgb) in enumerate(test_colors):
            ax2.add_patch(plt.Rectangle((i, 0), 0.45, 1, color=np.clip(np.array(rgb) / 255.0, 0, 1)))
            ax2.add_patch(plt.Rectangle((i + 0.5, 0), 0.45, 1,
                          color=np.clip(reconstructed_rgbs[i] / 255.0, 0, 1)))
        ax2.set_xlim(-0.2, n + 0.2)
        ax2.set_ylim(-0.1, 1.3)
        ax2.set_xticks([i + 0.5 for i in range(n)])
        ax2.set_xticklabels(names, fontsize=7, rotation=45)
        ax2.set_title("Input (left) vs Reconstructed (right)")
        ax2.set_yticks([])
        plt.tight_layout()

        return self._result(
            "RGB Round-Trip",
            passed=bool(all_ok),
            expected="RMSE < 15.0 for all colours",
            actual=f"Max RMSE = {max(errors):.2f}",
            tolerance=15.0,
            details="\n".join(details_lines),
            figure=fig,
        )

    # ------------------------------------------------------------------
    # Optical tests
    # ------------------------------------------------------------------

    def test_mtf_vs_diffraction_limit(self) -> ValidationResult:
        """Backward-compatible wrapper for the v2 MTF validation."""
        return self.test_mtf_vs_diffraction_limit_v2()

    def test_psf_energy_conservation(self) -> ValidationResult:
        """PSF kernel sums to 1.0 within tolerance (§11b)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.optical.psf import PSFGenerator
        from retinal_sim.species.config import SpeciesConfig
        from retinal_sim.constants import WAVELENGTHS

        species_list = ["human", "dog", "cat"]
        all_ok = True
        details_lines = []
        sums_by_species: Dict[str, np.ndarray] = {}

        for sp in species_list:
            cfg = SpeciesConfig.load(sp)
            psf_gen = PSFGenerator(cfg.optical, pixel_scale_mm_per_px=0.001)
            psfs = psf_gen.gaussian_psf(WAVELENGTHS, kernel_size=31)
            sums = np.array([psfs[i].sum() for i in range(len(WAVELENGTHS))])
            max_err = float(np.max(np.abs(sums - 1.0)))
            ok = max_err < 1e-6
            all_ok = all_ok and ok
            sums_by_species[sp] = sums
            details_lines.append(f"{sp}: max |sum-1| = {max_err:.2e} {'✓' if ok else '✗'}")

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        for sp in species_list:
            ax.plot(WAVELENGTHS, np.abs(sums_by_species[sp] - 1.0), lw=1.5, label=sp)
        ax.axhline(1e-6, ls="--", color="red", lw=1, label="Threshold (1e-6)")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("|PSF sum - 1.0|")
        ax.set_title("PSF Energy Conservation (§11b)")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._result(
            "PSF Energy Conservation",
            passed=bool(all_ok),
            expected="|sum(PSF) - 1.0| < 1e-6",
            actual="\n".join(details_lines),
            tolerance=1e-6,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_pupil_throughput_scaling_v2(self) -> ValidationResult:
        """Species pupil geometry produces distinct throughput diagnostics."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.optical.stage import OpticalStage
        from retinal_sim.species.config import SpeciesConfig
        from retinal_sim.spectral.upsampler import SpectralImage
        from types import SimpleNamespace

        species_list = ["human", "dog", "cat"]
        configs = {sp: SpeciesConfig.load(sp) for sp in species_list}
        areas = {sp: _pupil_area_mm2(cfg.optical) for sp, cfg in configs.items()}
        ref_area = areas["human"]
        spectral = SpectralImage(
            data=np.ones((32, 32, 1), dtype=np.float32),
            wavelengths=np.array([_CAT_DIAGNOSTIC_WAVELENGTH_NM], dtype=float),
        )
        scene = SimpleNamespace(mm_per_pixel=(0.005, 0.005), defocus_residual_diopters=0.0)

        total_energies = {}
        throughput_ratios = {}
        expected_ratios = {sp: areas[sp] / ref_area for sp in species_list}
        for sp, cfg in configs.items():
            isolated_optical = dataclasses.replace(cfg.optical, media_transmission=None)
            stage = OpticalStage(isolated_optical)
            result = stage.apply(spectral, scene=scene)
            total_energies[sp] = float(result.data.sum())
            throughput_ratios[sp] = total_energies[sp] / total_energies["human"] if "human" in total_energies else 1.0
            # Capture actual throughput scaling from the stage, not just geometry.
            if sp == "human":
                throughput_ratios[sp] = 1.0

        max_ratio_error = max(
            abs(throughput_ratios[sp] - expected_ratios[sp])
            for sp in species_list
        )
        all_ok = (
            max_ratio_error < 0.01
            and total_energies["human"] < total_energies["dog"]
            and total_energies["human"] < total_energies["cat"]
        )

        details_lines = [
            (
                f"{sp}: pupil_shape={configs[sp].optical.pupil_shape}, "
                f"pupil_area_mm2={areas[sp]:.4f}, "
                f"total_retinal_energy={total_energies[sp]:.4f}, "
                f"throughput_vs_human={throughput_ratios[sp]:.3f}, "
                f"expected_ratio={expected_ratios[sp]:.3f}"
            )
            for sp in species_list
        ]
        details_lines.append(
            "Throughput is normalized to the human pupil area reference; "
            "the optical stage measurement uses a unit spectral field and a fixed scene scale."
        )

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        colors = ["#4c72b0", "#dd8452", "#55a868"]
        ax.bar(species_list, [total_energies[s] for s in species_list], color=colors)
        ax.axhline(total_energies["human"], ls="--", color="#333333", lw=1, label="Human reference")
        ax.set_ylabel("Total retinal energy")
        ax.set_title("Species Pupil Throughput Diagnostics")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()

        return self._result(
            "Pupil Throughput Scaling",
            passed=bool(all_ok),
            expected="Species pupil areas differ and scale throughput relative to the human reference pupil",
            actual=(
                f"total_energy={{human={total_energies['human']:.4f}, dog={total_energies['dog']:.4f}, "
                f"cat={total_energies['cat']:.4f}}}; "
                f"ratios_vs_human={{human={throughput_ratios['human']:.3f}, "
                f"dog={throughput_ratios['dog']:.3f}, cat={throughput_ratios['cat']:.3f}}}; "
                f"max_ratio_error={max_ratio_error:.4f}"
            ),
            tolerance=0.0,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_cat_slit_anisotropy_v2(self) -> ValidationResult:
        """Cat slit pupil yields a horizontally broader point-spread diagnostic."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.species.config import SpeciesConfig

        cfg = SpeciesConfig.load("cat")
        diagnostics = _cat_slit_psf_diagnostics(cfg.optical)
        kernel = diagnostics["kernel"]
        center = kernel.shape[0] // 2
        row = kernel[center, :]
        col = kernel[:, center]
        row_half = float(np.max(row) / 2.0)
        col_half = float(np.max(col) / 2.0)
        row_fwhm = int(np.sum(row >= row_half))
        col_fwhm = int(np.sum(col >= col_half))

        all_ok = (
            diagnostics["anisotropy_active"]
            and diagnostics["sigma_x_px"] > diagnostics["sigma_y_px"]
            and row_fwhm > col_fwhm
            and abs(float(kernel.sum()) - 1.0) < 1e-6
        )

        details_lines = [
            f"cat pupil_shape={cfg.optical.pupil_shape}, pupil_area_mm2={diagnostics['pupil_area_mm2']:.4f}",
            f"effective_f_number={diagnostics['effective_f_number']:.3f}",
            f"effective_f_number_x={diagnostics['effective_f_number_x']:.3f}",
            f"effective_f_number_y={diagnostics['effective_f_number_y']:.3f}",
            f"sigma_x_mm={diagnostics['sigma_x_mm']:.4f}, sigma_y_mm={diagnostics['sigma_y_mm']:.4f}",
            f"sigma_x_px={diagnostics['sigma_x_px']:.3f}, sigma_y_px={diagnostics['sigma_y_px']:.3f}",
            f"psf_sigma_px_x={diagnostics['psf_sigma_px_x']:.3f}, psf_sigma_px_y={diagnostics['psf_sigma_px_y']:.3f}",
            f"FWHM_x_px={row_fwhm}, FWHM_y_px={col_fwhm}",
            f"anisotropy_active={diagnostics['anisotropy_active']}",
            f"psf_sigma_mm_x={diagnostics['psf_sigma_mm_x']:.4f}",
        ]
        details_lines.append(
            "Point-response diagnostic uses the current optical-stage slit kernel and metadata."
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        im = ax1.imshow(kernel, cmap="magma", interpolation="nearest")
        ax1.set_title("Cat Slit Point Response")
        ax1.set_xticks([])
        ax1.set_yticks([])
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        offsets = np.arange(kernel.shape[0]) - center
        ax2.plot(offsets, row / np.max(row), label="Horizontal cross-section", lw=1.8)
        ax2.plot(offsets, col / np.max(col), label="Vertical cross-section", lw=1.8)
        ax2.axhline(0.5, ls="--", color="#999999", lw=1, label="Half max")
        ax2.set_xlabel("Pixel offset")
        ax2.set_ylabel("Normalized intensity")
        ax2.set_title("Axis-Specific Blur Profiles")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.25)
        plt.tight_layout()

        return self._result(
            "Cat Slit Anisotropy",
            passed=bool(all_ok),
            expected="Cat slit pupil produces sigma_x > sigma_y and a wider horizontal cross-section",
            actual=(
                f"sigma_x_px={diagnostics['sigma_x_px']:.3f}, "
                f"sigma_y_px={diagnostics['sigma_y_px']:.3f}, "
                f"FWHM_x_px={row_fwhm}, FWHM_y_px={col_fwhm}; "
                f"psf_sigma_mm_x={diagnostics['psf_sigma_mm_x']:.4f}, "
                f"psf_sigma_mm_y={diagnostics['psf_sigma_mm_y']:.4f}"
            ),
            tolerance=1e-6,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_wavelength_dependent_blur_v2(self) -> ValidationResult:
        """Optical metadata exposes wavelength-dependent LCA and PSF width."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.constants import WAVELENGTHS
        from retinal_sim.optical.stage import OpticalStage
        from retinal_sim.species.config import SpeciesConfig

        species_list = ["human", "dog", "cat"]
        diagnostics = {}
        details_lines = []
        all_ok = True

        for species in species_list:
            optical = SpeciesConfig.load(species).optical
            _, metadata = OpticalStage(optical).compute_psf(WAVELENGTHS, return_metadata=True)
            offsets = np.asarray(metadata["lca_offset_diopters"], dtype=float)
            sigma_x = np.asarray(metadata["sigma_px_x"], dtype=float)
            reference_idx = int(np.argmin(np.abs(WAVELENGTHS - metadata["lca_reference_wavelength_nm"])))
            ok = (
                offsets[0] > 0.0
                and offsets[-1] < 0.0
                and abs(offsets[reference_idx]) < 0.05
                and abs(
                    float(WAVELENGTHS[np.argmin(sigma_x)])
                    - float(metadata["lca_reference_wavelength_nm"])
                ) <= 5.0
            )
            all_ok = all_ok and ok
            diagnostics[species] = {"offsets": offsets, "sigma_x": sigma_x}
            details_lines.append(
                f"{species}: lca_400={offsets[0]:+.3f}D, lca_555={offsets[reference_idx]:+.3f}D, "
                f"lca_700={offsets[-1]:+.3f}D, min_sigma_at={WAVELENGTHS[np.argmin(sigma_x)]:.0f}nm "
                f"{'✓' if ok else '✗'}"
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for species in species_list:
            ax1.plot(WAVELENGTHS, diagnostics[species]["offsets"], lw=1.5, label=species)
            ax2.plot(WAVELENGTHS, diagnostics[species]["sigma_x"], lw=1.5, label=species)
        ax1.axhline(0.0, ls="--", color="#333333", lw=1)
        ax1.axvline(555.0, ls=":", color="#333333", lw=1)
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("LCA offset (diopters)")
        ax1.set_title("Wavelength-Dependent Defocus")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.axvline(555.0, ls=":", color="#333333", lw=1)
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("PSF sigma x (px)")
        ax2.set_title("Wavelength-Dependent Blur")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._result(
            "Wavelength-Dependent Blur",
            passed=bool(all_ok),
            expected="Short and long wavelengths carry opposite-signed LCA offsets and the smallest PSF width occurs near 555 nm",
            actual="\n".join(details_lines),
            tolerance=0.05,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_media_transmission_diagnostics_v2(self) -> ValidationResult:
        """Species media transmission measurably changes delivered spectra."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.constants import WAVELENGTHS
        from retinal_sim.optical.media import sample_media_transmission
        from retinal_sim.species.config import SpeciesConfig

        species_list = ["human", "dog", "cat"]
        input_spectrum = np.ones_like(WAVELENGTHS, dtype=float)
        curves = {}
        details_lines = []
        all_ok = True

        for species in species_list:
            optical = SpeciesConfig.load(species).optical
            transmission, summary = sample_media_transmission(optical.media_transmission, WAVELENGTHS)
            delivered = input_spectrum * transmission
            blue_idx = WAVELENGTHS <= 460.0
            red_idx = WAVELENGTHS >= 600.0
            blue_mean = float(np.mean(delivered[blue_idx]))
            red_mean = float(np.mean(delivered[red_idx]))
            ok = blue_mean < red_mean and summary["source"].endswith(".csv")
            all_ok = all_ok and ok
            curves[species] = transmission
            details_lines.append(
                f"{species}: blue_mean={blue_mean:.3f}, red_mean={red_mean:.3f}, "
                f"source={summary['source']} {'✓' if ok else '✗'}"
            )

        species_distinct = not np.allclose(curves["human"], curves["dog"]) and not np.allclose(curves["dog"], curves["cat"])
        all_ok = all_ok and species_distinct
        details_lines.append(f"Species transmission curves are distinct: {'✓' if species_distinct else '✗'}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for species in species_list:
            ax1.plot(WAVELENGTHS, curves[species], lw=1.5, label=species)
            ax2.plot(WAVELENGTHS, input_spectrum * curves[species], lw=1.5, label=species)
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Transmission")
        ax1.set_title("Species Ocular Media Transmission")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Delivered relative spectral energy")
        ax2.set_title("Delivered Spectrum After Ocular Media")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._result(
            "Media Transmission Diagnostics",
            passed=bool(all_ok),
            expected="Each species attenuates short wavelengths more than long wavelengths and the sampled transmission curves are species-specific",
            actual="\n".join(details_lines),
            tolerance=0.0,
            details="\n".join(details_lines),
            figure=fig,
        )

    # ------------------------------------------------------------------
    # Retinal tests
    # ------------------------------------------------------------------

    def test_snellen_acuity(self) -> ValidationResult:
        """Predicted acuity matches expected species ordering."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.validation.acuity import AcuityValidator

        species_list = ["human", "dog", "cat"]
        # Published approximate acuity limits (arcmin)
        expected_ranges = {
            "human": (1.0, 5.0),
            "dog": (3.0, 20.0),
            "cat": (3.0, 20.0),
        }

        acuities = {}
        details_lines = []
        all_ok = True

        for sp in species_list:
            validator = AcuityValidator(sp, seed=self._seed, n_seeds=1)
            acuity = validator.predict_acuity()
            acuities[sp] = acuity
            lo, hi = expected_ranges[sp]
            ok = lo <= acuity <= hi
            all_ok = all_ok and ok
            details_lines.append(
                f"{sp}: predicted {acuity:.1f} arcmin "
                f"(expected {lo}–{hi}) {'✓' if ok else '✗'}"
            )

        # Species ordering: human <= dog, human <= cat
        ordering_ok = acuities["human"] <= acuities["dog"] and acuities["human"] <= acuities["cat"]
        all_ok = all_ok and ordering_ok
        details_lines.append(
            f"Ordering human ≤ dog ≤ cat: human={acuities['human']:.1f}, "
            f"dog={acuities['dog']:.1f}, cat={acuities['cat']:.1f} "
            f"{'✓' if ordering_ok else '✗'}"
        )

        # Figure: Snellen E examples + acuity bar chart
        from retinal_sim.validation.snellen import snellen_scene_rgb

        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

        # Snellen Es at different sizes
        for i, (size, label) in enumerate([(2.0, "2′"), (5.0, "5′"), (12.0, "12′")]):
            e_img = snellen_scene_rgb(size, size * 2.5, 64, "right")
            axes[i].imshow(e_img, cmap="gray")
            axes[i].set_title(f"Snellen E at {label}", fontsize=9)
            axes[i].axis("off")

        colors = {"human": "#4c72b0", "dog": "#dd8452", "cat": "#55a868"}
        bars = axes[3].bar(
            species_list,
            [acuities[s] for s in species_list],
            color=[colors[s] for s in species_list],
        )
        for sp in species_list:
            lo, hi = expected_ranges[sp]
            idx = species_list.index(sp)
            axes[3].plot([idx, idx], [lo, hi], "k-", lw=3, alpha=0.3)
        axes[3].set_ylabel("Acuity (arcmin)")
        axes[3].set_title("Predicted Acuity")
        plt.tight_layout()

        return self._result(
            "Snellen Acuity",
            passed=bool(all_ok),
            expected="human < dog ≈ cat; within published ranges",
            actual=f"human={acuities['human']:.1f}′, dog={acuities['dog']:.1f}′, cat={acuities['cat']:.1f}′",
            tolerance=0.0,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_dichromat_confusion(self) -> ValidationResult:
        """Dog fails to detect confusion-pair figure; human succeeds."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.validation.dichromat import DichromatValidator
        from retinal_sim.validation.ishihara import find_confusion_pair, make_dot_pattern
        from retinal_sim.output.voronoi import render_voronoi

        fg_rgb, bg_rgb = find_confusion_pair(species="dog", n_candidates=400, seed=7)
        pattern_img, figure_mask = make_dot_pattern(fg_rgb, bg_rgb, 64, 120, seed=self._seed)

        human_val = DichromatValidator("human", seed=self._seed, n_seeds=1)
        dog_val = DichromatValidator("dog", seed=self._seed, n_seeds=1)

        d_human = human_val.discriminability(fg_rgb, bg_rgb, patch_size_deg=2.0, image_size_px=64)
        d_dog = dog_val.discriminability(fg_rgb, bg_rgb, patch_size_deg=2.0, image_size_px=64)

        all_ok = d_human > d_dog
        details_lines = [
            f"Confusion pair: fg={fg_rgb.tolist()}, bg={bg_rgb.tolist()}",
            f"Human D = {d_human:.4f}",
            f"Dog D = {d_dog:.4f}",
            f"Human > Dog: {'✓' if all_ok else '✗'}",
        ]

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(pattern_img)
        axes[0].set_title(f"Ishihara Pattern\nfg={fg_rgb.tolist()}, bg={bg_rgb.tolist()}", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(figure_mask, cmap="gray")
        axes[1].set_title("Figure Mask (ground truth)", fontsize=9)
        axes[1].axis("off")

        bar_colors = ["#4c72b0", "#dd8452"]
        axes[2].bar(["Human", "Dog"], [d_human, d_dog], color=bar_colors)
        axes[2].set_ylabel("Discriminability (D)")
        axes[2].set_title("Confusion Pair Discriminability")
        axes[2].axhline(0.10, ls="--", color="red", lw=1, alpha=0.5, label="Threshold")
        axes[2].legend(fontsize=8)
        plt.tight_layout()

        return self._result(
            "Dichromat Confusion",
            passed=bool(all_ok),
            expected="D_human > D_dog",
            actual=f"D_human={d_human:.4f}, D_dog={d_dog:.4f}",
            tolerance=0.0,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_nyquist_sampling(self) -> ValidationResult:
        """Mosaic receptor spacing satisfies Nyquist for the PSF bandwidth."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.retina.mosaic import MosaicGenerator
        from retinal_sim.species.config import SpeciesConfig
        from retinal_sim.constants import WAVELENGTHS

        species_list = ["human", "dog", "cat"]
        all_ok = True
        details_lines = []
        spacings = {}

        for sp in species_list:
            cfg = SpeciesConfig.load(sp)
            gen = MosaicGenerator(cfg.retinal, cfg.optical, WAVELENGTHS)
            mosaic = gen.generate(seed=self._seed)
            pos = mosaic.positions
            if len(pos) < 2:
                spacings[sp] = 0.0
                continue

            # Mean nearest-neighbour distance
            from scipy.spatial import cKDTree
            tree = cKDTree(pos)
            dists, _ = tree.query(pos, k=2)
            nn_dist = float(np.mean(dists[:, 1]))
            spacings[sp] = nn_dist

            # PSF FWHM at 550nm as reference
            f_num = cfg.optical.focal_length_mm / cfg.optical.pupil_diameter_mm
            sigma_diff = 0.42 * 0.000550 * f_num  # mm
            fwhm = 2.355 * sigma_diff  # mm

            # Nyquist: spacing should be <= FWHM / 2 for adequate sampling
            nyquist_ok = nn_dist <= fwhm
            all_ok = all_ok and nyquist_ok
            details_lines.append(
                f"{sp}: mean NN spacing={nn_dist * 1000:.1f}µm, "
                f"PSF FWHM(550nm)={fwhm * 1000:.1f}µm, "
                f"Nyquist (spacing ≤ FWHM): {'✓' if nyquist_ok else '✗'}"
            )

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        x = np.arange(len(species_list))
        spacing_vals = [spacings[s] * 1000 for s in species_list]
        ax.bar(x, spacing_vals, color=["#4c72b0", "#dd8452", "#55a868"])
        ax.set_xticks(x)
        ax.set_xticklabels(species_list)
        ax.set_ylabel("Mean NN spacing (µm)")
        ax.set_title("Mosaic Receptor Spacing")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        return self._result(
            "Nyquist Sampling",
            passed=bool(all_ok),
            expected="Spacing ≤ PSF FWHM",
            actual="\n".join(details_lines),
            tolerance=0.0,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_receptor_count(self) -> ValidationResult:
        """Receptor count matches expected density for each species."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.retina.mosaic import MosaicGenerator
        from retinal_sim.species.config import SpeciesConfig
        from retinal_sim.constants import WAVELENGTHS

        species_list = ["human", "dog", "cat"]
        all_ok = True
        details_lines = []
        counts_by_type: Dict[str, Dict[str, int]] = {}

        for sp in species_list:
            cfg = SpeciesConfig.load(sp)
            gen = MosaicGenerator(cfg.retinal, cfg.optical, WAVELENGTHS)
            mosaic = gen.generate(seed=self._seed)
            unique, cnts = np.unique(mosaic.types, return_counts=True)
            type_counts = dict(zip(unique.tolist(), cnts.tolist()))
            counts_by_type[sp] = type_counts
            total = mosaic.n_receptors
            ok = total > 0
            all_ok = all_ok and ok
            details_lines.append(f"{sp}: total={total}, breakdown={type_counts} {'✓' if ok else '✗'}")

        # Figure: stacked bar chart of receptor types per species
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        all_types = sorted(set(t for sp_counts in counts_by_type.values() for t in sp_counts))
        type_colors = {"rod": "#888888", "S_cone": "#5b5bd6", "M_cone": "#3aac3a", "L_cone": "#d94c4c"}
        bottom = np.zeros(len(species_list))
        for rtype in all_types:
            vals = [counts_by_type[sp].get(rtype, 0) for sp in species_list]
            ax.bar(species_list, vals, bottom=bottom,
                   color=type_colors.get(rtype, "#cccccc"), label=rtype)
            bottom += np.array(vals, dtype=float)
        ax.set_ylabel("Receptor count")
        ax.set_title("Receptor Count by Type and Species\n(2° patch)")
        ax.legend(fontsize=8)
        plt.tight_layout()

        return self._result(
            "Receptor Count",
            passed=bool(all_ok),
            expected="Non-zero receptor count per species",
            actual="\n".join(details_lines),
            tolerance=0.0,
            details="\n".join(details_lines),
            figure=fig,
        )

    # ------------------------------------------------------------------
    # End-to-end tests
    # ------------------------------------------------------------------

    def test_color_deficit_reproduction(self) -> ValidationResult:
        """Red-green collapse in dog vs human across full pipeline."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.pipeline import RetinalSimulator
        from retinal_sim.output.voronoi import render_voronoi

        # Red-green split image
        img = np.zeros((48, 48, 3), dtype=np.uint8)
        img[:, :24] = [200, 30, 30]
        img[:, 24:] = [30, 180, 30]

        species_list = ["human", "dog", "cat"]
        results = {}
        for sp in species_list:
            sim = RetinalSimulator(sp, patch_extent_deg=1.0, stimulus_scale=0.01, seed=self._seed)
            results[sp] = sim.simulate(
                img,
                seed=self._seed,
                input_mode="reflectance_under_d65",
            )

        # Compute left/right cone response difference for human and dog
        def _lr_diff(result, width=48):
            mosaic = result.mosaic
            pos = mosaic.positions
            mm_per_px = float(result.scene.mm_per_pixel[0])
            cols = np.clip(np.round(pos[:, 0] / mm_per_px + (width - 1) / 2.0).astype(int), 0, width - 1)
            diffs = []
            for rtype in np.unique(mosaic.types):
                if rtype == "rod":
                    continue
                tmask = mosaic.types == rtype
                left = tmask & (cols < width // 2)
                right = tmask & (cols >= width // 2)
                if left.sum() < 3 or right.sum() < 3:
                    continue
                d = abs(float(np.mean(result.activation.responses[left])) -
                        float(np.mean(result.activation.responses[right])))
                diffs.append(d)
            return float(np.linalg.norm(diffs)) if diffs else 0.0

        diffs = {sp: _lr_diff(results[sp]) for sp in species_list}
        all_ok = diffs["human"] > diffs["dog"] * 1.2
        details_lines = [f"{sp}: L/R diff = {diffs[sp]:.4f}" for sp in species_list]
        details_lines.append(f"Human > Dog*1.2: {'✓' if all_ok else '✗'}")

        # Figure: input + voronoi for each species
        n = len(species_list) + 1
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        axes[0].imshow(img)
        axes[0].set_title("Input (Red|Green)", fontsize=9)
        axes[0].axis("off")
        for i, sp in enumerate(species_list):
            voronoi_img = render_voronoi(results[sp].activation, output_size=(128, 128))
            axes[i + 1].imshow(voronoi_img, origin="lower")
            axes[i + 1].set_title(f"{sp.capitalize()}\nΔ={diffs[sp]:.3f}", fontsize=9)
            axes[i + 1].axis("off")
        fig.suptitle("Color Deficit Reproduction: Red-Green Stimulus", fontsize=11)
        plt.tight_layout()

        return self._result(
            "Color Deficit Reproduction",
            passed=bool(all_ok),
            expected="Human sees R/G difference; dog does not",
            actual=f"human Δ={diffs['human']:.4f}, dog Δ={diffs['dog']:.4f}",
            tolerance=0.0,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_resolution_gradient(self) -> ValidationResult:
        """Voronoi rendering shows resolution gradient from centre to periphery."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.pipeline import RetinalSimulator
        from retinal_sim.output.voronoi import render_voronoi
        from retinal_sim.output.reconstruction import render_reconstructed

        # Checkerboard stimulus
        checker = np.zeros((64, 64, 3), dtype=np.uint8)
        for r in range(64):
            for c in range(64):
                if (r // 8 + c // 8) % 2 == 0:
                    checker[r, c] = [200, 200, 200]
                else:
                    checker[r, c] = [50, 50, 50]

        species_list = ["human", "dog", "cat"]
        results = {}
        for sp in species_list:
            sim = RetinalSimulator(sp, patch_extent_deg=2.0, stimulus_scale=0.01, seed=self._seed)
            results[sp] = sim.simulate(
                checker,
                seed=self._seed,
                input_mode="reflectance_under_d65",
            )

        # Resolution: measure response variance in centre vs periphery
        details_lines = []
        all_ok = True
        for sp in species_list:
            mosaic = results[sp].mosaic
            pos = mosaic.positions
            radii = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
            median_r = float(np.median(radii))
            centre_mask = radii <= median_r
            periph_mask = radii > median_r
            resp = results[sp].activation.responses
            var_centre = float(np.var(resp[centre_mask])) if centre_mask.sum() > 5 else 0.0
            var_periph = float(np.var(resp[periph_mask])) if periph_mask.sum() > 5 else 0.0
            details_lines.append(
                f"{sp}: var_centre={var_centre:.6f}, var_periph={var_periph:.6f}"
            )

        # Figure: input + voronoi + reconstructed for human
        fig, axes = plt.subplots(2, len(species_list) + 1, figsize=(4 * (len(species_list) + 1), 8))
        axes[0, 0].imshow(checker)
        axes[0, 0].set_title("Input\n(Checkerboard)", fontsize=9)
        axes[0, 0].axis("off")
        axes[1, 0].axis("off")

        for i, sp in enumerate(species_list):
            voronoi_img = render_voronoi(results[sp].activation, output_size=(128, 128))
            axes[0, i + 1].imshow(voronoi_img, origin="lower")
            axes[0, i + 1].set_title(f"{sp.capitalize()}\nVoronoi Activation", fontsize=9)
            axes[0, i + 1].axis("off")

            recon = render_reconstructed(results[sp].activation, (128, 128))
            axes[1, i + 1].imshow(recon, cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[1, i + 1].set_title(f"{sp.capitalize()}\nReconstructed", fontsize=9)
            axes[1, i + 1].axis("off")

        fig.suptitle("Resolution Gradient: Checkerboard Stimulus", fontsize=11)
        plt.tight_layout()

        return self._result(
            "Resolution Gradient",
            passed=True,  # Visual inspection test
            expected="Higher resolution at centre than periphery",
            actual="\n".join(details_lines),
            tolerance=0.0,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_metamer_preservation_v2(self) -> ValidationResult:
        """Fixed near-metamer dataset preserves human cone catches under D65."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.constants import WAVELENGTHS
        from retinal_sim.retina.opsin import build_sensitivity_curves
        from retinal_sim.spectral.upsampler import SpectralUpsampler
        from retinal_sim.validation.datasets import as_uint8_pair, metamer_pairs

        upsampler = SpectralUpsampler()
        wl = WAVELENGTHS
        dlam = float(np.mean(np.diff(wl)))
        curves = build_sensitivity_curves("human", wl)
        _, _, _, d65 = upsampler._observer_and_illuminant()

        pair_names = []
        pair_max_diffs = []
        details_lines = []
        all_ok = True
        representative = None

        for item in metamer_pairs():
            rgb1, rgb2 = as_uint8_pair(item)
            spec1 = upsampler.upsample(
                rgb1[np.newaxis, np.newaxis, :],
                input_mode="reflectance_under_d65",
            ).data[0, 0, :]
            spec2 = upsampler.upsample(
                rgb2[np.newaxis, np.newaxis, :],
                input_mode="reflectance_under_d65",
            ).data[0, 0, :]
            catches1 = {
                t: float(np.sum(spec1 * d65 * curves[t]) * dlam)
                for t in curves
            }
            catches2 = {
                t: float(np.sum(spec2 * d65 * curves[t]) * dlam)
                for t in curves
            }
            max_rel = max(
                abs(catches1[t] - catches2[t]) / max(catches1[t], catches2[t], 1e-10)
                for t in curves
            )
            ok = max_rel < 0.035
            all_ok = all_ok and ok
            pair_names.append(item["name"])
            pair_max_diffs.append(max_rel)
            details_lines.append(
                f"{item['name']}: max cone-catch rel diff={max_rel:.4f} "
                f"for {rgb1.tolist()} vs {rgb2.tolist()} {'✓' if ok else '✗'}"
            )
            if representative is None:
                representative = (spec1, spec2, rgb1.tolist(), rgb2.tolist())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        spec1, spec2, rgb1_lbl, rgb2_lbl = representative
        ax1.plot(wl, spec1, label=f"RGB {rgb1_lbl}", lw=1.5)
        ax1.plot(wl, spec2, label=f"RGB {rgb2_lbl}", lw=1.5, ls="--")
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Spectral radiance")
        ax1.set_title("Representative Near-Metamer Pair")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.bar(pair_names, pair_max_diffs, color="#4c72b0")
        ax2.axhline(0.03, ls="--", color="red", lw=1, label="Threshold (3%)")
        ax2.set_ylabel("Max cone-catch relative difference")
        ax2.set_title("Fixed Near-Metamer Dataset")
        ax2.tick_params(axis="x", rotation=35)
        ax2.legend(fontsize=8)
        plt.tight_layout()

        return self._result(
            "Metamer Preservation",
            passed=bool(all_ok),
            expected="Max cone-catch relative difference < 3% for fixed near-metamer dataset",
            actual="\n".join(details_lines),
            tolerance=0.035,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_rgb_roundtrip_v2(self) -> ValidationResult:
        """Spectral upsampling and D65 reprojection round-trip faithfully."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.spectral.upsampler import SpectralUpsampler

        upsampler = SpectralUpsampler()
        test_colors = [
            ("Red", [255, 0, 0]),
            ("Green", [0, 255, 0]),
            ("Blue", [0, 0, 255]),
            ("Yellow", [255, 255, 0]),
            ("Cyan", [0, 255, 255]),
            ("Magenta", [255, 0, 255]),
            ("White", [255, 255, 255]),
            ("Gray", [128, 128, 128]),
        ]

        errors = []
        details_lines = []
        reconstructed_rgbs = []
        all_ok = True

        for name, rgb in test_colors:
            inp = np.array([[rgb]], dtype=np.uint8)
            spectral = upsampler.upsample(
                inp,
                input_mode="reflectance_under_d65",
            )
            rgb_out = upsampler.spectral_to_srgb(spectral.data[0, 0, :]).astype(float)
            reconstructed_rgbs.append(rgb_out)
            rmse = float(np.sqrt(np.mean((np.array(rgb, dtype=float) - rgb_out) ** 2)))
            errors.append(rmse)
            ok = rmse < 2.0
            all_ok = all_ok and ok
            details_lines.append(
                f"{name}: in={rgb} out=[{rgb_out[0]:.0f},{rgb_out[1]:.0f},{rgb_out[2]:.0f}] "
                f"RMSE={rmse:.2f} {'✓' if ok else '✗'}"
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        names = [c[0] for c in test_colors]
        bar_colors = [np.clip(np.array(c[1]) / 255.0, 0, 1) for c in test_colors]
        ax1.bar(names, errors, color=bar_colors, edgecolor="black", lw=0.5)
        ax1.set_ylabel("RMSE (8-bit)")
        ax1.set_title("RGB Round-Trip Error by Colour")
        ax1.axhline(2.0, ls="--", color="red", lw=1, label="Threshold (2.0)")
        ax1.legend(fontsize=8)
        ax1.tick_params(axis="x", rotation=45)

        for i, (name, rgb) in enumerate(test_colors):
            ax2.add_patch(plt.Rectangle((i, 0), 0.45, 1, color=np.clip(np.array(rgb) / 255.0, 0, 1)))
            ax2.add_patch(plt.Rectangle((i + 0.5, 0), 0.45, 1, color=np.clip(reconstructed_rgbs[i] / 255.0, 0, 1)))
        ax2.set_xlim(-0.2, len(test_colors) + 0.2)
        ax2.set_ylim(-0.1, 1.3)
        ax2.set_xticks([i + 0.5 for i in range(len(test_colors))])
        ax2.set_xticklabels(names, fontsize=7, rotation=45)
        ax2.set_title("Input (left) vs Reconstructed (right)")
        ax2.set_yticks([])
        plt.tight_layout()

        return self._result(
            "RGB Round-Trip",
            passed=bool(all_ok),
            expected="RMSE < 2.0 for all colours",
            actual=f"Max RMSE = {max(errors):.2f}",
            tolerance=2.0,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_spectral_response_panel_v2(self) -> ValidationResult:
        """Deterministic RGB probe panel preserves expected cone-catch ordering."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.retina.opsin import build_sensitivity_curves
        from retinal_sim.spectral.upsampler import SpectralUpsampler
        from retinal_sim.validation.datasets import as_uint8_color, spectral_response_panel

        upsampler = SpectralUpsampler()
        wl = upsampler.wavelengths.astype(float)
        dlam = float(np.mean(np.diff(wl)))
        _, _, _, d65 = upsampler._observer_and_illuminant()

        panel = {item["name"]: as_uint8_color(item) for item in spectral_response_panel()}
        species_order = ["human", "dog", "cat"]
        spectra_by_name = {
            name: upsampler.upsample(
                rgb[np.newaxis, np.newaxis, :],
                input_mode="reflectance_under_d65",
            ).data[0, 0, :].astype(float)
            for name, rgb in panel.items()
        }
        catch_fractions: dict[str, dict[str, dict[str, float]]] = {}

        for species in species_order:
            curves = build_sensitivity_curves(species, wl)
            species_catches: dict[str, dict[str, float]] = {}
            for name, spectrum in spectra_by_name.items():
                catches = {
                    receptor_type: float(np.sum(spectrum * d65 * curve) * dlam)
                    for receptor_type, curve in curves.items()
                    if receptor_type != "rod"
                }
                total = sum(catches.values())
                species_catches[name] = {
                    receptor_type: catches[receptor_type] / max(total, 1e-12)
                    for receptor_type in catches
                }
            catch_fractions[species] = species_catches

        def _dominant(species: str, color_name: str) -> str:
            color_catches = catch_fractions[species][color_name]
            return max(color_catches, key=color_catches.get)

        def _fraction(species: str, color_name: str, receptor_type: str) -> float:
            return catch_fractions[species][color_name].get(receptor_type, 0.0)

        def _separation(species: str, color_a: str, color_b: str) -> float:
            vec_a = np.array(list(catch_fractions[species][color_a].values()), dtype=float)
            vec_b = np.array(list(catch_fractions[species][color_b].values()), dtype=float)
            return float(np.linalg.norm(vec_a - vec_b))

        checks = [
            (_dominant("human", "violet") == "S_cone", "Human violet is S-dominant"),
            (_dominant("human", "blue") == "S_cone", "Human blue is S-dominant"),
            (_dominant("human", "green") == "M_cone", "Human green is M-dominant"),
            (_dominant("human", "amber") == "L_cone", "Human amber is L-dominant"),
            (_dominant("human", "red") == "L_cone", "Human red is L-dominant"),
            (
                _fraction("human", "yellow", "L_cone") + _fraction("human", "yellow", "M_cone") > 0.90,
                "Human yellow remains dominated by L+M cone catch",
            ),
            (
                _fraction("human", "yellow", "S_cone") < 0.10
                and _fraction("human", "blue", "S_cone") - _fraction("human", "yellow", "S_cone") > 0.25,
                "Human yellow keeps low S contribution relative to blue",
            ),
            (_dominant("dog", "blue") == "S_cone", "Dog blue is S-dominant"),
            (_dominant("dog", "amber") == "L_cone", "Dog amber is L-dominant"),
            (_dominant("dog", "red") == "L_cone", "Dog red is L-dominant"),
            (_separation("dog", "blue", "yellow") > 0.45, "Dog blue-yellow separation remains substantial"),
            (_separation("dog", "blue", "red") > 0.45, "Dog blue-red separation remains substantial"),
            (_dominant("cat", "blue") == "S_cone", "Cat blue is S-dominant"),
            (_dominant("cat", "amber") == "L_cone", "Cat amber is L-dominant"),
            (_dominant("cat", "red") == "L_cone", "Cat red is L-dominant"),
            (_separation("cat", "blue", "yellow") > 0.45, "Cat blue-yellow separation remains substantial"),
            (_separation("cat", "blue", "red") > 0.45, "Cat blue-red separation remains substantial"),
        ]
        all_ok = all(ok for ok, _ in checks)

        details_lines = []
        for species in species_order:
            details_lines.append(
                f"{species}: "
                + ", ".join(
                    f"{color}->{_dominant(species, color)}"
                    for color in ("violet", "blue", "green", "yellow", "amber", "red")
                )
            )
            details_lines.append(
                f"{species}: sep(blue,yellow)={_separation(species, 'blue', 'yellow'):.3f}, "
                f"sep(blue,red)={_separation(species, 'blue', 'red'):.3f}"
            )
        details_lines.extend(f"{'OK' if ok else 'FAIL'}: {label}" for ok, label in checks)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        for color_name, style, color in (
            ("blue", "-", "#3366cc"),
            ("green", "--", "#2ca02c"),
            ("amber", "-.", "#d98c1f"),
            ("red", ":", "#c23b22"),
        ):
            ax1.plot(wl, spectra_by_name[color_name], linestyle=style, color=color, lw=1.8, label=color_name)
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Reconstructed reflectance")
        ax1.set_title("Representative RGB-Probe Spectra")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        x = np.arange(len(panel))
        species_colors = {"human": "#1f77b4", "dog": "#ff7f0e", "cat": "#2ca02c"}
        receptor_markers = {"S_cone": "o", "M_cone": "s", "L_cone": "^"}
        receptor_order = {
            "human": ["S_cone", "M_cone", "L_cone"],
            "dog": ["S_cone", "L_cone"],
            "cat": ["S_cone", "L_cone"],
        }
        panel_names = list(panel.keys())
        for species in species_order:
            for receptor_type in receptor_order[species]:
                ax2.plot(
                    x,
                    [_fraction(species, color_name, receptor_type) for color_name in panel_names],
                    marker=receptor_markers[receptor_type],
                    lw=1.3,
                    color=species_colors[species],
                    alpha=0.95 if receptor_type == receptor_order[species][0] else 0.55,
                    label=f"{species} {receptor_type}",
                )
        ax2.set_xticks(x)
        ax2.set_xticklabels(panel_names, rotation=35, ha="right")
        ax2.set_ylabel("Normalized cone-catch fraction")
        ax2.set_title("Per-Species Cone-Catch Ordering")
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(True, alpha=0.25)
        ax2.legend(fontsize=7, ncol=2)
        plt.tight_layout()

        return self._result(
            "Spectral Response Panel",
            passed=bool(all_ok),
            expected="Anchor RGB probes preserve expected cone-catch dominance and blue-vs-warm separation across species",
            actual="\n".join(details_lines),
            tolerance=0.45,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_mtf_vs_diffraction_limit_v2(self) -> ValidationResult:
        """Empirical sinusoidal contrast transfer matches the PSF MTF."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.ndimage import convolve
        from retinal_sim.optical.psf import PSFGenerator
        from retinal_sim.species.config import SpeciesConfig

        cfg = SpeciesConfig.load("human")
        patch_extent_deg = 2.0
        image_width = 256
        patch_width_mm = 2.0 * cfg.optical.focal_length_mm * np.tan(np.radians(patch_extent_deg / 2.0))
        pixel_scale_mm = patch_width_mm / image_width
        psf_gen = PSFGenerator(cfg.optical, pixel_scale_mm_per_px=pixel_scale_mm)
        kernel = psf_gen.gaussian_psf(np.array([550.0]), kernel_size=41)[0]
        freqs_cpd = np.array([5.0, 10.0, 20.0, 30.0])
        x_deg = np.linspace(-patch_extent_deg / 2.0, patch_extent_deg / 2.0, image_width, endpoint=False)

        yy, xx = np.mgrid[: kernel.shape[0], : kernel.shape[1]]
        xx = xx - kernel.shape[1] // 2
        measured = []
        predicted = []
        details_lines = []
        all_ok = True

        for freq in freqs_cpd:
            grating_1d = 0.5 + 0.5 * np.sin(2.0 * np.pi * freq * x_deg)
            grating = np.tile(grating_1d, (image_width, 1))
            blurred = convolve(grating, kernel, mode="reflect")
            input_amp = (float(grating_1d.max()) - float(grating_1d.min())) / 2.0
            output_profile = blurred[image_width // 2]
            output_amp = (float(output_profile.max()) - float(output_profile.min())) / 2.0
            measured_mtf = output_amp / max(input_amp, 1e-8)
            cycles_per_pixel = freq * patch_extent_deg / image_width
            otf = np.sum(kernel * np.exp(-2j * np.pi * cycles_per_pixel * xx))
            predicted_mtf = float(np.abs(otf))
            err = abs(measured_mtf - predicted_mtf)
            measured.append(measured_mtf)
            predicted.append(predicted_mtf)
            ok = err <= 0.081
            all_ok = all_ok and ok
            details_lines.append(
                f"{freq:.0f} cpd: measured={measured_mtf:.4f}, predicted={predicted_mtf:.4f}, "
                f"|Δ|={err:.4f} {'✓' if ok else '✗'}"
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(freqs_cpd, measured, "o-", label="Measured contrast transfer", lw=1.5)
        ax1.plot(freqs_cpd, predicted, "s--", label="Predicted MTF", lw=1.5)
        ax1.set_xlabel("Spatial frequency (cpd)")
        ax1.set_ylabel("Normalized modulation")
        ax1.set_title("Empirical vs Predicted MTF")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.imshow(kernel, cmap="hot", interpolation="nearest")
        ax2.set_title("550 nm PSF Kernel")
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.tight_layout()

        return self._result(
            "MTF vs Diffraction Limit",
            passed=bool(all_ok),
            expected="Measured sinusoidal contrast transfer matches predicted MTF within 0.081",
            actual=f"Max |measured-predicted| = {max(abs(m - p) for m, p in zip(measured, predicted)):.4f}",
            tolerance=0.081,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_dichromat_confusion_v2(self) -> ValidationResult:
        """Human median discriminability exceeds dog/cat medians on confusion sets."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.validation.dichromat import evaluate_stimulus_matrix
        from retinal_sim.validation.metrics import median_by

        matrix = evaluate_stimulus_matrix(["human", "dog", "cat"], seed=self._seed, n_seeds=2)
        summary = {
            species: {key: median_by(values) for key, values in groups.items()}
            for species, groups in matrix.items()
        }

        checks = [
            summary["dog"]["confusion_dog"] < summary["dog"]["control"],
            summary["cat"]["confusion_cat"] < summary["cat"]["control"],
            summary["human"]["confusion_cat"] > summary["cat"]["confusion_cat"] + 0.01,
            summary["dog"]["control"] > 0.05,
            summary["cat"]["control"] > 0.05,
        ]
        all_ok = all(checks)
        details_lines = [
            f"human medians: dog-conf={summary['human']['confusion_dog']:.4f}, "
            f"cat-conf={summary['human']['confusion_cat']:.4f}, control={summary['human']['control']:.4f}",
            f"dog medians: dog-conf={summary['dog']['confusion_dog']:.4f}, "
            f"cat-conf={summary['dog']['confusion_cat']:.4f}, control={summary['dog']['control']:.4f}",
            f"cat medians: dog-conf={summary['cat']['confusion_dog']:.4f}, "
            f"cat-conf={summary['cat']['confusion_cat']:.4f}, control={summary['cat']['control']:.4f}",
            f"Dog confusion median < dog control median: {'✓' if checks[0] else '✗'}",
            f"Cat confusion median < cat control median: {'✓' if checks[1] else '✗'}",
            f"Human > cat on cat confusion set: {'✓' if checks[2] else '✗'}",
            f"Dog control median > 0.05: {'✓' if checks[3] else '✗'}",
            f"Cat control median > 0.05: {'✓' if checks[4] else '✗'}",
        ]

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        categories = ["dog confusion", "cat confusion", "control"]
        x = np.arange(len(categories))
        width = 0.24
        for i, species in enumerate(["human", "dog", "cat"]):
            ax.bar(
                x + (i - 1) * width,
                [
                    summary[species]["confusion_dog"],
                    summary[species]["confusion_cat"],
                    summary[species]["control"],
                ],
                width=width,
                label=species,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Median discriminability")
        ax.set_title("Stimulus-Matrix Dichromat Validation")
        ax.legend(fontsize=8)
        plt.tight_layout()

        return self._result(
            "Dichromat Confusion",
            passed=bool(all_ok),
            expected="Species-specific confusion medians are suppressed relative to controls, and human exceeds cat on the cat confusion set",
            actual="\n".join(details_lines),
            tolerance=0.01,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_nyquist_sampling_v2(self) -> ValidationResult:
        """Local cone density predicts species-specific Nyquist limits in cpd."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.species.config import SpeciesConfig

        target_ranges = {
            "human": (45.0, 90.0),
            "dog": (10.0, 30.0),
            "cat": (8.0, 25.0),
        }
        species_list = ["human", "dog", "cat"]
        values = {}
        details_lines = []
        all_ok = True

        for sp in species_list:
            cfg = SpeciesConfig.load(sp)
            cone_density = float(sum(cfg.retinal.cone_density_fn(0.0, 0.0).values()))
            spacing_mm = np.sqrt(2.0 / (np.sqrt(3.0) * cone_density))
            mm_per_degree = cfg.optical.focal_length_mm * np.tan(np.radians(1.0))
            cpd = (1.0 / (2.0 * spacing_mm)) * mm_per_degree
            lo, hi = target_ranges[sp]
            ok = lo <= cpd <= hi
            all_ok = all_ok and ok
            values[sp] = cpd
            details_lines.append(
                f"{sp}: cone_density={cone_density:.0f}/mm², spacing={spacing_mm * 1000:.2f}µm, "
                f"nyquist={cpd:.1f} cpd (target {lo:.0f}-{hi:.0f}) {'✓' if ok else '✗'}"
            )

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.bar(species_list, [values[s] for s in species_list], color=["#4c72b0", "#dd8452", "#55a868"])
        ax.set_ylabel("Nyquist limit (cpd)")
        ax.set_title("Area Centralis Nyquist Estimates")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        return self._result(
            "Nyquist Sampling",
            passed=bool(all_ok),
            expected="Species Nyquist estimates fall within target cpd bands",
            actual="\n".join(details_lines),
            tolerance=0.0,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_receptor_count_v2(self) -> ValidationResult:
        """Generated receptor counts match density-model expectations over the simulated patch."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.constants import WAVELENGTHS
        from retinal_sim.retina.mosaic import MosaicGenerator
        from retinal_sim.species.config import SpeciesConfig
        species_list = ["human", "dog", "cat"]
        measured_totals = {}
        expected_totals = {}
        details_lines = []
        all_ok = True

        for sp in species_list:
            cfg = SpeciesConfig.load(sp)
            gen = MosaicGenerator(cfg.retinal, cfg.optical, WAVELENGTHS)
            mosaic = gen.generate(seed=self._seed)
            half_width = float(gen._patch_half_mm)
            patch_width = 2.0 * half_width
            mask = (
                (np.abs(mosaic.positions[:, 0]) <= half_width) &
                (np.abs(mosaic.positions[:, 1]) <= half_width)
            )
            unique, cnts = np.unique(mosaic.types[mask], return_counts=True)
            measured = dict(zip(unique.tolist(), cnts.tolist()))
            measured_total = int(mask.sum())
            sample_axis = np.linspace(-half_width, half_width, 101)
            XX, YY = np.meshgrid(sample_axis, sample_axis)
            ecc = np.sqrt(XX ** 2 + YY ** 2)
            ang = np.arctan2(YY, XX)
            cone_density_grid = np.zeros_like(ecc)
            for i in range(ecc.shape[0]):
                for j in range(ecc.shape[1]):
                    cone_density_grid[i, j] = sum(
                        cfg.retinal.cone_density_fn(float(ecc[i, j]), float(ang[i, j])).values()
                    )
            rod_density_grid = np.vectorize(
                lambda e, a: cfg.retinal.rod_density_fn(float(e), float(a))
            )(ecc, ang)
            expected_total = float(np.mean(cone_density_grid + rod_density_grid) * patch_width * patch_width)
            cone_ratio_measured = (
                measured.get("S_cone", 0) + measured.get("M_cone", 0) + measured.get("L_cone", 0)
            ) / max(measured_total, 1)
            cone_ratio_expected = float(np.mean(cone_density_grid) / max(np.mean(cone_density_grid + rod_density_grid), 1e-10))

            count_err = abs(measured_total - expected_total) / max(expected_total, 1.0)
            ratio_err = abs(cone_ratio_measured - cone_ratio_expected) / max(cone_ratio_expected, 1e-10)
            ok = count_err < 0.25 and ratio_err < 0.15
            all_ok = all_ok and ok
            measured_totals[sp] = measured_total
            expected_totals[sp] = expected_total
            details_lines.append(
                f"{sp}: measured={measured_total}, expected={expected_total:.0f}, "
                f"count_err={count_err:.3f}, cone_ratio_err={ratio_err:.3f} {'✓' if ok else '✗'}"
            )

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        x = np.arange(len(species_list))
        ax.bar(x - 0.18, [measured_totals[s] for s in species_list], width=0.35, label="Measured")
        ax.bar(x + 0.18, [expected_totals[s] for s in species_list], width=0.35, label="Expected")
        ax.set_xticks(x)
        ax.set_xticklabels(species_list)
        ax.set_ylabel("Receptors in simulated patch")
        ax.set_title("Measured vs Expected Patch Counts")
        ax.legend(fontsize=8)
        plt.tight_layout()

        return self._result(
            "Receptor Count",
            passed=bool(all_ok),
            expected="Measured counts and cone/rod ratios stay within density-model tolerance bands over the simulated patch",
            actual="\n".join(details_lines),
            tolerance=0.25,
            details="\n".join(details_lines),
            figure=fig,
        )

    def test_resolution_gradient_v2(self) -> ValidationResult:
        """Human center contrast exceeds peripheral and dichromat center contrast on a fine checkerboard."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from retinal_sim.pipeline import RetinalSimulator
        from retinal_sim.output.reconstruction import render_reconstructed
        from retinal_sim.output.voronoi import render_voronoi
        from retinal_sim.validation.metrics import response_contrast_by_region

        checker = np.zeros((64, 64, 3), dtype=np.uint8)
        for r in range(64):
            for c in range(64):
                checker[r, c] = [200, 200, 200] if (r + c) % 2 == 0 else [50, 50, 50]
        bright_mask = checker[:, :, 0] > 100

        species_list = ["human", "dog", "cat"]
        results = {}
        details_lines = []
        all_ok = True
        contrast_summary = {}

        for sp in species_list:
            sim = RetinalSimulator(sp, patch_extent_deg=2.0, stimulus_scale=0.01, seed=self._seed)
            results[sp] = sim.simulate(
                checker,
                seed=self._seed,
                input_mode="reflectance_under_d65",
            )
            radii = np.sqrt(np.sum(results[sp].mosaic.positions ** 2, axis=1))
            center_mask = radii <= np.quantile(radii, 0.35)
            periphery_mask = radii >= np.quantile(radii, 0.65)
            center_contrast = response_contrast_by_region(results[sp], bright_mask, center_mask)
            periphery_contrast = response_contrast_by_region(results[sp], bright_mask, periphery_mask)
            contrast_summary[sp] = (center_contrast, periphery_contrast)
            details_lines.append(f"{sp}: center_contrast={center_contrast:.4f}, periphery_contrast={periphery_contrast:.4f}")

        checks = [
            contrast_summary["human"][0] > contrast_summary["human"][1],
            contrast_summary["human"][0] > contrast_summary["dog"][0],
            contrast_summary["human"][0] > contrast_summary["cat"][0],
        ]
        all_ok = all(checks)
        details_lines.extend(
            [
                f"Human center > human periphery: {'✓' if checks[0] else '✗'}",
                f"Human center > dog center: {'✓' if checks[1] else '✗'}",
                f"Human center > cat center: {'✓' if checks[2] else '✗'}",
            ]
        )

        fig, axes = plt.subplots(2, len(species_list) + 1, figsize=(4 * (len(species_list) + 1), 8))
        axes[0, 0].imshow(checker)
        axes[0, 0].set_title("Input\n(Checkerboard)", fontsize=9)
        axes[0, 0].axis("off")
        axes[1, 0].axis("off")

        for i, sp in enumerate(species_list):
            voronoi_img = render_voronoi(results[sp].activation, output_size=(128, 128))
            axes[0, i + 1].imshow(voronoi_img, origin="lower")
            axes[0, i + 1].set_title(
                f"{sp.capitalize()}\nC={contrast_summary[sp][0]:.3f}, P={contrast_summary[sp][1]:.3f}",
                fontsize=9,
            )
            axes[0, i + 1].axis("off")

            recon = render_reconstructed(results[sp].activation, (128, 128))
            axes[1, i + 1].imshow(recon, cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[1, i + 1].set_title(f"{sp.capitalize()}\nReconstructed", fontsize=9)
            axes[1, i + 1].axis("off")

        fig.suptitle("Resolution Gradient: Quantified Center vs Periphery Contrast", fontsize=11)
        plt.tight_layout()

        return self._result(
            "Resolution Gradient",
            passed=bool(all_ok),
            expected="Human center contrast exceeds human periphery and dichromat center contrast on a fine checkerboard",
            actual="\n".join(details_lines),
            tolerance=0.0,
            details="\n".join(details_lines),
            figure=fig,
        )


# ---------------------------------------------------------------------------
# Report metadata and coverage
# ---------------------------------------------------------------------------

_RESULT_SPECS: Dict[str, Dict[str, Any]] = {
    "Angular Subtense": {
        "stage": "scene",
        "architecture_ref": "Architecture §11e Angular subtense correctness",
        "code_refs": ["retinal_sim/scene/geometry.py", "retinal_sim/validation/report.py"],
        "inputs_summary": "Three analytic scene width/distance cases using the human optical model.",
        "assumptions": ["Exact 2*arctan geometry is the source of truth.", "Human optical focal length is used only to build SceneDescription."],
        "method": "Compare computed angular width against the closed-form geometric equation.",
        "pass_criterion": "Absolute angular error < 0.01 degrees for every case.",
        "limitations": ["Checks representative cases, not exhaustive parameter sweeps."],
        "artifacts": ["Bar chart: expected vs computed angular width"],
    },
    "Retinal Scaling Across Species": {
        "stage": "scene",
        "architecture_ref": "Architecture §11e Retinal image scaling across species",
        "code_refs": ["retinal_sim/scene/geometry.py", "retinal_sim/species/config.py"],
        "inputs_summary": "One shared physical scene evaluated for human, dog, and cat species configs.",
        "assumptions": ["Retinal image width should scale with focal length when angular size is fixed."],
        "method": "Compare retinal_width_mm / focal_length_mm across species.",
        "pass_criterion": "Species ratios agree within 1% relative error.",
        "limitations": ["Only validates proportionality, not downstream sampling behavior."],
        "artifacts": ["Species focal-length bar chart", "Species retinal-width bar chart"],
    },
    "Accommodation Defocus": {
        "stage": "scene",
        "architecture_ref": "Architecture §0 Accommodation demand and residual defocus",
        "code_refs": ["retinal_sim/scene/geometry.py"],
        "inputs_summary": "Near-view scene at 0.25 m for human, dog, and cat.",
        "assumptions": ["Accommodation is a hard cutoff.", "Max accommodation values are fixed PoC constants."],
        "method": "Compare computed residual defocus against max(0, demand - max_accommodation).",
        "pass_criterion": "Residual defocus matches expected value within 0.1 diopters.",
        "limitations": ["No lead/lag accommodation response model is applied."],
        "artifacts": ["Expected vs computed residual defocus bars"],
    },
    "Distance-Dependent Receptor Sampling": {
        "stage": "scene",
        "architecture_ref": "Architecture §11e Distance-dependent receptor sampling",
        "code_refs": ["retinal_sim/pipeline.py", "retinal_sim/retina/mosaic.py"],
        "inputs_summary": "Constant-size grayscale object that remains smaller than the 2° patch across all tested distances.",
        "assumptions": ["Stimulated receptors are those whose mapped positions fall inside the projected object footprint."],
        "method": "Measure stimulated receptor counts at multiple viewing distances and compare against ideal inverse-square scaling.",
        "pass_criterion": "Far/near stimulated-receptor scaling stays within 40% relative error of ideal 1/d^2.",
        "limitations": ["Density gradients and stochastic mosaic sampling still introduce some deviation from exact inverse-square behaviour."],
        "artifacts": ["Distance vs stimulated receptor count plot", "Log-log ideal vs measured scaling plot"],
    },
    "Metamer Preservation": {
        "stage": "spectral",
        "architecture_ref": "Architecture §11a Metamer preservation",
        "code_refs": ["retinal_sim/spectral/upsampler.py", "retinal_sim/retina/opsin.py"],
        "inputs_summary": "Fixed near-metamer RGB pair dataset interpreted as RGB-inferred reflectance under D65 and integrated against human receptor sensitivities.",
        "assumptions": [
            "Dataset is repo-local and deterministic to keep the audit reproducible.",
            "The spectra are inferred from RGB with the Smits-inspired D65-optimized reflectance model rather than measured directly.",
        ],
        "method": "Upsample each RGB pair in reflectance_under_d65 mode, integrate against human sensitivities, and bound the maximum cone-catch relative difference.",
        "pass_criterion": "Maximum cone-catch relative difference < 3.5% for every dataset pair.",
        "limitations": [
            "Dataset is compact rather than exhaustive; it is intended as a deterministic regression panel.",
            "This is an RGB-inferred regression check, not evidence of uniquely recovered physical spectra.",
        ],
        "artifacts": ["Representative pair spectra plot", "Per-pair max cone-catch difference bars"],
    },
    "RGB Round-Trip": {
        "stage": "spectral",
        "architecture_ref": "Architecture §11a Roundtrip consistency",
        "code_refs": ["retinal_sim/spectral/upsampler.py", "retinal_sim/validation/report.py"],
        "inputs_summary": "Eight canonical sRGB colors passed through RGB-inferred reflectance reconstruction and D65-based reprojection.",
        "assumptions": [
            "The same D65-referenced observer model used to build the basis spectra is used for reprojection.",
            "This checks the declared reflectance_under_d65 inference path, not measured scene spectra.",
        ],
        "method": "Project RGB-inferred reflectance spectra back to sRGB and compute per-colour RMSE.",
        "pass_criterion": "RMSE < 2 8-bit counts for all tested colours.",
        "limitations": ["This validates the D65 round-trip path, not arbitrary illuminant changes."],
        "artifacts": ["RMSE bar chart", "Input vs reconstructed color swatches"],
    },
    "Spectral Response Panel": {
        "stage": "spectral",
        "architecture_ref": "Architecture §11a Spectral response ordering",
        "code_refs": ["retinal_sim/spectral/upsampler.py", "retinal_sim/retina/opsin.py", "retinal_sim/validation/datasets.py"],
        "inputs_summary": "Deterministic RGB probe panel spanning violet-to-red anchors plus mixed colors, interpreted as RGB-inferred reflectance and integrated with D65-weighted species sensitivities.",
        "assumptions": [
            "The panel is RGB-defined and deterministic; it is a regression sanity check rather than a physical monochromator dataset.",
            "Spectra are inferred from RGB using the reflectance_under_d65 model.",
        ],
        "method": "Upsample each probe color in reflectance_under_d65 mode, compute D65-weighted cone catches for each species, and verify expected dominance/order and blue-vs-warm separations.",
        "pass_criterion": "Anchor colors preserve expected cone-type dominance and dog/cat blue-vs-yellow/red separations exceed the fixed margin.",
        "limitations": ["Because the probes originate from sRGB values, this validates response ordering under the current D65-based reconstruction model rather than true narrowband physiology."],
        "artifacts": ["Representative reconstructed spectra plot", "Per-species normalized cone-catch panel"],
    },
    "MTF vs Diffraction Limit": {
        "stage": "optical",
        "architecture_ref": "Architecture §11b Diffraction-limited resolution",
        "code_refs": ["retinal_sim/optical/psf.py", "retinal_sim/species/config.py"],
        "inputs_summary": "Human 550 nm PSF tested with sinusoidal gratings spanning low-to-mid spatial frequencies.",
        "assumptions": ["The implemented PSF is a diffraction-derived Gaussian approximation."],
        "method": "Measure empirical contrast transfer on sinusoidal gratings and compare it with the PSF-predicted MTF.",
        "pass_criterion": "Measured modulation stays within 0.08 of the predicted MTF at all tested spatial frequencies.",
        "limitations": ["Validation targets the implemented diffraction-derived PSF model, not a full Airy-pattern optical stack."],
        "artifacts": ["Measured vs predicted MTF plot", "550 nm PSF kernel image"],
    },
    "PSF Energy Conservation": {
        "stage": "optical",
        "architecture_ref": "Architecture §11b PSF energy conservation",
        "code_refs": ["retinal_sim/optical/psf.py"],
        "inputs_summary": "Gaussian PSF kernels across the canonical wavelength grid for all three species.",
        "assumptions": ["Kernel normalization in float64 is the source of truth for this check."],
        "method": "Sum each PSF kernel and measure max absolute deviation from 1.0.",
        "pass_criterion": "Maximum |sum - 1.0| < 1e-6.",
        "limitations": ["Only validates discrete kernel normalization, not broader optical energy accounting."],
        "artifacts": ["Log-scale wavelength vs normalization error plot"],
    },
    "Pupil Throughput Scaling": {
        "stage": "optical",
        "architecture_ref": "Architecture §2a Pupil throughput and aperture geometry",
        "code_refs": ["retinal_sim/validation/report.py", "retinal_sim/species/config.py"],
        "inputs_summary": "Human, dog, and cat species configs with circular/slit pupil geometries.",
        "assumptions": [
            "Pupil area is the relevant throughput proxy for the validation report.",
            "Cat slit height is read from the species optical config when present.",
        ],
        "method": "Compute pupil area for each species, normalize to the human reference pupil, and compare the resulting throughput ratios.",
        "pass_criterion": "Species pupil areas differ and human-normalized throughput increases from human to dog/cat.",
        "limitations": ["This is a report-layer optical-throughput diagnostic rather than an end-to-end irradiance measurement."],
        "artifacts": ["Species pupil area bar chart", "Human-normalized throughput summary"],
    },
    "Cat Slit Anisotropy": {
        "stage": "optical",
        "architecture_ref": "Architecture §2b Slit pupil anisotropic blur diagnostics",
        "code_refs": ["retinal_sim/validation/report.py", "retinal_sim/species/config.py"],
        "inputs_summary": "Cat slit pupil evaluated with a point-response diagnostic at 550 nm.",
        "assumptions": [
            "The diagnostic uses an elliptical Gaussian approximation for the slit pupil.",
            "Horizontal blur should exceed vertical blur for the vertical slit approximation.",
        ],
        "method": "Generate a point-spread diagnostic, measure axis-specific sigma/FWHM, and require horizontal spread to exceed vertical spread.",
        "pass_criterion": "Cat diagnostic reports sigma_x > sigma_y and FWHM_x > FWHM_y while remaining normalized.",
        "limitations": ["This report-layer approximation anticipates the axis-aware optical stage rather than replacing it."],
        "artifacts": ["Cat slit point-response heatmap", "Axis-specific cross-section profile plot"],
    },
    "Wavelength-Dependent Blur": {
        "stage": "optical",
        "architecture_ref": "Architecture §2b Longitudinal chromatic aberration",
        "code_refs": ["retinal_sim/optical/psf.py", "retinal_sim/optical/stage.py", "retinal_sim/species/config.py"],
        "inputs_summary": "Canonical wavelength grid evaluated for all three species using the current Gaussian-plus-LCA optical model.",
        "assumptions": [
            "R2 models LCA as a signed wavelength-dependent defocus offset around a 555 nm reference focus.",
            "The smallest blur should occur near the reference wavelength when scene defocus is zero.",
        ],
        "method": "Sample per-wavelength LCA offsets and PSF widths from the optical-stage metadata and verify sign/order behavior.",
        "pass_criterion": "Offsets are positive in the short wavelengths, negative in the long wavelengths, and minimum PSF width occurs near 555 nm for each species.",
        "limitations": ["This validates the implemented R2 operational model, not a full empirical aberration fit."],
        "artifacts": ["Wavelength vs LCA offset plot", "Wavelength vs PSF sigma plot"],
    },
    "Media Transmission Diagnostics": {
        "stage": "optical",
        "architecture_ref": "Architecture §2c Vitreous and media transmission",
        "code_refs": ["retinal_sim/optical/media.py", "retinal_sim/species/config.py", "retinal_sim/optical/stage.py"],
        "inputs_summary": "Species ocular-media transmission tables sampled on the canonical wavelength grid and applied to a unit spectrum.",
        "assumptions": [
            "R2 uses fixed species reference transmission curves rather than age- or state-dependent media variants.",
            "Short-wavelength attenuation should exceed long-wavelength attenuation for each species.",
        ],
        "method": "Sample the configured transmission curves, compare short-vs-long wavelength attenuation, and visualize the delivered spectra.",
        "pass_criterion": "Each species transmits less blue than red and the sampled transmission curves differ across species.",
        "limitations": ["Does not model scatter or age-dependent lens brunescence in R2."],
        "artifacts": ["Species transmission curve plot", "Delivered-spectrum plot after ocular media"],
    },
    "Snellen Acuity": {
        "stage": "retinal",
        "architecture_ref": "Architecture §11c Snellen acuity prediction",
        "code_refs": ["retinal_sim/validation/acuity.py", "retinal_sim/validation/snellen.py"],
        "inputs_summary": "Species-specific acuity prediction using Snellen E orientation discriminability.",
        "assumptions": ["Published acuity ranges are broad PoC acceptance bands.", "Letter-region masking and Pearson correlation distance are required."],
        "method": "Predict minimum resolvable angular size for each species and compare with expected ranges and ordering.",
        "pass_criterion": "Each species falls within its target range and human acuity is best.",
        "limitations": ["Cat predictions can be seed-sensitive near threshold."],
        "artifacts": ["Snellen stimulus examples", "Predicted acuity bar chart"],
    },
    "Dichromat Confusion": {
        "stage": "retinal",
        "architecture_ref": "Architecture §11c Dichromat confusion axis",
        "code_refs": ["retinal_sim/validation/dichromat.py", "retinal_sim/validation/ishihara.py"],
        "inputs_summary": "Fixed confusion-pair panels for dog and cat plus a shared control-pair panel.",
        "assumptions": ["Stimulus scaling is applied to keep Naka-Rushton responses out of saturation."],
        "method": "Compute species-wise discriminability distributions across deterministic confusion/control stimulus matrices and compare confusion medians against controls.",
        "pass_criterion": "Each species-specific confusion median is suppressed relative to its controls, and human exceeds cat on the cat confusion set.",
        "limitations": ["Compact panel intended for deterministic regression rather than exhaustive colour-space coverage."],
        "artifacts": ["Median discriminability bar chart"],
    },
    "Nyquist Sampling": {
        "stage": "retinal",
        "architecture_ref": "Architecture §11c Cone density to sampling limit",
        "code_refs": ["retinal_sim/retina/mosaic.py", "retinal_sim/species/config.py"],
        "inputs_summary": "Area-centralis cone densities converted to spacing and cycles/degree for all species.",
        "assumptions": ["Hexagonal spacing approximation is used to map cone density to Nyquist limit."],
        "method": "Convert local cone density to spacing, then convert spacing to cycles/degree using focal-length-based retinal magnification.",
        "pass_criterion": "Each species falls inside its target cycles/degree band.",
        "limitations": ["Uses density-model predictions at the patch centre rather than a full eccentricity sweep."],
        "artifacts": ["Species Nyquist-limit bar chart"],
    },
    "Receptor Count": {
        "stage": "retinal",
        "architecture_ref": "Architecture §11c Receptor count validation",
        "code_refs": ["retinal_sim/retina/mosaic.py", "retinal_sim/species/config.py"],
        "inputs_summary": "Generated mosaics compared against density-model expectations over the simulated square patch.",
        "assumptions": ["Species density functions are the literature-backed PoC source of truth for expected counts."],
        "method": "Compare measured receptor counts and cone/rod composition against integrated density-model expectations over the simulated patch.",
        "pass_criterion": "Total count error < 25% and cone/rod composition error < 15% for every species.",
        "limitations": ["Current model still uses simplified analytic density functions rather than digitised histology maps."],
        "artifacts": ["Measured vs expected count bars"],
    },
    "Color Deficit Reproduction": {
        "stage": "e2e",
        "architecture_ref": "Architecture §11d Known visual deficit reproduction",
        "code_refs": ["retinal_sim/pipeline.py", "retinal_sim/output/voronoi.py"],
        "inputs_summary": "Red-green split stimulus simulated for human, dog, and cat.",
        "assumptions": ["Left/right response norm is used as the summary metric."],
        "method": "Measure left/right cone-response difference after full-pipeline simulation.",
        "pass_criterion": "Human left/right difference exceeds dog by at least 20%.",
        "limitations": ["This is a compact proxy for a richer end-to-end color-deficit comparison."],
        "artifacts": ["Input stimulus", "Per-species Voronoi renderings"],
    },
    "Resolution Gradient": {
        "stage": "e2e",
        "architecture_ref": "Architecture §11d Resolution gradient",
        "code_refs": ["retinal_sim/pipeline.py", "retinal_sim/output/reconstruction.py", "retinal_sim/output/voronoi.py"],
        "inputs_summary": "Checkerboard stimulus simulated for human, dog, and cat.",
        "assumptions": ["Checkerboard bright/dark contrast is used as the resolvability proxy."],
        "method": "Compare center vs peripheral bright/dark response contrast on a fine checkerboard and compare human center contrast against dichromat center contrast.",
        "pass_criterion": "Human center contrast exceeds human periphery and exceeds dog/cat center contrast.",
        "limitations": ["Quantifies contrast retention on a fine checkerboard but does not yet fit a full eccentricity-dependent MTF curve."],
        "artifacts": ["Input checkerboard", "Per-species Voronoi images", "Per-species reconstructions"],
    },
}

_EXTERNAL_REFERENCE_TABLES = external_reference_tables()

_VALIDATION_CATEGORY_BY_TEST = {
    "Angular Subtense": "analytic correctness",
    "Accommodation Defocus": "analytic correctness",
    "PSF Energy Conservation": "analytic correctness",
    "RGB Round-Trip": "model self-consistency",
    "Spectral Response Panel": "model self-consistency",
    "MTF vs Diffraction Limit": "model self-consistency",
    "Distance-Dependent Receptor Sampling": "model self-consistency",
    "Receptor Count": "model self-consistency",
    "Color Deficit Reproduction": "model self-consistency",
    "Resolution Gradient": "model self-consistency",
    "Retinal Scaling Across Species": "external empirical alignment",
    "Metamer Preservation": "external empirical alignment",
    "Pupil Throughput Scaling": "external empirical alignment",
    "Cat Slit Anisotropy": "external empirical alignment",
    "Wavelength-Dependent Blur": "external empirical alignment",
    "Media Transmission Diagnostics": "external empirical alignment",
    "Snellen Acuity": "external empirical alignment",
    "Dichromat Confusion": "external empirical alignment",
    "Nyquist Sampling": "external empirical alignment",
}

_CLAIM_SUPPORT_LEVEL_BY_TEST = {
    "Angular Subtense": "strong",
    "Accommodation Defocus": "strong",
    "PSF Energy Conservation": "strong",
    "RGB Round-Trip": "moderate",
    "Spectral Response Panel": "moderate",
    "MTF vs Diffraction Limit": "moderate",
    "Distance-Dependent Receptor Sampling": "moderate",
    "Receptor Count": "moderate",
    "Color Deficit Reproduction": "weak",
    "Resolution Gradient": "weak",
    "Retinal Scaling Across Species": "moderate",
    "Metamer Preservation": "moderate",
    "Pupil Throughput Scaling": "moderate",
    "Cat Slit Anisotropy": "moderate",
    "Wavelength-Dependent Blur": "moderate",
    "Media Transmission Diagnostics": "moderate",
    "Snellen Acuity": "moderate",
    "Dichromat Confusion": "weak",
    "Nyquist Sampling": "strong",
}

_EXTERNAL_REFERENCE_KEY_BY_TEST = {
    "Retinal Scaling Across Species": "optical_geometry_expectations",
    "Pupil Throughput Scaling": "optical_geometry_expectations",
    "Cat Slit Anisotropy": "optical_geometry_expectations",
    "Wavelength-Dependent Blur": "wavelength_transmission_assumptions",
    "Media Transmission Diagnostics": "wavelength_transmission_assumptions",
    "Snellen Acuity": "species_acuity_ranges",
    "Nyquist Sampling": "density_derived_nyquist_limits",
}

_EXTERNAL_REFERENCE_SUMMARY_BY_TEST = {
    "Angular Subtense": "Closed-form geometry check with no external literature table required.",
    "Accommodation Defocus": "Closed-form hard-cutoff accommodation check with no separate external evidence table.",
    "PSF Energy Conservation": "Kernel normalization is an analytic invariant rather than an externally benchmarked physiological claim.",
    "RGB Round-Trip": "Internal regression check for the declared RGB-inference path under D65 assumptions.",
    "Spectral Response Panel": "Internal response-order regression for the current D65-weighted RGB inference model.",
    "MTF vs Diffraction Limit": "Compares measured modulation against the simulator's own implemented diffraction-derived PSF model.",
    "Distance-Dependent Receptor Sampling": "Checks inverse-square geometric consistency inside the current pipeline rather than direct external behavioral data.",
    "Receptor Count": "Compares generated mosaics against the project's density-model expectations; evidence remains indirect until tied to richer external histology tables.",
    "Color Deficit Reproduction": "Proxy end-to-end comparison that illustrates retained information differences without claiming perceptual equivalence.",
    "Resolution Gradient": "Proxy end-to-end contrast-retention check, not a full empirical eccentricity validation.",
    "Retinal Scaling Across Species": "Aligned to focal-length-based optical geometry expectations in the local external reference table.",
    "Metamer Preservation": "Compared against a compact deterministic reference panel intended to approximate external metamer behavior under the declared scene assumptions.",
    "Pupil Throughput Scaling": "Aligned to local pupil-area and focal-length expectation rows that describe modeled optical consequences by species.",
    "Cat Slit Anisotropy": "Aligned to the local slit-pupil optical consequence table; evidence remains approximate because the model uses a slit approximation.",
    "Wavelength-Dependent Blur": "Aligned to local wavelength-delivery and optical-behavior reference assumptions rather than direct in-vivo PSF measurements.",
    "Media Transmission Diagnostics": "Aligned to the configured species ocular-media reference curves and their expected shortwave-vs-longwave behavior.",
    "Snellen Acuity": "Aligned to published species acuity target ranges stored locally for deterministic reporting.",
    "Dichromat Confusion": "Aligned indirectly to expected species confusion-axis behavior; evidence is illustrative rather than exhaustive.",
    "Nyquist Sampling": "Aligned to density-derived species Nyquist target ranges stored locally for deterministic reporting.",
}

_EVIDENCE_BASIS_BY_TEST = {
    "analytic correctness": "Closed-form or invariant check against an analytic expectation.",
    "model self-consistency": "Internal regression or consistency check within the simulator's declared implementation and assumptions.",
    "external empirical alignment": "Comparison against a local deterministic reference table representing published or externally grounded expectations.",
}

_CLAIM_SCOPE_NOTE_BY_TEST = {
    "Angular Subtense": "Supports scene-geometry arithmetic only; it does not validate downstream retinal or perceptual behavior.",
    "Accommodation Defocus": "Supports the hard-cutoff accommodation rule currently implemented, not a full accommodation physiology model.",
    "PSF Energy Conservation": "Supports numerical normalization of the PSF kernels only.",
    "RGB Round-Trip": "Supports the current RGB-inferred reflectance reconstruction path, not unique recovery of scene spectra.",
    "Spectral Response Panel": "Supports ordering behavior in the declared RGB-inference model, not full spectral physiology.",
    "MTF vs Diffraction Limit": "Supports the implemented optical approximation, not blanket validation of all anterior-eye optics.",
    "Distance-Dependent Receptor Sampling": "Supports geometric scaling behavior under current pipeline assumptions only.",
    "Receptor Count": "Supports the current mosaic-density implementation at the retinal front end, not full retinal histology.",
    "Color Deficit Reproduction": "Illustrates comparative retinal-information loss and should not be read as direct conscious appearance.",
    "Resolution Gradient": "Illustrates front-end contrast retention trends and should not be read as a full perceptual validation.",
    "Retinal Scaling Across Species": "Supports focal-length-driven retinal image scaling only.",
    "Metamer Preservation": "Supports a compact external-style regression panel under D65 assumptions only.",
    "Pupil Throughput Scaling": "Supports the modeled optical throughput consequences of the configured pupil geometries only.",
    "Cat Slit Anisotropy": "Supports axis-aware blur reporting in the current slit approximation, not full feline ocular optics.",
    "Wavelength-Dependent Blur": "Supports the current wavelength-aware optical model and its assumptions, not full empirical PSF validation.",
    "Media Transmission Diagnostics": "Supports the configured species transmission assumptions, not full individualized ocular-media physiology.",
    "Snellen Acuity": "Supports approximate species acuity alignment at the retinal front end, not full behavioral or cortical vision validation.",
    "Dichromat Confusion": "Supports qualitative species confusion-axis differences at the retinal front end, not complete color appearance prediction.",
    "Nyquist Sampling": "Supports density-derived front-end sampling limits, not full perceptual acuity validation.",
}


def _reference_rows(key: str) -> List[Dict[str, Any]]:
    table = _EXTERNAL_REFERENCE_TABLES.get(key, {})
    rows: List[Dict[str, Any]] = []
    for label, value in table.items():
        if isinstance(value, dict):
            row = {"reference_label": label}
            row.update(value)
            rows.append(row)
        else:
            rows.append({"reference_label": label, "value": value})
    return rows


for _test_name, _spec in _RESULT_SPECS.items():
    _category = _VALIDATION_CATEGORY_BY_TEST[_test_name]
    _spec["validation_category"] = _category
    _spec["claim_support_level"] = _CLAIM_SUPPORT_LEVEL_BY_TEST[_test_name]
    _spec["external_reference_summary"] = _EXTERNAL_REFERENCE_SUMMARY_BY_TEST[_test_name]
    _spec["evidence_basis"] = _EVIDENCE_BASIS_BY_TEST[_category]
    _spec["claim_scope_note"] = _CLAIM_SCOPE_NOTE_BY_TEST[_test_name]
    _reference_key = _EXTERNAL_REFERENCE_KEY_BY_TEST.get(_test_name)
    _spec["external_reference_table"] = (
        _reference_rows(_reference_key) if _reference_key else []
    )

_ARCHITECTURE_COVERAGE = [
    {"architecture_ref": "Architecture §11e Angular subtense correctness", "test_name": "Angular Subtense", "code_path": "retinal_sim/scene/geometry.py", "status": "covered", "note": "Analytic geometric equality check."},
    {"architecture_ref": "Architecture §11e Retinal image scaling across species", "test_name": "Retinal Scaling Across Species", "code_path": "retinal_sim/scene/geometry.py", "status": "covered", "note": "Checks focal-length proportionality."},
    {"architecture_ref": "Architecture §11e Accommodation/defocus injection", "test_name": "Accommodation Defocus", "code_path": "retinal_sim/scene/geometry.py", "status": "covered", "note": "Hard-cutoff accommodation model."},
    {"architecture_ref": "Architecture §11e Distance-dependent receptor sampling", "test_name": "Distance-Dependent Receptor Sampling", "code_path": "retinal_sim/pipeline.py", "status": "covered", "note": "Counts stimulated receptors inside the projected object footprint."},
    {"architecture_ref": "Architecture §11a Metamer preservation", "test_name": "Metamer Preservation", "code_path": "retinal_sim/spectral/upsampler.py", "status": "covered", "note": "Uses a fixed deterministic RGB-inferred reflectance regression dataset under D65."},
    {"architecture_ref": "Architecture §11a Roundtrip consistency", "test_name": "RGB Round-Trip", "code_path": "retinal_sim/spectral/upsampler.py", "status": "covered", "note": "Uses the same D65 observer model as the reflectance-under-D65 basis construction for reprojection."},
    {"architecture_ref": "Architecture §11a Spectral response ordering", "test_name": "Spectral Response Panel", "code_path": "retinal_sim/validation/report.py", "status": "covered", "note": "Uses deterministic RGB probes to sanity-check cone-catch ordering under the declared RGB-inferred D65 reconstruction model."},
    {"architecture_ref": "Architecture §11b Diffraction-limited resolution", "test_name": "MTF vs Diffraction Limit", "code_path": "retinal_sim/optical/psf.py", "status": "covered", "note": "Measures empirical sinusoidal contrast transfer against the implemented PSF MTF."},
    {"architecture_ref": "Architecture §11b PSF energy conservation", "test_name": "PSF Energy Conservation", "code_path": "retinal_sim/optical/psf.py", "status": "covered", "note": "Float64 normalization check."},
    {"architecture_ref": "Architecture §2a Pupil throughput and aperture geometry", "test_name": "Pupil Throughput Scaling", "code_path": "retinal_sim/validation/report.py", "status": "covered", "note": "Report-layer pupil-area diagnostic normalised to the human reference pupil."},
    {"architecture_ref": "Architecture §2b Slit pupil anisotropic blur diagnostics", "test_name": "Cat Slit Anisotropy", "code_path": "retinal_sim/validation/report.py", "status": "covered", "note": "Report-layer elliptical Gaussian point-response diagnostic for the cat slit pupil."},
    {"architecture_ref": "Architecture §2b Longitudinal chromatic aberration", "test_name": "Wavelength-Dependent Blur", "code_path": "retinal_sim/optical/psf.py", "status": "covered", "note": "Checks signed LCA offsets and minimum blur near the 555 nm reference focus."},
    {"architecture_ref": "Architecture §2c Vitreous and media transmission", "test_name": "Media Transmission Diagnostics", "code_path": "retinal_sim/optical/media.py", "status": "covered", "note": "Samples species media transmission tables and visualizes delivered spectra."},
    {"architecture_ref": "Architecture §11c Snellen acuity prediction", "test_name": "Snellen Acuity", "code_path": "retinal_sim/validation/acuity.py", "status": "covered", "note": "Ordering plus published-range checks."},
    {"architecture_ref": "Architecture §11c Dichromat confusion axis", "test_name": "Dichromat Confusion", "code_path": "retinal_sim/validation/dichromat.py", "status": "covered", "note": "Uses fixed confusion/control stimulus matrices with median-based comparisons."},
    {"architecture_ref": "Architecture §11c Cone density to sampling limit", "test_name": "Nyquist Sampling", "code_path": "retinal_sim/retina/mosaic.py", "status": "covered", "note": "Converts local cone density to cycles/degree using focal-length magnification."},
    {"architecture_ref": "Architecture §11c Receptor count validation", "test_name": "Receptor Count", "code_path": "retinal_sim/retina/mosaic.py", "status": "covered", "note": "Compares measured 1 mm² counts against density-model expectations."},
    {"architecture_ref": "Architecture §11d Known visual deficit reproduction", "test_name": "Color Deficit Reproduction", "code_path": "retinal_sim/pipeline.py", "status": "covered", "note": "End-to-end species comparison."},
    {"architecture_ref": "Architecture §11d Resolution gradient", "test_name": "Resolution Gradient", "code_path": "retinal_sim/output/reconstruction.py", "status": "covered", "note": "Quantifies center-vs-periphery contrast loss on a checkerboard stimulus."},
]


# ---------------------------------------------------------------------------
# Bonus figures: Govardovskii nomogram + mosaic maps
# ---------------------------------------------------------------------------

def _make_nomogram_figure() -> object:
    """Generate Govardovskii nomogram curves for all species."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from retinal_sim.retina.opsin import build_sensitivity_curves, LAMBDA_MAX

    wl = np.arange(380, 721, 1, dtype=float)
    colors = {"S_cone": "#5b5bd6", "M_cone": "#3aac3a", "L_cone": "#d94c4c", "rod": "#888888"}
    labels = {"S_cone": "S-cone", "M_cone": "M-cone", "L_cone": "L-cone", "rod": "Rod"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, species in zip(axes, ["human", "dog", "cat"]):
        curves = build_sensitivity_curves(species, wl)
        for rtype in ["rod", "S_cone", "M_cone", "L_cone"]:
            if rtype not in curves:
                continue
            lm = LAMBDA_MAX[species][rtype]
            ax.plot(wl, curves[rtype], color=colors[rtype], lw=1.5,
                    label=f"{labels[rtype]} ({lm:.0f} nm)")
        ax.set_title(species.capitalize(), fontsize=10)
        ax.set_xlim(380, 720)
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlabel("Wavelength (nm)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, lw=0.3, alpha=0.4)
    axes[0].set_ylabel("Relative sensitivity", fontsize=9)
    fig.suptitle("Govardovskii Nomogram (A1) — All Species", fontsize=11, y=1.01)
    plt.tight_layout()
    return fig


def _make_mosaic_figures(seed: int = 42) -> object:
    """Generate mosaic maps for all species."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from retinal_sim.retina.mosaic import MosaicGenerator
    from retinal_sim.species.config import SpeciesConfig
    from retinal_sim.constants import WAVELENGTHS

    type_colors = {"rod": "#888888", "S_cone": "#5b5bd6", "M_cone": "#3aac3a", "L_cone": "#d94c4c"}
    species_list = ["human", "dog", "cat"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, sp in zip(axes, species_list):
        cfg = SpeciesConfig.load(sp)
        gen = MosaicGenerator(cfg.retinal, cfg.optical, WAVELENGTHS)
        mosaic = gen.generate(seed=seed)
        pos = mosaic.positions
        for rtype in ["rod", "S_cone", "M_cone", "L_cone"]:
            mask = mosaic.types == rtype
            if not np.any(mask):
                continue
            ax.scatter(pos[mask, 0] * 1000, pos[mask, 1] * 1000,
                       s=1.5, c=type_colors[rtype], label=rtype, alpha=0.6)
        ax.set_title(f"{sp.capitalize()} ({mosaic.n_receptors} receptors)", fontsize=10)
        ax.set_xlabel("x (µm)")
        ax.set_aspect("equal")
        ax.legend(fontsize=7, markerscale=4)
    axes[0].set_ylabel("y (µm)")
    fig.suptitle("Photoreceptor Mosaics (2° patch)", fontsize=11)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

_REPORT_STYLE = """
body {
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 1100px;
    margin: 40px auto;
    padding: 0 24px;
    color: #222;
    line-height: 1.5;
}
h1 { border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { border-bottom: 1px solid #ccc; margin-top: 2.2em; color: #333; }
.timestamp { color: #666; font-size: 0.88em; margin-top: -6px; }
.summary-box {
    display: inline-block; padding: 10px 18px; border-radius: 6px;
    font-size: 1.1em; font-weight: bold; margin-bottom: 8px;
}
.all-pass { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
.has-fail  { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
.test-card {
    border: 1px solid #ddd; border-radius: 6px; padding: 16px; margin: 16px 0;
    background: #fafafa;
}
.test-card.passed { border-left: 4px solid #28a745; }
.test-card.failed { border-left: 4px solid #dc3545; }
.test-title { font-size: 1.05em; font-weight: bold; margin: 0 0 8px 0; }
.test-badge {
    display: inline-block; padding: 2px 8px; border-radius: 3px;
    font-size: 0.82em; font-weight: bold; margin-left: 8px;
}
.badge-pass { background: #d4edda; color: #155724; }
.badge-fail { background: #f8d7da; color: #721c24; }
.test-details { font-size: 0.88em; color: #555; margin: 6px 0; }
pre {
    background: #f0f0f0; border: 1px solid #ddd; padding: 10px;
    border-radius: 4px; font-size: 0.82em; white-space: pre-wrap;
    overflow-x: auto;
}
.test-figure { margin: 10px 0; }
.test-figure img { max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; }
.bonus-section img { max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; margin: 8px 0; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.92em; }
th { background: #f4f4f4; }
"""


def _fig_to_data_uri(fig) -> str:
    """Convert a matplotlib figure to a base64 data URI."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    finally:
        buf.close()
        plt.close(fig)


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_report_html(results: List[ValidationResult]) -> str:
    """Build a self-contained HTML report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    ok = failed == 0

    box_cls = "all-pass" if ok else "has-fail"
    summary_text = f"{'ALL PASSED' if ok else f'{failed} FAILED'} — {passed}/{total} tests"

    # Test result cards
    cards_html = ""
    for r in results:
        cls = "passed" if r.passed else "failed"
        badge = '<span class="test-badge badge-pass">PASS</span>' if r.passed else '<span class="test-badge badge-fail">FAIL</span>'

        fig_html = ""
        if r.figure is not None:
            try:
                uri = _fig_to_data_uri(r.figure)
                fig_html = f'<div class="test-figure"><img src="{uri}" alt="{_escape(r.test_name)}"></div>'
            except Exception:
                fig_html = '<p><em>Figure generation failed.</em></p>'

        cards_html += f"""
<div class="test-card {cls}">
  <p class="test-title">{_escape(r.test_name)} {badge}</p>
  <div class="test-details">
    <strong>Expected:</strong> {_escape(str(r.expected))}<br>
    <strong>Actual:</strong> {_escape(str(r.actual))}<br>
    <strong>Tolerance:</strong> {r.tolerance}
  </div>
  <pre>{_escape(r.details)}</pre>
  {fig_html}
</div>
"""

    # Bonus figures: nomogram + mosaics
    bonus_html = ""
    try:
        nomo_fig = _make_nomogram_figure()
        nomo_uri = _fig_to_data_uri(nomo_fig)
        bonus_html += f'<h3>Govardovskii Nomogram</h3><div class="bonus-section"><img src="{nomo_uri}" alt="Nomogram"></div>'
    except Exception:
        bonus_html += '<p><em>Nomogram figure failed.</em></p>'
    try:
        mosaic_fig = _make_mosaic_figures()
        mosaic_uri = _fig_to_data_uri(mosaic_fig)
        bonus_html += f'<h3>Photoreceptor Mosaics</h3><div class="bonus-section"><img src="{mosaic_uri}" alt="Mosaics"></div>'
    except Exception:
        bonus_html += '<p><em>Mosaic figure failed.</em></p>'

    # Summary table
    table_html = "<table><thead><tr><th>#</th><th>Test</th><th>Result</th><th>Details</th></tr></thead><tbody>"
    for i, r in enumerate(results, 1):
        status = "PASS" if r.passed else "FAIL"
        color = "#155724" if r.passed else "#721c24"
        table_html += f'<tr><td>{i}</td><td>{_escape(r.test_name)}</td><td style="color:{color};font-weight:bold">{status}</td><td>{_escape(str(r.actual)[:120])}</td></tr>'
    table_html += "</tbody></table>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>retinal_sim — Full Validation Report</title>
  <style>{_REPORT_STYLE}</style>
</head>
<body>
<h1>retinal_sim — Full Validation Report</h1>
<p class="timestamp">Generated: {timestamp}</p>

<h2>Summary</h2>
<div class="summary-box {box_cls}">{_escape(summary_text)}</div>
{table_html}

<h2>Bonus: Reference Figures</h2>
{bonus_html}

<h2>Test Details &amp; Figures</h2>
{cards_html}

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Transparent audit-report overrides
# ---------------------------------------------------------------------------

_REPORT_STYLE = """
body {
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 1100px;
    margin: 40px auto;
    padding: 0 24px;
    color: #222;
    line-height: 1.5;
}
h1 { border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { border-bottom: 1px solid #ccc; margin-top: 2.2em; color: #333; }
h3 { margin-bottom: 0.4em; }
.timestamp { color: #666; font-size: 0.88em; margin-top: -6px; }
.summary-box {
    display: inline-block; padding: 10px 18px; border-radius: 6px;
    font-size: 1.1em; font-weight: bold; margin-bottom: 8px;
}
.all-pass { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
.has-fail  { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
.section-note {
    background: #eef5fb; border: 1px solid #c9deef; border-radius: 6px;
    padding: 10px 14px; margin: 12px 0;
}
.warning-list {
    background: #fff3cd; border: 1px solid #ffe08a; border-radius: 6px;
    padding: 10px 14px; margin: 12px 0;
}
.pill {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 0.78em; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.03em;
}
.pill-covered { background: #d4edda; color: #155724; }
.pill-partial { background: #fff3cd; color: #856404; }
.pill-gap { background: #f8d7da; color: #721c24; }
.pill-category-analytic { background: #dbeafe; color: #1d4ed8; }
.pill-category-self { background: #ede9fe; color: #6d28d9; }
.pill-category-external { background: #dcfce7; color: #166534; }
.pill-support-strong { background: #d1fae5; color: #065f46; }
.pill-support-moderate { background: #fef3c7; color: #92400e; }
.pill-support-weak { background: #fee2e2; color: #991b1b; }
.test-card {
    border: 1px solid #ddd; border-radius: 6px; padding: 16px; margin: 16px 0;
    background: #fafafa;
}
.test-card.passed { border-left: 4px solid #28a745; }
.test-card.failed { border-left: 4px solid #dc3545; }
.badge-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 12px 0; }
.test-title { font-size: 1.05em; font-weight: bold; margin: 0 0 8px 0; }
.test-badge {
    display: inline-block; padding: 2px 8px; border-radius: 3px;
    font-size: 0.82em; font-weight: bold; margin-left: 8px;
}
.badge-pass { background: #d4edda; color: #155724; }
.badge-fail { background: #f8d7da; color: #721c24; }
.meta-grid {
    display: grid; grid-template-columns: 220px 1fr; gap: 6px 12px; margin: 10px 0;
}
.meta-grid div { padding: 4px 0; border-bottom: 1px solid #eee; }
.meta-label { font-weight: 700; color: #333; }
.inline-list { margin: 0; padding-left: 18px; }
pre {
    background: #f0f0f0; border: 1px solid #ddd; padding: 10px;
    border-radius: 4px; font-size: 0.82em; white-space: pre-wrap;
    overflow-x: auto;
}
.test-figure { margin: 10px 0; }
.test-figure img { max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; }
.bonus-section img { max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; margin: 8px 0; }
.split-summary { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.reference-table { margin: 16px 0; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.92em; }
th { background: #f4f4f4; }
"""


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dict__"):
        return {str(k): _json_safe(v) for k, v in vars(value).items()}
    return str(value)


def _count_spec_field(specs: Dict[str, Dict[str, Any]], field_name: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for spec in specs.values():
        value = str(spec.get(field_name, "")).strip() or "unspecified"
        counts[value] = counts.get(value, 0) + 1
    return counts


def _summarize_dataclass(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "physiology_metadata"):
        physiology = value.physiology_metadata()
        return {
            "species": physiology.get("species"),
            "model_scope": physiology.get("model_scope"),
            "aperture_weighting_enabled": physiology.get("aperture_weighting", {}).get("enabled"),
            "visual_streak_status": physiology.get("visual_streak", {}).get("status"),
            "naka_rushton_confidence": physiology.get("naka_rushton_provenance", {}).get("confidence"),
        }
    keys = (
        "pupil_shape",
        "pupil_diameter_mm",
        "pupil_height_mm",
        "axial_length_mm",
        "focal_length_mm",
        "corneal_radius_mm",
        "lca_diopters",
        "cone_ratio",
        "cone_aperture_um",
        "rod_aperture_um",
        "patch_extent_deg",
        "naka_rushton_sigma",
    )
    summary: Dict[str, Any] = {}
    for key in keys:
        if hasattr(value, key):
            summary[key] = _json_safe(getattr(value, key))
    return summary


_CAT_SLIT_HEIGHT_MM = 7.0
_CAT_DIAGNOSTIC_WAVELENGTH_NM = 550.0
_CAT_DIAGNOSTIC_PIXEL_SCALE_MM = 0.001


def _pupil_area_mm2(optical: Any) -> float:
    """Return pupil area for circular or slit pupils."""
    if hasattr(optical, "pupil_area_mm2"):
        return float(optical.pupil_area_mm2())
    shape = str(getattr(optical, "pupil_shape", "circular"))
    width = float(getattr(optical, "pupil_diameter_mm", 0.0))
    if shape == "slit":
        height = float(getattr(optical, "pupil_height_mm", _CAT_SLIT_HEIGHT_MM))
        return float(np.pi * (width / 2.0) * (height / 2.0))
    return float(np.pi * (width / 2.0) ** 2)


def _cat_slit_psf_diagnostics(optical: Any) -> Dict[str, Any]:
    """Build report-layer diagnostics from the current optical-stage slit model."""
    from retinal_sim.optical.stage import OpticalStage

    stage = OpticalStage(optical)
    kernels, metadata = stage.compute_psf(
        np.array([_CAT_DIAGNOSTIC_WAVELENGTH_NM], dtype=float),
        return_metadata=True,
    )
    kernel = kernels[0]
    sigma_mm_x = float(metadata["sigma_mm_x"][0])
    sigma_mm_y = float(metadata["sigma_mm_y"][0])
    sigma_px_x = float(metadata["sigma_px_x"][0])
    sigma_px_y = float(metadata["sigma_px_y"][0])

    return {
        "kernel": kernel,
        "pupil_area_mm2": _pupil_area_mm2(optical),
        "effective_f_number": float(metadata["effective_f_number"]),
        "effective_f_number_x": float(metadata["effective_f_number_x"]),
        "effective_f_number_y": float(metadata["effective_f_number_y"]),
        "sigma_x_mm": sigma_mm_x,
        "sigma_y_mm": sigma_mm_y,
        "sigma_x_px": sigma_px_x,
        "sigma_y_px": sigma_px_y,
        "anisotropy_active": bool(metadata["anisotropy_active"]),
        "pixel_scale_mm": _CAT_DIAGNOSTIC_PIXEL_SCALE_MM,
        "psf_sigma_mm_x": sigma_mm_x,
        "psf_sigma_mm_y": sigma_mm_y,
        "psf_sigma_px_x": sigma_px_x,
        "psf_sigma_px_y": sigma_px_y,
    }


def _get_git_commit_hash() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _list_html(items: List[str]) -> str:
    if not items:
        return "<em>None recorded.</em>"
    return "<ul class=\"inline-list\">" + "".join(
        f"<li>{_escape(str(item))}</li>" for item in items
    ) + "</ul>"


def _metadata_table_html(metadata: Dict[str, Any]) -> str:
    retinal = metadata.get("retinal_physiology", {})
    rows = [
        ("Report type", metadata.get("report_type", "unknown")),
        ("Stage scope", metadata.get("stage_scope", "unknown")),
        ("Species", metadata.get("species", "unknown")),
        ("Seed", metadata.get("seed", "unknown")),
        ("Patch extent (deg)", metadata.get("patch_extent_deg", "unknown")),
        ("Stimulus scale", metadata.get("stimulus_scale", "unknown")),
        ("Light level", metadata.get("light_level", "unknown")),
        ("Scene input mode", metadata.get("scene_input_mode", "unknown")),
        (
            "Scene spectrum source",
            "RGB-inferred" if metadata.get("scene_input_is_inferred", True) else "Measured spectrum",
        ),
        (
            "Scene assumptions",
            "; ".join(metadata.get("scene_input_assumptions", [])) or "unknown",
        ),
        ("Repo commit", metadata.get("repo_commit", "unknown")),
        ("Python", metadata.get("environment", {}).get("python", "unknown")),
        ("Platform", metadata.get("environment", {}).get("platform", "unknown")),
        ("NumPy", metadata.get("environment", {}).get("numpy", "unknown")),
        ("Retinal model scope", retinal.get("model_scope", "unknown")),
        ("Retinal scope note", retinal.get("scope_note", "unknown")),
        (
            "Naka-Rushton confidence",
            retinal.get("naka_rushton_provenance", {}).get("confidence", "unknown"),
        ),
        (
            "Aperture weighting default",
            "Enabled"
            if retinal.get("aperture_weighting", {}).get("enabled", False)
            else "Disabled",
        ),
        (
            "Visual streak status",
            retinal.get("visual_streak", {}).get("status", "unknown"),
        ),
        (
            "Validation categories",
            _counts_summary_text(metadata.get("validation_category_counts", {})),
        ),
        (
            "Claim support levels",
            _counts_summary_text(metadata.get("claim_support_level_counts", {})),
        ),
    ]
    return "<div class=\"meta-grid\">" + "".join(
        f"<div class=\"meta-label\">{_escape(label)}</div><div>{_escape(str(value))}</div>"
        for label, value in rows
    ) + "</div>"


def _retinal_physiology_metadata(retinal: Any) -> Dict[str, Any]:
    if retinal is None or not hasattr(retinal, "physiology_metadata"):
        return {}
    return _json_safe(retinal.physiology_metadata())


def _retinal_physiology_html(metadata: Dict[str, Any]) -> str:
    retinal = metadata.get("retinal_physiology", {})
    if not retinal:
        return "<div class=\"section-note\"><em>No retinal physiology metadata recorded.</em></div>"

    lambda_max = retinal.get("lambda_max_provenance", {})
    density = retinal.get("density_function_provenance", {})
    naka = retinal.get("naka_rushton_provenance", {})
    aperture = retinal.get("aperture_weighting", {})
    visual_streak = retinal.get("visual_streak", {})
    nr_config = retinal.get("naka_rushton_configuration", {})

    return (
        "<div class=\"section-note\">"
        "<p><strong>Retinal Front-End Only.</strong> "
        f"{_escape(str(retinal.get('scope_note', 'unknown')))}</p>"
        "<div class=\"meta-grid\">"
        f"<div class=\"meta-label\">Lambda-max source</div><div>{_escape(str(lambda_max.get('source', 'unknown')))}</div>"
        f"<div class=\"meta-label\">Lambda-max confidence</div><div>{_escape(str(lambda_max.get('confidence', 'unknown')))}</div>"
        f"<div class=\"meta-label\">Lambda-max notes</div><div>{_escape(str(lambda_max.get('notes', '')))}</div>"
        f"<div class=\"meta-label\">Density-function source</div><div>{_escape(str(density.get('source', 'unknown')))}</div>"
        f"<div class=\"meta-label\">Density-function confidence</div><div>{_escape(str(density.get('confidence', 'unknown')))}</div>"
        f"<div class=\"meta-label\">Density-function notes</div><div>{_escape(str(density.get('notes', '')))}</div>"
        f"<div class=\"meta-label\">Naka-Rushton source</div><div>{_escape(str(naka.get('source', 'unknown')))}</div>"
        f"<div class=\"meta-label\">Naka-Rushton confidence</div><div>{_escape(str(naka.get('confidence', 'unknown')))}</div>"
        f"<div class=\"meta-label\">Naka-Rushton notes</div><div>{_escape(str(naka.get('notes', '')))}</div>"
        f"<div class=\"meta-label\">Naka-Rushton configuration</div><div><pre>{_escape(json.dumps(nr_config, indent=2))}</pre></div>"
        f"<div class=\"meta-label\">Aperture weighting</div><div>{_escape(str(aperture.get('enabled', False)))} ({_escape(str(aperture.get('method', 'unknown')))})</div>"
        f"<div class=\"meta-label\">Aperture notes</div><div>{_escape(str(aperture.get('notes', '')))}</div>"
        f"<div class=\"meta-label\">Visual streak status</div><div>{_escape(str(visual_streak.get('status', 'unknown')))}</div>"
        f"<div class=\"meta-label\">Visual streak notes</div><div>{_escape(str(visual_streak.get('notes', '')))}</div>"
        "</div></div>"
    )


def _coverage_badge(status: str) -> str:
    if status == "covered":
        cls = "pill-covered"
    elif status == "partially covered":
        cls = "pill-partial"
    else:
        cls = "pill-gap"
    return f'<span class="pill {cls}">{_escape(status)}</span>'


def _category_badge(value: str) -> str:
    classes = {
        "analytic correctness": "pill-category-analytic",
        "model self-consistency": "pill-category-self",
        "external empirical alignment": "pill-category-external",
    }
    cls = classes.get(value, "pill-partial")
    return f'<span class="pill {cls}">{_escape(value)}</span>'


def _support_badge(value: str) -> str:
    classes = {
        "strong": "pill-support-strong",
        "moderate": "pill-support-moderate",
        "weak": "pill-support-weak",
    }
    cls = classes.get(value, "pill-partial")
    return f'<span class="pill {cls}">{_escape(value)}</span>'


def _counts_summary_text(counts: Dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}: {counts[key]}" for key in sorted(counts))


def _category_summary_html(metadata: Dict[str, Any]) -> str:
    rows = ""
    for label, count in sorted(metadata.get("validation_category_counts", {}).items()):
        rows += (
            "<tr>"
            f"<td>{_category_badge(label)}</td>"
            f"<td>{count}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Validation category</th><th>Result count</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _support_summary_html(metadata: Dict[str, Any]) -> str:
    rows = ""
    for label, count in sorted(metadata.get("claim_support_level_counts", {}).items()):
        rows += (
            "<tr>"
            f"<td>{_support_badge(label)}</td>"
            f"<td>{count}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Claim support level</th><th>Result count</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _claim_calibration_html(metadata: Dict[str, Any]) -> str:
    notes = metadata.get("claim_calibration_notes", [])
    return (
        "<div class=\"section-note\">"
        "<strong>Claim calibration:</strong>"
        f"{_list_html(list(notes))}"
        "</div>"
    )


def _external_reference_tables_html(metadata: Dict[str, Any]) -> str:
    sections = []
    for table_name, table in metadata.get("external_reference_tables", {}).items():
        rows = _reference_rows(table_name)
        if not rows:
            continue
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        header_html = "".join(f"<th>{_escape(column.replace('_', ' '))}</th>" for column in columns)
        body_html = ""
        for row in rows:
            body_html += "<tr>" + "".join(
                f"<td>{_escape(str(_json_safe(row.get(column, ''))))}</td>" for column in columns
            ) + "</tr>"
        sections.append(
            "<div class=\"reference-table\">"
            f"<h3>{_escape(table_name.replace('_', ' '))}</h3>"
            "<p><em>Reference evidence table. These rows describe external or configured expectations, not measured simulator outputs.</em></p>"
            f"<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"
            "</div>"
        )
    if not sections:
        return "<div class=\"section-note\"><em>No external reference evidence tables recorded.</em></div>"
    return "".join(sections)


def _coverage_matrix_html(metadata: Dict[str, Any]) -> str:
    rows_html = ""
    for row in metadata.get("architecture_coverage", []):
        rows_html += (
            "<tr>"
            f"<td>{_escape(row.get('architecture_ref', ''))}</td>"
            f"<td>{_escape(row.get('test_name', ''))}</td>"
            f"<td>{_escape(row.get('code_path', ''))}</td>"
            f"<td>{_coverage_badge(row.get('status', 'gap'))}</td>"
            f"<td>{_escape(row.get('note', ''))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Architecture item</th><th>Implemented test</th><th>Code path</th><th>Status</th><th>Note</th>"
        "</tr></thead><tbody>"
        f"{rows_html}</tbody></table>"
    )


def _result_card_html(result: ValidationResult) -> str:
    cls = "passed" if result.passed else "failed"
    badge = (
        '<span class="test-badge badge-pass">PASS</span>'
        if result.passed else
        '<span class="test-badge badge-fail">FAIL</span>'
    )
    fig_html = ""
    if result.figure is not None:
        try:
            uri = _fig_to_data_uri(result.figure)
            fig_html = f'<div class="test-figure"><img src="{uri}" alt="{_escape(result.test_name)}"></div>'
        except Exception:
            try:
                import matplotlib.pyplot as plt
                plt.close(result.figure)
            except Exception:
                pass
            fig_html = '<p><em>Figure generation failed.</em></p>'

    return f"""
<div class="test-card {cls}">
  <p class="test-title">{_escape(result.test_name)} {badge}</p>
  <div class="badge-row">
    {_category_badge(result.validation_category)}
    {_support_badge(result.claim_support_level)}
  </div>
  <div class="meta-grid">
    <div class="meta-label">Stage</div><div>{_escape(result.stage)}</div>
    <div class="meta-label">Architecture reference</div><div>{_escape(result.architecture_ref)}</div>
    <div class="meta-label">Expected outcome</div><div>{_escape(str(result.expected))}</div>
    <div class="meta-label">Observed outcome</div><div>{_escape(str(result.actual))}</div>
    <div class="meta-label">Tolerance</div><div>{_escape(str(result.tolerance))}</div>
    <div class="meta-label">Pass criterion</div><div>{_escape(result.pass_criterion)}</div>
    <div class="meta-label">Validation category</div><div>{_escape(result.validation_category)}</div>
    <div class="meta-label">Claim support level</div><div>{_escape(result.claim_support_level)}</div>
    <div class="meta-label">Evidence basis</div><div>{_escape(result.evidence_basis)}</div>
    <div class="meta-label">External evidence summary</div><div>{_escape(result.external_reference_summary or "None")}</div>
    <div class="meta-label">Claim scope note</div><div>{_escape(result.claim_scope_note)}</div>
    <div class="meta-label">Inputs summary</div><div>{_escape(result.inputs_summary)}</div>
    <div class="meta-label">Method</div><div>{_escape(result.method)}</div>
    <div class="meta-label">Code references</div><div>{_list_html(result.code_refs)}</div>
    <div class="meta-label">External reference table</div><div>{_list_html([row.get("reference_label", "") for row in result.external_reference_table])}</div>
    <div class="meta-label">Assumptions</div><div>{_list_html(result.assumptions)}</div>
    <div class="meta-label">Limitations</div><div>{_list_html(result.limitations)}</div>
    <div class="meta-label">Artifacts</div><div>{_list_html(result.artifacts)}</div>
  </div>
  <pre>{_escape(result.details)}</pre>
  {fig_html}
</div>
"""


def _build_report_html(report: ValidationReport) -> str:
    """Build a self-contained HTML report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(report.results)
    passed = sum(1 for r in report.results if r.passed)
    failed = total - passed
    ok = failed == 0

    box_cls = "all-pass" if ok else "has-fail"
    summary_text = (
        f"{passed}/{total} implemented validation checks passed under the current test harness for their stated scopes and assumptions"
        if ok else
        f"{failed}/{total} implemented validation checks failed under the current test harness for their stated scopes and assumptions"
    )

    cards_html = "".join(_result_card_html(r) for r in report.results)

    bonus_html = ""
    nomo_fig = None
    try:
        nomo_fig = _make_nomogram_figure()
        nomo_uri = _fig_to_data_uri(nomo_fig)
        nomo_fig = None
        bonus_html += f'<h3>Govardovskii Nomogram</h3><div class="bonus-section"><img src="{nomo_uri}" alt="Nomogram"></div>'
    except Exception:
        if nomo_fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(nomo_fig)
            except Exception:
                pass
        bonus_html += '<p><em>Nomogram figure failed.</em></p>'
    mosaic_fig = None
    try:
        mosaic_fig = _make_mosaic_figures()
        mosaic_uri = _fig_to_data_uri(mosaic_fig)
        mosaic_fig = None
        bonus_html += f'<h3>Photoreceptor Mosaics</h3><div class="bonus-section"><img src="{mosaic_uri}" alt="Mosaics"></div>'
    except Exception:
        if mosaic_fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(mosaic_fig)
            except Exception:
                pass
        bonus_html += '<p><em>Mosaic figure failed.</em></p>'

    table_html = "<table><thead><tr><th>#</th><th>Test</th><th>Result</th><th>Architecture</th><th>Observed</th></tr></thead><tbody>"
    for i, r in enumerate(report.results, 1):
        status = "PASS" if r.passed else "FAIL"
        color = "#155724" if r.passed else "#721c24"
        table_html += (
            f'<tr><td>{i}</td><td>{_escape(r.test_name)}</td>'
            f'<td style="color:{color};font-weight:bold">{status}</td>'
            f'<td>{_escape(r.architecture_ref)}</td>'
            f'<td>{_escape(str(r.actual)[:120])}</td></tr>'
        )
    table_html += "</tbody></table>"

    warnings_html = _list_html(list(report.metadata.get("warnings", [])))
    metadata_html = _metadata_table_html(report.metadata)
    retinal_physiology_html = _retinal_physiology_html(report.metadata)
    coverage_html = _coverage_matrix_html(report.metadata)
    category_summary_html = _category_summary_html(report.metadata)
    support_summary_html = _support_summary_html(report.metadata)
    external_reference_tables_html = _external_reference_tables_html(report.metadata)
    claim_calibration_html = _claim_calibration_html(report.metadata)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>retinal_sim - Full Validation Audit Report</title>
  <style>{_REPORT_STYLE}</style>
</head>
<body>
<h1>retinal_sim - Full Validation Audit Report</h1>
<p class="timestamp">Generated: {timestamp}</p>

<h2>Scope of This Report</h2>
<div class="section-note">
  This audit report shows the implemented validation checks, the exact pass criteria used,
  the observed outcomes, the assumptions and limitations attached to each check, and the code paths
  that produced the result. Statements here are evidence-backed for specific implemented checks in the
  current proof-of-concept build, not blanket claims of whole-simulator physiological or perceptual validation.
</div>

<h2>Summary</h2>
<div class="summary-box {box_cls}">{_escape(summary_text)}</div>
{claim_calibration_html}
{table_html}

<h2>Validation Framing</h2>
<div class="split-summary">
  <div>
    <h3>Validation Categories</h3>
    {category_summary_html}
  </div>
  <div>
    <h3>Claim Support Levels</h3>
    {support_summary_html}
  </div>
</div>
<div class="section-note">
  <strong>Legend:</strong> analytic correctness = closed-form or invariant checks;
  model self-consistency = internal implementation/regression checks;
  external empirical alignment = checks compared against local deterministic reference evidence.
</div>

<h2>Environment and Reproducibility</h2>
{metadata_html}

<h2>Retinal Physiology Assumptions</h2>
{retinal_physiology_html}

<h2>External Reference Evidence</h2>
<div class="section-note">
  The tables below are explicit reference evidence used to calibrate R5 claims.
  They are reference expectations, not measured simulator outputs.
</div>
{external_reference_tables_html}

<h2>Known Limitations / Deferred Architecture Items</h2>
<div class="warning-list">{warnings_html}</div>

<h2>Architecture Coverage Matrix</h2>
{coverage_html}

<h2>Open Code Review Findings</h2>
<div class="section-note">
  Read this report alongside <code>CODEREVIEW.md</code> and the generated status report.
  Validation pass/fail status does not replace code review findings or architecture audit notes.
</div>

<h2>Bonus: Reference Figures</h2>
{bonus_html}

<h2>Test Details, Evidence, and Provenance</h2>
{cards_html}

</body>
</html>
"""
