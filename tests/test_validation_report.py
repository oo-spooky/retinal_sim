"""Phase 13: Full validation report generator tests.

Tests are grouped into five classes:

TestValidationResult
    Dataclass construction and field access.

TestValidationReport
    Report summary, HTML generation, and save_html.

TestValidationSuiteConstruction
    Suite init, stage dispatch, run_all/run_stage structure.

TestValidationSuiteTests
    Individual validation test methods return correct types and produce
    figures for visual inspection.

TestHTMLOutput
    End-to-end HTML report generation and content verification.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from retinal_sim.validation.report import (
    ValidationResult,
    ValidationReport,
    ValidationSuite,
)

def workspace_path(*parts: str) -> Path:
    base = Path("tests/.tmp_validation_report")
    base.mkdir(parents=True, exist_ok=True)
    return base.joinpath(*parts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def suite() -> ValidationSuite:
    """Create a ValidationSuite with a human RetinalSimulator."""
    from retinal_sim.pipeline import RetinalSimulator
    sim = RetinalSimulator("human", patch_extent_deg=1.0, stimulus_scale=0.01, seed=42)
    return ValidationSuite(sim, seed=42)


@pytest.fixture(scope="module")
def sample_result() -> ValidationResult:
    return ValidationResult(
        test_name="Sample Test",
        passed=True,
        expected=1.0,
        actual=0.99,
        tolerance=0.05,
        details="Sample details",
        figure=None,
    )


@pytest.fixture(scope="module")
def sample_result_with_figure() -> ValidationResult:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return ValidationResult(
        test_name="Figure Test",
        passed=True,
        expected="line",
        actual="line",
        tolerance=0.0,
        details="Has a figure",
        figure=fig,
    )


# ---------------------------------------------------------------------------
# TestValidationResult
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_construction(self, sample_result: ValidationResult):
        assert sample_result.test_name == "Sample Test"
        assert sample_result.passed is True
        assert sample_result.expected == 1.0
        assert sample_result.actual == 0.99
        assert sample_result.tolerance == 0.05
        assert sample_result.details == "Sample details"
        assert sample_result.figure is None

    def test_failed_result(self):
        r = ValidationResult(
            test_name="Fail",
            passed=False,
            expected=1.0,
            actual=2.0,
            tolerance=0.1,
            details="Over tolerance",
        )
        assert r.passed is False

    def test_default_figure_is_none(self):
        r = ValidationResult(
            test_name="No Fig",
            passed=True,
            expected=0,
            actual=0,
            tolerance=0,
            details="",
        )
        assert r.figure is None

    def test_result_with_figure(self, sample_result_with_figure: ValidationResult):
        assert sample_result_with_figure.figure is not None
        assert hasattr(sample_result_with_figure.figure, "savefig")

    def test_result_fields_are_any(self):
        r = ValidationResult(
            test_name="Mixed types",
            passed=True,
            expected={"key": "val"},
            actual=[1, 2, 3],
            tolerance=0.0,
            details="Arbitrary types",
        )
        assert isinstance(r.expected, dict)
        assert isinstance(r.actual, list)

    def test_new_metadata_fields_default(self):
        r = ValidationResult("Meta", True, 1, 1, 0.0, "ok")
        assert r.stage == ""
        assert r.architecture_ref == ""
        assert r.code_refs == []
        assert r.assumptions == []
        assert r.limitations == []


# ---------------------------------------------------------------------------
# TestValidationReport
# ---------------------------------------------------------------------------

class TestValidationReport:
    def test_empty_report_summary(self):
        report = ValidationReport()
        assert "0/0" in report.summary()

    def test_all_pass_summary(self, sample_result: ValidationResult):
        report = ValidationReport([sample_result])
        s = report.summary()
        assert "PASSED" in s
        assert "1/1" in s

    def test_has_fail_summary(self):
        r_pass = ValidationResult("A", True, 0, 0, 0, "")
        r_fail = ValidationResult("B", False, 1, 2, 0, "")
        report = ValidationReport([r_pass, r_fail])
        s = report.summary()
        assert "FAIL" in s
        assert "1/2" in s

    def test_results_list(self, sample_result: ValidationResult):
        report = ValidationReport([sample_result])
        assert len(report.results) == 1
        assert report.results[0] is sample_result
        assert report.metadata == {}

    def test_save_html_creates_file(self, sample_result: ValidationResult):
        report = ValidationReport([sample_result])
        path = workspace_path("report_basic.html")
        report.save_html(str(path))
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Sample Test" in content

    def test_save_html_with_figure(self, sample_result_with_figure: ValidationResult):
        report = ValidationReport([sample_result_with_figure])
        path = workspace_path("report_figure.html")
        report.save_html(str(path))
        content = path.read_text(encoding="utf-8")
        assert "data:image/png;base64," in content

    def test_save_html_creates_parent_dirs(self, sample_result: ValidationResult):
        report = ValidationReport([sample_result])
        path = workspace_path("nested", "dir", "report.html")
        report.save_html(str(path))
        assert path.exists()

    def test_save_html_escapes_html_entities(self):
        r = ValidationResult(
            test_name="<script>alert('xss')</script>",
            passed=True,
            expected="<b>expected</b>",
            actual="<i>actual</i>",
            tolerance=0.0,
            details="details with & < > chars",
        )
        report = ValidationReport([r])
        path = workspace_path("report_escape.html")
        report.save_html(str(path))
        content = path.read_text(encoding="utf-8")
        assert "<script>" not in content
        assert "&lt;script&gt;" in content

    def test_save_json_creates_file(self, sample_result: ValidationResult):
        report = ValidationReport([sample_result], metadata={"report_type": "unit"})
        path = workspace_path("report_unit.json")
        report.save_json(str(path))
        content = path.read_text(encoding="utf-8")
        assert '"report_type": "unit"' in content
        assert '"test_name": "Sample Test"' in content


# ---------------------------------------------------------------------------
# TestValidationSuiteConstruction
# ---------------------------------------------------------------------------

class TestValidationSuiteConstruction:
    def test_init(self, suite: ValidationSuite):
        assert suite._seed == 42
        assert suite._sim is not None

    def test_run_stage_invalid(self, suite: ValidationSuite):
        with pytest.raises(ValueError, match="Unknown stage"):
            suite.run_stage("nonexistent")

    def test_run_stage_returns_report(self, suite: ValidationSuite):
        report = suite.run_stage("spectral")
        assert isinstance(report, ValidationReport)
        assert len(report.results) == 3

    def test_run_stage_scene_count(self, suite: ValidationSuite):
        report = suite.run_stage("scene")
        assert isinstance(report, ValidationReport)
        assert len(report.results) == 4

    def test_run_stage_optical_count(self, suite: ValidationSuite):
        report = suite.run_stage("optical")
        assert isinstance(report, ValidationReport)
        assert len(report.results) == 6
        assert [r.test_name for r in report.results] == [
            "MTF vs Diffraction Limit",
            "PSF Energy Conservation",
            "Pupil Throughput Scaling",
            "Cat Slit Anisotropy",
            "Wavelength-Dependent Blur",
            "Media Transmission Diagnostics",
        ]

    def test_run_stage_retinal_count(self, suite: ValidationSuite):
        report = suite.run_stage("retinal")
        assert isinstance(report, ValidationReport)
        assert len(report.results) == 4

    def test_run_stage_e2e_count(self, suite: ValidationSuite):
        report = suite.run_stage("e2e")
        assert isinstance(report, ValidationReport)
        assert len(report.results) == 2

    def test_run_all_count(self, suite: ValidationSuite):
        report = suite.run_all()
        assert isinstance(report, ValidationReport)
        assert len(report.results) == 19

    def test_run_all_returns_validation_results(self, suite: ValidationSuite):
        report = suite.run_all()
        for r in report.results:
            assert isinstance(r, ValidationResult)

    def test_run_all_metadata_contains_provenance(self, suite: ValidationSuite):
        report = suite.run_all()
        assert report.metadata["report_type"] == "full_validation_audit"
        assert report.metadata["species"] == "human"
        assert "architecture_coverage" in report.metadata
        assert len(report.metadata["architecture_coverage"]) == 19
        assert report.metadata["stage_counts"]["optical"] == 6


# ---------------------------------------------------------------------------
# TestValidationSuiteTests — individual test methods
# ---------------------------------------------------------------------------

class TestValidationSuiteTests:
    def test_angular_subtense(self, suite: ValidationSuite):
        r = suite.test_angular_subtense()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Angular Subtense"
        assert r.passed is True
        assert r.figure is not None
        assert r.stage == "scene"
        assert "Architecture" in r.architecture_ref
        assert r.code_refs
        assert r.pass_criterion
        assert r.assumptions
        assert r.limitations

    def test_retinal_scaling(self, suite: ValidationSuite):
        r = suite.test_retinal_scaling_across_species()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Retinal Scaling Across Species"
        assert r.passed is True
        assert r.figure is not None

    def test_accommodation_defocus(self, suite: ValidationSuite):
        r = suite.test_accommodation_defocus()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Accommodation Defocus"
        assert r.figure is not None

    def test_distance_receptor_sampling(self, suite: ValidationSuite):
        r = suite.test_distance_receptor_sampling()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Distance-Dependent Receptor Sampling"
        assert r.figure is not None

    def test_metamer_preservation(self, suite: ValidationSuite):
        r = suite.test_metamer_preservation()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Metamer Preservation"
        assert r.passed is True
        assert r.figure is not None

    def test_rgb_roundtrip(self, suite: ValidationSuite):
        r = suite.test_rgb_roundtrip()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "RGB Round-Trip"
        assert r.figure is not None

    def test_spectral_response_panel(self, suite: ValidationSuite):
        r = suite.test_spectral_response_panel_v2()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Spectral Response Panel"
        assert r.passed is True
        assert r.figure is not None

    def test_mtf_vs_diffraction(self, suite: ValidationSuite):
        r = suite.test_mtf_vs_diffraction_limit()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "MTF vs Diffraction Limit"
        assert r.passed is True
        assert r.figure is not None

    def test_psf_energy_conservation(self, suite: ValidationSuite):
        r = suite.test_psf_energy_conservation()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "PSF Energy Conservation"
        assert r.passed is True
        assert r.figure is not None

    def test_pupil_throughput_scaling(self, suite: ValidationSuite):
        r = suite.test_pupil_throughput_scaling_v2()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Pupil Throughput Scaling"
        assert r.passed is True
        assert r.figure is not None
        assert "pupil_area_mm2" in r.details

    def test_cat_slit_anisotropy(self, suite: ValidationSuite):
        r = suite.test_cat_slit_anisotropy_v2()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Cat Slit Anisotropy"
        assert r.passed is True
        assert r.figure is not None
        assert "anisotropy_active" in r.details

    def test_wavelength_dependent_blur(self, suite: ValidationSuite):
        r = suite.test_wavelength_dependent_blur_v2()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Wavelength-Dependent Blur"
        assert r.passed is True
        assert r.figure is not None
        assert "lca_400" in r.details

    def test_media_transmission_diagnostics(self, suite: ValidationSuite):
        r = suite.test_media_transmission_diagnostics_v2()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Media Transmission Diagnostics"
        assert r.passed is True
        assert r.figure is not None
        assert "source=" in r.details

    def test_snellen_acuity(self, suite: ValidationSuite):
        r = suite.test_snellen_acuity()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Snellen Acuity"
        assert r.figure is not None

    def test_dichromat_confusion(self, suite: ValidationSuite):
        r = suite.test_dichromat_confusion()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Dichromat Confusion"
        assert r.figure is not None

    def test_nyquist_sampling(self, suite: ValidationSuite):
        r = suite.test_nyquist_sampling()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Nyquist Sampling"
        assert r.figure is not None

    def test_receptor_count(self, suite: ValidationSuite):
        r = suite.test_receptor_count()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Receptor Count"
        assert r.passed is True
        assert r.figure is not None

    def test_color_deficit(self, suite: ValidationSuite):
        r = suite.test_color_deficit_reproduction()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Color Deficit Reproduction"
        assert r.figure is not None

    def test_resolution_gradient(self, suite: ValidationSuite):
        r = suite.test_resolution_gradient()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Resolution Gradient"
        assert r.passed is True
        assert r.figure is not None


# ---------------------------------------------------------------------------
# TestHTMLOutput — end-to-end HTML report
# ---------------------------------------------------------------------------

class TestHTMLOutput:
    def test_full_report_html(self, suite: ValidationSuite):
        report = suite.run_all()
        path = workspace_path("full_report.html")
        report.save_html(str(path))
        content = path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Full Validation Audit Report" in content
        assert "Summary" in content
        assert "Architecture Coverage Matrix" in content
        assert "Environment and Reproducibility" in content
        for r in report.results:
            assert r.test_name in content
        assert content.count("data:image/png;base64,") >= len(report.results) + 2

    def test_report_contains_bonus_figures(self, suite: ValidationSuite):
        report = suite.run_all()
        path = workspace_path("report_bonus.html")
        report.save_html(str(path))
        content = path.read_text(encoding="utf-8")
        assert "Govardovskii Nomogram" in content
        assert "Photoreceptor Mosaics" in content

    def test_report_pass_fail_badges(self, suite: ValidationSuite):
        report = suite.run_all()
        path = workspace_path("report_badges.html")
        report.save_html(str(path))
        content = path.read_text(encoding="utf-8")
        assert "PASS" in content

    def test_report_summary_table(self, suite: ValidationSuite):
        report = suite.run_all()
        path = workspace_path("report_table.html")
        report.save_html(str(path))
        content = path.read_text(encoding="utf-8")
        assert "<table>" in content
        assert "<thead>" in content
        assert "Architecture" in content

    def test_all_figures_have_alt_text(self, suite: ValidationSuite):
        report = suite.run_all()
        path = workspace_path("report_alt.html")
        report.save_html(str(path))
        content = path.read_text(encoding="utf-8")
        import re
        img_tags = re.findall(r"<img\s[^>]*>", content)
        for tag in img_tags:
            assert 'alt="' in tag

    def test_report_contains_transparency_sections(self, suite: ValidationSuite):
        report = suite.run_all()
        path = workspace_path("report_transparency.html")
        report.save_html(str(path))
        content = path.read_text(encoding="utf-8")
        assert "Pass criterion" in content
        assert "Assumptions" in content
        assert "Limitations" in content
        assert "Code references" in content

    def test_report_json_contains_result_metadata(self, suite: ValidationSuite):
        report = suite.run_all()
        path = workspace_path("report_results.json")
        report.save_json(str(path))
        content = path.read_text(encoding="utf-8")
        assert '"architecture_ref"' in content
        assert '"code_refs"' in content
        assert '"pass_criterion"' in content
        assert '"Pupil Throughput Scaling"' in content
        assert '"Cat Slit Anisotropy"' in content
        assert "pupil_area_mm2" in content
        assert "anisotropy_active" in content
