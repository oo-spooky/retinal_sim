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


def _write_and_read_report_output(
    report: ValidationReport,
    filename: str,
    writer: str,
) -> str:
    path = workspace_path(filename)
    getattr(report, writer)(str(path))
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def cleanup_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")

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


@pytest.fixture(scope="module")
def retinal_report(suite: ValidationSuite) -> ValidationReport:
    return suite.run_stage("retinal")


@pytest.fixture(scope="module")
def full_report_bundle(suite: ValidationSuite) -> dict[str, object]:
    report = suite.run_all()
    html = _write_and_read_report_output(report, "full_report.html", "save_html")
    json_text = _write_and_read_report_output(report, "report_results.json", "save_json")
    return {
        "report": report,
        "html": html,
        "json": json_text,
    }


@pytest.fixture(scope="module")
def full_report(full_report_bundle: dict[str, object]) -> ValidationReport:
    return full_report_bundle["report"]  # type: ignore[return-value]


@pytest.fixture(scope="module")
def full_report_html(full_report_bundle: dict[str, object]) -> str:
    return full_report_bundle["html"]  # type: ignore[return-value]


@pytest.fixture(scope="module")
def full_report_json(full_report_bundle: dict[str, object]) -> str:
    return full_report_bundle["json"]  # type: ignore[return-value]


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
        assert r.validation_category == ""
        assert r.claim_support_level == ""
        assert r.external_reference_summary == ""
        assert r.external_reference_table == []
        assert r.evidence_basis == ""
        assert r.claim_scope_note == ""


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
        assert "stated scopes and assumptions" in s

    def test_has_fail_summary(self):
        r_pass = ValidationResult("A", True, 0, 0, 0, "")
        r_fail = ValidationResult("B", False, 1, 2, 0, "")
        report = ValidationReport([r_pass, r_fail])
        s = report.summary()
        assert "FAIL" in s
        assert "1/2" in s
        assert "stated scopes and assumptions" in s

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

@pytest.mark.slow
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

    def test_run_stage_retinal_count(self, retinal_report: ValidationReport):
        report = retinal_report
        assert isinstance(report, ValidationReport)
        assert len(report.results) == 4

    def test_run_stage_e2e_count(self, suite: ValidationSuite):
        report = suite.run_stage("e2e")
        assert isinstance(report, ValidationReport)
        assert len(report.results) == 2

    def test_run_all_count(self, full_report: ValidationReport):
        report = full_report
        assert isinstance(report, ValidationReport)
        assert len(report.results) == 19

    def test_run_all_returns_validation_results(self, full_report: ValidationReport):
        report = full_report
        for r in report.results:
            assert isinstance(r, ValidationResult)

    def test_run_all_metadata_contains_provenance(self, full_report: ValidationReport):
        report = full_report
        assert report.metadata["report_type"] == "full_validation_audit"
        assert report.metadata["species"] == "human"
        assert report.metadata["scene_input_mode"] == "reflectance_under_d65"
        assert report.metadata["scene_input_is_inferred"] is True
        assert report.metadata["retinal_physiology"]["model_scope"] == "retinal_front_end_only"
        assert report.metadata["retinal_physiology"]["naka_rushton_provenance"]["confidence"] == "low"
        assert report.metadata["retinal_physiology"]["aperture_weighting"]["enabled"] is False
        assert "architecture_coverage" in report.metadata
        assert len(report.metadata["architecture_coverage"]) == 19
        assert report.metadata["stage_counts"]["optical"] == 6
        assert report.metadata["validation_category_counts"]["analytic correctness"] == 3
        assert report.metadata["validation_category_counts"]["external empirical alignment"] >= 1
        assert report.metadata["claim_support_level_counts"]["strong"] >= 1
        assert "external_reference_tables" in report.metadata
        assert "species_acuity_ranges" in report.metadata["external_reference_tables"]
        assert "density_derived_nyquist_limits" in report.metadata["external_reference_tables"]
        assert "optical_geometry_expectations" in report.metadata["external_reference_tables"]
        assert "wavelength_transmission_assumptions" in report.metadata["external_reference_tables"]
        assert "claim_calibration_notes" in report.metadata


# ---------------------------------------------------------------------------
# TestValidationSuiteTests — individual test methods
# ---------------------------------------------------------------------------

@pytest.mark.slow
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
        assert r.validation_category == "analytic correctness"
        assert r.claim_support_level == "strong"
        assert r.evidence_basis
        assert r.claim_scope_note

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
        assert r.validation_category == "external empirical alignment"
        assert r.claim_support_level == "moderate"
        assert r.external_reference_summary
        assert r.external_reference_table

    def test_snellen_acuity(self, suite: ValidationSuite):
        r = suite.test_snellen_acuity()
        assert isinstance(r, ValidationResult)
        assert r.test_name == "Snellen Acuity"
        assert r.figure is not None
        assert r.validation_category == "external empirical alignment"
        assert r.claim_support_level == "moderate"
        assert r.external_reference_summary
        assert any(row["reference_label"] == "human" for row in r.external_reference_table)

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
        assert r.validation_category == "external empirical alignment"
        assert r.claim_support_level == "strong"
        assert r.external_reference_summary
        assert any(row["reference_label"] == "dog" for row in r.external_reference_table)

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

@pytest.mark.slow
class TestHTMLOutput:
    def test_full_report_html(self, full_report: ValidationReport, full_report_html: str):
        content = full_report_html
        assert "<!DOCTYPE html>" in content
        assert "Full Validation Audit Report" in content
        assert "Summary" in content
        assert "Architecture Coverage Matrix" in content
        assert "Environment and Reproducibility" in content
        for r in full_report.results:
            assert r.test_name in content
        assert content.count("data:image/png;base64,") >= len(full_report.results) + 2

    def test_report_contains_bonus_figures(self, full_report_html: str):
        content = full_report_html
        assert "Govardovskii Nomogram" in content
        assert "Photoreceptor Mosaics" in content

    def test_report_pass_fail_badges(self, full_report_html: str):
        content = full_report_html
        assert "PASS" in content

    def test_report_summary_table(self, full_report_html: str):
        content = full_report_html
        assert "<table>" in content
        assert "<thead>" in content
        assert "Architecture" in content

    def test_all_figures_have_alt_text(self, full_report_html: str):
        content = full_report_html
        import re
        img_tags = re.findall(r"<img\s[^>]*>", content)
        for tag in img_tags:
            assert 'alt="' in tag

    def test_report_contains_transparency_sections(self, full_report_html: str):
        content = full_report_html
        assert "Pass criterion" in content
        assert "Assumptions" in content
        assert "Limitations" in content
        assert "Code references" in content
        assert "Validation category" in content
        assert "Claim support level" in content
        assert "Evidence basis" in content
        assert "Claim scope note" in content
        assert "Scene input mode" in content
        assert "RGB-inferred" in content
        assert "Retinal Physiology Assumptions" in content
        assert "Retinal Front-End Only." in content
        assert "Naka-Rushton confidence" in content
        assert "Visual streak status" in content
        assert "analytic correctness" in content
        assert "model self-consistency" in content
        assert "external empirical alignment" in content
        assert "External Reference Evidence" in content
        assert "Reference evidence table" in content

    def test_report_is_claim_calibrated(self, full_report_html: str):
        content = full_report_html
        assert "not blanket claims of whole-simulator physiological or perceptual validation" in content
        assert "A passing report supports only the implemented checks and their declared claim scopes." in content
        assert "full physiological or perceptual validation" in content

    def test_report_json_contains_result_metadata(self, full_report_json: str):
        content = full_report_json
        assert '"architecture_ref"' in content
        assert '"code_refs"' in content
        assert '"pass_criterion"' in content
        assert '"validation_category"' in content
        assert '"claim_support_level"' in content
        assert '"external_reference_summary"' in content
        assert '"external_reference_table"' in content
        assert '"evidence_basis"' in content
        assert '"claim_scope_note"' in content
        assert '"scene_input_mode": "reflectance_under_d65"' in content
        assert '"retinal_physiology"' in content
        assert '"retinal_front_end_only"' in content
        assert '"Pupil Throughput Scaling"' in content
        assert '"Cat Slit Anisotropy"' in content
        assert "pupil_area_mm2" in content
        assert "anisotropy_active" in content
        assert '"validation_category_counts"' in content
        assert '"claim_support_level_counts"' in content
        assert '"external_reference_tables"' in content
