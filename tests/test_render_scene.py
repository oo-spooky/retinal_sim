"""Tests for the ad hoc render helper script."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts import render_scene
from retinal_sim.spectral.upsampler import SpectralImage


def _workspace_tmp(name: str) -> Path:
    path = Path("tests/.tmp_render_scene") / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fake_result(input_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        metadata={
            "scene_input_mode": input_mode,
            "scene_input_assumptions": ["unit-test assumption"],
        },
        summary_metrics={
            "stimulated_receptor_count": 4.0,
            "stimulated_receptor_fraction": 1.0,
            "mean_response": 0.25,
            "left_right_cone_discriminability": 0.1,
            "center_contrast": 0.2,
            "periphery_contrast": 0.05,
        },
        artifacts={
            "retinal_irradiance_diagnostics": {
                "traceability_note": "irradiance traceability",
                "irradiance_native_px": [4, 4],
                "delivered_spectrum_summary": {"delivered_total_energy": 1.0},
                "optical_stage_summary": {"pupil_throughput_scale": 1.0},
                "selected_wavelength_slices": [
                    {
                        "id": "irradiance_slice_420nm",
                        "label": "Retinal irradiance slice (420 nm)",
                        "target_wavelength_nm": 420.0,
                        "sampled_wavelength_nm": 420.0,
                        "image_data": np.ones((4, 4), dtype=np.float32) * 0.2,
                    },
                    {
                        "id": "irradiance_slice_530nm",
                        "label": "Retinal irradiance slice (530 nm)",
                        "target_wavelength_nm": 530.0,
                        "sampled_wavelength_nm": 530.0,
                        "image_data": np.ones((4, 4), dtype=np.float32) * 0.3,
                    },
                    {
                        "id": "irradiance_slice_650nm",
                        "label": "Retinal irradiance slice (650 nm)",
                        "target_wavelength_nm": 650.0,
                        "sampled_wavelength_nm": 650.0,
                        "image_data": np.ones((4, 4), dtype=np.float32) * 0.4,
                    },
                ],
                "band_composite": {
                    "image_data": np.ones((4, 4, 3), dtype=np.float32) * 0.25,
                },
            },
            "photoreceptor_activation_diagnostics": {
                "traceability_note": "activation traceability",
                "overall_summary": {"n_receptors": 4},
                "sampling_footprint_summary": {"stimulated_receptor_count": 4},
                "retinal_physiology_summary": {"model_scope": "retinal_front_end_only"},
                "mosaic_footprint_overlay": {
                    "image_data": np.ones((4, 4, 3), dtype=np.float32) * 0.5,
                },
            },
            "comparative_renderings": {
                "scope_note": "These are not direct perceptual or cortical reconstructions.",
                "activation_render_px": [4, 4],
                "items": [
                    {
                        "id": "comparative_activation_map",
                        "label": "Comparative rendering: receptor-coded activation map",
                        "description": "Unit-test activation map.",
                        "image_data": np.ones((4, 4, 3), dtype=np.float32) * 0.6,
                    },
                    {
                        "id": "retinal_information_rendering",
                        "label": "Retinal-information rendering",
                        "description": "Unit-test retinal-information rendering.",
                        "image_data": np.ones((4, 4), dtype=np.float32) * 0.7,
                    },
                ],
            },
            "input_shape": (4, 4),
            "artifact_render_shape": (4, 4),
        },
    )


def test_crop_input_to_patch_respects_landscape_patch_limit():
    rgb = np.zeros((40, 100, 3), dtype=np.float32)
    _, crop_info = render_scene._crop_input_to_patch(
        rgb,
        scene_width_m=1.0,
        distance_m=1.0,
        patch_deg=20.0,
    )

    assert crop_info["cropped"] is True
    assert crop_info["cropped_angular_width_deg"] <= 20.0 + 1e-6
    assert crop_info["cropped_angular_height_deg"] <= 20.0 + 1e-6


def test_crop_input_to_patch_respects_portrait_patch_limit():
    rgb = np.zeros((100, 40, 3), dtype=np.float32)
    _, crop_info = render_scene._crop_input_to_patch(
        rgb,
        scene_width_m=1.0,
        distance_m=1.0,
        patch_deg=20.0,
    )

    width_limited_fraction = 20.0 / crop_info["full_angular_width_deg"]
    assert crop_info["cropped"] is True
    assert crop_info["crop_fraction"] < width_limited_fraction
    assert crop_info["cropped_angular_width_deg"] <= 20.0 + 1e-6
    assert crop_info["cropped_angular_height_deg"] <= 20.0 + 1e-6


def test_main_propagates_explicit_input_mode(monkeypatch):
    captured: dict[str, object] = {}

    class FakeSimulator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def compare_species(
            self,
            input_image,
            species_list,
            scene_width_m,
            viewing_distance_m,
            input_mode,
            artifact_render_longest_edge_px=None,
        ):
            captured["input_mode"] = input_mode
            captured["scene_width_m"] = scene_width_m
            return {species: _fake_result(input_mode) for species in species_list}

    tmp_path = _workspace_tmp("input_mode")
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "render.png"

    monkeypatch.setattr(render_scene, "_load_input", lambda path, mode: np.zeros((8, 8, 3), dtype=np.float32))
    monkeypatch.setattr(render_scene, "RetinalSimulator", FakeSimulator)
    monkeypatch.setattr(
        render_scene,
        "render_perceptual_image",
        lambda result, grid_shape: np.zeros((*grid_shape, 3), dtype=np.float32),
    )

    rc = render_scene.main(
        [
            str(input_path),
            "--input-mode",
            "display_rgb",
            "--output",
            str(output_path),
        ]
    )

    assert rc == 0
    assert captured["input_mode"] == "display_rgb"
    assert output_path.exists()


def test_main_can_write_diagnostics_bundle(monkeypatch):
    class FakeSimulator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def compare_species(
            self,
            input_image,
            species_list,
            scene_width_m,
            viewing_distance_m,
            input_mode,
            artifact_render_longest_edge_px=None,
        ):
            return {species: _fake_result(input_mode) for species in species_list}

    tmp_path = _workspace_tmp("diagnostics")
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "render.png"
    diagnostics_dir = tmp_path / "diagnostics"

    monkeypatch.setattr(render_scene, "_load_input", lambda path, mode: np.zeros((8, 8, 3), dtype=np.float32))
    monkeypatch.setattr(render_scene, "RetinalSimulator", FakeSimulator)
    monkeypatch.setattr(
        render_scene,
        "render_perceptual_image",
        lambda result, grid_shape: np.zeros((*grid_shape, 3), dtype=np.float32),
    )

    rc = render_scene.main(
        [
            str(input_path),
            "--species",
            "human",
            "--diagnostics-dir",
            str(diagnostics_dir),
            "--output",
            str(output_path),
        ]
    )

    assert rc == 0
    manifest = json.loads((diagnostics_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_metadata"]["scene_input_mode"] == "reflectance_under_d65"
    assert (diagnostics_dir / "human" / "summary.json").exists()
    assert (diagnostics_dir / "human" / "activation_overlay.png").exists()
    assert (diagnostics_dir / "human" / "irradiance_band_composite.png").exists()


def test_main_run_dir_writes_standard_bundle_and_compatibility_outputs(monkeypatch):
    class FakeSimulator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def compare_species(
            self,
            input_image,
            species_list,
            scene_width_m,
            viewing_distance_m,
            input_mode,
            artifact_render_longest_edge_px=None,
        ):
            return {species: _fake_result(input_mode) for species in species_list}

    tmp_path = _workspace_tmp("run_bundle")
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"placeholder")
    run_dir = tmp_path / "bundle"
    output_path = tmp_path / "compat_render.png"
    diagnostics_dir = tmp_path / "compat_diagnostics"

    monkeypatch.setattr(render_scene, "_load_input", lambda path, mode: np.zeros((8, 8, 3), dtype=np.float32))
    monkeypatch.setattr(render_scene, "RetinalSimulator", FakeSimulator)
    monkeypatch.setattr(
        render_scene,
        "render_perceptual_image",
        lambda result, grid_shape: np.zeros((*grid_shape, 3), dtype=np.float32),
    )

    rc = render_scene.main(
        [
            str(input_path),
            "--species",
            "human",
            "--run-dir",
            str(run_dir),
            "--output",
            str(output_path),
            "--diagnostics-dir",
            str(diagnostics_dir),
        ]
    )

    assert rc == 0
    assert (run_dir / "comparison.png").exists()
    assert (run_dir / "input_patch.png").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "index.html").exists()
    assert (run_dir / "diagnostics" / "manifest.json").exists()
    assert (run_dir / "species" / "human" / "index.html").exists()
    assert (run_dir / "species" / "human" / "summary.json").exists()
    assert (run_dir / "species" / "human" / "irradiance_band_composite.png").exists()
    assert (run_dir / "species" / "human" / "irradiance_slice_420nm.png").exists()
    assert (run_dir / "species" / "human" / "irradiance_slice_530nm.png").exists()
    assert (run_dir / "species" / "human" / "irradiance_slice_650nm.png").exists()
    assert (run_dir / "species" / "human" / "activation_overlay.png").exists()
    assert (run_dir / "species" / "human" / "activation_map.png").exists()
    assert (run_dir / "species" / "human" / "retinal_information_rendering.png").exists()
    assert output_path.exists()
    assert (diagnostics_dir / "manifest.json").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["scene_input_mode"] == "reflectance_under_d65"
    assert summary["species"] == ["human"]
    assert summary["paths"]["comparison_png"] == "comparison.png"
    assert summary["paths"]["input_patch_png"] == "input_patch.png"
    assert summary["paths"]["diagnostics_manifest_json"] == "diagnostics/manifest.json"
    assert summary["native_input_patch_px"] == [1, 1]
    assert summary["activation_render_px"] == [1, 1]
    assert summary["irradiance_native_px"] == [1, 1]
    assert summary["species_summaries"]["human"]["stimulated_receptor_count"] == 4.0
    assert summary["species_summaries"]["human"]["diagnostics"]["summary_json"] == "diagnostics/human/summary.json"
    assert summary["species_summaries"]["human"]["report_html"] == "species/human/index.html"
    assert summary["species_summaries"]["human"]["report_assets"]["activation_map_png"] == "species/human/activation_map.png"

    index_html = (run_dir / "index.html").read_text(encoding="utf-8")
    assert "retinal-front-end comparison" in summary["scope_note"]
    assert "Open human report" in index_html
    assert "retinal-information renderings" in index_html
    assert "comparison strip remains an overview" in index_html

    species_html = (run_dir / "species" / "human" / "index.html").read_text(encoding="utf-8")
    assert "Scene and patch" in species_html
    assert "Retinal irradiance" in species_html
    assert "Sampling and activation" in species_html
    assert "Retinal-information outputs" in species_html


def test_main_run_dir_writes_default_three_species_reports(monkeypatch):
    class FakeSimulator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def compare_species(
            self,
            input_image,
            species_list,
            scene_width_m,
            viewing_distance_m,
            input_mode,
            artifact_render_longest_edge_px=None,
        ):
            return {species: _fake_result(input_mode) for species in species_list}

    tmp_path = _workspace_tmp("default_species")
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"placeholder")
    run_dir = tmp_path / "bundle"

    monkeypatch.setattr(render_scene, "_load_input", lambda path, mode: np.zeros((8, 8, 3), dtype=np.float32))
    monkeypatch.setattr(render_scene, "RetinalSimulator", FakeSimulator)
    monkeypatch.setattr(
        render_scene,
        "render_perceptual_image",
        lambda result, grid_shape: np.zeros((*grid_shape, 3), dtype=np.float32),
    )

    rc = render_scene.main(
        [
            str(input_path),
            "--run-dir",
            str(run_dir),
        ]
    )

    assert rc == 0
    for species in ("human", "dog", "cat"):
        assert (run_dir / "species" / species / "index.html").exists()


def test_main_run_dir_supports_measured_spectrum(monkeypatch):
    class FakeSimulator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def compare_species(
            self,
            input_image,
            species_list,
            scene_width_m,
            viewing_distance_m,
            input_mode,
            artifact_render_longest_edge_px=None,
        ):
            assert isinstance(input_image, SpectralImage)
            return {species: _fake_result(input_mode) for species in species_list}

    tmp_path = _workspace_tmp("measured_spectrum")
    input_path = tmp_path / "input.npz"
    input_path.write_bytes(b"placeholder")
    run_dir = tmp_path / "bundle"
    spectral = SpectralImage(
        data=np.ones((8, 8, 3), dtype=np.float32),
        wavelengths=np.array([420.0, 530.0, 650.0], dtype=np.float64),
        metadata={},
    )

    monkeypatch.setattr(render_scene, "_load_input", lambda path, mode: spectral)
    monkeypatch.setattr(render_scene, "RetinalSimulator", FakeSimulator)
    monkeypatch.setattr(
        render_scene,
        "render_perceptual_image",
        lambda result, grid_shape: np.zeros((*grid_shape, 3), dtype=np.float32),
    )

    rc = render_scene.main(
        [
            str(input_path),
            "--species",
            "human",
            "--input-mode",
            "measured_spectrum",
            "--run-dir",
            str(run_dir),
        ]
    )

    assert rc == 0
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["scene_input_mode"] == "measured_spectrum"
    assert summary["scene_input_is_inferred"] is False
    assert (run_dir / "comparison.png").exists()
    assert (run_dir / "species" / "human" / "index.html").exists()


def test_main_render_longest_edge_changes_output_size_and_summary(monkeypatch):
    captured: dict[str, object] = {}

    class FakeSimulator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def compare_species(
            self,
            input_image,
            species_list,
            scene_width_m,
            viewing_distance_m,
            input_mode,
            artifact_render_longest_edge_px=None,
        ):
            captured["artifact_render_longest_edge_px"] = artifact_render_longest_edge_px
            return {species: _fake_result(input_mode) for species in species_list}

    tmp_path = _workspace_tmp("render_longest_edge")
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"placeholder")
    run_dir = tmp_path / "bundle"
    output_path = tmp_path / "render.png"

    monkeypatch.setattr(
        render_scene,
        "_load_input",
        lambda path, mode: np.zeros((8, 16, 3), dtype=np.float32),
    )
    monkeypatch.setattr(render_scene, "RetinalSimulator", FakeSimulator)
    monkeypatch.setattr(
        render_scene,
        "render_perceptual_image",
        lambda result, grid_shape: np.zeros((*grid_shape, 3), dtype=np.float32),
    )

    rc = render_scene.main(
        [
            str(input_path),
            "--species",
            "human",
            "--output",
            str(output_path),
            "--run-dir",
            str(run_dir),
            "--render-longest-edge",
            "64",
        ]
    )

    assert rc == 0
    assert captured["artifact_render_longest_edge_px"] == 64

    output_size = Image.open(output_path).size
    bundle_size = Image.open(run_dir / "comparison.png").size
    assert output_size[1] == 32
    assert bundle_size[1] == 32

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["native_input_patch_px"] == [1, 2]
    assert summary["activation_render_px"] == [32, 64]
    assert summary["irradiance_native_px"] == [1, 2]
    assert summary["patch_geometry"]["patch_deg"] == 2.0
    assert summary["species_summaries"]["human"]["report_assets"]["retinal_information_rendering_png"] == "species/human/retinal_information_rendering.png"

    manifest = json.loads((run_dir / "diagnostics" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["species"]["human"]["species_report_html"] == "../species/human/index.html"
    assert manifest["species"]["human"]["irradiance_slice_pngs"]["irradiance_slice_420nm"] == "../species/human/irradiance_slice_420nm.png"


def test_render_shape_from_longest_edge_preserves_aspect_ratio():
    assert render_scene._render_shape_from_longest_edge((8, 16), 64) == (32, 64)
    assert render_scene._render_shape_from_longest_edge((16, 8), 64) == (64, 32)
