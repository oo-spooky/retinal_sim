"""Tests for the ad hoc render helper script."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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
                "delivered_spectrum_summary": {"delivered_total_energy": 1.0},
                "optical_stage_summary": {"pupil_throughput_scale": 1.0},
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
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "index.html").exists()
    assert (run_dir / "diagnostics" / "manifest.json").exists()
    assert output_path.exists()
    assert (diagnostics_dir / "manifest.json").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["scene_input_mode"] == "reflectance_under_d65"
    assert summary["species"] == ["human"]
    assert summary["paths"]["comparison_png"] == "comparison.png"
    assert summary["paths"]["diagnostics_manifest_json"] == "diagnostics/manifest.json"
    assert summary["species_summaries"]["human"]["stimulated_receptor_count"] == 4.0
    assert summary["species_summaries"]["human"]["diagnostics"]["summary_json"] == "diagnostics/human/summary.json"

    index_html = (run_dir / "index.html").read_text(encoding="utf-8")
    assert "retinal-front-end comparison" in index_html
    assert "retinal-information renderings" in index_html
    assert "not direct conscious-perception" in index_html


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
