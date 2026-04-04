"""Deterministic validation datasets used by reports and tests."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def _rgb_triplet(values: List[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.uint8)


VALIDATION_DATASET: Dict[str, object] = {
    "metamer_pairs": [
        {
            "name": "cyan_blue_pair",
            "rgb_a": [13, 150, 234],
            "rgb_b": [52, 148, 235],
        },
        {
            "name": "teal_pair",
            "rgb_a": [70, 192, 203],
            "rgb_b": [106, 189, 203],
        },
        {
            "name": "green_pair",
            "rgb_a": [61, 245, 158],
            "rgb_b": [24, 249, 158],
        },
        {
            "name": "violet_pair",
            "rgb_a": [127, 41, 209],
            "rgb_b": [101, 60, 207],
        },
    ],
    "control_pairs": [
        {"name": "blue_yellow", "rgb_a": [30, 80, 220], "rgb_b": [220, 210, 20]},
        {"name": "bright_dark", "rgb_a": [220, 220, 220], "rgb_b": [20, 20, 20]},
        {"name": "cyan_orange", "rgb_a": [30, 180, 210], "rgb_b": [220, 120, 20]},
        {"name": "purple_lime", "rgb_a": [150, 40, 220], "rgb_b": [170, 230, 30]},
    ],
    "confusion_pairs": {
        "dog": [
            {"name": "dog_conf_0", "rgb_a": [229, 101, 39], "rgb_b": [41, 159, 50]},
            {"name": "dog_conf_1", "rgb_a": [229, 113, 37], "rgb_b": [44, 163, 49]},
            {"name": "dog_conf_2", "rgb_a": [224, 119, 14], "rgb_b": [79, 165, 30]},
            {"name": "dog_conf_3", "rgb_a": [216, 171, 17], "rgb_b": [57, 202, 42]},
            {"name": "dog_conf_4", "rgb_a": [224, 168, 26], "rgb_b": [51, 201, 48]},
            {"name": "dog_conf_5", "rgb_a": [226, 59, 26], "rgb_b": [44, 143, 35]},
            {"name": "dog_conf_6", "rgb_a": [224, 84, 39], "rgb_b": [62, 146, 46]},
            {"name": "dog_conf_7", "rgb_a": [226, 186, 6], "rgb_b": [44, 218, 44]},
            {"name": "dog_conf_8", "rgb_a": [226, 129, 28], "rgb_b": [45, 174, 44]},
            {"name": "dog_conf_9", "rgb_a": [226, 55, 14], "rgb_b": [94, 129, 22]},
        ],
        "cat": [
            {"name": "cat_conf_0", "rgb_a": [226, 120, 31], "rgb_b": [47, 167, 4]},
            {"name": "cat_conf_1", "rgb_a": [225, 183, 29], "rgb_b": [71, 213, 8]},
            {"name": "cat_conf_2", "rgb_a": [225, 165, 34], "rgb_b": [56, 201, 0]},
            {"name": "cat_conf_3", "rgb_a": [223, 66, 44], "rgb_b": [51, 137, 24]},
            {"name": "cat_conf_4", "rgb_a": [222, 122, 37], "rgb_b": [42, 168, 15]},
            {"name": "cat_conf_5", "rgb_a": [228, 147, 45], "rgb_b": [50, 187, 28]},
            {"name": "cat_conf_6", "rgb_a": [219, 193, 27], "rgb_b": [55, 219, 11]},
            {"name": "cat_conf_7", "rgb_a": [225, 150, 37], "rgb_b": [104, 182, 21]},
            {"name": "cat_conf_8", "rgb_a": [228, 106, 45], "rgb_b": [61, 157, 28]},
            {"name": "cat_conf_9", "rgb_a": [227, 191, 35], "rgb_b": [45, 224, 5]},
        ],
    },
    "spectral_response_panel": [
        {"name": "violet", "rgb": [120, 40, 220]},
        {"name": "blue", "rgb": [40, 70, 230]},
        {"name": "cyan", "rgb": [30, 190, 220]},
        {"name": "green", "rgb": [40, 210, 60]},
        {"name": "yellow", "rgb": [230, 220, 30]},
        {"name": "amber", "rgb": [240, 150, 20]},
        {"name": "red", "rgb": [220, 40, 30]},
        {"name": "blue_green_mix", "rgb": [20, 150, 160]},
        {"name": "green_red_mix", "rgb": [180, 160, 20]},
    ],
}


def metamer_pairs() -> List[dict]:
    return list(VALIDATION_DATASET["metamer_pairs"])


def control_pairs() -> List[dict]:
    return list(VALIDATION_DATASET["control_pairs"])


def confusion_pairs(species: str) -> List[dict]:
    return list(VALIDATION_DATASET["confusion_pairs"][species])


def spectral_response_panel() -> List[dict]:
    return list(VALIDATION_DATASET["spectral_response_panel"])


def as_uint8_pair(item: dict) -> tuple[np.ndarray, np.ndarray]:
    return _rgb_triplet(item["rgb_a"]), _rgb_triplet(item["rgb_b"])


def as_uint8_color(item: dict) -> np.ndarray:
    return _rgb_triplet(item["rgb"])
