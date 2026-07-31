import cv2
import numpy as np

from src.vision.unified_pipeline import (
    _build_wire_mask,
    _extract_skeleton_from_mask,
)


def test_build_wire_mask_preserves_wire_and_masks_component_interior():
    gray = np.full((100, 100), 255, dtype=np.uint8)
    cv2.line(gray, (5, 10), (95, 10), 0, 2)
    cv2.rectangle(gray, (40, 40), (60, 60), 0, thickness=-1)

    mask = _build_wire_mask(gray, [{"xyxy": (35, 35, 65, 65)}])

    assert mask.shape == gray.shape
    assert mask[10, 20] == 255
    assert mask[50, 50] == 0


def test_extract_skeleton_from_mask_preserves_horizontal_wire_interior():
    binary_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(binary_mask, (10, 50), (90, 50), 255, 3)

    skeleton = _extract_skeleton_from_mask(binary_mask)

    assert skeleton[50, 25] == 255
    assert skeleton[50, 75] == 255


def test_build_wire_mask_clamps_edge_and_degenerate_component_boxes():
    gray = np.full((32, 32), 255, dtype=np.uint8)
    cv2.line(gray, (1, 16), (30, 16), 0, 1)

    mask = _build_wire_mask(
        gray,
        [
            {"xyxy": (-20, -20, 4, 4)},
            {"xyxy": (10, 10, 10, 20)},
            {"xyxy": (28, 28, 60, 60)},
        ],
    )

    assert mask.shape == gray.shape
    assert mask[16, 16] == 255
