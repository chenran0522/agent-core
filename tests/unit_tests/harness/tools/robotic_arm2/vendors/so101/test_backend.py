#!/usr/bin/env python
"""Tests for So101ArmBackend.segment/get_object_geometry -- pure PCA-over-point-cloud logic.

Builds a real ``So101ArmBackend`` with every heavy dependency (ikpy, MobileSAM,
pyrealsense2/lerobot) injected as a stub, since ``get_object_geometry`` and the
mask-caching half of ``segment`` never touch the IK solver, camera, or
segmenter -- only ``frame.points_m``/``frame.object_masks``, which are set up
directly per test.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from openjiuwen.harness.tools.robotic_arm2.vendors.so101.backend import (
    So101ArmBackend,
    _CachedFrame,
    _shape_hint,
)


def _build_backend(tmp_path) -> So101ArmBackend:
    camera_matrix = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    np.save(tmp_path / "camera_matrix.npy", camera_matrix)
    np.save(tmp_path / "depth_scale.npy", np.array([0.001]))
    np.save(tmp_path / "extrinsics.npy", np.eye(4))

    return So101ArmBackend(
        workspace_min=[-5.0, -5.0, -5.0],
        workspace_max=[5.0, 5.0, 5.0],
        camera_matrix_path=str(tmp_path / "camera_matrix.npy"),
        depth_scale_path=str(tmp_path / "depth_scale.npy"),
        extrinsics_path=str(tmp_path / "extrinsics.npy"),
        ik_solver=MagicMock(),
        camera=MagicMock(),
        segmenter=MagicMock(),
    )


def _line_mask_and_points(h: int, w: int, row: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    """A row of ``length`` points spaced 1cm apart along +x -- an elongated cluster."""
    points_m = np.full((h, w, 3), np.nan)
    mask = np.zeros((h, w), dtype=bool)
    for i in range(length):
        points_m[row, i] = [i * 0.01, 0.0, 0.0]
        mask[row, i] = True
    return mask, points_m


class TestShapeHint:
    def test_cylinder_elongated_with_round_cross_section(self) -> None:
        assert _shape_hint(np.array([0.30, 0.03, 0.028])) == "cylinder"

    def test_box_elongated_with_rectangular_cross_section(self) -> None:
        assert _shape_hint(np.array([0.30, 0.03, 0.01])) == "box_elongated"

    def test_disk_flat_with_round_face(self) -> None:
        assert _shape_hint(np.array([0.10, 0.095, 0.01])) == "disk"

    def test_plate_flat_with_rectangular_face(self) -> None:
        assert _shape_hint(np.array([0.10, 0.05, 0.01])) == "plate"

    def test_compact(self) -> None:
        assert _shape_hint(np.array([0.05, 0.045, 0.04])) == "compact"


class TestGetObjectGeometry:
    def test_measures_an_elongated_object_by_object_id(self, tmp_path) -> None:
        backend = _build_backend(tmp_path)
        mask, points_m = _line_mask_and_points(h=20, w=20, row=5, length=10)
        frame = _CachedFrame(
            rgb=np.zeros((20, 20, 3), dtype=np.uint8),
            depth_raw=np.zeros((20, 20), dtype=np.uint16),
            points_m=points_m,
        )
        frame.object_masks[0] = mask
        backend._frames["f1"] = frame

        geometry = backend.get_object_geometry("f1", object_id=0)

        assert geometry is not None
        assert geometry.point_count == 10
        # the two non-dominant axes are exactly zero-variance here (a perfect line), so
        # cross-section roundness is a numerical edge case -- assert the family, not the exact label.
        assert geometry.shape_hint in ("cylinder", "box_elongated")
        assert geometry.dimensions_m[0] > geometry.dimensions_m[1] >= geometry.dimensions_m[2]
        # principal axis should point along x (the line's direction), sign-agnostic
        assert abs(geometry.principal_axis_m[0]) == pytest.approx(1.0, abs=1e-6)

    def test_resolves_object_by_containing_pixel(self, tmp_path) -> None:
        backend = _build_backend(tmp_path)
        mask, points_m = _line_mask_and_points(h=20, w=20, row=5, length=10)
        frame = _CachedFrame(
            rgb=np.zeros((20, 20, 3), dtype=np.uint8),
            depth_raw=np.zeros((20, 20), dtype=np.uint16),
            points_m=points_m,
        )
        frame.object_masks[3] = mask
        backend._frames["f1"] = frame

        geometry = backend.get_object_geometry("f1", pixel_x=2, pixel_y=5)

        assert geometry is not None
        assert geometry.point_count == 10

    def test_returns_none_for_too_few_valid_points(self, tmp_path) -> None:
        backend = _build_backend(tmp_path)
        points_m = np.full((10, 10, 3), np.nan)
        mask = np.zeros((10, 10), dtype=bool)
        points_m[0, 0] = [0.0, 0.0, 0.0]
        mask[0, 0] = True
        frame = _CachedFrame(
            rgb=np.zeros((10, 10, 3), dtype=np.uint8), depth_raw=np.zeros((10, 10), dtype=np.uint16), points_m=points_m
        )
        frame.object_masks[0] = mask
        backend._frames["f1"] = frame

        assert backend.get_object_geometry("f1", object_id=0) is None

    def test_unknown_object_id_raises(self, tmp_path) -> None:
        backend = _build_backend(tmp_path)
        mask, points_m = _line_mask_and_points(h=10, w=10, row=0, length=5)
        frame = _CachedFrame(
            rgb=np.zeros((10, 10, 3), dtype=np.uint8), depth_raw=np.zeros((10, 10), dtype=np.uint16), points_m=points_m
        )
        frame.object_masks[0] = mask
        backend._frames["f1"] = frame

        with pytest.raises(ValueError, match="unknown object_id"):
            backend.get_object_geometry("f1", object_id=99)

    def test_requires_segment_scene_first(self, tmp_path) -> None:
        backend = _build_backend(tmp_path)
        frame = _CachedFrame(
            rgb=np.zeros((10, 10, 3), dtype=np.uint8),
            depth_raw=np.zeros((10, 10), dtype=np.uint16),
            points_m=np.full((10, 10, 3), np.nan),
        )
        backend._frames["f1"] = frame

        with pytest.raises(ValueError, match="segment_scene must be called"):
            backend.get_object_geometry("f1", object_id=0)

    def test_pixel_outside_every_mask_raises(self, tmp_path) -> None:
        backend = _build_backend(tmp_path)
        mask, points_m = _line_mask_and_points(h=10, w=10, row=0, length=5)
        frame = _CachedFrame(
            rgb=np.zeros((10, 10, 3), dtype=np.uint8), depth_raw=np.zeros((10, 10), dtype=np.uint16), points_m=points_m
        )
        frame.object_masks[0] = mask
        backend._frames["f1"] = frame

        with pytest.raises(ValueError, match="not inside any object mask"):
            backend.get_object_geometry("f1", pixel_x=9, pixel_y=9)

    def test_requires_object_id_or_pixel(self, tmp_path) -> None:
        backend = _build_backend(tmp_path)
        mask, points_m = _line_mask_and_points(h=10, w=10, row=0, length=5)
        frame = _CachedFrame(
            rgb=np.zeros((10, 10, 3), dtype=np.uint8), depth_raw=np.zeros((10, 10), dtype=np.uint16), points_m=points_m
        )
        frame.object_masks[0] = mask
        backend._frames["f1"] = frame

        with pytest.raises(ValueError, match="either object_id"):
            backend.get_object_geometry("f1")


class TestSegmentCachesMasks:
    def test_segment_caches_masks_keyed_by_renumbered_object_id(self, tmp_path) -> None:
        backend = _build_backend(tmp_path)
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        depth_raw = np.full((20, 20), 1000, dtype=np.uint16)  # uniform valid depth
        points_m = np.zeros((20, 20, 3))
        # keep the workspace filter happy: every point at the origin, inside [-1,1]^3
        frame = _CachedFrame(rgb=rgb, depth_raw=depth_raw, points_m=points_m)
        backend._frames["f1"] = frame

        mask_a = np.zeros((20, 20), dtype=bool)
        mask_a[1, 1] = True
        mask_b = np.zeros((20, 20), dtype=bool)
        mask_b[2, 2] = True
        backend._segmenter.segment.return_value = [((1, 1), mask_a), ((2, 2), mask_b)]

        result = backend.segment("f1")

        assert [obj.object_id for obj in result.objects] == [0, 1]
        assert frame.object_masks[0] is mask_a
        assert frame.object_masks[1] is mask_b
