#!/usr/bin/env python
"""Tests for the 8 robotic-arm2 tools against a fake in-memory ArmBackend.

Each tool is a thin adapter: parse inputs -> call one ``ArmBackend`` method ->
shape the result into ``ToolOutput.data["content"]`` (text the model reads)
plus, for the perception tools, ``ToolOutput.data["images"]`` (never
stringified into the model-visible text -- see ``rails/tool_image_rail.py``).
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from openjiuwen.harness.tools.robotic_arm2.config import (
    CapturedFrame,
    GripperPose,
    MotionResult,
    ObjectGeometry,
    Point3D,
    ReachabilityResult,
    SegmentationResult,
    SegmentedObject,
)
from openjiuwen.harness.tools.robotic_arm2.tools.geometry import GetObjectGeometryTool
from openjiuwen.harness.tools.robotic_arm2.tools.motion import GetGripperPoseTool, MoveToTool, SetGripperTool
from openjiuwen.harness.tools.robotic_arm2.tools.perception import (
    CapturePhotoTool,
    PixelTo3DTool,
    SegmentSceneTool,
)
from openjiuwen.harness.tools.robotic_arm2.tools.reachability import CheckReachabilityTool


class FakeArmBackend:
    """Deterministic stand-in for :class:`ArmBackend`; every method is scriptable."""

    def __init__(self) -> None:
        self.capture_calls = 0
        self.move_to_calls: list[tuple] = []
        self.set_gripper_calls: list[str] = []
        self.get_object_geometry_calls: list[tuple] = []
        self.backproject_result: Point3D | None = Point3D(x=0.1, y=-0.02, z=0.05)
        self.move_to_result = MotionResult(
            success=True, message="Moved.", ee_position_m=(0.1, -0.02, 0.05), ik_error_cm=1.2
        )
        self.set_gripper_result = MotionResult(
            success=True, message="Gripper closed.", ee_position_m=(0.1, -0.02, 0.05)
        )
        self.gripper_pose = GripperPose(ee_position_m=(0.1, -0.02, 0.05), gripper_state="open")
        self.object_geometry_result: ObjectGeometry | None = ObjectGeometry(
            centroid_m=(0.1, -0.02, 0.05),
            dimensions_m=(0.15, 0.03, 0.02),
            principal_axis_m=(1.0, 0.0, 0.0),
            shape_hint="cylinder",
            point_count=42,
        )

    def capture(self) -> CapturedFrame:
        self.capture_calls += 1
        return CapturedFrame(frame_id=f"f{self.capture_calls}", width=640, height=480, image_base64="AAAA")

    def segment(self, frame_id: str | None = None) -> SegmentationResult:
        resolved = frame_id or "f1"
        return SegmentationResult(
            frame_id=resolved,
            objects=(SegmentedObject(object_id=0, pixel_x=100, pixel_y=200),),
            overlay_image_base64="BBBB",
        )

    def get_object_geometry(
        self,
        frame_id: str | None = None,
        object_id: int | None = None,
        pixel_x: int | None = None,
        pixel_y: int | None = None,
    ) -> ObjectGeometry | None:
        self.get_object_geometry_calls.append((frame_id, object_id, pixel_x, pixel_y))
        return self.object_geometry_result

    def backproject(self, frame_id: str | None, pixel_x: int, pixel_y: int) -> Point3D | None:
        return self.backproject_result

    def check_reachability(
        self, point_m: Sequence[float], elevation_degs: Sequence[float], roll_deg: float
    ) -> dict[float, ReachabilityResult]:
        return {
            float(e): ReachabilityResult(reachable=e >= 45, ik_error_cm=1.0 if e >= 45 else 9.0) for e in elevation_degs
        }

    def move_to(self, point_m, elevation_deg, roll_deg, gripper=None) -> MotionResult:
        self.move_to_calls.append((tuple(point_m), elevation_deg, roll_deg, gripper))
        return self.move_to_result

    def set_gripper(self, state: str) -> MotionResult:
        self.set_gripper_calls.append(state)
        return self.set_gripper_result

    def get_gripper_pose(self) -> GripperPose:
        return self.gripper_pose


@pytest.fixture()
def backend() -> FakeArmBackend:
    return FakeArmBackend()


# -- capture_photo -----------------------------------------------------------


@pytest.mark.asyncio
async def test_capture_photo_returns_frame_id_and_image(backend: FakeArmBackend) -> None:
    tool = CapturePhotoTool(backend)

    result = await tool.invoke({})

    assert result.success is True
    assert "f1" in result.data["content"]
    assert result.data["frame_id"] == "f1"
    assert result.data["images"] == [{"label": "Photo (f1)", "image_base64": "AAAA"}]


@pytest.mark.asyncio
async def test_capture_photo_wraps_backend_exception(backend: FakeArmBackend) -> None:
    def _boom():
        raise RuntimeError("camera offline")

    backend.capture = _boom
    tool = CapturePhotoTool(backend)

    result = await tool.invoke({})

    assert result.success is False
    assert "camera offline" in result.error


# -- segment_scene -------------------------------------------------------------


@pytest.mark.asyncio
async def test_segment_scene_lists_objects_and_overlay(backend: FakeArmBackend) -> None:
    tool = SegmentSceneTool(backend)

    result = await tool.invoke({"frame_id": "f1"})

    assert result.success is True
    assert "id=0 pixel=(100, 200)" in result.data["content"]
    assert result.data["images"][0]["image_base64"] == "BBBB"


@pytest.mark.asyncio
async def test_segment_scene_reports_no_objects() -> None:
    backend = FakeArmBackend()
    backend.segment = lambda frame_id=None: SegmentationResult(frame_id="f1", objects=(), overlay_image_base64="BBBB")
    tool = SegmentSceneTool(backend)

    result = await tool.invoke({})

    assert "No objects detected" in result.data["content"]


# -- pixel_to_3d ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_pixel_to_3d_returns_point(backend: FakeArmBackend) -> None:
    tool = PixelTo3DTool(backend)

    result = await tool.invoke({"frame_id": "f1", "pixel_x": 100, "pixel_y": 200})

    assert result.success is True
    assert result.data["point_m"] == (0.1, -0.02, 0.05)


@pytest.mark.asyncio
async def test_pixel_to_3d_handles_invalid_depth(backend: FakeArmBackend) -> None:
    backend.backproject_result = None
    tool = PixelTo3DTool(backend)

    result = await tool.invoke({"pixel_x": 1, "pixel_y": 1})

    assert result.success is True
    assert "no valid depth" in result.data["content"]


@pytest.mark.asyncio
async def test_pixel_to_3d_requires_pixel_coords(backend: FakeArmBackend) -> None:
    tool = PixelTo3DTool(backend)

    result = await tool.invoke({"pixel_x": 1})

    assert result.success is False


# -- get_object_geometry ----------------------------------------------------------


@pytest.mark.asyncio
async def test_get_object_geometry_by_object_id(backend: FakeArmBackend) -> None:
    tool = GetObjectGeometryTool(backend)

    result = await tool.invoke({"frame_id": "f1", "object_id": 0})

    assert result.success is True
    assert backend.get_object_geometry_calls == [("f1", 0, None, None)]
    assert result.data["shape_hint"] == "cylinder"
    assert result.data["dimensions_m"] == (0.15, 0.03, 0.02)
    assert "cylinder" in result.data["content"]


@pytest.mark.asyncio
async def test_get_object_geometry_by_pixel(backend: FakeArmBackend) -> None:
    tool = GetObjectGeometryTool(backend)

    result = await tool.invoke({"pixel_x": 10, "pixel_y": 20})

    assert result.success is True
    assert backend.get_object_geometry_calls == [(None, None, 10, 20)]


@pytest.mark.asyncio
async def test_get_object_geometry_requires_object_id_or_pixel(backend: FakeArmBackend) -> None:
    tool = GetObjectGeometryTool(backend)

    result = await tool.invoke({})

    assert result.success is False
    assert backend.get_object_geometry_calls == []


@pytest.mark.asyncio
async def test_get_object_geometry_handles_none_result(backend: FakeArmBackend) -> None:
    backend.object_geometry_result = None
    tool = GetObjectGeometryTool(backend)

    result = await tool.invoke({"object_id": 0})

    assert result.success is True
    assert "Not enough valid depth points" in result.data["content"]


@pytest.mark.asyncio
async def test_get_object_geometry_wraps_backend_exception(backend: FakeArmBackend) -> None:
    def _boom(*args, **kwargs):
        raise ValueError("segment_scene must be called on this frame first")

    backend.get_object_geometry = _boom
    tool = GetObjectGeometryTool(backend)

    result = await tool.invoke({"object_id": 0})

    assert result.success is False
    assert "segment_scene must be called" in result.error


# -- check_reachability -----------------------------------------------------------


@pytest.mark.asyncio
async def test_check_reachability_defaults_elevations(backend: FakeArmBackend) -> None:
    tool = CheckReachabilityTool(backend)

    result = await tool.invoke({"point_m": [0.1, 0.0, 0.05]})

    assert result.success is True
    assert "elevation=90.0deg" in result.data["content"]
    assert "UNREACHABLE" in result.data["content"]  # elevation=0 scripted unreachable
    assert "OK" in result.data["content"]


@pytest.mark.asyncio
async def test_check_reachability_rejects_bad_point(backend: FakeArmBackend) -> None:
    tool = CheckReachabilityTool(backend)

    result = await tool.invoke({"point_m": [0.1, 0.0]})

    assert result.success is False


# -- move_to ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_move_to_success_passes_args_through(backend: FakeArmBackend) -> None:
    tool = MoveToTool(backend)

    result = await tool.invoke(
        {"point_m": [0.1, -0.02, 0.05], "elevation_deg": 90, "roll_deg": 45, "gripper": "closed"}
    )

    assert result.success is True
    assert backend.move_to_calls == [((0.1, -0.02, 0.05), 90.0, 45.0, "closed")]


@pytest.mark.asyncio
async def test_move_to_surfaces_refusal_as_error(backend: FakeArmBackend) -> None:
    backend.move_to_result = MotionResult(success=False, message="IK error too large", ik_error_cm=9.0)
    tool = MoveToTool(backend)

    result = await tool.invoke({"point_m": [1.0, 1.0, 1.0], "elevation_deg": 90})

    assert result.success is False
    assert "IK error too large" in result.error


@pytest.mark.asyncio
async def test_move_to_requires_point_and_elevation(backend: FakeArmBackend) -> None:
    tool = MoveToTool(backend)

    result = await tool.invoke({"point_m": [0.1, 0.0, 0.0]})

    assert result.success is False


# -- set_gripper / get_gripper_pose ------------------------------------------------


@pytest.mark.asyncio
async def test_set_gripper_valid_state(backend: FakeArmBackend) -> None:
    tool = SetGripperTool(backend)

    result = await tool.invoke({"state": "closed"})

    assert result.success is True
    assert backend.set_gripper_calls == ["closed"]


@pytest.mark.asyncio
async def test_set_gripper_rejects_invalid_state(backend: FakeArmBackend) -> None:
    tool = SetGripperTool(backend)

    result = await tool.invoke({"state": "half-open"})

    assert result.success is False


@pytest.mark.asyncio
async def test_get_gripper_pose_reports_state(backend: FakeArmBackend) -> None:
    tool = GetGripperPoseTool(backend)

    result = await tool.invoke({})

    assert result.success is True
    assert "gripper=open" in result.data["content"]
    assert result.data["ee_position_m"] == (0.1, -0.02, 0.05)
