# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Runtime settings and the ``ArmBackend`` protocol for the tool-based robotic-arm subagent.

Unlike ``robotic_arm.config.SubTaskExecutor`` (two methods -- ``capture()`` and
``execute(frame, sub_task)`` -- with all perception/CV/IK/trajectory reasoning
hidden inside ``execute()`` and driven automatically by a rail), ``ArmBackend``
exposes each capability as its own method, one per LLM-callable ``Tool`` (see
``tools/``). The model itself chooses which capability to call, with what
arguments, and in what order; nothing here is invoked automatically on the
model's behalf. Implementations own all cross-call state (frame cache,
current joint/gripper readout) as instance attributes -- the framework does
not thread ``ctx`` into ``Tool.invoke()``, so ``ctx.extra`` is not available
for this purpose (see ``rails/tool_image_rail.py`` for the one place
``ctx.extra`` *is* used, to carry pending images from a tool call to the
next model call, which is a different mechanism).
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


@dataclass(frozen=True)
class CapturedFrame:
    """One RGB-D capture, as returned by :meth:`ArmBackend.capture`.

    ``image_base64`` is a resized/compressed JPEG ready to inject into a
    model message; the raw RGB/depth/point-cloud arrays stay inside the
    backend, cached by ``frame_id``, and are never sent to the model.
    """

    frame_id: str
    width: int
    height: int
    image_base64: str


@dataclass(frozen=True)
class SegmentedObject:
    """One detected object from :meth:`ArmBackend.segment` -- a pixel, not yet a 3D point.

    Deliberately 2D-only: the model must call :meth:`ArmBackend.backproject`
    (the ``pixel_to_3d`` tool) on whichever pixel it wants -- one of these,
    or any other pixel it picks by eye from the photo -- to get a 3D point.
    """

    object_id: int
    pixel_x: int
    pixel_y: int


@dataclass(frozen=True)
class SegmentationResult:
    frame_id: str
    objects: tuple[SegmentedObject, ...]
    overlay_image_base64: str


@dataclass(frozen=True)
class Point3D:
    """A point in the robot base frame, metres."""

    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class ReachabilityResult:
    reachable: bool
    ik_error_cm: float


@dataclass(frozen=True)
class MotionResult:
    success: bool
    message: str
    ee_position_m: tuple[float, float, float] | None = None
    ik_error_cm: float | None = None


@dataclass(frozen=True)
class GripperPose:
    ee_position_m: tuple[float, float, float]
    gripper_state: str  # "open" | "closed"


@dataclass(frozen=True)
class ObjectGeometry:
    """PCA over one object's masked depth points, from :meth:`ArmBackend.get_object_geometry`.

    ``dimensions_m``/``principal_axis_m`` are ordered by extent, largest first
    -- e.g. for a pen lying on the table, ``dimensions_m[0]`` is its length
    and ``principal_axis_m`` points along it. ``shape_hint`` is a coarse,
    threshold-based label -- one of ``"cylinder"``/``"box_elongated"``
    (elongated, round vs. rectangular cross-section), ``"disk"``/``"plate"``
    (flat, round vs. rectangular face), or ``"compact"`` (no dominant axis;
    PCA extents alone can't distinguish a sphere from a cube, so this is
    never split further) -- meant as a starting point for the model's own
    reasoning about elevation/roll, not a verdict.
    """

    centroid_m: tuple[float, float, float]
    dimensions_m: tuple[float, float, float]
    principal_axis_m: tuple[float, float, float]
    shape_hint: str
    point_count: int


@runtime_checkable
class ArmBackend(Protocol):
    """User-supplied hardware/algorithm backend: one method per LLM-callable tool.

    Every method here corresponds 1:1 to a ``Tool`` in ``tools/`` -- the model
    decides which to call, with what arguments, and in what order. Unlike
    ``robotic_arm.config.SubTaskExecutor``, nothing here is invoked
    automatically by a rail; there is no hidden per-turn pipeline.

    ``move_to`` is a single-shot IK solve + move -- there is no built-in
    multi-waypoint/approach-then-descend trajectory. A caller that wants a
    safer approach path composes it itself with multiple ``move_to`` calls
    (e.g. hover point, then grasp point).
    """

    def capture(self) -> CapturedFrame:
        """Trigger a fresh RGB-D capture; cache it internally keyed by ``frame_id``."""
        ...

    def segment(self, frame_id: str | None = None) -> SegmentationResult:
        """Segment the given (default: most recently captured) frame into candidate objects."""
        ...

    def get_object_geometry(
        self,
        frame_id: str | None = None,
        object_id: int | None = None,
        pixel_x: int | None = None,
        pixel_y: int | None = None,
    ) -> ObjectGeometry | None:
        """PCA over one segmented object's masked depth points.

        Reference the object either by ``object_id`` (from a prior ``segment``
        call on this frame) or by a ``pixel_x``/``pixel_y`` that falls inside
        its mask -- ``segment`` must have already run on ``frame_id`` (its
        masks are cached, not the whole point cloud re-searched blind).
        Returns ``None`` if too few of the mask's pixels have valid depth to
        compute a meaningful shape.
        """
        ...

    def backproject(self, frame_id: str | None, pixel_x: int, pixel_y: int) -> Point3D | None:
        """Backproject one pixel on ``frame_id`` to a 3D point in the robot base frame (metres).

        Returns ``None`` if the depth reading at that pixel is invalid (e.g. a
        hole in the depth map) -- never raises for that case.
        """
        ...

    def check_reachability(
        self, point_m: Sequence[float], elevation_degs: Sequence[float], roll_deg: float
    ) -> dict[float, ReachabilityResult]:
        """IK-error / reachability for one target point at each candidate approach elevation."""
        ...

    def move_to(
        self,
        point_m: Sequence[float],
        elevation_deg: float,
        roll_deg: float,
        gripper: str | None = None,
    ) -> MotionResult:
        """Solve IK for one target pose and physically move the arm there.

        ``gripper`` is ``"open"``/``"closed"``/``None`` (leave the gripper
        state unchanged). Must refuse to move (return ``success=False``,
        no hardware command sent) when the IK error exceeds the backend's
        own tolerance -- this is the one safety gate in the whole pipeline.
        """
        ...

    def set_gripper(self, state: str) -> MotionResult:
        """Open or close the gripper without moving the arm."""
        ...

    def get_gripper_pose(self) -> GripperPose:
        """Current end-effector position + gripper state.

        The model uses this to reason about a held object's position after a
        grasp (e.g. "the object moved by the same delta as the EE since I
        grasped it") -- there is no automatic rigid-keypoint-follow logic in
        this backend, unlike ``robotic_arm``'s ``_update_held_keypoints``.
        """
        ...


@dataclass
class RoboticArm2RuntimeSettings:
    """The single settings object threaded through the robotic-arm2 subagent.

    As an alternative to passing an already-constructed ``backend=``, set
    ``backend_model`` to a name registered via :class:`ArmBackendRegistry`
    (see ``registry.py``) plus ``backend_params``.
    """

    backend: ArmBackend | None = None
    backend_model: str | None = None
    backend_params: dict = field(default_factory=dict)

    health_check: bool = True

    # Optional runtime-observability hooks, mirroring robotic_arm.config's --
    # neither is required; both MUST be async callables, and exceptions they
    # raise are caught and logged by the calling rail/tool, never allowed to
    # interrupt perception/execution.
    #
    # on_frame_captured(payload): fired whenever a photo (capture_photo or
    # segment_scene's overlay) is about to be injected into the model's next
    # message. payload = {"image_base64": str, "label": str}.
    #
    # on_tool_result(payload): fired after every tool call in the agent's turn
    # (not just this package's own tools -- mirrors on_frame_captured's scope).
    # payload = {"tool_name": str, "tool_args": Any (usually a dict, but may
    # be the raw JSON string the model emitted -- see ToolCallInputs.tool_args),
    # "result_text": str (exactly what the model reads as the tool result),
    # "success": bool}.
    on_frame_captured: Callable[[dict], Awaitable[None]] | None = None
    on_tool_result: Callable[[dict], Awaitable[None]] | None = None

    photo_max_width: int = field(default_factory=lambda: _get_int("ARM2_PHOTO_MAX_WIDTH", 1280))
    photo_jpeg_quality: int = field(default_factory=lambda: _get_int("ARM2_PHOTO_JPEG_QUALITY", 85))

    mcs_screenshots_to_keep: int = field(default_factory=lambda: _get_int("ARM2_MCS_SCREENSHOTS_TO_KEEP", 3))

    context_max_message_num: int = field(default_factory=lambda: _get_int("ARM2_CONTEXT_MAX_MESSAGES", 120))
    context_default_window_round_num: int = field(default_factory=lambda: _get_int("ARM2_CONTEXT_WINDOW_ROUNDS", 20))

    @classmethod
    def from_env(cls) -> RoboticArm2RuntimeSettings:
        return cls()


__all__ = [
    "ArmBackend",
    "CapturedFrame",
    "GripperPose",
    "MotionResult",
    "ObjectGeometry",
    "Point3D",
    "ReachabilityResult",
    "RoboticArm2RuntimeSettings",
    "SegmentationResult",
    "SegmentedObject",
]
