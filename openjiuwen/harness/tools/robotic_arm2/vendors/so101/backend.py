# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""``ArmBackend`` implementation for a real SO-101 + RealSense rig.

Registered as ``"so101"``. Owns all hardware I/O (RealSense capture, arm/gripper
drive via ``lerobot``) plus the pure-vision/geometry pieces (segmentation, IK) --
one method per :class:`~openjiuwen.harness.tools.robotic_arm2.config.ArmBackend`
capability, each backing exactly one LLM-callable tool in ``tools/``. Unlike
``robotic_arm.vendors.so101.rekep.executor.So101RekepExecutor``, there is no
``execute()`` that runs a fixed multi-step pipeline per call -- the model
decides the call sequence, and this class only ever does exactly what one
tool call asks for.

Cross-call state (the frame cache, in particular) lives on instance
attributes because the framework does not thread ``ctx`` into ``Tool.invoke()``
(see ``config.py``'s module docstring).
"""

from __future__ import annotations

import base64
import io
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import ToolError
from openjiuwen.core.common.logging import tool_logger
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
from openjiuwen.harness.tools.robotic_arm2.registry import ArmBackendRegistry
from openjiuwen.harness.tools.robotic_arm2.vendors.so101.kinematics import (
    IKSolver,
    approach_from_elevation,
    backproject_frame,
    backproject_pixel,
    in_workspace,
)
from openjiuwen.harness.tools.robotic_arm2.vendors.so101.kinematics import (
    check_reachability as _kinematics_check_reachability,
)
from openjiuwen.harness.tools.robotic_arm2.vendors.so101.perception import (
    So101Camera,
    So101Segmenter,
    draw_numbered_overlay,
)

_GRIPPER_STATES = ("open", "closed")


@dataclass
class _CachedFrame:
    rgb: np.ndarray
    depth_raw: np.ndarray
    points_m: np.ndarray  # [H, W, 3], robot base frame, NaN where depth is invalid
    # object_id -> boolean mask [H, W], populated by segment() and consumed by
    # get_object_geometry(); reset (not merged) on every segment() call since
    # object ids are renumbered each time.
    object_masks: dict[int, np.ndarray] = field(default_factory=dict)


def _roughly_equal(x: float, y: float, ratio: float = 1.3) -> bool:
    larger, smaller = max(x, y), min(x, y)
    return larger <= ratio * smaller


def _shape_hint(dimensions_m: np.ndarray) -> str:
    """Coarse label from PCA extents (largest, mid, smallest) -- a starting point for the
    model's own elevation/roll reasoning, not a verdict. Thresholds are simple ratios, not
    calibrated against any dataset.

    Six labels, not three: beyond "how dominant is one axis" (elongated/flat/compact),
    also checks whether the OTHER two axes are close to each other, i.e. whether the
    object's cross-section (for elongated) or flat face (for flat) is round or
    rectangular -- this matters for roll_deg (a round cross-section is roll-agnostic to
    grip; a rectangular one isn't). PCA extents alone cannot tell a sphere from a cube of
    the same bounding box, so "compact" is never split further.
    """
    largest, mid, smallest = (float(v) for v in dimensions_m)
    if largest > 3.0 * mid:
        # elongated -- cross-section is spanned by the two smaller axes (mid, smallest)
        return "cylinder" if _roughly_equal(mid, smallest) else "box_elongated"
    if smallest < 0.3 * mid:
        # flat -- the flat face is spanned by the two larger axes (largest, mid)
        return "disk" if _roughly_equal(largest, mid) else "plate"
    return "compact"


def _resize_and_encode_jpeg(rgb: np.ndarray, *, max_width: int, jpeg_quality: int) -> tuple[np.ndarray, str]:
    """Resize to ``max_width`` (if wider) and JPEG-encode; returns ``(possibly-resized rgb, base64 str)``."""
    img = Image.fromarray(rgb)
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return np.array(img), base64.b64encode(buf.getvalue()).decode("utf-8")


@ArmBackendRegistry.register("so101")
class So101ArmBackend:
    """SO-101 hardware I/O + SAM segmentation + IK, one method per tool."""

    def __init__(
        self,
        *,
        workspace_min: Sequence[float],
        workspace_max: Sequence[float],
        camera_matrix_path: str | None = None,
        depth_scale_path: str | None = None,
        extrinsics_path: str | None = None,
        urdf_path: str | None = None,
        ik_solver: IKSolver | None = None,
        tcp_offset_m: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        port: str | None = None,
        robot: Any | None = None,
        gripper_open_value: float = 50.0,
        gripper_closed_value: float = 0.0,
        ik_tolerance_cm: float = 5.0,
        color_width: int = 1280,
        color_height: int = 720,
        fps: int = 6,
        frame_timeout_ms: int = 5000,
        pre_capture_hook: Callable[[], None] | None = None,
        sam_checkpoint_path: str | None = None,
        sam_device: str | None = None,
        min_mask_pixels: int = 300,
        photo_max_width: int = 1280,
        photo_jpeg_quality: int = 85,
        max_cached_frames: int = 5,
        camera: So101Camera | None = None,
        segmenter: So101Segmenter | None = None,
    ) -> None:
        if ik_solver is not None:
            self._ik = ik_solver
        elif urdf_path is not None:
            self._ik = IKSolver(urdf_path, tcp_offset_m)
        else:
            raise ValueError("either ik_solver or urdf_path is required")

        if camera_matrix_path is None or depth_scale_path is None or extrinsics_path is None:
            raise ValueError("camera_matrix_path/depth_scale_path/extrinsics_path are required")
        self._camera_matrix = np.load(camera_matrix_path)
        self._depth_scale = float(np.load(depth_scale_path)[0])
        self._extrinsics_cm = np.load(extrinsics_path)
        self._workspace_min = np.asarray(workspace_min, dtype=float)
        self._workspace_max = np.asarray(workspace_max, dtype=float)

        self._port = port
        self._robot = robot
        self._gripper_open_value = gripper_open_value
        self._gripper_closed_value = gripper_closed_value
        self._ik_tolerance_cm = ik_tolerance_cm
        self._photo_max_width = photo_max_width
        self._photo_jpeg_quality = photo_jpeg_quality

        self._camera = camera or So101Camera(
            color_width=color_width,
            color_height=color_height,
            fps=fps,
            frame_timeout_ms=frame_timeout_ms,
            pre_capture_hook=pre_capture_hook,
        )
        if segmenter is not None:
            self._segmenter = segmenter
        else:
            if sam_checkpoint_path is None:
                raise ValueError("sam_checkpoint_path is required unless segmenter is injected")
            self._segmenter = So101Segmenter(
                sam_checkpoint_path=sam_checkpoint_path,
                device=sam_device,
                min_mask_pixels=min_mask_pixels,
            )

        self._frames: OrderedDict[str, _CachedFrame] = OrderedDict()
        self._frame_counter = 0
        self._max_cached_frames = max_cached_frames

    # -- ArmBackend protocol -----------------------------------------------

    def capture(self) -> CapturedFrame:
        rgb, depth_raw = self._camera.retry_capture()
        points_m = backproject_frame(depth_raw, self._camera_matrix, self._depth_scale, self._extrinsics_cm)

        self._frame_counter += 1
        frame_id = f"f{self._frame_counter}"
        self._frames[frame_id] = _CachedFrame(rgb=rgb, depth_raw=depth_raw, points_m=points_m)
        while len(self._frames) > self._max_cached_frames:
            self._frames.popitem(last=False)

        _, image_base64 = _resize_and_encode_jpeg(
            rgb, max_width=self._photo_max_width, jpeg_quality=self._photo_jpeg_quality
        )
        return CapturedFrame(frame_id=frame_id, width=rgb.shape[1], height=rgb.shape[0], image_base64=image_base64)

    def segment(self, frame_id: str | None = None) -> SegmentationResult:
        resolved_id, frame = self._resolve_frame(frame_id)

        raw = self._segmenter.segment(frame.rgb)
        kept: list[tuple[tuple[int, int], np.ndarray]] = []
        for (px, py), mask in raw:
            point = backproject_pixel(
                px,
                py,
                frame.depth_raw,
                camera_matrix=self._camera_matrix,
                depth_scale=self._depth_scale,
                extrinsics_cm=self._extrinsics_cm,
            )
            if point is not None and in_workspace(point, self._workspace_min, self._workspace_max):
                kept.append(((px, py), mask))

        # Renumbered every call -- object ids only make sense relative to this segment() result.
        frame.object_masks = {i: mask for i, (_, mask) in enumerate(kept)}

        overlay = draw_numbered_overlay(frame.rgb, [pixel for pixel, _ in kept])
        _, overlay_base64 = _resize_and_encode_jpeg(
            overlay, max_width=self._photo_max_width, jpeg_quality=self._photo_jpeg_quality
        )
        objects = tuple(SegmentedObject(object_id=i, pixel_x=px, pixel_y=py) for i, ((px, py), _) in enumerate(kept))
        return SegmentationResult(frame_id=resolved_id, objects=objects, overlay_image_base64=overlay_base64)

    def get_object_geometry(
        self,
        frame_id: str | None = None,
        object_id: int | None = None,
        pixel_x: int | None = None,
        pixel_y: int | None = None,
    ) -> ObjectGeometry | None:
        _, frame = self._resolve_frame(frame_id)
        mask = self._resolve_object_mask(frame, object_id, pixel_x, pixel_y)

        points = frame.points_m[mask]
        valid = points[~np.isnan(points).any(axis=1)]
        if len(valid) < 3:
            return None

        centroid = valid.mean(axis=0)
        centered = valid - centroid
        cov = (centered.T @ centered) / len(centered)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        projected = centered @ eigvecs
        dimensions = projected.max(axis=0) - projected.min(axis=0)

        return ObjectGeometry(
            centroid_m=tuple(float(v) for v in centroid),
            dimensions_m=tuple(float(v) for v in dimensions),
            principal_axis_m=tuple(float(v) for v in eigvecs[:, 0]),
            shape_hint=_shape_hint(dimensions),
            point_count=len(valid),
        )

    def _resolve_object_mask(
        self, frame: _CachedFrame, object_id: int | None, pixel_x: int | None, pixel_y: int | None
    ) -> np.ndarray:
        if not frame.object_masks:
            raise ValueError("segment_scene must be called on this frame before get_object_geometry")
        if object_id is not None:
            mask = frame.object_masks.get(object_id)
            if mask is None:
                raise ValueError(f"unknown object_id {object_id!r}; known ids: {sorted(frame.object_masks)}")
            return mask
        if pixel_x is not None and pixel_y is not None:
            for mask in frame.object_masks.values():
                if mask[pixel_y, pixel_x]:
                    return mask
            raise ValueError(
                f"pixel ({pixel_x}, {pixel_y}) is not inside any object mask from the last segment_scene call"
            )
        raise ValueError("either object_id, or both pixel_x and pixel_y, are required")

    def backproject(self, frame_id: str | None, pixel_x: int, pixel_y: int) -> Point3D | None:
        _, frame = self._resolve_frame(frame_id)
        if not (0 <= pixel_x < frame.rgb.shape[1] and 0 <= pixel_y < frame.rgb.shape[0]):
            raise ValueError(
                f"pixel ({pixel_x}, {pixel_y}) is outside the frame bounds ({frame.rgb.shape[1]}x{frame.rgb.shape[0]})"
            )
        point = backproject_pixel(
            pixel_x,
            pixel_y,
            frame.depth_raw,
            camera_matrix=self._camera_matrix,
            depth_scale=self._depth_scale,
            extrinsics_cm=self._extrinsics_cm,
        )
        if point is None:
            return None
        return Point3D(x=float(point[0]), y=float(point[1]), z=float(point[2]))

    def check_reachability(
        self, point_m: Sequence[float], elevation_degs: Sequence[float], roll_deg: float
    ) -> dict[float, ReachabilityResult]:
        point_arr = np.asarray(point_m, dtype=float)
        report: dict[float, ReachabilityResult] = {}
        for elevation_deg in elevation_degs:
            raw = _kinematics_check_reachability(point_arr, self._ik, elevation_deg=elevation_deg, roll_deg=roll_deg)
            report[float(elevation_deg)] = ReachabilityResult(
                reachable=raw["reachable"], ik_error_cm=raw["ik_error_cm"]
            )
        return report

    def move_to(
        self,
        point_m: Sequence[float],
        elevation_deg: float,
        roll_deg: float,
        gripper: str | None = None,
    ) -> MotionResult:
        if gripper is not None and gripper not in _GRIPPER_STATES:
            raise ValueError(f"gripper must be one of {_GRIPPER_STATES} or None, got {gripper!r}")

        point_arr = np.asarray(point_m, dtype=float)
        approach = approach_from_elevation(point_arr, elevation_deg)
        ik_result = self._ik.solve(point_arr, approach_dir=approach, roll_override=np.deg2rad(roll_deg))
        err_cm = ik_result.error_m * 100.0

        if err_cm > self._ik_tolerance_cm:
            return MotionResult(
                success=False,
                message=(
                    f"IK error {err_cm:.1f}cm exceeds tolerance {self._ik_tolerance_cm}cm at "
                    f"elevation={elevation_deg}deg roll={roll_deg}deg -- refusing to move. "
                    "Try a different elevation/roll, or re-check the target point."
                ),
                ik_error_cm=float(err_cm),
            )

        gripper_val = None
        if gripper == "open":
            gripper_val = self._gripper_open_value
        elif gripper == "closed":
            gripper_val = self._gripper_closed_value
        self._move_robot(ik_result.joints, gripper_val)

        ee_now = self._ik.forward_kinematics(ik_result.joints)
        return MotionResult(
            success=True,
            message=f"Moved to {np.round(ee_now, 3).tolist()}m (IK error {err_cm:.1f}cm).",
            ee_position_m=tuple(float(v) for v in ee_now),
            ik_error_cm=float(err_cm),
        )

    def set_gripper(self, state: str) -> MotionResult:
        if state not in _GRIPPER_STATES:
            raise ValueError(f"state must be one of {_GRIPPER_STATES}, got {state!r}")
        gripper_val = self._gripper_open_value if state == "open" else self._gripper_closed_value
        joints = self._read_current_joints()
        self._move_robot(joints, gripper_val)
        ee_now = self._ik.forward_kinematics(joints)
        return MotionResult(
            success=True,
            message=f"Gripper {state}.",
            ee_position_m=tuple(float(v) for v in ee_now),
        )

    def get_gripper_pose(self) -> GripperPose:
        obs = self._read_observation()
        joints = self._joints_from_observation(obs)
        ee_now = self._ik.forward_kinematics(joints)
        gripper_pos = float(obs["gripper.pos"])
        dist_open = abs(gripper_pos - self._gripper_open_value)
        dist_closed = abs(gripper_pos - self._gripper_closed_value)
        gripper_state = "open" if dist_open <= dist_closed else "closed"
        return GripperPose(ee_position_m=tuple(float(v) for v in ee_now), gripper_state=gripper_state)

    # -- frame cache ---------------------------------------------------------

    def _resolve_frame(self, frame_id: str | None) -> tuple[str, _CachedFrame]:
        if not self._frames:
            raise ToolError(
                StatusCode.TOOL_EXECUTION_ERROR, reason="no frame has been captured yet; call capture_photo first"
            )
        if frame_id is None:
            resolved_id = next(reversed(self._frames))
            return resolved_id, self._frames[resolved_id]
        frame = self._frames.get(frame_id)
        if frame is None:
            raise ValueError(f"unknown frame_id {frame_id!r}; known frames: {list(self._frames)}")
        return frame_id, frame

    # -- SO-101 hardware I/O (robot/gripper drive) ---------------------------

    def _ensure_robot(self) -> Any:
        if self._robot is not None:
            return self._robot
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except ImportError as e:
            raise ImportError(f"lerobot is not installed ({e})") from e
        if self._port is None:
            raise ValueError("either robot or port is required")
        self._robot = SO101Follower(SO101FollowerConfig(port=self._port, id="agent_core_so101"))
        self._robot.connect(calibrate=False)
        return self._robot

    def _move_robot(self, joints: np.ndarray, gripper_val: float | None) -> None:
        action = self._ik.joints_to_action(joints)
        if gripper_val is not None:
            action["gripper.pos"] = float(gripper_val)
        self._ensure_robot().send_action(action)

    def _read_observation(self) -> dict:
        return self._ensure_robot().get_observation()

    def _joints_from_observation(self, obs: dict) -> np.ndarray:
        q = np.zeros(7)
        q[1:6] = np.deg2rad(
            [
                obs["shoulder_pan.pos"],
                obs["shoulder_lift.pos"],
                obs["elbow_flex.pos"],
                obs["wrist_flex.pos"],
                obs["wrist_roll.pos"],
            ]
        )
        return self._ik.clamp_joints(q)

    def _read_current_joints(self) -> np.ndarray:
        return self._joints_from_observation(self._read_observation())

    def health_check(self) -> None:
        """Best-effort camera probe, mirroring ``robotic_arm``'s ``StepExecutorRail`` health check."""
        try:
            self.capture()
        except Exception:  # noqa: BLE001 -- best-effort probe, must never block agent construction
            tool_logger.exception("[So101ArmBackend] health check capture failed (continuing)")


__all__ = ["So101ArmBackend"]
