# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""``move_to`` / ``set_gripper`` / ``get_gripper_pose`` tools -- the only ones that touch hardware."""

from __future__ import annotations

import asyncio
from typing import Any

from openjiuwen.core.foundation.tool import Tool, ToolCard
from openjiuwen.harness.tools.base_tool import ToolOutput
from openjiuwen.harness.tools.robotic_arm2.config import ArmBackend

MOVE_TO_TOOL_CARD = ToolCard(
    id="tool.robotic_arm2.move_to",
    name="move_to",
    description=(
        "Solve IK for one target end-effector pose and physically move the arm there. This is a "
        "SINGLE-SHOT move to exactly the point/angle you give -- no path planning, no collision "
        "checking, no automatic waypoints. Avoid jumping straight from far away to a grasp/release "
        "point in one call; approach gradually across multiple move_to calls, re-checking with a "
        "fresh photo as you get closer -- you decide how many steps. Refuses to move (no hardware "
        "command sent) if the IK error exceeds the backend's safety tolerance -- call "
        "check_reachability first if you're not sure a point/elevation combination is feasible."
    ),
    input_params={
        "type": "object",
        "properties": {
            "point_m": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
                "description": "[x, y, z] in metres, robot base frame (e.g. from pixel_to_3d).",
            },
            "elevation_deg": {
                "type": "number",
                "description": "Approach elevation, degrees (90=straight down, 45=diagonal, 0=horizontal).",
            },
            "roll_deg": {
                "type": "number",
                "description": "Gripper roll around the approach axis, degrees. Defaults to 0.",
            },
            "gripper": {
                "type": "string",
                "enum": ["open", "closed"],
                "description": "Optionally open/close the gripper as part of this move. Omit to leave it unchanged.",
            },
        },
        "required": ["point_m", "elevation_deg"],
    },
)

SET_GRIPPER_TOOL_CARD = ToolCard(
    id="tool.robotic_arm2.set_gripper",
    name="set_gripper",
    description="Open or close the gripper without moving the arm (e.g. release after move_to already placed it).",
    input_params={
        "type": "object",
        "properties": {
            "state": {"type": "string", "enum": ["open", "closed"], "description": "Target gripper state."},
        },
        "required": ["state"],
    },
)

GET_GRIPPER_POSE_TOOL_CARD = ToolCard(
    id="tool.robotic_arm2.get_gripper_pose",
    name="get_gripper_pose",
    description=(
        "Read the current end-effector 3D position and gripper state. Useful after a grasp to "
        "reason about a held object's position as you move it -- e.g. 'the object moved by the "
        "same delta as the end-effector since I grasped it at point X'. There is no automatic "
        "held-object tracking; you do that reasoning yourself with this tool."
    ),
    input_params={"type": "object", "properties": {}},
)


class MoveToTool(Tool):
    def __init__(self, backend: ArmBackend, card: ToolCard = MOVE_TO_TOOL_CARD) -> None:
        super().__init__(card=card)
        self._backend = backend

    async def invoke(self, inputs: Any, **kwargs: Any) -> ToolOutput:
        if not isinstance(inputs, dict):
            return ToolOutput(success=False, error="point_m and elevation_deg are required")
        point_m = inputs.get("point_m")
        elevation_deg = inputs.get("elevation_deg")
        if not isinstance(point_m, (list, tuple)) or len(point_m) != 3 or elevation_deg is None:
            return ToolOutput(success=False, error="point_m ([x, y, z]) and elevation_deg are required")
        roll_deg = float(inputs.get("roll_deg") or 0.0)
        gripper = inputs.get("gripper")

        try:
            result = await asyncio.to_thread(self._backend.move_to, point_m, float(elevation_deg), roll_deg, gripper)
        except Exception as e:  # noqa: BLE001
            return ToolOutput(success=False, error=f"move_to failed: {e}")

        if not result.success:
            return ToolOutput(success=False, error=result.message)
        return ToolOutput(success=True, data={"content": result.message})

    async def stream(self, inputs: Any, **kwargs: Any):
        yield await self.invoke(inputs, **kwargs)


class SetGripperTool(Tool):
    def __init__(self, backend: ArmBackend, card: ToolCard = SET_GRIPPER_TOOL_CARD) -> None:
        super().__init__(card=card)
        self._backend = backend

    async def invoke(self, inputs: Any, **kwargs: Any) -> ToolOutput:
        state = inputs.get("state") if isinstance(inputs, dict) else None
        if state not in ("open", "closed"):
            return ToolOutput(success=False, error="state must be 'open' or 'closed'")
        try:
            result = await asyncio.to_thread(self._backend.set_gripper, state)
        except Exception as e:  # noqa: BLE001
            return ToolOutput(success=False, error=f"set_gripper failed: {e}")
        if not result.success:
            return ToolOutput(success=False, error=result.message)
        return ToolOutput(success=True, data={"content": result.message})

    async def stream(self, inputs: Any, **kwargs: Any):
        yield await self.invoke(inputs, **kwargs)


class GetGripperPoseTool(Tool):
    def __init__(self, backend: ArmBackend, card: ToolCard = GET_GRIPPER_POSE_TOOL_CARD) -> None:
        super().__init__(card=card)
        self._backend = backend

    async def invoke(self, inputs: Any, **kwargs: Any) -> ToolOutput:
        del inputs
        try:
            pose = await asyncio.to_thread(self._backend.get_gripper_pose)
        except Exception as e:  # noqa: BLE001
            return ToolOutput(success=False, error=f"get_gripper_pose failed: {e}")
        content = f"end_effector_m={[round(v, 4) for v in pose.ee_position_m]}, gripper={pose.gripper_state}."
        return ToolOutput(success=True, data={"content": content, "ee_position_m": pose.ee_position_m})

    async def stream(self, inputs: Any, **kwargs: Any):
        yield await self.invoke(inputs, **kwargs)


__all__ = [
    "GET_GRIPPER_POSE_TOOL_CARD",
    "MOVE_TO_TOOL_CARD",
    "SET_GRIPPER_TOOL_CARD",
    "GetGripperPoseTool",
    "MoveToTool",
    "SetGripperTool",
]
