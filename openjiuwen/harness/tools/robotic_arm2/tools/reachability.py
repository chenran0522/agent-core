# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""``check_reachability`` tool."""

from __future__ import annotations

import asyncio
from typing import Any

from openjiuwen.core.foundation.tool import Tool, ToolCard
from openjiuwen.harness.tools.base_tool import ToolOutput
from openjiuwen.harness.tools.robotic_arm2.config import ArmBackend

_DEFAULT_ELEVATIONS = (90.0, 60.0, 45.0, 30.0, 0.0)

CHECK_REACHABILITY_TOOL_CARD = ToolCard(
    id="tool.robotic_arm2.check_reachability",
    name="check_reachability",
    description=(
        "Check whether a 3D target point (metres, robot base frame) is actually reachable, at one "
        "or more candidate approach elevations (90=straight down, 45=diagonal, 0=horizontal). "
        "Returns the IK error in cm for each elevation. Call this BEFORE move_to to pick a "
        "feasible elevation/roll, instead of discovering infeasibility only after move_to refuses."
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
            "elevation_degs": {
                "type": "array",
                "items": {"type": "number"},
                "description": (
                    f"Candidate approach elevations to test, degrees (0-90). Defaults to "
                    f"{list(_DEFAULT_ELEVATIONS)} if omitted."
                ),
            },
            "roll_deg": {
                "type": "number",
                "description": "Gripper roll around the approach axis, degrees. Defaults to 0.",
            },
        },
        "required": ["point_m"],
    },
)


class CheckReachabilityTool(Tool):
    def __init__(self, backend: ArmBackend, card: ToolCard = CHECK_REACHABILITY_TOOL_CARD) -> None:
        super().__init__(card=card)
        self._backend = backend

    async def invoke(self, inputs: Any, **kwargs: Any) -> ToolOutput:
        if not isinstance(inputs, dict):
            return ToolOutput(success=False, error="point_m is required")
        point_m = inputs.get("point_m")
        if not isinstance(point_m, (list, tuple)) or len(point_m) != 3:
            return ToolOutput(success=False, error="point_m must be a 3-element [x, y, z] array")

        elevation_degs = inputs.get("elevation_degs") or list(_DEFAULT_ELEVATIONS)
        roll_deg = float(inputs.get("roll_deg") or 0.0)

        try:
            report = await asyncio.to_thread(self._backend.check_reachability, point_m, elevation_degs, roll_deg)
        except Exception as e:  # noqa: BLE001
            return ToolOutput(success=False, error=f"check_reachability failed: {e}")

        lines = [f"Reachability for point_m={list(point_m)}, roll_deg={roll_deg}:"]
        for elevation in sorted(report, reverse=True):
            r = report[elevation]
            status = "OK" if r.reachable else "UNREACHABLE"
            lines.append(f"  elevation={elevation}deg -> ik_error={r.ik_error_cm:.1f}cm [{status}]")
        return ToolOutput(success=True, data={"content": "\n".join(lines)})

    async def stream(self, inputs: Any, **kwargs: Any):
        yield await self.invoke(inputs, **kwargs)


__all__ = ["CHECK_REACHABILITY_TOOL_CARD", "CheckReachabilityTool"]
