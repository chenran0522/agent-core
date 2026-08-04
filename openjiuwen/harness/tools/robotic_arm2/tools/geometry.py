# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""``get_object_geometry`` tool.

The backend already caches a full per-pixel point cloud for every captured
frame (``backproject_frame``, used internally by ``segment`` for its
workspace filter) -- this tool is the first thing that exposes that 3D data
to the model as actual geometry (bounding dimensions + principal axis)
instead of leaving shape judgement entirely to reading the photo.
"""

from __future__ import annotations

import asyncio
from typing import Any

from openjiuwen.core.foundation.tool import Tool, ToolCard
from openjiuwen.harness.tools.base_tool import ToolOutput
from openjiuwen.harness.tools.robotic_arm2.config import ArmBackend

GET_OBJECT_GEOMETRY_TOOL_CARD = ToolCard(
    id="tool.robotic_arm2.get_object_geometry",
    name="get_object_geometry",
    description=(
        "Measure an object's real 3D shape from the depth points inside its segment_scene mask: "
        "bounding dimensions in metres (largest to smallest), the direction of its longest axis "
        "(robot base frame), and a rough shape_hint. shape_hint is one of: 'cylinder' (elongated, "
        "round cross-section, e.g. a pen/bottle -- roll_deg barely matters), 'box_elongated' "
        "(elongated, rectangular cross-section, e.g. a ruler -- roll_deg must align to its edges), "
        "'disk' (flat, round face, e.g. a coin/tape roll), 'plate' (flat, rectangular face, e.g. a "
        "book/card), or 'compact' (no dominant axis, e.g. roughly cube- or ball-shaped -- PCA can't "
        "tell those two apart). Use this instead of guessing shape from the photo alone when picking "
        "elevation_deg/roll_deg -- e.g. principal_axis_m is the object's real long-axis direction, "
        "useful for gripping perpendicular to it. Requires segment_scene to have already run on this "
        "frame. Reference the object either by its segment_scene id, or by a pixel inside its mask."
    ),
    input_params={
        "type": "object",
        "properties": {
            "frame_id": {
                "type": "string",
                "description": "Frame to measure on. Omit to use the most recent capture.",
            },
            "object_id": {
                "type": "integer",
                "description": "An id from segment_scene's last result on this frame.",
            },
            "pixel_x": {
                "type": "integer",
                "description": "Alternative to object_id: a pixel that falls inside the object's mask.",
            },
            "pixel_y": {
                "type": "integer",
                "description": "Alternative to object_id: a pixel that falls inside the object's mask.",
            },
        },
    },
)


class GetObjectGeometryTool(Tool):
    def __init__(self, backend: ArmBackend, card: ToolCard = GET_OBJECT_GEOMETRY_TOOL_CARD) -> None:
        super().__init__(card=card)
        self._backend = backend

    async def invoke(self, inputs: Any, **kwargs: Any) -> ToolOutput:
        if not isinstance(inputs, dict):
            inputs = {}
        frame_id = inputs.get("frame_id")
        object_id = inputs.get("object_id")
        pixel_x = inputs.get("pixel_x")
        pixel_y = inputs.get("pixel_y")
        if object_id is None and (pixel_x is None or pixel_y is None):
            return ToolOutput(success=False, error="either object_id, or both pixel_x and pixel_y, are required")

        try:
            geometry = await asyncio.to_thread(self._backend.get_object_geometry, frame_id, object_id, pixel_x, pixel_y)
        except Exception as e:  # noqa: BLE001
            return ToolOutput(success=False, error=f"get_object_geometry failed: {e}")

        if geometry is None:
            return ToolOutput(
                success=True,
                data={
                    "content": (
                        "Not enough valid depth points in this object's mask to measure its "
                        "geometry -- try recapturing, or pick a different object/pixel."
                    )
                },
            )

        content = (
            f"dimensions_m (largest->smallest)={[round(v, 3) for v in geometry.dimensions_m]}, "
            f"principal_axis_m={[round(v, 3) for v in geometry.principal_axis_m]}, "
            f"shape_hint={geometry.shape_hint}, centroid_m={[round(v, 3) for v in geometry.centroid_m]} "
            f"({geometry.point_count} depth points)."
        )
        return ToolOutput(
            success=True,
            data={
                "content": content,
                "dimensions_m": geometry.dimensions_m,
                "principal_axis_m": geometry.principal_axis_m,
                "shape_hint": geometry.shape_hint,
            },
        )

    async def stream(self, inputs: Any, **kwargs: Any):
        yield await self.invoke(inputs, **kwargs)


__all__ = ["GET_OBJECT_GEOMETRY_TOOL_CARD", "GetObjectGeometryTool"]
