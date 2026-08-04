# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""``capture_photo`` / ``segment_scene`` / ``pixel_to_3d`` tools.

Every tool's ``invoke()`` returns a :class:`ToolOutput` whose ``data["content"]``
is what the model actually reads as the tool result text -- images never go in
there (``ability_manager._build_tool_message_content`` stringifies everything,
so a base64 image dumped into the text channel would just be noise, not
something the model can see). Instead, image payloads ride in
``data["images"]``; ``rails/tool_image_rail.py`` reads them off
``ctx.inputs.tool_result`` (the raw return value, available to rails but never
stringified) and injects them as real ``image_url`` content blocks before the
model's next turn -- the same mechanism ``robotic_arm``'s
``VisionPerceptionRail`` uses, just triggered by an explicit tool call
instead of automatically every turn.
"""

from __future__ import annotations

import asyncio
from typing import Any

from openjiuwen.core.foundation.tool import Tool, ToolCard
from openjiuwen.harness.tools.base_tool import ToolOutput
from openjiuwen.harness.tools.robotic_arm2.config import ArmBackend

CAPTURE_PHOTO_TOOL_CARD = ToolCard(
    id="tool.robotic_arm2.capture_photo",
    name="capture_photo",
    description=(
        "Capture a fresh RGB-D photo of the workspace from the arm-mounted camera. Call this "
        "first, and again whenever the scene may have changed (after a move, a grasp/release, or "
        "if you're not sure the last photo still reflects reality). Returns a frame_id and the "
        "photo itself (attached to your next message) -- use the frame_id with segment_scene and "
        "pixel_to_3d."
    ),
    input_params={"type": "object", "properties": {}},
)

SEGMENT_SCENE_TOOL_CARD = ToolCard(
    id="tool.robotic_arm2.segment_scene",
    name="segment_scene",
    description=(
        "Segment a captured frame into distinct objects. Returns a numbered overlay photo "
        "(attached to your next message) plus each object's 2D PIXEL coordinate -- NOT a 3D "
        "position. Call pixel_to_3d on whichever pixel you actually want -- one of these ids, or "
        "any other pixel you pick by eye from the photo -- to get its 3D position; you are not "
        "restricted to the ids this tool proposes."
    ),
    input_params={
        "type": "object",
        "properties": {
            "frame_id": {
                "type": "string",
                "description": "Frame to segment, as returned by capture_photo. Omit to use the most recent capture.",
            },
        },
    },
)

PIXEL_TO_3D_TOOL_CARD = ToolCard(
    id="tool.robotic_arm2.pixel_to_3d",
    name="pixel_to_3d",
    description=(
        "Backproject one pixel on a captured frame to a 3D point in the robot base frame "
        "(metres, right-handed: +X forward, +Y left, +Z up). Works for ANY pixel, not just ones "
        "segment_scene proposed -- pick whatever point on the photo you actually want to reach "
        "(e.g. a handle, a rim, a corner) and pass its pixel coordinates here."
    ),
    input_params={
        "type": "object",
        "properties": {
            "frame_id": {
                "type": "string",
                "description": "Frame the pixel is on. Omit to use the most recent capture.",
            },
            "pixel_x": {"type": "integer", "description": "Pixel x coordinate (column, 0 = left edge)."},
            "pixel_y": {"type": "integer", "description": "Pixel y coordinate (row, 0 = top edge)."},
        },
        "required": ["pixel_x", "pixel_y"],
    },
)


class CapturePhotoTool(Tool):
    def __init__(self, backend: ArmBackend, card: ToolCard = CAPTURE_PHOTO_TOOL_CARD) -> None:
        super().__init__(card=card)
        self._backend = backend

    async def invoke(self, inputs: Any, **kwargs: Any) -> ToolOutput:
        del inputs
        try:
            frame = await asyncio.to_thread(self._backend.capture)
        except Exception as e:  # noqa: BLE001 -- surfaced to the model as a tool error, not raised
            return ToolOutput(success=False, error=f"capture_photo failed: {e}")
        return ToolOutput(
            success=True,
            data={
                "content": f"Captured photo frame_id={frame.frame_id} ({frame.width}x{frame.height}).",
                "images": [{"label": f"Photo ({frame.frame_id})", "image_base64": frame.image_base64}],
                "frame_id": frame.frame_id,
            },
        )

    async def stream(self, inputs: Any, **kwargs: Any):
        yield await self.invoke(inputs, **kwargs)


class SegmentSceneTool(Tool):
    def __init__(self, backend: ArmBackend, card: ToolCard = SEGMENT_SCENE_TOOL_CARD) -> None:
        super().__init__(card=card)
        self._backend = backend

    async def invoke(self, inputs: Any, **kwargs: Any) -> ToolOutput:
        frame_id: str | None = inputs.get("frame_id") if isinstance(inputs, dict) else None
        try:
            result = await asyncio.to_thread(self._backend.segment, frame_id)
        except Exception as e:  # noqa: BLE001
            return ToolOutput(success=False, error=f"segment_scene failed: {e}")

        if not result.objects:
            content = f"No objects detected on frame_id={result.frame_id}."
        else:
            lines = [f"Detected {len(result.objects)} object(s) on frame_id={result.frame_id}:"]
            lines.extend(f"  id={obj.object_id} pixel=({obj.pixel_x}, {obj.pixel_y})" for obj in result.objects)
            content = "\n".join(lines)

        return ToolOutput(
            success=True,
            data={
                "content": content,
                "images": [
                    {
                        "label": f"Segmentation overlay ({result.frame_id})",
                        "image_base64": result.overlay_image_base64,
                    }
                ],
                "frame_id": result.frame_id,
            },
        )

    async def stream(self, inputs: Any, **kwargs: Any):
        yield await self.invoke(inputs, **kwargs)


class PixelTo3DTool(Tool):
    def __init__(self, backend: ArmBackend, card: ToolCard = PIXEL_TO_3D_TOOL_CARD) -> None:
        super().__init__(card=card)
        self._backend = backend

    async def invoke(self, inputs: Any, **kwargs: Any) -> ToolOutput:
        if not isinstance(inputs, dict):
            return ToolOutput(success=False, error="pixel_x and pixel_y are required")
        pixel_x, pixel_y = inputs.get("pixel_x"), inputs.get("pixel_y")
        if pixel_x is None or pixel_y is None:
            return ToolOutput(success=False, error="pixel_x and pixel_y are required")
        frame_id = inputs.get("frame_id")

        try:
            point = await asyncio.to_thread(self._backend.backproject, frame_id, int(pixel_x), int(pixel_y))
        except Exception as e:  # noqa: BLE001
            return ToolOutput(success=False, error=f"pixel_to_3d failed: {e}")

        if point is None:
            return ToolOutput(
                success=True,
                data={
                    "content": (
                        f"pixel ({pixel_x}, {pixel_y}) has no valid depth reading -- try a nearby "
                        "pixel, or recapture if the scene may have changed."
                    )
                },
            )
        return ToolOutput(
            success=True,
            data={
                "content": (
                    f"pixel ({pixel_x}, {pixel_y}) -> point_m=[{point.x:.4f}, {point.y:.4f}, {point.z:.4f}] "
                    "(robot base frame, metres)."
                ),
                "point_m": point.as_tuple(),
            },
        )

    async def stream(self, inputs: Any, **kwargs: Any):
        yield await self.invoke(inputs, **kwargs)


__all__ = [
    "CAPTURE_PHOTO_TOOL_CARD",
    "PIXEL_TO_3D_TOOL_CARD",
    "SEGMENT_SCENE_TOOL_CARD",
    "CapturePhotoTool",
    "PixelTo3DTool",
    "SegmentSceneTool",
]
