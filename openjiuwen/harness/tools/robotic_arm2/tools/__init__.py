# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""The 8 LLM-callable tools that replace ``robotic_arm``'s single ``report_plan`` + rail pipeline."""

from __future__ import annotations

from openjiuwen.core.foundation.tool import Tool
from openjiuwen.harness.tools.robotic_arm2.config import ArmBackend
from openjiuwen.harness.tools.robotic_arm2.tools.geometry import (
    GET_OBJECT_GEOMETRY_TOOL_CARD,
    GetObjectGeometryTool,
)
from openjiuwen.harness.tools.robotic_arm2.tools.motion import (
    GET_GRIPPER_POSE_TOOL_CARD,
    MOVE_TO_TOOL_CARD,
    SET_GRIPPER_TOOL_CARD,
    GetGripperPoseTool,
    MoveToTool,
    SetGripperTool,
)
from openjiuwen.harness.tools.robotic_arm2.tools.perception import (
    CAPTURE_PHOTO_TOOL_CARD,
    PIXEL_TO_3D_TOOL_CARD,
    SEGMENT_SCENE_TOOL_CARD,
    CapturePhotoTool,
    PixelTo3DTool,
    SegmentSceneTool,
)
from openjiuwen.harness.tools.robotic_arm2.tools.reachability import (
    CHECK_REACHABILITY_TOOL_CARD,
    CheckReachabilityTool,
)


def build_robotic_arm2_tools(backend: ArmBackend) -> list[Tool]:
    return [
        CapturePhotoTool(backend),
        SegmentSceneTool(backend),
        PixelTo3DTool(backend),
        GetObjectGeometryTool(backend),
        CheckReachabilityTool(backend),
        MoveToTool(backend),
        SetGripperTool(backend),
        GetGripperPoseTool(backend),
    ]


__all__ = [
    "CAPTURE_PHOTO_TOOL_CARD",
    "CHECK_REACHABILITY_TOOL_CARD",
    "GET_GRIPPER_POSE_TOOL_CARD",
    "GET_OBJECT_GEOMETRY_TOOL_CARD",
    "MOVE_TO_TOOL_CARD",
    "PIXEL_TO_3D_TOOL_CARD",
    "SEGMENT_SCENE_TOOL_CARD",
    "SET_GRIPPER_TOOL_CARD",
    "CapturePhotoTool",
    "CheckReachabilityTool",
    "GetGripperPoseTool",
    "GetObjectGeometryTool",
    "MoveToTool",
    "PixelTo3DTool",
    "SegmentSceneTool",
    "SetGripperTool",
    "build_robotic_arm2_tools",
]
