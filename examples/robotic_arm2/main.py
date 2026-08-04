#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Minimal end-to-end test script for the robotic-arm2 agent (SO-101, tool-driven).

Runs ``create_robotic_arm2_agent`` as the top-level agent against a real
``So101ArmBackend`` (real RealSense camera + real SO-101 arm). Unlike
``examples/robotic_arm/main.py``, there is only ONE model involved: the same
vision-capable planning LLM calls ``capture_photo``/``segment_scene``/
``pixel_to_3d``/``check_reachability``/``move_to``/``set_gripper``/
``get_gripper_pose`` itself -- there is no separate constraint-generation VLM.

All configuration lives in ``.env`` (copy ``.env.example`` -> ``.env`` and
fill in real values for your rig; nothing is hardcoded here or defaulted
silently -- a missing variable raises ``KeyError`` immediately).

Requires: pip install 'openjiuwen[robotic-arm-so101-rekep]'
(same extra as ``robotic_arm`` -- MobileSAM + torch + RealSense + ikpy; this
rewrite drops the DINOv2/transformers dependency but the extra name is
unchanged). Also needs MobileSAM weights (see vendors/so101/perception.py)
and a physical SO-101 connected at ``ROBOT_PORT``.

Run from the repository root::

    cp examples/robotic_arm2/.env.example examples/robotic_arm2/.env
    # edit examples/robotic_arm2/.env with your rig's real values
    uv run python examples/robotic_arm2/main.py
"""

from __future__ import annotations

import asyncio
import os
import uuid

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from openjiuwen.core.foundation.llm import init_model  # noqa: E402
from openjiuwen.core.runner import Runner  # noqa: E402
from openjiuwen.harness.subagents.robotic_arm2_agent import create_robotic_arm2_agent  # noqa: E402
from openjiuwen.harness.tools.robotic_arm2.config import RoboticArm2RuntimeSettings  # noqa: E402
from openjiuwen.harness.tools.robotic_arm2.vendors.so101.backend import So101ArmBackend  # noqa: E402

# The manipulation goal to hand to the agent -- set in .env, no silent default.
QUERY = os.environ["ROBOTIC_ARM_QUERY"]


def build_model():
    """The single vision-capable planning LLM -- drives every tool call itself."""
    return init_model(
        provider=os.environ["MODEL_PROVIDER"],
        model_name=os.environ["MODEL_NAME"],
        api_key=os.environ["API_KEY"],
        api_base=os.environ["API_BASE"],
    )


def build_backend() -> So101ArmBackend:
    """Real SO-101 + RealSense + MobileSAM backend -- see config.py's ArmBackend protocol."""
    workspace_min = [float(x) for x in os.environ["ARM_WORKSPACE_MIN"].split(",")]
    workspace_max = [float(x) for x in os.environ["ARM_WORKSPACE_MAX"].split(",")]
    return So101ArmBackend(
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        camera_matrix_path=os.environ["CAMERA_MATRIX_PATH"],
        depth_scale_path=os.environ["DEPTH_SCALE_PATH"],
        extrinsics_path=os.environ["EXTRINSICS_PATH"],
        urdf_path=os.environ["URDF_PATH"],
        port=os.environ["ROBOT_PORT"],
        sam_checkpoint_path=os.environ["SAM_CHECKPOINT_PATH"],
    )


async def main() -> None:
    model = build_model()
    settings = RoboticArm2RuntimeSettings(backend=build_backend(), context_default_window_round_num=1)
    agent = create_robotic_arm2_agent(model=model, settings=settings, max_iterations=60, language="en")

    print(f"query={QUERY!r}")

    await Runner.start()
    try:
        await agent.ensure_initialized()
        result = await Runner.run_agent(
            agent,
            {"query": QUERY, "conversation_id": f"robotic_arm2_{uuid.uuid4().hex[:12]}"},
        )
    finally:
        await Runner.stop()

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
