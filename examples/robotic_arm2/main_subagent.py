#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Run robotic_arm2_agent as a subagent behind a coordinator, to exercise the
on_frame_captured / on_tool_result runtime-observability callbacks.

Unlike ``main.py`` (robotic_arm2_agent as the top-level agent), this script
builds a small coordinator DeepAgent with robotic_arm2_agent registered as a
subagent via ``task_tool`` -- the same shape a UI-facing wrapper would use to
delegate manipulation goals (see ``build_robotic_arm2_agent_config`` +
``create_deep_agent(subagents=[...])``). Mirrors
``examples/robotic_arm/main_subagent.py``, adapted for the tool-driven
architecture: there is no ``sub_tasks``/``debug`` payload to log anymore
(``robotic_arm2`` has no ``report_plan``/hidden pipeline) -- instead
``on_tool_result`` fires after every one of the 8 tool calls the model makes
(``capture_photo``/``segment_scene``/``pixel_to_3d``/``get_object_geometry``/
``check_reachability``/``move_to``/``set_gripper``/``get_gripper_pose``),
and ``on_frame_captured`` fires once per image any of those tools produced.
``_CaptureLogger`` below saves every such image to disk and appends a
full-detail JSON record of every callback to ``events.jsonl``, so the whole
run's tool-call sequence is available afterwards, not just what scrolled past
on the console -- every console line (this script's own prints, plus
stdout/stderr from anything else in the process) is also mirrored into
``console.log`` in the same run folder, via ``_TeeStream``.

``ctx.session`` ends up ``None`` inside a delegated subagent (TaskTool
doesn't forward the parent session), so ``on_frame_captured``/``on_tool_result``
are wired via closures on ``RoboticArm2RuntimeSettings`` instead of relying on
that session -- specifically so they still fire through this delegated path.

Same ``.env`` as ``main.py`` -- see that file's docstring for setup
(``cp examples/robotic_arm2/.env.example examples/robotic_arm2/.env`` and
fill in real values for your rig).

Requires: pip install 'openjiuwen[robotic-arm-so101-rekep]'
Also needs MobileSAM weights (see vendors/so101/perception.py) and a
physical SO-101 connected at ``ROBOT_PORT``.

Run from the repository root::

    uv run python examples/robotic_arm2/main_subagent.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from openjiuwen.core.foundation.llm import init_model  # noqa: E402
from openjiuwen.core.runner import Runner  # noqa: E402
from openjiuwen.core.single_agent.schema.agent_card import AgentCard  # noqa: E402
from openjiuwen.harness.factory import create_deep_agent  # noqa: E402
from openjiuwen.harness.subagents.robotic_arm2_agent import build_robotic_arm2_agent_config  # noqa: E402
from openjiuwen.harness.tools.robotic_arm2.config import RoboticArm2RuntimeSettings  # noqa: E402
from openjiuwen.harness.tools.robotic_arm2.vendors.so101.backend import So101ArmBackend  # noqa: E402

# The manipulation goal handed to the coordinator -- set in .env, no silent default.
QUERY = os.environ["ROBOTIC_ARM_QUERY"]

_COORDINATOR_SYSTEM_PROMPT = (
    "You are a coordinator with exactly one specialized subagent, robotic_arm2_agent, for any "
    "physical manipulation goal. Delegate the whole goal to it via task_tool in a single call; "
    "do not try to plan, perceive, or ground the manipulation yourself."
)


def build_model():
    """Shared model config for both the coordinator and the robotic-arm subagent.

    In production these would typically be two different models (a cheap
    text-only model for the coordinator, a vision-capable one for the arm) --
    kept identical here to keep this test script to one required model config.
    """
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


class _TeeStream:
    """Mirror every write to this stream into a log file as well as the original stream."""

    def __init__(self, original: Any, log_file: Any) -> None:
        self._original = original
        self._log_file = log_file

    def write(self, text: str) -> None:
        self._original.write(text)
        self._log_file.write(text)
        self._log_file.flush()

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def _safe_filename(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label) or "image"


class _CaptureLogger:
    """Records the full detail of every on_frame_captured/on_tool_result callback.

    Every image (``capture_photo``'s frame, ``segment_scene``'s numbered
    overlay) is written to ``out_dir`` as an individual JPEG; every tool
    call's name, arguments, result text, and success flag is both printed in
    full and appended to ``events.jsonl`` (one JSON object per line, image
    bytes replaced by the file path) so the whole run's tool-call sequence
    can be inspected afterwards, not just skimmed live off the console.
    """

    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self._out_dir / "events.jsonl"
        self._frame_count = 0
        self._tool_call_count = 0
        print(f"[capture] writing images/events.jsonl under {self._out_dir}")

    def _append_event(self, event: dict) -> None:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    async def on_frame_captured(self, payload: dict) -> None:
        """Fires from ToolImageRail just before an image is injected into the model's next message."""
        self._frame_count += 1
        label = payload.get("label") or "image"
        image_path = self._out_dir / f"frame_{self._frame_count:03d}_{_safe_filename(label)}.jpg"
        image_path.write_bytes(base64.b64decode(payload["image_base64"]))

        print(f"[callback:on_frame_captured] #{self._frame_count} label={label!r} -> {image_path}")

        self._append_event(
            {"type": "frame_captured", "seq": self._frame_count, "label": label, "image_path": str(image_path)}
        )

    async def on_tool_result(self, payload: dict) -> None:
        """Fires from ToolImageRail after every tool call in the agent's turn (all 8 robotic-arm2
        tools, plus any other tool the agent happens to have -- see ``config.py``'s docstring).
        """
        self._tool_call_count += 1
        print(
            f"[callback:on_tool_result] #{self._tool_call_count} {payload['tool_name']}"
            f"({payload['tool_args']}) success={payload['success']}"
        )
        print(f"[callback:on_tool_result] result_text: {payload['result_text']}")

        self._append_event(
            {
                "type": "tool_result",
                "seq": self._tool_call_count,
                "tool_name": payload["tool_name"],
                "tool_args": payload["tool_args"],
                "result_text": payload["result_text"],
                "success": payload["success"],
            }
        )


async def main() -> None:
    run_dir = Path(__file__).resolve().parent / "_captures" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    console_log = (run_dir / "console.log").open("a", encoding="utf-8")
    sys.stdout = _TeeStream(sys.stdout, console_log)
    sys.stderr = _TeeStream(sys.stderr, console_log)

    capture_logger = _CaptureLogger(run_dir)

    settings = RoboticArm2RuntimeSettings(
        backend=build_backend(),
        context_default_window_round_num=1,
        on_frame_captured=capture_logger.on_frame_captured,
        on_tool_result=capture_logger.on_tool_result,
    )

    subagents = [
        build_robotic_arm2_agent_config(
            model=build_model(),
            settings=settings,
            max_iterations=60,
            language="en",
        ),
    ]

    coordinator = create_deep_agent(
        model=build_model(),
        card=AgentCard(
            name="robotic_arm2_coordinator",
            description="Coordinator that delegates physical manipulation goals to robotic_arm2_agent.",
        ),
        system_prompt=_COORDINATOR_SYSTEM_PROMPT,
        subagents=subagents,
        max_iterations=10,
        language="en",
    )

    print(f"query={QUERY!r}")

    await Runner.start()
    try:
        await coordinator.ensure_initialized()
        result = await Runner.run_agent(
            coordinator,
            {"query": QUERY, "conversation_id": f"robotic_arm2_coordinator_{uuid.uuid4().hex[:12]}"},
        )
    finally:
        await Runner.stop()

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
