#!/usr/bin/env python
"""Tests for ToolImageRail: after_tool_call stashes images, before_model_call injects them.

Mirrors ``robotic_arm``'s ``test_vision_perception_rail.py`` fake-context
pattern, adapted for the tool-triggered (rather than every-turn) injection
mechanism.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openjiuwen.core.foundation.llm import ToolMessage, UserMessage
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext, InvokeInputs, ModelCallInputs, ToolCallInputs
from openjiuwen.harness.tools.base_tool import ToolOutput
from openjiuwen.harness.tools.robotic_arm2.config import RoboticArm2RuntimeSettings
from openjiuwen.harness.tools.robotic_arm2.rails.tool_image_rail import ToolImageRail


class _FakeModelContext:
    def __init__(self, messages: list) -> None:
        self._messages = messages

    def get_messages(self):
        return self._messages

    def set_messages(self, messages):
        self._messages = messages

    def pop_messages(self, n: int):
        popped, self._messages = self._messages[-n:], self._messages[:-n]
        return popped

    async def add_messages(self, msg):
        self._messages.append(msg)


def _tool_call_ctx(
    tool_result, *, tool_name: str = "capture_photo", tool_args=None, tool_msg=None
) -> AgentCallbackContext:
    return AgentCallbackContext(
        agent=MagicMock(),
        inputs=ToolCallInputs(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            tool_msg=tool_msg if tool_msg is not None else ToolMessage(content=str(tool_result), tool_call_id="c1"),
        ),
        extra={},
    )


def _model_call_ctx(*, context=None, extra=None) -> AgentCallbackContext:
    return AgentCallbackContext(
        agent=MagicMock(),
        inputs=ModelCallInputs(messages=[]),
        extra=extra if extra is not None else {},
        context=context,
    )


@pytest.mark.asyncio
async def test_after_tool_call_ignores_non_tool_call_events() -> None:
    rail = ToolImageRail(RoboticArm2RuntimeSettings())
    ctx = AgentCallbackContext(agent=MagicMock(), inputs=InvokeInputs(query="go"), extra={})

    await rail.after_tool_call(ctx)

    assert "robotic_arm2_pending_images" not in ctx.extra


@pytest.mark.asyncio
async def test_after_tool_call_ignores_results_without_images() -> None:
    rail = ToolImageRail(RoboticArm2RuntimeSettings())
    ctx = _tool_call_ctx(ToolOutput(success=True, data={"content": "no images here"}))

    await rail.after_tool_call(ctx)

    assert ctx.extra.get("robotic_arm2_pending_images") in (None, [])


@pytest.mark.asyncio
async def test_after_tool_call_stashes_images_and_notifies() -> None:
    notified = []

    async def on_frame_captured(payload):
        notified.append(payload)

    settings = RoboticArm2RuntimeSettings(on_frame_captured=on_frame_captured)
    rail = ToolImageRail(settings)
    images = [{"label": "Photo (f1)", "image_base64": "AAAA"}]
    ctx = _tool_call_ctx(ToolOutput(success=True, data={"content": "captured", "images": images}))

    await rail.after_tool_call(ctx)

    assert ctx.extra["robotic_arm2_pending_images"] == images
    assert notified == images


@pytest.mark.asyncio
async def test_after_tool_call_notifies_on_tool_result_for_every_call() -> None:
    notified = []

    async def on_tool_result(payload):
        notified.append(payload)

    settings = RoboticArm2RuntimeSettings(on_tool_result=on_tool_result)
    rail = ToolImageRail(settings)
    ctx = _tool_call_ctx(
        ToolOutput(success=True, data={"content": "moved"}),
        tool_name="move_to",
        tool_args={"point_m": [0.1, 0.0, 0.05], "elevation_deg": 90},
        tool_msg=ToolMessage(content="Moved to [0.1, 0.0, 0.05]m.", tool_call_id="c1"),
    )

    await rail.after_tool_call(ctx)

    assert notified == [
        {
            "tool_name": "move_to",
            "tool_args": {"point_m": [0.1, 0.0, 0.05], "elevation_deg": 90},
            "result_text": "Moved to [0.1, 0.0, 0.05]m.",
            "success": True,
        }
    ]


@pytest.mark.asyncio
async def test_after_tool_call_on_tool_result_reflects_failure() -> None:
    notified = []

    async def on_tool_result(payload):
        notified.append(payload)

    settings = RoboticArm2RuntimeSettings(on_tool_result=on_tool_result)
    rail = ToolImageRail(settings)
    ctx = _tool_call_ctx(
        ToolOutput(success=False, error="IK error too large"),
        tool_name="move_to",
        tool_msg=ToolMessage(content="IK error too large", tool_call_id="c1"),
    )

    await rail.after_tool_call(ctx)

    assert notified[0]["success"] is False
    assert notified[0]["result_text"] == "IK error too large"


@pytest.mark.asyncio
async def test_after_tool_call_on_tool_result_disabled_by_default() -> None:
    rail = ToolImageRail(RoboticArm2RuntimeSettings())
    ctx = _tool_call_ctx(ToolOutput(success=True, data={"content": "ok"}))

    await rail.after_tool_call(ctx)  # must not raise with no hook configured


@pytest.mark.asyncio
async def test_before_model_call_no_op_without_pending_images() -> None:
    rail = ToolImageRail(RoboticArm2RuntimeSettings())
    fake_context = _FakeModelContext([UserMessage(content="hello")])
    ctx = _model_call_ctx(context=fake_context)

    await rail.before_model_call(ctx)

    assert fake_context.get_messages() == [UserMessage(content="hello")]


@pytest.mark.asyncio
async def test_before_model_call_merges_into_trailing_user_message() -> None:
    rail = ToolImageRail(RoboticArm2RuntimeSettings())
    fake_context = _FakeModelContext([UserMessage(content="what should I do next?")])
    ctx = _model_call_ctx(
        context=fake_context,
        extra={"robotic_arm2_pending_images": [{"label": "Photo (f1)", "image_base64": "AAAA"}]},
    )
    ctx.inputs.messages = list(fake_context.get_messages())

    await rail.before_model_call(ctx)

    merged = fake_context.get_messages()
    assert len(merged) == 1
    blocks = merged[0].content
    assert {"type": "text", "text": "what should I do next?"} in blocks
    assert any(b.get("type") == "image_url" for b in blocks)
    assert ctx.extra["robotic_arm2_pending_images"] == []


@pytest.mark.asyncio
async def test_before_model_call_appends_new_message_when_no_trailing_user_turn() -> None:
    rail = ToolImageRail(RoboticArm2RuntimeSettings())
    fake_context = _FakeModelContext([])
    ctx = _model_call_ctx(
        context=fake_context,
        extra={"robotic_arm2_pending_images": [{"label": "Photo (f1)", "image_base64": "AAAA"}]},
    )
    ctx.inputs.messages = []

    await rail.before_model_call(ctx)

    merged = fake_context.get_messages()
    assert len(merged) == 1
    assert merged[0].role == "user"
    assert any(b.get("type") == "image_url" for b in merged[0].content)
