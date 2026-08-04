#!/usr/bin/env python
"""Tests for ContextSummarizerRail (ported unchanged from ``robotic_arm``)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openjiuwen.core.foundation.llm import UserMessage
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext, ModelCallInputs
from openjiuwen.harness.tools.robotic_arm2.rails.context_summarizer_rail import (
    ARCHIVED_FRAME_PLACEHOLDER,
    ContextSummarizerRail,
)


class _FakeModelContext:
    def __init__(self, messages: list) -> None:
        self._messages = messages

    def get_messages(self):
        return self._messages

    def set_messages(self, messages):
        self._messages = messages


def _image_message(tag: str) -> UserMessage:
    return UserMessage(
        content=[
            {"type": "text", "text": tag},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{tag}"}},
        ]
    )


@pytest.mark.asyncio
async def test_keeps_only_the_most_recent_n_photos() -> None:
    messages = [_image_message(f"f{i}") for i in range(5)]
    fake_context = _FakeModelContext(list(messages))
    ctx = AgentCallbackContext(agent=MagicMock(), inputs=ModelCallInputs(messages=[]), extra={}, context=fake_context)
    rail = ContextSummarizerRail(frames_to_keep=2)

    await rail.before_model_call(ctx)

    result = fake_context.get_messages()
    has_image = [any(b.get("type") == "image_url" for b in m.content) for m in result]
    assert has_image == [False, False, False, True, True]
    assert result[0].content[-1]["text"] == ARCHIVED_FRAME_PLACEHOLDER


@pytest.mark.asyncio
async def test_no_op_when_under_the_keep_threshold() -> None:
    messages = [_image_message("f0"), _image_message("f1")]
    fake_context = _FakeModelContext(list(messages))
    ctx = AgentCallbackContext(agent=MagicMock(), inputs=ModelCallInputs(messages=[]), extra={}, context=fake_context)
    rail = ContextSummarizerRail(frames_to_keep=3)

    await rail.before_model_call(ctx)

    result = fake_context.get_messages()
    assert all(any(b.get("type") == "image_url" for b in m.content) for m in result)


@pytest.mark.asyncio
async def test_ignores_non_model_call_events() -> None:
    fake_context = _FakeModelContext([_image_message("f0")])
    from openjiuwen.core.single_agent.rail.base import InvokeInputs

    ctx = AgentCallbackContext(agent=MagicMock(), inputs=InvokeInputs(query="go"), extra={}, context=fake_context)
    rail = ContextSummarizerRail(frames_to_keep=0)

    await rail.before_model_call(ctx)

    assert any(b.get("type") == "image_url" for b in fake_context.get_messages()[0].content)
