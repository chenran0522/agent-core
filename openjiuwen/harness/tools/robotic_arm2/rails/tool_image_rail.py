# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Carries image payloads from a tool result to the model's next message, and
fires the caller-supplied observability hooks (``on_frame_captured``/``on_tool_result``).

``Tool.invoke()`` return values always get stringified into the ``ToolMessage``
the model reads (see ``ability_manager._build_tool_message_content``) -- there
is no way for a tool to put an actual image where the model can see it.
Images must still be injected as ``image_url`` content blocks on a
``UserMessage``, the same mechanism ``robotic_arm``'s ``VisionPerceptionRail``
uses -- the only difference here is *when*: instead of capturing/injecting a
photo unconditionally before every model call, this rail only injects
whatever a tool call actually produced (``capture_photo``'s photo,
``segment_scene``'s numbered overlay), and only after the model chose to
call one of those tools.

Mechanism:

1. ``after_tool_call`` reads ``ctx.inputs.tool_result`` -- the tool's raw
   Python return value (a ``ToolOutput``), NOT yet stringified -- and, if its
   ``data["images"]`` is present, stashes those images on
   ``ctx.extra["robotic_arm2_pending_images"]``. It also fires
   ``on_tool_result`` for every tool call (not just robotic-arm2's own tools,
   mirroring ``on_frame_captured``'s scope -- a caller-supplied UI wrapper
   generally wants to see everything the agent does, not just this
   package's 7 tools), and ``on_frame_captured`` once per image.
2. ``before_model_call`` (the model's next turn) drains the pending-images
   list, merges the images into the trailing user message (or appends a new
   one), then clears it.

This works because ``ctx.extra`` is the SAME dict object across the isolated
per-tool-call context and the outer inner-agent context (``ability_manager``
passes ``extra=ctx.extra``, not a copy) -- see ``config.py``'s module
docstring for why tools can't rely on this for their OWN cross-call state,
but it is exactly the right channel for this one-way handoff.
"""

from __future__ import annotations

from typing import Any

from openjiuwen.core.common.logging import logger
from openjiuwen.core.foundation.llm import UserMessage
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext, AgentRail, ModelCallInputs, ToolCallInputs
from openjiuwen.harness.tools.robotic_arm2.config import RoboticArm2RuntimeSettings

_PENDING_IMAGES_KEY = "robotic_arm2_pending_images"


class ToolImageRail(AgentRail):
    """Stash images produced by a tool call; inject them before the model's next turn."""

    priority: int = 90

    def __init__(self, settings: RoboticArm2RuntimeSettings) -> None:
        super().__init__()
        self._on_frame_captured = settings.on_frame_captured
        self._on_tool_result = settings.on_tool_result

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        if not isinstance(ctx.inputs, ToolCallInputs):
            return

        await self._notify_tool_result(ctx.inputs)

        tool_result = ctx.inputs.tool_result
        data = getattr(tool_result, "data", None)
        if not isinstance(data, dict):
            return
        images = data.get("images")
        if not images:
            return
        pending: list[dict] = ctx.extra.setdefault(_PENDING_IMAGES_KEY, [])
        pending.extend(images)
        await self._notify_frame_captured(images)

    async def _notify_tool_result(self, inputs: ToolCallInputs) -> None:
        if self._on_tool_result is None:
            return
        tool_result = inputs.tool_result
        success = getattr(tool_result, "success", None)
        payload = {
            "tool_name": inputs.tool_name,
            "tool_args": inputs.tool_args,
            "result_text": self._extract_result_text(inputs),
            "success": True if success is None else bool(success),
        }
        try:
            await self._on_tool_result(payload)
        except Exception:  # noqa: BLE001 -- caller's hook must never break the perception/action loop
            logger.exception("[ToolImageRail] on_tool_result callback failed")

    @staticmethod
    def _extract_result_text(inputs: ToolCallInputs) -> str:
        content = getattr(inputs.tool_msg, "content", None)
        if isinstance(content, str):
            return content
        return "" if content is None else str(content)

    async def _notify_frame_captured(self, images: list[dict]) -> None:
        if self._on_frame_captured is None:
            return
        for image in images:
            try:
                await self._on_frame_captured(image)
            except Exception:  # noqa: BLE001 -- caller's hook must never break the perception/action loop
                logger.exception("[ToolImageRail] on_frame_captured callback failed")

    async def before_model_call(self, ctx: AgentCallbackContext) -> None:
        if not isinstance(ctx.inputs, ModelCallInputs):
            return
        pending: list[dict] = ctx.extra.get(_PENDING_IMAGES_KEY) or []
        if not pending:
            return
        ctx.extra[_PENDING_IMAGES_KEY] = []

        content_blocks: list[dict] = []
        for image in pending:
            image_base64 = image.get("image_base64")
            if not image_base64:
                continue
            label = image.get("label") or "Photo"
            content_blocks.append({"type": "text", "text": f"[{label}]"})
            content_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "high"},
                }
            )
        if not content_blocks:
            return

        await self._inject_images(ctx, content_blocks)
        logger.info("[ToolImageRail] injected %s image(s) before next model call", len(pending))

    async def _inject_images(self, ctx: AgentCallbackContext, content_blocks: list[dict]) -> None:
        if not ctx.context:
            ctx.inputs.messages = list(ctx.inputs.messages) + [UserMessage(content=content_blocks)]
            return

        ctx_messages = ctx.context.get_messages()
        last_msg = ctx_messages[-1] if ctx_messages else None
        if last_msg is not None and last_msg.role == "user":
            popped = ctx.context.pop_messages(1)
            last_user_msg = popped[0]
            last_user_msg.content = self._to_content_blocks(last_user_msg.content) + content_blocks
            await ctx.context.add_messages(last_user_msg)

            new_inputs = list(ctx.inputs.messages)
            for i in range(len(new_inputs) - 1, -1, -1):
                if new_inputs[i].role == "user":
                    new_inputs[i] = last_user_msg
                    break
            ctx.inputs.messages = new_inputs
            return

        new_msg = UserMessage(content=content_blocks)
        await ctx.context.add_messages(new_msg)
        ctx.inputs.messages = list(ctx.inputs.messages) + [new_msg]

    @staticmethod
    def _to_content_blocks(content: Any) -> list[dict]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            return list(content)
        return [{"type": "text", "text": str(content)}]


__all__ = ["ToolImageRail"]
