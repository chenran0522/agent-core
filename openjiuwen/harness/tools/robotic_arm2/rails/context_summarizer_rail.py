# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Replace ``image_url`` on older photo user turns with a fixed placeholder.

Ported unchanged from ``robotic_arm.rails.context_summarizer_rail`` -- the
logic doesn't depend on how images got into the message history (auto-capture
every turn, vs. tool-triggered injection via ``ToolImageRail`` here), only on
the fact that ``user`` messages can carry ``image_url`` blocks that pile up
over a long manipulation sequence and need pruning to bound context size/cost.
"""

from __future__ import annotations

from openjiuwen.core.common.logging import logger
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext, AgentRail, ModelCallInputs

ARCHIVED_FRAME_PLACEHOLDER = (
    "Photo from an earlier step is no longer attached to save context. The scene may have "
    "changed since then; rely on the latest image in this conversation for the current state, "
    "or call capture_photo again if you need a fresh one."
)


def _has_image_url(msg) -> bool:
    if not isinstance(msg.content, list):
        return False
    return any(isinstance(b, dict) and b.get("type") == "image_url" for b in msg.content)


def _replace_archived_frame_images(msg) -> None:
    if not isinstance(msg.content, list):
        return
    next_content: list = []
    for block in msg.content:
        if isinstance(block, dict) and block.get("type") == "image_url":
            next_content.append({"type": "text", "text": ARCHIVED_FRAME_PLACEHOLDER})
        else:
            next_content.append(block)
    msg.content = next_content or [{"type": "text", "text": ARCHIVED_FRAME_PLACEHOLDER}]


class ContextSummarizerRail(AgentRail):
    """Keep only the most recent ``frames_to_keep`` photos as real images."""

    priority: int = 85

    def __init__(self, frames_to_keep: int = 3) -> None:
        super().__init__()
        self._frames_to_keep = frames_to_keep

    async def before_model_call(self, ctx: AgentCallbackContext) -> None:
        if not isinstance(ctx.inputs, ModelCallInputs):
            return
        if ctx.context is None:
            return
        self._archive_old_frame_images(ctx)

    def _archive_old_frame_images(self, ctx: AgentCallbackContext) -> None:
        all_msgs = ctx.context.get_messages()
        if not all_msgs:
            return

        frame_indices = [i for i, msg in enumerate(all_msgs) if msg.role == "user" and _has_image_url(msg)]
        if len(frame_indices) <= self._frames_to_keep:
            return

        old_indices = frame_indices[: -self._frames_to_keep]
        replaced = 0
        for idx in old_indices:
            _replace_archived_frame_images(all_msgs[idx])
            replaced += 1

        if replaced:
            ctx.context.set_messages(all_msgs)
            logger.info("[ContextSummarizerRail] replaced photo images in %s older turn(s)", replaced)


__all__ = ["ContextSummarizerRail"]
