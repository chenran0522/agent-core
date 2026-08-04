# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Factory helpers for the robotic-arm2 (tool-driven perception + action) subagent.

Use ``factory_name="robotic_arm2_agent"`` on :class:`SubAgentConfig` so
:class:`DeepAgent` dispatches to :func:`create_robotic_arm2_agent`.

Compare with ``robotic_arm_agent.py``: there the model only calls
``report_plan`` and a rail runs a hidden capture->CV->VLM->IK->drive pipeline
automatically; here the model calls ``capture_photo``/``segment_scene``/
``pixel_to_3d``/``check_reachability``/``move_to``/``set_gripper``/
``get_gripper_pose`` itself, in whatever order it decides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openjiuwen.core.context_engine.schema.config import ContextEngineConfig
from openjiuwen.core.foundation.llm.model import Model
from openjiuwen.core.foundation.tool import McpServerConfig, Tool, ToolCard
from openjiuwen.core.single_agent.rail.base import AgentRail
from openjiuwen.core.single_agent.schema.agent_card import AgentCard
from openjiuwen.core.sys_operation import SysOperation
from openjiuwen.harness.deep_agent import DeepAgent
from openjiuwen.harness.factory import create_deep_agent
from openjiuwen.harness.schema.config import SubAgentConfig
from openjiuwen.harness.tools.robotic_arm2.arm_task_prompt import build_robotic_arm2_system_prompt
from openjiuwen.harness.tools.robotic_arm2.config import RoboticArm2RuntimeSettings
from openjiuwen.harness.tools.robotic_arm2.rails_factory import build_robotic_arm2_rails
from openjiuwen.harness.tools.robotic_arm2.tools import build_robotic_arm2_tools

if TYPE_CHECKING:
    from openjiuwen.harness.workspace.workspace import Workspace

try:
    from openjiuwen.harness.prompts import resolve_language
except ImportError:

    def resolve_language(language: str | None = None) -> str:  # type: ignore[misc]
        return language if language in {"cn", "en"} else "cn"


DEFAULT_ROBOTIC_ARM2_DESCRIPTION_EN = (
    "Dedicated robotic-arm subagent (tool-driven): captures photos, segments objects, "
    "backprojects pixels to 3D, checks reachability, and moves/grips the arm via explicit tool "
    "calls -- the model itself decides target points and approach angles, there is no hidden "
    "auto-execution pipeline."
)
DEFAULT_ROBOTIC_ARM2_DESCRIPTION_CN = (
    "专用机械臂子代理（工具驱动）：拍照、分割物体、把像素反投影为 3D 坐标、检查可达性、"
    "通过显式工具调用移动/控制夹爪——目标点和接近角度均由模型自己决定，没有隐藏的自动执行流水线。"
)

DEFAULT_ROBOTIC_ARM2_DESCRIPTION: dict[str, str] = {
    "cn": DEFAULT_ROBOTIC_ARM2_DESCRIPTION_CN,
    "en": DEFAULT_ROBOTIC_ARM2_DESCRIPTION_EN,
}


def _resolve_runtime_settings(settings: RoboticArm2RuntimeSettings | None) -> RoboticArm2RuntimeSettings:
    if settings is not None:
        return settings
    return RoboticArm2RuntimeSettings.from_env()


def build_robotic_arm2_agent_config(
    model: Model,
    *,
    card: AgentCard | None = None,
    system_prompt: str | None = None,
    tools: list[Tool | ToolCard] | None = None,
    mcps: list[McpServerConfig] | None = None,
    rails: list[AgentRail] | None = None,
    enable_task_loop: bool = False,
    max_iterations: int = 30,
    workspace: str | Workspace | None = None,
    skills: list[str] | None = None,
    backend: Any | None = None,
    sys_operation: SysOperation | None = None,
    language: str | None = None,
    prompt_mode: str | None = None,
    settings: RoboticArm2RuntimeSettings | None = None,
) -> SubAgentConfig:
    """Build a SubAgentConfig that materializes as :func:`create_robotic_arm2_agent`.

    ``settings`` (see :class:`RoboticArm2RuntimeSettings`) is where callers supply the
    ``backend`` a specific rig needs -- there is no generic default hardware to fall back to.
    """
    resolved_language = resolve_language(language)
    resolved_settings = _resolve_runtime_settings(settings)
    default_prompt = build_robotic_arm2_system_prompt(resolved_settings)
    return SubAgentConfig(
        agent_card=card
        or AgentCard(
            name="robotic_arm2_agent",
            description=DEFAULT_ROBOTIC_ARM2_DESCRIPTION.get(resolved_language, DEFAULT_ROBOTIC_ARM2_DESCRIPTION["cn"]),
        ),
        system_prompt=system_prompt or default_prompt,
        tools=list(tools or []),
        mcps=list(mcps or []),
        model=model,
        rails=rails,
        skills=skills,
        backend=backend,
        workspace=workspace,
        sys_operation=sys_operation,
        language=resolved_language,
        prompt_mode=prompt_mode,
        enable_task_loop=enable_task_loop,
        max_iterations=max_iterations,
        factory_name="robotic_arm2_agent",
        factory_kwargs={"settings": resolved_settings},
    )


def create_robotic_arm2_agent(
    model: Model,
    *,
    card: AgentCard | None = None,
    system_prompt: str | None = None,
    tools: list[Tool | ToolCard] | None = None,
    mcps: list[McpServerConfig] | None = None,
    subagents: list[SubAgentConfig | DeepAgent] | None = None,
    rails: list[AgentRail] | None = None,
    enable_task_loop: bool = False,
    max_iterations: int = 30,
    workspace: str | Workspace | None = None,
    skills: list[str] | None = None,
    backend: Any | None = None,
    sys_operation: SysOperation | None = None,
    language: str | None = None,
    prompt_mode: str | None = None,
    settings: RoboticArm2RuntimeSettings | None = None,
    **config_kwargs: Any,
) -> DeepAgent:
    """Create a DeepAgent wired with the robotic-arm2 perception/action tools.

    Args:
        model: Pre-constructed Model instance (must support image inputs).
        settings: The ``backend`` pipeline -- see :class:`RoboticArm2RuntimeSettings`.
            Required field (``backend`` or ``backend_model``) has no default and
            must be supplied by the caller for this specific rig.
        **config_kwargs: Extra fields forwarded to DeepAgentConfig.

    Returns:
        Configured DeepAgent instance ready for invoke()/stream().
    """
    resolved_language = resolve_language(language)
    resolved_settings = _resolve_runtime_settings(settings)

    final_card = card or AgentCard(
        name="robotic_arm2_agent",
        description=DEFAULT_ROBOTIC_ARM2_DESCRIPTION.get(resolved_language, DEFAULT_ROBOTIC_ARM2_DESCRIPTION["cn"]),
    )
    default_prompt = build_robotic_arm2_system_prompt(resolved_settings)
    final_prompt = system_prompt or default_prompt

    injected_rails = build_robotic_arm2_rails(resolved_settings)
    injected_tools = build_robotic_arm2_tools(resolved_settings.backend)
    final_tools: list[Tool | ToolCard] = list(tools or []) + injected_tools
    final_rails: list[AgentRail] = list(rails or []) + injected_rails

    extra_config: dict[str, Any] = dict(config_kwargs)
    if extra_config.get("context_engine_config") is None:
        extra_config["context_engine_config"] = ContextEngineConfig(
            max_context_message_num=resolved_settings.context_max_message_num,
            default_window_round_num=resolved_settings.context_default_window_round_num,
        )

    return create_deep_agent(
        model=model,
        card=final_card,
        system_prompt=final_prompt,
        tools=final_tools,
        mcps=mcps,
        subagents=subagents,
        rails=final_rails,
        enable_task_loop=enable_task_loop,
        max_iterations=max_iterations,
        workspace=workspace,
        skills=skills,
        backend=backend,
        sys_operation=sys_operation,
        language=resolved_language,
        prompt_mode=prompt_mode,
        **extra_config,
    )


__all__ = [
    "build_robotic_arm2_agent_config",
    "create_robotic_arm2_agent",
]
