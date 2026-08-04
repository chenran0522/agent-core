# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Resolve ``backend_model`` -> instance and assemble the two rails.

Only two rails remain here, versus ``robotic_arm``'s three: there is no
``StepExecutorRail`` equivalent, because there is no hidden pipeline for a
rail to run automatically -- every capability is a tool the model calls
itself (see ``tools/``).
"""

from __future__ import annotations

from openjiuwen.core.single_agent.rail.base import AgentRail
from openjiuwen.harness.tools.robotic_arm2.config import RoboticArm2RuntimeSettings
from openjiuwen.harness.tools.robotic_arm2.rails.context_summarizer_rail import ContextSummarizerRail
from openjiuwen.harness.tools.robotic_arm2.rails.tool_image_rail import ToolImageRail
from openjiuwen.harness.tools.robotic_arm2.registry import ArmBackendRegistry


def resolve_backend(settings: RoboticArm2RuntimeSettings) -> None:
    """Resolve ``backend_model`` to an instance once, ahead of any tool/rail use.

    Idempotent: only fills in when ``settings.backend`` is still ``None``.
    """
    if settings.backend is None and settings.backend_model is not None:
        settings.backend = ArmBackendRegistry.create(settings.backend_model, **settings.backend_params)


def build_robotic_arm2_rails(settings: RoboticArm2RuntimeSettings) -> list[AgentRail]:
    resolve_backend(settings)
    if settings.backend is None:
        raise ValueError(
            "RoboticArm2RuntimeSettings.backend or backend_model is required: either supply an "
            "ArmBackend instance directly, or set backend_model to a name registered via "
            "ArmBackendRegistry.register(...)."
        )
    if settings.health_check and hasattr(settings.backend, "health_check"):
        settings.backend.health_check()
    return [
        ToolImageRail(settings),
        ContextSummarizerRail(settings.mcs_screenshots_to_keep),
    ]


__all__ = ["build_robotic_arm2_rails", "resolve_backend"]
