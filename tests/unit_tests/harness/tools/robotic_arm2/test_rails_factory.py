#!/usr/bin/env python
"""Tests for build_robotic_arm2_rails: backend_model resolution and rail assembly."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openjiuwen.harness.tools.robotic_arm2.config import RoboticArm2RuntimeSettings
from openjiuwen.harness.tools.robotic_arm2.rails.context_summarizer_rail import ContextSummarizerRail
from openjiuwen.harness.tools.robotic_arm2.rails.tool_image_rail import ToolImageRail
from openjiuwen.harness.tools.robotic_arm2.rails_factory import build_robotic_arm2_rails
from openjiuwen.harness.tools.robotic_arm2.registry import ArmBackendRegistry


@pytest.fixture(autouse=True)
def _cleanup_registry():
    before = dict(ArmBackendRegistry._registry)
    yield
    ArmBackendRegistry._registry = before


def test_direct_backend_is_untouched() -> None:
    backend = MagicMock()
    settings = RoboticArm2RuntimeSettings(backend=backend, health_check=False)

    build_robotic_arm2_rails(settings)

    assert settings.backend is backend


def test_backend_model_resolves_and_backfills_settings() -> None:
    @ArmBackendRegistry.register("unit-test-rig")
    class FakeBackend:
        def __init__(self, arm_ip: str) -> None:
            self.arm_ip = arm_ip

    settings = RoboticArm2RuntimeSettings(
        backend_model="unit-test-rig", backend_params={"arm_ip": "10.0.0.5"}, health_check=False
    )

    build_robotic_arm2_rails(settings)

    assert isinstance(settings.backend, FakeBackend)
    assert settings.backend.arm_ip == "10.0.0.5"


def test_returns_both_rails() -> None:
    settings = RoboticArm2RuntimeSettings(backend=MagicMock(), health_check=False)

    rails = build_robotic_arm2_rails(settings)

    assert any(isinstance(r, ToolImageRail) for r in rails)
    assert any(isinstance(r, ContextSummarizerRail) for r in rails)
    assert len(rails) == 2


def test_missing_backend_and_model_raises() -> None:
    settings = RoboticArm2RuntimeSettings()

    with pytest.raises(ValueError, match="backend or backend_model"):
        build_robotic_arm2_rails(settings)


def test_health_check_calls_backend_when_enabled() -> None:
    backend = MagicMock()
    settings = RoboticArm2RuntimeSettings(backend=backend, health_check=True)

    build_robotic_arm2_rails(settings)

    backend.health_check.assert_called_once()
