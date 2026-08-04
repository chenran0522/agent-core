#!/usr/bin/env python
"""Tests for the ArmBackend model-name registry."""

from __future__ import annotations

import pytest

from openjiuwen.harness.tools.robotic_arm2.registry import ArmBackendRegistry


@pytest.fixture(autouse=True)
def _cleanup_registry():
    before = dict(ArmBackendRegistry._registry)
    yield
    ArmBackendRegistry._registry = before


def test_register_and_create() -> None:
    @ArmBackendRegistry.register("test-rig")
    class FakeBackend:
        def __init__(self, arm_ip: str) -> None:
            self.arm_ip = arm_ip

    backend = ArmBackendRegistry.create("test-rig", arm_ip="10.0.0.1")

    assert isinstance(backend, FakeBackend)
    assert backend.arm_ip == "10.0.0.1"


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backend model"):
        ArmBackendRegistry.create("does-not-exist")
