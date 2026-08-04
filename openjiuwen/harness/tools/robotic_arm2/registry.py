# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Name-keyed registry for ``ArmBackend`` vendor adapters.

Mirrors ``robotic_arm.registry.SubTaskExecutorRegistry``: a concrete vendor
backend class registers itself under a short model name via a decorator;
callers select a model by name instead of importing and constructing the
vendor class directly. Passing an already-constructed ``backend=`` on
:class:`RoboticArm2RuntimeSettings` remains the escape hatch for one-off or
unregistered rigs.
"""

from __future__ import annotations

from typing import Any, ClassVar


class ArmBackendRegistry:
    """Registry mapping a rig name to its ``ArmBackend`` implementation class."""

    _registry: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, model_name: str):
        def decorator(backend_cls: type) -> type:
            cls._registry[model_name] = backend_cls
            return backend_cls

        return decorator

    @classmethod
    def create(cls, model_name: str, **kwargs: Any) -> Any:
        backend_cls = cls._registry.get(model_name)
        if backend_cls is None:
            raise ValueError(f"Unknown backend model {model_name!r}. Registered models: {sorted(cls._registry)}")
        return backend_cls(**kwargs)


__all__ = ["ArmBackendRegistry"]
