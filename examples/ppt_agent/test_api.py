#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最简 create_deep_agent 测试：用 .env 里的 API_KEY / API_BASE / MODEL_NAME / MODEL_PROVIDER
搭一个最小 DeepAgent（无 skills、无 web tools），跑一句简单 query，验证真实链路是否可用。

用法：
    export PYTHONPATH=.:$PYTHONPATH
    python examples/ppt_agent/test_api.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from openjiuwen.core.foundation.llm import init_model
from openjiuwen.core.runner import Runner
from openjiuwen.core.single_agent.schema.agent_card import AgentCard
from openjiuwen.harness import create_deep_agent
from openjiuwen.harness.workspace.workspace import Workspace


async def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")

    api_base = os.getenv("API_BASE", "")
    api_key = os.getenv("API_KEY", "")
    model_name = os.getenv("MODEL_NAME", "")
    model_provider = os.getenv("MODEL_PROVIDER", "")

    print(f"provider={model_provider!r} model={model_name!r} api_base={api_base!r}")

    model = init_model(
        provider=model_provider,
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
    )

    workspace_dir = str(Path(__file__).resolve().parent / "workspace_test")
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)

    agent = create_deep_agent(
        model,
        card=AgentCard(name="test_agent", description="Minimal test agent"),
        system_prompt="你是一个测试助手，直接回答用户问题即可。",
        tools=[],
        rails=[],
        enable_task_planning=False,
        add_general_purpose_agent=False,
        max_iterations=1,
        workspace=Workspace(root_path=workspace_dir, language="cn"),
        language="cn",
    )

    await Runner.start()
    try:
        result = await Runner.run_agent(agent, {"query": "回复 pong 即可"})
        print(f"[OK] {result.get('output', result)!r}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {type(e).__name__}: {e}")
    finally:
        await Runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
