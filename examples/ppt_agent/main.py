#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""PPT Agent 示例：基于 DeepAgent（harness）构建的 PPT 生成助手。

装配的能力：
- 核心文件/终端工具（SysOperationRail）：
  read_file / write_file / edit_file / glob / list_files / grep / bash / powershell
- 任务派发 + 用户询问：task（SubagentRail，经由 add_general_purpose_agent 注入）、ask_user（AskUserRail）
- Todo 管理（TaskPlanningRail，经由 enable_task_planning 注入）：
  todo_create / todo_list / todo_get / todo_modify
- Skill 系统 + 技能演进链路：
  SkillUseRail（scan `<workspace>/skills/` 目录）+ configure_skill_evolution（
  prepare_skill_evolution / evolve_review_task / list_skill_experiences /
  read_skill_experiences / evolve_skill_experiences / simplify_skill_experiences）
- 向量记忆（MemoryRail，需 EMBEDDING_* 环境变量）：
  memory_search / memory_get / write_memory / edit_memory / read_memory
- 联网搜索：WebFreeSearchTool / WebPaidSearchTool（bocha，需 BOCHA_API_KEY）/ WebFetchWebpageTool

PPT 生成技能本身不在此示例中提供，请把技能包放进 `<workspace>/skills/` 目录（workspace 是
SysOperation 的沙箱根目录。skills 目录必须落在里面——SkillUseRail 的描述预读、模型的
read_file、skill_tool 内部读取全部走同一个沙箱化文件访问器，一旦 skills 目录落在沙箱外，
三条路径会全部被拒绝，技能包彻底不可用，不是"仅报个 WARNING"那么轻）。

不需要先运行一次 main.py 才能有 workspace 目录：手动 mkdir 出
`<workspace>/skills/<skill-name>/` 再把技能包放进去即可，main.py 里的
`mkdir(parents=True, exist_ok=True)` 是幂等的，不会清空已有内容。

运行方式：
    export PYTHONPATH=.:$PYTHONPATH
    python examples/ppt_agent/main.py
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# 日志路径固定到本示例目录下，不随运行时的当前工作目录漂移（./logs/ 默认按 CWD 解析）。
# 每次运行（= 一次 query = 一个 session）单独用一个按本地时间命名的子目录，
# 这次运行产生的 llm.log / run/jiuwen.log / runner.log / session.log /
# sys_operation.log / tool.log / interface/ / performance/ 全部落在这个子目录里，
# 不会跟其它运行的日志混在一起、也不会散落在 logs/ 根目录下。
# 必须在其它 openjiuwen.* 导入之前完成配置，避免早期的 import-time 日志散落到默认位置。
from openjiuwen.core.common.logging.default.constant import DEFAULT_INNER_LOG_CONFIG
from openjiuwen.core.common.logging.log_config import configure_log_config

_RUN_STARTED_AT = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_LOG_DIR = Path(__file__).resolve().parent / "logs" / _RUN_STARTED_AT
configure_log_config({
    **DEFAULT_INNER_LOG_CONFIG,
    "log_path": str(_LOG_DIR),
})

from openjiuwen.core.common.logging import logger  # noqa: E402
from openjiuwen.core.foundation.llm import init_model  # noqa: E402
from openjiuwen.core.runner import Runner  # noqa: E402
from openjiuwen.core.single_agent.schema.agent_card import AgentCard  # noqa: E402
from openjiuwen.harness import create_deep_agent  # noqa: E402
from openjiuwen.harness.rails import AskUserRail, MemoryRail, SkillUseRail, configure_skill_evolution  # noqa: E402
from openjiuwen.harness.rails.sys_operation_rail import SysOperationRail  # noqa: E402
from openjiuwen.harness.tools import create_web_tools  # noqa: E402
from openjiuwen.harness.workspace.workspace import Workspace  # noqa: E402


def _build_memory_rail(api_key: str, api_base: str) -> Optional[MemoryRail]:
    from openjiuwen.core.memory.lite.embeddings import resolve_embedding_config_from_env

    embedding_config = resolve_embedding_config_from_env(
        model_name="text-embedding-3-small",
        fallback_base_url=api_base,
        fallback_api_key=api_key,
    )
    if embedding_config is None:
        return None
    return MemoryRail(embedding_config=embedding_config)


async def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")

    api_base = os.getenv("API_BASE", "")
    api_key = os.getenv("API_KEY", "")
    model_name = os.getenv("MODEL_NAME", "")
    model_provider = os.getenv("MODEL_PROVIDER", "")
    language = os.getenv("AGENT_LANGUAGE", "cn")
    max_iterations = int(os.getenv("MAX_ITERATIONS", "40"))
    workspace_dir = os.getenv("WORKSPACE_DIR") or str(Path(__file__).resolve().parent / "workspace")
    # skills 目录必须落在 workspace 沙箱根目录内部（Workspace 的标准 SKILLS 节点），
    # 否则 SysOperation 的 sandbox 校验会拒绝 SkillUseRail/read_file/skill_tool 的读取。
    skills_dir = Path(workspace_dir) / "skills"

    # DEBUG 开关：逐项排查是否某个 rail/tool 触发 401，默认全部保持原行为（启用）
    disable_skills = os.getenv("DEBUG_DISABLE_SKILLS", "0") == "1"
    disable_web_tools = os.getenv("DEBUG_DISABLE_WEB_TOOLS", "0") == "1"
    disable_task_planning = os.getenv("DEBUG_DISABLE_TASK_PLANNING", "0") == "1"
    disable_subagent = os.getenv("DEBUG_DISABLE_SUBAGENT", "0") == "1"

    model = init_model(
        provider=model_provider,
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
    )

    materials_dir = Path(workspace_dir) / "source"
    # 产出物（pptx / 生成脚本等）按本次运行的时间戳单独建目录，避免下一次运行用
    # 同样的文件名把这次的结果覆盖掉。复用 _RUN_STARTED_AT，方便跟 logs/ 里对应的
    # 那次运行日志对上号。workspace 根目录本身（AGENT.md/记忆/skills/source）
    # 仍然是所有运行共享、持续累积的，不跟着按次重建。
    output_dir = Path(workspace_dir) / "output" / _RUN_STARTED_AT

    Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    skills_dir.mkdir(parents=True, exist_ok=True)
    materials_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 把 source/ 里实际有哪些文件直接列进 system_prompt（而不是只说"去看那个目录"）。
    # 之前跑过发现：顶层 agent 确实会 list_files 看一眼 source/，但把研究工作委托给
    # task 工具派生的子智能体后，子智能体是全新会话上下文，不会自动继承"source/ 里有
    # 这几个文件"这条信息，结果重新联网搜索/下载了一遍同样的论文，source/ 里的 PDF
    # 全程没被 read_file 过一次。直接把文件路径列出来 + 明确要求把路径转发给子智能体，
    # 比只说"生成前看一下目录"更能让文件真正被用上。
    materials_files = sorted(p.name for p in materials_dir.iterdir() if p.is_file())
    if materials_files:
        materials_listing = "\n".join(f"  - {materials_dir / name}" for name in materials_files)
        materials_instruction = (
            f"参考资料目录 '{materials_dir}' 下已有以下文件，生成前必须先用 read_file 逐个读取，"
            f"把它们作为主要信息来源，不要重新联网搜索/下载同主题的资料：\n{materials_listing}\n"
            "如果把资料研究工作委托给子智能体（task 工具），必须把上面这些具体文件的完整路径"
            "原样写进子智能体的任务描述里，要求子智能体直接读这些本地文件，不要自己重新搜索。"
        )
    else:
        materials_instruction = (
            f"用户提供的参考资料（如 PDF）放在 '{materials_dir}' 目录下，生成前请先查看该目录；"
            "目录为空的话可以自行联网搜索资料。"
        )

    system_prompt = (
        "你是一个 PPT 生成助手，擅长根据用户需求梳理大纲、组织内容并生成演示文稿文件。\n"
        f"{materials_instruction}\n"
        f"将生成的所有文件放入 '{output_dir}' 目录（本次运行专属的输出目录，不要写到其它位置，"
        "避免覆盖之前运行生成的文件）。\n"
    )

    rails: list[Any] = [
        SysOperationRail(),
        AskUserRail(),
    ]
    if not disable_skills:
        rails.append(SkillUseRail(skills_dir=[str(skills_dir)], skill_mode="all", include_tools=False))

    memory_rail = _build_memory_rail(api_key, api_base)
    if memory_rail is not None:
        rails.append(memory_rail)

    agent = create_deep_agent(
        model,
        card=AgentCard(name="ppt_agent", description="PPT Generation Agent"),
        system_prompt=system_prompt,
        tools=[] if disable_web_tools else create_web_tools(language=language),
        rails=rails,
        enable_task_planning=not disable_task_planning,
        add_general_purpose_agent=not disable_subagent,
        max_iterations=max_iterations,
        workspace=Workspace(root_path=workspace_dir, language=language),
        language=language,
    )

    if not disable_skills:
        configure_skill_evolution(
            agent,
            skills_dir=[str(skills_dir)],
            llm=model,
            model=model_name,
            auto_save=False,
            language=language,
        )

    await Runner.start()
    try:
        query = "Please generate a technical presentation on the alignment of LLMs using human feedback. Compare the methodological evolution from the initial specific-task reward modeling (2017) to the generalized PPO approach used in InstructGPT. Specifically, analyze how the 'Reward Model' architecture and the 'Policy Optimization' phase differ across these papers to reduce toxicity and hallucination."
        result = await Runner.run_agent(agent, {"query": query})
        logger.info(result.get("output", result))
    finally:
        await Runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
