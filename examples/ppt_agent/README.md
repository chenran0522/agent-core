# PPT Agent 示例

基于 DeepAgent（harness）构建的 PPT 生成助手。给定一个技术主题，agent 会检索参考资料、
梳理大纲，并调用 `workspace/skills/` 里的 PPT 生成技能包产出可编辑的 `.pptx` 文件。

## 1. 安装环境

在仓库根目录（`agent-core/`）下安装依赖：

```bash
uv sync
```

PPT 生成技能包自己的 Python 依赖也要单独装（`python-pptx`、`beautifulsoup4`、`lxml`、
`playwright`、`latex2mathml`、`mathml2omml`、`pillow` 等，不在 `agent-core` 核心依赖里）：

```bash
uv pip install -r examples/ppt_agent/workspace/skills/pptagent-pipeline/requirements.txt
python -m playwright install chromium
```

`playwright install chromium` 是给 `html-to-pptx`/`designer` 子技能里的排版探测、HTML 转
图片流程用的无头浏览器，装完 pip 包并不会自动带上浏览器内核，必须单独跑这一步。

## 2. 配置 `.env`

在 `examples/ppt_agent/.env` 下配置：

```bash
# 必填：模型
API_BASE=...
API_KEY=...
MODEL_NAME=...          # 例如 anthropic/claude-sonnet-4.5、openai/gpt-5.5
MODEL_PROVIDER=...      # 例如 OpenRouter

# 可选
AGENT_LANGUAGE=cn        # 默认 cn
MAX_ITERATIONS=40        # 默认 40

# 可选：向量记忆（两个都配才会启用 MemoryRail，任一没配就跳过，不报错）
EMBEDDING_MODEL_NAME=...
EMBEDDING_BASE_URL=...
EMBEDDING_API_KEY=...

# 可选：付费网络搜索兜底（不设也能用 WebFreeSearchTool 免费搜索）
BOCHA_API_KEY=...
FREE_SEARCH_DDG_ENABLED=true
# 可选调试开关：排查是哪个 rail/tool 出问题时用，默认都是 "0"（保持原行为）
DEBUG_DISABLE_SKILLS=0
DEBUG_DISABLE_WEB_TOOLS=0
DEBUG_DISABLE_TASK_PLANNING=0
DEBUG_DISABLE_SUBAGENT=0
```

## 3. 放置 Skill 包 和 参考资料

`workspace/` 目录不会预先存在（第一次用这个示例时是空的），`skills/` 和 `source/`
**都需要自己手动创建目录**，再把对应内容放进去——`main.py` 只会在运行时补建空目录，
不会帮你把内容放进去。

### Skill 包

PPT 生成技能本身不在这个示例里提供：

```bash
mkdir -p examples/ppt_agent/workspace/skills/pptagent-pipeline
# 然后把技能包内容（SKILL.md 及其它文件）复制/解压到这个目录下
```

最终要落成这个路径：

```
examples/ppt_agent/workspace/skills/<skill-name>/SKILL.md
```

**必须放在 `workspace/skills/` 下面，不能放在跟 `workspace/` 同级的独立 `skills/` 目录。**
`workspace/` 是 `SysOperation` 的沙箱根目录，`SkillUseRail` 的描述预读、模型的
`read_file`、`skill_tool` 内部读取全部走同一个沙箱化文件访问器——一旦 skill 目录落在
沙箱外，三条路径会全部被拒绝访问，技能包彻底不可用。

### 参考资料（PDF 等）

同样需要自己建目录再放文件进去：

```bash
mkdir -p examples/ppt_agent/workspace/source
# 把参考 PDF 直接放进这个目录，不用建子文件夹，文件名随意
```

`materials_dir = workspace_dir / "source"`，agent 生成前会先 `list_files` 检查这个目录：
放了 PDF 就会读你提供的材料，没放（目录为空）agent 会自己联网搜索/下载论文来凑材料。

两者都要**在跑 `python examples/ppt_agent/main.py` 之前**放好——`SkillUseRail` 只在启动时
扫描一次 skill 目录，`source/` 也只在这次运行开始时检查一次，跑起来之后中途再放不会
被这次运行用上，只能等下一次运行生效。

## 4. 跑脚本

在仓库根目录下执行（`PYTHONPATH` 要指到仓库根目录，这样才能 `import openjiuwen`）：

```bash
export PYTHONPATH=.:$PYTHONPATH
```

**先跑一次连通性测试**（不带 skills/web tools，最小化排查模型链路是否通）：

```bash
python examples/ppt_agent/test_api.py
```

看到 `[OK] 'pong'` 说明 API_BASE/API_KEY/MODEL_NAME/MODEL_PROVIDER 配置没问题，再跑正式流程：

```bash
python examples/ppt_agent/main.py
```

`main.py` 里的 query 是写死的一句 RLHF 对齐技术演进的 PPT 生成需求，改需求直接编辑
`main.py` 里 `query` 变量。

## 5. 在哪里查看结果

### LLM / 运行日志

`examples/ppt_agent/logs/<运行开始时间>/` —— 每次运行单独一个按本地时间命名
（`YYYY-MM-DD_HH-MM-SS`）的文件夹，不会跟其它运行的日志混在一起：

| 文件 | 内容 |
|---|---|
| `llm.log` | 每次 LLM 请求/响应的完整记录（含 messages、tool 定义、token 用量、报错） |
| `run/jiuwen.log` | 主执行日志：ReAct 迭代、工具调用、最终回复摘要 |
| `runner.log` | Runner 启停、任务调度相关事件 |
| `session.log` | session/checkpoint 相关事件 |
| `sys_operation.log` | 每次文件/终端操作的详细记录（含沙箱校验结果，`"code": 0` 是成功） |
| `tool.log` | 工具调用相关事件 |
| `interface/`、`performance/` | 内置的接口日志、性能指标日志（这个 example 目前不产生 performance 数据，文件会是空的） |

### PPT 产出物

`examples/ppt_agent/workspace/output/<运行开始时间>/` —— 同样按运行时间戳分文件夹，
里面是生成的 `.pptx`、生成脚本、大纲等文件。时间戳跟对应的 `logs/<时间戳>/` 文件夹是
同一个，可以对着查。

