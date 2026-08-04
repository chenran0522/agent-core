# 机械臂操作流水线 2.0（工具驱动）/ Robotic-Arm Manipulation Pipeline 2.0 (Tool-Driven)

代码路径 · Code root: `openjiuwen/harness/tools/robotic_arm2/`

本文档面向需要理解本模块架构与设计决策的同事，中英对照。与 `robotic_arm`（见
[`docs/robotic-arm-so101-pipeline.md`](./robotic-arm-so101-pipeline.md)）并列存在，
两者互不依赖，`robotic_arm` 未被修改或删除。

This document is for teammates who need to understand this module's architecture and
design decisions, in parallel Chinese/English. It lives alongside `robotic_arm` (see
[`docs/robotic-arm-so101-pipeline.md`](./robotic-arm-so101-pipeline.md)) -- the two
packages have no runtime dependency on each other, and `robotic_arm` is unmodified.

---

## 0. 总览 · Overview

**中文**：`robotic_arm` 让模型只报"做什么"（`report_plan`），拍照、关键点检测、二级
VLM 生成约束代码、坐标变换、IK 求解、机械臂驱动全部由三个 rail 在幕后自动完成，模型永
远不接触任何坐标或角度。`robotic_arm2` 把这条流水线的每一步拆成一个显式的、模型自己调
用的工具：模型自己拍照、自己看图分割物体、自己把像素反投影成 3D 坐标、自己检查可达
性、自己选择接近角度、自己发出移动/夹爪指令。没有隐藏的自动执行环节，也没有二级约束生
成 VLM——空间推理完全交给主模型（必须具备视觉能力）。

**English**: `robotic_arm` lets the model only report *what* to do (`report_plan`);
photo capture, keypoint detection, a secondary constraint-generation VLM, coordinate
transforms, IK solving, and the physical drive all happen automatically behind three
rails -- the model never touches a coordinate or an angle. `robotic_arm2` breaks every
step of that pipeline into an explicit tool the model calls itself: it captures its own
photo, segments objects by looking at the image, backprojects whichever pixel it wants
to 3D, checks reachability, picks an approach angle, and issues the move/gripper command
itself. There is no hidden auto-execution step and no secondary constraint-generation
VLM -- all spatial reasoning lives in the main (vision-capable) model.

---

## 1. 架构地图 · Architecture Map

### 编排层 · Orchestration

| 文件 File | 职责 Role |
|---|---|
| [`config.py`](../openjiuwen/harness/tools/robotic_arm2/config.py) | `ArmBackend` 协议（每个能力一个方法）+ `RoboticArm2RuntimeSettings`。<br>The `ArmBackend` protocol (one method per capability) + runtime settings. |
| [`tools/`](../openjiuwen/harness/tools/robotic_arm2/tools/) | 8 个模型可调用工具的 `Tool`/`ToolCard` 定义。<br>The 8 LLM-callable `Tool`/`ToolCard` definitions. |
| [`arm_task_prompt.py`](../openjiuwen/harness/tools/robotic_arm2/arm_task_prompt.py) | 系统提示词：解释工具用法顺序、如何选接近角度、动作后要重新感知。<br>System prompt: tool usage order, how to pick an approach angle, re-perceive after acting. |
| [`rails_factory.py`](../openjiuwen/harness/tools/robotic_arm2/rails_factory.py) | 把 `backend_model` 解析成实例，装配两个 rail。<br>Resolves `backend_model` → instance, assembles the two rails. |
| [`registry.py`](../openjiuwen/harness/tools/robotic_arm2/registry.py) | `ArmBackendRegistry`：名称到厂商实现类的映射。<br>Name → vendor-class registry. |
| [`subagents/robotic_arm2_agent.py`](../openjiuwen/harness/subagents/robotic_arm2_agent.py) | 入口 `create_robotic_arm2_agent()`：组装成一个 `DeepAgent`。<br>Entry point wiring everything into a `DeepAgent`. |

### 工具层 · Tools（模型的完整能力面 · the model's entire capability surface）

| 工具 Tool | 对应 `ArmBackend` 方法 | 作用 Role |
|---|---|---|
| `capture_photo` | `capture()` | 拍一张新 RGB-D 照片。<br>Capture a fresh RGB-D photo. |
| `segment_scene` | `segment()` | 分割出候选物体的 2D 像素坐标 + 编号叠加图。<br>Candidate objects' 2D pixels + a numbered overlay. |
| `pixel_to_3d` | `backproject()` | 任意像素 → 机械臂基座系 3D 坐标（米）。<br>Any pixel → a 3D point in the robot base frame (metres). |
| `get_object_geometry` | `get_object_geometry()` | 对 `segment_scene` 某个物体 mask 内的深度点云做 PCA，返回包围盒尺寸 + 主轴方向 + 形状提示。<br>PCA over one `segment_scene` object mask's depth points: bounding dimensions + principal axis + a shape hint. |
| `check_reachability` | `check_reachability()` | 给定 3D 点，测试若干接近角度的 IK 可行性。<br>Test IK feasibility at candidate elevations for a point. |
| `move_to` | `move_to()` | 单次 IK 求解 + 真正移动机械臂。<br>Single-shot IK solve + physical move. |
| `set_gripper` | `set_gripper()` | 独立开合夹爪。<br>Open/close the gripper independently. |
| `get_gripper_pose` | `get_gripper_pose()` | 读取当前末端位置 + 夹爪状态。<br>Read current EE position + gripper state. |

### Rails · 行为拦截层（只剩两个，而且都不做决策 · only two, neither makes decisions）

| 文件 File | 职责 Role |
|---|---|
| [`rails/tool_image_rail.py`](../openjiuwen/harness/tools/robotic_arm2/rails/tool_image_rail.py) | 把 `capture_photo`/`segment_scene` 产出的图片，从工具结果搬到下一条模型消息里。<br>Carries images from a tool result into the model's next message. |
| [`rails/context_summarizer_rail.py`](../openjiuwen/harness/tools/robotic_arm2/rails/context_summarizer_rail.py) | 超过 `mcs_screenshots_to_keep` 张的旧照片换成文字占位符（与 `robotic_arm` 逻辑相同）。<br>Replaces old in-context photos past a keep-count with a text placeholder (same logic as `robotic_arm`). |

`robotic_arm` 的 `StepExecutorRail`（隐藏的"移动工具"）在这里没有对应物——因为没有隐藏
流水线需要被自动触发；`VisionPerceptionRail` 的"每轮自动拍照"逻辑变成了显式的
`capture_photo` 工具，只有图片注入这一半机制被保留为 rail（原因见 §2）。
`StepExecutorRail` (the hidden "movement tool") has no counterpart here -- there is no
hidden pipeline left to auto-trigger; `VisionPerceptionRail`'s "auto-capture every turn"
became the explicit `capture_photo` tool, only the image-injection half of the mechanism
survives as a rail (see §2 for why).

### 执行层 · Execution（`vendors/so101/`，SO-101 参考实现 · the SO-101 reference implementation）

| 文件 File | 职责 Role |
|---|---|
| [`vendors/so101/kinematics.py`](../openjiuwen/harness/tools/robotic_arm2/vendors/so101/kinematics.py) | 从 `robotic_arm/vendors/so101/_kinematics.py` **原样迁移**（数学不变）：像素↔3D、`IKSolver`、可达性检查。<br>Ported **unchanged** from `robotic_arm`: pixel↔3D math, `IKSolver`, reachability checks. |
| [`vendors/so101/perception.py`](../openjiuwen/harness/tools/robotic_arm2/vendors/so101/perception.py) | `So101Camera`（RealSense 采集，原样迁移）+ `So101Segmenter`（**仅 MobileSAM**，去掉了 DINOv2，见 §4）。<br>`So101Camera` (RealSense capture, ported unchanged) + `So101Segmenter` (**MobileSAM only**, DINOv2 dropped, see §4). |
| [`vendors/so101/backend.py`](../openjiuwen/harness/tools/robotic_arm2/vendors/so101/backend.py) | `So101ArmBackend`（注册名 `"so101"`）：`ArmBackend` 协议的唯一实现，帧缓存 + IK + 硬件 I/O。`segment()` 时顺带缓存每个物体的 mask，`get_object_geometry()` 对 mask 内已缓存的深度点云做 PCA。<br>`So101ArmBackend`: the one `ArmBackend` implementation, frame cache + IK + hardware I/O. `segment()` also caches each object's mask; `get_object_geometry()` runs PCA over that mask's already-cached depth points. |

---

## 2. 逐轮数据流 · Turn-by-Turn Data Flow

一个典型任务（"把胶带卷放到书上"）的工具调用序列：
A typical task's ("place the tape roll on the book") tool-call sequence:

```
capture_photo()                                    -> frame_id=f1 (照片注入下一条消息 / photo injected)
segment_scene(frame_id="f1")                        -> 编号叠加图 + 候选像素列表 / overlay + candidate pixels
pixel_to_3d(frame_id="f1", pixel_x=412, pixel_y=233) -> point_m=[0.182,-0.045,0.061]  (胶带卷 / tape roll)
get_object_geometry(frame_id="f1", object_id=0)      -> dimensions_m=[0.03,0.03,0.018], shape_hint="compact"
                                                        (模型据此判断该怎么接近 / informs the approach choice)
check_reachability(point_m=..., elevation_degs=[90,60,45,30]) -> 每个角度的 IK 误差 / IK error per angle

# 模型自己判断要不要先走一步过渡点、走几步——没有固定公式，靠它自己看距离和照片决定
# The model decides for itself whether/how many intermediate moves to take -- no fixed
# formula, judged from distance and fresh photos
move_to(point_m=<模型自选的过渡点 / a point the model chose en route>, elevation_deg=45, roll_deg=90)
capture_photo()                                     -> 靠近后重新拍照确认 / re-check with a fresh photo once closer
move_to(point_m=[0.182,-0.045,0.061], elevation_deg=45, roll_deg=90, gripper="closed")
get_gripper_pose()                                  -> 记录末端位置，用于推理"抓着的物体怎么移动" / for held-object reasoning

# 放置同理：模型自己决定要不要分步，而不是直接从远处扎到放置点
# Placement: same idea, the model decides whether to break it up rather than diving straight in
move_to(point_m=<书上方 above the book>, elevation_deg=90, roll_deg=90)
move_to(point_m=<书上放置点 placement point>, elevation_deg=90, roll_deg=90)
set_gripper(state="open")
```

`move_to` 本身没有"过渡点""分几步"这些概念——它只是单次 IK 求解 + 移动。上面每一次
要不要拆成多步、拆成几步、要不要中途重新拍照，都是**模型自己在 `arm_task_prompt.py`
的"Approach Gradually"一节引导下决定的**，没有固定公式，也不是代码层面强制或自动完
成的（这与之前把 `move_to` 定为"单次 IK 求解、不内置多阶段轨迹"的决策一致，见 §3）。

`move_to` itself has no concept of "intermediate points" or "how many steps" -- it is
only ever a single IK solve + move. Whether/how to break an approach into multiple steps,
and when to re-capture a photo along the way, is entirely the **model's own judgment**,
guided (not prescribed by a fixed formula) by the "Approach Gradually" section of
`arm_task_prompt.py` -- nothing enforces or automates it at the code level (consistent
with keeping `move_to` single-shot, see §3).

**图片如何真正到达模型 · How images actually reach the model**（框架级约束，非设计选
择 · a framework-level constraint, not a design choice）：`Tool.invoke()` 的返回值最
终都会被 `ability_manager._build_tool_message_content` 套壳成纯文本
`ToolMessage`——任何工具都不能把图片直接放进模型能看见的地方。所以：

1. `CapturePhotoTool`/`SegmentSceneTool` 的 `ToolOutput.data["images"]` 携带图片
   base64，但这不会进入模型看到的文本。
2. `ToolImageRail.after_tool_call()` 读取 `ctx.inputs.tool_result`（工具的**原始**
   Python 返回值，框架不会把它字符串化），把 `data["images"]` 暂存进
   `ctx.extra["robotic_arm2_pending_images"]`。
3. `ToolImageRail.before_model_call()`（模型的下一轮）取出暂存的图片，合并进即将发
   送的用户消息（`image_url` content block），然后清空暂存区。

这与 `robotic_arm` 的 `VisionPerceptionRail` 用的是**同一套注入机制**，只是触发时机从
"每轮自动"改成了"工具调用后按需"。`ctx.extra` 之所以能跨 tool-call 和下一次
model-call 传递，是因为 `ability_manager` 在构造隔离的 per-tool-call 上下文时用的是
`extra=ctx.extra`（同一个字典引用，非拷贝）。

*(English)* `Tool.invoke()`'s return value always gets stringified into a plain-text
`ToolMessage` -- no tool can put an image where the model can see it directly. So:
`ToolOutput.data["images"]` never reaches the model-visible text; `ToolImageRail`
reads the tool's *raw* return value off `ctx.inputs.tool_result` (never stringified,
available to rails only) in `after_tool_call`, stashes the images on
`ctx.extra["robotic_arm2_pending_images"]`, and `before_model_call` (the model's next
turn) drains that list and merges it into the outgoing user message. This is the exact
same injection mechanism `robotic_arm`'s `VisionPerceptionRail` uses -- only the trigger
changed from "every turn" to "after whichever tool call produced an image". `ctx.extra`
survives the trip because `ability_manager` passes `extra=ctx.extra` (the same dict
object, not a copy) into each isolated per-tool-call context.

---

## 3. 关键设计决策 · Key Design Decisions

| 决策 Decision | 原因 Why |
|---|---|
| 去掉 `report_plan`，模型自己维护任务分解<br>Drop `report_plan`; the model tracks decomposition itself | 工具粒度已经足够细，标准 ReAct 循环（先说明推理，再调用工具）就能表达"做什么/在哪/怎么做"，不需要一个专门的计划工具。<br>Tools are already fine-grained enough that a standard ReAct loop (reason, then call) expresses what/where/how without a dedicated plan tool. |
| 去掉二级 ReKep 约束生成 VLM<br>Drop the secondary ReKep constraint-generation VLM | 抓取点、接近角度现在由主模型直接通过工具参数决定，不需要模型再写 Python 约束代码交给沙箱求解。省一次模型调用，逻辑更透明，但要求主模型本身空间推理能力足够强。<br>Grasp point/approach angle are now direct tool arguments from the main model; no more VLM-authored constraint code solved in a sandbox. One fewer model call, more transparent, but leans on the main model's own spatial reasoning. |
| `segment_scene` 每物体只给 1 个质心点，去掉 DINOv2<br>`segment_scene` gives ONE centroid per object, DINOv2 dropped | 旧版每物体给最多 5 个候选点，是因为下游 VLM 只能引用检测到的关键点。新版 `pixel_to_3d` 对任意像素开放，模型可以自己看图选点，"多候选点"的价值大幅下降；去掉 DINOv2 也砍掉了 torch/transformers 依赖和 PCA/KMeans/MeanShift 聚类。<br>The old pipeline needed up to 5 candidates per object because its VLM could only reference detected keypoints. Since `pixel_to_3d` now accepts any pixel, the model can just look and point -- "many candidates" lost most of its value; dropping DINOv2 also removes the torch/transformers dependency and PCA/KMeans/MeanShift clustering. |
| `move_to` 是单次 IK 求解，不内置多阶段轨迹<br>`move_to` is single-shot IK, no built-in multi-stage trajectory | 保持工具语义简单可预测；需要"分步靠近"这种更安全的动作时，由模型自己判断要不要拆、拆成几次 `move_to` 调用。`arm_task_prompt.py`"Approach Gradually"一节要求模型不要从远处直接一步扎到抓取/放置点，但故意不给固定的"退后几厘米"公式——具体走几步、什么时候重新拍照，都留给模型自己判断；这仅是 prompt 层面的引导，代码不做强制或校验。<br>Keeps the tool's semantics simple and predictable; a safer multi-step motion is composed by the model as separate `move_to` calls, at its own discretion. `arm_task_prompt.py`'s "Approach Gradually" section asks the model not to jump straight from far away to a grasp/release point, but deliberately gives no fixed "back off by N cm" formula -- how many steps and when to re-photograph are left entirely to the model's judgment; this is prompt-level guidance only, never enforced or validated in code. |
| 去掉"抓取后关键点跟随末端刚性平移"的自动逻辑<br>Drop automatic "held keypoint follows the EE" logic | 旧版 `_update_held_keypoints` 靠 `self._ee_at_grasp` 跨调用状态自动完成；新版没有这层自动化，模型需要自己用 `get_gripper_pose` 推理"抓着的物体现在大概在哪"，或者直接重新拍照分割。这是复杂度换灵活性的取舍，见 §5 已知限制。<br>The old `_update_held_keypoints` did this automatically via cross-call `self._ee_at_grasp` state; there is no such automation here -- the model reasons it out itself via `get_gripper_pose`, or just re-captures/re-segments. A complexity-for-flexibility tradeoff, see known limitations in §5. |
| 图片仍然靠 rail 注入，不能从工具返回值直接给模型看<br>Images still injected via a rail, never directly from a tool's return value | 框架限制，非设计选择：`ability_manager._build_tool_message_content` 总是把工具结果字符串化。见 §2。<br>A framework constraint, not a design choice: tool results are always stringified. See §2. |
| 跨工具状态（帧缓存等）存在 `ArmBackend` 实例属性上，不用 `ctx.extra`<br>Cross-tool state (frame cache, etc.) lives on the `ArmBackend` instance, not `ctx.extra` | 框架不会把 `ctx` 传进 `Tool.invoke()`（`ability_manager._execute_single_tool_call` 只传 `session`），所以工具之间的状态没法指望 `ctx.extra`；`ctx.extra` 只用于 §2 描述的"图片搬运"这一种单向传递。<br>The framework never threads `ctx` into `Tool.invoke()`; `ctx.extra` is only used for the one-way image handoff described in §2, never for inter-tool state. |
| 新增 `get_object_geometry`，暴露已缓存的点云做 PCA<br>Added `get_object_geometry`, exposing the already-cached point cloud via PCA | `capture()` 本来就会算并缓存整帧点云（`backproject_frame`），过去只在 `segment()` 内部用于工作空间过滤，模型拿不到。加这个工具不是新增感知能力，是把已经算出来的 3D 数据暴露出来，让模型选 `elevation_deg`/`roll_deg` 时有真实测量数据可用，而不是纯粹看图猜形状。返回值只给尺寸/主轴/粗略形状标签，不直接给出建议的 roll_deg——具体怎么用这些数据仍由模型自己推理，延续本文档里"给原始数据、不替模型下结论"的一贯做法。<br>`capture()` already computes and caches a full per-frame point cloud (`backproject_frame`); it was previously only used internally by `segment()` for its workspace filter, never exposed to the model. This tool doesn't add new sensing -- it surfaces 3D data already computed, so the model has a real measurement to ground `elevation_deg`/`roll_deg` choices in, instead of guessing shape from the photo alone. It deliberately returns raw dimensions/axis/a coarse label, not a suggested roll_deg -- how to use the numbers is still the model's own reasoning, consistent with this package's "give primitives, not answers" pattern elsewhere. |

---

## 4. 与 `robotic_arm` 的差异速查 · Diff Cheat-Sheet vs. `robotic_arm`

| | `robotic_arm` | `robotic_arm2` |
|---|---|---|
| 模型可调用工具数 Tools | 1（`report_plan`） | 8 |
| 模型是否碰坐标/角度 Model touches coordinates/angles | 否，完全不碰 No, never | 是，全权决定 Yes, fully in control |
| 拍照 Photo capture | rail 每轮自动 Automatic every turn (rail) | 显式工具 `capture_photo` Explicit tool |
| 关键点检测 Keypoint detection | DINOv2 + MobileSAM，每物体最多 5 点 DINOv2 + MobileSAM, up to 5/object | 仅 MobileSAM，每物体 1 个质心点 MobileSAM only, 1 centroid/object |
| 物体几何测量 Object geometry | 无（仅隐含在约束代码里） None (only implicit in constraint code) | 显式工具 `get_object_geometry`（PCA 包围盒+主轴） Explicit tool `get_object_geometry` (PCA bounding box + axis) |
| 目标点/角度决策 Target point/angle decision | 二级 ReKep VLM 写约束代码 Secondary ReKep VLM writes constraint code | 主模型直接给工具参数 Main model, direct tool args |
| 抓取后物体跟随 Held-object follow | 自动（`_ee_at_grasp` 刚体投影) Automatic (rigid projection) | 手动（`get_gripper_pose` + 模型推理) Manual (`get_gripper_pose` + reasoning) |
| Rail 数量 Rails | 3 | 2 |
| 依赖 Dependencies | + torch/transformers(DINOv2) + openai(VLM) | 不需要 transformers/openai 客户端 No transformers/openai client needed |

---

## 5. 已知限制 · Known Limitations

- **没有真正的轨迹规划**：`move_to` 是单次 IK 求解，没有碰撞检测、没有防碰撞路径。
  `arm_task_prompt.py`"Approach Gradually"一节要求模型不要从远处直接一步扎到抓取/放
  置点，但故意不给固定公式，走几步、什么时候重新拍照全靠模型自己判断——这只是提示词层
  面的引导，模型仍可能因为疏忽、上下文压缩丢失指令，或单纯判断错误而直接单步移动到目
  标点，系统本身不强制也不校验。
  **No real trajectory planning**: `move_to` is a single IK solve with no collision
  checking. `arm_task_prompt.py`'s "Approach Gradually" section asks the model not to
  jump straight from far away to a grasp/release point, but deliberately gives no fixed
  formula -- how many steps and when to re-photograph are the model's own judgment call.
  This is prompt-level guidance only -- the model can still skip it (inattention, context
  compression losing the instruction, or a plain misjudgment) and move_to a target
  directly; nothing in the code enforces or validates this.
- **抓取物体的位置不再自动跟踪**：模型需要自己用 `get_gripper_pose` 或重新
  拍照/分割来判断"抓着的物体现在在哪"，尤其当物体被夹爪遮挡、从相机视角看不见时。
  **Held-object position is not auto-tracked**: the model must reason it out via
  `get_gripper_pose` or a fresh capture/segment, especially when the gripper occludes
  the object from the camera.
- **`segment_scene` 每物体只给 1 个点**：复杂形状（带把手的杯子）的最佳抓取点可能不
  是质心；模型需要自己看叠加图判断，必要时直接对着照片上别的像素调用 `pixel_to_3d`。
  **`segment_scene` gives one point per object**: the best grasp point on a complex
  shape (a mug's handle) may not be the centroid; the model must look at the overlay
  and, if needed, call `pixel_to_3d` on a different pixel of its own choosing.
- **`get_object_geometry` 不是真正的抓取规划器**：它只是对一个 mask 内的深度点做
  PCA，得到的是"这坨点云大致的长宽高和主轴方向"，不是语义理解（不知道哪端是"杯把
  手"、哪端是"杯身"），也不做防遮挡/防碰撞的抓取点选择。`shape_hint` 现在有 5 档
  （`cylinder`/`box_elongated`/`disk`/`plate`/`compact`，见 §1 工具表），仍然只是几个
  比例阈值的组合判断，不是形状分类模型；`compact` 这一档尤其分不出球和方块（同样大小
  的两者包围盒完全一样，PCA 看不出棱角）。深度读数本身的噪声（反光/透明/边缘）也会直
  接影响 PCA 结果。它能提供的是"这个物体真实的长宽高数字"，不能替模型回答"应该抓哪
  里"。
  **`get_object_geometry` is not a real grasp planner**: it's PCA over one mask's depth
  points -- it reports the point cloud's rough bounding dimensions and principal axis,
  not semantic understanding (it doesn't know which end is a mug's "handle" vs. "body"),
  and does no occlusion-aware or collision-aware grasp-point selection. `shape_hint` now
  has 5 labels (`cylinder`/`box_elongated`/`disk`/`plate`/`compact`, see the tool table in
  §1), but is still just a combination of ratio thresholds, not a shape classifier --
  `compact` in particular can't tell a sphere from a cube (identical bounding boxes, PCA
  can't see corners). Noisy depth (reflective/transparent/edge pixels) feeds directly into
  the PCA result. It gives real dimension
  numbers; it does not answer "where should I grasp" for the model.
