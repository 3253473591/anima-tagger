# Anima/Danbooru 二次元 LoRA 训练自动打标 Agent 开发补充设计

## 1. 文档定位与约束

本文是原始规范 `Anima.docx` 的实现补充，而非替代或修订。（该规范为生产资产，未随本仓库分发。）原文中已经定义的项目目标、技术栈、运行平台、前端调参项、工作流、暂停语义、输出文件命名、UI 规范、导出规则、异常规则、角色/画风 LoRA 提示词及禁用词，均为不可变需求（以下简称“原始规范”）。本文仅定义原始规范未明确的内部数据结构、状态、接口契约、边界和容错落地方式。

如本文与原始规范有冲突，以原始规范为准；实现不得通过“补充设计”改变原始规范的外部可见行为。

### 1.1 必须保持不变的外部行为

- 运行平台仅为 Windows 11 x64，Python 3.11+、PyQt6、llama.cpp-python/Gemma4-12B GGUF、OpenAI 兼容云端 API、SQLite + 本地 txt 的技术方向不变。
- 前端必须可配置：Gemma4 GGUF 模型位置、一审本地推理并发数、二审 Gemini API 并发数、Gemini Key、代理、模型版本、缩略图每行数量。
- 全局工作流固定为“一审 -> 二审 -> 人工复核 -> 导出”；一审完成并成功写入 `图片名-1.txt` 后，立即进入二审队列，两阶段流式并行。
- 一审完成判断固定为 `图片名-1.txt` 存在且非空；二审完成判断固定为 `图片名-2.txt` 存在且非空。它们只影响各自阶段。
- 暂停为优雅暂停：立即停止派发新任务，已开始的推理/API 请求完成并成功落盘后才停；不得强杀正在运行的任务。
- 最终导出固定为 `图片名.txt`；`-1.txt`、`-2.txt` 永久保留，不被导出流程覆盖或删除。
- 原始规范第 10 节的四份提示词及第 13 节禁用词应原样作为提示词资产保存和加载，不在代码中手写缩减版。

## 2. 总体架构

```text
PyQt6 UI
  |-- 配置/文件夹/操作命令
  v
Application Service (RunController)
  |-- Scanner ------> SQLite 元数据
  |-- Stage 1 Scheduler ---> LocalInferenceAdapter ---> 图片名-1.txt
  |                                      | 成功即投递
  |-- Stage 2 Scheduler ---> CloudReviewAdapter -----> 图片名-2.txt
  |-- Review Service <---- -1/-2.txt + SQLite ------> 标签编辑器
  `-- Export Service --------------------------------> 图片名.txt

HardwareMonitor (1 秒) --> UI 状态栏
Event/Log Service -------> SQLite + 本地滚动日志
```

### 2.1 分层职责

| 层 | 职责 | 不得承担的职责 |
|---|---|---|
| UI | 收集配置、展示状态、提交操作、编辑审核数据 | 直接执行模型推理、直接写阶段结果 |
| RunController | 创建/恢复运行、协调扫描器和两个调度器 | 解析模型响应细节 |
| Stage Scheduler | 并发限制、暂停闸门、投递、重试、状态迁移 | 操作 UI 控件 |
| Adapter | 调用 llama.cpp 或 OpenAI 兼容 API，返回原始结果 | 写数据库、决定任务状态 |
| Caption Validator | 格式、触发词、禁用词、结构化响应校验 | 判断图片视觉事实 |
| Repository | SQLite 事务、查询和状态持久化 | 调用外部模型 |
| File Store | 原子读写 `-1.txt`、`-2.txt`、最终 `.txt` | 修改已有有效阶段文件 |
| Review/Export Service | 合并标签、人工编辑、生成最终文本 | 改写原始一审/二审证据文件 |

## 3. 领域数据结构

以下为 Python 3.11 `dataclass`/`Enum` 的逻辑模型；字段名同时是 SQLite 列名和 JSON 键的推荐命名。

```python
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

class LoraMode(StrEnum):
    CHARACTER = "character"
    STYLE = "style"

class Stage(StrEnum):
    FIRST_REVIEW = "first_review"
    SECOND_REVIEW = "second_review"

class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    RETRY_WAIT = "retry_wait"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"

class TagSource(StrEnum):
    FIRST_REVIEW = "first_review"
    SECOND_REVIEW = "second_review"
    MANUAL = "manual"

@dataclass(frozen=True)
class RunConfig:
    folder: Path
    lora_mode: LoraMode
    trigger_word: str
    gemma4_gguf_path: Path
    first_review_concurrency: int
    second_review_concurrency: int
    gemini_api_key_ref: str  # 仅保存凭据标识，不保存明文 Key
    proxy_url: str | None
    cloud_model: str
    thumbnail_columns: int

@dataclass(frozen=True)
class ImageAsset:
    image_id: str
    folder_id: str
    absolute_path: str
    relative_path: str
    stem: str
    extension: str
    size_bytes: int
    modified_ns: int
    content_fingerprint: str

@dataclass
class StageJob:
    job_id: str
    image_id: str
    stage: Stage
    status: JobStatus
    attempt_count: int
    max_attempts: int
    next_retry_at: str | None
    started_at: str | None
    finished_at: str | None
    error_code: str | None
    error_message: str | None

@dataclass
class TagRow:
    tag_id: str
    image_id: str
    source: TagSource
    en: str
    cn: str | None
    suggestion: str | None
    enabled: bool
    position: int
    is_natural_language: bool
    manually_overridden: bool
```

### 3.1 配置边界

`RunConfig` 在点击“一键启动 / 一键恢复 / 仅启动一审 / 仅启动二审”时生成不可变快照并关联到本次运行。运行中的任务不得读取正在编辑的 UI 配置；后续改动只对下一次启动生效。这样可保证同一批任务的模型、触发词和并发策略可追溯。

Gemini Key 仅由 Windows Credential Manager（推荐）或受权限保护的本地密钥存储保存。SQLite、日志、`-1.txt`、`-2.txt`、导出文件和 UI 截图中不得出现明文 Key、`Authorization` 头或完整请求体。

### 3.2 文件命名和图像身份

对于图片 `D:\dataset\a\sample.png`：

| 文件 | 作用 | 格式 |
|---|---|---|
| `sample.png` | 原图 | 不修改 |
| `sample-1.txt` | 一审原始 caption | UTF-8，无 BOM，单行文本 |
| `sample-2.txt` | 二审结构化审核结果 | UTF-8，无 BOM，JSON 对象 |
| `sample.txt` | 人工确认后的训练 caption | UTF-8，无 BOM，单行文本 |

图片身份以规范化绝对路径加 `size_bytes + modified_ns` 的 SHA-256 指纹为主；同名图片位于不同子目录时是不同图片。扫描时只接受配置的图片扩展名（默认 `.png`、`.jpg`、`.jpeg`、`.webp`、`.bmp`），并排除 `*-1.txt`、`*-2.txt`、最终 `.txt` 等辅助文件。

若图片在任务运行期间发生指纹变化，旧任务标为 `blocked`（`IMAGE_CHANGED`），不得把旧结果写到新图片上；重新扫描后按新版本创建任务。

## 4. SQLite 持久化设计

数据库建议位于用户应用数据目录，例如 `%LOCALAPPDATA%\AnimaTagger\anima_tagger.sqlite3`，不放在用户数据集文件夹中。启用 WAL、`foreign_keys=ON`、`busy_timeout=5000`，所有状态变更使用短事务。

| 表 | 主键 | 核心字段与用途 |
|---|---|---|
| `schema_meta` | `version` | 数据库迁移版本 |
| `folders` | `folder_id` | 根目录、最近扫描时间、默认模式 |
| `images` | `image_id` | 图像路径、相对路径、指纹、扫描状态 |
| `runs` | `run_id` | 配置快照 JSON、启动方式、创建/结束时间 |
| `stage_jobs` | `job_id` | `image_id + stage` 唯一、状态、重试、错误、运行归属 |
| `review_results` | `image_id` | 二审分数、评语、最终 caption、状态、原始 JSON 摘要 |
| `tag_rows` | `tag_id` | 标签四列、来源、排序、启用状态、人工覆盖标志 |
| `events` | `event_id` | 操作审计、状态迁移、异常、文件写入记录 |
| `ui_preferences` | `key` | 分栏比例、缩略图每行数量等非密钥偏好 |

关键唯一约束：`images(folder_id, relative_path)`、`stage_jobs(image_id, stage)`、`review_results(image_id)`。写入任务状态时应带版本号或以 `WHERE status IN (...)` 条件更新，避免重复调度导致同一任务被两个 worker 领取。

数据库不是阶段完成的唯一事实来源：恢复时必须重新检查原始规范规定的 `-1.txt`/`-2.txt` 是否存在且非空；数据库用于调度、审计和 UI，阶段文件用于跨重启的完成判定。

## 5. 阶段文件与 API 契约

### 5.1 一审文件

`图片名-1.txt` 必须是通过原始规范对应模式一审提示词和本地校验的最终 caption。写入前校验：非空、单行、以当前 `{trigger_word},` 开头、字符编码有效、无禁用词/不允许的语法、无重复或冲突标签。校验失败视为一审失败，不生成文件。

### 5.2 二审文件

为落实“结构化内容”的既定要求，`图片名-2.txt` 定义为如下 JSON。`final_caption` 是二审优化后的完整 caption，`first_caption_validity` 是二审逐项判断一审标签是否有效，用于前端自动设置状态。

```json
{
  "schema_version": 1,
  "score": 8,
  "comment": "Missing visible hat; remove unsupported blush.",
  "final_caption": "{trigger_word}, 1girl, ...",
  "first_caption_validity": [
    {"tag": "1girl", "valid": true, "suggestion": null},
    {"tag": "blush", "valid": false, "suggestion": "not visible"}
  ],
  "first_review_status": "needs_revision"
}
```

规则：`score` 为 0--10 整数；`first_review_status` 只能是 `pass` 或 `needs_revision`，并须与原始规范的分数范围一致；`comment` 必须为字符串；`final_caption` 必须通过二审 caption 校验；每个一审标签最多对应一条有效性结论。解析失败、缺字段、类型错误、分数状态不一致均视为二审响应无效并按 API 失败处理。

云端 Adapter 应要求模型输出 JSON，不接受 Markdown 代码块。若供应商只返回文本，Adapter 可提取唯一 JSON 对象，但不得猜测或补造字段。

### 5.3 最终导出与来源合并

导出前，`Review Service` 从 `tag_rows` 按 `position` 收集 `enabled=true` 的行：二审标签优先、有效一审标签其次、人工新增标签随后；英文标签相同只导出一次。中文名、建议、来源和状态均不得写入最终 `.txt`。自然语言行按原始规范的画风模式规则置于标签末尾，并与标签使用 `. ` 分隔。

导出前必须再次进行 caption 校验。若校验失败，不覆盖已有 `图片名.txt`，并提示用户具体违规项。导出只影响最终文件，不改变 `-1.txt`、`-2.txt`。

## 6. 状态机与调度

### 6.1 单阶段任务状态机

```text
PENDING --领取--> RUNNING --成功且原子落盘--> SUCCEEDED
   |                  |--可重试错误--> RETRY_WAIT --到期--> PENDING
   |                  |--不可重试/耗尽重试--> FAILED
   |--发现有效阶段文件--> SKIPPED
   `--图片变更/缺少前置--> BLOCKED

PENDING/RETRY_WAIT --暂停闸门关闭--> PAUSED --恢复--> PENDING
RUNNING 不直接迁移为 PAUSED；当前工作完成并落盘后才停。
```

`SKIPPED` 表示启动时已有对应非空阶段文件，外部表现等同该阶段完成；`SUCCEEDED` 表示本次程序成功生成；二者均可满足下一阶段的前置条件。`FAILED` 和 `BLOCKED` 必须在 UI 可见，并可由用户执行明确的“重试失败项”或修复原因后重新扫描；不得自动无限重试。

### 6.2 运行级状态与操作边界

| 用户操作 | 允许状态 | 行为边界 |
|---|---|---|
| 一键启动 | 空闲、已完成、失败后 | 扫描文件夹，分别按阶段文件规则建队列，启动两阶段调度器 |
| 一键恢复 | 暂停、异常退出后 | 重扫并恢复未完成项；不重做有效阶段文件 |
| 仅启动一审 | 空闲、暂停、已完成 | 只调度一审；一审成功仍允许按原始工作流投递二审，若用户要求完全隔离需使用“二审暂停” |
| 仅启动二审 | 空闲、暂停、已完成 | 仅选择存在有效 `-1.txt` 且不存在有效 `-2.txt` 的图片 |
| 暂停一审/二审 | 对应调度器运行中 | 关闭该阶段领取闸门，不取消正在运行的请求 |
| 全局暂停 | 任一阶段运行中 | 同时关闭两个领取闸门；每阶段独立等待在途工作结束 |
| 恢复 | 阶段暂停 | 打开指定闸门，`PAUSED` 任务回到 `PENDING` |
| 人工编辑标签 | 任意非导出写入时 | 更新 `tag_rows`；设置 `manually_overridden=true`，后续自动同步不得覆盖该行 |
| 导出 | 单图或文件夹 | 只读取审核标签并写最终文件；不启动模型任务 |

同一图片同一阶段在任一时刻最多一个 `RUNNING` job；同一个文件夹最多一个活动 `RunController`。再次点击启动时，UI 应聚焦已有运行并提示，而不是创建竞争运行。

### 6.3 流式投递算法

1. 扫描结束后，为每张图独立计算一审和二审完成状态。
2. 有效 `-1.txt` 的图片直接满足二审前置条件；无有效 `-1.txt` 的图片入一审队列。
3. 一审 worker 成功写入并提交 `SUCCEEDED` 后，在同一完成回调中创建/唤醒该图片的二审 job。
4. 二审 worker 只领取“存在有效 `-1.txt` 且没有有效 `-2.txt`”的任务。
5. 所有入队、领取和完成动作通过 SQLite 条件更新去重；内存队列丢失后可由数据库重建。

并发数必须严格不超过对应前端配置。由于本地模型通常共享 GPU，同一 Gemma 模型实例由单一 `LocalInferenceService` 管理；配置大于 1 时，必须先验证 llama.cpp 后端和显存容量，无法安全并发则在 UI 告警并将实际并发降至 1，记录事件。

## 7. 文件系统操作边界

- 只递归读取用户选择的文件夹内、允许扩展名的图片；不得扫描父目录、盘符其他目录或符号链接指向的外部目录。
- 所有输出文件必须与原图同目录；构造输出路径后应验证其解析路径仍在所选根目录内。
- 写入统一采用“同目录临时文件 -> flush/fsync -> 原子 replace”流程，例如 `sample-1.txt.<uuid>.tmp` 写完后替换目标。异常时只删除本次临时文件。
- 如果目标 `-1.txt` 或 `-2.txt` 已存在且非空，除非用户在未来明确提供“强制重跑/覆盖”操作，否则不得覆盖。当前原始规范只要求跳过和补跑。
- 如果最终 `sample.txt` 已存在，导出前显示覆盖确认；确认后仍使用原子替换。阶段原始文件永不因最终导出而删除。
- 文件正在被占用、无写权限或目录不可访问时，记录 `FILE_LOCKED`/`PERMISSION_DENIED`，跳过当前任务，不影响其他图片。

## 8. 异常分类、重试与恢复

| 类别 | 错误码示例 | 是否重试 | 处理 |
|---|---|---:|---|
| 本地模型初始化 | `MODEL_NOT_FOUND`、`MODEL_LOAD_FAILED`、`GPU_OOM` | 否 | 禁止启动一审，保留未完成队列，提示修复模型路径/显存 |
| 本地单图推理 | `LOCAL_INFERENCE_FAILED`、`INVALID_FIRST_CAPTION` | 否（默认） | 标记该图一审失败，不写破损文件，不影响队列其余项 |
| 网络瞬态错误 | `NETWORK_TIMEOUT`、连接重置、5xx | 是，最多 5 次 | 指数退避加随机抖动，最终标记二审失败 |
| API 限流 | `RATE_LIMITED`/429 | 是，最多 5 次 | 优先遵从 `Retry-After`，否则退避；不占用新的并发槽 |
| API 认证/配额 | `AUTH_FAILED`、`QUOTA_EXHAUSTED` | 否 | 暂停二审新领取并提示修复；已有在途请求可完成 |
| API 内容/结构 | `INVALID_SECOND_REVIEW_JSON`、`CAPTION_VALIDATION_FAILED` | 是，最多 5 次 | 附带纠错提示重新请求；仍无效则失败 |
| 文件系统 | `FILE_LOCKED`、`PERMISSION_DENIED`、`DISK_FULL` | 否自动重试 | 任务失败/阻塞并记录路径；用户处理后可重试 |
| 图片读取 | `UNSUPPORTED_IMAGE`、`CORRUPTED_IMAGE` | 否 | 标记图片不可处理，保留原图 |
| 数据库 | `DB_BUSY` | 短暂重试 | 事务级重试；持续失败时停止派发新任务，保护状态一致性 |
| 进程异常退出 | `UNEXPECTED_SHUTDOWN` | 恢复时处理 | 启动恢复时把遗留 `RUNNING` 任务重置为 `PENDING`，并以阶段文件为准重新判定 |

二审的“五次”严格指一次二审任务最多五次外部 API 尝试，包含首次请求；不可重试错误不消耗后续次数。每次失败记录尝试序号、HTTP 状态（如有）、安全化错误摘要和下次重试时间，但不记录 Key 或原始图片内容。

## 9. 标签解析、审核与人工覆盖

### 9.1 标签标准化

解析 caption 时，以英文逗号加可选空白切分，修剪空格，保留原始 `{trigger_word}` 大小写，其余按原始规范校验小写。对显示与去重使用 Unicode NFC 和首尾空白规范化；不得擅自同义词归一化或翻译英文标签，因为那会改变训练语义。

`-1.txt` 的标签创建为 `source=first_review`。`-2.txt` 的 `final_caption` 创建为 `source=second_review`；二审的 `first_caption_validity` 只自动更新尚未人工覆盖的一审标签启用状态。人工新增为 `source=manual`，永远不被二审同步删除。

### 9.2 编辑边界

- 所有四列双击编辑并在回车/失焦时事务保存，保存失败必须恢复单元格旧值并显示错误。
- 修改英文标签、启用状态或新增行会设置人工覆盖标志；仅修改中文名、建议不会影响导出。
- 触发词标签必须保持启用且位于首位。若用户修改导致违反原始规范，允许保存草稿但禁止导出，并明确显示校验错误。
- 二审分数和评语是审核证据，UI 可展示但默认不可直接编辑；若未来增加编辑能力，应另存人工备注，不得改写 `-2.txt`。

## 10. 硬件监控与资源保护

`HardwareMonitor` 使用 `pynvml + psutil` 每秒采样一次 GPU 利用率、显存使用/总量、GPU 核心温度、系统内存使用/总量。采样失败仅显示“不可用”，不得影响打标任务。

当 GPU 温度、显存或系统内存达到未来配置的安全阈值时，控制器只停止派发新的本地一审任务，允许在途任务优雅结束，并在 UI/事件日志中标识 `RESOURCE_GUARD_PAUSED`。二审是否继续运行遵循原始的分阶段暂停语义。阈值具体数值须在前端明确配置后才能启用，不能硬编码为隐式产品规则。

## 11. 日志、隐私与可观测性

日志放在 `%LOCALAPPDATA%\AnimaTagger\logs`，按日期滚动，保留策略可配置。每条事件至少包含时间、级别、`run_id`、`image_id`、阶段、事件码、耗时与安全化消息。

必须记录：扫描数量、跳过原因、状态迁移、实际并发、重试、文件原子写入结果、模型/API 错误、导出结果、人工覆盖。不得记录 API Key、完整授权头、完整 base64 图片、用户未要求上传的文件内容。诊断包若实现，必须由用户显式导出并默认脱敏。

## 12. 验收测试清单

- 配置页能够设置并持久化原始规范列出的全部前端调参项，尤其是 Gemma4 GGUF 模型位置；密钥不以明文落入 SQLite 或日志。
- 100 张图片的一审运行中，任一图片 `-1.txt` 原子落盘后对应二审无需等待其余 99 张完成。
- 预先存在非空 `-1.txt` 时跳过一审但仍能进入二审；预先存在非空 `-2.txt` 时仅跳过二审。
- 暂停一审、暂停二审、全局暂停均不取消在途任务、不生成半截文件，恢复后无重复处理。
- API 超时、429、5xx 最多尝试五次；认证失败不盲目重试；失败单图不阻塞其他图片。
- 模拟程序崩溃后重启，遗留 `RUNNING` 任务可恢复，且已存在有效阶段文件不被重复覆盖。
- 二审 JSON 缺字段、分数非法、状态不匹配或 caption 校验失败时不生成 `-2.txt`。
- 人工禁用一审标签后，重新加载二审结果不会覆盖人工选择；人工新增标签可导出。
- 最终导出只包含启用标签，并保留 `-1.txt` 与 `-2.txt` 原样不动。
- 大批量缩略图采用虚拟滚动；默认每行 10 张；单图页初始 50/50、范围 30/70、10px 图片边距及比例记忆符合原始规范。

## 13. 建议的实现目录

```text
src/anima_tagger/
  app.py
  domain/          # 枚举、dataclass、状态机、校验规则
  application/     # RunController、ReviewService、ExportService
  infrastructure/  # SQLite、文件原子写入、llama/API adapters、日志
  workers/         # 一审/二审调度器、暂停闸门、重试策略
  ui/              # PyQt6 配置、网格、编辑页、状态栏
  prompts/         # 原始规范第 10 节提示词资产，按模式/阶段原样存放
  resources/       # 禁用词、默认扩展名等非密钥静态资源
tests/
  unit/
  integration/
  fixtures/
```

提示词资产和禁用词资产必须来自原始规范的完整原文，应用层只负责选择 `角色/画风 x 一审/二审` 模板并替换 `{trigger_word}`、`{first_caption}`；不得在调度层修改提示词文本。
