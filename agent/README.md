# 国画 Agent：LangGraph 框架

`local-chat` 使用本机 Ollama 做多轮聊天和创作纲要预览。`assess` 提供单条消息评估。原有 LangGraph `chat` 和 `create` 命令继续使用模拟服务。

## 已实现

- LangGraph `StateGraph`：入口、陪伴、规划、选择题等待、执行、评审，以及固定兜底/发布处理器。
- 一道单选题，2–4 个选项，无预选；支持自由输入、跳过和严格答案归属校验。
- `interrupt()` / `Command(resume=...)`，SQLite checkpoint；退出进程后仍可继续回答。
- 后台线程池执行创作，聊天使用独立 `chat:{session_id}` 状态，创作使用 `art:{task_id}` 状态。
- 程序控制每任务最多一道题、两次生成、一次评审修订和一次兜底。
- 无效答案保持等待；重复提交相同答案不会重复生成；mock 生成操作记录持久化。
- 候选不包含在服务的公开结果里；只返回已通过模拟评审与发布门禁的记录。
- 任务取消、总超时、新消息风险分支和最终发布核查；修改任务可关联同会话已发布作品。

## 启动

在 PowerShell 中进入仓库的 `agent` 目录并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

普通聊天不会提交创作任务：

```powershell
.\.venv\Scripts\python.exe -m ink_agent.cli chat "今天有点累"
```

明确提交创作：

```powershell
.\.venv\Scripts\python.exe -m ink_agent.cli create "画一幅安静的远山"
```

演示选择题，输出包含 `task_id` 和题目；将下面的 `<task_id>` 替换为真实值：

```powershell
.\.venv\Scripts\python.exe -m ink_agent.cli create "画一幅舒服的画" --scenario ask
.\.venv\Scripts\python.exe -m ink_agent.cli status <task_id>
.\.venv\Scripts\python.exe -m ink_agent.cli answer <task_id> --choice bamboo
```

也可对尚未回答的任务使用以下任意一种方式；不要对同一任务依次执行三个回答命令：

```powershell
.\.venv\Scripts\python.exe -m ink_agent.cli answer <task_id> --text "淡墨竹林，不要花"
.\.venv\Scripts\python.exe -m ink_agent.cli answer <task_id> --skip
.\.venv\Scripts\python.exe -m ink_agent.cli cancel <task_id>
```

`--scenario revise` 模拟评审不断要求修改：程序最多提交两次生成，然后因没有可用精选库明确失败，用于观察预算保护。`--scenario retrieve` 模拟检索未命中后转生成；`--scenario fallback` 模拟评审失败后兜底未命中。场景参数只是确定性测试开关，并非自动理解用户语言。

默认状态文件写入当前目录下的 `runtime/`；需要指定路径时将全局参数放在子命令前：

```powershell
.\.venv\Scripts\python.exe -m ink_agent.cli --db-dir runtime/demo create "画一幅画" --scenario ask
```

## 用本地 Qwen3.5-4B 预览创作纲要

安装并启动 [Ollama](https://ollama.com/download)。模型由 Ollama 从 [Qwen3.5 模型库](https://ollama.com/library/qwen3.5:4b) 下载。聊天程序通过 `127.0.0.1:11434` 连接 Ollama。从 `agent` 目录运行：

```powershell
ollama pull qwen3.5:4b
.\.venv\Scripts\python.exe -m ink_agent.cli local-chat
```

可以先聊近况，再分轮说出想要的题材和氛围。输入「开始创作」、`/create` 或「帮我画一幅竹林图」时，程序结合最近八轮与当前指令，输出六轴和创作纲要预览。普通聊天阶段只回复对话。输入 `/new` 清空对话，输入 `/quit` 退出。

### 用 DeepSeek 生成国画提示词

在启动聊天的 PowerShell 中设置 `DEEPSEEK_API_KEY`。下面的命令会提示输入密钥，输入过程隐藏，密钥只在本次 PowerShell 会话及其子进程中有效：

```powershell
$env:DEEPSEEK_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'DeepSeek API Key' -AsSecureString)).Password
.\.venv\Scripts\python.exe -m ink_agent.cli local-chat
```

明确开始创作并得到画面纲要后，程序用 DeepSeek V4.1 Flash（API 模型名 `deepseek-flash`）把画面描述和已指定的六轴改写为国画提示词。普通聊天、情绪评估和情绪依据留在本地。终端显示的 `{"prompt":"..."}` 可以直接作为 Z-Image `/generate` 的请求体，尺寸和推理步数由图像服务使用默认值。缺少密钥或 API 失败时，仍可以查看本地纲要和六轴。

需要测试单条消息时，可用 `assess`：

```powershell
.\.venv\Scripts\python.exe -m ink_agent.cli assess '我很紧张，但想看一幅明亮、有生机的画'
```

输出包含 `emotion_assessment`（空 `feelings` 表示未知）和 `support_plan`。明确要求创作时，输出额外包含 `brief` 和 `target_experience_axes`。评估使用原始消息，六轴只读取从原文摘出的创作要求。用户未指定的维度保留 `null`。服务不可用、超时或校验失败时，命令向标准错误输出原因并返回失败退出码。评估和纲要预览均为只读操作。

真实推理样例见 [ollama-assess-sample.json](examples/ollama-assess-sample.json)。

## 在前端接入后台任务

`AgentService` 应在应用生命周期内保持存活，例如由 Streamlit 资源缓存持有；不要每次刷新页面都新建线程池。CLI 为方便演示会等待任务到达完成或等待回答状态。

```python
from ink_agent.service import AgentService

with AgentService("runtime") as app:
    task_id = app.submit("画一幅舒服的画", session_id="user-session", scenario="ask")
    waiting = app.wait(task_id)  # 测试/CLI 可等待；网页应轮询 status()
    print(app.chat("我想再聊一会儿", session_id="user-session"))
    app.answer(task_id, {
        "task_id": task_id,
        "question_id": waiting["question"]["question_id"],
        "message_id": "user-message-2",
        "answer_type": "choice",
        "option_id": "bamboo",
    })
    print(app.wait(task_id))
```

## 模块位置

```text
src/ink_agent/
  config.py       V1 上限与超时配置
  schemas.py      Pydantic 契约及 LangGraph 状态
  ports.py        模型与图像后端接口
  mock.py         确定性模拟服务
  graph.py        节点、条件路由、暂停恢复和发布门禁
  storage.py      任务生命周期与 mock 操作记录
  service.py      后台线程池、任务接口、公开结果过滤
  cli.py          命令行演示
tests/
  test_workflow.py
```

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

`requirements-lock.txt` 记录本次 Windows / Python 3.14 环境安装版本，包含平台相关依赖；其他平台优先从 `pyproject.toml` 重新解析依赖。

## 当前边界与下一步

1. **`create` 和 `chat` 仍使用 mock。** 调用方选择工作流入口；`assess` 独立提供真实模型的单条消息评估和纲要预览。陪伴不具备持续对话推理或偏好跨任务继承能力。
2. **不生成图片，也不做实际视觉或内容安全检查。** `published` 仅表示模拟记录通过流程，`image_path=null`，不是可展示的真实作品。精选库为空，因此兜底会明确失败。
3. **风险接口只是测试替身。** `[risk]` 和 `[uncertain]` 标记演示阻断分支，不是可部署的风险识别系统。接真实用户前须接入独立风险检查、支持流程及实际图片检查。
4. **后台队列是本进程线程池。** 不提供跨进程 worker 锁或自动重启调度；等待题目可直接恢复，已 checkpoint 的非等待任务可显式 `recover <task_id>`。首个 checkpoint 之前进程退出需重新建任务。每个数据库目录同时只运行一个应用服务实例。
5. **幂等记录仅适用于确定性 mock。** 真实远程生成需补持久化提交状态、后端任务查询和幂等键，避免“后端已接单、本地未落库”时重复收费；不能直接将 mock 操作缓存当作生产保证。
6. 下一步实现 `ports.py` 的真实模型客户端与 SDXL 适配器，并扩展 `Candidate` 的后端/真实文件契约、内容检查和发布验证。随后接规则库、偏好持久化和 Streamlit；`.env.example` 当前仅作接口预留，程序不会自动加载。

实现参考：[LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)、[Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)。
