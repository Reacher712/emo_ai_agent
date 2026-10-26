# 国画 Agent：LangGraph 框架

按 `项目修改方案-改.md` 的简化架构实现：一个创作 Agent、一个固定评审节点和独立陪伴通道。当前是 **可运行的 mock 骨架**，不需要 API Key 或 GPU。

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

在 PowerShell 中进入本目录。当前工作区已经建立 `.venv` 并安装依赖；重新搭建时运行：

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

1. **全部模型输出为 mock。** 规划尚未真正理解自然语言；创建/聊天由调用方明确选择入口，尚无真实意图识别。陪伴不具备持续对话推理或偏好跨任务继承能力。
2. **不生成图片，也不做实际视觉或内容安全检查。** `published` 仅表示模拟记录通过流程，`image_path=null`，不是可展示的真实作品。精选库为空，因此兜底会明确失败。
3. **风险接口只是测试替身。** `[risk]` 和 `[uncertain]` 标记演示阻断分支，不是可部署的风险识别系统。接真实用户前须接入独立风险检查、支持流程及实际图片检查。
4. **后台队列是本进程线程池。** 不提供跨进程 worker 锁或自动重启调度；等待题目可直接恢复，已 checkpoint 的非等待任务可显式 `recover <task_id>`。首个 checkpoint 之前进程退出需重新建任务。每个数据库目录同时只运行一个应用服务实例。
5. **幂等记录仅适用于确定性 mock。** 真实远程生成需补持久化提交状态、后端任务查询和幂等键，避免“后端已接单、本地未落库”时重复收费；不能直接将 mock 操作缓存当作生产保证。
6. 下一步实现 `ports.py` 的真实模型客户端与 SDXL 适配器，并扩展 `Candidate` 的后端/真实文件契约、内容检查和发布验证。随后接规则库、偏好持久化和 Streamlit；`.env.example` 当前仅作接口预留，程序不会自动加载。

实现参考：[LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)、[Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)。
