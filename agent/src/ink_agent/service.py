from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import RLock
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .config import Settings
from .graph import build_graph, validate_answer
from .mock import MockImageBackend, MockModelClient
from .storage import TaskStore


class AgentService:
    """Local single-process task runner with durable checkpoints.

    Queue execution is in-process; restart recovery is explicit via recover().
    Public views never include unapproved candidates.
    """

    def __init__(self, directory="runtime", *, model=None, backend=None, settings=None):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.settings = settings or Settings()
        self.stack = ExitStack()
        self.store = TaskStore(root / "tasks.sqlite")
        saver = self.stack.enter_context(SqliteSaver.from_conn_string(str(root / "checkpoints.sqlite")))
        self.graph = build_graph(model or MockModelClient(), backend or MockImageBackend(),
                                 self.store, self.settings, saver)
        self.pool = ThreadPoolExecutor(max_workers=self.settings.workers, thread_name_prefix="ink-task")
        self.futures = {}
        self.locks = {}
        self.lock = RLock()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.pool.shutdown(wait=True)
        self.stack.close()
        self.store.close()

    def _config(self, task_id):
        return {"configurable": {"thread_id": f"art:{task_id}"},
                "recursion_limit": self.settings.recursion_limit}

    def _task_lock(self, task_id):
        with self.lock:
            return self.locks.setdefault(task_id, RLock())

    def _invoke(self, task_id, payload):
        with self._task_lock(task_id):
            if isinstance(payload, Command):
                state = self.graph.get_state(self._config(task_id)).values
                if state.get("answer") == payload.resume:
                    return self.status(task_id)
                if state.get("status") != "waiting_answer":
                    raise ValueError("Task is no longer waiting for an answer")
            self.graph.invoke(payload, self._config(task_id))
            return self.status(task_id)

    def submit(self, message, *, session_id="demo", scenario="direct", parent_artifact_id=None):
        if not message.strip():
            raise ValueError("Message cannot be empty")
        if scenario not in {"direct", "ask", "revise", "fallback", "retrieve"}:
            raise ValueError("Unknown mock scenario")
        if parent_artifact_id and not self.store.owns_artifact(session_id, parent_artifact_id):
            raise ValueError("Parent artifact must be published in the same session")
        task_id = uuid4().hex
        self.store.create(task_id, session_id, self.settings.task_timeout_seconds)
        initial = dict(session_id=session_id, task_id=task_id, mode="create", message=message,
                       message_id=uuid4().hex, scenario=scenario, status="queued", question=None,
                       answer=None, evidence=[], clarification_count=0, generation_count=0,
                       revision_count=0, fallback_count=0, candidate=None, published=None,
                       parent_artifact_id=parent_artifact_id,
                       revision_request=message if parent_artifact_id else None)
        self.futures[task_id] = self.pool.submit(self._invoke, task_id, initial)
        return task_id

    def wait(self, task_id, timeout=30):
        return self.futures[task_id].result(timeout=timeout)

    def status(self, task_id):
        values = self.graph.get_state(self._config(task_id)).values
        blocked = self.store.guard(task_id)
        return {"task_id": task_id, "status": blocked or values.get("status", "queued"),
                "question": values.get("question") if not blocked and values.get("status") == "waiting_answer" else None,
                "published": values.get("published") if not blocked else None,
                "reply": values.get("reply", ""), "error": values.get("error"),
                "clarification_count": values.get("clarification_count", 0),
                "generation_count": values.get("generation_count", 0),
                "revision_count": values.get("revision_count", 0)}

    def answer(self, task_id, payload):
        with self._task_lock(task_id):
            state = self.graph.get_state(self._config(task_id))
            if not state.values:
                raise ValueError("Task has not started")
            if self.store.guard(task_id):
                return self.status(task_id)
            # Validate before resuming so a bad answer leaves the question waiting.
            answer = validate_answer(payload, state.values)
            if state.values.get("answer") == answer.model_dump():
                return self.status(task_id)
            if state.values.get("status") != "waiting_answer" or not state.next:
                raise ValueError("Task is not waiting for an answer")
            self.futures[task_id] = self.pool.submit(self._invoke, task_id, Command(resume=answer.model_dump()))
        return {"task_id": task_id, "status": "resuming"}

    def cancel(self, task_id):
        self.store.cancel(task_id)
        return self.status(task_id)

    def recover(self, task_id):
        with self._task_lock(task_id):
            state = self.graph.get_state(self._config(task_id))
            if not state.values:
                raise ValueError("No checkpoint; resubmit as a new task")
            if not state.next or state.values.get("status") == "waiting_answer" or self.store.guard(task_id):
                return self.status(task_id)
            self.futures[task_id] = self.pool.submit(self._invoke, task_id, None)
            return {"task_id": task_id, "status": "recovering"}

    def chat(self, message, *, session_id="demo"):
        if not message.strip():
            raise ValueError("Message cannot be empty")
        result = self.graph.invoke(
            {"mode": "chat", "message": message, "session_id": session_id,
             "reply": "", "published": None, "error": ""},
            {"configurable": {"thread_id": f"chat:{session_id}"}},
        )
        return {"status": result["status"], "reply": result.get("reply", ""), "error": result.get("error")}

