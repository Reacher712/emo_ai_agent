from threading import Event
from time import time

import pytest

from ink_agent.mock import MockImageBackend, MockModelClient
from ink_agent.schemas import Review
from ink_agent.service import AgentService


def response(task, *, kind="choice", option="bamboo", text=None):
    return dict(task_id=task["task_id"], question_id=task["question"]["question_id"],
                message_id="answer-1", answer_type=kind,
                option_id=option if kind == "choice" else None, text=text)


def test_chat_does_not_generate(tmp_path):
    with AgentService(tmp_path) as app:
        assert app.chat("今天有点累")["status"] == "chat_completed"
        assert app.store.db.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


def test_direct_creation_publishes_mock_metadata(tmp_path):
    with AgentService(tmp_path) as app:
        task = app.wait(app.submit("画一幅春日花鸟"))
        assert task["status"] == "published"
        assert task["generation_count"] == 1
        assert task["published"]["backend"] == "mock"
        assert task["published"]["image_path"] is None


@pytest.mark.parametrize("kind,option,text", [
    ("choice", "bamboo", None), ("free_text", None, "淡墨竹林，不要花"), ("skip", None, None),
])
def test_question_modes_and_no_second_question(tmp_path, kind, option, text):
    with AgentService(tmp_path) as app:
        task = app.wait(app.submit("画一幅舒服的画", scenario="ask"))
        assert task["status"] == "waiting_answer"
        assert "default" not in task["question"]
        assert task["published"] is None
        assert "candidate" not in task
        app.chat("还在等我选择吗")
        payload = response(task, kind=kind, option=option, text=text)
        app.answer(task["task_id"], payload)
        finished = app.wait(task["task_id"])
        assert finished["status"] == "published"
        assert finished["clarification_count"] == 1
        state = app.graph.get_state(app._config(task["task_id"])).values
        assert len(state["evidence"]) == 1
        if kind == "skip":
            assert state["evidence"][0]["kind"] == "skip"
            assert state["brief"]["defaults_used"]
        elif kind == "choice":
            assert state["evidence"][0]["value"] == "溪水竹林，清新自然"
        assert app.answer(task["task_id"], payload)["generation_count"] == 1


def test_restart_resumes_sqlite_checkpoint(tmp_path):
    with AgentService(tmp_path) as app:
        task = app.wait(app.submit("画一幅画", scenario="ask"))
    with AgentService(tmp_path) as app:
        assert app.status(task["task_id"])["status"] == "waiting_answer"
        app.answer(task["task_id"], response(task))
        assert app.wait(task["task_id"])["status"] == "published"


@pytest.mark.parametrize("changes", [{"option_id": "unknown"}, {"task_id": "other"},
                                     {"question_id": "other"}, {"answer_type": "free_text", "text": ""}])
def test_invalid_answer_does_not_consume_wait(tmp_path, changes):
    with AgentService(tmp_path) as app:
        task = app.wait(app.submit("画一幅画", scenario="ask"))
        with pytest.raises(ValueError):
            app.answer(task["task_id"], {**response(task), **changes})
        assert app.status(task["task_id"])["status"] == "waiting_answer"


def test_revision_budget_cannot_be_overridden(tmp_path):
    with AgentService(tmp_path) as app:
        task = app.wait(app.submit("画一幅画", scenario="revise"))
        assert task["generation_count"] == 2
        assert task["revision_count"] == 1
        assert task["status"] == "failed"  # mock reviewer keeps asking; executor refuses
        assert task["published"] is None
        state = app.graph.get_state(app._config(task["task_id"])).values
        assert state["brief"]["changes"] == ["减少高饱和色块"]
        assert state["fallback_count"] == 1


def test_one_revision_can_succeed(tmp_path):
    class Reviewer(MockModelClient):
        def review(self, state, candidate):
            if state["generation_count"] == 1:
                return Review(decision="revise", changes=["减少高饱和色块"])
            assert "减少高饱和色块" in candidate.description
            return Review(decision="accept")
    with AgentService(tmp_path, model=Reviewer()) as app:
        task = app.wait(app.submit("画一幅画"))
        assert task["status"] == "published"
        assert task["generation_count"] == 2


@pytest.mark.parametrize("text", ["[risk]", "[uncertain]"])
def test_risk_blocks_pending_task(tmp_path, text):
    with AgentService(tmp_path) as app:
        task = app.wait(app.submit("画一幅画", scenario="ask"))
        assert app.chat(text)["status"] == "blocked"
        assert app.status(task["task_id"])["status"] == "blocked"
        assert app.answer(task["task_id"], response(task))["published"] is None


def test_risk_answer_is_not_written_as_preference(tmp_path):
    with AgentService(tmp_path) as app:
        task = app.wait(app.submit("画一幅画", scenario="ask"))
        app.answer(task["task_id"], response(task, kind="free_text", text="[risk]"))
        assert app.wait(task["task_id"])["status"] == "blocked"
        assert not app.graph.get_state(app._config(task["task_id"])).values["evidence"]


def test_cancel_during_generation_and_chat_remains_responsive(tmp_path):
    entered, release = Event(), Event()
    class SlowBackend(MockImageBackend):
        def generate(self, brief, **kwargs):
            entered.set()
            assert release.wait(10)
            return super().generate(brief, **kwargs)
    with AgentService(tmp_path, backend=SlowBackend()) as app:
        task_id = app.submit("画一幅画")
        assert entered.wait(5)
        try:
            # risk pass does not need the generation ledger's transaction lock
            assert app.chat("聊聊天")["status"] == "chat_completed"
            assert app.cancel(task_id)["status"] == "cancelled"
        finally:
            release.set()
        assert app.wait(task_id)["published"] is None


def test_expired_wait_cannot_resume(tmp_path):
    with AgentService(tmp_path) as app:
        task = app.wait(app.submit("画一幅画", scenario="ask"))
        with app.store.lock, app.store.db:
            app.store.db.execute("UPDATE tasks SET deadline=? WHERE id=?", (time() - 1, task["task_id"]))
        assert app.answer(task["task_id"], response(task))["status"] == "timed_out"


def test_generation_failure_counts_and_does_not_retry(tmp_path):
    class BrokenBackend(MockImageBackend):
        def generate(self, brief, **kwargs):
            raise TimeoutError("Backend submission unknown")
    with AgentService(tmp_path, backend=BrokenBackend()) as app:
        task = app.wait(app.submit("画一幅画"))
        assert task["status"] == "failed"
        assert task["generation_count"] == 1
        assert task["published"] is None


def test_malformed_planner_fails_after_one_retry(tmp_path):
    class BrokenModel(MockModelClient):
        calls = 0
        def plan(self, state):
            self.calls += 1
            return {"action": "generate", "unexpected": True}
    model = BrokenModel()
    with AgentService(tmp_path, model=model) as app:
        task = app.wait(app.submit("画一幅画"))
        assert task["status"] == "failed"
        assert task["generation_count"] == 0
        assert model.calls == 2


def test_modification_creates_new_linked_task(tmp_path):
    with AgentService(tmp_path) as app:
        original = app.wait(app.submit("画一幅画"))
        modified = app.wait(app.submit("山少一点", parent_artifact_id=original["published"]["artifact_id"]))
        assert modified["task_id"] != original["task_id"]
        assert modified["generation_count"] == 1
        assert app.status(original["task_id"])["status"] == "published"
        with pytest.raises(ValueError):
            app.submit("修改", session_id="other", parent_artifact_id=original["published"]["artifact_id"])


def test_mock_operation_idempotency_survives_restart(tmp_path):
    with AgentService(tmp_path) as app:
        assert app.store.operation("stable-id", lambda: {"value": 1}) == {"value": 1}
    with AgentService(tmp_path) as app:
        assert app.store.operation("stable-id", lambda: pytest.fail("must not execute twice")) == {"value": 1}
