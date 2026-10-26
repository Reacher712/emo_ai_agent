"""Six business nodes plus deterministic fallback/publication handlers."""

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from .schemas import Answer, Brief, Candidate, Plan, Question, Review, State


def validate_answer(raw, state):
    answer = Answer.model_validate(raw)
    question = Question.model_validate(state["question"])
    if answer.task_id != state["task_id"] or answer.question_id != question.question_id:
        raise ValueError("Answer does not belong to this task/question")
    if answer.answer_type == "choice" and answer.option_id not in {o.id for o in question.options}:
        raise ValueError("Unknown option ID")
    return answer


def build_graph(model, backend, store, settings, checkpointer):
    def guard(state):
        return store.guard(state["task_id"]) if state["mode"] == "create" else None

    def risk(text, session_id):
        try:
            result = model.risk(text)
        except Exception:
            result = "uncertain"
        if result not in {"pass", "high", "uncertain"}:
            result = "uncertain"
        if result != "pass":
            store.block_session(session_id)
        return result

    def checked_call(schema, callback):
        for attempt in range(2):
            try:
                return schema.model_validate(callback())
            except (ValidationError, ValueError):
                if attempt:
                    raise

    def intake(state):
        result = risk(state["message"], state["session_id"])
        if result != "pass":
            return {"risk_status": result, "status": "blocked", "route": "stop",
                    "reply": "[演示] 风险检查未通过，艺术任务已停止；真实支持流程待接入。"}
        blocked = guard(state)
        return {"risk_status": result, "status": blocked or "planning",
                "route": "stop" if blocked else ("chat" if state["mode"] == "chat" else "plan")}

    def companion(state):
        try:
            return {"status": "chat_completed", "reply": model.companion(state["message"])}
        except Exception as exc:
            return {"status": "failed", "error": type(exc).__name__}

    def plan(state):
        blocked = guard(state)
        if blocked:
            return {"status": blocked, "route": "stop"}
        try:
            result = checked_call(Plan, lambda: model.plan(state))
            if result.action == "stop":
                return {"status": "failed", "error": result.reason, "route": "stop"}
            if result.action == "ask":
                if state["clarification_count"] >= settings.max_clarifications:
                    # A model cannot override the one-question cap.
                    return {"status": "failed", "error": "Planner requested a second question", "route": "stop"}
                return {"question": result.question.model_dump(), "clarification_count": 1,
                        "status": "waiting_answer", "route": "ask"}
            return {"brief": result.brief.model_dump(), "action": result.action,
                    "status": "executing", "route": "execute"}
        except Exception as exc:
            return {"status": "failed", "error": f"Planning failed: {type(exc).__name__}", "route": "stop"}

    def ask(state):
        # No external side effects before interrupt: this node restarts on resume.
        raw = interrupt({"task_id": state["task_id"], **state["question"]})
        blocked = guard(state)
        if blocked:
            return {"status": blocked, "route": "stop"}
        try:
            answer = validate_answer(raw, state)
        except (ValidationError, ValueError) as exc:
            return {"status": "failed", "error": str(exc), "route": "stop"}
        selected = next((o["text"] for o in state["question"]["options"]
                         if o["id"] == answer.option_id), "")
        value = answer.text if answer.answer_type == "free_text" else selected
        result = risk(value or "", state["session_id"])
        if result != "pass":
            return {"risk_status": result, "status": "blocked", "route": "stop"}
        evidence = list(state["evidence"])
        evidence.append({"kind": "skip" if answer.answer_type == "skip" else "preference",
                         "value": value or None, "message_id": answer.message_id,
                         "question_id": answer.question_id})
        return {"answer": answer.model_dump(), "evidence": evidence,
                "status": "planning", "risk_status": result, "route": "plan"}

    def execute(state):
        blocked = guard(state)
        if blocked:
            return {"status": blocked, "route": "stop"}
        count = state["generation_count"]
        try:
            brief = Brief.model_validate(state["brief"])
            if state["action"] == "retrieve":
                found = backend.retrieve(brief)
                if found is not None:
                    candidate = Candidate.model_validate(found)
                    return {"candidate": candidate.model_dump(), "route": "review"}
                # Exactly one retrieval-to-generation transition, no model loop.
            if count >= settings.max_generations:
                return {"route": "fallback"}
            count += 1
            operation_id = f"{state['task_id']}:generate:{count}"
            payload = store.operation(operation_id, lambda: Candidate.model_validate(
                backend.generate(brief, operation_id=operation_id, seed=42 + count - 1)
            ).model_dump())
            return {"candidate": payload, "generation_count": count,
                    "action": "generate", "status": "reviewing", "route": "review"}
        except Exception as exc:
            return {"candidate": None, "generation_count": count,
                    "error": f"Execution failed: {type(exc).__name__}", "route": "fallback"}

    def review(state):
        blocked = guard(state)
        if blocked:
            return {"status": blocked, "route": "stop"}
        try:
            candidate = Candidate.model_validate(state["candidate"])
            result = checked_call(Review, lambda: model.review(state, candidate))
            update = {"review": result.model_dump()}
            if result.decision == "accept":
                return {**update, "route": "publish"}
            if (result.decision == "revise" and any(c.strip() for c in result.changes)
                    and state["revision_count"] < settings.max_revisions
                    and state["generation_count"] < settings.max_generations):
                brief = Brief.model_validate(state["brief"])
                brief.changes.extend(result.changes)
                return {**update, "brief": brief.model_dump(), "action": "generate",
                        "revision_count": state["revision_count"] + 1, "route": "execute"}
            return {**update, "route": "fallback"}
        except Exception as exc:
            return {"error": f"Review failed: {type(exc).__name__}", "route": "fallback"}

    def fallback(state):
        blocked = guard(state)
        if blocked:
            return {"status": blocked, "route": "stop"}
        if state["fallback_count"]:
            return {"status": "failed", "route": "stop"}
        try:
            found = backend.retrieve(Brief.model_validate(state["brief"]))
            if found is not None:
                candidate = Candidate.model_validate(found)
                # Independent acceptance check, without another revision loop.
                temporary = {**state, "candidate": candidate.model_dump()}
                result = checked_call(Review, lambda: model.review(temporary, candidate))
                if result.decision == "accept":
                    return {"fallback_count": 1, "candidate": candidate.model_dump(),
                            "review": result.model_dump(), "route": "publish"}
            return {"fallback_count": 1, "status": "failed", "route": "stop",
                    "error": state.get("error") or "No acceptable curated artwork available"}
        except Exception as exc:
            return {"fallback_count": 1, "status": "failed", "route": "stop",
                    "error": f"Fallback failed: {type(exc).__name__}"}

    def publish(state):
        if state["risk_status"] != "pass" or state["review"]["decision"] != "accept":
            return {"status": "failed", "published": None, "error": "Publication gate rejected"}
        artifact = Candidate.model_validate(state["candidate"]).model_dump()
        blocked = store.publish(state["task_id"], artifact)
        if blocked:
            return {"status": blocked, "published": None}
        return {"status": "published", "published": artifact,
                "reply": "[Mock 完成] 已通过模拟评审，仅产出元数据，没有生成真实图片。"}

    graph = StateGraph(State)
    nodes = {"intake": intake, "companion": companion, "plan": plan, "ask": ask,
             "execute": execute, "review": review, "fallback": fallback, "publish": publish}
    for name, node in nodes.items():
        graph.add_node(name, node)
    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", lambda s: s["route"], {"chat": "companion", "plan": "plan", "stop": END})
    graph.add_edge("companion", END)
    for name, routes in {
        "plan": {"ask": "ask", "execute": "execute", "stop": END},
        "ask": {"plan": "plan", "stop": END},
        "execute": {"review": "review", "fallback": "fallback", "stop": END},
        "review": {"publish": "publish", "execute": "execute", "fallback": "fallback", "stop": END},
        "fallback": {"publish": "publish", "stop": END},
    }.items():
        graph.add_conditional_edges(name, lambda s: s["route"], routes)
    graph.add_edge("publish", END)
    return graph.compile(checkpointer=checkpointer)

