"""Deterministic fixtures, NOT a psychological risk detector or visual model."""

from hashlib import sha256

from .schemas import Brief, Candidate, Option, Plan, Question, Review


class MockModelClient:
    def risk(self, text):
        # Explicit test markers only. Do not use this adapter for real users.
        if "[risk]" in text:
            return "high"
        if "[uncertain]" in text:
            return "uncertain"
        return "pass"

    def companion(self, text):
        return f"[Mock 陪伴回复] 收到：{text}。此消息不会触发出图。"

    def plan(self, state):
        if state["scenario"] == "ask" and not state["clarification_count"]:
            return Plan(action="ask", question=Question(
                question_id=f"{state['task_id']}:scene",
                text="这幅画你更喜欢哪种景象？",
                options=[Option(id="mountain", text="云雾远山，安静疏朗"),
                         Option(id="bamboo", text="溪水竹林，清新自然"),
                         Option(id="spring", text="春日花鸟，明亮有生机")],
            ))
        answer = state.get("answer")
        chosen = next((e["value"] for e in state["evidence"] if e["kind"] == "preference"), None)
        scene = chosen or "淡墨远山、疏朗构图、较多留白"
        defaults = [] if chosen else [scene]
        return Plan(
            action="retrieve" if state["scenario"] == "retrieve" else "generate",
            brief=Brief(request=state["message"], scene=scene, defaults_used=defaults),
            reason="mock default" if answer is None else "mock answer applied",
        )

    def review(self, state, candidate):
        if state["scenario"] == "fallback":
            return Review(decision="fallback", reason="模拟评审不通过")
        if state["scenario"] == "revise":
            return Review(decision="revise", changes=["减少高饱和色块"], reason="模拟修订")
        return Review(decision="accept", reason="Mock 验收通过，不代表实际视觉质量")


class MockImageBackend:
    def generate(self, brief, *, operation_id, seed):
        return Candidate(
            artifact_id=f"mock-{sha256(operation_id.encode()).hexdigest()[:16]}",
            description=f"[模拟作品，无真实图片] {brief.scene}；调整：{'、'.join(brief.changes) or '无'}",
            seed=seed,
        )

    def retrieve(self, brief):
        # No real curated collection has been supplied yet.
        return None

