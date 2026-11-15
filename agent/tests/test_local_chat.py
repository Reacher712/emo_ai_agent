import pytest

from ink_agent import conversation
from ink_agent.schemas import AssessmentPreview, EmotionAssessment, SupportPlan


@pytest.mark.parametrize(("message", "expected"), [
    ("开始创作。", True),
    ("/create 明亮的竹林", True),
    ("帮我画一幅竹林，不要画人", True),
    ("我想聊聊画画", False),
    ("帮我画一幅竹林，先别画", False),
    ("帮我生成一张表格", False),
    ("开始创作需要准备什么？", False),
])
def test_creation_gate_requires_an_explicit_art_request(message, expected):
    assert conversation.is_creation_command(message) is expected


def test_chat_waits_for_command_and_preserves_user_turns_only(monkeypatch, capsys):
    inputs = iter(["我最近很累", "想要明亮一点", "开始创作", "/new", "/create", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    calls = []

    def reply(history):
        calls.append(("chat", history[-1]["content"]))
        return "我们可以再聊聊。"

    def preview(messages):
        calls.append(("preview", messages))
        return AssessmentPreview(
            emotion_assessment=EmotionAssessment(feelings=[]),
            support_plan=SupportPlan(suggestion="收到。"),
        )

    monkeypatch.setattr(conversation, "model_reply", reply)
    monkeypatch.setattr(conversation, "preview_creation", preview)
    conversation.run_chat()
    assert calls == [
        ("chat", "我最近很累"),
        ("chat", "想要明亮一点"),
        ("preview", ["我最近很累", "想要明亮一点", "开始创作"]),
        ("preview", ["/create"]),
    ]
    assert capsys.readouterr().out.count("创作纲要预览") == 2
