import json
import sys
from urllib.error import URLError

import pytest

from ink_agent import cli, emotion_support
from ink_agent.schemas import Brief


def run_assess(monkeypatch, message):
    monkeypatch.setattr(sys, "argv", ["ink-agent", "assess", message])
    cli.main()


def test_cli_previews_shared_axes_without_creating_tasks(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)

    def chat(payload):
        assert payload["model"] == "qwen3.5:4b"
        assert payload["think"] is False
        assert payload["format"]["type"] == "object"
        if payload["format"]["title"] == "ExperienceRating":
            assert payload["messages"][-1]["content"] == "想看一幅明亮、有生机的画"
            return json.dumps({
                "target_quotes": ["明亮、有生机"],
                "axes": {"brightness": 4, "vitality": 4},
            })
        return json.dumps({
            "emotion_assessment": {"feelings": [
                {"label": "焦虑", "evidence": "我很紧张", "source": "user_reported"},
            ]},
            "support_plan": {"suggestion": "如果你愿意，我们可以从明亮、有生机的画面开始。"},
            "creation": {
                "request_quote": "想看一幅明亮、有生机的画",
                "scene": "明亮、有生机的画面",
            },
        })

    monkeypatch.setattr(emotion_support, "_chat", chat)
    run_assess(monkeypatch, "我很紧张，但想看一幅明亮、有生机的画")
    result = json.loads(capsys.readouterr().out)
    assert Brief.model_validate(result["brief"]).target_experience_axes.model_dump() == result["target_experience_axes"]
    assert result["target_experience_axes"]["target_source"] == "user_explicit"
    assert result["target_experience_axes"]["calmness"] is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("failure", "expected"), [
    (ConnectionRefusedError(), "无法连接本地 Ollama"),
    (TimeoutError(), "等待本地 Ollama 响应超时"),
])
def test_cli_reports_failed_connection(monkeypatch, capsys, failure, expected):
    class OfflineOllama:
        def open(self, request, *, timeout):
            raise URLError(failure)

    monkeypatch.setattr(emotion_support, "build_opener", lambda *args: OfflineOllama())
    with pytest.raises(SystemExit) as exc:
        run_assess(monkeypatch, "今天好累")
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert captured.out == ""
    assert expected in captured.err


def test_cli_rejects_out_of_range_axes(monkeypatch, capsys):
    replies = iter([
        {"emotion_assessment": None, "support_plan": {"suggestion": "可以从明亮的颜色开始。"},
         "creation": {"request_quote": "想看明亮的画", "scene": "明亮的画"}},
        {"target_quotes": ["明亮"], "axes": {"brightness": 5}},
    ])
    monkeypatch.setattr(emotion_support, "_chat", lambda payload: json.dumps(next(replies)))
    with pytest.raises(SystemExit) as exc:
        run_assess(monkeypatch, "想看明亮的画")
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert captured.out == ""
    assert "评估 JSON 校验失败" in captured.err
    assert "axes.brightness" in captured.err
