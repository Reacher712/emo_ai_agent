import io
import json

import pytest

from ink_agent import painting_prompt
from ink_agent.schemas import Brief, TargetExperienceAxes


def test_deepseek_receives_only_the_art_brief_and_returns_zimage_request(monkeypatch):
    class FakeApi:
        def open(self, request, *, timeout):
            assert request.full_url == 'https://api.deepseek.com/chat/completions'
            assert request.get_header('Authorization') == 'Bearer test-key'
            payload = json.loads(request.data)
            assert payload['model'] == 'deepseek-flash'
            assert payload['response_format'] == {'type': 'json_object'}
            assert json.loads(payload['messages'][-1]['content']) == {
                'scene': '明亮竹林，无题字',
                'target_experience_axes': {'brightness': 3},
            }
            return io.StringIO(json.dumps({'choices': [{
                'finish_reason': 'stop',
                'message': {'content': '{"prompt":"清晨竹林，以淡墨写新叶，无题字。"}'},
            }]}))

    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    monkeypatch.setattr(painting_prompt, 'build_opener', lambda *handlers: FakeApi())
    brief = Brief(request='这段原话有个人近况', scene='明亮竹林，无题字',
                  target_experience_axes=TargetExperienceAxes(brightness=3, target_source='user_explicit'))
    assert painting_prompt.compile_painting_prompt(brief).model_dump() == {
        'prompt': '清晨竹林，以淡墨写新叶，无题字。',
    }


def test_blank_deepseek_response_is_reported(monkeypatch):
    class EmptyApi:
        def open(self, request, *, timeout):
            return io.StringIO('{"choices":[{"finish_reason":"stop","message":{"content":"   "}}]}')

    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    monkeypatch.setattr(painting_prompt, 'build_opener', lambda *handlers: EmptyApi())
    with pytest.raises(painting_prompt.PaintingPromptError, match='提示词为空或格式无效'):
        painting_prompt.compile_painting_prompt(Brief(request='开始创作', scene='竹林'))
