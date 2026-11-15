"""Assess one message with the local Ollama model and preview a creation brief."""

import json
from typing import TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from pydantic import Field, ValidationError

from .schemas import (
    AssessmentPreview,
    Brief,
    Contract,
    EmotionAssessment,
    SupportPlan,
    TargetExperienceAxes,
)

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3.5:4b"
TIMEOUT_SECONDS = 180

SYSTEM_PROMPT = """分析用户的一条中文消息。按 JSON schema 回答。
1. 先判断用户是否透露了自己的当下感受。只有别人的感受、已过去或被否定的感受、绘画愿望时，emotion_assessment 必须是 null。‘其他’用于本人当前确实表达了感受但不属于前五类，例如疲倦或烦躁。紧张归为‘焦虑’。每项 evidence 必须是用户原文中的连续片段。明确自述用 user_reported，有原文依据的推断用 inferred。
2. support_plan.suggestion 用一句自然的中文回应，避免诊断或承诺疗效。如果用户没有提出创作，围绕当前话题交流。用户要求创作但没说明体验时，可以提一个待确认的方向。
3. creation 只在用户明确想要画作时填写。request_quote 原样摘录连续的创作要求，scene 用简短中文保留题材、偏好与禁忌。普通聊天用 null。想要的画面体验不能作为情绪证据。
示例‘朋友心情不好’或‘我现在已经不担心了’：{"emotion_assessment":null,"support_plan":{"suggestion":"如果你想聊聊，我在听。"},"creation":null}
示例‘请画一张竹林图’：{"emotion_assessment":null,"support_plan":{"suggestion":"如果你愿意，可以再说说希望画面有什么氛围。"},"creation":{"request_quote":"请画一张竹林图","scene":"竹林图"}}
示例‘我心里挺烦，想看一幅安静的竹林图’：{"emotion_assessment":{"feelings":[{"label":"其他","evidence":"我心里挺烦","source":"user_reported"}]},"support_plan":{"suggestion":"听起来你现在有些烦，想看看安静的画面。"},"creation":{"request_quote":"想看一幅安静的竹林图","scene":"安静的竹林图"}}"""

AXES_PROMPT = """只分析用户的画面要求，按 JSON schema 回答。综合按时间顺序提供的所有行，同一维度有冲突时使用后面更新的要求。target_quotes 收集支持评分的原文片段，每项都从同一行连续摘录，多行偏好分别放进不同的数组元素。axes 只给原话直接支持的维度评分。0 很低，1 偏低，2 中等，3 偏高，4 很高，未指定为 null。明亮度、活力度、舒缓度、温暖感、开阔感、新奇感各自判断。不要用一种体验推断其他体验。只有题材或无明确体验时，target_quotes 为 []，axes 为 {}。target_source 由程序填写。
示例两行分别为‘画面明亮一点’、‘希望有生机’：{"target_quotes":["明亮一点","有生机"],"axes":{"brightness":3,"vitality":3}}
示例‘我想要安静的竹林图’：{"target_quotes":["安静"],"axes":{"calmness":3}}
示例‘画一张竹林图’：{"target_quotes":[],"axes":{}}"""


class CreationRequest(Contract):
    request_quote: str = Field(min_length=1)
    scene: str = Field(min_length=1)


class ModelAssessment(Contract):
    emotion_assessment: EmotionAssessment | None
    support_plan: SupportPlan
    creation: CreationRequest | None


class ExperienceRating(Contract):
    target_quotes: list[str]
    axes: TargetExperienceAxes


ResponseContract = TypeVar("ResponseContract", bound=Contract)


class AssessmentError(RuntimeError):
    """An error reported by the assessment CLI."""


def _chat(payload: dict) -> str:
    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            result = json.load(response)
    except HTTPError as exc:
        with exc:
            if exc.code == 404:
                raise AssessmentError(f"本机 Ollama 未提供 {MODEL}（HTTP 404）。请检查 Ollama 的模型列表。") from exc
            raise AssessmentError(f"本地 Ollama 请求失败（HTTP {exc.code}）。") from exc
    except (TimeoutError, URLError, OSError) as exc:
        cause = exc.reason if isinstance(exc, URLError) else exc
        if isinstance(cause, TimeoutError):
            raise AssessmentError(f"等待本地 Ollama 响应超时（{TIMEOUT_SECONDS} 秒）。") from exc
        raise AssessmentError("无法连接本地 Ollama（127.0.0.1:11434）。请先启动 Ollama。") from exc
    except ValueError as exc:
        raise AssessmentError("本地 Ollama 返回了无效的 API JSON。") from exc

    if not isinstance(result, dict):
        raise AssessmentError("本地 Ollama 的 API 响应格式无效。")
    if result.get("done_reason") == "length":
        raise AssessmentError("模型达到输出长度上限，评估 JSON 未生成完整。")
    message = result.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if result.get("done") is not True or not isinstance(content, str) or not content.strip():
        raise AssessmentError("本地 Ollama 未返回完整的评估内容。")
    return content


def model_reply(messages: list[dict[str, str]], schema: dict | None = None) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "options": {"num_ctx": 4096, "num_predict": 512, "temperature": 0 if schema is not None else 0.7},
    }
    if schema is not None:
        payload["format"] = schema
    return _chat(payload)


def structured_reply(schema: type[ResponseContract], prompt: str, message: str) -> ResponseContract:
    content = model_reply([
        {"role": "system", "content": prompt},
        {"role": "user", "content": message},
    ], schema.model_json_schema())
    try:
        return schema.model_validate_json(content)
    except ValidationError as exc:
        error = exc.errors(include_input=False)[0]
        field = ".".join(map(str, error["loc"])) or "response"
        raise AssessmentError(f"评估 JSON 校验失败（{field}：{error['msg']}）。") from exc


def validate_emotion_evidence(emotion: EmotionAssessment, messages: list[str]) -> None:
    for feeling in emotion.feelings:
        if not any(feeling.evidence in message for message in messages):
            raise AssessmentError("评估依据包含原文中不存在的情绪引文。")


def rate_axes(quotes: list[str]) -> TargetExperienceAxes | None:
    if not quotes:
        return None
    rating = structured_reply(ExperienceRating, AXES_PROMPT, "\n".join(quotes))
    if any(not target or not any(target in quote for quote in quotes) for target in rating.target_quotes):
        raise AssessmentError("目标体验依据必须来自用户的创作要求。")
    scores = rating.axes.model_dump(exclude={"axis_version", "target_source"}, exclude_none=True)
    if not scores:
        return None
    if not rating.target_quotes:
        raise AssessmentError("六轴评分缺少用户明确表达的目标体验依据。")
    return TargetExperienceAxes(target_source="user_explicit", **scores)


def assess(message: str) -> AssessmentPreview:
    message = message.strip()
    if not message:
        raise ValueError("评估文本不能为空。")
    assessment = structured_reply(ModelAssessment, SYSTEM_PROMPT, message)
    emotion = assessment.emotion_assessment or EmotionAssessment(feelings=[])
    validate_emotion_evidence(emotion, [message])

    preview = AssessmentPreview(
        emotion_assessment=emotion,
        support_plan=assessment.support_plan,
    )
    creation = assessment.creation
    if creation is None:
        return preview
    if creation.request_quote not in message:
        raise AssessmentError("创作要求包含原文中不存在的引文。")
    preview.target_experience_axes = rate_axes([creation.request_quote])
    preview.brief = Brief(
        request=creation.request_quote,
        scene=creation.scene,
        target_experience_axes=preview.target_experience_axes,
    )
    return preview
