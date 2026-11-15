"""Local conversation with an explicit creation gate."""

import json
import re
import sys

from pydantic import Field

from .emotion_support import (
    AssessmentError,
    model_reply,
    rate_axes,
    structured_reply,
)
from .painting_prompt import PaintingPromptError, compile_painting_prompt
from .schemas import AssessmentPreview, Brief, Contract, EmotionAssessment, SupportPlan

CHAT_PROMPT = """你是国画情感陪伴项目的中文对话助手。结合近期对话自然回应，尊重用户纠正和不想继续的意愿。不作心理诊断，不宣称疗效。普通情绪交流先陪用户聊天。用户讨论画面偏好时可以倾听与澄清，等用户下明确指令后再进入创作。聊天阶段不要输出六轴、评分或创作纲要，不声称作品已生成。回复简短，避免每次都用问题结尾。"""

SUMMARY_PROMPT = """用户已明确启动创作。分析以下按时间从旧到新排列的用户原话，按 JSON schema 回答。
emotion_assessment 分析用户本人当下的感受，优先采用较新的表达。忽略已否定或过去的情绪以及他人的感受。当下不明就用 null。紧张归入焦虑，疲惫归入其他。每项 evidence 摘录某一轮的连续原话，明确自述用 user_reported，有依据的推断用 inferred。
support_plan.suggestion 用一句自然的回复承接用户的感受和已经开始的创作预览，不再要求用户确认开始。不作诊断或承诺疗效。
art_quotes 收集跨轮仍有效的题材、目标体验、画面偏好及禁忌。每项都原样摘自某一轮，跳过仅表示启动的命令和没有画面要求的情绪陈述。用户更正某项时只采用更正后的原话，保留没变的其他要求。禁止把用户当下的情绪变成画面偏好。没有可靠的画面要求时 art_quotes 用 []，scene 用 null。否则 scene 用一句中文汇总 art_quotes，保留限制，不追加未确认的设定。
例：先说‘我想要明亮的竹林’，再说‘改成昏暗的竹林’，最后说‘开始创作’。emotion_assessment 为 null，art_quotes 只有 ["改成昏暗的竹林"]，scene 为‘昏暗的竹林’。"""

MAX_ROUNDS = 8
UNSPECIFIED_SCENE = "画面要求待确认"
START_COMMANDS = {
    "/create", "画画", "作画", "开始画画", "开始画吧", "开始创作", "开始创作吧",
    "可以开始创作了", "生成图片", "现在出图",
}
DIRECT_REQUEST = re.compile(
    r"^(?:现在[，, ]*)?(?:(?:请)?(?:帮我|给我)|请)?"
    r"(?:画(?:一[幅张]|[幅张]|个|国画|水墨画)|(?:创作|生成)(?:一[幅张])?(?:国画|水墨画|图片|画作))"
)
START_WITH_DETAILS = re.compile(r"^(?:/create\s+|(?:开始创作|开始画画|现在出图)[：:，,]\s*)\S", re.IGNORECASE)
HESITATION = re.compile(
    r"(?:(?:先|暂时|现在)?(?:别|不要)|不用)(?:帮我)?(?:画画|画|创作|出图|生成)(?=$|[，,。！？！；;\s]|了|吧)|还没决定"
)


class ConversationSummary(Contract):
    emotion_assessment: EmotionAssessment | None
    support_plan: SupportPlan
    art_quotes: list[str]
    scene: str | None = Field(min_length=1)


def is_creation_command(message: str) -> bool:
    text = message.strip().rstrip("。！! ").casefold()
    if text in START_COMMANDS:
        return True
    if text.endswith(("?", "？")) or HESITATION.search(text):
        return False
    return bool(START_WITH_DETAILS.match(text) or DIRECT_REQUEST.match(text))


def preview_creation(user_messages: list[str]) -> AssessmentPreview:
    summary = structured_reply(ConversationSummary, SUMMARY_PROMPT, json.dumps(user_messages, ensure_ascii=False))
    quotes = list(dict.fromkeys(summary.art_quotes))
    quote_turns = {}
    for quote in quotes:
        if not quote or not any(quote in message for message in user_messages):
            raise AssessmentError("创作纲要包含对话中找不到的用户原话。")
        quote_turns[quote] = max(index for index, message in enumerate(user_messages) if quote in message)
    quotes.sort(key=quote_turns.__getitem__)
    emotion = summary.emotion_assessment or EmotionAssessment(feelings=[])
    emotion = EmotionAssessment(feelings=[
        feeling for feeling in emotion.feelings
        if any(feeling.evidence in message for message in user_messages)
        and not any(feeling.evidence in quote for quote in quotes)
    ])
    scene = summary.scene if quotes else UNSPECIFIED_SCENE
    if scene is None:
        raise AssessmentError("模型没有把画面要求整理成创作纲要。")
    axes = rate_axes(quotes)
    return AssessmentPreview(
        emotion_assessment=emotion,
        support_plan=summary.support_plan,
        target_experience_axes=axes,
        brief=Brief(
            request=user_messages[-1],
            scene=scene,
            target_experience_axes=axes,
        ),
    )


def run_chat() -> None:
    history: list[dict[str, str]] = []
    print("可以先聊聊。准备好时输入「开始创作」或 /create 查看六轴和创作纲要。/new 清空，/quit 退出。")
    while True:
        try:
            message = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if message == "/quit":
            return
        if message == "/new":
            history.clear()
            print("已清空对话。")
            continue
        if not message:
            continue

        turn = {"role": "user", "content": message}
        try:
            if is_creation_command(message):
                user_messages = [item["content"] for item in history if item["role"] == "user"]
                preview = preview_creation([*user_messages, message])
                print("创作纲要预览")
                print(preview.model_dump_json(indent=2))
                if preview.brief is not None and preview.brief.scene != UNSPECIFIED_SCENE:
                    try:
                        painting_prompt = compile_painting_prompt(preview.brief)
                    except PaintingPromptError as exc:
                        print(f"国画提示词暂未生成。{exc}")
                    else:
                        print("DeepSeek 国画提示词（Z-Image 请求 JSON）")
                        print(painting_prompt.model_dump_json(indent=2))
                else:
                    print("可以先补充想画的题材或氛围，再输入「开始创作」生成国画提示词。")
                reply = preview.support_plan.suggestion
            else:
                reply = model_reply([{"role": "system", "content": CHAT_PROMPT}, *history, turn])
                print(f"助手：{reply}")
        except (AssessmentError, ValueError) as exc:
            print(f"处理失败：{exc}", file=sys.stderr)
            continue
        history.extend([turn, {"role": "assistant", "content": reply}])
        history = history[-2 * MAX_ROUNDS:]
