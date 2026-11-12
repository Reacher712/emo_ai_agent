from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Option(Contract):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class Question(Contract):
    question_id: str
    question_type: Literal["single_choice"] = "single_choice"
    text: str = Field(min_length=1)
    options: list[Option] = Field(min_length=2, max_length=4)
    allow_free_text: Literal[True] = True
    allow_skip: Literal[True] = True

    @model_validator(mode="after")
    def unique_options(self):
        if len({o.id for o in self.options}) != len(self.options):
            raise ValueError("Option IDs must be unique")
        return self


class Answer(Contract):
    task_id: str
    question_id: str
    message_id: str = Field(min_length=1)
    answer_type: Literal["choice", "free_text", "skip"]
    option_id: str | None = None
    text: str | None = None

    @model_validator(mode="after")
    def valid_payload(self):
        if self.answer_type == "choice" and (not self.option_id or self.text is not None):
            raise ValueError("Choice requires only option_id")
        if self.answer_type == "free_text" and (not self.text or self.option_id is not None):
            raise ValueError("Free text requires only nonempty text")
        if self.answer_type == "skip" and (self.option_id is not None or self.text is not None):
            raise ValueError("Skip cannot contain a preference")
        return self


ExperienceScore = Annotated[int, Field(strict=True, ge=0, le=4)]


class TargetExperienceAxes(Contract):
    """画面希望带来的体验。0 为低，2 为中，4 为高，1 和 3 为中间档。None 表示未指定。"""

    axis_version: Literal["v1"] = "v1"
    target_source: Literal["user_explicit", "user_confirmed", "system_suggested"] | None = Field(
        default=None,
        description="目标来源：用户明确表达、用户确认的建议或系统建议。",
    )
    brightness: ExperienceScore | None = Field(
        default=None,
        description="明亮度：0 暮色、深墨；2 柔和日光；4 通透晨光、亮部清晰。",
    )
    vitality: ExperienceScore | None = Field(
        default=None,
        description="活力度：0 静水、枝叶安静；2 微风、缓流；4 新生枝叶、飞鸟、流动的溪水。",
    )
    calmness: ExperienceScore | None = Field(
        default=None,
        description="舒缓度：0 强对比、节奏鲜明；2 过渡自然；4 柔和对比、笔触平稳。",
    )
    warmth: ExperienceScore | None = Field(
        default=None,
        description="温暖感：0 清冷月色、疏离氛围；2 中性气氛；4 灯火、小屋、柔暖光线。",
    )
    openness: ExperienceScore | None = Field(
        default=None,
        description="开阔感：0 近景庭院；2 溪岸中景；4 远山、宽阔水面、大片留白。",
    )
    novelty: ExperienceScore | None = Field(
        default=None,
        description="新奇感：0 熟悉的山水意象；2 熟悉场景中有新细节；4 少见视角、富想象力的意象。",
    )


class Brief(Contract):
    request: str = Field(min_length=1)
    scene: str = Field(min_length=1)
    target_experience_axes: TargetExperienceAxes | None = None
    defaults_used: list[str] = Field(default_factory=list)
    changes: list[str] = Field(default_factory=list)


class Plan(Contract):
    action: Literal["ask", "retrieve", "generate", "stop"]
    brief: Brief | None = None
    question: Question | None = None
    reason: str = ""

    @model_validator(mode="after")
    def required_fields(self):
        if self.action == "ask" and self.question is None:
            raise ValueError("ask requires a question")
        if self.action in {"generate", "retrieve"} and self.brief is None:
            raise ValueError("Execution requires a brief")
        return self


class Candidate(Contract):
    artifact_id: str
    backend: Literal["mock"] = "mock"
    description: str
    seed: int
    # No fabricated image URL: the skeleton only produces metadata.
    image_path: str | None = None


class Review(Contract):
    decision: Literal["accept", "revise", "fallback"]
    changes: list[str] = Field(default_factory=list)
    reason: str = ""


class State(TypedDict, total=False):
    session_id: str
    task_id: str
    mode: Literal["chat", "create"]
    message: str
    message_id: str
    scenario: str
    parent_artifact_id: str | None
    revision_request: str | None
    status: str
    route: str
    reply: str
    error: str
    question: dict[str, Any] | None
    answer: dict[str, Any] | None
    evidence: list[dict[str, Any]]
    brief: dict[str, Any]
    action: str
    candidate: dict[str, Any] | None
    review: dict[str, Any]
    published: dict[str, Any] | None
    clarification_count: int
    generation_count: int
    revision_count: int
    fallback_count: int
    risk_status: str

