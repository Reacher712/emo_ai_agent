from typing import Any, Literal, TypedDict

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


class Brief(Contract):
    request: str = Field(min_length=1)
    scene: str = Field(min_length=1)
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

