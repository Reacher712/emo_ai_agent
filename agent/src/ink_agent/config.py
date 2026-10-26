from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    max_clarifications: int = 1
    max_generations: int = 2
    max_revisions: int = 1
    task_timeout_seconds: float = 300
    recursion_limit: int = 30
    workers: int = 2

    def __post_init__(self):
        if self.max_clarifications != 1 or self.max_generations != 2 or self.max_revisions != 1:
            raise ValueError("V1 limits are one question, two generations, one revision")
        if self.task_timeout_seconds <= 0 or self.workers < 1:
            raise ValueError("Timeout and worker count must be positive")

