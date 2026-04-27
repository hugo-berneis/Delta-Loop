from typing import TypedDict

from deltaloop.storage.models import BenchmarkTask


class AgentState(TypedDict):
    task: BenchmarkTask
    iteration: int
    reasoning_steps: list[str]
    tool_calls: list[dict]
    retrieved_context: str
    multimodal_output: str
    final_answer: str
    is_complete: bool
    error: str | None
