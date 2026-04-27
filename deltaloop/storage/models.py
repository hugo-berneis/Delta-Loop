from datetime import UTC, datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class BenchmarkTask(SQLModel, table=True):
    id: str = Field(primary_key=True)  # e.g. "mmqa_001"
    category: str  # multimodal_qa | document_comprehension | web_navigation
    question: str
    context_type: str  # image | document | html
    context_path: Optional[str] = None  # path to asset file
    ground_truth: str
    scoring_rubric: str  # exact_match | partial_match
    difficulty: str  # easy | medium | hard
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReasoningTrace(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: str = Field(foreign_key="benchmarktask.id")
    iteration: int  # which training loop iteration
    reasoning_steps: str  # JSON array of strings
    tool_calls: str  # JSON array of dicts
    final_answer: str
    is_correct: bool
    score: float  # 0.0 to 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PreferencePair(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: str = Field(foreign_key="benchmarktask.id")
    iteration: int
    chosen_trace: str  # JSON of corrected reasoning steps
    rejected_trace: str  # JSON of original failed steps
    failure_mode: str  # from FAILURE_MODES taxonomy
    failure_explanation: str  # one-sentence explanation from critic
    quality_score: float  # (1 - original_score) * critic_confidence
    cluster_label: int = Field(default=-1)  # -1 = unclustered
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
