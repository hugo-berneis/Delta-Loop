import json
from dataclasses import dataclass

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from deltaloop.agent.ollama_client import OllamaClient
from deltaloop.config import settings
from deltaloop.critic.prompts import EVAL_PROMPT, EVAL_PROMPT_STRICT
from deltaloop.storage.models import BenchmarkTask, ReasoningTrace

FAILURE_MODES = {
    "WRONG_RETRIEVAL",
    "WRONG_REASONING",
    "INCOMPLETE_ANSWER",
    "HALLUCINATION",
    "TOOL_MISUSE",
}

_REQUIRED_FIELDS = {"is_correct", "score", "failure_mode", "failure_explanation"}


@dataclass
class TraceEvaluation:
    is_correct: bool
    score: float
    failure_mode: str | None
    failure_explanation: str | None
    critic_confidence: float  # abs(score - 0.5) * 2


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def parse_eval_response(raw: str) -> TraceEvaluation | None:
    """Parse a critic JSON response into a TraceEvaluation. Returns None on any error."""
    if not raw:
        return None

    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    # Validate required fields present
    if not _REQUIRED_FIELDS.issubset(data.keys()):
        logger.debug(f"parse_eval_response: missing fields {_REQUIRED_FIELDS - data.keys()}")
        return None

    score = data["score"]
    if not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 1.0):
        logger.debug(f"parse_eval_response: score out of range: {score}")
        return None

    failure_mode = data["failure_mode"]
    if failure_mode is not None and failure_mode not in FAILURE_MODES:
        logger.debug(f"parse_eval_response: invalid failure_mode '{failure_mode}'")
        return None

    score_f = float(score)
    critic_confidence = abs(score_f - 0.5) * 2.0

    return TraceEvaluation(
        is_correct=bool(data["is_correct"]),
        score=score_f,
        failure_mode=failure_mode,
        failure_explanation=data["failure_explanation"],
        critic_confidence=critic_confidence,
    )


def compute_quality_score(trace: ReasoningTrace, evaluation: TraceEvaluation) -> float:
    """quality_score = (1.0 - trace.score) * critic_confidence"""
    return (1.0 - trace.score) * evaluation.critic_confidence


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def evaluate_trace(
    trace: ReasoningTrace,
    task: BenchmarkTask,
    client: OllamaClient,
) -> TraceEvaluation | None:
    """Call the critic model to evaluate a reasoning trace. Returns None on parse failure."""
    prompt = EVAL_PROMPT.format(
        task_question=task.question,
        ground_truth=task.ground_truth,
        reasoning_steps=trace.reasoning_steps,
        final_answer=trace.final_answer,
    )

    with logger.contextualize(task_id=task.id, iteration=trace.iteration):
        raw = await client.complete(
            settings.critic_model,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        result = parse_eval_response(raw)

        if result is None:
            logger.warning("evaluate_trace: malformed JSON from critic, retrying with strict prompt")
            strict_prompt = EVAL_PROMPT_STRICT.format(
                task_question=task.question,
                ground_truth=task.ground_truth,
                reasoning_steps=trace.reasoning_steps,
                final_answer=trace.final_answer,
            )
            raw = await client.complete(
                settings.critic_model,
                [{"role": "user", "content": strict_prompt}],
                temperature=0.3,
            )
            result = parse_eval_response(raw)

        if result is None:
            logger.error(
                f"evaluate_trace: critic returned invalid JSON twice for task {task.id}, skipping"
            )
            return None

        logger.info(
            f"evaluate_trace: is_correct={result.is_correct} score={result.score:.2f} "
            f"failure_mode={result.failure_mode} confidence={result.critic_confidence:.2f}"
        )
        return result
