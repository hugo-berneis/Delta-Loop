import json

from loguru import logger

from deltaloop.agent.ollama_client import OllamaClient
from deltaloop.config import settings
from deltaloop.critic.evaluator import TraceEvaluation, compute_quality_score
from deltaloop.critic.prompts import CORRECTION_PROMPT, CORRECTION_PROMPT_STRICT
from deltaloop.storage.models import BenchmarkTask, PreferencePair, ReasoningTrace


def _parse_corrected_trace(raw: str) -> list[str] | None:
    """Parse a JSON list of strings from the critic's correction response."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list) or not data:
        return None
    if not all(isinstance(s, str) for s in data):
        return None

    return data  # type: ignore[return-value]


async def synthesize_preference_pair(
    trace: ReasoningTrace,
    evaluation: TraceEvaluation,
    task: BenchmarkTask,
    client: OllamaClient,
) -> PreferencePair | None:
    """Generate a corrected trace and build a DPO preference pair.

    Returns None if:
    - The trace was already correct
    - quality_score <= 0.4 (too noisy)
    - Critic returns invalid JSON twice
    - Corrected trace is empty
    """
    if evaluation.is_correct:
        return None

    quality_score = compute_quality_score(trace, evaluation)
    if quality_score <= 0.4:
        logger.info(
            f"synthesize_preference_pair: skipping low-quality pair for task {task.id} "
            f"(quality_score={quality_score:.3f})"
        )
        return None

    with logger.contextualize(task_id=task.id, iteration=trace.iteration):
        prompt = CORRECTION_PROMPT.format(
            failure_mode=evaluation.failure_mode,
            failure_explanation=evaluation.failure_explanation,
            ground_truth=task.ground_truth,
            task_question=task.question,
            reasoning_steps=trace.reasoning_steps,
            final_answer=trace.final_answer,
        )

        raw = await client.complete(
            settings.critic_model,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        corrected = _parse_corrected_trace(raw)

        if corrected is None:
            logger.warning(
                "synthesize_preference_pair: malformed correction JSON, retrying with strict prompt"
            )
            strict_prompt = CORRECTION_PROMPT_STRICT.format(
                task_question=task.question,
                failure_mode=evaluation.failure_mode,
                ground_truth=task.ground_truth,
            )
            raw = await client.complete(
                settings.critic_model,
                [{"role": "user", "content": strict_prompt}],
                temperature=0.3,
            )
            corrected = _parse_corrected_trace(raw)

        if corrected is None:
            logger.error(
                f"synthesize_preference_pair: critic returned invalid JSON twice for task {task.id}"
            )
            return None

        logger.info(
            f"synthesize_preference_pair: created pair quality_score={quality_score:.3f} "
            f"chosen_steps={len(corrected)}"
        )

        return PreferencePair(
            task_id=task.id,
            iteration=trace.iteration,
            chosen_trace=json.dumps(corrected),
            rejected_trace=trace.reasoning_steps,
            failure_mode=evaluation.failure_mode or "UNKNOWN",
            failure_explanation=evaluation.failure_explanation or "",
            quality_score=quality_score,
            cluster_label=-1,
        )
