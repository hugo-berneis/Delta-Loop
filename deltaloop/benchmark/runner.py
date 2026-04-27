"""Main orchestration loop — ties together agent, critic, clustering, and training."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

from deltaloop.agent.graph import run_agent
from deltaloop.agent.ollama_client import OllamaClient
from deltaloop.benchmark.loader import load_tasks
from deltaloop.benchmark.scorer import score_task
from deltaloop.clustering.failure_clusterer import run_clustering
from deltaloop.config import settings
from deltaloop.critic.evaluator import TraceEvaluation, evaluate_trace
from deltaloop.critic.synthesizer import synthesize_preference_pair
from deltaloop.storage.models import BenchmarkTask, PreferencePair, ReasoningTrace

if TYPE_CHECKING:
    from deltaloop.observability.mlflow_tracker import MLflowTracker
    from deltaloop.training.adapter_manager import AdapterManager


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class RunnerState(Enum):
    IDLE = "IDLE"
    BENCHMARKING = "BENCHMARKING"
    EVALUATING = "EVALUATING"
    SYNTHESIZING = "SYNTHESIZING"
    CLUSTERING = "CLUSTERING"
    TRAINING = "TRAINING"
    SWAPPING_ADAPTER = "SWAPPING_ADAPTER"


# ---------------------------------------------------------------------------
# IterationMetrics
# ---------------------------------------------------------------------------

@dataclass
class IterationMetrics:
    iteration: int
    task_success_rate: float
    avg_score: float
    failure_mode_distribution: dict[str, int] = field(default_factory=dict)
    pairs_generated: int = 0
    pairs_stored: int = 0
    pairs_filtered_low_quality: int = 0
    training_triggered: bool = False
    adapter_path: str | None = None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    def __init__(
        self,
        repo,
        mlflow_tracker: MLflowTracker | None = None,
        adapter_manager: AdapterManager | None = None,
    ) -> None:
        self._repo = repo
        self._mlflow = mlflow_tracker
        self._adapter_manager = adapter_manager
        self._client = OllamaClient()
        self.state = RunnerState.IDLE
        self.current_iteration: int = 0
        self.current_task_id: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_iteration(
        self,
        iteration: int,
        category: str | None = None,
        task_limit: int | None = None,
    ) -> IterationMetrics:
        """Run one full benchmark iteration and return metrics."""
        self.current_iteration = iteration
        logger.info(f"Runner.run_iteration: starting iteration={iteration}")

        # ---- 1. Load tasks ----
        tasks = await load_tasks(category=category, limit=task_limit)
        task_map: dict[str, BenchmarkTask] = {t.id: t for t in tasks}

        # Persist tasks to DB (idempotent — existing rows are skipped)
        for task in tasks:
            existing = await self._repo.get_task(task.id)
            if existing is None:
                await self._repo.save_task(task)

        # ---- 2. Benchmarking: run agent on every task ----
        self.state = RunnerState.BENCHMARKING
        traces: list[ReasoningTrace] = []

        for task in tasks:
            self.current_task_id = task.id
            with logger.contextualize(iteration=iteration, task_id=task.id):
                try:
                    agent_state = await run_agent(task, iteration)
                    final_answer = agent_state.get("final_answer", "")
                    score = score_task(final_answer, task)

                    trace = ReasoningTrace(
                        task_id=task.id,
                        iteration=iteration,
                        reasoning_steps=json.dumps(agent_state.get("reasoning_steps", [])),
                        tool_calls=json.dumps(agent_state.get("tool_calls", [])),
                        final_answer=final_answer,
                        is_correct=agent_state.get("is_complete", False),
                        score=score,
                    )
                    saved_trace = await self._repo.save_trace(trace)
                    traces.append(saved_trace)
                    logger.info(f"task done score={score:.2f} is_correct={trace.is_correct}")

                except Exception as exc:
                    logger.error(f"agent failed on task {task.id}: {exc}")
                    # Record a zero-score failed trace so the critic can generate a correction
                    trace = ReasoningTrace(
                        task_id=task.id,
                        iteration=iteration,
                        reasoning_steps=json.dumps([]),
                        tool_calls=json.dumps([]),
                        final_answer="",
                        is_correct=False,
                        score=0.0,
                    )
                    saved_trace = await self._repo.save_trace(trace)
                    traces.append(saved_trace)

        # ---- 3. Evaluating: critic scores every trace ----
        self.state = RunnerState.EVALUATING
        evaluations: list[TraceEvaluation | None] = []

        for trace in traces:
            task = task_map.get(trace.task_id)
            if task is None:
                evaluations.append(None)
                continue
            with logger.contextualize(iteration=iteration, task_id=trace.task_id):
                try:
                    eval_ = await evaluate_trace(trace, task, self._client)
                    evaluations.append(eval_)
                except Exception as exc:
                    logger.error(f"critic failed on task {trace.task_id}: {exc}")
                    evaluations.append(None)

        # ---- 4. Synthesizing: generate preference pairs for failures ----
        self.state = RunnerState.SYNTHESIZING
        pairs_generated = 0
        pairs_stored = 0

        for trace, eval_ in zip(traces, evaluations):
            if eval_ is None or eval_.is_correct:
                continue
            task = task_map.get(trace.task_id)
            if task is None:
                continue
            pairs_generated += 1
            with logger.contextualize(iteration=iteration, task_id=trace.task_id):
                try:
                    pair = await synthesize_preference_pair(trace, eval_, task, self._client)
                    if pair is not None:
                        saved = await self._repo.save_pair(pair)
                        if saved is not None:
                            pairs_stored += 1
                except Exception as exc:
                    logger.error(f"synthesis failed on task {trace.task_id}: {exc}")

        pairs_filtered = pairs_generated - pairs_stored

        # ---- 5. Clustering: update cluster labels in DB ----
        self.state = RunnerState.CLUSTERING
        try:
            await run_clustering(self._repo, k=settings.failure_cluster_k)
        except Exception as exc:
            logger.error(f"clustering failed: {exc}")

        # ---- 6. Compute metrics ----
        metrics = _compute_metrics(
            iteration=iteration,
            traces=traces,
            evaluations=evaluations,
            pairs_generated=pairs_generated,
            pairs_stored=pairs_stored,
        )

        # ---- 7. Log to MLflow ----
        if self._mlflow is not None:
            try:
                self._mlflow.log_iteration(iteration, metrics)
            except Exception as exc:
                logger.warning(f"MLflow logging failed: {exc}")

        # ---- 8. Trigger fine-tuning if threshold reached ----
        all_pairs = await self._repo.get_all_pairs()
        if len(all_pairs) >= settings.preference_pair_threshold:
            logger.info(
                f"Threshold reached ({len(all_pairs)} pairs ≥ "
                f"{settings.preference_pair_threshold}). Triggering fine-tuning."
            )
            metrics.training_triggered = True
            try:
                self.state = RunnerState.TRAINING
                adapter_path = await _trigger_fine_tuning(
                    iteration, self._repo, self._mlflow
                )
                metrics.adapter_path = adapter_path

                if self._adapter_manager is not None:
                    self.state = RunnerState.SWAPPING_ADAPTER
                    self._adapter_manager.swap(adapter_path)
                    logger.info(f"Adapter swapped to {adapter_path}")

            except Exception as exc:
                logger.error(
                    f"Fine-tuning failed: {exc}. Continuing with current adapter."
                )
                metrics.training_triggered = False

        self.state = RunnerState.IDLE
        self.current_task_id = None
        logger.info(
            f"Runner.run_iteration: done iteration={iteration} "
            f"success_rate={metrics.task_success_rate:.2f} "
            f"avg_score={metrics.avg_score:.2f}"
        )
        return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_metrics(
    iteration: int,
    traces: list[ReasoningTrace],
    evaluations: list[TraceEvaluation | None],
    pairs_generated: int,
    pairs_stored: int,
) -> IterationMetrics:
    if not traces:
        return IterationMetrics(
            iteration=iteration,
            task_success_rate=0.0,
            avg_score=0.0,
            pairs_generated=pairs_generated,
            pairs_stored=pairs_stored,
            pairs_filtered_low_quality=pairs_generated - pairs_stored,
        )

    n_correct = sum(1 for t in traces if t.is_correct)
    avg_score = sum(t.score for t in traces) / len(traces)

    failure_dist: dict[str, int] = {}
    for eval_ in evaluations:
        if eval_ is not None and eval_.failure_mode:
            failure_dist[eval_.failure_mode] = failure_dist.get(eval_.failure_mode, 0) + 1

    return IterationMetrics(
        iteration=iteration,
        task_success_rate=n_correct / len(traces),
        avg_score=avg_score,
        failure_mode_distribution=failure_dist,
        pairs_generated=pairs_generated,
        pairs_stored=pairs_stored,
        pairs_filtered_low_quality=pairs_generated - pairs_stored,
    )


async def _trigger_fine_tuning(
    iteration: int,
    repo,
    mlflow_tracker: MLflowTracker | None,
) -> str:
    from deltaloop.training.data_builder import build_dpo_dataset
    from deltaloop.training.dpo_trainer import run_dpo_training

    dataset = await build_dpo_dataset(repo)
    if len(dataset) == 0:
        raise RuntimeError("No training data available after build_dpo_dataset")

    return await run_dpo_training(dataset, iteration, mlflow_tracker)


# ---------------------------------------------------------------------------
# CLI entry point: python -m deltaloop.benchmark.runner
# ---------------------------------------------------------------------------

async def _main() -> None:
    import argparse

    from deltaloop.storage.repository import Repository, get_session, init_db

    parser = argparse.ArgumentParser(description="Run DeltaLoop benchmark iteration")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=None, help="Limit task count (CI mode)")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--ci-mode", action="store_true")
    args = parser.parse_args()

    await init_db()

    mlflow_tracker = None
    try:
        from deltaloop.observability.mlflow_tracker import MLflowTracker
        mlflow_tracker = MLflowTracker()
    except Exception as exc:
        logger.warning(f"MLflow unavailable: {exc}")

    async with get_session() as session:
        repo = Repository(session)
        runner = Runner(repo, mlflow_tracker=mlflow_tracker)
        metrics = await runner.run_iteration(
            iteration=args.iteration,
            category=args.category,
            task_limit=args.tasks,
        )

    logger.info(
        f"Iteration {metrics.iteration} complete — "
        f"success_rate={metrics.task_success_rate:.3f} "
        f"avg_score={metrics.avg_score:.3f} "
        f"pairs_stored={metrics.pairs_stored}"
    )

    if args.ci_mode:
        import json as _json
        with open("ci_metrics.json", "w") as f:
            _json.dump({
                "task_success_rate": metrics.task_success_rate,
                "avg_score": metrics.avg_score,
                "pairs_generated": metrics.pairs_generated,
            }, f)


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
