from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from deltaloop.config import settings
from deltaloop.critic.evaluator import FAILURE_MODES

if TYPE_CHECKING:
    from deltaloop.benchmark.runner import IterationMetrics


class MLflowTracker:
    def __init__(self) -> None:
        import mlflow  # type: ignore[import-untyped]

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("deltaloop")
        self._mlflow = mlflow
        logger.info(f"MLflowTracker: tracking_uri={settings.mlflow_tracking_uri}")

    def log_iteration(self, iteration: int, metrics: IterationMetrics) -> str:
        """Log all iteration metrics and return the MLflow run ID."""
        with self._mlflow.start_run(run_name=f"iteration_{iteration}") as run:
            self._mlflow.log_metrics({
                "task_success_rate": metrics.task_success_rate,
                "avg_score": metrics.avg_score,
                "pairs_generated": float(metrics.pairs_generated),
                "pairs_stored": float(metrics.pairs_stored),
                "pairs_filtered_low_quality": float(metrics.pairs_filtered_low_quality),
                "training_triggered": float(metrics.training_triggered),
            })

            self._mlflow.log_params({
                "agent_model": settings.agent_model,
                "critic_model": settings.critic_model,
                "preference_pair_threshold": settings.preference_pair_threshold,
                "lora_r": settings.lora_r,
                "lora_alpha": settings.lora_alpha,
                "training_epochs": settings.training_epochs,
                "failure_cluster_k": settings.failure_cluster_k,
            })

            confusion = {
                mode: metrics.failure_mode_distribution.get(mode, 0)
                for mode in FAILURE_MODES
            }
            self._mlflow.log_dict(confusion, "confusion_matrix.json")

            if metrics.adapter_path:
                try:
                    self._mlflow.log_artifact(metrics.adapter_path)
                except Exception as exc:
                    logger.warning(f"MLflowTracker: failed to log adapter artifact: {exc}")

            run_id: str = run.info.run_id
            logger.info(f"MLflowTracker: logged iteration {iteration} run_id={run_id}")
            return run_id

    def log_training_run(
        self,
        iteration: int,
        adapter_path: str,
        log_history: list[dict],
    ) -> None:
        """Log training loss curve as a nested MLflow run."""
        with self._mlflow.start_run(
            run_name=f"training_iteration_{iteration}", nested=True
        ):
            for step in log_history:
                if "loss" in step:
                    self._mlflow.log_metric(
                        "train_loss", step["loss"], step=int(step.get("step", 0))
                    )
            try:
                self._mlflow.log_artifact(adapter_path)
            except Exception as exc:
                logger.warning(f"MLflowTracker: failed to log training artifact: {exc}")
