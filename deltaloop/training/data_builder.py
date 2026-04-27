import json

import numpy as np
from datasets import Dataset
from loguru import logger

from deltaloop.clustering.failure_clusterer import sample_balanced
from deltaloop.config import settings
from deltaloop.storage.models import BenchmarkTask, PreferencePair


def format_pair(pair: PreferencePair, task: BenchmarkTask) -> dict[str, str]:
    """Format a PreferencePair into the prompt/chosen/rejected dict required by TRL DPO trainer."""
    chosen_steps: list[str] = json.loads(pair.chosen_trace)
    rejected_steps: list[str] = json.loads(pair.rejected_trace)

    if not chosen_steps:
        raise ValueError(f"chosen_trace is empty for pair task_id={pair.task_id}")
    if not rejected_steps:
        raise ValueError(f"rejected_trace is empty for pair task_id={pair.task_id}")

    prompt = f"Task: {task.question}\nContext type: {task.context_type}"
    chosen = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(chosen_steps))
    rejected = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(rejected_steps))

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


async def build_dpo_dataset(
    repo,
    n: int | None = None,
) -> Dataset:
    """Fetch all pairs from DB, apply cluster-balanced sampling, return HF Dataset.

    Args:
        repo: Repository instance.
        n: Number of pairs to sample. Defaults to all available pairs.
    """
    pairs = await repo.get_all_pairs()
    if not pairs:
        logger.warning("build_dpo_dataset: no preference pairs found in DB")
        return Dataset.from_list([])

    # Cluster-balanced sampling
    labels = np.array([p.cluster_label for p in pairs])
    n_sample = n if n is not None else len(pairs)
    balanced = sample_balanced(pairs, labels, n=n_sample,
                               max_cluster_fraction=0.30)

    logger.info(
        f"build_dpo_dataset: {len(pairs)} total pairs → "
        f"{len(balanced)} after balanced sampling (n={n_sample})"
    )

    # Format each pair — fetch task for prompt construction
    records: list[dict[str, str]] = []
    for pair in balanced:
        task = await repo.get_task(pair.task_id)
        if task is None:
            logger.warning(f"build_dpo_dataset: task {pair.task_id} not found, skipping pair")
            continue
        try:
            records.append(format_pair(pair, task))
        except ValueError as exc:
            logger.warning(f"build_dpo_dataset: skipping malformed pair: {exc}")

    logger.info(f"build_dpo_dataset: built dataset with {len(records)} records")
    return Dataset.from_list(records)
