from deltaloop.storage.models import BenchmarkTask, ReasoningTrace


def score_exact(answer: str, ground_truth: str) -> float:
    return 1.0 if answer.strip().lower() == ground_truth.strip().lower() else 0.0


def score_partial(answer: str, ground_truth: str) -> float:
    """Jaccard similarity on word token sets."""
    answer_tokens = set(answer.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    if not truth_tokens:
        return 0.0
    return len(answer_tokens & truth_tokens) / len(answer_tokens | truth_tokens)


def score_task(answer: str, task: BenchmarkTask) -> float:
    """Apply the rubric-appropriate scorer. Critic score overrides this when available."""
    if task.scoring_rubric == "exact_match":
        return score_exact(answer, task.ground_truth)
    return score_partial(answer, task.ground_truth)
