from __future__ import annotations

from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from deltaloop.api.main import app
from deltaloop.benchmark.runner import Runner, RunnerState
from deltaloop.storage.models import BenchmarkTask, PreferencePair, ReasoningTrace
from deltaloop.storage.repository import Repository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trace(task_id="t1", iteration=1, is_correct=True, score=0.9):
    return ReasoningTrace(
        task_id=task_id,
        iteration=iteration,
        reasoning_steps="step",
        final_answer="ans",
        is_correct=is_correct,
        score=score,
        created_at=datetime.now(UTC),
    )


def _pair(task_id="t1", iteration=1, cluster_label=0):
    return PreferencePair(
        task_id=task_id,
        iteration=iteration,
        failure_mode="WRONG_REASONING",
        failure_explanation="bad logic",
        quality_score=0.8,
        cluster_label=cluster_label,
        chosen_trace='["step"]',
        rejected_trace='["bad"]',
        created_at=datetime.now(UTC),
    )


def _mock_repo(
    iteration_numbers=None,
    traces=None,
    pairs=None,
):
    repo = AsyncMock(spec=Repository)
    repo.get_iteration_numbers.return_value = iteration_numbers or []
    repo.get_traces_for_iteration.return_value = traces or []
    repo.get_pairs_for_iteration.return_value = pairs or []
    return repo


def _mock_runner(state=RunnerState.IDLE, iteration=0, task_id=None, adapter=None):
    runner = MagicMock(spec=Runner)
    runner.state = state
    runner.current_iteration = iteration
    runner.current_task_id = task_id
    runner._adapter_manager = MagicMock()
    runner._adapter_manager.current_adapter = adapter
    return runner


# ---------------------------------------------------------------------------
# GET /api/iterations
# ---------------------------------------------------------------------------


def test_list_iterations_empty():
    repo = _mock_repo()
    with patch("deltaloop.api.deps.get_repo", return_value=repo), \
         patch("deltaloop.api.deps.get_runner", return_value=_mock_runner()):
        client = TestClient(app)
        with patch("deltaloop.api.routes.benchmark.get_repo", lambda: repo):
            resp = client.get("/api/iterations")
    # FastAPI dependency override is cleaner
    app.dependency_overrides = {}


def test_list_iterations_returns_summary():
    repo = _mock_repo(
        iteration_numbers=[1],
        traces=[_trace(is_correct=True, score=0.9), _trace(is_correct=False, score=0.4)],
        pairs=[_pair()],
    )
    runner = _mock_runner()

    from deltaloop.api.deps import get_repo, get_runner
    app.dependency_overrides[get_repo] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: runner

    client = TestClient(app)
    resp = client.get("/api/iterations")
    app.dependency_overrides = {}

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["iteration"] == 1
    assert data[0]["task_success_rate"] == pytest.approx(0.5)
    assert data[0]["avg_score"] == pytest.approx(0.65)
    assert data[0]["pairs_stored"] == 1


# ---------------------------------------------------------------------------
# GET /api/iterations/{n}/traces
# ---------------------------------------------------------------------------


def test_get_traces_pagination():
    traces = [_trace(task_id=f"t{i}", score=0.8) for i in range(25)]
    repo = _mock_repo(traces=traces)

    from deltaloop.api.deps import get_repo, get_runner
    app.dependency_overrides[get_repo] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: _mock_runner()

    client = TestClient(app)
    resp = client.get("/api/iterations/1/traces?page=2&page_size=10")
    app.dependency_overrides = {}

    assert resp.status_code == 200
    body = resp.json()
    assert body["page"] == 2
    assert body["total"] == 25
    assert len(body["items"]) == 10


def test_get_traces_filter_failed():
    traces = [_trace(is_correct=True), _trace(task_id="t2", is_correct=False, score=0.2)]
    repo = _mock_repo(traces=traces)

    from deltaloop.api.deps import get_repo, get_runner
    app.dependency_overrides[get_repo] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: _mock_runner()

    client = TestClient(app)
    resp = client.get("/api/iterations/1/traces?filter=failed")
    app.dependency_overrides = {}

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["task_id"] == "t2"


# ---------------------------------------------------------------------------
# POST /api/run
# ---------------------------------------------------------------------------


def test_trigger_run_starts_when_idle():
    runner = _mock_runner(state=RunnerState.IDLE, iteration=2)
    runner.run_iteration = AsyncMock()
    repo = _mock_repo()

    from deltaloop.api.deps import get_repo, get_runner
    app.dependency_overrides[get_repo] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: runner

    client = TestClient(app)
    resp = client.post("/api/run")
    app.dependency_overrides = {}

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "started"
    assert body["run_id"] == "iter_003"


def test_trigger_run_returns_409_when_busy():
    runner = _mock_runner(state=RunnerState.BENCHMARKING)
    runner.run_iteration = AsyncMock()
    repo = _mock_repo()

    from deltaloop.api.deps import get_repo, get_runner
    app.dependency_overrides[get_repo] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: runner

    client = TestClient(app)
    resp = client.post("/api/run")
    app.dependency_overrides = {}

    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# GET /api/status
# ---------------------------------------------------------------------------


def test_get_status():
    runner = _mock_runner(state=RunnerState.TRAINING, iteration=3, task_id="task_42", adapter="adapters/iteration_2")

    from deltaloop.api.deps import get_runner
    app.dependency_overrides[get_runner] = lambda: runner

    client = TestClient(app)
    resp = client.get("/api/status")
    app.dependency_overrides = {}

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "TRAINING"
    assert body["current_iteration"] == 3
    assert body["current_task"] == "task_42"
    assert body["adapter"] == "adapters/iteration_2"


# ---------------------------------------------------------------------------
# GET /api/learning-curve
# ---------------------------------------------------------------------------


def test_learning_curve_single_iteration():
    repo = _mock_repo(
        iteration_numbers=[1],
        traces=[_trace(is_correct=True, score=1.0), _trace(task_id="t2", is_correct=True, score=0.8)],
    )

    from deltaloop.api.deps import get_repo
    app.dependency_overrides[get_repo] = lambda: repo

    client = TestClient(app)
    resp = client.get("/api/learning-curve")
    app.dependency_overrides = {}

    assert resp.status_code == 200
    body = resp.json()
    assert body["iterations"] == [1]
    assert body["task_success_rate"] == [pytest.approx(1.0)]
    assert body["avg_score"] == [pytest.approx(0.9)]


def test_learning_curve_empty():
    repo = _mock_repo(iteration_numbers=[])

    from deltaloop.api.deps import get_repo
    app.dependency_overrides[get_repo] = lambda: repo

    client = TestClient(app)
    resp = client.get("/api/learning-curve")
    app.dependency_overrides = {}

    assert resp.status_code == 200
    body = resp.json()
    assert body["iterations"] == []


# ---------------------------------------------------------------------------
# GET /api/iterations/{n}/pairs
# ---------------------------------------------------------------------------


def test_get_pairs_no_filter():
    pairs = [_pair(cluster_label=0), _pair(task_id="t2", cluster_label=1)]
    repo = _mock_repo(pairs=pairs)

    from deltaloop.api.deps import get_repo
    app.dependency_overrides[get_repo] = lambda: repo

    client = TestClient(app)
    resp = client.get("/api/iterations/1/pairs")
    app.dependency_overrides = {}

    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_get_pairs_with_cluster_filter():
    pairs = [_pair(cluster_label=0), _pair(task_id="t2", cluster_label=1)]
    repo = _mock_repo(pairs=pairs)

    from deltaloop.api.deps import get_repo
    app.dependency_overrides[get_repo] = lambda: repo

    client = TestClient(app)
    resp = client.get("/api/iterations/1/pairs?cluster_label=1")
    app.dependency_overrides = {}

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["task_id"] == "t2"
