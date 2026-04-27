import { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import LearningCurve from "./components/LearningCurve";
import FailureExplorer from "./components/FailureExplorer";
import TrainingLog from "./components/TrainingLog";
import AnimatedGradientBackground from "./components/ui/animated-gradient-background";

const POLL_MS = 5000;

function useStatus() {
  const [status, setStatus] = useState(null);
  useEffect(() => {
    const tick = () => api.getStatus().then(setStatus).catch(() => {});
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => clearInterval(id);
  }, []);
  return status;
}

export default function App() {
  const status = useStatus();
  const [iterations, setIterations] = useState([]);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState(null);

  useEffect(() => {
    api.getIterations()
      .then((list) => setIterations(list.map((s) => s.iteration)))
      .catch(() => {});
  }, [status]);

  const handleRun = useCallback(async () => {
    setRunError(null);
    setRunning(true);
    try {
      await api.triggerRun();
    } catch (e) {
      setRunError(e.message);
    } finally {
      setRunning(false);
    }
  }, []);

  const isBusy = status && status.state !== "idle";

  return (
    <>
      <div style={{ position: "relative", overflow: "hidden", minHeight: 80 }}>
        <AnimatedGradientBackground
          Breathing={true}
          gradientColors={["#0f172a", "#1e1b4b", "#312e81", "#4338ca", "#6366f1", "#818cf8", "#c7d2fe"]}
          gradientStops={[20, 40, 55, 70, 82, 92, 100]}
          startingGap={130}
          breathingRange={6}
          animationSpeed={0.015}
        />
        <header className="app-header" style={{ position: "relative", zIndex: 1, background: "transparent", borderBottom: "1px solid rgba(99,102,241,0.3)" }}>
        <h1>DeltaLoop</h1>
        <div className="status-bar">
          {status && (
            <>
              <span className={`status-dot ${isBusy ? "busy" : ""}`} />
              <span>{status.state}</span>
              {status.current_task && <span>· {status.current_task}</span>}
              {status.adapter && <span>· {status.adapter}</span>}
            </>
          )}
          <button
            className="run-btn"
            onClick={handleRun}
            disabled={running || isBusy}
          >
            {isBusy ? "Running…" : "Run Iteration"}
          </button>
          {runError && <span className="error">{runError}</span>}
        </div>
      </header>
      </div>
      <main>
        <LearningCurve />
        <TrainingLog />
        {iterations.length > 0 && <FailureExplorer iterations={iterations} />}
      </main>
    </>
  );
}
