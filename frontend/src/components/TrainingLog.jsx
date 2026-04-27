import { useEffect, useState } from "react";
import { api } from "../api/client";

function Delta({ value }) {
  if (value == null) return <span>—</span>;
  const pos = value >= 0;
  return (
    <span style={{ color: pos ? "#22c55e" : "#ef4444", fontWeight: 600 }}>
      {pos ? "+" : ""}{(value * 100).toFixed(1)}%
    </span>
  );
}

export default function TrainingLog() {
  const [summaries, setSummaries] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getIterations()
      .then(setSummaries)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="error">Failed to load: {error}</p>;
  if (summaries.length === 0) return <p className="loading">No iterations yet.</p>;

  const rows = summaries.map((s, i) => {
    const prev = summaries[i - 1];
    const delta = prev ? s.task_success_rate - prev.task_success_rate : null;
    return { ...s, delta };
  });

  return (
    <section className="card">
      <h2>Training Log</h2>
      <table className="data-table">
        <thead>
          <tr>
            <th>Iter</th>
            <th>Success Rate</th>
            <th>Δ</th>
            <th>Avg Score</th>
            <th>Pairs</th>
            <th>Fine-tuned</th>
            <th>Completed</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.iteration}>
              <td>{r.iteration}</td>
              <td>{(r.task_success_rate * 100).toFixed(1)}%</td>
              <td><Delta value={r.delta} /></td>
              <td>{(r.avg_score * 100).toFixed(1)}%</td>
              <td>{r.pairs_stored}</td>
              <td>{r.training_triggered ? "✓" : "—"}</td>
              <td style={{ fontSize: 12, color: "#6b7280" }}>{r.completed_at ? new Date(r.completed_at).toLocaleString() : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
