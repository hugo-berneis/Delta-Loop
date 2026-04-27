import { useEffect, useState } from "react";
import { api } from "../api/client";

const BADGE_COLORS = {
  WRONG_RETRIEVAL: "#ef4444",
  WRONG_REASONING: "#f97316",
  INCOMPLETE_ANSWER: "#eab308",
  HALLUCINATION: "#8b5cf6",
  TOOL_MISUSE: "#06b6d4",
};

function Badge({ mode }) {
  const color = BADGE_COLORS[mode] ?? "#6b7280";
  return (
    <span style={{ background: color, color: "#fff", borderRadius: 4, padding: "2px 6px", fontSize: 11, fontWeight: 600 }}>
      {mode}
    </span>
  );
}

function DiffRow({ pair, expanded, onToggle }) {
  return (
    <>
      <tr onClick={onToggle} style={{ cursor: "pointer" }}>
        <td>{pair.task_id}</td>
        <td><Badge mode={pair.failure_mode} /></td>
        <td>{pair.failure_explanation}</td>
        <td>{pair.quality_score.toFixed(3)}</td>
        <td>{pair.cluster_label}</td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={5}>
            <div className="diff-grid">
              <div>
                <strong>Rejected</strong>
                <pre>{pair.rejected_trace}</pre>
              </div>
              <div>
                <strong>Chosen</strong>
                <pre>{pair.chosen_trace}</pre>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function FailureExplorer({ iterations }) {
  const [selectedIter, setSelectedIter] = useState(iterations[0] ?? null);
  const [clusterFilter, setClusterFilter] = useState("");
  const [pairs, setPairs] = useState([]);
  const [expandedId, setExpandedId] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (selectedIter == null) return;
    const label = clusterFilter !== "" ? parseInt(clusterFilter) : undefined;
    api.getPairs(selectedIter, label)
      .then(setPairs)
      .catch((e) => setError(e.message));
  }, [selectedIter, clusterFilter]);

  if (error) return <p className="error">Failed to load: {error}</p>;

  return (
    <section className="card">
      <h2>Failure Explorer</h2>
      <div className="toolbar">
        <label>
          Iteration{" "}
          <select value={selectedIter ?? ""} onChange={(e) => setSelectedIter(Number(e.target.value))}>
            {iterations.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <label>
          Cluster{" "}
          <input
            type="number"
            min={0}
            value={clusterFilter}
            onChange={(e) => setClusterFilter(e.target.value)}
            placeholder="all"
            style={{ width: 60 }}
          />
        </label>
      </div>
      {pairs.length === 0 ? (
        <p>No preference pairs found.</p>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Task</th>
              <th>Failure Mode</th>
              <th>Explanation</th>
              <th>Quality</th>
              <th>Cluster</th>
            </tr>
          </thead>
          <tbody>
            {pairs.map((p) => (
              <DiffRow
                key={p.id ?? p.task_id}
                pair={p}
                expanded={expandedId === p.id}
                onToggle={() => setExpandedId(expandedId === p.id ? null : p.id)}
              />
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
