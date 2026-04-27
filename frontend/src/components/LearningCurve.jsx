import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";

export default function LearningCurve() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getLearningCurve()
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="error">Failed to load: {error}</p>;
  if (!data) return <p className="loading">Loading…</p>;

  const chartData = data.iterations.map((iter, i) => ({
    iteration: iter,
    success_rate: +(data.task_success_rate[i] * 100).toFixed(1),
    avg_score: +(data.avg_score[i] * 100).toFixed(1),
  }));

  return (
    <section className="card">
      <h2>Learning Curve</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="iteration" label={{ value: "Iteration", position: "insideBottom", offset: -4 }} />
          <YAxis unit="%" domain={[0, 100]} />
          <Tooltip formatter={(v) => `${v}%`} />
          <Legend verticalAlign="top" />
          {data.fine_tuning_events.map((n) => (
            <ReferenceLine
              key={n}
              x={n}
              stroke="#f59e0b"
              strokeDasharray="4 2"
              label={{ value: "FT", position: "top", fill: "#f59e0b", fontSize: 11 }}
            />
          ))}
          <Line type="monotone" dataKey="success_rate" stroke="#6366f1" name="Task Success Rate" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="avg_score" stroke="#22c55e" name="Avg Score" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </section>
  );
}
