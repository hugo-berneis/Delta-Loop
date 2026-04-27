const BASE = "";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function post(path) {
  const res = await fetch(`${BASE}${path}`, { method: "POST" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  getLearningCurve: () => get("/api/learning-curve"),
  getIterations: () => get("/api/iterations"),
  getTraces: (n, { page = 1, pageSize = 20, filter } = {}) => {
    const params = new URLSearchParams({ page, page_size: pageSize });
    if (filter) params.set("filter", filter);
    return get(`/api/iterations/${n}/traces?${params}`);
  },
  getPairs: (n, clusterLabel) => {
    const params = clusterLabel != null ? `?cluster_label=${clusterLabel}` : "";
    return get(`/api/iterations/${n}/pairs${params}`);
  },
  getStatus: () => get("/api/status"),
  triggerRun: () => post("/api/run"),
};
