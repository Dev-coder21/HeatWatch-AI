// Thin client for the HeatWatch API (contract: docs/API.md).

export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function getJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return body;
}

const point = (lat, lon) => `lat=${Number(lat).toFixed(4)}&lon=${Number(lon).toFixed(4)}`;

export const api = {
  featured: () => getJson("/api/featured"),
  risk: (lat, lon) => getJson(`/api/risk?${point(lat, lon)}`),
  explain: (lat, lon) => getJson(`/api/explain?${point(lat, lon)}`),
  search: (q) => getJson(`/api/search?q=${encodeURIComponent(q)}`),
  reverse: (lat, lon) => getJson(`/api/reverse?${point(lat, lon)}`),
};

// cool -> intelligence cyan -> heat
const RAMP = ["#0ea5e9", "#67e8f9", "#fde68a", "#fbbf24", "#f97316", "#ef4444", "#dc2626"];

function mix(a, b, t) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

function ramp(t) {
  const x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x));
  return mix(RAMP[i], RAMP[i + 1], x - i);
}

export const tempColor = (t) => ramp((t - 18) / (44 - 18));
export const riskColor = (score) => ramp(0.25 + (score / 100) * 0.75);
export const departureColor = (d) => ramp(0.33 + d / 12);

export const RISK_BANDS = [
  { name: "Low", min: 0, color: "#38bdf8" },
  { name: "Moderate", min: 20, color: "#fbbf24" },
  { name: "High", min: 40, color: "#f97316" },
  { name: "Very High", min: 60, color: "#ef4444" },
  { name: "Extreme", min: 80, color: "#dc2626" },
];

export const bandColor = (category) =>
  (RISK_BANDS.find((b) => b.name === category) || RISK_BANDS[0]).color;

export const severityColor = (label) =>
  !label || label === "No Heatwave" ? "var(--muted)" : /severe/i.test(label) ? "#ef4444" : "#f97316";

export const fmt = (v, d = 1) => (v === null || v === undefined || !Number.isFinite(Number(v)) ? "—" : Number(v).toFixed(d));
export const signed = (v, d = 1) => (Number(v) > 0 ? "+" : "") + fmt(v, d);

export function weekday(iso, opts = { weekday: "short" }) {
  return new Date(`${iso}T00:00:00`).toLocaleDateString("en-IN", opts);
}
