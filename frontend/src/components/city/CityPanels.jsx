import { motion } from "framer-motion";
import { MapContainer, Rectangle, TileLayer } from "react-leaflet";
import {
  Area,
  Bar,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  Pie,
  PieChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { AXIS, ChartTip, GRID, Panel } from "../ui";
import { bandColor, fmt, severityColor, signed, tempColor, weekday } from "../../api";
import { IMAGERY, gridBounds } from "./AerialScene";

export const COMPONENTS = [
  { k: "temperature", name: "Temperature", w: 0.4, color: "#f97316" },
  { k: "heatwave_probability", name: "HW prob.", w: 0.25, color: "#ef4444" },
  { k: "anomaly", name: "Anomaly", w: 0.2, color: "#fbbf24" },
  { k: "persistence", name: "Persistence", w: 0.1, color: "#dc2626" },
  { k: "humidity", name: "Humidity", w: 0.05, color: "#22d3ee" },
];

const IMD_MIN = { plains: 40, coastal: 37, hilly: 30 };

/* ------------------------------------------------------------------
   Left: city profile (context map, description, stats, radar, donut)
   ------------------------------------------------------------------ */
export function ProfilePanel({ risk, name, delay }) {
  const loc = risk.location;
  const p = risk.prediction;
  const pts = COMPONENTS.map((c) => ({ ...c, pts: p.risk_components[c.k] * c.w }));
  const total = pts.reduce((s, c) => s + c.pts, 0) || 1;
  const terrainText = {
    plains: "an inland plains location",
    coastal: `a coastal location ${fmt(loc.distance_to_coast_km, 0)} km from the sea`,
    hilly: `a hill location at ${fmt(loc.elevation_m, 0)} m`,
  }[loc.terrain_type];

  return (
    <Panel title="City profile" from="left" delay={delay} className="profile-panel">
      <div className="mini-map">
        <MapContainer
          center={[loc.grid_lat, loc.grid_lon]}
          zoom={8}
          zoomControl={false}
          dragging={false}
          scrollWheelZoom={false}
          doubleClickZoom={false}
          touchZoom={false}
          keyboard={false}
          attributionControl={false}
          style={{ height: "100%" }}
        >
          <TileLayer url={IMAGERY} />
          <Rectangle bounds={gridBounds(loc.grid_key)} pathOptions={{ color: "#67e8f9", weight: 1.5, fillOpacity: 0.15 }} />
        </MapContainer>
        <span className="mini-map-tag mono">CELL {loc.grid_key}</span>
      </div>
      <h3 className="profile-name">{name}</h3>
      <p className="profile-text">
        {name} is {terrainText} in {loc.state || "India"}. HeatWatch assesses the ~11 km grid cell around it, so the
        IMD criteria for {loc.terrain_type} stations apply.
      </p>
      <div className="stat-boxes">
        <div>
          <span className="label">Elevation (m)</span>
          <b className="num">{fmt(loc.elevation_m, 0)}</b>
        </div>
        <div>
          <span className="label">From coast (km)</span>
          <b className="num">{fmt(loc.distance_to_coast_km, 0)}</b>
        </div>
      </div>

      <div className="radar-wrap">
        <ResponsiveContainer>
          <RadarChart data={COMPONENTS.map((c) => ({ name: c.name, value: p.risk_components[c.k] }))} outerRadius="70%">
            <PolarGrid stroke="rgba(125,211,252,0.18)" />
            <PolarAngleAxis dataKey="name" tick={{ fill: "#a9bccb", fontSize: 9, fontFamily: "Barlow" }} />
            <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
            <Radar dataKey="value" stroke="#38bdf8" strokeWidth={1.5} fill="#38bdf8" fillOpacity={0.22} animationDuration={1600} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      <div className="donut-head">
        <span className="hw-panel-title">Risk index composition</span>
        <span className="label">{weekday(risk.forecast_date, { day: "2-digit", month: "short" })}</span>
      </div>
      <div className="donut-row">
        <div className="donut">
          <ResponsiveContainer>
            <PieChart>
              <Pie data={pts} dataKey="pts" innerRadius="68%" outerRadius="92%" stroke="none" startAngle={90} endAngle={-270} animationDuration={1600}>
                {pts.map((c) => (
                  <Cell key={c.k} fill={c.color} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="donut-center">
            <b className="num" style={{ color: bandColor(p.risk_category) }}>{fmt(p.risk_score, 0)}</b>
            <span className="label">{p.risk_category}</span>
          </div>
        </div>
        <ul className="donut-legend">
          {pts.map((c) => (
            <li key={c.k}>
              <i style={{ background: c.color }} />
              {c.name}
              <b className="mono">{fmt((c.pts / total) * 100, 1)}%</b>
            </li>
          ))}
        </ul>
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------
   Right: 7-day forecast (bars = max temp, line = departure) + outlook list
   ------------------------------------------------------------------ */
export function ForecastPanel({ risk, delay }) {
  const p = risk.prediction;
  const data = risk.outlook.map((o, i) => ({
    date: o.date,
    tmax: i === 0 ? p.predicted_tmax : o.forecast_tmax,
    departure: i === 0 ? p.departure : o.departure,
    normal: o.normal_tmax,
  }));
  return (
    <Panel title="7-day heat forecast" aside={<span className="label">{weekday(risk.issued_date, { day: "2-digit", month: "short" })} issue</span>} from="right" delay={delay} className="forecast-panel">
      <span className="bracket">Max temperature (bars) · departure from normal (line)</span>
      <div className="combo">
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 10, right: 0, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="fc-bar" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0" stopColor="#38bdf8" stopOpacity={0.95} />
                <stop offset="1" stopColor="#0ea5e9" stopOpacity={0.15} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="date" {...AXIS} tickFormatter={(d) => weekday(d)} />
            <YAxis yAxisId="t" {...AXIS} width={30} domain={["dataMin - 4", "dataMax + 2"]} tickFormatter={(v) => v.toFixed(0)} />
            <YAxis yAxisId="d" orientation="right" {...AXIS} width={30} tickFormatter={(v) => v.toFixed(0)} />
            <ReferenceLine yAxisId="d" y={4.5} stroke="#f97316" strokeDasharray="3 3" />
            <Tooltip content={<ChartTip labelFormat={(d) => weekday(d, { weekday: "long", day: "numeric", month: "short" })} />} cursor={{ fill: "rgba(125,211,252,0.05)" }} />
            <Bar yAxisId="t" dataKey="tmax" name="Max temp" barSize={12} fill="url(#fc-bar)" animationDuration={1400}>
              {data.map((d, i) => (
                <Cell key={d.date} fill={i === 0 ? tempColor(d.tmax) : "url(#fc-bar)"} />
              ))}
            </Bar>
            <Line yAxisId="d" dataKey="departure" name="Departure" stroke="#fbbf24" strokeWidth={1.8} dot={{ r: 2.5, fill: "#fbbf24" }} animationDuration={1800} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="combo-key mono">
        <span><i style={{ background: "#38bdf8" }} /> max temp °C</span>
        <span><i style={{ background: "#fbbf24" }} /> departure °C</span>
        <span><i className="dash" /> IMD +4.5</span>
      </div>

      <div className="outlook-head">
        <span className="hw-panel-title">Outlook &amp; conditions</span>
        <span className="label">confidence</span>
      </div>
      <ul className="outlook-list">
        {risk.outlook.map((o, i) => {
          const t = i === 0 ? p.predicted_tmax : o.forecast_tmax;
          return (
            <motion.li
              key={o.date}
              className={i === 0 ? "on" : ""}
              initial={{ opacity: 0, x: 12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: delay + 0.4 + i * 0.06 }}
            >
              <span className="mono ol-day">{weekday(o.date)} {o.date.slice(8)}</span>
              <b className="num" style={{ color: tempColor(t) }}>{fmt(t, 0)}°</b>
              <span className="mono ol-meta">{fmt(o.humidity, 0)}% · {fmt(o.precipitation_mm, 1)} mm</span>
              {o.alert_label && <span className="ol-alert" style={{ color: severityColor(o.alert_label) }}>{o.alert_label}</span>}
              <span className={`ol-conf conf-${o.confidence}`}>{o.confidence}</span>
            </motion.li>
          );
        })}
      </ul>
    </Panel>
  );
}

/* ------------------------------------------------------------------
   Bottom-left: city data card (key/value table)
   ------------------------------------------------------------------ */
export function CityCard({ risk, delay, className = "" }) {
  const loc = risk.location;
  const p = risk.prediction;
  const d = risk.data;
  const rows = [
    ["Grid cell", loc.grid_key, "Terrain rule", `${loc.terrain_type} · ≥ ${IMD_MIN[loc.terrain_type]} °C`],
    ["State", loc.state || "—", "IMD result tomorrow", p.imd_rule_severity],
    ["Normal (bias-corrected)", `${fmt(p.normal_tmax)} °C`, "Model severity", p.model_severity],
    ["Archive normal", `${fmt(p.archive_normal_tmax)} °C`, "Bias correction", `${signed(d.normal_bias_correction_c, 2)} °C · ${d.normal_bias_overlap_days} d`],
    ["Raw forecast", `${fmt(p.raw_forecast_tmax)} °C`, "Weather age", d.stale ? "cached (stale)" : `${Math.round(d.weather_age_s / 60)} min`],
  ];
  return (
    <Panel title="City data card" from="up" delay={delay} className={`card-panel ${className}`} bodyClass="card-body">
      <table className="card-table">
        <tbody>
          {rows.map((r) => (
            <tr key={r[0]}>
              <th>{r[0]}</th>
              <td className="mono">{r[1]}</td>
              <th>{r[2]}</th>
              <td className="mono" style={r[2].includes("severity") || r[2].includes("IMD") ? { color: severityColor(r[3]) } : undefined}>{r[3]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}

/* ------------------------------------------------------------------
   Bottom-right: 15-day temperature (observed + forecast + normal)
   ------------------------------------------------------------------ */
export function fifteenDays(risk) {
  const p = risk.prediction;
  const data = [
    ...risk.recent_days.map((d) => ({ date: d.date, observed: d.tmax, normal: d.normal_tmax, band: [d.normal_tmax + 4.5, d.normal_tmax + 6.5] })),
    ...risk.outlook.map((o, i) => ({
      date: o.date,
      forecast: i === 0 ? p.predicted_tmax : o.forecast_tmax,
      normal: o.normal_tmax,
      band: [o.normal_tmax + 4.5, o.normal_tmax + 6.5],
      alert: o.alert_label ? (i === 0 ? p.predicted_tmax : o.forecast_tmax) : null,
    })),
  ];
  const today = risk.recent_days.length - 1;
  data[today].forecast = data[today].observed;
  return { data, todayDate: data[today].date };
}

export function TrendPanel({ risk, delay }) {
  const { data, todayDate } = fifteenDays(risk);
  return (
    <Panel title="Temperature · last 8 days and next 7" from="up" delay={delay} className="trend-panel" bodyClass="trend-body">
      <div className="trend-chart">
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 8, right: 6, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="t15-obs" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0" stopColor="#fbbf24" stopOpacity={0.5} />
                <stop offset="1" stopColor="#fbbf24" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="t15-fc" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0" stopColor="#22d3ee" stopOpacity={0.4} />
                <stop offset="1" stopColor="#22d3ee" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="date" {...AXIS} tickFormatter={(d) => `${d.slice(5, 7)}.${d.slice(8)}`} interval={2} />
            <YAxis {...AXIS} width={30} domain={["dataMin - 2", "dataMax + 2"]} tickFormatter={(v) => v.toFixed(0)} />
            <Tooltip content={<ChartTip labelFormat={(d) => weekday(d, { weekday: "short", day: "numeric", month: "short" })} />} />
            <Area dataKey="band" name="Heatwave band" stroke="none" fill="rgba(239,68,68,0.12)" />
            <ReferenceLine x={todayDate} stroke="rgba(230,241,248,0.35)" strokeDasharray="2 3" />
            <Line dataKey="normal" name="Normal" stroke="#7dd3fc" strokeDasharray="3 3" strokeWidth={1} dot={false} />
            <Area dataKey="observed" name="Observed" stroke="#fbbf24" strokeWidth={1.8} fill="url(#t15-obs)" animationDuration={1600} />
            <Area dataKey="forecast" name="Forecast" stroke="#22d3ee" strokeWidth={1.8} fill="url(#t15-fc)" animationDuration={1600} />
            <Line dataKey="alert" name="IMD flag" stroke="none" dot={{ r: 4, fill: "#ef4444", stroke: "#fecaca", strokeWidth: 1 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="combo-key mono">
        <span><i style={{ background: "#fbbf24" }} /> observed</span>
        <span><i style={{ background: "#22d3ee" }} /> forecast</span>
        <span><i className="dash" /> normal</span>
        <span><i style={{ background: "rgba(239,68,68,0.5)" }} /> heatwave band (+4.5 to +6.5)</span>
      </div>
    </Panel>
  );
}
