import { motion, useReducedMotion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import { AXIS, ChartTip, GRID, Legend, Reveal } from "../ui";
import { departureColor, fmt, signed, weekday } from "../../api";

const TERRAIN = { plains: "#f97316", coastal: "#22d3ee", hilly: "#a5b4fc" };
const COMPONENTS = [
  { k: "temperature", name: "Temperature", w: 0.4, color: "#f97316" },
  { k: "heatwave_probability", name: "Heatwave probability", w: 0.25, color: "#ef4444" },
  { k: "anomaly", name: "Anomaly", w: 0.2, color: "#fbbf24" },
  { k: "persistence", name: "Persistence", w: 0.1, color: "#dc2626" },
  { k: "humidity", name: "Humidity", w: 0.05, color: "#22d3ee" },
];

function SectionHead({ index, title, children }) {
  return (
    <Reveal className="section-head">
      <span className="label">{index}</span>
      <h2>{title}</h2>
      <p>{children}</p>
    </Reveal>
  );
}

function Frame({ title, note, children, delay = 0, className = "" }) {
  return (
    <Reveal delay={delay} className={`hw-panel ${className}`}>
      <header className="hw-panel-head">
        <h3 className="hw-panel-title">{title}</h3>
      </header>
      <div className="hw-panel-body">
        {note && <p className="frame-note">{note}</p>}
        {children}
      </div>
    </Reveal>
  );
}

export default function NationalSections({ rows, dates, onPick }) {
  const ranked = [...rows].sort((a, b) => (b.risk ?? 0) - (a.risk ?? 0));
  return (
    <section className="national">
      <SectionHead index="01 · Week ahead" title="Every city, every day">
        Maximum temperature for each featured city over the next seven days, coloured by departure from that date’s normal.
        Outlined cells meet IMD heatwave criteria. Select a row to open the city.
      </SectionHead>
      <Frame title="7-day heat matrix" className="matrix-frame">
        <HeatMatrix rows={ranked} dates={dates} onPick={onPick} />
      </Frame>

      <SectionHead index="02 · Tomorrow" title="How far above normal, and why the score">
        The model’s prediction for tomorrow against the normal for the date, and the five weighted components that make up
        each city’s 0–100 heat-risk index.
      </SectionHead>
      <div className="duo">
        <Frame title="Predicted maximum vs normal" note="Each bar runs from the normal to tomorrow’s prediction.">
          <Dumbbell rows={ranked} />
        </Frame>
        <Frame title="Risk index composition" note="Weighted points per component." delay={0.1}>
          <RiskStack rows={ranked} />
        </Frame>
      </div>

      <div className="duo">
        <Frame title="Departure vs heatwave probability" note="Bubble size: risk index. Colour: terrain. Dashed line: IMD +4.5 °C departure.">
          <ProbScatter rows={rows} />
          <Legend items={Object.entries(TERRAIN).map(([k, c]) => ({ label: k, color: c, round: true }))} />
        </Frame>
        <Frame title="Method" delay={0.1}>
          <Method />
        </Frame>
      </div>

      <footer className="national-foot">
        Weather: Open-Meteo forecast and ERA5 archive. Heatwave rules: India Meteorological Department criteria for plains,
        coastal and hill stations, including the 45/47 °C absolute rule. Town and district scale, not street level.
      </footer>
    </section>
  );
}

function HeatMatrix({ rows, dates, onPick }) {
  const reduce = useReducedMotion();
  if (!dates.length) return <p className="frame-note">Loading the 7-day outlook…</p>;
  return (
    <div className="matrix" role="table" aria-label="Seven-day maximum temperature by city">
      <div className="matrix-row matrix-head" role="row">
        <span className="label" role="columnheader">City</span>
        {dates.map((d, i) => (
          <span key={d} role="columnheader" className="mono">
            {weekday(d)} {d.slice(8)}
            <small>{i === 0 ? "model" : i <= 2 ? "forecast" : "low conf."}</small>
          </span>
        ))}
      </div>
      {rows.map((r, ri) => (
        <button type="button" key={r.slug} className="matrix-row" role="row" onClick={() => onPick(r)}>
          <span role="rowheader" className="matrix-city">
            {r.name}
            <small className="label">{r.terrain}</small>
          </span>
          {r.days.map((d, i) => (
            <motion.span
              key={d.date}
              role="cell"
              className={`matrix-cell${d.alert ? " alert" : ""}`}
              style={{ "--c": departureColor(d.departure ?? 0), opacity: i >= 3 ? 0.8 : 1 }}
              title={`${r.name} ${d.date}: ${fmt(d.tmax)} °C, ${signed(d.departure)} °C vs normal${d.alert ? ` — ${d.alert}` : ""}`}
              initial={reduce ? false : { opacity: 0, scaleY: 0.2 }}
              whileInView={{ opacity: i >= 3 ? 0.8 : 1, scaleY: 1 }}
              viewport={{ once: true }}
              transition={{ delay: ri * 0.025 + i * 0.03, duration: 0.5 }}
            >
              <b className="num">{fmt(d.tmax, 0)}°</b>
              <small className="mono">{signed(d.departure)}</small>
            </motion.span>
          ))}
        </button>
      ))}
    </div>
  );
}

function Dumbbell({ rows }) {
  const data = rows
    .filter((r) => r.detail)
    .map((r) => ({ name: r.name, range: [r.detail.prediction.normal_tmax, r.detail.prediction.predicted_tmax], dep: r.detail.prediction.departure }))
    .sort((a, b) => b.range[1] - a.range[1]);
  return (
    <div style={{ height: Math.max(280, data.length * 24) }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 20, left: 6, bottom: 0 }}>
          <CartesianGrid stroke={GRID} horizontal={false} />
          <XAxis type="number" domain={["dataMin - 2", "dataMax + 1"]} {...AXIS} tickFormatter={(v) => `${v.toFixed(0)}°`} />
          <YAxis type="category" dataKey="name" width={80} {...AXIS} />
          <Tooltip content={<ChartTip />} cursor={{ fill: "rgba(125,211,252,0.04)" }} />
          <Bar dataKey="range" name="Normal → predicted" barSize={8} radius={1} animationDuration={1400}>
            {data.map((d) => (
              <Cell key={d.name} fill={departureColor(d.dep)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function RiskStack({ rows }) {
  const data = rows
    .filter((r) => r.detail)
    .map((r) => {
      const c = r.detail.prediction.risk_components;
      return Object.fromEntries([["name", r.name], ...COMPONENTS.map((m) => [m.k, +(c[m.k] * m.w).toFixed(2)])]);
    });
  return (
    <>
      <div style={{ height: Math.max(280, data.length * 24) }}>
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 6, bottom: 0 }}>
            <CartesianGrid stroke={GRID} horizontal={false} />
            <XAxis type="number" {...AXIS} domain={[0, "auto"]} />
            <YAxis type="category" dataKey="name" width={80} {...AXIS} />
            <Tooltip content={<ChartTip unit=" pts" />} cursor={{ fill: "rgba(125,211,252,0.04)" }} />
            {COMPONENTS.map((m) => (
              <Bar key={m.k} dataKey={m.k} name={m.name} stackId="r" fill={m.color} barSize={8} animationDuration={1400} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <Legend items={COMPONENTS.map((m) => ({ label: `${m.name} ×${m.w}`, color: m.color }))} />
    </>
  );
}

function ProbScatter({ rows }) {
  const data = rows
    .filter((r) => r.detail)
    .map((r) => ({ name: r.name, x: r.detail.prediction.departure, y: r.detail.prediction.heatwave_probability * 100, z: r.risk, terrain: r.terrain }));
  return (
    <div style={{ height: 290 }}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 14, right: 16, left: -8, bottom: 4 }}>
          <CartesianGrid stroke={GRID} />
          <XAxis type="number" dataKey="x" name="Departure" unit="°" {...AXIS} domain={[(min) => Math.min(-1, Math.floor(min)), (max) => Math.max(6, Math.ceil(max))]} />
          <YAxis type="number" dataKey="y" name="Heatwave probability" unit="%" {...AXIS} />
          <ZAxis type="number" dataKey="z" range={[50, 520]} />
          <ReferenceLine x={4.5} stroke="#f97316" strokeDasharray="4 4" />
          <Tooltip
            cursor={false}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <div className="chart-tip">
                  <b>{payload[0].payload.name}</b>
                  <div className="chart-tip-row"><span>Departure</span><b className="mono">{signed(payload[0].payload.x)} °C</b></div>
                  <div className="chart-tip-row"><span>Heatwave prob.</span><b className="mono">{fmt(payload[0].payload.y, 0)}%</b></div>
                  <div className="chart-tip-row"><span>Risk</span><b className="mono">{fmt(payload[0].payload.z, 0)}</b></div>
                </div>
              ) : null
            }
          />
          <Scatter data={data} animationDuration={1400}>
            {data.map((d) => (
              <Cell key={d.name} fill={TERRAIN[d.terrain]} fillOpacity={0.7} stroke="#e6f1f8" strokeOpacity={0.5} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function Method() {
  const steps = [
    ["Live weather", "Last 14 days and the next 7 from Open-Meteo for any point in India."],
    ["Normals", "Ten years of ERA5 reanalysis around each date, shifted by the measured forecast–archive bias."],
    ["Models", "Gradient boosting predicts tomorrow’s maximum; a Random Forest estimates heatwave probability and severity."],
    ["IMD rules", "Plains, coastal and hill criteria, including the 45/47 °C absolute rule for plains."],
    ["Explanation", "SHAP values show which inputs pushed each prediction up or down."],
  ];
  return (
    <ol className="method">
      {steps.map(([t, d], i) => (
        <motion.li key={t} initial={{ opacity: 0, x: 16 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.1 }}>
          <span className="mono">{String(i + 1).padStart(2, "0")}</span>
          <div>
            <b>{t}</b>
            <p>{d}</p>
          </div>
        </motion.li>
      ))}
    </ol>
  );
}
