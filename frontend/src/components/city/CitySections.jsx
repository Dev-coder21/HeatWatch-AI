import { useMemo, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CheckCircle2, CircleSlash } from "lucide-react";

import { AXIS, GRID, Reveal } from "../ui";
import { bandColor, fmt, severityColor, signed } from "../../api";

const IMD_MIN = { plains: 40, coastal: 37, hilly: 30 };
const AUDIENCES = [
  ["general", "Everyone"],
  ["outdoor_workers", "Outdoor workers"],
  ["elderly", "Elderly"],
  ["children", "Children"],
];

function Frame({ title, aside, children, delay = 0, className = "" }) {
  return (
    <Reveal delay={delay} className={`hw-panel ${className}`}>
      <header className="hw-panel-head">
        <h3 className="hw-panel-title">{title}</h3>
        {aside}
      </header>
      <div className="hw-panel-body">{children}</div>
    </Reveal>
  );
}

function SectionHead({ index, title, children }) {
  return (
    <Reveal className="section-head">
      <span className="label">{index}</span>
      <h2>{title}</h2>
      <p>{children}</p>
    </Reveal>
  );
}

export default function CitySections({ risk, explain, name, children }) {
  return (
    <section className="city-sections">
      {children}
      <SectionHead index="01 · Explainable AI" title={explain ? `Why ${fmt(explain.predicted_tmax.prediction)} °C tomorrow?` : "Why this prediction?"}>
        SHAP values show how each input moved the models away from their average day for {name}. Warm colours push the
        prediction up; blue pulls it down.
      </SectionHead>
      <div className="duo">
        <Frame title="Temperature model · contribution waterfall" aside={<span className="label">°C</span>}>
          <Waterfall block={explain?.predicted_tmax} />
        </Frame>
        <Frame title="Heatwave probability · biggest pushes" aside={<span className="label">percentage points</span>} delay={0.1}>
          <Pushes block={explain?.heatwave_probability} />
        </Frame>
      </div>

      <SectionHead index="02 · Rules and action" title="What IMD would call it, and what to do">
        The India Meteorological Department’s criteria applied to tomorrow’s prediction, and advice matched to the risk
        level for the people most exposed to heat.
      </SectionHead>
      <div className="duo">
        <RuleCheck risk={risk} />
        <Advice risk={risk} />
      </div>

      <p className="city-foot">
        {risk.data.source}. Resolution: {risk.data.grid_resolution}. Decision support, not an official IMD warning.
      </p>
    </section>
  );
}

function Waterfall({ block }) {
  const rows = useMemo(() => {
    if (!block) return [];
    const top = block.contributions.slice(0, 8);
    const rest = block.contributions.slice(8).reduce((s, c) => s + c.shap, 0);
    let run = block.base_value;
    const items = [...top, { label: "All other inputs", shap: rest, value: null }];
    return [
      { label: "Model average day", range: [0, block.base_value], kind: "base", v: block.base_value },
      ...items.map((c) => {
        const start = run;
        run += c.shap;
        return { label: c.label, range: [Math.min(start, run), Math.max(start, run)], v: c.shap, value: c.value, kind: c.shap >= 0 ? "up" : "down" };
      }),
      { label: "Prediction", range: [0, run], kind: "total", v: run },
    ];
  }, [block]);
  if (!block) return <p className="frame-note">Computing SHAP values…</p>;
  const lo = Math.floor(Math.min(...rows.map((r) => r.range[0] || r.range[1])) - 1);
  const domainMin = Math.min(lo, Math.floor(block.prediction - 6));
  return (
    <div style={{ height: 380 }}>
      <ResponsiveContainer>
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 30, left: 4, bottom: 0 }}>
          <CartesianGrid stroke={GRID} horizontal={false} />
          <XAxis type="number" {...AXIS} domain={[domainMin, "dataMax + 1"]} allowDataOverflow tickFormatter={(v) => `${v.toFixed(0)}°`} />
          <YAxis type="category" dataKey="label" width={170} {...AXIS} fontFamily="Barlow" fontSize={11} />
          <Tooltip
            cursor={{ fill: "rgba(125,211,252,0.05)" }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <div className="chart-tip">
                  <b>{payload[0].payload.label}</b>
                  <div className="chart-tip-row">
                    <span>{payload[0].payload.kind === "up" || payload[0].payload.kind === "down" ? "Contribution" : "Value"}</span>
                    <b className="mono">{payload[0].payload.kind === "up" || payload[0].payload.kind === "down" ? signed(payload[0].payload.v, 2) : fmt(payload[0].payload.v, 2)} °C</b>
                  </div>
                  {payload[0].payload.value !== null && payload[0].payload.value !== undefined && (
                    <div className="chart-tip-row"><span>Input value</span><b className="mono">{fmt(payload[0].payload.value, 2)}</b></div>
                  )}
                </div>
              ) : null
            }
          />
          <Bar dataKey="range" barSize={12} animationDuration={1400}>
            {rows.map((r) => (
              <Cell key={r.label} fill={r.kind === "up" ? "#f97316" : r.kind === "down" ? "#38bdf8" : r.kind === "total" ? "#fbbf24" : "#1e3a4c"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Pushes({ block }) {
  const reduce = useReducedMotion();
  if (!block) return <p className="frame-note">Computing…</p>;
  const items = block.contributions.slice(0, 9);
  const max = Math.max(...items.map((c) => Math.abs(c.shap)), 0.01);
  return (
    <>
      <p className="frame-note">
        From the model’s {fmt(block.base_value * 100, 0)}% baseline to {fmt(block.prediction * 100, 0)}% for tomorrow.
      </p>
      <ul className="pushes">
        {items.map((c, i) => (
          <li key={c.feature}>
            <span className="push-label">{c.label}</span>
            <span className="push-track">
              <motion.i
                initial={reduce ? false : { scaleX: 0 }}
                whileInView={{ scaleX: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.9, delay: 0.15 + i * 0.06 }}
                style={{
                  width: `${(Math.abs(c.shap) / max) * 50}%`,
                  [c.shap >= 0 ? "left" : "right"]: "50%",
                  transformOrigin: c.shap >= 0 ? "left" : "right",
                  background: c.shap >= 0 ? "linear-gradient(90deg, #f97316, #ef4444)" : "linear-gradient(270deg, #38bdf8, #0ea5e9)",
                }}
              />
            </span>
            <b className="mono">{signed(c.shap * 100, 1)}</b>
          </li>
        ))}
      </ul>
    </>
  );
}

function RuleCheck({ risk }) {
  const p = risk.prediction;
  const terrain = risk.location.terrain_type;
  const min = IMD_MIN[terrain];
  const checks = [
    { label: `Maximum temperature ≥ ${min} °C`, value: `${fmt(p.predicted_tmax)} °C`, met: p.predicted_tmax >= min },
    { label: "Departure from normal ≥ +4.5 °C", value: `${signed(p.departure)} °C`, met: p.departure >= 4.5 },
    { label: "Departure ≥ +6.5 °C (severe)", value: `${signed(p.departure)} °C`, met: p.departure >= 6.5 },
  ];
  if (terrain === "plains") checks.push({ label: "Absolute rule: ≥ 45 °C (≥ 47 severe)", value: `${fmt(p.predicted_tmax)} °C`, met: p.predicted_tmax >= 45 });
  const result = p.imd_rule_severity;
  return (
    <Frame title={`IMD heatwave criteria · ${terrain} station`}>
      <ul className="rules">
        {checks.map((c, i) => (
          <motion.li key={c.label} initial={{ opacity: 0, x: -12 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.12 }}>
            {c.met ? <CheckCircle2 size={16} color="#f97316" /> : <CircleSlash size={16} color="#3b4f5f" />}
            <span>{c.label}</span>
            <b className="mono" style={{ color: c.met ? "#fdba74" : "var(--text-2)" }}>{c.value}</b>
          </motion.li>
        ))}
      </ul>
      <div className="rule-verdict" style={{ borderColor: severityColor(result), color: result === "No Heatwave" ? "var(--cyan-1)" : severityColor(result) }}>
        <span className="label">Result for tomorrow</span>
        <b>{result === "No Heatwave" ? "No heatwave" : result}</b>
        {p.model_severity !== result && <small>Model alone: {p.model_severity.toLowerCase()}</small>}
      </div>
    </Frame>
  );
}

function Advice({ risk }) {
  const [who, setWho] = useState("general");
  const a = risk.advisory;
  const text = who === "general" ? a.general : a.audiences[who];
  return (
    <Frame title="Advisory" aside={<span className="label" style={{ color: bandColor(risk.prediction.risk_category) }}>level · {a.level.replace("_", " ")}</span>} delay={0.1}>
      <div className="tabs" role="tablist" aria-label="Audience">
        {AUDIENCES.map(([k, label]) => (
          <button key={k} type="button" role="tab" aria-selected={who === k} className={who === k ? "on" : ""} onClick={() => setWho(k)}>
            {label}
          </button>
        ))}
      </div>
      <motion.p key={who} className="advice" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
        {text}
      </motion.p>
      {a.context.length > 0 && <p className="frame-note">{a.context.join(" ")}</p>}
    </Frame>
  );
}
