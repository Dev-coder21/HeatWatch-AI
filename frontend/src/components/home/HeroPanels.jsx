import { useMemo } from "react";
import { motion } from "framer-motion";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { AXIS, ChartTip, GRID, Panel } from "../ui";
import { bandColor, fmt, riskColor, tempColor, weekday } from "../../api";

const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
const finite = (xs) => xs.filter(Number.isFinite);

/* ------------------------------------------------------------------
   City ranking (rank / city / tomorrow / risk)
   ------------------------------------------------------------------ */
export function RankingTable({ rows, onPick, hoverSlug }) {
  const ranked = [...rows].sort((a, b) => (b.risk ?? 0) - (a.risk ?? 0));
  return (
    <table className="rank-table">
      <thead>
        <tr>
          <th className="label">Rank</th>
          <th className="label">City</th>
          <th className="label num-col">Tmax</th>
          <th className="label num-col">Risk</th>
        </tr>
      </thead>
      <tbody>
        {ranked.map((r, i) => (
          <motion.tr
            key={r.slug}
            className={r.slug === hoverSlug ? "hot" : ""}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 2.6 + i * 0.04 }}
            onClick={() => onPick(r)}
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && onPick(r)}
            aria-label={`${r.name}: risk ${fmt(r.risk, 0)}, tomorrow ${fmt(r.predicted)} degrees. Open city view.`}
          >
            <td className="mono rank-no">{String(i + 1).padStart(2, "0")}</td>
            <td>
              <span className="rank-city">{r.name}</span>
              <span className="rank-bar">
                <motion.i
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.max(3, r.risk ?? 0)}%` }}
                  transition={{ delay: 2.8 + i * 0.04, duration: 1.1, ease: [0.22, 1, 0.36, 1] }}
                  style={{ background: riskColor(r.risk ?? 0) }}
                />
              </span>
            </td>
            <td className="mono num-col" style={{ color: tempColor(r.predicted) }}>{fmt(r.predicted)}</td>
            <td className="mono num-col" style={{ color: bandColor(r.category) }}>{fmt(r.risk, 0)}</td>
          </motion.tr>
        ))}
      </tbody>
    </table>
  );
}

/* ------------------------------------------------------------------
   Heat progression: 15-day national series (observed + forecast)
   ------------------------------------------------------------------ */
export function ProgressionPanel({ rows, selectedDate }) {
  const series = useMemo(() => {
    if (!rows.length) return [];
    return rows[0].series.map((s, i) => {
      const vals = rows.map((r) => r.series[i]).filter(Boolean);
      const t = mean(finite(vals.map((v) => v.tmax)));
      const dep = mean(finite(vals.map((v) => v.departure)));
      const observed = s.kind === "observed";
      const isToday = i === 7;
      return {
        date: s.date,
        kind: s.kind,
        normal: mean(finite(vals.map((v) => v.normal))),
        observed: observed ? t : null,
        forecast: !observed || isToday ? t : null,
        depObs: observed ? dep : null,
        depFc: !observed || isToday ? dep : null,
        flagged: vals.filter((v) => v.flagged).length,
      };
    });
  }, [rows]);

  const today = series[7]?.date;
  const tick = (d) => (d ? `${d.slice(5, 7)}.${d.slice(8)}` : "");
  const labelFormat = (d) => weekday(d, { weekday: "short", day: "numeric", month: "short" });
  const marks = (
    <>
      <CartesianGrid stroke={GRID} vertical={false} />
      <XAxis dataKey="date" {...AXIS} tickFormatter={tick} interval={7} padding={{ left: 4, right: 10 }} />
      {today && <ReferenceLine x={today} stroke="rgba(230,241,248,0.3)" strokeDasharray="2 3" />}
      {selectedDate && <ReferenceLine x={selectedDate} stroke="#22d3ee" strokeOpacity={0.85} />}
    </>
  );
  const margin = { top: 6, right: 4, left: 0, bottom: 0 };

  return (
    <Panel title="Heat progression trend" from="right" delay={2.3} className="g-panel g-prog" bodyClass="g-prog-body">
      <div className="g-chart">
        <span className="bracket">Mean max temperature · 14 cities</span>
        <div className="g-chart-area">
          <ResponsiveContainer>
            <AreaChart data={series} margin={margin}>
              <defs>
                <linearGradient id="pp-obs" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="1" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="pp-fc" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0" stopColor="#22d3ee" stopOpacity={0.35} />
                  <stop offset="1" stopColor="#22d3ee" stopOpacity={0} />
                </linearGradient>
              </defs>
              {marks}
              <YAxis {...AXIS} width={26} domain={["dataMin - 1", "dataMax + 1"]} tickFormatter={(v) => v.toFixed(0)} />
              <Tooltip content={<ChartTip labelFormat={labelFormat} />} />
              <Area type="monotone" dataKey="normal" name="Normal" stroke="#7dd3fc" strokeOpacity={0.6} strokeDasharray="3 3" strokeWidth={1} fill="none" />
              <Area type="monotone" dataKey="observed" name="Observed" stroke="#ef4444" strokeWidth={1.5} fill="url(#pp-obs)" animationDuration={1800} />
              <Area type="monotone" dataKey="forecast" name="Prediction" stroke="#22d3ee" strokeWidth={1.5} fill="url(#pp-fc)" animationDuration={1800} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="g-key mono">
          <span><i style={{ background: "#ef4444" }} />observed</span>
          <span><i style={{ background: "#22d3ee" }} />prediction</span>
        </div>
      </div>

      <div className="g-chart">
        <span className="bracket">Mean departure from normal</span>
        <div className="g-chart-area">
          <ResponsiveContainer>
            <AreaChart data={series} margin={margin}>
              <defs>
                <linearGradient id="pp-dep" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0" stopColor="#3b82f6" stopOpacity={0.55} />
                  <stop offset="1" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              {marks}
              <YAxis {...AXIS} width={26} tickFormatter={(v) => v.toFixed(1)} />
              <ReferenceLine y={0} stroke="rgba(230,241,248,0.2)" />
              <Tooltip content={<ChartTip labelFormat={labelFormat} />} />
              <Area type="monotone" dataKey="depObs" name="Observed" stroke="#60a5fa" strokeWidth={1.5} fill="url(#pp-dep)" animationDuration={1800} />
              <Area type="monotone" dataKey="depFc" name="Prediction" stroke="#3b82f6" strokeDasharray="4 3" strokeWidth={1.5} fill="url(#pp-dep)" animationDuration={1800} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="g-chart">
        <span className="bracket">Cities meeting IMD heatwave criteria</span>
        <div className="g-chart-area">
          <ResponsiveContainer>
            <BarChart data={series} margin={margin}>
              {marks}
              <YAxis {...AXIS} width={26} allowDecimals={false} domain={[0, (max) => Math.max(3, max)]} />
              <Tooltip content={<ChartTip unit=" cities" labelFormat={labelFormat} />} cursor={{ fill: "rgba(125,211,252,0.05)" }} />
              <Bar dataKey="flagged" name="Cities" barSize={6} animationDuration={1500}>
                {series.map((s) => (
                  <Cell key={s.date} fill={s.date === selectedDate ? "#fbbf24" : s.kind === "observed" ? "rgba(239,68,68,0.85)" : "rgba(34,211,238,0.55)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------
   Timeline: PAST ── TODAY ── FORECAST ── +7 DAYS
   ------------------------------------------------------------------ */
export function PlanetTimeline({ timeline, index, onChange }) {
  const n = timeline.length || 15;
  const pct = (i) => `${(i / (n - 1)) * 100}%`;
  const onKey = (e) => {
    if (e.key === "ArrowRight") onChange(Math.min(n - 1, index + 1));
    if (e.key === "ArrowLeft") onChange(Math.max(0, index - 1));
  };
  const short = (d) => (d ? `${d.slice(5, 7)}-${d.slice(8)}` : "");
  const sel = timeline[index];
  return (
    <motion.div className="tl" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 3, duration: 0.9 }}>
      <div className="tl-chip mono">{sel ? weekday(sel.date, { weekday: "short", day: "2-digit", month: "short" }).toUpperCase() : "15 DAYS"}</div>
      <div
        className="tl-track"
        role="slider"
        tabIndex={0}
        aria-label="Day shown on the planet"
        aria-valuemin={-7}
        aria-valuemax={7}
        aria-valuenow={index - 7}
        aria-valuetext={sel ? `${sel.kind} ${sel.date}` : ""}
        onKeyDown={onKey}
      >
        <span className="tl-line tl-past" style={{ width: pct(7) }} />
        <span className="tl-line tl-future" style={{ left: pct(7), width: `calc(100% - ${pct(7)})` }} />
        {Array.from({ length: n }, (_, i) => (
          <button
            key={i}
            type="button"
            className={`tl-tick${i === 7 ? " today" : ""}${i === index ? " on" : ""}`}
            style={{ left: pct(i) }}
            onClick={() => onChange(i)}
            aria-label={timeline[i] ? `${timeline[i].kind} ${timeline[i].date}` : `day ${i - 7}`}
          />
        ))}
        <motion.span className="tl-handle" animate={{ left: pct(index) }} transition={{ type: "spring", stiffness: 220, damping: 28 }} />
      </div>
      <div className="tl-labels mono">
        <span style={{ left: 0 }}>PAST · {short(timeline[0]?.date)}</span>
        <span style={{ left: pct(7) }} className="mid">TODAY</span>
        <span style={{ left: pct(10) }} className="mid">FORECAST</span>
        <span style={{ right: 0 }}>+7 D · {short(timeline[n - 1]?.date)}</span>
      </div>
    </motion.div>
  );
}
