import { useEffect, useRef, useState } from "react";
import { animate, motion, useInView, useReducedMotion } from "framer-motion";

import { RISK_BANDS, bandColor } from "../api";

const EASE = [0.22, 1, 0.36, 1];

/* Glass panel with a title bar. `delay` staggers the entrance. */
export function Panel({ title, aside, children, className = "", bodyClass = "", delay = 0, style, from = "up" }) {
  const reduce = useReducedMotion();
  const offset = { up: { y: 24 }, left: { x: -36 }, right: { x: 36 }, none: {} }[from];
  return (
    <motion.section
      className={`hw-panel ${className}`}
      style={style}
      initial={reduce ? false : { opacity: 0, ...offset, clipPath: "inset(0 0 100% 0)" }}
      animate={{ opacity: 1, x: 0, y: 0, clipPath: "inset(0 0 0% 0)" }}
      transition={{ duration: 0.9, delay, ease: EASE }}
    >
      {title && (
        <header className="hw-panel-head">
          <h2 className="hw-panel-title">{title}</h2>
          {aside}
        </header>
      )}
      <div className={`hw-panel-body ${bodyClass}`}>{children}</div>
    </motion.section>
  );
}

/* Scroll-triggered reveal. */
export function Reveal({ children, delay = 0, className = "", ...rest }) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial={reduce ? false : { opacity: 0, y: 36 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.15 }}
      transition={{ duration: 0.9, delay, ease: EASE }}
      {...rest}
    >
      {children}
    </motion.div>
  );
}

/* Number that counts up when shown. */
export function CountUp({ value, decimals = 1, prefix = "", suffix = "" }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(0);
  const target = Number(value);
  useEffect(() => {
    if (!inView || !Number.isFinite(target)) return undefined;
    if (reduce) {
      setShown(target);
      return undefined;
    }
    const controls = animate(0, target, { duration: 1.6, ease: EASE, onUpdate: setShown });
    return () => controls.stop();
  }, [inView, target, reduce]);
  if (!Number.isFinite(target)) return <span ref={ref}>—</span>;
  return (
    <span ref={ref}>
      {prefix}
      {shown.toFixed(decimals)}
      {suffix}
    </span>
  );
}

/* Big number with label and optional delta chip, as in the reference metrics. */
export function Metric({ label, value, decimals = 1, unit, delta, deltaUnit = "", deltaGood, size = "lg", tone }) {
  const up = Number(delta) > 0;
  const deltaColor = deltaGood === undefined ? (up ? "var(--heat-3)" : "var(--sky-2)") : deltaGood ? "var(--cyan-2)" : "var(--heat-3)";
  return (
    <div className={`metric metric-${size}`}>
      <span className="label">{label}</span>
      <div className="metric-row">
        <span className="metric-value num" style={{ color: tone }}>
          <CountUp value={value} decimals={decimals} />
          {unit && <small>{unit}</small>}
        </span>
        {Number.isFinite(Number(delta)) && (
          <span className="delta" style={{ color: deltaColor }}>
            {Math.abs(Number(delta)).toFixed(1)}
            {deltaUnit} {up ? "↑" : "↓"}
          </span>
        )}
      </div>
    </div>
  );
}

/* 0-100 risk ring with the five bands. */
export function RiskRing({ score = 0, category = "Low", size = 140 }) {
  const reduce = useReducedMotion();
  const r = size / 2 - 12;
  const c = 2 * Math.PI * r;
  const outer = r + 8;
  const color = bandColor(category);
  return (
    <div style={{ width: size, height: size, position: "relative", flex: "none" }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Risk ${score} of 100, ${category}`}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(125,211,252,0.08)" strokeWidth="7" />
        {RISK_BANDS.map((b, i) => {
          const start = b.min / 100;
          const end = (RISK_BANDS[i + 1]?.min ?? 100) / 100;
          const len = 2 * Math.PI * outer;
          return (
            <circle
              key={b.name}
              cx={size / 2}
              cy={size / 2}
              r={outer}
              fill="none"
              stroke={b.color}
              strokeOpacity="0.55"
              strokeWidth="1.5"
              strokeDasharray={`${(end - start) * len - 3} ${len}`}
              strokeDashoffset={-start * len}
              transform={`rotate(-90 ${size / 2} ${size / 2})`}
            />
          );
        })}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="7"
          strokeDasharray={c}
          initial={{ strokeDashoffset: reduce ? c * (1 - Math.min(100, score) / 100) : c }}
          animate={{ strokeDashoffset: c * (1 - Math.min(100, score) / 100) }}
          transition={{ duration: 1.8, ease: EASE, delay: 0.4 }}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ filter: `drop-shadow(0 0 6px ${color})` }}
        />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", textAlign: "center" }}>
        <div>
          <div className="num" style={{ fontSize: size * 0.27, fontWeight: 500, lineHeight: 1 }}>
            <CountUp value={score} decimals={0} />
          </div>
          <div className="label" style={{ color, fontSize: 10 }}>{category}</div>
        </div>
      </div>
    </div>
  );
}

/* Swatch legend. */
export function Legend({ items, vertical = false }) {
  return (
    <div className={`legend${vertical ? " legend-v" : ""}`}>
      {items.map((it) => (
        <span key={it.label}>
          <i style={{ background: it.color, borderRadius: it.round ? "50%" : 1 }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

/* Recharts tooltip in the panel style. */
export function ChartTip({ active, payload, label, unit = "°C", labelFormat }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tip">
      <div className="label">{labelFormat ? labelFormat(label) : label}</div>
      {payload
        .filter((p) => p.value !== null && p.value !== undefined && p.name)
        .map((p) => (
          <div key={p.dataKey} className="chart-tip-row">
            <span style={{ color: p.color || p.stroke || p.fill }}>{p.name}</span>
            <b className="mono">
              {Array.isArray(p.value) ? p.value.map((v) => Number(v).toFixed(1)).join("–") : Number(p.value).toFixed(1)}
              {unit}
            </b>
          </div>
        ))}
    </div>
  );
}

export const AXIS = { stroke: "#6b8295", fontSize: 10, tickLine: false, axisLine: false, fontFamily: "JetBrains Mono" };
export const GRID = "rgba(125,211,252,0.07)";
