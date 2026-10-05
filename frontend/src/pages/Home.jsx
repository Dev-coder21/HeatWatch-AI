import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle } from "lucide-react";

import PlanetView from "../components/globe/PlanetView";
import CommandBar from "../components/home/CommandBar";
import { ProgressionPanel, RankingTable, PlanetTimeline } from "../components/home/HeroPanels";
import NationalSections from "../components/home/NationalSections";
import { Legend, Metric, Panel } from "../components/ui";
import { api, departureColor, fmt, riskColor, signed, tempColor, weekday } from "../api";
import { openCity, useData } from "../state";
import "./Home.css";

const TEMP_BINS = [
  { label: "< 26", v: 24 },
  { label: "26 – 30", v: 28 },
  { label: "30 – 33", v: 31.5 },
  { label: "33 – 36", v: 34.5 },
  { label: "36 – 40", v: 38 },
  { label: "40 – 45", v: 42 },
  { label: "≥ 45 °C", v: 46 },
];
const DEP_BINS = [
  { label: "≤ −2", v: -2.5 },
  { label: "−2 – 0", v: -1 },
  { label: "0 – +2", v: 1 },
  { label: "+2 – +4.5", v: 3.2 },
  { label: "+4.5 – +6.5", v: 5.5 },
  { label: "≥ +6.5 °C", v: 7.5 },
];

const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);

/*
 * One row per featured city with a 15-day series:
 * 8 observed days (7 past + today) then 7 forecast days (day 1 = model prediction).
 */
function useCityRows(featured, details) {
  return useMemo(
    () =>
      (featured || []).map((p) => {
        const d = details[p.slug];
        const recent = (d?.recent_days || []).map((r) => ({
          date: r.date,
          tmax: r.tmax,
          normal: r.normal_tmax,
          departure: r.departure,
          flagged: r.imd_rule_severity !== "No Heatwave",
          kind: "observed",
        }));
        const ahead = (d?.outlook || []).map((o, i) => ({
          date: o.date,
          tmax: i === 0 ? d.prediction.predicted_tmax : o.forecast_tmax,
          normal: o.normal_tmax,
          departure: i === 0 ? d.prediction.departure : o.departure,
          flagged: Boolean(o.alert_label),
          alert: o.alert_label,
          confidence: o.confidence,
          kind: "forecast",
        }));
        return {
          slug: p.slug,
          name: p.name,
          state: p.state,
          lat: p.lat,
          lon: p.lon,
          terrain: p.terrain_type,
          risk: p.risk_score,
          category: p.risk_category,
          predicted: p.predicted_tmax,
          probability: p.heatwave_probability,
          severity: p.severity,
          detail: d,
          today: d?.today,
          recent: d?.recent_days || [],
          days: ahead,
          series: [...recent, ...ahead],
        };
      }),
    [featured, details],
  );
}

export default function Home() {
  const { featured, details, error } = useData();
  const rows = useCityRows(featured, details);
  const [metric, setMetric] = useState("temp");
  const [index, setIndex] = useState(8); // 0..7 observed, 8 = tomorrow, 14 = +7 days
  const [pickError, setPickError] = useState("");
  const [leaving, setLeaving] = useState(false);
  const [hover, setHover] = useState(null);
  const planetRef = useRef(null);

  const loaded = rows.filter((r) => r.series.length === 15);
  const timeline = loaded[0]?.series.map((s) => ({ date: s.date, kind: s.kind })) || [];
  const selected = timeline[index];

  useEffect(() => {
    if (!pickError) return undefined;
    const t = setTimeout(() => setPickError(""), 4500);
    return () => clearTimeout(t);
  }, [pickError]);

  const colorOf = useCallback((v) => (metric === "temp" ? tempColor(v) : departureColor(v)), [metric]);
  const heightOf = useCallback((v) => (metric === "temp" ? (v - 24) / 18 : (v + 1.5) / 8), [metric]);

  const planetCities = useMemo(
    () =>
      rows.map((r) => {
        const s = r.series[index];
        const fallback = index === 8 && metric === "temp" ? r.predicted : NaN;
        const value = s ? (metric === "temp" ? s.tmax : s.departure) : fallback;
        return {
          slug: r.slug,
          name: r.name,
          lat: r.lat,
          lon: r.lon,
          risk: r.risk,
          value,
          color: riskColor(r.risk ?? 0),
          label: Number.isFinite(value) ? (metric === "temp" ? `${fmt(value)} °C` : `${signed(value)} °C`) : "—",
        };
      }),
    [rows, index, metric],
  );

  // Planet -> region -> city.
  const goToCity = useCallback(
    async (place) => {
      if (leaving) return;
      setLeaving(true);
      await planetRef.current?.flyTo(place.lat, place.lon, place.slug);
      openCity(place);
    },
    [leaving],
  );
  const onPickCity = useCallback((c) => goToCity({ lat: c.lat, lon: c.lon, name: c.name, slug: c.slug }), [goToCity]);
  const onPickPoint = useCallback(
    (lat, lon) => {
      setPickError("");
      api
        .reverse(lat, lon)
        .then((place) => goToCity({ lat: place.lat, lon: place.lon, name: place.name }))
        .catch((e) => setPickError(e.message));
    },
    [goToCity],
  );

  // Headline numbers: tomorrow, with change from today.
  const peak = rows.length ? rows.reduce((a, b) => ((b.predicted ?? -99) > (a.predicted ?? -99) ? b : a)) : null;
  const todayPeak = loaded.length ? Math.max(...loaded.map((r) => r.today?.tmax ?? -99)) : NaN;
  const depTomorrow = mean(loaded.map((r) => r.days[0]?.departure).filter(Number.isFinite));
  const depToday = mean(loaded.map((r) => r.today?.departure).filter(Number.isFinite));
  const alertCities = loaded.filter((r) => r.days.some((d) => d.alert)).length;
  const maxProb = rows.length ? Math.max(...rows.map((r) => r.probability ?? 0)) : NaN;
  const meanRisk = mean(rows.map((r) => r.risk).filter(Number.isFinite));
  const flaggedNow = selected ? loaded.filter((r) => r.series[index]?.flagged).length : NaN;
  const ready = rows.length > 0 && loaded.length === rows.length;

  const bins = metric === "temp" ? TEMP_BINS.map((b) => ({ label: b.label, color: tempColor(b.v) })) : DEP_BINS.map((b) => ({ label: b.label, color: departureColor(b.v) }));
  const stage = leaving ? { opacity: 0, transition: { duration: 0.7 } } : { opacity: 1 };
  const status = !selected ? "Loading" : selected.kind === "observed" ? (index === 7 ? "Today · observed" : "Observed") : index === 8 ? "HeatWatch prediction" : "Forecast";

  return (
    <div className="global">
      <div className="global-stage">
        <PlanetView ref={planetRef} cities={planetCities} colorOf={colorOf} heightOf={heightOf} onPickCity={onPickCity} onPickPoint={onPickPoint} onHover={setHover} />
        <div className="global-shade" aria-hidden="true" />

        <motion.div className="global-ui" animate={stage}>
          <CommandBar metric={metric} onMetric={setMetric} />

          <aside className="g-left">
            <Panel title="Heat status · India" from="left" delay={2.2} className="g-panel">
              <Metric
                label="Peak predicted maximum · tomorrow"
                value={peak?.predicted}
                unit="°C"
                delta={Number.isFinite(todayPeak) && peak ? peak.predicted - todayPeak : undefined}
                deltaUnit="°"
                tone={peak ? tempColor(peak.predicted) : undefined}
              />
              <p className="g-where">{peak ? `${peak.name} · ${peak.state}` : "Acquiring live data…"}</p>
              <div className="g-metrics">
                <Metric size="md" label="Mean departure" value={depTomorrow} unit="°C" delta={Number.isFinite(depToday) ? depTomorrow - depToday : undefined} deltaUnit="°" />
                <Metric size="md" label="Heatwave alerts · 7 d" value={ready ? alertCities : NaN} decimals={0} unit={`/ ${rows.length}`} />
                <Metric size="md" label="Max heatwave prob." value={maxProb * 100} decimals={0} unit="%" />
                <Metric size="md" label="Mean risk index" value={meanRisk} decimals={0} unit="/ 100" />
              </div>
            </Panel>
            <Panel title="City heat-risk rankings" from="left" delay={2.4} className="g-panel g-rank" bodyClass="g-rank-body">
              <RankingTable rows={rows} onPick={onPickCity} hoverSlug={hover?.slug} />
            </Panel>
          </aside>

          <aside className="g-right">
            <ProgressionPanel rows={loaded} selectedDate={selected?.date} />
          </aside>

          <motion.div className="g-status" initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 2.8, duration: 0.8 }}>
            <span className="g-status-dot" />
            <span>{status}</span>
            <small className="mono">
              {selected ? weekday(selected.date, { weekday: "short", day: "2-digit", month: "short" }) : ""}
              {Number.isFinite(flaggedNow) ? ` · ${flaggedNow} cities meet IMD criteria` : ""}
            </small>
          </motion.div>

          <motion.div className="g-legend" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 3 }}>
            <span className="label">{metric === "temp" ? "Max temp · °C" : "Vs normal · °C"}</span>
            <Legend vertical items={bins} />
          </motion.div>

          <PlanetTimeline timeline={timeline} index={index} onChange={setIndex} />

          <p className="g-foot">
            Heat surface interpolated between {rows.length || "the"} featured cities (inverse-distance weighting) and drawn on the
            planet; city values are HeatWatch model output and Open-Meteo forecasts at ~10–25 km resolution. Decision support,
            not an official IMD warning.
          </p>

          <AnimatePresence>
            {(pickError || error) && (
              <motion.div className="g-error" role="alert" initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <AlertTriangle size={14} /> {pickError || `HeatWatch API unreachable: ${error}`}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      <NationalSections rows={loaded} dates={loaded[0]?.days.map((d) => d.date) || []} onPick={onPickCity} />
    </div>
  );
}
