import { useEffect, useState } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { ChevronLeft, CloudSun } from "lucide-react";

import AerialScene from "../components/city/AerialScene";
import { CityCard, ForecastPanel, ProfilePanel, TrendPanel } from "../components/city/CityPanels";
import CitySections from "../components/city/CitySections";
import { CountUp } from "../components/ui";
import { api, bandColor, fmt, severityColor, signed, tempColor } from "../api";
import { navigate } from "../state";
import "./City.css";

function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now;
}

export default function City({ params }) {
  const lat = Number(params.lat);
  const lon = Number(params.lon);
  const [risk, setRisk] = useState(null);
  const [explain, setExplain] = useState(null);
  const [error, setError] = useState("");
  const [leaving, setLeaving] = useState(false);
  const now = useClock();

  useEffect(() => {
    let cancelled = false;
    api.risk(lat, lon).then((r) => !cancelled && setRisk(r)).catch((e) => !cancelled && setError(e.message));
    api.explain(lat, lon).then((x) => !cancelled && setExplain(x)).catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [lat, lon]);

  const name = params.name || risk?.location.name || "";
  const { scrollY } = useScroll();
  const sceneY = useTransform(scrollY, [0, 800], [0, 220]);
  const sceneOpacity = useTransform(scrollY, [0, 700], [1, 0.3]);

  function back() {
    setLeaving(true);
    setTimeout(() => navigate("/india"), 500);
  }

  const p = risk?.prediction;
  const base = 0.6; // panel entrance starts while the camera is still diving

  return (
    <div className="city">
      <section className="city-stage">
        <motion.div className="city-scene" style={{ y: sceneY, opacity: sceneOpacity }}>
          <AerialScene lat={lat} lon={lon} risk={risk} />
        </motion.div>

        <motion.div className="city-ui" animate={leaving ? { opacity: 0, scale: 0.98, filter: "blur(6px)" } : {}} transition={{ duration: 0.5 }}>
          {/* Curved title banner */}
          <motion.header className="banner" initial={{ opacity: 0, y: -30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 1, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}>
            <svg className="banner-arc" viewBox="0 0 1000 90" preserveAspectRatio="none" aria-hidden="true">
              <defs>
                <linearGradient id="arc-fill" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0" stopColor="#060b10" stopOpacity="0.95" />
                  <stop offset="1" stopColor="#0e1820" stopOpacity="0.75" />
                </linearGradient>
              </defs>
              <path d="M0 0 H1000 V48 Q500 104 0 48 Z" fill="url(#arc-fill)" />
              <path d="M0 48 Q500 104 1000 48" fill="none" stroke="rgba(125,211,252,0.55)" strokeWidth="1.2" />
            </svg>
            <button type="button" className="banner-nav" onClick={back}>
              <ChevronLeft size={15} /> India overview
            </button>
            <div className="banner-title">
              <span className="label">City heat intelligence</span>
              <h1>{name || "Locating…"}</h1>
            </div>
            <div className="banner-clock">
              <CloudSun size={20} color="#fbbf24" />
              <span className="mono">{now.toLocaleDateString("en-CA")}</span>
              <b className="mono">{now.toLocaleTimeString("en-GB")}</b>
            </div>
          </motion.header>

          {error && (
            <div className="city-error hw-panel" role="alert">
              <div className="hw-panel-body">
                <b>{error}</b>
                <button type="button" className="hw-btn" onClick={back}>Back to India</button>
              </div>
            </div>
          )}

          {!risk && !error && <p className="city-loading mono">Acquiring live weather · running models…</p>}

          {risk && (
            <>
              <div className="city-left">
                <ProfilePanel risk={risk} name={name} delay={base + 0.2} />
              </div>
              <div className="city-right">
                <ForecastPanel risk={risk} delay={base + 0.3} />
              </div>

              {/* Callout anchored to the assessed point */}
              <motion.div className="callout" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 2.6, duration: 0.8 }}>
                <div className="callout-box">
                  <span className="label">Tomorrow · predicted max</span>
                  <div className="callout-temp num" style={{ color: tempColor(p.predicted_tmax) }}>
                    <CountUp value={p.predicted_tmax} />
                    <small>°C</small>
                  </div>
                  <div className="callout-row">
                    <span className="delta" style={{ color: p.departure >= 0 ? "#f97316" : "#38bdf8" }}>{signed(p.departure)} ° vs normal</span>
                    <span className="delta" style={{ color: bandColor(p.risk_category) }}>risk {fmt(p.risk_score, 0)} · {p.risk_category}</span>
                  </div>
                  <div className="callout-row mono">
                    <span>HW prob {fmt(p.heatwave_probability * 100, 0)}%</span>
                    <span style={{ color: severityColor(p.severity) }}>{p.severity}</span>
                  </div>
                </div>
                <span className="callout-leader" aria-hidden="true" />
                <span className="callout-pin" aria-hidden="true" />
              </motion.div>

              <div className="city-bottom">
                <CityCard risk={risk} delay={base + 0.5} className="card-hero" />
                <TrendPanel risk={risk} delay={base + 0.6} />
              </div>
            </>
          )}
        </motion.div>
      </section>

      {risk && (
        <CitySections risk={risk} explain={explain} name={name}>
          <CityCard risk={risk} delay={0} className="card-scroll" />
        </CitySections>
      )}
    </div>
  );
}
