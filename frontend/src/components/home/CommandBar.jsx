import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

import { api } from "../../api";
import { navigate, openCity } from "../../state";

/* Minimal top bar in the reference's style: brand · system title ........ command items */
export default function CommandBar({ metric, onMetric }) {
  const [searchOpen, setSearchOpen] = useState(false);
  return (
    <motion.header className="cmd" initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1.6, duration: 0.9 }}>
      <button type="button" className="cmd-logo" onClick={() => navigate("/")} aria-label="HeatWatch AI — back to start">
        <span className="cmd-logo-orb" aria-hidden="true" />
        <span className="cmd-logo-text">
          HeatWatch<b>AI</b>
        </span>
      </button>
      <span className="cmd-sep" aria-hidden="true" />
      <div className="cmd-system">
        <strong>Heat Intelligence System</strong>
        <span>live forecast &amp; IMD heatwave criteria</span>
      </div>

      <nav className="cmd-items" aria-label="View controls">
        <button type="button" className={`cmd-item${metric === "temp" ? " on" : ""}`} aria-pressed={metric === "temp"} onClick={() => onMetric("temp")}>
          <span className="ic ic-ring" /> Max temperature
        </button>
        <button type="button" className={`cmd-item${metric === "departure" ? " on" : ""}`} aria-pressed={metric === "departure"} onClick={() => onMetric("departure")}>
          <span className="ic ic-half" /> Departure mode
        </button>
        <div className="cmd-find">
          <button type="button" className={`cmd-item${searchOpen ? " on" : ""}`} aria-expanded={searchOpen} onClick={() => setSearchOpen((o) => !o)}>
            <span className="ic ic-plus" /> Find a place
          </button>
          <AnimatePresence>{searchOpen && <FindBox onClose={() => setSearchOpen(false)} />}</AnimatePresence>
        </div>
      </nav>
    </motion.header>
  );
}

function FindBox({ onClose }) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState([]);
  const [active, setActive] = useState(0);
  const boxRef = useRef(null);

  useEffect(() => {
    const term = q.trim();
    if (term.length < 2) return undefined;
    const t = setTimeout(() => {
      api
        .search(term)
        .then((d) => {
          setResults(d.results);
          setActive(0);
        })
        .catch(() => setResults([]));
    }, 250);
    return () => clearTimeout(t);
  }, [q]);

  useEffect(() => {
    const close = (e) => boxRef.current && !boxRef.current.parentElement.contains(e.target) && onClose();
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [onClose]);

  const shown = q.trim().length >= 2 ? results : [];
  const pick = (p) => openCity({ lat: p.lat, lon: p.lon, name: p.name });

  return (
    <motion.div ref={boxRef} className="find-box" initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}>
      <input
        autoFocus
        type="search"
        value={q}
        placeholder="Any town or city in India"
        aria-label="Search any place in India"
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Escape") onClose();
          if (!shown.length) return;
          if (e.key === "ArrowDown") setActive((a) => (a + 1) % shown.length);
          if (e.key === "ArrowUp") setActive((a) => (a - 1 + shown.length) % shown.length);
          if (e.key === "Enter") pick(shown[active]);
        }}
      />
      {shown.length > 0 && (
        <ul role="listbox">
          {shown.map((p, i) => (
            <li key={`${p.name}-${p.lat}-${p.lon}`} role="option" aria-selected={i === active}>
              <button type="button" className={i === active ? "active" : ""} onMouseEnter={() => setActive(i)} onClick={() => pick(p)}>
                {p.name}
                <small>{[p.district, p.state].filter(Boolean).join(" · ")}</small>
              </button>
            </li>
          ))}
        </ul>
      )}
    </motion.div>
  );
}
