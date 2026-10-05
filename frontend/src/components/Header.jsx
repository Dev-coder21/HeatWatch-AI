import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search } from "lucide-react";

import { api } from "../api";
import { navigate, openCity } from "../state";
import "./Header.css";

function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now;
}

function SearchBox() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const boxRef = useRef(null);

  useEffect(() => {
    const q = query.trim();
    if (q.length < 2) return undefined;
    const timer = setTimeout(() => {
      api
        .search(q)
        .then((data) => {
          setResults(data.results);
          setActive(0);
          setOpen(true);
        })
        .catch(() => setResults([]));
    }, 250);
    return () => clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    const close = (e) => boxRef.current && !boxRef.current.contains(e.target) && setOpen(false);
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, []);

  const shown = query.trim().length >= 2 ? results : [];

  function pick(place) {
    setOpen(false);
    setQuery("");
    openCity({ lat: place.lat, lon: place.lon, name: place.name });
  }

  function onKey(e) {
    if (!open || !shown.length) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => (a + 1) % shown.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => (a - 1 + shown.length) % shown.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      pick(shown[active]);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div className="cmd-search" ref={boxRef}>
      <Search size={14} aria-hidden="true" />
      <input
        type="search"
        value={query}
        placeholder="Search any place in India"
        aria-label="Search any place in India"
        role="combobox"
        aria-expanded={open}
        aria-controls="cmd-search-list"
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => shown.length && setOpen(true)}
        onKeyDown={onKey}
      />
      <AnimatePresence>
        {open && shown.length > 0 && (
          <motion.ul
            id="cmd-search-list"
            role="listbox"
            className="cmd-search-list"
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
          >
            {shown.map((p, i) => (
              <li key={`${p.name}-${p.lat}-${p.lon}`} role="option" aria-selected={i === active}>
                <button type="button" className={i === active ? "active" : ""} onMouseEnter={() => setActive(i)} onClick={() => pick(p)}>
                  <span>{p.name}</span>
                  <small>{[p.district, p.state].filter(Boolean).join(" · ")}</small>
                </button>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function Header({ title, subtitle, children }) {
  const now = useClock();
  return (
    <motion.header
      className="cmd-bar"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
    >
      <button type="button" className="cmd-brand" onClick={() => navigate("/")} aria-label="HeatWatch AI — back to start">
        <span className="cmd-orb" aria-hidden="true" />
        <span className="cmd-brand-text">
          HeatWatch <b>AI</b>
        </span>
      </button>
      <span className="cmd-divider" aria-hidden="true" />
      <div className="cmd-title">
        <h1>{title}</h1>
        <p className="label">{subtitle}</p>
      </div>

      <div className="cmd-right">
        {children}
        <SearchBox />
        <div className="cmd-clock mono" aria-label="Current date and time">
          <span>{now.toLocaleDateString("en-CA")}</span>
          <b>{now.toLocaleTimeString("en-GB")}</b>
        </div>
      </div>
      <span className="cmd-rule" aria-hidden="true" />
    </motion.header>
  );
}
