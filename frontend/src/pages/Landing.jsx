import { useEffect, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";

import { GlobeCollection } from "../shaders/globe/GlobeCollection";
import { navigate, useData } from "../state";
import { fmt } from "../api";
import "./Landing.css";

const EASE = [0.22, 1, 0.36, 1];

export default function Landing() {
  const { featured } = useData();
  const [leaving, setLeaving] = useState(false);
  const reduce = useReducedMotion();
  const leavingRef = useRef(false);

  const peak = featured?.length ? featured.reduce((a, b) => (b.predicted_tmax > a.predicted_tmax ? b : a)) : null;

  function enter() {
    if (leavingRef.current) return;
    leavingRef.current = true;
    setLeaving(true);
    // Orb swells into a flash, then the planet view takes over.
    setTimeout(() => navigate("/india"), reduce ? 0 : 1250);
  }

  // Scrolling down also enters.
  useEffect(() => {
    const onWheel = (e) => e.deltaY > 30 && enter();
    window.addEventListener("wheel", onWheel, { passive: true });
    return () => window.removeEventListener("wheel", onWheel);
  });

  return (
    <main className="landing">
      <motion.div
        className="shader-frame"
        initial={reduce ? false : { opacity: 0, scale: 0.86 }}
        animate={leaving ? { scale: 2.6, opacity: 0 } : { opacity: 1, scale: 1 }}
        transition={leaving ? { duration: 1.25, ease: [0.7, 0, 0.84, 0] } : { duration: 2.4, ease: EASE }}
      >
        <GlobeCollection
          variant="energy-orb"
          speed={1.00}
          scale={1.00}
          smokeScale={1.00}
          smokeStrength={1.00}
          smokeSpeed={1.00}
          hue={0}
          saturation={1.00}
          glow={1.00}
          starDensity={1.00}
          starSpeed={1.00}
          starSize={1.00}
          brightness={1.00}
          opacity={1.00}
        />
      </motion.div>

      <motion.div
        className="landing-flash"
        initial={{ opacity: 0 }}
        animate={{ opacity: leaving ? 1 : 0 }}
        transition={{ duration: 0.9, delay: leaving ? 0.45 : 0 }}
        aria-hidden="true"
      />

      <motion.div
        className="landing-copy"
        animate={leaving ? { opacity: 0, y: 24, filter: "blur(6px)" } : {}}
        transition={{ duration: 0.5 }}
      >
        <h1 className="landing-title" aria-label="HeatWatch AI">
          {"HEATWATCH".split("").map((ch, i) => (
            <motion.span
              key={i}
              initial={reduce ? false : { opacity: 0, y: 18, filter: "blur(8px)" }}
              animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
              transition={{ delay: 1.0 + i * 0.07, duration: 0.9, ease: EASE }}
            >
              {ch}
            </motion.span>
          ))}
          <motion.span
            className="landing-ai"
            initial={reduce ? false : { opacity: 0, filter: "blur(10px)" }}
            animate={{ opacity: 1, filter: "blur(0px)" }}
            transition={{ delay: 1.8, duration: 1.1 }}
          >
            AI
          </motion.span>
        </h1>

        <motion.p
          className="landing-sub"
          initial={reduce ? false : { opacity: 0, letterSpacing: "0.9em" }}
          animate={{ opacity: 1, letterSpacing: "0.42em" }}
          transition={{ delay: 2.1, duration: 1.4, ease: EASE }}
        >
          Heatwave intelligence
        </motion.p>

        <motion.button
          type="button"
          className="landing-enter"
          onClick={enter}
          initial={reduce ? false : { opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 2.6, duration: 0.8 }}
          autoFocus
        >
          <span>Enter</span>
          <i aria-hidden="true" />
        </motion.button>

        <motion.p className="landing-ticker" initial={{ opacity: 0 }} animate={{ opacity: peak ? 1 : 0 }} transition={{ delay: 3, duration: 1 }}>
          {peak && (
            <>
              <span className="tick-dot" /> LIVE · {featured.length} cities monitored · tomorrow’s peak {fmt(peak.predicted_tmax)} °C in {peak.name}
            </>
          )}
        </motion.p>
      </motion.div>
    </main>
  );
}
