import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

import { DataContext, parseHash, useNationalData } from "./state";
import Landing from "./pages/Landing";

// The globe, maps and charts load only after the landing page.
const Home = lazy(() => import("./pages/Home"));
const City = lazy(() => import("./pages/City"));

/* ------------------------------------------------------------------
   Page transitions
   ------------------------------------------------------------------ */

// Pages choreograph their own entrances; the shell only cross-fades.
const pageVariants = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: { duration: 0.6 } },
  exit: { opacity: 0, transition: { duration: 0.45 } },
};

export default function App() {
  const [route, setRoute] = useState(parseHash);
  const data = useNationalData();

  useEffect(() => {
    const onHash = () => {
      setRoute(parseHash());
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // Start loading national data as soon as the app opens, so Home is ready.
  useEffect(() => data.load(), [data.load]);

  const key = route.path === "/city" ? `city-${route.params.lat}-${route.params.lon}` : route.path;

  const page = useMemo(() => {
    if (route.path === "/india") return <Home />;
    if (route.path === "/city" && route.params.lat && route.params.lon) return <City params={route.params} />;
    return <Landing />;
  }, [route]);

  return (
    <DataContext.Provider value={data}>
      <AnimatePresence mode="wait">
        <motion.div key={key} variants={pageVariants} initial="initial" animate="animate" exit="exit">
          <Suspense fallback={<div style={{ height: "100vh", background: "var(--bg)" }} />}>{page}</Suspense>
        </motion.div>
      </AnimatePresence>
    </DataContext.Provider>
  );
}
