import { createContext, useCallback, useContext, useEffect, useState } from "react";

import { api } from "./api";

/* ------------------------------------------------------------------
   Routing: #/ (landing), #/india (home), #/city?lat=&lon=&name=
   ------------------------------------------------------------------ */

export function parseHash() {
  const raw = window.location.hash.replace(/^#/, "") || "/";
  const [path, query = ""] = raw.split("?");
  return { path, params: Object.fromEntries(new URLSearchParams(query)) };
}

export function navigate(path, params) {
  const query = params ? `?${new URLSearchParams(params)}` : "";
  window.location.hash = `${path}${query}`;
}

export function openCity({ lat, lon, name, slug }) {
  navigate("/city", { lat: Number(lat).toFixed(4), lon: Number(lon).toFixed(4), ...(name ? { name } : {}), ...(slug ? { slug } : {}) });
}

/* ------------------------------------------------------------------
   Shared data: featured places + each one's full assessment
   ------------------------------------------------------------------ */

export const DataContext = createContext(null);
export const useData = () => useContext(DataContext);

export function useNationalData() {
  const [featured, setFeatured] = useState(null);
  const [details, setDetails] = useState({});
  const [error, setError] = useState("");
  const [started, setStarted] = useState(false);

  const load = useCallback(() => setStarted(true), []);

  useEffect(() => {
    if (!started) return undefined;
    let cancelled = false;
    api
      .featured()
      .then((data) => {
        if (cancelled) return;
        const places = data.places.filter((p) => !p.error);
        setFeatured(places);
        // Full assessments (7-day outlook etc.); the server caches them per cell.
        places.forEach((p) =>
          api
            .risk(p.lat, p.lon)
            .then((r) => !cancelled && setDetails((prev) => ({ ...prev, [p.slug]: r })))
            .catch(() => {}),
        );
      })
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [started]);

  return { featured, details, error, load };
}

