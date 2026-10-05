import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { geoDistance } from "d3-geo";

import EarthScene from "./EarthScene";
import "./PlanetView.css";

/* Inverse-distance weighting from the featured cities onto the heat lattice. */
function interpolate(cells, cities) {
  const pts = cities.filter((c) => Number.isFinite(c.value));
  if (!pts.length) return null;
  return cells.map(([lon, lat]) => {
    let num = 0;
    let den = 0;
    for (const p of pts) {
      const d = geoDistance([lon, lat], [p.lon, p.lat]) + 1e-4;
      const w = 1 / (d * d * d);
      num += w * p.value;
      den += w;
    }
    return num / den;
  });
}

/**
 * Three.js Earth with HeatWatch data projected onto it.
 * Props: cities [{slug,name,lat,lon,risk,value,color,label}], colorOf(v), heightOf(v)
 * Ref: flyTo(lat, lon, slug) -> Promise, home()
 */
const PlanetView = forwardRef(function PlanetView({ cities, colorOf, heightOf, onPickCity, onPickPoint, onHover }, ref) {
  const hostRef = useRef(null);
  const labelRef = useRef(null);
  const sceneRef = useRef(null);
  const handlers = useRef({});
  handlers.current = { onPickCity, onPickPoint, onHover };

  useEffect(() => {
    const scene = new EarthScene(hostRef.current, labelRef.current, {
      onPickCity: (c) => handlers.current.onPickCity?.(c),
      onPickPoint: (lat, lon) => handlers.current.onPickPoint?.(lat, lon),
      onHover: (c) => handlers.current.onHover?.(c),
    });
    sceneRef.current = scene;
    return () => {
      scene.dispose();
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    scene.setCities(cities);
    scene.setField(interpolate(scene.cellCoordinates(), cities), colorOf, heightOf);
  }, [cities, colorOf, heightOf]);

  useImperativeHandle(ref, () => ({
    flyTo: (lat, lon, slug) => sceneRef.current?.flyTo(lat, lon, slug) ?? Promise.resolve(),
    home: () => sceneRef.current?.home(),
  }));

  return (
    <div className="planet" role="img" aria-label="Earth seen from orbit, centred on India, with HeatWatch heat data projected onto the surface. Drag to rotate, scroll to zoom, click a city or any point in India.">
      <div ref={hostRef} className="planet-host" />
      <div ref={labelRef} className="planet-labels" />
    </div>
  );
});

export default PlanetView;
