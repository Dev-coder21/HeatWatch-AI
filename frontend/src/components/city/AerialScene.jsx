import { useEffect } from "react";
import { MapContainer, Rectangle, TileLayer, useMap } from "react-leaflet";
import { useReducedMotion } from "framer-motion";
import "leaflet/dist/leaflet.css";

import { bandColor } from "../../api";

export const IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
export const LABELS = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}";

/* Camera dive: start high over the region, fly down to the city. */
function Dive({ lat, lon, zoom }) {
  const map = useMap();
  const reduce = useReducedMotion();
  useEffect(() => {
    if (reduce) {
      map.setView([lat, lon], zoom, { animate: false });
      return undefined;
    }
    map.setView([lat, lon], 6, { animate: false });
    const t = setTimeout(() => map.flyTo([lat, lon], zoom, { duration: 3.2, easeLinearity: 0.15 }), 250);
    return () => clearTimeout(t);
  }, [map, lat, lon, zoom, reduce]);
  return null;
}

/* The 0.1-degree cell HeatWatch assessed (profile, weather and normals share it). */
export function gridBounds(key) {
  const [la, lo] = key.split(",").map(Number);
  return [
    [la - 0.05, lo - 0.05],
    [la + 0.05, lo + 0.05],
  ];
}

export default function AerialScene({ lat, lon, risk }) {
  const color = risk ? bandColor(risk.prediction.risk_category) : "#38bdf8";
  return (
    <div className="aerial">
      <div className="aerial-tilt">
        <MapContainer
          center={[lat, lon]}
          zoom={6}
          zoomControl={false}
          dragging={false}
          scrollWheelZoom={false}
          doubleClickZoom={false}
          touchZoom={false}
          boxZoom={false}
          keyboard={false}
          attributionControl={false}
          className="aerial-map"
        >
          <TileLayer url={IMAGERY} maxZoom={18} />
          <TileLayer url={LABELS} maxZoom={18} opacity={0.6} />
          <Dive lat={lat} lon={lon} zoom={12} />
          {risk && (
            <Rectangle
              bounds={gridBounds(risk.location.grid_key)}
              pathOptions={{ color, weight: 2, fillColor: color, fillOpacity: 0.12, dashArray: "6 6", className: "grid-cell" }}
            />
          )}
        </MapContainer>
        <div className="aerial-grade" />
        <div className="aerial-heat" style={{ "--heat": color }} />
        <div className="aerial-grid" />
      </div>
      <div className="aerial-fog" />
      <span className="aerial-credit">Imagery © Esri, Maxar, Earthstar Geographics</span>
    </div>
  );
}
