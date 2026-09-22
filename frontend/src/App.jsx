import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Bell,
  BrainCircuit,
  ChevronRight,
  CircleDot,
  CloudSun,
  Database,
  Gauge,
  Globe2,
  Map,
  Menu,
  ShieldAlert,
  Thermometer,
  TrendingUp,
  X,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import {
  BarChart,
  Bar,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  CircleMarker,
  MapContainer,
  Popup,
  TileLayer,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";


const API_BASE = "http://localhost:8000";


function MapViewController({ locations }) {
  const map = useMap();

  useEffect(() => {
    const timer = setTimeout(() => {
      map.invalidateSize();

      const validLocations = locations.filter(
        (location) =>
          Number.isFinite(Number(location.latitude)) &&
          Number.isFinite(Number(location.longitude))
      );

      if (validLocations.length === 1) {
        map.setView(
          [
            Number(validLocations[0].latitude),
            Number(validLocations[0].longitude),
          ],
          6
        );
      }

      if (validLocations.length > 1) {
        const bounds = validLocations.map((location) => [
          Number(location.latitude),
          Number(location.longitude),
        ]);

        map.fitBounds(bounds, {
          padding: [40, 40],
          maxZoom: 6,
        });
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [map, locations]);

  return null;
}


function riskLabel(score) {
  const value = Number(score || 0);

  if (value >= 80) return "Extreme";
  if (value >= 60) return "Very High";
  if (value >= 40) return "High";
  if (value >= 20) return "Moderate";
  return "Low";
}


function riskClass(score) {
  return riskLabel(score)
    .toLowerCase()
    .replace(" ", "-");
}


function formatNumber(value, decimals = 1) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "—";
  }

  return number.toFixed(decimals);
}


function formatPercent(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "—";
  }

  const normalized = number > 1 ? number / 100 : number;
  return `${(normalized * 100).toFixed(1)}%`;
}

function firstFiniteNumber(...values) {
  for (const value of values) {
    const number = Number(value);
    if (Number.isFinite(number)) {
      return number;
    }
  }

  return null;
}

function getPredictedTmax(location) {
  if (!location) return null;

  return firstFiniteNumber(
    location.predicted_tmax,
    location.predicted_temperature,
    location.predicted_tmax_c,
    location.predicted_temperature_c,
    location.forecast_tmax,
    location.forecast_temperature,
    location.tmax_forecast,
    location.prediction?.predicted_tmax,
    location.prediction?.predicted_temperature,
  );
}

function getHumidity(location) {
  if (!location) return null;

  const rawHumidity = firstFiniteNumber(
    location.humidity,
    location.relative_humidity,
    location.humidity_percent,
    location.relative_humidity_percent,
  );

  // A zero humidity value in the API is treated as missing when the
  // deterministic risk engine contains a valid humidity component.
  // The risk engine normalizes humidity as: (humidity - 40) / 40 * 100.
  // Inverting that transformation gives the raw humidity used by the
  // score, while avoiding a misleading 0% display.
  if (rawHumidity !== null && rawHumidity > 0) {
    return rawHumidity;
  }

  const humidityComponent = firstFiniteNumber(
    location.humidity_component,
  );

  if (humidityComponent !== null) {
    return Math.max(0, Math.min(100, 40 + humidityComponent * 0.4));
  }

  return rawHumidity;
}

function getHeatwaveProbability(location) {
  if (!location) return 0;

  const value = firstFiniteNumber(
    location.heatwave_probability,
    location.heatwave_prob,
    location.heatwave_probability_pct,
    location.prediction?.heatwave_probability,
  );

  if (value === null) return 0;
  return value > 1 ? value / 100 : value;
}

function getRiskScore(location) {
  if (!location) return 0;

  return firstFiniteNumber(
    location.risk_score,
    location.heat_risk_score,
    location.prediction?.risk_score,
  ) ?? 0;
}

function getForecastDate(location, explanation) {
  return (
    location?.forecast_date ||
    location?.prediction_date ||
    explanation?.forecast_date ||
    explanation?.prediction_date ||
    null
  );
}


function App() {
  const [locations, setLocations] = useState([]);
  const [selectedLocation, setSelectedLocation] = useState(null);

  const [explanationData, setExplanationData] = useState(null);

  const [loading, setLoading] = useState(true);
  const [explanationLoading, setExplanationLoading] =
    useState(false);

  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);


  /* ==========================================================
     LOAD PREDICTIONS
     ========================================================== */

  useEffect(() => {
    async function loadPredictions() {
      try {
        setLoading(true);
        setError("");

        const response = await fetch(
          `${API_BASE}/api/predictions`
        );

        if (!response.ok) {
          throw new Error(
            "Unable to load prediction data."
          );
        }

        const data = await response.json();

        const predictionRows =
          data.predictions || [];

        setLocations(predictionRows);

        if (predictionRows.length > 0) {
          setSelectedLocation(
            predictionRows[0]
          );
        }
      } catch (err) {
        console.error(err);

        setError(
          "Unable to connect to the HeatWatch AI backend."
        );
      } finally {
        setLoading(false);
      }
    }

    loadPredictions();
  }, []);


  /* ==========================================================
     LOAD SHAP EXPLANATION
     ========================================================== */

  useEffect(() => {
    async function loadExplanation() {
      if (!selectedLocation?.location_id) {
        setExplanationData(null);
        return;
      }

      try {
        setExplanationLoading(true);

        const response = await fetch(
          `${API_BASE}/api/explanations/${selectedLocation.location_id}`
        );

        if (!response.ok) {
          throw new Error(
            "Unable to load SHAP explanation."
          );
        }

        const data = await response.json();

        setExplanationData(data);
      } catch (err) {
        console.error(err);
        setExplanationData(null);
      } finally {
        setExplanationLoading(false);
      }
    }

    loadExplanation();
  }, [selectedLocation]);


  /* ==========================================================
     GLOBAL METRICS
     ========================================================== */

  const metrics = useMemo(() => {
    if (!locations.length) {
      return {
        systemRisk: 0,
        forecastTemp: 0,
        heatwaveProbability: 0,
        monitoredLocations: 0,
      };
    }

    const riskValues = locations.map(getRiskScore);
    const temperatureValues = locations
      .map(getPredictedTmax)
      .filter((value) => value !== null);
    const probabilityValues = locations.map(getHeatwaveProbability);

    return {
      // Headline risk is the highest risk among monitored locations.
      systemRisk: Math.max(...riskValues),
      forecastTemp: temperatureValues.length
        ? Math.max(...temperatureValues)
        : 0,
      heatwaveProbability: probabilityValues.length
        ? Math.max(...probabilityValues)
        : 0,
      monitoredLocations: locations.length,
    };
  }, [locations]);


  /* ==========================================================
     RISK RANKING
     ========================================================== */

  const rankedLocations = useMemo(() => {
    return [...locations].sort(
      (a, b) =>
        getRiskScore(b) -
        getRiskScore(a)
    );
  }, [locations]);

  const priorityLocation =
    rankedLocations[0] || selectedLocation || null;


  /* ==========================================================
     FORECAST DATA
     ========================================================== */

  const temperatureChartData = useMemo(() => {
    return locations.map((location) => ({
      city:
        location.city ||
        location.location_id,
      temperature: getPredictedTmax(location) ?? 0,
    }));
  }, [locations]);


  const probabilityChartData = useMemo(() => {
    return locations.map((location) => ({
      city:
        location.city ||
        location.location_id,
      probability:
        getHeatwaveProbability(location) * 100,
    }));
  }, [locations]);


  const riskChartData = useMemo(() => {
    return rankedLocations.map((location) => ({
      city:
        location.city ||
        location.location_id,
      risk: Number(
        location.risk_score || 0
      ),
    }));
  }, [rankedLocations]);

  const hottestLocation = useMemo(() => {
    return [...locations]
      .filter((location) => getPredictedTmax(location) !== null)
      .sort(
        (a, b) =>
          getPredictedTmax(b) -
          getPredictedTmax(a)
      )[0] || null;
  }, [locations]);

  const highestProbabilityLocation = useMemo(() => {
    return [...locations].sort(
      (a, b) =>
        getHeatwaveProbability(b) -
        getHeatwaveProbability(a)
    )[0] || null;
  }, [locations]);


  /* ==========================================================
     SHAP DATA
     ========================================================== */

  const shapFeatures = useMemo(() => {
    if (!explanationData?.features) {
      return [];
    }

    return [...explanationData.features]
      .sort(
        (a, b) =>
          Math.abs(
            Number(b.shap_value || 0)
          ) -
          Math.abs(
            Number(a.shap_value || 0)
          )
      )
      .slice(0, 10);
  }, [explanationData]);


  const strongestPositiveShap = useMemo(() => {
    const positive = shapFeatures.filter(
      (feature) =>
        Number(feature.shap_value || 0) > 0
    );

    if (!positive.length) {
      return null;
    }

    return positive.reduce(
      (strongest, current) =>
        Number(current.shap_value) >
        Number(strongest.shap_value)
          ? current
          : strongest
    );
  }, [shapFeatures]);


  const strongestNegativeShap = useMemo(() => {
    const negative = shapFeatures.filter(
      (feature) =>
        Number(feature.shap_value || 0) < 0
    );

    if (!negative.length) {
      return null;
    }

    return negative.reduce(
      (strongest, current) =>
        Number(current.shap_value) <
        Number(strongest.shap_value)
          ? current
          : strongest
    );
  }, [shapFeatures]);


  /* ==========================================================
     RISK ENGINE COMPONENTS
     ========================================================== */

  const riskComponents = useMemo(() => {
    if (!selectedLocation) {
      return [];
    }

    return [
      {
        name: "Temperature",
        value: Number(
          selectedLocation.temperature_component || 0
        ),
        weight: "40%",
      },
      {
        name: "Heatwave Probability",
        value: Number(
          selectedLocation.heatwave_probability_component || 0
        ),
        weight: "25%",
      },
      {
        name: "Temperature Anomaly",
        value: Number(
          selectedLocation.anomaly_component || 0
        ),
        weight: "20%",
      },
      {
        name: "Persistence",
        value: Number(
          selectedLocation.persistence_component || 0
        ),
        weight: "10%",
      },
      {
        name: "Humidity",
        value: Number(
          selectedLocation.humidity_component || 0
        ),
        weight: "5%",
      },
    ];
  }, [selectedLocation]);


  /* ==========================================================
     NAVIGATION
     ========================================================== */

  function scrollToSection(id) {
    const element =
      document.getElementById(id);

    if (element) {
      element.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }

    setSidebarOpen(false);
  }


  /* ==========================================================
     LOADING
     ========================================================== */

  if (loading) {
    return (
      <div className="app-loading">

        <div className="loading-orb">
          <Activity size={28} />
        </div>

        <h2>
          HeatWatch AI
        </h2>

        <p>
          Loading heat intelligence system...
        </p>

      </div>
    );
  }


  return (
    <div className="app-shell">

      {/* ======================================================
          SIDEBAR
          ====================================================== */}

      <aside
        className={`sidebar ${
          sidebarOpen
            ? "sidebar-open"
            : ""
        }`}
      >

        <div className="sidebar-brand">

          <div className="brand-mark">
            <Thermometer size={22} />
          </div>

          <div>
            <div className="brand-name">
              HeatWatch
            </div>

            <div className="brand-ai">
              AI
            </div>
          </div>

        </div>


        <div className="sidebar-system">

          <span className="status-dot" />

          SYSTEM OPERATIONAL

        </div>


        <nav className="sidebar-nav">

          <button
            className="nav-item active"
            onClick={() =>
              scrollToSection("overview")
            }
          >
            <Gauge size={18} />
            <span>Overview</span>
          </button>


          <button
            className="nav-item"
            onClick={() =>
              scrollToSection("risk-map")
            }
          >
            <Map size={18} />
            <span>Heat Risk Map</span>
          </button>


          <button
            className="nav-item"
            onClick={() =>
              scrollToSection("locations")
            }
          >
            <Globe2 size={18} />
            <span>Locations</span>
          </button>


          <button
            className="nav-item"
            onClick={() =>
              scrollToSection("forecast")
            }
          >
            <TrendingUp size={18} />
            <span>Forecast</span>
          </button>


          <button
            className="nav-item"
            onClick={() =>
              scrollToSection("explain")
            }
          >
            <BrainCircuit size={18} />
            <span>Explainable AI</span>
          </button>

        </nav>


        <div className="sidebar-bottom">

          <div className="sidebar-model">

            <div className="sidebar-model-icon">
              <BrainCircuit size={17} />
            </div>

            <div>

              <div className="sidebar-model-label">
                MODEL STATUS
              </div>

              <div className="sidebar-model-value">
                Random Forest
              </div>

            </div>

          </div>

        </div>

      </aside>


      {/* ======================================================
          MOBILE OVERLAY
          ====================================================== */}

      <AnimatePresence>

        {sidebarOpen && (

          <motion.div
            className="sidebar-overlay"
            initial={{
              opacity: 0,
            }}
            animate={{
              opacity: 1,
            }}
            exit={{
              opacity: 0,
            }}
            onClick={() =>
              setSidebarOpen(false)
            }
          />

        )}

      </AnimatePresence>


      {/* ======================================================
          MAIN CONTENT
          ====================================================== */}

      <main className="main-content">

        {/* ====================================================
            TOP BAR
            ==================================================== */}

        <header className="topbar">

          <button
            className="mobile-menu"
            onClick={() =>
              setSidebarOpen(
                !sidebarOpen
              )
            }
          >

            {sidebarOpen ? (
              <X size={21} />
            ) : (
              <Menu size={21} />
            )}

          </button>


          <div className="topbar-left">

            <div className="topbar-title">
              Heat Intelligence
            </div>

            <div className="topbar-subtitle">
              Hyperlocal early warning system
            </div>

          </div>


          <div className="topbar-right">

            <div className="live-status">

              <span className="live-dot" />

              FORECAST SYSTEM ACTIVE

            </div>


            <div className="model-badge">

              <BrainCircuit size={15} />

              MODEL ACTIVE

            </div>


            <button className="notification-button">
              <Bell size={18} />
            </button>

          </div>

        </header>


        {error && (

          <div className="error-banner">

            <ShieldAlert size={18} />

            {error}

          </div>

        )}


        {/* ====================================================
            OVERVIEW
            ==================================================== */}

        <section
          id="overview"
          className="dashboard-section"
        >

          <div className="hero-heading">

            <div>

              <div className="eyebrow">
                AI-POWERED CLIMATE INTELLIGENCE
              </div>

              <h1>
                Heat Risk Command Center
              </h1>

              <p>
                Predictive heat monitoring,
                risk scoring and explainable
                early warning intelligence.
              </p>

            </div>


            <div className="hero-date">

              <CloudSun size={19} />

              <div>

                <span>
                  FORECAST HORIZON
                </span>

                <strong>
                  Next Day
                </strong>

              </div>

            </div>

          </div>


          {/* METRICS */}

          <div className="metric-grid">

            <motion.div
              className="metric-card"
              whileHover={{ y: -4 }}
            >

              <div className="metric-card-top">

                <span>
                  SYSTEM RISK
                </span>

                <Gauge size={18} />

              </div>

              <div className="metric-value">

                {formatNumber(
                  metrics.systemRisk,
                  1
                )}

              </div>

              <div className="metric-foot">
                / 100 composite score
              </div>

            </motion.div>


            <motion.div
              className="metric-card"
              whileHover={{ y: -4 }}
            >

              <div className="metric-card-top">

                <span>
                  PEAK FORECAST
                </span>

                <Thermometer size={18} />

              </div>

              <div className="metric-value">

                {formatNumber(
                  metrics.forecastTemp,
                  1
                )}

                <small>
                  °C
                </small>

              </div>

              <div className="metric-foot">
                highest predicted Tmax
              </div>

            </motion.div>


            <motion.div
              className="metric-card"
              whileHover={{ y: -4 }}
            >

              <div className="metric-card-top">

                <span>
                  HEATWAVE PROBABILITY
                </span>

                <ShieldAlert size={18} />

              </div>

              <div className="metric-value">

                {formatPercent(
                  metrics.heatwaveProbability
                )}

              </div>

              <div className="metric-foot">
                highest location probability
              </div>

            </motion.div>


            <motion.div
              className="metric-card"
              whileHover={{ y: -4 }}
            >

              <div className="metric-card-top">

                <span>
                  MONITORED LOCATIONS
                </span>

                <Database size={18} />

              </div>

              <div className="metric-value">
                {metrics.monitoredLocations}
              </div>

              <div className="metric-foot">
                active forecast points
              </div>

            </motion.div>

          </div>


          {/* HIGHEST RISK LOCATION */}

          {priorityLocation && (

            <div className="priority-layout">

              <div className="priority-card">

                <div className="section-kicker">
                  PRIORITY LOCATION
                </div>


                <div className="priority-main">

                  <div>

                    <div className="location-title-row">

                      <h2>
                        {priorityLocation.city ||
                          priorityLocation.location_id}
                      </h2>

                      <span
                        className={`risk-pill ${riskClass(
                          priorityLocation.risk_score
                        )}`}
                      >
                        {riskLabel(
                          priorityLocation.risk_score
                        )}
                      </span>

                    </div>


                    <div className="location-region">

                      {priorityLocation.state ||
                        "India"}

                      {" • "}

                      {priorityLocation.region_type ||
                        "region"}

                    </div>


                    <div className="forecast-temp-large">

                      {formatNumber(
                        getPredictedTmax(priorityLocation),
                        1
                      )}

                      <span>
                        °C
                      </span>

                    </div>


                    <div className="forecast-label">
                      predicted next-day maximum temperature
                    </div>

                  </div>


                  {/* GAUGE */}

                  <div className="risk-gauge-wrapper">

                    <svg
                      viewBox="0 0 180 180"
                      className="risk-gauge"
                    >

                      <circle
                        cx="90"
                        cy="90"
                        r="68"
                        className="gauge-track"
                      />

                      <motion.circle
                        cx="90"
                        cy="90"
                        r="68"
                        className="gauge-progress"
                        strokeDasharray="427.3"
                        initial={{
                          strokeDashoffset:
                            427.3,
                        }}
                        animate={{
                          strokeDashoffset:
                            427.3 -
                            (427.3 *
                              Math.min(
                                getRiskScore(priorityLocation),
                                100
                              )) /
                              100,
                        }}
                        transition={{
                          duration: 1,
                          ease: "easeOut",
                        }}
                      />

                    </svg>


                    <div className="gauge-center">

                      <strong>
                        {formatNumber(
                          priorityLocation.risk_score,
                          1
                        )}
                      </strong>

                      <span>
                        RISK SCORE
                      </span>

                    </div>

                  </div>

                </div>


                <div className="priority-stats">

                  <div>

                    <span>
                      Heatwave Probability
                    </span>

                    <strong>
                      {formatPercent(
                        priorityLocation.heatwave_probability
                      )}
                    </strong>

                  </div>


                  <div>

                    <span>
                      Temperature Anomaly
                    </span>

                    <strong>
                      {formatNumber(
                        priorityLocation.departure,
                        1
                      )}
                      °C
                    </strong>

                  </div>


                  <div>

                    <span>
                      Persistence
                    </span>

                    <strong>
                      {formatNumber(
                        priorityLocation.consecutive_heatwave_days,
                        0
                      )}{" "}
                      days
                    </strong>

                  </div>

                </div>

              </div>


              {/* LOCATION SELECTOR */}

              <div className="location-selector-card">

                <div className="section-kicker">
                  LOCATION INTELLIGENCE
                </div>

                <h3>
                  Select a location
                </h3>

                <p>
                  Explore forecast and risk
                  intelligence for each monitored
                  point.
                </p>


                <div className="location-selector-list">

                  {locations.map(
                    (location) => (

                      <button
                        key={
                          location.location_id
                        }
                        className={`location-selector ${
                          selectedLocation?.location_id ===
                          location.location_id
                            ? "selected"
                            : ""
                        }`}
                        onClick={() =>
                          setSelectedLocation(
                            location
                          )
                        }
                      >

                        <span className="location-selector-dot">
                          <CircleDot size={14} />
                        </span>

                        <span className="location-selector-name">

                          <strong>
                            {location.city ||
                              location.location_id}
                          </strong>

                          <small>
                            {formatNumber(
                              getPredictedTmax(location),
                              1
                            )}
                            °C
                          </small>

                        </span>

                        <span
                          className={`mini-risk ${riskClass(
                            location.risk_score
                          )}`}
                        >
                          {formatNumber(
                            location.risk_score,
                            0
                          )}
                        </span>

                        <ChevronRight
                          size={16}
                        />

                      </button>

                    )
                  )}

                </div>

              </div>

              <div
                className="advisory-panel"
                style={{
                  marginTop: "22px",
                  padding: "18px 20px",
                  border: "1px solid rgba(139,226,139,0.16)",
                  borderRadius: "14px",
                  background: "rgba(139,226,139,0.045)",
                }}
              >

                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                    marginBottom: "8px",
                  }}
                >

                  <ShieldAlert size={18} />

                  <div
                    className="component-title"
                    style={{ margin: 0 }}
                  >
                    Current Heat Advisory
                  </div>

                </div>

                <p
                  style={{
                    margin: 0,
                    lineHeight: 1.65,
                    color: "#c7d5cc",
                  }}
                >
                  {selectedLocation.advisory ||
                    "No advisory is available for this location."}
                </p>

                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "8px",
                    marginTop: "12px",
                  }}
                >

                  {[
                    selectedLocation.top_factor_1,
                    selectedLocation.top_factor_2,
                    selectedLocation.top_factor_3,
                  ]
                    .filter(Boolean)
                    .map((factor) => (
                      <span
                        key={factor}
                        style={{
                          padding: "5px 9px",
                          borderRadius: "999px",
                          background: "rgba(255,255,255,0.055)",
                          border: "1px solid rgba(255,255,255,0.08)",
                          color: "#9fb3a7",
                          fontSize: "11px",
                        }}
                      >
                        {factor}
                      </span>
                    ))}

                </div>

              </div>

            </div>

          )}

        </section>


        {/* ====================================================
            MAP
            ==================================================== */}

        <section
          id="risk-map"
          className="dashboard-section"
        >

          <div className="section-heading-row">

            <div>

              <div className="eyebrow">
                SPATIAL RISK INTELLIGENCE
              </div>

              <h2>
                Hyperlocal Heat Risk Map
              </h2>

              <p>
                Geographic distribution of predicted
                heat risk across monitored locations.
              </p>

            </div>


            <div className="map-legend">

              <span>
                <i className="legend-low" />
                Low
              </span>

              <span>
                <i className="legend-moderate" />
                Moderate
              </span>

              <span>
                <i className="legend-high" />
                High+
              </span>

            </div>

          </div>


          <div className="map-card">

            <MapContainer
              center={[
                20.5937,
                78.9629,
              ]}
              zoom={5}
              scrollWheelZoom={true}
              className="risk-map"
            >

              {/* =================================================
                  FIXED TILE PROVIDER
                  ================================================= */}

              <TileLayer
                attribution='&copy; OpenStreetMap contributors &copy; OpenStreetMap France'
                url="https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png"
                minZoom={1}
                maxZoom={20}
              />

              <MapViewController
                locations={locations}
              />


              {locations.map(
                (location) => {

                  const latitude =
                    Number(
                      location.latitude
                    );

                  const longitude =
                    Number(
                      location.longitude
                    );

                  if (
                    !Number.isFinite(
                      latitude
                    ) ||
                    !Number.isFinite(
                      longitude
                    )
                  ) {
                    return null;
                  }

                  const score =
                    Number(
                      location.risk_score || 0
                    );

                  return (

                    <CircleMarker
                      key={
                        location.location_id
                      }
                      center={[
                        latitude,
                        longitude,
                      ]}
                      radius={
                        priorityLocation?.location_id ===
                        location.location_id
                          ? 14
                          : 10
                      }
                      pathOptions={{
                        className: `risk-marker ${riskClass(
                          score
                        )}`,
                      }}
                      eventHandlers={{
                        click: () =>
                          setSelectedLocation(
                            location
                          ),
                      }}
                    >

                      <Popup>

                        <div className="map-popup">

                          <strong>
                            {location.city ||
                              location.location_id}
                          </strong>

                          <span>
                            Risk Score:{" "}
                            {formatNumber(
                              score,
                              1
                            )}
                          </span>

                          <span>
                            Forecast:{" "}
                            {formatNumber(
                              getPredictedTmax(location),
                              1
                            )}
                            °C
                          </span>

                          <span>
                            Heatwave:{" "}
                            {formatPercent(
                              location.heatwave_probability
                            )}
                          </span>

                        </div>

                      </Popup>

                    </CircleMarker>

                  );
                }
              )}

            </MapContainer>


            <div className="map-overlay-card">

              <span>
                ACTIVE MONITORING
              </span>

              <strong>
                {locations.length} locations
              </strong>

              <small>
                Click a point to inspect risk
              </small>

            </div>

          </div>

        </section>


        {/* ====================================================
            LOCATIONS
            ==================================================== */}

        <section
          id="locations"
          className="dashboard-section"
        >

          <div className="section-heading-row">

            <div>

              <div className="eyebrow">
                RISK PRIORITIZATION
              </div>

              <h2>
                Location Risk Ranking
              </h2>

              <p>
                Locations ordered by composite
                Hyperlocal Heat Risk Score.
              </p>

            </div>

          </div>


          <div className="risk-table-card">

            <div className="risk-table-header">

              <span>RANK</span>
              <span>LOCATION</span>
              <span>FORECAST</span>
              <span>HEATWAVE</span>
              <span>RISK SCORE</span>
              <span>STATUS</span>

            </div>


            {rankedLocations.map(
              (location, index) => (

                <button
                  className={`risk-table-row ${
                    priorityLocation?.location_id ===
                    location.location_id
                      ? "selected"
                      : ""
                  }`}
                  key={
                    location.location_id
                  }
                  onClick={() =>
                    setSelectedLocation(
                      location
                    )
                  }
                >

                  <span className="rank-number">
                    {String(
                      index + 1
                    ).padStart(2, "0")}
                  </span>


                  <span className="table-location">

                    <strong>
                      {location.city ||
                        location.location_id}
                    </strong>

                    <small>
                      {location.state ||
                        "India"}
                    </small>

                  </span>


                  <span className="table-value">

                    {formatNumber(
                      location.predicted_tmax ??
                        location.predicted_temperature,
                      1
                    )}
                    °C

                  </span>


                  <span className="table-value">

                    {formatPercent(
                      location.heatwave_probability
                    )}

                  </span>


                  <span className="table-risk">

                    <strong>
                      {formatNumber(
                        location.risk_score,
                        1
                      )}
                    </strong>

                    <div className="table-risk-bar">

                      <span
                        style={{
                          width: `${Math.min(
                            getRiskScore(location),
                            100
                          )}%`,
                        }}
                      />

                    </div>

                  </span>


                  <span>

                    <span
                      className={`risk-pill ${riskClass(
                        location.risk_score
                      )}`}
                    >
                      {riskLabel(
                        location.risk_score
                      )}
                    </span>

                  </span>

                </button>

              )
            )}

          </div>


          {/* SELECTED LOCATION */}

          {selectedLocation && (

            <div
              id="selected-intelligence"
              className="intelligence-card"
            >

              <div className="intelligence-header">

                <div>

                  <div className="section-kicker">
                    SELECTED LOCATION
                  </div>

                  <h3>
                    {selectedLocation.city ||
                      selectedLocation.location_id}
                  </h3>

                </div>


                <span
                  className={`risk-pill ${riskClass(
                    selectedLocation.risk_score
                  )}`}
                >
                  {riskLabel(
                    selectedLocation.risk_score
                  )}
                </span>

                <span
                  className="risk-pill"
                  style={{
                    marginLeft: "8px",
                    opacity: 0.9,
                  }}
                >
                  {selectedLocation.severity || "Unknown"}
                </span>

              </div>


              <div className="intelligence-grid">

                <div className="intelligence-stat">

                  <span>
                    Predicted Tmax
                  </span>

                  <strong>
                    {formatNumber(
                      getPredictedTmax(selectedLocation),
                      1
                    )}
                    °C
                  </strong>

                </div>


                <div className="intelligence-stat">

                  <span>
                    Heatwave Probability
                  </span>

                  <strong>
                    {formatPercent(
                      selectedLocation.heatwave_probability
                    )}
                  </strong>

                </div>


                <div className="intelligence-stat">

                  <span>
                    Risk Score
                  </span>

                  <strong>
                    {formatNumber(
                      selectedLocation.risk_score,
                      1
                    )}
                    /100
                  </strong>

                </div>


                <div className="intelligence-stat">

                  <span>
                    Humidity
                  </span>

                  <strong>
                    {formatNumber(
                      getHumidity(selectedLocation),
                      0
                    )}
                    %
                  </strong>

                </div>

                <div className="intelligence-stat">

                  <span>
                    Severity
                  </span>

                  <strong>
                    {selectedLocation.severity || "Unknown"}
                  </strong>

                </div>

              </div>


              <div className="component-section">

                <div className="component-title">
                  Risk Engine Components
                </div>

                <p className="component-description">
                  Deterministic components used to
                  construct the final 0–100 risk score.
                  These are separate from SHAP.
                </p>


                <div className="component-bars">

                  {riskComponents.map(
                    (component) => (

                      <div
                        className="component-row"
                        key={
                          component.name
                        }
                      >

                        <div className="component-label">

                          <span>
                            {component.name}
                          </span>

                          <small>
                            weight{" "}
                            {component.weight}
                          </small>

                        </div>


                        <div className="component-bar">

                          <motion.span
                            initial={{
                              width: 0,
                            }}
                            animate={{
                              width: `${Math.min(
                                component.value,
                                100
                              )}%`,
                            }}
                            transition={{
                              duration: 0.7,
                            }}
                          />

                        </div>


                        <strong>
                          {formatNumber(
                            component.value,
                            0
                          )}
                        </strong>

                      </div>

                    )
                  )}

                </div>

              </div>

            </div>

          )}

        </section>


        {/* ====================================================
            FORECAST
            ==================================================== */}

        <section
          id="forecast"
          className="dashboard-section"
        >

          <div className="section-heading-row">

            <div>

              <div className="eyebrow">
                PREDICTIVE ANALYTICS
              </div>

              <h2>
                Forecast Analytics
              </h2>

              <p>
                Model-generated temperature and
                heatwave probability forecasts.
              </p>

            </div>

          </div>


          <div className="analytics-grid">

            {/* TEMPERATURE */}

            <div className="chart-card">

              <div className="chart-card-header">

                <div>

                  <span>
                    TEMPERATURE FORECAST
                  </span>

                  <h3>
                    Predicted Tmax by City
                  </h3>

                </div>

                <Thermometer size={19} />

              </div>


              <div className="chart-container">

                <ResponsiveContainer
                  width="100%"
                  height="100%"
                >

                  <BarChart
                    data={
                      temperatureChartData
                    }
                    margin={{
                      top: 10,
                      right: 10,
                      left: -10,
                      bottom: 5,
                    }}
                  >

                    <CartesianGrid
                      strokeDasharray="3 3"
                      vertical={false}
                      stroke="rgba(255,255,255,0.08)"
                    />

                    <XAxis
                      dataKey="city"
                      tick={{
                        fill: "#8fa59a",
                        fontSize: 11,
                      }}
                      axisLine={false}
                      tickLine={false}
                    />

                    <YAxis
                      tick={{
                        fill: "#8fa59a",
                        fontSize: 11,
                      }}
                      axisLine={false}
                      tickLine={false}
                    />

                    <Tooltip
                      cursor={false}
                      contentStyle={{
                        background:
                          "#101c17",
                        border:
                          "1px solid rgba(139,226,139,0.2)",
                        borderRadius:
                          "10px",
                        color: "#fff",
                      }}
                    />

                    <Bar
                      dataKey="temperature"
                      fill="#8BE28B"
                      activeBar={{ fill: "#8BE28B", stroke: "none" }}
                      radius={[
                        6,
                        6,
                        0,
                        0,
                      ]}
                    >

                      {temperatureChartData.map(
                        (_, index) => (
                          <Cell
                            key={index}
                            fill="#8BE28B"
                          />
                        )
                      )}

                    </Bar>

                  </BarChart>

                </ResponsiveContainer>

              </div>

            </div>


            {/* HEATWAVE PROBABILITY */}

            <div className="chart-card">

              <div className="chart-card-header">

                <div>

                  <span>
                    CLASSIFICATION OUTPUT
                  </span>

                  <h3>
                    Heatwave Probability
                  </h3>

                </div>

                <ShieldAlert size={19} />

              </div>


              <div className="chart-container">

                <ResponsiveContainer
                  width="100%"
                  height="100%"
                >

                  <BarChart
                    data={
                      probabilityChartData
                    }
                    margin={{
                      top: 10,
                      right: 10,
                      left: -10,
                      bottom: 5,
                    }}
                  >

                    <CartesianGrid
                      strokeDasharray="3 3"
                      vertical={false}
                      stroke="rgba(255,255,255,0.08)"
                    />

                    <XAxis
                      dataKey="city"
                      tick={{
                        fill: "#8fa59a",
                        fontSize: 11,
                      }}
                      axisLine={false}
                      tickLine={false}
                    />

                    <YAxis
                      domain={[
                        0,
                        100,
                      ]}
                      tick={{
                        fill: "#8fa59a",
                        fontSize: 11,
                      }}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(value) =>
                        `${value}%`
                      }
                    />

                    <Tooltip
                      cursor={false}
                      formatter={(value) =>
                        `${Number(
                          value
                        ).toFixed(1)}%`
                      }
                      contentStyle={{
                        background:
                          "#101c17",
                        border:
                          "1px solid rgba(139,226,139,0.2)",
                        borderRadius:
                          "10px",
                        color: "#fff",
                      }}
                    />

                    <Bar
                      dataKey="probability"
                      fill="#8BE28B"
                      activeBar={{ fill: "#8BE28B", stroke: "none" }}
                      radius={[
                        6,
                        6,
                        0,
                        0,
                      ]}
                    >

                      {probabilityChartData.map(
                        (_, index) => (
                          <Cell
                            key={index}
                            fill="#8BE28B"
                          />
                        )
                      )}

                    </Bar>

                  </BarChart>

                </ResponsiveContainer>

              </div>

            </div>


            {/* RISK SCORE */}

            <div className="chart-card chart-card-wide">

              <div className="chart-card-header">

                <div>

                  <span>
                    COMPOSITE RISK ENGINE
                  </span>

                  <h3>
                    Hyperlocal Risk Score Ranking
                  </h3>

                </div>

                <Gauge size={19} />

              </div>


              <div className="chart-container chart-container-wide">

                <ResponsiveContainer
                  width="100%"
                  height="100%"
                >

                  <BarChart
                    data={
                      riskChartData
                    }
                    layout="vertical"
                    margin={{
                      top: 10,
                      right: 20,
                      left: 20,
                      bottom: 5,
                    }}
                  >

                    <CartesianGrid
                      strokeDasharray="3 3"
                      horizontal={false}
                      stroke="rgba(255,255,255,0.08)"
                    />

                    <XAxis
                      type="number"
                      domain={[
                        0,
                        100,
                      ]}
                      tick={{
                        fill: "#8fa59a",
                        fontSize: 11,
                      }}
                      axisLine={false}
                      tickLine={false}
                    />

                    <YAxis
                      type="category"
                      dataKey="city"
                      width={85}
                      tick={{
                        fill: "#d8e4dc",
                        fontSize: 11,
                      }}
                      axisLine={false}
                      tickLine={false}
                    />

                    <Tooltip
                      cursor={false}
                      formatter={(value) =>
                        `${Number(
                          value
                        ).toFixed(1)} / 100`
                      }
                      contentStyle={{
                        background:
                          "#101c17",
                        border:
                          "1px solid rgba(139,226,139,0.2)",
                        borderRadius:
                          "10px",
                        color: "#fff",
                      }}
                    />

                    <Bar
                      dataKey="risk"
                      fill="#8BE28B"
                      activeBar={{ fill: "#8BE28B", stroke: "none" }}
                      radius={[
                        0,
                        6,
                        6,
                        0,
                      ]}
                    >

                      {riskChartData.map(
                        (_, index) => (
                          <Cell
                            key={index}
                            fill="#8BE28B"
                          />
                        )
                      )}

                    </Bar>

                  </BarChart>

                </ResponsiveContainer>

              </div>

            </div>

          </div>


          {/* FORECAST SUMMARY */}

          <div className="forecast-summary-grid">

            <div className="forecast-summary-card">

              <span>
                HOTTEST FORECAST
              </span>

              <strong>
                {hottestLocation?.city ||
                  "—"}
              </strong>

              <small>
                {formatNumber(
                  getPredictedTmax(hottestLocation),
                  1
                )}
                °C predicted Tmax
              </small>

            </div>


            <div className="forecast-summary-card">

              <span>
                HIGHEST RISK
              </span>

              <strong>
                {rankedLocations[0]?.city ||
                  "—"}
              </strong>

              <small>
                {formatNumber(
                  rankedLocations[0]?.risk_score,
                  1
                )}
                /100 composite score
              </small>

            </div>


            <div className="forecast-summary-card">

              <span>
                PEAK PROBABILITY
              </span>

              <strong>

                {highestProbabilityLocation?.city || "—"}

              </strong>

              <small>

                {formatPercent(
                  highestProbabilityLocation
                    ? getHeatwaveProbability(
                        highestProbabilityLocation
                      )
                    : 0
                )}

                {" "}heatwave probability

              </small>

            </div>


            <div className="forecast-summary-card">

              <span>
                FORECAST POINTS
              </span>

              <strong>
                {locations.length}
              </strong>

              <small>
                monitored locations
              </small>

            </div>

          </div>

        </section>


        {/* ====================================================
            EXPLAINABLE AI
            ==================================================== */}

        <section
          id="explain"
          className="dashboard-section xai-section"
        >

          <div className="xai-heading">

            <div>

              <div className="xai-eyebrow">
                MODEL TRANSPARENCY
              </div>

              <h2>
                Explainable AI
              </h2>

              <p>
                Understand which input features
                influenced the Random Forest's
                next-day temperature prediction.
              </p>

            </div>


            <div className="xai-model-badge">

              <BrainCircuit size={17} />

              <span>
                SHAP • TREE EXPLAINER
              </span>

            </div>

          </div>


          {explanationLoading && (

            <div className="xai-loading">

              <div className="loading-orb small">
                <BrainCircuit size={20} />
              </div>

              <span>
                Calculating model explanation...
              </span>

            </div>

          )}


          {!explanationLoading &&
            explanationData && (

              <>

                {/* XAI TOP CARDS */}

                <div className="xai-layout">

                  <div className="xai-location-card">

                    <div className="section-kicker">
                      SELECTED MODEL PREDICTION
                    </div>

                    <div className="xai-location-title">

                      <div>

                        <h3>
                          {explanationData.city ||
                            selectedLocation?.city ||
                            explanationData.location_id}
                        </h3>

                        <span>
                          {getForecastDate(
                            selectedLocation,
                            explanationData
                          ) ||
                            explanationData.date ||
                            "Date unavailable"}
                        </span>

                      </div>


                      <div className="xai-risk-score">

                        <strong>
                          {formatNumber(
                            selectedLocation?.risk_score,
                            1
                          )}
                        </strong>

                        <small>
                          RISK SCORE
                        </small>

                      </div>

                    </div>


                    <div className="xai-prediction">

                      <div>

                        <span>
                          BASE VALUE
                        </span>

                        <strong>
                          {formatNumber(
                            explanationData.base_value,
                            2
                          )}
                          °C
                        </strong>

                      </div>


                      <div className="xai-arrow">
                        →
                      </div>


                      <div>

                        <span>
                          PREDICTED Tmax
                        </span>

                        <strong>
                          {formatNumber(
                            explanationData.predicted_temperature,
                            2
                          )}
                          °C
                        </strong>

                      </div>

                    </div>


                    <div className="xai-note">

                      <BrainCircuit size={15} />

                      <span>
                        SHAP values show how each
                        feature moves the model's
                        prediction away from the
                        baseline value.
                      </span>

                    </div>

                  </div>


                  <div className="xai-factors-card">

                    <div className="section-kicker">
                      STRONGEST MODEL SIGNALS
                    </div>

                    <h3>
                      What influenced the forecast?
                    </h3>


                    <div className="xai-signal-grid">

                      <div className="xai-signal positive">

                        <span>
                          STRONGEST UPWARD SIGNAL
                        </span>

                        <strong>
                          {strongestPositiveShap?.feature ||
                            "—"}
                        </strong>

                        <small>

                          {strongestPositiveShap
                            ? `+${formatNumber(
                                strongestPositiveShap.shap_value,
                                3
                              )}°C contribution`
                            : "No positive contribution"}

                        </small>

                      </div>


                      <div className="xai-signal negative">

                        <span>
                          STRONGEST DOWNWARD SIGNAL
                        </span>

                        <strong>
                          {strongestNegativeShap?.feature ||
                            "—"}
                        </strong>

                        <small>

                          {strongestNegativeShap
                            ? `${formatNumber(
                                strongestNegativeShap.shap_value,
                                3
                              )}°C contribution`
                            : "No negative contribution"}

                        </small>

                      </div>

                    </div>

                  </div>

                </div>


                {/* SHAP CONTRIBUTIONS */}

                <div className="xai-chart-card">

                  <div className="xai-chart-header">

                    <div>

                      <div className="section-kicker">
                        LOCAL SHAP EXPLANATION
                      </div>

                      <h3>
                        Feature Contributions
                      </h3>

                      <p>
                        Positive values push the
                        predicted temperature higher;
                        negative values push it lower.
                      </p>

                    </div>

                    <BrainCircuit size={20} />

                  </div>


                  <div className="xai-feature-list">

                    {shapFeatures.map(
                      (feature, index) => {

                        const shapValue =
                          Number(
                            feature.shap_value ||
                              0
                          );

                        const magnitude =
                          Math.abs(
                            shapValue
                          );

                        const maxMagnitude =
                          Math.max(
                            ...shapFeatures.map(
                              (item) =>
                                Math.abs(
                                  Number(
                                    item.shap_value ||
                                      0
                                  )
                                )
                            ),
                            0.001
                          );

                        const width =
                          (magnitude /
                            maxMagnitude) *
                          100;

                        const isPositive =
                          shapValue > 0;

                        return (

                          <motion.div
                            className={`xai-feature ${
                              isPositive
                                ? "positive"
                                : "negative"
                            }`}
                            key={`${feature.feature}-${index}`}
                            initial={{
                              opacity: 0,
                              x: -10,
                            }}
                            whileInView={{
                              opacity: 1,
                              x: 0,
                            }}
                            viewport={{
                              once: true,
                            }}
                            transition={{
                              delay:
                                index * 0.04,
                            }}
                          >

                            <div className="xai-feature-info">

                              <strong>
                                {feature.feature}
                              </strong>

                              <span>
                                Value:{" "}
                                {formatNumber(
                                  feature.value,
                                  2
                                )}
                              </span>

                            </div>


                            <div className="xai-feature-track">

                              <motion.div
                                className="xai-feature-bar"
                                initial={{
                                  width: 0,
                                }}
                                whileInView={{
                                  width: `${width}%`,
                                }}
                                viewport={{
                                  once: true,
                                }}
                                transition={{
                                  duration: 0.6,
                                  delay:
                                    index * 0.04,
                                }}
                              />

                            </div>


                            <div className="xai-feature-value">

                              <span>
                                {isPositive
                                  ? "+"
                                  : ""}
                                {formatNumber(
                                  shapValue,
                                  3
                                )}
                                °C
                              </span>

                              <small>
                                {isPositive
                                  ? "raises"
                                  : "lowers"}
                              </small>

                            </div>

                          </motion.div>

                        );
                      }
                    )}

                  </div>

                </div>


                {/* INTERPRETATION */}

                <div className="xai-interpretation">

                  <div className="xai-interpretation-icon">
                    <BrainCircuit size={20} />
                  </div>

                  <div>

                    <span>
                      MODEL INTERPRETATION
                    </span>

                    <p>

                      For{" "}

                      <strong>
                        {explanationData.city ||
                          selectedLocation?.city ||
                          explanationData.location_id}
                      </strong>

                      , the model's strongest
                      upward signal was{" "}

                      <strong>
                        {strongestPositiveShap?.feature ||
                          "not identified"}
                      </strong>

                      {strongestPositiveShap
                        ? `, contributing approximately +${formatNumber(
                            strongestPositiveShap.shap_value,
                            3
                          )}°C`
                        : ""}

                      . The strongest downward
                      signal was{" "}

                      <strong>
                        {strongestNegativeShap?.feature ||
                          "not identified"}
                      </strong>

                      {strongestNegativeShap
                        ? `, contributing approximately ${formatNumber(
                            strongestNegativeShap.shap_value,
                            3
                          )}°C`
                        : ""}

                      .

                    </p>

                  </div>

                </div>


                {/* ACADEMIC NOTE */}

                <div className="xai-academic-note">

                  <ShieldAlert size={17} />

                  <div>

                    <strong>
                      Interpretation note
                    </strong>

                    <p>
                      SHAP explains the behavior of
                      the Random Forest temperature
                      model. These contributions
                      describe model behavior and
                      should not be interpreted as
                      causal effects. The final
                      0–100 Heat Risk Score is
                      calculated separately by the
                      deterministic risk engine.
                    </p>

                  </div>

                </div>

              </>

            )}


          {!explanationLoading &&
            !explanationData && (

              <div className="xai-empty">

                <BrainCircuit size={26} />

                <h3>
                  SHAP explanation unavailable
                </h3>

                <p>
                  Run the explainability pipeline
                  before viewing local model
                  explanations.
                </p>

              </div>

            )}

        </section>


        {/* ====================================================
            FOOTER
            ==================================================== */}

        <footer className="dashboard-footer">

          <div>

            <strong>
              HeatWatch AI
            </strong>

            <span>
              Explainable Hyperlocal Heat Intelligence
            </span>

          </div>


          <div className="footer-status">

            <span className="status-dot" />

            Backend Connected

          </div>

        </footer>

      </main>

    </div>
  );
}


export default App;