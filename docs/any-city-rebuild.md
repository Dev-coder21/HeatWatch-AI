# HeatWatch AI — Any-City Rebuild (functionality only)

You are working in my existing project `heatwave-intelligence` (FastAPI backend in `backend/`, ML pipeline in `src/`, React + Vite dashboard in `frontend/`, tests in `tests/`). Read the README and the code in `src/`, `backend/main.py` and `frontend/src/App.jsx` before changing anything.

**Goal:** turn HeatWatch AI from a fixed 5-city, frozen-2024 snapshot into a **live heat-risk system that works for any location in India** — searched by name or clicked on a map — while keeping the IMD-based heatwave rules, the composite risk score, the advisories and the SHAP explanations.

**Out of scope for this round:** visual design. I will redesign the entire UI later. Only touch the frontend enough to wire up the new functionality and prove it works (see Phase 6). Don't restyle anything.

## Ground rules

- Create and work on a new git branch `any-city-rebuild`. Never commit to `main`. Commit after each phase with a clear message.
- Python stays the backend language; keep FastAPI, scikit-learn and SHAP. Add dependencies only when needed, and pin them in `requirements.txt`. Remove unused ones (streamlit, streamlit-folium, the Jupyter/ipykernel stack if notebooks don't need it — check first).
- All config (thresholds, risk weights, terrain cut-offs, cache TTLs, API base URLs) lives in `config/`, not hard-coded.
- Be honest in metrics and docs. Report recall, precision, PR-AUC and Brier score for the rare heatwave class, not just accuracy or weighted F1. Never cherry-pick the best split.
- **Stop and report to me at the two checkpoints marked ⏸.** Don't keep going past them without my OK.

## Current problems you are fixing

1. Locations are hard-coded in `config/locations.csv` (Pune, Mumbai, Delhi, Bengaluru, Shimla), including their terrain type.
2. Data ends on 2024-12-31, so the "forecast" is stale. Every city currently shows Low risk.
3. The models use `latitude` and `longitude` as features but were trained on only 5 points. They have effectively memorised those cities and won't generalise.
4. The heatwave classifier catches only 10 of 32 heatwave days on the 2024 test set (recall 0.31), and 0 of 4 on 2023 validation. The README overstates this.
5. Monthly "normals" come from the same 2015–2022 window used for training, and are monthly steps rather than a smooth seasonal curve.
6. The IMD absolute criterion is missing for plains: actual Tmax ≥ 45 °C → heatwave and ≥ 47 °C → severe, regardless of departure.
7. Empty files: `app/app.py`, `src/advisory.py`, `src/config.py`. The advisory logic actually lives inside `predict.py`.
8. `models/*.joblib` is gitignored, so a fresh clone can't run the API without rerunning the whole pipeline.

## Target architecture

```
user types "Jodhpur" or clicks the map
  → geocode (Open-Meteo Geocoding API, countryCode=IN)
  → location profile (cached): elevation, terrain type, 30-yr daily normals
  → live weather: last ~14 days observed + next 7 days forecast (Open-Meteo Forecast API, past_days + forecast_days)
  → features per forecast day
  → ML: (a) bias-corrected Tmax for each day, (b) heatwave probability, (c) severity
  → IMD rule check on the corrected forecast + composite risk score (0–100) + advisory
  → SHAP explanation for the selected day
  → JSON to the dashboard
```

---

## Phase 0 — Verify the data sources and plan ⏸

Check, with real requests, what each Open-Meteo API gives and over which date ranges:

- Geocoding: `https://geocoding-api.open-meteo.com/v1/search`
- Elevation: `https://api.open-meteo.com/v1/elevation`
- Historical weather / reanalysis: `https://archive-api.open-meteo.com/v1/archive`. I need 1991–2020 for normals and about 2015 to the present for training.
- Forecast with `past_days`: `https://api.open-meteo.com/v1/forecast`
- **Historical Forecast API** (`historical-forecast-api.open-meteo.com`) and **Previous Runs API** (`previous-runs-api.open-meteo.com`). These archive what the weather models *forecast* in the past, which is what lets the ML learn from forecast errors. Find out exactly which years and lead days (day 1–7) are available for daily Tmax over India.

Also note the free-tier rate limits and the grid resolution of the archive data. Don't call it "hyperlocal" in a way the resolution can't support.

**Report back:** what's available, the date ranges, which ML design below is feasible with that data, the list of training sites you propose (see Phase 2), and an estimate of how many API calls and how long the full data download takes. Then wait for me.

## Phase 1 — Location profile service (any point in India)

Create `src/locations/`, or similar, with:

- `geocode(query)` → candidates `{name, district/admin1, state, lat, lon, elevation}`. India only. Also support reverse lookup for map clicks: nearest named place, or a "lat, lon" label.
- `terrain_type(lat, lon, elevation)` → `hilly | coastal | plains`:
  - **Hilly** if elevation ≥ the configured threshold (default 1000 m, kept in config).
  - **Coastal** if the distance to India's coastline ≤ the configured distance (default 50 km). Bundle a small simplified coastline file (e.g. Natural Earth 1:10m coastline clipped to India) in `data/static/`. Compute distance with haversine or shapely; no external API.
  - Otherwise **plains**.
  - Hilly takes priority over coastal.
  - Write unit tests: Shimla, Darjeeling, Ooty → hilly; Mumbai, Chennai, Kochi, Puri → coastal; Delhi, Jaipur, Nagpur, Lucknow → plains.
- `normals(lat, lon)` → a **smoothed daily climatological normal of Tmax for 1991–2020** (day-of-year mean with a ±7-day window, or harmonic fit). This matches IMD's normal period better than the current monthly means.
- **Cache** profiles on disk (SQLite or JSON) keyed by coordinates rounded to the data grid (≈0.1°), so a city costs one archive call the first time and none afterwards.

Make `config/locations.csv` a list of *featured* places (keep the original 5, add a handful more) rather than the only places the system knows.

## Phase 2 — Training dataset across India

The models must learn general heat behaviour, not city identities.

- Pick about **60–100 training sites** spread across India's climate zones: Thar desert, Indo-Gangetic plain, central India, Deccan plateau, the west and east coasts, Himalayan hill stations, the Northeast and the islands. Include known heatwave hotspots (e.g. Churu, Phalodi, Banda, Nagpur, Chandrapur, Titlagarh, Prayagraj, Delhi) and keep a balance across terrain types. Store them in `config/training_sites.csv` with their computed terrain type.
- Write a download script with throttling, retries and **resume**: skip files already downloaded. It should produce, per site:
  - observed daily weather (Tmax, Tmin, mean RH, max wind, precipitation, surface pressure), and
  - if Phase 0 confirmed availability, the model-forecast Tmax at lead days 1–7 for the same dates.
- Re-label every day with the corrected IMD rules (fix #6), using the 1991–2020 daily normals from Phase 1.
- **Hold out sites as well as time.** Test on a set of sites the model never saw, so we measure the "any city" claim, plus a time split (e.g. train ≤2022, validate 2023, test 2024–2025). Report both.

## Phase 3 — Models (Option A: use the weather-model forecast as an input)

Raw weather-model forecasts beat a Random Forest that only looks at past days, so our ML adds value on top of them.

**a) Tmax bias-correction model** (a regression model, in the spirit of classic Model Output Statistics):
- **Inputs:** raw forecast Tmax for the target day; lead day; recent observed lags (Tmax lag 1–3, 3- and 7-day means, Tmin lag 1, RH lag 1, wind lag 1, 3-day precipitation, 3-day trend); normal Tmax for the target day; elevation; terrain type; distance to coast; seasonal encoding (sin/cos of day-of-year).
- **No raw latitude or longitude.**
- **Target:** observed Tmax on the target day.
- **Compare against two baselines:** persistence (tomorrow = today) and the raw weather-model forecast. Report MAE and RMSE per lead day and per terrain type, on unseen sites. If the correction doesn't beat the raw forecast, say so and use the raw forecast instead.
- If historical forecasts turned out not to be available, fall back to "perfect-prog" training: use observed Tmax with lead-dependent noise as a stand-in for the forecast. Document that clearly.

**b) Heatwave probability and severity classifiers:**
- **Inputs:** the same features as (a), plus forecast departure from normal.
- **Target:** the *observed* IMD label on the target day. This is what makes the classifier useful: it learns how often a forecast like this one actually turns into a heatwave, not just the threshold rule again.
- Handle the rare class properly: class weights, and choose the decision threshold on validation to give a sensible recall/precision balance. Calibrate the probabilities (isotonic or Platt).
- **Report:** recall, precision, F1 and PR-AUC for heatwave; Brier score; reliability; macro-F1 for severity. Give results per terrain type and on unseen sites.
- Try at least one gradient-boosting model (e.g. sklearn HistGradientBoosting, or LightGBM) alongside Random Forest, and pick by validation results.

Save models together with a small metadata JSON in `models/`: feature list, training date range, metrics and version. If the files are small enough (< ~50 MB total), commit them. Remove `models/*.joblib` from `.gitignore`, so the API works straight after clone.

**⏸ Checkpoint:** show me the metrics tables (baselines vs our models, seen vs unseen sites) before building the API on top.

## Phase 4 — Live inference service

Create `src/service/` (pure Python, no FastAPI imports) with `assess(lat, lon)`, which:
1. loads or creates the location profile;
2. fetches the last ~14 days plus the next 7 forecast days (cache the result about 1 hour per grid cell);
3. builds features for each of the next 7 days;
4. runs the models;
5. applies the IMD rules to the corrected Tmax;
6. computes the composite risk score.

Keep the 5 weighted components and weights from `config/thresholds.yaml`. Persistence should use observed consecutive heatwave days carried forward through the forecasted ones.

Its output is, per day: `predicted_tmax`, `raw_forecast_tmax`, `normal_tmax`, `departure`, `heatwave_probability`, `severity`, `risk_score`, `risk_category`, `risk_components`, `advisory` and `top_factors`. Add SHAP values for each day, computed on demand and cached.

- Move the advisory text into `src/advisory.py`. Write audience-specific advice for outdoor workers, the elderly, and children.
- Delete the empty `app/app.py` and `src/config.py`, or give them a real purpose.
- Make `src/pipeline.py` a single command that runs the whole offline flow (download → clean → label → features → train → evaluate). Every step should be skippable if its output already exists.

## Phase 5 — API (FastAPI)

New endpoints (typed with Pydantic response models, with clear 4xx errors):

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/search?q=` | Place autocomplete, India only |
| GET | `/api/reverse?lat=&lon=` | Name for a map click |
| GET | `/api/risk?lat=&lon=` | Full 7-day assessment for a point (main endpoint) |
| GET | `/api/risk/{place_slug}` | Same, for featured places |
| GET | `/api/explain?lat=&lon=&day=` | SHAP breakdown for one forecast day |
| GET | `/api/featured` | Today's risk for featured places |
| GET | `/api/national` | Today's risk for all training sites (for a national map). Precompute it in the background or on first request; cache ~1 h |
| GET | `/api/health` | Models loaded, model version, cache status, upstream API reachable |
| GET | `/api/meta` | Thresholds, risk bands, model metrics summary (for an "About the model" panel later) |

Requirements:
- Return 400 for points outside India.
- Return a clear error if Open-Meteo is unreachable, and serve the last cached data with a `stale: true` flag when possible.
- Make CORS origins configurable via `.env`.
- Write `docs/API.md` with example responses for every endpoint. **I'll use this as the contract when I redesign the UI, so make it complete and accurate.**

Keep the old endpoints working only if the current frontend needs them during Phase 6. Otherwise remove them.

## Phase 6 — Minimal frontend wiring (no redesign)

In the existing `frontend/src/App.jsx`, with no visual work beyond what's needed:
- Read the API base URL from a Vite env variable instead of the hard-coded `http://localhost:8000`.
- Add a place search box (with autocomplete) and click-anywhere-on-the-map to assess a point.
- Show the featured places as quick picks.
- Show the 7-day outlook for the selected place.
- Keep the existing map, ranking, charts and SHAP panel working with the new API shape.

That's all. The full redesign comes later.

## Phase 7 — Tests, docs, verification

- pytest:
  - unit tests for terrain classification, normals, the IMD rules (including the 45/47 °C absolute rule and every region's edge cases), risk score bounds and advisory selection;
  - service tests with Open-Meteo mocked (no network in tests);
  - API tests with FastAPI's TestClient;
  - a test that no feature list contains raw `latitude` or `longitude`.
- Rewrite the README with honest metrics from Phase 3, the real data resolution, how "any city" works, one-command setup, and known limitations.
- **End-to-end check:** start the backend and frontend. Assess Delhi, Jaisalmer, Mumbai, Shimla, Guwahati and Port Blair, plus a random map click in rural Madhya Pradesh. Confirm each one returns a plausible 7-day outlook, the right terrain type, and a SHAP explanation. Also hit the API directly for a point outside India and confirm the 400.
- Paste a short summary for me: what changed, final metrics, anything you couldn't do and why.
