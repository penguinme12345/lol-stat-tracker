# LoL Coach Tracker

Local-first League of Legends analytics app with:

- Python ML pipeline (ingest, features, Random Forest training, insights)
- FastAPI backend for desktop integration
- Electron desktop shell UI
- Timeline-aware contextual coaching intelligence

## Python Setup

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set:

- `RIOT_API_KEY` (required only for direct Riot mode)
- `TRACKER_PUUID` (optional; only needed if automatic inference picks the wrong player)
- `RIOT_PROXY_URL` (optional; if set, app uses proxy instead of direct Riot calls)
- `RIOT_PROXY_ACCESS_TOKEN` (required when `RIOT_PROXY_URL` is set)

## CLI Usage

```bash
python main.py ingest --game-name "<gameName>" --tag-line "<tagLine>" --region americas --count 100
python main.py build-dataset
python main.py train
python main.py report
```

`ingest` now fetches and caches both match payloads and Riot timeline payloads.

## FastAPI Backend

Run backend:

```bash
python main.py serve-api --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health`
- `POST /sync`
- `POST /build`
- `POST /train`
- `GET /api/intelligence/report`

`/sync` request body example:

```json
{
  "game_name": "SEAMIN DEAMIN XD",
  "tag_line": "NA1",
  "region": "americas",
  "count": 100
}
```

`/api/intelligence/report` response shape:

```json
{
  "performance_index": 72,
  "confidence": "High",
  "win_probability_last_game": 0.64,
  "focus_goal": "Keep deaths ≤ 3",
  "top_improvements": [
    "Keep deaths ≤ 3 (Estimated +17% win chance)",
    "Target gold/min ≥ 420 (Estimated +9% win chance)",
    "Target damage/min ≥ 550 (Estimated +6% win chance)"
  ],
  "weekly_trend": "Your performance improved 8% this week.",
  "ai_feedback": "..."
}
```

## Riot Proxy (for sharing app without exposing your Riot key)

Run this service on a public HTTPS host (Render/Fly.io/Railway/AWS):

```bash
uvicorn lol_stat_tracker.proxy_server:app --host 0.0.0.0 --port 8080
```

Set proxy server environment variables:

- `RIOT_API_KEY` = your Riot key (server-only)
- `DEMO_ACCESS_TOKEN` = shared access token expected from desktop app

Set desktop/backend runtime environment variables (client side):

- `RIOT_PROXY_URL` = your proxy base URL (for example, `https://your-proxy.example.com`)
- `RIOT_PROXY_ACCESS_TOKEN` = same token as `DEMO_ACCESS_TOKEN`

Security notes:

- Proxy never returns Riot API key.
- Proxy requires `Authorization: Bearer <token>`.
- Proxy applies simple IP rate limiting.
- Keep `RIOT_API_KEY` only on server, never in installer or repo.

## Electron Desktop Shell

From `desktop/`:

```bash
npm install
npm start
```

The Electron app starts a local backend (`uvicorn`) and opens a desktop UI for sync/build/train/report actions.

## Build a Downloadable Windows App (.exe)

Prerequisite on target laptop: Python 3.11+ installed and available on PATH.

On first launch, the app automatically creates a local virtual environment and installs backend Python dependencies. This may take up to a minute.

From `desktop/`:

```bash
npm install
npm run dist:win
```

Installer outputs are written to `desktop/dist/` (NSIS installer + portable `.exe`).

After installing on a laptop, create this file for your API key:

- `%APPDATA%/LoL Coach Tracker/runtime/.env`

Contents:

```env
RIOT_API_KEY=RGAPI-...
```

## Output Artifacts

- Raw matches: `data/raw/matches/*.json`
- Raw timelines: `data/raw/timelines/*.json`
- Manifest: `data/raw/manifest.json`
- Processed dataset: `data/processed/matches.csv`
- Model: `models/win_rf.pkl`
- Metrics: `models/win_rf_metrics.json`
- Reports:
  - `reports/last_game_report.md`
  - `reports/weekly_summary.md`

## Intelligence Notes (v2)

- Performance index uses context-aware filtering (`champion+role`, fallback `role`, fallback global).
- Index uses rolling 20-game values to reduce volatility.
- Coaching targets use win-percentile thresholds (60th for increase, 40th for decrease) instead of simple win means.
- Win-state analytics include lead conversion, comeback rate, throw rate, and snowball strength.

## Intelligence Notes (v4 in progress)

- Primary post-game explainer now trains as LightGBM with a Random Forest baseline agreement signal.
- Early outlook model uses leakage-safe features focused on pre-15 minute information.
- Context hierarchy is stricter: champion+role (>=25), role (>=40), else global.
- Intelligence payload now includes counterfactual win-rate deltas, contribution percentages, and behavioral dimension scores.
