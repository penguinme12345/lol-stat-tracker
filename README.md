# LoL Coach Tracker

Local-first League of Legends analytics app with:

- Python ML pipeline (ingest, features, Random Forest training, insights)
- FastAPI backend for desktop integration
- Electron desktop shell UI

## Python Setup

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set:

- `RIOT_API_KEY` (used by backend and desktop sync)
- `TRACKER_PUUID` (optional; only needed if automatic inference picks the wrong player)

## CLI Usage

```bash
python main.py ingest --game-name "<gameName>" --tag-line "<tagLine>" --region americas --count 100
python main.py build-dataset
python main.py train
python main.py report
```

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
- `GET /last-game`
- `GET /weekly`
- `GET /metrics`

`/sync` request body example:

```json
{
  "game_name": "SEAMIN DEAMIN XD",
  "tag_line": "NA1",
  "region": "americas",
  "count": 100
}
```

## Electron Desktop Shell

From `desktop/`:

```bash
npm install
npm start
```

The Electron app starts a local backend (`uvicorn`) and opens a desktop UI for sync/build/train/report actions.

## Build a Downloadable Windows App (.exe)

Prerequisite on target laptop: Python 3.11+ installed and available on PATH.

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
- Manifest: `data/raw/manifest.json`
- Processed dataset: `data/processed/matches.csv`
- Model: `models/win_rf.pkl`
- Metrics: `models/win_rf_metrics.json`
- Reports:
  - `reports/last_game_report.md`
  - `reports/weekly_summary.md`
