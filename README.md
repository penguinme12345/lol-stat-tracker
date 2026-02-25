# LoL Coach Tracker MVP

Local-first League of Legends stat tracker that ingests Riot match data, builds a per-match dataset, trains a personalized Random Forest win model, and generates coaching reports.

## Setup

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set:

- `RIOT_API_KEY`
- `TRACKER_PUUID` (optional; only needed if automatic inference picks the wrong player)

## CLI flow

```bash
python main.py ingest --game-name <gameName> --tag-line <tagLine> --region americas --count 100
python main.py build-dataset
python main.py train
python main.py report
```

## Outputs

- Raw match JSON: `data/raw/matches/*.json`
- Manifest: `data/raw/manifest.json`
- Processed dataset: `data/processed/matches.csv`
- Model: `models/win_rf.pkl`
- Metrics + feature importance: `models/win_rf_metrics.json`
- Reports:
  - `reports/last_game_report.md`
  - `reports/weekly_summary.md`

