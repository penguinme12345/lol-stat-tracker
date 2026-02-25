PRD — LoL Stat Tracker MVP with Random Forest Win Probability Model
1) Product overview

Product name (working): LoL Coach Tracker
One-liner: A League of Legends stat tracker that pulls match data from Riot’s API, trains a personalized Random Forest model to estimate win probability, and highlights the user’s biggest “win-rate leaks” with concrete improvement targets.

2) Goals
Primary goal

Increase a user’s win rate over time by identifying statistically meaningful weak spots in their gameplay and turning them into measurable goals.

Secondary goals

Make match history + trends easy to view (champion pool, role performance, phase performance).

Provide explainable ML outputs (not “black box” vibes): feature importance + what to focus on next.

3) Non-goals (MVP)

Real-time in-game coaching/overlays (“rotate now” live calls).

Deep VOD/computer-vision analysis.

Perfect prediction accuracy (MVP focuses on actionable insights, not esports-level forecasting).

Multi-user SaaS with payments/deployment (local-first MVP is fine).

4) Target users

Solo players who want to improve and like data (ranked grinders).

Especially strong fit for players who spam a small champ pool (better signal).

5) User stories

As a player, I want to import my last 50–500 matches so I can track performance.

As a player, I want to see which stats correlate most with my wins so I know what to improve.

As a player, I want a “last game report” telling me my biggest mistakes and 1–3 goals for next game.

As a player, I want a weekly summary showing whether I’m improving in the metrics that matter.

6) MVP deliverables
A) Data ingestion (Riot API → local storage)

Input: Riot ID (gameName + tagLine), routing region (americas/europe/asia)

Fetch:

account-v1 → PUUID

match-v5 → match IDs for PUUID

match-v5 → match details JSON for each match

Store raw match JSON locally with an idempotent manifest (don’t redownload).

B) Dataset builder (raw JSON → structured table)

Parse raw match JSON into a dataset with 1 row per match for that user.

Minimum columns/features (post-game stats):

match_id, timestamp, queue_id, duration_min

champion, role/lane

win (0/1)

kills, deaths, assists, KDA

CS total, CS/min

damageToChampions, damage/min

goldEarned, gold/min

visionScore, vision/min

(optional) killParticipation, teamDamageShare if easy to compute

Output: data/processed/matches.csv (or DB table)

C) ML training: Random Forest win probability model

Train a personalized model per user (their history).

Model type: RandomForestClassifier

Target: win (binary classification)

Split: time-based (older matches train, newest 20% test) to avoid “future leakage.”

Metrics:

ROC-AUC (preferred) + accuracy (secondary)

Save artifacts:

models/win_rf.pkl

models/win_rf_metrics.json

feature importance list

D) Insight engine (turn model into coaching)

Generate:

Top 3 Win-Rate Drivers (from feature importances + user distributions)

Weak spot detection: identify “below-threshold” metrics vs user’s own wins, e.g.

“In wins: deaths/min avg = 0.08; in losses: 0.15 → primary leak.”

Actionable goals (measurable):

“Keep deaths ≤ 2 before 15”

“Hit 7 CS/min to 10”

“Vision/min above X”

Output formats (MVP):

CLI + Markdown report: reports/last_game_report.md

Weekly summary: reports/weekly_summary.md

E) Minimal UI (choose one for MVP)

CLI-first (fastest) + Markdown reports
or

Streamlit dashboard (nice but optional in MVP)

7) Product behavior
“Last Game Report”

Includes:

predicted win probability (based on that match’s feature profile)

your key stats + percentile compared to your last N games

top 2–3 improvement targets

one “focus goal” for next match

“Weekly Summary”

Includes:

win rate trend

champ pool performance

metrics trend for the top 3 win-rate drivers

whether you improved on last week’s assigned goals

8) Requirements
Functional requirements

Import matches reliably without duplicates.

Build a dataset and train the model in under ~30 seconds for a few hundred games.

Produce reports deterministically (same inputs → same results).

Non-functional requirements

Privacy: API key never committed; store data locally by default.

Reliability: handle Riot rate limits (429) with backoff; recover gracefully.

Maintainability: clean modules, type hints, docstrings; tests for parsing.

9) Data + ML design notes
Why Random Forest here

Strong baseline for tabular stats

Robust with minimal tuning

Gives feature importance for explainability

Key ML pitfalls to avoid

“Champion” can dominate predictions if one-trick; treat carefully (one-hot or limit to top champs).

Small datasets: don’t overclaim accuracy; focus on consistent insights.

Use time-based split; avoid leaking info from future games.

Evaluation success criteria (MVP)

Model trains end-to-end and outputs:

stable metrics (not NaN / broken)

sensible feature importances (e.g., deaths/min should matter)

Reports produce at least 1–3 actionable recommendations that match intuition.

10) Architecture (MVP)

Modules

riot_client — API requests, retries, routing region handling

ingest — fetch IDs, download matches, manifest

features — parse match JSON → dataframe rows

train — train RF, evaluate, save artifacts

insights — compute weak spots + goals

report — render Markdown reports

Storage

Raw: JSON files + manifest

Processed: CSV (MVP)
(DB like SQLite/Postgres can be Phase 2)

11) Risks & mitigations

Riot API key expires → document regeneration step; .env loading.

Rate limiting → caching raw JSON; backoff.

Low signal early on → require minimum N matches for “confidence” labels.

Interpretability → always show “why” (feature importance + win/loss stat deltas).

12) Milestones

M1 (Day 1–2): ingestion + raw JSON saved
M2 (Day 2–3): build dataset CSV + basic stats summaries
M3 (Day 3–4): train RF + save model + print feature importances
M4 (Day 4–5): last game report + weekly summary in Markdown
M5 (optional): Streamlit dashboard view

If you want, I can also generate:

the repo folder structure + file list for this PRD

acceptance criteria checklist (what “done” means for MVP)

a simple test plan (parser tests + “does model train” test)