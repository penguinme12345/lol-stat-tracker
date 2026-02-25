# PRD — LoL Coach Tracker v3 (Single-User Riot API + ML Coaching)

## 1) Product Summary
LoL Coach Tracker is a personal analytics + coaching tool that syncs a player’s last N matches from Riot, builds a structured dataset (match + timeline-derived features), trains a win-outcome model, and produces:
- **Last Game Report** (what happened + top drivers + targets)
- **Weekly Summary** (trend + champion pool + win-state analytics)
- **Coaching Snapshot (v2)**: performance index (0–100), confidence, focus goal, top improvements, weekly momentum narrative

Target user (v3): **you**, midlane assassin player (Zed/Fizz/Akali vibes), wanting repeatable improvements across ~100–200 matches.

---

## 2) Goals
### Primary goals
1) **Actionable coaching** based on your own history (not generic advice)
2) **Reliable early-game coaching** using timeline @10/@15 diffs + discipline signals
3) **Clear, honest confidence + sample size** so you know when to trust it

### Success criteria
- Snapshot outputs are **consistent**, **interpretable**, and **don’t overpromise** (“+25% win chance” only if justified)
- Timeline-derived features appear in dataset for ≥95% of synced matches (with safe fallbacks)
- Weekly summary includes **lead conversion, comeback, throw rate, snowball strength** and updates as you sync

---

## 3) Current Pipeline (What you have)
### Data ingestion
- `get_puuid(gameName, tagLine)`
- `get_match_ids(puuid, count)`
- `get_match(matchId)`
- `get_match_timeline(matchId)`
- Save:
  - `RAW_DIR/{matchId}.json`
  - `TIMELINE_DIR/{matchId}.json`
  - `manifest.json` with `match_ids`, `timeline_ids`, `target_puuid`

### Dataset build
- Parse match JSON → participant stats, team stats, objective stats
- Parse timeline JSON → @10/@15 frames + event timestamps
- Write `MATCHES_CSV_PATH` sorted by timestamp

### Model training
- RandomForest pipeline:
  - Numeric passthrough
  - OneHot champion/role
  - Train/test split by time (first 80% train, last 20% test)
  - Save `MODEL_PATH` + `METRICS_PATH` (AUC/ACC + top 25 feature importances)

### Reports
- `Last Game Report` markdown
- `Weekly Summary` markdown (includes `compute_win_state_analytics`)

### Coaching v2 snapshot
- Context selection: champion+role (>=12 games) else role (>=12) else global
- Rolling value (last 20 games) compared to win-target percentile
- Generates: performance index, confidence, win prob last game, focus goal, top improvements, weekly momentum

---

## 4) What v3 Adds (Required)
### 4.1 Timeline Endpoint (New)
Add a timeline-focused interface so you can debug and extend timeline-derived coaching without rebuilding everything.

**Endpoint/Command (pick your style):**
- CLI:
  - `lol-tracker timeline --match-id <id> --minute 10 --participant self`
  - `lol-tracker timeline --match-id <id> --export raw|features`
- OR local API:
  - `GET /api/matches/{matchId}/timeline/raw`
  - `GET /api/matches/{matchId}/timeline/features`
  - `GET /api/matches/{matchId}/timeline/summary` (human-readable)

**Outputs**
1) `raw`: returns stored `TIMELINE_DIR/{matchId}.json`
2) `features`: returns `extract_timeline_features(...)` output
3) `summary`: adds extra derived insights (see §5) + sanity checks (missing frames, missing opp, etc.)

**Acceptance criteria**
- Works offline using cached JSON
- If timeline missing → returns defaults + “timeline_missing: true”
- Includes `participant_id`, inferred opponent id, and team id used

---

## 5) Extract Even More Data From Timeline JSON (Expansion Plan)
Right now you extract:
- gold/xp/cs @10
- diff @10 (vs inferred lane opponent)
- level diff @10
- ahead/behind @10 and @15 (based on gold)
- deaths near objectives (±60s of dragon/baron)
- late deaths post20
- death_rate_after_15
- time_to_first_tower_min (team tower kills)

### 5.1 Add more frame-based metrics (low risk, high value)
From `participantFrames` at minutes 5/10/15:
- `gold_5`, `xp_5`, `cs_5`, diffs vs opponent
- `gold_15`, `xp_15`, `cs_15`, diffs vs opponent
- `level_diff_15`
- **delta curves**:
  - `gold_diff_5_to_10`, `gold_diff_10_to_15`
  - same for xp/cs
These let coaching say: “you start fine then bleed at 10–15” vs only one snapshot.

### 5.2 Add event-based discipline signals (medium risk, big coaching value)
From `frames[*].events`:
- **First blood involvement**
  - `first_blood_participation` (1/0)
  - `first_death_time_min` (if you died first)
- **Recall timing**
  - `first_recall_min` (look for `ITEM_PURCHASED` patterns or `LEVEL_UP`+gold swing proxies; if too messy, skip)
- **Objective window deaths (strong coaching)**
  - Expand beyond dragon/baron:
    - `deaths_before_herald_60s`
    - `deaths_before_soul_point_60s` (if you track drake count)
- **Tower/plate windows**
  - Track turret plate events if present (some timelines/events provide plate info indirectly; if not reliable, keep it optional)
- **Ward behavior from events**
  - `wards_placed_early` (0–10)
  - `control_wards_placed_early` (0–10) (if distinguishable)
  - `wards_cleared_early` (0–10)
If the timeline doesn’t cleanly expose this for your patch/queue, mark as optional and only compute when events exist.

### 5.3 Opponent inference improvements (critical for diff accuracy)
Current opponent selection:
- match participant whose `teamPosition/individualPosition == your role` on enemy team

Add v3 logic:
1) If `teamPosition` is empty or role-match fails → infer lane opponent using early-game lane presence:
   - Use timeline participantFrames positions (if available) or cs patterns by lane (if positions not available, fallback)
2) If still ambiguous → compute diffs vs **enemy team mid-average** instead of a single opponent, and flag:
   - `opponent_inferred: "team_average" | "role_match" | "unknown"`

Acceptance criteria:
- `gold_diff_10` etc. include an `diff_reference` label to prevent silent wrong math.

---

## 6) Coaching Snapshot Improvements (Your output is good — make it smarter)
Current snapshot example:
- Performance 53/100, confidence Moderate
- Win prob last game 17%
- Focus: gold_diff_10 ≥ 656, xp_diff_10 ≥ 441, cs_diff_10 ≥ 11
- Weekly momentum + win-state metrics

### v3 upgrades
1) **Show context bucket explicitly**
- `Context: champion_role (n=18)` or `role (n=42)` or `global (n=100)`
2) **Show “current vs target vs win/loss baseline”**
For each top lever:
- `Current (rolling20): +220 gold@10`
- `Target (60th pct of wins): +656`
- `Loss avg: -140`
3) **Add early/late split**
- If your early leads are fine but you lose anyway: highlight throw rate + post15 deaths
4) **Stop repeating the same lever twice**
Your narrative currently can repeat the same improvement in “supporting notes”. Ensure top_improvements are unique.

---

## 7) The 3 Big Logic Leaks (Critiques) + Fixes (Required in v3)

### Leak #1 — “Win Probability” is mostly post-game, not coaching-actionable
**What’s happening**
Your model uses full-match features (gold_earned, damage_to_champions, kills, etc.). That makes win probability for a completed match somewhat circular: you’re predicting win with features heavily influenced by the win itself.

**Why it’s a problem**
- “Last game win probability 17%” can be misleading because after the game ends, lots of features already encode the outcome.
- It also confuses users: are we predicting pre-game? in-game@10? post-game?

**Fix (v3 requirement)**
Split into two models (or two modes):
1) **Early Model (@10/@15 Coaching Model)**  
   Features allowed: timeline + early rates only  
   Example features:
   - gold/xp/cs diffs @10/@15, level diff
   - early deaths, first tower time, objective-window deaths
   - vision early signals if available  
   Output: **Early Win Outlook** + actionable levers
2) **Post-Game Explainer (No “probability” claim)**  
   Keep RF but present as “what drove wins historically” + SHAP/importance-style explanation, not as a win predictor.

Acceptance criteria:
- Snapshot clearly labels: `Early Outlook` vs `Post-game Explanation`
- Remove “win probability last game” from the coaching snapshot OR rename to `model_score` unless it’s the early model.

---

### Leak #2 — “Estimated +25% win chance” uplift is not causal and can be wrong
**What’s happening**
Uplift is a heuristic:
- uses gap/scale and importance weights
- capped 3%–25%
- displayed like a real win-probability gain

**Why it’s a problem**
Users will trust it like science. But it’s not measuring causal effect.

**Fix (v3 requirement)**
Replace the claim with one of these (pick one):
A) **Rename it**: “Priority impact score” (0–100) and stop using “win chance” language  
B) **Empirical uplift via binning** (recommended):
- For each feature, bin into quantiles
- compute win rate per bin (within context bucket)
- uplift = win_rate(top_bin) - win_rate(bottom_bin)
- report as “historical swing in your data”
C) **Calibration step**:
- Fit a simple logistic regression on the early-model features to get smoother, calibrated probabilities

Acceptance criteria:
- Output text changes from:
  - “Estimated +25% win chance”
  - to: “Historical win-rate swing: +12% (context: midlane, last 100 games)”
- If sample size in bin is too small → show “insufficient data” instead of a number

---

### Leak #3 — Opponent matching can silently break lane-diff features
**What’s happening**
Lane opponent is inferred by role match. This fails in:
- swaps, roaming supports, weird assignments
- “UNKNOWN” roles
- off-meta lanes

**Why it’s a problem**
Your top levers are literally `gold_diff_10/xp_diff_10/cs_diff_10`. If opponent inference is wrong, your #1 coaching message becomes noise.

**Fix (v3 requirement)**
- Add `diff_reference` + `opponent_resolution_quality`
- Improve opponent inference using timeline signals (see §5.3)
- If confidence low: compute diffs vs enemy team average and say so

Acceptance criteria:
- Every diff feature includes metadata:
  - `diff_reference: role_match | team_avg | unknown`
  - `opponent_found: true/false`
- Snapshot warns if top lever is based on low-quality opponent inference

---

## 8) Reporting Requirements (Markdown v3)
### Last Game Report
- Match meta: champ/role/result, early outlook (if early model exists)
- Key stats + percentiles (keep)
- Top drivers (keep)
- Targets (keep) but must be **context-aware** and **non-duplicated**
- Add: `timeline_integrity` (timeline present, opponent resolved, frames ok)

### Weekly Summary
- Win rate trend
- Champion pool performance
- Top drivers
- Progress targets
- Win-state analytics:
  - lead_conversion_rate
  - comeback_rate
  - throw_rate
  - snowball_strength
- Add: “Where games are lost” breakdown:
  - ahead@15 but lose %
  - behind@15 but win %

---

## 9) Data Quality + Safety
### Required data checks
- Confirm `timestamp` exists and is ms
- Confirm features exist; otherwise set safe defaults (you already do this in `_load_data`)
- Add warnings if:
  - < 40 matches → confidence forced “Low”
  - wins or losses missing in context bucket → auto-fallback to global (you already do)

---

## 10) Non-Goals (v3)
- Multi-user auth
- Cloud deployment
- Perfect causal inference
- Real-time in-match coaching overlays

---

## 11) Final v3 Deliverables Checklist
- [ ] Timeline endpoint/command returning raw + features + summary
- [ ] Expanded timeline feature extraction (@5/@10/@15 + event signals where reliable)
- [ ] Opponent inference quality labels + fallback behavior
- [ ] Two-mode modeling: Early Outlook vs Post-game Explainer (or remove probability claims)
- [ ] Replace “+win chance” with empirical swing or priority score
- [ ] Snapshot includes context bucket + current vs target vs loss baseline
- [ ] Reports updated (last game + weekly) with timeline integrity + improved win-state breakdown