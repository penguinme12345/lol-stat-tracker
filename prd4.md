📘 PRODUCT REQUIREMENTS DOCUMENT
LoL Coach Tracker v4 — Behavioral Performance Intelligence Engine
1️⃣ Product Objective

Upgrade LoL Coach Tracker into a:

Leak-free, explainable, ML-driven behavioral coaching system
That converts gameplay data into actionable, calibrated percentages
While gamifying improvement through archetypes and dynamic tags

The system must:

Avoid data leakage

Avoid circular prediction logic

Maintain interpretability

Keep backwards compatibility

Preserve UI stability

2️⃣ Model Architecture Upgrade
2.1 Core Model

Primary Model: LightGBM (binary classification)
Target: Win (1) / Loss (0)

Why:

Strong on tabular data

Handles nonlinear thresholds

Good probability estimates

Compatible with SHAP

Works well with 100–500 samples

Random Forest retained as:

Baseline sanity check

Agreement-based confidence signal

3️⃣ Preventing Data Leakage (Critical)
3.1 What is Data Leakage?

Data leakage occurs when the model learns from information that would not be available at prediction time.

Examples of leakage:

Using post-game stats to predict win

Using team total gold after game ends

Using match duration to predict win

Using features derived from win condition itself

3.2 Feature Segmentation Rules
Early Model (≤15min)

Allowed:

gold_diff_10

xp_diff_10

cs_diff_10

level_diff_10

deaths_before_15

first_blood_involvement

early_vision

early_objective_presence

Forbidden:

total_gold

total_damage

kills

final_kda

post20 stats

Full Game Model

Allowed:

all behavior metrics

all engineered indices

Forbidden:

win-derived aggregates

future-derived rolling windows

target leakage variables

3.3 Time-Based Train/Test Split

Must use chronological split:

Train = first 80% of matches
Test = last 20%

No random shuffle.

4️⃣ Model Context Explained

Model Context refers to:

The subset of historical matches used to compute baselines and percentile comparisons.

Hierarchy:

Champion + Role context (if ≥25 matches)

Role-only context (if ≥40 matches)

Global context

Context is used for:

Tier grading

Percentile targets

Tag detection

Quest thresholds

Context is NOT used for:

Target variable construction

Leakage-prone features

Context size must always be displayed.

If context < 20 matches:

Confidence auto-downgraded

Percentile thresholds revert to broader bucket

5️⃣ Actionable Percentage Engine
5.1 SHAP Contribution %

For each match:

Output:

+X% win impact from gold_diff_10

-Y% from deaths_after_15

+Z% from farming_discipline

These are local additive contributions.

5.2 Counterfactual Simulation

Simulate controlled changes:

Example:

Original:
Win probability = 58%

Modify:
deaths = 3 instead of 6

New:
Win probability = 72%

Impact:
+14%

Rules:

Only modify 1 feature at a time

Only within realistic percentile range

Never exceed observed data bounds

6️⃣ Behavioral Dimensions (ML-Derived Scores)

Replace raw metrics with model-compressed dimensions:

⚔️ Early Pressure Score

🔁 Lead Stability Score

🧠 Late Game Control Score

🎯 Objective Fight Impact Score

💥 Combat Efficiency Score

🛡️ Consistency Score

🏹 Map Presence Score

🔥 Snowball Efficiency Score

🎲 Risk Index

🧱 Clutch Reliability Score

Each scaled 0–100.

Derived from:
SHAP-weighted feature subsets.

7️⃣ Expanded Archetypes

Primary archetype determined via clustering on:

SHAP pattern vectors

Phase performance scores

Behavioral indices

Archetype Options

🗡️ High-Pressure Assassin

👑 Solo Carry

🔥 Snowball Specialist

🧠 Strategic Closer

🎯 Objective General

⚔️ Duelist

🌾 Scaling Farmer

🧱 Lead Protector

🎲 High-Variance Fighter

🏹 Map Manipulator

💣 Early Game Bully

🛡️ Teamfight Anchor

🧪 Experimental Flex

🕵️ Opportunistic Roamer

🧭 Macro Stabilizer

🧨 Glass Cannon

☠️ Late Game Risk Taker

🧊 Cold Closer

🔮 Comeback Specialist

🎮 Chaos Controller

Only 1 Primary.
Up to 3 Secondary.

8️⃣ Massive Player Tag Expansion

Tags must be:

Data-backed

Threshold-based

Context-aware

Auto-expiring

8.1 Strength Tags

🔥 Lane Dominator

🧨 Burst Specialist

🏹 Objective Hunter

🧠 Vision Controller

💥 Damage Engine

🎯 Precision Finisher

🧱 Stable Backbone

🔥 Momentum Rider

🧭 Map Pressure Creator

🎮 Snowball Architect

🛡️ Low Death Specialist

🧠 Cool Under Pressure

🗡️ Solo Kill Artist

📈 Scaling Monster

🏆 Gold Efficiency Master

🎯 CS Perfectionist

🧊 Late Game Stabilizer

🏹 Baron Closer

🐉 Dragon Closer

💎 High Value Farmer

8.2 Risk / Weakness Tags

☠️ Late Game Risk

⚠️ Lead Gambler

🎲 Coinflip Fighter

🪤 Lead Dropper

🐉 Dragon Donor

🧭 Map Neglect

🔥 Overforce Habit

💥 Damage Without Conversion

🧨 Glass Cannon

⏳ Slow Rotator

🧠 Vision Blindspot

⚔️ Overextend Tendency

🧊 Passive Mid Game

💀 Baron Window Risk

🎯 CS Drop Off Post-15

📉 Inconsistent Snowball

🧱 Hesitation After Lead

🎮 Overconfidence Spike

8.3 Momentum Tags

📈 Climbing

❄️ Slump Detected

🔥 Hot Streak

🧊 Cooling Off

🏁 Consistency Streak

🎯 Quest Completion Streak

9️⃣ Tier Rating System (Rebuilt)

Each behavioral dimension graded:

🔴 Weak

🟡 Developing

🟢 Strong

🔥 Elite

💎 Exceptional

Tier is determined by:

Context percentile

Stability over last 20 matches

Variance weighting

If context small:
Tier confidence indicator added.

🔟 Niche Improvement Engine

Instead of generic “reduce deaths,” system generates:

Reduce deaths after 15 only when ahead

Convert first kill into plate within 2 minutes

Avoid death within 60s of 2nd dragon

Push wave before roaming

Increase early recall timing efficiency

Maintain gold_diff between 10–15

Improve lead extension after turret

Reduce damage taken per gold spent

Increase kill participation when ahead

Avoid side-lane isolation deaths

Improve objective presence when snowballing

Reduce unnecessary baron contest deaths

Increase CS retention after first back

Limit overchasing after winning fight

Maintain death ≤3 in 3 consecutive games

These are generated from SHAP weakness ranking.

1️⃣1️⃣ Safety + Stability

No raw stat replaced without being stored

All ML-derived metrics logged

All predictions bounded within historical min/max

Calibration curve checked quarterly

RF used to monitor divergence

Model drift detection on rolling AUC

1️⃣2️⃣ Final Philosophy

System must feel:

🎮 Fun
🧠 Intelligent
📊 Legitimate
🔥 Competitive

But:

No black box magic
No fake percentages
No inflated win chance claims

All actionable percentages must be:

Empirically grounded

Counterfactually validated

Context-aware

Confidence-weighted