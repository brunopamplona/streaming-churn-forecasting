# 📉 Streaming Churn Forecasting — Behavioral Churn Model with Intervention Optimization

> **End-to-end churn prediction pipeline** with behavioral engagement features, cohort-level calibration, and intervention threshold optimization — built on a synthetic dataset inspired by a real Brazilian streaming platform with 1.2M+ MAU.

---

## Overview

This project demonstrates a production-grade churn modeling workflow for a subscription streaming business. The model identifies subscribers at risk of cancellation before the renewal decision, quantifies the behavioral signals driving disengagement, and determines the optimal intervention threshold that maximizes retention ROI — balancing the cost of false positives (wasted offers) against the revenue impact of missed churners.

**Why this matters for streaming businesses:** Churn in SVOD is structurally different from e-commerce attrition. Subscribers don't "abandon carts" — they quietly disengage weeks before canceling. Models that only react to the cancellation event are already too late. This pipeline targets the **disengagement window**, enabling proactive retention at the moment it's still actionable.

---

## Key Features

- **Behavioral Feature Engineering** — Recency, frequency, and depth-of-engagement features derived from session-level activity (days active, content diversity, binge sessions, login gaps)
- **Cohort-Level Calibration** — Probability calibration applied per subscription plan cohort to correct for class imbalance and plan-specific churn base rates
- **Intervention Threshold Optimization** — Custom decision threshold tuned to maximize expected retention value, not just classification accuracy
- **Survival Analysis Layer** — Time-to-churn estimation (Kaplan-Meier + Cox PH) to support cohort lifecycle understanding
- **Feature Importance & SHAP** — Global and local explainability to identify actionable churn drivers per segment

---

## Model Architecture

```
Raw Event / Subscription Data
          │
          ▼
Feature Engineering (rolling windows: 7d, 14d, 30d)
          │
          ▼
Train / Validation Split (time-based, no leakage)
          │
          ▼
Gradient Boosting Classifier (XGBoost / LightGBM)
          │
          ├──► Cohort Calibration (Platt scaling per plan)
          │
          ├──► Threshold Optimizer (maximize retention ROI)
          │              │
          │              └──► Expected Value = P(churn) × LTV − cost(intervention)
          │
          └──► SHAP Explainability (global importance + subscriber-level drivers)
```

---

## Dataset

The dataset is **fully synthetic**, generated to reflect realistic behavioral patterns of a Brazilian SVOD platform across tiered subscription plans:

| Plan | Price | Base Churn Rate | Key Churn Signals |
|---|---|---|---|
| Família | R$ 80/mo | ~4% | Drop in co-viewing sessions, content diversity decline |
| Premiere (Football) | R$ 60/mo | ~7% | Post-season disengagement, login gap spikes |
| Telecine (Movies) | R$ 38/mo | ~9% | Catalog exhaustion pattern, reduced weekend sessions |
| HBO Max bundle | R$ 35/mo | ~11% | Short tenure, low content breadth, price sensitivity |

Behavioral signals simulated: **session frequency, days active (L7/L14/L30), content categories consumed, binge flags, login gap (days since last session), device switches, support contacts, plan upgrade/downgrade history**

---

## Feature Engineering

| Feature Group | Features |
|---|---|
| Recency | `days_since_last_session`, `login_gap_trend` |
| Frequency | `sessions_l7`, `sessions_l14`, `sessions_l30`, `active_days_l30` |
| Engagement Depth | `avg_watch_time`, `binge_sessions_l14`, `content_categories_l30` |
| Content Diversity | `unique_titles_l30`, `genre_entropy` |
| Account Signals | `days_since_signup`, `plan_changes`, `support_contacts_l30` |
| Billing | `payment_failures`, `days_to_renewal` |

> All features computed as of a **prediction point** set 14 days before the renewal date, targeting the actionable intervention window.

---

## Results Snapshot

| Metric | Value |
|---|---|
| AUC-ROC | 0.87 |
| Precision @ optimal threshold | 72% |
| Recall @ optimal threshold | 68% |
| Intervention threshold | 0.38 |
| Expected retention lift vs. no model | +18% |

**Top churn drivers (SHAP):**
1. `login_gap_trend` — accelerating gap between sessions
2. `content_categories_l30` — narrowing content breadth
3. `days_to_renewal` — proximity to billing date
4. `sessions_l14` — recent engagement drop
5. `plan` — HBO bundle highest baseline risk

---

## Threshold Optimization

Standard classifiers optimize for accuracy or F1. For retention interventions, the right objective is **expected value**:

```
EV(threshold) = Σ [P(churn|subscriber) > threshold] ×
                    [P(churn) × LTV − (1 − P(churn)) × offer_cost]
```

The optimizer sweeps across thresholds and selects the point that maximizes net retention revenue, explicitly penalizing false positives (unnecessary discount offers sent to subscribers who would have stayed).

---

## Project Structure

```
streaming-churn-forecasting/
│
├── data/
│   ├── generate_synthetic_data.py       # Behavioral dataset generator
│   └── streaming_churn_dataset.csv      # Generated dataset
│
├── notebooks/
│   ├── 01_eda_cohort_analysis.ipynb     # Churn rates by plan, tenure, engagement
│   ├── 02_feature_engineering.ipynb     # Rolling window features, leakage audit
│   ├── 03_model_training.ipynb          # XGBoost training, calibration, CV
│   ├── 04_threshold_optimization.ipynb  # Expected value threshold sweep
│   └── 05_shap_explainability.ipynb     # Global importance + subscriber-level SHAP
│
├── src/
│   ├── features.py                      # Feature engineering pipeline
│   ├── model.py                         # Training, calibration, evaluation wrapper
│   └── threshold_optimizer.py           # EV-based threshold optimizer
│
├── outputs/
│   ├── roc_curve.png
│   ├── threshold_ev_curve.png
│   ├── shap_summary.png
│   └── cohort_churn_rates.png
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Feature Engineering | Python, pandas, NumPy |
| Modeling | XGBoost, LightGBM, scikit-learn |
| Calibration | `CalibratedClassifierCV` (Platt scaling) |
| Survival Analysis | lifelines (Kaplan-Meier, Cox PH) |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn, Plotly |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/streaming-churn-forecasting.git
cd streaming-churn-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic dataset
python data/generate_synthetic_data.py

# 4. Run notebooks in order
jupyter notebook notebooks/
```

---

## Methodology Notes

**Time-based train/validation split**
The dataset is split by subscription cohort date — future cohorts are never used to train the model, preventing temporal leakage that would inflate performance metrics on subscriber behavior data.

**Cohort calibration rationale**
Raw XGBoost probabilities are poorly calibrated, especially under class imbalance. Platt scaling is applied per subscription plan cohort because each plan has a structurally different churn base rate — a 0.5 probability means different things for a Família subscriber vs. an HBO bundle subscriber.

**Identified Limitations**
- No causal identification: the model predicts churn risk but cannot estimate the true incremental effect of an intervention (offer) on that subscriber — a holdout experiment would be required for that
- Session-level granularity is simulated; real pipelines would require event stream ingestion (Kafka, Kinesis)
- Survival analysis layer is descriptive; no competing risks model implemented

---

## Roadmap

- [ ] Uplift modeling layer (causal forest) to estimate intervention incrementality per subscriber
- [ ] Real-time scoring pipeline (FastAPI + feature store integration)
- [ ] A/B test framework for intervention strategy validation
- [ ] Multi-touch retention journey modeling (sequence of interventions)

---

## Context

This project is part of a **Growth & Marketing Science portfolio** built to demonstrate applied ML in a subscription business context. The churn modeling problem was chosen for its direct connection to LTV management, retention budget allocation, and growth forecasting — core responsibilities in senior Growth Data Science and Marketing Science roles.

---

## Author

**Bruno** — Technical Architect & Decision Science Lead  
Focused on Growth Analytics, churn modeling, and causal inference for subscription businesses.

[LinkedIn](https://linkedin.com/in/your-profile) • [GitHub](https://github.com/your-username)

---

*Synthetic data. All figures are illustrative and do not represent any real company's performance.*

