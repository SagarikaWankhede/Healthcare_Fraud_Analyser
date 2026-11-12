# Healthcare Provider Fraud Detection & Audit Prioritization System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-red.svg)](https://xgboost.readthedocs.io/)

> **Executive Summary:** An end-to-end machine learning system built on **558,211 Medicare insurance claims** across **5,410 healthcare providers** that solves the constrained audit resource dilemma. Instead of stopping at classification accuracy, this system optimizes for **expected fraudulent dollars recovered** under a fixed 100-provider audit budget, recovering **93.75% of all fraudulent reimbursement** ($67.48M) compared to **9.54%** for random audits.

---

## 📌 The Real-World Problem & Novelty

In healthcare insurance fraud detection:
1. **Severe Imbalance:** Only ~9.35% of providers are fraudulent. A naive model predicting "no fraud" achieves 90.65% accuracy while recovering \$0.
2. **The "High-Probability, Zero-Dollar" Trap:** Auditing a small clinic with a 99% fraud probability that only billed \$200 wastes a human auditor's time. Auditing a large hospital system with an 80% fraud probability that billed \$30,000,000 can recover millions.
3. **Capacity Constraints:** Insurance special investigation units (SIUs) can only audit a tiny fraction of providers (e.g., 100 per year).

### The Solution: Expected Dollar Recovery Ranking
We rank providers by **Expected Fraud Dollars Recovered**:

$$\text{Priority Score}_i = P(\text{Fraud}_i) \times \text{Total Reimbursement}_i$$

The audit team simply works down the ranked list from rank 1 to 100.

---

## 🔬 Multi-Technique Methodology (5 Syllabus Pillars)

```
[4 Raw Tables: 558,211 Claims]
              │
              ▼
    [Data Integration & Cleaning] ──► (Inpatient, Outpatient, Beneficiary, Labels)
              │
              ▼
   [Provider-Level Train/Test Split] ──► (80/20 Stratified by Provider, Zero Leakage)
              │
              ├─► 1. Preprocessing: 48 volume, financial, demographic, doctor network features
              ├─► 2. Regression: HistGradientBoosting predicts standard claim cost; computes overbilling residuals
              ├─► 3. Clustering: Unsupervised Hierarchical (4 clinic scales) + DBSCAN (density anomaly flags)
              └─► 4. Association Rules: In-fold Apriori discovers suspicious diagnosis/procedure code pairs
              │
              ▼
   [Enriched Provider Matrix] ──► (64 synthesized features)
              │
              ▼
    [Classification Benchmark] ──► (Logistic Regression, Random Forest, AdaBoost, XGBoost)
              │
              ▼
    [Audit Resource Optimization] ──► (Top-100 Budget Showdown vs. Random & Volume Baselines)
```

### 1. Preprocessing & Relational Aggregation
* **Integrity Guarantee:** Partitioning is performed strictly at the **Provider level** (4,328 train, 1,082 test). No single provider's claims ever cross the train/test boundary.
* Aggregates 558K individual bills into 48 provider metrics: claim volume, inpatient/outpatient ratios, chronic disease prevalence (11 CMS conditions), patient age profiles, physician network density, and same-physician billing behavior.

### 2. Regression: Claim Cost & Overbilling Residuals
* Trains a `HistGradientBoostingRegressor` on claim-level attributes ($R^2 = 0.5763$) to estimate the standard expected reimbursement:
  $$\text{Residual} = \text{Actual Reimbursement} - \widehat{\text{Expected Cost}}$$
* Providers whose claims consistently generate positive residuals receive high "excess reimbursement" penalty scores.

### 3. Clustering: Unsupervised DBSCAN & Hierarchical Typology
* **Hierarchical Clustering (Ward linkage):** Automatically groups providers into 4 operational archetypes (small walk-in clinics, medium practices, diagnostic centers, large hospital systems).
* **DBSCAN:** Flags isolated behavioral anomalies that do not conform to any standard cluster density.

### 4. Association Rule Mining: In-Fold Apriori Code Combos
* Formulates claims as transaction baskets of diagnosis codes (`D_*`) and procedure codes (`P_*`).
* **Leakage-Free Execution:** Mined *strictly within the training fold* using fraud labels to isolate code combinations over-represented in fraudulent claims (e.g., `D_2851` + `D_4019` with **2.52x Lift**).
* Calculates how many claims each provider submits with these flagged combinations.

### 5. Supervised Classification
* Compares 4 algorithms with balanced class weights:
  * **Logistic Regression** (L2-regularized with feature standardization)
  * **Random Forest** (150 trees, max depth 12)
  * **AdaBoost** (100 sequential weak learners)
  * **XGBoost** (`scale_pos_weight` tuned to negative-to-positive ratio)

---

## 📊 Experimental Results

### 1. Classification Benchmark (Held-Out Test Set: 1,082 Providers)

| Model | ROC-AUC | PR-AUC (Avg Precision) | F1 Score | Precision | Recall | True Positives | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.9684** | **0.7654** | **0.6584** | 56.3% | 79.2% | **80 / 101** | 62 | 21 |
| **XGBoost** | 0.9666 | 0.7591 | 0.6780 | 59.3% | 79.2% | 80 / 101 | 55 | 21 |
| **Logistic Regression** | 0.9641 | 0.7585 | 0.5987 | 44.8% | **90.1%** | 91 / 101 | 112 | 10 |
| **AdaBoost** | 0.9642 | 0.7543 | 0.6400 | **75.7%** | 55.4% | 56 / 101 | 18 | 45 |

---

### 2. The 100-Provider Audit Showdown (The Core Novelty)

Under a fixed budget where the insurer can only investigate **100 providers** out of 1,082 test candidates:

| Audit Strategy | Fraud Caught (of 101) | Precision @ 100 | Fraud Dollars Recovered | **Fraud Capture Rate** | Total Payout Audited | **Audit Yield** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed ML System (Expected Recovery)** | **72 / 101** | **72.0%** | **$67,478,710.00** | **93.75%** | $82,160,090.00 | **82.1%** |
| **Highest Billers (Naive Heuristic)** | 69 / 101 | 69.0% | $67,143,690.00 | 93.28% | $83,269,890.00 | 80.6% |
| **Pure Risk Ranking (Probability-Only)** | 71 / 101 | 71.0% | $66,199,110.00 | 91.97% | $79,560,200.00 | 83.2% |
| **Random Selection (Chance Baseline)** | 9.3 / 101 | 9.3% | $6,865,598.40 | **9.54%** | $11,614,320.70 | 59.1% |

> **Key Business Takeaways:**
> - The proposed ML system captured **93.75% of all fraudulent reimbursement** ($67.48M out of $71.98M) while auditing only **9.2% of providers**.
> - **Outperformed Random Selection by 9.8x** in dollar recovery ($67.5M vs. $6.9M).
> - **Outperformed the "Highest Billers" heuristic** by capturing more true fraudulent providers (72 vs. 69) and generating a higher audit yield (82.1% vs. 80.6%), avoiding false alarms on honest high-volume hospitals.

---

### 3. Multi-Technique Feature Ablation Study

| Pipeline Stage | Features | PR-AUC | ROC-AUC | Top-100 Providers Caught | Top-100 Dollars Recovered | Dollar Capture Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Base Billing & Demographics** | 48 | 0.7538 | 0.9660 | 72 | $67,459,740.00 | 93.72% |
| **2. + Regression Residuals** | 54 | 0.7699 | 0.9674 | 71 | $67,218,240.00 | 93.39% |
| **3. + Clustering (DBSCAN/Hierarchical)** | 59 | 0.7694 | 0.9662 | 71 | $67,218,240.00 | 93.39% |
| **4. Full Pipeline (+ Apriori Rules)** | **62** | **0.7742** | **0.9674** | **72** | **$67,459,740.00** | **93.72%** |

*Adding overbilling residuals and Apriori association rules increased PR-AUC from **0.7538 to 0.7742 (+2.04% gain)**, sharpening risk separation.*

---

## 📈 Generated Visualizations

All high-resolution figures are exported to `outputs/figures/`:
1. `recovery_curve_comparison.png`: Cumulative dollar recovery across audit budgets (1 to 250 providers).
2. `roc_pr_curves.png`: Comparative ROC curves and Precision-Recall curves across all 4 classifiers.
3. `feature_importance.png`: Top 15 predictive features ranked by information gain.
4. `clustering_outliers_tsne.png`: 2D PCA projection of DBSCAN density anomalies and Hierarchical operational clusters.

---

## 📁 Repository Structure

```
Healthcare_Fraud_Analyser/
├── data/
│   ├── raw/                      # Inpatient, Outpatient, Beneficiary, Labels CSVs
│   └── processed/                # Processed train/test feature matrices (64 columns)
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Relational merging & date cleaning
│   ├── feature_engineering.py    # 48 provider volume & demographic aggregations
│   ├── regression_residuals.py   # HistGradientBoosting claim cost & residual model
│   ├── clustering.py             # DBSCAN anomaly detector & Hierarchical clustering
│   ├── apriori_mining.py         # Leakage-free in-fold diagnosis/procedure miner
│   ├── classifiers.py            # Model training, calibration, and benchmarking
│   ├── audit_ranking.py          # Expected recovery scoring & baseline simulation
│   └── ablation.py               # Incremental technique evaluation
├── outputs/
│   ├── figures/                  # Publication-ready plots (.png)
│   ├── reports/                  # Detailed CSV audit manifest & benchmark tables
│   └── models/                   # Serialized best trained model (.joblib)
├── requirements.txt              # Pinned Python package dependencies
├── run_pipeline.py               # Master end-to-end execution script
└── README.md                     # Comprehensive project documentation
```

---

## 🚀 How to Run

### 1. Clone & Activate Environment
```bash
git clone https://github.com/SagarikaWankhede/Healthcare_Fraud_Analyser.git
cd Healthcare_Fraud_Analyser

# Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Master Pipeline
```bash
python run_pipeline.py
```
*The full pipeline will run data loading, feature engineering, regression, clustering, Apriori mining, classifier benchmarking, audit simulation, and ablation in ~35 seconds.*

### 3. Review Deliverables
* Check `outputs/reports/top_100_audit_manifest.csv` for the prioritized audit checklist with Provider IDs, fraud risk scores, expected recovery values, and primary anomaly reason codes.
* Check `outputs/figures/` for all comparative plots.

---

## 💼 Exact Resume Bullet

> **“Built an audit-prioritization system on 558K insurance claims combining association-rule mining, density-based anomaly detection and gradient boosting; recovered 93.8% of fraudulent reimbursement at a 100-provider audit budget vs. 9.5% for random selection.”**
