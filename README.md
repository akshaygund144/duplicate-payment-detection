# 🔍 Duplicate Payment Detection & Causal Analysis
### Enterprise AP Analytics | Difference-in-Differences · Chi-Square · Rule-Based Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-green)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Statistics-orange)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

In enterprise ERP environments, **duplicate payments** — where the same vendor invoice is settled more than once — are a persistent source of financial leakage. They spike most sharply around **system migration events**, when vendor master data is re-seeded and approval controls are temporarily relaxed.

This project presents a **production-grade analytical pipeline** that:

1. **Detects** duplicate payments using SQL-equivalent window function logic (implemented in Pandas)
2. **Causally attributes** post-migration spikes using a **Difference-in-Differences (DiD)** quasi-experimental framework
3. **Validates** statistical significance via chi-square tests and bootstrap confidence intervals (1,000 iterations)
4. **Quantifies** financial exposure and surfaces the highest-risk vendor segments

The analysis replicates a real-world problem solved at an enterprise scale — all data is synthetic but structurally faithful to Oracle ERP AP datasets.

---

## 📊 Key Results

| Metric | Value |
|---|---|
| **DiD Causal Estimate** | +11.1 percentage points |
| **95% Bootstrap CI** | [9.5pp, 12.8pp] — excludes zero |
| **Chi-Square p-value** | < 0.001 ✅ |
| **Detection Precision** | 99.7% |
| **Detection Recall** | 82.3% |
| **F1 Score** | 90.2% |
| **Financial Exposure Flagged** | $19.8M out of $23.5M true exposure |
| **Highest-Risk Segment** | IT Services × Affected BUs (24.2% post-migration) |

---

## 🏗️ Project Structure

```
duplicate_payment_detection/
│
├── data/
│   ├── ap_transactions.csv                   # Full synthetic AP dataset (15,387 rows)
│   ├── ap_transactions_pre_migration.csv     # Pre-migration subset
│   ├── ap_transactions_post_migration.csv    # Post-migration subset
│   ├── summary_by_bu_period.csv              # Aggregated duplicate rates by BU & period
│   └── ap_transactions_scored.csv            # Detection output with duplicate scores
│
├── src/
│   ├── generate_data.py                      # Synthetic AP data generation
│   ├── detection_pipeline.py                # Rule-based duplicate scoring (R1/R2/R3)
│   ├── causal_analysis.py                   # DiD + Bootstrap CI + Chi-Square
│   └── visualizations.py                    # All 5 report figures
│
├── notebooks/
│   └── analysis_codebook.ipynb              # Full analysis codebook (open here first)
│
├── reports/
│   ├── fig1_monthly_trend.png               # Duplicate rate over time: treated vs control
│   ├── fig2_did_bars.png                    # DiD bar chart
│   ├── fig3_category_heatmap.png            # Vendor category × BU heatmap
│   ├── fig4_score_distribution.png          # Duplicate score distribution
│   ├── fig5_financial_exposure.png          # Financial exposure summary
│   └── analysis_summary.json               # Machine-readable results
│
├── run_analysis.py                           # ⚡ Master orchestration script
├── requirements.txt                          # Python dependencies
└── README.md
```

---

## ⚙️ Methodology

### Detection Pipeline (3-Rule Scoring)

Replicates SQL window function logic in Pandas:

```sql
-- SQL equivalent of Rule R1
SELECT *,
  LAG(transaction_date) OVER (
    PARTITION BY vendor_id, amount 
    ORDER BY transaction_date
  ) AS prev_txn_date
FROM ap_transactions
HAVING DATEDIFF(day, prev_txn_date, transaction_date) BETWEEN 0 AND 3;
```

| Rule | Logic | Weight |
|---|---|---|
| **R1** | Same vendor + exact amount + date within 3 days | 1 |
| **R2** | Same vendor + same invoice number (hard match) | 1 |
| **R3** | Same vendor + amount within 1% tolerance + date within 7 days | 1 |

**Duplicate Score** = R1 + R2 + R3. Score ≥ 2 → predicted duplicate.

---

### Causal Framework: Difference-in-Differences

| | Pre-Migration | Post-Migration | Δ |
|---|---|---|---|
| **Treated BUs** (BU_North, BU_South) | 2.97% | 15.15% | +12.19pp |
| **Control BUs** (Others) | 3.40% | 4.47% | +1.07pp |
| | | **DiD Estimate** | **+11.11pp** |

The DiD isolates the **causal effect of the migration** by differencing out any time trend common to both groups.

**Key assumption (Parallel Trends):** Both groups followed similar duplicate rate trends pre-migration. Visual inspection (Fig 1) confirms this holds.

---

### Statistical Validation

- **Chi-square test**: χ²(1) = 275.5, p < 0.001 → reject H₀ of independence
- **Bootstrap CI** (1,000 iterations): 95% CI = [9.5pp, 12.8pp] — entirely above zero

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/akshaygund144/duplicate-payment-detection.git
cd duplicate-payment-detection
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python run_analysis.py
```

This generates all datasets, scores transactions, runs causal analysis, and produces all figures in one command.

### 3. Open the Codebook

```bash
jupyter notebook notebooks/analysis_codebook.ipynb
```

Step through the full analysis with explanations, worked calculations, and inline visualizations.

---

## 📦 Dependencies

```
pandas>=2.0
numpy>=1.24
scipy>=1.11
matplotlib>=3.7
seaborn>=0.12
nbformat>=5.9
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📈 Visualizations

### Fig 1 — Monthly Duplicate Rate: Treated vs Control
Treated BUs spike sharply post-migration while control BUs remain flat — supporting the parallel trends assumption.

### Fig 2 — DiD Bar Chart
Side-by-side comparison of pre/post rates per group. The divergence between groups is the DiD causal estimate.

### Fig 3 — Vendor Category Heatmap
IT Services in treated BUs post-migration reaches 24.2% — 8× the baseline. Root cause: vendor master records not re-validated post-migration.

### Fig 4 — Duplicate Score Distribution
Most transactions score 0 (clean). Score ≥ 2 transactions are flagged for review.

### Fig 5 — Financial Exposure
The model flags $19.8M of $23.5M true duplicate exposure at 99.7% precision.

---

## 💡 Business Recommendations

1. **Implement mandatory vendor master data validation** within 30 days of any ERP migration event
2. **Prioritise IT Services vendors** in affected BUs for immediate post-migration audit
3. **Deploy this pipeline as a scheduled job** for the first 90 days post-migration
4. **Add a vendor re-submission cooldown** in AP workflow: block same-vendor + same-amount payments within 7 days unless explicitly approved

---

## 📚 References & Sources

- Angrist, J. & Pischke, J.S. (2009). *Mostly Harmless Econometrics.* Princeton University Press.
- Card, D. & Krueger, A. (1994). Minimum Wages and Employment. *American Economic Review.* — Seminal DiD paper.
- Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.
- [scipy.stats.chi2_contingency](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)
- Oracle ERP AP data model — all figures, companies, and amounts in this project are **synthetic and fictional**.

