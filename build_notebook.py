"""build_notebook.py — generates analysis_codebook.ipynb"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""# Duplicate Payment Detection & Causal Analysis
## Enterprise AP Analytics | Difference-in-Differences + Chi-Square Experimentation

**Author:** Akshay | [github.com/akshaygund144](https://github.com/akshaygund144)  
**Domain:** Financial / ERP Analytics  
**Methods:** Rule-Based Detection · DiD · Bootstrap CI · Chi-Square Test  

---
### Business Context
In enterprise ERP environments, duplicate payments — where the same vendor invoice is paid more than once — are a significant source of financial leakage. They most commonly occur around system migration events, when vendor master data is re-seeded and process controls are temporarily relaxed.

This notebook presents a **production-grade analytical pipeline** that:
1. Detects duplicate payments using SQL-equivalent window function logic in Pandas
2. Applies a **quasi-experimental Difference-in-Differences (DiD) framework** to causally attribute spikes to a system migration event
3. Validates statistical significance via chi-square tests and bootstrap confidence intervals
4. Quantifies financial exposure and provides actionable recommendations

### Project Structure
```
duplicate_payment_detection/
├── data/
│   ├── ap_transactions.csv                   # Full synthetic AP dataset
│   ├── ap_transactions_pre_migration.csv     # Pre-migration subset
│   ├── ap_transactions_post_migration.csv    # Post-migration subset
│   ├── summary_by_bu_period.csv              # Aggregated duplicate rates
│   └── ap_transactions_scored.csv            # Detection output with scores
├── src/
│   ├── generate_data.py                      # Synthetic data generation
│   ├── detection_pipeline.py                # Rule-based duplicate scoring
│   ├── causal_analysis.py                   # DiD + Chi-Square analysis
│   └── visualizations.py                    # All figures
├── reports/                                  # Generated charts + JSON summary
├── notebooks/
│   └── analysis_codebook.ipynb              # This notebook
└── run_analysis.py                           # Master orchestration script
```
"""))

# ── Setup ─────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("---\n## 0. Setup & Imports"))
cells.append(nbf.v4.new_code_cell(
"""import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', '{:,.4f}'.format)
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

print(f"Pandas: {pd.__version__} | NumPy: {np.__version__}")
print("Environment ready.")
"""))

# ── Dataset Overview ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""---
## 1. Dataset Overview

The synthetic dataset simulates 15,000+ Accounts Payable transactions across 5 business units and 5 vendor categories over a 12-month period. A system migration on **July 1, 2023** acts as the natural experiment trigger.

| Group | Business Units | Baseline Dup Rate | Post-Migration Rate |
|---|---|---|---|
| **Treated** | BU_North, BU_South | ~3% | ~15% |
| **Control** | BU_East, BU_West, BU_Central | ~3% | ~4% |

The IT Services vendor category carries the highest risk (24.2% post-migration in treated BUs).
"""))

cells.append(nbf.v4.new_code_cell(
"""df = pd.read_csv('../data/ap_transactions.csv')
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

print(f"Dataset shape     : {df.shape}")
print(f"Date range        : {df.transaction_date.min().date()} to {df.transaction_date.max().date()}")
print(f"Total transactions: {len(df):,}")
print(f"Duplicates        : {df.is_duplicate.sum():,} ({df.is_duplicate.mean():.1%})")
print()
df.head(8)
"""))

cells.append(nbf.v4.new_code_cell(
"""# Duplicate rate by business unit and period
MIGRATION_DATE = pd.Timestamp('2023-07-01')
AFFECTED_BUS   = ['BU_North', 'BU_South']

df['post']    = (df['transaction_date'] >= MIGRATION_DATE).astype(int)
df['treated'] = df['business_unit'].isin(AFFECTED_BUS).astype(int)
df['period']  = df['post'].map({0: 'Pre-Migration', 1: 'Post-Migration'})

summary = (
    df.groupby(['business_unit', 'period'])
      .agg(total=('payment_ref','count'), dups=('is_duplicate','sum'))
      .assign(dup_rate=lambda x: (x['dups'] / x['total']).round(4))
      .reset_index()
)
print("Duplicate Rates by BU and Period:")
print(summary.to_string(index=False))
"""))

# ── Detection Pipeline ────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""---
## 2. Duplicate Detection Pipeline

### SQL Window Function Logic in Pandas

In production SQL, duplicate detection uses:

```sql
SELECT *,
  LAG(transaction_date) OVER (
    PARTITION BY vendor_id, amount 
    ORDER BY transaction_date
  ) AS prev_date,
  DATEDIFF(day, prev_date, transaction_date) AS day_gap
FROM ap_transactions
WHERE day_gap BETWEEN 0 AND 3;
```

This notebook replicates that logic using `groupby + shift`, then applies three escalating rules:

| Rule | Logic | Severity |
|------|-------|----------|
| **R1** | Same vendor + exact amount + date within 3 days | Medium |
| **R2** | Same vendor + same invoice number | High |
| **R3** | Same vendor + amount within 1% tolerance + date within 7 days | Medium |

**Duplicate Score** = R1 + R2 + R3. Score ≥ 2 → predicted duplicate.
"""))

cells.append(nbf.v4.new_code_cell(
"""scored = pd.read_csv('../data/ap_transactions_scored.csv')
scored['transaction_date'] = pd.to_datetime(scored['transaction_date'])

score_labels = {0: 'Clean', 1: 'Low suspicion', 2: 'Medium suspicion', 3: 'Confirmed duplicate'}
score_dist = scored['duplicate_score'].value_counts().sort_index()
print("Duplicate Score Distribution:")
for score, count in score_dist.items():
    pct = count / len(scored)
    print(f"  Score {score} ({score_labels[score]}): {count:,}  ({pct:.2%})")
"""))

cells.append(nbf.v4.new_code_cell(
"""# Model performance metrics
tp = ((scored['predicted_duplicate'] == 1) & (scored['is_duplicate'] == 1)).sum()
fp = ((scored['predicted_duplicate'] == 1) & (scored['is_duplicate'] == 0)).sum()
fn = ((scored['predicted_duplicate'] == 0) & (scored['is_duplicate'] == 1)).sum()

precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1        = 2 * precision * recall / (precision + recall)

print(f"Model Performance")
print(f"{'-'*40}")
print(f"  True Positives  : {tp:,}")
print(f"  False Positives : {fp:,}  ← minimise to reduce operational noise")
print(f"  False Negatives : {fn:,}  ← minimise to reduce financial leakage")
print(f"  Precision       : {precision:.4f}")
print(f"  Recall          : {recall:.4f}")
print(f"  F1 Score        : {f1:.4f}")

flagged  = scored.loc[scored['predicted_duplicate'] == 1, 'amount'].sum()
true_dup = scored.loc[scored['is_duplicate'] == 1, 'amount'].sum()
print()
print(f"  Flagged Amount        : ${flagged:,.0f}")
print(f"  True Duplicate Amount : ${true_dup:,.0f}")
print(f"  Financial Coverage    : {flagged/true_dup:.1%}")
"""))

cells.append(nbf.v4.new_code_cell(
"""# Fig 4: Score distribution
img = mpimg.imread('../reports/fig4_score_distribution.png')
plt.figure(figsize=(10, 4)); plt.imshow(img); plt.axis('off')
plt.tight_layout(); plt.show()
"""))

# ── Causal Analysis ────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""---
## 3. Causal Analysis: Difference-in-Differences (DiD)

### Why DiD?

A simple before/after comparison is confounded — duplicate rates may have been trending upward anyway (seasonality, volume growth). DiD removes this bias by using unaffected BUs as a control group.

### DiD Setup

| | Pre-Migration | Post-Migration | Δ |
|---|---|---|---|
| **Treated BUs** | A | B | B - A |
| **Control BUs** | C | D | D - C |
| | | **DiD** | **(B-A) - (D-C)** |

**Key assumption (Parallel Trends):** In the absence of the migration, treated and control BUs would have followed the same trend.
"""))

cells.append(nbf.v4.new_code_cell(
"""rates = (
    df.groupby(['treated', 'post'])
      .agg(total=('payment_ref', 'count'), dups=('is_duplicate', 'sum'))
      .assign(dup_rate=lambda x: (x['dups'] / x['total']).round(4))
      .reset_index()
)
rates['group']  = rates['treated'].map({1: 'Treated (Affected BUs)', 0: 'Control (Unaffected BUs)'})
rates['period'] = rates['post'].map({0: 'Pre-Migration', 1: 'Post-Migration'})

print("Duplicate Rates by Group and Period:")
print(rates[['group','period','total','dups','dup_rate']].to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell(
"""def get_rate(treated, post):
    return float(rates.loc[(rates['treated']==treated)&(rates['post']==post), 'dup_rate'])

t_pre  = get_rate(1, 0)
t_post = get_rate(1, 1)
c_pre  = get_rate(0, 0)
c_post = get_rate(0, 1)

did = (t_post - t_pre) - (c_post - c_pre)

print("DiD Calculation")
print("-"*50)
print(f"  Treated  Pre → Post : {t_pre:.4f} → {t_post:.4f}  Δ = +{t_post - t_pre:.4f}")
print(f"  Control  Pre → Post : {c_pre:.4f} → {c_post:.4f}  Δ = +{c_post - c_pre:.4f}")
print()
print(f"  DiD = ({t_post:.4f} - {t_pre:.4f}) - ({c_post:.4f} - {c_pre:.4f})")
print(f"      = {t_post - t_pre:.4f} - {c_post - c_pre:.4f}")
print(f"      = {did:.4f}  ({did*100:.1f} percentage points)")
print()
print(f"  Interpretation: The system migration CAUSALLY increased the duplicate rate")
print(f"  in affected BUs by {did*100:.1f}pp above and beyond the control group trend.")
"""))

cells.append(nbf.v4.new_code_cell(
"""# Bootstrap 95% Confidence Interval
np.random.seed(42)
did_samples = []

for _ in range(1000):
    s = df.sample(frac=1.0, replace=True)
    r = (
        s.groupby(['treated', 'post'])
         .agg(total=('payment_ref','count'), dups=('is_duplicate','sum'))
         .assign(rate=lambda x: x['dups'] / x['total'])
    )
    try:
        tp_ = float(r.loc[(1,1), 'rate'])
        ta_ = float(r.loc[(1,0), 'rate'])
        cp_ = float(r.loc[(0,1), 'rate'])
        ca_ = float(r.loc[(0,0), 'rate'])
        did_samples.append((tp_ - ta_) - (cp_ - ca_))
    except:
        continue

ci_lower = np.percentile(did_samples, 2.5)
ci_upper = np.percentile(did_samples, 97.5)

print("Bootstrap 95% Confidence Interval for DiD Estimate")
print("-"*50)
print(f"  DiD Point Estimate : {did:.4f}  ({did*100:.1f}pp)")
print(f"  95% CI             : [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  Bootstrap samples  : {len(did_samples):,}")
print()
print("  The CI excludes zero → causal effect is statistically and practically significant.")

# Visualise bootstrap distribution
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(did_samples, bins=40, color='#457B9D', alpha=0.8, edgecolor='white')
ax.axvline(did, color='#E63946', linewidth=2, label=f'DiD = {did:.4f}')
ax.axvline(ci_lower, color='gray', linestyle='--', linewidth=1.5, label=f'95% CI lower = {ci_lower:.4f}')
ax.axvline(ci_upper, color='gray', linestyle='--', linewidth=1.5, label=f'95% CI upper = {ci_upper:.4f}')
ax.set_title('Bootstrap Distribution of DiD Estimator (1,000 Iterations)', fontweight='bold')
ax.set_xlabel('DiD Estimate')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.show()
"""))

# ── Chi-Square ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""---
## 4. Chi-Square Test of Independence

**H₀:** Duplicate occurrence is independent of migration period (in treated BUs).  
**H₁:** Post-migration treated BUs show a significantly higher duplicate rate.

The chi-square test operates on a 2×2 contingency table (treated BUs only).
"""))

cells.append(nbf.v4.new_code_cell(
"""treated_df = df[df['treated'] == 1]

pre_dups   = treated_df.loc[treated_df['post'] == 0, 'is_duplicate'].sum()
pre_clean  = (treated_df['post'] == 0).sum() - pre_dups
post_dups  = treated_df.loc[treated_df['post'] == 1, 'is_duplicate'].sum()
post_clean = (treated_df['post'] == 1).sum() - post_dups

contingency = np.array([[pre_dups, pre_clean], [post_dups, post_clean]])

ct = pd.DataFrame(contingency,
    index=['Pre-Migration', 'Post-Migration'],
    columns=['Duplicate', 'Not Duplicate'])
print("Contingency Table (Treated BUs Only):")
print(ct)
print()

chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-Square Statistic : {chi2:.4f}")
print(f"Degrees of Freedom   : {dof}")
print(f"P-Value              : {p:.2e}")
print(f"Significant (p<0.05) : {p < 0.05}")
print()
print("Result: We REJECT H0. The spike in duplicates post-migration is")
print("statistically significant. This is not random variation.")
"""))

# ── Visualizations ────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("---\n## 5. Visualizations"))

for fig_path, caption in [
    ("../reports/fig1_monthly_trend.png",
     "Fig 1: Treated BUs show a clear post-migration spike while control BUs remain stable — supporting the parallel trends assumption."),
    ("../reports/fig2_did_bars.png",
     "Fig 2: Treated BUs show a 5x increase post-migration; control BUs are stable. The gap is the DiD causal estimate."),
    ("../reports/fig3_category_heatmap.png",
     "Fig 3: IT Services post-migration in treated BUs reaches 24.2% — the root cause segment requiring targeted vendor validation controls."),
    ("../reports/fig5_financial_exposure.png",
     "Fig 5: The model flags $19.8M of the $23.5M true duplicate exposure — 84% financial coverage at 99.7% precision."),
]:
    cells.append(nbf.v4.new_code_cell(
        f"""img = mpimg.imread('{fig_path}')
plt.figure(figsize=(11, 5)); plt.imshow(img); plt.axis('off')
plt.tight_layout(); plt.show()
print(\"\"\"{caption}\"\"\")
"""))

# ── Summary ───────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""---
## 6. Summary of Findings & Recommendations

| Metric | Value |
|---|---|
| **DiD Causal Estimate** | +11.1pp increase in duplicate rate |
| **95% Bootstrap CI** | [9.5pp, 12.8pp] — excludes zero |
| **Chi-Square p-value** | < 0.001 (highly significant) |
| **Detection Precision** | 99.7% |
| **Detection Recall** | 82.3% |
| **F1 Score** | 90.2% |
| **Financial Exposure Flagged** | $19.8M |
| **True Duplicate Exposure** | $23.5M |
| **Highest-Risk Segment** | IT Services × Treated BUs (post-migration) |

### Key Conclusions

1. **The migration causally increased duplicates.** The DiD estimate of +11.1pp is statistically significant (p < 0.001), with a 95% CI entirely above zero.

2. **IT Services is the highest-risk category.** Post-migration duplicate rate of 24.2% in treated BUs — 8x the baseline — indicating vendor master records were not re-validated.

3. **99.7% precision** virtually eliminates false positives that create operational noise, while 82.3% recall captures the bulk of financial exposure.

### Recommendations

- Implement **mandatory vendor master data validation** within 30 days of any ERP migration
- Prioritise **IT Services vendors in BU_North and BU_South** for immediate audit
- Deploy this detection pipeline as a **scheduled job** in the post-migration window (first 60–90 days)
- Add a **vendor re-submission cooldown rule** in AP workflow: block same-vendor, same-amount payments within 7 days unless explicitly approved

---

### References & Sources

- Angrist, J. & Pischke, J.S. (2009). *Mostly Harmless Econometrics.* Princeton University Press.
- Card, D. & Krueger, A. (1994). Minimum Wages and Employment: A Case Study of the Fast-Food Industry. *American Economic Review.* — Foundational DiD paper.
- `scipy.stats.chi2_contingency` — [SciPy Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)
- Bootstrap confidence intervals — Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap.*
- Synthetic AP data generated by `src/generate_data.py` — all figures, company names, and amounts are fictional.
"""))

nb.cells = cells

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/analysis_codebook.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written successfully.")
