"""
causal_analysis.py
-------------------
Difference-in-Differences (DiD) causal analysis of duplicate payment rates
around a system migration event.

Framework
---------
  Treated group  : Affected business units (BU_North, BU_South)
  Control group  : Unaffected business units (BU_East, BU_West, BU_Central)
  Pre period     : Jan 2023 – Jun 2023
  Post period    : Jul 2023 – Dec 2023
  Outcome        : Duplicate payment rate (duplicates / total transactions)

DiD Estimator
-------------
  DiD = (Treated_Post - Treated_Pre) - (Control_Post - Control_Pre)

Statistical Tests
-----------------
  Chi-square test of independence: duplicate occurrence vs period × group
  Confidence intervals on DiD estimate via bootstrap (1000 iterations)
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


MIGRATION_DATE = pd.Timestamp("2023-07-01")
AFFECTED_BUS   = ["BU_North", "BU_South"]


# ── Data prep ────────────────────────────────────────────────────────────────

def prepare_did_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["post"]    = (df["transaction_date"] >= MIGRATION_DATE).astype(int)
    df["treated"] = df["business_unit"].isin(AFFECTED_BUS).astype(int)
    return df


# ── Aggregate rates ──────────────────────────────────────────────────────────

def compute_group_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns duplicate rate per group (treated/control) × period (pre/post).
    """
    agg = (
        df.groupby(["treated", "post"])
          .agg(
              total_txns = ("payment_ref",  "count"),
              dup_count  = ("is_duplicate", "sum"),
          )
          .assign(dup_rate=lambda x: x["dup_count"] / x["total_txns"])
          .reset_index()
    )
    agg["group"]  = agg["treated"].map({1: "Treated (Affected BUs)",
                                         0: "Control (Unaffected BUs)"})
    agg["period"] = agg["post"].map({0: "Pre-Migration", 1: "Post-Migration"})
    return agg


# ── DiD estimator ─────────────────────────────────────────────────────────────

def compute_did(rates: pd.DataFrame) -> dict:
    """
    Computes the DiD point estimate.
    """
    def _rate(treated: int, post: int) -> float:
        row = rates[(rates["treated"] == treated) & (rates["post"] == post)]
        return float(row["dup_rate"].values[0])

    treated_pre  = _rate(1, 0)
    treated_post = _rate(1, 1)
    control_pre  = _rate(0, 0)
    control_post = _rate(0, 1)

    did = (treated_post - treated_pre) - (control_post - control_pre)

    return {
        "treated_pre":   round(treated_pre,  4),
        "treated_post":  round(treated_post, 4),
        "control_pre":   round(control_pre,  4),
        "control_post":  round(control_post, 4),
        "delta_treated": round(treated_post - treated_pre, 4),
        "delta_control": round(control_post - control_pre, 4),
        "did_estimate":  round(did, 4),
    }


# ── Bootstrap confidence interval ────────────────────────────────────────────

def bootstrap_did_ci(df: pd.DataFrame,
                     n_iterations: int = 1000,
                     ci: float = 0.95) -> dict:
    """
    Bootstrap confidence interval for the DiD estimator.
    """
    did_samples = []

    for _ in range(n_iterations):
        sample = df.sample(frac=1.0, replace=True, random_state=None)
        rates  = compute_group_rates(sample)
        try:
            did_val = compute_did(rates)["did_estimate"]
            did_samples.append(did_val)
        except (IndexError, KeyError):
            continue

    alpha = 1 - ci
    lower = np.percentile(did_samples, 100 * alpha / 2)
    upper = np.percentile(did_samples, 100 * (1 - alpha / 2))

    return {
        "did_mean":   round(np.mean(did_samples), 4),
        "ci_lower":   round(lower, 4),
        "ci_upper":   round(upper, 4),
        "ci_level":   f"{int(ci*100)}%",
        "n_bootstrap": n_iterations,
    }


# ── Chi-square test ───────────────────────────────────────────────────────────

def chi_square_test(df: pd.DataFrame) -> dict:
    """
    Chi-square test of independence.
    H0: Duplicate occurrence is independent of (group × period).
    H1: Treated BUs post-migration have a significantly different duplicate rate.
    """
    # Focus on treated BUs pre vs post
    treated_pre  = df[(df["treated"] == 1) & (df["post"] == 0)]
    treated_post = df[(df["treated"] == 1) & (df["post"] == 1)]

    contingency = np.array([
        [treated_pre["is_duplicate"].sum(),
         len(treated_pre) - treated_pre["is_duplicate"].sum()],
        [treated_post["is_duplicate"].sum(),
         len(treated_post) - treated_post["is_duplicate"].sum()],
    ])

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    return {
        "chi2_statistic":   round(chi2, 4),
        "p_value":          round(p_value, 6),
        "degrees_of_freedom": int(dof),
        "significant_at_5pct": p_value < 0.05,
        "contingency_table": contingency.tolist(),
    }


# ── Vendor category breakdown ─────────────────────────────────────────────────

def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicate rate by vendor category × period × group.
    Isolates which category drives the spike.
    """
    breakdown = (
        df.groupby(["vendor_category", "treated", "post"])
          .agg(
              total   = ("payment_ref",  "count"),
              dups    = ("is_duplicate", "sum"),
          )
          .assign(dup_rate=lambda x: (x["dups"] / x["total"]).round(4))
          .reset_index()
    )
    breakdown["group"]  = breakdown["treated"].map({1: "Treated", 0: "Control"})
    breakdown["period"] = breakdown["post"].map({0: "Pre", 1: "Post"})
    return breakdown


# ── Main runner ───────────────────────────────────────────────────────────────

def run_causal_analysis(input_path: str = "data/ap_transactions.csv") -> dict:
    print(f"\n{'='*60}")
    print("  Causal Analysis: Difference-in-Differences")
    print(f"{'='*60}")

    df    = pd.read_csv(input_path)
    df    = prepare_did_data(df)
    rates = compute_group_rates(df)

    print("\n── Group Rates ──────────────────────────────────────────")
    print(rates[["group", "period", "total_txns", "dup_count", "dup_rate"]].to_string(index=False))

    did = compute_did(rates)
    print("\n── DiD Estimates ────────────────────────────────────────")
    for k, v in did.items():
        print(f"  {k:<20} {v}")

    print("\n── Bootstrap CI (1000 iterations) ──────────────────────")
    ci = bootstrap_did_ci(df, n_iterations=1000)
    for k, v in ci.items():
        print(f"  {k:<20} {v}")

    print("\n── Chi-Square Test ──────────────────────────────────────")
    chi = chi_square_test(df)
    for k, v in chi.items():
        if k != "contingency_table":
            print(f"  {k:<28} {v}")
    print(f"  {'p < 0.05 (significant)?':<28} {chi['significant_at_5pct']}")

    print("\n── Vendor Category Breakdown ────────────────────────────")
    cat = category_breakdown(df)
    print(cat.to_string(index=False))

    return {"group_rates": rates, "did": did, "ci": ci, "chi_square": chi,
            "category_breakdown": cat}


if __name__ == "__main__":
    run_causal_analysis()
