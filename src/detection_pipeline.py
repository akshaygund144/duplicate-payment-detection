"""
detection_pipeline.py
----------------------
Modular duplicate payment detection pipeline.

Replicates SQL window-function logic (PARTITION BY + LAG/LEAD) in Pandas,
applies rule-based flagging, and outputs a scored transaction file.

Rules applied
-------------
R1 – Same vendor + same amount + invoice within 3-day window
R2 – Same vendor + same amount + same invoice number (hard duplicate)
R3 – Same vendor + amount within 1% tolerance + within 7-day window
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import os


# ── Rule definitions ──────────────────────────────────────────────────────────

def rule_r1_exact_proximity(df: pd.DataFrame) -> pd.Series:
    """
    R1: Same vendor_id + exact amount + transaction_date within 3 days.
    Mimics SQL:
        PARTITION BY vendor_id, amount
        ORDER BY transaction_date
        LAG(transaction_date) → date_diff
    """
    df = df.sort_values(["vendor_id", "amount", "transaction_date"])
    df["_lag_date"] = df.groupby(["vendor_id", "amount"])["transaction_date"].shift(1)
    df["_date_diff"] = (df["transaction_date"] - df["_lag_date"]).dt.days
    flag = (df["_date_diff"].between(0, 3, inclusive="both"))
    df.drop(columns=["_lag_date", "_date_diff"], inplace=True)
    return flag.fillna(False)


def rule_r2_exact_invoice(df: pd.DataFrame) -> pd.Series:
    """
    R2: Same vendor_id + same invoice_no (hard duplicate regardless of date).
    """
    dup_mask = df.duplicated(subset=["vendor_id", "invoice_no"], keep=False)
    return dup_mask


def rule_r3_fuzzy_amount(df: pd.DataFrame, tolerance: float = 0.01) -> pd.Series:
    """
    R3: Same vendor + amount within tolerance% + within 7-day window.
    Tolerance default = 1%.
    """
    df = df.sort_values(["vendor_id", "transaction_date"])
    flags = pd.Series(False, index=df.index)

    for vendor, group in df.groupby("vendor_id"):
        group = group.sort_values("transaction_date").reset_index()
        for i in range(1, len(group)):
            for j in range(i - 1, -1, -1):
                date_diff = (group.loc[i, "transaction_date"] - group.loc[j, "transaction_date"]).days
                if date_diff > 7:
                    break
                amt_i = group.loc[i, "amount"]
                amt_j = group.loc[j, "amount"]
                if amt_j == 0:
                    continue
                if abs(amt_i - amt_j) / amt_j <= tolerance:
                    flags.at[group.loc[i, "index"]] = True
                    flags.at[group.loc[j, "index"]] = True

    return flags


def score_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all rules and produce a composite duplicate_score (0–3).
    Score interpretation:
        0 = Clean
        1 = Low suspicion (1 rule)
        2 = Medium suspicion (2 rules)
        3 = High suspicion / confirmed duplicate (all 3 rules)
    """
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    print("  Applying Rule R1: Exact proximity...")
    df["flag_r1"] = rule_r1_exact_proximity(df).astype(int)

    print("  Applying Rule R2: Exact invoice match...")
    df["flag_r2"] = rule_r2_exact_invoice(df).astype(int)

    print("  Applying Rule R3: Fuzzy amount proximity...")
    df["flag_r3"] = rule_r3_fuzzy_amount(df).astype(int)

    df["duplicate_score"] = df["flag_r1"] + df["flag_r2"] + df["flag_r3"]
    df["predicted_duplicate"] = (df["duplicate_score"] >= 2).astype(int)

    return df


# ── Evaluation metrics ────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame) -> dict:
    """
    Compare predicted_duplicate vs is_duplicate ground truth.
    Returns precision, recall, F1, and financial exposure.
    """
    tp = ((df["predicted_duplicate"] == 1) & (df["is_duplicate"] == 1)).sum()
    fp = ((df["predicted_duplicate"] == 1) & (df["is_duplicate"] == 0)).sum()
    fn = ((df["predicted_duplicate"] == 0) & (df["is_duplicate"] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    flagged_amount = df.loc[df["predicted_duplicate"] == 1, "amount"].sum()
    true_dup_amount = df.loc[df["is_duplicate"] == 1, "amount"].sum()

    return {
        "true_positives":        int(tp),
        "false_positives":       int(fp),
        "false_negatives":       int(fn),
        "precision":             round(precision, 4),
        "recall":                round(recall, 4),
        "f1_score":              round(f1, 4),
        "flagged_amount_usd":    round(flagged_amount, 2),
        "true_dup_amount_usd":   round(true_dup_amount, 2),
    }


# ── Main pipeline entry point ─────────────────────────────────────────────────

def run_pipeline(input_path: str = "data/ap_transactions.csv",
                 output_dir: str = "data") -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("  Duplicate Payment Detection Pipeline")
    print(f"{'='*60}")

    print(f"\n[1/4] Loading transactions from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"      {len(df):,} transactions loaded.")

    print("\n[2/4] Scoring transactions...")
    scored = score_transactions(df)

    print("\n[3/4] Evaluating model performance...")
    metrics = evaluate(scored)
    print(f"\n  {'Metric':<28} {'Value':>10}")
    print(f"  {'-'*40}")
    for k, v in metrics.items():
        print(f"  {k:<28} {v:>10}")

    print("\n[4/4] Saving scored output...")
    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/ap_transactions_scored.csv"
    scored.to_csv(out_path, index=False)
    print(f"      Saved → {out_path}")

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"{'='*60}\n")

    return scored


if __name__ == "__main__":
    run_pipeline()
