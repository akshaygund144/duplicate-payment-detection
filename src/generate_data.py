"""
generate_data.py
----------------
Generates a synthetic Accounts Payable (AP) transaction dataset
simulating an ERP environment with a system migration event.

The migration acts as a natural experiment:
  - Pre-migration: baseline duplicate rate ~2%
  - Post-migration (affected BUs): duplicate rate spikes ~6%
  - Post-migration (unaffected BUs): duplicate rate stays ~2%

This enables a Difference-in-Differences (DiD) causal analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Configuration ─────────────────────────────────────────────────────────────
MIGRATION_DATE     = datetime(2023, 7, 1)
START_DATE         = datetime(2023, 1, 1)
END_DATE           = datetime(2023, 12, 31)
N_TRANSACTIONS     = 15_000

BUSINESS_UNITS     = ["BU_North", "BU_South", "BU_East", "BU_West", "BU_Central"]
AFFECTED_BUS       = ["BU_North", "BU_South"]          # treated group in DiD

VENDOR_CATEGORIES  = ["Office Supplies", "IT Services", "Logistics", "Consulting", "Facilities"]
HIGH_RISK_CATEGORY = "IT Services"                     # concentrated duplicate risk

AMOUNT_RANGES = {
    "Office Supplies": (500,   5_000),
    "IT Services":     (5_000, 80_000),
    "Logistics":       (1_000, 15_000),
    "Consulting":      (3_000, 50_000),
    "Facilities":      (800,   10_000),
}

# ── Helper: duplicate rate by condition ──────────────────────────────────────
def get_dup_rate(bu: str, txn_date: datetime, vendor_cat: str) -> float:
    post  = txn_date >= MIGRATION_DATE
    treat = bu in AFFECTED_BUS
    high  = vendor_cat == HIGH_RISK_CATEGORY

    if treat and post and high:
        return 0.18   # severe spike in treated + high-risk
    elif treat and post:
        return 0.06   # general spike in treated BUs
    elif high and post:
        return 0.04   # slight elevation even in control BUs
    else:
        return 0.02   # baseline


# ── Main generation ──────────────────────────────────────────────────────────
def generate_transactions(n: int = N_TRANSACTIONS) -> pd.DataFrame:
    date_range_days = (END_DATE - START_DATE).days
    records = []

    for i in range(n):
        txn_date     = START_DATE + timedelta(days=random.randint(0, date_range_days))
        bu           = random.choice(BUSINESS_UNITS)
        vendor_cat   = random.choices(
            VENDOR_CATEGORIES,
            weights=[0.15, 0.30, 0.20, 0.25, 0.10]   # IT Services over-represented
        )[0]
        vendor_id    = f"V{random.randint(1000, 1099):04d}"
        lo, hi       = AMOUNT_RANGES[vendor_cat]
        amount       = round(np.random.uniform(lo, hi), 2)
        invoice_no   = f"INV-{random.randint(10000, 99999)}"
        payment_ref  = f"PMT-{i+1:06d}"
        dup_rate     = get_dup_rate(bu, txn_date, vendor_cat)
        is_duplicate = int(np.random.rand() < dup_rate)
        post_mig     = int(txn_date >= MIGRATION_DATE)
        treated      = int(bu in AFFECTED_BUS)

        records.append({
            "payment_ref":       payment_ref,
            "invoice_no":        invoice_no,
            "vendor_id":         vendor_id,
            "vendor_category":   vendor_cat,
            "business_unit":     bu,
            "transaction_date":  txn_date.strftime("%Y-%m-%d"),
            "amount":            amount,
            "is_duplicate":      is_duplicate,
            "post_migration":    post_mig,
            "treated_bu":        treated,
        })

    df = pd.DataFrame(records)

    # ── Inject realistic duplicates: copy rows with slight variation ─────────
    dup_rows   = df[df["is_duplicate"] == 1].copy()
    dup_inject = dup_rows.sample(frac=0.7, random_state=SEED).copy()
    dup_inject["payment_ref"] = [
        f"PMT-DUP-{i:06d}" for i in range(len(dup_inject))
    ]
    # Slight date shift (1–3 days) to simulate re-submission
    dup_inject["transaction_date"] = pd.to_datetime(
        dup_inject["transaction_date"]
    ).apply(lambda d: (d + timedelta(days=random.randint(1, 3))).strftime("%Y-%m-%d"))

    df = pd.concat([df, dup_inject], ignore_index=True)
    df = df.sort_values("transaction_date").reset_index(drop=True)

    return df


def save_datasets(df: pd.DataFrame, out_dir: str = "data"):
    os.makedirs(out_dir, exist_ok=True)

    # Full dataset
    df.to_csv(f"{out_dir}/ap_transactions.csv", index=False)
    print(f"[✓] Saved ap_transactions.csv  ({len(df):,} rows)")

    # Pre / Post splits for SQL-style analysis
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    pre  = df[df["transaction_date"] <  MIGRATION_DATE]
    post = df[df["transaction_date"] >= MIGRATION_DATE]
    pre.to_csv(f"{out_dir}/ap_transactions_pre_migration.csv",  index=False)
    post.to_csv(f"{out_dir}/ap_transactions_post_migration.csv", index=False)
    print(f"[✓] Saved pre-migration  ({len(pre):,} rows)")
    print(f"[✓] Saved post-migration ({len(post):,} rows)")

    # Summary table
    summary = (
        df.groupby(["business_unit", "post_migration", "vendor_category"])
          .agg(
              total_txns   = ("payment_ref",  "count"),
              total_amount = ("amount",        "sum"),
              dup_count    = ("is_duplicate",  "sum"),
          )
          .assign(dup_rate=lambda x: (x["dup_count"] / x["total_txns"]).round(4))
          .reset_index()
    )
    summary.to_csv(f"{out_dir}/summary_by_bu_period.csv", index=False)
    print(f"[✓] Saved summary_by_bu_period.csv")


if __name__ == "__main__":
    print("Generating synthetic AP transaction data...")
    df = generate_transactions()
    save_datasets(df, out_dir="data")
    print("\nData generation complete.")
