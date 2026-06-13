"""
visualizations.py
------------------
Produces all charts for the duplicate payment detection project.

Figures generated
-----------------
  Fig 1 – Duplicate rate over time (treated vs control)
  Fig 2 – DiD bar chart (pre/post × group)
  Fig 3 – Vendor category heatmap (duplicate rate)
  Fig 4 – Duplicate score distribution
  Fig 5 – Financial exposure: flagged vs confirmed duplicates
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

MIGRATION_DATE = pd.Timestamp("2023-07-01")
AFFECTED_BUS   = ["BU_North", "BU_South"]
REPORTS_DIR    = "reports"

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"treated": "#E63946", "control": "#457B9D",
          "pre": "#A8DADC",    "post":    "#1D3557"}


def _ensure_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)


# ── Fig 1: Monthly duplicate rate – treated vs control ───────────────────────

def fig1_monthly_trend(df: pd.DataFrame):
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["month"] = df["transaction_date"].dt.to_period("M")
    df["group"] = df["business_unit"].apply(
        lambda x: "Treated (BU_North, BU_South)" if x in AFFECTED_BUS else "Control (Others)"
    )

    monthly = (
        df.groupby(["month", "group"])
          .agg(total=("payment_ref", "count"), dups=("is_duplicate", "sum"))
          .assign(dup_rate=lambda x: x["dups"] / x["total"])
          .reset_index()
    )
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 5))
    for grp, color in [("Treated (BU_North, BU_South)", COLORS["treated"]),
                        ("Control (Others)",              COLORS["control"])]:
        sub = monthly[monthly["group"] == grp]
        ax.plot(sub["month_dt"], sub["dup_rate"], marker="o",
                label=grp, color=color, linewidth=2)

    ax.axvline(MIGRATION_DATE, color="black", linestyle="--", linewidth=1.5,
               label="Migration Date (Jul 2023)")
    ax.fill_betweenx([0, 0.25], MIGRATION_DATE,
                     monthly["month_dt"].max(),
                     alpha=0.06, color="orange", label="Post-Migration Window")
    ax.set_title("Fig 1 — Monthly Duplicate Payment Rate: Treated vs Control BUs",
                 fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Duplicate Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    fig.tight_layout()
    _ensure_dir()
    fig.savefig(f"{REPORTS_DIR}/fig1_monthly_trend.png", dpi=150)
    plt.close(fig)
    print("[✓] Fig 1 saved")


# ── Fig 2: DiD bar chart ─────────────────────────────────────────────────────

def fig2_did_bars(rates: pd.DataFrame):
    pivot = rates.pivot_table(index="group", columns="period",
                               values="dup_rate").reset_index()

    x      = np.arange(len(pivot))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    bars_pre  = ax.bar(x - width/2, pivot["Pre-Migration"],  width,
                       label="Pre-Migration",  color=COLORS["pre"],  edgecolor="white")
    bars_post = ax.bar(x + width/2, pivot["Post-Migration"], width,
                       label="Post-Migration", color=COLORS["post"], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(pivot["group"], fontsize=10)
    ax.set_ylabel("Duplicate Rate")
    ax.set_title("Fig 2 — DiD: Duplicate Rate Pre vs Post Migration by Group",
                 fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.legend()

    for bar in list(bars_pre) + list(bars_post):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                f"{h:.2%}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/fig2_did_bars.png", dpi=150)
    plt.close(fig)
    print("[✓] Fig 2 saved")


# ── Fig 3: Heatmap – vendor category × BU ───────────────────────────────────

def fig3_category_heatmap(df: pd.DataFrame):
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    post_df = df[df["transaction_date"] >= MIGRATION_DATE]

    pivot = (
        post_df.groupby(["business_unit", "vendor_category"])
               .agg(total=("payment_ref","count"), dups=("is_duplicate","sum"))
               .assign(dup_rate=lambda x: x["dups"] / x["total"])
               .reset_index()
               .pivot(index="business_unit", columns="vendor_category",
                      values="dup_rate")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlOrRd",
                linewidths=0.5, ax=ax)
    ax.set_title("Fig 3 — Post-Migration Duplicate Rate: BU × Vendor Category",
                 fontweight="bold")
    ax.set_xlabel("Vendor Category")
    ax.set_ylabel("Business Unit")
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/fig3_category_heatmap.png", dpi=150)
    plt.close(fig)
    print("[✓] Fig 3 saved")


# ── Fig 4: Duplicate score distribution ──────────────────────────────────────

def fig4_score_distribution(scored_df: pd.DataFrame):
    counts = scored_df["duplicate_score"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=["#2EC4B6", "#FFBF69", "#FF9F1C", "#E63946"][:len(counts)],
                  edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 50,
                f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Fig 4 — Distribution of Duplicate Scores (0 = Clean, 3 = Confirmed)",
                 fontweight="bold")
    ax.set_xlabel("Duplicate Score")
    ax.set_ylabel("Number of Transactions")
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/fig4_score_distribution.png", dpi=150)
    plt.close(fig)
    print("[✓] Fig 4 saved")


# ── Fig 5: Financial exposure ────────────────────────────────────────────────

def fig5_financial_exposure(scored_df: pd.DataFrame):
    flagged_amt    = scored_df.loc[scored_df["predicted_duplicate"] == 1, "amount"].sum()
    true_dup_amt   = scored_df.loc[scored_df["is_duplicate"] == 1,        "amount"].sum()
    total_amt      = scored_df["amount"].sum()
    clean_amt      = total_amt - flagged_amt

    labels  = ["Clean Transactions", "Flagged (Predicted Duplicate)",
               "Confirmed Duplicates (Ground Truth)"]
    amounts = [clean_amt, flagged_amt, true_dup_amt]
    colors  = [COLORS["control"], COLORS["treated"], "#F4A261"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, [a / 1e6 for a in amounts], color=colors, edgecolor="white")
    for bar, amt in zip(bars, amounts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"${amt/1e6:.1f}M", va="center", fontsize=10)
    ax.set_xlabel("Amount (USD Millions)")
    ax.set_title("Fig 5 — Financial Exposure: Flagged vs Confirmed Duplicates",
                 fontweight="bold")
    ax.set_xlim(0, max(a/1e6 for a in amounts) * 1.2)
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/fig5_financial_exposure.png", dpi=150)
    plt.close(fig)
    print("[✓] Fig 5 saved")


# ── Main ─────────────────────────────────────────────────────────────────────

def generate_all_figures(transactions_path: str = "data/ap_transactions.csv",
                         scored_path: str       = "data/ap_transactions_scored.csv",
                         rates_df: pd.DataFrame = None):
    print("\nGenerating visualizations...")
    df        = pd.read_csv(transactions_path)
    scored_df = pd.read_csv(scored_path)

    fig1_monthly_trend(df)
    if rates_df is not None:
        fig2_did_bars(rates_df)
    fig3_category_heatmap(df)
    fig4_score_distribution(scored_df)
    fig5_financial_exposure(scored_df)
    print(f"\nAll figures saved to /{REPORTS_DIR}/")


if __name__ == "__main__":
    from causal_analysis import run_causal_analysis
    results = run_causal_analysis()
    generate_all_figures(rates_df=results["group_rates"])
