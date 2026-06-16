import json
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(".")
STEP3_BIN = BASE_DIR / "bin" / "step3_mm"
FEATURES_CSV = BASE_DIR / "output" / "step2_hawkes" / "feature_table_step2_1s.csv"
OUT_ROOT = BASE_DIR / "output" / "step4_analysis"

BASE_ARGS = {
    "gamma": 0.10,
    "inventory_limit": 500,
    "max_order_size": 25,
    "tick_size": 0.01,
    "min_spread_ticks": 2,
    "ofi_skew_coef": 0.03,
    "microprice_skew_coef": 0.10,
    "latency_budget_us": 500,
    "maker_fee_bps": 0.20,
    "queue_haircut_base": 0.25,
    "toxicity_haircut_strength": 0.60,
    "toxic_ofi_threshold": 0.20,
    "toxic_side_widen_ticks": 3,
}

A_GRID = [0.05, 0.08, 0.12]
K_GRID = [4.0, 6.0, 8.0]
MARKOUT_GRID = [1, 3, 5]
USE_HAWKES_GRID = [True, False]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def run_step3(outdir: Path, A: float, k: float, markout_horizon: int, use_hawkes: bool):
    cmd = [
        str(STEP3_BIN),
        "--features", str(FEATURES_CSV),
        "--outdir", str(outdir),
        "--gamma", str(BASE_ARGS["gamma"]),
        "--k", str(k),
        "--A", str(A),
        "--inventory-limit", str(BASE_ARGS["inventory_limit"]),
        "--max-order-size", str(BASE_ARGS["max_order_size"]),
        "--tick-size", str(BASE_ARGS["tick_size"]),
        "--min-spread-ticks", str(BASE_ARGS["min_spread_ticks"]),
        "--ofi-skew-coef", str(BASE_ARGS["ofi_skew_coef"]),
        "--microprice-skew-coef", str(BASE_ARGS["microprice_skew_coef"]),
        "--latency-budget-us", str(BASE_ARGS["latency_budget_us"]),
        "--maker-fee-bps", str(BASE_ARGS["maker_fee_bps"]),
        "--queue-haircut-base", str(BASE_ARGS["queue_haircut_base"]),
        "--toxicity-haircut-strength", str(BASE_ARGS["toxicity_haircut_strength"]),
        "--toxic-ofi-threshold", str(BASE_ARGS["toxic_ofi_threshold"]),
        "--toxic-side-widen-ticks", str(BASE_ARGS["toxic_side_widen_ticks"]),
        "--markout-horizon-steps", str(markout_horizon),
    ]
    if use_hawkes:
        cmd.append("--use-hawkes")
    subprocess.run(cmd, check=True)

def load_summary_and_results(run_dir: Path):
    with open(run_dir / "step3_summary.json", "r") as f:
        summary = json.load(f)
    df = pd.read_csv(run_dir / "engine_results.csv")
    return summary, df

def compute_drawdown(series: pd.Series):
    running_max = series.cummax()
    dd = running_max - series
    return float(dd.max())

def compute_extra_metrics(df: pd.DataFrame):
    out = {}

    out["final_pnl"] = float(df["mtm_pnl"].iloc[-1])
    out["max_drawdown"] = compute_drawdown(df["mtm_pnl"])
    out["inventory_mean_abs"] = float(df["inventory"].abs().mean())
    out["inventory_std"] = float(df["inventory"].std())
    out["latency_p99_empirical"] = float(df["latency_us"].quantile(0.99))
    out["fill_rate_bid_empirical"] = float(df["bid_fill"].mean())
    out["fill_rate_ask_empirical"] = float(df["ask_fill"].mean())
    out["avg_fee_per_step"] = float(df["fee_paid"].mean())
    out["avg_adverse_cost_per_step"] = float(df["adverse_selection_cost"].mean())
    out["avg_realized_spread_per_step"] = float(df["realized_spread_component"].mean())
    out["quote_pull_rate_bid"] = float(1.0 - df["quote_bid_active"].mean())
    out["quote_pull_rate_ask"] = float(1.0 - df["quote_ask_active"].mean())

    bid_markouts = df.loc[df["bid_fill"] == 1, "bid_markout"]
    ask_markouts = df.loc[df["ask_fill"] == 1, "ask_markout"]
    all_markouts = pd.concat([bid_markouts, ask_markouts], ignore_index=True)

    if len(all_markouts):
        out["avg_markout_all_fills"] = float(all_markouts.mean())
        out["median_markout_all_fills"] = float(all_markouts.median())
        out["p10_markout_all_fills"] = float(all_markouts.quantile(0.10))
        out["p90_markout_all_fills"] = float(all_markouts.quantile(0.90))
    else:
        out["avg_markout_all_fills"] = np.nan
        out["median_markout_all_fills"] = np.nan
        out["p10_markout_all_fills"] = np.nan
        out["p90_markout_all_fills"] = np.nan

    if "buy_pressure" in df.columns and "sell_pressure" in df.columns:
        out["avg_abs_pressure_diff"] = float((df["buy_pressure"] - df["sell_pressure"]).abs().mean())
    else:
        out["avg_abs_pressure_diff"] = np.nan

    return out

def regime_breakdown(df: pd.DataFrame):
    df = df.copy()

    if "buy_pressure" in df.columns and "sell_pressure" in df.columns:
        df["toxicity_proxy"] = (df["buy_pressure"] - df["sell_pressure"]).abs()
    else:
        df["toxicity_proxy"] = 0.0

    q1 = df["toxicity_proxy"].quantile(0.33)
    q2 = df["toxicity_proxy"].quantile(0.66)

    def label_regime(x):
        if x <= q1:
            return "low_toxicity"
        elif x <= q2:
            return "medium_toxicity"
        return "high_toxicity"

    df["toxicity_regime"] = df["toxicity_proxy"].apply(label_regime)

    rows = []
    for regime, g in df.groupby("toxicity_regime"):
        bid_markouts = g.loc[g["bid_fill"] == 1, "bid_markout"]
        ask_markouts = g.loc[g["ask_fill"] == 1, "ask_markout"]
        all_markouts = pd.concat([bid_markouts, ask_markouts], ignore_index=True)

        rows.append({
            "toxicity_regime": regime,
            "n_steps": int(len(g)),
            "mean_pnl": float(g["mtm_pnl"].mean()),
            "final_pnl_last_step": float(g["mtm_pnl"].iloc[-1]),
            "fill_rate_bid": float(g["bid_fill"].mean()),
            "fill_rate_ask": float(g["ask_fill"].mean()),
            "avg_adverse_selection_cost": float(g["adverse_selection_cost"].mean()),
            "avg_realized_spread_component": float(g["realized_spread_component"].mean()),
            "avg_markout": float(all_markouts.mean()) if len(all_markouts) else np.nan,
            "median_markout": float(all_markouts.median()) if len(all_markouts) else np.nan,
            "mean_inventory_abs": float(g["inventory"].abs().mean()),
        })

    order = {"low_toxicity": 0, "medium_toxicity": 1, "high_toxicity": 2}
    regime_df = pd.DataFrame(rows)
    if len(regime_df):
        regime_df["sort_key"] = regime_df["toxicity_regime"].map(order)
        regime_df = regime_df.sort_values("sort_key").drop(columns="sort_key")
    return regime_df

def plot_top_pnl_paths(master_df: pd.DataFrame, results_map: dict, outdir: Path):
    plt.figure(figsize=(12, 7))
    top = master_df.head(6)
    for _, row in top.iterrows():
        rid = row["run_id"]
        label = f"{rid}"
        plt.plot(results_map[rid]["mtm_pnl"].values, label=label, linewidth=1.4)
    plt.title("Top 6 Runs by Risk-Adjusted Score")
    plt.xlabel("Step")
    plt.ylabel("MTM PnL")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "top6_pnl_paths.png", dpi=180)
    plt.close()

def plot_heatmap(master_df: pd.DataFrame, outdir: Path, use_hawkes: bool):
    sub = master_df[master_df["use_hawkes"] == use_hawkes].copy()
    if sub.empty:
        return
    pivot = sub.groupby(["A", "k"])["final_mtm_pnl"].mean().reset_index()
    heat = pivot.pivot(index="A", columns="k", values="final_mtm_pnl")

    plt.figure(figsize=(8, 6))
    plt.imshow(heat.values, aspect="auto", cmap="viridis")
    plt.xticks(range(len(heat.columns)), [str(c) for c in heat.columns])
    plt.yticks(range(len(heat.index)), [str(i) for i in heat.index])
    plt.xlabel("k")
    plt.ylabel("A")
    plt.title(f"Average Final MTM PnL by A and k ({'Hawkes' if use_hawkes else 'Baseline'})")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            plt.text(j, i, f"{heat.values[i, j]:.0f}", ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(label="Final MTM PnL")
    plt.tight_layout()
    name = "pnl_heatmap_A_k_hawkes.png" if use_hawkes else "pnl_heatmap_A_k_baseline.png"
    plt.savefig(outdir / name, dpi=180)
    plt.close()

def plot_markout_by_horizon(master_df: pd.DataFrame, outdir: Path):
    agg = (
        master_df.groupby(["use_hawkes", "markout_horizon_steps"])["avg_markout_all_fills"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(8, 5))
    for use_hawkes, g in agg.groupby("use_hawkes"):
        label = "Hawkes" if use_hawkes else "Baseline"
        plt.plot(g["markout_horizon_steps"], g["avg_markout_all_fills"], marker="o", label=label)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Markout Horizon Steps")
    plt.ylabel("Average Markout Across Fills")
    plt.title("Average Markout vs Horizon: Hawkes vs Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "avg_markout_by_horizon_hawkes_vs_baseline.png", dpi=180)
    plt.close()

def plot_hawkes_vs_baseline_box(master_df: pd.DataFrame, outdir: Path):
    hawkes = master_df.loc[master_df["use_hawkes"] == True, "final_mtm_pnl"].values
    baseline = master_df.loc[master_df["use_hawkes"] == False, "final_mtm_pnl"].values
    plt.figure(figsize=(7, 5))
    plt.boxplot([hawkes, baseline], labels=["Hawkes", "Baseline"])
    plt.ylabel("Final MTM PnL")
    plt.title("Final MTM PnL Distribution: Hawkes vs Baseline")
    plt.tight_layout()
    plt.savefig(outdir / "hawkes_vs_baseline_pnl_boxplot.png", dpi=180)
    plt.close()

def plot_regime_comparison(regime_master: pd.DataFrame, outdir: Path):
    if regime_master.empty:
        return

    agg = (
        regime_master.groupby(["use_hawkes", "toxicity_regime"])
        .agg(
            avg_markout=("avg_markout", "mean"),
            avg_adverse_selection_cost=("avg_adverse_selection_cost", "mean"),
            mean_pnl=("mean_pnl", "mean"),
        )
        .reset_index()
    )

    regimes = ["low_toxicity", "medium_toxicity", "high_toxicity"]
    x = np.arange(len(regimes))
    width = 0.35

    for metric, filename, title in [
        ("avg_markout", "regime_avg_markout_hawkes_vs_baseline.png", "Average Markout by Toxicity Regime"),
        ("avg_adverse_selection_cost", "regime_adverse_cost_hawkes_vs_baseline.png", "Adverse Selection Cost by Toxicity Regime"),
        ("mean_pnl", "regime_mean_pnl_hawkes_vs_baseline.png", "Mean PnL by Toxicity Regime"),
    ]:
        hawkes_vals = []
        base_vals = []
        for r in regimes:
            h = agg[(agg["toxicity_regime"] == r) & (agg["use_hawkes"] == True)]
            b = agg[(agg["toxicity_regime"] == r) & (agg["use_hawkes"] == False)]
            hawkes_vals.append(h[metric].iloc[0] if len(h) else np.nan)
            base_vals.append(b[metric].iloc[0] if len(b) else np.nan)

        plt.figure(figsize=(8, 5))
        plt.bar(x - width/2, hawkes_vals, width, label="Hawkes")
        plt.bar(x + width/2, base_vals, width, label="Baseline")
        plt.xticks(x, regimes)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=180)
        plt.close()

def build_master_table():
    ensure_dir(OUT_ROOT)
    runs_dir = OUT_ROOT / "runs"
    ensure_dir(runs_dir)

    master_rows = []
    results_map = {}

    for A, k, horizon, use_hawkes in product(A_GRID, K_GRID, MARKOUT_GRID, USE_HAWKES_GRID):
        mode = "hawkes" if use_hawkes else "baseline"
        run_id = f"{mode}__A_{A:.2f}__k_{k:.1f}__h_{horizon}"
        run_dir = runs_dir / run_id
        ensure_dir(run_dir)

        print(f"Running {run_id} ...")
        run_step3(run_dir, A, k, horizon, use_hawkes)

        summary, df = load_summary_and_results(run_dir)
        extra = compute_extra_metrics(df)
        regime_df = regime_breakdown(df)
        regime_df["run_id"] = run_id
        regime_df["use_hawkes"] = use_hawkes
        regime_df["A"] = A
        regime_df["k"] = k
        regime_df["markout_horizon_steps"] = horizon
        regime_df.to_csv(run_dir / "regime_breakdown.csv", index=False)

        row = {
            "run_id": run_id,
            "use_hawkes": use_hawkes,
            "model_type": "hawkes" if use_hawkes else "baseline",
            "A": A,
            "k": k,
            "markout_horizon_steps": horizon,
            "final_mtm_pnl": summary["final_mtm_pnl"],
            "mean_mtm_pnl": summary["mean_mtm_pnl"],
            "std_mtm_pnl": summary["std_mtm_pnl"],
            "fill_count_total": summary["fill_count_total"],
            "avg_bid_fill_prob": summary["avg_bid_fill_prob"],
            "avg_ask_fill_prob": summary["avg_ask_fill_prob"],
            "total_fees_paid": summary["total_fees_paid"],
            "total_adverse_selection_cost": summary["total_adverse_selection_cost"],
            "total_realized_spread_component": summary["total_realized_spread_component"],
            "avg_bid_markout_per_fill": summary["avg_bid_markout_per_fill"],
            "avg_ask_markout_per_fill": summary["avg_ask_markout_per_fill"],
            "mean_latency_us": summary["mean_latency_us"],
            "p99_latency_us": summary["p99_latency_us"],
            "max_latency_us": summary["max_latency_us"],
            "latency_budget_exceeded_count": summary["latency_budget_exceeded_count"],
            **extra,
        }

        master_rows.append(row)
        results_map[run_id] = df

    master_df = pd.DataFrame(master_rows)

    master_df["risk_adjusted_score"] = (
        master_df["final_mtm_pnl"]
        - 0.5 * master_df["max_drawdown"]
        - 50.0 * master_df["avg_adverse_cost_per_step"]
    )

    master_df = master_df.sort_values(
        ["risk_adjusted_score", "final_mtm_pnl"],
        ascending=[False, False]
    ).reset_index(drop=True)

    master_df.to_csv(OUT_ROOT / "step4_master_summary.csv", index=False)
    master_df.head(10).to_csv(OUT_ROOT / "step4_top10_runs.csv", index=False)

    hawkes_vs_baseline = (
        master_df.groupby("model_type")
        .agg(
            mean_final_mtm_pnl=("final_mtm_pnl", "mean"),
            median_final_mtm_pnl=("final_mtm_pnl", "median"),
            mean_risk_adjusted_score=("risk_adjusted_score", "mean"),
            mean_avg_markout_all_fills=("avg_markout_all_fills", "mean"),
            mean_total_adverse_selection_cost=("total_adverse_selection_cost", "mean"),
            mean_fill_count_total=("fill_count_total", "mean"),
            mean_p99_latency_us=("p99_latency_us", "mean"),
        )
        .reset_index()
    )
    hawkes_vs_baseline.to_csv(OUT_ROOT / "step4_hawkes_vs_baseline_summary.csv", index=False)

    plot_top_pnl_paths(master_df, results_map, OUT_ROOT)
    plot_heatmap(master_df, OUT_ROOT, use_hawkes=True)
    plot_heatmap(master_df, OUT_ROOT, use_hawkes=False)
    plot_markout_by_horizon(master_df, OUT_ROOT)
    plot_hawkes_vs_baseline_box(master_df, OUT_ROOT)

    return master_df

def build_regime_master(master_df: pd.DataFrame):
    rows = []
    for run_id in master_df["run_id"]:
        path = OUT_ROOT / "runs" / run_id / "regime_breakdown.csv"
        if path.exists():
            rows.append(pd.read_csv(path))
    if not rows:
        return pd.DataFrame()

    regime_master = pd.concat(rows, ignore_index=True)
    regime_master.to_csv(OUT_ROOT / "step4_regime_master.csv", index=False)

    regime_agg = (
        regime_master.groupby(["use_hawkes", "toxicity_regime"])
        .agg(
            avg_markout=("avg_markout", "mean"),
            mean_pnl=("mean_pnl", "mean"),
            avg_adverse_selection_cost=("avg_adverse_selection_cost", "mean"),
            fill_rate_bid=("fill_rate_bid", "mean"),
            fill_rate_ask=("fill_rate_ask", "mean"),
            mean_inventory_abs=("mean_inventory_abs", "mean"),
        )
        .reset_index()
    )
    regime_agg["model_type"] = regime_agg["use_hawkes"].map({True: "hawkes", False: "baseline"})
    regime_agg.to_csv(OUT_ROOT / "step4_regime_aggregate.csv", index=False)

    plot_regime_comparison(regime_master, OUT_ROOT)
    return regime_master

def write_report(master_df: pd.DataFrame):
    hawkes_summary = pd.read_csv(OUT_ROOT / "step4_hawkes_vs_baseline_summary.csv")
    best = master_df.iloc[0]
    best_hawkes = master_df[master_df["use_hawkes"] == True].iloc[0]
    best_baseline = master_df[master_df["use_hawkes"] == False].iloc[0]

    lines = []
    lines.append("STEP 4 COMPLETE")
    lines.append("")
    lines.append(f"Overall best run: {best['run_id']}")
    lines.append(f"Overall best final MTM PnL: {best['final_mtm_pnl']:.2f}")
    lines.append(f"Overall best risk-adjusted score: {best['risk_adjusted_score']:.2f}")
    lines.append("")

    lines.append("Best Hawkes run")
    lines.append(f"  run_id: {best_hawkes['run_id']}")
    lines.append(f"  final_mtm_pnl: {best_hawkes['final_mtm_pnl']:.2f}")
    lines.append(f"  risk_adjusted_score: {best_hawkes['risk_adjusted_score']:.2f}")
    lines.append(f"  avg_markout_all_fills: {best_hawkes['avg_markout_all_fills']:.6f}")
    lines.append("")

    lines.append("Best Baseline run")
    lines.append(f"  run_id: {best_baseline['run_id']}")
    lines.append(f"  final_mtm_pnl: {best_baseline['final_mtm_pnl']:.2f}")
    lines.append(f"  risk_adjusted_score: {best_baseline['risk_adjusted_score']:.2f}")
    lines.append(f"  avg_markout_all_fills: {best_baseline['avg_markout_all_fills']:.6f}")
    lines.append("")

    lines.append("Hawkes vs Baseline Aggregate")
    for _, row in hawkes_summary.iterrows():
        lines.append(
            f"  {row['model_type']}: mean_final_mtm_pnl={row['mean_final_mtm_pnl']:.2f}, "
            f"median_final_mtm_pnl={row['median_final_mtm_pnl']:.2f}, "
            f"mean_risk_adjusted_score={row['mean_risk_adjusted_score']:.2f}, "
            f"mean_avg_markout_all_fills={row['mean_avg_markout_all_fills']:.6f}, "
            f"mean_total_adverse_selection_cost={row['mean_total_adverse_selection_cost']:.2f}"
        )

    with open(OUT_ROOT / "step4_report.txt", "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))

def main():
    if not STEP3_BIN.exists():
        raise FileNotFoundError(f"Missing Step 3 binary: {STEP3_BIN}")
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing Step 2 features CSV: {FEATURES_CSV}")

    ensure_dir(OUT_ROOT)
    master_df = build_master_table()
    regime_master = build_regime_master(master_df)
    write_report(master_df)

if __name__ == "__main__":
    main()