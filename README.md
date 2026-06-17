# Hawkes Market Maker

Hawkes Market Maker is a low-latency market-making research and engineering project that combines microstructure-driven feature engineering, Hawkes-process order flow modeling, and an event-driven C++ quoting engine into one end-to-end pipeline. The project starts from raw market data, builds replayable features, calibrates Hawkes intensities, runs a queue- and toxicity-aware market-making engine, and then evaluates the strategy against a baseline through PnL, markout, regime, and latency analysis. For the full technical write-up, detailed interpretation of the plots, and a deep dive into the design decisions and results, see [`Report.md`](./Report.md).

## Why this project

Most trading projects stop at either research code or systems code. This project was built to show both.

It combines:

- microstructure-aware feature engineering,
- Hawkes-based modeling of clustered order flow,
- a replayable C++ event-driven engine,
- explicit inventory, queue, fee, and toxicity logic,
- parameter sweeps and regime analysis,
- and benchmarked runtime performance.

## Pipeline

The project is organized as a four-stage pipeline.

### Step 1: Feature engineering

Raw market data is transformed into a structured feature table. This stage produces time-aligned features such as midprice, microprice, volatility proxies, and order flow imbalance so the strategy can reason about market state efficiently during replay.

**Outputs**

- `output/step1_features/feature_table_1s.csv`
- `output/step1_features/feature_table_1s.parquet`
- `output/step1_features/ohlcv_1s.csv`
- `output/step1_features/ohlcv_1s.parquet`

### Step 2: Hawkes calibration

This stage labels trades, estimates order-flow activity, and generates Hawkes-based intensity features. The point of this layer is to capture self-excitation in order arrivals so the quoting engine can respond not only to the current book state but also to recent clustering in flow.

**Outputs**

- `output/step2_hawkes/feature_table_step2_1s.csv`
- `output/step2_hawkes/feature_table_step2_1s.parquet`
- `output/step2_hawkes/hawkes_intensities_1s.parquet`
- `output/step2_hawkes/baseline_intensities_1s.parquet`
- `output/step2_hawkes/event_counts_1s.parquet`
- `output/step2_hawkes/trades_labeled.parquet`
- `output/step2_hawkes/step2_summary.json`

### Step 3: Event-driven C++ engine

The core market-making engine is implemented in C++ and compiled as `bin/step3_mm`. It replays the feature stream event by event, extracts signals, generates quotes, applies toxicity and queue logic, enforces inventory/risk constraints, simulates fills, measures latency, and writes detailed results.

**Core source**

- `engine/event_driven_mm_engine.cpp`
- `bin/step3_mm`

**Outputs**

- `output/step3_engine_cpp/engine_results.csv`
- `output/step3_engine_cpp/step3_summary.json`

### Step 4: Analysis and validation

This stage aggregates run outputs into comparative summaries and plots. It is where Hawkes versus baseline performance is evaluated through final PnL, markout, adverse selection, regime analysis, and latency profiling.

**Outputs**

- `output/step4_analysis/step4_hawkes_vs_baseline_summary.csv`
- `output/step4_analysis/step4_master_summary.csv`
- `output/step4_analysis/step4_regime_aggregate.csv`
- `output/step4_analysis/step4_regime_master.csv`
- `output/step4_analysis/step4_report.txt`
- `output/step4_analysis/step4_top10_runs.csv`
- `plots/`

## Repository structure

```text
Hawkes Market Maker/
├── benchmark_results/
│   ├── runs/
│   ├── benchmark_raw_runs.csv
│   ├── benchmark_report.md
│   └── benchmark_summary.json
├── bin/
│   └── step3_mm
├── data/
│   └── databento/
│       ├── aapl_mbp-10_2024-06-03.csv
│       └── aapl_trades_2024-06-03.csv
├── engine/
│   └── event_driven_mm_engine.cpp
├── output/
│   ├── step1_features/
│   ├── step2_hawkes/
│   ├── step3_engine_cpp/
│   └── step4_analysis/
├── plots/
├── scripts/
│   ├── benchmark_engine.py
│   ├── plot_graphs.py
│   ├── Script.py
│   ├── step1_build_feature_table.py
│   ├── step2_trade_label_and_hawkes.py
│   └── step4_analysis.py
└── Report.md
```

## Strategy design

The strategy uses an Avellaneda-Stoikov style market-making core and augments it with microstructure-aware adjustments.

The engine incorporates:

- reservation-price and spread logic,
- order flow imbalance and microprice skew,
- Hawkes intensity signals,
- inventory limits,
- queue haircut logic,
- toxicity haircuts,
- maker fees,
- fill probability modeling,
- markout tracking,
- and latency profiling.

This combination is important because it makes the simulator far more realistic than a naive quoting model that ignores execution quality and inventory risk.

## Key results

The project compares a Hawkes-informed market maker against a baseline strategy under the same evaluation framework.

### Best-run comparison

| Metric | Hawkes | Baseline |
|---|---:|---:|
| Best run ID | `hawkes__A_0.12__k_4.0__h_5` | `baseline__A_0.12__k_4.0__h_5` |
| Final MTM PnL | 11,592.47 | 10,723.18 |
| Risk-adjusted score | 11,450.01 | 10,527.80 |

### Engine benchmark summary

| Metric | Value |
|---|---:|
| Measured runs | 20 |
| Mean wall time | 221.887 ms |
| p95 wall time | 224.271 ms |
| Mean throughput | 105,464.13 events/sec |
| Mean peak RSS | 16,758.8 KB |
| Median engine p50 latency | 0.076 us |
| Median engine p95 latency | 0.090 us |
| Median engine p99 latency | 0.102 us |
| Worst observed max latency | 9.308 us |
| Latency budget exceedances | 0 |

These results show that the Hawkes-driven variant improves final PnL while the C++ engine remains fast, stable, and comfortably within the configured latency budget.

## Plots

The `plots/` directory contains the main visual outputs used to evaluate the strategy.

- `01_pnl_path_hawkes_vs_baseline.jpg` — best-run PnL path comparison.
- `02_inventory_path_hawkes_vs_baseline.jpg` — inventory evolution under Hawkes and baseline.
- `03_markout_by_horizon.jpg` — average markout by horizon.
- `04_pnl_decomposition.jpg` — spread capture, adverse selection, fees, and final PnL.
- `05_mean_pnl_by_regime.jpg` — mean PnL by toxicity regime.
- `05_avg_markout_by_regime.jpg` — markout comparison by regime.
- `05_avg_adverse_selection_cost_by_regime.jpg` — adverse selection cost by regime.
- `06_latency_cdf.jpg` — latency distribution and tail behavior.
- `engine_arch.jpg` and `arch_diag.jpg` — architecture diagrams.

## Running the project

The repository is designed as a staged workflow.

### 1. Build features

```bash
python scripts/step1_build_feature_table.py
```

### 2. Calibrate Hawkes and label trades

```bash
python scripts/step2_trade_label_and_hawkes.py
```

### 3. Build the C++ engine

```bash
g++ -O3 -std=c++17 -o bin/step3_mm engine/event_driven_mm_engine.cpp
```

### 4. Run the engine

Run the engine on the Step 2 feature table and write outputs to `output/step3_engine_cpp/`.

```bash
./bin/step3_mm \
  --features output/step2_hawkes/feature_table_step2_1s.csv \
  --outdir output/step3_engine_cpp
```

### 5. Run analysis

```bash
python scripts/step4_analysis.py
python scripts/plot_graphs.py
```

### 6. Benchmark the engine

```bash
python scripts/benchmark_engine.py
```


## Next directions

Natural extensions for this project include:

- binary replay instead of CSV input,
- finer-grained engine microbenchmarks,
- broader fill-model sensitivity analysis,
- multi-instrument evaluation,
- and more explicit testing around deterministic engine outputs.


