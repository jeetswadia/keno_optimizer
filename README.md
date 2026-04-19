# Keno Optimizer

A Python toolkit for analyzing Massachusetts Lottery Keno draw data and exploring whether
machine learning models can find any predictive signal in what is — by design — a random game.

The repo combines a **Streamlit dashboard** for historical frequency analysis and ROI
backtesting with a set of **experimental ML scripts** (LSTM, Random Forest, GPU-accelerated
optimizer) that attempt to model the draw data. The honest answer the project converges on:
Keno draws are independent and uniform, and no model in this repo (or anywhere else) reliably
predicts the next draw. The value is in the workflow — clean data pipelines, calibrated payout
math, and a reproducible backtest framework.

> ⚠️ **Disclaimer.** This project is for research, education, and entertainment only. Lottery
> outcomes are random. Past frequencies do not predict future draws. Do not use this code to
> inform real wagers.

---

## What's in the repo

| File | What it does |
|------|--------------|
| `data analysis.py` | Streamlit dashboard: dynamically reconstructs per-draw timestamps (400 draws/day, 5:04 AM start, 3 min apart), filters by date / day-of-week / hour, and runs a historical ROI backtest against the official MA Keno payout tables. |
| `keno_optimizer.py` | Core optimizer logic — searches for number combinations with the best historical hit rate over the dataset. |
| `Keno_optimizer_gpu.py` | GPU-accelerated variant of the optimizer (CUDA / PyTorch or CuPy) for faster combinatorial search. |
| `keno_ai_engine.py` | Higher-level engine that wraps the models and exposes a single entry point for predictions / scoring. |
| `Keno_LSTM.py` | Sequence model (LSTM) that treats the draw history as a time series and predicts likely next-draw numbers. |
| `Keno_Random_Forest.py` | Tree-based baseline model — predicts per-number draw probability from time and recency features. |
| `gpuCheck.py` | Quick sanity-check script that prints whether a CUDA-capable GPU is visible to PyTorch / TensorFlow. |
| `pdfExtractor.py` | Utility for pulling Keno results out of PDF reports (used to build the CSVs). |
| `Keno_data.csv` | Smaller / sample MA Keno draw history. |
| `Keno_data_year.csv` | Full one-year MA Keno draw history (used by the dashboard and models). |
| `LICENSE` | MIT. |

> If a script's behavior differs from the description above, treat the description as a
> placeholder and update — these were inferred from filenames.

---

## Data schema

The CSVs follow the MA State Lottery export format:

| Column | Type | Description |
|--------|------|-------------|
| `drawNumber` | int | Unique ID for the draw within the day. |
| `bonus` | int | Bonus multiplier for that game (1 if no bonus hit, otherwise 3 / 4 / 5 / 10). |
| `drawDate` | date | Lottery operating date (YYYY-MM-DD). |
| `winningNumbers` | string | Comma-separated list of the 20 winning numbers (1–80). |

Each `drawDate` contains exactly **400 draws**. The first draw is at **5:04 AM ET**, with each
subsequent draw 3 minutes later — meaning the 400th draw lands at **1:01 AM the following day**.
The dashboard reconstructs precise per-draw timestamps from this rule.

---

## Quick start

### 1. Clone and set up

```bash
git clone https://github.com/jeetswadia/keno_optimizer.git
cd keno_optimizer
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

There's no `requirements.txt` checked in yet. The dashboard alone needs:

```bash
pip install streamlit pandas plotly altair
```

For the ML scripts you'll additionally want:

```bash
pip install scikit-learn torch tensorflow numpy tqdm
```

For PDF extraction:

```bash
pip install pdfplumber
```

### 3. Run the dashboard

```bash
streamlit run "data analysis.py"
```

If `streamlit` isn't on your PATH (common on Windows):

```bash
py -m streamlit run "data analysis.py"
```

The browser opens at `http://localhost:8501`. The sidebar lets you:

- Upload a CSV or auto-load `Keno_data_year.csv` from the repo root
- Filter by date range, day of week, and a specific hour (0–23)
- View number frequency with hot/cold shading
- Drill into a weekly trend heatmap when a specific hour is selected
- Run an ROI backtest with 1–12 picked numbers against the filtered draws

---

## ROI calculator — what it actually computes

Picks 1–12 distinct numbers and replays them against the filtered historical draws.

- **Cost per draw:** $1 base + $1 Bonus = $2 for spots 1–9. Spots 10/11/12 are not Bonus
  eligible per MA rules, so cost drops to $1/draw and no multiplier is applied.
- **Payouts:** uses the official MA State Lottery prize table for each spot size, including
  the Match-0 prizes on 10/11/12-spot games.
- **Bonus multiplier:** when Bonus is active, the historical `bonus` column for each draw is
  applied as a multiplier (1 / 3 / 4 / 5 / 10) on the base prize.
- **Outputs:** total wagered, total won, net P/L, ROI %, biggest single win, and a hit
  histogram (matched-N → times-hit).

The expected ROI is meaningfully negative for every spot size — that's the whole point of how
the lottery is priced. The calculator just puts a number on it for the slice of history you
selected.

---

## ML scripts — what to expect

The Random Forest, LSTM, and GPU optimizer scripts are exploratory. They train on past draw
sequences and try to score which numbers are more likely to appear next.

**Key honest caveat:** MA Keno uses a certified hardware RNG. Each draw is independent and
uniform over the 80-number space. No amount of feature engineering, model capacity, or GPU
horsepower will produce a predictor that beats the house edge over a long enough horizon.
These scripts are useful for:

- Practicing time-series modeling on a clean, well-defined dataset
- Demonstrating that ML accuracy converges to chance on truly random data
- Benchmarking GPU vs CPU pipelines

Run any of them directly:

```bash
python Keno_LSTM.py
python Keno_Random_Forest.py
python Keno_optimizer_gpu.py
```

Check GPU visibility first:

```bash
python gpuCheck.py
```

---

## Project layout

```
keno_optimizer/
├── data analysis.py            # Streamlit dashboard
├── keno_optimizer.py           # Core optimizer
├── Keno_optimizer_gpu.py       # GPU-accelerated optimizer
├── keno_ai_engine.py           # Model orchestration
├── Keno_LSTM.py                # LSTM sequence model
├── Keno_Random_Forest.py       # Random Forest baseline
├── gpuCheck.py                 # GPU sanity check
├── pdfExtractor.py             # PDF → CSV utility
├── Keno_data.csv               # Sample dataset
├── Keno_data_year.csv          # Full year dataset
├── LICENSE                     # MIT
└── README.md
```

---

## Roadmap / ideas

- Add a `requirements.txt` and pin versions
- Containerize the dashboard with a `Dockerfile`
- Add unit tests for the payout math (the trickiest part to get right)
- Persist backtest results so they can be diffed across data refreshes
- Document the PDF extractor's expected input format
- Add a "Monte Carlo expected ROI" view to compare empirical vs theoretical EV per spot

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Author

Built by [Jeet Swadia](https://github.com/jeetswadia).
