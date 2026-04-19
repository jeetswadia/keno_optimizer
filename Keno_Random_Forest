# ============================================================
# RF LOTTERY / KENO PREDICTOR 
# Google Colab  |  scikit-learn RandomForestRegressor
# ============================================================
# USAGE:  Edit the parameters in run_rf_pipeline() at the
#         bottom of this cell, then press Shift+Enter.
# ============================================================

import numpy as np
import pandas as pd
import time
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING & PARSING
# ──────────────────────────────────────────────────────────────

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the CSV file.
    Expected columns: drawNumber, winningNumbers
    winningNumbers example: "16,46,60,26,73,12,55,3,68,34,..."
    The dataset is already pre-sorted chronologically.
    NO sorting is applied here.
    """
    print(f"[1/6] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)

    # Validate required columns exist
    required = {"drawNumber", "winningNumbers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"      Loaded {len(df):,} rows  |  "
          f"Draw range: {df['drawNumber'].min()} → {df['drawNumber'].max()}")
    return df


# ──────────────────────────────────────────────────────────────
# SECTION 2 — MULTI-HOT ENCODING  (int8 — memory critical)
# ──────────────────────────────────────────────────────────────

def parse_winning_numbers(s: str) -> list:
    """
    Parse a comma-separated string of integers into a Python list.
    Numbers are NOT sorted — preserved exactly as they appear in the CSV.
    Example: "16,46,60" → [16, 46, 60]
    """
    return [int(x.strip()) for x in s.split(",")]


def multihot_encode(numbers: list, pool_size: int = 80) -> np.ndarray:
    """
    Convert a list of drawn numbers to an 80-length binary vector.
    Index n-1 is set to 1 for each drawn number n (1-indexed pool).
    Dtype is int8 (1 byte per element) — critical for RAM efficiency.
    """
    vec = np.zeros(pool_size, dtype=np.int8)
    for n in numbers:
        if 1 <= n <= pool_size:
            vec[n - 1] = np.int8(1)
    return vec


def encode_dataset(df: pd.DataFrame, pool_size: int = 80) -> np.ndarray:
    """
    Encode all rows in the DataFrame.
    Returns shape: (num_draws, pool_size) with dtype=int8.

    Memory comparison for 440,000 rows × 80 columns:
      float64  →  ~275 MB    (default numpy)
      int8     →   ~34 MB    ← what we use
    """
    print(f"[2/6] Encoding {len(df):,} draws into {pool_size}-dim multi-hot vectors (int8)...")
    t0 = time.time()

    encoded = np.vstack([
        multihot_encode(parse_winning_numbers(row), pool_size)
        for row in df["winningNumbers"]
    ]).astype(np.int8)  # ← CRITICAL: explicit int8 cast after stack

    mem_mb = encoded.nbytes / (1024 ** 2)
    print(f"      Encoded matrix shape : {encoded.shape}")
    print(f"      Dtype                : {encoded.dtype}")
    print(f"      Memory footprint     : {mem_mb:.1f} MB")
    print(f"      Completed in         : {time.time() - t0:.2f}s")
    return encoded


# ──────────────────────────────────────────────────────────────
# SECTION 3 — SLIDING WINDOW FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────

def create_sliding_window(
    encoded_matrix: np.ndarray,
    window_size: int = 10
) -> tuple:
    """
    Generate (X, y) pairs using a sliding window over the encoded draws.

    For each position i (from window_size to end):
      X[i] = encoded_matrix[i-window_size : i].flatten()
            → shape (window_size * pool_size,)   e.g. 10 * 80 = 800 features
      y[i] = encoded_matrix[i]
            → shape (pool_size,)                 e.g. 80 binary targets

    Random Forests need 2D input (n_samples, n_features), so the window
    must be flattened — unlike LSTMs which accept 3D sequences.

    Both X and y are cast as int8 to minimise RAM usage.
    NO sorting is applied to numbers within any draw.
    """
    print(f"[3/6] Building sliding window  (window_size={window_size})...")
    t0 = time.time()

    n_rows, pool_size = encoded_matrix.shape
    n_samples = n_rows - window_size
    n_features = window_size * pool_size

    # Pre-allocate int8 arrays (avoids repeated reallocation)
    X = np.empty((n_samples, n_features), dtype=np.int8)
    y = np.empty((n_samples, pool_size),  dtype=np.int8)

    for i in range(n_samples):
        X[i] = encoded_matrix[i : i + window_size].flatten()
        y[i] = encoded_matrix[i + window_size]

    x_mem = X.nbytes / (1024 ** 2)
    y_mem = y.nbytes / (1024 ** 2)
    print(f"      X shape   : {X.shape}  ({x_mem:.1f} MB, dtype={X.dtype})")
    print(f"      y shape   : {y.shape}  ({y_mem:.1f} MB, dtype={y.dtype})")
    print(f"      Completed : {time.time() - t0:.2f}s")
    return X, y


# ──────────────────────────────────────────────────────────────
# SECTION 4 — MODEL TRAINING
# ──────────────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 50,
    max_depth: int    = 15
) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor on the sliding-window dataset.

    Why Regressor (not Classifier)?
      The regressor outputs floats in [0.0, 1.0] — the fraction of trees
      that voted "1" for each number. This is a natural probability proxy
      that allows clean ranking without a hard 0/1 decision.

    Key parameters:
      n_estimators : number of trees (more = better accuracy, more RAM/time)
      max_depth    : max tree depth  (limits RAM; 15 is a good Colab default)
      n_jobs=-1    : use ALL available CPU cores (essential on Colab)
    """
    print(f"[4/6] Training RandomForestRegressor...")
    print(f"      n_estimators = {n_estimators}")
    print(f"      max_depth    = {max_depth}")
    print(f"      n_jobs       = -1  (all cores)")
    print(f"      Training samples: {X_train.shape[0]:,}")
    print(f"      Feature count   : {X_train.shape[1]:,}")
    t0 = time.time()

    model = RandomForestRegressor(
        n_estimators = n_estimators,
        max_depth    = max_depth,
        n_jobs       = -1,
        random_state = 42,
        verbose      = 0
    )

    # scikit-learn RandomForestRegressor natively supports multi-output
    # (y with shape n_samples × n_targets) — no wrapper needed.
    model.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"      Training complete in {elapsed:.1f}s  ({elapsed/60:.2f} min)")
    return model


# ──────────────────────────────────────────────────────────────
# SECTION 5 — MODEL EVALUATION
# ──────────────────────────────────────────────────────────────

def evaluate_model(
    model: RandomForestRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Evaluate the trained model on held-out test data.

    Metrics:
      RMSE     — Root Mean Squared Error (lower is better)
      ROC-AUC  — Macro-averaged AUC across all 80 binary targets
                 (higher is better; 0.5 = random, 1.0 = perfect)
    """
    print("[5/6] Evaluating model on test set...")
    t0 = time.time()

    y_pred = model.predict(X_test)  # shape: (n_test, pool_size), floats

    # RMSE (global, across all targets)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Macro-averaged ROC-AUC (per number, then averaged)
    auc_scores = []
    for col in range(y_test.shape[1]):
        col_true = y_test[:, col]
        col_pred = y_pred[:, col]
        if col_true.sum() > 0:  # skip columns with no positive examples
            try:
                auc_scores.append(roc_auc_score(col_true, col_pred))
            except Exception:
                pass

    macro_auc = float(np.mean(auc_scores)) if auc_scores else float("nan")

    print(f"      RMSE (global)    : {rmse:.5f}")
    print(f"      ROC-AUC (macro)  : {macro_auc:.5f}")
    print(f"      Evaluation done  : {time.time() - t0:.2f}s")
    return {"rmse": rmse, "roc_auc_macro": macro_auc}


# ──────────────────────────────────────────────────────────────
# SECTION 6 — AUTOREGRESSIVE FUTURE DRAW PREDICTION
# ──────────────────────────────────────────────────────────────

def predict_future_draws(
    model:          RandomForestRegressor,
    last_window:    np.ndarray,
    future_draws:   int = 10,
    num_to_select:  int = 10,
    pool_size:      int = 80
) -> list:
    """
    Predict N future draws using an autoregressive (chained) strategy.

    The model has never seen actual future draws.  Instead, after each
    prediction step, we synthesise a "draw" from the model's top-20
    picks, append it to the rolling window, drop the oldest draw, and
    use the updated window to predict the next step.

    Algorithm:
      window  ← last real window_size draws  (shape: window_size × pool_size)
      for step in 1..future_draws:
          flat_input  ← window.flatten()             → 1D (window*pool,)
          scores      ← model.predict([flat_input])  → (1, pool_size) floats
          top20_idx   ← argsort(scores)[-20:]        → indices of top 20
          synthetic   ← zeros(pool_size, int8); synthetic[top20_idx] = 1
          topK_idx    ← argsort(scores)[-num_to_select:]
          topK_nums   ← topK_idx + 1  (convert 0-indexed → 1-indexed)
          window      ← vstack(window[1:], synthetic)   # slide forward

    Parameters
    ----------
    last_window    : np.ndarray, shape (window_size, pool_size) — seed context
    future_draws   : number of future draws to predict
    num_to_select  : top-K numbers to include in each prediction output
    pool_size      : size of the number pool (80 for Keno)

    Returns
    -------
    List of dicts, one per future draw:
      { "draw": T+n, "picks": [int, ...], "scores": [float, ...] }
    """
    window = last_window.copy().astype(np.int8)  # (window_size, pool_size)
    results = []

    print(f"\n[6/6] Autoregressive prediction: {future_draws} future draw(s), "
          f"top {num_to_select} picks each")
    print("=" * 60)

    for step in range(1, future_draws + 1):
        # ── a) Flatten window to 1D feature vector
        flat_input = window.flatten().reshape(1, -1)  # (1, window*pool)

        # ── b) Model prediction: continuous scores ∈ [0.0, 1.0]
        scores = model.predict(flat_input)[0]  # (pool_size,)

        # ── c) Rank all numbers by score (descending)
        ranked_idx = np.argsort(scores)[::-1]  # highest score first

        # ── d) Build synthetic draw from top-20 (Keno draw size)
        #      This keeps the window's statistical signature realistic
        synthetic = np.zeros(pool_size, dtype=np.int8)
        top20_idx = ranked_idx[:20]
        synthetic[top20_idx] = np.int8(1)

        # ── e) Extract top-K picks for output (user-facing selection)
        topK_idx   = ranked_idx[:num_to_select]
        topK_nums  = sorted((topK_idx + 1).tolist())  # 1-indexed, sorted for display
        topK_scores = [round(float(scores[i]), 4) for i in topK_idx]

        results.append({
            "draw"  : f"T+{step}",
            "picks" : topK_nums,
            "scores": topK_scores
        })

        # ── f) Slide the window forward: drop oldest, append synthetic
        window = np.vstack([window[1:], synthetic])

        # ── Pretty print this step
        print(f"  ══ Future Draw T+{step} ══")
        print(f"     Top {num_to_select} picks  : {topK_nums}")
        score_str = ", ".join(f"{s:.4f}" for s in topK_scores)
        print(f"     RF scores    : [{score_str}]")
        print()

    return results


# ──────────────────────────────────────────────────────────────
# MASTER PIPELINE FUNCTION
# ──────────────────────────────────────────────────────────────

def run_rf_pipeline(
    filepath:       str = "Keno_data_year_april18.csv",
    window_size:    int = 10,
    n_estimators:   int = 50,
    max_depth:      int = 15,
    num_to_select:  int = 10,
    future_draws:   int = 10,
    pool_size:      int = 80
) -> dict:
    """
    End-to-end pipeline:
      1. Load CSV
      2. Multi-hot encode draws (int8)
      3. Build sliding-window X/y dataset (int8)
      4. Chronological train/test split (80/20, shuffle=False)
      5. Train RandomForestRegressor (n_jobs=-1)
      6. Evaluate on test set (RMSE + ROC-AUC)
      7. Predict future_draws steps autogressively
      8. Return all results as a dictionary

    Parameters
    ----------
    filepath      : path to CSV (local Colab or Google Drive)
    window_size   : number of past draws used as context
    n_estimators  : trees in the forest (speed/accuracy trade-off)
    max_depth     : max tree depth (memory/accuracy trade-off)
    num_to_select : top-K numbers to display per future draw
    future_draws  : how many future draw steps to predict
    pool_size     : size of the number pool (80 for standard Keno)
    """
    print("=" * 60)
    print("  RF KENO PREDICTOR — PIPELINE START")
    print("=" * 60)
    pipeline_start = time.time()

    # ── 1. Load ──────────────────────────────────────────────
    df = load_dataset(filepath)

    # ── 2. Encode ────────────────────────────────────────────
    encoded = encode_dataset(df, pool_size=pool_size)

    # ── 3. Sliding Window ────────────────────────────────────
    X, y = create_sliding_window(encoded, window_size=window_size)

    # ── 4. Chronological Train / Test Split (NO shuffle) ─────
    print(f"[4a] Splitting data (80% train / 20% test, shuffle=False)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.20,
        shuffle      = False,   # CRITICAL: preserve temporal order
        random_state = 42
    )
    print(f"      Train samples : {X_train.shape[0]:,}")
    print(f"      Test  samples : {X_test.shape[0]:,}")

    # ── 5. Train ─────────────────────────────────────────────
    model = train_model(
        X_train,
        y_train,
        n_estimators = n_estimators,
        max_depth    = max_depth
    )

    # ── 6. Evaluate ──────────────────────────────────────────
    metrics = evaluate_model(model, X_test, y_test)

    # ── 7. Autoregressive Prediction ─────────────────────────
    #  Seed window = the very last window_size real draws
    last_window = encoded[-window_size:]          # shape: (window_size, pool_size)

    predictions = predict_future_draws(
        model         = model,
        last_window   = last_window,
        future_draws  = future_draws,
        num_to_select = num_to_select,
        pool_size     = pool_size
    )

    # ── 8. Summary ───────────────────────────────────────────
    total_time = time.time() - pipeline_start
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Total wall time : {total_time:.1f}s  ({total_time/60:.2f} min)")
    print(f"  RMSE            : {metrics['rmse']:.5f}")
    print(f"  ROC-AUC (macro) : {metrics['roc_auc_macro']:.5f}")
    print("=" * 60)
    print("\n📌 PREDICTED NUMBERS SUMMARY")
    print("-" * 40)
    for r in predictions:
        print(f"  Draw {r['draw']}  →  {r['picks']}")
    print("-" * 40)

    return {
        "model"      : model,
        "metrics"    : metrics,
        "predictions": predictions,
        "encoded"    : encoded,
        "X_train"    : X_train,
        "X_test"     : X_test,
        "y_train"    : y_train,
        "y_test"     : y_test
    }


# ──────────────────────────────────────────────────────────────
# ▶  ENTRY POINT — edit parameters below and run the cell
# ──────────────────────────────────────────────────────────────

# Optional: mount Google Drive if your CSV is stored there
# from google.colab import drive
# drive.mount('/content/drive')
# filepath = '/content/drive/MyDrive/keno_draws.csv'

results = run_rf_pipeline(
    filepath      = "Keno_data_year_april18.csv",
    window_size   = 10,
    n_estimators  = 50,
    max_depth     = 15,
    num_to_select = 10,
    future_draws  = 10,
    pool_size     = 80
)

# Access results programmatically:
# results["model"]       → trained RandomForestRegressor
# results["metrics"]     → {"rmse": ..., "roc_auc_macro": ...}
# results["predictions"] → list of {"draw": "T+1", "picks": [...], "scores": [...]}
