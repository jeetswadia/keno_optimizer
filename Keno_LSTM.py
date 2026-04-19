
# ============================================================
#  KENO / LOTTERY LSTM PREDICTOR  —  Single Google Colab Cell
#  TensorFlow 2.x | Keras | GPU-Optimised (float32 enforced)
# ============================================================
#
#  USAGE:
#    1. Upload your CSV to Colab (or mount Google Drive).
#    2. Set `filepath` in run_pipeline() to your CSV path.
#    3. Runtime → Change runtime type → GPU (T4 recommended).
#    4. Run this cell.
#
#  CSV FORMAT EXPECTED:
#    - Column `drawNumber`    : integer draw ID (descending OK)
#    - Column `winningNumbers`: comma-separated string of 20
#                               integers in range [1, 80]
#      e.g.  "16,46,60,26,73,5,12,33,55,68,2,9,44,71,80,7,19,38,62,77"
#
#  ⚠ MEMORY NOTE:
#    If you increase window_size beyond 150, LOWER batch_size
#    (e.g. 16 or 8) to avoid CUDA_ERROR_INVALID_HANDLE or
#    "Dst tensor is not initialized" errors on Colab GPUs.
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ════════════════════════════════════════════════════════════
#  SECTION 1 — DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════

def load_and_preprocess(filepath: str, pool_size: int = 80) -> np.ndarray:
    """
    Load the CSV, sort chronologically, parse winning numbers,
    and multi-hot encode each draw into a float32 binary vector.

    Args:
        filepath  : Path to the CSV file.
        pool_size : Total number pool (default 80 for Keno).

    Returns:
        encoded   : np.ndarray of shape (n_draws, pool_size),
                    dtype=float32. STRICTLY float32 to prevent
                    GPU memory overflow on 440k+ row datasets.
    """
    print("📂 Loading dataset …")
    df = pd.read_csv(filepath)

    # ── Validate required columns ────────────────────────────
    required = {"drawNumber", "winningNumbers"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # ── Sort ascending (oldest draw first) ───────────────────
    df = df.sort_values("drawNumber", ascending=True).reset_index(drop=True)
    print(f"   ✔ Rows loaded & sorted: {len(df):,}")
    print(f"   ✔ Draw range: {df['drawNumber'].iloc[0]} → {df['drawNumber'].iloc[-1]}")

    # ── Parse winning numbers ────────────────────────────────
    def parse_numbers(raw: str) -> list[int]:
        nums = [int(x.strip()) for x in raw.split(",")]
        if len(nums) != 20:
            raise ValueError(f"Expected 20 numbers, got {len(nums)}: {raw}")
        for n in nums:
            if not (1 <= n <= pool_size):
                raise ValueError(f"Number {n} out of range [1, {pool_size}]")
        return sorted(nums)          # sort for consistency

    print("🔢 Parsing & sorting winning numbers …")
    df["parsed"] = df["winningNumbers"].apply(parse_numbers)

    # ── Multi-hot encode → STRICTLY float32 ─────────────────
    #    CRITICAL: np.zeros defaults to float64.
    #    On 440k rows × 80 cols that is ~282 MB in float64
    #    vs ~141 MB in float32. During tf.constant() / tensor
    #    copying this difference can crash the Colab GPU RAM.
    print("🔥 Multi-hot encoding (float32 enforced) …")
    n_draws  = len(df)
    encoded  = np.zeros((n_draws, pool_size), dtype=np.float32)   # ← float32!

    for i, numbers in enumerate(df["parsed"]):
        for num in numbers:
            encoded[i, num - 1] = 1.0   # zero-indexed: num 1 → index 0

    mem_mb = encoded.nbytes / (1024 ** 2)
    print(f"   ✔ Encoded array shape : {encoded.shape}")
    print(f"   ✔ Dtype               : {encoded.dtype}   ← ✅ must be float32")
    print(f"   ✔ Memory footprint     : {mem_mb:.1f} MB")

    return encoded


# ════════════════════════════════════════════════════════════
#  SECTION 2 — SLIDING WINDOW DATASET BUILDER
# ════════════════════════════════════════════════════════════

def build_sequences(
    encoded:     np.ndarray,
    window_size: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create supervised learning pairs using a sliding window.

        X[i] = encoded[i : i + window_size]       shape (window_size, 80)
        y[i] = encoded[i + window_size]            shape (80,)

    Both arrays are cast to float32 explicitly.

    Args:
        encoded     : Multi-hot float32 array, shape (n_draws, 80).
        window_size : Number of past draws used to predict the next.

    Returns:
        X : np.ndarray, shape (n_samples, window_size, 80), float32
        y : np.ndarray, shape (n_samples, 80),              float32
    """
    print(f"\n🪟 Building sliding-window sequences (window={window_size}) …")
    n        = len(encoded)
    n_samples = n - window_size

    if n_samples <= 0:
        raise ValueError(
            f"window_size ({window_size}) >= total draws ({n}). "
            "Reduce window_size or supply more data."
        )

    # Pre-allocate with float32 to avoid any implicit upcast
    X = np.empty((n_samples, window_size, encoded.shape[1]), dtype=np.float32)
    y = np.empty((n_samples, encoded.shape[1]),              dtype=np.float32)

    for i in range(n_samples):
        X[i] = encoded[i : i + window_size]
        y[i] = encoded[i + window_size]

    print(f"   ✔ X shape: {X.shape}  dtype={X.dtype}")
    print(f"   ✔ y shape: {y.shape}  dtype={y.dtype}")
    print(f"   ✔ Training samples: {n_samples:,}")
    return X, y


# ════════════════════════════════════════════════════════════
#  SECTION 3 — MODEL DEFINITION
# ════════════════════════════════════════════════════════════

def build_model(
    window_size: int = 100,
    pool_size:   int = 80
) -> keras.Model:
    """
    Build a two-layer stacked LSTM model with an explicit
    Input layer (avoids Keras input_shape deprecation warnings).

    Architecture:
        Input  (window_size, pool_size)
        LSTM   128 units, return_sequences=True
        Dropout 0.2
        LSTM   64  units, return_sequences=False
        Dropout 0.2
        Dense  pool_size units, activation='sigmoid'

    Loss : binary_crossentropy
        → Each of the 80 output nodes is an independent
          probability that the corresponding number is drawn.

    Optimizer : Adam (default lr=1e-3)
    """
    print("\n🏗️  Building model …")

    inputs = keras.Input(shape=(window_size, pool_size), name="draw_sequence")

    x = layers.LSTM(128, return_sequences=True, name="lstm_1")(inputs)
    x = layers.Dropout(0.2, name="dropout_1")(x)

    x = layers.LSTM(64, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)

    outputs = layers.Dense(pool_size, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="KenoLSTM")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


# ════════════════════════════════════════════════════════════
#  SECTION 4 — TRAINING
# ════════════════════════════════════════════════════════════

def train_model(
    model:      keras.Model,
    X:          np.ndarray,
    y:          np.ndarray,
    epochs:     int = 35,
    batch_size: int = 32,
    val_split:  float = 0.1
) -> keras.callbacks.History:
    """
    Train the model with EarlyStopping and ReduceLROnPlateau.

    Args:
        model      : Compiled Keras model.
        X          : Input sequences, shape (n, window_size, 80).
        y          : Target draws,    shape (n, 80).
        epochs     : Max training epochs (EarlyStopping may stop earlier).
        batch_size : Mini-batch size.
                     ⚠ Lower this (16 or 8) if you raise window_size
                       to prevent CUDA_ERROR_INVALID_HANDLE errors.
        val_split  : Fraction of data used for validation.

    Returns:
        history    : Keras History object.
    """
    print(f"\n🚀 Training  |  epochs={epochs}  batch_size={batch_size}  val_split={val_split}")
    print(f"   GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    t0      = time.time()
    history = model.fit(
        X, y,
        epochs          = epochs,
        batch_size      = batch_size,
        validation_split= val_split,
        callbacks       = callbacks,
        verbose         = 1
    )
    elapsed = time.time() - t0
    print(f"\n   ✔ Training complete in {elapsed:.1f}s")
    print(f"   ✔ Best val_loss : {min(history.history['val_loss']):.6f}")
    return history


# ════════════════════════════════════════════════════════════
#  SECTION 5 — AUTOREGRESSIVE MULTI-STEP PREDICTION
# ════════════════════════════════════════════════════════════

def predict_future_draws(
    model:          keras.Model,
    seed_window:    np.ndarray,
    future_draws:   int = 3,
    num_to_select:  int = 7,
    pool_size:      int = 80,
    numbers_per_draw: int = 20
) -> list[list[int]]:
    """
    Predict N future draws autoregressively.

    At each step T:
      1. Feed the current window (shape 1, window_size, 80) to the model.
      2. Get an 80-dim sigmoid probability vector.
      3. Select the top `numbers_per_draw` (20) probabilities
         → encode as a synthetic draw (80-dim float32 binary vector).
      4. Slide the window: drop oldest draw, append synthetic draw.
      5. From the same probability vector, extract the top
         `num_to_select` (e.g. 7) numbers as the prediction output.

    Args:
        model            : Trained Keras model.
        seed_window      : np.ndarray shape (window_size, 80), float32.
                           The last `window_size` real draws.
        future_draws     : How many future draws to predict.
        num_to_select    : How many top numbers to output per draw.
        pool_size        : Number pool size (80).
        numbers_per_draw : Numbers drawn per game (20 for Keno).

    Returns:
        predictions : List of lists. Each inner list contains
                      `num_to_select` predicted numbers (1-indexed),
                      sorted ascending.
    """
    print(f"\n🔮 Predicting {future_draws} future draw(s)  "
          f"[top {num_to_select} numbers each] …\n")

    # Work on a copy to avoid mutating caller's array
    window = seed_window.astype(np.float32).copy()   # (window_size, 80)

    predictions = []

    for step in range(1, future_draws + 1):

        # ── Forward pass ─────────────────────────────────────
        x_input  = window[np.newaxis, :, :]           # (1, window_size, 80)
        probs    = model.predict(x_input, verbose=0)  # (1, 80)
        probs    = probs[0]                            # (80,)

        # ── Top-K output (user-facing prediction) ────────────
        top_k_idx    = np.argsort(probs)[-num_to_select:]
        top_k_numbers= sorted([idx + 1 for idx in top_k_idx])  # 1-indexed

        # ── Synthetic draw for window slide ──────────────────
        top20_idx    = np.argsort(probs)[-numbers_per_draw:]
        synthetic    = np.zeros(pool_size, dtype=np.float32)
        synthetic[top20_idx] = 1.0

        # ── Slide window: drop oldest, append synthetic ───────
        window = np.vstack([window[1:], synthetic[np.newaxis, :]])

        predictions.append(top_k_numbers)
        print(f"   Draw T+{step:02d}  →  {top_k_numbers}")

    return predictions


# ════════════════════════════════════════════════════════════
#  SECTION 6 — PIPELINE ENTRY POINT
# ════════════════════════════════════════════════════════════

def run_pipeline(
    filepath:       str   = "keno_draws.csv",
    num_to_select:  int   = 7,      # top numbers to output per draw
    future_draws:   int   = 3,      # how many future draws to predict
    window_size:    int   = 100,    # past draws used as context
    epochs:         int   = 35,     # max training epochs
    batch_size:     int   = 32,     # ⚠ lower to 16/8 if window_size > 150
    pool_size:      int   = 80      # keno number pool: 1–80
) -> dict:
    """
    Full end-to-end pipeline:
        load → encode → sequences → model → train → predict

    Args:
        filepath       : Path to your CSV file.
        num_to_select  : Top N numbers to display per predicted draw.
        future_draws   : Number of future draws to forecast.
        window_size    : Sliding-window length (past draws).
        epochs         : Training epochs (EarlyStopping may reduce).
        batch_size     : Training batch size.
                         ⚠ IMPORTANT: If you increase window_size,
                           DECREASE batch_size proportionally to avoid
                           CUDA_ERROR_INVALID_HANDLE or
                           "Dst tensor is not initialized" errors on
                           Colab GPU. E.g.:
                             window_size=150  → batch_size=16
                             window_size=200  → batch_size=8
        pool_size      : Total number pool (80 for Keno).

    Returns:
        result : dict with keys:
                   'model'       → trained Keras model
                   'history'     → training history
                   'predictions' → list of predicted draw lists
                   'seed_window' → the seed window used
    """
    print("=" * 60)
    print("   KENO LSTM PREDICTOR — PIPELINE START")
    print("=" * 60)

    # ── Step 1: Load & encode ────────────────────────────────
    encoded = load_and_preprocess(filepath, pool_size=pool_size)

    # ── Step 2: Build sequences ──────────────────────────────
    X, y = build_sequences(encoded, window_size=window_size)

    # ── Step 3: Build model ──────────────────────────────────
    model = build_model(window_size=window_size, pool_size=pool_size)

    # ── Step 4: Train ────────────────────────────────────────
    history = train_model(
        model,
        X, y,
        epochs     = epochs,
        batch_size = batch_size
    )

    # ── Step 5: Predict future draws (autoregressive) ────────
    #    Seed = the very last `window_size` real draws
    seed_window  = encoded[-window_size:]          # shape: (window_size, 80)
    predictions  = predict_future_draws(
        model,
        seed_window,
        future_draws     = future_draws,
        num_to_select    = num_to_select,
        pool_size        = pool_size,
        numbers_per_draw = 20
    )

    # ── Step 6: Pretty-print results ─────────────────────────
    print("\n" + "=" * 60)
    print("   📊 FINAL PREDICTIONS")
    print("=" * 60)
    for i, draw in enumerate(predictions, 1):
        stars = "  ".join([f"[ {n:02d} ]" for n in draw])
        print(f"  Future Draw T+{i:02d}:  {stars}")
    print("=" * 60 + "\n")

    return {
        "model":       model,
        "history":     history,
        "predictions": predictions,
        "seed_window": seed_window
    }


# ════════════════════════════════════════════════════════════
#  SECTION 7 — RUN  (edit parameters here)
# ════════════════════════════════════════════════════════════

# ⚠ MEMORY REMINDER:
#   If you raise window_size, LOWER batch_size to match:
#     window_size=100  → batch_size=32   (default, safe)
#     window_size=150  → batch_size=16
#     window_size=200  → batch_size=8
#   This prevents CUDA_ERROR_INVALID_HANDLE and
#   "Dst tensor is not initialized" errors on Colab GPUs.

results = run_pipeline(
    filepath      = "/content/Keno_data_year_april17_sorted.csv",   # ← change to your file path
    num_to_select = 7,                  # top numbers to show per draw
    future_draws  = 10,                  # how many draws ahead to predict
    window_size   = 100,                # sliding window size
    epochs        = 12,                 # max epochs (EarlyStopping active)
    batch_size    = 32,                 # ⚠ lower if window_size > 150
    pool_size     = 80                  # keno pool: 1–80
)

# ── Access results programmatically if needed ────────────────
trained_model = results["model"]
train_history = results["history"]
my_predictions= results["predictions"]   # list of lists

# Example: save the model
# trained_model.save("keno_lstm_model.keras")
