"""
keno_optimizer_gpu.py
=====================
GPU-accelerated Keno optimizer for NVIDIA RTX 2060.

Three levels of GPU acceleration:
  1. LSTM training/inference → TensorFlow GPU or PyTorch CUDA
  2. Matrix operations → CuPy (GPU NumPy)
  3. Batch scoring → vectorized GPU kernels

Install:
    pip install tensorflow==2.10.1    # Windows native GPU
    pip install cupy-cuda12x           # GPU numpy (optional)
    pip install pdfplumber pandas numpy

Usage:
    python keno_optimizer_gpu.py your_data.csv 8
"""
import sys
sys.modules['cupy'] = None

import numpy as np
import pandas as pd
import sys
import os
import re
import time
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from math import comb
from itertools import combinations

# ─── GPU Setup ───────────────────────────────────────────

# Try CuPy (GPU NumPy replacement)
try:
    import cupy as cp
    HAS_CUPY = True
    xp = cp          # Use GPU arrays
    print("✅ CuPy detected — NumPy ops will use GPU")
except ImportError:
    HAS_CUPY = False
    xp = np           # Fallback to CPU NumPy

# Try TensorFlow GPU
HAS_TF_GPU = False
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        HAS_TF_GPU = True
        print(f"✅ TensorFlow GPU: {gpus[0].name}")
    else:
        print("⚠️  TensorFlow found but no GPU detected")
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("❌ TensorFlow not installed")

# Try PyTorch GPU (alternative to TF)
HAS_TORCH_GPU = False
try:
    import torch
    import torch.nn as nn
    if torch.cuda.is_available():
        TORCH_DEVICE = torch.device('cuda')
        HAS_TORCH_GPU = True
        print(f"✅ PyTorch CUDA: {torch.cuda.get_device_name(0)}")
    else:
        TORCH_DEVICE = torch.device('cpu')
        print("⚠️  PyTorch found but no CUDA")
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    TORCH_DEVICE = None


def get_gpu_status() -> str:
    """Return summary of GPU capabilities."""
    parts = []
    if HAS_TF_GPU:
        parts.append("TF-GPU")
    if HAS_TORCH_GPU:
        parts.append("PyTorch-CUDA")
    if HAS_CUPY:
        parts.append("CuPy")
    
    if parts:
        return f"GPU Active: {', '.join(parts)}"
    else:
        return "CPU Only (install tensorflow==2.10.1 or torch for GPU)"


# ═══════════════════════════════════════════════════════════
#  DATA LOADER (same as before, CPU is fine here)
# ═══════════════════════════════════════════════════════════

class KenoDataLoader:
    """Handles CSV with winningNumbers column."""
    
    @staticmethod
    def load_csv(path: str) -> np.ndarray:
        df = pd.read_csv(path)
        print(f"📄 CSV: {path} ({len(df)} rows)")
        
        history = []
        
        # Find the winning numbers column
        for col in df.columns:
            sample = str(df[col].iloc[0])
            if ',' in sample and any(c.isdigit() for c in sample):
                nums_test = [int(x) for x in sample.split(',') if x.strip().isdigit()]
                if len(nums_test) >= 15 and all(1 <= n <= 80 for n in nums_test):
                    print(f"   Found numbers in column: '{col}'")
                    for _, row in df.iterrows():
                        raw = str(row[col]).strip()
                        nums = [int(x.strip()) for x in raw.split(',') 
                                if x.strip().isdigit() and 1 <= int(x.strip()) <= 80]
                        if len(nums) == 20:
                            history.append(sorted(nums))
                    break
        
        if not history:
            # Try multi-column format
            ncols = [c for c in df.columns if re.match(r'^n\d+$', c)]
            if ncols:
                for _, row in df.iterrows():
                    history.append(sorted([int(row[c]) for c in ncols]))
        
        if not history:
            raise ValueError(f"Cannot parse CSV. Columns: {list(df.columns)}")
        
        # Check order (reverse if newest-first)
        draw_col = [c for c in df.columns if 'draw' in c.lower() and 'date' not in c.lower()]
        if draw_col and len(history) == len(df):
            if df[draw_col[0]].iloc[0] > df[draw_col[0]].iloc[-1]:
                history = history[::-1]
                print("   ↕ Reversed to chronological order")
        
        valid = [h for h in history if len(h) == 20 and len(set(h)) == 20]
        print(f"   ✅ {len(valid)} valid draws")
        
        return np.array(valid)


# ═══════════════════════════════════════════════════════════
#  GPU-ACCELERATED SCORER
# ═══════════════════════════════════════════════════════════

class GPUKenoScorer:
    """
    Scores numbers 1-80 using GPU where possible.
    
    GPU is used for:
    - Multi-hot matrix operations (CuPy)
    - LSTM training and inference (TF/PyTorch)
    - Batch Markov transition computation (CuPy)
    
    CPU is used for:
    - Simple loop-based models (faster on CPU for small data)
    """
    
    def __init__(self, history: np.ndarray):
        self.history = history  # always numpy on CPU
        self.n_draws = len(history)
        
        # ── Build multi-hot matrix on GPU ──
        # Shape: (n_draws, 80) — 1 if number appeared, 0 otherwise
        t0 = time.time()
        
        mh_cpu = np.zeros((self.n_draws, 80), dtype=np.float32)
        for i, draw in enumerate(history):
            for num in draw:
                mh_cpu[i, num - 1] = 1.0
        
        if HAS_CUPY:
            self.mh_gpu = cp.asarray(mh_cpu)  # on GPU
            self.mh_cpu = mh_cpu
            print(f"   📦 Multi-hot matrix on GPU: {self.mh_gpu.shape} ({self.mh_gpu.nbytes/1e6:.1f} MB VRAM)")
        else:
            self.mh_gpu = None
            self.mh_cpu = mh_cpu
        
        # Pre-compute per-number appearance vectors
        # Shape: (80, n_draws) — row i = whether number i+1 appeared in each draw
        if HAS_CUPY:
            self.app_matrix_gpu = self.mh_gpu.T  # (80, n_draws) on GPU
        
        self.app_matrix_cpu = mh_cpu.T  # (80, n_draws) on CPU
        
        print(f"   Cache built in {time.time()-t0:.2f}s")
    
    def score_all(self) -> Dict[str, Dict[int, float]]:
        """Run all models."""
        print("   Running scoring models...")
        scores = {}
        
        t0 = time.time()
        scores['short_freq']  = self._short_term_frequency_gpu()
        print(f"     short_freq:  {time.time()-t0:.2f}s")
        
        t0 = time.time()
        scores['momentum']    = self._momentum_gpu()
        print(f"     momentum:    {time.time()-t0:.2f}s")
        
        t0 = time.time()
        scores['markov']      = self._markov_gpu()
        print(f"     markov:      {time.time()-t0:.2f}s")
        
        t0 = time.time()
        scores['gap']         = self._gap_analysis()
        print(f"     gap:         {time.time()-t0:.2f}s")
        
        t0 = time.time()
        scores['pair']        = self._pair_following_gpu()
        print(f"     pair:        {time.time()-t0:.2f}s")
        
        t0 = time.time()
        scores['anti_streak'] = self._anti_streak_gpu()
        print(f"     anti_streak: {time.time()-t0:.2f}s")
        
        t0 = time.time()
        lstm_scores = self._lstm_gpu()
        if lstm_scores:
            scores['lstm'] = lstm_scores
            print(f"     lstm:        {time.time()-t0:.2f}s [GPU]")
        
        return {k: v for k, v in scores.items() if v}
    
    # ── GPU-Accelerated Models ──
    
    def _short_term_frequency_gpu(self) -> Dict[int, float]:
        """Vectorized short-term frequency using GPU matrix ops."""
        if HAS_CUPY:
            app = self.app_matrix_gpu  # (80, n_draws) on GPU
            n = self.n_draws
            
            # Compute windowed sums in one shot
            w5  = cp.sum(app[:, -5:],  axis=1) / 5   if n >= 5  else cp.full(80, 0.25)
            w10 = cp.sum(app[:, -10:], axis=1) / 10  if n >= 10 else cp.full(80, 0.25)
            w20 = cp.sum(app[:, -20:], axis=1) / 20  if n >= 20 else cp.full(80, 0.25)
            w50 = cp.sum(app[:, -50:], axis=1) / 50  if n >= 50 else cp.full(80, 0.25)
            
            combined = 0.40 * w5 + 0.30 * w10 + 0.20 * w20 + 0.10 * w50
            result = cp.asnumpy(combined)  # back to CPU
        else:
            app = self.app_matrix_cpu  # (80, n_draws)
            n = self.n_draws
            
            w5  = np.sum(app[:, -5:],  axis=1) / 5   if n >= 5  else np.full(80, 0.25)
            w10 = np.sum(app[:, -10:], axis=1) / 10  if n >= 10 else np.full(80, 0.25)
            w20 = np.sum(app[:, -20:], axis=1) / 20  if n >= 20 else np.full(80, 0.25)
            w50 = np.sum(app[:, -50:], axis=1) / 50  if n >= 50 else np.full(80, 0.25)
            
            result = 0.40 * w5 + 0.30 * w10 + 0.20 * w20 + 0.10 * w50
        
        return self._normalize_array(result)
    
    def _momentum_gpu(self) -> Dict[int, float]:
        """GPU-vectorized momentum calculation."""
        if HAS_CUPY:
            app = self.app_matrix_gpu
            n = self.n_draws
            
            signals = cp.zeros(80, dtype=cp.float32)
            count = 0
            
            if n >= 10:
                new5 = cp.sum(app[:, -5:], axis=1) / 5
                old5 = cp.sum(app[:, -10:-5], axis=1) / 5
                signals += 0.5 * (new5 - old5)
                count += 1
            
            if n >= 20:
                new10 = cp.sum(app[:, -10:], axis=1) / 10
                old10 = cp.sum(app[:, -20:-10], axis=1) / 10
                signals += 0.3 * (new10 - old10)
                count += 1
            
            if n >= 50:
                new25 = cp.sum(app[:, -25:], axis=1) / 25
                old25 = cp.sum(app[:, -50:-25], axis=1) / 25
                signals += 0.2 * (new25 - old25)
                count += 1
            
            result = cp.asnumpy(signals)
        else:
            app = self.app_matrix_cpu
            n = self.n_draws
            
            signals = np.zeros(80)
            
            if n >= 10:
                signals += 0.5 * (np.sum(app[:, -5:], axis=1)/5 - np.sum(app[:, -10:-5], axis=1)/5)
            if n >= 20:
                signals += 0.3 * (np.sum(app[:, -10:], axis=1)/10 - np.sum(app[:, -20:-10], axis=1)/10)
            if n >= 50:
                signals += 0.2 * (np.sum(app[:, -25:], axis=1)/25 - np.sum(app[:, -50:-25], axis=1)/25)
            
            result = signals
        
        return self._normalize_array(result)
    
    def _markov_gpu(self) -> Dict[int, float]:
        """
        GPU-accelerated Markov transitions using matrix multiplication.
        
        Transition matrix T[i,j] = P(j appears in draw t+1 | i appeared in draw t)
        Then score = T^T @ last_draw_vector (normalized)
        """
        recent = min(500, self.n_draws)
        start = self.n_draws - recent
        
        if HAS_CUPY:
            mh = self.mh_gpu[start:]  # (recent, 80)
            
            # Build transition counts via matrix mult:
            # T = prev_draws^T @ next_draws
            # Where prev = mh[:-1] and next = mh[1:]
            prev = mh[:-1]  # (recent-1, 80)
            nxt  = mh[1:]   # (recent-1, 80)
            
            # T[i,j] = sum over draws where i appeared, j appeared next
            T = prev.T @ nxt  # (80, 80) — GPU matmul!
            
            # Normalize rows: T[i,j] /= sum(prev[:,i])
            row_sums = cp.sum(prev, axis=0).reshape(80, 1) + 1e-8
            T = T / row_sums
            
            # Score: multiply by last draw vector
            last_vec = self.mh_gpu[-1]  # (80,)
            scores_gpu = T.T @ last_vec  # (80,)
            
            result = cp.asnumpy(scores_gpu)
        else:
            mh = self.mh_cpu[start:]
            
            prev = mh[:-1]
            nxt = mh[1:]
            
            T = prev.T @ nxt  # (80, 80)
            row_sums = np.sum(prev, axis=0).reshape(80, 1) + 1e-8
            T = T / row_sums
            
            last_vec = self.mh_cpu[-1]
            result = T.T @ last_vec
        
        return self._normalize_array(result)
    
    def _pair_following_gpu(self) -> Dict[int, float]:
        """GPU-accelerated pair co-occurrence following."""
        recent = min(300, self.n_draws)
        start = self.n_draws - recent
        
        if HAS_CUPY:
            mh = self.mh_gpu[start:]
            
            # For each draw where numbers co-occurred with last draw's pairs,
            # what followed? Approximate with:
            # similarity = mh[:-1] @ last_vec → how similar each past draw is to last
            # weighted_next = similarity^T @ mh[1:] → weighted sum of following draws
            
            last_vec = self.mh_gpu[-1]  # (80,)
            
            past = mh[:-1]  # (recent-1, 80)
            future = mh[1:]  # (recent-1, 80)
            
            # Similarity of each past draw to last draw
            similarity = past @ last_vec  # (recent-1,) — dot products on GPU
            
            # Weight future draws by similarity
            similarity_norm = similarity / (cp.sum(similarity) + 1e-8)
            weighted = similarity_norm.reshape(-1, 1) * future  # broadcast
            scores_gpu = cp.sum(weighted, axis=0)  # (80,)
            
            result = cp.asnumpy(scores_gpu)
        else:
            mh = self.mh_cpu[start:]
            last_vec = self.mh_cpu[-1]
            
            past = mh[:-1]
            future = mh[1:]
            
            similarity = past @ last_vec
            similarity_norm = similarity / (np.sum(similarity) + 1e-8)
            weighted = similarity_norm.reshape(-1, 1) * future
            result = np.sum(weighted, axis=0)
        
        return self._normalize_array(result)
    
    def _anti_streak_gpu(self) -> Dict[int, float]:
        """GPU-vectorized streak detection."""
        if HAS_CUPY:
            app = self.app_matrix_gpu  # (80, n_draws)
            n = self.n_draws
            
            # Last state for each number
            last_state = app[:, -1]  # (80,) — 1 if appeared in last draw
            
            # Count streak length — need to iterate (hard to fully vectorize)
            # But we can do it in chunks on GPU
            streak_lens = cp.ones(80, dtype=cp.float32)
            
            for k in range(1, min(n, 30)):  # max 30 draws back
                same_state = (app[:, -(k+1)] == last_state).astype(cp.float32)
                # Only extend streak if all previous were same
                still_going = cp.prod(
                    cp.stack([app[:, -(j+1)] == last_state for j in range(k)]), 
                    axis=0
                ) if k <= 15 else same_state  # limit stack depth
                streak_lens += still_going
            
            # Score: absent streaks → overdue (higher), present streaks → slight boost
            scores = cp.where(
                last_state > 0.5,
                0.5 + 0.1 * cp.minimum(streak_lens, 5),   # present streak
                1 - cp.exp(-0.3 * streak_lens)              # absent streak
            )
            
            result = cp.asnumpy(scores)
        else:
            # CPU fallback
            result = np.zeros(80)
            for num_idx in range(80):
                app = self.app_matrix_cpu[num_idx]
                streak = 0
                state = app[-1] > 0.5
                for k in range(self.n_draws - 1, -1, -1):
                    if (app[k] > 0.5) == state:
                        streak += 1
                    else:
                        break
                
                if not state:
                    result[num_idx] = 1 - np.exp(-0.3 * streak)
                else:
                    result[num_idx] = 0.5 + 0.1 * min(streak, 5)
        
        return self._normalize_array(result)
    
    def _gap_analysis(self) -> Dict[int, float]:
        """Gap analysis (CPU — sequential by nature)."""
        scores = np.zeros(80)
        
        for num_idx in range(80):
            app = self.app_matrix_cpu[num_idx]
            recent_start = max(0, self.n_draws - 200)
            recent = app[recent_start:]
            indices = np.where(recent > 0.5)[0]
            
            if len(indices) < 4:
                scores[num_idx] = 0.5
                continue
            
            gaps = np.diff(indices).astype(float)
            recent_gaps = gaps[-6:] if len(gaps) >= 6 else gaps
            avg = np.mean(recent_gaps)
            std = np.std(recent_gaps) + 0.5
            current = len(recent) - 1 - indices[-1]
            
            z = (current - avg) / std
            scores[num_idx] = 1 / (1 + np.exp(-1.5 * z))
        
        return self._normalize_array(scores)
    
    def _lstm_gpu(self) -> Dict[int, float]:
        """
        LSTM on GPU — tries PyTorch first (better GPU util on RTX),
        falls back to TensorFlow.
        """
        if HAS_TORCH_GPU:
            return self._lstm_pytorch()
        elif HAS_TF_GPU:
            return self._lstm_tensorflow()
        elif HAS_TF:
            return self._lstm_tensorflow()  # CPU TF as last resort
        return {}
    
    def _lstm_pytorch(self) -> Dict[int, float]:
        """PyTorch LSTM on GPU."""
        print("     [PyTorch CUDA]", end=" ")
        
        seq_len = 15
        recent_n = min(3000, self.n_draws)
        mh = self.mh_cpu[-recent_n:]  # numpy
        
        if len(mh) < seq_len + 20:
            return {}
        
        # Build sequences
        X_np = np.array([mh[i-seq_len:i] for i in range(seq_len, len(mh))])
        y_np = np.array([mh[i] for i in range(seq_len, len(mh))])
        
        # To GPU tensors
        X = torch.FloatTensor(X_np).to(TORCH_DEVICE)
        y = torch.FloatTensor(y_np).to(TORCH_DEVICE)
        
        # Split train/val
        split = int(len(X) * 0.85)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Define model
        class KenoLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm1 = nn.LSTM(80, 128, batch_first=True, 
                                     bidirectional=True, num_layers=2, 
                                     dropout=0.3)
                self.dropout = nn.Dropout(0.3)
                self.fc1 = nn.Linear(256, 128)  # 128*2 for bidirectional
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(128, 80)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                lstm_out, _ = self.lstm1(x)
                last = lstm_out[:, -1, :]  # last timestep
                x = self.dropout(last)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                return self.sigmoid(self.fc2(x))
        
        model = KenoLSTM().to(TORCH_DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Training loop
        batch_size = 128  # RTX 2060 can handle this easily
        best_val_loss = float('inf')
        patience = 8
        patience_counter = 0
        best_state = None
        
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                              shuffle=True, drop_last=True)
        
        for epoch in range(60):
            model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Load best weights
        if best_state:
            model.load_state_dict(best_state)
        
        # Predict
        model.eval()
        with torch.no_grad():
            last_seq = torch.FloatTensor(mh[-seq_len:]).unsqueeze(0).to(TORCH_DEVICE)
            pred = model(last_seq).cpu().numpy()[0]
        
        return self._normalize_array(pred)
    
    def _lstm_tensorflow(self) -> Dict[int, float]:
        """TensorFlow LSTM (GPU if available)."""
        print("     [TensorFlow" + (" GPU" if HAS_TF_GPU else " CPU") + "]", end=" ")
        
        seq_len = 15
        recent_n = min(3000, self.n_draws)
        mh = self.mh_cpu[-recent_n:]
        
        if len(mh) < seq_len + 20:
            return {}
        
        X = np.array([mh[i-seq_len:i] for i in range(seq_len, len(mh))])
        y = np.array([mh[i] for i in range(seq_len, len(mh))])
        
        # Force GPU placement
        if HAS_TF_GPU:
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        else:
            strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        
        with strategy.scope():
            model = tf.keras.Sequential([
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(128, return_sequences=True),
                    input_shape=(seq_len, 80)
                ),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(80, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy')
        
        model.fit(X, y, epochs=50, batch_size=128,
                  validation_split=0.15,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      patience=8, restore_best_weights=True
                  )],
                  verbose=0)
        
        last_seq = mh[-seq_len:].reshape(1, seq_len, 80)
        pred = model.predict(last_seq, verbose=0)[0]
        
        return self._normalize_array(pred)
    
    # ── Utilities ──
    
    @staticmethod
    def _normalize_array(arr) -> Dict[int, float]:
        """Normalize array of 80 scores to {1-80: score}."""
        if isinstance(arr, dict):
            vals = np.array(list(arr.values()))
        else:
            vals = np.asarray(arr).flatten()[:80]
        
        if len(vals) != 80:
            return {n: 0.5 for n in range(1, 81)}
        
        mean = vals.mean()
        std = vals.std()
        
        if std < 1e-10:
            return {n: 0.5 for n in range(1, 81)}
        
        z = (vals - mean) / std
        sigmoid = 1 / (1 + np.exp(-z))
        
        return {i + 1: float(sigmoid[i]) for i in range(80)}


# ═══════════════════════════════════════════════════════════
#  PICK SELECTOR (CPU is fine, tiny computation)
# ═══════════════════════════════════════════════════════════

class PickSelector:
    """Select picks with diversity enforcement."""
    
    PAYOUTS = {
        5: {0:0,1:0,2:1,3:2,4:18,5:420},
        6: {0:1,1:0,2:0,3:1,4:7,5:50,6:1600},
        7: {0:0,1:0,2:0,3:1,4:3,5:17,6:100,7:7000},
        8: {0:0,1:0,2:0,3:0,4:2,5:10,6:50,7:500,8:15000},
    }
    
    WEIGHTS = {
        'short_freq':  0.15,
        'momentum':    0.18,
        'markov':      0.20,
        'gap':         0.15,
        'pair':        0.10,
        'anti_streak': 0.10,
        'lstm':        0.12,
    }
    
    def __init__(self, scores: Dict[str, Dict[int, float]]):
        self.scores = scores
    
    def select(self, pick_count: int) -> Dict:
        # Ensemble
        final = defaultdict(float)
        total_w = defaultdict(float)
        
        for model, mscores in self.scores.items():
            w = self.WEIGHTS.get(model, 0.05)
            for num, s in mscores.items():
                final[num] += w * s
                total_w[num] += w
        
        ensemble = {n: final[n] / (total_w[n] + 1e-8) for n in final}
        
        # Model consensus bonus
        vote = defaultdict(int)
        for model, mscores in self.scores.items():
            top = sorted(mscores.items(), key=lambda x: x[1], reverse=True)
            for n, _ in top[:pick_count * 2]:
                vote[n] += 1
        n_models = len(self.scores)
        for n in ensemble:
            ensemble[n] = 0.7 * ensemble[n] + 0.3 * (vote.get(n, 0) / n_models)
        
        ranked = sorted(ensemble.items(), key=lambda x: x[1], reverse=True)
        
        # Diverse selection
        picks = []
        for num, score in ranked:
            if len(picks) >= pick_count:
                break
            if not self._bad_run(picks, num):
                picks.append(num)
        
        while len(picks) < pick_count:
            for num, _ in ranked:
                if num not in picks:
                    picks.append(num)
                    break
        
        max_s = ranked[0][1] if ranked else 1
        conf = {n: round(ensemble[n] / max_s * 100, 1) for n in picks}
        
        model_picks = {}
        for m, ms in self.scores.items():
            top = sorted(ms.items(), key=lambda x: x[1], reverse=True)
            model_picks[m] = [n for n, _ in top[:pick_count]]
        
        return {
            'picks': sorted(picks),
            'confidence': conf,
            'model_picks': model_picks,
            'ranked': ranked,
        }
    
    @staticmethod
    def _bad_run(picks, num, max_consecutive=2):
        test = sorted(picks + [num])
        run = 1
        for i in range(1, len(test)):
            if test[i] == test[i-1] + 1:
                run += 1
                if run > max_consecutive:
                    return True
            else:
                run = 1
        return False


# ═══════════════════════════════════════════════════════════
#  MAIN AGENT
# ═══════════════════════════════════════════════════════════

class KenoGPUAgent:
    """
    GPU-accelerated Keno optimizer.
    
    Usage:
        agent = KenoGPUAgent()
        result = agent.run("data.csv", pick_count=8)
        bt = agent.backtest("data.csv", pick_count=8)
    """
    
    def run(self, data_source: str, pick_count: int = 8, 
            future_draws: int = 10) -> Dict:
        
        print("\n" + "═" * 65)
        print("  🎰  KENO GPU OPTIMIZER  🎰")
        print("  " + get_gpu_status())
        print("═" * 65)
        
        history = KenoDataLoader.load_csv(data_source)
        print(f"\n  {len(history)} draws | Pick {pick_count} | "
              f"Predict {future_draws} draws")
        print("─" * 65)
        
        # Score
        print("\n🔬 GPU Scoring...")
        t_total = time.time()
        scorer = GPUKenoScorer(history)
        scores = scorer.score_all()
        scoring_time = time.time() - t_total
        print(f"\n   ⏱️  Total scoring: {scoring_time:.2f}s")
        
        # Select
        selector = PickSelector(scores)
        result = selector.select(pick_count)
        
        # Per-draw predictions
        draw_preds = []
        ranked = result['ranked']
        pool = ranked[:pick_count * 3]
        
        for di in range(future_draws):
            adj = {}
            for num, base in pool:
                gap_s = scores.get('gap', {}).get(num, 0.5)
                mom_s = scores.get('momentum', {}).get(num, 0.5)
                adj[num] = base * 0.5 + mom_s * max(0.1, 0.5 - 0.04*di) + gap_s * min(0.5, 0.2 + 0.03*di)
            
            top = sorted(adj.items(), key=lambda x: x[1], reverse=True)
            dp = []
            for n, _ in top:
                if len(dp) >= pick_count:
                    break
                if not PickSelector._bad_run(dp, n):
                    dp.append(n)
            draw_preds.append(sorted(dp))
        
        result['draw_predictions'] = draw_preds
        result['scoring_time'] = scoring_time
        result['gpu_status'] = get_gpu_status()
        
        # Print
        self._print(result, pick_count, scores)
        
        return result
    
    def backtest(self, data_source: str, pick_count: int = 8,
                 n_tests: int = 300) -> Dict:
        """GPU-accelerated backtesting."""
        history = KenoDataLoader.load_csv(data_source)
        n = len(history)
        test_start = max(100, n - n_tests)
        actual = n - test_start
        
        payouts = PickSelector.PAYOUTS.get(pick_count, {})
        hits_list = []
        payout_list = []
        
        print(f"\n🧪 GPU BACKTEST: {actual} draws, Pick {pick_count}")
        t0 = time.time()
        
        for ti in range(test_start, n):
            train = history[:ti]
            actual_draw = set(history[ti])
            
            scorer = GPUKenoScorer.__new__(GPUKenoScorer)
            scorer.history = train
            scorer.n_draws = len(train)
            
            mh = np.zeros((len(train), 80), dtype=np.float32)
            for i, draw in enumerate(train):
                for num in draw:
                    mh[i, num - 1] = 1.0
            
            scorer.mh_cpu = mh
            scorer.app_matrix_cpu = mh.T
            if HAS_CUPY:
                scorer.mh_gpu = cp.asarray(mh)
                scorer.app_matrix_gpu = scorer.mh_gpu.T
            
            # Fast models only
            fast_scores = {
                'short_freq': scorer._short_term_frequency_gpu(),
                'momentum': scorer._momentum_gpu(),
                'markov': scorer._markov_gpu(),
                'gap': scorer._gap_analysis(),
                'anti_streak': scorer._anti_streak_gpu(),
            }
            
            sel = PickSelector(fast_scores)
            picks = set(sel.select(pick_count)['picks'])
            
            hits = len(picks & actual_draw)
            hits_list.append(hits)
            payout_list.append(payouts.get(hits, 0))
            
            if (ti - test_start + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (ti - test_start + 1) / elapsed
                print(f"   [{ti-test_start+1}/{actual}] "
                      f"avg: {np.mean(hits_list):.2f} hits | "
                      f"{rate:.0f} draws/sec")
        
        total_time = time.time() - t0
        hits_arr = np.array(hits_list)
        payout_arr = np.array(payout_list)
        random_exp = pick_count * 20 / 80
        
        bt = {
            'tests': actual,
            'avg_hits': round(float(np.mean(hits_arr)), 3),
            'random': round(random_exp, 3),
            'improvement': round(float(np.mean(hits_arr)) / random_exp, 3),
            'distribution': dict(Counter(hits_arr.astype(int).tolist())),
            'avg_payout': round(float(np.mean(payout_arr)), 3),
            'total_payout': round(float(np.sum(payout_arr)), 2),
            'net': round(float(np.sum(payout_arr)) - actual, 2),
            'roi': round((float(np.mean(payout_arr)) - 1) * 100, 1),
            'time': round(total_time, 1),
            'draws_per_sec': round(actual / total_time, 1),
        }
        
        print(f"\n  {'═'*55}")
        print(f"  BACKTEST RESULTS — Pick {pick_count}")
        print(f"  {'═'*55}")
        print(f"  Avg hits/draw:    {bt['avg_hits']:.2f}  (random: {bt['random']:.2f})")
        print(f"  Improvement:      {bt['improvement']:.3f}x")
        print(f"  Distribution:     {bt['distribution']}")
        print(f"  Avg payout:       ${bt['avg_payout']:.2f} per $1 bet")
        print(f"  Net ($1/draw):    ${bt['net']:.2f}")
        print(f"  ROI:              {bt['roi']}%")
        print(f"  Speed:            {bt['draws_per_sec']:.0f} draws/sec")
        print(f"  Total time:       {bt['time']:.1f}s")
        print(f"  {'═'*55}")
        
        return bt
    
    def _print(self, result, pick_count, scores):
        picks = result['picks']
        conf = result['confidence']
        
        print("\n" + "═" * 65)
        print(f"  🎯  YOUR {pick_count} PICKS")
        print("═" * 65)
        
        nums_str = "  ".join(f"[{n:2d}]" for n in picks)
        print(f"\n  ▶  {nums_str}  ◀")
        
        print(f"\n  Confidence:")
        for n in picks:
            c = conf.get(n, 0)
            bar = "█" * int(c/5) + "░" * (20 - int(c/5))
            print(f"    {n:3d}  [{bar}] {c:.1f}%")
        
        print(f"\n  📅 Next {len(result['draw_predictions'])} Draws:")
        for i, dp in enumerate(result['draw_predictions'], 1):
            print(f"    Draw {i:2d}: {dp}")
        
        print(f"\n  🤖 Model Picks:")
        for m, mp in result['model_picks'].items():
            print(f"    {m:15s} → {mp}")
        
        print(f"\n  ⏱️  Scoring time: {result['scoring_time']:.2f}s")
        print(f"  🖥️  {result['gpu_status']}")
        print("═" * 65)


# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else "keno_data.csv"
    pc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    agent = KenoGPUAgent()
    result = agent.run(data, pick_count=pc)
    
    print("\n" + "─" * 65)
    bt = agent.backtest(data, pick_count=pc, n_tests=300)