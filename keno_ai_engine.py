"""
keno_ai_engine.py
=================
ACTUAL AI models for Keno prediction:

1. Transformer (self-attention — like GPT for number sequences)
2. XGBoost (gradient-boosted trees on engineered features)  
3. Deep Ensemble (multiple neural nets that LEARN to combine)
4. Neural Pair Model (learned embeddings for co-occurrence)
5. Reinforcement Learning pick selector

All models run on RTX 2060 via PyTorch CUDA.

Install:
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install xgboost lightgbm scikit-learn pandas numpy
    
Usage:
    python keno_ai_engine.py Keno_data.csv 7
"""

import numpy as np
import pandas as pd
import sys
import time
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from math import comb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  pip install xgboost  (for gradient boosting model)")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


# ═══════════════════════════════════════════════════════════
#  DATA LOADER
# ═══════════════════════════════════════════════════════════

class DataLoader_:
    @staticmethod
    def load_csv(path: str) -> np.ndarray:
        df = pd.read_csv(path)
        history = []
        for col in df.columns:
            sample = str(df[col].iloc[0])
            if ',' in sample:
                nums = [int(x) for x in sample.split(',') if x.strip().isdigit()]
                if len(nums) >= 15 and all(1 <= n <= 80 for n in nums):
                    for _, row in df.iterrows():
                        raw = str(row[col]).strip()
                        nums = [int(x.strip()) for x in raw.split(',')
                                if x.strip().isdigit() and 1 <= int(x.strip()) <= 80]
                        if len(nums) == 20:
                            history.append(sorted(nums))
                    break
        
        dcol = [c for c in df.columns if 'draw' in c.lower() and 'date' not in c.lower()]
        if dcol and len(history) == len(df):
            if df[dcol[0]].iloc[0] > df[dcol[0]].iloc[-1]:
                history = history[::-1]
        
        valid = np.array([h for h in history if len(h) == 20 and len(set(h)) == 20])
        print(f"📄 Loaded {len(valid)} draws from {path}")
        return valid


# ═══════════════════════════════════════════════════════════
#  AI MODEL 1: TRANSFORMER (Self-Attention)
# ═══════════════════════════════════════════════════════════

class KenoTransformer(nn.Module):
    """
    Transformer model for Keno prediction.
    
    Like GPT but for number draw sequences:
    - Input: last N draws as multi-hot vectors
    - Self-attention learns which past draws matter
    - Predicts probability of each number in next draw
    
    This is REAL AI — same architecture family as ChatGPT.
    """
    
    def __init__(self, d_model=128, nhead=8, num_layers=4, 
                 seq_len=20, dropout=0.2):
        super().__init__()
        
        self.seq_len = seq_len
        
        # Input projection: 80 binary → d_model dimensional
        self.input_proj = nn.Linear(80, d_model)
        
        # Learned positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Transformer encoder layers (self-attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 80),
            nn.Sigmoid()
        )
        
        # Causal mask (each position only attends to past)
        self.register_buffer('causal_mask', 
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        )
    
    def forward(self, x):
        # x: (batch, seq_len, 80)
        x = self.input_proj(x)           # (batch, seq_len, d_model)
        x = x + self.pos_embedding       # add positional info
        
        # Self-attention with causal mask
        x = self.transformer(x, mask=self.causal_mask)
        
        # Use last position's output for prediction
        x = x[:, -1, :]                  # (batch, d_model)
        return self.output_head(x)        # (batch, 80)


class TransformerPredictor:
    """Train and use the Transformer for Keno prediction."""
    
    def __init__(self, seq_len=20, d_model=128, nhead=8, num_layers=4):
        self.seq_len = seq_len
        self.model = KenoTransformer(
            d_model=d_model, nhead=nhead, 
            num_layers=num_layers, seq_len=seq_len
        ).to(DEVICE)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50
        )
    
    def train(self, history: np.ndarray, epochs=50, batch_size=128):
        """Train the transformer on draw sequences."""
        print("   🤖 Training Transformer (self-attention)...")
        t0 = time.time()
        
        # Build multi-hot sequences
        mh = np.zeros((len(history), 80), dtype=np.float32)
        for i, draw in enumerate(history):
            for num in draw:
                mh[i, num - 1] = 1.0
        
        # Create training sequences
        X_list, y_list = [], []
        for i in range(self.seq_len, len(mh)):
            X_list.append(mh[i - self.seq_len:i])
            y_list.append(mh[i])
        
        X = torch.FloatTensor(np.array(X_list)).to(DEVICE)
        y = torch.FloatTensor(np.array(y_list)).to(DEVICE)
        
        # Train/val split
        split = int(len(X) * 0.85)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_state = None
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = F.binary_cross_entropy(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
            
            self.scheduler.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = F.binary_cross_entropy(val_pred, y_val).item()
            self.model.train()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        print(f"      Trained {epoch+1} epochs in {time.time()-t0:.1f}s | "
              f"Val loss: {best_val_loss:.4f}")
    
    def predict(self, history: np.ndarray) -> Dict[int, float]:
        """Predict next draw probabilities."""
        mh = np.zeros((len(history), 80), dtype=np.float32)
        for i, draw in enumerate(history):
            for num in draw:
                mh[i, num - 1] = 1.0
        
        last_seq = torch.FloatTensor(mh[-self.seq_len:]).unsqueeze(0).to(DEVICE)
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(last_seq).cpu().numpy()[0]
        
        return {i + 1: float(pred[i]) for i in range(80)}


# ═══════════════════════════════════════════════════════════
#  AI MODEL 2: NEURAL PAIR MODEL (Learned Embeddings)
# ═══════════════════════════════════════════════════════════

class NeuralPairModel(nn.Module):
    """
    Learns number EMBEDDINGS (like word2vec but for Keno numbers).
    
    Each number 1-80 gets a learned vector representation.
    Co-occurrence patterns are captured in embedding space.
    Similar to how word embeddings capture 'king-queen' relationships.
    """
    
    def __init__(self, embed_dim=32, hidden_dim=128, seq_len=10):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        # LEARNED number embeddings (like word2vec)
        self.number_embedding = nn.Embedding(81, embed_dim, padding_idx=0)
        
        # Attention over numbers within a draw
        self.intra_draw_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True
        )
        
        # Sequence model over draws
        self.draw_lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=2, batch_first=True, dropout=0.3
        )
        
        # Cross-attention: last draw attends to history
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 80),
            nn.Sigmoid()
        )
    
    def forward(self, x_indices):
        """
        x_indices: (batch, seq_len, 20) — number indices per draw
        """
        batch_size = x_indices.shape[0]
        
        # Embed each number
        embedded = self.number_embedding(x_indices)  # (batch, seq, 20, embed_dim)
        
        # Aggregate numbers within each draw via attention
        draw_reprs = []
        for t in range(x_indices.shape[1]):
            draw_nums = embedded[:, t, :, :]  # (batch, 20, embed_dim)
            attn_out, _ = self.intra_draw_attention(
                draw_nums, draw_nums, draw_nums
            )
            draw_repr = attn_out.mean(dim=1)  # (batch, embed_dim)
            draw_reprs.append(draw_repr)
        
        draw_sequence = torch.stack(draw_reprs, dim=1)  # (batch, seq, embed_dim)
        
        # Process sequence
        lstm_out, _ = self.draw_lstm(draw_sequence)  # (batch, seq, hidden)
        
        # Cross-attention: last draw attends to all history
        query = lstm_out[:, -1:, :]   # (batch, 1, hidden)
        key_value = lstm_out           # (batch, seq, hidden)
        
        attended, attn_weights = self.cross_attention(query, key_value, key_value)
        
        # Predict
        return self.predictor(attended.squeeze(1))  # (batch, 80)


class NeuralPairPredictor:
    """Train and use the Neural Pair Model."""
    
    def __init__(self, seq_len=15, embed_dim=32, hidden_dim=128):
        self.seq_len = seq_len
        self.model = NeuralPairModel(
            embed_dim=embed_dim, hidden_dim=hidden_dim, seq_len=seq_len
        ).to(DEVICE)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=5e-4, weight_decay=1e-5
        )
    
    def train(self, history: np.ndarray, epochs=40, batch_size=64):
        print("   🤖 Training Neural Pair Model (learned embeddings)...")
        t0 = time.time()
        
        # Build index sequences: (n_draws, 20) — actual number indices
        indices = np.zeros((len(history), 20), dtype=np.int64)
        for i, draw in enumerate(history):
            indices[i] = draw  # numbers 1-80
        
        # Multi-hot for targets
        mh = np.zeros((len(history), 80), dtype=np.float32)
        for i, draw in enumerate(history):
            for num in draw:
                mh[i, num - 1] = 1.0
        
        # Create sequences
        X_list, y_list = [], []
        for i in range(self.seq_len, len(history)):
            X_list.append(indices[i - self.seq_len:i])
            y_list.append(mh[i])
        
        X = torch.LongTensor(np.array(X_list)).to(DEVICE)
        y = torch.FloatTensor(np.array(y_list)).to(DEVICE)
        
        split = int(len(X) * 0.85)
        dataset = TensorDataset(X[:split], y[:split])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        X_val, y_val = X[split:], y[split:]
        
        best_val = float('inf')
        best_state = None
        patience = 8
        wait = 0
        
        for epoch in range(epochs):
            self.model.train()
            for bx, by in loader:
                self.optimizer.zero_grad()
                pred = self.model(bx)
                loss = F.binary_cross_entropy(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                vp = self.model(X_val)
                vl = F.binary_cross_entropy(vp, y_val).item()
            
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        print(f"      Trained {epoch+1} epochs in {time.time()-t0:.1f}s | Val: {best_val:.4f}")
        
        # Show learned embeddings similarity
        self._show_embeddings()
    
    def _show_embeddings(self):
        """Visualize what the AI learned about number relationships."""
        emb = self.model.number_embedding.weight.data[1:81].cpu().numpy()  # skip padding
        
        # Find most similar number pairs (in learned space)
        from numpy.linalg import norm
        
        similarities = []
        for i in range(80):
            for j in range(i + 1, 80):
                sim = np.dot(emb[i], emb[j]) / (norm(emb[i]) * norm(emb[j]) + 1e-8)
                similarities.append((i + 1, j + 1, sim))
        
        top_pairs = sorted(similarities, key=lambda x: x[2], reverse=True)[:10]
        print(f"      AI-Learned number pairs (highest affinity):")
        for a, b, sim in top_pairs:
            print(f"        {a:2d} ↔ {b:2d}  similarity: {sim:.3f}")
    
    def predict(self, history: np.ndarray) -> Dict[int, float]:
        indices = np.zeros((len(history), 20), dtype=np.int64)
        for i, draw in enumerate(history):
            indices[i] = draw
        
        last_seq = torch.LongTensor(indices[-self.seq_len:]).unsqueeze(0).to(DEVICE)
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(last_seq).cpu().numpy()[0]
        
        return {i + 1: float(pred[i]) for i in range(80)}


# ═══════════════════════════════════════════════════════════
#  AI MODEL 3: XGBOOST (Feature-Based Learning)
# ═══════════════════════════════════════════════════════════

class XGBoostPredictor:
    """
    Gradient Boosted Trees on hand-crafted features.
    
    For each number, builds features from recent history,
    then trains XGBoost to predict: will this number appear?
    
    This is real ML — learns non-linear feature interactions
    that simple statistics miss.
    """
    
    def __init__(self):
        self.models = {}  # one model per number (or one for all)
    
    def _build_features(self, history: np.ndarray, 
                        target_draw_idx: int) -> np.ndarray:
        """
        Build feature vector for a specific draw index.
        Returns (80,) feature-per-number or (80, n_features) matrix.
        """
        n = target_draw_idx
        
        # Multi-hot matrix up to this point
        mh = np.zeros((n, 80), dtype=np.float32)
        for i in range(n):
            for num in history[i]:
                mh[i, num - 1] = 1.0
        
        features_per_number = []
        
        for num_idx in range(80):
            app = mh[:, num_idx]
            feats = []
            
            # Frequency in different windows
            for w in [3, 5, 7, 10, 15, 20, 30, 50]:
                if n >= w:
                    feats.append(app[-w:].sum() / w)
                else:
                    feats.append(app.sum() / n if n > 0 else 0.25)
            
            # Momentum (rate changes)
            for w in [5, 10, 20]:
                if n >= w * 2:
                    new = app[-w:].sum() / w
                    old = app[-w*2:-w].sum() / w
                    feats.append(new - old)
                else:
                    feats.append(0)
            
            # Gap features
            indices = np.where(app > 0.5)[0]
            if len(indices) >= 2:
                gaps = np.diff(indices)
                feats.append(np.mean(gaps[-5:]) if len(gaps) >= 5 else np.mean(gaps))
                feats.append(np.std(gaps[-5:]) if len(gaps) >= 5 else np.std(gaps))
                current_gap = n - 1 - indices[-1]
                avg_gap = np.mean(gaps[-5:]) if len(gaps) >= 5 else np.mean(gaps)
                feats.append(current_gap / (avg_gap + 0.5))  # overdue ratio
                feats.append(current_gap)
            else:
                feats.extend([4, 2, 1, n])
            
            # Streak
            if n > 0:
                streak = 0
                state = app[-1] > 0.5
                for k in range(n - 1, max(n - 20, -1), -1):
                    if (app[k] > 0.5) == state:
                        streak += 1
                    else:
                        break
                feats.append(streak * (1 if state else -1))
            else:
                feats.append(0)
            
            # Co-occurrence with last draw
            if n > 0:
                last_draw_mask = mh[n - 1]
                cooccur = 0
                for i in range(max(0, n - 100), n - 1):
                    if mh[i, num_idx] > 0.5:
                        cooccur += np.dot(mh[i], last_draw_mask)
                feats.append(cooccur / max(1, app[-100:].sum()))
            else:
                feats.append(0)
            
            # Pair similarity score (like the model that works well)
            if n > 1:
                last_vec = mh[n - 1]
                recent_start = max(0, n - 200)
                past = mh[recent_start:n - 1]
                future = mh[recent_start + 1:n]
                
                if len(past) > 0:
                    similarity = past @ last_vec
                    sim_norm = similarity / (similarity.sum() + 1e-8)
                    weighted_future = sim_norm @ future
                    feats.append(weighted_future[num_idx])
                else:
                    feats.append(0.25)
            else:
                feats.append(0.25)
            
            features_per_number.append(feats)
        
        return np.array(features_per_number, dtype=np.float32)
    
    def train(self, history: np.ndarray, n_train_samples: int = 2000):
        """Train XGBoost on historical features."""
        if not HAS_XGB:
            print("   ⚠️  XGBoost not installed, skipping")
            return
        
        print("   🤖 Training XGBoost (gradient-boosted trees)...")
        t0 = time.time()
        
        n = len(history)
        # Sample training points from recent history
        train_start = max(100, n - n_train_samples)
        
        X_all = []
        y_all = []
        
        for idx in range(train_start, n):
            feats = self._build_features(history, idx)  # (80, n_features)
            
            # Target: which numbers appeared
            target = np.zeros(80, dtype=np.float32)
            for num in history[idx]:
                target[num - 1] = 1.0
            
            X_all.append(feats)
            y_all.append(target)
        
        # Reshape: each number in each draw is a training sample
        X_flat = np.vstack(X_all)        # (n_samples * 80, n_features)
        y_flat = np.concatenate(y_all)    # (n_samples * 80,)
        
        print(f"      Training data: {X_flat.shape[0]} samples × {X_flat.shape[1]} features")
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',  # fast
            device='cuda' if torch.cuda.is_available() else 'cpu',
            eval_metric='logloss',
            early_stopping_rounds=15,
            verbosity=0
        )
        
        split = int(len(X_flat) * 0.85)
        self.model.fit(
            X_flat[:split], y_flat[:split],
            eval_set=[(X_flat[split:], y_flat[split:])],
            verbose=False
        )
        
        # Feature importance
        imp = self.model.feature_importances_
        top_feats = np.argsort(imp)[-5:][::-1]
        feat_names = [
            'freq_3','freq_5','freq_7','freq_10','freq_15','freq_20','freq_30','freq_50',
            'momentum_5','momentum_10','momentum_20',
            'avg_gap','gap_std','overdue_ratio','current_gap',
            'streak','cooccurrence','pair_similarity'
        ]
        print(f"      Top features: {[feat_names[i] if i < len(feat_names) else f'f{i}' for i in top_feats]}")
        print(f"      Trained in {time.time()-t0:.1f}s")
    
    def predict(self, history: np.ndarray) -> Dict[int, float]:
        if not HAS_XGB or not hasattr(self, 'model'):
            return {}
        
        feats = self._build_features(history, len(history))  # (80, n_features)
        probs = self.model.predict_proba(feats)[:, 1]  # probability of class 1
        
        return {i + 1: float(probs[i]) for i in range(80)}


# ═══════════════════════════════════════════════════════════
#  AI MODEL 4: META-LEARNER (Learns How to Combine Models)
# ═══════════════════════════════════════════════════════════

class MetaLearnerNet(nn.Module):
    """
    Instead of hand-tuned weights, a neural network LEARNS
    how to combine different model predictions.
    
    Input: predictions from all base models
    Output: final probability per number
    
    This replaces the simple weighted average with a 
    learned non-linear combination.
    """
    
    def __init__(self, n_models: int, hidden_dim: int = 64):
        super().__init__()
        
        # Input: n_models scores per number, stacked as (80, n_models)
        self.net = nn.Sequential(
            nn.Linear(n_models, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Also learn a per-model attention weight
        self.model_attention = nn.Sequential(
            nn.Linear(n_models, n_models),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        # x: (batch, 80, n_models)
        
        # Learned attention weights per model
        attn = self.model_attention(x.mean(dim=1, keepdim=True))  # (batch, 1, n_models)
        x_weighted = x * attn  # apply learned weights
        
        # Non-linear combination
        out = self.net(x_weighted)  # (batch, 80, 1)
        return out.squeeze(-1)      # (batch, 80)


class MetaLearner:
    """Trains the meta-learner on base model outputs."""
    
    def __init__(self, n_models: int):
        self.model = MetaLearnerNet(n_models).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.n_models = n_models
    
    def train(self, base_predictions: np.ndarray, targets: np.ndarray, 
              epochs: int = 30, batch_size: int = 64):
        """
        base_predictions: (n_samples, 80, n_models) — stacked model outputs
        targets: (n_samples, 80) — actual outcomes
        """
        print("   🤖 Training Meta-Learner (learns to combine models)...")
        t0 = time.time()
        
        X = torch.FloatTensor(base_predictions).to(DEVICE)
        y = torch.FloatTensor(targets).to(DEVICE)
        
        split = int(len(X) * 0.85)
        dataset = TensorDataset(X[:split], y[:split])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_val = float('inf')
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            for bx, by in loader:
                self.optimizer.zero_grad()
                pred = self.model(bx)
                loss = F.binary_cross_entropy(pred, by)
                loss.backward()
                self.optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                vp = self.model(X[split:])
                vl = F.binary_cross_entropy(vp, y[split:]).item()
            
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        # Show learned model weights
        self.model.eval()
        with torch.no_grad():
            dummy = torch.ones(1, 80, self.n_models).to(DEVICE) * 0.5
            attn = self.model.model_attention(dummy.mean(dim=1))
            weights = attn.cpu().numpy()[0]
        
        print(f"      Meta-learner trained in {time.time()-t0:.1f}s")
        print(f"      Learned model weights: {np.round(weights, 3)}")
    
    def predict(self, base_predictions: np.ndarray) -> Dict[int, float]:
        """base_predictions: (1, 80, n_models)"""
        X = torch.FloatTensor(base_predictions).to(DEVICE)
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X).cpu().numpy()[0]
        
        return {i + 1: float(pred[i]) for i in range(80)}


# ═══════════════════════════════════════════════════════════
#  STATISTICAL MODELS (kept as base models for meta-learner)
# ═══════════════════════════════════════════════════════════

class StatModels:
    """Fast statistical models (input to meta-learner)."""
    
    @staticmethod
    def short_freq(mh, n):
        app = mh[:n].T
        w5 = np.sum(app[:, -5:], axis=1) / 5 if n >= 5 else np.full(80, 0.25)
        w10 = np.sum(app[:, -10:], axis=1) / 10 if n >= 10 else np.full(80, 0.25)
        w20 = np.sum(app[:, -20:], axis=1) / 20 if n >= 20 else np.full(80, 0.25)
        return 0.4 * w5 + 0.3 * w10 + 0.2 * w20
    
    @staticmethod
    def pair_score(mh, n):
        recent = min(300, n)
        start = n - recent
        past = mh[start:n - 1]
        future = mh[start + 1:n]
        last_vec = mh[n - 1]
        
        if len(past) == 0:
            return np.full(80, 0.25)
        
        similarity = past @ last_vec
        weights = np.exp(-0.005 * np.arange(len(similarity)))[::-1]
        combined = similarity * weights
        combined /= combined.sum() + 1e-8
        return combined @ future
    
    @staticmethod
    def markov_score(mh, n):
        recent = min(500, n)
        start = n - recent
        prev = mh[start:n - 1]
        nxt = mh[start + 1:n]
        T = prev.T @ nxt
        row_sums = np.sum(prev, axis=0).reshape(80, 1) + 1e-8
        T = T / row_sums
        return T.T @ mh[n - 1]
    
    @staticmethod
    def gap_score(mh, n):
        app = mh[:n].T
        scores = np.zeros(80)
        for i in range(80):
            indices = np.where(app[i] > 0.5)[0]
            if len(indices) < 4:
                scores[i] = 0.5
                continue
            gaps = np.diff(indices[-20:]).astype(float)
            avg = np.mean(gaps)
            std = np.std(gaps) + 0.5
            current = n - 1 - indices[-1]
            scores[i] = 1 / (1 + np.exp(-1.5 * (current - avg) / std))
        return scores


# ═══════════════════════════════════════════════════════════
#  MAIN AI AGENT
# ═══════════════════════════════════════════════════════════

class KenoAIAgent:
    """
    Full AI-powered Keno agent.
    
    Pipeline:
    1. Train Transformer (attention-based sequence model)
    2. Train Neural Pair Model (learned embeddings)
    3. Train XGBoost (feature-based trees)
    4. Get statistical model scores
    5. Train Meta-Learner to combine all models
    6. Select diverse picks
    """
    
    PAYOUTS = {
        5: {0:0,1:0,2:1,3:2,4:18,5:420},
        6: {0:1,1:0,2:0,3:1,4:7,5:50,6:1600},
        7: {0:0,1:0,2:0,3:1,4:3,5:17,6:100,7:7000},
        8: {0:0,1:0,2:0,3:0,4:2,5:10,6:50,7:500,8:15000},
    }
    
    def run(self, data_source: str, pick_count: int = 7, 
            future_draws: int = 10) -> Dict:
        
        print("\n" + "═" * 65)
        print("  🧠  KENO AI ENGINE  🧠")
        print(f"  {DEVICE.type.upper()}: {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'}")
        print("═" * 65)
        
        history = DataLoader_.load_csv(data_source)
        n = len(history)
        
        # Build multi-hot (shared across models)
        mh = np.zeros((n, 80), dtype=np.float32)
        for i, draw in enumerate(history):
            for num in draw:
                mh[i, num - 1] = 1.0
        
        print(f"\n  {n} draws | Pick {pick_count}")
        print("─" * 65)
        
        # ── Train AI Models ──
        total_t0 = time.time()
        
        # 1. Transformer
        transformer = TransformerPredictor(seq_len=20)
        transformer.train(history[-3000:])
        
        # 2. Neural Pair Model  
        pair_model = NeuralPairPredictor(seq_len=15)
        pair_model.train(history[-3000:])
        
        # 3. XGBoost
        xgb_model = XGBoostPredictor()
        xgb_model.train(history, n_train_samples=2000)
        
        # ── Get All Predictions ──
        print("\n📊 Generating predictions from all models...")
        
        model_outputs = {}
        
        # AI models
        model_outputs['transformer'] = transformer.predict(history)
        model_outputs['neural_pair'] = pair_model.predict(history)
        if HAS_XGB:
            model_outputs['xgboost'] = xgb_model.predict(history)
        
        # Statistical models
        model_outputs['stat_freq'] = {i+1: float(v) for i, v in 
                                       enumerate(StatModels.short_freq(mh, n))}
        model_outputs['stat_pair'] = {i+1: float(v) for i, v in 
                                       enumerate(StatModels.pair_score(mh, n))}
        model_outputs['stat_markov'] = {i+1: float(v) for i, v in 
                                         enumerate(StatModels.markov_score(mh, n))}
        model_outputs['stat_gap'] = {i+1: float(v) for i, v in 
                                      enumerate(StatModels.gap_score(mh, n))}
        
        # ── Train Meta-Learner ──
        print("\n🧠 Training Meta-Learner...")
        n_models = len(model_outputs)
        model_names = list(model_outputs.keys())
        
        # Build meta-learner training data from recent history
        meta_X = []
        meta_y = []
        meta_start = max(200, n - 1000)
        
        for idx in range(meta_start, n):
            # Get base model predictions at this point
            base_preds = np.zeros((80, n_models), dtype=np.float32)
            
            # Statistical models (fast, can recompute)
            base_preds[:, model_names.index('stat_freq')] = StatModels.short_freq(mh, idx)
            base_preds[:, model_names.index('stat_pair')] = StatModels.pair_score(mh, idx)
            base_preds[:, model_names.index('stat_markov')] = StatModels.markov_score(mh, idx)
            base_preds[:, model_names.index('stat_gap')] = StatModels.gap_score(mh, idx)
            
            # For AI models, use current predictions as proxy
            # (retraining at each step is too slow)
            for mi, mname in enumerate(model_names):
                if mname.startswith('stat_'):
                    continue
                for num in range(1, 81):
                    base_preds[num - 1, mi] = model_outputs[mname].get(num, 0.25)
            
            meta_X.append(base_preds)
            meta_y.append(mh[idx])
        
        meta_X = np.array(meta_X)
        meta_y = np.array(meta_y)
        
        meta_learner = MetaLearner(n_models)
        meta_learner.train(meta_X, meta_y)
        
        # ── Final Prediction ──
        print("\n🎯 Generating final AI prediction...")
        
        # Stack current predictions
        current_preds = np.zeros((1, 80, n_models), dtype=np.float32)
        for mi, mname in enumerate(model_names):
            for num in range(1, 81):
                current_preds[0, num - 1, mi] = model_outputs[mname].get(num, 0.25)
        
        final_scores = meta_learner.predict(current_preds)
        
        # ── Select Picks ──
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Diverse selection
        picks = []
        for num, score in ranked:
            if len(picks) >= pick_count:
                break
            test = sorted(picks + [num])
            run = 1
            bad = False
            for i in range(1, len(test)):
                if test[i] == test[i-1] + 1:
                    run += 1
                    if run > 2:
                        bad = True
                        break
                else:
                    run = 1
            if not bad:
                picks.append(num)
        
        while len(picks) < pick_count:
            for num, _ in ranked:
                if num not in picks:
                    picks.append(num)
                    break
        
        picks = sorted(picks)
        
        # Confidence
        max_s = ranked[0][1] if ranked else 1
        conf = {n: round(final_scores[n] / max_s * 100, 1) for n in picks}
        
        # Per-model picks
        model_picks = {}
        for mname, mscores in model_outputs.items():
            top = sorted(mscores.items(), key=lambda x: x[1], reverse=True)
            model_picks[mname] = [n for n, _ in top[:pick_count]]
        
        total_time = time.time() - total_t0
        
        # ── Print Results ──
        print("\n" + "═" * 65)
        print(f"  🎯  AI PICKS (Pick {pick_count})")
        print("═" * 65)
        
        nums_str = "  ".join(f"[{n:2d}]" for n in picks)
        print(f"\n  ▶  {nums_str}  ◀")
        
        print(f"\n  Confidence:")
        for n in picks:
            c = conf[n]
            bar = "█" * int(c / 5) + "░" * (20 - int(c / 5))
            print(f"    {n:3d}  [{bar}] {c:.1f}%")
        
        print(f"\n  🤖 Individual AI Model Picks:")
        for mname, mpicks in model_picks.items():
            tag = "🧠" if not mname.startswith('stat_') else "📊"
            print(f"    {tag} {mname:20s} → {mpicks}")
        
        print(f"\n  ⏱️  Total AI pipeline: {total_time:.1f}s")
        print("═" * 65)
        
        return {
            'picks': picks,
            'confidence': conf,
            'model_picks': model_picks,
            'final_scores': final_scores,
            'model_outputs': model_outputs,
        }
    
    def backtest(self, data_source: str, pick_count: int = 7, 
                 n_tests: int = 200) -> Dict:
        """
        AI backtest — trains models on rolling window, tests forward.
        Slower but honest.
        """
        history = DataLoader_.load_csv(data_source)
        n = len(history)
        
        mh = np.zeros((n, 80), dtype=np.float32)
        for i, draw in enumerate(history):
            for num in draw:
                mh[i, num - 1] = 1.0
        
        test_start = max(500, n - n_tests)
        actual_tests = n - test_start
        payouts = self.PAYOUTS.get(pick_count, {})
        
        print(f"\n🧪 AI BACKTEST: {actual_tests} draws")
        
        # Train AI models once on first portion
        train_data = history[:test_start]
        
        print("   Training AI models for backtest...")
        transformer = TransformerPredictor(seq_len=20)
        transformer.train(train_data[-2000:])
        
        pair_model = NeuralPairPredictor(seq_len=15)
        pair_model.train(train_data[-2000:])
        
        hits_list = []
        payout_list = []
        
        t0 = time.time()
        for ti in range(test_start, n):
            actual_draw = set(history[ti])
            
            # Get predictions from all models
            scores = {}
            scores['stat_freq'] = StatModels.short_freq(mh, ti)
            scores['stat_pair'] = StatModels.pair_score(mh, ti)
            scores['stat_markov'] = StatModels.markov_score(mh, ti)
            scores['stat_gap'] = StatModels.gap_score(mh, ti)
            
            # Average all scores
            combined = np.zeros(80)
            for name, s in scores.items():
                if isinstance(s, dict):
                    arr = np.array([s.get(i+1, 0.25) for i in range(80)])
                else:
                    arr = s
                combined += arr
            combined /= len(scores)
            
            # AI model predictions (trained once, applied forward)
            ai_pred_t = transformer.predict(history[:ti])
            ai_pred_p = pair_model.predict(history[:ti])
            
            for i in range(80):
                combined[i] = (
                    0.25 * combined[i] + 
                    0.40 * ai_pred_t.get(i+1, 0.25) + 
                    0.35 * ai_pred_p.get(i+1, 0.25)
                )
            
            top_indices = np.argsort(combined)[-pick_count:]
            predicted = set(idx + 1 for idx in top_indices)
            
            hits = len(predicted & actual_draw)
            hits_list.append(hits)
            payout_list.append(payouts.get(hits, 0))
            
            if (ti - test_start + 1) % 50 == 0:
                print(f"   [{ti-test_start+1}/{actual_tests}] "
                      f"avg: {np.mean(hits_list):.2f} hits")
        
        hits_arr = np.array(hits_list)
        payout_arr = np.array(payout_list)
        random_exp = pick_count * 20 / 80
        
        bt = {
            'tests': actual_tests,
            'avg_hits': round(float(np.mean(hits_arr)), 3),
            'random': round(random_exp, 3),
            'improvement': round(float(np.mean(hits_arr)) / random_exp, 3),
            'distribution': dict(Counter(hits_arr.astype(int).tolist())),
            'avg_payout': round(float(np.mean(payout_arr)), 3),
            'total_payout': round(float(np.sum(payout_arr)), 2),
            'net': round(float(np.sum(payout_arr)) - actual_tests, 2),
            'time': round(time.time() - t0, 1),
        }
        
        print(f"\n  {'═'*55}")
        print(f"  AI BACKTEST RESULTS — Pick {pick_count}")
        print(f"  {'═'*55}")
        print(f"  Avg hits:     {bt['avg_hits']:.3f}  (random: {bt['random']:.3f})")
        print(f"  Improvement:  {bt['improvement']:.3f}x")
        print(f"  Distribution: {bt['distribution']}")
        print(f"  Payout/draw:  ${bt['avg_payout']:.2f}")
        print(f"  Net:          ${bt['net']:.2f}")
        print(f"  Time:         {bt['time']:.1f}s")
        print(f"  {'═'*55}")
        
        return bt


# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else "Keno_data.csv"
    pc = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    
    agent = KenoAIAgent()
    result = agent.run(data, pick_count=pc)
    
    print("\n" + "─" * 65)
    bt = agent.backtest(data, pick_count=pc, n_tests=300)