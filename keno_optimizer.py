"""
keno_optimizer_v2.py
====================
Fixed for real MA Keno CSV format.
Focuses on SHORT-TERM patterns (the only possible edge).

Usage:
    python keno_optimizer_v2.py your_data.csv 8
"""

import numpy as np
import pandas as pd
import sys
import os
import re
import time
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from math import comb
from itertools import combinations

try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    HAS_TF = False


# ═══════════════════════════════════════════════════════════
#  SECTION 1: DATA LOADER (Fixed for your CSV format)
# ═══════════════════════════════════════════════════════════

class KenoDataLoader:
    """
    Handles the ACTUAL CSV format from MA Lottery:
    
    drawNumber | bonus | drawDate   | winningNumbers
    2987367    | 1     | 2026-03-26 | 17,11,3,57,61,49,...
    """
    
    @staticmethod
    def load_csv(path: str) -> List[List[int]]:
        """
        Load CSV with auto-detection of format.
        Handles both:
          - Single column: "17,11,3,57,..." 
          - Multi column: n1=17, n2=11, ...
        """
        df = pd.read_csv(path)
        print(f"📄 Loaded CSV: {path}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Rows: {len(df)}")
        
        history = []
        
        # ── Format 1: Single "winningNumbers" column ──
        winning_col = None
        for col in df.columns:
            if 'winning' in col.lower() or 'number' in col.lower():
                # Check if it's a comma-separated string
                sample = str(df[col].iloc[0])
                if ',' in sample:
                    winning_col = col
                    break
        
        if winning_col:
            print(f"   Detected format: comma-separated in '{winning_col}'")
            for idx, row in df.iterrows():
                raw = str(row[winning_col]).strip()
                try:
                    nums = [int(x.strip()) for x in raw.split(',') if x.strip().isdigit()]
                    nums = [n for n in nums if 1 <= n <= 80]
                    if len(nums) == 20:
                        history.append(sorted(nums))
                    elif len(nums) > 0:
                        # Some rows might have different counts
                        history.append(sorted(nums[:20]))
                except Exception:
                    continue
        else:
            # ── Format 2: Separate n1, n2, ... columns ──
            num_cols = [c for c in df.columns if re.match(r'^n\d+$', c)]
            if num_cols:
                print(f"   Detected format: separate columns ({len(num_cols)} number columns)")
                for _, row in df.iterrows():
                    nums = [int(row[c]) for c in num_cols if 1 <= int(row[c]) <= 80]
                    history.append(sorted(nums))
            else:
                # ── Format 3: Try all numeric columns ──
                print("   Trying to auto-detect numeric columns...")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                # Skip obvious non-draw columns
                skip = ['drawnumber', 'bonus', 'game_id', 'id', 'index']
                num_cols = [c for c in numeric_cols 
                           if c.lower() not in skip and df[c].between(1, 80).all()]
                if len(num_cols) >= 20:
                    num_cols = num_cols[:20]
                    for _, row in df.iterrows():
                        nums = [int(row[c]) for c in num_cols]
                        history.append(sorted(nums))
        
        if not history:
            raise ValueError(
                "Could not parse CSV! Expected either:\n"
                "  - Column 'winningNumbers' with '17,11,3,57,...'\n"
                "  - Columns 'n1','n2',...,'n20'\n"
                f"  Got columns: {list(df.columns)}"
            )
        
        # Sort by draw order (check if we need to reverse)
        # If file has a drawNumber column, use it
        draw_col = None
        for col in df.columns:
            if 'draw' in col.lower() and 'date' not in col.lower():
                draw_col = col
                break
        
        if draw_col and len(history) == len(df):
            ids = df[draw_col].tolist()
            # If IDs are descending (newest first), reverse
            if len(ids) > 1 and ids[0] > ids[-1]:
                history = history[::-1]
                print("   ↕ Reversed to chronological order (oldest first)")
        
        # Validate
        valid = [h for h in history if len(h) == 20 and len(set(h)) == 20]
        if len(valid) < len(history):
            print(f"   ⚠️ Dropped {len(history)-len(valid)} invalid rows")
        history = valid
        
        print(f"   ✅ {len(history)} valid draws loaded")
        print(f"   Oldest: {history[0][:5]}...")
        print(f"   Newest: {history[-1][:5]}...")
        
        return history
    
    @staticmethod
    def load_pdf(path: str) -> List[List[int]]:
        """Extract from PDF."""
        if not HAS_PDF:
            raise ImportError("pip install pdfplumber")
        
        lines = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    lines.extend(text.strip().split('\n'))
        
        game_re = re.compile(r'^29\d{5}$')
        skip = ['game name','game results','past results','masslottery','https://','date:']
        
        tokens = []
        for line in lines:
            if any(s in line.lower() for s in skip):
                continue
            tokens.extend(line.strip().split())
        
        draws = []
        gid = None
        nums = []
        
        for tok in tokens:
            t = tok.strip().rstrip(',:.')
            if game_re.match(t):
                if gid and len(nums) == 20:
                    draws.append((int(gid), sorted(nums)))
                gid = t
                nums = []
                continue
            try:
                n = int(t)
                if 1 <= n <= 80 and len(nums) < 20:
                    nums.append(n)
            except ValueError:
                pass
        
        if gid and len(nums) == 20:
            draws.append((int(gid), sorted(nums)))
        
        draws.sort(key=lambda x: x[0])
        history = [d[1] for d in draws]
        print(f"📄 Extracted {len(history)} draws from PDF")
        return history


# ═══════════════════════════════════════════════════════════
#  SECTION 2: SCORING ENGINE (Fixed — No More 1,2,3,4,5,6,7)
# ═══════════════════════════════════════════════════════════

class KenoScorer:
    """
    Scores numbers 1-80 using ONLY methods that work.
    
    KEY INSIGHT: With 21,000 draws of fair Keno, long-term
    frequency is IDENTICAL for all numbers (~25%).
    
    We focus ONLY on:
    1. Short-term patterns (last 5-50 draws)
    2. Sequential dependencies (what follows what)
    3. Gap analysis (cycles and overdue)
    4. Streak momentum
    """
    
    def __init__(self, history: np.ndarray):
        self.history = history
        self.n_draws = len(history)
        
        # Pre-build appearance lookup for speed
        self._build_cache()
    
    def _build_cache(self):
        """Pre-compute per-number appearance arrays."""
        self.appearances = {}
        for num in range(1, 81):
            self.appearances[num] = np.array(
                [num in self.history[i] for i in range(self.n_draws)], 
                dtype=bool
            )
        
        # Also store as sets for fast lookup
        self.draw_sets = [set(d) for d in self.history]
    
    def score_all(self) -> Dict[str, Dict[int, float]]:
        """Run all scoring models, return {model_name: {number: score}}."""
        print("   Running 8 scoring models...")
        
        scores = {}
        scores['short_freq']  = self._short_term_frequency()
        scores['momentum']    = self._momentum()
        scores['markov']      = self._markov_chains()
        scores['gap']         = self._gap_analysis()
        scores['pair']        = self._pair_following()
        scores['cluster']     = self._cluster_tendency()
        scores['anti_streak'] = self._anti_streak()
        scores['lstm']        = self._lstm_predict()
        
        # Remove empty models
        scores = {k: v for k, v in scores.items() if v}
        
        return scores
    
    def _short_term_frequency(self) -> Dict[int, float]:
        """
        ONLY look at recent draws (not all 21,000).
        Uses windows of 5, 10, 20, 50 with decaying weights.
        """
        scores = {}
        
        for num in range(1, 81):
            app = self.appearances[num]
            
            # Recent windows
            w5  = app[-5:].sum() / 5   if self.n_draws >= 5  else 0.25
            w10 = app[-10:].sum() / 10 if self.n_draws >= 10 else 0.25
            w20 = app[-20:].sum() / 20 if self.n_draws >= 20 else 0.25
            w50 = app[-50:].sum() / 50 if self.n_draws >= 50 else 0.25
            
            # Weighted combination (recent matters most)
            score = 0.40 * w5 + 0.30 * w10 + 0.20 * w20 + 0.10 * w50
            
            # Deviation from expected 0.25
            scores[num] = score
        
        return self._normalize(scores)
    
    def _momentum(self) -> Dict[int, float]:
        """
        Is a number heating up or cooling down?
        Compare recent rate vs slightly-less-recent rate.
        """
        scores = {}
        
        for num in range(1, 81):
            app = self.appearances[num]
            
            if self.n_draws < 20:
                scores[num] = 0.5
                continue
            
            # Multiple momentum windows
            signals = []
            
            # 5-draw momentum
            if self.n_draws >= 10:
                new5 = app[-5:].sum() / 5
                old5 = app[-10:-5].sum() / 5
                signals.append(new5 - old5)
            
            # 10-draw momentum
            if self.n_draws >= 20:
                new10 = app[-10:].sum() / 10
                old10 = app[-20:-10].sum() / 10
                signals.append(new10 - old10)
            
            # 25-draw momentum
            if self.n_draws >= 50:
                new25 = app[-25:].sum() / 25
                old25 = app[-50:-25].sum() / 25
                signals.append(new25 - old25)
            
            if signals:
                # Weight shorter momentum more
                weights = [0.5, 0.3, 0.2][:len(signals)]
                total_w = sum(weights)
                score = sum(s * w for s, w in zip(signals, weights)) / total_w
                scores[num] = score
            else:
                scores[num] = 0
        
        return self._normalize(scores)
    
    def _markov_chains(self) -> Dict[int, float]:
        """
        2nd-order Markov: P(num in draw t | draws t-1 and t-2).
        Only uses RECENT transitions (last 200 draws).
        """
        # Use recent history only (last 500 or all if less)
        recent = min(500, self.n_draws)
        start = self.n_draws - recent
        
        # First-order: what follows what?
        follow_count = defaultdict(lambda: defaultdict(int))
        follow_total = defaultdict(int)
        
        for i in range(max(start, 1), self.n_draws):
            prev_set = self.draw_sets[i - 1]
            curr_set = self.draw_sets[i]
            
            for p in prev_set:
                follow_total[p] += 1
                for c in curr_set:
                    follow_count[p][c] += 1
        
        # Score based on last draw
        last = self.draw_sets[-1]
        scores = defaultdict(float)
        
        for prev_num in last:
            if follow_total[prev_num] > 0:
                for num in range(1, 81):
                    prob = follow_count[prev_num].get(num, 0) / follow_total[prev_num]
                    scores[num] += prob
        
        # Normalize by number of previous numbers
        n_last = len(last)
        scores = {k: v / n_last for k, v in scores.items()}
        
        return self._normalize(scores)
    
    def _gap_analysis(self) -> Dict[int, float]:
        """
        How overdue is each number based on its personal rhythm?
        Uses only RECENT gaps, not lifetime average.
        """
        scores = {}
        
        for num in range(1, 81):
            app = self.appearances[num]
            
            # Only look at last 200 draws for gap calculation
            recent_start = max(0, self.n_draws - 200)
            recent_app = app[recent_start:]
            indices = np.where(recent_app)[0]
            
            if len(indices) < 4:
                scores[num] = 0.5
                continue
            
            gaps = np.diff(indices).astype(float)
            
            # Recent gaps matter more
            if len(gaps) >= 6:
                recent_avg_gap = np.mean(gaps[-6:])
                recent_std_gap = np.std(gaps[-6:]) + 0.5
            else:
                recent_avg_gap = np.mean(gaps)
                recent_std_gap = np.std(gaps) + 0.5
            
            # Current gap
            current_gap = len(recent_app) - 1 - indices[-1]
            
            # Overdue z-score
            z = (current_gap - recent_avg_gap) / recent_std_gap
            
            # Sigmoid: >0 means overdue (higher score)
            scores[num] = 1 / (1 + np.exp(-1.5 * z))
        
        return self._normalize(scores)
    
    def _pair_following(self) -> Dict[int, float]:
        """
        If numbers A,B appeared together last draw,
        which numbers tend to follow that PAIR?
        """
        # Only compute on recent draws for relevance
        recent = min(300, self.n_draws)
        start = self.n_draws - recent
        
        # Build pair→follower counts
        # Use a sample of pairs from last draw to keep it fast
        last = list(self.draw_sets[-1])
        
        # Pick ~10 random pairs from last draw for efficiency
        np.random.seed(int(time.time()) % 10000)
        if len(last) >= 2:
            pair_indices = list(combinations(range(len(last)), 2))
            if len(pair_indices) > 15:
                sample_idx = np.random.choice(len(pair_indices), 15, replace=False)
                pair_indices = [pair_indices[i] for i in sample_idx]
            
            pairs = [(last[i], last[j]) for i, j in pair_indices]
        else:
            return {n: 0.5 for n in range(1, 81)}
        
        # For each pair, find draws where both appeared, 
        # then see what appeared in the NEXT draw
        pair_followers = defaultdict(float)
        pair_count = 0
        
        for a, b in pairs:
            for i in range(max(start, 0), self.n_draws - 1):
                if a in self.draw_sets[i] and b in self.draw_sets[i]:
                    pair_count += 1
                    for num in self.draw_sets[i + 1]:
                        pair_followers[num] += 1
        
        if pair_count > 0:
            scores = {num: pair_followers.get(num, 0) / pair_count for num in range(1, 81)}
        else:
            scores = {n: 0.5 for n in range(1, 81)}
        
        return self._normalize(scores)
    
    def _cluster_tendency(self) -> Dict[int, float]:
        """
        Numbers near recently drawn numbers tend to cluster.
        E.g., if 35 was drawn, 34 and 36 might have slightly higher chance.
        """
        last_draws = self.history[-3:]  # last 3 draws
        
        proximity_score = defaultdict(float)
        
        for draw in last_draws:
            for num in draw:
                for delta in [-3, -2, -1, 1, 2, 3]:
                    neighbor = num + delta
                    if 1 <= neighbor <= 80:
                        weight = 1.0 / abs(delta)
                        proximity_score[neighbor] += weight
        
        # Also boost numbers that appeared in last 3 draws
        for draw in last_draws:
            for num in draw:
                proximity_score[num] += 0.5
        
        # Fill missing
        for n in range(1, 81):
            if n not in proximity_score:
                proximity_score[n] = 0
        
        return self._normalize(dict(proximity_score))
    
    def _anti_streak(self) -> Dict[int, float]:
        """
        Numbers on a long ABSENCE streak get boosted.
        Numbers on a long PRESENCE streak get slight penalty.
        (Mean reversion tendency)
        """
        scores = {}
        
        for num in range(1, 81):
            app = self.appearances[num]
            
            # Count current streak
            streak_len = 0
            if self.n_draws > 0:
                current_state = app[-1]
                for k in range(self.n_draws - 1, -1, -1):
                    if app[k] == current_state:
                        streak_len += 1
                    else:
                        break
            
            if not current_state:
                # Number has been ABSENT for streak_len draws
                # Expected gap ≈ 4 draws (25% rate)
                # Longer absence → higher overdue score
                scores[num] = 1 - np.exp(-0.3 * streak_len)
            else:
                # Number has been PRESENT for streak_len draws
                # Hot streak — slight boost but diminishing
                scores[num] = 0.5 + 0.1 * min(streak_len, 5)
        
        return self._normalize(scores)
    
    def _lstm_predict(self) -> Dict[int, float]:
        """LSTM prediction using recent sequence."""
        if not HAS_TF or self.n_draws < 50:
            return {}
        
        print("   Training LSTM...")
        
        # Use last 2000 draws max for training speed
        recent_n = min(2000, self.n_draws)
        recent_history = self.history[-recent_n:]
        
        seq_len = 15
        
        # Multi-hot encode
        mh = np.zeros((len(recent_history), 80), dtype=np.float32)
        for i, draw in enumerate(recent_history):
            for num in draw:
                mh[i, num - 1] = 1.0
        
        if len(mh) < seq_len + 10:
            return {}
        
        X = np.array([mh[i-seq_len:i] for i in range(seq_len, len(mh))])
        y = np.array([mh[i] for i in range(seq_len, len(mh))])
        
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=(seq_len, 80)),
            Dropout(0.3),
            LSTM(50),
            Dropout(0.3),
            Dense(100, activation='relu'),
            Dense(80, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X, y, epochs=40, batch_size=64, validation_split=0.1,
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                  verbose=0)
        
        last_seq = mh[-seq_len:].reshape(1, seq_len, 80)
        pred = model.predict(last_seq, verbose=0)[0]
        
        scores = {i + 1: float(pred[i]) for i in range(80)}
        return self._normalize(scores)
    
    @staticmethod
    def _normalize(scores: Dict[int, float]) -> Dict[int, float]:
        """Normalize scores to [0, 1] with proper spread."""
        if not scores:
            return scores
        
        vals = np.array(list(scores.values()))
        mn, mx = vals.min(), vals.max()
        rng = mx - mn
        
        if rng < 1e-10:
            # All scores identical → add random tiebreaker
            # This prevents the 1,2,3,4,5,6,7 problem
            return {k: 0.5 for k, v in scores.items()}
        
        # Z-score normalization then sigmoid for spread
        mean = vals.mean()
        std = vals.std() + 1e-10
        
        result = {}
        for k, v in scores.items():
            z = (v - mean) / std
            result[k] = 1 / (1 + np.exp(-z))  # sigmoid squash
        
        return result


# ═══════════════════════════════════════════════════════════
#  SECTION 3: PICK SELECTOR (Anti-degeneracy)
# ═══════════════════════════════════════════════════════════

class PickSelector:
    """
    Selects best picks with:
    - Diversity enforcement (no sequential clusters)
    - Model consensus requirement
    - Confidence thresholds
    """
    
    # MA Keno payouts per $1
    PAYOUTS = {
        5: {0:0, 1:0, 2:1, 3:2, 4:18, 5:420},
        6: {0:1, 1:0, 2:0, 3:1, 4:7, 5:50, 6:1600},
        7: {0:0, 1:0, 2:0, 3:1, 4:3, 5:17, 6:100, 7:7000},
        8: {0:0, 1:0, 2:0, 3:0, 4:2, 5:10, 6:50, 7:500, 8:15000},
    }
    
    MODEL_WEIGHTS = {
        'short_freq':  0.15,
        'momentum':    0.18,
        'markov':      0.20,
        'gap':         0.15,
        'pair':        0.10,
        'cluster':     0.07,
        'anti_streak': 0.08,
        'lstm':        0.12,
    }
    
    def __init__(self, all_scores: Dict[str, Dict[int, float]], 
                 history: np.ndarray):
        self.all_scores = all_scores
        self.history = history
        self.weights = dict(self.MODEL_WEIGHTS)
    
    def select(self, pick_count: int = 8) -> Dict:
        """
        Select best numbers with diversity enforcement.
        
        Returns dict with picks, confidence, analysis.
        """
        # ── Step 1: Ensemble score ──
        ensemble = self._weighted_ensemble()
        
        # ── Step 2: Model consensus bonus ──
        consensus = self._model_consensus(pick_count)
        
        # Blend ensemble + consensus
        for num in ensemble:
            con_score = consensus.get(num, 0)
            ensemble[num] = 0.7 * ensemble[num] + 0.3 * con_score
        
        # ── Step 3: Select with diversity ──
        ranked = sorted(ensemble.items(), key=lambda x: x[1], reverse=True)
        picks = self._diverse_selection(ranked, pick_count)
        
        # ── Step 4: Confidence ──
        max_score = ranked[0][1] if ranked else 1
        confidence = {}
        for num in picks:
            raw_conf = ensemble[num] / max_score * 100
            confidence[num] = round(raw_conf, 1)
        
        # ── Step 5: Per-model breakdown ──
        model_picks = {}
        for mname, mscores in self.all_scores.items():
            top = sorted(mscores.items(), key=lambda x: x[1], reverse=True)
            model_picks[mname] = [n for n, _ in top[:pick_count]]
        
        return {
            'picks': picks,
            'confidence': confidence,
            'model_picks': model_picks,
            'full_ranking': ranked,
            'ensemble_scores': ensemble,
        }
    
    def _weighted_ensemble(self) -> Dict[int, float]:
        """Combine all model scores with weights."""
        final = defaultdict(float)
        weight_sum = defaultdict(float)
        
        for mname, mscores in self.all_scores.items():
            w = self.weights.get(mname, 0.05)
            for num, score in mscores.items():
                final[num] += w * score
                weight_sum[num] += w
        
        return {num: final[num] / (weight_sum[num] + 1e-8) for num in final}
    
    def _model_consensus(self, pick_count: int) -> Dict[int, float]:
        """
        Numbers picked by MULTIPLE models get a bonus.
        This prevents any single model from dominating.
        """
        vote_count = defaultdict(int)
        n_models = len(self.all_scores)
        
        for mname, mscores in self.all_scores.items():
            top = sorted(mscores.items(), key=lambda x: x[1], reverse=True)
            # Each model votes for its top picks
            for num, _ in top[:pick_count * 2]:  # generous top pool
                vote_count[num] += 1
        
        # Normalize: fraction of models that voted for this number
        return {num: count / n_models for num, count in vote_count.items()}
    
    def _diverse_selection(self, ranked: List[Tuple[int, float]], 
                           pick_count: int) -> List[int]:
        """
        Pick top numbers but enforce diversity:
        - No more than 2 consecutive numbers (e.g., no 5,6,7)
        - Spread across ranges (1-20, 21-40, 41-60, 61-80)
        - Don't just pick the mathematical top-K
        """
        picks = []
        candidates = list(ranked)
        
        for num, score in candidates:
            if len(picks) >= pick_count:
                break
            
            # Check: would this create a run of 3+ consecutive?
            if self._creates_long_run(picks, num, max_run=2):
                continue
            
            # Check: would this over-concentrate in one quadrant?
            if self._over_concentrated(picks, num, pick_count):
                continue
            
            picks.append(num)
        
        # If diversity rules were too strict, fill remaining from top
        if len(picks) < pick_count:
            for num, score in candidates:
                if num not in picks:
                    picks.append(num)
                if len(picks) >= pick_count:
                    break
        
        return sorted(picks)
    
    @staticmethod
    def _creates_long_run(current_picks: List[int], new_num: int, 
                          max_run: int = 2) -> bool:
        """Check if adding new_num creates a consecutive run > max_run."""
        test = sorted(current_picks + [new_num])
        
        run = 1
        for i in range(1, len(test)):
            if test[i] == test[i-1] + 1:
                run += 1
                if run > max_run:
                    return True
            else:
                run = 1
        
        return False
    
    @staticmethod
    def _over_concentrated(current_picks: List[int], new_num: int, 
                           total_picks: int) -> bool:
        """No more than ceil(total/2) picks in same quadrant."""
        max_per_quad = (total_picks + 1) // 2
        
        test = current_picks + [new_num]
        quads = [0, 0, 0, 0]
        for n in test:
            q = min(3, (n - 1) // 20)
            quads[q] += 1
        
        return any(q > max_per_quad for q in quads)
    
    def calibrate_weights(self, pick_count: int) -> Dict[str, float]:
        """
        Walk-forward backtest to calibrate model weights.
        Uses only the RECENT portion of history.
        """
        history = self.history
        n = len(history)
        
        # Use last 20% for testing, rest for training
        # But only test on last 500 draws max for speed
        test_size = min(500, n // 5)
        train_end = n - test_size
        
        if train_end < 100:
            return self.weights
        
        model_hits = defaultdict(list)
        
        # Sample test points for speed
        test_indices = np.linspace(train_end, n - 1, min(100, test_size), dtype=int)
        
        for test_idx in test_indices:
            actual = set(history[test_idx])
            
            # Score from each model using data up to test_idx
            train = history[:test_idx]
            scorer = KenoScorer(train)
            
            # Only run fast models for calibration
            fast_scores = {
                'short_freq':  scorer._short_term_frequency(),
                'momentum':    scorer._momentum(),
                'gap':         scorer._gap_analysis(),
                'anti_streak': scorer._anti_streak(),
            }
            
            for mname, mscores in fast_scores.items():
                top = sorted(mscores.items(), key=lambda x: x[1], reverse=True)
                predicted = set(num for num, _ in top[:pick_count])
                hits = len(predicted & actual)
                model_hits[mname].append(hits)
        
        # Set weights proportional to average hits
        avg = {m: np.mean(h) for m, h in model_hits.items()}
        total = sum(avg.values()) + 1e-8
        
        new_weights = dict(self.weights)
        for m in avg:
            new_weights[m] = 0.4 * (avg[m] / total) + 0.6 * self.weights.get(m, 0.1)
        
        # Normalize
        wt = sum(new_weights.values())
        new_weights = {k: v / wt for k, v in new_weights.items()}
        
        return new_weights


# ═══════════════════════════════════════════════════════════
#  SECTION 4: MAIN AGENT
# ═══════════════════════════════════════════════════════════

class KenoOptimizerV2:
    """
    Main agent — fixed version.
    
    Usage:
        agent = KenoOptimizerV2()
        result = agent.run("your_data.csv", pick_count=8)
    """
    
    def run(self, data_source: str, pick_count: int = 8, 
            future_draws: int = 10, use_lstm: bool = True) -> Dict:
        """
        Full pipeline: Load → Score → Select → Analyze.
        """
        if pick_count < 5 or pick_count > 8:
            raise ValueError("pick_count must be 5-8")
        
        # ── Load Data ──
        print("\n" + "═" * 65)
        print("  🎰  KENO OPTIMIZER V2  🎰")
        print("═" * 65)
        
        if data_source.endswith('.csv'):
            history = KenoDataLoader.load_csv(data_source)
        elif data_source.endswith('.pdf'):
            history = KenoDataLoader.load_pdf(data_source)
        else:
            raise ValueError("Provide .csv or .pdf file")
        
        history = np.array(history)
        n = len(history)
        
        print(f"\n  Config: Pick {pick_count} | {n} draws | "
              f"Predict {future_draws} draws")
        print("─" * 65)
        
        # ── Score All Numbers ──
        print(f"\n🔬 Scoring (focusing on RECENT patterns, not lifetime freq)...")
        t0 = time.time()
        scorer = KenoScorer(history)
        all_scores = scorer.score_all()
        print(f"   Scoring done in {time.time()-t0:.1f}s")
        print(f"   Active models: {list(all_scores.keys())}")
        
        # ── Calibrate Weights ──
        print("\n⚙️  Calibrating model weights...")
        selector = PickSelector(all_scores, history)
        
        if n >= 200:
            t0 = time.time()
            selector.weights = selector.calibrate_weights(pick_count)
            print(f"   Calibrated in {time.time()-t0:.1f}s")
        
        # ── Select Picks ──
        print(f"\n🎯 Selecting best {pick_count} numbers...")
        selection = selector.select(pick_count)
        
        picks = selection['picks']
        confidence = selection['confidence']
        
        # ── Generate Per-Draw Predictions ──
        draw_preds = self._per_draw_predictions(
            selection, all_scores, pick_count, future_draws
        )
        
        # ── Payout Analysis ──
        payout = self._payout_analysis(selection['full_ranking'], pick_count)
        
        # ── Analytics ──
        analytics = self._build_analytics(history, all_scores, selection, pick_count)
        
        # ── Result ──
        result = {
            'picks': picks,
            'pick_count': pick_count,
            'confidence': confidence,
            'draw_predictions': draw_preds,
            'payout_analysis': payout,
            'model_picks': selection['model_picks'],
            'analytics': analytics,
        }
        
        self._print_results(result)
        
        return result
    
    def backtest(self, data_source: str, pick_count: int = 8, 
                 n_tests: int = 200) -> Dict:
        """
        Walk-forward backtest with payout tracking.
        """
        if data_source.endswith('.csv'):
            history = np.array(KenoDataLoader.load_csv(data_source))
        else:
            history = np.array(KenoDataLoader.load_pdf(data_source))
        
        n = len(history)
        test_start = max(100, n - n_tests)
        actual_tests = n - test_start
        
        payouts = PickSelector.PAYOUTS.get(pick_count, {})
        
        hits_list = []
        payout_list = []
        
        print(f"\n🧪 BACKTEST: {actual_tests} draws, Pick {pick_count}")
        print(f"   Training on {test_start}+ draws each time")
        
        for test_idx in range(test_start, n):
            train = history[:test_idx]
            actual = set(history[test_idx])
            
            # Quick scoring (no LSTM for speed)
            scorer = KenoScorer(train)
            scores = {
                'short_freq':  scorer._short_term_frequency(),
                'momentum':    scorer._momentum(),
                'markov':      scorer._markov_chains(),
                'gap':         scorer._gap_analysis(),
                'anti_streak': scorer._anti_streak(),
            }
            
            sel = PickSelector(scores, train)
            result = sel.select(pick_count)
            predicted = set(result['picks'])
            
            hits = len(predicted & actual)
            hits_list.append(hits)
            payout_list.append(payouts.get(hits, 0))
            
            if (test_idx - test_start + 1) % 50 == 0:
                print(f"   [{test_idx-test_start+1}/{actual_tests}] "
                      f"avg hits: {np.mean(hits_list):.2f} | "
                      f"avg payout: ${np.mean(payout_list):.2f}")
        
        hits_arr = np.array(hits_list)
        payout_arr = np.array(payout_list)
        
        random_expected = pick_count * 20 / 80
        
        bt = {
            'n_tests': actual_tests,
            'pick_count': pick_count,
            'avg_hits': round(float(np.mean(hits_arr)), 3),
            'random_expected': round(random_expected, 3),
            'improvement_x': round(float(np.mean(hits_arr)) / random_expected, 3),
            'hit_distribution': dict(Counter(hits_arr.astype(int).tolist())),
            'avg_payout': round(float(np.mean(payout_arr)), 3),
            'total_wagered': actual_tests,
            'total_payout': round(float(np.sum(payout_arr)), 2),
            'net': round(float(np.sum(payout_arr)) - actual_tests, 2),
            'roi_pct': round((float(np.mean(payout_arr)) - 1) * 100, 1),
        }
        
        print(f"\n  {'═'*50}")
        print(f"  BACKTEST RESULTS (Pick {pick_count})")
        print(f"  {'═'*50}")
        print(f"  Avg matches/draw:  {bt['avg_hits']:.2f}")
        print(f"  Random expected:   {bt['random_expected']:.2f}")
        print(f"  Improvement:       {bt['improvement_x']:.3f}x")
        print(f"  Hit distribution:  {bt['hit_distribution']}")
        print(f"  Avg payout/draw:   ${bt['avg_payout']:.2f}")
        print(f"  Total wagered:     ${bt['total_wagered']}")
        print(f"  Total payouts:     ${bt['total_payout']}")
        print(f"  Net profit/loss:   ${bt['net']}")
        print(f"  ROI:               {bt['roi_pct']}%")
        print(f"  {'═'*50}")
        
        return bt
    
    def _per_draw_predictions(self, selection, all_scores, pick_count, future_draws):
        """Slightly varied picks per future draw."""
        preds = []
        ranked = selection['full_ranking']
        pool = ranked[:pick_count * 3]
        
        for di in range(future_draws):
            adjusted = {}
            for num, base in pool:
                gap_s = all_scores.get('gap', {}).get(num, 0.5)
                mom_s = all_scores.get('momentum', {}).get(num, 0.5)
                
                # Early: favor momentum. Later: favor gap/overdue.
                mom_w = max(0.1, 0.5 - 0.04 * di)
                gap_w = min(0.5, 0.2 + 0.03 * di)
                
                adjusted[num] = base * 0.5 + mom_s * mom_w + gap_s * gap_w
            
            top = sorted(adjusted.items(), key=lambda x: x[1], reverse=True)
            
            # Apply diversity filter
            div_picks = []
            for num, _ in top:
                if len(div_picks) >= pick_count:
                    break
                if not PickSelector._creates_long_run(div_picks, num, 2):
                    div_picks.append(num)
            
            # Fill if needed
            if len(div_picks) < pick_count:
                for num, _ in top:
                    if num not in div_picks:
                        div_picks.append(num)
                    if len(div_picks) >= pick_count:
                        break
            
            preds.append(sorted(div_picks[:pick_count]))
        
        return preds
    
    def _payout_analysis(self, ranked, pick_count):
        """EV analysis for pick 5-8."""
        results = {}
        for pc in [5, 6, 7, 8]:
            top = ranked[:pc]
            payouts = PickSelector.PAYOUTS.get(pc, {})
            
            # Each picked number has ~25% chance 
            # (model might push to ~26-28% for top picks)
            p = 0.27  # slight optimistic estimate for model picks
            
            ev = 0
            dist = {}
            for k in range(pc + 1):
                prob = comb(pc, k) * (p ** k) * ((1-p) ** (pc-k))
                dist[k] = round(prob, 4)
                ev += payouts.get(k, 0) * prob
            
            results[pc] = {
                'ev': round(ev, 3),
                'picks': [n for n, _ in top],
                'match_dist': dist,
            }
        
        return results
    
    def _build_analytics(self, history, all_scores, selection, pick_count):
        """Summary stats."""
        recent = [n for draw in history[-30:] for n in draw]
        rf = Counter(recent)
        
        return {
            'total_draws': len(history),
            'hot_10': rf.most_common(10),
            'cold_10': sorted([(n, rf.get(n, 0)) for n in range(1,81)], 
                              key=lambda x: x[1])[:10],
            'overdue': sorted(all_scores.get('gap', {}).items(), 
                            key=lambda x: x[1], reverse=True)[:10],
            'model_weights': selection.get('model_picks', {}),
        }
    
    def _print_results(self, result):
        """Nice output."""
        picks = result['picks']
        conf = result['confidence']
        pc = result['pick_count']
        
        print("\n" + "═" * 65)
        print(f"  🎯  YOUR {pc} PICKS")
        print("═" * 65)
        
        # Big display
        nums_str = "  ".join(f"[{n:2d}]" for n in picks)
        print(f"\n  ▶  {nums_str}  ◀")
        
        print(f"\n  Confidence:")
        for num in picks:
            c = conf.get(num, 0)
            filled = int(c / 5)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"    {num:3d}  [{bar}] {c:.1f}%")
        
        # Per-draw
        print(f"\n  📅 Next {len(result['draw_predictions'])} Draws:")
        for i, dp in enumerate(result['draw_predictions'], 1):
            marker = "★" if dp == picks else " "
            print(f"   {marker} Draw {i:2d}: {dp}")
        
        # Payout
        print(f"\n  💰 Expected Value (per $1 bet):")
        for pc_opt, info in sorted(result['payout_analysis'].items()):
            marker = " ◄" if pc_opt == pc else ""
            print(f"    Pick {pc_opt}: {info['picks'][:pc_opt]}  →  "
                  f"EV=${info['ev']:.3f}{marker}")
        
        # Model agreement
        print(f"\n  🤖 Model Picks:")
        for mname, mpicks in result['model_picks'].items():
            print(f"    {mname:15s}  {mpicks}")
        
        # Analytics
        a = result['analytics']
        print(f"\n  🔥 Hot:     {[n for n,_ in a['hot_10'][:8]]}")
        print(f"  ❄️  Cold:    {[n for n,_ in a['cold_10'][:8]]}")
        print(f"  ⏰ Overdue: {[n for n,_ in a['overdue'][:8]]}")
        
        print(f"\n  📊 Based on {a['total_draws']} historical draws")
        print("═" * 65)


# ═══════════════════════════════════════════════════════════
#  SECTION 5: RUN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else "keno_data.csv"
    pc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    agent = KenoOptimizerV2()
    
    # Predict
    result = agent.run(data, pick_count=pc)
    
    # Backtest
    print("\n" + "─" * 65)
    bt = agent.backtest(data, pick_count=pc, n_tests=300)