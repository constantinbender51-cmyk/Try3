import requests
import pandas as pd
import numpy as np
from collections import Counter
import os
import itertools
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = '/app/data'

PARAMS = {
    'SYMBOL': ['BTCUSDT', 'ETHUSDT', 'XRPUSDT'],
    'TIMEFRAME': ['30m', '1h', '4h', '1d'],
    'START_YEAR': ['2020', '2021', '2022', '2023'],
    'END_YEAR': ['2024', '2025', '2026'],
    'A_ROUND': [1.024, 0.512, 0.256, 0.128, 0.064, 0.032, 0.016, 0.008, 0.004, 0.002],
    'C_TOP': [2, 4, 8, 16, 32],
    'D_LEN': [2, 3, 4, 5],
    'E_SIM': [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32],
    'B_SPLIT': [70]
}

# Global cache for workers (Read-only after init)
# Structure: { (SYMBOL, TIMEFRAME): (timestamps_array, ohlc_numpy_array) }
GLOBAL_DATA_CACHE = {}

# ==========================================
# DATA & PRE-PROCESSING
# ==========================================

def get_pandas_freq(tf):
    if tf.endswith('m'): return tf.replace('m', 'min')
    if tf.endswith('h'): return tf
    if tf.endswith('d'): return tf.replace('d', 'D')
    return '1h'

def fetch_1m_data(symbol, start_year=2020, end_year=2026):
    ensure_data_dir()
    file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    
    if os.path.exists(file_path):
        print(f"[DATA] Loading {symbol} from cache...")
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df

    print(f"[DATA] Downloading {symbol} (2020-2026)...")
    start_ts = int(datetime(start_year, 1, 1).timestamp() * 1000)
    end_ts = int(datetime(end_year, 12, 31).timestamp() * 1000)
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start = start_ts
    
    while current_start < end_ts:
        params = {'symbol': symbol, 'interval': '1m', 'startTime': current_start, 'limit': 1000}
        try:
            r = requests.get(base_url, params=params)
            klines = r.json()
            if not klines: break
            for k in klines:
                all_data.append([pd.to_datetime(k[0], unit='ms'), float(k[1]), float(k[2]), float(k[3]), float(k[4])])
            current_start = klines[-1][6] + 1
            time.sleep(0.05) # Nice to API
        except Exception as e:
            print(f"Error: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=['datetime', 'open', 'high', 'low', 'close'])
    df.set_index('datetime', inplace=True)
    df.to_csv(file_path)
    return df

def ensure_data_dir():
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

def precompute_all_data():
    """
    Loads 1m data and resamples to ALL target timeframes beforehand.
    Stores pure NumPy arrays in memory to avoid Pandas overhead in the loop.
    """
    print("[INIT] Pre-computing resampled data for all timeframes...")
    
    for symbol in PARAMS['SYMBOL']:
        # Load 1m data once
        df_1m = fetch_1m_data(symbol)
        
        for tf in PARAMS['TIMEFRAME']:
            freq = get_pandas_freq(tf)
            agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
            try:
                # Resample
                resampled = df_1m.resample(freq).agg(agg).dropna()
                
                # Store as tuple: (Time Index Array, OHLC Float32 Array)
                # Using float32 saves memory and is faster for SIMD
                times = resampled.index.values
                ohlc = resampled[['open', 'high', 'low', 'close']].values.astype(np.float32)
                
                GLOBAL_DATA_CACHE[(symbol, tf)] = (times, ohlc)
            except Exception as e:
                print(f"Resample failed for {symbol} {tf}: {e}")

# ==========================================
# VECTORIZED ALGORITHMS
# ==========================================

def vectorized_deriveround(ohlc, a):
    """
    Vectorized calculation of percentage change and rounding.
    Input: ohlc (N, 4) numpy array
    Output: derived (N-1, 4) numpy array
    """
    # Calculate % Change: (Curr - Prev) / Prev * 100
    prev = ohlc[:-1]
    curr = ohlc[1:]
    
    # Handle division by zero safely
    with np.errstate(divide='ignore', invalid='ignore'):
        change = ((curr - prev) / prev) * 100.0
        
    change[~np.isfinite(change)] = 0.0  # Replace inf/nan with 0
    
    # Rounding logic: round(x/a)*a
    # Using np.round (ties to even) is standard
    derived = np.round(change / a) * a
    return derived

def get_sliding_windows(arr, window_size):
    """
    Efficiently create sliding windows view using stride tricks.
    Input: (N, 4)
    Output: (N-W+1, W, 4)
    """
    return np.lib.stride_tricks.sliding_window_view(arr, window_shape=(window_size, 4))

def vectorized_backtest(test_data, top_seqs, d, e):
    """
    Fully vectorized backtest using broadcasting.
    test_data: (N, 4) array
    top_seqs: list of (d, 4) tuples
    """
    if len(test_data) < d:
        return 0, 0, 0.0

    begin_len = d - 1
    # 1. Create windows from test data: Shape (Num_Windows, begin_len, 4)
    # These are the "inputs" we check against patterns
    windows = np.lib.stride_tricks.sliding_window_view(test_data[:-1], window_shape=(begin_len, 4))
    # We strip the last row from view inputs because we need the NEXT row as outcome
    # Actually, if test_data is length N.
    # Window i starts at i. Ends at i+begin_len. Outcome is at i+begin_len.
    # We need windows up to N-d.
    
    # Re-logic:
    # We need windows of length (d-1).
    # Window 0: indices 0..d-2. Outcome: index d-1.
    # Last Window: indices ... N-2. Outcome: index N-1.
    
    windows = np.lib.stride_tricks.sliding_window_view(test_data[:-1], window_shape=(begin_len, 4))
    # Outcomes are the 'close' change (col 3) of the row immediately following the window
    # test_data has N rows. Windows view reduces size by begin_len-1.
    # Correct slicing for outcomes to align with windows:
    outcomes = test_data[begin_len:, 3]
    
    # Ensure shapes match
    n_samples = min(len(windows), len(outcomes))
    windows = windows[:n_samples]
    outcomes = outcomes[:n_samples]
    
    predictions = np.zeros(n_samples)
    has_prediction = np.zeros(n_samples, dtype=bool)

    # 2. Iterate through patterns. 
    # (Iterating patterns is better than iterating data because patterns are few ~10-30, data is ~50k)
    # We want "First Match" priority.
    
    for seq in top_seqs:
        # seq is tuple of tuples. Convert to array.
        # Pattern to match: first d-1 rows
        pattern = np.array(seq[:begin_len], dtype=np.float32)
        pred_val = seq[begin_len][3]
        
        # Vectorized Similarity Check
        # Condition: abs(w - p) / abs(p) < e
        # Rewrite to avoid div/0: abs(w - p) < e * abs(p)
        # Handle zeros: if p==0, w must be 0.
        
        abs_pattern = np.abs(pattern)
        
        # Calculate diffs for all windows at once (Broadcasting)
        # windows: (N, d-1, 4), pattern: (d-1, 4) -> Broadcasts
        diffs = np.abs(windows - pattern)
        
        # Threshold matrix
        thresholds = e * abs_pattern
        
        # Check 1: Value similarity
        # We use a small epsilon for float comparison safety if e=0
        is_close = diffs <= (thresholds + 1e-9)
        
        # Check 2: Zero handling (if pattern is 0, window must be 0)
        # If pattern != 0, the ratio check covers it.
        # If pattern == 0, thresholds is 0. diffs must be 0 (window == pattern).
        # However, logic in original: if val1(pattern)==0: if val2!=0 return False.
        # This implies if pattern is 0, window MUST be 0.
        # Our diff check `abs(w - 0) <= 0` -> `w=0`. It handles it correctly.
        
        # Combine across all elements in the window (d-1) and all metrics (4)
        # Axis (1,2) reduces (N, d-1, 4) to (N,)
        matches = np.all(is_close, axis=(1, 2))
        
        # Apply predictions to spots that haven't matched yet (First Match Priority)
        # "mask_to_update" = matches AND (NOT has_prediction)
        update_mask = matches & (~has_prediction)
        
        predictions[update_mask] = pred_val
        has_prediction[update_mask] = True
        
        # Optimization: If all filled, break
        if np.all(has_prediction):
            break
            
    # 3. Calculate Stats
    # Filter only where we had a prediction and prediction != 0 and actual != 0
    valid_mask = has_prediction & (predictions != 0) & (outcomes != 0)
    
    if not np.any(valid_mask):
        return 0, 0, 0.0
        
    final_preds = predictions[valid_mask]
    final_actuals = outcomes[valid_mask]
    
    correct = ((final_preds > 0) & (final_actuals > 0)) | ((final_preds < 0) & (final_actuals < 0))
    n_correct = np.count_nonzero(correct)
    n_trades = len(final_preds)
    
    # Vectorized PnL
    dirs = np.sign(final_preds)
    pnl = np.sum(dirs * final_actuals)
    
    accuracy = n_correct / n_trades
    return accuracy, n_trades, pnl

# ==========================================
# WORKER PROCESS
# ==========================================

def run_grid_task(params):
    """
    Executed by worker process.
    params: dict of current combination
    """
    # 1. Retrieve Pre-computed Data
    # GLOBAL_DATA_CACHE is available in the process memory (Copy-on-Write on Linux)
    key = (params['SYMBOL'], params['TIMEFRAME'])
    if key not in GLOBAL_DATA_CACHE:
        return None
        
    times, ohlc_data = GLOBAL_DATA_CACHE[key]
    
    # 2. Slice by Date (Vectorized)
    # Convert 'YYYY' to numpy datetime64
    t_start = np.datetime64(f"{params['START_YEAR']}-01-01")
    t_end = np.datetime64(f"{params['END_YEAR']}-12-31")
    
    # Searchsorted is faster than boolean indexing for ranges on sorted arrays
    idx_start = np.searchsorted(times, t_start)
    idx_end = np.searchsorted(times, t_end, side='right')
    
    if idx_end - idx_start < 100: return None
    
    raw_slice = ohlc_data[idx_start:idx_end]
    
    # 3. Derive & Round (Vectorized)
    derived = vectorized_deriveround(raw_slice, params['A_ROUND'])
    
    # 4. Split
    split_idx = int(len(derived) * (params['B_SPLIT'] / 100.0))
    train_data = derived[:split_idx]
    test_data = derived[split_idx:]
    
    # 5. Get Top Sequences
    # Convert rolling window to tuples for Counter (Python part, but strictly limited by C_TOP)
    d = params['D_LEN']
    if len(train_data) < d: return None
    
    # Get all sequences in train
    train_windows = get_sliding_windows(train_data, d) # Shape (N, d, 4)
    
    # Convert to list of tuples for hashing/counting
    # This loop is fast because N is reduced by resample and operations are simple
    # but strictly speaking, we can optimize. For now, this is okay.
    # To speed up: Use map
    seqs_as_tuples = [tuple(map(tuple, w)) for w in train_windows]
    
    if not seqs_as_tuples: return None
    
    counts = Counter(seqs_as_tuples)
    unique = counts.most_common()
    limit = max(1, int(len(unique) * (params['C_TOP'] / 100.0)))
    top_seqs = [x[0] for x in unique[:limit]]
    
    # 6. Backtest (Vectorized)
    acc, trades, pnl = vectorized_backtest(test_data, top_seqs, d, params['E_SIM'])
    
    score = acc * trades
    
    return {
        'params': params,
        'score': score,
        'accuracy': acc,
        'trades': trades,
        'pnl': pnl
    }

# ==========================================
# MAIN
# ==========================================

def main():
    # 1. Pre-load data into global memory
    # (Must be done before creating the ProcessPool so workers inherit the data)
    precompute_all_data()
    
    # 2. Generate Param Grid
    keys = list(PARAMS.keys())
    combinations = list(itertools.product(*[PARAMS[k] for k in keys]))
    total_combos = len(combinations)
    
    print(f"\n[GRID SEARCH] Starting optimization on {total_combos} combinations.")
    print(f"[SYSTEM] Using ProcessPoolExecutor with {os.cpu_count()} cores.")
    
    best_score = -999999.0
    best_result = None
    
    start_time = time.time()
    
    # 3. Run Parallel Execution
    # Chunksize helps reduce IPC overhead for very fast tasks
    with ProcessPoolExecutor() as executor:
        # Prepare dictionaries
        task_list = [dict(zip(keys, combo)) for combo in combinations]
        
        # Filter invalid dates before submitting
        task_list = [p for p in task_list if int(p['END_YEAR']) > int(p['START_YEAR'])]
        
        print(f"[GRID SEARCH] executing {len(task_list)} valid tasks...")
        
        # Submit all tasks
        futures = [executor.submit(run_grid_task, p) for p in task_list]
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            
            if res and res['score'] > best_score:
                best_score = res['score']
                best_result = res
                print(f"[*] NEW BEST: Score {res['score']:.2f} | Acc: {res['accuracy']:.2%} | Trades: {res['trades']} | PnL: {res['pnl']:.2f}")
                # print(f"    Params: {res['params']}")
            
            if completed % 5000 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                print(f"Progress: {completed}/{len(task_list)} ({completed/len(task_list):.1%}) | Rate: {rate:.1f} tasks/sec")

    print("\n==========================================")
    print("GRID SEARCH COMPLETE")
    print("==========================================")
    if best_result:
        print(f"BEST SCORE: {best_score:.4f}")
        print(f"ACCURACY:   {best_result['accuracy']:.2%}")
        print(f"TRADES:     {best_result['trades']}")
        print(f"TOTAL PNL:  {best_result['pnl']:.2f}%")
        print("------------------------------------------")
        print("OPTIMAL PARAMETERS:")
        for k, v in best_result['params'].items():
            print(f"{k}: {v}")
    else:
        print("No valid results found.")

if __name__ == "__main__":
    main()
