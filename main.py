import requests
import math
import pandas as pd
from collections import Counter
import os
import itertools
import time
from datetime import datetime

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================
# Data Storage
DATA_DIR = '/app/data'

# Grid Search Parameter Space
PARAMS = {
    'SYMBOL': ['BTCUSDT', 'ETHUSDT', 'XRPUSDT'],
    # Updated: Timeframes start from 1h onward
    'TIMEFRAME': ['1h', '2h', '4h', '6h', '8h', '12h', '1d'],
    'START_YEAR': ['2020', '2021', '2022', '2023'],
    'END_YEAR': ['2024', '2025', '2026'],
    'A_ROUND': [1.024, 0.512, 0.256, 0.128, 0.064, 0.032, 0.016, 0.008, 0.004, 0.002],
    'C_TOP': [2, 4, 8, 16, 32],
    'D_LEN': [2, 3, 4, 5],
    'E_SIM': [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32],
    'B_SPLIT': [70]
}

# ==========================================
# DATA MANAGEMENT
# ==========================================

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def get_pandas_freq(tf):
    """Maps custom timeframe string to Pandas offset alias."""
    # Minutes: '1m' -> '1min' (Newer Pandas prefers 'min' over 'T')
    if tf.endswith('m'): 
        return tf.replace('m', 'min')
    
    # Hours: '1h' -> '1h' (Newer Pandas requires lowercase 'h', 'H' is removed)
    if tf.endswith('h'): 
        return tf # already '1h', '4h', etc.
    
    # Days: '1d' -> '1D' (Calendar day uses uppercase 'D')
    if tf.endswith('d'): 
        return tf.replace('d', 'D')
    
    return '1h'

def fetch_1m_data(symbol, start_year=2020, end_year=2026):
    """
    Fetches 1m data from Binance for the entire range (2020-2026 + buffer).
    Checks local cache first.
    """
    file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    
    # Return cached if exists
    if os.path.exists(file_path):
        print(f"[DATA] Loading cached data for {symbol}...")
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df

    print(f"[DATA] Fetching 1m data for {symbol} from {start_year} to {end_year} (this may take time)...")
    
    # Convert years to timestamps
    start_ts = int(datetime(start_year, 1, 1).timestamp() * 1000)
    end_ts = int(datetime(end_year, 12, 31).timestamp() * 1000)
    
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start = start_ts
    
    while current_start < end_ts:
        params = {
            'symbol': symbol, 
            'interval': '1m',
            'startTime': current_start, 
            'limit': 1000
        }
        try:
            r = requests.get(base_url, params=params)
            if r.status_code != 200:
                print(f"API Error: {r.status_code}")
                break
                
            klines = r.json()
            if not klines: 
                break
                
            for k in klines:
                # Store: Open Time, Open, High, Low, Close
                ts = pd.to_datetime(k[0], unit='ms')
                row = [ts, float(k[1]), float(k[2]), float(k[3]), float(k[4])]
                all_data.append(row)
                
            # Move time forward
            current_start = klines[-1][6] + 1
            
            # Rate limit safety
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=['datetime', 'open', 'high', 'low', 'close'])
    df.set_index('datetime', inplace=True)
    
    print(f"[DATA] Saving {len(df)} rows to {file_path}...")
    df.to_csv(file_path)
    return df

def get_resampled_data(df_1m, timeframe, start_date_str, end_date_str):
    """
    Resamples 1m data to target timeframe and slices by date.
    """
    # 1. Slice by date first to speed up resampling
    mask = (df_1m.index >= start_date_str) & (df_1m.index <= end_date_str)
    df_subset = df_1m.loc[mask].copy()
    
    if df_subset.empty:
        return []

    # 2. Resample
    freq = get_pandas_freq(timeframe)
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }
    
    try:
        # dropna() removes empty bins (e.g. maintenance gaps)
        df_resampled = df_subset.resample(freq).agg(agg_dict).dropna()
    except ValueError as e:
        print(f"Resample Error with freq '{freq}': {e}")
        return []

    # 3. Convert to list of lists [O, H, L, C] for existing logic
    return df_resampled[['open', 'high', 'low', 'close']].values.tolist()

# ==========================================
# ALGORITHM CORE
# ==========================================

def deriveround(ohlc_data, a):
    """
    Standard rounding: round(change/a) * a
    """
    derived = []
    for i in range(1, len(ohlc_data)):
        curr = ohlc_data[i]
        prev = ohlc_data[i-1]
        d_row = []
        for j in range(4): 
            if prev[j] == 0: change = 0.0
            else: change = ((curr[j] - prev[j]) / prev[j]) * 100.0
            
            # Standard rounding logic
            rounded = round(change / a) * a
            d_row.append(rounded)
            
        derived.append(tuple(d_row))
    return derived

def split(derived_data, b):
    split_idx = int(len(derived_data) * (b / 100.0))
    return derived_data[:split_idx], derived_data[split_idx:]

def gettop(train_data, c, d):
    sequences = []
    if len(train_data) < d: return []
    for i in range(len(train_data) - d + 1):
        sequences.append(tuple(train_data[i : i+d]))
    if not sequences: return []
    
    counts = Counter(sequences)
    unique_seqs = sorted(list(counts.items()), key=lambda x: x[1], reverse=True)
    
    # c% of top sequences
    limit = max(1, int(len(unique_seqs) * (c / 100.0)))
    return [item[0] for item in unique_seqs[:limit]]

def is_similar(seq1, seq2, e):
    if len(seq1) != len(seq2): return False
    for k in range(len(seq1)):
        for val1, val2 in zip(seq1[k], seq2[k]):
            if val1 == 0:
                if val2 != 0: return False
                continue
            if (abs(val2 - val1) / abs(val1)) >= e: return False
    return True

def backtest(test_data, top_sequences, d, e):
    """
    Returns stats dict: {accuracy, trades, pnl}
    """
    valid = 0
    correct = 0
    pnl = 0.0
    
    begin_len = d - 1
    if len(test_data) <= begin_len:
         return {'accuracy': 0, 'trades': 0, 'pnl': 0}

    for i in range(len(test_data) - d + 1):
        window = test_data[i : i + begin_len]
        outcome = test_data[i + begin_len] # The d-th candle (contains outcome close)
        
        pred = None
        for seq in top_sequences:
            if is_similar(seq[:begin_len], window, e):
                pred = seq[begin_len][3] # Close change of the sequence
                break 
        
        if pred is not None and pred != 0:
            actual = outcome[3]
            if actual == 0: continue
            
            valid += 1
            is_cor = (pred > 0 and actual > 0) or (pred < 0 and actual < 0)
            if is_cor: correct += 1
            
            direction = 1 if pred > 0 else -1
            pnl += direction * actual

    accuracy = (correct / valid) if valid > 0 else 0
    return {'accuracy': accuracy, 'trades': valid, 'pnl': pnl}

# ==========================================
# GRID SEARCH MAIN
# ==========================================

def main():
    ensure_data_dir()
    
    # 1. Fetch/Load Master Data for all Symbols (1m Data)
    master_data = {}
    for sym in PARAMS['SYMBOL']:
        # Fetching strictly 2020-2026 as base
        master_data[sym] = fetch_1m_data(sym, 2020, 2026)

    # 2. Generate Parameter Combinations
    keys = [
        'SYMBOL', 'TIMEFRAME', 'START_YEAR', 'END_YEAR',
        'A_ROUND', 'C_TOP', 'D_LEN', 'E_SIM', 'B_SPLIT'
    ]
    
    combinations = list(itertools.product(*[PARAMS[k] for k in keys]))
    total_combos = len(combinations)
    
    print(f"\n[GRID SEARCH] Starting optimization on {total_combos} combinations.")
    print(f"[TARGET] Maximizing (Accuracy * Trades)")
    
    best_score = -999999.0
    best_params = None
    best_stats = None
    
    start_time = time.time()
    
    # 3. Loop
    for idx, combo in enumerate(combinations):
        p = dict(zip(keys, combo))
        
        # Validation: End Year > Start Year
        if int(p['END_YEAR']) <= int(p['START_YEAR']):
            continue
            
        # Prepare Data (Resample 1m to Target Timeframe)
        start_date = f"{p['START_YEAR']}-01-01"
        end_date = f"{p['END_YEAR']}-12-31"
        
        raw_data = get_resampled_data(
            master_data[p['SYMBOL']], 
            p['TIMEFRAME'], 
            start_date, 
            end_date
        )
        
        if len(raw_data) < 100: # Skip if insufficient data
            continue
            
        # Core Logic
        derived = deriveround(raw_data, p['A_ROUND'])
        train, test = split(derived, p['B_SPLIT'])
        
        top_seqs = gettop(train, p['C_TOP'], p['D_LEN'])
        
        if not top_seqs:
            continue
            
        stats = backtest(test, top_seqs, p['D_LEN'], p['E_SIM'])
        
        # Optimization Metric: Accuracy * Trades
        score = stats['accuracy'] * stats['trades']
        
        if score > best_score:
            best_score = score
            best_params = p
            best_stats = stats
            print(f"[*] NEW BEST: Score {score:.2f} | Acc: {stats['accuracy']:.2%} | Trades: {stats['trades']} | PnL: {stats['pnl']:.2f}")
            print(f"    Params: {p}")
            
        # Progress Log (every 1000 iters)
        if idx % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {idx}/{total_combos} ({idx/total_combos:.1%}) - {elapsed:.0f}s elapsed")

    print("\n==========================================")
    print("GRID SEARCH COMPLETE")
    print("==========================================")
    if best_params:
        print(f"BEST SCORE: {best_score:.4f}")
        print(f"ACCURACY:   {best_stats['accuracy']:.2%}")
        print(f"TRADES:     {best_stats['trades']}")
        print(f"TOTAL PNL:  {best_stats['pnl']:.2f}%")
        print("------------------------------------------")
        print("OPTIMAL PARAMETERS:")
        for k, v in best_params.items():
            print(f"{k}: {v}")
    else:
        print("No valid results found.")

if __name__ == "__main__":
    main()
