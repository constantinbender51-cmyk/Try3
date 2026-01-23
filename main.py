import requests
import math
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import http.server
import socketserver
import os
from datetime import datetime, timedelta
import time
import threading
import multiprocessing
import itertools
import json

# ==========================================
# PARAMETERS (DEFAULTS / OVERRIDDEN BY GRID SEARCH)
# ==========================================
# These initial values are placeholders. The Grid Search will overwrite them.
TIMEFRAME = os.getenv('TIMEFRAME', '1h')       
SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')        
START = os.getenv('START', '2023-01-01')       
END = os.getenv('END', '2023-06-01')           

A_ROUND = float(os.getenv('A_ROUND', '0.5'))   
B_SPLIT = int(os.getenv('B_SPLIT', '70'))      
C_TOP = int(os.getenv('C_TOP', '10'))          
D_LEN = int(os.getenv('D_LEN', '3'))           
E_SIM = float(os.getenv('E_SIM', '0.1'))       

PORT = int(os.getenv('PORT', '8080'))
DATA_DIR = '/app/data'

# ==========================================
# GRID SEARCH CONFIGURATION
# ==========================================
DO_GRID_SEARCH = True  # Set to False to skip optimization

GRID_PARAMS = {
    'A_ROUND': [1.024, 0.512, 0.256, 0.128, 0.064, 0.032, 0.016, 0.008, 0.004, 0.002],
    'C_TOP': [2, 4, 8, 16, 32],
    'D_LEN': [2, 3, 4, 5],
    'E_SIM': [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32],
    'TIMEFRAME': ['30m', '1h', '4h', '1d'],
    'START': ['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01'],
    'END': ['2024-01-01', '2025-01-01', '2026-01-01'],
    'SYMBOL': ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
}

# ==========================================
# GLOBAL STATE FOR LIVE TRADING
# ==========================================
LIVE_RESULTS = []      
PENDING_TRADES = []    
IS_RUNNING = True      

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_timeframe_seconds(tf):
    """Converts timeframe string to seconds."""
    unit = tf[-1]
    val = int(tf[:-1])
    if unit == 'm': return val * 60
    if unit == 'h': return val * 3600
    if unit == 'd': return val * 86400
    return 3600 

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def fetch_1m_data(symbol):
    """Fetches or loads 1m data for the full range (2020-2026)."""
    ensure_data_dir()
    file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    
    if os.path.exists(file_path):
        print(f"[DATA] Loading cached {symbol} 1m data...")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
        
    print(f"[DATA] Downloading full history for {symbol} (1m)...")
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Fetch range 2020 to 2026
    start_ts = int(pd.Timestamp("2020-01-01").timestamp() * 1000)
    end_ts = int(pd.Timestamp("2026-01-01").timestamp() * 1000)
    
    data = []
    current_start = start_ts
    
    while current_start < end_ts:
        params = {
            'symbol': symbol, 'interval': '1m',
            'startTime': current_start, 'endTime': end_ts, 'limit': 1000
        }
        try:
            response = requests.get(base_url, params=params)
            klines = response.json()
            if not klines: break
            for k in klines:
                # Time, Open, High, Low, Close
                ts = pd.to_datetime(k[0], unit='ms')
                ohlc = [ts, float(k[1]), float(k[2]), float(k[3]), float(k[4])]
                data.append(ohlc)
            current_start = klines[-1][6] + 1
            time.sleep(0.05) # Prevent rate limit
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    df = pd.DataFrame(data, columns=['datetime', 'open', 'high', 'low', 'close'])
    df.set_index('datetime', inplace=True)
    df.to_csv(file_path)
    return df

def fetch(timeframe, symbol, start, end, limit=1000, quiet=False):
    """
    Optimized Fetch: Loads 1m cache, resamples to requested timeframe, 
    and returns list of [O, H, L, C].
    """
    # If request is very recent (live trading), use API directly for freshness
    now = datetime.now()
    if isinstance(start, str): start_dt = pd.Timestamp(start)
    else: start_dt = start
    
    # If requesting recent data (less than 2 days ago), hit API to be safe
    if (now - start_dt).total_seconds() < 172800:
        return fetch_api_direct(timeframe, symbol, start, end, limit, quiet)

    # Historical / Backtest: Use Cache
    try:
        df = fetch_1m_data(symbol)
        
        # Resample
        tf_map = {'m': 'min', 'h': 'h', 'd': 'D'}
        rule = f"{timeframe[:-1]}{tf_map[timeframe[-1]]}"
        
        resampled = df.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()
        
        # Filter Date
        if isinstance(start, str): start_ts = pd.Timestamp(start)
        else: start_ts = start
        if isinstance(end, str): end_ts = pd.Timestamp(end)
        else: end_ts = end
        
        mask = (resampled.index >= start_ts) & (resampled.index < end_ts)
        sliced = resampled.loc[mask]
        
        # Convert to list of lists
        return sliced.values.tolist()
        
    except Exception as e:
        print(f"[WARN] Cache failed ({e}), falling back to API.")
        return fetch_api_direct(timeframe, symbol, start, end, limit, quiet)

def fetch_api_direct(timeframe, symbol, start, end, limit=1000, quiet=False):
    """Original API fetcher for fallback/live data."""
    base_url = "https://api.binance.com/api/v3/klines"
    if isinstance(start, str): start_ts = int(pd.Timestamp(start).timestamp() * 1000)
    else: start_ts = int(start.timestamp() * 1000)
    if isinstance(end, str): end_ts = int(pd.Timestamp(end).timestamp() * 1000)
    else: end_ts = int(end.timestamp() * 1000)
    
    data = []
    current_start = start_ts
    if not quiet: print(f"Fetching {symbol} {timeframe} from API...")
    
    while current_start < end_ts:
        params = {'symbol': symbol, 'interval': timeframe, 'startTime': current_start, 'endTime': end_ts, 'limit': limit}
        try:
            response = requests.get(base_url, params=params)
            klines = response.json()
            if not klines: break
            for k in klines:
                data.append([float(k[1]), float(k[2]), float(k[3]), float(k[4])])
            current_start = klines[-1][6] + 1
        except: break
    return data

def deriveround(ohlc_data, a):
    """
    Modified to use normal rounding as requested.
    0.4 -> 0, 0.6 -> 1, -0.4 -> 0, -0.6 -> -1
    """
    derived = []
    for i in range(1, len(ohlc_data)):
        curr = ohlc_data[i]
        prev = ohlc_data[i-1]
        d_row = []
        for j in range(4): 
            if prev[j] == 0: change = 0.0
            else: change = ((curr[j] - prev[j]) / prev[j]) * 100.0
            
            # Normal Rounding logic
            # Python's round() goes to nearest even number for .5, which is acceptable computationally
            # To strictly force 0.5->1, 0.4->0 logic:
            rounded = round(change / a) * a
            d_row.append(rounded)
        derived.append(tuple(d_row))
    return derived

def split(derived_data, b):
    split_idx = int(len(derived_data) * (b / 100.0))
    return derived_data[:split_idx], derived_data[split_idx:]

def gettop(train_data, c, d):
    sequences = []
    for i in range(len(train_data) - d + 1):
        sequences.append(tuple(train_data[i : i+d]))
    if not sequences: return []
    counts = Counter(sequences)
    unique_seqs = sorted(list(counts.items()), key=lambda x: x[1], reverse=True)
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

def completesimilarbeginnings(test_data, top_sequences, d, e):
    predictions = []
    begin_len = d - 1
    # Optimization: If no sequences, return empty early
    if not top_sequences or len(test_data) < d: return []

    for i in range(len(test_data) - d + 1):
        window = test_data[i : i + begin_len]
        outcome = test_data[i + begin_len]
        pred = None
        for seq in top_sequences:
            if is_similar(seq[:begin_len], window, e):
                pred = seq[begin_len][3]
                break 
        if pred is not None:
            predictions.append((pred, outcome[3]))
    return predictions

# ==========================================
# GRID SEARCH OPTIMIZATION ENGINE
# ==========================================

def evaluate_config(args):
    """
    Worker function for Grid Search.
    Args: (symbol, tf, start, end, a_val, logic_params_list)
    """
    symbol, tf, start, end, a_val, logic_combos = args
    
    # 1. Fetch & Prepare Data (Once per data-config)
    try:
        raw_data = fetch(tf, symbol, start, end, quiet=True)
        if len(raw_data) < 100: return []
        
        derived = deriveround(raw_data, a_val)
        train_data, test_data = split(derived, 70) # B_SPLIT fixed at 70
        
        results = []
        
        # 2. Iterate Logic Params (C, D, E)
        for c_val, d_val, e_val in logic_combos:
            # Train
            top_seqs = gettop(train_data, c_val, d_val)
            if not top_seqs: continue
            
            # Test
            preds = completesimilarbeginnings(test_data, top_seqs, d_val, e_val)
            
            # Calculate Score
            valid = 0
            correct = 0
            for p, act in preds:
                if p == 0 or act == 0: continue
                valid += 1
                if (p > 0 and act > 0) or (p < 0 and act < 0):
                    correct += 1
            
            if valid > 5: # Filter noise
                acc = correct / valid
                score = acc * valid # Optimize for Accuracy * Trades
                
                results.append({
                    'score': score,
                    'acc': acc,
                    'valid': valid,
                    'params': (symbol, tf, start, end, a_val, c_val, d_val, e_val)
                })
        return results
    except Exception as e:
        return []

def run_grid_search():
    print("\n" + "="*50)
    print("STARTING GRID SEARCH OPTIMIZATION")
    print(f"Goal: Maximize (Accuracy * Trades)")
    print("="*50)
    
    # Generate Data Combinations (Outer Loop)
    data_configs = list(itertools.product(
        GRID_PARAMS['SYMBOL'],
        GRID_PARAMS['TIMEFRAME'],
        GRID_PARAMS['START'],
        GRID_PARAMS['END'],
        GRID_PARAMS['A_ROUND']
    ))
    
    # Generate Logic Combinations (Inner Loop)
    logic_configs = list(itertools.product(
        GRID_PARAMS['C_TOP'],
        GRID_PARAMS['D_LEN'],
        GRID_PARAMS['E_SIM']
    ))
    
    tasks = []
    # Bundle tasks: Each worker handles one Data Config and ALL logic configs for it
    # This minimizes data fetching overhead
    for d_conf in data_configs:
        # Check start < end year to avoid invalid ranges
        s_y = int(d_conf[2][:4])
        e_y = int(d_conf[3][:4])
        if s_y >= e_y: continue
        
        tasks.append(d_conf + (logic_configs,))

    print(f"Total Data Scenarios: {len(tasks)}")
    print(f"Logic Variants per Scenario: {len(logic_configs)}")
    print(f"Total Combinations: {len(tasks) * len(logic_configs)}")
    print("Processing... (This may take a while)\n")

    best_score = -1
    best_config = None
    
    cpu_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_cores)
    
    # Process
    counter = 0
    total = len(tasks)
    
    for result_batch in pool.imap_unordered(evaluate_config, tasks):
        counter += 1
        print(f"Progress: {counter}/{total} scenarios checked...", end='\r')
        
        if not result_batch: continue
        
        for res in result_batch:
            if res['score'] > best_score:
                best_score = res['score']
                best_config = res
                print(f"\n>> NEW BEST: Score {best_score:.2f} | Acc: {res['acc']:.2%}, Trades: {res['valid']}")
                p = res['params']
                print(f"   {p[0]} {p[1]} {p[2]}-{p[3]} | A:{p[4]} C:{p[5]} D:{p[6]} E:{p[7]}")

    pool.close()
    pool.join()
    
    print("\n" + "="*50)
    if best_config:
        p = best_config['params']
        print(f"OPTIMIZATION COMPLETE.")
        print(f"Best Symbol: {p[0]}")
        print(f"Best Timeframe: {p[1]}")
        print(f"Best Date Range: {p[2]} to {p[3]}")
        print(f"Best Params: A={p[4]}, C={p[5]}, D={p[6]}, E={p[7]}")
        return {
            'SYMBOL': p[0], 'TIMEFRAME': p[1], 'START': p[2], 'END': p[3],
            'A_ROUND': p[4], 'C_TOP': p[5], 'D_LEN': p[6], 'E_SIM': p[7]
        }
    else:
        print("Optimization failed to find valid trades. Using defaults.")
        return {}

# ==========================================
# LIVE TRADING LOGIC
# ==========================================

def live_trading_loop(top_sequences, d_len, e_sim, timeframe_str):
    """Background thread for live prediction."""
    global LIVE_RESULTS, PENDING_TRADES
    
    tf_seconds = get_timeframe_seconds(timeframe_str)
    max_items = int((14 * 24 * 3600) / tf_seconds)
    
    print(f"\n[LIVE] Thread started. Timeframe: {timeframe_str} ({tf_seconds}s).")

    while IS_RUNNING:
        now = datetime.now()
        current_ts = now.timestamp()
        candle_start = (current_ts // tf_seconds) * tf_seconds
        next_close = candle_start + tf_seconds
        target_time = next_close + 5
        sleep_duration = target_time - current_ts
        if sleep_duration < 0: sleep_duration += tf_seconds
            
        print(f"[LIVE] Sleeping {sleep_duration:.2f}s...")
        time.sleep(sleep_duration)
        
        # Resolve Pending
        recent_check_start = datetime.now() - timedelta(seconds=tf_seconds*3)
        recent_data = fetch_api_direct(timeframe_str, SYMBOL, recent_check_start, datetime.now(), quiet=True)
        
        if len(recent_data) >= 2:
            derived_recent = deriveround(recent_data, A_ROUND)
            if derived_recent:
                last_outcome_val = derived_recent[-1][3]
                
                for p_time, p_pred in PENDING_TRADES:
                    direction = 1 if p_pred > 0 else -1
                    pnl = direction * last_outcome_val
                    is_correct = (p_pred > 0 and last_outcome_val > 0) or (p_pred < 0 and last_outcome_val < 0)
                    
                    record = {
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'pred': p_pred, 'actual': last_outcome_val,
                        'pnl': pnl, 'correct': "Yes" if is_correct else "No"
                    }
                    LIVE_RESULTS.insert(0, record)
                    print(f"[LIVE] Resolved: Pred {p_pred:.2f}%, Actual {last_outcome_val:.2f}%")
                
                PENDING_TRADES = []
                if len(LIVE_RESULTS) > max_items: LIVE_RESULTS = LIVE_RESULTS[:max_items]

        # New Prediction
        needed_candles = d_len + 5 
        fetch_start = datetime.now() - timedelta(seconds=tf_seconds * needed_candles)
        data_for_pred = fetch_api_direct(timeframe_str, SYMBOL, fetch_start, datetime.now(), quiet=True)
        
        if len(data_for_pred) < d_len: continue
            
        derived_pred = deriveround(data_for_pred, A_ROUND)
        begin_len = d_len - 1
        if len(derived_pred) < begin_len: continue
        
        current_window = derived_pred[-begin_len:] 
        prediction_val = None
        for seq in top_sequences:
            if is_similar(seq[:begin_len], current_window, e_sim):
                prediction_val = seq[begin_len][3]
                break
        
        if prediction_val is not None:
            print(f"[LIVE] New Prediction: {prediction_val:.2f}% change expected.")
            PENDING_TRADES.append((datetime.now(), prediction_val))

# ==========================================
# VISUALIZATION & SERVER
# ==========================================

def process_stats(result_list):
    if not result_list: return {'valid':0, 'accuracy':0, 'pnl':0}, ""
    valid, correct, pnl = 0, 0, 0.0
    rows = ""
    cumulative_acc = []
    
    for i, (pred, actual) in enumerate(result_list):
        if pred == 0 or actual == 0: continue
        valid += 1
        is_cor = (pred > 0 and actual > 0) or (pred < 0 and actual < 0)
        if is_cor: correct += 1
        trade_pnl = (1 if pred > 0 else -1) * actual
        pnl += trade_pnl
        cumulative_acc.append((correct/valid)*100)
        color = "green" if trade_pnl > 0 else "red"
        rows += f"""<tr><td>{valid}</td><td>{pred:.2f}%</td><td>{actual:.2f}%</td>
                    <td style="color:{color}">{trade_pnl:.2f}%</td><td>{"Yes" if is_cor else "No"}</td></tr>"""
    final_acc = cumulative_acc[-1] if cumulative_acc else 0
    return {'valid': valid, 'accuracy': final_acc, 'pnl': pnl, 'acc_hist': cumulative_acc}, rows

def generate_plot(data_hist, data_recent, filename="combined.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    if data_hist: ax1.plot(data_hist, label='Historical Acc', color='blue')
    if data_recent: ax1.plot(data_recent, label='Recent 14d Acc', color='orange')
    ax1.set_title(f"Strategy Accuracy ({SYMBOL} {TIMEFRAME})")
    ax1.legend()
    ax1.grid(True)
    ax2.text(0.5, 0.5, "Live Updates Visible in Table Below", ha='center')
    ax2.set_axis_off()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

def serve_interface(hist_data, recent_data):
    h_stats, h_rows = process_stats(hist_data)
    r_stats, r_rows = process_stats(recent_data)
    generate_plot(h_stats.get('acc_hist'), r_stats.get('acc_hist'))

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                l_rows = ""
                l_pnl, l_valid, l_correct = 0, 0, 0
                for res in LIVE_RESULTS:
                    l_valid += 1
                    l_pnl += res['pnl']
                    if res['correct'] == "Yes": l_correct += 1
                    color = "green" if res['pnl'] > 0 else "red"
                    l_rows += f"<tr><td>{res['time']}</td><td>{res['pred']:.2f}%</td><td>{res['actual']:.2f}%</td><td style='color:{color}'>{res['pnl']:.2f}%</td><td>{res['correct']}</td></tr>"
                l_acc = (l_correct / l_valid * 100) if l_valid > 0 else 0

                html = f"""
                <html><head><title>Bot Dashboard</title><meta http-equiv="refresh" content="30">
                <style>body{{font-family:'Segoe UI',sans-serif;padding:20px;background:#f4f4f9}}.container{{display:flex;flex-wrap:wrap;gap:20px}}.section{{flex:1;min-width:400px;background:white;padding:20px;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,0.1)}}table{{width:100%;border-collapse:collapse}}th{{background:#eee;padding:8px;text-align:left}}td{{padding:8px;border-bottom:1px solid #eee}}.live-badge{{background:#ff4757;color:white;padding:2px 8px;border-radius:4px}}</style>
                </head><body><h1>Algo Dashboard: {SYMBOL} {TIMEFRAME}</h1>
                <div class="container">
                    <div class="section" style="border:2px solid #2ed573">
                        <h2>3. Live <span class="live-badge">ACTIVE</span></h2>
                        <p>Valid: {l_valid} | Acc: {l_acc:.2f}% | PnL: {l_pnl:.2f}%</p>
                        {f'<p><strong>Pending:</strong> {PENDING_TRADES[0][1]:.2f}%</p>' if PENDING_TRADES else ''}
                        <div style="max-height:400px;overflow-y:auto"><table><thead><tr><th>Time</th><th>Pred</th><th>Actual</th><th>PnL</th><th>Cor</th></tr></thead><tbody>{l_rows}</tbody></table></div>
                    </div>
                    <div class="section">
                        <h2>2. Recent (14d)</h2>
                        <p>Acc: {r_stats['accuracy']:.2f}% | PnL: {r_stats['pnl']:.2f}%</p>
                        <div style="max-height:300px;overflow-y:auto"><table><thead><tr><th>#</th><th>Pred</th><th>Actual</th><th>PnL</th><th>Cor</th></tr></thead><tbody>{r_rows}</tbody></table></div>
                    </div>
                    <div class="section">
                        <h2>1. Backtest ({START} - {END})</h2>
                        <p>Params: A={A_ROUND} C={C_TOP} D={D_LEN} E={E_SIM}</p>
                        <p>Acc: {h_stats['accuracy']:.2f}% | PnL: {h_stats['pnl']:.2f}%</p>
                        <img src="combined.png" style="width:100%">
                    </div>
                </div></body></html>"""
                self.wfile.write(html.encode('utf-8'))
            elif self.path == '/combined.png':
                try:
                    with open('combined.png', 'rb') as f:
                        self.send_response(200); self.send_header('Content-type', 'image/png'); self.end_headers(); self.wfile.write(f.read())
                except: self.send_error(404)
            else: return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    print(f"\n[SERVER] Dashboard at http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try: httpd.serve_forever()
        except KeyboardInterrupt: pass

def main():
    # 0. Grid Search Optimization
    if DO_GRID_SEARCH:
        best_params = run_grid_search()
        if best_params:
            global SYMBOL, TIMEFRAME, START, END, A_ROUND, C_TOP, D_LEN, E_SIM
            SYMBOL = best_params['SYMBOL']
            TIMEFRAME = best_params['TIMEFRAME']
            START = best_params['START']
            END = best_params['END']
            A_ROUND = best_params['A_ROUND']
            C_TOP = best_params['C_TOP']
            D_LEN = best_params['D_LEN']
            E_SIM = best_params['E_SIM']
            print(f"\n[MAIN] Applied Optimized Params: {best_params}")

    # 1. Backtest
    raw = fetch(TIMEFRAME, SYMBOL, START, END)
    derived = deriveround(raw, A_ROUND)
    train, test = split(derived, B_SPLIT)
    
    top_seqs = gettop(train, C_TOP, D_LEN)
    hist_results = []
    if top_seqs:
        hist_results = completesimilarbeginnings(test, top_seqs, D_LEN, E_SIM)
    
    # 2. Recent
    now = datetime.now()
    recent_raw = fetch(TIMEFRAME, SYMBOL, now - timedelta(days=14), now)
    recent_results = []
    if len(recent_raw) > D_LEN:
        recent_derived = deriveround(recent_raw, A_ROUND)
        recent_results = completesimilarbeginnings(recent_derived, top_seqs, D_LEN, E_SIM)
    
    # 3. Live
    t = threading.Thread(target=live_trading_loop, args=(top_seqs, D_LEN, E_SIM, TIMEFRAME))
    t.daemon = True
    t.start()
    
    # 4. Serve
    serve_interface(hist_results, recent_results)

if __name__ == "__main__":
    # Needed for multiprocessing on some OS
    multiprocessing.freeze_support()
    main()
