import requests
import math
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import http.server
import socketserver
import os
from datetime import datetime, timedelta
import time
import threading

# ==========================================
# PARAMETERS
# ==========================================
TIMEFRAME = os.getenv('TIMEFRAME', '1h')       # e.g., '1m', '5m', '1h', '1d'
SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')        # Binance symbol
START = os.getenv('START', '2023-01-01')       # Training/Backtest Start
END = os.getenv('END', '2023-06-01')           # Training/Backtest End

A_ROUND = float(os.getenv('A_ROUND', '0.5'))   # a%: Rounding step (floor)
B_SPLIT = int(os.getenv('B_SPLIT', '70'))      # b%: Percentage of data for Split 1 (Training)
C_TOP = int(os.getenv('C_TOP', '10'))          # c%: Percentage of top frequent sequences to keep
D_LEN = int(os.getenv('D_LEN', '3'))           # d: Total length of candle sequence
E_SIM = float(os.getenv('E_SIM', '0.1'))       # e%: Similarity threshold (0.1 = 10% difference)

PORT = int(os.getenv('PORT', '8080'))

# ==========================================
# GLOBAL STATE FOR LIVE TRADING
# ==========================================
LIVE_RESULTS = []      # Stores completed live trades
PENDING_TRADES = []    # Stores active predictions waiting for outcome
IS_RUNNING = True      # Thread control

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
    return 3600 # default 1h

def fetch(timeframe, symbol, start, end, limit=1000, quiet=False):
    """Fetches OHLC data from Binance."""
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Handle string or datetime
    if isinstance(start, str): start_ts = int(pd.Timestamp(start).timestamp() * 1000)
    else: start_ts = int(start.timestamp() * 1000)
    
    if isinstance(end, str): end_ts = int(pd.Timestamp(end).timestamp() * 1000)
    else: end_ts = int(end.timestamp() * 1000)
    
    data = []
    current_start = start_ts
    
    if not quiet: print(f"Fetching {symbol} {timeframe}...")
    
    while current_start < end_ts:
        params = {
            'symbol': symbol, 'interval': timeframe,
            'startTime': current_start, 'endTime': end_ts, 'limit': limit
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            klines = response.json()
            if not klines: break
            for k in klines:
                # O, H, L, C
                ohlc = [float(k[1]), float(k[2]), float(k[3]), float(k[4])]
                data.append(ohlc)
            current_start = klines[-1][6] + 1
        except Exception as e:
            print(f"Error: {e}")
            break
            
    return data

def deriveround(ohlc_data, a):
    derived = []
    for i in range(1, len(ohlc_data)):
        curr = ohlc_data[i]
        prev = ohlc_data[i-1]
        d_row = []
        for j in range(4): 
            if prev[j] == 0: change = 0.0
            else: change = ((curr[j] - prev[j]) / prev[j]) * 100.0
            rounded = math.floor(change / a) * a
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
# LIVE TRADING LOGIC
# ==========================================

def live_trading_loop(top_sequences, d_len, e_sim, timeframe_str):
    """Background thread for live prediction."""
    global LIVE_RESULTS, PENDING_TRADES
    
    tf_seconds = get_timeframe_seconds(timeframe_str)
    
    # Calculate max live results items (2 weeks)
    # 2 weeks * 24h * 3600s / tf_seconds
    max_items = int((14 * 24 * 3600) / tf_seconds)
    
    print(f"\n[LIVE] Thread started. Timeframe: {timeframe_str} ({tf_seconds}s).")
    print(f"[LIVE] Holding max {max_items} records.")

    while IS_RUNNING:
        now = datetime.now()
        
        # 1. Calculate time to next sync (Candle Close + 5s)
        # Determine current candle start
        current_ts = now.timestamp()
        # Floor division to get start of current candle
        candle_start = (current_ts // tf_seconds) * tf_seconds
        next_close = candle_start + tf_seconds
        
        target_time = next_close + 5 # 5 seconds after close
        sleep_duration = target_time - current_ts
        
        # If we are already past the +5s mark for this candle (e.g. startup), wait for next
        if sleep_duration < 0:
            sleep_duration += tf_seconds
            
        print(f"[LIVE] Sleeping {sleep_duration:.2f}s until next candle check...")
        time.sleep(sleep_duration)
        
        print(f"[LIVE] Waking up at {datetime.now().strftime('%H:%M:%S')}. Processing...")

        # ---------------------------------
        # A. Resolve Pending Predictions
        # ---------------------------------
        # Fetch just enough recent data to resolve the pending trade
        # We need the candle that JUST closed.
        recent_check_start = datetime.now() - timedelta(seconds=tf_seconds*3)
        recent_data = fetch(timeframe_str, SYMBOL, recent_check_start, datetime.now(), quiet=True)
        
        if len(recent_data) >= 2:
            # The last closed candle is likely index -1 (since we are 5s into new one)
            # Derive it
            derived_recent = deriveround(recent_data, A_ROUND)
            if derived_recent:
                last_outcome_val = derived_recent[-1][3] # Close change
                
                # Check pending list
                new_pending = []
                for p_time, p_pred in PENDING_TRADES:
                    # p_time is the timestamp when the prediction *should* be resolved
                    # Roughly: prediction made at T, outcome valid at T + tf_seconds
                    
                    # Logic: The PENDING trade was made 1 cycle ago. 
                    # We simply clear the queue because we run exactly on cycle.
                    
                    direction = 1 if p_pred > 0 else -1
                    pnl = direction * last_outcome_val
                    is_correct = (p_pred > 0 and last_outcome_val > 0) or (p_pred < 0 and last_outcome_val < 0)
                    
                    record = {
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'pred': p_pred,
                        'actual': last_outcome_val,
                        'pnl': pnl,
                        'correct': "Yes" if is_correct else "No"
                    }
                    
                    LIVE_RESULTS.insert(0, record) # Prepend newest
                    print(f"[LIVE] Resolved trade: Pred {p_pred:.2f}%, Actual {last_outcome_val:.2f}%")
                
                PENDING_TRADES = [] # Cleared handled trades
                
                # Prune list
                if len(LIVE_RESULTS) > max_items:
                    LIVE_RESULTS = LIVE_RESULTS[:max_items]

        # ---------------------------------
        # B. Make NEW Prediction
        # ---------------------------------
        # We need a sequence of length d-1 to match the "beginning"
        # The fetch above might be enough, but let's be safe and fetch needed amount
        needed_candles = d_len + 5 
        fetch_start = datetime.now() - timedelta(seconds=tf_seconds * needed_candles)
        
        data_for_pred = fetch(timeframe_str, SYMBOL, fetch_start, datetime.now(), quiet=True)
        
        if len(data_for_pred) < d_len:
            print("[LIVE] Not enough data to predict.")
            continue
            
        derived_pred = deriveround(data_for_pred, A_ROUND)
        
        # The "window" is the last (d-1) derived candles
        begin_len = d_len - 1
        if len(derived_pred) < begin_len: continue
        
        current_window = derived_pred[-begin_len:] 
        
        prediction_val = None
        for seq in top_sequences:
            top_beginning = seq[:begin_len]
            if is_similar(top_beginning, current_window, e_sim):
                prediction_val = seq[begin_len][3]
                break
        
        if prediction_val is not None:
            print(f"[LIVE] New Prediction: {prediction_val:.2f}% change expected.")
            PENDING_TRADES.append((datetime.now(), prediction_val))
        else:
            print("[LIVE] No pattern match found.")

# ==========================================
# VISUALIZATION & SERVER
# ==========================================

def process_stats(result_list):
    """Calculates stats for a simple list of (pred, actual) tuples."""
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
    
    ax1.set_title("Strategy Accuracy Over Time")
    ax1.legend()
    ax1.grid(True)
    
    # Just a placeholder for PnL or Counts on ax2
    ax2.text(0.5, 0.5, "Live Updates Visible in Table Below", ha='center')
    ax2.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

def serve_interface(hist_data, recent_data):
    # Process static data once
    h_stats, h_rows = process_stats(hist_data)
    r_stats, r_rows = process_stats(recent_data)
    
    generate_plot(h_stats.get('acc_hist'), r_stats.get('acc_hist'))

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                # Generate Live Rows dynamically
                l_rows = ""
                l_pnl = 0
                l_valid = 0
                l_correct = 0
                
                for res in LIVE_RESULTS:
                    l_valid += 1
                    l_pnl += res['pnl']
                    if res['correct'] == "Yes": l_correct += 1
                    
                    color = "green" if res['pnl'] > 0 else "red"
                    l_rows += f"""
                    <tr>
                        <td>{res['time']}</td>
                        <td>{res['pred']:.2f}%</td>
                        <td>{res['actual']:.2f}%</td>
                        <td style="color:{color}; font-weight:bold;">{res['pnl']:.2f}%</td>
                        <td>{res['correct']}</td>
                    </tr>
                    """
                
                l_acc = (l_correct / l_valid * 100) if l_valid > 0 else 0

                html = f"""
                <html>
                <head>
                    <title>Live Trading Bot</title>
                    <meta http-equiv="refresh" content="30"> <style>
                        body {{ font-family: 'Segoe UI', sans-serif; padding: 20px; background: #f4f4f9; }}
                        .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                        .section {{ flex: 1; min-width: 400px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                        h1 {{ text-align: center; color: #333; }}
                        h2 {{ border-bottom: 2px solid #ddd; padding-bottom: 10px; color: #555; }}
                        table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
                        th {{ background: #eee; padding: 8px; text-align: left; }}
                        td {{ padding: 8px; border-bottom: 1px solid #eee; }}
                        .metric {{ font-size: 1.1em; margin: 5px 0; }}
                        .live-badge {{ background: #ff4757; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.6em; vertical-align: middle; }}
                        .pending {{ background: #eccc68; padding: 10px; border-radius: 5px; margin-bottom: 15px; }}
                    </style>
                </head>
                <body>
                    <h1>Algo Trading Dashboard</h1>
                    
                    <div class="container">
                        <div class="section" style="border: 2px solid #2ed573;">
                            <h2>3. Live Forward Test <span class="live-badge">ACTIVE</span></h2>
                            <div class="metric"><strong>Status:</strong> Waiting for next candle...</div>
                            <div class="metric"><strong>Valid Trades:</strong> {l_valid}</div>
                            <div class="metric"><strong>Accuracy:</strong> {l_acc:.2f}%</div>
                            <div class="metric"><strong>Total PnL:</strong> {l_pnl:.2f}%</div>
                            
                            {f'<div class="pending"><strong>Pending Prediction:</strong> {PENDING_TRADES[0][1]:.2f}% (Waiting for close)</div>' if PENDING_TRADES else ''}
                            
                            <h3>Live Log (Max 2 Weeks)</h3>
                            <div style="max-height: 400px; overflow-y: auto;">
                                <table>
                                    <thead><tr><th>Time</th><th>Pred</th><th>Actual</th><th>PnL</th><th>Correct</th></tr></thead>
                                    <tbody>{l_rows}</tbody>
                                </table>
                            </div>
                        </div>

                        <div class="section">
                            <h2>2. Recent Performance (14 Days)</h2>
                            <div class="metric"><strong>Accuracy:</strong> {r_stats['accuracy']:.2f}% | <strong>PnL:</strong> {r_stats['pnl']:.2f}%</div>
                            <div style="max-height: 300px; overflow-y: auto;">
                                <table>
                                    <thead><tr><th>#</th><th>Pred</th><th>Actual</th><th>PnL</th><th>Correct</th></tr></thead>
                                    <tbody>{r_rows}</tbody>
                                </table>
                            </div>
                        </div>

                        <div class="section">
                            <h2>1. Backtest ({START} - {END})</h2>
                            <div class="metric"><strong>Accuracy:</strong> {h_stats['accuracy']:.2f}% | <strong>PnL:</strong> {h_stats['pnl']:.2f}%</div>
                            <img src="combined.png" style="width:100%; margin-top:10px;">
                        </div>
                    </div>
                </body>
                </html>
                """
                self.wfile.write(html.encode('utf-8'))
            elif self.path == '/combined.png':
                try:
                    with open('combined.png', 'rb') as f:
                        self.send_response(200)
                        self.send_header('Content-type', 'image/png')
                        self.end_headers()
                        self.wfile.write(f.read())
                except:
                    self.send_error(404)
            else:
                return http.server.SimpleHTTPRequestHandler.do_GET(self)

    print(f"\n[SERVER] Dashboard at http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass

def main():
    # 1. Backtest
    raw = fetch(TIMEFRAME, SYMBOL, START, END)
    derived = deriveround(raw, A_ROUND)
    train, test = split(derived, B_SPLIT)
    
    # Train Model
    top_seqs = gettop(train, C_TOP, D_LEN)
    if not top_seqs: return
    
    # Validate Historical
    hist_results = completesimilarbeginnings(test, top_seqs, D_LEN, E_SIM)
    
    # 2. Recent 14 Days
    now = datetime.now()
    recent_raw = fetch(TIMEFRAME, SYMBOL, now - timedelta(days=14), now)
    recent_results = []
    if len(recent_raw) > D_LEN:
        recent_derived = deriveround(recent_raw, A_ROUND)
        recent_results = completesimilarbeginnings(recent_derived, top_seqs, D_LEN, E_SIM)
    
    # 3. Start Live Thread
    t = threading.Thread(target=live_trading_loop, args=(top_seqs, D_LEN, E_SIM, TIMEFRAME))
    t.daemon = True # Kills thread when main program exits
    t.start()
    
    # 4. Serve
    serve_interface(hist_results, recent_results)

if __name__ == "__main__":
    main()
  
