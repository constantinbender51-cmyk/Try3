import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import io
import base64
import http.server
import socketserver
import threading
import time
from datetime import datetime, timedelta

# --- Configuration ---
TIMEFRAME = '1h'
SYMBOL = 'BTC/USDT'
START = '2024-01-01 00:00:00'
END = '2024-06-01 00:00:00'

# Parameters
A = 0.0          # (Unused as rounding is removed, kept for signature consistency if needed)
B = 0.7          # Split % (70% training, 30% testing)
C = 0.1          # Top % most frequent (densest) sequences to keep
D = 4            # Sequence length (candles)
E = 0.001        # Similarity threshold (0.1%)

# Global State
results_html = "<h1>Initializing...</h1>"
live_outcomes = []  # Stores live prediction results
model_sequences = None # Stores the trained patterns
data_global = None     # Stores the derived dataframe
recent_prediction = None

# --- Functions ---

def fetch(timeframe, symbol, start_str, end_str):
    """Fetches OHLCV data from Binance."""
    print(f"Fetching {symbol} {timeframe} from {start_str} to {end_str}...")
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(start_str)
    end_ts = exchange.parse8601(end_str)
    
    ohlc = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not candles:
                break
            
            # Filter out candles beyond end_ts
            candles = [c for c in candles if c[0] < end_ts]
            if not candles:
                break

            ohlc += candles
            current_ts = candles[-1][0] + 1
            time.sleep(0.1) # Rate limit respect
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def deriveround(df, a=None):
    """
    Applies candle(i)-candle(i-1)/candle(i-1) logic.
    Specific user request: Open relative to prev Open, Close to prev Close, etc.
    Rounding is REMOVED as requested.
    """
    df = df.copy()
    # Calculate returns relative to previous candle of the SAME type
    cols = ['open', 'high', 'low', 'close']
    for col in cols:
        df[f'{col}_ret'] = df[col].pct_change()
    
    df.dropna(inplace=True)
    return df

def split(df, b):
    """Splits data into b% (Split 1) and remaining (Split 2)."""
    split_idx = int(len(df) * b)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def gettop(df_split1, c, d):
    """
    Finds the top c% most frequent (densest) sequences in Split 1 using E similarity.
    Treats sequences of length d as vectors.
    """
    print("Training model (finding dense patterns)...")
    
    # Extract relevant columns
    data_cols = ['open_ret', 'high_ret', 'low_ret', 'close_ret']
    data_values = df_split1[data_cols].values
    
    # Create sliding windows of length D
    # Shape: (N_windows, D, 4)
    windows = np.lib.stride_tricks.sliding_window_view(data_values, window_shape=(d, 4))
    
    # Flatten the inner dimensions for distance calculation
    # Shape: (N_windows, D*4)
    N = windows.shape[0]
    flat_windows = windows.reshape(N, -1)
    
    # To be computationally efficient, we cannot do N*N comparisons if N is huge.
    # We will sample if N > 10000, otherwise exact.
    # Here we calculate how many neighbors each window has within E distance.
    
    densities = np.zeros(N, dtype=int)
    
    # Optimization: Chunked broadcasting to avoid memory overflow
    chunk_size = 1000
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        # Compare chunk (chunk_size, Features) vs All (N, Features)
        # Using Manhattan distance (sum of absolute diffs) or Max diff? 
        # User said: (open2-open1)/open1 < e%. This implies checking every component.
        # We check if absolute difference of ALL components is < E.
        
        batch = flat_windows[i:end]
        # Diff shape: (chunk_size, N, Features) -> Very large.
        # Alternative: Loop through batch
        for j in range(len(batch)):
            # Check deviation
            diff = np.abs(flat_windows - batch[j])
            # Match if ALL points in the sequence are within E
            matches = np.all(diff < E, axis=1)
            densities[i+j] = np.sum(matches)
            
    # Select top c%
    top_n = int(N * c)
    if top_n == 0: top_n = 1
    
    # Get indices of top densities
    top_indices = np.argsort(densities)[-top_n:]
    
    # Return the actual sequences (unflattened)
    return windows[top_indices]

def completesimilarbeginnings(df_target, model_patterns, e, d):
    """
    Predicts on df_target using the top patterns.
    Matches first d-1 candles, predicts d-th.
    """
    print("Running predictions...")
    data_cols = ['open_ret', 'high_ret', 'low_ret', 'close_ret']
    target_values = df_target[data_cols].values
    timestamps = df_target['timestamp'].values
    actual_returns = df_target['close_ret'].values # Outcome
    
    # Create windows from target data
    # We need windows of length D to verify accuracy (historical testing)
    # Shape: (M, D, 4)
    target_windows = np.lib.stride_tricks.sliding_window_view(target_values, window_shape=(d, 4))
    target_ts = np.lib.stride_tricks.sliding_window_view(timestamps, window_shape=(d,))
    
    predictions = []
    
    # Model Patterns Shape: (K, D, 4)
    # We split model patterns into:
    # Context: First D-1 candles
    # Outcome: Last candle (index D-1)
    model_context = model_patterns[:, :d-1, :] # (K, D-1, 4)
    model_outcome = model_patterns[:, -1, :]   # (K, 4) -> We specifically care about Close vs Open or Close direction
    
    # Flatten contexts for comparison
    model_context_flat = model_context.reshape(model_context.shape[0], -1)
    
    # Iterate through target windows
    for i in range(len(target_windows)):
        current_window = target_windows[i] # (D, 4)
        current_context = current_window[:d-1, :] # (D-1, 4)
        current_context_flat = current_context.reshape(-1)
        
        # Find matches in model patterns
        # Efficient broadcasting
        diff = np.abs(model_context_flat - current_context_flat)
        matches_idx = np.where(np.all(diff < e, axis=1))[0]
        
        if len(matches_idx) > 0:
            # We have matches. Aggregate predictions.
            # Strategy: Average outcome direction of matched patterns
            matched_outcomes = model_outcome[matches_idx] # (Num_Matches, 4)
            
            # Predict Direction: Close_ret of the D-th candle
            # Or specifically: Is the D-th candle Green (Close > Open) or Up (Close > PrevClose)?
            # User prompt implies "Directional Prediction". Let's use Close Return sign.
            
            # Vote: Sum of close returns of matches
            avg_return = np.mean(matched_outcomes[:, 3]) # Column 3 is close_ret
            
            predicted_dir = 1 if avg_return > 0 else -1
            if avg_return == 0: predicted_dir = 0
            
            # Actual outcome
            actual_ret = current_window[-1, 3] # Close ret of D-th candle
            actual_dir = 1 if actual_ret > 0 else -1
            if actual_ret == 0: actual_dir = 0
            
            # Timestamp of the predicted candle (the end of the sequence)
            ts = pd.to_datetime(target_ts[i, -1])
            
            predictions.append({
                'timestamp': ts,
                'predicted_dir': predicted_dir,
                'actual_ret': actual_ret, # Actual return of the candle
                'actual_dir': actual_dir,
                'is_correct': (predicted_dir == actual_dir) and (predicted_dir != 0)
            })
            
    return pd.DataFrame(predictions)

def printaccuracy(predictions_df):
    """Calculates metrics and prepares plots."""
    if predictions_df.empty:
        return "<h3>No predictions made (adjust E or C)</h3>", None

    # Filter flat predictions
    active = predictions_df[predictions_df['predicted_dir'] != 0].copy()
    
    if active.empty:
        return "<h3>No non-flat predictions</h3>", None

    total = len(active)
    correct = active['is_correct'].sum()
    accuracy = (correct / total) * 100
    
    # Calculate PnL: Predicted Direction * Actual Return
    active['pnl'] = active['predicted_dir'] * active['actual_ret']
    active['cum_pnl'] = active['pnl'].cumsum()
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(active['timestamp'], active['cum_pnl'], label='Cumulative Strategy Return')
    plt.title(f'Strategy Performance (Acc: {accuracy:.2f}%)')
    plt.grid(True)
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # HTML Table
    table_html = """
    <table border="1">
    <tr><th>Date</th><th>Pred</th><th>Actual Ret</th><th>Outcome</th><th>PnL</th></tr>
    """
    # Show last 50
    for _, row in active.tail(50).iterrows():
        p_str = "UP" if row['predicted_dir'] > 0 else "DOWN"
        color = "green" if row['is_correct'] else "red"
        table_html += f"<tr><td>{row['timestamp']}</td><td>{p_str}</td><td>{row['actual_ret']:.4f}</td><td style='color:{color}'>{row['is_correct']}</td><td>{row['pnl']:.4f}</td></tr>"
    table_html += "</table>"
    
    return f"<h3>Accuracy: {accuracy:.2f}% ({correct}/{total})</h3><img src='data:image/png;base64,{plot_url}'/><br>{table_html}", active

def predict_on_recent(model_patterns, df_recent, e, d):
    """Specific function for recent 14 days and live prediction."""
    # Similar logic to completesimilarbeginnings but optimized for live/recent
    # We only need the prediction for the existing data
    preds = completesimilarbeginnings(df_recent, model_patterns, e, d)
    return preds

# --- Live Loop Logic ---

def get_seconds_to_sleep(timeframe):
    now = datetime.utcnow()
    # Simple parser for m/h/d
    unit = timeframe[-1]
    val = int(timeframe[:-1])
    
    if unit == 'm':
        delta = timedelta(minutes=val)
    elif unit == 'h':
        delta = timedelta(hours=val)
    elif unit == 'd':
        delta = timedelta(days=val)
    else:
        delta = timedelta(hours=1) # Fallback

    # Calculate next close
    # Assuming aligned to hour/minute
    # This is a simplification; production would need robust aligning
    next_interval = now + delta # Rough approx, alignment logic below
    
    if unit == 'h':
        next_hour = now.replace(minute=0, second=0, microsecond=0) + delta
        while next_hour < now: next_hour += delta
        target = next_hour
    elif unit == 'm':
        next_min = now.replace(second=0, microsecond=0) + timedelta(minutes=val - (now.minute % val))
        if next_min <= now: next_min += timedelta(minutes=val)
        target = next_min
    else:
        target = now + delta

    seconds = (target - now).total_seconds() + 5 # +5 seconds buffer
    return seconds

def live_loop():
    global live_outcomes
    while True:
        try:
            sec = get_seconds_to_sleep(TIMEFRAME)
            print(f"Live Loop: Sleeping {sec:.1f}s until next close...")
            time.sleep(sec)
            
            print("Live Loop: Fetching latest data...")
            # Fetch enough data for one sequence + buffer
            # We need D candles. Fetching 2*D to be safe.
            # We need to reconstruct the dataframe to compute returns properly
            exchange = ccxt.binance()
            limit = D * 5
            candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
            df_live = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_live['timestamp'] = pd.to_datetime(df_live['timestamp'], unit='ms')
            
            # Derive
            df_derived = deriveround(df_live)
            
            # Take the last D-1 candles to predict the NEXT one
            # The candle that just closed is at index -1. 
            # We want to predict the candle that is CURRENTLY open (which isn't in derived fully yet if we only look at closed).
            # Actually, the user says "predicts on the most recent CLOSED candles... Records outcome one timeframe later".
            
            # Step 1: Check outcome of PREVIOUS prediction if exists
            if len(live_outcomes) > 0 and 'outcome' not in live_outcomes[-1]:
                # The last prediction was for the candle that JUST closed
                last_pred = live_outcomes[-1]
                actual_close_ret = df_derived.iloc[-1]['close_ret']
                actual_dir = 1 if actual_close_ret > 0 else -1
                if actual_close_ret == 0: actual_dir = 0
                
                last_pred['outcome'] = (last_pred['pred_dir'] == actual_dir)
                last_pred['actual_ret'] = actual_close_ret
                last_pred['pnl'] = last_pred['pred_dir'] * actual_close_ret
            
            # Step 2: Make NEW prediction for the candle that just opened
            # We use the LAST D-1 closed candles as the pattern
            if len(df_derived) >= D-1:
                recent_context = df_derived.iloc[-(D-1):][['open_ret', 'high_ret', 'low_ret', 'close_ret']].values
                recent_context_flat = recent_context.reshape(-1)
                
                # Model Matching
                model_context = model_sequences[:, :D-1, :]
                model_context_flat = model_context.reshape(model_context.shape[0], -1)
                
                diff = np.abs(model_context_flat - recent_context_flat)
                matches_idx = np.where(np.all(diff < E, axis=1))[0]
                
                if len(matches_idx) > 0:
                    model_outcomes = model_sequences[matches_idx, -1, :]
                    avg_ret = np.mean(model_outcomes[:, 3])
                    pred_dir = 1 if avg_ret > 0 else -1
                    if avg_ret == 0: pred_dir = 0
                    
                    live_outcomes.append({
                        'time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                        'pred_dir': pred_dir,
                        'matches': len(matches_idx)
                    })
                else:
                    live_outcomes.append({
                        'time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                        'pred_dir': 0,
                        'matches': 0,
                        'note': 'Flat/No Match'
                    })
            
            # Prune list (max 2 weeks approx)
            # 2 weeks of 1h candles = 336
            if len(live_outcomes) > 336:
                live_outcomes.pop(0)
                
        except Exception as e:
            print(f"Live loop error: {e}")
            time.sleep(60)

# --- Web Server ---

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global results_html, live_outcomes
        
        # Build Live Section
        live_html = "<h2>Live Outcomes (Last 2 weeks)</h2><table border='1'><tr><th>Time</th><th>Pred</th><th>Matches</th><th>Outcome</th><th>PnL</th></tr>"
        for item in reversed(live_outcomes):
            outcome_str = item.get('outcome', 'Pending...')
            pnl_str = f"{item.get('pnl', 0):.4f}" if 'pnl' in item else "-"
            pred_s = "UP" if item['pred_dir'] == 1 else ("DOWN" if item['pred_dir'] == -1 else "FLAT")
            live_html += f"<tr><td>{item['time']}</td><td>{pred_s}</td><td>{item['matches']}</td><td>{outcome_str}</td><td>{pnl_str}</td></tr>"
        live_html += "</table>"
        
        full_page = f"""
        <html><head><title>Pattern Matcher</title>
        <meta http-equiv="refresh" content="30">
        </head><body>
        <h1>Market Pattern Matcher: {SYMBOL} {TIMEFRAME}</h1>
        {results_html}
        <hr>
        {live_html}
        </body></html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(full_page.encode())

def run_server():
    with socketserver.TCPServer(("", 8080), Handler) as httpd:
        print("Serving on port 8080")
        httpd.serve_forever()

# --- Main ---

def main():
    global model_sequences, results_html, data_global
    
    # 1. Fetch
    df = fetch(TIMEFRAME, SYMBOL, START, END)
    
    # 2. Derive
    df_derived = deriveround(df, A)
    data_global = df_derived # Store for referencing
    
    # 3. Split
    split1, split2 = split(df_derived, B)
    print(f"Split 1 size: {len(split1)}, Split 2 size: {len(split2)}")
    
    # 4. Get Top (Train)
    model_sequences = gettop(split1, C, D)
    print(f"Model trained. {len(model_sequences)} patterns retained.")
    
    # 5. Predict on Split 2
    preds = completesimilarbeginnings(split2, model_sequences, E, D)
    
    # 6. Predict on Recent 14 Days (Overlap with Split 2 or new fetch)
    # Re-fetching recent 14 days to ensure freshness
    recent_start = (datetime.utcnow() - timedelta(days=14)).isoformat()
    recent_end = datetime.utcnow().isoformat()
    df_recent = fetch(TIMEFRAME, SYMBOL, recent_start, recent_end)
    df_recent_derived = deriveround(df_recent)
    preds_recent = predict_on_recent(model_sequences, df_recent_derived, E, D)
    
    # 7. Generate Output
    html_split2, _ = printaccuracy(preds)
    html_recent, _ = printaccuracy(preds_recent)
    
    results_html = f"""
    <h2>Backtest (Split 2)</h2>
    {html_split2}
    <hr>
    <h2>Recent 14 Days Performance</h2>
    {html_recent}
    """
    
    # 8. Start Live Loop
    t_live = threading.Thread(target=live_loop)
    t_live.daemon = True
    t_live.start()
    
    # 9. Start Server
    run_server()

if __name__ == "__main__":
    main()
