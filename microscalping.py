import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = ['timestamp', 'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def get_random_24h_window(df):
    start = df['datetime'].min()
    end = df['datetime'].max() - pd.Timedelta(hours=24)
    random_start = np.random.choice(pd.date_range(start, end, freq='min'))
    return df[(df['datetime'] >= random_start) & (df['datetime'] < random_start + pd.Timedelta(hours=24))]

def performance_metrics(trades, initial_balance=100):
    if trades.empty:
        return {
            'Saldo Iniziale': initial_balance,
            'Saldo Finale': initial_balance,
            'ROI (%)': 0.0,
            'Max Drawdown (%)': 0.0,
            'Win Rate (%)': 0.0,
            'Profitto Medio per Trade': 0.0,
            'Sharpe Ratio': 0.0
        }
    
    balance_history = [initial_balance + trades['profit'].iloc[:i+1].sum() for i in range(len(trades))]
    max_drawdown = (min(balance_history) - initial_balance) / initial_balance * 100
    
    return {
        'Saldo Iniziale': initial_balance,
        'Saldo Finale': round(balance_history[-1], 2),
        'ROI (%)': round((balance_history[-1] - initial_balance) / initial_balance * 100, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Win Rate (%)': round((trades['profit'] > 0).mean() * 100, 2),
        'Profitto Medio per Trade': round(trades['profit'].mean(), 4),
        'Sharpe Ratio': round(trades['profit'].mean() / trades['profit'].std(), 2) if trades['profit'].std() != 0 else 0
    }

def microscalping_strategy(df, initial_balance=100, risk_per_trade=0.01, rr_ratio=1.5):
    df = df.copy()
    
    # Calcolo indicatori
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['Volume_MA50'] = df['volume'].rolling(50).mean()
    df['ATR_MA100'] = df['ATR'].rolling(100).mean()
    
    balance = initial_balance
    trades = []
    
    for i in range(100, len(df)-1):  # Skip primi 100 periodi per indicatori
        if balance <= initial_balance * 0.8:
            break
            
        if pd.isna(df['ATR_MA100'].iloc[i]) or pd.isna(df['Volume_MA50'].iloc[i]):
            continue
            
        current_atr = df['ATR'].iloc[i]
        atr_ma100 = df['ATR_MA100'].iloc[i]
        volume_cond = df['volume'].iloc[i] > df['Volume_MA50'].iloc[i]
        
        # Filtro trend 4 ore
        hourly_trend = df['close'].iloc[max(0,i-240):i].mean()
        
        if volume_cond and (current_atr > atr_ma100):
            position_size = (balance * risk_per_trade) / (current_atr + 1e-8)
            
            long_cond = (df['close'].iloc[i] > df['MA20'].iloc[i]) and (df['close'].iloc[i] > hourly_trend)
            short_cond = (df['close'].iloc[i] < df['MA20'].iloc[i]) and (df['close'].iloc[i] < hourly_trend)
            
            if long_cond or short_cond:
                entry = df['close'].iloc[i]
                direction = 1 if long_cond else -1
                spread = df['high'].iloc[i] - df['low'].iloc[i]
                
                if spread < current_atr * 0.2:
                    continue
                
                # Calcolo SL/TP con slippage
                sl = entry - current_atr * 0.3 * direction
                tp = entry + current_atr * rr_ratio * direction
                executed_entry = entry + (spread * 0.0005 * (-direction))
                
                # Simula chiusura nella candela successiva
                exit_price = df['close'].iloc[i+1] + (spread * 0.0005 * direction)
                exit_price = np.clip(exit_price, sl, tp) if direction == 1 else np.clip(exit_price, tp, sl)
                
                profit = (exit_price - executed_entry) * direction * position_size
                balance += profit
                
                trades.append({'profit': profit})
    
    return pd.DataFrame(trades)

def run_tests(file_path, num_tests=100, initial_balance=100):
    df = load_data(file_path)
    results = []
    
    for test_num in range(1, num_tests+1):
        df_24h = get_random_24h_window(df)
        trades = microscalping_strategy(df_24h, initial_balance=initial_balance)
        metrics = performance_metrics(trades, initial_balance=initial_balance)
        results.append(metrics)
        
        print(f"\nTest {test_num}/{num_tests}")
        print(f"Periodo: {df_24h['datetime'].iloc[0]} - {df_24h['datetime'].iloc[-1]}")
        print(f"Trade eseguiti: {len(trades)}")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("-" * 40)
    
    results_df = pd.DataFrame(results)
    print("\nStatistiche Aggregate:")
    print(results_df.describe())
    
    # Grafico performance
    plt.figure(figsize=(12,6))
    plt.plot(results_df['ROI (%)'].cumsum(), marker='o', color='#2ecc71')
    plt.title("Performance Cumulativa")
    plt.xlabel("Numero Test")
    plt.ylabel("ROI Cumulativo (%)")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return results_df

if __name__ == "__main__":
    results = run_tests("Binance_BNBUSDT_2024_minute.csv")