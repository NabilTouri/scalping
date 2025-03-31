import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta  # Usa pandas_ta al posto di talib

# Parametri di configurazione
TAKER_FEE = 0.00075  # 0.075% (commissioni per ordini market)
MAKER_FEE = 0.0002   # 0.02% (commissioni per ordini limit)
SLIPPAGE = 0.0002    # 0.02% (slippage medio)
MIN_TRADE_SIZE = 10  # $10 (dimensione minima del trade)

def load_data(file_path):
    """
    Carica i dati di mercato da un file CSV.
    """
    df = pd.read_csv(file_path, header=None)
    df.columns = ['timestamp', 'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def get_random_24h_window(df):
    """
    Seleziona una finestra casuale di 24 ore dai dati.
    """
    start = df['datetime'].min()
    end = df['datetime'].max() - pd.Timedelta(hours=24)
    random_start = np.random.choice(pd.date_range(start, end, freq='min'))
    return df[(df['datetime'] >= random_start) & (df['datetime'] < random_start + pd.Timedelta(hours=24))]

def performance_metrics(trades, initial_balance=100):
    """
    Calcola le metriche di performance dei trade eseguiti.
    """
    if trades.empty:
        return {
            'Saldo Iniziale': initial_balance,
            'Saldo Finale': initial_balance,
            'ROI (%)': 0.0,
            'Max Drawdown (%)': 0.0,
            'Win Rate (%)': 0.0,
            'Profitto Medio per Trade': 0.0,
            'Sharpe Ratio': 0.0,
            'Costi Totali (%)': 0.0,
            'Profit Factor': 0.0
        }
    
    balance_history = [initial_balance + trades['profit'].iloc[:i+1].sum() for i in range(len(trades))]
    max_drawdown = (min(balance_history) - initial_balance) / initial_balance * 100
    
    total_fees = trades['fees'].sum()
    gross_profit = trades[trades['profit'] > 0]['profit'].sum()
    gross_loss = abs(trades[trades['profit'] < 0]['profit'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    return {
        'Saldo Iniziale': initial_balance,
        'Saldo Finale': round(balance_history[-1], 2),
        'ROI (%)': round((balance_history[-1] - initial_balance) / initial_balance * 100, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Win Rate (%)': round((trades['profit'] > 0).mean() * 100, 2),
        'Profitto Medio per Trade': round(trades['profit'].mean(), 4),
        'Sharpe Ratio': round(trades['profit'].mean() / trades['profit'].std(), 2) if trades['profit'].std() != 0 else 0,
        'Costi Totali (%)': round((total_fees / initial_balance) * 100, 2),
        'Profit Factor': round(profit_factor, 2)
    }

def microscalping_strategy(df, initial_balance=100, risk_per_trade=0.01, rr_ratio=3.0, max_trades=10):
    """
    Esegue una strategia di microscalping su un DataFrame di dati di mercato.
    """
    df = df.copy()
    
    # Calcolo indicatori tecnici
    df['ATR'] = (
        pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1).rolling(14).mean()
    )
    df['MA20'] = df['close'].rolling(20).mean()
    df['Volume_MA50'] = df['volume'].rolling(50).mean()
    df['ATR_MA100'] = df['ATR'].rolling(100).mean()
    df['RSI'] = ta.rsi(df['close'], length=14)  # Calcola RSI con pandas_ta
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)  # Calcola MACD con pandas_ta
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    
    balance = initial_balance
    trades = []
    trade_count = 0
    
    for i in range(100, len(df)-1):
        if balance <= initial_balance * 0.8 or trade_count >= max_trades: break
        
        if pd.isna(df['ATR_MA100'].iloc[i]) or pd.isna(df['Volume_MA50'].iloc[i]): continue
        
        current_atr = df['ATR'].iloc[i]
        volume_cond = df['volume'].iloc[i] > df['Volume_MA50'].iloc[i]
        hourly_trend = df['close'].iloc[max(0,i-240):i].mean()
        
        if volume_cond and (df['ATR'].iloc[i] > df['ATR_MA100'].iloc[i]):
            position_size = max((balance * risk_per_trade) / (current_atr + 1e-8), MIN_TRADE_SIZE)
            units = position_size / df['close'].iloc[i]
            
            long_cond = (df['close'].iloc[i] > df['MA20'].iloc[i]) and (df['close'].iloc[i] > hourly_trend) and (df['RSI'].iloc[i] < 70) and (df['MACD'].iloc[i] > df['MACD_signal'].iloc[i])
            short_cond = (df['close'].iloc[i] < df['MA20'].iloc[i]) and (df['close'].iloc[i] < hourly_trend) and (df['RSI'].iloc[i] > 30) and (df['MACD'].iloc[i] < df['MACD_signal'].iloc[i])
            
            if long_cond or short_cond:
                direction = 1 if long_cond else -1
                spread_cost = (df['high'].iloc[i] - df['low'].iloc[i]) * SLIPPAGE
                entry_price = df['close'].iloc[i] + (spread_cost * (-direction))
                exit_price = df['close'].iloc[i+1] + (spread_cost * direction)
                
                entry_fee = entry_price * MAKER_FEE * units
                exit_fee = exit_price * MAKER_FEE * units
                total_cost = entry_fee + exit_fee
                
                profit = (exit_price - entry_price) * direction * units - total_cost
                balance += profit
                trade_count += 1
                
                trades.append({
                    'datetime': df['datetime'].iloc[i],
                    'units': units,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit': profit,
                    'fees': total_cost
                })
    
    return pd.DataFrame(trades)

def run_tests(file_path, num_tests=10, initial_balance=100):
    """
    Esegue test multipli della strategia su finestre casuali di 24 ore.
    """
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
    results = run_tests("Binance_BTCEUR_2022_minute.csv")