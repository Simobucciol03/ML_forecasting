import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# ✅ Imposta il percorso del progetto
current_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.append(project_root)  

# 📌 Importa i dati dal database
from ..connection_database.login_mysql import DataProvider

dp = DataProvider()
da = dp.main()  
df = da[0]

# ✅ Ordina i dati per data
df = df.sort_values(by="date").reset_index(drop=True)

# ✅ Funzione per calcolare l'EMA manualmente
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

# ✅ Funzione per calcolare l'ATR manualmente
def atr(high, low, close, period=14):
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    return tr.rolling(window=period).mean()

# ✅ Funzione per calcolare l'RSI manualmente
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ✅ Aggiungi gli indicatori al DataFrame
df["EMA_20"] = ema(df["close"], 20)
df["ATR"] = atr(df["high"], df["low"], df["close"], period=14)
df["RSI"] = rsi(df["close"], period=14)

# 📌 🔥 Miglioramento: Finestra dinamica basata sulla volatilità
order_value = int(df["ATR"].mean() * 5)
order_value = max(order_value, 10)  # Minimo 10 per evitare errori

# 📌 🔥 Identificazione di massimi e minimi locali
df["Local_Min"] = df.iloc[argrelextrema(df["EMA_20"].values, np.less_equal, order=order_value)[0]]["close"]
df["Local_Max"] = df.iloc[argrelextrema(df["EMA_20"].values, np.greater_equal, order=order_value)[0]]["close"]

# 📌 🔥 Creazione del target di previsione per il modello ML
df["Position"] = np.nan

# 🔹 Condizioni di ingresso migliorate:
df.loc[(df["Local_Min"].notna()) & (df["RSI"] < 30), "Position"] = 1  # Long se RSI < 30
df.loc[(df["Local_Max"].notna()) & (df["RSI"] > 70), "Position"] = 0  # Short se RSI > 70

# Converti tutti i minimi locali in segnali di acquisto (Long) e i massimi locali in segnali di vendita (Short)
df.loc[df["Local_Min"].notna(), "Position"] = 1  # Long su minimi locali
df.loc[df["Local_Max"].notna(), "Position"] = 0  # Short su massimi locali

# Se la posizione non è un segnale di acquisto o vendita, riempi con il valore precedente

# Create future returns (target variable for training)
# df['returns'] = df['close'].pct_change()
# df['Future_Returns_1d'] = df['returns'].shift(-1)

# Generate position based on future returns (1 for buy, 0 for sell)
# df['position'] = np.where(df['Future_Returns_1d'] > 0, 1, 0)





df["position"].fillna(method="ffill", inplace=True)

# ✅ Stampa della colonna "Position"
print(df["position"])

# 🔹 GRAFICO: Prezzo e posizioni

# Crea un grafico con due sottotrame
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# 🔹 Grafico del prezzo di chiusura
ax1.plot(df["close"], label="Close Price", color="blue")
# ax1.plot(df["Local_Min"], marker="o", linestyle="", color="green", label="Local Min (Long)")
# ax1.plot(df["Local_Max"], marker="o", linestyle="", color="red", label="Local Max (Short)")
# ax1.set_title("Close Price History with Buy/Sell Signals")
# ax1.set_ylabel("Price")
# ax1.legend()

# 🔹 Grafico delle posizioni
ax2.plot(df["position"], label="Position (Long/Short)", color="orange")
ax2.set_title("Optimal Long/Short Position History")
ax2.set_ylabel("Position")
ax2.set_xlabel("Date")
ax2.legend()

plt.show()


y = df["position"]



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Set project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# Import data from database
from connection_database.login_mysql import DataProvider

dp = DataProvider()
da = dp.main()
df = da[0]

# Sort data by date
df = df.sort_values(by="date").reset_index(drop=True)

# Function to identify trends using only close prices
def identify_trend_direction(prices, window_size=20, threshold_pct=0.03):
    """
    Identifies trend direction in price data using only closing prices
    
    Parameters:
    - prices: Series of closing prices
    - window_size: Size of the rolling window to identify trends (default: 20 periods)
    - threshold_pct: Percentage change threshold to confirm a trend change (default: 3%)
    
    Returns:
    - Series with trend values (1 for uptrend, -1 for downtrend, 0 for no clear trend)
    """
    # Calculate rolling average
    rolling_avg = prices.rolling(window=window_size).mean()
    
    # Calculate rate of change over different periods
    roc_short = prices.pct_change(window_size // 2)
    roc_medium = prices.pct_change(window_size)
    roc_long = prices.pct_change(window_size * 2)
    
    # Initialize trend with NaN values
    trend = pd.Series(index=prices.index, dtype=float)
    
    # Initial state
    current_trend = None
    
    # Identify trends
    for i in range(window_size * 2, len(prices)):
        # Price relative to moving average
        price_vs_avg = prices.iloc[i] / rolling_avg.iloc[i] - 1
        
        # Rate of change signals
        short_signal = 1 if roc_short.iloc[i] > threshold_pct else (-1 if roc_short.iloc[i] < -threshold_pct else 0)
        medium_signal = 1 if roc_medium.iloc[i] > threshold_pct else (-1 if roc_medium.iloc[i] < -threshold_pct else 0)
        long_signal = 1 if roc_long.iloc[i] > threshold_pct else (-1 if roc_long.iloc[i] < -threshold_pct else 0)
        
        # Combined signal with more weight to longer timeframes
        combined_signal = (0.2 * short_signal + 0.3 * medium_signal + 0.5 * long_signal)
        
        # Set initial trend
        if pd.isna(current_trend):
            if combined_signal > 0.3:
                current_trend = 1  # Uptrend
            elif combined_signal < -0.3:
                current_trend = -1  # Downtrend
            else:
                current_trend = 0  # No clear trend
                
        # Check for trend change
        if current_trend == 1 and combined_signal < -0.4:
            # Strong evidence of trend reversal from up to down
            current_trend = -1
        elif current_trend == -1 and combined_signal > 0.4:
            # Strong evidence of trend reversal from down to up
            current_trend = 1
        elif current_trend == 0:
            # If currently no trend, check if a new trend is forming
            if combined_signal > 0.3:
                current_trend = 1
            elif combined_signal < -0.3:
                current_trend = -1
                
        trend.iloc[i] = current_trend
        
    return trend

# Apply trend identification with multiple timeframes
short_window = 9
medium_window = 20
long_window = 40

df['trend_short'] = identify_trend_direction(df['close'], window_size=short_window, threshold_pct=0.02)
df['trend_medium'] = identify_trend_direction(df['close'], window_size=medium_window, threshold_pct=0.03)
df['trend_long'] = identify_trend_direction(df['close'], window_size=long_window, threshold_pct=0.05)

# Create combined trend signal
df['trend_combined'] = (0.2 * df['trend_short'].fillna(0) + 
                        0.3 * df['trend_medium'].fillna(0) + 
                        0.5 * df['trend_long'].fillna(0))

# Position logic: 
# 1 for long (strong uptrend)
# -1 for short (strong downtrend)
# 0 for no position (unclear trend)
df['position'] = np.where(df['trend_combined'] > 0.3, 1, 
                         np.where(df['trend_combined'] < -0.3, -1, 0))

# Forward fill any missing positions
df['position'] = df['position'].fillna(method='ffill').fillna(0)

# Create trade signals column (entry and exit points)
df['signal'] = df['position'].diff()

# Print positions
print("Position Distribution:")
print(df['position'].value_counts())

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Close price chart
ax1.plot(df['close'], label='Close Price', color='blue')
ax1.set_title('Price History with Trend Positions')
ax1.set_ylabel('Price')

# Mark entry and exit points
long_entries = df[df['signal'] == 1].index
long_exits = df[df['signal'] == -1].index
short_entries = df[df['signal'] == -2].index
short_exits = df[df['signal'] == 2].index

ax1.plot(long_entries, df.loc[long_entries, 'close'], '^', markersize=10, color='green', label='Long Entry')
ax1.plot(long_exits, df.loc[long_exits, 'close'], 'v', markersize=10, color='red', label='Long Exit')
ax1.plot(short_entries, df.loc[short_entries, 'close'], 'v', markersize=10, color='purple', label='Short Entry')
ax1.plot(short_exits, df.loc[short_exits, 'close'], '^', markersize=10, color='orange', label='Short Exit')

ax1.legend()

# Position chart
ax2.plot(df['position'], label='Position (1=Long, -1=Short, 0=Cash)', color='blue')
ax2.set_title('Position History')
ax2.set_ylabel('Position')
ax2.set_xlabel('Date')
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels(['Short', 'Cash', 'Long'])
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# Calculate performance metrics
df['returns'] = df['close'].pct_change()
df['strategy_returns'] = df['position'].shift(1) * df['returns']

# Calculate cumulative returns
df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
df['strategy_cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1

# Plot performance comparison
plt.figure(figsize=(15, 8))
plt.plot(df['cumulative_returns'], label='Buy & Hold', color='blue')
plt.plot(df['strategy_cumulative_returns'], label='Trend Strategy', color='green')
plt.title('Strategy Performance Comparison')
plt.xlabel('Time')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

# Print performance statistics
print("\nPerformance Metrics:")
print(f"Total Return: {df['strategy_cumulative_returns'].iloc[-1]:.4f}")
print(f"Annual Return: {df['strategy_returns'].mean() * 252:.4f}")
print(f"Sharpe Ratio: {(df['strategy_returns'].mean() / df['strategy_returns'].std()) * np.sqrt(252):.4f}")
print(f"Max Drawdown: {(df['strategy_cumulative_returns'].cummax() - df['strategy_cumulative_returns']).max():.4f}")

# Calculate drawdowns
df['drawdown'] = df['strategy_cumulative_returns'].cummax() - df['strategy_cumulative_returns']

# Additional analysis
print("\nPosition Analysis:")
print(f"Average holding period (days): {1 / abs(df['signal']).mean():.1f}")
print(f"Number of trades: {abs(df['signal']).sum() / 2:.0f}")
print(f"Win rate: {(df.loc[df['signal'] != 0, 'strategy_returns'] > 0).mean():.2f}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# Set project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# Import data from database
from connection_database.login_mysql import DataProvider

dp = DataProvider()
da = dp.main()
df = da[0]

# Sort data by date
df = df.sort_values(by="date").reset_index(drop=True)

# Function to calculate price momentum with reduced lag
def directional_movement(prices, window=10):
    """
    Calculates price direction and momentum using linear regression slope
    
    Parameters:
    - prices: Series of closing prices
    - window: Lookback window for slope calculation
    
    Returns:
    - Series with slope values (positive = uptrend, negative = downtrend)
    """
    slopes = pd.Series(index=prices.index, dtype=float)
    
    for i in range(window, len(prices)):
        # Get price segment
        segment = prices.iloc[i-window:i]
        
        # Calculate linear regression
        x = np.arange(len(segment))
        slope, _, r_value, _, _ = linregress(x, segment.values)
        
        # Store slope value multiplied by R-squared for confidence weighting
        slopes.iloc[i] = slope * (r_value ** 2)
    
    return slopes

# Function to detect price breakouts
def detect_breakouts(prices, short_window=5, long_window=20, sensitivity=1.0):
    """
    Detects price breakouts using volatility-based thresholds
    
    Parameters:
    - prices: Series of closing prices
    - short_window: Window for recent price range
    - long_window: Window for volatility calculation
    - sensitivity: Multiplier for breakout threshold
    
    Returns:
    - DataFrame with breakout signals
    """
    # Calculate volatility (Average True Range concept but with close prices)
    volatility = prices.rolling(window=long_window).std()
    
    # Calculate recent highest and lowest prices
    recent_high = prices.rolling(window=short_window).max()
    recent_low = prices.rolling(window=short_window).min()
    
    # Calculate breakout thresholds
    upside_threshold = recent_high + volatility * sensitivity
    downside_threshold = recent_low - volatility * sensitivity
    
    # Generate breakout signals
    upside_breakout = prices > prices.shift(1).rolling(window=long_window).max()
    downside_breakout = prices < prices.shift(1).rolling(window=long_window).min()
    
    result = pd.DataFrame(index=prices.index)
    result['upside_breakout'] = upside_breakout
    result['downside_breakout'] = downside_breakout
    
    return result

# Function to identify momentum divergences
def detect_momentum_shift(prices, window=10, threshold=0.02):
    """
    Detects shifts in price momentum for early trend change signals
    
    Parameters:
    - prices: Series of closing prices
    - window: Lookback window
    - threshold: Percentage threshold for momentum shift
    
    Returns:
    - Series with momentum shift signals (1=positive shift, -1=negative shift, 0=no shift)
    """
    # Calculate rate of change
    roc = prices.pct_change(window)
    
    # Calculate acceleration (second derivative of price)
    acceleration = roc.diff(window)
    
    # Identify momentum shifts
    positive_shift = (roc > 0) & (acceleration > threshold)
    negative_shift = (roc < 0) & (acceleration < -threshold)
    
    result = pd.Series(0, index=prices.index)
    result[positive_shift] = 1
    result[negative_shift] = -1
    
    return result

# Apply the three different methods to the close prices
df['slope'] = directional_movement(df['close'], window=10)
breakouts = detect_breakouts(df['close'], short_window=5, long_window=20, sensitivity=0.5)
df['momentum_shift'] = detect_momentum_shift(df['close'], window=5, threshold=0.01)

# Merge breakout signals
df['upside_breakout'] = breakouts['upside_breakout']
df['downside_breakout'] = breakouts['downside_breakout']

# Generate combined signal
df['signal'] = 0

# Rules for long position:
# 1. Positive slope (uptrend direction)
# 2. Either an upside breakout OR positive momentum shift
df.loc[(df['slope'] > 0) & (df['upside_breakout'] | (df['momentum_shift'] == 1)), 'signal'] = 1

# Rules for short position:
# 1. Negative slope (downtrend direction)
# 2. Either a downside breakout OR negative momentum shift
df.loc[(df['slope'] < 0) & (df['downside_breakout'] | (df['momentum_shift'] == -1)), 'signal'] = -1

# Implement a trend confirmation filter to reduce whipsaws
confirmation_window = 3
df['confirmed_signal'] = np.where(
    df['signal'].rolling(window=confirmation_window).sum() >= 2, 1,  # Long if majority of recent signals are positive
    np.where(
        df['signal'].rolling(window=confirmation_window).sum() <= -2, -1,  # Short if majority of recent signals are negative
        0  # Otherwise, no position
    )
)

# Generate position column (with reduced noise)
df['position'] = df['confirmed_signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)

# Add exit rules to improve responsiveness
# Exit long position when slope turns negative and we have a downside breakout
long_exit = (df['position'] == 1) & (df['slope'] < 0) & (df['downside_breakout'] | (df['momentum_shift'] == -1))
# Exit short position when slope turns positive and we have an upside breakout
short_exit = (df['position'] == -1) & (df['slope'] > 0) & (df['upside_breakout'] | (df['momentum_shift'] == 1))

# Apply exits
df.loc[long_exit, 'position'] = 0
df.loc[short_exit, 'position'] = 0

# Fill forward positions
df['position'] = df['position'].fillna(method='ffill').fillna(0)

# Calculate trade signals (for visualization)
df['trade_signal'] = df['position'].diff()

# Plotting results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Plot price and position
ax1.plot(df['close'], label='Close Price', color='blue', alpha=0.7)

# Highlight position periods
long_periods = df['position'] == 1
short_periods = df['position'] == -1

# Create long/short background shading
for i in range(len(df)):
    if df['position'].iloc[i] == 1:
        ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='green')
    elif df['position'].iloc[i] == -1:
        ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='red')

# Mark entry and exit points
long_entries = df[df['trade_signal'] == 1].index
long_exits = df[(df['trade_signal'] < 0) & (df['position'].shift(1) == 1)].index
short_entries = df[df['trade_signal'] == -1].index
short_exits = df[(df['trade_signal'] > 0) & (df['position'].shift(1) == -1)].index

ax1.plot(long_entries, df.loc[long_entries, 'close'], '^', color='green', markersize=10, label='Long Entry')
ax1.plot(long_exits, df.loc[long_exits, 'close'], 'v', color='orange', markersize=8, label='Long Exit')
ax1.plot(short_entries, df.loc[short_entries, 'close'], 'v', color='red', markersize=10, label='Short Entry')
ax1.plot(short_exits, df.loc[short_exits, 'close'], '^', color='magenta', markersize=8, label='Short Exit')

ax1.set_title('Price Action with Responsive Trend Signals')
ax1.set_ylabel('Price')
ax1.legend()

# Plot slope indicator to show trend strength
ax2.bar(df.index, df['slope'], color=np.where(df['slope'] > 0, 'green', 'red'), label='Price Direction (Slope)')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Add breakout markers
up_breaks = df[df['upside_breakout'] == True].index
down_breaks = df[df['downside_breakout'] == True].index
ax2.scatter(up_breaks, [0.001] * len(up_breaks), marker='^', color='blue', s=50, label='Upside Breakout')
ax2.scatter(down_breaks, [-0.001] * len(down_breaks), marker='v', color='purple', s=50, label='Downside Breakout')

ax2.set_title('Trend Direction and Breakout Signals')
ax2.set_ylabel('Slope')
ax2.legend()

plt.tight_layout()
plt.show()

# Calculate performance metrics
df['returns'] = df['close'].pct_change()
df['strategy_returns'] = df['position'].shift(1) * df['returns']

# Calculate cumulative returns
df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
df['strategy_cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1

# Plot performance comparison
plt.figure(figsize=(15, 8))
plt.plot(df['cumulative_returns'], label='Buy & Hold', color='blue')
plt.plot(df['strategy_cumulative_returns'], label='Trend Strategy', color='green')
plt.title('Strategy Performance Comparison')
plt.xlabel('Time')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

# Calculate drawdowns
df['drawdown'] = df['strategy_cumulative_returns'].cummax() - df['strategy_cumulative_returns']
max_drawdown = df['drawdown'].max()

# Print performance statistics
print("\nPerformance Metrics:")
print(f"Total Return: {df['strategy_cumulative_returns'].iloc[-1]:.4f}")
print(f"Annual Return: {df['strategy_returns'].mean() * 252:.4f}")
print(f"Sharpe Ratio: {(df['strategy_returns'].mean() / df['strategy_returns'].std()) * np.sqrt(252):.4f}")
print(f"Max Drawdown: {max_drawdown:.4f}")
print(f"Win Rate: {(df['strategy_returns'] > 0).mean():.4f}")
print(f"Average Winning Trade: {df['strategy_returns'][df['strategy_returns'] > 0].mean():.4f}")
print(f"Average Losing Trade: {df['strategy_returns'][df['strategy_returns'] < 0].mean():.4f}")
print(f"Profit Factor: {abs(df['strategy_returns'][df['strategy_returns'] > 0].sum() / df['strategy_returns'][df['strategy_returns'] < 0].sum()):.4f}")
print(f"Number of Trades: {abs(df['trade_signal']).sum() / 2:.0f}")






###################################################################### NEW CODE ##############################################################################


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import sys
# import os
# from scipy.stats import linregress
# import warnings
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.model_selection import TimeSeriesSplit, cross_val_score
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.pipeline import Pipeline
# warnings.filterwarnings('ignore')

# # Set project path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, ".."))
# sys.path.append(project_root)

# # Import data from database
# from connection_database.login_mysql import DataProvider

# dp = DataProvider()
# da = dp.main()
# df = da[0]

# # Sort data by date
# df = df.sort_values(by="date").reset_index(drop=True)

# # Improved Technical Feature Generation
# # =====================================

# # 1. Enhanced Directional Movement
# def enhanced_directional_movement(prices, windows=[5, 10, 20]):
#     """
#     Calculates price direction and momentum using linear regression slope at multiple timeframes
    
#     Parameters:
#     - prices: Series of closing prices
#     - windows: List of lookback windows for slope calculation
    
#     Returns:
#     - DataFrame with slope values for different timeframes
#     """
#     result = pd.DataFrame(index=prices.index)
    
#     for window in windows:
#         slopes = pd.Series(index=prices.index, dtype=float)
        
#         for i in range(window, len(prices)):
#             # Get price segment
#             segment = prices.iloc[i-window:i]
            
#             # Calculate linear regression
#             x = np.arange(len(segment))
#             slope, _, r_value, _, _ = linregress(x, segment.values)
            
#             # Store slope value multiplied by R-squared for confidence weighting
#             slopes.iloc[i] = slope * (r_value ** 2)
        
#         result[f'slope_{window}'] = slopes
    
#     # Add slope ratios for trend persistence detection
#     if len(windows) > 1:
#         for i in range(len(windows)-1):
#             short_window = windows[i]
#             long_window = windows[i+1]
#             result[f'slope_ratio_{short_window}_{long_window}'] = result[f'slope_{short_window}'] / result[f'slope_{long_window}']
    
#     return result

# # 2. Volume-Price Relationship
# def volume_price_analysis(prices, volume, windows=[5, 10, 20]):
#     """
#     Analyzes relationship between price movements and volume
    
#     Parameters:
#     - prices: Series of closing prices
#     - volume: Series of trading volumes
#     - windows: List of lookback windows
    
#     Returns:
#     - DataFrame with volume-price indicators
#     """
#     result = pd.DataFrame(index=prices.index)
    
#     # Calculate price changes
#     price_change = prices.pct_change()
    
#     # Normalize volume
#     normalized_volume = volume / volume.rolling(window=50).mean()
    
#     for window in windows:
#         # Volume momentum
#         result[f'volume_momentum_{window}'] = normalized_volume.pct_change(window)
        
#         # Price-volume correlation (positive correlation in uptrend is bullish)
#         result[f'price_volume_corr_{window}'] = (
#             price_change.rolling(window=window).corr(normalized_volume)
#         )
        
#         # Volume trend
#         result[f'volume_trend_{window}'] = (
#             normalized_volume.rolling(window=window).mean() / 
#             normalized_volume.rolling(window=window*2).mean() - 1
#         )
        
#         # On-balance volume
#         obv = (price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * normalized_volume).cumsum()
#         result[f'obv_slope_{window}'] = enhanced_directional_movement(obv, [window])[f'slope_{window}']
    
#     return result

# # 3. Enhanced Breakout Detection
# def enhanced_breakout_detection(prices, volume=None, windows=[5, 10, 20, 50]):
#     """
#     Detects price breakouts with multiple confirmation factors
    
#     Parameters:
#     - prices: Series of closing prices
#     - volume: Series of trading volumes (optional)
#     - windows: List of lookback windows
    
#     Returns:
#     - DataFrame with breakout signals and strength indicators
#     """
#     result = pd.DataFrame(index=prices.index)
    
#     for window in windows:
#         # Calculate volatility (Average True Range concept but with close prices)
#         volatility = prices.rolling(window=window).std()
        
#         # Calculate recent highest and lowest prices
#         upper_band = prices.rolling(window=window).max()
#         lower_band = prices.rolling(window=window).min()
        
#         # Measure distance to bands
#         result[f'upper_band_dist_{window}'] = (prices - upper_band) / volatility
#         result[f'lower_band_dist_{window}'] = (prices - lower_band) / volatility
        
#         # Normalized price position within range
#         price_range = (upper_band - lower_band)
#         result[f'range_position_{window}'] = np.where(
#             price_range == 0, 
#             0, 
#             (prices - lower_band) / price_range
#         )
        
#         # Generate breakout signals (continuous rather than binary)
#         result[f'upside_breakout_{window}'] = np.maximum(0, (prices / upper_band.shift(1) - 1) * 100)
#         result[f'downside_breakout_{window}'] = np.minimum(0, (prices / lower_band.shift(1) - 1) * 100)
        
#         # Add volume confirmation if available
#         if volume is not None:
#             norm_volume = volume / volume.rolling(window=window*2).mean()
#             result[f'upside_vol_confirm_{window}'] = result[f'upside_breakout_{window}'] * norm_volume
#             result[f'downside_vol_confirm_{window}'] = result[f'downside_breakout_{window}'] * norm_volume
    
#     return result

# # 4. Advanced Momentum Indicators
# def advanced_momentum_indicators(prices, windows=[5, 10, 20]):
#     """
#     Calculates advanced momentum indicators for trend strength and reversals
    
#     Parameters:
#     - prices: Series of closing prices
#     - windows: List of lookback windows
    
#     Returns:
#     - DataFrame with momentum indicators
#     """
#     result = pd.DataFrame(index=prices.index)
    
#     for window in windows:
#         # Rate of change
#         result[f'roc_{window}'] = prices.pct_change(window)
        
#         # Acceleration (second derivative)
#         result[f'acceleration_{window}'] = result[f'roc_{window}'].diff(window//2)
        
#         # Smoothed momentum
#         ema_fast = prices.ewm(span=window).mean()
#         ema_slow = prices.ewm(span=window*2).mean()
#         result[f'macd_{window}'] = ema_fast - ema_slow
#         result[f'macd_signal_{window}'] = result[f'macd_{window}'].ewm(span=window//2).mean()
#         result[f'macd_hist_{window}'] = result[f'macd_{window}'] - result[f'macd_signal_{window}']
        
#         # RSI calculation
#         delta = prices.diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
#         rs = gain / loss
#         result[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
#         # RSI momentum
#         result[f'rsi_momentum_{window}'] = result[f'rsi_{window}'].diff(window//2)
    
#     return result

# # 5. Market Regime Identification
# def market_regime(prices, windows=[20, 50, 200]):
#     """
#     Identifies market regime (trend/range/volatile) based on multiple metrics
    
#     Parameters:
#     - prices: Series of closing prices
#     - windows: List of lookback windows
    
#     Returns:
#     - DataFrame with regime indicators
#     """
#     result = pd.DataFrame(index=prices.index)
    
#     # Moving averages
#     for window in windows:
#         result[f'ma_{window}'] = prices.rolling(window=window).mean()
    
#     # Moving average crossovers
#     for i in range(len(windows)-1):
#         short_window = windows[i]
#         long_window = windows[i+1]
#         result[f'ma_cross_{short_window}_{long_window}'] = result[f'ma_{short_window}'] - result[f'ma_{long_window}']
    
#     # Volatility regime
#     for window in windows:
#         result[f'volatility_{window}'] = prices.rolling(window=window).std() / prices.rolling(window=window).mean()
    
#     # Trend strength (ADX-like)
#     for window in windows:
#         # +DM and -DM
#         high_change = prices.diff()
#         low_change = -prices.diff()
        
#         pos_dm = np.where((high_change > 0) & (high_change > low_change), high_change, 0)
#         neg_dm = np.where((low_change > 0) & (low_change > high_change), low_change, 0)
        
#         # Smoothed +DM and -DM
#         smoothed_pos_dm = pd.Series(pos_dm).rolling(window=window).mean()
#         smoothed_neg_dm = pd.Series(neg_dm).rolling(window=window).mean()
        
#         # Average true range (simplified)
#         atr = prices.rolling(window=window).std()
        
#         # +DI and -DI
#         pos_di = 100 * smoothed_pos_dm / atr
#         neg_di = 100 * smoothed_neg_dm / atr
        
#         # Directional index
#         dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        
#         # Average directional index
#         result[f'adx_{window}'] = dx.rolling(window=window).mean()
        
#         # IMPORTANT CHANGE: Convert string values to numeric values for trend_strength
#         result[f'trend_strength_{window}'] = np.where(
#             result[f'adx_{window}'] < 20, 0,  # 'range' -> 0
#             np.where(result[f'adx_{window}'] > 40, 2, 1)  # 'trend' -> 2, 'weak_trend' -> 1
#         )
    
#     return result

# # Generate all technical features
# def generate_all_features(df):
#     """
#     Generates comprehensive feature set for the ML model
    
#     Parameters:
#     - df: DataFrame with OHLCV data
    
#     Returns:
#     - DataFrame with all technical features
#     """
#     # Generate features
#     slopes = enhanced_directional_movement(df['close'], windows=[5, 10, 20, 50])
#     volume_features = volume_price_analysis(df['close'], df['volume'], windows=[5, 10, 20])
#     breakout_features = enhanced_breakout_detection(df['close'], df['volume'], windows=[5, 10, 20, 50])
#     momentum_features = advanced_momentum_indicators(df['close'], windows=[5, 10, 20])
#     regime_features = market_regime(df['close'], windows=[20, 50, 200])
    
#     # Combine all features
#     features = pd.concat([
#         slopes,
#         volume_features,
#         breakout_features,
#         momentum_features,
#         regime_features
#     ], axis=1)
    
#     return features

# # Generate features
# features = generate_all_features(df)
# df = pd.concat([df, features], axis=1)

# # Fill NaN values (from calculation windows)
# df = df.fillna(method='bfill').fillna(0)

# # Machine Learning Integration
# # ===========================

# # Define a function to create lagged target for prediction
# def create_target(df, n_forward=5, threshold=0.01):
#     """
#     Creates target variable for ML prediction based on future returns
    
#     Parameters:
#     - df: DataFrame with price data
#     - n_forward: Number of periods to look ahead
#     - threshold: Return threshold for signal generation
    
#     Returns:
#     - Series with target labels: 1 (buy), -1 (sell), 0 (neutral)
#     """
#     # Calculate future returns
#     future_returns = df['close'].pct_change(n_forward).shift(-n_forward)
    
#     # Generate target signal
#     target = pd.Series(0, index=df.index)
#     target[future_returns > threshold] = 1     # Long signal
#     target[future_returns < -threshold] = -1   # Short signal
    
#     return target

# # Create target for different forward periods
# df['target_5d'] = create_target(df, n_forward=5, threshold=0.01)
# df['target_10d'] = create_target(df, n_forward=10, threshold=0.02)
# df['target_20d'] = create_target(df, n_forward=20, threshold=0.03)

# # Select features (excluding target, date, and original price data)
# feature_columns = [col for col in df.columns 
#                   if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 
#                                 'target_5d', 'target_10d', 'target_20d']]

# # Create training function for model
# def train_ml_model(df, feature_cols, target_col, test_size=0.2, random_state=42):
#     """
#     Trains ML model for position prediction
    
#     Parameters:
#     - df: DataFrame with features and target
#     - feature_cols: List of feature column names
#     - target_col: Target column name
#     - test_size: Proportion of data to use for testing
#     - random_state: Random seed for reproducibility
    
#     Returns:
#     - Trained model, feature importances, and performance metrics
#     """
#     # Prepare data
#     X = df[feature_cols].copy()
#     y = df[target_col].copy()
    
#     # Remove rows with NaN values in target
#     valid_idx = ~y.isna()
#     X = X[valid_idx]
#     y = y[valid_idx]
    
#     # Split data by time (not randomly)
#     split_idx = int(len(X) * (1 - test_size))
#     X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
#     y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
#     # Define model pipeline
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('model', GradientBoostingClassifier(random_state=random_state))
#     ])
    
#     # Train model
#     pipeline.fit(X_train, y_train)
    
#     # Make predictions
#     y_pred = pipeline.predict(X_test)
    
#     # Calculate metrics
#     metrics = {
#         'accuracy': accuracy_score(y_test, y_pred),
#         'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
#         'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
#         'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
#     }
    
#     # Get feature importances
#     feature_importances = pd.Series(
#         pipeline.named_steps['model'].feature_importances_,
#         index=feature_cols
#     ).sort_values(ascending=False)
    
#     return pipeline, feature_importances, metrics

# # Train model for each time horizon
# models = {}
# importances = {}
# metrics = {}

# for target in ['target_5d', 'target_10d', 'target_20d']:
#     models[target], importances[target], metrics[target] = train_ml_model(
#         df, feature_columns, target
#     )
#     print(f"\nModel metrics for {target}:")
#     for metric_name, value in metrics[target].items():
#         print(f"{metric_name}: {value:.4f}")

# # Generate model predictions
# for target in ['target_5d', 'target_10d', 'target_20d']:
#     df[f'pred_{target}'] = models[target].predict(df[feature_columns])

# # Create ensemble prediction
# df['ensemble_prediction'] = (
#     df['pred_target_5d'] * 0.5 + 
#     df['pred_target_10d'] * 0.3 + 
#     df['pred_target_20d'] * 0.2
# )

# # Apply decision threshold
# df['ml_signal'] = np.where(
#     df['ensemble_prediction'] > 0.5, 1,
#     np.where(df['ensemble_prediction'] < -0.5, -1, 0)
# )

# # Add technical confirmation filter (combine ML and technical signals)
# # 1. Slope direction confirms ML signal
# slope_confirm = df['slope_10'] * df['ml_signal'] > 0
# # 2. Breakout confirms ML signal
# breakout_confirm = (
#     (df['ml_signal'] > 0) & (df['upside_breakout_10'] > 0) |
#     (df['ml_signal'] < 0) & (df['downside_breakout_10'] < 0)
# )
# # 3. Regime is favorable for signal
# regime_confirm = (
#     (df['ml_signal'] > 0) & (df['trend_strength_20'] != 'range') |
#     (df['ml_signal'] < 0) & (df['trend_strength_20'] != 'range')
# )

# # Only take signals with sufficient confirmation
# df['confirmed_ml_signal'] = np.where(
#     (df['ml_signal'] != 0) & (slope_confirm | breakout_confirm) & regime_confirm,
#     df['ml_signal'],
#     0
# )

# # Generate position with filtering of low-confidence signals
# df['confidence'] = abs(df['ensemble_prediction'])
# high_confidence = df['confidence'] > 0.7

# # Create the position column first, then use it in the np.where logic
# df['position'] = 0  # Initialize with zeros

# # Then use the np.where logic
# df['position'] = np.where(
#     high_confidence,
#     df['confirmed_ml_signal'],
#     np.where(
#         df['confirmed_ml_signal'] == 0,
#         0,
#         df['position'].shift(1)  # Now this will work because the column exists
#     )
# )
# # Fill NaN values in position
# df['position'] = df['position'].fillna(0)

# # Apply risk management rules
# # 1. Exit signal based on stop-loss
# stop_loss_pct = 0.02  # 2% stop loss
# df['cum_return'] = (1 + df['close'].pct_change()).cumprod()
# df['trade_return'] = df['position'].shift(1) * df['close'].pct_change()
# df['trade_cumret'] = 0
# df['trade_cumret'] = np.where(df['position'] != df['position'].shift(1), 0, df['trade_cumret'].shift(1) + df['trade_return'])

# # Apply stop loss
# long_stop_loss = (df['position'] == 1) & (df['trade_cumret'] < -stop_loss_pct)
# short_stop_loss = (df['position'] == -1) & (df['trade_cumret'] < -stop_loss_pct)
# df.loc[long_stop_loss | short_stop_loss, 'position'] = 0

# # Calculate trade signals (for visualization)
# df['trade_signal'] = df['position'].diff()

# # Plotting results
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18))

# # Plot price and position
# ax1.plot(df['close'], label='Close Price', color='blue', alpha=0.7)

# # Highlight position periods
# long_periods = df['position'] == 1
# short_periods = df['position'] == -1

# # Create long/short background shading
# for i in range(len(df)):
#     if df['position'].iloc[i] == 1:
#         ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='green')
#     elif df['position'].iloc[i] == -1:
#         ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='red')

# # Mark entry and exit points
# long_entries = df[df['trade_signal'] == 1].index
# long_exits = df[(df['trade_signal'] < 0) & (df['position'].shift(1) == 1)].index
# short_entries = df[df['trade_signal'] == -1].index
# short_exits = df[(df['trade_signal'] > 0) & (df['position'].shift(1) == -1)].index

# ax1.plot(long_entries, df.loc[long_entries, 'close'], '^', color='green', markersize=10, label='Long Entry')
# ax1.plot(long_exits, df.loc[long_exits, 'close'], 'v', color='orange', markersize=8, label='Long Exit')
# ax1.plot(short_entries, df.loc[short_entries, 'close'], 'v', color='red', markersize=10, label='Short Entry')
# ax1.plot(short_exits, df.loc[short_exits, 'close'], '^', color='magenta', markersize=8, label='Short Exit')

# ax1.set_title('Price Action with ML-Enhanced Trend Signals')
# ax1.set_ylabel('Price')
# ax1.legend()

# # Plot ML signal and confidence
# ax2.bar(df.index, df['ml_signal'], color=np.where(df['ml_signal'] > 0, 'green', np.where(df['ml_signal'] < 0, 'red', 'gray')), alpha=0.7, label='ML Signal')
# ax2.plot(df.index, df['confidence'], 'b--', alpha=0.5, label='Signal Confidence')
# ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
# ax2.axhline(y=0.7, color='blue', linestyle='--', alpha=0.3, label='Confidence Threshold')
# ax2.set_title('Machine Learning Signal and Confidence')
# ax2.set_ylabel('Signal Strength')
# ax2.legend()

# # Plot feature importance
# top_features = importances['target_10d'].head(10)
# ax3.barh(top_features.index, top_features.values, color='skyblue')
# ax3.set_title('Top 10 Most Important Features')
# ax3.set_xlabel('Feature Importance')

# plt.tight_layout()
# plt.savefig('ml_enhanced_strategy.png')
# plt.show()

# # Calculate performance metrics
# df['returns'] = df['close'].pct_change()
# df['strategy_returns'] = df['position'].shift(1) * df['returns']

# # Calculate cumulative returns
# df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
# df['strategy_cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1

# # Plot performance comparison
# plt.figure(figsize=(15, 8))
# plt.plot(df['cumulative_returns'], label='Buy & Hold', color='blue')
# plt.plot(df['strategy_cumulative_returns'], label='ML Trend Strategy', color='green')
# plt.title('Strategy Performance Comparison')
# plt.xlabel('Time')
# plt.ylabel('Cumulative Returns')
# plt.legend()
# plt.grid(True)
# plt.savefig('ml_strategy_performance.png')
# plt.show()

# # Calculate drawdowns
# df['drawdown'] = df['strategy_cumulative_returns'].cummax() - df['strategy_cumulative_returns']
# max_drawdown = df['drawdown'].max()

# # Print performance statistics
# print("\nPerformance Metrics:")
# print(f"Total Return: {df['strategy_cumulative_returns'].iloc[-1]:.4f}")
# print(f"Annual Return: {df['strategy_returns'].mean() * 252:.4f}")
# print(f"Sharpe Ratio: {(df['strategy_returns'].mean() / df['strategy_returns'].std()) * np.sqrt(252):.4f}")
# print(f"Max Drawdown: {max_drawdown:.4f}")
# print(f"Win Rate: {(df['strategy_returns'] > 0).mean():.4f}")
# print(f"Average Winning Trade: {df['strategy_returns'][df['strategy_returns'] > 0].mean():.4f}")
# print(f"Average Losing Trade: {df['strategy_returns'][df['strategy_returns'] < 0].mean():.4f}")
# print(f"Profit Factor: {abs(df['strategy_returns'][df['strategy_returns'] > 0].sum() / df['strategy_returns'][df['strategy_returns'] < 0].sum()):.4f}")
# print(f"Number of Trades: {abs(df['trade_signal']).sum() / 2:.0f}")

# # # Save predictions for future use
# # df[['date', 'close', 'position', 'ml_signal', 'confidence']].to_csv('ml_predictions.csv', index=False)
# # print(df['position'])
# # print(df['ml_signal'])


# y = df["position"]




# ################################################################################### NEW CODE ###################################################################################
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import warnings
# from scipy.stats import linregress
# from scipy.signal import argrelextrema
# warnings.filterwarnings('ignore')


# # 📌 Importa i dati dal database
# from ..connection_database.login_mysql import DataProvider

# dp = DataProvider()
# da = dp.main()  
# df = da[0]

# # ✅ Ordina i dati per data
# df = df.sort_values(by="date").reset_index(drop=True)


# class OptimalPositionsAnalyzer:
#     def __init__(self, df):
#         """
#         Inizializza l'analizzatore con il dataframe contenente i dati di prezzo
        
#         Parameters:
#         df (pandas.DataFrame): DataFrame con colonne 'date', 'close', 'volume'
#         """
#         self.df = df.copy()
#         # Assicuriamoci che i dati siano ordinati per data
#         self.df = self.df.sort_values(by="date").reset_index(drop=True)
#         # Inizializza colonna delle posizioni
#         self.df['position'] = 0
        
#     def analyze(self):
#         """
#         Esegue l'analisi completa e genera le posizioni ottimali
        
#         Returns:
#         pandas.DataFrame: DataFrame originale con posizioni ottimali aggiunte
#         """
#         # Esegui tutte le analisi
#         self._identify_market_regime()
#         self._find_key_support_resistance()
#         self._analyze_price_patterns()
#         self._detect_volume_patterns()
#         self._find_optimal_positions()
        
#         return self.df
    
#     def _identify_market_regime(self, window=50):
#         """
#         Identifica il regime di mercato usando deviazione dalla distribuzione dei rendimenti
#         """
#         print("Identificando regime di mercato...")
#         prices = self.df['close']
#         returns = prices.pct_change()
        
#         # Calcola metriche di distribuzione
#         regime = pd.DataFrame(index=prices.index)
        
#         # Skewness (asimmetria) dei rendimenti
#         regime['skew'] = returns.rolling(window=window).skew()
        
#         # Kurtosis (curtosi) dei rendimenti
#         regime['kurt'] = returns.rolling(window=window).kurt()
        
#         # Variazione di volatilità
#         regime['vol'] = returns.rolling(window=window).std()
#         regime['vol_ratio'] = regime['vol'] / regime['vol'].rolling(window=window*2).mean()
        
#         # Autocorrelazione (misura il momentum/mean-reversion)
#         regime['autocorr'] = returns.rolling(window=window).apply(
#             lambda x: x.autocorr(lag=1) if len(x.dropna()) > 1 else np.nan, raw=False
#         )
        
#         # Regime classification basata su queste statistiche
#         # Positive autocorrelation = trend, Negative autocorrelation = mean-reversion
#         regime['regime'] = np.where(
#             regime['autocorr'] > 0.10, 1,  # regime di trend 
#             np.where(regime['autocorr'] < -0.10, -1, 0)  # regime di mean-reversion o neutrale
#         )
        
#         # Aggiungi al dataframe principale
#         self.df = pd.concat([self.df, regime], axis=1)
    
#     def _find_key_support_resistance(self, window=20, prominence=0.02):
#         """
#         Identifica livelli di supporto e resistenza usando i punti estremali locali
#         """
#         print("Calcolando livelli di supporto e resistenza...")
#         prices = self.df['close']
#         result = pd.DataFrame(index=prices.index)
        
#         # Normalizza i prezzi per lavorare con le variazioni percentuali
#         norm_prices = prices / prices.iloc[0]
        
#         # Trova massimi locali (resistenze potenziali)
#         max_idx = argrelextrema(norm_prices.values, np.greater, order=window)[0]
#         self.resistance_levels = pd.Series(index=prices.index, dtype=float)
#         self.resistance_levels.iloc[max_idx] = norm_prices.iloc[max_idx]
        
#         # Trova minimi locali (supporti potenziali)
#         min_idx = argrelextrema(norm_prices.values, np.less, order=window)[0]
#         self.support_levels = pd.Series(index=prices.index, dtype=float)
#         self.support_levels.iloc[min_idx] = norm_prices.iloc[min_idx]
        
#         # Calcola distanza dai livelli
#         if len(max_idx) > 0:
#             resistances = pd.Series(norm_prices.iloc[max_idx].values, index=max_idx)
            
#             # Per ogni prezzo, calcola la distanza dalla resistenza più vicina sopra di esso
#             dist_to_res = []
#             for i, p in enumerate(norm_prices):
#                 # Resistenze al di sopra del prezzo corrente
#                 res_above = resistances[resistances.index > i]
#                 if len(res_above) > 0:
#                     nearest_res = res_above.iloc[0]
#                     dist_pct = (nearest_res - p) / p
#                     dist_to_res.append(dist_pct)
#                 else:
#                     dist_to_res.append(np.nan)
#             result['dist_to_res'] = dist_to_res
        
#         if len(min_idx) > 0:
#             supports = pd.Series(norm_prices.iloc[min_idx].values, index=min_idx)
            
#             # Per ogni prezzo, calcola la distanza dal supporto più vicino sotto di esso
#             dist_to_sup = []
#             for i, p in enumerate(norm_prices):
#                 # Supporti al di sotto del prezzo corrente
#                 sup_below = supports[supports.index < i]
#                 if len(sup_below) > 0:
#                     nearest_sup = sup_below.iloc[-1]
#                     dist_pct = (p - nearest_sup) / p
#                     dist_to_sup.append(dist_pct)
#                 else:
#                     dist_to_sup.append(np.nan)
#             result['dist_to_sup'] = dist_to_sup
        
#         # Crea segnali basati sulla vicinanza ai livelli chiave
#         result['support_signal'] = np.where(
#             result['dist_to_sup'] < prominence, 1, 0  # Vicino al supporto = segnale rialzista
#         )
        
#         result['resistance_signal'] = np.where(
#             result['dist_to_res'] < prominence, -1, 0  # Vicino alla resistenza = segnale ribassista
#         )
        
#         # Aggiungi al dataframe principale
#         self.df = pd.concat([self.df, result], axis=1)
    
#     def _analyze_price_patterns(self):
#         """
#         Analizza pattern di prezzo basandosi su proprietà matematiche delle serie temporali
#         """
#         print("Analizzando pattern di prezzo...")
#         prices = self.df['close']
#         result = pd.DataFrame(index=prices.index)
        
#         # Log dei prezzi per lavorare con le variazioni proporzionali
#         log_prices = np.log(prices)
        
#         # Calcoliamo le variazioni scomponendole in frequenze diverse
#         returns = prices.pct_change()
        
#         # Deviazione dalla trend-line di lungo periodo (regressione lineare)
#         # usando finestre di diversa lunghezza
#         for window in [20, 50, 100]:
#             # Evita NaN all'inizio
#             if window >= len(prices):
#                 continue
                
#             # Per ogni punto, calcola la regressione lineare delle ultime 'window' osservazioni
#             deviations = []
#             slopes = []
            
#             for i in range(window, len(prices)):
#                 # Segmento di prezzo
#                 y = log_prices.iloc[i-window:i].values
#                 x = np.arange(window)
                
#                 # Calcola la regressione lineare
#                 slope, intercept, r_value, p_value, std_err = linregress(x, y)
                
#                 # Proiezione del trend line
#                 trend_line = intercept + slope * (window - 1)
                
#                 # Deviazione del prezzo attuale dalla trend-line
#                 actual = log_prices.iloc[i-1]
#                 deviation = actual - trend_line
                
#                 deviations.append(deviation)
#                 slopes.append(slope)
            
#             # Aggiungi padding per i valori iniziali
#             padding = [np.nan] * window
#             deviations = padding + deviations
#             slopes = padding + slopes
            
#             # Aggiungi al dataframe
#             result[f'dev_{window}'] = deviations
#             result[f'slope_{window}'] = slopes
            
#             # Normalizza la deviazione rispetto alla volatilità
#             rolling_std = log_prices.rolling(window=window).std()
#             result[f'norm_dev_{window}'] = result[f'dev_{window}'] / rolling_std
        
#         # Identifica pattern di inversione basati su statistiche
#         result['reversal_signal'] = 0
        
#         # Condizioni di ipercomprato/ipervenduto basate sulle deviazioni normalizzate
#         if 'norm_dev_50' in result.columns:
#             # Segnale di inversione rialzista: forte deviazione negativa + pendenza che inizia a salire
#             result['reversal_signal'] = np.where(
#                 (result['norm_dev_50'] < -2) & (result['slope_20'] > result['slope_20'].shift(1)),
#                 1,
#                 result['reversal_signal']
#             )
            
#             # Segnale di inversione ribassista: forte deviazione positiva + pendenza che inizia a scendere
#             result['reversal_signal'] = np.where(
#                 (result['norm_dev_50'] > 2) & (result['slope_20'] < result['slope_20'].shift(1)),
#                 -1,
#                 result['reversal_signal']
#             )
        
#         # Pattern di continuazione
#         result['continuation_signal'] = 0
        
#         # Segnale di continuazione rialzista: deviazione moderata e pendenza positiva
#         if 'norm_dev_50' in result.columns and 'slope_50' in result.columns:
#             result['continuation_signal'] = np.where(
#                 (result['norm_dev_50'] > 0) & (result['norm_dev_50'] < 1.5) & (result['slope_50'] > 0),
#                 1,
#                 result['continuation_signal']
#             )
            
#             # Segnale di continuazione ribassista: deviazione moderata e pendenza negativa
#             result['continuation_signal'] = np.where(
#                 (result['norm_dev_50'] < 0) & (result['norm_dev_50'] > -1.5) & (result['slope_50'] < 0),
#                 -1,
#                 result['continuation_signal']
#             )
        
#         # Aggiungi al dataframe principale
#         self.df = pd.concat([self.df, result], axis=1)
    
#     def _detect_volume_patterns(self):
#         """
#         Analizza i pattern di volume e le divergenze prezzo-volume
#         """
#         print("Analizzando pattern di volume...")
#         if 'volume' not in self.df.columns:
#             print("Dati di volume non disponibili, analisi volume saltata.")
#             return
            
#         result = pd.DataFrame(index=self.df.index)
#         prices = self.df['close']
#         volume = self.df['volume']
#         returns = prices.pct_change()
        
#         # Calcola volume relativo (rispetto alla media mobile)
#         rel_volume = volume / volume.rolling(window=50).mean()
#         result['rel_volume'] = rel_volume
        
#         # Volume crescente/decrescente (trend a 5 giorni)
#         result['volume_trend'] = rel_volume.diff(5)
        
#         # Divergenze tra prezzo e volume
#         price_direction = returns.rolling(window=5).mean()
#         volume_direction = result['volume_trend']
        
#         # Divergenze rialziste: prezzo scende ma volume diminuisce
#         result['bullish_divergence'] = np.where(
#             (price_direction < 0) & (volume_direction < 0),
#             1, 0
#         )
        
#         # Divergenze ribassiste: prezzo sale ma volume diminuisce
#         result['bearish_divergence'] = np.where(
#             (price_direction > 0) & (volume_direction < 0),
#             -1, 0
#         )
        
#         # Aggiungi al dataframe principale
#         self.df = pd.concat([self.df, result], axis=1)
    
#     def _find_optimal_positions(self):
#         """
#         Determina le posizioni ottimali basate su tutti i segnali calcolati
#         """
#         print("Calcolando posizioni ottimali...")
#         # Inizializza il segnale combinato
#         self.df['signal'] = 0
        
#         # 1. Dai priorità ai segnali di inversione (hanno maggiore precisione)
#         if 'reversal_signal' in self.df.columns:
#             self.df['signal'] = self.df['reversal_signal']
        
#         # 2. Se non c'è un segnale di inversione, considera supporti/resistenze
#         if 'support_signal' in self.df.columns and 'resistance_signal' in self.df.columns:
#             self.df['signal'] = np.where(
#                 self.df['signal'] == 0,
#                 self.df['support_signal'] + self.df['resistance_signal'],
#                 self.df['signal']
#             )
        
#         # 3. Se ancora non c'è un segnale e siamo in un regime di trend, usa i segnali di continuazione
#         if 'continuation_signal' in self.df.columns and 'regime' in self.df.columns:
#             self.df['signal'] = np.where(
#                 (self.df['signal'] == 0) & (self.df['regime'] == 1),
#                 self.df['continuation_signal'],
#                 self.df['signal']
#             )
        
#         # 4. Considera divergenze di volume (se disponibili)
#         if 'bullish_divergence' in self.df.columns and 'bearish_divergence' in self.df.columns:
#             self.df['signal'] = np.where(
#                 self.df['signal'] == 0,
#                 self.df['bullish_divergence'] + self.df['bearish_divergence'],
#                 self.df['signal']
#             )
        
#         # Applica un filtro di forza del segnale (riduce falsi segnali)
#         # Segnali devono coincidere con trend di prezzo locale
#         if 'slope_20' in self.df.columns:
#             # Indebolisci segnali che vanno contro il trend recente
#             self.df['signal'] = np.where(
#                 (self.df['signal'] > 0) & (self.df['slope_20'] < 0),
#                 0, # Riduci segnali long durante trend discendenti
#                 np.where(
#                     (self.df['signal'] < 0) & (self.df['slope_20'] > 0),
#                     0, # Riduci segnali short durante trend ascendenti
#                     self.df['signal']
#                 )
#             )
        
#         # Applica smoothing ai segnali (riduci segnali che durano solo un giorno)
#         self.df['signal_smooth'] = self.df['signal'].rolling(window=3, center=True).mean()
        
#         # Genera le posizioni ottimali finali
#         # Implementa una logica di posizione che tenga conto anche della redditività
#         # Logica: Mantenere la posizione finché non appare un segnale opposto
#         for i in range(1, len(self.df)):
#             # Se c'è un nuovo segnale forte, apri posizione
#             if abs(self.df['signal'].iloc[i]) >= 1:
#                 self.df.loc[self.df.index[i], 'position'] = np.sign(self.df['signal'].iloc[i])
#             elif self.df['signal'].iloc[i] != 0:
#                 # Se il segnale ha segno opposto rispetto alla posizione attuale, chiudi
#                 if self.df['signal'].iloc[i] * self.df['position'].iloc[i-1] < 0:
#                     self.df.loc[self.df.index[i], 'position'] = 0
#                 else:
#                     # Altrimenti mantieni la posizione corrente
#                     self.df.loc[self.df.index[i], 'position'] = self.df['position'].iloc[i-1]
#             else:
#                 # Se non ci sono segnali, mantieni la posizione precedente
#                 self.df.loc[self.df.index[i], 'position'] = self.df['position'].iloc[i-1]
        
#         # Ora miglioriamo le posizioni con un'ottimizzazione a posteriori
#         # Per ogni posizione, verifichiamo se è effettivamente redditizia
#         returns = self.df['close'].pct_change()
        
#         # Calcola rendimenti cumulativi per ogni posizione aperta
#         position_starts = self.df.index[self.df['position'].diff() != 0].tolist()
        
#         if len(position_starts) > 1:
#             for i in range(len(position_starts) - 1):
#                 start_idx = position_starts[i]
#                 end_idx = position_starts[i+1]
                
#                 if start_idx >= len(self.df) or end_idx >= len(self.df):
#                     continue
                
#                 position_type = self.df.loc[start_idx, 'position']
                
#                 # Calcola rendimento della posizione
#                 if position_type == 1:  # Long
#                     position_return = (self.df.loc[end_idx, 'close'] / self.df.loc[start_idx, 'close']) - 1
#                 elif position_type == -1:  # Short
#                     position_return = 1 - (self.df.loc[end_idx, 'close'] / self.df.loc[start_idx, 'close'])
#                 else:
#                     position_return = 0
                
#                 # Se la posizione è perdente, rimuoviamola (imposta a 0)
#                 if position_return < 0:
#                     self.df.loc[start_idx:end_idx, 'position'] = 0
        
#         # Finalizza le posizioni: riduciamo a +1 (compra) e -1 (vendi) solo
#         self.df['final_position'] = np.sign(self.df['position'])
        
#         # Calcola i cambi di posizione per visualizzazione
#         self.df['trade_signal'] = self.df['final_position'].diff()
    
#     def plot_results(self, filename=None):
#         """
#         Visualizza i risultati dell'analisi
        
#         Parameters:
#         filename (str): Nome del file per salvare il grafico (opzionale)
#         """
#         print("Creando grafici...")
#         # Crea il grafico
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
        
#         # Plot prezzo e posizione
#         ax1.plot(self.df['close'], label='Prezzo di Chiusura', color='blue', alpha=0.7)
        
#         # Evidenzia periodi con posizioni
#         for i in range(len(self.df)):
#             if self.df['final_position'].iloc[i] == 1:
#                 ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='green')
#             elif self.df['final_position'].iloc[i] == -1:
#                 ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='red')
        
#         # Contrassegna punti di ingresso
#         buy_entries = self.df[self.df['trade_signal'] == 1].index
#         sell_entries = self.df[self.df['trade_signal'] == -1].index
        
#         # Plotta i punti di ingresso/uscita
#         ax1.plot(buy_entries, self.df.loc[buy_entries, 'close'], '^', color='green', markersize=10, label='Posizione Long (+1)')
#         ax1.plot(sell_entries, self.df.loc[sell_entries, 'close'], 'v', color='red', markersize=10, label='Posizione Short (-1)')
        
#         # Plotta livelli chiave se disponibili
#         if hasattr(self, 'resistance_levels') and hasattr(self, 'support_levels'):
#             for idx in self.resistance_levels.dropna().index:
#                 if idx < len(self.df):
#                     ax1.axhline(y=self.df['close'].iloc[0] * self.resistance_levels[idx], 
#                                color='red', linestyle='--', alpha=0.3, label='_nolegend_')
            
#             for idx in self.support_levels.dropna().index:
#                 if idx < len(self.df):
#                     ax1.axhline(y=self.df['close'].iloc[0] * self.support_levels[idx], 
#                                color='green', linestyle='--', alpha=0.3, label='_nolegend_')
        
#         ax1.set_title('Posizioni Ottimali Basate su Pattern di Prezzo')
#         ax1.set_ylabel('Prezzo')
#         ax1.legend()
        
#         # Plot 2: Segnali e regime
#         if 'signal' in self.df.columns:
#             ax2.bar(self.df.index, self.df['signal'], 
#                    color=np.where(self.df['signal'] > 0, 'green', 
#                                  np.where(self.df['signal'] < 0, 'red', 'gray')), 
#                    alpha=0.7, label='Forza del Segnale')
        
#         if 'regime' in self.df.columns:
#             ax2.plot(self.df.index, self.df['regime'], 'b--', alpha=0.5, 
#                     label='Regime (1=Trend, -1=Mean-Reversion)')
        
#         ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
#         ax2.set_title('Segnali e Regime di Mercato')
#         ax2.set_ylabel('Intensità')
#         ax2.set_xlabel('Periodo')
#         ax2.legend()
        
#         plt.tight_layout()
        
#         if filename:
#             plt.savefig(filename)
        
#         plt.show()
    
#     def get_optimal_positions(self):
#         """
#         Restituisce solo le posizioni ottimali identificate
        
#         Returns:
#         pandas.DataFrame: DataFrame con data, prezzo e posizione
#         """
#         result = self.df[['date', 'close', 'final_position']].copy()
#         result.columns = ['Data', 'Prezzo', 'Posizione']
#         return result

# # Funzione principale per eseguire l'analisi
# def analyze_optimal_positions(df, plot=True, save_file=None):
#     """
#     Analizza un dataframe di dati di prezzo e trova le posizioni ottimali
    
#     Parameters:
#     df (pandas.DataFrame): DataFrame con colonne 'date', 'close', 'volume'
#     plot (bool): Se True, visualizza i grafici
#     save_file (str): Nome del file per salvare il grafico (opzionale)
    
#     Returns:
#     pandas.DataFrame: DataFrame con posizioni ottimali
#     """
#     # Verifica che i dati necessari siano presenti
#     required_columns = ['date', 'close']
#     for col in required_columns:
#         if col not in df.columns:
#             raise ValueError(f"Colonna '{col}' non trovata nel dataframe. Colonne richieste: {required_columns}")
    
#     # Inizializza l'analizzatore
#     analyzer = OptimalPositionsAnalyzer(df)
    
#     # Esegui l'analisi
#     result_df = analyzer.analyze()
    
#     # Visualizza i risultati
#     if plot:
#         analyzer.plot_results(filename=save_file)
    
#     # Restituisci le posizioni ottimali
#     return analyzer.get_optimal_positions()

# # Esempio di utilizzo:
# if __name__ == "__main__":
#     # Assumi che 'df' contenga già i dati nel formato corretto
#     # ad esempio, caricati da un file CSV o database
    
#     # Se si importano dati da un CSV:
#     # import pandas as pd
#     # df = pd.read_csv('dati_prezzi.csv')
    
#     # Esegui l'analisi
#     optimal_positions = analyze_optimal_positions(df, save_file='optimal_positions.png')
    
#     # Stampa le posizioni ottimali
#     print("\nPosizioni Ottimali Identificate:")
#     print(optimal_positions[optimal_positions['Posizione'] != 0])
    
#     # Salva il risultato su file
#     optimal_positions.to_csv('posizioni_ottimali.csv', index=False)