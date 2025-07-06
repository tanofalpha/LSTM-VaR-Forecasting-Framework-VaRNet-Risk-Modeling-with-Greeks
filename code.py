import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. Download NIFTYBEES data
ticker = yf.download('NIFTYBEES.NS', start='2015-01-01', end='2023-01-01')
close = ticker['Close']
returns = close.pct_change()

# 2. Compute basic market features
data = pd.DataFrame()
data['returns'] = returns
data['volatility'] = returns.rolling(20).std()
data['momentum'] = close.pct_change(10)

# 3. Compute implied volatility proxy (rolling std)
sigma_series = returns.rolling(20).std() * np.sqrt(252)
sigma_series = sigma_series.dropna().values

# 4. Simulate constant YTM curve with shock
min_len = len(sigma_series)
S_series = close[-min_len:].values
ytm_series = 0.06 + 0.004 * np.cos(np.linspace(0, 2 * np.pi, min_len))
ytm_series[80:100] += 0.01  # market shock

# 5. Black-Scholes and bond pricing functions
def bs_delta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)

def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T)/(sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def bs_gamma(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.pdf(d1)/(S * sigma * np.sqrt(T))

def bond_price(face, rate, T, ytm):
    return sum([face * rate * np.exp(-ytm*t) for t in range(1, T+1)]) + face * np.exp(-ytm*T)

def dv01(face, rate, T, ytm):
    price = bond_price(face, rate, T, ytm)
    price_up = bond_price(face, rate, T, ytm + 0.0001)
    return price - price_up

# 6. Compute Greeks + DV01
call = {'K': 200, 'T': 0.5, 'r': 0.05, 'option_type': 'call'}
bond = {'face': 1000, 'rate': 0.06, 'T': 5}

greeks_dv01_data = []
for i in range(min_len):
    S = S_series[i]
    sigma = sigma_series[i]
    ytm = ytm_series[i]
    
    delta = bs_delta(S, call['K'], call['T'], call['r'], sigma, call['option_type'])
    vega = bs_vega(S, call['K'], call['T'], call['r'], sigma)
    gamma = bs_gamma(S, call['K'], call['T'], call['r'], sigma)
    dv01_val = dv01(bond['face'], bond['rate'], bond['T'], ytm)
    
    greeks_dv01_data.append([delta, vega, gamma, dv01_val])

greeks_df = pd.DataFrame(greeks_dv01_data, columns=["Delta", "Vega", "Gamma", "DV01"])

# 7. Align market + greeks data
market_df = data.dropna().iloc[-min_len:].reset_index(drop=True)
df_full = pd.concat([market_df.reset_index(drop=True), greeks_df.reset_index(drop=True)], axis=1)

# 8. Compute realistic VaR as target
def historical_var(returns, window=252, confidence=0.95):
    var_list = []
    for i in range(window, len(returns)):
        var = -np.percentile(returns[i - window:i], (1 - confidence) * 100)
        var_list.append(var)
    return np.array(var_list)

returns_clean = df_full['returns'].values
hist_var = historical_var(returns_clean)

# Align dataset
df_full = df_full.iloc[-len(hist_var):].copy()
df_full['VaR'] = hist_var

# 9. Prepare LSTM sequences
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_full.drop(columns=['VaR']))

sequence_length = 10
X_seq, y_seq = [], []

for i in range(len(X_scaled) - sequence_length):
    X_seq.append(X_scaled[i:i + sequence_length])
    y_seq.append(df_full['VaR'].iloc[i + sequence_length])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# 10. Build and train LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, 7)),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, callbacks=[early_stop], verbose=1)

# 11. Evaluation
y_pred = model.predict(X_test).flatten()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f" MSE: {mse:.6f}")
print(f" MAE: {mae:.6f}")
print(f" RMSE: {rmse:.6f}")
print(f" R² Score: {r2:.4f}")

# 12. Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test[:100], label='Actual VaR')
plt.plot(y_pred[:100], label='Predicted VaR')
plt.title('Hybrid LSTM VaR Prediction (NIFTYBEES + Greeks)')
plt.xlabel('Sample')
plt.ylabel('VaR')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
