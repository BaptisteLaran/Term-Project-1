import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def read_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    return df

df_train = read_data('data/kospi_train.csv')
df_test = read_data('data/kospi_test.csv')


def create_lag_features(df, lags):
    df_lagged = df.copy()
    for i in range(1, lags + 1):
        df_lagged[f'Open_lag{i}'] = df_lagged['Open'].shift(i)
        df_lagged[f'High_lag{i}'] = df_lagged['High'].shift(i)
        df_lagged[f'Low_lag{i}'] = df_lagged['Low'].shift(i)
        df_lagged[f'Close_lag{i}'] = df_lagged['Close'].shift(i)
        df_lagged[f'Volume_lag{i}'] = df_lagged['Volume'].shift(i)
    df_lagged['Target'] = df_lagged['Close'].shift(-1)
    return df_lagged.dropna()

def cross_validate():
    best_score = float('inf')
    best_mae = float('inf')
    best_r2 = float('-inf')
    best_lag = None

    for lag in range(1, 11):
        df_lagged = create_lag_features(df_train, lags=lag)
        X_train = df_lagged[[col for col in df_lagged.columns if 'lag' in col]]
        y_train = df_lagged['Target']
        tscv = TimeSeriesSplit(n_splits=5)
        
        model_lr = LinearRegression()
        scores_lr = cross_val_score(model_lr, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
        model_lr.fit(X_train, y_train)
        rmse_lr = np.sqrt(-scores_lr.mean())
        mae_lr = np.mean(np.abs(scores_lr))
        r2_lr = r2_score(y_train, model_lr.predict(X_train))
        
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        scores_rf = cross_val_score(model_rf, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
        model_rf.fit(X_train, y_train)
        rmse_rf = np.sqrt(-scores_rf.mean())
        mae_rf = np.mean(np.abs(scores_rf))
        r2_rf = r2_score(y_train, model_rf.predict(X_train))

        model_svr = SVR(kernel='poly', C=10, epsilon=0.5)
        scores_svr = cross_val_score(model_svr, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
        model_svr.fit(X_train, y_train)
        rmse_svr = np.sqrt(-scores_svr.mean())
        mae_svr = np.mean(np.abs(scores_svr))
        r2_svr = r2_score(y_train, model_svr.predict(X_train))
        
        print(f'Lags={lag}, CV RMSE (LinearRegression)={rmse_lr:.2f}, MAE={mae_lr:.2f}, R²={r2_lr:.4f}')
        print(f'Lags={lag}, CV RMSE (RandomForest)={rmse_rf:.2f}, MAE={mae_rf:.2f}, R²={r2_rf:.4f}')
        print(f'Lags={lag}, CV RMSE (SVR)={rmse_svr:.2f}, MAE={mae_svr:.2f}, R²={r2_svr:.4f}')
        
        if rmse_lr < best_score and rmse_lr < best_mae and r2_lr > best_r2:
            best_score = rmse_lr
            best_mae = mae_lr
            best_r2 = r2_lr
            best_lag = lag
            best_model = "LinearRegression"

        if rmse_rf < best_score and rmse_rf < best_mae and r2_rf > best_r2:
            best_score = rmse_rf
            best_mae = mae_rf
            best_r2 = r2_rf
            best_lag = lag
            best_model = "RandomForest"

        if rmse_svr < best_score and rmse_svr < best_mae and r2_svr > best_r2:
            best_score = rmse_svr
            best_mae = mae_svr
            best_r2 = r2_svr
            best_lag = lag
            best_model = "SVR"
    
    print(f'Best lag: {best_lag}, with RMSE: {best_score:.2f}, MAE: {best_mae:.2f}, R²: {best_r2:.4f}')
    print(f'Best model: {best_model}')
    return best_lag, best_model


lags, best_model = 1, "LinearRegression"  # cross_validate()
df_lagged = create_lag_features(df_train, lags)
X_train = df_lagged[[col for col in df_lagged.columns if 'lag' in col]]
y_train = df_lagged['Target']


if best_model == "LinearRegression":
    model = LinearRegression()
    model.fit(X_train, y_train)
elif best_model == "RandomForest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
elif best_model == "SVR":
    model = SVR(kernel='poly', C=10, epsilon=0.5)
    model.fit(X_train, y_train)


def generate_lag_row(df, lags):
    lag_data = {}
    for i in range(1, lags + 1):
        lag_data[f'Open_lag{i}'] = df.iloc[-i]['Open']
        lag_data[f'High_lag{i}'] = df.iloc[-i]['High']
        lag_data[f'Low_lag{i}'] = df.iloc[-i]['Low']
        lag_data[f'Close_lag{i}'] = df.iloc[-i]['Close']
        lag_data[f'Volume_lag{i}'] = df.iloc[-i]['Volume']
    return pd.DataFrame([lag_data])

history = df_train.copy()
predictions = []

for i in range(len(df_test)):
    X_next = generate_lag_row(history, lags)
    y_next = model.predict(X_next)[0]
    predictions.append(y_next)

    next_row = df_test.iloc[i].copy()
    next_row['Close'] = y_next
    history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)


y_test_true = df_test['Close'].iloc[1:].reset_index(drop=True)
y_pred = predictions[:-1]

rmse = mean_squared_error(y_test_true, y_pred)
mae = mean_absolute_error(y_test_true, y_pred)
r2 = r2_score(y_test_true, y_pred)

print(f'Test RMSE: {rmse:.2f}')
print(f'Test MAE: {mae:.2f}')
print(f'Test R²: {r2:.4f}')


plt.figure(figsize=(12, 6))
plt.plot(y_test_true.values, label='Actual Close', marker='o')
plt.plot(y_pred, label='Predicted Close', marker='x')
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Time Step')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
