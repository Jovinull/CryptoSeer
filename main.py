import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# === CONFIGURAÇÕES ===
crypto_currency = 'DOGE'
against_currency = "USD"
start = "2014-01-01"
end = dt.datetime.now().strftime("%Y-%m-%d")
prediction_days = 60
future_day = 30

# === OBTENDO DADOS ===
ticker = f"{crypto_currency}-{against_currency}"
data = yf.download(ticker, start=start, end=end)

# === ESCALONAMENTO (usando fit apenas uma vez) ===
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data['Close'].values.reshape(-1, 1))
scaled_data = scaler.transform(data['Close'].values.reshape(-1, 1))

# === TREINAMENTO ===
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data) - future_day):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x + future_day, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# === TESTES ===
test_start = dt.datetime(2018, 1, 1)
test_end = dt.datetime.now()

test_data = yf.download(ticker, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

# Reutilizando o mesmo scaler
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

# === AVALIAÇÃO QUANTITATIVA ===
mse = mean_squared_error(actual_prices, prediction_prices)
mae = mean_absolute_error(actual_prices, prediction_prices)
r2 = r2_score(actual_prices, prediction_prices)

print(f"\nAvaliação do modelo:")
print(f" - Mean Squared Error (MSE): {mse:.4f}")
print(f" - Mean Absolute Error (MAE): {mae:.4f}")
print(f" - R² Score: {r2:.4f}")

# === PLOTAGEM COM DATAS ===
plt.figure(figsize=(12,6))
plt.plot(test_data.index, actual_prices, color='black', label='Actual Prices')
plt.plot(test_data.index, prediction_prices, color='green', label='Predicted Prices')
plt.title(f'{crypto_currency} Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === PREVISÃO DE UM ÚNICO PONTO FUTURO ===
real_data = model_inputs[-prediction_days:]
real_data = np.array([real_data])
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

future_prediction = model.predict(real_data)
future_prediction = scaler.inverse_transform(future_prediction)

print(f"\nPrevisão para {future_day} dias à frente: ${future_prediction[0][0]:.2f}")
