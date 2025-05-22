import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def plot_predictions(dates, actual, predicted, label):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, color='black', label='Actual Prices')
    plt.plot(dates, predicted, color='green', label='Predicted Prices')
    plt.title(f'{label} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.tight_layout()
    plt.show()

def plot_residuals(dates, actual, predicted):
    residuals = np.array(actual) - np.array(predicted)
    plt.figure(figsize=(12, 4))
    plt.plot(dates, residuals, color='red', label='Residuals')
    plt.axhline(0, linestyle='--', color='black')
    plt.title("Gráfico de Resíduos")
    plt.xlabel('Date')
    plt.ylabel('Erro (Real - Previsto)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def predict_future(model, model_inputs, scaler, prediction_days, y_scaler, base_price):
    real_data = model_inputs[-prediction_days:]
    real_data = np.expand_dims(real_data, axis=0)
    delta_pred = model.predict(real_data, verbose=0)[0][0]
    delta_pred = y_scaler.inverse_transform([[delta_pred]])[0][0]
    return base_price * (1 + delta_pred)

def recursive_forecast(model, last_sequence, future_steps, scaler, y_scaler, base_price):
    predictions = []
    input_seq = last_sequence.copy()
    current_price = base_price

    for _ in range(future_steps):
        input_seq_reshaped = np.expand_dims(input_seq, axis=0)
        delta_pred = model.predict(input_seq_reshaped, verbose=0)[0][0]
        delta_pred = y_scaler.inverse_transform([[delta_pred]])[0][0]
        next_price = current_price * (1 + delta_pred)
        predictions.append([next_price])
        current_price = next_price

        next_step = np.zeros_like(input_seq[0])
        next_step[0] = delta_pred  # ainda em escala percentual
        input_seq = np.vstack((input_seq[1:], next_step))

    return predictions
