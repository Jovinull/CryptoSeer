import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    print(f"\n📊 Avaliação do modelo:")
    print(f" - MSE: {mse:.4f}")
    print(f" - MAE: {mae:.4f}")
    print(f" - R²: {r2:.4f}")


def plot_predictions(dates, actual, predicted, label):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, color='black', label='Actual Prices')
    plt.plot(dates, predicted, color='green', label='Predicted Prices')
    plt.title(f'{label} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def predict_future(model, model_inputs, scaler, prediction_days):
    real_data = model_inputs[-prediction_days:]
    real_data = np.array([real_data])
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    prediction = model.predict(real_data)
    return scaler.inverse_transform(prediction)[0][0]


def recursive_forecast(model, last_sequence, future_steps, scaler):
    predictions = []
    input_seq = last_sequence.copy()

    for _ in range(future_steps):
        input_seq_reshaped = input_seq.reshape(1, input_seq.shape[0], 1)
        pred = model.predict(input_seq_reshaped, verbose=0)
        predictions.append(pred[0, 0])
        input_seq = np.append(input_seq[1:], pred[0, 0])

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))