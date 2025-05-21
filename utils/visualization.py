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

def predict_future(model, model_inputs, scaler, prediction_days):
    real_data = model_inputs[-prediction_days:]
    real_data = np.expand_dims(real_data, axis=0)
    prediction = model.predict(real_data, verbose=0)
    full_input = np.zeros((1, scaler.scale_.shape[0]))
    full_input[0, 0] = prediction[0][0]
    return scaler.inverse_transform(full_input)[0][0]

def recursive_forecast(model, last_sequence, future_steps, scaler):
    predictions = []
    input_seq = last_sequence.copy()
    for _ in range(future_steps):
        input_seq_reshaped = np.expand_dims(input_seq, axis=0)
        pred = model.predict(input_seq_reshaped, verbose=0)
        full_input = np.zeros((1, scaler.scale_.shape[0]))
        full_input[0, 0] = pred[0][0]
        pred_inversed = scaler.inverse_transform(full_input)[0][0]
        predictions.append([pred_inversed])
        next_step = np.zeros_like(input_seq[0])
        next_step[0] = pred[0][0]
        input_seq = np.vstack((input_seq[1:], next_step))
    return predictions
