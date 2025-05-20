import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.dates as mdates


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
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    plt.show()


def predict_future(model, model_inputs, scaler, prediction_days):
    real_data = model_inputs[-prediction_days:]  # shape: (60, N)
    real_data = np.expand_dims(real_data, axis=0)  # shape: (1, 60, N)

    prediction = model.predict(real_data, verbose=0)
    
    # Inverte a escala apenas da coluna de interesse (ex: 'Close')
    full_input = np.zeros((1, scaler.scale_.shape[0]))
    full_input[0, 0] = prediction[0][0]
    return scaler.inverse_transform(full_input)[0][0]


def recursive_forecast(model, last_sequence, future_steps, scaler):
    predictions = []
    input_seq = last_sequence.copy()  # shape: (prediction_days, n_features)

    for _ in range(future_steps):
        input_seq_reshaped = np.expand_dims(input_seq, axis=0)  # (1, prediction_days, n_features)
        pred = model.predict(input_seq_reshaped, verbose=0)
        
        # Cria vetor com a mesma estrutura do scaler para inverter
        full_input = np.zeros((1, scaler.scale_.shape[0]))
        full_input[0, 0] = pred[0][0]
        pred_inversed = scaler.inverse_transform(full_input)[0][0]
        predictions.append([pred_inversed])

        # Adiciona a previsão como nova linha e remove a mais antiga
        next_step = np.zeros_like(input_seq[0])
        next_step[0] = pred[0][0]
        input_seq = np.vstack((input_seq[1:], next_step))

    return predictions