from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    print(f"\n📊 Avaliação do modelo:")
    print(f" - MSE: {mse:.4f}")
    print(f" - MAE: {mae:.4f}")
    print(f" - R²: {r2:.4f}")