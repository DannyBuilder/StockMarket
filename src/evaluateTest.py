from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def print_evaluation(evaluation):
    print(f"MAE: {evaluation['MAE']:.2f} USD")
    print(f"RMSE: {evaluation['RMSE']:.2f} USD")
    print(f"R^2 Score: {evaluation['R2']:.4f}")
