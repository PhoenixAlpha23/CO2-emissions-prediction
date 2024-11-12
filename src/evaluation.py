import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_statistics(y):
    """
    Calculate basic statistics of CO2 emissions.
    
    Args:
        y: Target variable (CO2 emissions)
    
    Returns:
        dict: Dictionary containing statistics
    """
    return {
        'min': y.min(),
        'max': y.max(),
        'mean': y.mean(),
        'std': y.std()
    }

def evaluate_model(y_true, y_pred):
    """
    Calculate various evaluation metrics for the model.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def print_evaluation_metrics(metrics):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
    """
    print("Evaluation Metrics:")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R2 Score: {metrics['r2']:.4f}")
