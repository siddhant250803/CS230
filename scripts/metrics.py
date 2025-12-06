"""
Evaluation metrics for volatility prediction.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        
    Returns:
        Dictionary of metrics
    """
    # Ensure 1D arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    # MAE
    mae = mean_absolute_error(targets, predictions)
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

