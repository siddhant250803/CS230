"""Create baseline linear regression model for comparison."""

import numpy as np
from sklearn.linear_model import LinearRegression


def train_linear_regression_baseline(X_volatility_train: np.ndarray, y_train: np.ndarray, 
                                     target_stock_idx: int, lookback: int = 12):
    """
    Train a linear regression model using previous lookback volatilities.
    
    Args:
        X_volatility_train: Training volatility sequences (n_train, lookback, n_stocks)
        y_train: Training targets (n_train, n_stocks)
        target_stock_idx: Index of target stock
        lookback: Number of previous volatilities to use
        
    Returns:
        Trained LinearRegression model
    """
    # Extract target stock's volatility history and targets
    X_train = X_volatility_train[:, :, target_stock_idx]  # (n_train, lookback)
    y_train_target = y_train[:, target_stock_idx]  # (n_train,)
    
    # Fit linear regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train_target)
    
    return lr_model


def predict_linear_regression(lr_model: LinearRegression, X_volatility: np.ndarray, 
                              target_stock_idx: int) -> np.ndarray:
    """
    Make predictions using linear regression model.
    
    Args:
        lr_model: Trained LinearRegression model
        X_volatility: Volatility sequences (n_samples, lookback, n_stocks)
        target_stock_idx: Index of target stock
        
    Returns:
        Predictions (n_samples)
    """
    X = X_volatility[:, :, target_stock_idx]  # (n_samples, lookback)
    return lr_model.predict(X)

