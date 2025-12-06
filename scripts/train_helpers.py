"""
Helper functions for training DeltaLag attention model.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer import SimpleTrainer
from baseline import predict_linear_regression


def evaluate_model(trainer: SimpleTrainer, val_loader: DataLoader, target_idx: int):
    """
    Evaluate model on validation set and return predictions.
    
    Args:
        trainer: SimpleTrainer instance
        val_loader: Validation DataLoader
        target_idx: Target stock index
        
    Returns:
        Tuple of (val_predictions, val_targets, X_volatility_val)
    """
    trainer.model.eval()
    val_predictions = []
    val_targets = []
    X_volatility_val_list = []
    
    with torch.no_grad():
        for X_returns, X_volatility, y in val_loader:
            X_returns = X_returns.to(trainer.device)
            X_volatility = X_volatility.to(trainer.device)
            preds, _ = trainer.model(
                X_returns, X_volatility, target_idx, return_attention=False
            )
            val_predictions.append(preds.squeeze().cpu().numpy())
            val_targets.append(y.numpy())
            X_volatility_val_list.append(X_volatility.cpu().numpy())
    
    return (
        np.concatenate(val_predictions),
        np.concatenate(val_targets),
        np.concatenate(X_volatility_val_list, axis=0)
    )


def extract_attention_info(trainer: SimpleTrainer, val_loader: DataLoader, target_idx: int):
    """
    Extract attention weights and lag probabilities from model.
    
    Args:
        trainer: SimpleTrainer instance
        val_loader: Validation DataLoader
        target_idx: Target stock index
        
    Returns:
        Tuple of (attention_weights, lag_probs, expected_lags)
    """
    trainer.model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        X_ret_sample = sample_batch[0][:100].to(trainer.device)
        X_vol_sample = sample_batch[1][:100].to(trainer.device)
        
        _, attention_info = trainer.model(
            X_ret_sample, X_vol_sample, target_idx, return_attention=True
        )
        
        attention_weights = attention_info['attention_weights'].cpu().numpy()
        lag_probs = attention_info['lag_probs'].cpu().numpy()
        expected_lags = attention_info['expected_lags']
    
    return attention_weights, lag_probs, expected_lags


def calculate_baseline_metrics(lr_predictions: np.ndarray, val_targets: np.ndarray) -> dict:
    """
    Calculate metrics for linear regression baseline.
    
    Args:
        lr_predictions: Linear regression predictions
        val_targets: Validation targets
        
    Returns:
        Dictionary of metrics
    """
    lr_mse = np.mean((lr_predictions - val_targets) ** 2)
    lr_rmse = np.sqrt(lr_mse)
    lr_mae = np.mean(np.abs(lr_predictions - val_targets))
    lr_r2 = 1 - np.sum((val_targets - lr_predictions) ** 2) / np.sum(
        (val_targets - np.mean(val_targets)) ** 2
    )
    lr_mean_abs_percent_error = np.mean(
        np.abs((lr_predictions - val_targets) / (val_targets + 1e-10))
    ) * 100
    
    return {
        'mse': lr_mse,
        'rmse': lr_rmse,
        'mae': lr_mae,
        'r2': lr_r2,
        'mean_abs_percent_error': lr_mean_abs_percent_error
    }

