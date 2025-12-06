"""
Train DeltaLag attention model and save results for analysis.
"""

import os
import sys
import pickle
import numpy as np
from torch.utils.data import DataLoader

# Ensure scripts directory is on sys.path
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from config import Config
from dataloader import StockDataLoader, VolatilityDataset
from trainer import SimpleTrainer
from utils import build_stock_universe, calculate_train_end_idx
from baseline import train_linear_regression_baseline, predict_linear_regression
from train_helpers import evaluate_model, extract_attention_info, calculate_baseline_metrics


def train_deltalag_model(
    target_stock: str = "APA",
    other_stocks: list = None,
    market_stock: str = "SPY",
    epochs: int = 30,
    dropout: float = 0.0,
    mlp_hidden1: int = 128,
    mlp_hidden2: int = 64,
    attention_entropy_weight: float = 0.001,
    results_dir: str = "results",
):
    """Train DeltaLag attention model and save results for analysis."""
    if other_stocks is None:
        # 7 oil + 7 tech stocks
        other_stocks = [
            "XOM", "CVX", "SLB", "HAL", "BP", "SHEL", "COP",  # Oil
            "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "ADBE"  # Tech
        ]
    
    print("=" * 80)
    print(f"TRAINING DELTALAG ATTENTION MODEL: {target_stock} Volatility Prediction")
    print("=" * 80)
    print(f"\nTarget stock: {target_stock}")
    print(f"Other stocks: {', '.join(other_stocks)}")
    print(f"Market proxy: {market_stock}")
    print(f"Epochs: {epochs}")
    print(f"Dropout: {dropout}")
    print(f"MLP hidden sizes: [{mlp_hidden1}, {mlp_hidden2}]")
    print(f"Attention entropy weight: {attention_entropy_weight}")
    print("=" * 80)
    
    # Load data
    config = Config()
    config.NUM_EPOCHS = epochs
    data_loader = StockDataLoader(config.DATA_PATH, config)
    print("\nStep 1: Loading data...")
    data_loader.load_data()
    
    # Build stock universe
    stock_universe, sector_labels = build_stock_universe(
        data_loader, target_stock, other_stocks, market_stock
    )
    
    # Subset data (dates are preserved - they're per row, not per stock)
    indices = [data_loader.stock_names.index(s) for s in stock_universe]
    data_loader.stock_names = stock_universe
    data_loader.returns = data_loader.returns[:, indices]
    
    # Preprocess
    print("\nStep 2: Computing realized volatility...")
    data_loader.compute_realized_volatility()
    
    n_timesteps = len(data_loader.returns)
    
    # Calculate training end index for normalization
    train_end_idx = calculate_train_end_idx(data_loader, n_timesteps)
    data_loader.normalize_data(train_end_idx=train_end_idx)
    
    print("\nStep 4: Creating sequences...")
    X_returns, X_volatility, y = data_loader.create_sequences()
    
    sequence_start_idx = data_loader.config.LOOKBACK_WINDOW
    
    print("\nStep 5: Splitting data...")
    train_data, val_data, test_data = data_loader.split_data(
        X_returns, X_volatility, y, 
        use_recent_split=True,
        sequence_start_idx=sequence_start_idx
    )
    
    # Get target index
    target_idx = stock_universe.index(target_stock)
    
    # Create datasets
    train_dataset = VolatilityDataset(*train_data, target_stock_idx=target_idx)
    val_dataset = VolatilityDataset(*val_data, target_stock_idx=target_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Create test dataset and loader
    test_dataset = VolatilityDataset(*test_data, target_stock_idx=target_idx)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Train model
    print("\nStep 6: Training model...")
    trainer = SimpleTrainer(config, target_idx, target_stock)
    trainer.attention_entropy_weight = attention_entropy_weight
    trainer.setup_model(n_stocks=len(stock_universe), dropout=dropout, 
                        mlp_hidden1=mlp_hidden1, mlp_hidden2=mlp_hidden2)
    history = trainer.train(train_loader, val_loader)
    
    # Train linear regression baseline
    print("\n" + "=" * 80)
    print("TRAINING LINEAR REGRESSION BASELINE")
    print("=" * 80)
    X_volatility_train, y_train = train_data[1], train_data[2]
    lr_model = train_linear_regression_baseline(
        X_volatility_train, y_train, target_idx, lookback=config.LOOKBACK_WINDOW
    )
    
    # Final validation evaluation
    print("\n" + "=" * 80)
    print("FINAL VALIDATION RESULTS")
    print("=" * 80)
    val_losses, val_metrics = trainer.validate(val_loader)
    print(f"Validation MSE:  {val_losses['mse_loss']:.6f}")
    print(f"Validation RMSE: {val_metrics['rmse']:.6f}")
    print(f"Validation MAE:  {val_metrics['mae']:.6f}")
    
    # Get validation predictions
    val_predictions, val_targets, X_volatility_val = evaluate_model(trainer, val_loader, target_idx)
    
    # Get test predictions
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    test_losses, test_metrics = trainer.validate(test_loader)
    print(f"Test MSE:  {test_losses['mse_loss']:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Test MAE:  {test_metrics['mae']:.6f}")
    
    test_predictions, test_targets, X_volatility_test = evaluate_model(trainer, test_loader, target_idx)
    
    # Get linear regression predictions and metrics for validation
    lr_predictions = predict_linear_regression(lr_model, X_volatility_val, target_idx)
    lr_metrics = calculate_baseline_metrics(lr_predictions, val_targets)
    nn_mean_abs_percent_error = np.mean(np.abs((val_predictions - val_targets) / (val_targets + 1e-10))) * 100
    
    # Get linear regression predictions and metrics for test
    lr_test_predictions = predict_linear_regression(lr_model, X_volatility_test, target_idx)
    lr_test_metrics = calculate_baseline_metrics(lr_test_predictions, test_targets)
    
    print(f"\nLinear Regression Baseline Results:")
    print(f"  Validation MSE:  {lr_metrics['mse']:.6f}")
    print(f"  Validation RMSE: {lr_metrics['rmse']:.6f}")
    print(f"  Validation MAE:  {lr_metrics['mae']:.6f}")
    print(f"  Validation R²:   {lr_metrics['r2']:.6f}")
    print(f"  Mean |% Error|:   {lr_metrics['mean_abs_percent_error']:.2f}%")
    print(f"\nNeural Network Results:")
    print(f"  Mean |% Error|:   {nn_mean_abs_percent_error:.2f}%")
    
    # Extract attention information
    attention_weights, lag_probs, expected_lags = extract_attention_info(trainer, val_loader, target_idx)
    
    print(f"\n{'─'*80}")
    print("LAG DISTRIBUTION SUMMARY")
    print(f"{'─'*80}")
    for i, stock in enumerate(stock_universe):
        print(f"{stock}: Expected lag = {expected_lags[i]:.2f} intervals")
        print(f"  Top 3 lags: {np.argsort(lag_probs[i])[-3:][::-1]} "
              f"with probs {lag_probs[i][np.argsort(lag_probs[i])[-3:][::-1]]}")
    
    # Save results for analysis
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/{target_stock}_deltalag_results.pkl"
    
    results = {
        'target_stock': target_stock,
        'stock_universe': stock_universe,
        'sector_labels': sector_labels,
        'target_idx': target_idx,
        'config': {
            'max_lag': config.MAX_LAG,
            'lookback_window': config.LOOKBACK_WINDOW,
        },
        'history': history,
        'val_predictions': val_predictions,
        'val_targets': val_targets,
        'test_predictions': test_predictions,
        'test_targets': test_targets,
        'lr_predictions': lr_predictions,
        'lr_test_predictions': lr_test_predictions,
        'attention_weights': attention_weights,
        'lag_probs': lag_probs,
        'expected_lags': expected_lags,
        'metrics': {
            'nn_mse': float(val_losses['mse_loss']),
            'nn_rmse': float(val_metrics['rmse']),
            'nn_mae': float(val_metrics['mae']),
            'nn_mean_abs_percent_error': float(nn_mean_abs_percent_error),
            'nn_test_mse': float(test_losses['mse_loss']),
            'nn_test_rmse': float(test_metrics['rmse']),
            'nn_test_mae': float(test_metrics['mae']),
            'lr_mse': float(lr_metrics['mse']),
            'lr_rmse': float(lr_metrics['rmse']),
            'lr_mae': float(lr_metrics['mae']),
            'lr_r2': float(lr_metrics['r2']),
            'lr_mean_abs_percent_error': float(lr_metrics['mean_abs_percent_error']),
            'lr_test_mse': float(lr_test_metrics['mse']),
            'lr_test_rmse': float(lr_test_metrics['rmse']),
            'lr_test_mae': float(lr_test_metrics['mae']),
        },
        'model_path': f"checkpoints/{target_stock}_deltalag_best.pt",
    }
    
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {results_file}")
    print(f"Model saved to: checkpoints/{target_stock}_deltalag_best.pt")
    print(f"\nRun analyze_deltalag_results.py to generate plots and analysis.")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train DeltaLag attention model"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="APA",
        help="Target stock to predict (default: APA)",
    )
    parser.add_argument(
        "--stocks",
        type=str,
        default="XOM,CVX,SLB,HAL,BP,SHEL,COP,AAPL,MSFT,GOOGL,NVDA,AMZN,META,ADBE",
        help="Comma-separated list of other stock tickers (default: 7 oil + 7 tech)",
    )
    parser.add_argument(
        "--market",
        type=str,
        default="SPY",
        help="Market proxy stock (default: SPY)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate (default: 0.0)",
    )
    parser.add_argument(
        "--mlp_hidden1",
        type=int,
        default=128,
        help="First MLP hidden layer size (default: 128)",
    )
    parser.add_argument(
        "--mlp_hidden2",
        type=int,
        default=64,
        help="Second MLP hidden layer size (default: 64)",
    )
    parser.add_argument(
        "--attention_entropy_weight",
        type=float,
        default=0.001,
        help="Weight for attention entropy penalty (encourages balanced attention, default: 0.001, set to 0 to disable)",
    )
    
    args = parser.parse_args()
    
    stock_list = [s.strip() for s in args.stocks.split(",") if s.strip()]
    
    train_deltalag_model(
        target_stock=args.target,
        other_stocks=stock_list,
        market_stock=args.market,
        epochs=args.epochs,
        dropout=args.dropout,
        mlp_hidden1=args.mlp_hidden1,
        mlp_hidden2=args.mlp_hidden2,
        attention_entropy_weight=args.attention_entropy_weight,
    )

