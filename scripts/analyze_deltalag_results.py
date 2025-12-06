"""
Analyze DeltaLag attention model results and generate visualizations.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Ensure scripts directory is on sys.path
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from plots import (
    plot_lag_probability_distributions,
    plot_averaged_lag_distribution,
    plot_attention_weights,
    plot_loss_history,
    plot_test_predictions_and_accuracy,
)


def analyze_deltalag_results(
    results_file: str,
    output_dir: str = "results",
    plot_dir: str = "results",
):
    """Analyze DeltaLag attention model results and generate visualizations."""
    
    # Load results
    print("=" * 80)
    print("ANALYZING DELTALAG ATTENTION MODEL RESULTS")
    print("=" * 80)
    print(f"\nLoading results from: {results_file}")
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    target_stock = results['target_stock']
    stock_universe = results['stock_universe']
    sector_labels = results['sector_labels']
    history = results['history']
    val_predictions = results['val_predictions']
    val_targets = results['val_targets']
    lr_predictions = results['lr_predictions']
    attention_weights = results['attention_weights']
    lag_probs = results['lag_probs']
    expected_lags = results['expected_lags']
    metrics = results['metrics']
    max_lag = results['config']['max_lag']
    
    # Load test data if available
    test_predictions = results.get('test_predictions', None)
    test_targets = results.get('test_targets', None)
    lr_test_predictions = results.get('lr_test_predictions', None)
    
    print(f"Target stock: {target_stock}")
    print(f"Stock universe: {', '.join(stock_universe)}")
    
    # Print metrics summary
    print(f"\n{'─'*80}")
    print("METRICS SUMMARY")
    print(f"{'─'*80}")
    print(f"Neural Network:")
    print(f"  MSE:  {metrics['nn_mse']:.6f}")
    print(f"  RMSE: {metrics['nn_rmse']:.6f}")
    print(f"  MAE:  {metrics['nn_mae']:.6f}")
    print(f"  Mean |% Error|: {metrics['nn_mean_abs_percent_error']:.2f}%")
    print(f"\nLinear Regression Baseline:")
    print(f"  MSE:  {metrics['lr_mse']:.6f}")
    print(f"  RMSE: {metrics['lr_rmse']:.6f}")
    print(f"  MAE:  {metrics['lr_mae']:.6f}")
    print(f"  R²:   {metrics['lr_r2']:.6f}")
    print(f"  Mean |% Error|: {metrics['lr_mean_abs_percent_error']:.2f}%")
    
    improvement = ((metrics['lr_mse'] - metrics['nn_mse']) / metrics['lr_mse']) * 100
    print(f"\nNN Improvement over LR: {improvement:.2f}%")
    
    # Print lag distribution summary
    print(f"\n{'─'*80}")
    print("LAG DISTRIBUTION SUMMARY")
    print(f"{'─'*80}")
    for i, stock in enumerate(stock_universe):
        print(f"{stock}: Expected lag = {expected_lags[i]:.2f} intervals")
        print(f"  Top 3 lags: {np.argsort(lag_probs[i])[-3:][::-1]} "
              f"with probs {lag_probs[i][np.argsort(lag_probs[i])[-3:][::-1]]}")
    
    # Generate plots
    print(f"\n{'─'*80}")
    print("GENERATING PLOTS")
    print(f"{'─'*80}")
    
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create PDF
    pdf_path = f"{plot_dir}/{target_stock}_deltalag_analysis.pdf"
    with PdfPages(pdf_path) as pdf:
        # Plot 1: Lag probability distributions
        fig_path = f"{plot_dir}/{target_stock}_lag_distributions.png"
        fig1 = plot_lag_probability_distributions(
            stock_universe, lag_probs, sector_labels, max_lag, fig_path
        )
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)
        
        # Plot 2: Averaged lag distribution
        fig_path = f"{plot_dir}/{target_stock}_averaged_lag_distribution.png"
        fig2_avg = plot_averaged_lag_distribution(
            lag_probs, max_lag, stock_universe, fig_path
        )
        pdf.savefig(fig2_avg, bbox_inches='tight')
        plt.close(fig2_avg)
        
        # Plot 3: Attention weights
        fig_path = f"{plot_dir}/{target_stock}_attention_weights.png"
        fig3 = plot_attention_weights(stock_universe, attention_weights, sector_labels, fig_path)
        pdf.savefig(fig3, bbox_inches='tight')
        plt.close(fig3)
        
        # Plot 4: Loss history
        fig_path = f"{plot_dir}/{target_stock}_loss_history.png"
        fig4 = plot_loss_history(history, fig_path)
        pdf.savefig(fig4, bbox_inches='tight')
        plt.close(fig4)
        
        # Plot 5: Validation and test predictions and squared error (with linear regression comparison)
        fig_path = f"{plot_dir}/{target_stock}_validation_predictions_error.png"
        fig5, mse_results = plot_test_predictions_and_accuracy(
            val_predictions, val_targets, target_stock, fig_path, 
            lr_predictions=lr_predictions,
            test_predictions=test_predictions,
            test_targets=test_targets,
            lr_test_predictions=lr_test_predictions
        )
        pdf.savefig(fig5, bbox_inches='tight')
        plt.close(fig5)
        
        # Print final MSE comparison
        print(f"\n{'─'*80}")
        print("FINAL MEAN SQUARED ERROR COMPARISON")
        print(f"{'─'*80}")
        print(f"Neural Network MSE: {mse_results['nn_mse']:.6f}")
        if 'lr_mse' in mse_results:
            print(f"Linear Regression MSE: {mse_results['lr_mse']:.6f}")
            improvement = ((mse_results['lr_mse'] - mse_results['nn_mse']) / mse_results['lr_mse']) * 100
            print(f"NN Improvement over LR: {improvement:.2f}%")
        
        d = pdf.infodict()
        d['Title'] = f'{target_stock} DeltaLag Attention Analysis'
        d['Author'] = 'CS230 Project'
    
    print(f"\n✓ All plots saved to PDF: {pdf_path}")
    
    # Save results CSV
    os.makedirs(output_dir, exist_ok=True)
    avg_attention = attention_weights.mean(axis=0) if len(attention_weights.shape) > 1 else attention_weights
    results_df = pd.DataFrame({
        'stock': stock_universe,
        'sector': [sector_labels.get(s, 'UNKNOWN') for s in stock_universe],
        'attention_weight': avg_attention,
        'expected_lag_intervals': expected_lags,
        'expected_lag_minutes': expected_lags * 5,
    })
    csv_path = f"{output_dir}/{target_stock}_deltalag_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to: {csv_path}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nOutputs:")
    print(f"  - PDF: {pdf_path}")
    print(f"  - CSV: {csv_path}")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze DeltaLag attention model results and generate visualizations"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results pickle file (e.g., results/APA_deltalag_results.pkl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory for output CSV files (default: results)",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="results",
        help="Directory for output plots (default: results)",
    )
    
    args = parser.parse_args()
    
    analyze_deltalag_results(
        results_file=args.results,
        output_dir=args.output_dir,
        plot_dir=args.plot_dir,
    )

