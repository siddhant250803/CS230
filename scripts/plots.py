"""
Plotting functions for DeltaLag attention model analysis.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_lag_probability_distributions(
    stock_names: list,
    lag_probs: np.ndarray,
    sector_labels: dict,
    max_lag: int,
    output_path: str = None,
):
    """Plot lag probability distributions for each stock."""
    n_stocks = len(stock_names)
    fig, axes = plt.subplots(n_stocks, 1, figsize=(12, 2 * n_stocks))
    if n_stocks == 1:
        axes = [axes]
    
    lag_values = np.arange(max_lag)
    colors = {
        'TARGET': 'red',
        'MARKET': 'purple',
        'OIL': 'green',
        'TECH': 'blue',
        'OTHER': 'gray',
    }
    
    for idx, stock in enumerate(stock_names):
        ax = axes[idx]
        probs = lag_probs[idx]
        sector = sector_labels.get(stock, 'OTHER')
        color = colors.get(sector, 'gray')
        
        ax.bar(lag_values, probs, alpha=0.7, color=color, edgecolor='black')
        ax.set_xlabel('Lag (intervals)', fontsize=10)
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_title(f'{stock} Lag Distribution', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(lag_values[::2])
        
        expected_lag = np.sum(lag_values * probs)
        ax.axvline(expected_lag, color='red', linestyle='--', linewidth=2, 
                   label=f'Expected: {expected_lag:.2f}')
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Lag probability distributions saved to: {output_path}")
    return fig


def plot_averaged_lag_distribution(
    lag_probs: np.ndarray,
    max_lag: int,
    stock_names: list = None,
    output_path: str = None,
):
    """Plot the averaged lag probability distribution across all stocks."""
    avg_lag_probs = np.mean(lag_probs, axis=0)
    std_lag_probs = np.std(lag_probs, axis=0)
    lag_values = np.arange(max_lag)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.bar(lag_values, avg_lag_probs, alpha=0.7, color='#1f77b4', edgecolor='black',
           yerr=std_lag_probs, capsize=5)
    
    ax.set_xlabel('Lag (intervals)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Probability', fontsize=12, fontweight='bold')
    ax.set_title('Averaged Lag Distribution Across All Stocks', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(lag_values[::2])
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Averaged lag distribution saved to: {output_path}")
    return fig


def plot_attention_weights(
    stock_names: list,
    attention_weights: np.ndarray,
    sector_labels: dict,
    output_path: str = None,
):
    """Plot attention weights."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    avg_attention = attention_weights.mean(axis=0) if len(attention_weights.shape) > 1 else attention_weights
    
    colors = {
        'TARGET': 'red',
        'MARKET': 'purple',
        'OIL': 'green',
        'TECH': 'blue',
        'OTHER': 'gray',
    }
    
    for sector in ['TARGET', 'MARKET', 'OIL', 'TECH', 'OTHER']:
        sector_stocks = [s for s in stock_names if sector_labels.get(s) == sector]
        if not sector_stocks:
            continue
        
        indices = [stock_names.index(s) for s in sector_stocks]
        sector_weights = avg_attention[indices]
        
        ax.scatter(
            indices, sector_weights,
            c=colors[sector], marker='o' if sector != 'TARGET' else 's',
            s=200, alpha=0.8, edgecolors='black', linewidth=2,
            label=f'{sector} ({len(sector_stocks)})',
            zorder=3
        )
        
        for idx, stock in zip(indices, sector_stocks):
            ax.annotate(
                stock, (idx, avg_attention[idx]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                         edgecolor=colors[sector])
            )
    
    ax.set_xlabel('Stock Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
    ax.set_title('Attention Weights: Influence of Each Stock', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_xticks(range(len(stock_names)))
    ax.set_xticklabels(stock_names, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Attention weights plot saved to: {output_path}")
    return fig


def plot_loss_history(history: dict, output_path: str = None):
    """Plot loss history with train/validation."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.semilogy(epochs, history['train_loss'], 'o-', 
                label='Train Loss', linewidth=2, markersize=6, color='#1f77b4')
    ax.semilogy(epochs, history['val_loss'], 's-', 
                label='Validation Loss', linewidth=2, markersize=6, color='#ff7f0e')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_title('Training History: Loss (Log Scale)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Loss history plot saved to: {output_path}")
    return fig


def plot_test_predictions_and_accuracy(
    predictions: np.ndarray, 
    targets: np.ndarray, 
    stock_name: str, 
    output_path: str = None,
    lr_predictions: np.ndarray = None,
    test_predictions: np.ndarray = None,
    test_targets: np.ndarray = None,
    lr_test_predictions: np.ndarray = None,
):
    """
    Plot predicted vs actual volatilities and cumulative MSE over time.
    Includes both validation and test sets if provided.
    
    Returns:
        fig: matplotlib figure
        result: dict with 'nn_mse' and optionally 'lr_mse'
    """
    has_test = test_predictions is not None and test_targets is not None
    n_samples = len(predictions)
    time_points = np.arange(n_samples)
    
    # Calculate number of plots: validation (3-4) + test (3-4 if available)
    val_plots = 4 if lr_predictions is not None else 3
    test_plots = 4 if (has_test and lr_test_predictions is not None) else (3 if has_test else 0)
    n_plots = val_plots + test_plots
    
    fig = plt.figure(figsize=(16, 6 * n_plots))
    zoom_size = min(1000, n_samples)
    
    plot_idx = 1
    
    # ========== VALIDATION SET PLOTS ==========
    # Plot 1: Validation - First 1000 points
    ax1 = plt.subplot(n_plots, 1, plot_idx)
    plot_idx += 1
    ax1.plot(time_points[:zoom_size], predictions[:zoom_size], label='NN Predicted', 
             linewidth=3, color='#ff7f0e', alpha=0.8)
    if lr_predictions is not None:
        ax1.plot(time_points[:zoom_size], lr_predictions[:zoom_size], label='LR Predicted', 
                 linewidth=2.5, color='red', alpha=0.8, linestyle=':')
    ax1.plot(time_points[:zoom_size], targets[:zoom_size], label='Actual', 
             linewidth=2, color='#2ca02c', alpha=0.8)
    ax1.set_xlabel('Time Point (First 1000)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Volatility', fontsize=11, fontweight='bold')
    ax1.set_title(f'VALIDATION SET - Predicted vs Actual Volatility: {stock_name} - First 1000 Points', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Plot 2: Validation - Last 1000 points
    ax2 = plt.subplot(n_plots, 1, plot_idx)
    plot_idx += 1
    if n_samples > zoom_size:
        start_idx = max(0, n_samples - zoom_size)
        ax2.plot(time_points[start_idx:], predictions[start_idx:], label='NN Predicted', 
                 linewidth=3, color='#ff7f0e', alpha=0.8)
        if lr_predictions is not None:
            ax2.plot(time_points[start_idx:], lr_predictions[start_idx:], label='LR Predicted', 
                     linewidth=2.5, color='red', alpha=0.8, linestyle=':')
        ax2.plot(time_points[start_idx:], targets[start_idx:], label='Actual', 
                 linewidth=2, color='#2ca02c', alpha=0.8)
        ax2.set_xlabel(f'Time Point (Last 1000, starting at {start_idx})', 
                      fontsize=11, fontweight='bold')
    else:
        ax2.plot(time_points, predictions, label='NN Predicted', 
                 linewidth=3, color='#ff7f0e', alpha=0.8)
        if lr_predictions is not None:
            ax2.plot(time_points, lr_predictions, label='LR Predicted', 
                     linewidth=1, color='red', alpha=0.8, linestyle=':')
        ax2.plot(time_points, targets, label='Actual', 
                 linewidth=2, color='#2ca02c', alpha=0.8)
        ax2.set_xlabel('Time Point (All Points)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Volatility', fontsize=11, fontweight='bold')
    ax2.set_title(f'VALIDATION SET - Predicted vs Actual Volatility: {stock_name} - Last 1000 Points', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Calculate validation errors
    squared_errors = (predictions - targets) ** 2
    nn_mse = np.mean(squared_errors)
    nn_cumulative_mse = np.cumsum(squared_errors) / np.arange(1, n_samples + 1)
    
    # Plot 3: Validation - Neural Network Cumulative MSE
    ax3 = plt.subplot(n_plots, 1, plot_idx)
    plot_idx += 1
    ax3.plot(time_points, nn_cumulative_mse, linewidth=2, color='#1f77b4', alpha=0.8, 
             label=f'NN Cumulative MSE (Final: {nn_mse:.6f})')
    ax3.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Mean Squared Error', fontsize=12, fontweight='bold')
    ax3.set_title(f'VALIDATION SET - Neural Network Cumulative MSE Over Time: {stock_name}', 
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Plot 4: Validation - Linear Regression Cumulative MSE (if available)
    if lr_predictions is not None:
        lr_squared_errors = (lr_predictions - targets) ** 2
        lr_mse = np.mean(lr_squared_errors)
        lr_cumulative_mse = np.cumsum(lr_squared_errors) / np.arange(1, n_samples + 1)
        
        ax4 = plt.subplot(n_plots, 1, plot_idx)
        plot_idx += 1
        ax4.plot(time_points, lr_cumulative_mse, linewidth=2, color='red', alpha=0.8, 
                 label=f'LR Cumulative MSE (Final: {lr_mse:.6f})')
        ax4.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Cumulative Mean Squared Error', fontsize=12, fontweight='bold')
        ax4.set_title(f'VALIDATION SET - Linear Regression Cumulative MSE Over Time: {stock_name}', 
                      fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # ========== TEST SET PLOTS ==========
    if has_test:
        n_test_samples = len(test_predictions)
        test_time_points = np.arange(n_test_samples)
        test_zoom_size = min(1000, n_test_samples)
        
        # Plot: Test - First 1000 points
        ax_test1 = plt.subplot(n_plots, 1, plot_idx)
        plot_idx += 1
        ax_test1.plot(test_time_points[:test_zoom_size], test_predictions[:test_zoom_size], 
                     label='NN Predicted', linewidth=3, color='#ff7f0e', alpha=0.8)
        if lr_test_predictions is not None:
            ax_test1.plot(test_time_points[:test_zoom_size], lr_test_predictions[:test_zoom_size], 
                         label='LR Predicted', linewidth=2.5, color='red', alpha=0.8, linestyle=':')
        ax_test1.plot(test_time_points[:test_zoom_size], test_targets[:test_zoom_size], 
                     label='Actual', linewidth=2, color='#2ca02c', alpha=0.8)
        ax_test1.set_xlabel('Time Point (First 1000)', fontsize=11, fontweight='bold')
        ax_test1.set_ylabel('Volatility', fontsize=11, fontweight='bold')
        ax_test1.set_title(f'TEST SET - Predicted vs Actual Volatility: {stock_name} - First 1000 Points', 
                          fontsize=12, fontweight='bold')
        ax_test1.grid(True, alpha=0.3, linestyle='--')
        ax_test1.legend(loc='best', fontsize=10, framealpha=0.9)
        
        # Plot: Test - Last 1000 points
        ax_test2 = plt.subplot(n_plots, 1, plot_idx)
        plot_idx += 1
        if n_test_samples > test_zoom_size:
            test_start_idx = max(0, n_test_samples - test_zoom_size)
            ax_test2.plot(test_time_points[test_start_idx:], test_predictions[test_start_idx:], 
                         label='NN Predicted', linewidth=3, color='#ff7f0e', alpha=0.8)
            if lr_test_predictions is not None:
                ax_test2.plot(test_time_points[test_start_idx:], lr_test_predictions[test_start_idx:], 
                             label='LR Predicted', linewidth=2.5, color='red', alpha=0.8, linestyle=':')
            ax_test2.plot(test_time_points[test_start_idx:], test_targets[test_start_idx:], 
                         label='Actual', linewidth=2, color='#2ca02c', alpha=0.8)
            ax_test2.set_xlabel(f'Time Point (Last 1000, starting at {test_start_idx})', 
                              fontsize=11, fontweight='bold')
        else:
            ax_test2.plot(test_time_points, test_predictions, label='NN Predicted', 
                         linewidth=3, color='#ff7f0e', alpha=0.8)
            if lr_test_predictions is not None:
                ax_test2.plot(test_time_points, lr_test_predictions, label='LR Predicted', 
                             linewidth=1, color='red', alpha=0.8, linestyle=':')
            ax_test2.plot(test_time_points, test_targets, label='Actual', 
                         linewidth=2, color='#2ca02c', alpha=0.8)
            ax_test2.set_xlabel('Time Point (All Points)', fontsize=11, fontweight='bold')
        ax_test2.set_ylabel('Volatility', fontsize=11, fontweight='bold')
        ax_test2.set_title(f'TEST SET - Predicted vs Actual Volatility: {stock_name} - Last 1000 Points', 
                          fontsize=12, fontweight='bold')
        ax_test2.grid(True, alpha=0.3, linestyle='--')
        ax_test2.legend(loc='best', fontsize=10, framealpha=0.9)
        
        # Calculate test errors
        test_squared_errors = (test_predictions - test_targets) ** 2
        nn_test_mse = np.mean(test_squared_errors)
        nn_test_cumulative_mse = np.cumsum(test_squared_errors) / np.arange(1, n_test_samples + 1)
        
        # Plot: Test - Neural Network Cumulative MSE
        ax_test3 = plt.subplot(n_plots, 1, plot_idx)
        plot_idx += 1
        ax_test3.plot(test_time_points, nn_test_cumulative_mse, linewidth=2, color='#1f77b4', alpha=0.8, 
                     label=f'NN Cumulative MSE (Final: {nn_test_mse:.6f})')
        ax_test3.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax_test3.set_ylabel('Cumulative Mean Squared Error', fontsize=12, fontweight='bold')
        ax_test3.set_title(f'TEST SET - Neural Network Cumulative MSE Over Time: {stock_name}', 
                          fontsize=13, fontweight='bold')
        ax_test3.grid(True, alpha=0.3, linestyle='--')
        ax_test3.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Plot: Test - Linear Regression Cumulative MSE (if available)
        if lr_test_predictions is not None:
            lr_test_squared_errors = (lr_test_predictions - test_targets) ** 2
            lr_test_mse = np.mean(lr_test_squared_errors)
            lr_test_cumulative_mse = np.cumsum(lr_test_squared_errors) / np.arange(1, n_test_samples + 1)
            
            ax_test4 = plt.subplot(n_plots, 1, plot_idx)
            plot_idx += 1
            ax_test4.plot(test_time_points, lr_test_cumulative_mse, linewidth=2, color='red', alpha=0.8, 
                         label=f'LR Cumulative MSE (Final: {lr_test_mse:.6f})')
            ax_test4.set_xlabel('Time Step', fontsize=12, fontweight='bold')
            ax_test4.set_ylabel('Cumulative Mean Squared Error', fontsize=12, fontweight='bold')
            ax_test4.set_title(f'TEST SET - Linear Regression Cumulative MSE Over Time: {stock_name}', 
                              fontsize=13, fontweight='bold')
            ax_test4.grid(True, alpha=0.3, linestyle='--')
            ax_test4.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if has_test:
            print(f"✓ Validation and test predictions plot saved to: {output_path}")
        else:
            print(f"✓ Validation predictions and squared error plot saved to: {output_path}")
    
    result = {'nn_mse': nn_mse}
    if lr_predictions is not None:
        result['lr_mse'] = lr_mse
    if has_test:
        result['nn_test_mse'] = nn_test_mse
        if lr_test_predictions is not None:
            result['lr_test_mse'] = lr_test_mse
    return fig, result

