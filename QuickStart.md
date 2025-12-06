# Quick Start Guide: Running the DeltaLag Attention Model

This guide explains how to train the model and generate analysis reports.

## Prerequisites

1. **Data File**: Ensure `data/HF_Returns_Stocks.csv` exists
2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Step 1: Train the Model

Train a DeltaLag attention model to predict volatility for a target stock.

### Basic Usage

```bash
python scripts/train_deltalag_model.py --target APA --epochs 30
```

This will:
- Train a model to predict volatility for **APA** stock
- Use default stock universe (7 oil + 7 tech stocks + SPY)
- Train for 30 epochs
- Save results to `results/APA_deltalag_results.pkl`
- Save model checkpoint to `checkpoints/APA_deltalag_best.pt`

### Custom Stock Universe

```bash
python scripts/train_deltalag_model.py \
    --target APA \
    --stocks XOM,CVX,SLB,HAL,BP,SHEL,COP,AAPL,MSFT,GOOGL,NVDA,AMZN,META,ADBE \
    --market SPY \
    --epochs 50
```

### All Available Options

```bash
python scripts/train_deltalag_model.py \
    --target APA                    # Target stock to predict (default: APA)
    --stocks XOM,CVX,SLB,...        # Comma-separated list of other stocks
    --market SPY                     # Market proxy stock (default: SPY)
    --epochs 30                      # Number of training epochs (default: 30)
    --dropout 0.0                    # Dropout rate (default: 0.0)
    --mlp_hidden1 128                # First MLP hidden layer size (default: 128)
    --mlp_hidden2 64                 # Second MLP hidden layer size (default: 64)
    --attention_entropy_weight 0.001 # Attention entropy penalty weight (default: 0.001)
```

### Example: Quick Test Run

```bash
# Quick test with 1 epoch
python scripts/train_deltalag_model.py --target APA --epochs 1 --stocks XOM,CVX
```

## Step 2: Generate Analysis and Visualizations

After training, generate plots and analysis reports.

### Basic Usage

```bash
python scripts/analyze_deltalag_results.py --results results/APA_deltalag_results.pkl
```

This will:
- Load the saved results from training
- Generate all visualizations (lag distributions, attention weights, loss history, predictions)
- Create a PDF report: `results/APA_deltalag_analysis.pdf`
- Save individual plots to `results/` directory
- Generate CSV summary: `results/APA_deltalag_results.csv`

### Custom Output Directories

```bash
python scripts/analyze_deltalag_results.py \
    --results results/APA_deltalag_results.pkl \
    --output_dir results \
    --plot_dir results
```

### All Available Options

```bash
python scripts/analyze_deltalag_results.py \
    --results results/APA_deltalag_results.pkl  # Path to results pickle file (required)
    --output_dir results                         # Directory for CSV files (default: results)
    --plot_dir results                           # Directory for plots (default: results)
```

## Complete Workflow Example

### Example 1: Train and Analyze APA Stock

```bash
# Step 1: Train the model
python scripts/train_deltalag_model.py --target APA --epochs 30

# Step 2: Generate analysis
python scripts/analyze_deltalag_results.py --results results/APA_deltalag_results.pkl
```

### Example 2: Train Multiple Stocks

```bash
# Train AAPL
python scripts/train_deltalag_model.py --target AAPL --epochs 30

# Train NVDA
python scripts/train_deltalag_model.py --target NVDA --epochs 30

# Analyze AAPL
python scripts/analyze_deltalag_results.py --results results/AAPL_deltalag_results.pkl

# Analyze NVDA
python scripts/analyze_deltalag_results.py --results results/NVDA_deltalag_results.pkl
```

### Example 3: Custom Configuration

```bash
# Train with custom parameters
python scripts/train_deltalag_model.py \
    --target MUR \
    --stocks XOM,CVX,SLB,HAL,BP,SHEL,COP \
    --market SPY \
    --epochs 50 \
    --dropout 0.1 \
    --mlp_hidden1 256 \
    --mlp_hidden2 128 \
    --attention_entropy_weight 0.0001

# Analyze results
python scripts/analyze_deltalag_results.py --results results/MUR_deltalag_results.pkl
```

## Output Files

After running both scripts, you'll have:

### From Training (`train_deltalag_model.py`):
- **Model checkpoint**: `checkpoints/{STOCK}_deltalag_best.pt`
- **Results pickle**: `results/{STOCK}_deltalag_results.pkl`

### From Analysis (`analyze_deltalag_results.py`):
- **PDF report**: `results/{STOCK}_deltalag_analysis.pdf`
- **Individual plots**:
  - `results/{STOCK}_lag_distributions.png`
  - `results/{STOCK}_averaged_lag_distribution.png`
  - `results/{STOCK}_attention_weights.png`
  - `results/{STOCK}_loss_history.png`
  - `results/{STOCK}_validation_predictions_error.png`
- **CSV summary**: `results/{STOCK}_deltalag_results.csv`

## Understanding the Output

### Training Output
- **Model metrics**: MSE, RMSE, MAE for neural network
- **Baseline comparison**: Linear regression baseline metrics
- **Lag distribution**: Expected lag for each stock in the universe
- **Training history**: Loss curves over epochs

### Analysis Output
- **Lag probability distributions**: How the model learns temporal dependencies
- **Attention weights**: Which stocks influence the target most
- **Loss history**: Training and validation loss over time
- **Prediction plots**: Model predictions vs. actual values with error analysis
- **Metrics summary**: Comparison between neural network and linear regression

## Troubleshooting

### Common Issues

1. **File not found error**:
   - Ensure `data/HF_Returns_Stocks.csv` exists
   - Check that the results file path is correct

2. **Out of memory**:
   - Reduce batch size in `scripts/config.py`
   - Use fewer stocks in the universe

3. **Import errors**:
   - Make sure you're running from the project root directory
   - Check that all dependencies are installed: `pip install -r requirements.txt`

### Getting Help

Check the terminal output for:
- Training progress and loss values
- Validation metrics
- Any error messages

## Next Steps

- Experiment with different stock universes
- Adjust hyperparameters (dropout, MLP sizes, attention entropy weight)
- Compare results across different target stocks
- Analyze the lag distributions to understand temporal dependencies

