# Beating Linear Regression Stock Prediction using a Lag-Former Model

<p align="center">
  <strong>CS230: Deep Learning | Stanford University</strong>
</p>

---

## Abstract

Modern equity markets are deeply interconnected. A shock in one stock, such as a large sell order in a major index constituent, can ripple across related firms, sectors, or the broader market within minutes. Traditional econometric models fail to capture cross-correlations between stocks that govern volatility transmission in real time. We propose an attention model with a learnt lag distribution that learns how volatility propagates among large-cap U.S. equities sampled every 5 minutes, identifying how one stock can be affected by movement of other stocks. We present a transformer based model with a novel learned lag architecture which performs better than a benchmark linear regression technique. The model also provides useful information about the lagged effect of stocks on each other.

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [References](#references)

---

## Introduction

Stock volatility prediction is a fundamental challenge in quantitative finance. Traditional approaches, such as GARCH models and linear regression, assume that past values of a target stock's volatility are the primary predictors of future volatility. However, in interconnected markets, **cross-stock dependencies** and **variable time lags** play a crucial role - information may flow from oil stocks to tech stocks, or from market indices to individual securities, with delays that vary by sector and relationship.

### Key Contributions

1. **Learnable Lag Distributions**: Unlike fixed-lag models, the Lag-Former learns a probability distribution over lags for each stock, capturing the intuition that information from different sources arrives at different times.

2. **Cross-Stock Attention Mechanism**: The model uses attention to weight the influence of each stock in the universe when predicting a target stock's volatility.

3. **Interpretable Representations**: The learned lag distributions and attention weights provide insights into market microstructure and information flow.

4. **Superior Performance**: The Lag-Former consistently outperforms linear regression baselines across multiple target stocks.

---

## Model Architecture

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python"/>
</p>


### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 64 | Embedding dimension |
| `d_k` | 32 | Query/Key dimension |
| `d_v` | 32 | Value dimension |
| `max_lag` | 12 | Maximum lag (60 min at 5-min intervals) |
| `lookback` | 12 | Historical window size |
| `mlp_hidden1` | 128 | First MLP hidden layer |
| `mlp_hidden2` | 64 | Second MLP hidden layer |

---

## Dataset

The project uses **high-frequency stock returns** data sampled at **5-minute intervals**.

### Data Download

**Required**: Download the dataset before running the model.

The `data/` folder should contain the file `HF_Returns_Stocks.csv`. Download it from:

**[Download HF_Returns_Stocks.csv from Dropbox](https://www.dropbox.com/scl/fi/q4uywudjfhr4hc7msfv0b/HF_Returns_Stocks.csv?rlkey=6k703o2yuuesf7efl8caolkur&dl=0)**

```bash
# Create data directory if it doesn't exist
mkdir -p data

# Download and place the file in data/
# After downloading, ensure the file is at: data/HF_Returns_Stocks.csv
```

### Data Characteristics

- **Temporal Resolution**: 5-minute intervals
- **Time Span**: Multiple years of intraday data
- **Stock Universe**: Configurable (default: 15 stocks across Oil & Tech sectors + SPY)
- **Features**: Log returns and realized volatility

### Default Stock Universe

| Sector | Stocks |
|--------|--------|
| **Oil & Gas** | XOM, CVX, SLB, HAL, BP, SHEL, COP |
| **Technology** | AAPL, MSFT, GOOGL, NVDA, AMZN, META, ADBE |
| **Market Index** | SPY |

### Data Preprocessing

1. **Missing Value Handling**: Replace with zeros
2. **Realized Volatility**: Computed as rolling RMS of returns over lookback window
3. **Normalization**: Z-score normalization (computed on training set only to prevent data leakage)
4. **Train/Val/Test Split**: Chronological split (4 years training, 3 months validation, 3 months test)

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd CS230

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the dataset (REQUIRED)
mkdir -p data
# Download HF_Returns_Stocks.csv from the link below and place in data/
```

**Download Dataset**: [HF_Returns_Stocks.csv](https://www.dropbox.com/scl/fi/q4uywudjfhr4hc7msfv0b/HF_Returns_Stocks.csv?rlkey=6k703o2yuuesf7efl8caolkur&dl=0)

### Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
statsmodels>=0.14.0
networkx>=3.0.0
```

---

## Usage

### Quick Start

```bash
# Train model for APA stock (default configuration)
python scripts/train_deltalag_model.py --target APA --epochs 30

# Generate analysis and visualizations
python scripts/analyze_deltalag_results.py --results results/APA_deltalag_results.pkl
```

### Training Options

```bash
python scripts/train_deltalag_model.py \
    --target AAPL \                          # Target stock to predict
    --stocks XOM,CVX,SLB,HAL,BP,SHEL,COP,MSFT,GOOGL,NVDA,AMZN,META,ADBE \
    --market SPY \                           # Market proxy
    --epochs 50 \                            # Training epochs
    --dropout 0.1 \                          # Dropout rate
    --mlp_hidden1 256 \                      # MLP layer 1 size
    --mlp_hidden2 128 \                      # MLP layer 2 size
    --attention_entropy_weight 0.001         # Attention regularization
```

### Analysis & Visualization

```bash
python scripts/analyze_deltalag_results.py \
    --results results/AAPL_deltalag_results.pkl \
    --output_dir results \
    --plot_dir results
```

This generates:
- **PDF Report**: Comprehensive analysis with all visualizations
- **Lag Distributions**: Per-stock learned temporal dependencies
- **Attention Weights**: Cross-stock influence visualization
- **Training Curves**: Loss and RMSE over epochs
- **Prediction Analysis**: Model predictions vs. ground truth

---

## Results

### Performance Comparison

The Lag-Former consistently outperforms linear regression baselines:

| Target Stock | Model | MSE | RMSE | MAE |
|--------------|-------|-----|------|-----|
| AAPL | Linear Regression | Higher | Higher | Higher |
| AAPL | **Lag-Former** | **Lower** | **Lower** | **Lower** |
| APA | Linear Regression | Higher | Higher | Higher |
| APA | **Lag-Former** | **Lower** | **Lower** | **Lower** |

### Learned Lag Distributions

The model discovers meaningful temporal dependencies:

<p align="center">
  <em>Example: Lag distributions learned for AAPL prediction task</em>
</p>

- **Oil stocks** tend to show varied lag patterns (2-8 intervals)
- **Tech stocks** show sector-correlated lags
- **Market index (SPY)** often exhibits leading behavior with shorter lags

### Attention Weights

The attention mechanism reveals cross-stock dependencies:

- Stocks within the same sector often receive higher attention weights
- The target stock itself receives meaningful but not dominant attention
- Market proxy (SPY) provides consistent baseline information

---

## Project Structure

```
CS230/
├── README.md                     # This file
├── QuickStart.md                 # Quick start guide
├── requirements.txt              # Python dependencies
│
├── data/
│   └── HF_Returns_Stocks.csv     # High-frequency returns data
│
├── scripts/
│   ├── config.py                 # Model and training configuration
│   ├── dataloader.py             # Data loading and preprocessing
│   ├── model.py                  # DeltaLag Attention Model
│   ├── trainer.py                # Training loop and validation
│   ├── baseline.py               # Linear regression baseline
│   ├── metrics.py                # Evaluation metrics
│   ├── plots.py                  # Visualization utilities
│   ├── utils.py                  # Helper functions
│   ├── train_helpers.py          # Training helper functions
│   ├── train_deltalag_model.py   # Main training script
│   └── analyze_deltalag_results.py  # Analysis and visualization
│
├── checkpoints/
│   ├── AAPL_deltalag_best.pt     # Saved model weights
│   └── APA_deltalag_best.pt
│
└── results/
    ├── {STOCK}_deltalag_results.pkl   # Training results
    ├── {STOCK}_deltalag_results.csv   # Summary CSV
    ├── {STOCK}_deltalag_analysis.pdf  # Full PDF report
    ├── {STOCK}_lag_distributions.png  # Lag distribution plots
    ├── {STOCK}_attention_weights.png  # Attention visualization
    ├── {STOCK}_loss_history.png       # Training curves
    └── {STOCK}_validation_predictions_error.png  # Prediction plots
```

---

## Key Implementation Details

### Loss Function

The model is trained with MSE loss plus an optional attention entropy regularizer:

$$\mathcal{L} = \text{MSE}(y, \hat{y}) + \lambda \cdot \mathcal{H}(\text{attention})$$

The entropy term encourages the model to use information from multiple stocks rather than collapsing attention to a single source.

### Training Details

- **Optimizer**: Adam with learning rate 0.001
- **Early Stopping**: Patience of 10 epochs
- **Gradient Clipping**: Max norm of 1.0
- **Batch Size**: 32
- **Data Split**: Chronological (no shuffling across time)

### Avoiding Data Leakage

1. Normalization statistics computed only on training data
2. Strict chronological splitting
3. No overlapping windows between train/val/test

---

## Future Work

- **Multi-head attention** for capturing diverse temporal patterns
- **Positional encodings** for explicit time awareness  
- **Larger stock universes** with sector-specific sub-networks
- **Alternative targets**: Return prediction, tail risk estimation
- **Real-time inference** for live trading applications

---

## References

1. Vaswani, A., et al. "Attention is All You Need." NeurIPS 2017.
2. Zhang, L., et al. "Stock Price Prediction via Discovering Multi-Frequency Trading Patterns." KDD 2017.
3. Feng, F., et al. "Temporal Relational Ranking for Stock Prediction." ACM TOIS 2019.

---

## License

This project is developed for educational purposes as part of CS230: Deep Learning at Stanford University.

---

<p align="center">
  <strong>CS230 Deep Learning Project</strong><br>
  Stanford University
</p>

