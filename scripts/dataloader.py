"""
Data loading and preprocessing for high-frequency stock returns.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
from config import Config


class StockDataLoader:
    """Loads and preprocesses high-frequency stock returns data."""
    
    def __init__(self, data_path: str, config: Config):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV file
            config: Configuration object
        """
        self.config = config
        self.data_path = data_path
        self.data = None
        self.stock_names = None
        self.dates = None
        self.returns = None
        self.volatility = None
        self.normalized_returns = None
        self.normalized_volatility = None
        
    def load_data(self):
        """Load data from CSV file."""
        print("Loading data...")
        # Read CSV, skip the first row which is a description
        # Use latin-1 encoding to handle copyright symbol in header
        self.data = pd.read_csv(self.data_path, skiprows=1, low_memory=False, encoding='latin-1')
        
        # Extract date column
        self.dates = self.data['Date'].values
        
        # Extract stock names (all columns except Date and Time)
        self.stock_names = [col for col in self.data.columns if col not in ['Date', 'Time']]
        
        # Extract returns data
        self.returns = self.data[self.stock_names].values.astype(np.float32)
        
        # Replace missing values with 0
        self.returns[np.abs(self.returns - self.config.MISSING_VALUE) < 1e-10] = 0
        
        # Handle NaN values (replace with 0)
        nan_count = np.isnan(self.returns).sum()
        if nan_count > 0:
            print(f"Warning: Found {nan_count} NaN values, replacing with 0")
            self.returns = np.nan_to_num(self.returns, nan=0.0)
        
        print(f"Loaded data: {len(self.dates)} time steps, {len(self.stock_names)} stocks")
        print(f"Date range: {self.dates[0]} to {self.dates[-1]}")
        
    def compute_realized_volatility(self, window: int = None):
        """
        Compute realized volatility over a rolling window.
        
        Args:
            window: Rolling window size (default: from config)
        """
        if window is None:
            window = self.config.LOOKBACK_WINDOW
            
        print(f"Computing realized volatility with window={window}...")
        
        n_timesteps, n_stocks = self.returns.shape
        self.volatility = np.zeros_like(self.returns)
        
        # Compute rolling standard deviation
        for i in range(window, n_timesteps):
            # RMS of returns over window (realized volatility)
            self.volatility[i] = np.sqrt(np.mean(self.returns[i-window:i]**2, axis=0))
        
        # Leave initial values as zero - these samples will be excluded when creating sequences
        # This prevents artificial patterns from contaminating the data
            
        print(f"Volatility computed. Mean: {np.mean(self.volatility):.6f}, Std: {np.std(self.volatility):.6f}")
        
    def normalize_data(self, train_end_idx: Optional[int] = None):
        """
        Normalize returns and volatility per stock using z-score.
        
        Args:
            train_end_idx: If provided, compute statistics only on training data [:train_end_idx]
                          to avoid data leakage. If None, use all data (for backward compatibility).
        """
        print("Normalizing data...")
        
        # Determine range for computing statistics
        if train_end_idx is None:
            stat_end = len(self.returns)
            print("WARNING: Computing statistics on all data (including test set)")
        else:
            stat_end = train_end_idx
            print(f"Computing statistics on training data only (first {stat_end} samples)")
        
        # Normalize returns per stock
        self.normalized_returns = np.zeros_like(self.returns)
        for i in range(len(self.stock_names)):
            mean = np.mean(self.returns[:stat_end, i])
            std = np.std(self.returns[:stat_end, i])
            if std > 0:
                self.normalized_returns[:, i] = (self.returns[:, i] - mean) / std
            else:
                self.normalized_returns[:, i] = 0
                
        # Normalize volatility per stock
        self.normalized_volatility = np.zeros_like(self.volatility)
        for i in range(len(self.stock_names)):
            mean = np.mean(self.volatility[:stat_end, i])
            std = np.std(self.volatility[:stat_end, i])
            if std > 0:
                self.normalized_volatility[:, i] = (self.volatility[:, i] - mean) / std
            else:
                self.normalized_volatility[:, i] = 0
                
        print("Normalization complete.")
        
    def create_sequences(self, lookback: int = None, volatility_window: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for training.
        
        Args:
            lookback: Number of past intervals to include
            volatility_window: Window size used for volatility computation (to skip initial samples)
            
        Returns:
            X_returns: (n_samples, lookback, n_stocks) - historical returns
            X_volatility: (n_samples, lookback, n_stocks) - historical volatility
            y: (n_samples, n_stocks) - target volatility (next interval)
        """
        if lookback is None:
            lookback = self.config.LOOKBACK_WINDOW
        if volatility_window is None:
            volatility_window = self.config.LOOKBACK_WINDOW
            
        print(f"Creating sequences with lookback={lookback}...")
        
        n_timesteps, n_stocks = self.normalized_returns.shape
        
        # Skip initial samples where volatility was not computed properly
        start_idx = volatility_window
        n_samples = n_timesteps - lookback - start_idx
        
        print(f"Skipping first {start_idx} samples (volatility burn-in period)")
        
        X_returns = np.zeros((n_samples, lookback, n_stocks), dtype=np.float32)
        X_volatility = np.zeros((n_samples, lookback, n_stocks), dtype=np.float32)  # Historical volatility
        y = np.zeros((n_samples, n_stocks), dtype=np.float32)
        
        for i in range(n_samples):
            actual_idx = start_idx + i
            X_returns[i] = self.normalized_returns[actual_idx:actual_idx+lookback]
            X_volatility[i] = self.normalized_volatility[actual_idx:actual_idx+lookback]
            y[i] = self.normalized_volatility[actual_idx+lookback]
            
        print(f"Created {n_samples} sequences.")
        return X_returns, X_volatility, y
    
    def split_data(self, X_returns, X_volatility, y, use_recent_split: bool = True, 
                  sequence_start_idx: int = 0) -> Tuple:
        """
        Split data into train, validation, and test sets chronologically.
        
        Args:
            use_recent_split: If True, train on last 4 years, validation on next 3 months, test on next 3 months.
                             If False, use config splits (70/15/15).
        
        Returns:
            Tuple of (train, val, test) datasets
        """
        n_samples = len(X_returns)
        
        if use_recent_split and hasattr(self, 'dates') and self.dates is not None:
            # Find the last 4 years + 6 months of data (4 years train + 3 months validation + 3 months test)
            # Dates are in format YYYYMMDD (e.g., 20161230)
            # Note: sequences start at sequence_start_idx in original data
            last_date = int(self.dates[-1])
            last_year = last_date // 10000  # Extract year (2016)
            last_month = (last_date // 100) % 100  # Extract month (12)
            
            # Calculate start date: 4 years + 6 months before last date
            # Go back 4 years
            start_year = last_year - 4
            # Go back 6 months from last month
            start_month = last_month - 6
            if start_month <= 0:
                start_month += 12
                start_year -= 1
            
            start_date = start_year * 10000 + start_month * 100 + 1  # e.g., 20120701
            end_date = last_date  # e.g., 20161230
            
            # Find indices in ORIGINAL data for this period
            period_indices_original = []
            for i in range(len(self.dates)):
                date_int = int(self.dates[i])
                if start_date <= date_int <= end_date:
                    period_indices_original.append(i)
            
            # Map to sequence indices (subtract sequence_start_idx)
            period_indices = [idx - sequence_start_idx for idx in period_indices_original 
                            if idx >= sequence_start_idx and idx - sequence_start_idx < n_samples]
            
            if len(period_indices) == 0:
                print("⚠️  Warning: Could not find recent period, using percentage split")
                use_recent_split = False
            else:
                # period_indices are now sequence indices
                period_start_seq_idx = period_indices[0]
                period_end_seq_idx = period_indices[-1] + 1
                period_samples = period_end_seq_idx - period_start_seq_idx
                
                # Find the boundary between 4 years (train), 3 months (validation), and 3 months (test)
                # 4 years from start
                train_end_year = start_year + 4
                train_end_month = start_month
                train_end_date = train_end_year * 10000 + train_end_month * 100 + 1
                
                # Validation end: 3 months after train end
                val_end_year = train_end_year
                val_end_month = train_end_month + 3
                if val_end_month > 12:
                    val_end_month -= 12
                    val_end_year += 1
                val_end_date = val_end_year * 10000 + val_end_month * 100 + 1
                
                # Find sequence indices for the boundaries
                train_end_seq_idx = None
                val_end_seq_idx = None
                
                for seq_idx in range(period_start_seq_idx, min(period_end_seq_idx, n_samples)):
                    # Map sequence index back to original data index
                    orig_idx = sequence_start_idx + seq_idx
                    if orig_idx < len(self.dates):
                        date_int = int(self.dates[orig_idx])
                        if train_end_seq_idx is None and date_int >= train_end_date:
                            train_end_seq_idx = seq_idx
                        if val_end_seq_idx is None and date_int >= val_end_date:
                            val_end_seq_idx = seq_idx
                            break
                
                # Fallback to percentage if exact dates not found
                if train_end_seq_idx is None:
                    # 4 years out of 4.5 years = ~88.9%
                    train_end_seq_idx = period_start_seq_idx + int(period_samples * 0.889)
                if val_end_seq_idx is None:
                    # 3 months out of 6 months = 50% of remaining period
                    remaining_samples = period_end_seq_idx - train_end_seq_idx
                    val_end_seq_idx = train_end_seq_idx + int(remaining_samples * 0.5)
                
                # Ensure we don't go beyond period_end_seq_idx or n_samples
                train_end_seq_idx = min(train_end_seq_idx, period_end_seq_idx, n_samples)
                val_end_seq_idx = min(val_end_seq_idx, period_end_seq_idx, n_samples)
                
                # Get original data indices for printing
                orig_period_start = sequence_start_idx + period_start_seq_idx
                orig_train_end = sequence_start_idx + train_end_seq_idx
                orig_val_end = sequence_start_idx + val_end_seq_idx
                orig_period_end = min(sequence_start_idx + period_end_seq_idx, len(self.dates))
                
                print(f"Using last 4 years + 6 months for split:")
                print(f"  Period: {self.dates[orig_period_start]} to {self.dates[orig_period_end-1]}")
                print(f"  Train (4 years): {self.dates[orig_period_start]} to {self.dates[orig_train_end-1]} ({train_end_seq_idx - period_start_seq_idx} samples)")
                print(f"  Validation (3 months): {self.dates[orig_train_end]} to {self.dates[orig_val_end-1]} ({val_end_seq_idx - train_end_seq_idx} samples)")
                print(f"  Test (3 months): {self.dates[orig_val_end]} to {self.dates[orig_period_end-1]} ({period_end_seq_idx - val_end_seq_idx} samples)")
                
                # Train: last 4 years (using sequence indices)
                X_ret_train = X_returns[period_start_seq_idx:train_end_seq_idx]
                X_vol_train = X_volatility[period_start_seq_idx:train_end_seq_idx]
                y_train = y[period_start_seq_idx:train_end_seq_idx]
                
                # Validation: 3 months after training
                X_ret_val = X_returns[train_end_seq_idx:val_end_seq_idx]
                X_vol_val = X_volatility[train_end_seq_idx:val_end_seq_idx]
                y_val = y[train_end_seq_idx:val_end_seq_idx]
                
                # Test: 3 months after validation
                X_ret_test = X_returns[val_end_seq_idx:period_end_seq_idx]
                X_vol_test = X_volatility[val_end_seq_idx:period_end_seq_idx]
                y_test = y[val_end_seq_idx:period_end_seq_idx]
                
                print(f"Data split: Train={len(X_ret_train)}, Dev={len(X_ret_val)}, Test={len(X_ret_test)}")
                
                return (X_ret_train, X_vol_train, y_train), \
                       (X_ret_val, X_vol_val, y_val), \
                       (X_ret_test, X_vol_test, y_test)
        
        # Fallback to percentage-based split
        if not use_recent_split:
            train_end = int(n_samples * self.config.TRAIN_SPLIT)
            val_end = int(n_samples * (self.config.TRAIN_SPLIT + 0.15))
        else:
            # If use_recent_split but dates not available, use 90/5/5 split on all data
            train_end = int(n_samples * 0.90)
            val_end = int(n_samples * 0.95)
        
        # Train
        X_ret_train = X_returns[:train_end]
        X_vol_train = X_volatility[:train_end]
        y_train = y[:train_end]
        
        # Validation
        X_ret_val = X_returns[train_end:val_end]
        X_vol_val = X_volatility[train_end:val_end]
        y_val = y[train_end:val_end]
        
        # Test
        X_ret_test = X_returns[val_end:]
        X_vol_test = X_volatility[val_end:]
        y_test = y[val_end:]
        
        print(f"Data split: Train={len(X_ret_train)}, Val={len(X_ret_val)}, Test={len(X_ret_test)}")
        
        return (X_ret_train, X_vol_train, y_train), \
               (X_ret_val, X_vol_val, y_val), \
               (X_ret_test, X_vol_test, y_test)


class VolatilityDataset(Dataset):
    """PyTorch Dataset for volatility prediction."""
    
    def __init__(self, X_returns, X_volatility, y, target_stock_idx: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            X_returns: Historical returns (n_samples, lookback, n_stocks)
            X_volatility: Historical volatility (n_samples, lookback, n_stocks)
            y: Target volatility (n_samples, n_stocks)
            target_stock_idx: If specified, only predict for this stock
        """
        self.X_returns = torch.FloatTensor(X_returns)
        self.X_volatility = torch.FloatTensor(X_volatility)
        self.y = torch.FloatTensor(y)
        self.target_stock_idx = target_stock_idx
        
    def __len__(self):
        return len(self.X_returns)
    
    def __getitem__(self, idx):
        if self.target_stock_idx is not None:
            # Return only the target stock's prediction
            return (self.X_returns[idx], 
                   self.X_volatility[idx], 
                   self.y[idx, self.target_stock_idx])
        else:
            return (self.X_returns[idx], 
                   self.X_volatility[idx], 
                   self.y[idx])

