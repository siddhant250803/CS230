"""
Configuration file for the DeltaLag attention model.
"""

import torch

class Config:
    """Configuration parameters for the model and training."""
    
    # Data parameters
    DATA_PATH = "data/HF_Returns_Stocks.csv"
    MISSING_VALUE = -1.04e-07  # Placeholder for missing data in the dataset
    LOOKBACK_WINDOW = 12  # Number of 5-min intervals to look back (60 minutes)
    
    # Model parameters
    D_MODEL = 64  # Dimension of embeddings
    D_K = 32  # Dimension of query/key vectors
    D_V = 32  # Dimension of value vectors
    MAX_LAG = 12  # Maximum lag to consider (in 5-min intervals = 60 minutes)
    DROPOUT = 0.1
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    TRAIN_SPLIT = 0.70  # Used only in fallback split
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __repr__(self):
        return f"Config(d_model={self.D_MODEL}, batch_size={self.BATCH_SIZE}, device={self.DEVICE})"

