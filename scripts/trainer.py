"""
Trainer class for DeltaLag attention model.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from model import DeltaLagAttentionModel
from metrics import calculate_metrics


class SimpleTrainer:
    """Trainer for DeltaLag attention model."""
    
    def __init__(self, config: Config, target_stock_idx: int, stock_name: str, patience: int = 10):
        self.config = config
        self.target_stock_idx = target_stock_idx
        self.stock_name = stock_name
        self.device = config.DEVICE
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.attention_entropy_weight = 0.0001
        self.best_val_loss = float('inf')
        self.patience = patience
        self.patience_counter = 0
        
    def setup_model(self, n_stocks: int, dropout: float = 0.0, 
                    mlp_hidden1: int = 128, mlp_hidden2: int = 64):
        """Setup model and optimizer."""
        self.model = DeltaLagAttentionModel(
            n_stocks=n_stocks,
            lookback=self.config.LOOKBACK_WINDOW,
            d_model=self.config.D_MODEL,
            d_k=self.config.D_K,
            d_v=self.config.D_V,
            max_lag=self.config.MAX_LAG,
            dropout=dropout,
            mlp_hidden1=mlp_hidden1,
            mlp_hidden2=mlp_hidden2
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        pbar = tqdm(train_loader, desc='Training')
        for X_returns, X_volatility, y in pbar:
            X_returns = X_returns.to(self.device)
            X_volatility = X_volatility.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            predictions, attention_info = self.model(
                X_returns, X_volatility, self.target_stock_idx, return_attention=True
            )
            
            mse_loss = self.criterion(predictions.squeeze(), y)
            
            total_loss = mse_loss
            mean_entropy = 0.0
            if attention_info and self.attention_entropy_weight > 0:
                attn_weights = attention_info['attention_weights']
                entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=1)
                mean_entropy = torch.mean(entropy)
                max_entropy = np.log(attn_weights.shape[1])
                entropy_penalty = -mean_entropy / max_entropy
                total_loss = mse_loss + self.attention_entropy_weight * entropy_penalty
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            pbar.set_postfix({
                'loss': mse_loss.item(), 
                'entropy': mean_entropy.item() if isinstance(mean_entropy, torch.Tensor) else 0
            })
        
        # Compute training loss in eval mode
        self.model.eval()
        train_losses = []
        with torch.no_grad():
            for X_returns, X_volatility, y in train_loader:
                X_returns = X_returns.to(self.device)
                X_volatility = X_volatility.to(self.device)
                y = y.to(self.device)
                
                predictions, _ = self.model(
                    X_returns, X_volatility, self.target_stock_idx, return_attention=False
                )
                loss = self.criterion(predictions.squeeze(), y)
                train_losses.append(loss.item())
        
        return {'total_loss': np.mean(train_losses)}
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """Validate the model."""
        self.model.eval()
        losses = []
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for X_returns, X_volatility, y in val_loader:
                X_returns = X_returns.to(self.device)
                X_volatility = X_volatility.to(self.device)
                y = y.to(self.device)
                
                predictions, _ = self.model(
                    X_returns, X_volatility, self.target_stock_idx, return_attention=False
                )
                loss = self.criterion(predictions.squeeze(), y)
                
                losses.append(loss.item())
                predictions_list.append(predictions.squeeze().cpu().numpy())
                targets_list.append(y.cpu().numpy())
        
        all_predictions = np.concatenate(predictions_list)
        all_targets = np.concatenate(targets_list)
        metrics = calculate_metrics(all_predictions, all_targets)
        
        return {'mse_loss': np.mean(losses)}, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Train the model."""
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': []
        }
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{self.config.NUM_EPOCHS}")
            
            train_losses = self.train_epoch(train_loader)
            history['train_loss'].append(train_losses['total_loss'])
            
            val_losses, val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_losses['mse_loss'])
            history['val_rmse'].append(val_metrics['rmse'])
            
            # Calculate train RMSE
            self.model.eval()
            train_preds = []
            train_targets = []
            with torch.no_grad():
                for X_returns, X_volatility, y in train_loader:
                    X_returns = X_returns.to(self.device)
                    X_volatility = X_volatility.to(self.device)
                    preds, _ = self.model(
                        X_returns, X_volatility, self.target_stock_idx, return_attention=False
                    )
                    train_preds.append(preds.squeeze().cpu().numpy())
                    train_targets.append(y.numpy())
            train_metrics = calculate_metrics(
                np.concatenate(train_preds), np.concatenate(train_targets)
            )
            history['train_rmse'].append(train_metrics['rmse'])
            
            print(f"Train Loss: {train_losses['total_loss']:.6f}, "
                  f"Validation Loss: {val_losses['mse_loss']:.6f}")
            print(f"Validation RMSE: {val_metrics['rmse']:.6f}")
            
            # Save best model and check for early stopping
            if val_losses['mse_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['mse_loss']
                self.patience_counter = 0  # Reset patience counter
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(
                    self.model.state_dict(), 
                    f"checkpoints/{self.stock_name}_deltalag_best.pt"
                )
                print(f"✓ Best model saved (val_loss: {self.best_val_loss:.6f})")
            else:
                self.patience_counter += 1
                print(f"  No improvement. Patience: {self.patience_counter}/{self.patience}")
                
                # Early stopping
                if self.patience_counter >= self.patience:
                    print(f"\n{'='*80}")
                    print(f"Early stopping triggered after {epoch} epochs")
                    print(f"Best validation loss: {self.best_val_loss:.6f}")
                    print(f"{'='*80}\n")
                    break
        
        # Load best model
        self.model.load_state_dict(
            torch.load(f"checkpoints/{self.stock_name}_deltalag_best.pt")
        )
        
        return history

