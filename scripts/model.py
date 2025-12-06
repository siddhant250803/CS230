""" DeltaLag style inspired attention model with learned lag distributions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DeltaLagAttention(nn.Module):
    """
    DeltaLag-style like attention: implements the lag-based attention computation, 
    learns a distribution over lags for each stock, creates lag-stacked embeddings
    and performs cross-stock attention using resulting embeddings.
    """
    
    def __init__(self, n_stocks: int, d_model: int, d_k: int, d_v: int, max_lag: int):
        super().__init__()
        self.n_stocks = n_stocks
        self.d_k = d_k
        self.d_v = d_v
        self.max_lag = max_lag
        
        # Q/K/V  linear layers initialization
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        
        # Lag logits per stock to learn the lag distribution for each stock
        self.lag_logits = nn.Parameter(torch.randn(n_stocks, max_lag))
    
    def forward(self, query_stock_emb: torch.Tensor, 
                all_stock_embs: torch.Tensor,
                return_attention: bool = False):

        """
        Args:
            query_stock_emb: (batch, d_model)
            all_stock_embs: (batch, lookback, n_stocks, d_model)
        
        Returns:
            context: (batch, d_v)
            attention_weights: (batch, n_stocks) if return_attention
            lag_probs: (n_stocks, max_lag) if return_attention
        """
        batch_size, lookback, n_stocks, d_model = all_stock_embs.shape
        
        # Query for target stock
        q = self.W_q(query_stock_emb)  # (batch, d_k)
        
        # Use softmax to convert lag logits into categorical distribution
        lag_probs = F.softmax(self.lag_logits, dim=1)
        
        values = []
        scores = []
        
        # Loop over # of stocks (n_stocks)
        for i in range(n_stocks):

            # Past embeddings for stock i: (batch, lookback, d_model)
            past_i = all_stock_embs[:, :, i, :]
            
            # Construct lag-stacked views x_(t-l,i) 
            stacked = []
            for lag_val in range(self.max_lag):
                # ensure no negative index
                idx = max(0, min(lookback - 1 - lag_val, lookback - 1)) # this fixes index issues
                stacked.append(past_i[:, idx, :])
            stacked = torch.stack(stacked, dim=1)  # shape here is (batch, max_lag, d_model)
            
            # expected embedding under lag distribution
            stock_emb = torch.sum(stacked * lag_probs[i].view(1, -1, 1), dim=1)  # calculates the EXPECTED value of the embeddings
            
            # project lag-weighted stock embedding into key and value vectors for attention
            k = self.W_k(stock_emb)
            v = self.W_v(stock_emb)
            
            # attention score
            score = torch.sum(q * k, dim=-1) / np.sqrt(self.d_k)
            
            # append score and value to list
            scores.append(score)
            values.append(v)
        
        # calculate scores, attention weights, values, and context
        scores = torch.stack(scores, dim=1)  # (batch, n_stocks)
        attention_weights = F.softmax(scores, dim=1) 
        values = torch.stack(values, dim=1)  # (batch, n_stocks, d_v)
        context = torch.sum(attention_weights.unsqueeze(-1) * values, dim=1)
        
        ## REMOVE FROM FUNCTION PARAMETER
        if True:#return_attention:
            return context, attention_weights, lag_probs
        # else:
            # return context


class DeltaLagAttentionModel(nn.Module):
    """Full model using DeltaLag-style lag attention."""
    
    def __init__(self, n_stocks: int, lookback: int,
                 d_model: int = 64, d_k: int = 32, d_v: int = 32,
                 max_lag: int = 12, dropout: float = 0.0,
                 mlp_hidden1: int = 128, mlp_hidden2: int = 64):
        #note: default values is set for our runs
        
        
        super().__init__()
        self.n_stocks = n_stocks
        self.lookback = lookback
        self.d_model = d_model
        self.max_lag = max_lag
        

        # define linear layer for stock embedding (return, volatility)
        self.stock_embedding = nn.Linear(2, d_model) # 2 because we have return and volatility
        
        # define layers for target-stock embedding
        self.target_embedding = nn.Sequential(
            nn.Linear(lookback * 2, 128),  # 2 because we have return and volatility
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        
        # create instance of the DeltaLagAttention class
        self.attention = DeltaLagAttention(n_stocks, d_model, d_k, d_v, max_lag)
        
        # Sequential layers with activation relu 
        self.mlp = nn.Sequential(
            nn.Linear(d_v + d_model, mlp_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden1, mlp_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden2, 1)
        )
    

    def forward(self, X_returns: torch.Tensor, X_volatility: torch.Tensor, 
                target_stock_idx: int, return_attention: bool = False):
        # combine returns  and volatility: (batch, lookback, n_stocks, 2)
        stock_features = torch.stack([X_returns, X_volatility], dim=-1)
        
        # Stocks embeddings
        all_stock_embs = self.stock_embedding(stock_features)
        
        # Target stock embedding
        target_stock_returns = X_returns[:, :, target_stock_idx]
        target_stock_vol = X_volatility[:, :, target_stock_idx]
        target_info = torch.cat([target_stock_returns, target_stock_vol], dim=1)
        target_emb = self.target_embedding(target_info)
        
        # Apply DeltaLag attention
        context, attn_weights, lag_probs = self.attention(target_emb, all_stock_embs, return_attention)
        
        # concatenate context with target embedding to info from other stocks and target stock
        concat_context_target_emb = torch.cat([context, target_emb], dim=-1)
        
        # MLP layer to predict volatility
        prediction = self.mlp(concat_context_target_emb)
        
        
        attn_lag_info = {
            "attention_weights": attn_weights,      # (batch, n_stocks)
            "lag_probs": lag_probs,                # (n_stocks, max_lag)
            "expected_lags": self.get_expected_lags(lag_probs)
            }

        ## REMOVE FROM FUNCTION PARAMETER
        if True:# return_attention:
            return prediction, attn_lag_info
        # else:
        #     return prediction
    

    def get_expected_lags(self, lag_probs: torch.Tensor) -> np.ndarray:
        """compute expected lag for each stock"""
        lags = torch.arange(self.max_lag, device=lag_probs.device).float()
        expected_lags = (lag_probs * lags[None, :]).sum(dim=1)
        return expected_lags.detach().cpu().numpy()