"""
Utility functions for stock data processing and sector classification.
"""


def get_sector_label(stock: str) -> str:
    """
    Get sector label for a stock.
    
    Args:
        stock: Stock ticker symbol
        
    Returns:
        Sector label: 'OIL', 'TECH', or 'OTHER'
    """
    oil_set = {"XOM", "CVX", "SLB", "HAL", "BP", "SHEL", "COP"}
    tech_set = {"AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "ADBE"}
    
    if stock in oil_set:
        return 'OIL'
    elif stock in tech_set:
        return 'TECH'
    else:
        return 'OTHER'


def build_stock_universe(data_loader, target_stock: str, other_stocks: list, 
                         market_stock: str = "SPY") -> tuple:
    """
    Build stock universe with target, market, and other stocks.
    
    Args:
        data_loader: StockDataLoader instance
        target_stock: Target stock ticker
        other_stocks: List of other stock tickers
        market_stock: Market proxy stock (default: SPY)
        
    Returns:
        Tuple of (stock_universe_list, sector_labels_dict)
    """
    available = data_loader.stock_names
    
    def filter_existing(candidates: list) -> list:
        present = []
        missing = []
        for ticker in candidates:
            if ticker in available:
                if ticker not in present:
                    present.append(ticker)
            else:
                missing.append(ticker)
        if missing:
            print(f"⚠️  Missing tickers: {', '.join(missing)}")
        return present
    
    other_present = filter_existing(other_stocks)
    
    # Build final list: target + market + other stocks
    final_list = []
    sector_labels = {}
    
    # Add target stock
    if target_stock in available:
        final_list.append(target_stock)
        sector_labels[target_stock] = 'TARGET'
    else:
        raise ValueError(f"Target stock '{target_stock}' not found")
    
    # Add market stock
    if market_stock in available:
        final_list.append(market_stock)
        sector_labels[market_stock] = 'MARKET'
    
    # Add other stocks with sector labels
    for stock in other_present:
        if stock not in final_list:
            final_list.append(stock)
            sector_labels[stock] = get_sector_label(stock)
    
    print(f"\nFinal stock universe ({len(final_list)} stocks):")
    print(f"  Target: {target_stock}")
    print(f"  Market: {market_stock if market_stock in final_list else 'N/A'}")
    print(f"  Other ({len(other_present)}): {', '.join(other_present)}")
    
    return final_list, sector_labels


def calculate_train_end_idx(data_loader, n_timesteps: int) -> int:
    """
    Calculate training end index for normalization (last 4 years).
    
    Args:
        data_loader: StockDataLoader instance
        n_timesteps: Total number of timesteps
        
    Returns:
        Training end index
    """
    if hasattr(data_loader, 'dates') and data_loader.dates is not None:
        last_date = int(data_loader.dates[-1])
        last_year = last_date // 10000
        last_month = (last_date // 100) % 100
        
        start_year = last_year - 4
        start_month = last_month - 6
        if start_month <= 0:
            start_month += 12
            start_year -= 1
        
        start_date = start_year * 10000 + start_month * 100 + 1
        train_end_year = start_year + 4
        train_end_month = start_month
        train_end_date = train_end_year * 10000 + train_end_month * 100 + 1
        
        period_indices = []
        for i in range(n_timesteps):
            date_int = int(data_loader.dates[i])
            if start_date <= date_int <= last_date:
                period_indices.append(i)
        
        if len(period_indices) > 0:
            period_start_idx = period_indices[0]
            train_end_idx = None
            for i in range(period_start_idx, len(data_loader.dates)):
                date_int = int(data_loader.dates[i])
                if train_end_idx is None and date_int >= train_end_date:
                    train_end_idx = i
                    break
            
            if train_end_idx is None:
                train_end_idx = period_start_idx + int(len(period_indices) * 0.889)
            
            print(f"\nStep 3: Normalizing using last 4 years training data "
                  f"({train_end_idx - period_start_idx} samples, "
                  f"{data_loader.dates[period_start_idx]} to {data_loader.dates[train_end_idx-1]})...")
            return train_end_idx
    
    # Fallback
    train_end_idx = int(n_timesteps * 0.889)
    print(f"\nStep 3: Normalizing using first {train_end_idx} timesteps (88.9%)...")
    return train_end_idx

