import pandas as pd
from decimal import Decimal, getcontext

# Set precision for Decimal
getcontext().prec = 10

# Normalization function with expanding windows

def normalize_data(prices):
    normalized = (prices - prices.expanding(min_periods=1).mean()) / prices.expanding(min_periods=1).std()
    return normalized

# Signal validation with boundary checking and epsilon tolerance

def validate_signal(signal, lower_bound, upper_bound, epsilon=1e-10):
    if signal < (lower_bound + epsilon) or signal > (upper_bound - epsilon):
        raise ValueError(f"Signal out of bounds: {signal}")
    return True

# Price data validation function

def validate_price_data(prices):
    if prices.isnull().any():
        raise ValueError("Price data contains null values")
    if (prices < 0).any():
        raise ValueError("Price data contains negative values")

# Precision PnL calculation using Decimal arithmetic

def calculate_pnl(entry_price, exit_price, quantity):
    entry_price = Decimal(entry_price)
    exit_price = Decimal(exit_price)
    quantity = Decimal(quantity)
    pnl = (exit_price - entry_price) * quantity
    return pnl

# Example usage
if __name__ == '__main__':
    prices = pd.Series([100.0, 102.0, 98.0, 105.0])
    normalized_prices = normalize_data(prices)
    print(normalized_prices)
