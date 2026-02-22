import numpy as np
import pandas as pd
from typing import List, Optional

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic technical indicators for trading research"""
    # SMA
    df['sma_20'] = df['price'].rolling(window=20).mean()
    df['sma_50'] = df['price'].rolling(window=50).mean()

    # RSI
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Returns
    df['returns'] = df['price'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate basic trading signals based on SMA crossover"""
    df['signal'] = 0
    df.loc[df['sma_20'] > df['sma_50'], 'signal'] = 1
    df.loc[df['sma_20'] < df['sma_50'], 'signal'] = -1
    return df
