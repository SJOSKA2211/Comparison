"""
BS-Opt Test Suite
Tests for ML feature engineering
"""

import pandas as pd
import numpy as np
from src.ml.feature_engineering import calculate_technical_indicators, generate_signals

class TestTechnicalIndicators:
    """Tests for technical indicator calculations"""

    def test_calculate_technical_indicators_happy_path(self):
        """Test with sufficient data to calculate all indicators"""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = np.linspace(100, 200, 100) + np.random.normal(0, 1, 100)
        df = pd.DataFrame({'price': prices}, index=dates)

        # Run calculation
        result = calculate_technical_indicators(df)

        # Verify columns exist
        expected_columns = ['sma_20', 'sma_50', 'rsi', 'returns', 'volatility']
        for col in expected_columns:
            assert col in result.columns

        # Verify no unexpected NaNs (after initial window)
        # SMA 50 needs 50 periods. So from index 49 onwards, it should be valid.
        assert not result['sma_50'].iloc[49:].isna().any()
        assert not result['sma_20'].iloc[19:].isna().any()

        # RSI needs 14 periods.
        assert not result['rsi'].iloc[14:].isna().any()

    def test_calculate_technical_indicators_insufficient_data(self):
        """Test with fewer rows than window size"""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        df = pd.DataFrame({'price': np.random.randn(10)}, index=dates)

        result = calculate_technical_indicators(df)

        # Columns should exist but contain mostly NaNs
        assert 'sma_50' in result.columns
        assert result['sma_50'].isna().all()  # window is 50, data is 10
        assert 'sma_20' in result.columns
        assert result['sma_20'].isna().all()  # window is 20, data is 10

    def test_calculate_technical_indicators_empty_df(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame({'price': []})
        result = calculate_technical_indicators(df)

        assert result.empty
        assert 'sma_20' in result.columns
        assert 'sma_50' in result.columns

    def test_rsi_calculation_increasing(self):
        """Test RSI for strictly increasing prices"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        prices = np.linspace(100, 200, 50)  # Strictly increasing
        df = pd.DataFrame({'price': prices}, index=dates)

        result = calculate_technical_indicators(df)

        # RSI should be 100 (or very close to it depending on implementation details of initial window)
        # Check last value
        assert result['rsi'].iloc[-1] == 100.0

    def test_rsi_calculation_decreasing(self):
        """Test RSI for strictly decreasing prices"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        prices = np.linspace(200, 100, 50)  # Strictly decreasing
        df = pd.DataFrame({'price': prices}, index=dates)

        result = calculate_technical_indicators(df)

        # RSI should be 0
        # Gain is 0. RS is 0. RSI is 0.
        assert result['rsi'].iloc[-1] == 0.0

    def test_constant_prices(self):
        """Test indicators for constant prices"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        df = pd.DataFrame({'price': [100.0] * 50}, index=dates)

        result = calculate_technical_indicators(df)

        # SMA should equal price
        assert (result['sma_20'].iloc[19:] == 100.0).all()
        assert (result['sma_50'].iloc[49:] == 100.0).all()

        # Returns should be 0
        assert (result['returns'].iloc[1:] == 0.0).all()

        # Volatility should be 0
        assert (result['volatility'].iloc[20:] == 0.0).all()


class TestSignalGeneration:
    """Tests for signal generation logic"""

    def test_generate_signals(self):
        """Test basic crossover logic"""
        df = pd.DataFrame({
            'sma_20': [100, 105, 110],
            'sma_50': [102, 102, 102]
        })
        # 0: 100 < 102 -> -1
        # 1: 105 > 102 -> 1
        # 2: 110 > 102 -> 1

        result = generate_signals(df)

        assert result['signal'].iloc[0] == -1
        assert result['signal'].iloc[1] == 1
        assert result['signal'].iloc[2] == 1

    def test_signal_neutral(self):
        """Test neutral signal (exact match)"""
        df = pd.DataFrame({
            'sma_20': [100],
            'sma_50': [100]
        })

        result = generate_signals(df)
        # Should remain 0 (initialized to 0)
        assert result['signal'].iloc[0] == 0
