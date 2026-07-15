"""
Unit tests for the Auto-DCA package.

Tests cover:
1. Data preprocessing (decline-aware IQR filtering, shutin removal, normalization)
2. Decline curve models (hyperbolic, harmonic, exponential)
3. Error metrics (SMAPE)
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dca.utils import DataPreprocessor, MetricsCalculator
from dca.models import DeclineModels


class TestDataPreprocessing(unittest.TestCase):
    """Test decline-aware preprocessing: outlier filtering and normalization."""

    def setUp(self):
        """Create synthetic declining production data for testing."""
        # Simulate 10 years of quarterly declining production
        np.random.seed(42)
        n_points = 40  # 10 years quarterly

        # Create declining trend with some noise
        t = np.arange(n_points)
        # Hyperbolic-like decline: q(t) = qi / (1 + b*di*t)^(1/b)
        qi = 100  # Initial rate
        di = 0.3  # Decline rate
        b = 0.5   # Hyperbolic exponent

        base_rates = qi / (1 + b * di * t) ** (1/b)
        noise = np.random.normal(0, 2, n_points)
        rates = base_rates + noise

        # Add some realistic anomalies:
        # - Near-zero shutins (should be removed)
        rates[15] = 0.005  # Shutin
        rates[25] = 0.008  # Another shutin

        # - High early production (should be KEPT)
        rates[0] = 120  # Slightly higher than expected initial rate
        rates[1] = 115

        dates = pd.date_range(start='2010-01-01', periods=n_points, freq='Q')

        self.production_data = pd.DataFrame({
            'ZONE_PROD_PERIOD': dates,
            'GAS_PRODUCTION_RATE': rates,
            'GAS_PRODUCTION_E3M3': rates * 91.25  # Volume = rate * days
        })

    def test_preserves_high_production_values(self):
        """Test that high early production values are NOT removed."""
        preprocessor = DataPreprocessor()

        # Get the initial high value
        initial_high = self.production_data['GAS_PRODUCTION_RATE'].iloc[0]

        # Process with default settings (remove_upper_outliers=False)
        processed = preprocessor.preprocess_data(self.production_data.copy())

        # The high initial value should be preserved
        self.assertIn(initial_high, processed['GAS_PRODUCTION_RATE'].values,
                      "High initial production should be preserved")

    def test_removes_shutin_values(self):
        """Test that near-zero shutin values ARE removed."""
        preprocessor = DataPreprocessor()

        # Count near-zero values in original (may have more due to noise)
        original_near_zero = (self.production_data['GAS_PRODUCTION_RATE'] <= 0.01).sum()
        self.assertGreater(original_near_zero, 0, "Should have some shutin values in test data")

        # Process data
        processed = preprocessor.preprocess_data(self.production_data.copy())

        # Near-zero values should be removed
        processed_near_zero = (processed['GAS_PRODUCTION_RATE'] <= 0.01).sum()
        self.assertEqual(processed_near_zero, 0, "Shutin values should be removed")

    def test_default_preserves_more_than_aggressive_filtering(self):
        """Test that default settings preserve more data than aggressive IQR filtering."""
        preprocessor = DataPreprocessor()

        # Default settings (upper outliers preserved)
        processed_default = preprocessor.preprocess_data(self.production_data.copy())

        # Aggressive filtering (upper and lower outliers removed, tighter IQR bounds)
        processed_aggressive = preprocessor.preprocess_data(
            self.production_data.copy(),
            remove_upper_outliers=True,
            iqr_multiplier=1.5
        )

        # Default settings should preserve >= data points
        self.assertGreaterEqual(
            len(processed_default), len(processed_aggressive),
            "Default settings should preserve at least as many data points"
        )

    def test_normalization_works(self):
        """Test that normalization produces values in [0, 1] range."""
        preprocessor = DataPreprocessor()
        processed = preprocessor.preprocess_data(self.production_data.copy())

        # Check normalized values are in [0, 1]
        self.assertTrue(
            (processed['GAS_PRODUCTION_RATE_NORMALIZED'] >= 0).all(),
            "Normalized values should be >= 0"
        )
        self.assertTrue(
            (processed['GAS_PRODUCTION_RATE_NORMALIZED'] <= 1).all(),
            "Normalized values should be <= 1"
        )

        # Maximum should be exactly 1.0
        self.assertAlmostEqual(
            processed['GAS_PRODUCTION_RATE_NORMALIZED'].max(), 1.0,
            places=6, msg="Max normalized value should be 1.0"
        )

    def test_iqr_multiplier_parameter(self):
        """Test that IQR multiplier parameter affects filtering."""
        preprocessor = DataPreprocessor()

        # More aggressive (smaller multiplier)
        processed_aggressive = preprocessor.preprocess_data(
            self.production_data.copy(),
            iqr_multiplier=1.0
        )

        # Less aggressive (larger multiplier)
        processed_conservative = preprocessor.preprocess_data(
            self.production_data.copy(),
            iqr_multiplier=3.0
        )

        # Conservative should have >= data points
        self.assertGreaterEqual(
            len(processed_conservative), len(processed_aggressive),
            "Higher IQR multiplier should preserve more data"
        )


class TestDeclineModels(unittest.TestCase):
    """Test the decline curve model equations."""

    def test_hyperbolic_model(self):
        """Test hyperbolic decline model."""
        models = DeclineModels()

        # Test hyperbolic model
        t = np.array([0, 1, 2, 3, 4, 5])
        qi, di, b = 1.0, 0.2, 0.5
        q = models.hyperbolic(t, qi, di, b)

        # Should return declining values
        self.assertTrue(all(np.diff(q) <= 0), "Hyperbolic should decline over time")
        self.assertEqual(q[0], qi, "Initial rate should equal qi")

    def test_exponential_model(self):
        """Test exponential decline model."""
        models = DeclineModels()

        t = np.array([0, 1, 2, 3, 4, 5])
        qi, di = 1.0, 0.1
        q = models.exponential(t, qi, di)

        # Should return declining values
        self.assertTrue(all(np.diff(q) < 0), "Exponential should strictly decline")

    def test_harmonic_model(self):
        """Test harmonic decline model."""
        models = DeclineModels()

        t = np.array([0, 1, 2, 3, 4, 5])
        qi, di = 1.0, 0.1
        q = models.harmonic(t, qi, di)

        # Should return declining values
        self.assertTrue(all(np.diff(q) < 0), "Harmonic should strictly decline")


class TestMetricsCalculator(unittest.TestCase):
    """Test metrics calculations."""

    def test_smape_calculation(self):
        """Test SMAPE calculation."""
        calc = MetricsCalculator()

        y_true = np.array([100, 80, 60, 40, 20])
        y_pred = np.array([95, 85, 55, 45, 18])

        smape = calc.symmetric_mean_absolute_percentage_error(y_true, y_pred)

        # SMAPE should be between 0 and 100
        self.assertGreaterEqual(smape, 0, "SMAPE should be >= 0")
        self.assertLessEqual(smape, 100, "SMAPE should be <= 100")

    def test_smape_perfect_prediction(self):
        """Test SMAPE with perfect predictions."""
        calc = MetricsCalculator()

        y_true = np.array([100, 80, 60, 40, 20])
        y_pred = y_true.copy()

        smape = calc.symmetric_mean_absolute_percentage_error(y_true, y_pred)
        self.assertEqual(smape, 0, "Perfect prediction should have SMAPE = 0")


class TestDeclineCurveConstraints(unittest.TestCase):
    """Test decline curve fitting constraints."""

    def test_b_factor_bounds(self):
        """Test that b-factor stays within expected bounds."""
        models = DeclineModels()

        # For valid hyperbolic decline, b should be in [0, 1] typically
        # (can be higher for unconventional wells)
        t = np.linspace(0, 10, 50)

        # Test with b = 0 (exponential limit)
        q_exp_like = models.hyperbolic(t, 1.0, 0.1, 0.001)
        self.assertTrue(all(q_exp_like > 0), "Production should always be positive")

        # Test with b = 1 (harmonic)
        q_harmonic_like = models.hyperbolic(t, 1.0, 0.1, 1.0)
        self.assertTrue(all(q_harmonic_like > 0), "Production should always be positive")


if __name__ == '__main__':
    unittest.main(verbosity=2)
