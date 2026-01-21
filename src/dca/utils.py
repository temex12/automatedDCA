"""
Utility Classes for Decline Curve Analysis (DCA)

This module provides essential utility functions for DCA calculations and data preprocessing:

1. MetricsCalculator:
   - Error metrics calculation (SMAPE)
   - Decline rate calculations
   - Rate smoothing functions

2. DataPreprocessor:
   - Production data cleaning
   - Outlier removal
   - Data normalization
   - Switch time detection for terminal decline

Key Features:
- Robust error metric calculations
- Automated data cleaning and preprocessing
- Terminal decline detection algorithms

Notes:
- All calculations handle edge cases and invalid data
- Methods are designed for production time series analysis
- Supports both conventional and unconventional well analysis
"""

import numpy as np
import pandas as pd


class MetricsCalculator:
    """
    Utility class for calculating DCA-related metrics and error measurements.
    
    Features:
    - Symmetric Mean Absolute Percentage Error (SMAPE) calculation
    - Instantaneous decline rate calculations
    - Time series smoothing functions
    
    All methods are implemented as static methods for easy access.
    """
    
    @staticmethod
    def symmetric_mean_absolute_percentage_error(y_true, y_pred):
        """
        Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
        
        SMAPE is preferred over MAPE because it:
        - Handles zero and near-zero values better
        - Treats over and under predictions more fairly
        - Has a bounded range of 0-100%
        
        Args:
            y_true (array-like): Actual production values
            y_pred (array-like): Model predicted values
            
        Returns:
            float: SMAPE value as percentage (0-100%)
            
        Notes:
            - More robust than traditional MAPE
            - Symmetric around zero
            - Less biased for small values
        """
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    @staticmethod
    def calculate_decline_rate(best_func, t1, q1, best_params):
        """
        Calculate instantaneous nominal decline rate at a specific time.
        
        Args:
            best_func (callable): Decline curve function (hyperbolic, harmonic, exponential)
            t1 (float): Time point for calculation (years)
            q1 (float): Production rate at time t1
            best_params (tuple): Model parameters (varies by model type):
                - Hyperbolic: (qi, di, b)
                - Harmonic: (qi, di)
                - Exponential: (qi, di)
            
        Returns:
            float: Absolute decline rate (fraction per year)
            
        Raises:
            ValueError: If model type is not supported
            
        Notes:
            - Hyperbolic: D(t) = di / (1 + b * di * t)
            - Harmonic: D(t) = di / (1 + di * t)
            - Exponential: D(t) = di (constant)
        """
        if best_func.__name__ == 'hyperbolic':
            qi, di, b = best_params
            decline_rate = (di / (1 + b * di * t1))
        elif best_func.__name__ == 'harmonic':
            qi, di = best_params
            decline_rate = (di / (1 + di * t1))
        elif best_func.__name__ == 'exponential':
            qi, di = best_params
            decline_rate = di
        else:
            raise ValueError(f"Unsupported model type: {best_func.__name__}")
        return np.abs(decline_rate)
    
    @staticmethod
    def smooth_decline_rates(decline_rates, window=3):
        """
        Smooth decline rates using centered moving average.
        
        Args:
            decline_rates (array-like): Raw decline rate values
            window (int): Smoothing window size (odd number recommended)
            
        Returns:
            numpy.ndarray: Smoothed decline rates
            
        Notes:
            - Uses centered moving average
            - Handles edges using forward/backward fill
            - Reduces noise while preserving trends
        """
        return pd.Series(decline_rates).rolling(
            window=window, 
            center=True
        ).mean().fillna(method='bfill').fillna(method='ffill').values


class DataPreprocessor:
    """
    Utility class for preprocessing production data before DCA analysis.
    
    Features:
    - Outlier detection and removal
    - Production rate normalization
    - Terminal decline detection
    - Data validation and cleaning
    
    All methods are implemented as static methods for easy use.
    """
    
    @staticmethod
    def preprocess_data(data, remove_upper_outliers=False, iqr_multiplier=2.0):
        """
        Clean and normalize production data for analysis.

        This method uses a decline-curve-aware approach to outlier removal:
        - For declining production, high early values are expected (not outliers)
        - Only clearly anomalous low values (shutins, near-zero) are removed by default
        - Upper outlier removal is disabled by default to preserve peak production data

        Args:
            data (pd.DataFrame): Raw production data with 'GAS_PRODUCTION_RATE' column
            remove_upper_outliers (bool): If True, also remove upper outliers using IQR.
                                         Default False to preserve high production values.
            iqr_multiplier (float): Multiplier for IQR bounds (default 2.0, more conservative
                                   than traditional 1.5 to preserve more data points)

        Returns:
            pd.DataFrame: Cleaned DataFrame with additional columns:
                - GAS_PRODUCTION_RATE_NORMALIZED: Production rates scaled to [0,1]

        Notes:
            Data Cleaning Steps:
            1. Remove negligible rates (≤ 0.01) - captures shutins and zero production
            2. Remove only LOWER outliers by default (anomalous low production during
               what should be productive periods)
            3. Optionally remove upper outliers if remove_upper_outliers=True
            4. Normalize remaining rates to maximum value

        The modified IQR approach is more appropriate for decline curves where:
        - High production early in well life is expected, not anomalous
        - Very low production (near-zero) indicates operational issues (shutins, workovers)
        - The decline trend itself should be preserved
        """
        # Remove negligible production rates (shutins, zero production periods)
        data = data[data['GAS_PRODUCTION_RATE'] > 0.01].copy()

        # Calculate outlier bounds using IQR method with configurable multiplier
        Q1 = data['GAS_PRODUCTION_RATE'].quantile(0.25)
        Q3 = data['GAS_PRODUCTION_RATE'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR

        # Ensure lower bound doesn't go negative
        lower_bound = max(lower_bound, 0.01)

        # Remove only lower outliers by default (anomalous low production)
        data = data[data['GAS_PRODUCTION_RATE'] >= lower_bound]

        # Optionally remove upper outliers (disabled by default for decline curves)
        if remove_upper_outliers:
            upper_bound = Q3 + iqr_multiplier * IQR
            data = data[data['GAS_PRODUCTION_RATE'] <= upper_bound]

        # Normalize to maximum rate
        max_production = data['GAS_PRODUCTION_RATE'].max()
        data['GAS_PRODUCTION_RATE_NORMALIZED'] = data['GAS_PRODUCTION_RATE'] / max_production

        return data
    
    @staticmethod
    def find_switch_time(forecast_data, best_func, t_forecast, best_params, decline_threshold,
                        min_switch_time=2, max_switch_time=10, validation_period=1, 
                        smoothing_window=3):
        """
        Determine optimal time to switch from hyperbolic to terminal decline.
        
        This method identifies when the decline rate stabilizes below a threshold,
        indicating transition to terminal decline phase.
        
        Args:
            forecast_data (pd.DataFrame): Production forecast data
            best_func (callable): Best-fit decline function
            t_forecast (array-like): Time points for forecast
            best_params (tuple): Parameters of best-fit model
            decline_threshold (float): Terminal decline rate threshold
            min_switch_time (int): Minimum years before considering switch
            max_switch_time (int): Maximum years before forcing switch
            validation_period (float): Years of sustained decline required
            smoothing_window (int): Window for decline rate smoothing
        
        Returns:
            float: Optimal switch time in years
            
        Algorithm Steps:
        1. Calculate decline rates along forecast
        2. Smooth rates to reduce noise
        3. Find sustained decline below threshold
        4. Validate for specified period
        5. Apply min/max constraints
        
        Notes:
        - Important for unconventional wells
        - Prevents unrealistic late-time forecasts
        - Considers both technical and economic factors
        """
        metrics_calc = MetricsCalculator()
        
        # Calculate instantaneous decline rates
        decline_rates = forecast_data.apply(
            lambda row: metrics_calc.calculate_decline_rate(
                best_func,
                row['Years'],
                best_func(row['Years'], *best_params),
                best_params
            ),
            axis=1
        ).values
        
        # Smooth decline rates to reduce noise
        smooth_rates = metrics_calc.smooth_decline_rates(decline_rates, window=smoothing_window)
        forecast_data['Decline_Rate'] = smooth_rates
        
        # Apply minimum time constraint
        valid_data = forecast_data[forecast_data['Years'] >= min_switch_time]
        
        # Find periods below threshold
        below_threshold = valid_data[valid_data['Decline_Rate'] <= decline_threshold]
        
        if len(below_threshold) > 0:
            # Look for sustained decline below threshold
            potential_switches = []
            current_run_start = None
            
            for idx, row in below_threshold.iterrows():
                if current_run_start is None:
                    current_run_start = row['Years']
                elif row['Years'] - current_run_start >= validation_period:
                    potential_switches.append(current_run_start)
                    break
                
            if potential_switches:
                # Return earliest valid switch time within constraints
                switch_time = min(potential_switches[0], max_switch_time)
                return switch_time
        
        # Default to maximum switch time if no valid point found
        return max_switch_time