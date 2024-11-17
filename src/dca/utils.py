"""
Utility classes for Decline Curve Analysis (DCA).
Contains metrics calculation and data preprocessing functionality.
"""
import numpy as np
import pandas as pd


class MetricsCalculator:
    """
    Class containing static methods for calculating various DCA-related metrics.
    """
    
    @staticmethod
    def symmetric_mean_absolute_percentage_error(y_true, y_pred):
        """
        Calculate the symmetric mean absolute percentage error between true and predicted values.
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            
        Returns:
            float: Symmetric MAPE value as a percentage
        """
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    @staticmethod
    def calculate_decline_rate(best_func, t1, q1, best_params):
        """
        Calculate instantaneous decline rate at time t1.
        
        Args:
            best_func (callable): The decline curve function (hyperbolic, harmonic, or exponential)
            t1 (float): Time point at which to calculate decline rate
            q1 (float): Production rate at time t1
            best_params (tuple): Parameters of the decline curve function
            
        Returns:
            float: Absolute decline rate at time t1
            
        Raises:
            ValueError: If the function type is not supported
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
        Apply smoothing to decline rates using rolling average.
        
        Args:
            decline_rates (array-like): Array of decline rates
            window (int): Size of the smoothing window
            
        Returns:
            numpy.ndarray: Smoothed decline rates
        """
        return pd.Series(decline_rates).rolling(
            window=window, 
            center=True
        ).mean().fillna(method='bfill').fillna(method='ffill').values


class DataPreprocessor:
    """
    Class containing static methods for preprocessing production data.
    """
    
    @staticmethod
    def preprocess_data(data):
        """
        Preprocess production data by removing outliers and normalizing production rates.
        
        Args:
            data (pd.DataFrame): DataFrame containing production data with 'GAS_PRODUCTION_RATE' column
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame with outliers removed and normalized production rates
            
        Notes:
            - Removes data points with production rate <= 0.01
            - Removes outliers using 1.5 * IQR method
            - Adds 'GAS_PRODUCTION_RATE_NORMALIZED' column with normalized production rates
        """
        # Remove very low production rates
        data = data[data['GAS_PRODUCTION_RATE'] > 0.01].copy()
        
        # Calculate IQR bounds
        Q1 = data['GAS_PRODUCTION_RATE'].quantile(0.25)
        Q3 = data['GAS_PRODUCTION_RATE'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers
        data = data[(data['GAS_PRODUCTION_RATE'] >= lower_bound) & 
                   (data['GAS_PRODUCTION_RATE'] <= upper_bound)]
        
        # Normalize production rates
        max_production = data['GAS_PRODUCTION_RATE'].max()
        data['GAS_PRODUCTION_RATE_NORMALIZED'] = data['GAS_PRODUCTION_RATE'] / max_production
        
        return data
    
    @staticmethod
    def find_switch_time(forecast_data, best_func, t_forecast, best_params, decline_threshold,
                        min_switch_time=2, max_switch_time=10, validation_period=1, 
                        smoothing_window=3):
        """
        Find the time point where decline rate drops below threshold with additional constraints.
        
        Args:
            forecast_data (pd.DataFrame): DataFrame with forecast data
            best_func (callable): Best fitting decline function
            t_forecast (array-like): Time array for forecast
            best_params (tuple): Parameters of the best fitting function
            decline_threshold (float): Threshold for switching to exponential
            min_switch_time (int): Minimum years before allowing switch (default: 2)
            max_switch_time (int): Maximum years to enforce switch (default: 10)
            validation_period (float): Years to validate sustained decline below threshold (default: 1)
            smoothing_window (int): Window size for decline rate smoothing (default: 3)
        
        Returns:
            float: Switch time in years
        """
        metrics_calc = MetricsCalculator()
        
        # Calculate decline rates
        decline_rates = forecast_data.apply(
            lambda row: metrics_calc.calculate_decline_rate(
                best_func,
                row['Years'],
                best_func(row['Years'], *best_params),
                best_params
            ),
            axis=1
        ).values
        
        # Apply smoothing to decline rates
        smooth_rates = metrics_calc.smooth_decline_rates(decline_rates, window=smoothing_window)
        forecast_data['Decline_Rate'] = smooth_rates
        
        # Filter data based on minimum switch time
        valid_data = forecast_data[forecast_data['Years'] >= min_switch_time]
        
        # Find points below threshold
        below_threshold = valid_data[valid_data['Decline_Rate'] <= decline_threshold]
        
        if len(below_threshold) > 0:
            # Check for sustained decline below threshold
            potential_switches = []
            current_run_start = None
            
            for idx, row in below_threshold.iterrows():
                if current_run_start is None:
                    current_run_start = row['Years']
                elif row['Years'] - current_run_start >= validation_period:
                    potential_switches.append(current_run_start)
                    break
                
            if potential_switches:
                switch_time = min(potential_switches[0], max_switch_time)
                return switch_time
        
        # If no valid switch point found, return max_switch_time
        return max_switch_time