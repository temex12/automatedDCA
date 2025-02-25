"""
Decline Curve Analysis (DCA) Module for Oil & Gas Well Production Analysis

This module provides the core DCA functionality through the DCAAnalyzer class, which handles:
1. Automated decline curve analysis for both conventional and unconventional wells
2. Production data preprocessing and aggregation
3. Multiple decline curve model fitting (Hyperbolic, Harmonic, Exponential)
4. Automated change point detection for optimal analysis periods
5. Economic limit calculations and reserves estimation
6. Batch processing capabilities for multiple wells

Key Features:
- Supports both database and CSV-based data sources
- Implements adaptive DCA cutoff detection
- Handles different well types with appropriate parameter adjustments
- Provides comprehensive error metrics and visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from typing import Dict, List, Tuple, Union
from sqlalchemy import create_engine
from .models import DeclineModels
from .utils import MetricsCalculator, DataPreprocessor
from .visualization import DCAVisualizer
from .data_loader import DCADataLoader

def calculate_remaining_reserves(forecast_rates, days_in_period, time_step, economic_limit_daily):
    """
    Calculate remaining reserves by integrating forecast rates over the actual time step.

    This method converts each forecast rate (in e3m3/day) into volume over the
    corresponding time interval (days_in_period * time_step). It then sums only
    the volumes for points where the rate exceeds the economic limit.

    Args:
        forecast_rates (np.ndarray): Array of forecast production rates (e3m3/day).
        days_in_period (float): Number of days in each base period (e.g., 91.25 for quarterly).
        time_step (float): Time step between forecast points in years (0.25 for quarterly).
        economic_limit_daily (float): Economic production limit (e3m3/day).

    Returns:
        float: Total remaining reserves above the economic limit (in e3m3).
    """
    # Convert daily rates into volumes for each forecast interval
    period_volumes = forecast_rates * days_in_period * time_step

    # Only sum volumes for the portion of the forecast above the economic limit
    above_limit_mask = (forecast_rates > economic_limit_daily)
    return period_volumes[above_limit_mask].sum()

class DCAAnalyzer:
    def __init__(self, source, time_period='Q'):
        """
        Initialize DCA Analyzer with either a database engine or data loader.
        
        Args:
            source: Either SQLAlchemy engine for database connections or DCADataLoader instance for file-based data
            time_period: Time period for analysis aggregation ('Q' for quarterly, 'M' for monthly)
        """
        self.source = source
        self.time_period = time_period
        self.visualizer = DCAVisualizer()
        self.models = DeclineModels()
        self.metrics = MetricsCalculator()
        self.preprocessor = DataPreprocessor()
        
        # Determine data source type for appropriate data loading method
        self.using_loader = isinstance(source, DCADataLoader)
    
    def _get_well_type(self, well_authorization_number):
        """
        Determine if a well is conventional or unconventional based on its horizontal flag.
        
        Args:
            well_authorization_number: Unique identifier for the well
            
        Returns:
            bool: True if unconventional (horizontal), False if conventional
        """
        if self.using_loader:
            _, well_info = self.source.load_well_data(well_authorization_number)
            well_type = well_info.iloc[0]['HZ_FLAG']
        else:
            well_type_query = f"""
            SELECT HZ_FLAG
            FROM well_index
            WHERE WA = '{well_authorization_number}'
            """
            well_type = pd.read_sql_query(well_type_query, self.source).iloc[0]['HZ_FLAG']
            
        return well_type == 'Y'

    def improved_adaptive_dca_cutoff(self, results, min_coverage=0.3, max_error=0.1, 
                                   error_tolerance_factor=0.1, trend_window=5):
        """
        Determine the optimal DCA cutoff point using an improved adaptive approach that balances
        data coverage and model error.
        
        The method finds a cutoff point that:
        1. Maximizes data coverage while maintaining acceptable error levels
        2. Considers the trade-off between coverage and error using a tolerance line
        3. Adapts to different well behaviors and production patterns
        
        Additionally, if the global minimum error is found with data coverage below 20% of the total
        historical data, the function will choose a global minimum among those points that cover at least 20%.
        
        Args:
            results: List of dictionaries containing analysis results for different time periods
            min_coverage: Minimum required fraction of data to be included
            max_error: Maximum acceptable model error
            error_tolerance_factor: Controls the slope of the error tolerance line
            trend_window: Window size for trend analysis in time periods
            
        Returns:
            tuple: (best_coverage, best_error, best_start_date, relative_slope)
                - best_coverage: Fraction of data included in the best fit
                - best_error: Model error at the best fit point
                - best_start_date: Optimal date to begin DCA analysis
                - relative_slope: Slope of the error tolerance line
        """
        coverages = [r['data_coverage'] for r in results]
        errors = [r['error'] for r in results]
        dates = [r['start_date'] for r in results]

        # Sort results chronologically
        sorted_results = sorted(zip(dates, coverages, errors))
        dates, coverages, errors = zip(*sorted_results)

        # Determine global minimum based on error
        global_min_index = np.argmin(errors)
        global_min_error = errors[global_min_index]
        global_min_coverage = coverages[global_min_index]

        # If the global minimum is found with less than 20% coverage,
        # then select the best candidate among those with at least 20% coverage.
        if global_min_coverage < 0.20:
            indices = [i for i, cov in enumerate(coverages) if cov >= 0.20]
            if indices:
                filtered_errors = [errors[i] for i in indices]
                new_index = indices[np.argmin(filtered_errors)]
                global_min_index = new_index
                global_min_error = errors[global_min_index]
                global_min_coverage = coverages[global_min_index]

        # Calculate the error tolerance line slope
        error_range = max(errors) - min(errors)
        coverage_range = max(coverages) - min(coverages)
        relative_slope = (error_range / coverage_range) * error_tolerance_factor
        
        # Define the error tolerance line function
        tolerance_line = lambda x: global_min_error + relative_slope * (x - global_min_coverage)

        # Find all points below the tolerance line with sufficient coverage
        points_below_line = [(i, coverage, error) for i, (coverage, error) 
                            in enumerate(zip(coverages, errors))
                            if error <= tolerance_line(coverage) and coverage >= global_min_coverage]

        if points_below_line:
            # Select the point with maximum coverage among acceptable points
            best_index, best_coverage, best_error = max(points_below_line, key=lambda x: x[1])
        else:
            # Default to global minimum if no points meet criteria
            best_index, best_coverage, best_error = (global_min_index, global_min_coverage, 
                                                   global_min_error)

        return best_coverage, best_error, dates[best_index], relative_slope

    def year_based_automated_change_detection(self, production_data, min_years=1, max_years=40,
                                            min_coverage=0.3, max_error=0.1,
                                            error_tolerance_factor=0.1, trend_window=5):
        """
        Detect significant changes in production data using a year-based sliding window approach.
        This method helps identify the optimal starting point for DCA analysis by:
        1. Testing different time windows
        2. Fitting decline curves to each window
        3. Evaluating the quality of fit
        4. Finding the optimal balance between data coverage and model accuracy
        
        Args:
            production_data: DataFrame containing production time series
            min_years: Minimum years of data to consider
            max_years: Maximum years of data to consider
            min_coverage: Minimum required fraction of data to be included
            max_error: Maximum acceptable model error
            error_tolerance_factor: Controls the slope of the error tolerance line
            trend_window: Window size for trend analysis
            
        Returns:
            tuple: (best_start_date, results, relative_slope, best_coverage, best_error)
        """
        production_data = production_data.sort_values('ZONE_PROD_PERIOD')
        
        # Drop rows with missing dates to ensure valid calculations
        production_data = production_data.dropna(subset=['ZONE_PROD_PERIOD'])
        if production_data.empty:
            raise ValueError("Production data is empty or has no valid 'ZONE_PROD_PERIOD' values.")
        
        start_date = production_data['ZONE_PROD_PERIOD'].min()
        end_date = production_data['ZONE_PROD_PERIOD'].max()
        
        if pd.isnull(start_date) or pd.isnull(end_date):
            raise ValueError("Missing start or end date in production data.")
        
        total_years = (end_date - start_date).days / 365.25
        if np.isnan(total_years) or total_years <= 0:
            raise ValueError("Invalid total years calculated from production data.")
        
        # Ensure that max_years is set correctly
        max_years = min(int(total_years), max_years)
        
        results = []
        
        # Test different time windows
        for years_back in range(min_years, max_years + 1):
            current_start_date = end_date - pd.DateOffset(years=years_back)
            subset_data = production_data[production_data['ZONE_PROD_PERIOD'] >= current_start_date]
            
            if len(subset_data) < 4:  # Ensure enough data points for fitting
                continue
            
            # Preprocess the subset
            subset_data = self.preprocessor.preprocess_data(subset_data)
            t = subset_data['Years']
            q = subset_data['GAS_PRODUCTION_RATE_NORMALIZED']
            
            # Fit all decline curve models
            try:
                models_local = {
                    'Hyperbolic': (self.models.hyperbolic, 
                                curve_fit(self.models.hyperbolic, t, q, 
                                            bounds=((0, 0, 0), (np.inf, np.inf, 1)), 
                                            maxfev=5000, method='trf')[0])
                }
                
                # Calculate errors for each model
                errors = []
                for name, (func, params) in models_local.items():
                    y_pred = func(t, *params)
                    error = mean_squared_error(q, y_pred)
                    errors.append(error)
                
                # Store results using the best model's error
                min_error = min(errors)
                data_coverage = len(subset_data) / len(production_data)
                
                results.append({
                    'start_date': current_start_date,
                    'error': min_error,
                    'data_coverage': data_coverage
                })
                
            except RuntimeError:
                continue  # Skip if curve fitting fails
        
        # Find optimal cutoff point
        best_coverage, best_error, best_start_date, relative_slope = self.improved_adaptive_dca_cutoff(
            results, min_coverage, max_error, error_tolerance_factor, trend_window
        )
        
        if best_start_date is None:
            print("No suitable cutoff found. Using default best error approach.")
            best_result = min(results, key=lambda x: x['error'])
            best_start_date = best_result['start_date']
            best_coverage = best_result['data_coverage']
            best_error = best_result['error']
        
        return best_start_date, results, relative_slope, best_coverage, best_error

    def perform_dca_analysis(self, well_authorization_number, min_coverage=0.3, max_error=0.1, 
                            error_tolerance_factor=0.1, trend_window=5, mse_scale_factor=1e3):
        """
        Performs comprehensive Decline Curve Analysis for a given well.
        
        This method:
        1. Determines well type (conventional/unconventional)
        2. Loads and preprocesses production data
        3. Detects optimal analysis start point
        4. Fits multiple decline curve models
        5. Selects the best model based on error metrics
        6. Generates production forecasts
        7. Calculates reserves and EUR
        8. Creates visualization plots (including the new production rate vs cumulative production plot)
        
        Args:
            well_authorization_number: Unique identifier for the well
            min_coverage: Minimum required fraction of data to be included
            max_error: Maximum acceptable model error
            error_tolerance_factor: Controls the slope of the error tolerance line
            trend_window: Window size for trend analysis
            mse_scale_factor: Scaling factor for mean squared error visualization
            
        Returns:
            dict: Comprehensive results including:
                - Best model and parameters
                - Switch time and decline parameters
                - Production metrics (EUR, reserves)
                - Error metrics
                - Forecast data
                - Original production data
        """
        # Determine well type and set appropriate parameters
        is_unconventional = self._get_well_type(well_authorization_number)
        
        if is_unconventional:
            decline_threshold = 0.15  # 15% for unconventional wells
            min_switch_time = 5  # Longer minimum time for unconventional wells
            max_switch_time = 15  # Longer maximum time for unconventional wells
            well_character = 'Unconventional'
        else:
            decline_threshold = 0.05  # 5% for conventional wells
            min_switch_time = 2  # Shorter minimum time for conventional wells
            max_switch_time = 10  # Shorter maximum time for conventional wells
            well_character = 'Conventional'

        # Load production data
        if self.using_loader:
            production_data, _ = self.source.load_well_data(well_authorization_number)
        else:
            production_query = f"""
            SELECT ZONE_PROD_PERIOD, GAS_PRODUCTION_E3M3
            FROM bc_total_production
            WHERE WELL_AUTHORIZATION_NUMBER = '{well_authorization_number}'
            """
            production_data = pd.read_sql_query(production_query, self.source)
            
        production_data['ZONE_PROD_PERIOD'] = pd.to_datetime(production_data['ZONE_PROD_PERIOD'])

        # Aggregate to specified time period
        production_data['TIME_PERIOD'] = production_data['ZONE_PROD_PERIOD'].dt.to_period(self.time_period)
        production_data = production_data.groupby('TIME_PERIOD').agg(
            {'GAS_PRODUCTION_E3M3': 'sum'}).reset_index()
        production_data['ZONE_PROD_PERIOD'] = production_data['TIME_PERIOD'].dt.to_timestamp()
        production_data = production_data.sort_values('ZONE_PROD_PERIOD')

        # Calculate production rates
        days_in_period = 91.25 if self.time_period == 'Q' else 30.44
        production_data['GAS_PRODUCTION_RATE'] = production_data['GAS_PRODUCTION_E3M3'] / days_in_period
        production_data['Years'] = (
            production_data['ZONE_PROD_PERIOD'] - production_data['ZONE_PROD_PERIOD'].min()
        ).dt.days / 365.25

        # Find optimal analysis start date
        best_start_date, change_detection_results, relative_slope, best_coverage, best_error = (
            self.year_based_automated_change_detection(
                production_data,
                min_years=1,
                max_years=40,
                min_coverage=min_coverage,
                max_error=max_error,
                error_tolerance_factor=error_tolerance_factor,
                trend_window=trend_window
            )
        )

        # Visualize error vs coverage analysis
        self.visualizer.plot_error_vs_data_coverage(
            change_detection_results,
            best_coverage,
            best_error,
            relative_slope,
            well_authorization_number,
            scale_factor=mse_scale_factor
        )

        # Prepare data for DCA
        dca_data = production_data[production_data['ZONE_PROD_PERIOD'] >= best_start_date]
        complete_data = self.preprocessor.preprocess_data(dca_data)
        complete_data['Years'] = (
            complete_data['ZONE_PROD_PERIOD'] - complete_data['ZONE_PROD_PERIOD'].min()
        ).dt.days / 365.25

        t = complete_data['Years']
        q = complete_data['GAS_PRODUCTION_RATE_NORMALIZED']

        # Set decline curve parameters based on well type and life stage
        if is_unconventional:
            if t.max() <= 3:  # Early life of unconventional well
                b_bounds = (0.8, 1.2)  # Higher b-factor for early production
            else:  # Late life of unconventional well
                b_bounds = (0, 0.5)  # Lower b-factor for stabilized production
        else:
            b_bounds = (0, 0.5)  # Conservative bounds for conventional wells

        # Fit all decline curve models using the fit_model method
        models = {}
        try:
            # Use the fit_model method instead of direct curve_fit
            hyperbolic_params = self.models.fit_model(
                t, q, 
                model_type='hyperbolic',
                bounds=((0, 0, b_bounds[0]), (float('inf'), float('inf'), b_bounds[1])),
                maxfev=5000
            )
            
            # If b is very close to zero (< 0.01), use exponential model instead
            if hyperbolic_params[2] < 0.01:
                print(f"B factor {hyperbolic_params[2]:.6f} is effectively zero, using exponential model")
                exponential_params = self.models.fit_model(t, q, model_type='exponential', maxfev=5000)
                models['Exponential'] = (self.models.exponential, exponential_params)
            else:
                models['Hyperbolic'] = (self.models.hyperbolic, hyperbolic_params)
            
            # Fit other models
            harmonic_params = self.models.fit_model(t, q, model_type='harmonic', maxfev=5000)
            exponential_params = self.models.fit_model(t, q, model_type='exponential', maxfev=5000)
            
            if 'Hyperbolic' not in models:  # Only add if not already switched to exponential
                models['Exponential'] = (self.models.exponential, exponential_params)
            models['Harmonic'] = (self.models.harmonic, harmonic_params)
        
        except Exception as e:
            print(f"Error during model fitting: {str(e)}")
            raise

        # Evaluate models using multiple error metrics
        results = {}
        for name, (func, params) in models.items():
            y_pred = func(t, *params)
            mae = mean_absolute_error(q, y_pred)
            rmse = np.sqrt(mean_squared_error(q, y_pred))
            mape = mean_absolute_percentage_error(q, y_pred)
            smape = self.metrics.symmetric_mean_absolute_percentage_error(q, y_pred)
            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'SMAPE': smape
            }

        # Select best model based on RMSE
        best_model = min(results, key=lambda x: results[x]['RMSE'])
        best_func, best_params = models[best_model]

        # Enforce a minimum decline rate of 3% per year
        best_params = np.array(best_params)  # Ensure mutability
        if best_params[1] < 0.03:
            print(
                f"Decline rate {best_params[1]:.4f} is below minimum threshold "
                f"for well {well_authorization_number}. Setting to 0.03."
            )
            best_params[1] = 0.03

        # Compute fitted decline curve on the historical period for visualization
        fitted_rates = best_func(complete_data['Years'], *best_params) * complete_data['GAS_PRODUCTION_RATE'].max()
        fitted_cumulative = np.cumsum(fitted_rates * days_in_period)
        fitted_decline_df = pd.DataFrame({
            'CUMULATIVE_PRODUCTION': fitted_cumulative,
            'GAS_PRODUCTION_RATE': fitted_rates
        })

        # Generate forecast timepoints (quarterly)
        max_years = 100
        t_forecast = np.arange(0, max_years, 0.25)
        
        # Create forecast dates for switch time calculation
        forecast_data = pd.DataFrame({
            'Years': t_forecast,
            'Date': pd.date_range(
                start=complete_data['ZONE_PROD_PERIOD'].min(), 
                periods=len(t_forecast), 
                freq='Q'
            )
        })

        # Find optimal switch time for terminal decline
        switch_time = self.preprocessor.find_switch_time(
            forecast_data, 
            best_func, 
            t_forecast, 
            best_params, 
            decline_threshold,
            min_switch_time=min_switch_time,
            max_switch_time=max_switch_time,
            validation_period=1,
            smoothing_window=3
        )
        
        if switch_time is not None:
            print(f"Switch time detected at {switch_time:.2f} years for {well_character} well")
        else:
            switch_time = max_years  # Use end of forecast if no switch point found
            print(f"No switch time detected within forecast period for {well_character} well")

        # Generate forecast with terminal decline switching logic
        if best_model == 'Harmonic' or (best_model == 'Hyperbolic' and best_params[2] > 0.5):
            if best_model == 'Harmonic':
                qi, di = best_params
                b = 1
            else:
                qi, di, b = best_params
            forecast = self.models.combined_decline_c1(t_forecast, qi, di, b, switch_time) * \
                    complete_data['GAS_PRODUCTION_RATE'].max()
        else:
            forecast = best_func(t_forecast, *best_params) * \
                    complete_data['GAS_PRODUCTION_RATE'].max()

        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': pd.date_range(
                start=complete_data['ZONE_PROD_PERIOD'].min(), 
                periods=len(t_forecast), 
                freq='Q'
            ),
            'GAS_PRODUCTION_RATE': forecast
        })

        # Define economic limit based on well type
        if is_unconventional:
            economic_limit_daily = 2.8  # e3m3/day for unconventional wells
        else:
            economic_limit_daily = 1.4  # e3m3/day for conventional wells

        # Generate and save visualization plots
        self.visualizer.save_dca_plots(
            production_data, forecast_df, well_authorization_number, 
            best_start_date, economic_limit_daily
        )
        self.visualizer.save_rate_vs_cumulative_plot(
            production_data, well_authorization_number, 
            decline_curve_df=fitted_decline_df
        )

        # Calculate reserves and EUR (FIXED to account for time step)
        cumulative_production = production_data['GAS_PRODUCTION_E3M3'].sum()
        remaining_reserves = calculate_remaining_reserves(
            forecast_rates=forecast,
            days_in_period=days_in_period,
            time_step=0.25,  # Because t_forecast increments in 0.25-year steps
            economic_limit_daily=economic_limit_daily
        )
        eur = cumulative_production + remaining_reserves

        # Calculate error metrics for the entire dataset
        full_forecast = best_func(complete_data['Years'], *best_params) * \
                        complete_data['GAS_PRODUCTION_RATE'].max()
        full_mae = mean_absolute_error(complete_data['GAS_PRODUCTION_RATE'], full_forecast)
        full_rmse = np.sqrt(mean_squared_error(complete_data['GAS_PRODUCTION_RATE'], full_forecast))
        full_mape = mean_absolute_percentage_error(complete_data['GAS_PRODUCTION_RATE'], full_forecast)
        full_smape = self.metrics.symmetric_mean_absolute_percentage_error(
            complete_data['GAS_PRODUCTION_RATE'], full_forecast
        )

        # Calculate end-of-life date if we want to track it
        time_to_limit = None
        if forecast is not None and economic_limit_daily is not None:
            above_limit_mask = forecast > economic_limit_daily
            if np.any(above_limit_mask):
                time_to_limit = np.sum(above_limit_mask) * 0.25

        if time_to_limit is not None:
            end_of_life_date = pd.to_datetime(best_start_date) + pd.DateOffset(days=int(time_to_limit * 365.25))
        else:
            end_of_life_date = None

        # Compile and return analysis results
        dca_results = {
            'best_model': best_model,
            'best_params': best_params,
            'switch_time': switch_time,
            'decline_threshold': decline_threshold,
            'well_type': well_character,
            'cumulative_production': cumulative_production,
            'remaining_reserves': remaining_reserves,
            'eur': eur,
            'economic_limit_daily': economic_limit_daily,
            'days_in_period': days_in_period,
            'best_start_date': best_start_date,
            'full_mae': full_mae,
            'full_rmse': full_rmse,
            'full_mape': full_mape,
            'full_smape': full_smape,
            'forecast_df': forecast_df,
            'production_data': production_data
        }

        # Extract and organize the parameters
        parameter_summary = {
            'well_authorization_number': well_authorization_number,
            'dca_start_date': best_start_date,
            'model': best_model,
            'q_model': best_params[0] * complete_data['GAS_PRODUCTION_RATE'].max(),
            'd_model': best_params[1],
            'b_model': (
                best_params[2] if best_model == 'Hyperbolic'
                else (1 if best_model == 'Harmonic' else 0)
            ),
            't_model': (
                switch_time
                if (best_model == 'Harmonic' or (best_model == 'Hyperbolic' and best_params[2] > 0.5))
                else time_to_limit
            ),
            'has_transition': (
                'yes' if (best_model == 'Harmonic' or
                        (best_model == 'Hyperbolic' and best_params[2] > 0.5))
                else 'no'
            ),
            'end_of_life_date': end_of_life_date
        }

        # If there's a transition, compute transition parameters
        if parameter_summary['has_transition'] == 'yes':
            if best_model == 'Harmonic':
                qi, di = best_params
                b = 1
            else:
                qi, di, b = best_params
            
            t_array = np.array([0, switch_time])
            hyp_decline = self.models.hyperbolic(t_array, qi, di, b)
            exp_start_rate = np.interp(switch_time, t_array, hyp_decline)
            hyp_slope_at_switch = self.models.hyperbolic_derivative(switch_time, qi, di, b)
            di_exp = -hyp_slope_at_switch / exp_start_rate
            
            exp_start_rate_denorm = exp_start_rate * complete_data['GAS_PRODUCTION_RATE'].max()
            t_after_switch = t_forecast[t_forecast >= switch_time]
            exp_decline = self.models.exponential(t_after_switch - switch_time, exp_start_rate, di_exp)
            exp_decline_denorm = exp_decline * complete_data['GAS_PRODUCTION_RATE'].max()
            
            above_limit_mask = exp_decline_denorm > economic_limit_daily
            if np.any(above_limit_mask):
                transition_time = np.sum(above_limit_mask) * 0.25
            else:
                transition_time = len(t_after_switch) * 0.25
            
            parameter_summary.update({
                'transition_start_date': (
                    pd.to_datetime(best_start_date)
                    + pd.Timedelta(days=int(switch_time * 365.25))
                ),
                'q_transition': exp_start_rate_denorm,
                'd_transition': di_exp,
                't_transition': transition_time
            })
        else:
            parameter_summary.update({
                'transition_start_date': None,
                'q_transition': None,
                'd_transition': None,
                't_transition': None
            })

        dca_results['parameter_summary'] = parameter_summary
        
        return dca_results

    def analyze_well_list(self, well_list, **kwargs):
        """
        Analyze multiple wells and generate comprehensive summary statistics.
        
        This method:
        1. Processes each well in the provided list
        2. Generates individual well analysis results
        3. Creates summary statistics
        4. Saves results to files
        
        Args:
            well_list: List of well authorization numbers to analyze
            **kwargs: Additional parameters passed to perform_dca_analysis
                - min_coverage: Minimum required data coverage
                - max_error: Maximum acceptable error
                - error_tolerance_factor: Controls error tolerance line slope
                - trend_window: Window size for trend analysis
                - mse_scale_factor: Scaling factor for MSE visualization
                
        Returns:
            pd.DataFrame: Summary statistics for all analyzed wells
            
        Outputs:
            - Normal scale plots: results/normal_scale/
            - Log scale plots: results/log_scale/
            - Error analysis plots: results/error_analysis/
            - Summary CSV: results/dca_analysis_summary.csv
            - Summary Excel: results/dca_analysis_summary.xlsx with multiple sheets:
                - Full Summary
                - Error Statistics
                - Error Ranges
        """
        # Create output directories
        for dir_name in ['normal_scale', 'log_scale', 'error_analysis']:
            Path(f'results/{dir_name}').mkdir(parents=True, exist_ok=True)

        # Initialize results storage
        summary_data = []
        
        # Process each well with error handling
        for well_number in well_list:
            print(f"\n{'='*50}")
            print(f"Processing Well {well_number}")
            print(f"{'='*50}")
            
            try:
                # Perform DCA analysis
                dca_results = self.perform_dca_analysis(
                    well_number,
                    min_coverage=kwargs.get('min_coverage', 0.3),
                    max_error=kwargs.get('max_error', 0.1),
                    error_tolerance_factor=kwargs.get('error_tolerance_factor', 0.5),
                    trend_window=kwargs.get('trend_window', 5),
                    mse_scale_factor=kwargs.get('mse_scale_factor', 1e3)
                )
                
                param_sum = dca_results['parameter_summary']
                
                # Store results in summary including new detailed parameters
                summary_data.append({
                    'Well': well_number,
                    'Best Model': dca_results['best_model'],
                    'Well Type': dca_results['well_type'],
                    'Cumulative Production (E3M3)': dca_results['cumulative_production'],
                    'Remaining Reserves (E3M3)': dca_results['remaining_reserves'],
                    'EUR (E3M3)': dca_results['eur'],
                    'Switch Time (years)': dca_results['switch_time'],
                    'MAE': dca_results['full_mae'],
                    'RMSE': dca_results['full_rmse'],
                    'MAPE': dca_results['full_mape'],
                    'SMAPE': dca_results['full_smape'],
                    'Analysis Start Date': dca_results['best_start_date'],
                    'End of Life Date': param_sum.get('end_of_life_date'),
                    'q_model': param_sum.get('q_model'),
                    'd_model': param_sum.get('d_model'),
                    'b_model': param_sum.get('b_model'),
                    't_model': param_sum.get('t_model'),
                    'Has Transition': param_sum.get('has_transition')
                })
                
                print(f"Analysis completed successfully for Well {well_number}")
                
            except Exception as e:
                # Handle and log errors
                print(f"Error processing Well {well_number}: {str(e)}")
                print("Traceback:")
                import traceback
                traceback.print_exc()
                
                summary_data.append({
                    'Well': well_number,
                    'Best Model': 'Failed',
                    'Well Type': 'Unknown',
                    'Cumulative Production (E3M3)': None,
                    'Remaining Reserves (E3M3)': None,
                    'EUR (E3M3)': None,
                    'Switch Time (years)': None,
                    'MAE': None,
                    'RMSE': None,
                    'MAPE': None,
                    'SMAPE': None,
                    'Analysis Start Date': None,
                    'End of Life Date': None,
                    'q_model': None,
                    'd_model': None,
                    'b_model': None,
                    't_model': None,
                    'Has Transition': None,
                    'Error': str(e)
                })
            
            print(f"{'='*50}\n")

        # Create and save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('results/dca_analysis_summary.csv', index=False)
        
        # Create an Excel file with multiple sheets
        with pd.ExcelWriter('results/dca_analysis_summary.xlsx') as writer:
            summary_df.to_excel(writer, sheet_name='Full Summary', index=False)
            summary_df.describe().to_excel(writer, sheet_name='Error Statistics')
            error_metrics = ['MAE', 'RMSE', 'MAPE', 'SMAPE']
            error_ranges = summary_df[error_metrics].agg(['min', 'max', 'median'])
            error_ranges.loc['range'] = error_ranges.loc['max'] - error_ranges.loc['min']
            error_ranges.to_excel(writer, sheet_name='Error Ranges')
        
        print("\nAnalysis complete. Results saved in results folder:")
        print("- Normal scale plots: results/normal_scale/")
        print("- Log scale plots: results/log_scale/")
        print("- Error analysis plots: results/error_analysis/")
        print("- Summary CSV: results/dca_analysis_summary.csv")
        print("- Summary Excel: results/dca_analysis_summary.xlsx")
        
        return summary_df
