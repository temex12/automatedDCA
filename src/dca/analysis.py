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

class DCAAnalyzer:
    def __init__(self, source, time_period='Q'):
        """
        Initialize DCA Analyzer with either a database engine or data loader.
        
        Args:
            source: Either SQLAlchemy engine or DCADataLoader instance
            time_period: Time period for analysis ('Q' for quarterly)
        """
        self.source = source
        self.time_period = time_period
        self.visualizer = DCAVisualizer()
        self.models = DeclineModels()
        self.metrics = MetricsCalculator()
        self.preprocessor = DataPreprocessor()
        
        # Determine if we're using a data loader or direct database connection
        self.using_loader = isinstance(source, DCADataLoader)
    
    def _get_well_type(self, well_authorization_number):
        """Get well type (conventional/unconventional) from data source."""
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
        Determine the optimal DCA cutoff point using an improved adaptive approach.
        
        Args:
            results (list): List of dictionaries containing analysis results
            min_coverage (float): Minimum required data coverage
            max_error (float): Maximum acceptable error
            error_tolerance_factor (float): Factor for error tolerance calculation
            trend_window (int): Window size for trend analysis
            
        Returns:
            tuple: (best_coverage, best_error, best_start_date, relative_slope)
        """
        coverages = [r['data_coverage'] for r in results]
        errors = [r['error'] for r in results]
        dates = [r['start_date'] for r in results]

        # Sort results by date (ascending)
        sorted_results = sorted(zip(dates, coverages, errors))
        dates, coverages, errors = zip(*sorted_results)

        # Find the global minimum error and its index
        global_min_index = np.argmin(errors)
        global_min_error = errors[global_min_index]
        global_min_coverage = coverages[global_min_index]

        # Calculate the relative error tolerance slope
        error_range = max(errors) - min(errors)
        coverage_range = max(coverages) - min(coverages)
        relative_slope = (error_range / coverage_range) * error_tolerance_factor
        
        # Calculate the error tolerance line
        tolerance_line = lambda x: global_min_error + relative_slope * (x - global_min_coverage)

        # Find all points below the error tolerance line
        points_below_line = [(i, coverage, error) for i, (coverage, error) 
                            in enumerate(zip(coverages, errors))
                            if error <= tolerance_line(coverage) and coverage >= global_min_coverage]

        if points_below_line:
            # Choose the point with the highest coverage among points below the line
            best_index, best_coverage, best_error = max(points_below_line, key=lambda x: x[1])
        else:
            # If no points are below the line, default to the global minimum
            best_index, best_coverage, best_error = (global_min_index, global_min_coverage, 
                                                   global_min_error)

        return best_coverage, best_error, dates[best_index], relative_slope

    def year_based_automated_change_detection(self, production_data, min_years=1, max_years=40,
                                            min_coverage=0.3, max_error=0.1,
                                            error_tolerance_factor=0.1, trend_window=5):
        """
        Detect changes in production data using a year-based approach.
        
        Args:
            production_data (pd.DataFrame): Production data
            min_years (int): Minimum years to consider
            max_years (int): Maximum years to consider
            min_coverage (float): Minimum required data coverage
            max_error (float): Maximum acceptable error
            error_tolerance_factor (float): Factor for error tolerance calculation
            trend_window (int): Window size for trend analysis
            
        Returns:
            tuple: (best_start_date, results, relative_slope, best_coverage, best_error)
        """
        production_data = production_data.sort_values('ZONE_PROD_PERIOD')
        end_date = production_data['ZONE_PROD_PERIOD'].max()
        start_date = production_data['ZONE_PROD_PERIOD'].min()
        total_years = (end_date - start_date).days / 365.25
        
        max_years = min(int(total_years), max_years)
        results = []
        
        for years_back in range(min_years, max_years + 1):
            current_start_date = end_date - pd.DateOffset(years=years_back)
            subset_data = production_data[production_data['ZONE_PROD_PERIOD'] >= current_start_date]
            
            if len(subset_data) < 4:  # Ensure enough data points for fitting
                continue
            
            # Preprocess the subset
            subset_data = self.preprocessor.preprocess_data(subset_data)
            t = subset_data['Years']
            q = subset_data['GAS_PRODUCTION_RATE_NORMALIZED']
            
            # Fit models
            try:
                models = {
                    'Hyperbolic': (self.models.hyperbolic, 
                                 curve_fit(self.models.hyperbolic, t, q, 
                                         bounds=((0, 0, 0), (np.inf, np.inf, 1)), 
                                         maxfev=5000, method='trf')[0]),
                    'Harmonic': (self.models.harmonic, 
                               curve_fit(self.models.harmonic, t, q, 
                                       maxfev=5000, method='trf')[0]),
                    'Exponential': (self.models.exponential, 
                                  curve_fit(self.models.exponential, t, q, 
                                          maxfev=5000, method='trf')[0])
                }
                
                # Evaluate models
                errors = []
                for name, (func, params) in models.items():
                    y_pred = func(t, *params)
                    error = mean_squared_error(q, y_pred)
                    errors.append(error)
                
                # Use the minimum error from all models
                min_error = min(errors)
                data_coverage = len(subset_data) / len(production_data)
                
                results.append({
                    'start_date': current_start_date,
                    'error': min_error,
                    'data_coverage': data_coverage
                })
                
            except RuntimeError:
                continue  # Skip if curve fitting fails
        
        # Use improved_adaptive_dca_cutoff to find the best cutoff
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
        Performs Decline Curve Analysis for a given well.
        """
        # Get well type
        is_unconventional = self._get_well_type(well_authorization_number)
        
        # Set decline threshold and switch time constraints based on well type
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

        # Aggregate production data
        production_data['TIME_PERIOD'] = production_data['ZONE_PROD_PERIOD'].dt.to_period(self.time_period)
        production_data = production_data.groupby('TIME_PERIOD').agg(
            {'GAS_PRODUCTION_E3M3': 'sum'}).reset_index()
        production_data['ZONE_PROD_PERIOD'] = production_data['TIME_PERIOD'].dt.to_timestamp()
        production_data = production_data.sort_values('ZONE_PROD_PERIOD')

        # Calculate production rate
        days_in_period = 91.25 if self.time_period == 'Q' else 30.44
        production_data['GAS_PRODUCTION_RATE'] = production_data['GAS_PRODUCTION_E3M3'] / days_in_period
        production_data['Years'] = (production_data['ZONE_PROD_PERIOD'] - 
                                production_data['ZONE_PROD_PERIOD'].min()).dt.days / 365.25
        
        # Find best start date
        best_start_date, change_detection_results, relative_slope, best_coverage, best_error = \
            self.year_based_automated_change_detection(
                production_data, min_years=1, max_years=40,
                min_coverage=min_coverage, max_error=max_error,
                error_tolerance_factor=error_tolerance_factor,
                trend_window=trend_window
            )
        
        # Plot error vs data coverage
        self.visualizer.plot_error_vs_data_coverage(
            change_detection_results, best_coverage, best_error,
            relative_slope, well_authorization_number, scale_factor=mse_scale_factor
        )

        # Filter data based on the detected start date
        dca_data = production_data[production_data['ZONE_PROD_PERIOD'] >= best_start_date]
        complete_data = self.preprocessor.preprocess_data(dca_data)
        complete_data['Years'] = (complete_data['ZONE_PROD_PERIOD'] - 
                                complete_data['ZONE_PROD_PERIOD'].min()).dt.days / 365.25

        t = complete_data['Years']
        q = complete_data['GAS_PRODUCTION_RATE_NORMALIZED']

        # Adjust decline curve expectations based on well type
        if is_unconventional:
            if t.max() <= 3:  # Early life
                b_bounds = (0.8, 1.2)
            else:  # Late life
                b_bounds = (0, 0.5)
        else:
            b_bounds = (0, 0.5)

        # Fit models
        models = {
            'Hyperbolic': (self.models.hyperbolic, curve_fit(self.models.hyperbolic, t, q, 
                        bounds=((0, 0, b_bounds[0]), (np.inf, np.inf, b_bounds[1])), 
                        maxfev=5000)[0]),
            'Harmonic': (self.models.harmonic, curve_fit(self.models.harmonic, t, q, maxfev=5000)[0]),
            'Exponential': (self.models.exponential, curve_fit(self.models.exponential, t, q, maxfev=5000)[0])
        }

        # Evaluate models
        results = {}
        for name, (func, params) in models.items():
            y_pred = func(t, *params)
            mae = mean_absolute_error(q, y_pred)
            mse = mean_squared_error(q, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(q, y_pred)
            smape = self.metrics.symmetric_mean_absolute_percentage_error(q, y_pred)
            results[name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape}

        # Select best model based on RMSE
        best_model = min(results, key=lambda x: results[x]['RMSE'])
        best_func, best_params = models[best_model]

        # Generate forecast timepoints
        max_years = 100
        t_forecast = np.arange(0, max_years, 0.25)
        
        # Create forecast data for switch time calculation
        forecast_data = pd.DataFrame({
            'Years': t_forecast,
            'Date': pd.date_range(start=complete_data['ZONE_PROD_PERIOD'].min(), 
                                periods=len(t_forecast), freq='Q')
        })

        # Find switch time with new parameters
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
            switch_time = max_years  # Default to end of forecast if no switch point found
            print(f"No switch time detected within forecast period for {well_character} well")

        # Generate forecast with switching logic
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
            'Date': pd.date_range(start=complete_data['ZONE_PROD_PERIOD'].min(), 
                                periods=len(t_forecast), freq='Q'),
            'GAS_PRODUCTION_RATE': forecast
        })

        # Economic limit parameters
        economic_limit_daily = 2.4  # e3m3 per day

        # Save DCA plots
        self.visualizer.save_dca_plots(production_data, forecast_df, well_authorization_number, 
                                    best_start_date, economic_limit_daily)

        # Calculate reserves and EUR
        cumulative_production = production_data['GAS_PRODUCTION_E3M3'].sum()
        remaining_reserves = forecast[forecast > economic_limit_daily].sum() * days_in_period
        eur = cumulative_production + remaining_reserves

        # Calculate error metrics for the entire dataset
        full_forecast = best_func(complete_data['Years'], *best_params) * \
                    complete_data['GAS_PRODUCTION_RATE'].max()
        full_mae = mean_absolute_error(complete_data['GAS_PRODUCTION_RATE'], full_forecast)
        full_rmse = np.sqrt(mean_squared_error(complete_data['GAS_PRODUCTION_RATE'], full_forecast))
        full_mape = mean_absolute_percentage_error(complete_data['GAS_PRODUCTION_RATE'], full_forecast)
        full_smape = self.metrics.symmetric_mean_absolute_percentage_error(
            complete_data['GAS_PRODUCTION_RATE'], full_forecast)

        # Prepare and return results
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

        return dca_results
    
    def analyze_well_list(self, well_list, **kwargs):
        """
        Analyze a list of wells and generate summary statistics.
        """
        # Create results directory if it doesn't exist
        for dir_name in ['normal_scale', 'log_scale', 'error_analysis']:
            Path(f'results/{dir_name}').mkdir(parents=True, exist_ok=True)

        # Initialize summary data list
        summary_data = []
        
        # Process each well
        for well_number in well_list:
            print(f"\n{'='*50}")
            print(f"Processing Well {well_number}")
            print(f"{'='*50}")
            
            try:
                # Perform DCA analysis with default or user-specified parameters
                dca_results = self.perform_dca_analysis(
                    well_number,
                    min_coverage=kwargs.get('min_coverage', 0.3),
                    max_error=kwargs.get('max_error', 0.1),
                    error_tolerance_factor=kwargs.get('error_tolerance_factor', 0.5),
                    trend_window=kwargs.get('trend_window', 5),
                    mse_scale_factor=kwargs.get('mse_scale_factor', 1e3)
                )
                
                # Add results to summary
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
                    'Analysis Start Date': dca_results['best_start_date']
                })
                
                print(f"Analysis completed successfully for Well {well_number}")
                
            except Exception as e:
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
                    'Error': str(e)
                })
            
            print(f"{'='*50}\n")

        # Create summary dataframe
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary to CSV
        summary_df.to_csv('results/dca_analysis_summary.csv', index=False)
        
        print("\nAnalysis complete. Results saved in results folder:")
        print("- Normal scale plots: results/normal_scale/")
        print("- Log scale plots: results/log_scale/")
        print("- Error analysis plots: results/error_analysis/")
        print("- Summary: results/dca_analysis_summary.csv")
        
        return summary_df