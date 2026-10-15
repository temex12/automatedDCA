"""
Decline Curve Analysis (DCA) Visualization Module

This module provides comprehensive visualization capabilities for DCA results, including:
1. Production rate plots (normal and log scale)
2. Error analysis visualization
3. Data coverage analysis plots

Key Features:
- Automated plot generation and saving
- Consistent styling and formatting
- Support for both normal and log-scale visualization
- Error analysis with tolerance lines
- Customizable output formats and directories

Output Structure:
results/
├── normal_scale/      # Linear scale production plots
├── log_scale/         # Logarithmic scale production plots
└── error_analysis/    # Error vs coverage analysis plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

class DCAVisualizer:
    """
    Visualization manager for Decline Curve Analysis results.
    
    Features:
    - Production rate visualization
    - Error analysis plots
    - Consistent styling across plots
    - Automated file management
    """
    
    def __init__(self, output_dir="results"):
        """
        Initialize visualization manager with output directory structure.
        
        Args:
            output_dir (str): Base directory for saving plots
                              Will create if doesn't exist
                            
        Directory Structure Created:
        - {output_dir}/normal_scale/
        - {output_dir}/log_scale/
        - {output_dir}/error_analysis/
        - {output_dir}/rate_vs_cumulative/
        """
        self.output_dir = output_dir
        self._create_output_directories()
        
        # Set default plotting parameters
        plt.style.use('seaborn-v0_8')  # Clean, modern style
        self.default_figsize = (16, 8)  # Widescreen format
        self.dpi = 300  # High resolution output
    
    def _create_output_directories(self):
        """
        Create directory structure for output files.
        Creates directories if they don't exist.
        """
        for subdir in ["normal_scale", "log_scale", "error_analysis", "rate_vs_cumulative"]:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
    
    def _save_figure(self, filepath):
        """
        Save figure with consistent settings and error handling.
        
        Args:
            filepath (str): Full path for saving figure
            
        Notes:
            - Includes tight bounding box
            - Closes figure after saving
        """
        try:
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error saving figure to {filepath}: {str(e)}")
            plt.close()
    
    def plot_error_vs_data_coverage(self, results, best_coverage, best_error, 
                                    relative_slope, well_number, scale_factor=1e3):
        """
        Create error analysis plot showing error vs data coverage relationship.
        
        This plot helps visualize the trade-off between:
        - Model error (accuracy of fit)
        - Data coverage (amount of data used)
        
        Args:
            results (list): Analysis results containing:
                - start_date: Analysis start date
                - error: Model error value
                - data_coverage: Fraction of data used
            best_coverage (float): Coverage at chosen analysis point
            best_error (float): Error at chosen analysis point
            relative_slope (float): Slope of error tolerance line
            well_number (int or str): Well identification number
            scale_factor (float): Factor for scaling error values
            
        Plot Elements:
        - Error vs coverage curve
        - Error tolerance line
        - Global minimum point
        - Chosen operating point
        - Year labels for key points
        """
        try:
            plt.figure(figsize=self.default_figsize)
            
            # Extract plot data
            dates = [r['start_date'] for r in results]
            errors = [r['error'] * scale_factor for r in results]
            coverages = [r['data_coverage'] for r in results]
            
            # Plot main curve
            plt.plot(coverages, errors, 'b-o', label='Error vs Coverage', markersize=5)
            
            # Calculate and plot global minimum
            global_min_index = np.argmin(errors)
            global_min_coverage = coverages[global_min_index]
            global_min_error = errors[global_min_index]
            max_coverage = max(coverages)

            # Determine the tolerance line anchor using the same rule as the
            # selection algorithm (improved_adaptive_dca_cutoff): when the global
            # minimum uses less than 20% of the data, the line is anchored at the
            # lowest-error point with at least 20% coverage.
            anchor_index = global_min_index
            if global_min_coverage < 0.20:
                eligible = [i for i, cov in enumerate(coverages) if cov >= 0.20]
                if eligible:
                    anchor_index = min(eligible, key=lambda i: errors[i])
            anchor_coverage = coverages[anchor_index]
            anchor_error = errors[anchor_index]

            # Add error tolerance line, drawn from the anchor the selection used
            tolerance_line_x = [anchor_coverage, max_coverage]
            tolerance_line_y = [
                anchor_error,
                anchor_error + (relative_slope * scale_factor) * (max_coverage - anchor_coverage)
            ]
            plt.plot(tolerance_line_x, tolerance_line_y, 'g--', label='Error Tolerance Line')

            # Highlight key points
            plt.plot(best_coverage, best_error * scale_factor, marker='s', color='green',
                     markersize=10, label='Chosen Point')
            plt.plot(global_min_coverage, global_min_error, marker='*', color='black',
                     markersize=12, label='Global Minimum')
            if anchor_index != global_min_index:
                plt.plot(anchor_coverage, anchor_error, marker='*', color='green',
                         markersize=12, label='Tolerance Anchor (>=20% coverage)')
                plt.annotate('TA', (anchor_coverage, anchor_error),
                             xytext=(5, -20), textcoords='offset points',
                             bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
                             arrowprops=dict(arrowstyle="->"))
            
            # Add annotations
            self._add_annotations(best_coverage, best_error, scale_factor, 
                                  global_min_coverage, global_min_error)
            
            # Style and format plot
            plt.xlabel('Data Coverage')
            plt.ylabel(f'MSE Error (x {scale_factor:,.0f})')
            plt.title(f'Error vs Data Coverage Analysis - Well {well_number}')
            plt.grid(True)
            plt.legend()
            
            # Format axes
            plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            self._add_year_labels(coverages, errors, dates)
            
            # Save plot
            plt.tight_layout()
            filepath = os.path.join(self.output_dir, "error_analysis", 
                                    f"well_{well_number}_error_analysis.png")
            self._save_figure(filepath)
            
        except Exception as e:
            print(f"Error creating error analysis plot for well {well_number}: {str(e)}")
            plt.close()
    
    def _add_annotations(self, best_coverage, best_error, scale_factor, 
                         global_min_coverage, global_min_error):
        """
        Add text annotations to error analysis plot.
        
        Annotations:
        - CP: Chosen Point
        - GM: Global Minimum
        
        Features:
        - Offset text boxes
        - Arrow connectors
        - Semi-transparent backgrounds
        """
        plt.annotate('CP', (best_coverage, best_error * scale_factor),
                     xytext=(5, 5), textcoords='offset points',
                     bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
                     arrowprops=dict(arrowstyle="->"))
        plt.annotate('GM', (global_min_coverage, global_min_error),
                     xytext=(25, 25), textcoords='offset points',
                     bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
                     arrowprops=dict(arrowstyle="->"))
    
    def _add_year_labels(self, coverages, errors, dates):
        """
        Add year labels to data points on error analysis plot.
        
        Args:
            coverages: Coverage values for each point
            errors: Error values for each point
            dates: Corresponding dates
            
        Notes:
            - Labels added every third point to avoid crowding
            - 30-degree rotation for better readability
            - Offset positioning to avoid overlap
        """
        for i, (coverage, error, date) in enumerate(zip(coverages, errors, dates)):
            if i % 3 == 0:  # Add label every third point
                plt.annotate(date.strftime('%Y'), (coverage, error), 
                             textcoords="offset points", xytext=(0, 10), 
                             ha='center', va='bottom', color='black', 
                             fontsize=8, rotation=30)
                
    def save_rate_vs_cumulative_plot(self, production_data, well_number, 
                                     decline_curve_df=None, best_start_date=None,
                                     economic_limit_daily=None):
        """
        Generate and save a plot of Gas Production Rate vs Cumulative Gas Production.

        The plot is generated using the historical production data. Cumulative production is 
        computed as the cumulative sum of 'GAS_PRODUCTION_E3M3'. The production rate is taken 
        from 'GAS_PRODUCTION_RATE'. If a decline curve is provided, it is overlaid on the plot.
        
        Optionally, if best_start_date is provided, a vertical line is added at the cumulative
        production corresponding to that date, to indicate the start of DCA analysis.
        
        Additionally, if economic_limit_daily is provided, a horizontal dashed black line 
        is added to represent the economic production limit.

        Args:
            production_data (DataFrame): Historical production data.
            well_number (int or str): Identifier for the well.
            decline_curve_df (DataFrame, optional): DataFrame with columns 
                'CUMULATIVE_PRODUCTION' and 'GAS_PRODUCTION_RATE' representing the fitted decline curve.
            best_start_date (datetime, optional): The DCA analysis start date to be annotated.
            economic_limit_daily (float, optional): Economic production rate limit (E3M3/day)
        """
        # Sort chronologically
        prod_data = production_data.sort_values('ZONE_PROD_PERIOD').copy()
        # Compute cumulative production
        prod_data['CUMULATIVE_PRODUCTION'] = prod_data['GAS_PRODUCTION_E3M3'].cumsum()

        plt.figure(figsize=self.default_figsize)

        # Plot HISTORICAL DATA as a LINE (blue)
        plt.plot(
            prod_data['CUMULATIVE_PRODUCTION'], 
            prod_data['GAS_PRODUCTION_RATE'],
            color='blue', 
            linestyle='-', 
            linewidth=2, 
            label='Historical Data'
        )
        
        # Plot the fitted decline curve (red line) if provided
        if decline_curve_df is not None:
            plt.plot(
                decline_curve_df['CUMULATIVE_PRODUCTION'], 
                decline_curve_df['GAS_PRODUCTION_RATE'],
                color='red', 
                linestyle='-', 
                linewidth=2, 
                label='Fitted Decline'
            )
        
        # Add economic limit line if provided
        if economic_limit_daily is not None:
            plt.axhline(y=economic_limit_daily, color='black', linestyle='--', 
                        linewidth=2, label=f'Economic Limit: {economic_limit_daily} E3M3/day')
        
        # Vertical line for DCA start date
        if best_start_date is not None:
            best_start_cum = prod_data.loc[
                prod_data['ZONE_PROD_PERIOD'] < best_start_date, 'GAS_PRODUCTION_E3M3'
            ].sum()
            plt.axvline(x=best_start_cum, color='green', linestyle='--', linewidth=2, label='DCA Start')

        plt.xlabel('Cumulative Gas Production (E3M3)')
        plt.ylabel('Gas Production Rate (E3M3/day)')
        plt.title(f'Gas Production Rate vs Cumulative Gas Production for Well {well_number}')
        plt.grid(True)
        plt.legend()

        rate_cum_dir = os.path.join(self.output_dir, "rate_vs_cumulative")
        os.makedirs(rate_cum_dir, exist_ok=True)
        filepath = os.path.join(rate_cum_dir, f"well_{well_number}_rate_vs_cumulative.png")
        self._save_figure(filepath)

    def save_dca_plots(self, production_data, forecast_df, well_number, 
                       best_start_date, economic_limit_daily):
        """
        Generate and save production rate plots in both normal and log scales.
        
        Creates two visualizations for each well:
        1. Normal (linear) scale plot
        2. Log scale plot
        
        Args:
            production_data: Historical production data DataFrame
            forecast_df: Forecast production DataFrame
            well_number: Well identification number
            best_start_date: Optimal DCA analysis start date
            economic_limit_daily: Economic production rate limit
            
        Plot Elements:
        - Historical production data points
        - DCA model fit
        - Production forecast
        - Economic limit line
        - Analysis start date line
        """
        last_data_date = production_data['ZONE_PROD_PERIOD'].max()
        
        # Create linear scale plot
        self._create_production_plot(
            production_data, forecast_df, well_number, best_start_date,
            economic_limit_daily, last_data_date, log_scale=False
        )
        
        # Create log scale plot
        self._create_production_plot(
            production_data, forecast_df, well_number, best_start_date,
            economic_limit_daily, last_data_date, log_scale=True
        )
    
    def _create_production_plot(self, production_data, forecast_df, well_number,
                                best_start_date, economic_limit_daily, last_data_date,
                                log_scale=False):
        """
        Create production rate visualization with specified scale.
        
        Args:
            production_data: Historical production data
            forecast_df: Forecast production data
            well_number: Well identification number
            best_start_date: DCA analysis start date
            economic_limit_daily: Economic production limit
            last_data_date: End date of historical data
            log_scale (bool): Whether to use logarithmic y-axis
            
        Plot Components:
        - Historical data points (blue scatter)
        - DCA model fit (red line)
        - Production forecast (green dashed line)
        - Economic limit (black dashed line)
        - Analysis start date (green vertical line)
        - Automatic truncation:
          1) By default, 20 years after last_data_date
          2) If forecast crosses economic limit in the future, 
             truncate at crossing + 5 years (or 20 years, whichever is earlier)
        """
        plt.figure(figsize=self.default_figsize)

        # Separate copies for plotting, so we can remove zero/negative values in log scale
        data_for_plot = production_data.copy()
        forecast_for_plot = forecast_df.copy()

        # Remove zero or negative rates for log scale
        if log_scale:
            data_for_plot = data_for_plot[data_for_plot['GAS_PRODUCTION_RATE'] > 0]
            forecast_for_plot = forecast_for_plot[forecast_for_plot['GAS_PRODUCTION_RATE'] > 0]
            plt.yscale('log')
        
        # Plot historical data as a line (ValNav style) instead of scatter points
        # Sort by date to ensure proper line connection
        data_for_plot = data_for_plot.sort_values('ZONE_PROD_PERIOD')
        plt.plot(data_for_plot['ZONE_PROD_PERIOD'], data_for_plot['GAS_PRODUCTION_RATE'],
                 color='blue', linewidth=1.5, label='Historical Data', alpha=0.8)
        
        # Split the forecast into "fit" portion (<= last_data_date) and "future" portion (> last_data_date)
        mask_fit = forecast_for_plot['Date'] <= last_data_date
        mask_forecast = forecast_for_plot['Date'] > last_data_date
        
        plt.plot(forecast_for_plot[mask_fit]['Date'], 
                 forecast_for_plot[mask_fit]['GAS_PRODUCTION_RATE'],
                 color='red', label='DCA Fit', linewidth=2)
        plt.plot(forecast_for_plot[mask_forecast]['Date'], 
                 forecast_for_plot[mask_forecast]['GAS_PRODUCTION_RATE'],
                 color='green', linestyle='--', label='DCA Forecast', linewidth=2)
        
        # Add economic limit line
        plt.axhline(y=economic_limit_daily, color='black', linestyle='--', 
                    linewidth=2, label=f'Economic Limit: {economic_limit_daily} E3M3/day')
        
        # Add analysis start date line
        plt.axvline(x=best_start_date, color='green', linestyle='--', linewidth=2, label='DCA Start Date')

        # -----------------------
        # DYNAMIC X-AXIS TRUNCATION
        # -----------------------
        # 1) x_min: earliest production date
        x_min = production_data['ZONE_PROD_PERIOD'].min()
        
        # 2) By default, show up to 20 years after the last historical date
        x_max_20_years = last_data_date + pd.DateOffset(years=20)
        
        # 3) Check if forecast crosses the economic limit AFTER last_data_date
        mask_below_limit = forecast_for_plot[
            (forecast_for_plot['Date'] > last_data_date) & 
            (forecast_for_plot['GAS_PRODUCTION_RATE'] <= economic_limit_daily)
        ]
        
        if not mask_below_limit.empty:
            # First crossing date after the last historical data
            crossing_date = mask_below_limit['Date'].iloc[0]
            # crossing_date + 5 years
            crossing_plus_5 = crossing_date + pd.DateOffset(years=5)
            # x_max is the earlier of crossing+5 or the 20-year mark
            x_max = min(x_max_20_years, crossing_plus_5)
        else:
            # If there's no crossing in the future or well is already below limit, 
            # just show up to the 20-year mark
            x_max = x_max_20_years

        # 4) Also ensure we do not extend beyond the actual forecast's max date
        actual_forecast_end = forecast_for_plot['Date'].max() if not forecast_for_plot.empty else last_data_date
        x_max = min(x_max, actual_forecast_end)

        # Final x-limits
        plt.xlim(left=x_min, right=x_max)

        # If log scale, set y-limits to ensure we see the economic limit and any low-rate forecasts
        if log_scale:
            relevant_rates = []
            if not data_for_plot.empty:
                relevant_rates.extend(data_for_plot['GAS_PRODUCTION_RATE'].tolist())
            if not forecast_for_plot.empty:
                relevant_rates.extend(forecast_for_plot['GAS_PRODUCTION_RATE'].tolist())
            relevant_rates.append(economic_limit_daily)

            # Filter out zero or negative rates
            relevant_rates = [r for r in relevant_rates if r > 0]

            # If everything is zero or negative (edge case), provide a fallback
            if not relevant_rates:
                relevant_rates = [1e-6, 1]

            min_val = min(relevant_rates)
            max_val = max(relevant_rates)

            # Add a margin so we don’t cut off the top or bottom
            y_min = min_val / 1.5
            y_max = max_val * 1.1

            # Ensure bottom is strictly > 0 for log scale
            if y_min <= 0:
                y_min = 1e-6

            plt.ylim(bottom=y_min, top=y_max)

        # Configure x-axis date formatting
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        # Adjust locator if you prefer different intervals, e.g., every 5 years
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))
        
        # Final styling
        scale_label = "Log Scale" if log_scale else "Normal Scale"
        plt.title(f'Gas Production Rate and DCA Forecast for Well {well_number} ({scale_label})')
        plt.xlabel('Date')
        plt.ylabel('Gas Production Rate (E3M3/day)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, which='both' if log_scale else 'major')
        
        # Save the plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, right=0.85)
        
        plot_type = "log" if log_scale else "normal"
        filepath = os.path.join(self.output_dir, f"{plot_type}_scale", 
                                f"well_{well_number}_{plot_type}.png")
        self._save_figure(filepath)
