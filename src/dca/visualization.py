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
├── log_scale/        # Logarithmic scale production plots
└── error_analysis/   # Error vs coverage analysis plots

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
        """
        self.output_dir = output_dir
        self._create_output_directories()
        
        # Set default plotting parameters
        plt.style.use('seaborn')  # Clean, modern style
        self.default_figsize = (16, 8)  # Widescreen format
        self.dpi = 300  # High resolution output
    
    def _create_output_directories(self):
        """
        Create directory structure for output files.
        Creates directories if they don't exist.
        """
        for subdir in ["normal_scale", "log_scale", "error_analysis"]:
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
            well_number (int): Well identification number
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
            
            # Add error tolerance line
            tolerance_line_x = [global_min_coverage, max_coverage]
            tolerance_line_y = [global_min_error, 
                              global_min_error + (relative_slope * scale_factor) * (max_coverage - global_min_coverage)]
            plt.plot(tolerance_line_x, tolerance_line_y, 'g--', label='Error Tolerance Line')
            
            # Highlight key points
            plt.plot(best_coverage, best_error * scale_factor, marker='s', color='green', 
                    markersize=10, label='Chosen Point')
            plt.plot(global_min_coverage, global_min_error, marker='*', color='black', 
                    markersize=12, label='Global Minimum')
            
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
        """
        plt.figure(figsize=(16, 8))
        
        # Handle data for log scale
        if log_scale:
            production_data = production_data[production_data['GAS_PRODUCTION_RATE'] > 0]
        
        # Plot historical data points
        plt.scatter(production_data['ZONE_PROD_PERIOD'], production_data['GAS_PRODUCTION_RATE'],
                   color='blue', label='Historical Data', alpha=0.6, s=40)
        
        # Split and plot DCA fit and forecast
        mask_fit = forecast_df['Date'] <= last_data_date
        mask_forecast = forecast_df['Date'] > last_data_date
        
        plt.plot(forecast_df[mask_fit]['Date'], forecast_df[mask_fit]['GAS_PRODUCTION_RATE'],
                color='red', label='DCA Fit', linewidth=2)
        plt.plot(forecast_df[mask_forecast]['Date'], forecast_df[mask_forecast]['GAS_PRODUCTION_RATE'],
                color='green', linestyle='--', label='DCA Forecast', linewidth=2)
        
        # Add reference lines
        plt.axhline(y=economic_limit_daily, color='black', linestyle='--', label='Economic Limit')
        plt.axvline(x=best_start_date, color='green', linestyle='--', linewidth=2, label='DCA Start Date')
        
        # Configure axis properties
        if log_scale:
            plt.yscale('log')
            y_min = max(production_data['GAS_PRODUCTION_RATE'].min(), 0.1)
            y_max = max(production_data['GAS_PRODUCTION_RATE'].max(),
                       forecast_df['GAS_PRODUCTION_RATE'].max())
            plt.ylim(bottom=y_min, top=y_max * 1.1)
            plot_type = "log"
            title_suffix = " (Log Scale)"
        else:
            plot_type = "normal"
            title_suffix = ""
        
        # Set x-axis range and format
        x_min = production_data['ZONE_PROD_PERIOD'].min()
        plt.xlim(left=x_min, right=pd.Timestamp('2060-01-01'))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(10))
        
        # Add labels and styling
        plt.title(f'Gas Production Rate and DCA Forecast for Well {well_number}{title_suffix}')
        plt.xlabel('Date')
        plt.ylabel('Gas Production Rate (E3M3/day)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, which='both' if log_scale else 'major')
        
        # Adjust layout and save plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, right=0.85)
        
        filepath = os.path.join(self.output_dir, f"{plot_type}_scale", 
                              f"well_{well_number}_{plot_type}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()