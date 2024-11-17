import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

class DCAVisualizer:
    """
    Class for visualizing Decline Curve Analysis (DCA) results.
    Handles creation of error analysis and production forecast plots.
    """
    
    def __init__(self, output_dir="results"):
        """
        Initialize DCA Visualizer with output directory structure.
        
        Args:
            output_dir (str): Base directory for saving output files
        """
        self.output_dir = output_dir
        self._create_output_directories()
        
        # Set default plotting style
        plt.style.use('seaborn')
        self.default_figsize = (16, 8)
        self.dpi = 300
    
    def _create_output_directories(self):
        """Create necessary output directories if they don't exist."""
        for subdir in ["normal_scale", "log_scale", "error_analysis"]:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
    
    def _save_figure(self, filepath):
        """
        Helper method to save figures with consistent settings.
        
        Args:
            filepath (str): Path where the figure should be saved
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
        Plot error vs data coverage analysis.
        
        Args:
            results (list): List of dictionaries containing analysis results
            best_coverage (float): Coverage value for chosen point
            best_error (float): Error value for chosen point
            relative_slope (float): Slope of error tolerance line
            well_number (int): Well authorization number
            scale_factor (float): Scale factor for error values
        """
        try:
            plt.figure(figsize=self.default_figsize)
            dates = [r['start_date'] for r in results]
            errors = [r['error'] * scale_factor for r in results]
            coverages = [r['data_coverage'] for r in results]
            
            # Plot combined results with coverage vs error
            plt.plot(coverages, errors, 'b-o', label='Error vs Coverage', markersize=5)
            
            # Find global minimum
            global_min_index = np.argmin(errors)
            global_min_coverage = coverages[global_min_index]
            global_min_error = errors[global_min_index]
            max_coverage = max(coverages)
            
            # Plot error tolerance line
            tolerance_line_x = [global_min_coverage, max_coverage]
            tolerance_line_y = [global_min_error, 
                              global_min_error + (relative_slope * scale_factor) * (max_coverage - global_min_coverage)]
            plt.plot(tolerance_line_x, tolerance_line_y, 'g--', label='Error Tolerance Line')
            
            # Highlight points
            plt.plot(best_coverage, best_error * scale_factor, marker='s', color='green', 
                    markersize=10, label='Chosen Point')
            plt.plot(global_min_coverage, global_min_error, marker='*', color='black', 
                    markersize=12, label='Global Minimum')
            
            # Add labels with offsets
            self._add_annotations(best_coverage, best_error, scale_factor, 
                                global_min_coverage, global_min_error)
            
            # Set labels and title
            plt.xlabel('Data Coverage')
            plt.ylabel(f'MSE Error (x {scale_factor:,.0f})')
            plt.title(f'Error vs Data Coverage - Well {well_number}')
            plt.grid(True)
            plt.legend()
            
            # Format x-axis as percentage
            plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            
            # Add year labels
            self._add_year_labels(coverages, errors, dates)
            
            plt.tight_layout()
            filepath = os.path.join(self.output_dir, "error_analysis", 
                                  f"well_{well_number}_error_analysis.png")
            self._save_figure(filepath)
            
        except Exception as e:
            print(f"Error creating error analysis plot for well {well_number}: {str(e)}")
            plt.close()
    
    def _add_annotations(self, best_coverage, best_error, scale_factor, 
                        global_min_coverage, global_min_error):
        """Helper method to add annotations to error analysis plot."""
        plt.annotate('CP', (best_coverage, best_error * scale_factor),
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
                    arrowprops=dict(arrowstyle="->"))
        plt.annotate('GM', (global_min_coverage, global_min_error),
                    xytext=(25, 25), textcoords='offset points',
                    bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
                    arrowprops=dict(arrowstyle="->"))
    
    def _add_year_labels(self, coverages, errors, dates):
        """Helper method to add year labels to error analysis plot."""
        for i, (coverage, error, date) in enumerate(zip(coverages, errors, dates)):
            if i % 3 == 0:  # Adjust density of labels
                plt.annotate(date.strftime('%Y'), (coverage, error), 
                           textcoords="offset points", xytext=(0, 10), 
                           ha='center', va='bottom', color='black', 
                           fontsize=8, rotation=30)
        
    def save_dca_plots(self, production_data, forecast_df, well_number, 
                      best_start_date, economic_limit_daily):
        """
        Save normal and log-scale DCA plots.
        
        Args:
            production_data (pd.DataFrame): Historical production data
            forecast_df (pd.DataFrame): Forecast data
            well_number (int): Well authorization number
            best_start_date (datetime): Start date for DCA analysis
            economic_limit_daily (float): Economic limit for production
        """
        last_data_date = production_data['ZONE_PROD_PERIOD'].max()
        
        # Create normal scale plot
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
        """Helper method to create production plots."""
        plt.figure(figsize=(16, 8))
        
        # Filter data for log scale if needed
        if log_scale:
            production_data = production_data[production_data['GAS_PRODUCTION_RATE'] > 0]
        
        # Plot historical data
        plt.scatter(production_data['ZONE_PROD_PERIOD'], production_data['GAS_PRODUCTION_RATE'],
                   color='blue', label='Historical Data', alpha=0.6, s=40)
        
        # Plot DCA fit and forecast
        mask_fit = forecast_df['Date'] <= last_data_date
        mask_forecast = forecast_df['Date'] > last_data_date
        
        plt.plot(forecast_df[mask_fit]['Date'], forecast_df[mask_fit]['GAS_PRODUCTION_RATE'],
                color='red', label='DCA Fit', linewidth=2)
        plt.plot(forecast_df[mask_forecast]['Date'], forecast_df[mask_forecast]['GAS_PRODUCTION_RATE'],
                color='green', linestyle='--', label='DCA Forecast', linewidth=2)
        
        # Add reference lines
        plt.axhline(y=economic_limit_daily, color='black', linestyle='--', label='Economic Limit')
        plt.axvline(x=best_start_date, color='green', linestyle='--', linewidth=2, label='DCA Start Date')
        
        # Set axis properties
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
        
        # Set x-axis limits and format
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
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, right=0.85)
        
        filepath = os.path.join(self.output_dir, f"{plot_type}_scale", 
                              f"well_{well_number}_{plot_type}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()