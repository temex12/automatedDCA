# Automated Decline Curve Analysis (AutoDCA)

A comprehensive Python package for automated Decline Curve Analysis (DCA) of oil and gas well production data. This tool provides robust capabilities for analyzing production decline trends, forecasting future production, and generating detailed visualization reports.

## Features

- **Automated Analysis**: Intelligent detection of optimal DCA start points using adaptive algorithms
- **Multiple Decline Models**: 
  - Hyperbolic decline
  - Harmonic decline
  - Exponential decline
  - Combined decline (hybrid model)
- **Flexible Data Sources**:
  - Direct database connection
  - CSV file-based analysis
- **Advanced Visualization**:
  - Production forecasts in normal and log scales
  - Error analysis plots
  - Data coverage analysis
  - Error metrics heatmap
- **Well Type Detection**:
  - Automatic differentiation between conventional and unconventional wells
  - Customized analysis parameters based on well type
- **Comprehensive Error Metrics**:
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Square Error)
  - MAPE (Mean Absolute Percentage Error)
  - SMAPE (Symmetric Mean Absolute Percentage Error)

## Project Structure

```
automatedDCA/
│
├── src/
│   ├── __init__.py
│   │
│   └── dca/
│       ├── __init__.py
│       ├── analysis.py          # DCAAnalyzer class
│       ├── data_loader.py       # DCADataLoader class
│       ├── models.py            # DeclineModels class
│       ├── utils.py             # MetricsCalculator and DataPreprocessor classes
│       └── visualization.py     # DCAVisualizer class
│
├── data/                        # Directory for extracted well data
│   ├── README.md               # Data documentation
│   ├── extraction_summary.csv  # Summary of data extraction
│   │
│   ├── well_3899/             # Example well directory
│   │   ├── production.csv     # Production time series data
│   │   ├── info.csv          # Well characteristics
│   │   └── metadata.txt       # Well metadata
│   │
│   └── well_[number]/         # Other well directories...
│
├── results/                    # Created when running analysis
│   ├── normal_scale/          # Normal scale DCA plots
│   ├── log_scale/            # Log scale DCA plots
│   ├── error_analysis/       # Error analysis plots
│   ├── model_distribution.png
│   ├── well_type_distribution.png
│   ├── error_metrics_heatmap.png
│   ├── dca_analysis_summary.csv
│   └── dca_analysis_summary.xlsx
│
├── requirements.txt           # Python package dependencies
├── extractWellData.ipynb      # Notebook for extracting data from database to CSV
├── dca_analysis.ipynb         # Main notebook for database analysis
└── dca_analysis_csv.ipynb     # Notebook for CSV-based analysis
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/automatedDCA.git
cd automatedDCA
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

The package includes three Jupyter notebooks for different analysis scenarios:

1. **dca_analysis.ipynb**: Main analysis notebook for working with database connections. This notebook:
   - Connects to a database source
   - Performs DCA analysis on specified wells
   - Generates visualizations and summary reports
   - Supports customization of analysis parameters

2. **dca_analysis_csv.ipynb**: Analysis notebook for working with CSV data files. This notebook:
   - Works with extracted well data in CSV format
   - Performs the same analysis as the database version
   - Includes data integrity verification
   - Generates identical outputs as the database version

3. **extractWellData.ipynb**: Utility notebook for extracting data from database to CSV format

### Analysis Parameters

Key parameters that can be adjusted in the notebooks:

- `min_coverage`: Minimum required data coverage (default: 0.3)
- `max_error`: Maximum acceptable error (default: 0.1)
- `error_tolerance_factor`: Factor for error tolerance calculation (default: 0.5)
- `trend_window`: Window size for trend analysis (default: 5)
- `mse_scale_factor`: Scale factor for MSE visualization (default: 1e3)

### Error Analysis Methodology

The error vs data coverage analysis uses multiple error metrics, with MSE and RMSE as the primary metrics for optimization. The MSE is scaled by a factor (default: 1e3) for visualization purposes. The analysis:

1. Calculates multiple error metrics (MSE, MAE, RMSE, MAPE, SMAPE) for different data coverage periods
2. Identifies the global minimum error point using MSE
3. Applies an error tolerance line based on the relative slope
4. Selects the optimal cutoff point that balances:
   - Minimizing error (primarily MSE)
   - Maximizing data coverage
   - Maintaining error within tolerance of the global minimum

## Output Files

The analysis generates the following output files in the `results` directory:

- `normal_scale/`: Production forecasts in normal scale
- `log_scale/`: Production forecasts in log scale
- `error_analysis/`: Error analysis plots showing MSE vs data coverage
- `model_distribution.png`: Distribution of best-fitting models
- `well_type_distribution.png`: Distribution of well types
- `error_metrics_heatmap.png`: Heatmap visualization of normalized error metrics
- `dca_analysis_summary.csv`: Summary of analysis results
- `dca_analysis_summary.xlsx`: Excel version of the summary with multiple sheets:
  - Full Summary: Complete analysis results
  - Error Statistics: Statistical breakdown of error metrics
  - Error Ranges: Min, max, range, and median for each error metric

## Data Requirements

### Database Structure
Required tables and columns for database analysis:
- `bc_total_production`:
  - WELL_AUTHORIZATION_NUMBER
  - ZONE_PROD_PERIOD
  - GAS_PRODUCTION_E3M3
- `well_index`:
  - WA
  - HZ_FLAG (for well type detection)

### CSV Structure
Required files in each well directory:
- `production.csv`:
  - ZONE_PROD_PERIOD
  - GAS_PRODUCTION_E3M3
- `info.csv`:
  - WA
  - HZ_FLAG

The analysis supports both database connections and CSV files, with identical functionality and outputs regardless of the data source used. The CSV-based approach is particularly useful for offline analysis or when database access is restricted.