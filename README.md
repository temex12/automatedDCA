# Automated Decline Curve Analysis (AutoDCA)

A comprehensive Python package for automated Decline Curve Analysis (DCA) of oil and gas well production data. This tool provides robust capabilities for analyzing production decline trends, forecasting future production, and generating detailed visualization reports. It is distributed as the Python package `autodca`.

## Features

- **Automated Analysis**: Intelligent detection of optimal DCA start points using adaptive algorithms
- **Multiple Decline Models**: 
  - Hyperbolic decline
  - Harmonic decline
  - Exponential decline
- **Flexible Data Sources**:
  - Direct database connection
  - CSV file-based analysis
- **Visualization**:
  - Production forecasts in normal and log scales
  - Error analysis plots
  - Data coverage analysis
  - Error metrics heatmap
- **Well Type Detection**:
  - Automatic differentiation between conventional and unconventional wells
  - Customized analysis parameters based on well type
- **Robust Curve Fitting**:
  - Power-law time weighting to emphasize recent production behavior
  - Flat-fit prevention during start date selection
  - Counter-trend spike detection (e.g., workovers, recompletions)
  - Hyperbolic-to-exponential terminal decline transition
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
│   └── autodca/                 # The autodca package
│       ├── __init__.py          # Public API and package version
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
│   ├── well_2596/             # Example well directory
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
├── tests/
│   └── test_dca.py            # Unit tests
│
├── pyproject.toml             # Package metadata and dependencies
├── requirements.txt           # Exact package versions used for the paper results
├── LICENSE                    # MIT License
├── extractWellData.ipynb      # Notebook for extracting data from database to CSV
├── dcaAnalysis.ipynb          # Main notebook for database analysis
├── dcaAnalysisCSV.ipynb       # Notebook for CSV-based analysis
├── dca_parameter_extraction.ipynb  # Notebook for extracting fitted DCA parameters
└── MLComparison.ipynb         # Comparison with machine learning forecasts on hidden production
```

## Installation

Install the package from GitHub:
```bash
pip install "autodca @ git+https://github.com/temex12/automatedDCA.git"
```

Optional extras:
- `mysql`: the PyMySQL driver, needed to read data from a MySQL database
- `ml`: XGBoost, TensorFlow and joblib, needed by `MLComparison.ipynb`

```bash
pip install "autodca[mysql] @ git+https://github.com/temex12/automatedDCA.git"
```

To use the notebooks and run the tests, clone the repository and install it in editable mode:
```bash
git clone https://github.com/temex12/automatedDCA.git
cd automatedDCA
pip install -e ".[mysql]"
```

`requirements.txt` lists the exact package versions used to produce the results in the paper. The package has been tested with Python 3.8 and 3.12. In an environment that already has pandas older than 2.0, keep SQLAlchemy at version 1.4, because those pandas versions do not support SQLAlchemy 2.

## Usage

### Quick start

The `data/` folder of the repository contains example wells in the CSV format described under [Data Requirements](#data-requirements). Run the following from the root of a clone:

```python
from autodca import DCAAnalyzer, DCADataLoader

loader = DCADataLoader("data")  # or a database URL, e.g. "mysql+pymysql://user:pass@host/db"
analyzer = DCAAnalyzer(loader, time_period="Q")

# Analyze a single well
results = analyzer.perform_dca_analysis(well_authorization_number=2596)
params = results["parameter_summary"]
print(results["best_model"], results["best_start_date"].date())
print(f"qi = {params['q_model']:.1f} E3M3/day, Di = {params['d_model']:.3f} per year, b = {params['b_model']}")
print(f"EUR = {results['eur']:,.0f} E3M3")

# Analyze several wells; returns a summary table and writes summary files to results/
summary = analyzer.analyze_well_list([2596, 2035, 153])
```

### Results

`perform_dca_analysis` returns a dictionary. Its main entries are:

- `best_model`: the selected decline model (`Hyperbolic`, `Harmonic` or `Exponential`)
- `best_start_date`: the selected start of the decline period
- `parameter_summary`: the fitted parameters in field units, including `q_model` (initial rate, E3M3/day), `d_model` (initial decline rate, per year) and `b_model` (b-factor), and the terminal decline parameters
- `switch_time`: the time of the switch to terminal (exponential) decline, in years from the start of the decline period
- `cumulative_production`, `remaining_reserves` and `eur`: volumes in E3M3
- `forecast_df`: the fitted and forecast rates by quarter (`GAS_PRODUCTION_RATE`, E3M3/day)
- `production_data`: the quarterly production used in the analysis
- `full_mae`, `full_rmse`, `full_mape` and `full_smape`: the errors of the fitted curve over the decline period

Production is aggregated to quarters by default (`time_period="Q"`). Volumes are in E3M3 (thousand cubic metres) and rates in E3M3/day.

Each call to `perform_dca_analysis` also saves four plots of the well in the `results/` folder of the working directory (see [Output Files](#output-files)). To save the plots in another folder, replace the analyzer's visualizer:

```python
from autodca import DCAVisualizer

analyzer.visualizer = DCAVisualizer(output_dir="my_results")
```

### Notebooks

The repository includes Jupyter notebooks for different analysis scenarios. The notebooks add `src/` to the Python path, so they run from a clone of the repository with or without the package installed.

1. **dcaAnalysis.ipynb**: Main analysis notebook for working with database connections. This notebook:
   - Connects to a database source
   - Performs DCA analysis on specified wells
   - Generates visualizations and summary reports
   - Supports customization of analysis parameters

2. **dcaAnalysisCSV.ipynb**: Analysis notebook for working with CSV data files. This notebook:
   - Works with extracted well data in CSV format
   - Performs the same analysis as the database version
   - Includes data integrity verification
   - Generates identical outputs as the database version

3. **extractWellData.ipynb**: Utility notebook for extracting data from database to CSV format

4. **dca_parameter_extraction.ipynb**: Utility notebook for extracting fitted DCA parameters into a summary table

5. **MLComparison.ipynb**: Comparison of Auto-DCA with XGBoost, random forest and LSTM forecasts on 40 horizontal Montney wells. This notebook:
   - Hides the most recent 15%, 25%, 50% or 75% of each well's quarterly history and scores all methods on the same hidden quarters
   - Trains the machine learning models on other Montney horizontal gas wells, using only production recorded before each forecast start date
   - Reads the database URL from the `AUTODCA_DB_URL` environment variable
   - Requires the `ml` extra (`pip install -e ".[ml]"`); the results were produced with xgboost 2.1.4, TensorFlow 2.13.0 and joblib 1.4.2
   - Writes its results to `ml_comparison_results/`

### Analysis Parameters

Key parameters that can be adjusted in the notebooks:

- `min_coverage`: Minimum required data coverage (default: 0.3)
- `max_error`: Maximum acceptable error (default: 0.1)
- `error_tolerance_factor`: Factor for error tolerance calculation (default: 0.5)
- `trend_window`: Window size for trend analysis (default: 5)
- `mse_scale_factor`: Scale factor for MSE visualization (default: 1e3)
- `recent_weight_factor`: Base weight multiplier for recent data (default: 1.5; set to 1.0 to disable time weighting)
- `recent_data_fraction`: Fraction of the data treated as recent (default: 0.15)
- `weight_exponent`: Power-law exponent controlling how strongly early data is downweighted (default: 2.0; use 3.0+ to emphasize recent data more aggressively)
- `min_decline_rate`: Minimum acceptable decline rate during start date selection (default: auto-set by well type - 0.10/yr for unconventional, 0.05/yr for conventional)
- `flat_fit_penalty`: Error penalty multiplier for fits below the minimum decline rate (default: 10.0)
- `detect_spikes`: Detect and downweight recent production spikes that run counter to the decline trend (default: True)
- `spike_threshold`: Standard deviations above the fitted trend for a point to be treated as a spike (default: 1.0; lower values detect smaller deviations)

### Curve Fitting Methodology

Model fitting uses a two-pass approach:

1. **Pass 1**: Fit with uniform weights to establish an unbiased baseline trend
2. **Spike detection**: Identify recent data points significantly above the baseline (counter-trend spikes from workovers, recompletions, or measurement anomalies)
3. **Pass 2**: Refit with power-law time weights, with detected spikes effectively removed from the fit

This ensures the fitted curve tracks recent production behavior without being distorted by short-lived production spikes.

### Error Analysis Methodology

The error vs data coverage analysis uses multiple error metrics, with MSE and RMSE as the primary metrics for optimization. The MSE is scaled by a factor (default: 1e3) for visualization purposes. The analysis:

1. Calculates multiple error metrics (MSE, MAE, RMSE, MAPE, SMAPE) for different data coverage periods
2. Identifies the global minimum error point using MSE
3. Applies an error tolerance line based on the relative slope
4. Selects the optimal cutoff point that balances:
   - Minimizing error (primarily MSE)
   - Maximizing data coverage
   - Maintaining error within tolerance of the global minimum

## Testing

Run the unit tests from the repository root:

```bash
python tests/test_dca.py
```

## Output Files

The analysis generates the following output files in the `results` directory:

- `normal_scale/`: Production forecasts in normal scale
- `log_scale/`: Production forecasts in log scale
- `error_analysis/`: Error analysis plots showing MSE vs data coverage
- `rate_vs_cumulative/`: Production rate vs cumulative production plots
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

`MLComparison.ipynb` also uses `FORMTN_CODE` in `bc_total_production`, `FLUID` and `FORM` in `well_index`, and the table `well_authorizations_issued` (WA_NUM, NAD_83_LATITUDE, NAD_83_LONGITUDE).

### CSV Structure
Required files in each well directory:
- `production.csv`:
  - ZONE_PROD_PERIOD
  - GAS_PRODUCTION_E3M3
- `info.csv`:
  - WA
  - HZ_FLAG

The analysis supports both database connections and CSV files, with identical functionality and outputs regardless of the data source used. The CSV-based approach is particularly useful for offline analysis or when database access is restricted.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
