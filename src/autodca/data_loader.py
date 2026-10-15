"""
Data Loader Module for Decline Curve Analysis (DCA)

This module provides flexible data loading capabilities for DCA analysis, supporting both:
1. Direct database connections (SQL databases)
2. File-based data access (CSV files in a structured directory)

Key Features:
- Automatic source type detection and initialization
- Comprehensive data validation
- Production and well information loading
- Data export capabilities
- Data integrity verification
- Metadata generation and management

Directory Structure (for file-based access):
automatedDCA/
└── data/
    ├── well_1234/
    │   ├── production.csv  (production time series)
    │   ├── info.csv       (well characteristics)
    │   └── metadata.txt   (data statistics and info)
    └── well_5678/
        └── ...

Required Data Columns:
- Production Data: ZONE_PROD_PERIOD, GAS_PRODUCTION_E3M3
- Well Info: WA, HZ_FLAG
"""

from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, inspect
from typing import Union, Optional, Tuple, List
import warnings

class DCADataLoader:
    def __init__(self, data_source: Union[str, Path]):
        """
        Initialize the data loader with either database connection or file path.
        
        Args:
            data_source: Either a database connection string (e.g., "mysql+pymysql://user:pass@localhost/db")
                      or path to extracted data directory containing well_* subdirectories
                      
        Raises:
            ConnectionError: If database connection fails
            FileNotFoundError: If data directory or required files are missing
        """
        self.data_source = data_source
        self._initialize_source()
    
    def _initialize_source(self):
        """
        Initialize and validate the data source.
        
        Detects source type (database/files) and performs initial validation:
        - For database: Tests connection
        - For files: Verifies directory structure and required files
        
        Raises:
            ConnectionError: If database connection fails
            FileNotFoundError: If directory structure is invalid
        """
        if isinstance(self.data_source, str) and "://" in self.data_source:
            # Handle database connection initialization
            self.source_type = "database"
            try:
                self.engine = create_engine(self.data_source)
                # Test connection
                with self.engine.connect() as conn:
                    pass
                print("Database connection successful!")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to database: {str(e)}")
        else:
            # Handle file system initialization
            self.source_type = "files"
            self.data_dir = Path(self.data_source)
            if not self.data_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
            # Verify well directories exist
            well_dirs = list(self.data_dir.glob('well_*'))
            if not well_dirs:
                raise FileNotFoundError(f"No well directories found in {self.data_dir}")
            
            # Validate required files in first well directory
            first_well = well_dirs[0]
            required_files = ['production.csv', 'info.csv']
            for file in required_files:
                if not (first_well / file).exists():
                    raise FileNotFoundError(f"Required file {file} not found in well directories")
            
            print(f"Using extracted data from: {self.data_dir}")
            print(f"Found {len(well_dirs)} well directories")
    
    def validate_well_data(self, production_data: pd.DataFrame, well_info: pd.DataFrame) -> None:
        """
        Validate the structure and required columns of well data.
        
        Args:
            production_data: DataFrame containing production time series
                Required columns: ZONE_PROD_PERIOD, GAS_PRODUCTION_E3M3
            well_info: DataFrame containing well characteristics
                Required columns: WA, HZ_FLAG
            
        Raises:
            ValueError: If any required columns are missing
        """
        # Validate production data columns
        required_prod_columns = ['ZONE_PROD_PERIOD', 'GAS_PRODUCTION_E3M3']
        missing_prod = [col for col in required_prod_columns if col not in production_data.columns]
        if missing_prod:
            raise ValueError(f"Missing required columns in production data: {missing_prod}")
        
        # Validate well info columns
        required_info_columns = ['WA', 'HZ_FLAG']
        missing_info = [col for col in required_info_columns if col not in well_info.columns]
        if missing_info:
            raise ValueError(f"Missing required columns in well info: {missing_info}")
    
    def get_well_list(self) -> List[int]:
        """
        Retrieve list of all available well authorization numbers.
        
        Returns:
            List of unique well authorization numbers (integers)
            
        Raises:
            ValueError: If no wells are found in the data source
        """
        if self.source_type == "database":
            # Query unique well numbers from database
            query = "SELECT DISTINCT WELL_AUTHORIZATION_NUMBER FROM bc_total_production"
            return pd.read_sql_query(query, self.engine)['WELL_AUTHORIZATION_NUMBER'].tolist()
        else:
            # Extract well numbers from directory names
            wells = [int(d.name.split('_')[1]) for d in self.data_dir.glob('well_*')]
            if not wells:
                raise ValueError("No wells found in data directory")
            return sorted(wells)
    
    def load_well_data(self, well_number: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load complete production and well information for a specific well.
        
        Args:
            well_number: Well authorization number to load
            
        Returns:
            Tuple containing:
            - production_data: DataFrame with production time series
            - well_info: DataFrame with well characteristics
            
        Raises:
            FileNotFoundError: If well data is not found
            ValueError: If data files are empty or malformed
            Exception: For other loading errors
        """
        try:
            if self.source_type == "database":
                # Load from database using SQL queries
                production_query = f"""
                SELECT 
                    ZONE_PROD_PERIOD,
                    GAS_PRODUCTION_E3M3
                FROM bc_total_production
                WHERE WELL_AUTHORIZATION_NUMBER = '{well_number}'
                ORDER BY ZONE_PROD_PERIOD
                """
                
                well_info_query = f"""
                SELECT 
                    WA,
                    WELL_NAME,
                    AREA,
                    OPER,
                    HZ_FLAG,
                    FORM,
                    POOL_SEQ,
                    UWI,
                    MAX_H2S_PERCENT,
                    BOTTOM_HOLE
                FROM well_index
                WHERE WA = '{well_number}'
                """
                
                production_data = pd.read_sql_query(production_query, self.engine)
                well_info = pd.read_sql_query(well_info_query, self.engine)
                
            else:
                # Load from CSV files
                well_dir = self.data_dir / f'well_{well_number}'
                if not well_dir.exists():
                    raise FileNotFoundError(f"Data not found for well {well_number}")
                
                try:
                    production_data = pd.read_csv(well_dir / 'production.csv')
                    well_info = pd.read_csv(well_dir / 'info.csv')
                except pd.errors.EmptyDataError:
                    raise ValueError(f"Empty data files for well {well_number}")
                except pd.errors.ParserError:
                    raise ValueError(f"Error parsing CSV files for well {well_number}")
            
            # Validate data structure
            self.validate_well_data(production_data, well_info)
            
            # Convert date column to datetime type
            production_data['ZONE_PROD_PERIOD'] = pd.to_datetime(production_data['ZONE_PROD_PERIOD'])
            
            # Ensure chronological order
            production_data = production_data.sort_values('ZONE_PROD_PERIOD')
            
            # Check for data quality issues
            if production_data['GAS_PRODUCTION_E3M3'].isna().any():
                warnings.warn(f"Found missing production values for well {well_number}")
            
            return production_data, well_info
            
        except Exception as e:
            raise Exception(f"Error loading data for well {well_number}: {str(e)}")
    
    def get_metadata(self, well_number: int) -> Optional[str]:
        """
        Retrieve metadata for a specific well from metadata.txt file.
        Only available for file-based data source.
        
        Args:
            well_number: Well authorization number
            
        Returns:
            String containing metadata if available, None otherwise
        """
        if self.source_type == "files":
            metadata_path = self.data_dir / f'well_{well_number}' / 'metadata.txt'
            if metadata_path.exists():
                return metadata_path.read_text()
        return None
    
    def export_data(self, well_list: List[int], output_dir: Union[str, Path]):
        """
        Export well data from database to CSV files.
        Only available when using database as data source.
        
        Creates a directory structure with:
        - One subdirectory per well
        - production.csv and info.csv files
        - metadata.txt containing data summaries
        - export_summary.csv tracking export status
        
        Args:
            well_list: List of well numbers to export
            output_dir: Directory where exported data will be saved
            
        Raises:
            ValueError: If not using database source
        """
        if self.source_type != "database":
            raise ValueError("Export is only available when using database source")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = []
        for well_number in well_list:
            try:
                # Load well data
                production_data, well_info = self.load_well_data(well_number)
                
                # Create well directory
                well_dir = output_dir / f'well_{well_number}'
                well_dir.mkdir(exist_ok=True)
                
                # Save CSV files
                production_data.to_csv(well_dir / 'production.csv', index=False)
                well_info.to_csv(well_dir / 'info.csv', index=False)
                
                # Generate metadata
                self._create_metadata(well_dir, production_data, well_info)
                
                print(f"Successfully exported data for well {well_number}")
                results.append({'well_number': well_number, 'status': 'success'})
                
            except Exception as e:
                print(f"Error exporting well {well_number}: {str(e)}")
                results.append({'well_number': well_number, 'status': 'failed', 'error': str(e)})
        
        # Save export summary
        pd.DataFrame(results).to_csv(output_dir / 'export_summary.csv', index=False)
    
    def _create_metadata(self, well_dir: Path, production_data: pd.DataFrame, 
                        well_info: pd.DataFrame):
        """
        Generate metadata file containing data summaries and statistics.
        
        Creates a metadata.txt file with:
        - Production data information and memory usage
        - Well information details
        - Production data statistical summary
        
        Args:
            well_dir: Directory where metadata file will be saved
            production_data: Production time series DataFrame
            well_info: Well characteristics DataFrame
        """
        with open(well_dir / 'metadata.txt', 'w') as f:
            f.write("Production Data Information:\n")
            f.write("-" * 50 + "\n")
            f.write(str(production_data.info(memory_usage='deep')))
            f.write("\n\nWell Information:\n")
            f.write("-" * 50 + "\n")
            f.write(str(well_info.info(memory_usage='deep')))
            f.write("\n\nProduction Data Statistics:\n")
            f.write("-" * 50 + "\n")
            f.write(str(production_data.describe()))
            
    def verify_data_integrity(self) -> pd.DataFrame:
        """
        Verify integrity of all well data files.
        Only available for file-based data source.
        
        Checks:
        - Presence of required files
        - Data loading capability
        - Basic data validity
        
        Returns:
            DataFrame containing verification results for each well:
            - well_number: Well authorization number
            - status: 'valid' or 'invalid'
            - production_rows: Number of production records (if valid)
            - date_range: Production date range (if valid)
            - error: Error message (if invalid)
            
        Raises:
            ValueError: If not using file-based source
        """
        if self.source_type != "files":
            raise ValueError("Data integrity verification only available for file-based data source")
        
        results = []
        for well_dir in self.data_dir.glob('well_*'):
            well_number = int(well_dir.name.split('_')[1])
            try:
                production_data, well_info = self.load_well_data(well_number)
                results.append({
                    'well_number': well_number,
                    'status': 'valid',
                    'production_rows': len(production_data),
                    'date_range': f"{production_data['ZONE_PROD_PERIOD'].min()} to {production_data['ZONE_PROD_PERIOD'].max()}"
                })
            except Exception as e:
                results.append({
                    'well_number': well_number,
                    'status': 'invalid',
                    'error': str(e)
                })
        
        return pd.DataFrame(results)