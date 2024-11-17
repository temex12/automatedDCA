from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, inspect
from typing import Union, Optional, Tuple, List
import warnings

class DCADataLoader:
    """
    Data loader class that handles both database and extracted data sources for DCA analysis.
    Supports seamless switching between database and file-based data access.
    """
    def __init__(self, data_source: Union[str, Path]):
        """
        Initialize the data loader.
        
        Args:
            data_source: Either a database connection string (e.g., "mysql+pymysql://user:pass@localhost/db")
                        or path to extracted data directory
        """
        self.data_source = data_source
        self._initialize_source()
    
    def _initialize_source(self):
        """Initialize the appropriate data source."""
        if isinstance(self.data_source, str) and "://" in self.data_source:
            # Database connection
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
            # File system
            self.source_type = "files"
            self.data_dir = Path(self.data_source)
            if not self.data_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
            # Verify directory structure
            well_dirs = list(self.data_dir.glob('well_*'))
            if not well_dirs:
                raise FileNotFoundError(f"No well directories found in {self.data_dir}")
            
            # Check first well directory for required files
            first_well = well_dirs[0]
            required_files = ['production.csv', 'info.csv']
            for file in required_files:
                if not (first_well / file).exists():
                    raise FileNotFoundError(f"Required file {file} not found in well directories")
            
            print(f"Using extracted data from: {self.data_dir}")
            print(f"Found {len(well_dirs)} well directories")
    
    def validate_well_data(self, production_data: pd.DataFrame, well_info: pd.DataFrame) -> None:
        """
        Validate the structure of loaded data.
        
        Args:
            production_data: Production data DataFrame
            well_info: Well info DataFrame
            
        Raises:
            ValueError: If required columns are missing
        """
        # Check required columns in production data
        required_prod_columns = ['ZONE_PROD_PERIOD', 'GAS_PRODUCTION_E3M3']
        missing_prod = [col for col in required_prod_columns if col not in production_data.columns]
        if missing_prod:
            raise ValueError(f"Missing required columns in production data: {missing_prod}")
        
        # Check required columns in well info
        required_info_columns = ['WA', 'HZ_FLAG']
        missing_info = [col for col in required_info_columns if col not in well_info.columns]
        if missing_info:
            raise ValueError(f"Missing required columns in well info: {missing_info}")
    
    def get_well_list(self) -> List[int]:
        """
        Get list of available wells.
        
        Returns:
            List of well authorization numbers
        """
        if self.source_type == "database":
            query = "SELECT DISTINCT WELL_AUTHORIZATION_NUMBER FROM bc_total_production"
            return pd.read_sql_query(query, self.engine)['WELL_AUTHORIZATION_NUMBER'].tolist()
        else:
            wells = [int(d.name.split('_')[1]) for d in self.data_dir.glob('well_*')]
            if not wells:
                raise ValueError("No wells found in data directory")
            return sorted(wells)
    
    def load_well_data(self, well_number: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load production and well info data for a specific well.
        
        Args:
            well_number: Well authorization number
            
        Returns:
            Tuple of (production_data, well_info) DataFrames
        """
        try:
            if self.source_type == "database":
                # Database queries
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
                # File system
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
            
            # Ensure date column is datetime
            production_data['ZONE_PROD_PERIOD'] = pd.to_datetime(production_data['ZONE_PROD_PERIOD'])
            
            # Sort production data by date
            production_data = production_data.sort_values('ZONE_PROD_PERIOD')
            
            # Check for missing values
            if production_data['GAS_PRODUCTION_E3M3'].isna().any():
                warnings.warn(f"Found missing production values for well {well_number}")
            
            return production_data, well_info
            
        except Exception as e:
            raise Exception(f"Error loading data for well {well_number}: {str(e)}")
    
    def get_metadata(self, well_number: int) -> Optional[str]:
        """
        Get metadata for a specific well (only available for extracted data).
        
        Args:
            well_number: Well authorization number
            
        Returns:
            Metadata string if available, None otherwise
        """
        if self.source_type == "files":
            metadata_path = self.data_dir / f'well_{well_number}' / 'metadata.txt'
            if metadata_path.exists():
                return metadata_path.read_text()
        return None
    
    def export_data(self, well_list: List[int], output_dir: Union[str, Path]):
        """
        Export data from database to file system (only available for database source).
        
        Args:
            well_list: List of well numbers to export
            output_dir: Directory to save exported data
        """
        if self.source_type != "database":
            raise ValueError("Export is only available when using database source")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = []
        for well_number in well_list:
            try:
                production_data, well_info = self.load_well_data(well_number)
                
                # Create well directory
                well_dir = output_dir / f'well_{well_number}'
                well_dir.mkdir(exist_ok=True)
                
                # Save data
                production_data.to_csv(well_dir / 'production.csv', index=False)
                well_info.to_csv(well_dir / 'info.csv', index=False)
                
                # Create metadata
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
        """Create metadata file for exported data."""
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
        Verify the integrity of all well data in the directory.
        Only available for file-based data source.
        
        Returns:
            DataFrame containing verification results
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