# Well Data for DCA Analysis

This directory contains production and well data for Decline Curve Analysis (DCA).

## Directory Structure
/data
    /well_[number]/
        - production.csv    # Production time series data
        - info.csv         # Well characteristics and information
        - metadata.txt     # Data descriptions and statistics
    extraction_summary.csv # Summary of data extraction results

## File Descriptions

### production.csv
- ZONE_PROD_PERIOD: Production period date
- GAS_PRODUCTION_E3M3: Gas production in thousand cubic meters

### info.csv
- WA: Well Authorization number
- WELL_NAME: Name of the well
- AREA: Area name
- OPER: Operator name
- HZ_FLAG: Horizontal well flag
- FORM: Formation
- POOL_SEQ: Pool sequence
- UWI: Unique Well Identifier
- MAX_H2S_PERCENT: Maximum H2S percentage
- BOTTOM_HOLE: Bottom hole location

### metadata.txt
Contains detailed information about the data including:
- Data structure information
- Basic statistics
- Column descriptions

## Usage
This data can be used with the DCA analysis code without requiring database access.
