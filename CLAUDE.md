# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **tax policy analysis project** that studies the effects of the "One Big Beautiful Bill Act" (OBBBA) on U.S. households. The project consists of two main components:

1. **Analysis Engine** (`tax_impacts/`): Python scripts that use PolicyEngine-US to calculate household-level tax impacts
2. **Interactive Dashboard** (`app.py`): Streamlit web application for exploring the analysis results

## Architecture

### Core Components

- **`tax_impacts/main.py`**: Entry point for running tax impact analysis. Generates both House and Senate versions of the bill impacts
- **`tax_impacts/analysis.py`**: Core microsimulation analysis using PolicyEngine-US with enhanced CPS 2024 dataset 
- **`tax_impacts/reforms.py`**: Defines the specific tax reforms being analyzed (not read but referenced)
- **`app.py`**: Streamlit dashboard with comprehensive household filtering, selection, and visualization

### Data Flow

1. Analysis scripts generate CSV files: `household_tax_income_changes.csv` (House) and `household_tax_income_changes_senate.csv` (Senate)
2. Streamlit app loads these CSVs and provides interactive exploration of household impacts
3. Users can filter households by demographics, income, geography, and view detailed breakdowns

## Key Dependencies

- **policyengine-us**: Tax policy microsimulation engine
- **policyengine-core**: Core policy modeling framework
- **streamlit**: Web dashboard framework  
- **pandas/numpy**: Data analysis
- **plotly**: Interactive charting

## Common Development Commands

### Running Analysis
```bash
# Generate household impact data (takes significant time)
python tax_impacts/main.py
```

### Running Dashboard
```bash
# Start Streamlit dashboard
streamlit run app.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Analysis Configuration

- **Year**: 2026 tax year analysis
- **Dataset**: Enhanced CPS 2024 from PolicyEngine
- **Reforms**: Multiple tax code changes including tax rates, deductions, credits, and benefits
- **Output**: Household-level dollar and percentage changes across different analysis types (Federal Taxes, State Taxes, Net Income, Benefits)

## Dashboard Features

The Streamlit app provides:
- **Filtering**: By household weight, income, state, marital status, dependents, age
- **Selection Methods**: Random sampling, specific household ID, or "interesting cases" (highest/lowest impacts)
- **Analysis Types**: Federal taxes, state taxes, net income, or benefits focus
- **Visualizations**: Waterfall charts showing reform component contributions
- **Detailed Breakdowns**: Component-wise impact analysis for each reform

## Code Structure

The application uses a clean object-oriented architecture with dataclasses:

- **Configuration Classes**: `AppConfig`, `UIConfig`, `FilterConfig` for settings
- **Data Models**: `HouseholdProfile`, `ReformImpact` for structured data
- **Managers**: `DataManager`, `FilterManager` for specific responsibilities
- **Analysis Engine**: `TaxAnalysisEngine` handles different analysis types
- **Rendering**: `VisualizationRenderer` for UI components
- **Main App**: `HouseholdDashboard` orchestrates the entire application

Key patterns:
- Extensive use of dataclasses for type safety and structure
- Clear separation of concerns between data, analysis, and presentation
- Streamlit caching for performance
- Comprehensive error handling throughout