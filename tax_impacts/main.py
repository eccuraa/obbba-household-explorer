#!/usr/bin/env python3
"""
Main script for analyzing tax reform impacts.
Analyzes 2026 tax year using enhanced CPS dataset.
"""

from datetime import datetime
from reforms import tcja_reform, get_all_reforms, get_all_senate_finance_reforms
from analysis import calculate_stacked_household_impacts


def main():
    print(f"Tax Reform Impact Analysis")
    print(f"========================")
    print(f"Analysis year: 2026")
    print(f"Dataset: Enhanced CPS 2024")
    print(f"Starting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Get baseline and reforms
    baseline_reform = tcja_reform()
    reforms = get_all_reforms()

    print(f"Analyzing {len(reforms)} reform components:")
    for i, reform_name in enumerate(reforms.keys(), 1):
        print(f"  {i}. {reform_name}")
    print()

    # Calculate household-level impacts
    df = calculate_stacked_household_impacts(
        reforms=reforms, baseline_reform=baseline_reform, year=2026
    )

    # Save results
    output_file = "household_tax_income_changes.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved results to '{output_file}'")
    print(f"Total households analyzed: {len(df):,}")

    # Display sample results
    print(f"\nFirst 5 rows of results:")
    print(df.head())

    print(f"\nAnalysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Senate Finance analysis
    print("\n" + "=" * 40)
    print("Senate Finance Tax Reform Impact Analysis")
    print("=" * 40)
    senate_reforms = get_all_senate_finance_reforms()
    print(f"Analyzing {len(senate_reforms)} Senate reform components:")
    for i, reform_name in enumerate(senate_reforms.keys(), 1):
        print(f"  {i}. {reform_name}")
    print()
    df_senate = calculate_stacked_household_impacts(
        reforms=senate_reforms, baseline_reform=baseline_reform, year=2026
    )
    senate_output_file = "household_tax_income_changes_senate.csv"
    df_senate.to_csv(senate_output_file, index=False)
    print(f"\nSaved Senate results to '{senate_output_file}'")
    print(f"Total households analyzed (Senate): {len(df_senate):,}")
    print(f"\nFirst 5 rows of Senate results:")
    print(df_senate.head())


if __name__ == "__main__":
    main()
