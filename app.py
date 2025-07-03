"""
Reconciliation Bill - Household Impact Dashboard

A Streamlit application for analyzing the impact of the reconciliation bill on individual households.
Provides interactive filtering, analysis type selection, and detailed impact visualization.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import os
from policyengine_core.charts import format_fig as format_fig_


def format_fig(fig):
    # Apply the original formatting
    formatted_fig = format_fig_(fig).update_layout(
        margin_r=120,
    )

    # Make the logo smaller by reducing its size
    if hasattr(formatted_fig, "layout") and hasattr(formatted_fig.layout, "images"):
        for image in formatted_fig.layout.images:
            # Reduce logo size from 0.15 to 0.1 (about 33% smaller)
            image.sizex = 0.1
            image.sizey = 0.1

    return formatted_fig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Supported analysis focus types for the dashboard."""

    NET_INCOME = "Net Income"
    FEDERAL_TAXES = "Federal Taxes"
    STATE_TAXES = "State Taxes"

    BENEFITS = "Benefits"


class AppConfig:
    """Application-wide configuration constants."""

    # CSV file mappings for different baselines and reform types
    CSV_FILES = {
        (
            "Current Law",
            "House",
        ): "household_tax_income_changes_current_law_baseline.csv",
        (
            "Current Law",
            "Senate",
        ): "household_tax_income_changes_senate_current_law_baseline.csv",
        ("Current Policy", "House"): "household_tax_income_changes_tcja_baseline.csv",
        (
            "Current Policy",
            "Senate",
        ): "household_tax_income_changes_senate_tcja_baseline.csv",
    }

    # Legacy filenames for backward compatibility
    HOUSE_CSV_FILENAME = "household_tax_income_changes.csv"
    SENATE_CSV_FILENAME = "household_tax_income_changes_senate.csv"
    INCOME_CHANGE_THRESHOLD = 0.01
    SIGNIFICANT_IMPACT_THRESHOLD = 1000
    MODERATE_IMPACT_THRESHOLD = 100
    MAX_DEPENDENTS = 11  # Based on IRS maximum dependent limit for tax modeling
    CHART_HEIGHT = 500

    # Reform column mappings: (Display Name, Column Suffix)
    REFORM_COLS = [
        ("Tax Rate Reform", "Tax Rate Reform"),
        ("Standard Deduction Reform", "Standard Deduction Reform"),
        ("Exemption Reform", "Exemption Reform"),
        ("Child Tax Credit Reform", "CTC Reform"),
        ("QBID Reform", "QBID Reform"),
        ("AMT Reform", "AMT Reform"),
        ("SALT Reform", "SALT Reform"),
        ("Estate Tax Reform", "Estate Tax Reform"),
        ("Tip Income Exemption", "Tip Income Exempt"),
        ("Senior Deduction Reform", "Senior Deduction Reform"),
        ("Overtime Income Exemption", "Overtime Income Exempt"),
        ("Auto Loan Interest Deduction", "Auto Loan Interest ALD"),
        ("Miscellaneous Reform", "Miscellaneous Reform"),
        ("Other Itemized Deductions Reform", "Other Itemized Deductions Reform"),
        (
            "Limitation on Itemized Deductions Reform",
            "Limitation on Itemized Deductions Reform",
        ),
        ("ACA Enhanced Subsidies Reform", "ACA Enhanced Subsidies Reform"),
        ("SNAP Reform", "SNAP Takeup Reform"),
        ("ACA Reform", "ACA Takeup Reform"),
        ("Medicaid Reform", "Medicaid Takeup Reform"),
    ]

    # Income source mappings for display
    INCOME_SOURCES = [
        ("Employment Income", "Employment Income"),
        ("Self-Employment Income", "Self-Employment Income"),
        ("Tip Income", "Tip Income"),
        ("Overtime Income", "Overtime Income"),
        ("Capital Gains", "Capital Gains"),
        ("Dividend Income", "Dividend Income"),
        ("Farm Income", "Farm Income"),
        ("Taxable Interest Income", "Taxable Interest Income"),
        ("Rental Income", "Rental Income"),
        ("Taxable Unemployment Compensation", "Taxable Unemployment Compensation"),
        ("Miscellaneous Income", "Miscellaneous Income"),
        ("Taxable Retirement Distributions", "Taxable Retirement Distributions"),
        ("Taxable Pension Income", "Taxable Pension Income"),
        ("Taxable Social Security", "Taxable Social Security"),
    ]


class UIConfig:
    """UI styling and configuration constants."""

    # Use transparent background with subtle backdrop
    CONTAINER_STYLE = "padding: 10px; border-radius: 5px; background-color: UIConfig.colors['BLUE_98']; backdrop-filter: brightness(0.95);"

    colors = {
        "BLACK": "#000000",
        "BLUE_98": "#F7FAFD",
        "BLUE": "#2C6496",
        "BLUE_LIGHT": "#D8E6F3",
        "BLUE_PRESSED": "#17354F",
        "DARK_BLUE_HOVER": "#1d3e5e",
        "DARK_GRAY": "#616161",
        "DARK_RED": "#b50d0d",
        "DARKEST_BLUE": "#0C1A27",
        "GRAY": "#808080",
        "LIGHT_GRAY": "#F2F2F2",
        "MEDIUM_DARK_GRAY": "#D2D2D2",
        "MEDIUM_LIGHT_GRAY": "#BDBDBD",
        "TEAL_ACCENT": "#39C6C0",
        "TEAL_LIGHT": "#F7FDFC",
        "TEAL_PRESSED": "#227773",
        "WHITE": "#FFFFFF",
    }


@dataclass
class FilterConfig:
    """Configuration for data filtering options."""

    weight_options: Dict[str, int]
    income_ranges: Dict[str, Tuple[float, float]]
    age_ranges: Dict[str, Tuple[int, int]]
    dependent_options: List[str]
    marital_options: List[str]
    single_tax_unit: bool

    @classmethod
    def default(cls) -> "FilterConfig":
        """Create default filter configuration."""
        return cls(
            weight_options={
                "All Households": 0,
                "Weight 1,000+": 1000,
                "Weight 5,000+": 5000,
                "Weight 10,000+": 10000,
                "Weight 25,000+": 25000,
                "Weight 50,000+": 50000,
            },
            income_ranges={
                "All Income Levels": (0, float("inf")),
                "Under $25k": (0, 25000),
                "$25k - $50k": (25000, 50000),
                "$50k - $100k": (50000, 100000),
                "$100k - $200k": (100000, 200000),
                "$200k+": (200000, float("inf")),
            },
            age_ranges={
                "All Ages": (0, 200),
                "Under 30": (0, 30),
                "30-40": (30, 40),
                "40-50": (40, 50),
                "50-60": (50, 60),
                "60-70": (60, 70),
                "70-80": (70, 80),
                "80+": (80, 200),
            },
            dependent_options=["All", "0", "1", "2", "3+"],
            marital_options=["All", "Married", "Single"],
            single_tax_unit=False,
        )


@dataclass
class ReformImpact:
    """Represents the financial impact of a single tax reform."""

    name: str
    total_change: float

    @property
    def is_significant(self) -> bool:
        """Check if the reform impact is significant enough to display."""
        return abs(self.total_change) > AppConfig.INCOME_CHANGE_THRESHOLD


@dataclass
class HouseholdProfile:
    """Demographic and financial profile of a household."""

    household_id: int
    state: str
    age_of_head: float
    age_of_spouse: Optional[float]
    number_of_dependents: int
    is_married: bool
    baseline_federal_tax: float
    baseline_net_income: float
    household_weight: float

    @classmethod
    def from_series(cls, series: pd.Series) -> "HouseholdProfile":
        """Create a HouseholdProfile from a pandas Series."""
        return cls(
            household_id=int(series["Household ID"]),
            state=series["State"],
            age_of_head=series["Age of Head"],
            age_of_spouse=series.get("Age of Spouse"),
            number_of_dependents=int(series["Number of Dependents"]),
            is_married=bool(series["Is Married"]),
            baseline_federal_tax=series["Baseline Federal Tax Liability"],
            baseline_net_income=series["Baseline Net Income"],
            household_weight=series["Household Weight"],
        )


class DataManager:
    """Handles data loading, validation, and caching."""

    @staticmethod
    @st.cache_data
    def load_data(csv_filename: str) -> pd.DataFrame:
        """
        Load and validate household data from CSV file.
        Args:
            csv_filename (str): Path to the CSV file to load
        Returns:
            pd.DataFrame: Validated household data
        Raises:
            FileNotFoundError: If CSV file is not found
            ValueError: If required columns are missing
        """
        try:
            df = pd.read_csv(csv_filename)
            # Tried to create the percent change column, but might not have worked.
            # For benefits: handle cases where baseline benefits is zero
            pct_benefits_change = np.zeros_like(df["Baseline Benefits"])
            mask = df["Baseline Benefits"] != 0
            pct_benefits_change[mask] = (
                df["Total Change in Benefits"][mask]
                / np.abs(df["Baseline Benefits"][mask])
            ) * 100
            df["Percentage Change in Benefits"] = pct_benefits_change
            DataManager._validate_data(df)
            logger.info(
                f"Successfully loaded {len(df)} household records from {csv_filename}"
            )
            return df
        except FileNotFoundError:
            st.error(f"Data file {csv_filename} not found")
            st.stop()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

    @staticmethod
    def _validate_data(df: pd.DataFrame) -> None:
        """Validate that all required columns exist in the dataframe."""
        required_columns = [
            "Household ID",
            "State",
            "Age of Head",
            "Number of Dependents",
            "Is Married",
            "Baseline Federal Tax Liability",
            "Baseline Net Income",
            "Household Weight",
            "Total Change in Federal Tax Liability",
            "Total Change in Net Income",
        ]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


class FilterManager:
    """Manages data filtering operations and UI rendering."""

    def __init__(self, config: FilterConfig):
        """Initialize with filter configuration."""
        self.config = config

    def render_and_apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Render filter UI in sidebar and apply selected filters to dataframe.

        Args:
            df: Original dataframe to filter

        Returns:
            pd.DataFrame: Filtered dataframe based on user selections
        """
        with st.sidebar.expander("🔍 Filters"):
            df_filtered = df.copy()

            # Apply each filter sequentially
            filter_methods = [
                self._apply_weight_filter,
                self._apply_income_filter,
                lambda x: self._apply_state_filter(x, df),
                self._apply_marital_filter,
                self._apply_dependents_filter,
                self._apply_age_filter,
                self._apply_tax_unit_filter,
            ]

            for filter_method in filter_methods:
                df_filtered = filter_method(df_filtered)

            self._display_filter_results(df_filtered, df)

        return df_filtered

    def _apply_weight_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply household weight filter."""
        selected = st.selectbox(
            "Minimum Household Weight:", list(self.config.weight_options.keys())
        )
        min_weight = self.config.weight_options[selected]
        return df[df["Household Weight"] >= min_weight] if min_weight > 0 else df

    def _apply_income_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply income range filter."""
        selected = st.selectbox("Net Income:", list(self.config.income_ranges.keys()))
        min_income, max_income = self.config.income_ranges[selected]
        if min_income > 0 or max_income < float("inf"):
            return df[
                (df["Baseline Net Income"] >= min_income)
                & (df["Baseline Net Income"] <= max_income)
            ]
        return df

    def _apply_state_filter(
        self, df_filtered: pd.DataFrame, df_original: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply state filter."""
        states = ["All States"] + sorted(df_original["State"].unique().tolist())
        selected = st.selectbox("State:", states)
        return (
            df_filtered[df_filtered["State"] == selected]
            if selected != "All States"
            else df_filtered
        )

    def _apply_marital_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply marital status filter."""
        selected = st.selectbox("Marital Status:", self.config.marital_options)
        if selected != "All":
            is_married = selected == "Married"
            return df[df["Is Married"] == is_married]
        return df

    def _apply_dependents_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply number of dependents filter."""
        selected = st.selectbox("Number of Dependents:", self.config.dependent_options)
        if selected != "All":
            if selected == "3+":
                return df[df["Number of Dependents"] >= 3]
            else:
                return df[df["Number of Dependents"] == int(selected)]
        return df

    def _apply_age_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply head of household age filter."""
        selected = st.selectbox(
            "Head of Household Age:", list(self.config.age_ranges.keys())
        )
        min_age, max_age = self.config.age_ranges[selected]
        if selected != "All Ages":
            return df[(df["Age of Head"] >= min_age) & (df["Age of Head"] < max_age)]
        return df

    def _apply_tax_unit_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply tax unit filter."""
        selected = st.checkbox(
            "Households with Only 1 Tax Unit", value=self.config.single_tax_unit
        )
        if selected:
            return df[df["Number of Tax Units"] == 1]
        return df

    def _display_filter_results(
        self, df_filtered: pd.DataFrame, df_original: pd.DataFrame
    ) -> None:
        """Display filter results summary."""
        st.caption(
            f"📊 Showing {len(df_filtered):,} of {len(df_original):,} households"
        )
        if len(df_filtered) == 0:
            st.error("No households match your filters!")
            st.stop()


class HouseholdSelector:
    """Handles household selection methods and UI."""

    @staticmethod
    def select_household(df_filtered: pd.DataFrame) -> int:
        """
        Render household selection UI and return selected household ID.

        Args:
            df_filtered: Filtered dataframe to select from

        Returns:
            int: Selected household ID
        """
        selection_method = st.sidebar.radio(
            "Selection Method:",
            ["Random Shuffle", "By Household ID", "Find Interesting Cases"],
        )

        selection_methods = {
            "By Household ID": HouseholdSelector._select_by_id,
            "Random Shuffle": HouseholdSelector._select_random,
            "Find Interesting Cases": HouseholdSelector._select_interesting_case,
        }

        return selection_methods[selection_method](df_filtered)

    @staticmethod
    def _select_by_id(df_filtered: pd.DataFrame) -> int:
        """Select household by ID."""
        return int(
            st.sidebar.selectbox(
                "Choose Household ID:", df_filtered["Household ID"].unique()
            )
        )

    @staticmethod
    def _select_random(df_filtered: pd.DataFrame) -> int:
        """Select random household with button to reshuffle, biased towards higher weights."""
        if st.sidebar.button("🎲 Get Random Household"):
            # Use household weight as sampling probability
            st.session_state.random_household = df_filtered.sample(
                1, weights="Household Weight"
            )["Household ID"].iloc[0]

        if "random_household" not in st.session_state:
            st.session_state.random_household = df_filtered.sample(
                1, weights="Household Weight"
            )["Household ID"].iloc[0]

        household_id = st.session_state.random_household
        st.sidebar.info(f"Random Household ID: {household_id}")
        return int(household_id)

    @staticmethod
    def _select_interesting_case(df_filtered: pd.DataFrame) -> int:
        """Select household from interesting cases (highest/lowest impacts)."""
        case_configs = {
            "Largest % Federal Tax Increase": (
                "nlargest",
                "Percentage Change in Federal Tax Liability",
            ),
            "Largest % Federal Tax Decrease": (
                "nsmallest",
                "Percentage Change in Federal Tax Liability",
            ),
            "Largest Federal Tax Increase": (
                "nlargest",
                "Total Change in Federal Tax Liability",
            ),
            "Largest Federal Tax Decrease": (
                "nsmallest",
                "Total Change in Federal Tax Liability",
            ),
            "Largest % Income Increase": (
                "nlargest",
                "Percentage Change in Net Income",
            ),
            "Largest % Income Decrease": (
                "nsmallest",
                "Percentage Change in Net Income",
            ),
            "Largest Income Increase": ("nlargest", "Total Change in Net Income"),
            "Largest Income Decrease": ("nsmallest", "Total Change in Net Income"),
        }

        case_type = st.sidebar.selectbox("Select Case Type:", list(case_configs.keys()))
        method, column = case_configs[case_type]

        try:
            top_households = getattr(df_filtered, method)(20, column)
            ranked_options, household_ids = HouseholdSelector._create_ranked_options(
                top_households, case_type
            )

            selected_option = st.sidebar.selectbox(
                f"Top 20 for {case_type}:", ranked_options
            )
            selected_index = ranked_options.index(selected_option)
            household_id = household_ids[selected_index]

            st.sidebar.info(f"Selected Household ID: {household_id}")
            return int(household_id)
        except Exception as e:
            logger.error(f"Error retrieving interesting cases: {str(e)}")
            st.error("Error retrieving households. Please try different filters.")
            st.stop()

    @staticmethod
    def _create_ranked_options(
        top_households: pd.DataFrame, case_type: str
    ) -> Tuple[List[str], List[int]]:
        """Create ranked options for interesting case selection."""
        ranked_options, household_ids = [], []

        for i, (_, row) in enumerate(top_households.iterrows(), 1):
            household_ids.append(row["Household ID"])

            if "%" in case_type:
                column = (
                    "Percentage Change in Federal Tax Liability"
                    if "Tax" in case_type
                    else "Percentage Change in Net Income"
                )
                value = row[column]
                ranked_options.append(f"#{i}: {value:+.1f}%")
            else:
                column = (
                    "Total Change in Federal Tax Liability"
                    if "Tax" in case_type
                    else "Total Change in Net Income"
                )
                value = row[column]
                ranked_options.append(f"#{i}: ${value:+,.0f}")

        return ranked_options, household_ids


class TaxAnalysisEngine:
    """Unified tax analysis engine that handles all analysis types."""

    def __init__(self, analysis_type: AnalysisType):
        """Initialize with specific analysis type."""
        self.analysis_type = analysis_type

    def get_reform_impacts(self, household_data: pd.Series) -> List[ReformImpact]:
        """
        Calculate reform impacts for the specific analysis type.

        Args:
            household_data: Household data series

        Returns:
            List[ReformImpact]: List of significant reform impacts
        """
        impacts = []
        column_prefix = self._get_column_prefix()

        for display_name, col_name in AppConfig.REFORM_COLS:
            try:
                change_value = household_data[f"{column_prefix} {col_name}"]
                impact = ReformImpact(name=display_name, total_change=change_value)

                if impact.is_significant:
                    impacts.append(impact)
            except KeyError:
                continue

        return impacts

    def get_chart_title(self) -> str:
        """Get the appropriate chart title for the analysis type."""
        titles = {
            AnalysisType.FEDERAL_TAXES: "Federal Taxes",
            AnalysisType.STATE_TAXES: "State Taxes",
            AnalysisType.NET_INCOME: "Net Income",
            AnalysisType.BENEFITS: "Benefits",
        }
        return titles[self.analysis_type]

    def get_baseline_info(
        self, profile: HouseholdProfile, household_data: pd.Series
    ) -> Tuple[float, str]:
        """
        Get baseline value and label for the analysis type.

        Returns:
            Tuple[float, str]: (baseline_value, baseline_label)
        """
        ## TODO: Make this mapping more consistent later
        mapping = {
            AnalysisType.FEDERAL_TAXES: (profile.baseline_federal_tax, "Federal Taxes"),
            AnalysisType.STATE_TAXES: (
                household_data["State Income Tax"],
                "State Taxes",
            ),
            AnalysisType.NET_INCOME: (profile.baseline_net_income, "Net Income"),
            AnalysisType.BENEFITS: (
                household_data["Baseline Total Benefits"],
                "Benefits",
            ),
        }
        return mapping[self.analysis_type]

    def get_change_info(
        self, household_data: pd.Series
    ) -> Tuple[float, float, str, str, float]:
        """
        Get change information for impact summary. Helps to calculate post-reform value, too.

        Returns:
            Tuple[float, float, str, str, float]: (change_value, pct_change, change_label, color, final_value)
        """
        mapping = {
            AnalysisType.FEDERAL_TAXES: (
                household_data["Total Change in Federal Tax Liability"],
                household_data["Percentage Change in Federal Tax Liability"],
                "Federal Tax Change",
                household_data["Baseline Federal Tax Liability"],
                "Reformed Federal Tax",
            ),
            AnalysisType.STATE_TAXES: (
                household_data["Total Change in State Tax Liability"],
                household_data["Percentage Change in State Tax Liability"],
                "State Tax Change",
                household_data.get("State Income Tax", 0),
                "Reformed State Tax",
            ),
            AnalysisType.NET_INCOME: (
                household_data["Total Change in Net Income"],
                household_data["Percentage Change in Net Income"],
                "Net Income Change",
                household_data["Baseline Net Income"],
                "Reformed Net Income",
            ),
            AnalysisType.BENEFITS: (
                household_data["Total Change in Benefits"],
                household_data["Total Change in Benefits"]
                / max(household_data["Baseline Total Benefits"], 1)
                * 100,
                "Benefits Change",
                household_data["Baseline Total Benefits"],
                "Reformed Benefits",
            ),
        }

        change_value, pct_change, change_label, baseline_value, final_label = mapping[
            self.analysis_type
        ]
        final_value = baseline_value + change_value

        # Determine color based on analysis type and value
        if abs(change_value) < 1:
            color = UIConfig.colors["BLACK"]
        elif self.analysis_type in [AnalysisType.NET_INCOME, AnalysisType.BENEFITS]:
            color = (
                UIConfig.colors["TEAL_PRESSED"]
                if change_value > 0
                else UIConfig.colors["GRAY"]
            )
        else:
            color = (
                UIConfig.colors["GRAY"]
                if change_value > 0
                else UIConfig.colors["TEAL_PRESSED"]
            )

        return change_value, pct_change, change_label, final_label, color, final_value

    def _get_column_prefix(self) -> str:
        """Get the column prefix for reform impact columns."""
        prefixes = {
            AnalysisType.FEDERAL_TAXES: "Change in Federal tax liability after",
            AnalysisType.STATE_TAXES: "Change in State tax liability after",
            AnalysisType.NET_INCOME: "Change in Net income after",
            AnalysisType.BENEFITS: "Change in Benefits after",
        }
        return prefixes[self.analysis_type]


class VisualizationRenderer:
    """Handles all UI rendering operations with consistent styling."""

    def __init__(
        self,
        analysis_engine: Optional[TaxAnalysisEngine],
        reform_type: str = "",
        baseline: str = "",
    ):
        """Initialize with analysis engine."""
        self.analysis_engine = analysis_engine
        self.reform_type = reform_type
        self.baseline = baseline

    def render_main_content(
        self, profile: HouseholdProfile, household_data: pd.Series
    ) -> None:
        """Render the main dashboard content in two columns."""
        col1, col2 = st.columns(2)

        with col1:
            self._render_household_attributes(profile, household_data)

        with col2:
            self._render_baseline_info(profile, household_data)
            self._render_impact_summary(household_data)

            # Add population weight in grey text
            weight = household_data["Household Weight"]
            st.markdown(
                f"<p style='color: {UIConfig.colors['GRAY']}; font-size: 14px; margin-top: 10px;'>"
                f"This household represents {math.ceil(weight):,} households in the US."
                f"</p>",
                unsafe_allow_html=True,
            )

        # Render analysis sections
        impacts = self.analysis_engine.get_reform_impacts(household_data)

        if impacts:
            self._render_waterfall_chart(impacts, household_data)
        else:
            st.info(
                f"The {self.reform_type} OBBB against {self.baseline.lower()} does not affect this household's {self.analysis_engine.analysis_type.value.lower()}."
            )

    def _render_household_attributes(
        self, profile: HouseholdProfile, household_data: pd.Series
    ) -> None:
        """Render household attributes in a styled container."""
        st.subheader("Household Attributes")

        content = self._build_household_attributes_content(profile, household_data)
        st.markdown(
            f"""
        <div style="{UIConfig.CONTAINER_STYLE}">
        {content}
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Add some space before the expander
        st.markdown("<br>", unsafe_allow_html=True)

        # Raw data expander outside the styled container
        with st.expander("Full Dataframe Row"):
            st.dataframe(household_data.to_frame().T, use_container_width=True)

    def _render_baseline_info(
        self, profile: HouseholdProfile, household_data: pd.Series
    ) -> None:
        """Render baseline tax/income information."""
        st.subheader("Reconciliation Bill Impact Summary")

        baseline_value, baseline_label = self.analysis_engine.get_baseline_info(
            profile, household_data
        )

        # Build content based on analysis type
        if self.analysis_engine.analysis_type == AnalysisType.BENEFITS:
            # For benefits analysis, show benefit components
            additional_content = self._build_benefit_baseline(household_data)
            # Also add tax information
            additional_content += self._build_additional_taxes_content(
                profile, household_data
            )
        else:
            # For other analysis types, show additional taxes
            additional_content = self._build_additional_taxes_content(
                profile, household_data
            )

        content = f"<p style='font-size: 18px; font-weight: bold; margin: 0; color: UIConfig.colors['TEAL_PRESSED'];'>{baseline_label}: ${baseline_value:,.0f}</p>"
        content += additional_content

        self._render_styled_container("Baseline Values", content)
        st.markdown("<br>", unsafe_allow_html=True)

    def _build_benefit_baseline(self, household_data: pd.Series) -> str:
        """Build content for benefit components breakdown."""
        # Define benefit columns to check
        benefit_columns = [
            "Baseline Benefits",
            "Baseline Medicaid",
            "Baseline ACA PTC",
            "Baseline CHIP",
        ]

        # Find benefit components with positive values
        benefit_components = [
            f"• {col.replace('Baseline ', 'Initial ')}: ${household_data[col]:,.0f}"
            for col in benefit_columns
            if col in household_data.index
            and pd.notna(household_data[col])
            and household_data[col] > 0
        ]

        if benefit_components:
            return "".join(
                f"<p style='margin: 2px 0 0 0; color: UIConfig.colors['TEAL_PRESSED'];'>{component}</p>"
                for component in benefit_components
            )
        return ""

    def _build_additional_taxes_content(
        self, profile: HouseholdProfile, household_data: pd.Series
    ) -> str:
        """Build content for additional taxes section."""
        state_tax = household_data.get("State Income Tax", 0)

        # Define additional taxes based on analysis type
        tax_mappings = {
            AnalysisType.FEDERAL_TAXES: [
                f"State Taxes: ${state_tax:,.0f}" if state_tax > 0 else None,
            ],
            AnalysisType.STATE_TAXES: [
                f"Federal Taxes: ${profile.baseline_federal_tax:,.0f}",
            ],
            AnalysisType.NET_INCOME: [
                f"Federal Taxes: ${profile.baseline_federal_tax:,.0f}",
                f"State Taxes: ${state_tax:,.0f}" if state_tax > 0 else None,
                f"Benefits: ${household_data['Baseline Benefits']:,.0f}",
            ],
            AnalysisType.BENEFITS: [
                f"Federal Taxes: ${profile.baseline_federal_tax:,.0f}",
                f"State Taxes: ${state_tax:,.0f}" if state_tax > 0 else None,
            ],
        }

        additional_taxes = [
            tax
            for tax in tax_mappings[self.analysis_engine.analysis_type]
            if tax is not None
        ]

        if additional_taxes:
            content = "<p style='margin: 10px 0 0 0; color: UIConfig.colors['TEAL_PRESSED'];'><strong>Additional Information:</strong></p>"
            content += "".join(
                f"<p style='margin: 2px 0 0 0; color: UIConfig.colors['TEAL_PRESSED'];'>• {tax}</p>"
                for tax in additional_taxes
            )
            return content
        return ""

    def _render_impact_summary(self, household_data: pd.Series) -> None:
        """Render impact summary with appropriate coloring."""
        change_value, pct_change, change_label, final_label, color, final_value = (
            self.analysis_engine.get_change_info(household_data)
        )

        # Override color for positive changes to use TEAL_ACCENT
        if change_value > 0:
            impact_color = UIConfig.colors["TEAL_PRESSED"]
        else:
            impact_color = UIConfig.colors["DARKEST_BLUE"]

        content = f"""
        <p style="color: {impact_color}; font-size: 18px; font-weight: bold;">
        {change_label}: ${change_value:,.0f} ({pct_change:+.1f}%)
        </p>
        <p style="font-size: 18px; font-weight: bold; margin-top: 10px; color: UIConfig.colors['TEAL_PRESSED'];">
        {final_label}: ${final_value:,.0f}
        </p>
        """

        self._render_styled_container("Overall Impact", content)

    def _render_waterfall_chart(
        self, impacts: List[ReformImpact], household_data: pd.Series
    ) -> None:
        """Render waterfall chart showing reform impacts."""
        chart_title = self.analysis_engine.get_chart_title()

        try:
            baseline_value, _ = self.analysis_engine.get_baseline_info(
                HouseholdProfile.from_series(household_data), household_data
            )

            change_value, _, _, _, _, _ = self.analysis_engine.get_change_info(
                household_data
            )

            # Prepare waterfall data
            waterfall_data = [
                (f"Baseline {chart_title}", baseline_value, baseline_value)
            ]
            running_total = baseline_value

            for impact in impacts:
                running_total += impact.total_change
                waterfall_data.append((impact.name, impact.total_change, running_total))

            final_value = baseline_value + change_value
            waterfall_data.append((f"Final {chart_title}", final_value, final_value))

            # Create and display chart
            fig = self._create_waterfall_figure(
                waterfall_data, chart_title, baseline_value, final_value
            )
            fig = format_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            logger.error(f"Error creating waterfall chart: {str(e)}")
            st.error(
                "Error creating waterfall chart. Please try a different household."
            )

    def _create_waterfall_figure(
        self,
        waterfall_data: List[Tuple],
        chart_title: str,
        baseline_value: float,
        final_value: float,
    ) -> go.Figure:
        """Create Plotly waterfall figure."""
        fig = go.Figure()

        # Determine colors based on analysis type, sorry for the redundancy! It was simplest like this.
        if self.analysis_engine.analysis_type in [
            AnalysisType.NET_INCOME,
            AnalysisType.BENEFITS,
        ]:
            # For net income: increases are good (teal), decreases are bad (dark gray)
            increasing_color = UIConfig.colors["TEAL_PRESSED"]
            decreasing_color = UIConfig.colors["GRAY"]
        else:
            increasing_color = UIConfig.colors["GRAY"]
            decreasing_color = UIConfig.colors["TEAL_PRESSED"]

        fig.add_trace(
            go.Waterfall(
                name=f"{chart_title} Impact",
                orientation="v",
                measure=["absolute"]
                + ["relative"] * (len(waterfall_data) - 2)
                + ["total"],
                x=[item[0] for item in waterfall_data],
                y=[item[1] for item in waterfall_data],
                text=[f"${item[1]:,.0f}" for item in waterfall_data],
                textposition="outside",
                textfont={"color": UIConfig.colors["GRAY"]},
                connector={"line": {"color": UIConfig.colors["DARKEST_BLUE"]}},
                increasing={"marker": {"color": increasing_color}},
                decreasing={"marker": {"color": decreasing_color}},
                totals={"marker": {"color": UIConfig.colors["BLUE_PRESSED"]}},
            )
        )

        # Create chart title with reform type and baseline
        if self.reform_type and self.baseline:
            chart_title_text = f"Impact of {self.reform_type} OBBB Against {self.baseline} by Provision on {self.analysis_engine.analysis_type.value}"
        else:
            chart_title_text = f"{chart_title} Changes"

        fig.update_layout(
            title={"text": chart_title_text, "font": {"size": 30}},
            yaxis_title=f"{chart_title} ($)",
            showlegend=False,
            height=AppConfig.CHART_HEIGHT,
            xaxis={"tickangle": -45},
            yaxis={
                "range": [
                    min(0, min([item[2] for item in waterfall_data]) * 1.15),
                    max(0, max([item[2] for item in waterfall_data]) * 1.15),
                ]
            },
        )

        return fig

    def render_analysis_info_card(self) -> None:
        """Render information card about current analysis scope."""
        st.markdown("---")

        # Format reform names list
        reform_names = [display_name for display_name, _ in AppConfig.REFORM_COLS]
        reforms_text = (
            ", ".join(reform_names[:-1]) + f", and {reform_names[-1]}"
            if len(reform_names) > 1
            else reform_names[0]
        )

        # Determine analysis focus
        focus_mapping = {
            AnalysisType.FEDERAL_TAXES: "Federal Taxes",
            AnalysisType.NET_INCOME: "Net Income overall",
            AnalysisType.STATE_TAXES: "State Taxes",
            AnalysisType.BENEFITS: "Benefits",
        }
        analysis_focus = focus_mapping[self.analysis_engine.analysis_type]

        income_sources = AppConfig.INCOME_SOURCES
        income_names = [display_name for display_name, _ in income_sources]
        income_list = ", ".join(income_names[:-1]) + f", and {income_names[-1]}"

        with st.expander("Analysis Scope", expanded=False):
            st.info(
                f"""
            We are currently analyzing the effects of {reforms_text} on {analysis_focus}. \n
            Currently, we are only displaying the {income_list} as featured income sources, with Employment Income including Tip and Overtime Income. 
            This is why these selected income sources will often not add up to the Gross Income total.
            """
            )

    def _render_styled_container(self, title: str, content: str) -> None:
        """Helper method to render consistently styled containers."""
        st.markdown(
            f"""
        <div style="{UIConfig.CONTAINER_STYLE}">
        <h4 style="color: UIConfig.colors['TEAL_PRESSED'];">{title}</h4>
        {content}
        </div>
        """,
            unsafe_allow_html=True,
        )

    def _get_dependent_ages(self, household_data: pd.Series) -> List[str]:
        """Extract dependent ages from household data."""
        dependent_ages = []
        for i in range(1, AppConfig.MAX_DEPENDENTS + 1):
            age_col = f"Age of Dependent {i}"
            if age_col in household_data.index:
                age = household_data[age_col]
                if pd.notna(age) and age > 0:
                    dependent_ages.append(f"{age:.0f}")
        return dependent_ages

    def _get_marital_info(
        self, profile: HouseholdProfile, household_data: pd.Series
    ) -> str:
        """Get formatted marital status information."""
        if profile.is_married and profile.age_of_spouse:
            return f"Married<br><strong>Spouse Age:</strong> {profile.age_of_spouse:.0f} years"
        else:
            return "Single"

    def _render_household_attributes_redesigned(
        self, profile: HouseholdProfile, household_data: pd.Series
    ) -> None:
        """Render household attributes in redesigned 2-column layout."""
        st.subheader("Household Attributes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic demographic info
            content1 = f"""
            <p style='color: var(--text-color, #000000);'><strong>State:</strong> {profile.state}</p>
            <p style='color: var(--text-color, #000000);'><strong>Household Size:</strong> {household_data['Household Size']:.0f}</p>
            <p style='color: var(--text-color, #000000);'><strong>Number of Tax Units:</strong> {household_data['Number of Tax Units']:.0f}</p>
            <p style='color: var(--text-color, #000000);'><strong>Head of Household Age:</strong> {profile.age_of_head:.0f} years</p>
            """
            
            # Add marital info
            content1 += f"<p style='color: var(--text-color, #000000);'><strong>Marital Status:</strong> {self._get_marital_info(profile, household_data)}</p>"
            
            st.markdown(content1, unsafe_allow_html=True)
        
        with col2:
            # Family and citizenship info
            content2 = f"""
            <p style='color: var(--text-color, #000000);'><strong>Number of Dependents:</strong> {profile.number_of_dependents:.0f}</p>
            """
            
            # Add children's ages if any
            if profile.number_of_dependents > 0:
                dependent_ages = self._get_dependent_ages(household_data)
                if dependent_ages:
                    content2 += f"<p style='color: var(--text-color, #000000);'><strong>Children's Ages:</strong> {', '.join(dependent_ages)} years</p>"
            
            content2 += f"""
            <p style='color: var(--text-color, #000000);'><strong>People with SSN Card (Citizen/EAD):</strong> {household_data['Num with SSN Card (Citizen/EAD)']:.0f}</p>
            <p style='color: var(--text-color, #000000);'><strong>People with SSN Card (Other/None):</strong> {household_data['Num with SSN Card (Other/None)']:.0f}</p>
            <p style='color: var(--text-color, #000000);'><strong>Market Income:</strong> ${household_data['Market Income']:,.0f}</p>
            """
            
            st.markdown(content2, unsafe_allow_html=True)
        
        # Add population weight below
        weight = household_data["Household Weight"]
        st.markdown(
            f"<p style='color: {UIConfig.colors['GRAY']}; font-size: 14px; margin-top: 10px; text-align: center;'>"
            f"This household represents {math.ceil(weight):,} households in the US."
            f"</p>",
            unsafe_allow_html=True,
        )
        
        # Raw data expander
        with st.expander("Full Dataframe Row"):
            st.dataframe(household_data.to_frame().T, use_container_width=True)

    def _render_impact_summary_table(
        self, profile: HouseholdProfile, household_data: pd.Series
    ) -> None:
        """Render the new 5x4 impact summary table."""
        st.markdown("---")
        st.subheader(f"Financial Impact Breakdown of {self.reform_type} OBBB Against {self.baseline}")
        
        # Prepare data for the table
        table_data = []
        
        # Market Income row
        market_income_baseline = household_data['Market Income']
        market_income_reform = market_income_baseline  # Market income doesn't change
        table_data.append([
            "Market Income",
            f"${market_income_baseline:,.0f}",
            f"${market_income_reform:,.0f}",
            "$0",
            "0.0%"
        ])
        
        # Federal Taxes row
        federal_tax_baseline = profile.baseline_federal_tax
        federal_tax_change = household_data["Total Change in Federal Tax Liability"]
        federal_tax_reform = federal_tax_baseline + federal_tax_change
        federal_tax_pct = household_data["Percentage Change in Federal Tax Liability"]
        table_data.append([
            "Federal Taxes",
            f"${federal_tax_baseline:,.0f}",
            f"${federal_tax_reform:,.0f}",
            f"${federal_tax_change:+,.0f}",
            f"{federal_tax_pct:+.1f}%"
        ])
        
        # State Taxes row
        state_tax_baseline = household_data.get("State Income Tax", 0)
        state_tax_change = household_data.get("Total Change in State Tax Liability", 0)
        state_tax_reform = state_tax_baseline + state_tax_change
        state_tax_pct = household_data.get("Percentage Change in State Tax Liability", 0)
        table_data.append([
            "State Taxes",
            f"${state_tax_baseline:,.0f}",
            f"${state_tax_reform:,.0f}",
            f"${state_tax_change:+,.0f}",
            f"{state_tax_pct:+.1f}%"
        ])
        
        # Benefits row
        benefits_baseline = household_data["Baseline Total Benefits"]
        benefits_change = household_data["Total Change in Benefits"]
        benefits_reform = benefits_baseline + benefits_change
        benefits_pct = household_data["Percentage Change in Benefits"]
        table_data.append([
            "Benefits",
            f"${benefits_baseline:,.0f}",
            f"${benefits_reform:,.0f}",
            f"${benefits_change:+,.0f}",
            f"{benefits_pct:+.1f}%"
        ])
        
        # Net Income row
        net_income_baseline = profile.baseline_net_income
        net_income_change = household_data["Total Change in Net Income"]
        net_income_reform = net_income_baseline + net_income_change
        net_income_pct = household_data["Percentage Change in Net Income"]
        table_data.append([
            "Net Income",
            f"${net_income_baseline:,.0f}",
            f"${net_income_reform:,.0f}",
            f"${net_income_change:+,.0f}",
            f"{net_income_pct:+.1f}%"
        ])
        
        # Create DataFrame for the table
        import pandas as pd
        df_table = pd.DataFrame(
            table_data,
            columns=["", "Baseline Value", "Reform Value", "Absolute Change", "% Change"]
        )
        
        # Style the table
        def style_table(x):
            styles = []
            for i, row in enumerate(x.itertuples()):
                row_styles = [''] * len(row[1:])  # Skip index
                
                # Color positive changes in green, negative in red for absolute and % change columns
                abs_change_val = row[4]  # Absolute Change column
                pct_change_val = row[5]  # % Change column
                
                if abs_change_val.startswith('+'):
                    if i in [3, 4]:  # Benefits and Net Income rows - positive is good
                        row_styles[3] = f'color: {UIConfig.colors["TEAL_PRESSED"]}; font-weight: bold'
                        row_styles[4] = f'color: {UIConfig.colors["TEAL_PRESSED"]}; font-weight: bold'
                    elif i in [1, 2]:  # Federal and State Taxes - positive is bad (more taxes)
                        row_styles[3] = f'color: {UIConfig.colors["GRAY"]}; font-weight: bold'
                        row_styles[4] = f'color: {UIConfig.colors["GRAY"]}; font-weight: bold'
                elif abs_change_val.startswith('-'):
                    if i in [3, 4]:  # Benefits and Net Income rows - negative is bad
                        row_styles[3] = f'color: {UIConfig.colors["GRAY"]}; font-weight: bold'
                        row_styles[4] = f'color: {UIConfig.colors["GRAY"]}; font-weight: bold'
                    elif i in [1, 2]:  # Federal and State Taxes - negative is good (less taxes)
                        row_styles[3] = f'color: {UIConfig.colors["TEAL_PRESSED"]}; font-weight: bold'
                        row_styles[4] = f'color: {UIConfig.colors["TEAL_PRESSED"]}; font-weight: bold'
                
                styles.append(row_styles)
            
            return pd.DataFrame(styles, index=x.index, columns=x.columns)
        
        # Display the styled table
        st.dataframe(
            df_table.style.apply(style_table, axis=None),
            use_container_width=True,
            hide_index=True
        )


class HouseholdDashboard:
    """Main dashboard application orchestrator."""

    def __init__(self):
        """Initialize dashboard with configuration and data loading."""
        self._configure_page()
        self.data_manager = DataManager()
        self.filter_manager = FilterManager(FilterConfig.default())

        # Add sidebar radio buttons for baseline and reform type selection
        st.sidebar.markdown("---")
        st.sidebar.subheader("Analysis Configuration")

        # Baseline selection
        self.baseline = st.sidebar.radio(
            "Choose Baseline:", ("Current Law", "Current Policy"), index=0
        )

        # Reform type selection
        self.reform_type = st.sidebar.radio(
            "Choose Reform Type:", ("House", "Senate"), index=0
        )

        # Load appropriate CSV file based on selections
        csv_filename = AppConfig.CSV_FILES.get((self.baseline, self.reform_type))
        if csv_filename and os.path.exists(csv_filename):
            self.df = self.data_manager.load_data(csv_filename)
        else:
            # Fallback to legacy files if new files don't exist
            if self.reform_type == "House":
                csv_filename = AppConfig.HOUSE_CSV_FILENAME
            else:
                csv_filename = AppConfig.SENATE_CSV_FILENAME
            self.df = self.data_manager.load_data(csv_filename)

            # Show warning if using fallback files
            st.warning(
                f"Using fallback file: {csv_filename}. The selected baseline ({self.baseline}) may not be available."
            )

    def _configure_page(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Reconciliation Bill Impact Dashboard", layout="wide"
        )

        # Add custom CSS for radio button styling
        st.markdown(
            """
        <style>
        .stRadio > div[role="radiogroup"] > label > div:first-child, .stCheckbox > div > label > div:first-child {
            background-color: #3378b2 !important;
        }
        .stRadio > div[role="radiogroup"] > label > div:first-child > div, .stCheckbox > div > label > div:first-child > div {
            border: 1px solid #3378b2 !important;
            background-color: white !important;
        }
        
        /* Button hover styling - only change text color, not background */
        .stButton > button:hover {
            border-color: #3378b2 !important;
            color: #3378b2 !important;
        }
        
        /* Button active/click styling - no color changes */
        .stButton > button:active {
            background-color: inherit !important;
            border-color: inherit !important;
            color: inherit !important;
        }
        
        /* Dropdown/selectbox hover styling */
        .stSelectbox > div > div:hover {
            border-color: #3378b2 !important;
        }
        
        /* Selectbox option hover - only change text color, not background */
        .stSelectbox > div > div > div > div:hover > div {
            color: #3378b2 !important;
        }
        
        /* Number input hover */
        .stNumberInput > div > div:hover {
            border-color: #3378b2 !important;
        }
        
        /* Text input hover */
        .stTextInput > div > div:hover {
            border-color: #3378b2 !important;
        }
        
        /* Dropdown button styling */
        .st-emotion-cache-p5msec {
            position: relative;
            display: flex;
            width: 100%;
            font-size: 14px;
            padding: 0.75rem 1rem;
            list-style-type: none;
            border-radius: 8px;
        }
        
        .st-emotion-cache-p5msec:hover {
            color: #3378b2 !important;
        }
        
        /* Checkbox styling */
        .stCheckbox > div > label > div:first-child > div {
            background-color: #3378b2 !important;
            border-color: #3378b2 !important;
        }
        
        .stCheckbox > div > label > div:first-child > div > div {
            background-color: white !important;
            border: 1px solid #3378b2 !important;
        }
        
        /* Specific checkbox class styling */
        .st-e9.st-bf.st-bg.st-ea.st-bd.st-b5.st-eb.st-ec.st-c6.st-cj.st-ck.st-cl.st-cm.st-cf.st-cg.st-ch.st-ci.st-ev.st-ew.st-ex.st-ey.st-au.st-av.st-ax.st-aw.st-eh.st-d9.st-bc.st-ez.st-az.st-ei.st-ej.st-ek.st-el.st-aq.st-ba.st-em.st-en {
            background-color: #3378b2 !important;
            border-color: #3378b2 !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    def run(self) -> None:
        """
        Run the main dashboard application.

        Orchestrates the entire dashboard flow: header, filters, household selection,
        analysis, and rendering.
        """
        try:
            self._render_header()

            # Apply filters and select household
            df_filtered = self.filter_manager.render_and_apply_filters(self.df)
            household_id = HouseholdSelector.select_household(df_filtered)
            household_data = self._get_household_data(df_filtered, household_id)
            profile = HouseholdProfile.from_series(household_data)

            # Change URL to include this household ID
            st.query_params.update({"ID": household_id})

            # Render main content in new layout
            renderer = VisualizationRenderer(
                None, self.reform_type, self.baseline  # Analysis engine will be set later
            )
            
            # Render household attributes in 2 columns
            renderer._render_household_attributes_redesigned(profile, household_data)
            
            # Render the new impact summary table
            renderer._render_impact_summary_table(profile, household_data)
            
            # Analysis type selector (moved from sidebar to main content)
            analysis_type = self._render_analysis_type_selector_main(household_data)
            
            # Set up analysis engine now that we have the analysis type
            analysis_engine = TaxAnalysisEngine(analysis_type)
            renderer.analysis_engine = analysis_engine

            # Render waterfall chart
            impacts = analysis_engine.get_reform_impacts(household_data)
            if impacts:
                renderer._render_waterfall_chart(impacts, household_data)
            else:
                st.info(
                    f"The {self.reform_type} OBBB against {self.baseline.lower()} does not affect this household's {analysis_type.value.lower()}."
                )

            # Add analysis info card (collapsible)
            renderer.render_analysis_info_card()

        except Exception as e:
            logger.error(f"Error running dashboard: {str(e)}")
            st.error(f"An error occurred: {str(e)}")

    def _render_header(self) -> None:
        """Render dashboard header and title."""
        st.title("OBBBA Household Explorer")
        st.markdown(
            f"*Explore how the reconciliation bill affects individual households across America*"
        )
        st.sidebar.header("Select Household")

    def _render_analysis_type_selector_main(self, household_data: pd.Series) -> AnalysisType:
        """Render analysis type selector in main content with percent changes."""
        st.markdown("---")

        type_mapping = {
            analysis_type.value: analysis_type for analysis_type in AnalysisType
        }

        options = []
        for display_value, enum_type in type_mapping.items():
            # Create temporary analysis engine for this type
            temp_engine = TaxAnalysisEngine(enum_type)
            _, pct, *_ = temp_engine.get_change_info(household_data)
            label = f"{display_value} ({pct:+.1f}%)"
            options.append(label)

        selected = st.radio("Select Impact Breakdown:", options, index=0, horizontal=True)
        selected_type = type_mapping[selected.split(" (")[0]]
        st.session_state.analysis_type = selected_type
        return selected_type

    def _get_household_data(
        self, df_filtered: pd.DataFrame, household_id: int
    ) -> pd.Series:
        """
        Get household data series for the selected household ID.

        Args:
            df_filtered: Filtered dataframe
            household_id: Selected household ID

        Returns:
            pd.Series: Household data series

        Raises:
            IndexError: If household ID not found
        """
        try:
            return df_filtered[df_filtered["Household ID"] == household_id].iloc[0]
        except IndexError:
            st.error(
                f"Household ID {household_id} not found. Please try different filters."
            )
            st.stop()


def main() -> None:
    """
    Application entry point.

    Initializes and runs the household dashboard with comprehensive error handling.
    """
    try:
        dashboard = HouseholdDashboard()
        dashboard.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        st.error("A fatal error occurred. Please refresh the page and try again.")


if __name__ == "__main__":
    main()
