"""
Feature Analysis Component for Feature Selection.

This component handles feature importance analysis, correlation analysis,
protected attributes detection, and related visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from components.feature_selection.utils.visualization_utils import (
    create_feature_importance_plot,
    create_low_importance_plot,
    create_correlation_network_plot,
    get_feature_importance_stats
)
from components.feature_selection.utils.correlation_utils import (
    build_correlation_groups,
    create_correlation_display_data,
    CORRELATION_DETECTION_THRESHOLD,
    CORRELATION_STRONG_THRESHOLD,
    CORRELATION_VERY_STRONG_THRESHOLD
)


class FeatureAnalysisComponent:
    """Handles feature analysis and visualization for the feature selection process."""

    def __init__(self, builder, logger=None):
        """
        Initialize the feature analysis component.

        Args:
            builder: Builder instance containing the dataset
            logger: Optional logger instance for tracking
        """
        self.builder = builder
        self.logger = logger
        self.analysis_result = None

    def render(self) -> bool:
        """
        Render the feature analysis section.

        Returns:
            bool: True if analysis was successful, False otherwise
        """
        # Run feature analysis
        self.analysis_result = self.builder.analyse_feature_importance()

        if not self.analysis_result["success"]:
            self._render_analysis_error()
            return False

        # Log successful analysis
        self._log_analysis_results()

        # Display analysis results
        st.write("## 📊 Feature Analysis Results")
        self._render_analysis_tabs()

        return True

    def _render_analysis_error(self) -> None:
        """Render error message when analysis fails."""
        st.error(self.analysis_result["message"])
        if self.logger:
            self.logger.log_error(
                "Feature Analysis Failed",
                {"error": self.analysis_result["message"]}
            )
        st.error("Unable to analyse feature importance. Please check your data and try again.")

    def _log_analysis_results(self) -> None:
        """Log the results of feature analysis."""
        if not self.logger:
            return

        result = self.analysis_result
        self.logger.log_calculation(
            "Feature Analysis",
            {
                "total_features": len(result["feature_scores"]),
                "feature_importance": {
                    feat["feature"]: feat["importance"]
                    for feat in result["feature_scores"][:5]  # Log top 5 features
                },
                "analysis_metrics": {
                    "low_importance_count": len(result["responsible_ai"]["low_importance_features"]),
                    "correlation_pairs": len(result["responsible_ai"]["correlations"]),
                    "quality_issues": len(result["responsible_ai"]["quality_issues"]),
                    "protected_attributes": len(result["responsible_ai"]["protected_attributes"])
                }
            }
        )

        # Log correlations and quality issues
        self.logger.log_calculation(
            "Feature Analysis Details",
            {
                "correlations": result["responsible_ai"]["correlations"],
                "quality_issues": result["responsible_ai"]["quality_issues"],
                "protected_attributes": result["responsible_ai"]["protected_attributes"]
            }
        )

    def _render_analysis_tabs(self) -> None:
        """Render the analysis results in organized tabs."""
        # Get shared variables
        feature_scores = pd.DataFrame(self.analysis_result["feature_scores"])
        protected_attributes = self.analysis_result["responsible_ai"]["protected_attributes"]
        correlations = self.analysis_result["responsible_ai"]["correlations"]

        # Calculate low importance features
        low_importance_features = [
            feat for feat, score in zip(feature_scores['feature'], feature_scores['importance'])
            if score <= 0.01
        ]

        # Create tabs for different analysis sections
        impact_tab, correlation_tab, protected_tab = st.tabs([
            "1. Low Impact Features",
            "2. Correlated Features",
            "3. Protected Attributes"
        ])

        with impact_tab:
            self._render_importance_analysis(feature_scores, low_importance_features)

        with correlation_tab:
            self._render_correlation_analysis(correlations)

        with protected_tab:
            self._render_protected_attributes_analysis(protected_attributes)

    def _render_importance_analysis(self, feature_scores: pd.DataFrame, low_importance_features: List[str]) -> None:
        """Render feature importance analysis."""
        st.write("### 📈 Feature Importance Scores")

        # Add explanation expander
        with st.expander("ℹ️ Understanding Feature Importance", expanded=False):
            explanation = self.builder.get_calculation_explanation("feature_importance")
            st.write("**Method:**")
            st.markdown(explanation["method"])
            st.write("**How to Interpret Results:**")
            st.markdown(explanation["interpretation"])
            st.markdown("""
            🎯 **Quick Guide:**
            - **High Importance (> 0.1)**: Strong predictive power
            - **Medium Importance (0.05 - 0.1)**: Moderate influence
            - **Low Importance (< 0.05)**: Weak or negligible impact

            💡 **Tips:**
            - Consider removing features with very low importance
            - Keep features that domain experts consider important
            - Balance statistical importance with business context
            """)

        # Create and display feature importance plot
        importance_fig = create_feature_importance_plot(feature_scores)
        st.plotly_chart(importance_fig, config={'responsive': True}, key="feature_importance_plot")

        # Display feature statistics
        self._render_importance_statistics(feature_scores)

        # Display low importance features section
        self._render_low_importance_section(feature_scores, low_importance_features)

    def _render_importance_statistics(self, feature_scores: pd.DataFrame) -> None:
        """Render feature importance statistics."""
        stats = get_feature_importance_stats(feature_scores)

        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        with stats_col1:
            st.metric("Total Features", stats["total_features"])
        with stats_col2:
            percentage = (stats["high_importance"] / stats["total_features"] * 100) if stats["total_features"] > 0 else 0
            st.metric("High Importance", stats["high_importance"], f"{percentage:.1f}%")
        with stats_col3:
            percentage = (stats["medium_importance"] / stats["total_features"] * 100) if stats["total_features"] > 0 else 0
            st.metric("Medium Importance", stats["medium_importance"], f"{percentage:.1f}%")
        with stats_col4:
            percentage = (stats["low_importance"] / stats["total_features"] * 100) if stats["total_features"] > 0 else 0
            st.metric("Low Importance", stats["low_importance"], f"{percentage:.1f}%")

        # Log importance distribution
        if self.logger:
            self.logger.log_calculation("Feature Importance Distribution", stats)

    def _render_low_importance_section(self, feature_scores: pd.DataFrame, low_importance_features: List[str]) -> None:
        """Render the low importance features section."""
        st.write("### ⚠️ Low Impact Features")

        with st.expander("ℹ️ Understanding Low Impact Features", expanded=False):
            st.markdown("""
            Low impact features contribute minimally to model predictions.

            **Why they matter:**
            - Model complexity
            - Processing overhead
            - Potential noise source

            **What to do:**
            1. Review for business relevance
            2. Consider removing
            3. Document decisions
            4. Monitor impact
            """)

        if low_importance_features:
            # Create visualization of low importance features
            low_imp_scores = feature_scores[
                feature_scores['feature'].isin(low_importance_features)
            ]

            low_imp_fig = create_low_importance_plot(low_imp_scores)
            st.plotly_chart(low_imp_fig, config={'responsive': True}, key="low_importance_plot")

            st.warning(f"Found {len(low_importance_features)} features with low predictive power:")
            for feat in low_importance_features:
                st.write(f"• {feat}")
        else:
            st.success("No low importance features detected")

    def _render_correlation_analysis(self, correlations: List[Dict[str, Any]]) -> None:
        """Render correlation analysis section."""
        st.write("### 🔗 Correlated Features")

        with st.expander("ℹ️ Understanding Correlations", expanded=False):
            st.markdown(f"""
            Feature correlations indicate how strongly pairs of features are related.

            **Interpretation:**
            - **Very Strong (≥ {CORRELATION_VERY_STRONG_THRESHOLD})**: Extremely high correlation, strongly consider removing one
            - **Strong (≥ {CORRELATION_STRONG_THRESHOLD})**: High correlation, consider removing one
            - **Moderate ({CORRELATION_DETECTION_THRESHOLD}-{CORRELATION_STRONG_THRESHOLD})**: Notable relationship, monitor
            - **Weak (< {CORRELATION_DETECTION_THRESHOLD})**: Limited relationship

            **Why it matters:**
            - Reduces redundancy
            - Improves model stability
            - Prevents multicollinearity

            **What to do:**
            1. Review highly correlated pairs
            2. Keep the more meaningful feature
            3. Document your decisions
            """)

        if correlations:
            # Create correlation network visualization
            st.write("#### 🔗 Feature Correlation Network")
            corr_fig = create_correlation_network_plot(correlations)
            st.plotly_chart(corr_fig, config={'responsive': True}, key="correlation_network_plot")

            # Display correlation details
            self._render_correlation_details(correlations)

            st.warning("Features with high correlation:")
            for corr in correlations:
                st.write(f"• {corr['feature1']} & {corr['feature2']} ({corr['correlation']:.2f})")
        else:
            st.success("✅ No highly correlated features detected")
            st.info(f"All features have low correlation with each other (< {CORRELATION_DETECTION_THRESHOLD})")

    def _render_correlation_details(self, correlations: List[Dict[str, Any]]) -> None:
        """Render detailed correlation information."""
        # Build correlation groups
        correlation_groups = build_correlation_groups(correlations)

        # Create correlation display data
        if self.builder.X_train is not None and len(self.builder.X_train.columns) > 1:
            feature_corr_matrix = self.builder.X_train.corr().abs()
            corr_display_data = create_correlation_display_data(correlation_groups, feature_corr_matrix)

            if corr_display_data:
                st.write("#### 📊 Correlation Values Within Groups")
                with st.expander("ℹ️ Understanding Group Correlations", expanded=False):
                    st.markdown(f"""
                    This section shows the exact correlation values between each pair of features within correlation groups.

                    **How to Read:**
                    - Values range from 0.0 to 1.0 (absolute correlation)
                    - Higher values indicate stronger relationships
                    - Values ≥ {CORRELATION_DETECTION_THRESHOLD} triggered group formation
                    - Diagonal values (feature with itself) are always 1.0

                    **Color Coding:**
                    - 🔴 Very Strong (≥ {CORRELATION_VERY_STRONG_THRESHOLD})
                    - 🟠 Strong (≥ {CORRELATION_STRONG_THRESHOLD})
                    - 🟡 Moderate (≥ {CORRELATION_DETECTION_THRESHOLD})
                    - 🟢 Weaker (< {CORRELATION_DETECTION_THRESHOLD})
                    """)

                # Group correlation data by group
                grouped_data = {}
                for item in corr_display_data:
                    group_num = item["Group"]
                    if group_num not in grouped_data:
                        grouped_data[group_num] = []
                    grouped_data[group_num].append(item)

                # Display each group
                for group_num in sorted(grouped_data.keys()):
                    group_data = grouped_data[group_num]
                    st.write(f"**Group {group_num}**")

                    # Create display dataframe for this group
                    group_df = pd.DataFrame([{
                        "Feature Pair": item["Feature Pair"],
                        "Correlation": item["Correlation"],
                        "Strength": item["Strength"]
                    } for item in group_data])

                    if len(group_df) > 0:
                        st.table(group_df)
                    else:
                        st.info("Only one feature in this group")

    def _render_protected_attributes_analysis(self, protected_attributes: List[str]) -> None:
        """Render protected attributes analysis."""
        st.write("### 🚨 Protected Attributes")

        with st.expander("ℹ️ Understanding Protected Attributes", expanded=False):
            st.markdown("""
            Protected attributes are features that could lead to discriminatory model behavior.

            **Examples include:**
            - Race, ethnicity
            - Gender, sexual orientation
            - Age, disability status
            - Religion, national origin

            **Why it matters:**
            - Legal compliance
            - Ethical AI practices
            - Fair decision-making

            **What to do:**
            1. Review identified attributes
            2. Consider legal requirements
            3. Document decisions
            4. Monitor for bias
            """)

        if protected_attributes:
            st.warning(
                "These features might introduce bias. Consider carefully whether "
                "they should be included in your model:"
            )
            for attr in protected_attributes:
                st.write(f"• {attr}")
            st.info("💡 Tip: Review these features for potential bias and legal/ethical implications")
        else:
            st.success("No protected attributes detected")

    def get_analysis_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get a summary of the analysis results.

        Returns:
            Dictionary with analysis summary or None if analysis not run
        """
        if not self.analysis_result or not self.analysis_result["success"]:
            return None

        feature_scores = pd.DataFrame(self.analysis_result["feature_scores"])
        stats = get_feature_importance_stats(feature_scores)

        return {
            "total_features": len(feature_scores),
            "importance_stats": stats,
            "correlations_found": len(self.analysis_result["responsible_ai"]["correlations"]),
            "protected_attributes_found": len(self.analysis_result["responsible_ai"]["protected_attributes"]),
            "low_importance_features": len([
                feat for feat, score in zip(feature_scores['feature'], feature_scores['importance'])
                if score <= 0.01
            ]),
            "analysis_successful": True
        }

    def get_feature_scores(self) -> Optional[pd.DataFrame]:
        """
        Get the feature scores DataFrame.

        Returns:
            DataFrame with feature scores or None if analysis not run
        """
        if not self.analysis_result or not self.analysis_result["success"]:
            return None

        return pd.DataFrame(self.analysis_result["feature_scores"])

    def get_correlations(self) -> List[Dict[str, Any]]:
        """
        Get the correlations list.

        Returns:
            List of correlation dictionaries
        """
        if not self.analysis_result or not self.analysis_result["success"]:
            return []

        return self.analysis_result["responsible_ai"]["correlations"]

    def get_protected_attributes(self) -> List[str]:
        """
        Get the protected attributes list.

        Returns:
            List of protected attribute names
        """
        if not self.analysis_result or not self.analysis_result["success"]:
            return []

        return self.analysis_result["responsible_ai"]["protected_attributes"]