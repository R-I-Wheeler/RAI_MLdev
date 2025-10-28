"""
Manual Selection Component for Feature Selection.

This component handles manual feature selection strategies, including
low importance removal, correlation-based selection, and custom selection.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional
from components.feature_selection.utils.selection_strategies import (
    get_available_selection_strategies,
    create_features_by_category,
    create_tiered_feature_analysis
)
from components.feature_selection.utils.correlation_utils import (
    build_correlation_groups,
    create_correlation_analysis_data,
    validate_correlation_group_selection,
    analyze_transitive_impact,
    create_network_positions,
    CORRELATION_DETECTION_THRESHOLD,
    CORRELATION_STRONG_THRESHOLD,
    CORRELATION_VERY_STRONG_THRESHOLD
)
from components.feature_selection.utils.visualization_utils import (
    create_correlation_network_plot,
    create_selection_summary_table,
    create_dataset_metrics_display
)
from components.feature_selection.utils.data_processing_utils import (
    check_and_remove_duplicates,
    clean_missing_values,
    synchronize_data_splits
)


class ManualSelectionComponent:
    """Handles manual feature selection interfaces and strategies."""

    def __init__(self, builder, logger=None, state_manager=None):
        """
        Initialize the manual selection component.

        Args:
            builder: Builder instance containing the dataset
            logger: Optional logger instance for tracking
            state_manager: Optional state manager for session state operations
        """
        self.builder = builder
        self.logger = logger
        self.state_manager = state_manager
        self.feature_scores = None
        self.correlations = None
        self.protected_attributes = None

    def set_analysis_data(self, feature_scores: pd.DataFrame,
                         correlations: List[Dict[str, Any]],
                         protected_attributes: List[str]) -> None:
        """
        Set the analysis data from the feature analysis component.

        Args:
            feature_scores: DataFrame with feature importance scores
            correlations: List of correlation dictionaries
            protected_attributes: List of protected attribute names
        """
        self.feature_scores = feature_scores
        self.correlations = correlations
        self.protected_attributes = protected_attributes

    def render(self) -> None:
        """Render the manual selection interface."""
        if self.feature_scores is None:
            st.error("Feature analysis data not available. Please run feature analysis first.")
            return

        # Render reset button
        self._render_reset_button()

        # Show current dataset statistics
        self._render_dataset_statistics()

        st.markdown("---")

        # Render feature categories for reference
        self._render_feature_categories()

    def _render_reset_button(self) -> None:
        """Render the reset to original features button."""
        if st.button("↩️ Reset to Original Features", type="primary", key="feature_reset_button"):
            if self.state_manager:
                reset_result = self.state_manager.reset_to_original_features()
                if reset_result["success"]:
                    st.success("✅ Features reset to original state!")
                    st.rerun()
                else:
                    st.error(f"Reset failed: {reset_result['message']}")
            else:
                st.error("State manager not available for reset operation.")

    def _render_dataset_statistics(self) -> None:
        """Render current dataset statistics."""
        st.write("### Current Dataset Statistics")
        metrics = create_dataset_metrics_display(self.builder)

        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.metric(
                "Total Features",
                metrics["total_features"],
                help="Current number of features in your dataset"
            )
        with stats_col2:
            st.metric(
                "Numerical Features",
                metrics["numerical_features"],
                f"{metrics['numerical_percentage']:.1f}%",
                help="Features containing numeric values"
            )
        with stats_col3:
            st.metric(
                "Categorical Features",
                metrics["categorical_features"],
                f"{metrics['categorical_percentage']:.1f}%",
                help="Features containing categorical values"
            )

    def _render_feature_categories(self) -> None:
        """Render feature categories for reference."""
        # Calculate low importance features
        low_importance_features = [
            feat for feat, score in zip(self.feature_scores['feature'], self.feature_scores['importance'])
            if score <= 0.01
        ]

        # Create features categorization
        features_by_category = create_features_by_category(
            self.protected_attributes,
            low_importance_features,
            self.correlations
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.write("### Feature Categories")
            for category, info in features_by_category.items():
                if info["features"]:
                    with st.expander(f"{info['icon']} {category}"):
                        st.markdown(f"_{info['description']}_")
                        for feat in info["features"]:
                            st.markdown(
                                f'<div style="padding: 5px; border-radius: 5px; '
                                f'border: 1px solid {info["color"]};">'
                                f'• {feat}</div>',
                                unsafe_allow_html=True
                            )

        with col2:
            st.write("### Manual Feature Selection")
            self._render_selection_interface(features_by_category)

    def _render_selection_interface(self, features_by_category: Dict[str, Dict[str, Any]]) -> None:
        """Render the main selection interface."""
        # Get available strategies
        available_strategies = get_available_selection_strategies(
            self.feature_scores, self.correlations
        )

        # Add selection guide
        with st.expander("📚 Feature Selection Guide", expanded=False):
            st.markdown("""
            ### How to Select Features

            1. **Review Analysis Results**
               - Check feature importance scores
               - Look for correlated features
               - Consider data quality issues
               - Review protected attributes

            2. **Selection Strategy**
               - Remove low importance features first
               - Choose one from highly correlated pairs
               - Address quality issues
               - Consider domain knowledge

            3. **Best Practices**
               - Start conservatively
               - Document your decisions
               - Consider business impact
               - Monitor performance changes
            """)

        # Selection strategy pills
        default_selection = st.session_state.get('previous_selection_strategy', available_strategies[0])
        if default_selection not in available_strategies:
            default_selection = available_strategies[0]

        selection_strategy = st.pills(
            "Selection Strategy",
            options=available_strategies,
            default=default_selection,
            help="Choose how you want to select features for removal",
            selection_mode="single"
        )

        # Store the current selection for next time
        st.session_state.previous_selection_strategy = selection_strategy

        # Log selection strategy change
        if self.logger:
            self.logger.log_user_action(
                "Selection Strategy Changed",
                {"strategy": selection_strategy}
            )

        # Render strategy-specific interface
        selected_features = self._render_strategy_interface(selection_strategy, features_by_category)

        # Handle feature application
        if selected_features:
            self._render_selection_summary_and_apply(selected_features, selection_strategy)

    def _render_strategy_interface(self, strategy: str, features_by_category: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Render the interface for the selected strategy.

        Args:
            strategy: Selected strategy name
            features_by_category: Dictionary categorizing features

        Returns:
            List of selected features for removal
        """
        if strategy == "Manual Selection":
            return st.multiselect(
                "Choose features to exclude:",
                options=list(self.builder.X_train.columns),
                help="Select features you want to remove from your model"
            )

        elif strategy == "Remove Low Importance":
            return self._render_low_importance_strategy()

        elif strategy == "Remove One from Correlated Groups":
            return self._render_correlation_strategy()

        return []

    def _render_low_importance_strategy(self) -> List[str]:
        """Render the enhanced low importance selection strategy."""
        with st.expander("📚 Understanding Enhanced Low Importance Selection", expanded=False):
            st.markdown("""
            ### Enhanced Low Importance Feature Selection with Tiered Thresholds

            This method uses **multiple threshold tiers** for more intelligent feature removal:

            #### 🚨 **Tier 1: Automatic Removal (Critical)**
            - **Importance < 0.001**: Essentially zero predictive power
            - **Total Correlation < 0.1**: Minimal relationship with any features
            - These features provide virtually no information and are automatically flagged

            #### ⚠️ **Tier 2: Enhanced Analysis (High Concern)**
            - **Importance 0.001-0.01**: Very low predictive power
            - Applies correlation analysis to determine redundancy
            - **Criteria**: Low importance AND above-average correlation

            #### 💭 **Tier 3: User Choice (Moderate Concern)**
            - **Importance 0.01-0.05**: Low but potentially useful
            - User decides based on domain knowledge

            #### 4. **Selection Logic**
            1. **Critical features** (< 0.001 importance OR < 0.1 total correlation): Automatic removal recommendation
            2. **High concern features**: Enhanced analysis with correlation
            3. **Moderate concern features**: User choice with warnings

            #### 5. **Why Tiered Thresholds Matter**
            - **Eliminates noise**: Removes truly useless features automatically
            - **Preserves potential**: Careful analysis for borderline cases
            - **Prevents over-removal**: Domain knowledge input for moderate cases
            - **More efficient**: Focuses detailed analysis where it matters most
            """)

        # Create tiered feature analysis
        feature_analysis_data = create_tiered_feature_analysis(self.builder.X_train, self.feature_scores)

        if feature_analysis_data:
            # Display summary metrics
            critical_count = len([f for f in feature_analysis_data if f["Category"] == "🚨 Critical"])
            high_concern_count = len([f for f in feature_analysis_data if f["Category"] == "⚠️ High Concern"])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🚨 Critical Features", critical_count)
            with col2:
                st.metric("⚠️ High Concern Features", high_concern_count)
            with col3:
                st.metric("📋 Total Flagged", len(feature_analysis_data))

            st.write("#### 📋 Interactive Feature Selection")
            st.info("Critical features are pre-selected for removal. Uncheck to keep them (not recommended). Check high concern features to remove them.")

            # Create interactive data editor
            edited_df = st.data_editor(
                pd.DataFrame(feature_analysis_data),
                column_config={
                    "Remove": st.column_config.CheckboxColumn(
                        "Remove",
                        help="Select features to remove from the dataset",
                        default=False,
                    ),
                    "Feature": st.column_config.TextColumn(
                        "Feature Name",
                        help="Name of the feature",
                        disabled=True,
                    ),
                    "Category": st.column_config.TextColumn(
                        "Category",
                        help="Feature risk category",
                        disabled=True,
                    ),
                    "Importance": st.column_config.TextColumn(
                        "Importance Score",
                        help="Feature importance score",
                        disabled=True,
                    ),
                    "Reason": st.column_config.TextColumn(
                        "Analysis",
                        help="Why this feature was flagged",
                        disabled=True,
                    ),
                },
                hide_index=True,
                width='stretch',
                key="feature_selection_table"
            )

            # Extract selected features
            selected_features = edited_df[edited_df["Remove"] == True]["Feature"].tolist()

            # Show selection summary
            if selected_features:
                critical_selected = [f for f in selected_features if any(
                    item["Feature"] == f and item["Category"] == "🚨 Critical"
                    for item in feature_analysis_data
                )]
                high_concern_selected = [f for f in selected_features if any(
                    item["Feature"] == f and item["Category"] == "⚠️ High Concern"
                    for item in feature_analysis_data
                )]

                st.write("#### 📊 Selection Summary")
                col1, col2 = st.columns(2)
                with col1:
                    if critical_selected:
                        st.success(f"✅ Removing {len(critical_selected)} critical features")
                    else:
                        st.warning("⚠️ No critical features selected (not recommended)")
                with col2:
                    if high_concern_selected:
                        st.info(f"ℹ️ Removing {len(high_concern_selected)} high concern features")
                    else:
                        st.success("✅ Keeping all high concern features")
            else:
                st.warning("⚠️ No features selected for removal")

            return selected_features
        else:
            st.success("✅ No problematic features detected!")
            st.info("All features have sufficient importance (> 0.01) and good correlation relationships.")
            return []

    def _render_correlation_strategy(self) -> List[str]:
        """Render the correlation-based selection strategy."""
        if not self.correlations:
            st.info("No correlated features found.")
            return []

        with st.expander("📚 Understanding Correlation Groups and Feature Selection", expanded=False):
            st.markdown(f"""
            ### How Correlation Groups are Formed

            1. **Initial Detection**:
               - Features with correlation > {CORRELATION_DETECTION_THRESHOLD} are identified as highly correlated pairs
               - Example: If A↔B = 0.75 and B↔C = 0.82, they form a group {{A,B,C}}

            2. **Advanced Group Merging** (Union-Find Algorithm):
               - Groups sharing any features are efficiently merged using Union-Find
               - Example: Groups {{A,B,C}} and {{C,D,E}} become {{A,B,C,D,E}}
               - Guarantees all transitively connected features are grouped correctly
               - Handles complex correlation graphs reliably

            ### Why Remove Features?

            1. **Multicollinearity Problems**:
               - Causes unstable model coefficients
               - Makes feature importance unreliable
               - Can lead to overfitting

            2. **Impact on Models**:
               - Linear models become unstable
               - Feature importance becomes misleading
               - Model interpretability decreases

            ### Enhanced Selection Algorithm

            For each correlation group, we calculate:

            1. **Weighted Correlation Score** (NEW):
               - Stronger correlations contribute more weight
               - Very strong (>{CORRELATION_VERY_STRONG_THRESHOLD}): 3x weight
               - Strong (>{CORRELATION_STRONG_THRESHOLD}): 2x weight
               - Moderate (>{CORRELATION_DETECTION_THRESHOLD}): 1x weight
               - Prioritizes breaking the strongest correlations

            2. **Feature Importance**:
               - Contribution to target prediction
               - Higher score = more valuable for prediction

            3. **High Correlation Count**:
               - Number of strong correlations (>{CORRELATION_STRONG_THRESHOLD})
               - Higher count = more redundant relationships

            4. **Transitive Impact Analysis** (NEW):
               - Detects "bridge" features that connect others
               - Warns if removing features may hide indirect relationships
               - Helps preserve important correlation chains

            ### Enhanced Selection Criteria

            Features are recommended for removal based on:
            1. **Primary**: High weighted correlation score AND low importance
            2. **Secondary**: Very high total correlation AND below-average importance
            3. **Fallback**: Highest weighted correlation in group

            ### 💡 Best Practices

            - Review feature importance before removing
            - Consider bridge feature warnings
            - Use domain knowledge for final decisions
            - Document removal decisions
            - At least one feature per group MUST be removed
            """)

        st.write("#### 📊 Correlation Analysis")
        st.info("Each correlation group needs at least one feature removed to address multicollinearity. Recommended features are pre-selected based on correlation and importance analysis.")

        # Create correlation network visualization
        st.write("#### 🔗 Feature Correlation Network")
        corr_fig = create_correlation_network_plot(self.correlations)
        st.plotly_chart(corr_fig, config={'responsive': True}, key="manual_correlation_network_plot")

        # Build correlation groups and analysis
        correlation_groups = build_correlation_groups(self.correlations)
        feature_corr_matrix = self.builder.X_train.corr().abs()
        correlation_analysis_data = create_correlation_analysis_data(
            correlation_groups, self.feature_scores, feature_corr_matrix
        )

        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔗 Correlation Groups", len(correlation_groups))
        with col2:
            st.metric("📋 Correlated Features", len(correlation_analysis_data))
        with col3:
            recommended_count = len([item for item in correlation_analysis_data if item["Remove"]])
            st.metric("🔴 Recommended Removals", recommended_count)

        # Interactive selection table
        st.write("#### 📋 Interactive Correlation Group Selection")
        st.info("Recommended features are pre-selected for removal. Each group needs at least one feature removed. Uncheck recommended features or check additional ones as needed.")

        edited_corr_df = st.data_editor(
            pd.DataFrame(correlation_analysis_data),
            column_config={
                "Remove": st.column_config.CheckboxColumn(
                    "Remove",
                    help="Select features to remove from the dataset",
                    default=False,
                ),
                "Feature": st.column_config.TextColumn(
                    "Feature Name",
                    help="Name of the feature",
                    disabled=True,
                ),
                "Group": st.column_config.TextColumn(
                    "Correlation Group",
                    help="Which correlation group this feature belongs to",
                    disabled=True,
                ),
                "Category": st.column_config.TextColumn(
                    "Recommendation",
                    help="Whether removal is recommended",
                    disabled=True,
                ),
                "Importance": st.column_config.TextColumn(
                    "Importance",
                    help="Feature importance score",
                    disabled=True,
                ),
                "Total Correlation": st.column_config.TextColumn(
                    "Total Correlation",
                    help="Sum of correlations with all features",
                    disabled=True,
                ),
                "High Corr Count": st.column_config.NumberColumn(
                    "High Corr Count",
                    help=f"Number of strong correlations (>{CORRELATION_STRONG_THRESHOLD})",
                    disabled=True,
                ),
                "Analysis": st.column_config.TextColumn(
                    "Analysis",
                    help="Detailed analysis of why this feature was categorized",
                    disabled=True,
                ),
            },
            hide_index=True,
            width='stretch',
            key="correlation_selection_table"
        )

        # Extract selected features
        selected_features = edited_corr_df[edited_corr_df["Remove"] == True]["Feature"].tolist()

        # Validate selection
        validation_result = validate_correlation_group_selection(correlation_groups, selected_features)
        self._render_correlation_validation(validation_result)

        # Perform transitive impact analysis if features are selected
        if selected_features:
            impact_analysis = analyze_transitive_impact(
                selected_features, correlation_groups, feature_corr_matrix
            )
            self._render_transitive_impact_analysis(impact_analysis)

        return selected_features

    def _render_correlation_validation(self, validation_result: Dict[str, Any]) -> None:
        """Render correlation group validation results."""
        st.write("#### 📊 Group Validation & Selection Summary")

        for group_name, validation in validation_result["group_validation"].items():
            col1, col2 = st.columns([1, 3])
            with col1:
                if validation["has_selection"]:
                    st.success(f"✅ {group_name}")
                else:
                    st.warning(f"⚠️ {group_name}")
            with col2:
                if validation["has_selection"]:
                    st.write(f"Removing {validation['selected_features']} of {validation['total_features']} features: {', '.join(validation['selected_list'])}")
                else:
                    st.write(f"⚠️ Keeping all {validation['total_features']} correlated features")

        # Show warnings if needed
        if validation_result["groups_with_no_selection"]:
            st.warning(f"""
            ⚠️ **Multicollinearity Warning**

            You have chosen to keep all features in {len(validation_result["groups_with_no_selection"])} correlation group(s): {', '.join(validation_result["groups_with_no_selection"])}

            **Potential Issues:**
            - **Multicollinearity**: Highly correlated features can make model coefficients unstable
            - **Redundancy**: Features may provide overlapping information
            - **Interpretation**: Feature importance scores may be unreliable
            - **Performance**: Model may be less generalizable

            **Recommendations:**
            - Consider removing at least one feature from each group
            - Monitor model performance carefully
            - Document your decision to keep correlated features

            You can still proceed, but we recommend addressing these correlations.
            """)

            # Log the decision to keep correlated features
            if self.logger:
                self.logger.log_user_action(
                    "Correlated Features Kept",
                    {
                        "groups_with_no_removal": validation_result["groups_with_no_selection"],
                        "total_groups": len(validation_result["group_validation"]),
                        "decision": "user_chose_to_keep_correlated_features",
                        "warning_displayed": True
                    }
                )
        else:
            if validation_result["group_validation"]:  # Only show if there are groups
                st.success("✅ All correlation groups addressed!")

    def _render_transitive_impact_analysis(self, impact_analysis: Dict[str, Any]) -> None:
        """Render transitive impact analysis results."""
        if impact_analysis["has_bridge_features"]:
            st.write("#### 🔗 Transitive Impact Analysis")

            st.warning(f"""
            **Bridge Feature Detection**

            {len(impact_analysis["bridge_features"])} feature(s) act as correlation bridges between other features:
            {', '.join(impact_analysis["bridge_features"])}

            **What this means:**
            - These features connect other features that have weak direct correlations
            - Removing them may hide indirect relationships in your data
            - Consider keeping at least one bridge feature if domain knowledge suggests the relationships are important
            """)

            if impact_analysis["affected_correlations"]:
                with st.expander("📋 View Affected Correlation Chains", expanded=False):
                    affected_df = pd.DataFrame(impact_analysis["affected_correlations"])
                    st.dataframe(
                        affected_df,
                        column_config={
                            "bridge": "Bridge Feature",
                            "feature1": "Feature 1",
                            "feature2": "Feature 2",
                            "direct_correlation": "Direct Correlation",
                            "transitive_strength": "Via Bridge",
                            "group": "Group"
                        },
                        hide_index=True,
                        width='stretch'
                    )

                    st.info("""
                    **How to interpret:**
                    - **Direct Correlation**: Correlation between Feature 1 and Feature 2 directly
                    - **Via Bridge**: Effective correlation through the bridge feature (product of correlations)
                    - A strong bridge has low direct correlation but high transitive strength
                    """)

            # Log the bridge feature detection
            if self.logger:
                self.logger.log_user_action(
                    "Bridge Features Detected",
                    {
                        "bridge_features": impact_analysis["bridge_features"],
                        "affected_correlation_count": len(impact_analysis["affected_correlations"]),
                        "user_notified": True
                    }
                )

    def _render_selection_summary_and_apply(self, selected_features: List[str], strategy: str) -> None:
        """Render selection summary and apply button."""
        if not selected_features:
            return

        st.markdown("### 📝 Features Selected for Removal")
        st.markdown("The following features will be removed:")

        # Create summary table
        features_by_category = create_features_by_category(
            self.protected_attributes,
            [feat for feat, score in zip(self.feature_scores['feature'], self.feature_scores['importance'])
             if score <= 0.01],
            self.correlations
        )

        summary_table = create_selection_summary_table(
            selected_features, features_by_category, self.feature_scores
        )
        st.table(summary_table)

        # Add impact warning
        curr_features = len(self.builder.X_train.columns)
        reduction_percentage = (len(selected_features) / curr_features) * 100
        st.warning(
            f"⚠️ This will remove {len(selected_features)} features "
            f"({reduction_percentage:.1f}% of total features)"
        )

        if st.button("Apply Selection", type="primary"):
            self._apply_feature_selection(selected_features, strategy)

    def _apply_feature_selection(self, selected_features: List[str], strategy: str) -> None:
        """Apply the feature selection."""
        try:
            # Add to history before making changes
            if self.state_manager:
                self.state_manager.add_to_feature_history(f"Manual selection: {strategy}")

            # Update features in the builder
            update_result = self.builder.update_features(selected_features)
            if not update_result["success"]:
                st.error(update_result["message"])
                if self.logger:
                    self.logger.log_error(
                        "Feature Selection Failed",
                        {"error": update_result["message"], "selected_features": selected_features}
                    )
                return

            # Update training and testing data
            remaining_features = list(self.builder.X_train.columns)
            self.builder.training_data = pd.concat(
                [self.builder.X_train[remaining_features], self.builder.y_train], axis=1
            )
            self.builder.testing_data = pd.concat(
                [self.builder.X_test[remaining_features], self.builder.y_test], axis=1
            )

            # Clean missing values if any
            clean_missing_values(self.builder)

            # Process for duplicates
            self._process_duplicates_after_selection()

            # Synchronize data splits
            sync_result = synchronize_data_splits(self.builder)
            if not sync_result["success"]:
                st.error(f"Data synchronization failed: {sync_result['message']}")
                return

            # Track the feature removal
            if self.state_manager:
                self.state_manager.track_manual_feature_removal(selected_features, strategy)

            # Log successful application
            if self.logger:
                self.logger.log_user_action(
                    "Feature Selection Applied",
                    {
                        "removed_features": selected_features,
                        "remaining_features": remaining_features,
                        "data_consistency": "verified",
                        "training_rows": len(self.builder.training_data),
                        "testing_rows": len(self.builder.testing_data),
                        "selection_strategy": strategy
                    }
                )

            # Update step and show success
            if self.state_manager:
                self.state_manager.set_current_step(2)

            st.success("✅ Feature selection applied successfully! Selected features have been updated.")
            st.rerun()

        except Exception as e:
            st.error(f"Error applying feature selection: {str(e)}")
            if self.logger:
                self.logger.log_error(
                    "Feature Selection Application Error",
                    {"error": str(e), "selected_features": selected_features}
                )

    def _process_duplicates_after_selection(self) -> None:
        """Process duplicates after feature selection."""
        # Process training and testing data for duplicates
        training_data = pd.concat([self.builder.X_train, self.builder.y_train], axis=1)
        testing_data = pd.concat([self.builder.X_test, self.builder.y_test], axis=1)

        # Use the reusable function for training data
        training_data, train_duplicate_stats = check_and_remove_duplicates(
            training_data,
            data_type="Training",
            target_column=self.builder.target_column
        )

        # Update the builder with cleaned training data
        self.builder.training_data = training_data

        # Use the reusable function for testing data
        testing_data, test_duplicate_stats = check_and_remove_duplicates(
            testing_data,
            data_type="Testing",
            target_column=self.builder.target_column
        )

        # Update the builder with cleaned testing data
        self.builder.testing_data = testing_data

        # Update X_train, X_test, y_train, y_test with cleaned data
        self.builder.X_train = training_data.drop(self.builder.target_column, axis=1)
        self.builder.X_test = testing_data.drop(self.builder.target_column, axis=1)
        self.builder.y_train = training_data[self.builder.target_column]
        self.builder.y_test = testing_data[self.builder.target_column]