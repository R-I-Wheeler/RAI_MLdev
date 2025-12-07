"""
Automated Feature Selection Component

This module provides a fully automated feature selection pipeline that replicates 100%
of the functionality from the manual feature selection page. It automatically applies
intelligent feature selection strategies and outputs ready-to-use data.

Key Features:
- Complete replication of manual feature selection functionality
- Automatic low importance feature removal with tiered analysis
- Correlation-based feature removal with group analysis
- Optional Boruta algorithm for advanced selection
- Protected attribute identification
- Duplicate detection and removal
- Data synchronization and validation
- Compatible output with all downstream stages

Usage:
    from utils.automated_feature_selection import AutomatedFeatureSelectionComponent

    auto_fs = AutomatedFeatureSelectionComponent(
        builder=st.session_state.builder,
        logger=st.session_state.logger,
        use_boruta=True,
        boruta_threshold=10
    )

    result = auto_fs.run()
    if result['success']:
        # Feature selection complete, ready for model selection
        print(f"Success! {result['summary']}")
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import traceback
from datetime import datetime

# Import existing feature selection components
from components.feature_selection.utils.correlation_utils import (
    build_correlation_groups,
    create_correlation_analysis_data
)
from components.feature_selection.utils.data_processing_utils import (
    check_and_remove_duplicates,
    synchronize_data_splits,
    clean_missing_values,
    validate_data_consistency
)
from components.feature_selection.utils.tracking_utils import (
    track_automated_feature_removal
)

# Import Builder for ModelStage
from Builder import ModelStage


class AutomatedFeatureSelectionComponent:
    """
    Automated feature selection component that replicates 100% of manual feature selection
    functionality, automatically applying intelligent selection strategies.

    This component uses the actual utility functions from the feature selection components
    to ensure complete fidelity with the manual process, including:
    - Tiered importance analysis with automatic removal
    - Correlation group analysis and removal
    - Optional Boruta algorithm for advanced selection
    - Protected attribute identification
    - Duplicate removal and data synchronization

    Attributes:
        builder: Builder instance containing data and model state
        logger: MLLogger instance for tracking operations
        show_analysis: Whether to display analysis summaries (default True)
        use_boruta: Whether to use Boruta algorithm (default False)
        boruta_threshold: Minimum features after selection to trigger Boruta (default 10)
        selection_summary: Dictionary containing feature selection results

    Example:
        >>> auto_fs = AutomatedFeatureSelectionComponent(builder, logger, use_boruta=True)
        >>> result = auto_fs.run()
        >>> if result['success']:
        >>>     # Ready for model selection
        >>>     st.switch_page("pages/5_Model_Selection.py")
    """

    def __init__(
        self,
        builder,
        logger,
        show_analysis: bool = True,
        use_boruta: bool = False,
        boruta_threshold: int = 10
    ):
        """
        Initialize the automated feature selection component.

        Args:
            builder: Builder instance with preprocessed data
            logger: MLLogger instance for tracking operations
            show_analysis: Whether to show analysis summaries (default True)
            use_boruta: Whether to use Boruta algorithm (default False)
            boruta_threshold: Min features after selection to trigger Boruta (default 10)
        """
        self.builder = builder
        self.logger = logger
        self.show_analysis = show_analysis
        self.use_boruta = use_boruta
        self.boruta_threshold = boruta_threshold

        # Initialize selection summary
        self.selection_summary = {
            'steps_completed': [],
            'steps_failed': [],
            'start_time': None,
            'end_time': None,
            'initial_features': None,
            'final_features': None,
            'features_removed': [],
            'transformations_applied': [],
            'low_importance_removed': [],
            'low_importance_details': {},  # {feature: {'importance': score, 'reason': text}}
            'correlation_removed': {},  # {group_id: [features]}
            'correlation_details': {},  # {feature: {'importance': score, 'correlation': value, 'group': id}}
            'protected_attributes': [],
            'duplicates_stats': {},
            'boruta_applied': False,
            'boruta_stats': {},
            'boruta_removed_details': {},  # {feature: {'importance': score, 'status': text}}
            'removed_features_stats': {}  # {feature: {min, max, mean, std, etc.}}
        }

        # Validate initial state
        self._validate_initial_state()

    def _validate_initial_state(self):
        """Validate that required data and configuration exist."""
        if self.builder.training_data is None:
            raise ValueError("Builder training data is None. Please complete preprocessing before feature selection.")

        if self.builder.testing_data is None:
            raise ValueError("Builder testing data is None. Please complete preprocessing before feature selection.")

        if self.builder.target_column is None:
            raise ValueError("Target column is not set. Please set target column before feature selection.")

        # Store initial feature count for reporting
        self.selection_summary['initial_features'] = len(self.builder.X_train.columns)

        self.logger.log_user_action(
            "Automated Feature Selection - Initialization",
            {
                "initial_features": len(self.builder.X_train.columns),
                "target_column": self.builder.target_column,
                "use_boruta": self.use_boruta,
                "boruta_threshold": self.boruta_threshold,
                "training_shape": self.builder.training_data.shape,
                "testing_shape": self.builder.testing_data.shape
            }
        )

    def run(self) -> Dict[str, Any]:
        """
        Run the complete automated feature selection pipeline.

        This method executes all feature selection steps in sequence:
        1. Initial validation
        2. Feature importance analysis
        3. Low importance feature removal
        4. Correlation-based feature removal
        5. Protected attributes review
        6. Data synchronization
        7. Optional Boruta selection
        8. Duplicate removal (after all feature analysis)
        9. Final validation
        10. Stage completion

        Returns:
            Dict containing:
                - success (bool): Whether feature selection completed successfully
                - summary (str): Human-readable summary of results
                - details (Dict): Detailed results from each step
                - errors (List): Any errors encountered during processing

        Example:
            >>> result = auto_fs.run()
            >>> if result['success']:
            >>>     st.success(result['summary'])
        """
        self.selection_summary['start_time'] = datetime.now()

        self.logger.log_stage_transition(
            "DATA_PREPROCESSING",
            "AUTOMATED_FEATURE_SELECTION_START"
        )

        # Only show header if showing step-by-step analysis
        if self.show_analysis:
            st.write("## 🎯 Automated Feature Selection")
            st.write("Running automated feature selection pipeline with intelligent strategies...")

        try:
            # Step 1: Initial validation (already done in __init__)

            # Step 2: Feature importance analysis
            self._step_2_feature_importance_analysis()

            # Step 3: Low importance feature removal
            self._step_3_low_importance_removal()

            # Step 4: Correlation-based feature removal
            self._step_4_correlation_removal()

            # Step 5: Protected attributes review
            self._step_5_protected_attributes_review()

            # Step 6: Data synchronization
            self._step_6_data_synchronization()

            # Step 7: Optional Boruta selection
            if self.use_boruta:
                self._step_7_boruta_selection()
            else:
                if self.show_analysis:
                    st.info("ℹ️ Boruta algorithm disabled per configuration.")

            # Step 8: Duplicate removal (after all feature analysis and removal)
            self._step_8_duplicate_removal()

            # Step 9: Final validation
            self._step_9_final_validation()

            # Step 10: Stage completion
            self._step_10_stage_completion()

            # Finalize selection
            self.selection_summary['end_time'] = datetime.now()
            self.selection_summary['final_features'] = len(self.builder.X_train.columns)

            # Mark feature selection stage as complete
            self.builder.stage_completion[ModelStage.FEATURE_SELECTION] = True

            # CRITICAL: Store final feature selection data in session state
            # This ensures downstream stages have access to the selected features
            st.session_state.final_feature_selection_training_data = self.builder.training_data.copy()
            st.session_state.final_feature_selection_testing_data = self.builder.testing_data.copy()

            self.logger.log_stage_transition(
                "AUTOMATED_FEATURE_SELECTION_END",
                "MODEL_SELECTION"
            )

            # Generate summary (always needed for return value)
            summary = self._generate_summary()

            # Don't show summary here - it will be shown in the dashboard
            # Only show if explicitly in step-by-step mode
            if self.show_analysis:
                st.success("✅ Automated feature selection completed successfully!")

            return {
                'success': True,
                'summary': summary,
                'details': self.selection_summary,
                'errors': self.selection_summary['steps_failed']
            }

        except Exception as e:
            error_msg = f"Automated feature selection failed: {str(e)}"
            self.logger.log_error(
                "Automated Feature Selection Failed",
                {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "steps_completed": self.selection_summary['steps_completed'],
                    "steps_failed": self.selection_summary['steps_failed']
                }
            )

            if self.show_analysis:
                st.error(f"❌ {error_msg}")
                with st.expander("View Error Details"):
                    st.code(traceback.format_exc())

            return {
                'success': False,
                'summary': error_msg,
                'details': self.selection_summary,
                'errors': [str(e)]
            }

    def run_with_report(self) -> Dict[str, Any]:
        """
        Run feature selection and generate a detailed report.

        Returns:
            Dict containing:
                - success (bool): Whether feature selection completed successfully
                - summary (str): Human-readable summary
                - report (str): Detailed markdown report
                - details (Dict): Detailed results from each step

        Example:
            >>> result = auto_fs.run_with_report()
            >>> st.markdown(result['report'])
        """
        result = self.run()
        result['report'] = self._generate_detailed_report()
        return result

    def _step_2_feature_importance_analysis(self):
        """
        Step 2: Feature Importance Analysis

        Analyzes feature importance, correlations, and protected attributes
        using the Builder's built-in analysis method.
        """
        step_name = "Feature Importance Analysis"

        try:
            if self.show_analysis:
                st.write(f"### Step 2: {step_name}")

            # Run feature importance analysis
            analysis_result = self.builder.analyse_feature_importance()

            if not analysis_result["success"]:
                raise Exception(f"Feature analysis failed: {analysis_result.get('message', 'Unknown error')}")

            # Store analysis results for later steps
            self.feature_scores = pd.DataFrame(analysis_result["feature_scores"])
            self.correlations = analysis_result["responsible_ai"]["correlations"]
            self.protected_attributes = analysis_result["responsible_ai"]["protected_attributes"]
            self.low_importance_features = analysis_result["responsible_ai"]["low_importance_features"]

            # Calculate statistics
            stats = {
                'total_features': len(self.feature_scores),
                'low_importance_count': len(self.low_importance_features),
                'correlation_pairs': len(self.correlations),
                'protected_attributes_count': len(self.protected_attributes)
            }

            self.selection_summary['steps_completed'].append({
                'step': step_name,
                'details': stats
            })

            if self.show_analysis:
                st.success(f"✅ {step_name} complete")
                st.write(f"- **Total features:** {stats['total_features']}")
                st.write(f"- **Low importance features:** {stats['low_importance_count']}")
                st.write(f"- **Correlation pairs:** {stats['correlation_pairs']}")
                st.write(f"- **Protected attributes:** {stats['protected_attributes_count']}")

            self.logger.log_calculation(
                f"Automated Feature Selection - {step_name}",
                stats
            )

        except Exception as e:
            self._handle_step_error(step_name, e)

    def _step_3_low_importance_removal(self):
        """
        Step 3: Low Importance Feature Removal

        Automatically removes features with low importance using tiered analysis:
        - Critical (< 0.001): Automatic removal
        - High concern (0.001-0.01): Correlation-based analysis
        - Minimal correlation (< 0.1): Automatic removal
        """
        step_name = "Low Importance Feature Removal"

        try:
            if self.show_analysis:
                st.write(f"### Step 3: {step_name}")

            # Use ALL low importance features identified by the builder
            # (not just the critical ones from tiered analysis)
            features_to_remove = self.low_importance_features

            if not features_to_remove:
                if self.show_analysis:
                    st.info("ℹ️ No low importance features detected - skipping this step.")

                self.selection_summary['steps_completed'].append({
                    'step': step_name,
                    'details': {'features_removed': 0, 'action': 'skipped'}
                })
                return

            # Calculate stats for features before removing them
            feature_stats = self._calculate_feature_stats(features_to_remove)
            self.selection_summary['removed_features_stats'].update(feature_stats)

            # Remove features from both training and testing data
            self.builder.X_train = self.builder.X_train.drop(columns=features_to_remove)
            self.builder.X_test = self.builder.X_test.drop(columns=features_to_remove)

            # Update training and testing data
            self.builder.training_data = pd.concat(
                [self.builder.X_train, self.builder.y_train], axis=1
            )
            self.builder.testing_data = pd.concat(
                [self.builder.X_test, self.builder.y_test], axis=1
            )

            # Update feature names
            self.builder.feature_names = list(self.builder.X_train.columns)

            # Track removal with detailed reasons
            self.selection_summary['low_importance_removed'] = features_to_remove
            self.selection_summary['features_removed'].extend(features_to_remove)
            self.selection_summary['transformations_applied'].append('low_importance_removal')

            # Store detailed reasons for each removed feature
            for feat in features_to_remove:
                # Get importance score
                feat_row = self.feature_scores[self.feature_scores['feature'] == feat]
                importance = feat_row['importance'].values[0] if len(feat_row) > 0 else 0

                # Determine category based on importance
                if importance < 0.001:
                    category = "🚨 Critical"
                    reason = f"Extremely low importance (score: {importance:.6f}, threshold: 0.001)"
                elif importance <= 0.01:
                    category = "⚠️ High Concern"
                    reason = f"Low importance (score: {importance:.6f}, threshold: 0.01)"
                else:
                    category = "Low Importance"
                    reason = f"Low importance (score: {importance:.6f})"

                self.selection_summary['low_importance_details'][feat] = {
                    'importance': importance,
                    'category': category,
                    'reason': reason
                }

            # Track in session state
            track_automated_feature_removal(
                removed_features=features_to_remove,
                method_name="Automated Low Importance Removal",
                addresses_low_importance=True,
                addresses_correlation=False
            )

            self.selection_summary['steps_completed'].append({
                'step': step_name,
                'details': {
                    'features_removed': len(features_to_remove),
                    'removed_features_list': features_to_remove,
                    'remaining_features': len(self.builder.X_train.columns)
                }
            })

            if self.show_analysis:
                st.success(f"✅ {step_name} complete")
                st.write(f"- **Features removed:** {len(features_to_remove)}")
                st.write(f"- **Remaining features:** {len(self.builder.X_train.columns)}")

                with st.expander("View removed features"):
                    for feat in features_to_remove:
                        st.write(f"• {feat}")

            self.logger.log_calculation(
                f"Automated Feature Selection - {step_name}",
                {
                    "features_removed": len(features_to_remove),
                    "removed_features": features_to_remove,
                    "remaining_features": len(self.builder.X_train.columns)
                }
            )

        except Exception as e:
            self._handle_step_error(step_name, e)

    def _step_4_correlation_removal(self):
        """
        Step 4: Correlation-Based Feature Removal

        Iteratively removes features from correlation groups using intelligent selection:
        - Build correlation groups (features with correlation > 0.7)
        - For each group, remove features with:
          - Above-average total correlation AND below-average importance
          - Or highest total correlation if no clear candidate
        - Ensures at least one feature removed per group
        - Repeats until no correlation groups remain (handles multiple groups)
        """
        step_name = "Correlation-Based Feature Removal"

        try:
            if self.show_analysis:
                st.write(f"### Step 4: {step_name}")

            # Check if correlations exist
            if not self.correlations:
                if self.show_analysis:
                    st.info("ℹ️ No correlated features detected - skipping this step.")

                self.selection_summary['steps_completed'].append({
                    'step': step_name,
                    'details': {'features_removed': 0, 'action': 'skipped'}
                })
                return

            if self.show_analysis:
                st.write(f"**Starting with {len(self.correlations)} correlation pairs**")

            # Track all removed features across iterations
            all_removed_features = []
            iteration_count = 0
            max_iterations = 10  # Safety limit

            # Iteratively process correlation groups
            while iteration_count < max_iterations:
                iteration_count += 1

                # Recalculate correlations on current features
                current_correlations = [
                    c for c in self.correlations
                    if c['feature1'] in self.builder.X_train.columns and
                       c['feature2'] in self.builder.X_train.columns
                ]

                if self.show_analysis:
                    st.write(f"  Iteration {iteration_count}: {len(current_correlations)} correlation pairs in current features")

                if not current_correlations:
                    # No more correlations to process
                    if self.show_analysis:
                        st.write("  → No correlations remain, stopping")
                    break

                # Build correlation groups
                correlation_groups = build_correlation_groups(current_correlations)

                if self.show_analysis:
                    st.write(f"  → Found {len(correlation_groups)} correlation group(s)")

                if not correlation_groups:
                    break

                # Get feature correlation matrix for current features
                feature_corr_matrix = self.builder.X_train.corr().abs()

                # Update feature scores for current features
                current_feature_scores = self.feature_scores[
                    self.feature_scores['feature'].isin(self.builder.X_train.columns)
                ]

                # Create correlation analysis data
                correlation_analysis_data = create_correlation_analysis_data(
                    correlation_groups,
                    current_feature_scores,
                    feature_corr_matrix
                )

                # Extract features recommended for removal - ONE PER GROUP only
                # This matches manual feature selection behavior
                features_to_remove = []
                for group_idx, group_features in enumerate(correlation_groups):
                    # Find recommended features in this group
                    group_recommendations = [
                        item["Feature"] for item in correlation_analysis_data
                        if item["Remove"] == True and item["Feature"] in group_features
                    ]

                    # Only take the first recommended feature from each group
                    if group_recommendations:
                        features_to_remove.append(group_recommendations[0])
                        if self.show_analysis:
                            st.write(f"    Group {group_idx + 1}: Removing {group_recommendations[0]}")

                if not features_to_remove:
                    # No more features recommended for removal
                    if self.show_analysis:
                        st.write("  → No features recommended for removal, stopping")
                    break

                # Calculate stats for features before removing them
                feature_stats = self._calculate_feature_stats(features_to_remove)
                self.selection_summary['removed_features_stats'].update(feature_stats)

                # Remove features from both training and testing data
                self.builder.X_train = self.builder.X_train.drop(columns=features_to_remove)
                self.builder.X_test = self.builder.X_test.drop(columns=features_to_remove)

                # Track removal by group with detailed reasons
                for group_idx, group_features in enumerate(correlation_groups):
                    group_id = f"Iteration {iteration_count} - Group {group_idx + 1}"
                    removed_in_group = [f for f in features_to_remove if f in group_features]
                    if removed_in_group:
                        self.selection_summary['correlation_removed'][group_id] = removed_in_group

                # Store detailed reasons for each removed feature
                for item in correlation_analysis_data:
                    if item["Remove"] == True:
                        feat = item["Feature"]
                        self.selection_summary['correlation_details'][feat] = {
                            'importance': item.get("Importance", 0),
                            'total_correlation': item.get("Total Correlation", 0),
                            'group': item.get("Group", "Unknown"),
                            'iteration': iteration_count,
                            'reason': self._get_correlation_reason(item)
                        }

                # Add to total removed features
                all_removed_features.extend(features_to_remove)

                if self.show_analysis:
                    st.write(f"**Iteration {iteration_count}:** Removed {len(features_to_remove)} features from {len(correlation_groups)} group(s)")

            # Final updates after all iterations
            if all_removed_features:
                # Update training and testing data
                self.builder.training_data = pd.concat(
                    [self.builder.X_train, self.builder.y_train], axis=1
                )
                self.builder.testing_data = pd.concat(
                    [self.builder.X_test, self.builder.y_test], axis=1
                )

                # Update feature names
                self.builder.feature_names = list(self.builder.X_train.columns)

                self.selection_summary['features_removed'].extend(all_removed_features)
                self.selection_summary['transformations_applied'].append('correlation_removal')

                # Track in session state
                track_automated_feature_removal(
                    removed_features=all_removed_features,
                    method_name="Automated Correlation Removal",
                    addresses_low_importance=False,
                    addresses_correlation=True
                )

                self.selection_summary['steps_completed'].append({
                    'step': step_name,
                    'details': {
                        'iterations': iteration_count,
                        'features_removed': len(all_removed_features),
                        'removed_features_list': all_removed_features,
                        'removed_by_group': self.selection_summary['correlation_removed'],
                        'remaining_features': len(self.builder.X_train.columns)
                    }
                })

                if self.show_analysis:
                    st.success(f"✅ {step_name} complete")
                    st.write(f"- **Iterations:** {iteration_count}")
                    st.write(f"- **Features removed:** {len(all_removed_features)}")
                    st.write(f"- **Remaining features:** {len(self.builder.X_train.columns)}")

                    with st.expander("View removed features by group"):
                        for group_id, removed in self.selection_summary['correlation_removed'].items():
                            st.write(f"**{group_id}:** {', '.join(removed)}")

                self.logger.log_calculation(
                    f"Automated Feature Selection - {step_name}",
                    {
                        "iterations": iteration_count,
                        "features_removed": len(all_removed_features),
                        "removed_features": all_removed_features,
                        "removed_by_group": self.selection_summary['correlation_removed']
                    }
                )
            else:
                if self.show_analysis:
                    st.info("ℹ️ Correlation groups detected but no features recommended for removal.")

                self.selection_summary['steps_completed'].append({
                    'step': step_name,
                    'details': {'features_removed': 0, 'action': 'no_automatic_removal'}
                })

        except Exception as e:
            self._handle_step_error(step_name, e)

    def _step_5_protected_attributes_review(self):
        """
        Step 5: Protected Attributes Review

        Identifies protected attributes for informational purposes.
        No automatic removal - just logging and reporting.
        """
        step_name = "Protected Attributes Review"

        try:
            if self.show_analysis:
                st.write(f"### Step 5: {step_name}")

            # Store protected attributes
            self.selection_summary['protected_attributes'] = self.protected_attributes

            if not self.protected_attributes:
                if self.show_analysis:
                    st.success("✅ No protected attributes detected")

                self.selection_summary['steps_completed'].append({
                    'step': step_name,
                    'details': {'protected_attributes_count': 0}
                })
            else:
                if self.show_analysis:
                    st.warning(f"⚠️ {len(self.protected_attributes)} protected attribute(s) detected:")
                    for attr in self.protected_attributes:
                        st.write(f"• {attr}")
                    st.info("💡 Protected attributes are flagged for review but not automatically removed.")

                self.selection_summary['steps_completed'].append({
                    'step': step_name,
                    'details': {
                        'protected_attributes_count': len(self.protected_attributes),
                        'protected_attributes_list': self.protected_attributes
                    }
                })

            self.logger.log_calculation(
                f"Automated Feature Selection - {step_name}",
                {
                    "protected_attributes_count": len(self.protected_attributes),
                    "protected_attributes": self.protected_attributes
                }
            )

        except Exception as e:
            self._handle_step_error(step_name, e)

    def _step_8_duplicate_removal(self):
        """
        Step 8: Duplicate Removal

        Detects and removes duplicates from training and testing datasets AFTER
        all feature analysis and removal (including Boruta):
        - Exact duplicates (all columns identical)
        - Partial duplicates (features identical, target different)
        """
        step_name = "Duplicate Removal"

        try:
            if self.show_analysis:
                st.write(f"### Step 8: {step_name}")

            # Process training data
            training_data_combined = pd.concat(
                [self.builder.X_train, self.builder.y_train], axis=1
            )
            training_cleaned, train_stats = check_and_remove_duplicates(
                training_data_combined,
                data_type="Training",
                target_column=self.builder.target_column
            )

            # Process testing data
            testing_data_combined = pd.concat(
                [self.builder.X_test, self.builder.y_test], axis=1
            )
            testing_cleaned, test_stats = check_and_remove_duplicates(
                testing_data_combined,
                data_type="Testing",
                target_column=self.builder.target_column
            )

            # Update builder data
            self.builder.training_data = training_cleaned
            self.builder.testing_data = testing_cleaned

            # Update X_train, X_test, y_train, y_test
            self.builder.X_train = training_cleaned.drop(columns=[self.builder.target_column])
            self.builder.X_test = testing_cleaned.drop(columns=[self.builder.target_column])
            self.builder.y_train = training_cleaned[self.builder.target_column]
            self.builder.y_test = testing_cleaned[self.builder.target_column]

            # Store duplicate statistics
            self.selection_summary['duplicates_stats'] = {
                'training': train_stats,
                'testing': test_stats
            }

            total_removed = train_stats['total_reduction'] + test_stats['total_reduction']

            if total_removed == 0:
                if self.show_analysis:
                    st.info("ℹ️ No duplicates found - skipping this step.")

                self.selection_summary['steps_completed'].append({
                    'step': step_name,
                    'details': {'action': 'skipped', 'total_removed': 0}
                })
            else:
                if self.show_analysis:
                    st.success(f"✅ {step_name} complete")
                    st.write(f"**Training Data:**")
                    st.write(f"- Exact duplicates: {train_stats['exact_duplicates_found']}")
                    st.write(f"- Partial duplicates: {train_stats['partial_duplicates_found']}")
                    st.write(f"- Rows removed: {train_stats['total_reduction']}")

                    st.write(f"**Testing Data:**")
                    st.write(f"- Exact duplicates: {test_stats['exact_duplicates_found']}")
                    st.write(f"- Partial duplicates: {test_stats['partial_duplicates_found']}")
                    st.write(f"- Rows removed: {test_stats['total_reduction']}")

                self.selection_summary['steps_completed'].append({
                    'step': step_name,
                    'details': {
                        'total_removed': total_removed,
                        'training_stats': train_stats,
                        'testing_stats': test_stats
                    }
                })

                self.logger.log_calculation(
                    f"Automated Feature Selection - {step_name}",
                    {
                        "total_removed": total_removed,
                        "training_stats": train_stats,
                        "testing_stats": test_stats
                    }
                )

        except Exception as e:
            self._handle_step_error(step_name, e)

    def _step_6_data_synchronization(self):
        """
        Step 6: Data Synchronization

        Ensures X_train, X_test, y_train, y_test are synchronized with
        training_data and testing_data.
        """
        step_name = "Data Synchronization"

        try:
            if self.show_analysis:
                st.write(f"### Step 6: {step_name}")

            # Synchronize data splits
            sync_result = synchronize_data_splits(self.builder)

            if not sync_result["success"]:
                raise Exception(f"Data synchronization failed: {sync_result['message']}")

            # Clean any missing values
            clean_result = clean_missing_values(self.builder)

            self.selection_summary['steps_completed'].append({
                'step': step_name,
                'details': {
                    'sync_result': sync_result['final_state'],
                    'missing_values_cleaned': clean_result['success']
                }
            })

            if self.show_analysis:
                st.success(f"✅ {step_name} complete")
                st.write(f"- **Training shape:** {self.builder.training_data.shape}")
                st.write(f"- **Testing shape:** {self.builder.testing_data.shape}")
                st.write(f"- **Data consistency:** Verified")

            self.logger.log_calculation(
                f"Automated Feature Selection - {step_name}",
                sync_result['final_state']
            )

        except Exception as e:
            self._handle_step_error(step_name, e)

    def _step_7_boruta_selection(self):
        """
        Step 7: Optional Boruta Selection

        Applies Boruta algorithm if:
        - use_boruta is True
        - Remaining features > boruta_threshold

        Uses "Confirmed Only" strategy (most conservative).
        """
        step_name = "Boruta Algorithm Selection"

        try:
            if self.show_analysis:
                st.write(f"### Step 7: {step_name}")

            current_features = len(self.builder.X_train.columns)

            # Check if we should run Boruta
            if current_features < self.boruta_threshold:
                if self.show_analysis:
                    st.info(
                        f"ℹ️ Boruta skipped: {current_features} features remaining "
                        f"(minimum threshold: {self.boruta_threshold})"
                    )

                self.selection_summary['steps_completed'].append({
                    'step': step_name,
                    'details': {
                        'action': 'skipped',
                        'reason': f'features ({current_features}) < threshold ({self.boruta_threshold})'
                    }
                })
                return

            if self.show_analysis:
                st.write(f"Running Boruta algorithm on {current_features} features...")

            # Run Boruta selection
            boruta_result = self._run_boruta_selection()

            if not boruta_result["success"]:
                raise Exception(f"Boruta selection failed: {boruta_result['message']}")

            # Apply "Confirmed Only" strategy (most conservative)
            # Remove tentative and rejected features
            features_to_remove = (
                boruta_result["info"]["tentative_features"] +
                boruta_result["info"]["rejected_features"]
            )

            if not features_to_remove:
                if self.show_analysis:
                    st.success("✅ Boruta confirmed all features!")

                self.selection_summary['boruta_applied'] = True
                self.selection_summary['boruta_stats'] = boruta_result["info"]["statistics"]
                self.selection_summary['steps_completed'].append({
                    'step': step_name,
                    'details': {
                        'features_removed': 0,
                        'boruta_stats': boruta_result["info"]["statistics"]
                    }
                })
                return

            # Calculate stats for features before removing them
            feature_stats = self._calculate_feature_stats(features_to_remove)
            self.selection_summary['removed_features_stats'].update(feature_stats)

            # Remove features from both training and testing data
            self.builder.X_train = self.builder.X_train.drop(columns=features_to_remove)
            self.builder.X_test = self.builder.X_test.drop(columns=features_to_remove)

            # Update training and testing data
            self.builder.training_data = pd.concat(
                [self.builder.X_train, self.builder.y_train], axis=1
            )
            self.builder.testing_data = pd.concat(
                [self.builder.X_test, self.builder.y_test], axis=1
            )

            # Update feature names
            self.builder.feature_names = list(self.builder.X_train.columns)

            # Track removal with detailed reasons
            self.selection_summary['features_removed'].extend(features_to_remove)
            self.selection_summary['transformations_applied'].append('boruta_selection')
            self.selection_summary['boruta_applied'] = True
            self.selection_summary['boruta_stats'] = boruta_result["info"]["statistics"]

            # Store detailed reasons for each removed feature
            for feat_info in boruta_result["info"]["feature_ranking"]:
                if feat_info["feature"] in features_to_remove:
                    self.selection_summary['boruta_removed_details'][feat_info["feature"]] = {
                        'importance': feat_info.get("importance", 0),
                        'rank': feat_info.get("rank", 0),
                        'status': feat_info.get("status", "Unknown"),
                        'reason': f"Boruta marked as {feat_info.get('status', 'Unknown')}"
                    }

            # Track in session state
            track_automated_feature_removal(
                removed_features=features_to_remove,
                method_name="Automated Boruta Selection (Confirmed Only)",
                addresses_low_importance=True,
                addresses_correlation=False
            )

            self.selection_summary['steps_completed'].append({
                'step': step_name,
                'details': {
                    'features_removed': len(features_to_remove),
                    'removed_features_list': features_to_remove,
                    'confirmed_features': boruta_result["info"]["confirmed_features"],
                    'tentative_features': boruta_result["info"]["tentative_features"],
                    'rejected_features': boruta_result["info"]["rejected_features"],
                    'boruta_stats': boruta_result["info"]["statistics"],
                    'remaining_features': len(self.builder.X_train.columns)
                }
            })

            if self.show_analysis:
                st.success(f"✅ {step_name} complete")
                st.write(f"- **Features removed:** {len(features_to_remove)}")
                st.write(f"- **Confirmed features:** {len(boruta_result['info']['confirmed_features'])}")
                st.write(f"- **Remaining features:** {len(self.builder.X_train.columns)}")

            self.logger.log_calculation(
                f"Automated Feature Selection - {step_name}",
                {
                    "features_removed": len(features_to_remove),
                    "removed_features": features_to_remove,
                    "boruta_stats": boruta_result["info"]["statistics"]
                }
            )

        except Exception as e:
            self._handle_step_error(step_name, e)

    def _step_9_final_validation(self):
        """
        Step 9: Final Validation

        Validates the final state before allowing progression:
        - Data consistency (feature/target length match)
        - No missing values
        - At least some features remain
        - Train/test column consistency
        """
        step_name = "Final Data Validation"

        try:
            if self.show_analysis:
                st.write(f"### Step 9: {step_name}")

            # Validate data consistency
            validation_result = validate_data_consistency(self.builder)

            if not validation_result["success"]:
                error_messages = [err["message"] for err in validation_result["errors"]]
                raise Exception(f"Validation failed: {'; '.join(error_messages)}")

            # Additional checks
            validation_checks = {
                'data_consistency': True,
                'no_missing_values': (
                    self.builder.training_data.isnull().sum().sum() == 0 and
                    self.builder.testing_data.isnull().sum().sum() == 0
                ),
                'features_remaining': len(self.builder.X_train.columns) > 0,
                'columns_match': (
                    set(self.builder.training_data.columns) ==
                    set(self.builder.testing_data.columns)
                )
            }

            # Check all validations passed
            if not all(validation_checks.values()):
                failed_checks = [k for k, v in validation_checks.items() if not v]
                raise Exception(f"Validation checks failed: {', '.join(failed_checks)}")

            self.selection_summary['steps_completed'].append({
                'step': step_name,
                'details': {
                    'validation_checks': validation_checks,
                    'final_training_shape': self.builder.training_data.shape,
                    'final_testing_shape': self.builder.testing_data.shape,
                    'final_features': len(self.builder.X_train.columns)
                }
            })

            if self.show_analysis:
                st.success(f"✅ {step_name} complete")
                st.write(f"- **Final training shape:** {self.builder.training_data.shape}")
                st.write(f"- **Final testing shape:** {self.builder.testing_data.shape}")
                st.write(f"- **Final features:** {len(self.builder.X_train.columns)}")
                st.write(f"- **Validation checks:**")
                for check, passed in validation_checks.items():
                    st.write(f"  - {check}: ✓" if passed else f"  - {check}: ✗")

            self.logger.log_calculation(
                f"Automated Feature Selection - {step_name}",
                {
                    'validation_checks': validation_checks,
                    'final_shapes': {
                        'training': self.builder.training_data.shape,
                        'testing': self.builder.testing_data.shape
                    }
                }
            )

        except Exception as e:
            self._handle_step_error(step_name, e)
            # Re-raise to stop the pipeline since validation failed
            raise

    def _step_10_stage_completion(self):
        """
        Step 10: Stage Completion

        Marks feature selection stage as complete and logs final metrics.
        """
        step_name = "Stage Completion"

        try:
            # Calculate final metrics
            initial_features = self.selection_summary['initial_features']
            final_features = len(self.builder.X_train.columns)
            features_removed = len(self.selection_summary['features_removed'])
            reduction_percentage = (features_removed / initial_features * 100) if initial_features > 0 else 0

            completion_metrics = {
                'initial_features': initial_features,
                'final_features': final_features,
                'features_removed': features_removed,
                'reduction_percentage': reduction_percentage,
                'removed_features_list': self.selection_summary['features_removed'],
                'low_importance_removed': len(self.selection_summary['low_importance_removed']),
                'correlation_groups_addressed': len(self.selection_summary['correlation_removed']),
                'protected_attributes_identified': len(self.selection_summary['protected_attributes']),
                'duplicates_removed': (
                    self.selection_summary['duplicates_stats'].get('training', {}).get('total_reduction', 0) +
                    self.selection_summary['duplicates_stats'].get('testing', {}).get('total_reduction', 0)
                ),
                'boruta_applied': self.selection_summary['boruta_applied']
            }

            self.selection_summary['steps_completed'].append({
                'step': step_name,
                'details': completion_metrics
            })

            self.logger.log_calculation(
                f"Automated Feature Selection - {step_name}",
                completion_metrics
            )

        except Exception as e:
            self._handle_step_error(step_name, e)

    def _run_boruta_selection(self) -> Dict[str, Any]:
        """
        Run Boruta feature selection algorithm.

        Returns:
            Dict containing:
            - success: bool indicating if the operation was successful
            - message: str explaining the result
            - info: Dict containing detailed results if successful
        """
        try:
            from boruta import BorutaPy
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            # Determine problem type and create appropriate estimator
            problem_type = self.builder.detect_problem_type()
            is_classification = problem_type in ["binary_classification", "multiclass_classification", "classification"]
            if is_classification:
                estimator = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=42)
            else:  # regression
                estimator = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)

            # Initialize Boruta
            boruta = BorutaPy(
                estimator=estimator,
                n_estimators='auto',
                max_iter=100,
                verbose=0,  # Silent mode for automated processing
                random_state=42
            )

            # Create processed copies of the data
            X_train_processed = self.builder.X_train.copy()
            y_train_processed = self.builder.y_train.copy()

            # Process each column based on its content
            for col in X_train_processed.columns:
                column_data = X_train_processed[col]

                # Try to convert to numeric first
                try:
                    numeric_data = pd.to_numeric(column_data, errors='raise')
                    X_train_processed[col] = numeric_data
                    continue
                except (ValueError, TypeError):
                    pass

                # If not numeric, handle categorical data
                unique_values = column_data.nunique()
                if unique_values <= 10:  # Threshold for categorical
                    X_train_processed[col] = pd.Categorical(column_data).codes
                else:
                    # For high cardinality strings, use hash encoding
                    X_train_processed[col] = pd.util.hash_array(column_data.fillna(''), num_items=100)

                X_train_processed[col] = X_train_processed[col].astype(float)

            # Handle missing values before Boruta
            X_train_processed = X_train_processed.fillna(X_train_processed.mean())

            # Fit Boruta
            boruta.fit(X_train_processed.values, y_train_processed.values)

            # Get feature ranking and support masks
            feature_ranks = boruta.ranking_
            confirmed_mask = boruta.support_
            tentative_mask = boruta.support_weak_
            rejected_mask = ~(confirmed_mask | tentative_mask)

            # Create feature importance ranking
            feature_ranking = []
            for idx, (feature, rank) in enumerate(zip(self.builder.X_train.columns, feature_ranks)):
                status = "Confirmed" if confirmed_mask[idx] else ("Tentative" if tentative_mask[idx] else "Rejected")
                feature_ranking.append({
                    "feature": feature,
                    "rank": int(rank),
                    "status": status,
                    "importance": float(
                        boruta.importance_history_[:, idx].mean() if boruta.importance_history_ is not None else 0)
                })

            # Sort by importance
            feature_ranking.sort(
                key=lambda x: (-1 if x["status"] == "Confirmed" else (0 if x["status"] == "Tentative" else 1),
                               x["importance"]),
                reverse=False)

            # Calculate statistics
            stats = {
                "total_features": len(self.builder.X_train.columns),
                "confirmed_features": int(confirmed_mask.sum()),
                "tentative_features": int(tentative_mask.sum()),
                "rejected_features": int(rejected_mask.sum()),
                "selection_ratio": float(confirmed_mask.sum() / len(self.builder.X_train.columns)),
                "iterations": 100,
            }

            # Get lists of features by status
            confirmed_features = [f["feature"] for f in feature_ranking if f["status"] == "Confirmed"]
            tentative_features = [f["feature"] for f in feature_ranking if f["status"] == "Tentative"]
            rejected_features = [f["feature"] for f in feature_ranking if f["status"] == "Rejected"]

            # Create importance history if available
            importance_history = None
            if hasattr(boruta, 'importance_history_') and boruta.importance_history_ is not None:
                importance_history = boruta.importance_history_.tolist()

            return {
                "success": True,
                "message": "Boruta feature selection completed successfully",
                "info": {
                    "feature_ranking": feature_ranking,
                    "statistics": stats,
                    "confirmed_features": confirmed_features,
                    "tentative_features": tentative_features,
                    "rejected_features": rejected_features,
                    "importance_history": importance_history
                }
            }
        except Exception as e:
            error_details = traceback.format_exc()
            return {
                "success": False,
                "message": f"Error running Boruta feature selection: {str(e)}\n\nDetails:\n{error_details}"
            }

    def _handle_step_error(self, step_name: str, error: Exception):
        """
        Handle errors that occur during a feature selection step.

        Args:
            step_name: Name of the step that failed
            error: Exception that was raised
        """
        error_details = {
            'step': step_name,
            'error': str(error),
            'traceback': traceback.format_exc()
        }

        self.selection_summary['steps_failed'].append(error_details)

        self.logger.log_error(
            f"Automated Feature Selection - {step_name} Failed",
            error_details
        )

        if self.show_analysis:
            st.warning(f"⚠️ {step_name} failed: {str(error)}")
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())
            st.info("Continuing with remaining steps...")

    def _get_low_importance_reason(self, item: Dict[str, Any]) -> str:
        """
        Generate a detailed reason for low importance feature removal.

        Args:
            item: Feature analysis item dictionary

        Returns:
            Human-readable reason string
        """
        importance = item.get("Importance", 0)
        category = item.get("Category", "Unknown")

        if category == "Critical (< 0.001)":
            return f"Critical low importance (score: {importance:.6f}, threshold: 0.001)"
        elif category == "High Concern (0.001-0.01)":
            max_corr = item.get("Max Correlation", 0)
            return f"High concern importance with minimal correlation (score: {importance:.6f}, max corr: {max_corr:.3f})"
        else:
            return f"Low importance (score: {importance:.6f})"

    def _get_correlation_reason(self, item: Dict[str, Any]) -> str:
        """
        Generate a detailed reason for correlation-based feature removal.

        Args:
            item: Correlation analysis item dictionary

        Returns:
            Human-readable reason string
        """
        # Values from correlation_analysis_data are already formatted strings
        importance = item.get("Importance", "0")
        total_corr = item.get("Total Correlation", "0")
        group = item.get("Group", "Unknown")

        return f"High correlation in {group} (importance: {importance}, total correlation: {total_corr})"

    def _format_step_details(self, step_name: str, details: Dict[str, Any]) -> str:
        """
        Format step details with expanded information about removed features.

        Args:
            step_name: Name of the step
            details: Dictionary of step details

        Returns:
            Formatted markdown string
        """
        formatted = "**Details:**\n\n"

        for key, value in details.items():
            # Special handling for removed features lists
            if key == 'removed_features_list' and isinstance(value, list):
                formatted += f"- **Features Removed ({len(value)}):**\n"
                for feat in value:
                    formatted += f"  - {feat}\n"

            # Special handling for correlation groups
            elif key == 'removed_by_group' and isinstance(value, dict):
                formatted += f"- **Features Removed by Correlation Group:**\n"
                for group_id, features in value.items():
                    formatted += f"  - **{group_id}:** {', '.join(features)}\n"

            # Special handling for Boruta results
            elif key in ['confirmed_features', 'tentative_features', 'rejected_features'] and isinstance(value, list):
                formatted += f"- **{key.replace('_', ' ').title()} ({len(value)}):**\n"
                if len(value) <= 20:  # Show all if reasonable number
                    for feat in value:
                        formatted += f"  - {feat}\n"
                else:  # Show count and first few
                    for feat in value[:10]:
                        formatted += f"  - {feat}\n"
                    formatted += f"  - ... and {len(value) - 10} more\n"

            # Special handling for protected attributes
            elif key == 'protected_attributes_list' and isinstance(value, list):
                formatted += f"- **Protected Attributes ({len(value)}):**\n"
                for attr in value:
                    formatted += f"  - {attr}\n"

            # Standard formatting for other values
            elif isinstance(value, dict):
                formatted += f"- **{key}:** {len(value)} items\n"
            elif isinstance(value, list):
                if len(value) <= 5:
                    formatted += f"- **{key}:** {', '.join(str(v) for v in value)}\n"
                else:
                    formatted += f"- **{key}:** {len(value)} items\n"
            else:
                formatted += f"- **{key}:** {value}\n"

        return formatted

    def _generate_feature_removal_summary(self) -> str:
        """
        Generate a comprehensive summary of all features removed with reasons.

        Returns:
            Formatted markdown string with complete removal details
        """
        summary = ""

        # Get feature importance scores for additional context
        feature_importance_map = {}
        if hasattr(self, 'feature_scores') and self.feature_scores is not None:
            feature_importance_map = dict(zip(
                self.feature_scores['feature'].values,
                self.feature_scores['importance'].values
            ))

        # 1. Low Importance Features
        if self.selection_summary['low_importance_removed']:
            summary += "### Low Importance Features Removed\n\n"
            summary += f"**Count:** {len(self.selection_summary['low_importance_removed'])}\n\n"
            summary += "**Reason:** Features with importance scores below critical threshold (< 0.001) or minimal correlation features with low importance.\n\n"
            summary += "| Feature | Importance Score | Category | Detailed Reason |\n"
            summary += "|---------|-----------------|----------|----------------|\n"
            for feat in self.selection_summary['low_importance_removed']:
                details = self.selection_summary['low_importance_details'].get(feat, {})
                importance = details.get('importance', feature_importance_map.get(feat, 'N/A'))
                category = details.get('category', 'Unknown')
                reason = details.get('reason', 'Low importance')

                if isinstance(importance, (int, float)):
                    importance_str = f"{importance:.6f}"
                else:
                    importance_str = str(importance)

                summary += f"| {feat} | {importance_str} | {category} | {reason} |\n"
            summary += "\n"

        # 2. Correlation-Based Removals
        if self.selection_summary['correlation_removed']:
            summary += "### Correlation-Based Features Removed\n\n"
            total_corr_removed = sum(len(feats) for feats in self.selection_summary['correlation_removed'].values())
            summary += f"**Count:** {total_corr_removed}\n\n"
            summary += "**Reason:** Features removed due to high correlation (> 0.7) with other features. Retained features with higher importance or lower total correlation.\n\n"

            for group_id, features in self.selection_summary['correlation_removed'].items():
                summary += f"**{group_id}:**\n\n"
                summary += "| Feature | Importance Score | Total Correlation | Detailed Reason |\n"
                summary += "|---------|-----------------|-------------------|----------------|\n"
                for feat in features:
                    details = self.selection_summary['correlation_details'].get(feat, {})
                    importance = details.get('importance', feature_importance_map.get(feat, 'N/A'))
                    total_corr = details.get('total_correlation', 'N/A')
                    reason = details.get('reason', 'High correlation within group')

                    if isinstance(importance, (int, float)):
                        importance_str = f"{importance:.6f}"
                    else:
                        importance_str = str(importance)

                    if isinstance(total_corr, (int, float)):
                        total_corr_str = f"{total_corr:.3f}"
                    else:
                        total_corr_str = str(total_corr)

                    summary += f"| {feat} | {importance_str} | {total_corr_str} | {reason} |\n"
                summary += "\n"

        # 3. Boruta Removals (if applied)
        if self.selection_summary['boruta_removed_details']:
            boruta_removed = list(self.selection_summary['boruta_removed_details'].keys())
            summary += "### Boruta Algorithm Removals\n\n"
            summary += f"**Count:** {len(boruta_removed)}\n\n"
            summary += "**Reason:** Features marked as 'Tentative' or 'Rejected' by Boruta algorithm (conservative 'Confirmed Only' strategy).\n\n"
            summary += "| Feature | Importance Score | Boruta Status | Rank | Detailed Reason |\n"
            summary += "|---------|-----------------|---------------|------|----------------|\n"
            for feat in boruta_removed:
                details = self.selection_summary['boruta_removed_details'].get(feat, {})
                importance = details.get('importance', feature_importance_map.get(feat, 'N/A'))
                status = details.get('status', 'Unknown')
                rank = details.get('rank', 'N/A')
                reason = details.get('reason', 'Boruta removal')

                if isinstance(importance, (int, float)):
                    importance_str = f"{importance:.6f}"
                else:
                    importance_str = str(importance)

                summary += f"| {feat} | {importance_str} | {status} | {rank} | {reason} |\n"
            summary += "\n"

        # 4. Summary table of all removals
        if self.selection_summary['features_removed']:
            summary += "### Complete Removal List\n\n"
            summary += f"**Total Features Removed:** {len(self.selection_summary['features_removed'])}\n\n"
            summary += "| # | Feature | Removal Method |\n"
            summary += "|---|---------|----------------|\n"

            for idx, feat in enumerate(self.selection_summary['features_removed'], 1):
                # Determine removal method
                if feat in self.selection_summary['low_importance_removed']:
                    method = "Low Importance"
                elif any(feat in feats for feats in self.selection_summary['correlation_removed'].values()):
                    method = "Correlation-Based"
                elif feat in boruta_removed:
                    method = "Boruta Algorithm"
                else:
                    method = "Other"

                summary += f"| {idx} | {feat} | {method} |\n"
            summary += "\n"

        # 5. Protected Attributes (identified but not removed)
        if self.selection_summary['protected_attributes']:
            summary += "### Protected Attributes (Identified, Not Removed)\n\n"
            summary += f"**Count:** {len(self.selection_summary['protected_attributes'])}\n\n"
            summary += "**Note:** These features were flagged as potentially sensitive (e.g., gender, race, age) but were NOT automatically removed.\n\n"
            for attr in self.selection_summary['protected_attributes']:
                summary += f"- {attr}\n"
            summary += "\n"

        return summary

    def _generate_summary(self) -> str:
        """
        Generate a human-readable summary of feature selection results.

        Returns:
            Formatted summary string
        """
        duration = (
            self.selection_summary['end_time'] - self.selection_summary['start_time']
        ).total_seconds()

        initial_features = self.selection_summary['initial_features']
        final_features = self.selection_summary['final_features']
        features_removed = len(self.selection_summary['features_removed'])
        reduction_percentage = (features_removed / initial_features * 100) if initial_features > 0 else 0

        summary = f"""
### Feature Selection Summary

**Duration:** {duration:.1f} seconds

**Steps Completed:** {len(self.selection_summary['steps_completed'])} / {len(self.selection_summary['steps_completed']) + len(self.selection_summary['steps_failed'])}

**Feature Changes:**
- Initial: {initial_features} features
- Final: {final_features} features
- Removed: {features_removed} features ({reduction_percentage:.1f}%)

**Key Actions:**
- Low importance features removed: {len(self.selection_summary['low_importance_removed'])}
- Correlation groups addressed: {len(self.selection_summary['correlation_removed'])}
- Protected attributes identified: {len(self.selection_summary['protected_attributes'])}
- Boruta algorithm applied: {'Yes' if self.selection_summary['boruta_applied'] else 'No'}
"""

        # Add detailed removal information
        if self.selection_summary['low_importance_removed']:
            summary += f"\n**Low Importance Features Removed ({len(self.selection_summary['low_importance_removed'])}):**\n"
            for feat in self.selection_summary['low_importance_removed']:
                summary += f"- {feat} (importance score < threshold)\n"

        if self.selection_summary['correlation_removed']:
            summary += f"\n**Correlation-Based Features Removed:**\n"
            for group_id, features in self.selection_summary['correlation_removed'].items():
                summary += f"- {group_id}: {', '.join(features)} (high correlation with group members)\n"

        if self.selection_summary['protected_attributes']:
            summary += f"\n**Protected Attributes Identified (not removed):**\n"
            for attr in self.selection_summary['protected_attributes']:
                summary += f"- {attr}\n"

        summary += "\n**Steps:**\n"

        for step in self.selection_summary['steps_completed']:
            summary += f"\n✅ {step['step']}"

        for step in self.selection_summary['steps_failed']:
            summary += f"\n❌ {step['step']} (failed)"

        return summary

    def _generate_detailed_report(self) -> str:
        """
        Generate a detailed markdown report of all feature selection operations.

        Returns:
            Detailed markdown report
        """
        report = "# Automated Feature Selection Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "## Summary\n\n"
        report += self._generate_summary()
        report += "\n\n## Detailed Results by Step\n\n"

        for step in self.selection_summary['steps_completed']:
            report += f"### {step['step']}\n\n"
            report += self._format_step_details(step['step'], step['details'])
            report += "\n"

        if self.selection_summary['steps_failed']:
            report += "## Failed Steps\n\n"
            for step in self.selection_summary['steps_failed']:
                report += f"### {step['step']}\n\n"
                report += f"**Error:** {step['error']}\n\n"

        # Add comprehensive feature removal summary
        report += "## Complete Feature Removal Summary\n\n"
        report += self._generate_feature_removal_summary()

        return report

    def _calculate_feature_stats(self, features_to_remove: List[str]) -> Dict[str, Dict]:
        """
        Calculate basic statistics for features before they are removed.
        
        Args:
            features_to_remove: List of feature names to calculate stats for
            
        Returns:
            Dictionary of feature stats keyed by feature name
        """
        stats = {}
        
        # Use training data for statistics
        df = self.builder.X_train
        
        for feature in features_to_remove:
            if feature not in df.columns:
                continue
                
            try:
                # Get series
                series = df[feature]
                
                # Basic descriptive stats
                feat_stats = {
                    'min': None,
                    'max': None,
                    'mean': None,
                    'std': None,
                    'missing_pct': float((series.isnull().sum() / len(series)) * 100),
                    'zero_pct': 0.0
                }
                
                # Handle numeric data
                if pd.api.types.is_numeric_dtype(series):
                    feat_stats['min'] = float(series.min()) if not pd.isna(series.min()) else None
                    feat_stats['max'] = float(series.max()) if not pd.isna(series.max()) else None
                    feat_stats['mean'] = float(series.mean()) if not pd.isna(series.mean()) else None
                    feat_stats['std'] = float(series.std()) if not pd.isna(series.std()) else None
                    
                    # Calculate zero percentage for numeric
                    if len(series) > 0:
                        zero_count = (series == 0).sum()
                        feat_stats['zero_pct'] = float((zero_count / len(series)) * 100)
                
                # Handle categorical/object data (limited stats)
                else:
                    # For categorical, min/max might correspond to alphabetical order or length
                    # but typically we just capture unique count or mode
                    try:
                        feat_stats['unique_count'] = series.nunique()
                        feat_stats['mode'] = str(series.mode().iloc[0]) if not series.mode().empty else None
                    except:
                        pass
                
                stats[feature] = feat_stats
                
            except Exception as e:
                # Don't fail the whole process if stats calculation fails for one feature
                stats[feature] = {'error': str(e)}
                
        return stats
