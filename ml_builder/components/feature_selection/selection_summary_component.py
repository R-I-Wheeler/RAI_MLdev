"""
Selection Summary Component for Feature Selection.

This component handles the final summary, tracking display, navigation,
and stage completion for the feature selection process.
"""

import streamlit as st
from typing import Dict, Any, Optional
from Builder import ModelStage
from utils.dataset_overview import DatasetOverviewComponent
from utils.dataframe_comparison import DataframeComparisonComponent
from utils.data_exploration_component import DataExplorationComponent

class SelectionSummaryComponent:
    """Handles selection summary, tracking, and navigation for feature selection."""

    def __init__(self, builder, logger=None, state_manager=None):
        """
        Initialize the selection summary component.

        Args:
            builder: Builder instance containing the dataset
            logger: Optional logger instance for tracking
            state_manager: Optional state manager for session state operations
        """
        self.builder = builder
        self.logger = logger
        self.state_manager = state_manager

    def render_dataset_review(self) -> None:
        """Render the dataset review section."""
        st.write("## 📋 Dataset Review")

        st.write("### Training Data Exploration")
        st.write(
            "NOTE: Using the data exploration component may cause the page to reload, any changes that you have applied will still be in effect.")

        @st.dialog(title="Data Exploration", width="large")
        def data_explorer_dialog():
            data_explorer = DataExplorationComponent(st.session_state.builder, st.session_state.logger,
                                                     data=st.session_state.builder.training_data,
                                                     target_column=st.session_state.builder.target_column)
            data_explorer.render()

        if st.button("Training Data Exploration", on_click=st.rerun):
            data_explorer_dialog()

        st.write("---")

        if self.builder.X_train is None:
            st.error("Training data not available for review.")
            return

        # Add toggle for switching between train and test sets
        preview_set = "Training" if st.toggle(
            "Switch off for Test Dataset Overview",
            value=True,
            help="Toggle between training and test dataset overviews, default is Training"
        ) else "Test"

        # Select the appropriate dataset based on toggle
        if preview_set == "Training":
            preview_X = self.builder.X_train
            preview_y = self.builder.y_train
        else:
            preview_X = self.builder.X_test
            preview_y = self.builder.y_test

        # Create preview data
        preview_data = preview_X.copy()
        preview_data[self.builder.target_column] = preview_y

        # Add custom CSS for better styled cards
        st.markdown("""
        <style>
        div.data-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            border-left: 4px solid #4e8df5;
        }
        div.data-card h4 {
            color: #1e3a8a;
            margin-top: 0;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Use the DatasetOverviewComponent to display dataset overview
        dataset_overview = DatasetOverviewComponent(
            preview_data, self.logger, keyidentifier="fs_dataset_overview"
        )
        dataset_overview.display_overview()

        # Display comparison with original data
        self._render_data_comparison()

    def _render_data_comparison(self) -> None:
        """Render comparison between original and current data."""
        if 'original_feature_data' in st.session_state:
            original_data = st.session_state.original_feature_data.get('training_data')
            if original_data is not None:
                comparison_component = DataframeComparisonComponent(
                    original_df=original_data,
                    modified_df=self.builder.training_data,
                    target_column=self.builder.target_column
                )
                comparison_component.render()

    def render_tracking_summary(self) -> None:
        """Render the feature selection tracking summary."""
        if not self.state_manager:
            return

        tracking_summary = self.state_manager.get_tracking_summary()

        with st.expander("🔍 Feature Selection Summary", expanded=False):
            st.write(f"**Initial Features:** {tracking_summary['initial_features']}")
            st.write(f"**Current Features:** {tracking_summary['current_features']}")
            st.write(f"**Manual Removals:** {tracking_summary['manual_removals']}")
            st.write(f"**Automated Removals:** {tracking_summary['automated_removals']}")
            st.write(f"**Total Removed:** {tracking_summary['total_removed']}")
            st.write(f"**Reduction:** {tracking_summary['reduction_percentage']:.1f}%")
            st.write(f"**Duplicates Removed:** {tracking_summary['duplicates_removed']}")
            st.write(f"**Correlation Addressed:** {tracking_summary['correlation_addressed']}")
            st.write(f"**Low Importance Addressed:** {tracking_summary['low_importance_addressed']}")
            st.write(f"**Methods Used:** {tracking_summary['methods_used']}")

    def render_navigation_and_summary(self) -> None:
        """Render the final navigation section and completion handling."""
        st.markdown("---")

        proceed_col1, proceed_col2 = st.columns([3, 2], vertical_alignment="bottom")

        with proceed_col1:
            st.write("Click 'Model Selection' to proceed with choosing and configuring your model.")
            self.render_tracking_summary()

        with proceed_col2:
            continue_enabled = self._check_continue_enabled()

            if continue_enabled:
                if st.button("Model Selection →", type="primary", width='stretch'):
                    self._handle_stage_completion()
            else:
                st.button(
                    "Model Selection →",
                    disabled=True,
                    help="Please address any remaining issues before continuing",
                    width='stretch'
                )

    def _check_continue_enabled(self) -> bool:
        """Check if the user can continue to the next stage."""
        # Basic validation checks
        if (self.builder.X_train is None or
            self.builder.X_test is None or
            self.builder.y_train is None or
            self.builder.y_test is None):
            return False

        # Check data consistency
        if (len(self.builder.X_train) != len(self.builder.y_train) or
            len(self.builder.X_test) != len(self.builder.y_test)):
            return False

        # Check that we have at least some features
        if len(self.builder.X_train.columns) == 0:
            return False

        return True

    def _handle_stage_completion(self) -> None:
        """Handle the completion of the feature selection stage."""
        # Get tracking data for completion metrics
        tracking_summary = self.state_manager.get_tracking_summary() if self.state_manager else {}

        # Prepare completion metrics
        completion_metrics = {
            "initial_features": tracking_summary.get('initial_features', 0),
            "final_features": tracking_summary.get('current_features', 0),
            "features_removed": tracking_summary.get('total_removed', 0),
            "removal_percentage": tracking_summary.get('reduction_percentage', 0),
            "removed_features_manual": tracking_summary.get('features_removed_manual', []),
            "removed_features_automated": tracking_summary.get('features_removed_automated', []),
            "removed_features_all": tracking_summary.get('features_removed_manual', []) + tracking_summary.get('features_removed_automated', []),
            "removal_methods_used": tracking_summary.get('methods_used', []),
            "quality_improvements": {
                "duplicates_removed": tracking_summary.get('duplicates_removed', False),
                "correlation_addressed": tracking_summary.get('correlation_addressed', False),
                "low_importance_addressed": tracking_summary.get('low_importance_addressed', False)
            }
        }

        # Mark stage as complete
        self.builder.stage_completion[ModelStage.FEATURE_SELECTION] = True

        # Log stage transition
        if self.logger:
            self.logger.log_stage_transition(
                "FEATURE_SELECTION",
                "MODEL_SELECTION",
                completion_metrics
            )

            # Log journey point
            # Format feature lists as strings to prevent truncation in journey viewer
            manual_features_str = ", ".join(completion_metrics["removed_features_manual"]) if completion_metrics["removed_features_manual"] else "None"
            automated_features_str = ", ".join(completion_metrics["removed_features_automated"]) if completion_metrics["removed_features_automated"] else "None"
            
            self.logger.log_journey_point(
                stage="FEATURE_SELECTION",
                decision_type="FEATURE_SELECTION",
                description="Feature selection completed",
                details={
                    "Initial Features": completion_metrics["initial_features"],
                    "Final Features": completion_metrics["final_features"],
                    "Features Removed": completion_metrics["features_removed"],
                    "Features Removed (Manual)": manual_features_str,
                    "Features Removed (Automated)": automated_features_str,
                    "Removal Methods Used": completion_metrics["removal_methods_used"],
                    "Quality Improvements": completion_metrics["quality_improvements"],
                    "Training Data Shape": self.builder.training_data.shape,
                    "Testing Data Shape": self.builder.testing_data.shape,
                },
                parent_id=None
            )

        # Navigate to next page
        st.switch_page("pages/5_Model_Selection.py")

    def render_progress_indicator(self) -> None:
        """Render a progress indicator for the feature selection process."""
        if not self.state_manager:
            return

        current_step = self.state_manager.get_current_step()
        total_steps = 3
        steps = ["Analysis", "Selection", "Review"]

        # Log progress state
        if self.logger:
            self.logger.log_page_state(
                "Feature_Selection_Progress",
                {
                    "current_step": current_step,
                    "steps_total": total_steps,
                    "steps": steps,
                    "completion_percentage": (current_step / total_steps) * 100
                }
            )

        # You could add a visual progress indicator here if desired
        # For now, we'll keep it simple and just log the state

    def get_completion_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the feature selection completion status.

        Returns:
            Dictionary with completion summary
        """
        if not self.state_manager:
            return {"error": "State manager not available"}

        tracking_summary = self.state_manager.get_tracking_summary()
        continue_enabled = self._check_continue_enabled()

        return {
            "stage_complete": continue_enabled,
            "tracking_summary": tracking_summary,
            "data_shapes": {
                "training_shape": self.builder.training_data.shape if self.builder.training_data is not None else None,
                "testing_shape": self.builder.testing_data.shape if self.builder.testing_data is not None else None,
                "X_train_shape": self.builder.X_train.shape if self.builder.X_train is not None else None,
                "X_test_shape": self.builder.X_test.shape if self.builder.X_test is not None else None
            },
            "validation_status": {
                "data_consistency": (
                    len(self.builder.X_train) == len(self.builder.y_train) and
                    len(self.builder.X_test) == len(self.builder.y_test)
                ) if self.builder.X_train is not None else False,
                "has_features": len(self.builder.X_train.columns) > 0 if self.builder.X_train is not None else False,
                "data_available": all([
                    self.builder.X_train is not None,
                    self.builder.X_test is not None,
                    self.builder.y_train is not None,
                    self.builder.y_test is not None
                ])
            }
        }

    def render_error_recovery_options(self, error_message: str) -> None:
        """
        Render error recovery options when issues are detected.

        Args:
            error_message: Description of the error
        """
        st.error(f"⚠️ Issue Detected: {error_message}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Reset to Original Features"):
                if self.state_manager:
                    reset_result = self.state_manager.reset_to_original_features()
                    if reset_result["success"]:
                        st.success("✅ Reset successful!")
                        st.rerun()
                    else:
                        st.error(f"Reset failed: {reset_result['message']}")

        with col2:
            if st.button("⬅️ Return to Preprocessing"):
                self.builder.current_stage = ModelStage.DATA_PREPROCESSING
                if self.logger:
                    self.logger.log_stage_transition("FEATURE_SELECTION", "DATA_PREPROCESSING")
                st.switch_page("pages/3_Data_Preprocessing.py")

    def validate_final_state(self) -> Dict[str, Any]:
        """
        Validate the final state before allowing progression.

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "success": True,
            "errors": [],
            "warnings": []
        }

        # Check data availability
        if self.builder.X_train is None:
            validation_results["success"] = False
            validation_results["errors"].append("Training features not available")

        if self.builder.X_test is None:
            validation_results["success"] = False
            validation_results["errors"].append("Testing features not available")

        if self.builder.y_train is None:
            validation_results["success"] = False
            validation_results["errors"].append("Training targets not available")

        if self.builder.y_test is None:
            validation_results["success"] = False
            validation_results["errors"].append("Testing targets not available")

        # Check data consistency
        if (self.builder.X_train is not None and self.builder.y_train is not None and
            len(self.builder.X_train) != len(self.builder.y_train)):
            validation_results["success"] = False
            validation_results["errors"].append("Training data length mismatch")

        if (self.builder.X_test is not None and self.builder.y_test is not None and
            len(self.builder.X_test) != len(self.builder.y_test)):
            validation_results["success"] = False
            validation_results["errors"].append("Testing data length mismatch")

        # Check feature count
        if (self.builder.X_train is not None and len(self.builder.X_train.columns) == 0):
            validation_results["success"] = False
            validation_results["errors"].append("No features remaining after selection")

        # Add warnings for potential issues
        if self.state_manager:
            tracking_summary = self.state_manager.get_tracking_summary()
            if tracking_summary["reduction_percentage"] > 90:
                validation_results["warnings"].append(
                    f"Very high feature reduction: {tracking_summary['reduction_percentage']:.1f}%"
                )

        return validation_results