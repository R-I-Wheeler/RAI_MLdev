"""
Feature Selection Page - Refactored Version

This module provides a streamlined interface for feature selection with
improved component architecture and separation of concerns.
"""

import streamlit as st
from Builder import Builder, ModelStage
from app import create_sidebar_navigation
from utils.logging.journey_viewer import render_journey_viewer
from utils.logging.log_viewer import render_log_viewer
from datetime import datetime

# Import new components
from components.feature_selection.feature_selection_state_manager import FeatureSelectionStateManager
from components.feature_selection.dataset_validation_component import DatasetValidationComponent
from components.feature_selection.feature_analysis_component import FeatureAnalysisComponent
from components.feature_selection.manual_selection_component import ManualSelectionComponent
from components.feature_selection.selection_summary_component import SelectionSummaryComponent
from components.feature_selection.boruta_feature_selection import AutomatedFeatureSelectionComponent
from components.feature_selection.utils.tracking_utils import track_automated_feature_removal


def render_page_header():
    """Render the page header and information."""
    # Get and render stage info
    stage_info = st.session_state.builder.get_current_stage_info()

    # Edge case handling: occasionally stage_info may not be ready on first call (e.g., on refresh)
    if not stage_info:
        if 'logger' in st.session_state:
            st.session_state.logger.log_error(
                "Stage Info Fetch - Retry",
                {
                    "stage": "FEATURE_SELECTION",
                    "reason": "stage_info was None or empty on first attempt",
                    "timestamp": datetime.now().isoformat()
                }
            )
        # Retry fetching once
        stage_info = st.session_state.builder.get_current_stage_info()

    # As a final safeguard, ensure stage_info has the expected structure
    if not stage_info:
        stage_info = {
            "title": "Feature Selection",
            "description": "Analyse feature importance, detect bias, and select an optimal subset of features before model training.",
            "requirements": [],
            "ethical_considerations": []
        }

    st.header(stage_info["title"])
    st.write(stage_info["description"])

    with st.expander("Functionality"):
        # Check if requirements is a list of strings or a list of formatted strings
        for req in stage_info["requirements"]:
            if "**" in req:  # If requirement is already formatted with markdown
                st.markdown(req)
            else:
                st.write(f"• {req}")

    with st.expander("Ethical Considerations"):
        # Check if ethical considerations is a list of strings or a list of formatted strings
        for consideration in stage_info["ethical_considerations"]:
            if "**" in consideration:  # If consideration is already formatted with markdown
                st.markdown(consideration)
            else:
                st.write(f"• {consideration}")

    # Add process overview and explanation
    st.markdown("""
    ### Process Overview

    This page will guide you through:

    1. **📊 Feature Analysis**
       - Importance scoring of each feature
       - Correlation analysis
       - Data quality assessment
       - Bias detection

    2. **🔍 Feature Selection**
       - Review and select features to remove
       - Visualize impact of selection
       - Compare before/after metrics
       - Duplicate detection and removal is performed before the data is used for training and testing the model

    4. **👀 Final Dataset Review**
       - Review the final dataset after all preprocessing steps have been completed
    """)


def render_navigation_pills():
    """Render navigation pills and return selected section."""
    st.write("---")

    # Create enhanced navigation using pills instead of tabs
    selected_section = st.pills(
        label="Navigate through the feature selection process:",
        options=[
            "📊 Analysis Results",
            "🎯 Feature Selection",
            "👀 Dataset Review"
        ],
        default="📊 Analysis Results"
    )


    return selected_section


def render_footer_components():
    """Render footer components including journey viewer and log viewer."""
    # At the end of each page's script
    if 'logger' in st.session_state:
        st.session_state.logger.flush_logs()

    render_journey_viewer(expanded=True)
    st.write("---")

    # Add log viewer before flushing logs
    render_log_viewer()

    # Bottom footer with version and copyright
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666666; padding: 10px;'>
        <small>Version 1.0.0 | Copyright © 2025, Richard Wheeler. All rights reserved.</small><br>
        <small>ML Model Development Guide</small>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    """Main function for the Feature Selection page."""
    # Initialize state manager and components
    state_manager = FeatureSelectionStateManager(
        st.session_state.builder,
        st.session_state.get('logger')
    )
    state_manager.initialize_session_state()

    # Make the tracking function available to other components
    st.session_state.track_automated_feature_removal = track_automated_feature_removal

    # Add consistent navigation
    create_sidebar_navigation()

    # Initialize builder if needed
    if 'builder' not in st.session_state:
        st.session_state.builder = Builder()

    # Ensure logger exists before using it
    if 'logger' in st.session_state:
        st.session_state.logger.log_stage_transition("START", "FEATURE_SELECTION")

    # Set current stage to FEATURE_SELECTION
    st.session_state.builder.current_stage = ModelStage.FEATURE_SELECTION

    # Log page access and state
    state_manager.log_page_state({
        "timestamp": datetime.now().isoformat(),
        "data_loaded": bool(st.session_state.get('data')),
        "target_selected": bool(st.session_state.get('target_column'))
    })

    # Render progress indicator
    state_manager.render_progress_indicator()

    # Validate prerequisites using validation component
    validator = DatasetValidationComponent(
        st.session_state.builder,
        st.session_state.get('logger')
    )
    if not validator.validate_and_handle_errors():
        return

    # Render page header and process overview
    render_page_header()

    # Handle main functionality based on selected section
    selected_section = render_navigation_pills()

    # Initialize components with shared state
    feature_analysis = FeatureAnalysisComponent(
        st.session_state.builder,
        st.session_state.get('logger')
    )
    manual_selection = ManualSelectionComponent(
        st.session_state.builder,
        st.session_state.get('logger'),
        state_manager
    )
    selection_summary = SelectionSummaryComponent(
        st.session_state.builder,
        st.session_state.get('logger'),
        state_manager
    )
    automated_selection = AutomatedFeatureSelectionComponent(
        st.session_state.builder,
        st.session_state.get('logger')
    )

    # Render content based on selected section
    if selected_section == "📊 Analysis Results":
        # Run feature analysis and display results
        analysis_success = feature_analysis.render()
        if not analysis_success:
            return

    elif selected_section == "🎯 Feature Selection":
        # Check if analysis has been run before, if not run it silently
        if feature_analysis.analysis_result is None:
            feature_analysis.analysis_result = st.session_state.builder.analyse_feature_importance()

        if not feature_analysis.analysis_result.get("success", False):
            st.error("Feature analysis failed. Please return to Analysis Results to investigate.")
            return

        # Get analysis data and pass to manual selection component
        feature_scores = feature_analysis.get_feature_scores()
        correlations = feature_analysis.get_correlations()
        protected_attributes = feature_analysis.get_protected_attributes()

        if feature_scores is not None:
            manual_selection.set_analysis_data(feature_scores, correlations, protected_attributes)
            manual_selection.render()

        # Render automated selection component
        st.write("---")
        st.subheader("Automated Feature Selection")
        automated_selection.render()

    elif selected_section == "👀 Dataset Review":
        # Disable automated selection when in review mode
        state_manager.set_auto_selection_active(False)

        # Render dataset review
        selection_summary.render_dataset_review()

    # Always render final navigation and summary
    selection_summary.render_navigation_and_summary()

    # Render footer components
    render_footer_components()


if __name__ == "__main__":
    main()