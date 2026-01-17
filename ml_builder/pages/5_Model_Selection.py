import streamlit as st
from Builder import Builder, ModelStage
from app import create_sidebar_navigation
from utils.logging.journey_viewer import render_journey_viewer
from utils.logging.log_viewer import render_log_viewer
from datetime import datetime
from components.model_selection.model_selection_interface import (
    render_model_explainer_section,
    render_performance_metrics_explainer,
    display_model_information
)
from components.model_selection.render_automated_model_selection_training import render_automated_model_selection_training
from utils.data_exploration_component import DataExplorationComponent

def main():
    st.title("Model Selection")

    # Add consistent navigation
    create_sidebar_navigation()

    # Check if we should navigate to training (after state is persisted)
    if st.session_state.get('navigate_to_training', False):
        st.session_state.navigate_to_training = False
        st.switch_page("pages/6_Model_Training.py")

    # Initialize session state if needed
    if 'builder' not in st.session_state:
        st.session_state.builder = Builder()
        st.session_state.logger.log_stage_transition("START", "MODEL_SELECTION")

    # Check for return navigation to clear automation results
    # If the user is navigating to this page (current_stage is not MODEL_SELECTION yet)
    # and automation was previously completed, we should clear the results to allow a fresh start
    if (st.session_state.builder.current_stage != ModelStage.MODEL_SELECTION and 
        st.session_state.get('automated_model_selection_training_completed', False)):
        
        # Clear automation state
        if 'automated_model_selection_training_completed' in st.session_state:
            del st.session_state.automated_model_selection_training_completed
        if 'automated_model_selection_training_result' in st.session_state:
            del st.session_state.automated_model_selection_training_result

        # Reset model selection and training stages
        st.session_state.builder.stage_completion[ModelStage.MODEL_SELECTION] = False
        st.session_state.builder.stage_completion[ModelStage.MODEL_TRAINING] = False

        # Clear training state
        keys_to_clear = [
            'training_complete',
            'training_results', 
            'selected_model_type',
            'selected_model_stability',
            'previous_model_selection'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        # Clear builder model state
        st.session_state.builder.model = None

        # Log the auto-clear action
        st.session_state.logger.log_user_action(
            "Automated Model Selection & Training Results Cleared",
            {"action": "auto_clear_on_return", "reason": "return_navigation"}
        )
        
        # Display a toast to inform the user
        st.toast("Previous automation results cleared for new selection", icon="🔄")

    # Set current stage to MODEL_SELECTION
    st.session_state.builder.current_stage = ModelStage.MODEL_SELECTION

    # Get and render stage info
    stage_info = st.session_state.builder.get_current_stage_info()

    # Edge case handling: occasionally stage_info may not be ready on first call (e.g., on refresh)
    if not stage_info:
        if 'logger' in st.session_state:
            st.session_state.logger.log_error(
                "Stage Info Fetch - Retry",
                {
                    "stage": "MODEL_SELECTION",
                    "reason": "stage_info was None or empty on first attempt",
                    "timestamp": datetime.now().isoformat()
                }
            )
        # Retry fetching once
        stage_info = st.session_state.builder.get_current_stage_info()

    # As a final safeguard, ensure stage_info has the expected structure
    if not stage_info:
        stage_info = {
            "title": "Model Selection",
            "description": "Compare candidate algorithms, review quick comparison results, and choose a model (or use automated selection) before training.",
            "requirements": [],
            "ethical_considerations": []
        }

    # Enhanced logging of page state with more details
    st.session_state.logger.log_page_state("Model_Selection", {
        "data_loaded": bool(st.session_state.get('data')),
        "target_selected": bool(st.session_state.get('target_column')),
        "current_stage": str(ModelStage.MODEL_SELECTION),
        "stage_completed": st.session_state.builder.stage_completion.get(ModelStage.MODEL_SELECTION, False)
    })

    st.header(stage_info["title"])
    st.write(stage_info["description"])

    with st.expander("Functionality"):
        for req in stage_info["requirements"]:
            if req.startswith("**"):
                st.markdown(req)
            else:
                st.markdown(f"• {req}")

    with st.expander("Ethical Considerations"):
        for consideration in stage_info["ethical_considerations"]:
            if consideration.startswith("**"):
                st.markdown(consideration)
            else:
                st.markdown(f"• {consideration}")

    # Get problem type and compatibility info
    problem_type = st.session_state.builder.detect_problem_type()
    xgboost_compatible = st.session_state.builder.check_xgboost_compatibility()
    problem_type_display = st.session_state.builder.get_problem_type_display()

    st.info(f"Detected problem type: {problem_type_display}")

    # Display XGBoost compatibility warning if needed
    if not xgboost_compatible:
        st.warning(
            "⚠️ **XGBoost Compatibility Issue**: Your target classes don't start from 0, which is required by XGBoost for multiclass classification. "
            f"Your classes are: {sorted(st.session_state.builder.training_data[st.session_state.builder.target_column].unique())}. "
            "XGBoost has been excluded from the available models. Consider relabeling your target classes to start from 0 if you specifically need XGBoost."
        )

    # Log problem type detection
    st.session_state.logger.log_calculation(
        "Problem Type Detection",
        {
            "type": problem_type,
            "timestamp": str(datetime.now()),
            "stage": "MODEL_SELECTION",
            "xgboost_compatible": xgboost_compatible
        }
    )

    # Get model options
    model_options = st.session_state.builder.get_model_options(xgboost_compatible)

    if not model_options:
        st.error("Unable to determine problem type. Please ensure data is properly loaded.")
        st.session_state.logger.log_error(
            "Model Selection Failed",
            {"reason": f"Unknown problem type: {problem_type}"}
        )
        return

    # Render comprehensive model explainer section
    render_model_explainer_section(st.session_state.builder.content_manager)

    # Display training data exploration widget
    st.write("---")
    st.write("### Training Data Exploration")
    st.write("NOTE: Using the data exploration component may cause the page to reload, any changes that you have applied will still be in effect.")

    @st.dialog(title="Data Exploration", width="large")
    def data_explorer_dialog():
        data_explorer = DataExplorationComponent(st.session_state.builder, st.session_state.logger,
                                                 data=st.session_state.builder.training_data,
                                                 target_column=st.session_state.builder.target_column)
        data_explorer.render()

    if st.button("Training Data Exploration", on_click=st.rerun):
        data_explorer_dialog()

    st.write("---")

    # Model recommendation section
    st.write("### Model Recommendation")
    recommended_model = None
    if st.session_state.builder.training_data is not None:
        # Get model recommendation
        recommendation = st.session_state.builder.get_model_recommendation()

        recommended_model = recommendation["recommended_model"]
        recommendation_reasons = recommendation["reasons"]

        # Display recommendation
        st.info(f"📊 Recommended Model: {model_options[recommended_model]}")
        st.write("Reasoning:")
        for reason in recommendation_reasons:
            st.write(f"• {reason}")

        # Log the recommendation
        st.session_state.logger.log_recommendation(
            "Model Recommendation",
            recommendation
        )

        st.write("---")

    # Quick Model Comparison Section
    st.write("### Quick Model Comparison")

    # Get explanation from content manager
    comparison_explanation = st.session_state.builder.content_manager.get_calculation_explanation("model_comparison_metrics")
    st.write(comparison_explanation.get("method", ""))

    # Add a warning about the sample size
    st.warning(comparison_explanation.get("interpretation", ""))

    # Add comprehensive metrics explainer
    render_performance_metrics_explainer(st.session_state.builder.content_manager, problem_type)

    # Perform quick comparison
    with st.spinner("Running quick model comparison..."):
        # Get the comparison results using training and testing data from session state
        results_df = st.session_state.builder.get_quick_model_comparison(
            sample_size=1000,
            exclude_xgboost=not xgboost_compatible
        )

        # Find the best performing model
        best_model, best_score, best_metric = st.session_state.builder.get_best_model_from_comparison(results_df)

        # Log the comparison results
        st.session_state.logger.log_calculation(
            "Quick Model Comparison",
            {
                "problem_type": problem_type,
                "sample_sizes": {
                    "training": len(st.session_state.builder.training_data),
                    "testing": len(st.session_state.builder.testing_data)
                },
                "best_model": best_model,
                "best_metric": best_metric,
                "best_score": best_score,
                "all_results": results_df.to_dict('records')
            }
        )

        # Style and display the results
        styled_results = st.session_state.builder.style_comparison_results(results_df, best_metric)
        st.dataframe(styled_results, width='stretch')

    st.write("---")

    # Model selection interface
    # Check if recommended model is available in the options
    if recommended_model and recommended_model in model_options:
        default_index = list(model_options.keys()).index(recommended_model)
    else:
        default_index = 0  # Default to first available model

    selected_model = st.selectbox(
        "Choose a model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=default_index
    )

    # Display model information
    if selected_model:
        display_model_information(st.session_state.builder.content_manager, selected_model)

        # Enhanced logging for model selection
        selection_details = {
            "model": selected_model,
            "matches_recommendation": selected_model == recommended_model if recommended_model else False,
            "problem_type": problem_type,
            "timestamp": str(datetime.now()),
        }

        st.session_state.logger.log_user_action(
            "Model Selected",
            {**selection_details, "xgboost_compatible": xgboost_compatible}
        )

    # Automated Model Selection & Training Section
    render_automated_model_selection_training()

    # Navigation
    st.markdown("---")

    # Check if automated workflow was completed
    automated_completed = st.session_state.get('automated_model_selection_training_completed', False)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Back to Feature Selection", key="back_to_feature_selection_bottom", type="primary"):
            st.session_state.logger.log_user_action("Navigation", {"direction": "back"})
            st.session_state.current_stage = ModelStage.FEATURE_SELECTION
            st.switch_page("pages/4_Feature_Selection.py")
    with col2:
        st.write(" ")
    with col3:
        # Dynamic button text and navigation based on automation status
        if automated_completed:
            button_text = "Continue to Model Evaluation"
            next_page = "7_Model_Evaluation"
        else:
            button_text = "Select Model and Continue"
            next_page = "6_Model_Training"

        if st.button(button_text, type="primary"):
            if automated_completed:
                # Automated workflow already completed - go directly to evaluation
                st.session_state.logger.log_user_action(
                    "Navigation",
                    {"direction": "forward", "to_stage": "MODEL_EVALUATION", "automated": True}
                )
                st.switch_page(f"pages/{next_page}.py")
            else:
                # Manual workflow - select model and go to training
                result = st.session_state.builder.select_model(selected_model)
                if result["success"]:
                    # Set stage completion flag
                    st.session_state.builder.stage_completion[ModelStage.MODEL_SELECTION] = True
                    
                    # Set a navigation flag to trigger page switch after rerun
                    st.session_state.navigate_to_training = True

                    # Enhanced logging for successful model selection
                    st.session_state.logger.log_stage_transition(
                        "MODEL_SELECTION",
                        "MODEL_TRAINING"
                    )

                    # Log final model selection details
                    st.session_state.logger.log_calculation(
                        "Final Model Selection",
                        {
                            "selected_model": selected_model,
                            "problem_type": problem_type,
                            "stage_completion": True,
                            "timestamp": str(datetime.now())
                        }
                    )

                    st.session_state.logger.log_journey_point(
                        stage="MODEL_SELECTION",
                        decision_type="MODEL_SELECTION",
                        description="Model selected",
                        details={"Problem Type": problem_type,
                                "Model Type": selected_model,
                                "Matches Recommendation": selected_model == recommended_model if recommended_model else False,
                                },
                        parent_id=None
                    )
                    
                    # Show success message and rerun to persist state
                    st.success(result["message"])
                    st.rerun()
                else:
                    st.error(result["message"])
                    # Enhanced error logging
                    st.session_state.logger.log_error(
                        "Model Selection Failed",
                        {
                            "error": result["message"],
                            "selected_model": selected_model,
                            "problem_type": problem_type,
                            "timestamp": str(datetime.now())
                        }
                    )

    # At the end of each page's script
    if 'logger' in st.session_state:
        st.session_state.logger.flush_logs()

    render_journey_viewer(expanded=True)
    st.write("---")

    # Add log viewer
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


if __name__ == "__main__":
    main()