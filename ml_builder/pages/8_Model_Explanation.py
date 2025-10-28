import streamlit as st
import pandas as pd
from Builder import Builder, ModelStage
from app import create_sidebar_navigation
from utils.logging.journey_viewer import render_journey_viewer
from utils.logging.log_viewer import render_log_viewer
import numpy as np
from datetime import datetime
from components.data_preprocessing.categorical_encoding import CategoricalEncodingComponent
from components.model_explanation.FeatureAnalysis import render_feature_analysis
from components.model_explanation.individual_prediction import render_individual_prediction_analysis
from components.model_explanation.what_if_analysis import render_what_if_analysis
from components.model_explanation.fairness_analysis import render_fairness_analysis
from components.model_explanation.model_limitations import render_model_limitations
from utils.dataset_overview import DatasetOverviewComponent
from streamlit_scroll_to_top import scroll_to_here

def scroll():
    st.session_state.scroll_to_top = True

@st.cache_data
def get_model_explanations(_builder: Builder, train_data_hash: str, test_data_hash: str, model_hash: str, problem_type: str) -> tuple:
    """
    Cached function to get model explanations and SHAP values.
    Returns a tuple of (explanation_result, limitations_result, importance_df)
    
    Args:
        _builder: The Builder instance (prefixed with underscore to indicate it's not used for caching)
        train_data_hash: Hash of training data to invalidate cache when data changes
        test_data_hash: Hash of test data to invalidate cache when data changes  
        model_hash: Hash of model to invalidate cache when model changes
    """
    explanation_result = _builder.explain_model()
    limitations_result = {"success": False, "message": "Analysis not started"}
    importance_df = None

    if explanation_result["success"]:
        try:
            # Get limitations analysis results
            limitations_result = _builder.analyse_model_limitations()
            
            # Try to get feature importance data from multiple sources
            if limitations_result["success"] and "limitation_plots" in limitations_result:
                if "feature_importance" in limitations_result["limitation_plots"]:
                    try:
                        fig_data = limitations_result["limitation_plots"]["feature_importance"].data[0]
                        importance_df = pd.DataFrame({
                            'feature': fig_data.x,
                            'importance': fig_data.y
                        }).sort_values('importance', ascending=False)
                    except Exception as e:
                        st.warning(f"Could not process feature importance data from limitations analysis: {str(e)}")
                        importance_df = None
            
            # If we couldn't get importance from limitations, try direct feature importance analysis
            if importance_df is None:
                try:
                    importance_result = _builder.analyse_feature_importance()
                    if importance_result and importance_result["success"]:
                        importance_df = pd.DataFrame(importance_result["feature_scores"])
                except Exception as e:
                    st.warning(f"Could not process direct feature importance analysis: {str(e)}")
                    importance_df = None

        except Exception as e:
            st.error(f"Error in limitations analysis: {str(e)}")
            limitations_result = {
                "success": False,
                "message": f"Error in limitations analysis: {str(e)}"
            }

    return explanation_result, limitations_result, importance_df

def main():
    st.title("Model Explanation")
    
    # Add consistent navigation
    create_sidebar_navigation()
    
    # Initialize session state if needed
    if 'builder' not in st.session_state:
        st.session_state.builder = Builder()
        st.session_state.logger.log_stage_transition("START", "MODEL_EXPLANATION")
    
    # Set current stage to MODEL_EXPLANATION
    st.session_state.builder.current_stage = ModelStage.MODEL_EXPLANATION
    
    # Get and render stage info
    stage_info = st.session_state.builder.get_current_stage_info()

    # Enhanced logging for page state
    st.session_state.logger.log_page_state("Model_Explanation", {
        "stage": "MODEL_EXPLANATION",
        "model_available": bool(st.session_state.builder.model),
        "data_loaded": bool(st.session_state.get('data')),
        "target_selected": bool(st.session_state.get('target_column')),
        "evaluation_completed": st.session_state.builder.stage_completion[ModelStage.MODEL_EVALUATION]
    })
    
    if stage_info:
        st.header(stage_info.get("title", "Model Training"))
        st.write(stage_info.get("description", "Train your model with automated hyperparameter tuning."))
        
        with st.expander("Functionality"):
            requirements = stage_info.get("requirements", [])
            for req in requirements:
                if isinstance(req, dict) and "title" in req and "items" in req:
                    # Handle dictionary format requirements
                    st.markdown(f"**{req['title']}**")
                    for item in req['items']:
                        st.markdown(f"• {item}")
                else:
                    # Handle string format requirements
                    st.markdown(f"• {req}")
            
        with st.expander("Ethical Considerations"):
            considerations = stage_info.get("ethical_considerations", [])
            for consideration in considerations:
                if isinstance(consideration, dict) and "title" in consideration and "items" in consideration:
                    # Handle dictionary format considerations
                    st.markdown(f"**{consideration['title']}**")
                    for item in consideration['items']:
                        st.markdown(f"• {item}")
                else:
                    # Handle string format considerations
                    st.markdown(f"• {consideration}")
        st.header("Model Explanation Overview")

        with st.expander("📚 Understanding Model Explanations", expanded=True):
            st.markdown("""
                ### Why Model Explanations Matter
                
                Model explanations are crucial for:
                - **Transparency**: Understand how your model arrives at decisions
                - **Trust Building**: Help stakeholders understand and trust model predictions
                - **Compliance**: Meet regulatory requirements for model interpretability
                - **Debugging**: Identify and fix model biases or performance issues
                - **Improvement**: Make data-driven decisions to enhance model performance
                
                ### 🗺️ Navigation Guide
                
                **🔄 Feature Analysis**
                - Overall feature importance rankings
                - Detailed feature impact study
                        
                **🎯 Individual Predictions**
                - Case-by-case prediction analysis using the test dataset
                - Feature contribution breakdown
                - Interactive explanation visualisations
                
                **🔮 What-If Analysis**
                - Enter your own feature values
                - Get predictions and explanations
                
                **⚖️ Fairness Analysis**
                - Analyse model fairness across different protected attributes
                - Identify and mitigate bias
                - Ensure fair treatment for all groups
                
                **⚠️ Limitations & Recommendations**
                - Known model constraints
                - Potential improvement areas
                - Best practices and suggestions
            """)

    # Check for model evaluation completion
    if not st.session_state.builder.stage_completion[ModelStage.MODEL_EVALUATION]:
        st.error("⚠️ Please complete model evaluation first")
        st.session_state.logger.log_error(
            "Model Explanation Access Denied",
            {
                "reason": "Model evaluation not completed",
                "stage_completion": st.session_state.builder.stage_completion,
                "current_stage": str(st.session_state.builder.current_stage)
            }
        )
        if st.button("Return to Model Evaluation", key="return_to_eval_top"):
            st.session_state.builder.current_stage = ModelStage.MODEL_EVALUATION
            st.rerun()
        return
    
    # Set position to scroll to the top of the page
    if st.session_state.scroll_to_top:
        scroll_to_here(0, key='top')  # Scroll to the top of the page, 0 means instantly, but you can add a delay (im milliseconds)
        st.session_state.scroll_to_top = False  # Reset the state after scrolling

    st.markdown("""
        #### 📊 View Test Data Summary
        """)
    @st.dialog(title="Data Exploration", width="large")
    def data_summary_dialog():
        dataset_overview = DatasetOverviewComponent(st.session_state.builder.testing_data, st.session_state.logger)
        dataset_overview.display_overview()
    if st.button("Test Data Summary"):
        data_summary_dialog()

    with st.expander("Encoded and Binned Feature Details", expanded=False):
        st.write("""
            #### Encoded and Binned Feature Details
            
            This section displays details about how categorical and numerical features were transformed:
        """)
        
        # First show encoding details if available
        # Check for encoding mappings from both categorical encoding step and target encoding during data loading
        encoding_mappings = CategoricalEncodingComponent.get_encoding_mappings()
        
        # If no mappings from categorical encoding, try direct access from session state (for target encodings)
        if not encoding_mappings and "encoding_mappings" in st.session_state:
            encoding_mappings = st.session_state.encoding_mappings
        
        if encoding_mappings:
            st.write("##### Categorical Encoding Mappings")
            
            for column, mapping_info in encoding_mappings.items():
                st.write(f"📊 {column} - {mapping_info['method']}")
                if mapping_info["method"] in ["Label Encoding", "Target Encoding", "Target Label Encoding", "Target Label Encoding (Numeric to Categorical)"]:
                    try:
                        # Create a DataFrame to display the mappings
                        if mapping_info["method"] in ["Label Encoding", "Target Label Encoding", "Target Label Encoding (Numeric to Categorical)"]:
                            # For Label Encoding, create a clean display of original→encoded
                            original_values = mapping_info.get("original_values", [])
                            
                            # Get the encoded values directly from the mapping
                            encoded_values = []
                            mapping = mapping_info.get("mapping", {})
                            
                            for val in original_values:
                                # Try different key formats in case of type mismatches
                                encoded_val = mapping.get(str(val), mapping.get(val, "N/A"))
                                encoded_values.append(encoded_val)
                            
                            mapping_df = pd.DataFrame({
                                "Original Value": original_values,
                                "Encoded Value": encoded_values
                            })
                        else:
                            # For Target Encoding, use the existing approach
                            original_values = mapping_info.get("original_values", [])
                            mapping = mapping_info.get("mapping", {})
                            
                            mapping_df = pd.DataFrame({
                                "Original Value": original_values,
                                "Encoded Value": [mapping.get(str(val), mapping.get(val, "N/A")) 
                                                for val in original_values]
                            })
                        
                        # Only display if we have data to show
                        if len(mapping_df) > 0 and not mapping_df.empty:
                            # Display the mapping table with styling
                            st.dataframe(
                                mapping_df.style.background_gradient(cmap='Blues', axis=0),
                                width='stretch',
                                hide_index=True
                            )
                        else:
                            st.warning("No mapping data available to display")
                            
                    except Exception as e:
                        st.error(f"Error displaying mapping for {column}: {str(e)}")
                        # Show debug information
                        with st.expander("Debug Information"):
                            st.write("**Mapping Info Structure:**")
                            st.json(mapping_info)
                            st.write("**Error Details:**")
                            st.code(str(e))
                elif mapping_info["method"] == "One-Hot Encoding":
                    st.write("**Original Values:**", ", ".join(map(str, mapping_info["original_values"])))
                    st.write("**New Columns Created:**", ", ".join(mapping_info["new_columns"]))
        
        # Then show binning details if available
        st.write("##### Binned Features")
        
        if hasattr(st.session_state, 'binning_info'):
            # Get the list of features actually used in the model
            model_features = st.session_state.builder.X_train.columns.tolist()
            
            #model_binned_features = {
            #    binned_col: info 
            #    for binned_col, info in st.session_state.binning_info.items() 
            #    if (binned_col in model_features or 
            #        any(feature for feature in model_features 
            #            if '_' in feature and  # Check if feature contains underscore
            #            'binned' in '_'.join(feature.split('_')[:2]).lower() and  # Check first two components
            #            feature.startswith(binned_col)))  # Check if feature starts with binned_col
            #}
            model_binned_features = st.session_state.binning_info
            if model_binned_features:
                for binned_col, info in model_binned_features.items():
                    st.write(f"\n**{info['original_feature']}** (binned as {binned_col})")
                    
                    if info['is_categorical']:
                        # Display categorical bin mappings
                        for bin_id, categories in sorted(info['bin_ranges'].items()):
                            if isinstance(categories, list):
                                st.write(f"- Bin {bin_id}: {', '.join(categories)}")
                            else:
                                st.write(f"- Bin {bin_id}: {', '.join(str(cat) for cat in categories)}")
                    else:
                        # Display numeric bin ranges
                        if isinstance(info['bin_ranges'], list):
                            for i, range_info in enumerate(info['bin_ranges']):
                                if isinstance(range_info, (list, tuple)) and len(range_info) == 2:
                                    lower, upper = range_info
                                    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                                        if (isinstance(lower, float) and np.isinf(lower)) and (isinstance(upper, float) and np.isinf(upper)):
                                            range_str = "All values"
                                        elif isinstance(lower, float) and np.isinf(lower):
                                            range_str = f"≤ {upper:.2f}"
                                        elif isinstance(upper, float) and np.isinf(upper):
                                            range_str = f"> {lower:.2f}"
                                        else:
                                            range_str = f"{lower:.2f} to {upper:.2f}"
                                    else:
                                        range_str = f"{str(lower)} to {str(upper)}"
                                    st.write(f"- Bin {i}: {range_str}")
                                else:
                                    st.write(f"- Bin {i}: {range_info}")
                        elif isinstance(info['bin_ranges'], dict):
                            # Handle dictionary format for numeric bins (sometimes used)
                            for bin_id, range_info in sorted(info['bin_ranges'].items()):
                                if isinstance(range_info, (list, tuple)) and len(range_info) == 2:
                                    lower, upper = range_info
                                    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                                        if (isinstance(lower, float) and np.isinf(lower)) and (isinstance(upper, float) and np.isinf(upper)):
                                            range_str = "All values"
                                        elif isinstance(lower, float) and np.isinf(lower):
                                            range_str = f"≤ {upper:.2f}"
                                        elif isinstance(upper, float) and np.isinf(upper):
                                            range_str = f"> {lower:.2f}"
                                        else:
                                            range_str = f"{lower:.2f} to {upper:.2f}"
                                    else:
                                        range_str = f"{str(lower)} to {str(upper)}"
                                    st.write(f"- Bin {bin_id}: {range_str}")
                                else:
                                    # Just display the value directly
                                    st.write(f"- Bin {bin_id}: {range_info}")
                        else:
                            st.info(f"Bin ranges for {binned_col} have an unexpected format.")
            else:
                st.info("No binned features are being used in the final model.")
        else:
            st.info("No binning information available. Features may not have been binned during preprocessing.")
        
        # Show message if neither encoding nor binning information is available
        if not encoding_mappings and not hasattr(st.session_state, 'binning_info'):
            st.warning("No encoding or binning information is available. Features may not have been transformed.")

    st.write("---")
    # Create pills for different sections
    selected_section = st.pills(label="Select Explanation Method:", options=[
            "🔄 Feature Analysis",
            "🎯 Individual Predictions",
            "🔮 What-If Analysis",
            "⚖️ Fairness Analysis",
            "⚠️ Limitations & Recommendations"
        ],
        default="🔄 Feature Analysis"
    )
    st.write(" ")
    # Get model limitations analysis results first
    if st.session_state.builder.model is not None:
        with st.spinner("Generating model explanations..."):
            # Generate hashes for cache invalidation
            train_data_hash = str(pd.util.hash_pandas_object(st.session_state.builder.X_train).sum()) if st.session_state.builder.X_train is not None else "no_train_data"
            test_data_hash = str(pd.util.hash_pandas_object(st.session_state.builder.X_test).sum()) if st.session_state.builder.X_test is not None else "no_test_data"
            model_hash = str(hash(str(st.session_state.builder.model.get("type", "unknown")) + str(st.session_state.builder.model.get("hyperparameters", {}))))
            
            # Get problem type from session state
            problem_type = getattr(st.session_state, 'problem_type', st.session_state.builder.model.get("problem_type", "unknown"))
            
            # Use the cached function to get explanations
            explanation_result, limitations_result, importance_df = get_model_explanations(
                st.session_state.builder, 
                train_data_hash, 
                test_data_hash, 
                model_hash,
                problem_type
            )
            
            if explanation_result["success"]:
                # Enhanced logging for model explanation
                st.session_state.logger.log_calculation(
                    "Model Explanation Started",
                    {
                        "model_type": st.session_state.builder.model["type"],
                        "feature_count": len(st.session_state.builder.X_train.columns),
                        "explanation_method": explanation_result.get("explanation_method", "SHAP"),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                st.session_state.logger.log_calculation(
                    "Explanation Overview",
                    {
                        "global_feature_importance": explanation_result["shap_values"]["feature_importance"],
                        "model_complexity": explanation_result.get("model_complexity", "unknown"),
                        "feature_interactions": explanation_result.get("feature_interactions", {}),
                        "explanation_quality": explanation_result.get("explanation_quality", "unknown")
                    }
                )   

                # Render content based on selected pill
                if selected_section == "🔄 Feature Analysis":
                    render_feature_analysis(st.session_state.builder, explanation_result, importance_df, limitations_result)
                
                elif selected_section == "🎯 Individual Predictions":
                    render_individual_prediction_analysis()
                
                elif selected_section == "🔮 What-If Analysis":
                    render_what_if_analysis()
                
                elif selected_section == "⚖️ Fairness Analysis":
                    render_fairness_analysis()
                
                elif selected_section == "⚠️ Limitations & Recommendations":
                    render_model_limitations(st.session_state.builder, limitations_result, importance_df)

    else:
        st.error("No model found. Please complete the previous stages first.")
        st.session_state.logger.log_error(
            "Explanation Failed",
            {"reason": "No model found"}
        )

    st.write(" ")
    st.write(" ")
    nav_cols = st.columns([1, 2, 1])
    with nav_cols[2]:
        st.button("Scroll to Top", on_click=scroll)

    # Navigation buttons at the bottom
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Back to Evaluation", key="back_to_eval_bottom", type="primary"):
            st.session_state.logger.log_user_action("Navigation", {"direction": "back"})
            st.session_state.current_stage = ModelStage.MODEL_EVALUATION
            st.switch_page("pages/7_Model_Evaluation.py")
    with col2:
        st.success("✅ Model explanation completed successfully!")
    with col3:
        st.write(" ")
        if st.button("Proceed to Summary", type="primary"):
            # Log the transition to summary
            st.session_state.logger.log_stage_transition(
                "MODEL_EXPLANATION",
                "SUMMARY"
            )
            st.session_state.builder.stage_completion[ModelStage.MODEL_EXPLANATION] = True
            next_page = "9_Summary"
            st.switch_page(f"pages/{next_page}.py")

     # Flush logs
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