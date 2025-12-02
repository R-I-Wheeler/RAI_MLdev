import streamlit as st
import numpy as np
import pandas as pd

# Import visualization utilities
from components.model_evaluation.evaluation_utils.eval_visualization_utils import (
    create_precision_recall_curve,
    create_multiclass_probability_distribution_plot,
    create_probability_distribution_plot,
    create_classification_error_by_confidence_plot,
    create_classification_confusion_by_features_plot,
    has_feature_importance_support,
    get_feature_importance_from_model,
    create_pure_feature_importance_plot,
    create_error_by_prediction_range_plot,
    create_cooks_distance_plot,
    create_prediction_intervals_plot,
    can_provide_confidence_intervals,
    highlight_high_influence
)

# All visualization creation functions are now imported from utils.visualization_utils

# ===== PERFORMANCE OPTIMIZATION: Caching Functions =====

@st.cache_data(show_spinner=False)
def get_cached_predictions(_model, X_test, X_test_hash):
    """Cache predictions for the test set."""
    return _model.predict(X_test)

@st.cache_data(show_spinner=False)
def get_cached_probabilities(_model, X_test, X_test_hash, has_proba):
    """Cache probability predictions for the test set."""
    if has_proba:
        return _model.predict_proba(X_test)
    return None

def get_data_hash(df):
    """Generate hash for DataFrame to use as cache key."""
    # Use shape and a sample of values for faster hashing
    return hash((df.shape, tuple(df.columns), df.iloc[0, 0] if len(df) > 0 else 0))

def create_feature_importance_plot(model, feature_names):
    """Create feature importance plot with session state logging wrapper."""
    try:
        # First use the pure utility function to create the plot
        fig = create_pure_feature_importance_plot(model, feature_names)
        
        if fig is None:
            return None
            
        # Add logging for journey tracking
        importance_values, display_type = get_feature_importance_from_model(model)
        if importance_values is not None:
            # Ensure we have the right number of feature names
            if len(feature_names) != len(importance_values):
                feature_names = [f"Feature_{i}" for i in range(len(importance_values))]
            
            # Create DataFrame for logging
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_values
            }).sort_values('Importance', ascending=True)
            
            # Log journey point for feature importance
            st.session_state.logger.log_journey_point(
                stage="MODEL_EVALUATION",
                decision_type="MODEL_EVALUATION",
                description="Model Feature Importance",
                details={"Model Type": st.session_state.builder.model['type'],
                        "Feature Importance": importance_df.to_dict(orient='records')},
                parent_id=None
            )
        
        return fig
        
    except Exception as e:
        print(f"Error creating feature importance plot: {e}")
        return None



# create_classification_error_by_confidence_plot is imported from utils

# create_classification_confusion_by_features_plot is imported from utils

# has_feature_importance_support is imported from utils

# get_feature_importance_from_model is imported from utils

# The original create_feature_importance_plot is replaced by our wrapper above

# create_error_by_prediction_range_plot is imported from utils

# create_cooks_distance_plot is imported from utils

# create_prediction_intervals_plot is imported from utils

# create_prediction_intervals_plot function body removed - using utils version
       
def display_classification_visualisations(result):
    # Get model and test data for additional visualizations
    model_instance = st.session_state.builder.model.get("active_model") or st.session_state.builder.model["model"]
    X_test = st.session_state.builder.X_test
    y_test = st.session_state.builder.y_test

    # ===== PERFORMANCE: Use cached predictions =====
    X_test_hash = get_data_hash(X_test)
    has_predict_proba = hasattr(model_instance, 'predict_proba')

    # Get cached predictions
    y_pred_cached = get_cached_predictions(model_instance, X_test, X_test_hash)

    # Get cached probabilities if available
    y_prob_matrix = get_cached_probabilities(model_instance, X_test, X_test_hash, has_predict_proba)

    # Check model capabilities
    has_feature_importance = has_feature_importance_support(model_instance)

    # Extract probability values for visualization
    y_prob = None
    if y_prob_matrix is not None:
        try:
            if y_prob_matrix.shape[1] == 2:  # Binary classification
                y_prob = y_prob_matrix[:, 1]
            else:  # Multi-class classification
                # For multi-class, we'll use the max probability for visualization
                y_prob = np.max(y_prob_matrix, axis=1)
        except:
            has_predict_proba = False
            y_prob_matrix = None
    
    # Determine which tabs to show based on model capabilities
    available_tabs = ["Confusion Matrix", "ROC Curve"]
    
    if has_predict_proba and y_prob is not None:
        available_tabs.extend(["Precision-Recall", "Probability Distribution"])
    
    # Always add Error Analysis as it works with basic predictions
    available_tabs.append("Error Analysis")

    if has_feature_importance:
        available_tabs.append("Feature Importance")
    
    # Create tabs for available visualisations
    #viz_tabs = st.tabs(available_tabs)
    viz_tabs = st.pills(label="Select Visualisation Method:", options=available_tabs, default="Confusion Matrix")

    # Tab 0: Confusion Matrix (always available)
    #with viz_tabs[0]:
    if viz_tabs=="Confusion Matrix":
        st.write("### 🎯 Confusion Matrix")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(result["confusion_matrix"], config={'responsive': True}, key="viz_confusion_matrix_plot")
        with col2:
            explanation = st.session_state.builder.get_calculation_explanation("confusion_matrix")
            with st.container():
                st.markdown("#### What am I looking at?")
                st.markdown("""---""")
                st.markdown(explanation["method"])
                with st.expander("📖 Detailed Interpretation"):
                    st.markdown("""---""")
                    st.markdown("**Key Points:**")
                    st.markdown(explanation["interpretation"])
    
    # Tab 1: ROC Curve (always available)
    #with viz_tabs[1]:
    elif viz_tabs=="ROC Curve":
        st.write("### 📈 ROC Curve")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(result["roc_curve"], config={'responsive': True}, key="roc_curve_plot")
        with col2:
            explanation = st.session_state.builder.get_calculation_explanation("roc_curve")
            with st.container():
                st.markdown("#### What am I looking at?")
                st.markdown("""---""")
                st.markdown(explanation["method"])
                with st.expander("📖 Detailed Interpretation"):
                    st.markdown("""---""")
                    st.markdown("**Key Points:**")
                    st.markdown(explanation["interpretation"])
    
    # Additional tabs only if model supports probability predictions
    if has_predict_proba and y_prob is not None:
        # Precision-Recall tab
        precision_recall_index = available_tabs.index("Precision-Recall")
        #with viz_tabs[precision_recall_index]:
        if viz_tabs=="Precision-Recall":
            st.write("### 📊 Precision-Recall Curve")
            col1, col2 = st.columns([2, 1])
            with col1:
                try:
                    # For multi-class, we need to handle differently
                    if len(np.unique(y_test)) > 2:
                        st.info("Precision-Recall curve is showing performance for the positive class in a one-vs-rest manner for multi-class classification.")
                        # Convert to binary for PR curve (positive class vs rest)
                        from sklearn.preprocessing import LabelBinarizer
                        lb = LabelBinarizer()
                        y_test_binary = lb.fit_transform(y_test)
                        if y_test_binary.shape[1] > 1:
                            # Take the first class as positive
                            y_test_binary = y_test_binary[:, 0]
                            y_prob_binary = y_prob_matrix[:, 0]  # Use cached probabilities
                        else:
                            y_test_binary = y_test_binary.ravel()
                            y_prob_binary = y_prob
                        pr_fig = create_precision_recall_curve(y_test_binary, y_prob_binary)
                    else:
                        pr_fig = create_precision_recall_curve(y_test, y_prob)
                    st.plotly_chart(pr_fig, config={'responsive': True}, key="pr_curve_plot")
                except Exception as e:
                    st.error(f"Unable to generate precision-recall curve: {str(e)}")
            with col2:
                explanation = st.session_state.builder.get_calculation_explanation("precision_recall_curve")
                with st.container():
                    st.markdown("#### What am I looking at?")
                    st.markdown("""---""")
                    st.markdown(explanation["method"])
                    with st.expander("📖 Detailed Interpretation"):
                        st.markdown("""---""")
                        st.markdown("**Key Points:**")
                        st.markdown(explanation["interpretation"])
        
        # Probability Distribution tab
        probability_distribution_index = available_tabs.index("Probability Distribution")
        #with viz_tabs[probability_distribution_index]:
        if viz_tabs=="Probability Distribution":
            st.write("### 📈 Probability Distribution")
            col1, col2 = st.columns([2, 1])
            with col1:
                try:
                    if len(np.unique(y_test)) > 2:
                        st.info("Showing probability distribution for multi-class classification. Each subplot shows how the model assigns probabilities to that specific class.")
                        # For multi-class, show distribution of class-specific probabilities
                        prob_fig = create_multiclass_probability_distribution_plot(y_test, y_prob_matrix)  # Use cached probabilities
                    else:
                        # Get optimal threshold info from session state
                        optimal_threshold = None
                        has_optimal_threshold = False
                        try:
                            if (hasattr(st.session_state, 'builder') and 
                                hasattr(st.session_state.builder, 'model') and 
                                st.session_state.builder.model.get("threshold_optimized", False)):
                                
                                optimal_threshold = st.session_state.builder.model.get("optimal_threshold", 0.5)
                                is_binary = st.session_state.builder.model.get("threshold_is_binary", True)
                                
                                if is_binary and optimal_threshold != 0.5:
                                    has_optimal_threshold = True
                        except:
                            pass
                            
                        prob_fig = create_probability_distribution_plot(
                            y_test, y_prob, optimal_threshold, has_optimal_threshold
                        )
                    st.plotly_chart(prob_fig, config={'responsive': True}, key="prob_dist_plot")
                except Exception as e:
                    st.error(f"Unable to generate probability distribution: {str(e)}")
            with col2:
                # Determine which explanation to show
                if len(np.unique(y_test)) > 2:
                    explanation_key = "probability_distribution_multiclass"
                else:
                    explanation_key = "probability_distribution_binary"
                
                explanation = st.session_state.builder.get_calculation_explanation(explanation_key)
                
                with st.container():
                    st.markdown("#### What am I looking at?")
                    st.markdown("""---""")
                    st.markdown(explanation["method"])
                    with st.expander("📖 Detailed Interpretation"):
                        st.markdown("""---""")
                        st.markdown("**Key Points:**")
                        st.markdown(explanation["interpretation"])
    
    # Error Analysis tab (always available)
    # Find the next available tab index
    #error_tab_index = len(available_tabs) - 1  # Error Analysis is always the last tab
    error_tab_index = available_tabs.index("Error Analysis")
    #with viz_tabs[error_tab_index]:
    if viz_tabs=="Error Analysis":
        st.write("### 📊 Error Analysis")
        
        # Error by confidence analysis (if probabilities available)
        if has_predict_proba and y_prob is not None:
            st.write("#### Error Rate by Prediction Confidence")
            col1, col2 = st.columns([2, 1])
            with col1:
                try:
                    # Use cached predictions and probabilities
                    confidence_fig, calibration_info = create_classification_error_by_confidence_plot(y_test, y_pred_cached, y_prob_matrix)
                    
                    if confidence_fig is not None:
                        st.plotly_chart(confidence_fig, config={'responsive': True}, key="classification_confidence_error_plot")
                        
                        # Display calibration metrics for both binary and multiclass classification
                        if calibration_info is not None:
                            is_binary_metrics = calibration_info.get('is_binary', True)
                            
                            # Check if there was an error calculating metrics
                            if 'error' in calibration_info:
                                if is_binary_metrics:
                                    st.write("##### 🎯 Model Calibration Metrics (Binary)")
                                else:
                                    st.write("##### 🎯 Model Calibration Metrics (Multiclass)")
                                
                                st.error(f"❌ **Calibration metrics unavailable**: {calibration_info['error']}")
                                
                                with st.expander("ℹ️ Why Calibration Metrics Might Be Missing"):
                                    st.markdown("""
                                    **Common reasons calibration metrics can't be calculated:**
                                    
                                    **For Multiclass Models:**
                                    - Class label encoding issues (non-numeric or inconsistent labels)
                                    - Mismatch between number of classes in predictions vs actual labels
                                    - Missing probability matrix (some models don't provide full probabilities)
                                    - Insufficient data in probability bins for reliable calculation
                                    
                                    **For Binary Models:**
                                    - Probability values all identical (no variation)
                                    - Missing or invalid probability predictions
                                    - Insufficient data samples
                                    
                                    **What this means:**
                                    - The confidence error plot is still valid and shows prediction patterns
                                    - You can still assess model performance using other metrics
                                    - Consider checking your data preprocessing and model configuration
                                    
                                    **Potential solutions:**
                                    - Ensure consistent class labeling (0, 1, 2, ... for multiclass)
                                    - Use models that provide full probability matrices
                                    - Check for sufficient data variety in your test set
                                    """)
                            else:
                                # Display metrics normally
                                if is_binary_metrics:
                                    st.write("##### 🎯 Model Calibration Metrics (Binary)")
                                else:
                                    st.write("##### 🎯 Model Calibration Metrics (Multiclass)")
                                
                                cal_col1, cal_col2, cal_col3 = st.columns(3)
                                
                                with cal_col1:
                                    brier_score = calibration_info['brier_score']
                                    if is_binary_metrics:
                                        # Binary thresholds
                                        brier_color = "green" if brier_score < 0.1 else "orange" if brier_score < 0.2 else "red"
                                        help_text = "Lower is better. Measures probability accuracy (0 = perfect, 0.25 = random)"
                                        status_text = 'Excellent' if brier_score < 0.1 else 'Good' if brier_score < 0.2 else 'Needs Improvement'
                                    else:
                                        # Multiclass thresholds (generally higher than binary)
                                        brier_color = "green" if brier_score < 0.2 else "orange" if brier_score < 0.4 else "red"
                                        help_text = "Lower is better. Multiclass Brier score measures overall probability accuracy"
                                        status_text = 'Excellent' if brier_score < 0.2 else 'Good' if brier_score < 0.4 else 'Needs Improvement'
                                    
                                    st.metric(
                                        "Brier Score", 
                                        f"{brier_score:.4f}",
                                        help=help_text
                                    )
                                    st.markdown(f"<span style='color: {brier_color}'>{status_text}</span>", unsafe_allow_html=True)
                                
                                with cal_col2:
                                    ece = calibration_info['ece']
                                    ece_color = "green" if ece < 0.05 else "orange" if ece < 0.1 else "red"
                                    st.metric(
                                        "Expected Calibration Error", 
                                        f"{ece:.4f}",
                                        help="Lower is better. Measures how well probabilities match actual outcomes"
                                    )
                                    st.markdown(f"<span style='color: {ece_color}'>{'Well Calibrated' if ece < 0.05 else 'Moderately Calibrated' if ece < 0.1 else 'Poorly Calibrated'}</span>", unsafe_allow_html=True)
                                
                                with cal_col3:
                                    st.write("**Calibration Quality:**")
                                    if is_binary_metrics:
                                        if ece < 0.05 and brier_score < 0.1:
                                            st.success("🎯 Excellent calibration!")
                                        elif ece < 0.1 and brier_score < 0.2:
                                            st.warning("⚠️ Moderate calibration")
                                        else:
                                            st.error("❌ Poor calibration")
                                    else:
                                        if ece < 0.05 and brier_score < 0.2:
                                            st.success("🎯 Excellent calibration!")
                                        elif ece < 0.1 and brier_score < 0.4:
                                            st.warning("⚠️ Moderate calibration")
                                        else:
                                            st.error("❌ Poor calibration")
                                            
                                # Add explanation for multiclass metrics
                                if not is_binary_metrics:
                                    with st.expander("ℹ️ Understanding Multiclass Calibration"):
                                        st.markdown("""
                                        **Multiclass Calibration Metrics:**
                                        
                                        **Brier Score:**
                                        - Measures how far predicted probabilities are from the true class
                                        - For multiclass, typically ranges from 0 to 2 (vs 0 to 1 for binary)
                                        - Good multiclass models usually have Brier scores < 0.3
                                        
                                        **Expected Calibration Error (ECE):**
                                        - Uses the maximum predicted probability and checks correctness
                                        - Same interpretation as binary: < 0.05 is well calibrated
                                        - Shows if the model is overconfident or underconfident
                                        
                                        **Why This Matters:**
                                        - Well-calibrated models: predicted probabilities match actual accuracy
                                        - Overconfident models: high probabilities but low accuracy
                                        - Underconfident models: low probabilities but high accuracy
                                        """)
                        else:
                            st.warning("⚠️ **Calibration metrics unavailable**: Could not calculate Brier score and ECE for this model configuration.")
                    else:
                        st.info("Unable to generate confidence error analysis")
                except Exception as e:
                    st.warning(f"Confidence error analysis not available: {str(e)}")
            with col2:
                explanation = st.session_state.builder.get_calculation_explanation("error_by_confidence")
                with st.container():
                    st.markdown("#### What am I looking at?")
                    st.markdown("""---""")
                    st.markdown(explanation["method"])
                    with st.expander("📖 Detailed Interpretation"):
                        st.markdown("""---""")
                        st.markdown("**Key Points:**")
                        st.markdown(explanation["interpretation"])
        
        # Error by feature ranges
        st.write("#### Error Patterns by Feature Values")
        col1, col2 = st.columns([2, 1])
        with col1:
            try:
                feature_error_fig = create_classification_confusion_by_features_plot(y_test, y_pred_cached, X_test, model_instance)  # Use cached predictions
                if feature_error_fig is not None:
                    st.plotly_chart(feature_error_fig, config={'responsive': True}, key="classification_feature_error_plot")
                else:
                    st.info("Unable to generate feature-based error analysis (no numeric features available)")
            except Exception as e:
                st.warning(f"Feature error analysis not available: {str(e)}")
        with col2:
            explanation = st.session_state.builder.get_calculation_explanation("error_by_feature")
            with st.container():
                st.markdown("#### What am I looking at?")
                st.markdown("""---""")
                st.markdown(explanation["method"])
                with st.expander("📖 Detailed Interpretation"):
                    st.markdown("""---""")
                    st.markdown("**Key Points:**")
                    st.markdown(explanation["interpretation"])

    # Feature Importance tab (only if model supports it)
    if has_feature_importance:
        feature_importance_index = available_tabs.index("Feature Importance")
        #with viz_tabs[feature_importance_index]:
        if viz_tabs=="Feature Importance":
            st.write("### 🎯 Feature Importance")
            col1, col2 = st.columns([2, 1])
            with col1:
                feature_names = list(X_test.columns)
                importance_fig = create_feature_importance_plot(model_instance, feature_names)
                if importance_fig is not None:
                    st.plotly_chart(importance_fig, config={'responsive': True}, key="feature_importance_plot")
                else:
                    st.error("Unable to generate feature importance plot")
            with col2:
                with st.container():
                    explanation = st.session_state.builder.get_calculation_explanation("feature_importance")
                    st.markdown("#### What am I looking at?")
                    st.markdown("""---""")
                    st.markdown(explanation["method"])
                    with st.expander("📖 Detailed Interpretation"):
                        st.markdown("""---""")
                        st.markdown("**Key Points:**")
                        st.markdown(explanation["interpretation"])
                            
    # Display model compatibility information
    st.markdown("---")
    st.markdown("### 🔧 Model Capabilities")
    
    model_type = st.session_state.builder.model.get("type", "Unknown")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Available Visualizations:**")
        for tab in available_tabs:
            st.markdown(f"✅ {tab}")
    
    with col2:
        st.markdown("**Model Information:**")
        st.markdown(f"- **Model Type**: {model_type}")
        is_calibrated = st.session_state.builder.model.get("is_calibrated", False)
        calibration_method = st.session_state.builder.model.get("calibration_method", "")
        if is_calibrated:
            st.markdown(f"- **Calibration**: ✅ Active ({calibration_method})")
        else:
            st.markdown(f"- **Calibration**: ❌ Not Applied")
        st.markdown(f"- **Probability Predictions**: {'✅ Supported' if has_predict_proba else '❌ Not Available'}")
        st.markdown(f"- **Feature Importance**: {'✅ Supported' if has_feature_importance else '❌ Not Available'}")
    
    if not has_predict_proba:
        st.info("💡 **Note**: Precision-Recall Curve and Probability Distribution are not available for this model type. Consider using Random Forest, Gradient Boosting, XGBoost, LightGBM, Logistic Regression, or MLP for these visualizations.")
    
    if not has_feature_importance:
        st.info("💡 **Note**: Feature Importance is not available for this model type. Consider using Decision Tree, Random Forest, Gradient Boosting, XGBoost, or LightGBM for feature importance analysis.")

 # Check model capabilities - improved detection for XGBoost and other boosting models
# can_provide_confidence_intervals is imported from utils

# Style the dataframe
# highlight_high_influence is imported from utils
       
def display_regression_visualisations(result):
    # Get model and test data for additional visualizations
    model_instance = st.session_state.builder.model.get("active_model") or st.session_state.builder.model["model"]
    X_test = st.session_state.builder.X_test
    y_test = st.session_state.builder.y_test

    # ===== PERFORMANCE: Use cached predictions =====
    X_test_hash = get_data_hash(X_test)
    y_pred = get_cached_predictions(model_instance, X_test, X_test_hash)
    
    has_ensemble_predictions = can_provide_confidence_intervals(model_instance)
    has_feature_importance = has_feature_importance_support(model_instance)
    
    # Determine which tabs to show based on model capabilities and data availability
    available_tabs = ["Predictions", "Residuals Analysis", "Error Analysis"]
    
    if has_ensemble_predictions:
        available_tabs.append("Confidence Intervals")
    
    if has_feature_importance:
        available_tabs.append("Feature Importance")
    
    # Create tabs for available visualisations
    #viz_tabs = st.tabs(available_tabs)
    viz_tabs = st.pills(label="", options=available_tabs, default="Predictions")

    # Tab 0: Predictions (always available)
    #with viz_tabs[0]:
    if viz_tabs=="Predictions":
        st.write("### 🎯 Actual vs Predicted Values")
        col1, col2 = st.columns([2, 1])
        with col1:
            if "prediction_plot" in result and result["prediction_plot"] is not None:
                st.plotly_chart(result["prediction_plot"], config={'responsive': True}, key="prediction_plot")
            else:
                st.error("Unable to generate prediction plot")
        with col2:
            explanation = st.session_state.builder.get_calculation_explanation("actual_vs_predicted")
            with st.container():
                st.markdown("#### What am I looking at?")
                st.markdown("""---""")
                st.markdown(explanation["method"])
                with st.expander("📖 Detailed Interpretation"):
                    st.markdown("""---""")
                    st.markdown("**Key Points:**")
                    st.markdown(explanation["interpretation"])

    # Tab 1: Residuals Analysis (always available)
    elif viz_tabs=="Residuals Analysis":
        st.write("### 📊 Residuals Analysis")
        
        # Add overview explanation
        with st.expander("ℹ️ What are Residuals?"):
            explanation = st.session_state.builder.get_calculation_explanation("residuals_analysis")
            st.markdown(explanation["method"])
        
        # Display the residuals plot which contains all four subplots
        if "residuals_plot" in result and result["residuals_plot"] is not None:
            st.plotly_chart(result["residuals_plot"], config={'responsive': True}, key="viz_residuals_plot")
        else:
            st.error("Unable to generate residuals plots")
            
        # Add individual plot explanations
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📊 Understanding Residuals vs Predicted"):
                st.markdown(st.session_state.builder.get_calculation_explanation("residuals_vs_predicted")["interpretation"])

            with st.expander("📉 Understanding Q-Q Plot"):
                st.markdown(st.session_state.builder.get_calculation_explanation("residuals_qq_plot")["interpretation"])
    
        with col2:
            
            with st.expander("📈 Understanding Residual Distribution"):
                st.markdown(st.session_state.builder.get_calculation_explanation("residuals_distribution")["interpretation"])
                
            with st.expander("📊 Understanding Scale-Location Plot"):
                st.markdown(st.session_state.builder.get_calculation_explanation("residuals_scale_location")["interpretation"])
        
        # Add overall interpretation guide
        with st.expander("📖 Overall Residuals Interpretation Guide"):
            explanation = st.session_state.builder.get_calculation_explanation("residuals_analysis")
            st.markdown(explanation["interpretation"])
    
    # Tab 2: Error Analysis (always available)
    #with viz_tabs[2]:
    elif viz_tabs=="Error Analysis":
        st.write("### 📊 Error Analysis")
        
        # Error by prediction range
        col1, col2 = st.columns([2, 1])
        with col1:
            try:
                error_range_fig = create_error_by_prediction_range_plot(y_test, y_pred)
                if error_range_fig is not None:
                    st.plotly_chart(error_range_fig, config={'responsive': True}, key="error_range_plot")
                else:
                    st.info("Unable to generate error range analysis (insufficient data variation)")
            except Exception as e:
                st.warning(f"Error range analysis not available: {str(e)}")
        with col2:
            explanation = st.session_state.builder.get_calculation_explanation("error_by_prediction_range")
            with st.container():
                st.markdown("#### What am I looking at?")
                st.markdown("""---""")
                st.markdown(explanation["method"])
                with st.expander("📖 Detailed Interpretation"):
                    st.markdown("""---""")
                    st.markdown("**Key Points:**")
                    st.markdown(explanation["interpretation"])
        
        # Influential points analysis
        st.write("### 🔍 Influential Points Analysis")
        col1, col2 = st.columns([2, 1])
        with col1:
            try:
                influence_fig = create_cooks_distance_plot(y_test, y_pred, X_test)
                if influence_fig is not None:
                    st.plotly_chart(influence_fig, config={'responsive': True}, key="influence_plot")
                else:
                    st.info("Unable to generate influential points analysis")
            except Exception as e:
                st.warning(f"Influential points analysis not available: {str(e)}")
        with col2:
            explanation = st.session_state.builder.get_calculation_explanation("influential_points")
            with st.container():
                st.markdown("#### What am I looking at?")
                st.markdown("""---""")
                st.markdown(explanation["method"])
                with st.expander("📖 Detailed Interpretation"):
                    st.markdown("""---""")
                    st.markdown("**Key Points:**")
                    st.markdown(explanation["interpretation"])
        
        # Add influential points dataframe
        try:
            residuals = y_test - y_pred
            
            # Calculate influence scores (same as in the plot function)
            standardized_residuals = np.abs(residuals) / np.std(residuals)
            prediction_leverage = np.abs(y_pred - np.mean(y_pred)) / np.std(y_pred)
            influence_score = standardized_residuals * (1 + prediction_leverage)
            
            # Calculate threshold
            threshold = np.mean(influence_score) + 2 * np.std(influence_score)
            
            # Find points above threshold
            influential_indices = np.where(influence_score > threshold)[0]
            
            if len(influential_indices) > 0:
                st.write(f"### 📋 High Influence Points (Above Threshold: {threshold:.3f})")
                
                # Create dataframe with influential points
                influential_data = []
                for idx in influential_indices:
                    # Get the original index in the test set
                    original_idx = X_test.index[idx] if hasattr(X_test, 'index') else idx
                    
                    row_data = {
                        'Sample_Index': original_idx,
                        'Actual_Value': y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx],
                        'Predicted_Value': y_pred[idx],
                        'Residual': residuals.iloc[idx] if hasattr(residuals, 'iloc') else residuals[idx],
                        'Influence_Score': influence_score[idx]
                    }
                    
                    # Add feature values (limited to avoid too wide table)
                    feature_values = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
                    if hasattr(feature_values, 'to_dict'):
                        feature_dict = feature_values.to_dict()
                    else:
                        feature_dict = {f'Feature_{i}': val for i, val in enumerate(feature_values)}
                    
                    # Limit to top 5 most important features if we have too many
                    if len(feature_dict) > 5:
                        # If we have feature importance available, show top features
                        try:
                            if hasattr(model_instance, 'feature_importances_'):
                                feature_importance = model_instance.feature_importances_
                                top_features = np.argsort(feature_importance)[-5:]
                                feature_names = list(X_test.columns)
                                top_feature_names = [feature_names[i] for i in top_features]
                                feature_dict = {name: feature_dict[name] for name in top_feature_names if name in feature_dict}
                            else:
                                # Just take first 5 features
                                feature_dict = dict(list(feature_dict.items())[:5])
                        except:
                            feature_dict = dict(list(feature_dict.items())[:5])
                    
                    row_data.update(feature_dict)
                    influential_data.append(row_data)
                
                influential_df = pd.DataFrame(influential_data)
                
                
                
                styled_df = influential_df.style.apply(highlight_high_influence, residuals=residuals, axis=1)
                
                # Display the dataframe
                st.dataframe(styled_df, width='stretch')
                
                # Add summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Influence Points", len(influential_indices))
                with col2:
                    st.metric("Percentage of Data", f"{len(influential_indices)/len(y_test)*100:.1f}%")
                with col3:
                    avg_influence = np.mean(influence_score[influential_indices])
                    st.metric("Avg Influence Score", f"{avg_influence:.3f}")
                
                # Add explanation
                explanation = st.session_state.builder.get_calculation_explanation("influential_points_table")
                with st.expander("ℹ️ Understanding the Influential Points Table"):
                    st.markdown(explanation["method"])
                    st.markdown(explanation["interpretation"])
            else:
                st.success("✅ No high influence points detected above the threshold!")
                st.info(f"Threshold: {threshold:.3f} | All points are below this value, indicating stable model performance.")
                
        except Exception as e:
            st.warning(f"Could not generate influential points table: {str(e)}")
    
    # Tab 3: Confidence Intervals (only for ensemble models)
    if has_ensemble_predictions:
        confidence_intervals_index = available_tabs.index("Confidence Intervals")
        #with viz_tabs[confidence_intervals_index]:
        if viz_tabs == "Confidence Intervals":
            st.write("### 🎯 Confidence Intervals")
            col1, col2 = st.columns([2, 1])
            with col1:
                try:
                    intervals_fig = create_prediction_intervals_plot(model_instance, X_test, y_test, y_pred)
                    if intervals_fig is not None:
                        st.plotly_chart(intervals_fig, config={'responsive': True}, key="intervals_plot")
                    else:
                        st.info("Unable to generate prediction intervals for this ensemble model")
                except Exception as e:
                    st.warning(f"Prediction intervals not available: {str(e)}")
            with col2:
                with st.container():
                    explanation = st.session_state.builder.get_calculation_explanation("prediction_intervals")
                    st.markdown("#### What am I looking at?")
                    st.markdown("""---""")
                    st.markdown(explanation["method"])
                    with st.expander("📖 Detailed Interpretation"):
                        st.markdown("""---""")
                        st.markdown("**Key Points:**")
                        st.markdown(explanation["interpretation"])
    
    # Tab 4: Feature Importance (only if model supports it)
    if has_feature_importance:
        feature_importance_index = available_tabs.index("Feature Importance")
        #with viz_tabs[feature_importance_index]:
        if viz_tabs == "Feature Importance":
            st.write("### 🎯 Feature Importance")
            col1, col2 = st.columns([2, 1])
            with col1:
                feature_names = list(X_test.columns)
                importance_fig = create_feature_importance_plot(model_instance, feature_names)
                if importance_fig is not None:
                    st.plotly_chart(importance_fig, config={'responsive': True}, key="feature_importance_plot")
                else:
                    st.error("Unable to generate feature importance plot")
            with col2:
                with st.container():
                    explanation = st.session_state.builder.get_calculation_explanation("feature_importance")
                    st.markdown("#### What am I looking at?")
                    st.markdown("""---""")
                    st.markdown(explanation["method"])
                    with st.expander("📖 Detailed Interpretation"):
                        st.markdown("""---""")
                        st.markdown("**Key Points:**")
                        st.markdown(explanation["interpretation"])
    
    # Display model compatibility information
    st.markdown("---")
    st.markdown("### 🔧 Model Capabilities")
    
    model_type = st.session_state.builder.model.get("type", "Unknown")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Available Visualizations:**")
        for tab in available_tabs:
            st.markdown(f"✅ {tab}")
    
    with col2:
        st.markdown("**Model Information:**")
        st.markdown(f"- **Model Type**: {model_type}")
        st.markdown(f"- **Ensemble Model**: {'✅ Yes' if has_ensemble_predictions else '❌ No'}")
        st.markdown(f"- **Feature Importance**: {'✅ Supported' if has_feature_importance else '❌ Not Available'}")
    
    if not has_ensemble_predictions:
        st.info("💡 **Note**: Confidence Intervals are only available for ensemble models like Random Forest, Extra Trees, Gradient Boosting, XGBoost, or LightGBM. These models can provide prediction uncertainty estimates through different ensemble techniques.")
    else:
        # Add explanation of how confidence intervals work for different model types
        model_type = st.session_state.builder.model.get("type", "Unknown")
        if "xgb" in model_type.lower() or "lightgbm" in model_type.lower():
            st.info("ℹ️ **Confidence Intervals**: For XGBoost/LightGBM models, intervals are generated using prediction variations with small input perturbations to simulate ensemble uncertainty.")
        elif any(ensemble_type in model_type.lower() for ensemble_type in ["forest", "trees", "gradient"]):
            st.info("ℹ️ **Confidence Intervals**: For ensemble models, intervals are calculated using predictions from individual estimators or boosting stages.")
    
    if not has_feature_importance:
        # Check if this is a calibrated model that we couldn't extract importance from
        is_calibrated = st.session_state.builder.model.get("is_calibrated", False)
        if is_calibrated:
            st.info("💡 **Note**: Feature Importance may not be available for some calibrated models due to wrapper complexity. The underlying model still retains its feature importance, but it may not be accessible through the calibration wrapper.")
        else:
            st.info("💡 **Note**: Feature Importance is not available for this model type. Consider using Decision Tree, Random Forest, Gradient Boosting, XGBoost, or LightGBM for feature importance analysis.")
