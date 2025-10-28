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
                with st.container():
                    st.markdown("#### What am I looking at?")
                    st.markdown("""---""")
                    st.markdown("""
                    **Precision-Recall Curve** shows the trade-off between precision and recall for different decision thresholds.
                    
                    - **X-axis (Recall)**: How many actual positive cases the model finds
                    - **Y-axis (Precision)**: How accurate the model is when it predicts positive
                    - **AP Score**: Average Precision - summarizes the curve (higher is better)
                    - **Red dashed line**: Random classifier baseline
                    """)
                    with st.expander("📖 Detailed Interpretation"):
                        st.markdown("""---""")
                        st.markdown("**Key Points:**")
                        st.markdown("""
                        - **High precision, low recall**: Model is very accurate but misses many positive cases
                        - **Low precision, high recall**: Model finds most positive cases but makes many false predictions
                        - **Curve close to top-right**: Excellent performance
                        - **Curve close to baseline**: Poor performance
                        
                        **When to use:**
                        - Imbalanced datasets (better than ROC curve)
                        - When false positives are costly
                        - When you need to understand precision-recall trade-offs
                        
                        **What good looks like:**
                        - Curve stays close to the top of the plot
                        - AP score > 0.7 is generally good
                        - Significantly above the red baseline
                        """)
        
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
                with st.container():
                    st.markdown("#### What am I looking at?")
                    st.markdown("""---""")
                    if len(np.unique(y_test)) > 2:
                        st.markdown("""
                        **Confidence Distribution** shows how confident the model is when making predictions for each true class.
                        
                        - **Each subplot**: One true class (what the samples actually are)
                        - **Green bars**: Correct predictions (model got it right)
                        - **Red bars**: Incorrect predictions (model got it wrong)
                        - **X-axis**: Model confidence level (0 = uncertain, 1 = very confident)
                        - **Accuracy %**: Shown in top-right corner of each subplot
                        """)
                    else:
                        st.markdown("""
                        **Probability Distribution** shows how confident the model is in its predictions for each class.
                        
                        - **Blue bars**: Distribution for actual negative class (0)
                        - **Red bars**: Distribution for actual positive class (1) 
                        - **X-axis**: Prediction probability (0 = confident negative, 1 = confident positive)
                        - **Black dashed line**: Default decision threshold (0.5)
                        """)
                    with st.expander("📖 Detailed Interpretation"):
                        st.markdown("""---""")
                        st.markdown("**Key Points:**")
                        if len(np.unique(y_test)) > 2:
                            st.markdown("""
                            **What good performance looks like:**
                            - **Green bars on the right**: Correct predictions with high confidence (> 0.8)
                            - **Red bars on the left**: Incorrect predictions with low confidence (< 0.6)
                            - **High accuracy %**: Most predictions are correct for this class
                            
                            **What poor performance looks like:**
                            - **Red bars on the right**: Incorrect but confident predictions (dangerous!)
                            - **Green bars in middle**: Correct but uncertain predictions
                            - **Low accuracy %**: Model struggles with this class
                            
                            **How to interpret each subplot:**
                            - **True Class X**: All samples that are actually class X
                            - **Green distribution**: When model correctly identifies class X samples
                            - **Red distribution**: When model incorrectly predicts something else
                            - **Ideal pattern**: Green bars clustered right (confident + correct), red bars clustered left (uncertain + wrong)
                            
                            **Actionable insights:**
                            - **Classes with low accuracy**: Need more training data or feature engineering
                            - **Confident but wrong (red on right)**: Review those specific samples for labeling errors
                            - **Uncertain but right (green in middle)**: Consider confidence thresholds for decision-making
                            """)
                        else:
                            st.markdown("""
                            **What good separation looks like:**
                            - Blue bars clustered near 0 (model confident about negatives)
                            - Red bars clustered near 1 (model confident about positives)
                            - Minimal overlap between the two distributions
                            
                            **What poor separation looks like:**
                            - Both distributions clustered around 0.5
                            - Heavy overlap between blue and red bars
                            - Similar shapes for both classes
                            
                            **How to use this:**
                            - Well-separated distributions → model has learned good decision boundaries
                            - Overlapping distributions → model struggles to distinguish classes
                            - Can help identify optimal probability thresholds
                            - Shows model confidence levels
                            """)
    
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
                with st.container():
                    st.markdown("#### What am I looking at?")
                    st.markdown("""---""")
                    st.markdown("""
                    **Error Rate by Confidence** shows how often the model makes mistakes based on how confident it is, with trend analysis and calibration metrics.
                    
                    - **Blue bars**: Error rate for each confidence level
                    - **Red trend line**: Overall relationship pattern
                    - **Calibration metrics**: How well probabilities match reality
                    """)
                    with st.expander("📖 Detailed Interpretation"):
                        st.markdown("""---""")
                        st.markdown("**Key Points:**")
                        st.markdown("""
                        **What good calibration looks like:**
                        - **Trend line slopes downward**: Higher confidence → Lower error rate
                        - **Low Brier Score** (< 0.1): Accurate probability predictions
                        - **Low ECE** (< 0.05): Predicted probabilities match actual outcomes
                        
                        **What poor calibration looks like:**
                        - **Flat trend line**: Confidence doesn't correlate with accuracy
                        - **High Brier Score** (> 0.2): Inaccurate probability predictions
                        - **High ECE** (> 0.1): Model overconfident or underconfident
                        
                        **Calibration Metrics Explained:**
                        - **Brier Score**: Overall accuracy of probability predictions (0 = perfect)
                        - **Expected Calibration Error (ECE)**: How much predicted probabilities deviate from actual frequencies
                        
                        **Business Implications:**
                        - **Well-calibrated**: Can trust probability thresholds for decisions
                        - **Overconfident**: Lower thresholds for high-stakes decisions
                        - **Underconfident**: Can be more aggressive with predictions
                        - **Poor calibration**: Consider probability calibration techniques
                        """)
        
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
            with st.container():
                st.markdown("#### What am I looking at?")
                st.markdown("""---""")
                st.markdown("""
                **Error by Feature Ranges** shows if the model struggles more in certain regions of your feature space.
                
                - **X-axis**: Feature value ranges
                - **Y-axis**: Error rate in that range
                - **Bars**: How often the model is wrong in each range
                """)
                with st.expander("📖 Detailed Interpretation"):
                    st.markdown("""---""")
                    st.markdown("**Key Points:**")
                    st.markdown("""
                    **What to look for:**
                    - **Even error rates**: Model performs consistently across feature ranges
                    - **High error spikes**: Model struggles in specific ranges
                    - **Edge effects**: Higher errors at extreme values
                    
                    **Common patterns:**
                    - **Data sparsity**: Higher errors where training data is sparse
                    - **Class imbalance**: Higher errors for underrepresented regions
                    - **Non-linear relationships**: Errors in regions where linear assumptions break down
                    
                    **How to improve:**
                    - Collect more data in high-error regions
                    - Consider feature transformations
                    - Review outliers in problematic ranges
                    - Adjust model complexity for those regions
                    """)

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
                    st.markdown("#### What am I looking at?")
                    st.markdown("""---""")
                    st.markdown("""
                    **Feature Importance** shows which features the model considers most important for making predictions.
                    
                    - **X-axis**: Importance score (higher = more important)
                    - **Y-axis**: Feature names
                    - **Bars**: Relative importance of each feature
                    
                    **What the numbers mean:**
                    - **Tree-based models** (Random Forest, XGBoost): Shows how often each feature was used to make decisions. Higher numbers = the model relied on this feature more often
                    - **Linear models** (Linear/Logistic Regression): Shows how strongly each feature influences the prediction. Higher numbers = this feature has a bigger impact on the result
                    - **Other models**: Shows which features the model considers most useful for making accurate predictions
                                
                    
                    """)
                    with st.expander("📖 Detailed Interpretation"):
                        st.markdown("""---""")
                        st.markdown("**Key Points:**")
                        st.markdown("""
                        **How to interpret:**
                        - **Longer bars**: More important features for the model
                        - **Shorter bars**: Less important features
                        - **Zero importance**: Features not used by the model
                        
                        **What this tells you:**
                        - Which features drive your model's decisions
                        - Potential features to focus on for data collection
                        - Features that might be removed to simplify the model
                        - Domain knowledge validation (do important features make sense?)
                        
                        **Important notes:**
                        - Based on how often features are used for splits
                        - High importance doesn't always mean causation
                        - Should be combined with domain expertise
                        
                        **Next steps:**
                        - Investigate why certain features are important
                        - Consider feature engineering based on insights
                        - Validate findings with domain experts
                        """)
                            
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
            st.markdown("""
                **Residuals** are the differences between actual and predicted values. They help us understand:
                - How well our model fits the data
                - Whether our model's assumptions are met
                - Where our model makes the biggest errors
                - If our model is biased in certain regions
            """)
        
        # Display the residuals plot which contains all four subplots
        if "residuals_plot" in result and result["residuals_plot"] is not None:
            st.plotly_chart(result["residuals_plot"], config={'responsive': True}, key="viz_residuals_plot")
        else:
            st.error("Unable to generate residuals plots")
            
        # Add individual plot explanations
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📊 Understanding Residuals vs Predicted"):
                st.markdown("""
                **What to Look For:**
                - Points should scatter randomly around the horizontal line at y=0
                - No clear patterns or trends
                - Even spread above and below the line
                
                **Red Flags:**
                - Curved patterns suggest non-linear relationships
                - Funnel shapes indicate heteroscedasticity
                - Clusters suggest missing variables
                - Outliers far from y=0
                
                **Ideal Pattern:**
                Random scatter with:
                - Even spread vertically
                - No obvious patterns
                - Most points between -2 and 2 on y-axis
                """)

            with st.expander("📉 Understanding Q-Q Plot"):
                st.markdown("""
                **What to Look For:**
                - Points following the diagonal line
                - Minimal deviation from line
                - Even spread along the line
                
                **Red Flags:**
                - S-shaped curve
                - Points far from diagonal
                - Heavy tails (ends deviate)
                
                **Why It Matters:**
                - Tests normality assumption
                - Shows outlier impact
                - Identifies systematic deviations
                """)
    
        with col2:
            
            with st.expander("📈 Understanding Residual Distribution"):
                st.markdown("""
                **What to Look For:**
                - Bell-shaped (normal) distribution
                - Centered at zero
                - Symmetric spread
                
                **Red Flags:**
                - Skewness (leaning left or right)
                - Multiple peaks
                - Long tails
                - Center not at zero
                
                **Why It Matters:**
                - Shows if errors are normally distributed
                - Helps identify bias in predictions
                - Validates regression assumptions
                """)
                
            with st.expander("📊 Understanding Scale-Location Plot"):
                st.markdown("""
                **What to Look For:**
                - Horizontal line with constant spread
                - Random scatter of points
                - No obvious patterns
                
                **Red Flags:**
                - Funnel shapes
                - Increasing/decreasing spread
                - Clusters or patterns
                
                **Why It Matters:**
                - Tests homoscedasticity
                - Shows if error variance is constant
                - Identifies prediction reliability
                """)
        
        # Add overall interpretation guide
        with st.expander("📖 Overall Residuals Interpretation Guide"):
            st.markdown("""
            **Good Model Characteristics:**
            1. Random scatter in Residuals vs Predicted
            2. Normal distribution of residuals
            3. Points following Q-Q line
            4. Constant spread in Scale-Location
            
            **Common Issues:**
            1. **Non-linearity**
                - Curved patterns in residual plots
                - Solution: Try non-linear transformations
            
            2. **Heteroscedasticity**
                - Funnel shapes in residual plots
                - Solution: Try variable transformation
            
            3. **Non-normal Errors**
                - Skewed distribution
                - Q-Q plot deviations
                - Solution: Check outliers, try transformations
            
            4. **Outliers**
                - Points far from others
                - Solution: Investigate and possibly remove
            
            **Next Steps if Issues Found:**
            - Review feature engineering
            - Consider data transformations
            - Check for outliers
            - Try different model types
            """)
    
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
            with st.container():
                st.markdown("#### What am I looking at?")
                st.markdown("""---""")
                st.markdown("""
                **Error by Prediction Range** shows how model accuracy varies across different prediction ranges.
                
                - **X-axis**: Prediction value ranges (grouped into bins)
                - **Y-axis**: Absolute error magnitude
                - **Box plots**: Distribution of errors within each range
                """)
                with st.expander("📖 Detailed Interpretation"):
                    st.markdown("""---""")
                    st.markdown("**Key Points:**")
                    st.markdown("""
                    **What good performance looks like:**
                    - Similar box heights across all ranges
                    - Low median error (box center) in all ranges
                    - Few outliers in any range
                    
                    **What poor performance looks like:**
                    - Much higher errors in certain ranges
                    - Many outliers in specific ranges
                    - Systematic patterns (increasing/decreasing errors)
                    
                    **How to use this:**
                    - Identify where your model struggles most
                    - Detect if model performs worse for high/low values
                    - Guide data collection efforts
                    - Inform confidence in predictions by range
                    
                    **Common patterns:**
                    - **Heteroscedasticity**: Errors increase with prediction values
                    - **Range limitation**: Poor performance at extreme values
                    - **Data sparsity**: High errors where training data is sparse
                    """)
        
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
            with st.container():
                st.markdown("#### What am I looking at?")
                st.markdown("""---""")
                st.markdown("""
                **Influential Points Analysis** identifies data points that have unusual impact on the model.
                
                - **X-axis**: Sample index (data point number)
                - **Y-axis**: Influence score (higher = more influential)
                - **Color**: Intensity shows influence level
                - **Red line**: High influence threshold
                """)
                with st.expander("📖 Detailed Interpretation"):
                    st.markdown("""---""")
                    st.markdown("**Key Points:**")
                    st.markdown("""
                    **What high influence means:**
                    - Points that strongly affect model fit
                    - Combination of unusual feature values and large residuals
                    - May be outliers or important edge cases
                    
                    **How to interpret:**
                    - **Points above red line**: Potentially problematic
                    - **Scattered low values**: Normal, healthy pattern
                    - **Few high spikes**: May indicate data quality issues
                    
                    **What to do with influential points:**
                    1. **Investigate**: Are these data entry errors?
                    2. **Domain check**: Do they represent valid but rare cases?
                    3. **Consider removal**: If they're errors or not representative
                    4. **Collect more data**: In regions with high influence
                    
                    **Important notes:**
                    - Not all influential points are bad
                    - Some may represent important edge cases
                    - Always investigate before removing data
                    """)
        
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
                with st.expander("ℹ️ Understanding the Influential Points Table"):
                    st.markdown("""
                    **Column Explanations:**
                    - **Sample_Index**: Original position in your test dataset
                    - **Actual_Value**: True target value for this sample
                    - **Predicted_Value**: Model's prediction for this sample
                    - **Residual**: Difference between actual and predicted (error)
                    - **Influence_Score**: Calculated influence measure (higher = more influential)
                    - **Feature Columns**: Values of the most important features for these samples
                    
                    **Color Coding:**
                    - 🔴 **Light Red**: Influence Score column (these are all high influence points)
                    - 🟡 **Light Yellow**: Residuals that are particularly large (> 2 standard deviations)
                    
                    **What to Look For:**
                    - **Unusual feature combinations**: Do these samples have strange feature values?
                    - **Data entry errors**: Are there any obviously incorrect values?
                    - **Edge cases**: Do these represent rare but valid scenarios?
                    - **Patterns**: Are the influential points clustered in certain feature ranges?
                    
                    **Next Steps:**
                    1. **Investigate each point**: Look at the feature values and check if they make sense
                    2. **Domain validation**: Consult with subject matter experts about these cases
                    3. **Data quality check**: Verify these aren't data collection or entry errors
                    4. **Consider action**: Decide whether to keep, correct, or remove these points
                    """)
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
                    st.markdown("#### What am I looking at?")
                    st.markdown("""---""")
                    st.markdown("""
                    **Prediction Intervals** show the uncertainty in model predictions using ensemble variation.
                    
                    - **Green shaded area**: 90% prediction interval
                    - **Red dots**: Actual values
                    - **Blue dots**: Model predictions
                    - **X-axis**: Sample index (sorted by actual value)
                    """)
                    with st.expander("📖 Detailed Interpretation"):
                        st.markdown("""---""")
                        st.markdown("**Key Points:**")
                        st.markdown("""
                        **What good intervals look like:**
                        - Most actual values (red dots) fall within green area
                        - Interval width reflects true uncertainty
                        - Predictions (blue dots) close to actual values
                        
                        **What poor intervals look like:**
                        - Many actual values outside green area
                        - Intervals too narrow or too wide
                        - Systematic bias in predictions
                        
                        **How to use this:**
                        - **Narrow intervals**: High model confidence
                        - **Wide intervals**: High uncertainty - be cautious
                        - **Values outside intervals**: May need investigation
                        - **Systematic patterns**: Suggest model improvements needed
                        
                        **Business applications:**
                        - Risk assessment for predictions
                        - Setting safety margins
                        - Identifying when to seek additional information
                        - Communicating uncertainty to stakeholders
                        
                        **Technical notes:**
                        - Available for ensemble models (Random Forest, Extra Trees) and boosting models (XGBoost, LightGBM, Gradient Boosting)
                        - Traditional ensembles: Based on variation across individual estimators
                        - Boosting models: Generated using prediction variations or input perturbations
                        - 90% interval means 90% of predictions should fall within bounds
                        """)
    
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
                    st.markdown("#### What am I looking at?")
                    st.markdown("""---""")
                    st.markdown("""
                    **Feature Importance** shows which features the model considers most important for making predictions.
                    
                    - **X-axis**: Importance score (higher = more important)
                    - **Y-axis**: Feature names
                    - **Bars**: Relative importance of each feature
                    
                    **What the numbers mean:**
                    - **Tree-based models** (Random Forest, XGBoost): Shows how often each feature was used to make decisions. Higher numbers = the model relied on this feature more often
                    - **Linear models** (Linear/Logistic Regression): Shows how strongly each feature influences the prediction. Higher numbers = this feature has a bigger impact on the result
                    - **Other models**: Shows which features the model considers most useful for making accurate predictions
                                
                    
                    """)
                    with st.expander("📖 Detailed Interpretation"):
                        st.markdown("""---""")
                        st.markdown("**Key Points:**")
                        st.markdown("""
                        **How to interpret:**
                        - **Longer bars**: More important features for the model
                        - **Shorter bars**: Less important features
                        - **Zero importance**: Features not used by the model
                        
                        **What this tells you:**
                        - Which features drive your model's decisions
                        - Potential features to focus on for data collection
                        - Features that might be removed to simplify the model
                        - Domain knowledge validation (do important features make sense?)
                        
                        **Important notes:**
                        - Based on how often features are used for splits
                        - High importance doesn't always mean causation
                        - Should be combined with domain expertise
                        
                        **Next steps:**
                        - Investigate why certain features are important
                        - Consider feature engineering based on insights
                        - Validate findings with domain experts
                        """)
    
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
