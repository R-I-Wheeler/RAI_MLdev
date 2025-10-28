import streamlit as st
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import plotly.graph_objects as go
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

@st.cache_data(ttl=300, show_spinner=False)
def _cached_calibration_analysis(model_params_hash, X_test_shape, y_test_hash, problem_type):
    """Cache calibration analysis results to avoid recalculation on every page load"""
    # This function will be populated by the main calibration analysis
    # The actual analysis will be done in analyze_current_calibration
    return None

@st.cache_data(ttl=300, show_spinner=False)
def _cached_calibration_plot(plot_data, plot_type):
    """Cache calibration plot creation"""
    # This will cache the plotly figure creation
    return None

def display_calibration_section():
    """Display the model calibration section in the training page."""

    # Import state manager for better component communication
    from components.model_training.utils.training_state_manager import TrainingStateManager

    # Validate training state
    validation = TrainingStateManager.validate_training_state()
    if not all(validation.values()):
        st.warning("Please complete model training first to access calibration analysis.")
        return
    # Check if we have a trained classification model
    if (not st.session_state.builder.model or 
        st.session_state.builder.model.get("problem_type") not in ["classification", "binary_classification", "multiclass_classification"]):
        return
    
    st.header("🎯 Model Calibration (Optional)")

    # Check problem type for targeted messaging
    problem_type = st.session_state.builder.model["problem_type"]
    is_binary = problem_type in ["classification", "binary_classification"]

    # Add multiclass-specific warning if applicable
    if not is_binary:
        st.warning("""
        📋 **Multiclass Classification Detected**

        **Important considerations for multiclass calibration:**
        - Calibration is more complex than binary classification
        - Results may vary significantly across different classes
        - Consider if calibration is necessary for your use case
        - Monitor per-class performance carefully
        """)

    # Add calibration explanation
    with st.expander("🔍 What is Model Calibration?", expanded=False):
        st.markdown("""
        ### Understanding Model Calibration
        
        **Model calibration** ensures that when your model predicts a 70% probability, 
        it's actually correct about 70% of the time.
        
        #### Why Calibration Matters:
        - **Medical Diagnosis**: "80% chance of disease" should be accurate
        - **Financial Risk**: Loan default probabilities must be reliable
        - **Weather Forecasting**: "30% chance of rain" should happen 3 out of 10 times
        - **Business Decisions**: Confidence levels guide important choices
        
        #### When to Use Calibration:
        ✅ **Always consider for**:
        - High-stakes decisions
        - When probability values matter (not just classifications)
        - Models that seem overconfident or underconfident
        
        ⚠️ **May not be needed for**:
        - Simple classification tasks where only the prediction matters
        - Well-calibrated models (check metrics below first)
        - Very small datasets
        
        #### Calibration Methods:
        - **Platt Scaling (Sigmoid)**: Fits a sigmoid curve, good for small datasets
        - **Isotonic Regression**: More flexible, better for larger datasets

        #### Special Considerations for Multiclass:
        - **Per-class complexity**: Each class may have different calibration needs
        - **Class imbalance effects**: Minority classes may be harder to calibrate
        - **Evaluation complexity**: Multiple metrics need to be considered
        - **Computational cost**: More expensive than binary calibration
        """)
    
    # Check current model calibration first
    st.subheader("📊 Current Model Calibration Analysis")
    
    with st.spinner("Analyzing current model calibration..."):
        calibration_analysis = analyze_current_calibration()
        
        if calibration_analysis["success"]:
            display_calibration_analysis(calibration_analysis)
            
            # Decide whether to recommend calibration
            recommend_calibration = should_recommend_calibration(calibration_analysis)

            if recommend_calibration:
                if is_binary:
                    st.warning("""
                    🎯 **Calibration Recommended**: Your model shows signs of poor calibration.
                    Consider applying calibration to improve probability reliability.
                    """)
                else:
                    st.warning("""
                    🎯 **Calibration Recommended**: Your multiclass model shows signs of poor calibration.

                    **For multiclass models:**
                    - Calibration can help improve probability reliability across all classes
                    - Pay attention to per-class calibration metrics below
                    - Consider if the complexity is worth the potential improvement
                    """)
            else:
                if is_binary:
                    st.success("""
                    ✅ **Well Calibrated**: Your model is already well-calibrated.
                    Calibration may provide minimal improvement.
                    """)
                else:
                    st.success("""
                    ✅ **Well Calibrated**: Your multiclass model is already reasonably calibrated.

                    **Multiclass note**: Overall metrics look good, but check per-class breakdown below
                    to ensure consistent calibration across all classes.
                    """)
            
            # Check if model is already calibrated and show status
            is_calibrated = st.session_state.builder.model.get("is_calibrated", False)
            
            if is_calibrated:
                # Show calibration status
                st.subheader("✅ Calibration Active")
                
                calibration_method = st.session_state.builder.model.get("calibration_method", "unknown")
                cv_folds = st.session_state.builder.model.get("calibration_cv_folds", "unknown")
                
                st.info(f"""
                🎯 **Current Calibration Status:**
                - **Method**: {calibration_method.title()}
                - **CV Folds**: {cv_folds}
                - **Status**: Active and applied to all predictions
                """)
                
                # Revert button for calibrated models
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button("🔄 Revert to Original Model", type="secondary"):
                        revert_calibration()
                        
                with col2:
                    st.info("Revert will restore the original uncalibrated model")
                
            else:
                # Show calibration options for uncalibrated models
                if is_binary:
                    st.subheader("🔧 Apply Calibration")
                else:
                    st.subheader("🔧 Apply Multiclass Calibration")
                    st.info("""
                    💡 **Multiclass Calibration Info**:
                    Calibration will be applied to all classes simultaneously. Monitor the per-class
                    breakdown above to understand how calibration affects individual classes.
                    """)

                # Get smart recommendation
                recommended_method, recommendation_explanation = recommend_calibration_method()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Find the index of the recommended method
                    method_options = ["isotonic", "sigmoid"]
                    recommended_index = method_options.index(recommended_method) if recommended_method in method_options else 0
                    
                    calibration_method = st.selectbox(
                        "Calibration Method",
                        options=method_options,
                        index=recommended_index,  # Set recommended method as default
                        format_func=lambda x: "Isotonic Regression" if x == "isotonic" else "Platt Scaling (Sigmoid)",
                        help="""
                        - Isotonic Regression: More flexible, better for larger datasets
                        - Platt Scaling: Simpler, better for smaller datasets
                        """
                    )
                    
                    # Show recommendation explanation
                    with st.expander("💡 Why this method?", expanded=False):
                        st.markdown(recommendation_explanation)
                
                with col2:
                    cv_folds_cal = st.selectbox(
                        "Cross-validation Folds for Calibration",
                        options=[3, 5, 10],
                        index=1,  # Default to 5
                        help="Number of folds to use for calibration cross-validation"
                    )
                
                # Calibration button
                if st.button("🎯 Apply Calibration", type="primary"):
                    apply_calibration(calibration_method, cv_folds_cal)
        else:
            st.error(f"Could not analyze calibration: {calibration_analysis['message']}")

def analyze_current_calibration() -> Dict[str, Any]:
    """Analyze the calibration of the current model."""
    try:
        # Use the same model access pattern as evaluation visualizations
        model = st.session_state.builder.model.get("active_model") or st.session_state.builder.model["model"]
        X_train = st.session_state.builder.X_train
        y_train = st.session_state.builder.y_train
        X_test = st.session_state.builder.X_test
        y_test = st.session_state.builder.y_test
        problem_type = st.session_state.builder.model["problem_type"]
        
        # Get predictions on test data for calibration analysis
        y_prob_test = model.predict_proba(X_test)
        
        # Handle binary vs multiclass
        is_binary = problem_type in ["classification", "binary_classification"]
        
        if is_binary:
            # For binary classification, use positive class probability
            if len(y_prob_test.shape) > 1:
                y_prob_positive = y_prob_test[:, 1]
            else:
                y_prob_positive = y_prob_test
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_prob_positive, n_bins=10
            )
            
            # Calculate Brier score
            brier_score = brier_score_loss(y_test, y_prob_positive)
            
            # Calculate Expected Calibration Error
            ece = calculate_ece_binary(y_test, y_prob_positive)
            
            # Create calibration plot
            calibration_plot = create_calibration_plot_binary(
                fraction_of_positives, mean_predicted_value, brier_score, ece
            )
            
            return {
                "success": True,
                "is_binary": True,
                "brier_score": brier_score,
                "ece": ece,
                "calibration_plot": calibration_plot,
                "reliability_data": {
                    "fraction_of_positives": fraction_of_positives,
                    "mean_predicted_value": mean_predicted_value
                }
            }
        else:
            # For multiclass classification
            # Calculate multiclass Brier score
            from sklearn.preprocessing import LabelBinarizer
            lb = LabelBinarizer()
            y_true_onehot = lb.fit_transform(y_test)

            if len(y_true_onehot.shape) == 1:
                y_true_onehot = np.column_stack([1 - y_true_onehot, y_true_onehot])

            brier_score = np.mean(np.sum((y_prob_test - y_true_onehot) ** 2, axis=1))

            # Calculate multiclass ECE using max probability
            max_probs = np.max(y_prob_test, axis=1)
            predicted_classes = np.argmax(y_prob_test, axis=1)

            # Convert y_test to numeric for comparison
            unique_classes = sorted(np.unique(y_test))
            y_test_numeric = np.array([unique_classes.index(y) for y in y_test])

            correct_predictions = (predicted_classes == y_test_numeric).astype(int)
            ece = calculate_ece_multiclass(max_probs, correct_predictions)

            # Calculate per-class calibration analysis
            per_class_analysis = calculate_per_class_calibration(
                y_test, y_prob_test, unique_classes
            )

            # Create multiclass calibration plot
            calibration_plot = create_calibration_plot_multiclass(
                max_probs, correct_predictions, brier_score, ece, per_class_analysis
            )

            return {
                "success": True,
                "is_binary": False,
                "brier_score": brier_score,
                "ece": ece,
                "calibration_plot": calibration_plot,
                "per_class_analysis": per_class_analysis,
                "unique_classes": unique_classes,
                "class_distribution": np.bincount(y_test_numeric) / len(y_test_numeric)
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }

def calculate_ece_binary(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error for binary classification."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def calculate_ece_multiclass(max_probs, correct_predictions, n_bins=10):
    """Calculate Expected Calibration Error for multiclass classification."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = correct_predictions[in_bin].mean()
            avg_confidence_in_bin = max_probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def calculate_per_class_calibration(y_true, y_prob, unique_classes):
    """Calculate per-class calibration metrics for multiclass classification."""
    per_class_metrics = {}

    for i, class_label in enumerate(unique_classes):
        # Create binary mask for this class
        y_true_binary = (y_true == class_label).astype(int)
        y_prob_class = y_prob[:, i]

        # Calculate calibration curve for this class
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_binary, y_prob_class, n_bins=10
            )

            # Calculate Brier score for this class
            brier_score_class = brier_score_loss(y_true_binary, y_prob_class)

            # Calculate ECE for this class
            ece_class = calculate_ece_binary(y_true_binary, y_prob_class)

            per_class_metrics[class_label] = {
                'brier_score': brier_score_class,
                'ece': ece_class,
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value,
                'n_samples': np.sum(y_true_binary),
                'class_frequency': np.sum(y_true_binary) / len(y_true_binary)
            }
        except Exception as e:
            # Handle edge cases (e.g., class not present in test set)
            per_class_metrics[class_label] = {
                'brier_score': np.nan,
                'ece': np.nan,
                'fraction_of_positives': np.array([]),
                'mean_predicted_value': np.array([]),
                'n_samples': 0,
                'class_frequency': 0,
                'error': str(e)
            }

    return per_class_metrics

def create_calibration_plot_binary(fraction_of_positives, mean_predicted_value, brier_score, ece):
    """Create calibration plot for binary classification."""
    fig = go.Figure()
    
    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash'),
        hovertemplate='Perfect Calibration<extra></extra>'
    ))
    
    # Actual calibration curve
    fig.add_trace(go.Scatter(
        x=mean_predicted_value,
        y=fraction_of_positives,
        mode='lines+markers',
        name='Model Calibration',
        line=dict(color='blue', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Model Calibration</b><br>' +
                     'Mean Predicted: %{x:.3f}<br>' +
                     'Fraction Positive: %{y:.3f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Calibration Plot<br><sub>Brier Score: {brier_score:.4f} | ECE: {ece:.4f}</sub>',
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        height=400,
        showlegend=True
    )
    
    return fig

def create_calibration_plot_multiclass(max_probs, correct_predictions, brier_score, ece, per_class_analysis=None):
    """Create calibration plot for multiclass classification."""
    from plotly.subplots import make_subplots

    # Create subplots: main plot and per-class plot if available
    if per_class_analysis and len(per_class_analysis) > 1:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f'Overall Calibration<br><sub>Brier: {brier_score:.4f} | ECE: {ece:.4f}</sub>',
                'Per-Class Calibration'
            ),
            horizontal_spacing=0.15
        )
    else:
        fig = go.Figure()

    # Main calibration plot
    # Bin the probabilities
    bin_boundaries = np.linspace(0, 1, 11)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    accuracies = []
    confidences = []
    counts = []

    for i in range(len(bin_boundaries) - 1):
        in_bin = (max_probs > bin_boundaries[i]) & (max_probs <= bin_boundaries[i + 1])

        if np.sum(in_bin) > 0:
            accuracies.append(correct_predictions[in_bin].mean())
            confidences.append(max_probs[in_bin].mean())
            counts.append(np.sum(in_bin))
        else:
            accuracies.append(0)
            confidences.append(bin_centers[i])
            counts.append(0)

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash'),
        showlegend=True
    ), row=1, col=1 if per_class_analysis and len(per_class_analysis) > 1 else None)

    # Actual calibration curve
    fig.add_trace(go.Scatter(
        x=confidences,
        y=accuracies,
        mode='lines+markers',
        name='Overall Calibration',
        line=dict(color='blue', width=3),
        marker=dict(size=[c/10 + 5 for c in counts], sizemode='diameter'),
        text=[f'Count: {c}' for c in counts],
        hovertemplate='<b>Overall Calibration</b><br>' +
                     'Confidence: %{x:.3f}<br>' +
                     'Accuracy: %{y:.3f}<br>' +
                     '%{text}<br>' +
                     '<extra></extra>',
        showlegend=True
    ), row=1, col=1 if per_class_analysis and len(per_class_analysis) > 1 else None)

    # Add per-class calibration curves if available
    if per_class_analysis and len(per_class_analysis) > 1:
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']

        # Perfect calibration line for second subplot
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect',
            line=dict(color='gray', dash='dash'),
            showlegend=False
        ), row=1, col=2)

        for i, (class_label, metrics) in enumerate(per_class_analysis.items()):
            if 'error' not in metrics and len(metrics['mean_predicted_value']) > 0:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=metrics['mean_predicted_value'],
                    y=metrics['fraction_of_positives'],
                    mode='lines+markers',
                    name=f'Class {class_label}',
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    hovertemplate=f'<b>Class {class_label}</b><br>' +
                                 'Mean Predicted: %{x:.3f}<br>' +
                                 'Fraction Positive: %{y:.3f}<br>' +
                                 f'Brier: {metrics["brier_score"]:.4f}<br>' +
                                 f'ECE: {metrics["ece"]:.4f}<br>' +
                                 '<extra></extra>',
                    showlegend=True
                ), row=1, col=2)

    # Update layout
    if per_class_analysis and len(per_class_analysis) > 1:
        fig.update_layout(
            height=500,
            showlegend=True,
            legend=dict(x=1.05, y=1)
        )
        fig.update_xaxes(title_text='Confidence', row=1, col=1)
        fig.update_yaxes(title_text='Accuracy', row=1, col=1)
        fig.update_xaxes(title_text='Mean Predicted Probability', row=1, col=2)
        fig.update_yaxes(title_text='Fraction of Positives', row=1, col=2)
    else:
        fig.update_layout(
            title=f'Multiclass Calibration Plot<br><sub>Brier Score: {brier_score:.4f} | ECE: {ece:.4f}</sub>',
            xaxis_title='Confidence (Max Probability)',
            yaxis_title='Accuracy',
            height=400,
            showlegend=True
        )

    return fig

def display_calibration_analysis(analysis: Dict[str, Any]):
    """Display the calibration analysis results."""
    
    # Display calibration plot with unique key
    st.plotly_chart(analysis["calibration_plot"], config={'responsive': True}, key="calibration_analysis_plot")
    
    _display_calibration_metrics(analysis)

def display_calibration_results(analysis: Dict[str, Any]):
    """Display the calibration results after applying calibration."""
    
    # Display calibration plot with unique key for results
    st.plotly_chart(analysis["calibration_plot"], config={'responsive': True}, key="calibration_results_plot")
    
    _display_calibration_metrics(analysis)

def _display_calibration_metrics(analysis: Dict[str, Any]):
    """Display calibration metrics (shared between analysis and results)."""

    # Display overall metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        brier_score = analysis["brier_score"]
        if analysis["is_binary"]:
            brier_color = "green" if brier_score < 0.1 else "orange" if brier_score < 0.2 else "red"
            brier_status = 'Excellent' if brier_score < 0.1 else 'Good' if brier_score < 0.2 else 'Needs Improvement'
        else:
            brier_color = "green" if brier_score < 0.2 else "orange" if brier_score < 0.4 else "red"
            brier_status = 'Excellent' if brier_score < 0.2 else 'Good' if brier_score < 0.4 else 'Needs Improvement'

        st.metric("Brier Score", f"{brier_score:.4f}")
        st.markdown(f"<span style='color: {brier_color}'>{brier_status}</span>", unsafe_allow_html=True)

    with col2:
        ece = analysis["ece"]
        ece_color = "green" if ece < 0.05 else "orange" if ece < 0.1 else "red"
        ece_status = 'Well Calibrated' if ece < 0.05 else 'Moderately Calibrated' if ece < 0.1 else 'Poorly Calibrated'

        st.metric("Expected Calibration Error", f"{ece:.4f}")
        st.markdown(f"<span style='color: {ece_color}'>{ece_status}</span>", unsafe_allow_html=True)

    with col3:
        # More nuanced overall assessment that matches individual metric thresholds
        if analysis["is_binary"]:
            # For binary: Well calibrated if ECE < 0.05 and Brier < 0.2 (matches "Good" threshold)
            excellent_calibration = ece < 0.05 and brier_score < 0.1
            good_calibration = ece < 0.1 and brier_score < 0.2
        else:
            # For multiclass: Well calibrated if ECE < 0.05 and Brier < 0.4 (matches "Good" threshold)
            excellent_calibration = ece < 0.05 and brier_score < 0.2
            good_calibration = ece < 0.1 and brier_score < 0.4

        if excellent_calibration:
            st.success("🎯 Excellent Calibration")
        elif good_calibration:
            st.warning("⚠️ Moderate Calibration")
        else:
            st.error("❌ Poor Calibration")

    # Display per-class metrics for multiclass
    if not analysis["is_binary"] and "per_class_analysis" in analysis:
        st.subheader("📋 Per-Class Calibration Breakdown")

        per_class_data = []
        for class_label, metrics in analysis["per_class_analysis"].items():
            if 'error' not in metrics:
                per_class_data.append({
                    'Class': class_label,
                    'Brier Score': f"{metrics['brier_score']:.4f}",
                    'ECE': f"{metrics['ece']:.4f}",
                    'Sample Count': metrics['n_samples'],
                    'Class Frequency': f"{metrics['class_frequency']:.1%}"
                })

        if per_class_data:
            df = pd.DataFrame(per_class_data)
            st.dataframe(df, width='stretch', hide_index=True)

            # Class imbalance warning
            frequencies = [float(item['Class Frequency'].rstrip('%'))/100 for item in per_class_data]
            if max(frequencies) / min(frequencies) > 3:  # 3:1 ratio threshold
                st.warning("⚠️ **Class Imbalance Detected**: Large differences in class frequencies may affect calibration quality for minority classes.")

        # Show class distribution
        if "class_distribution" in analysis:
            st.write("**Class Distribution in Test Set:**")
            dist_cols = st.columns(len(analysis["unique_classes"]))
            for i, (class_label, freq) in enumerate(zip(analysis["unique_classes"], analysis["class_distribution"])):
                with dist_cols[i]:
                    st.metric(f"Class {class_label}", f"{freq:.1%}")

def recommend_calibration_method() -> Tuple[str, str]:
    """Recommend the best calibration method based on dataset and model characteristics."""
    try:
        # Get dataset and model information
        X_train = st.session_state.builder.X_train
        y_train = st.session_state.builder.y_train
        model = st.session_state.builder.model.get("active_model") or st.session_state.builder.model["model"]
        model_type = st.session_state.builder.model.get("type", "Unknown").lower()
        problem_type = st.session_state.builder.model["problem_type"]
        
        # Factors for recommendation
        dataset_size = len(X_train)
        is_binary = problem_type in ["classification", "binary_classification"]
        n_classes = len(np.unique(y_train))
        
        # Model type classification
        tree_based_models = ["random forest", "extra trees", "decision tree", "xgboost", "lightgbm", "catboost", "gradient boosting"]
        linear_models = ["logistic regression", "linear", "ridge", "lasso"]
        neural_models = ["neural network", "mlp", "perceptron"]
        
        is_tree_based = any(tree_type in model_type for tree_type in tree_based_models)
        is_linear = any(linear_type in model_type for linear_type in linear_models)
        is_neural = any(neural_type in model_type for neural_type in neural_models)
        
        # Decision logic for method recommendation
        reasons = []
        
        # Size-based recommendation
        if dataset_size < 1000:
            method_vote = "sigmoid"
            reasons.append(f"Small dataset ({dataset_size:,} samples) - Platt scaling is more stable")
        elif dataset_size > 5000:
            method_vote = "isotonic"
            reasons.append(f"Large dataset ({dataset_size:,} samples) - Isotonic regression can capture complex patterns")
        else:
            method_vote = "isotonic"  # Default for medium datasets
            reasons.append(f"Medium dataset ({dataset_size:,} samples) - Isotonic regression offers good flexibility")
        
        # Model-based adjustment
        if is_tree_based:
            if dataset_size >= 1000:  # Only override for larger datasets
                method_vote = "isotonic"
                reasons.append("Tree-based model - Isotonic regression works well with non-linear model outputs")
        elif is_linear:
            method_vote = "sigmoid"
            reasons.append("Linear model - Platt scaling (sigmoid) complements linear model assumptions")
        elif is_neural:
            if dataset_size >= 2000:
                method_vote = "isotonic"
                reasons.append("Neural network - Isotonic regression can handle complex probability distributions")
            else:
                method_vote = "sigmoid"
                reasons.append("Neural network with smaller dataset - Platt scaling is more conservative")
        
        # Enhanced multiclass considerations
        if not is_binary:
            # Check for class imbalance in multiclass scenario
            class_counts = np.bincount([np.argmax(np.bincount(y_train))])
            class_balance_ratio = np.min(np.bincount(y_train)) / np.max(np.bincount(y_train))

            if n_classes > 5:
                # Many classes - be more conservative
                if dataset_size >= 3000:
                    method_vote = "isotonic"
                    reasons.append(f"Many classes ({n_classes}) with large dataset - Isotonic regression handles high-dimensional complexity")
                else:
                    method_vote = "sigmoid"
                    reasons.append(f"Many classes ({n_classes}) with smaller dataset - Platt scaling more stable")
            elif n_classes > 3:
                # Moderate number of classes
                if dataset_size >= 2000:
                    method_vote = "isotonic"
                    reasons.append(f"Multiclass problem ({n_classes} classes) - Isotonic regression handles complexity better")
                else:
                    method_vote = "sigmoid"
                    reasons.append(f"Multiclass with moderate dataset - Platt scaling for stability")

            # Adjust for class imbalance in multiclass
            if class_balance_ratio < 0.2:  # Severe imbalance
                method_vote = "isotonic"
                reasons.append(f"Severe class imbalance detected (ratio: {class_balance_ratio:.2f}) - Isotonic regression better for imbalanced multiclass")
            elif class_balance_ratio < 0.5:  # Moderate imbalance
                if dataset_size >= 1500:
                    method_vote = "isotonic"
                    reasons.append(f"Moderate class imbalance (ratio: {class_balance_ratio:.2f}) - Isotonic regression with sufficient data")

        # Get current calibration quality for final adjustment
        try:
            current_analysis = analyze_current_calibration()
            if current_analysis["success"]:
                ece = current_analysis["ece"]
                brier_score = current_analysis["brier_score"]

                # Adjusted thresholds for multiclass
                if is_binary:
                    well_calibrated_ece = 0.05
                    well_calibrated_brier = 0.15
                    poor_calibrated_ece = 0.15
                    poor_calibrated_brier = 0.25
                else:
                    well_calibrated_ece = 0.08  # More lenient for multiclass
                    well_calibrated_brier = 0.3
                    poor_calibrated_ece = 0.2
                    poor_calibrated_brier = 0.5

                # If already well calibrated, prefer gentler method
                if ece < well_calibrated_ece and brier_score < well_calibrated_brier:
                    method_vote = "sigmoid"
                    reasons.append("Model already reasonably calibrated - Platt scaling provides gentler adjustment")
                # If poorly calibrated, prefer more flexible method
                elif ece > poor_calibrated_ece or brier_score > poor_calibrated_brier:
                    method_vote = "isotonic"
                    reasons.append("Model poorly calibrated - Isotonic regression provides more flexible correction")

                # Special consideration for multiclass with per-class analysis
                if not is_binary and "per_class_analysis" in current_analysis:
                    per_class_eces = [metrics.get('ece', 1.0) for metrics in current_analysis["per_class_analysis"].values()
                                     if 'error' not in metrics]
                    if per_class_eces:
                        max_class_ece = max(per_class_eces)
                        if max_class_ece > 0.2:  # Some classes very poorly calibrated
                            method_vote = "isotonic"
                            reasons.append(f"Some classes poorly calibrated (max ECE: {max_class_ece:.3f}) - Isotonic regression for flexibility")
        except:
            pass  # If analysis fails, stick with previous recommendation
        
        # Create explanation
        explanation = f"**Recommended: {'Isotonic Regression' if method_vote == 'isotonic' else 'Platt Scaling'}**\n\n"
        explanation += "**Reasoning:**\n"
        for i, reason in enumerate(reasons, 1):
            explanation += f"{i}. {reason}\n"
        
        explanation += f"\n**Dataset:** {dataset_size:,} samples, {n_classes} classes"
        explanation += f"\n**Model:** {model_type.title()}"
        
        return method_vote, explanation
        
    except Exception as e:
        # Fallback recommendation
        return "isotonic", f"**Default recommendation: Isotonic Regression**\n\nUnable to analyze dataset characteristics: {str(e)}"

def should_recommend_calibration(analysis: Dict[str, Any]) -> bool:
    """Determine if calibration should be recommended."""
    ece = analysis["ece"]
    brier_score = analysis["brier_score"]
    is_binary = analysis["is_binary"]

    # Recommend calibration if model doesn't meet "good calibration" thresholds
    if is_binary:
        # For binary: recommend if ECE >= 0.1 OR Brier >= 0.2
        poor_calibration = ece >= 0.1 or brier_score >= 0.2
    else:
        # For multiclass: more lenient thresholds and additional considerations
        # Base thresholds: ECE >= 0.12 OR Brier >= 0.4
        poor_calibration = ece >= 0.12 or brier_score >= 0.4

        # Additional multiclass-specific checks
        if "per_class_analysis" in analysis and analysis["per_class_analysis"]:
            # Check if any individual class is poorly calibrated
            per_class_eces = [metrics.get('ece', 0) for metrics in analysis["per_class_analysis"].values()
                             if 'error' not in metrics]
            per_class_briers = [metrics.get('brier_score', 0) for metrics in analysis["per_class_analysis"].values()
                               if 'error' not in metrics]

            if per_class_eces:
                # If worst class ECE > 0.2, recommend calibration
                max_class_ece = max(per_class_eces)
                if max_class_ece > 0.2:
                    poor_calibration = True

                # If there's high variance in per-class calibration, recommend
                if len(per_class_eces) > 2:
                    ece_std = np.std(per_class_eces)
                    if ece_std > 0.1:  # High variance in per-class ECEs
                        poor_calibration = True

            # Check class imbalance effects
            if "class_distribution" in analysis:
                class_freqs = analysis["class_distribution"]
                if len(class_freqs) > 0:
                    imbalance_ratio = min(class_freqs) / max(class_freqs)
                    # For severely imbalanced multiclass, be more aggressive about calibration
                    if imbalance_ratio < 0.1 and ece >= 0.08:
                        poor_calibration = True

    return poor_calibration

def revert_calibration():
    """Revert the model back to its original uncalibrated state."""
    try:
        # Check if we have the original model stored
        if "original_model" not in st.session_state.builder.model:
            st.error("❌ Original model not found. Cannot revert calibration.")
            return
        
        # Get the original model from session state
        original_model = st.session_state.builder.model["original_model"]
        
        # Restore the original model
        st.session_state.builder.model["model"] = original_model
        st.session_state.builder.model["active_model"] = original_model
        
        # Clean up calibration-related state
        st.session_state.builder.model["is_calibrated"] = False
        
        # Remove calibration-specific keys
        calibration_keys_to_remove = [
            "calibrated_model", 
            "calibration_method", 
            "calibration_cv_folds"
        ]
        
        for key in calibration_keys_to_remove:
            if key in st.session_state.builder.model:
                del st.session_state.builder.model[key]
        
        # Log the reversion
        st.session_state.logger.log_user_action(
            "Calibration Reverted",
            {
                "action": "revert_to_original",
                "timestamp": pd.Timestamp.now().isoformat()
            }
        )
        
        st.session_state.logger.log_journey_point(
            stage="MODEL_TRAINING",
            decision_type="MODEL_TRAINING",
            description="Calibration reverted to original model",
            details={
                "Action": "Revert Calibration",
                "Status": "Reverted to original uncalibrated model"
            },
            parent_id=None
        )
        
        st.success("✅ Successfully reverted to original uncalibrated model!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error reverting calibration: {str(e)}")
        st.session_state.logger.log_error("Calibration Reversion Failed", {"error": str(e)})

def apply_calibration(method: str, cv_folds: int):
    """Apply calibration to the model."""
    try:
        # Get original calibration analysis before applying calibration
        original_analysis = analyze_current_calibration()
        
        with st.spinner(f"Applying {method} calibration..."):
            model = st.session_state.builder.model["model"]
            X_train = st.session_state.builder.X_train
            y_train = st.session_state.builder.y_train
            
            # Create calibrated classifier
            # Handle both old and new scikit-learn API versions
            try:
                # Try new API (sklearn >= 1.2)
                calibrated_model = CalibratedClassifierCV(
                    estimator=model,
                    method=method,
                    cv=cv_folds,
                    n_jobs=-1
                )
            except TypeError:
                # Fall back to old API (sklearn < 1.2)
                calibrated_model = CalibratedClassifierCV(
                    base_estimator=model,
                    method=method,
                    cv=cv_folds,
                    n_jobs=-1
                )
            
            # Fit the calibrated model
            calibrated_model.fit(X_train, y_train)
            
            # Store original model and set calibrated model
            original_model = st.session_state.builder.model["model"]
            st.session_state.builder.model["model"] = calibrated_model
            st.session_state.builder.model["active_model"] = calibrated_model  # Set active_model BEFORE analysis
            
            calibrated_analysis = analyze_current_calibration()
            
            if calibrated_analysis["success"]:
                # Store both models (active_model already set above)
                st.session_state.builder.model["original_model"] = original_model
                st.session_state.builder.model["calibrated_model"] = calibrated_model
                st.session_state.builder.model["is_calibrated"] = True
                st.session_state.builder.model["calibration_method"] = method
                st.session_state.builder.model["calibration_cv_folds"] = cv_folds
                
                # Log the calibration
                st.session_state.logger.log_user_action(
                    "Model Calibrated",
                    {
                        "method": method,
                        "cv_folds": cv_folds,
                        "original_brier": original_analysis["brier_score"] if original_analysis["success"] else None,
                        "calibrated_brier": calibrated_analysis["brier_score"],
                        "original_ece": original_analysis["ece"] if original_analysis["success"] else None,
                        "calibrated_ece": calibrated_analysis["ece"]
                    }
                )
                
                st.session_state.logger.log_journey_point(
                        stage="MODEL_TRAINING",
                        decision_type="MODEL_TRAINING",
                        description="Calibration applied",
                        details={
                                "Calibration Method": method,
                                "CV Folds": cv_folds,
                                "Original Brier Score": original_analysis["brier_score"] if original_analysis["success"] else None,
                                "Calibrated Brier Score": calibrated_analysis["brier_score"],
                                "Original ECE": original_analysis["ece"] if original_analysis["success"] else None,
                                "Calibrated ECE": calibrated_analysis["ece"]
                                },
                        parent_id=None
                    )
                st.success(f"✅ Calibration applied successfully using {method}!")
                
                # Show comparison
                st.subheader("📊 Calibration Results")
                # Pass a different context to avoid key conflicts
                calibrated_analysis_copy = calibrated_analysis.copy()
                if "calibration_plot" in calibrated_analysis_copy:
                    # Create a new plot with a different key context
                    display_calibration_results(calibrated_analysis_copy)
                
                st.info("🔄 Page will refresh to show the revert option at the top of the calibration section.")
                
                # Trigger page rerun to refresh the calibration status display
                st.rerun()
                    
            else:
                # Revert on error
                st.session_state.builder.model["model"] = original_model
                st.error(f"Calibration failed: {calibrated_analysis['message']}")
                
    except Exception as e:
        st.error(f"Error during calibration: {str(e)}")
        st.session_state.logger.log_error("Calibration Failed", {"error": str(e)})
