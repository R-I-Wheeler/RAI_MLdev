import streamlit as st
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
from datetime import datetime
import json
from typing import Dict, Any, List, Optional
from components.model_explanation.explanation_utils.report_generator import create_scenario_comparison_report
from components.model_explanation.explanation_utils.shap_visualization_utils import create_force_plot, CustomJSONEncoder
from components.model_explanation.explanation_utils.shap_computation_utils import get_class_names_with_indices, get_background_data, create_shap_explainer
from components.model_explanation.explanation_utils.form_generation_utils import create_input_form, create_feature_info

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# get_class_names_with_indices function moved to utils/shap_computation_utils.py

# calculate_slider_step function moved to explanation_utils/form_generation_utils.py

# get_feature_type function moved to explanation_utils/form_generation_utils.py

# process_original_data function moved to explanation_utils/form_generation_utils.py

# create_feature_info function moved to explanation_utils/form_generation_utils.py

# get_background_data function moved to utils/shap_computation_utils.py

# create_shap_explainer function moved to utils/shap_computation_utils.py

# calculate_shap_values function moved to utils/shap_computation_utils.py

# ScenarioManager class moved to utils/scenario_management_utils.py

# _parse_calculated_field function moved to explanation_utils/form_generation_utils.py



# create_input_form function moved to explanation_utils/form_generation_utils.py

@st.cache_data
def create_force_plot(explainer_expected_value: float, shap_values: np.ndarray, feature_names: List[str]) -> go.Figure:
    """Create and cache force plot for SHAP values."""
    # Ensure we have single-sample SHAP values (1D array)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
        shap_values = shap_values.flatten() if shap_values.shape[0] == 1 else shap_values[0]
    
    # Validate dimensions match
    # Validating force plot inputs
    
    if len(shap_values) != len(feature_names):
        # Fixing dimension mismatch between SHAP values and features
        # Try to fix by truncating or padding
        if len(shap_values) > len(feature_names):
            shap_values = shap_values[:len(feature_names)]
            # Truncated SHAP values to match features
        elif len(shap_values) < len(feature_names):
            feature_names = feature_names[:len(shap_values)]
            # Truncated feature names to match SHAP values
    
    try:
        # Try new SHAP v0.20+ API first
        try:
            force_plot = shap.plots.force(
                explainer_expected_value,
                shap_values,
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
        except (ValueError, TypeError) as e:
            # Force plot error with matplotlib=True, trying alternative
            if "multiple samples" in str(e).lower() or "length of features" in str(e).lower():
                # If matplotlib fails due to multiple samples or dimension mismatch, use matplotlib=False
                force_plot = shap.plots.force(
                    explainer_expected_value,
                    shap_values,
                    feature_names=feature_names,
                    matplotlib=False,
                    show=False
                )
            else:
                raise e
    except AttributeError:
        # Fallback to old API for older SHAP versions
        try:
            force_plot = shap.force_plot(
                explainer_expected_value,
                shap_values,
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
        except (ValueError, TypeError) as e:
            # Force plot error with old API, trying alternative
            if "multiple samples" in str(e).lower() or "length of features" in str(e).lower():
                # If matplotlib fails due to multiple samples or dimension mismatch, use matplotlib=False
                force_plot = shap.force_plot(
                    explainer_expected_value,
                    shap_values,
                    feature_names=feature_names,
                    matplotlib=False,
                    show=False
                )
            else:
                raise e
    return force_plot

@st.cache_data
def create_waterfall_chart(feature_importance: pd.DataFrame, selected_feature: str, 
                          baseline: float, hybrid: float, target: float) -> go.Figure:
    """Create and cache waterfall chart for feature importance visualization."""
    direction_color = 'rgba(50, 171, 96, 0.7)' if hybrid > baseline else 'rgba(219, 64, 82, 0.7)'
    feature_impact = hybrid - baseline
    
    scenarios = [f"Baseline", f"Only {selected_feature}<br>changed", "Final"]
    values = [baseline, hybrid, target]
    colors = ['rgba(55, 128, 191, 0.7)', direction_color, 'rgba(219, 64, 82, 0.7)']
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=scenarios,
        y=values,
        marker_color=colors,
        text=[f"{x:.4f}" for x in values],
        textposition='auto'
    ))
    
    # Add connecting line
    fig.add_trace(go.Scatter(
        x=scenarios,
        y=values,
        mode='lines',
        line=dict(color='black', width=1, dash='dot'),
        showlegend=False
    ))
    
    # Add annotations
    annotations = [
        dict(
            x=0,
            y=baseline,
            text=f"Baseline<br>{baseline:.4f}",
            showarrow=False,
            font=dict(color="black"),
            align="center",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        ),
        dict(
            x=1,
            y=hybrid,
            text=f"Impact of {selected_feature}<br>{feature_impact:+.4f}",
            showarrow=True,
            arrowhead=3,
            arrowcolor=direction_color,
            arrowwidth=2,
            arrowsize=1,
            font=dict(color="black"),
            align="center",
            bgcolor="white",
            bordercolor=direction_color,
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        ),
        dict(
            x=2,
            y=target,
            text=f"Final<br>{target:.4f}",
            showarrow=False,
            font=dict(color="black"),
            align="center",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )
    ]
    
    fig.update_layout(
        title=f"Impact of changing only '{selected_feature}'",
        xaxis_title="Scenarios",
        yaxis_title="Prediction Value",
        height=500,
        annotations=annotations,
        hovermode="x"
    )
    
    return fig

@st.cache_data
def create_radar_chart(features: List[str], normalized_importance: np.ndarray) -> go.Figure:
    """Create and cache radar chart for feature importance visualization."""
    radar_features = features.copy()
    radar_features.append(radar_features[0])
    radar_data = np.append(normalized_importance, normalized_importance[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_data,
        theta=radar_features,
        fill='toself',
        name="Feature Importance",
        line_color='rgba(50, 171, 96, 0.7)',
        fillcolor='rgba(50, 171, 96, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                title="Importance"
            )
        ),
        title="Feature Importance in Prediction Difference",
        height=500,
        showlegend=True
    )
    
    return fig

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling numpy and special types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return str(obj)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

@st.cache_data
def generate_comparison_report(scenario1: str, scenario2: str, data1: Dict[str, Any], 
                             data2: Dict[str, Any], feature_impacts: Dict[str, Any], 
                             shap_data: Optional[Dict[str, Any]] = None) -> str:
    """Generate and cache comparison report in JSON format."""
    report = {
        "metadata": {
            "report_type": "Scenario Comparison",
            "created_at": datetime.now().isoformat(),
            "model_type": st.session_state.builder.model.get("type", "unknown"),
            "problem_type": getattr(st.session_state, 'problem_type', st.session_state.builder.model.get("problem_type", "unknown")),
            "feature_count": len(set(list(data1['values'].keys()) + list(data2['values'].keys())))
        },
        "comparison_summary": {
            "scenario1": {
                "name": scenario1,
                "prediction": float(data1['prediction']),
                "feature_count": len(data1['values'])
            },
            "scenario2": {
                "name": scenario2,
                "prediction": float(data2['prediction']),
                "feature_count": len(data2['values'])
            },
            "prediction_difference": {
                "absolute": float(data2['prediction'] - data1['prediction']),
                "percentage": float((data2['prediction'] - data1['prediction']) / data1['prediction'] * 100 if data1['prediction'] != 0 else 0)
            }
        },
        "scenarios": {
            scenario1: data1['values'],
            scenario2: data2['values']
        },
        "feature_impacts": feature_impacts
    }
    
    if shap_data:
        report["shap_analysis"] = shap_data
    
    return json.dumps(report, indent=2, cls=CustomJSONEncoder)

def display_prediction_results(prediction: float, shap_values: np.ndarray, 
                             feature_names: List[str], input_values: Dict[str, Any],
                             explainer, problem_type: str):
    """Display prediction results with SHAP explanations."""
    st.markdown("### 🎯 Prediction Results")
    
    # Handle different prediction formats
    if problem_type == "multiclass_classification" and isinstance(prediction, np.ndarray):
        predicted_class = np.argmax(prediction)
        max_probability = np.max(prediction)
        prediction_scalar = max_probability  # Use max probability as scalar for display
    else:
        prediction_scalar = float(prediction) if isinstance(prediction, (np.ndarray, np.float64)) else prediction
    
    # Display prediction metrics
    col1, col2 = st.columns([1, 1])
    with col1:
        if problem_type in ["binary_classification", "multiclass_classification", "classification"]:
            if problem_type == "binary_classification":
                # Use optimal threshold if available
                if (hasattr(st.session_state, 'builder') and 
                    hasattr(st.session_state.builder, 'model') and 
                    st.session_state.builder.model.get("threshold_optimized", False) and
                    st.session_state.builder.model.get("threshold_is_binary", True)):
                    optimal_threshold = st.session_state.builder.model.get("optimal_threshold", 0.5)
                    predicted_class = "Positive" if prediction_scalar >= optimal_threshold else "Negative"
                else:
                    predicted_class = "Positive" if prediction_scalar >= 0.5 else "Negative"
                st.metric("Predicted Class", predicted_class)
            elif problem_type == "multiclass_classification":
                # Get actual class names if available
                class_names = get_class_names_with_indices()
                
                if isinstance(prediction, np.ndarray) and len(prediction) > 1:
                    if class_names and predicted_class < len(class_names):
                        predicted_class_display = class_names[predicted_class]
                    else:
                        predicted_class_display = f"Class {predicted_class}"
                    st.metric("Predicted Class", predicted_class_display)
                else:
                    pred_class_idx = int(prediction_scalar)
                    if class_names and pred_class_idx < len(class_names):
                        predicted_class_display = class_names[pred_class_idx]
                    else:
                        predicted_class_display = f"Class {pred_class_idx}"
                    st.metric("Predicted Class", predicted_class_display)
            else:  # fallback for legacy "classification"
                # Use optimal threshold if available
                if (hasattr(st.session_state, 'builder') and 
                    hasattr(st.session_state.builder, 'model') and 
                    st.session_state.builder.model.get("threshold_optimized", False) and
                    st.session_state.builder.model.get("threshold_is_binary", True)):
                    optimal_threshold = st.session_state.builder.model.get("optimal_threshold", 0.5)
                    predicted_class = "Positive" if prediction_scalar >= optimal_threshold else "Negative"
                else:
                    predicted_class = "Positive" if prediction_scalar >= 0.5 else "Negative"
                st.metric("Predicted Class", predicted_class)
        st.metric("Predicted Value", f"{prediction_scalar:.4f}")
    
    with col2:
        if problem_type == "binary_classification":
            st.metric("Positive Class Probability", f"{prediction_scalar:.1%}")
        elif problem_type == "multiclass_classification":
            if isinstance(prediction, np.ndarray) and len(prediction) > 1:
                st.metric("Max Class Probability", f"{max_probability:.1%}")
            else:
                # Get actual class names if available
                class_names = get_class_names_with_indices()
                pred_class_idx = int(prediction_scalar)
                if class_names and pred_class_idx < len(class_names):
                    predicted_class_display = class_names[pred_class_idx]
                else:
                    predicted_class_display = f"Class {pred_class_idx}"
                st.metric("Predicted Class", predicted_class_display)
        elif problem_type == "classification":  # fallback
            st.metric("Positive Class Probability", f"{prediction_scalar:.1%}")
    
    # Show class probabilities for multiclass
    if problem_type == "multiclass_classification" and isinstance(prediction, np.ndarray) and len(prediction) > 1:
        st.subheader("Class Probabilities")
        
        # Get actual class names if available
        class_names = get_class_names_with_indices()
        if class_names and len(class_names) == len(prediction):
            class_labels = class_names
        else:
            # Fallback to generic names
            class_labels = [f'Class {i}' for i in range(len(prediction))]
        
        prob_df = pd.DataFrame({
            'Class': class_labels,
            'Probability': prediction
        })
        prob_df['Percentage'] = prob_df['Probability'] * 100
        
        # Create a horizontal bar chart
        fig = go.Figure(go.Bar(
            x=prob_df['Probability'],
            y=prob_df['Class'],
            orientation='h',
            text=[f'{p:.1f}%' for p in prob_df['Percentage']],
            textposition='inside',
            marker_color=['#ff7f0e' if i == predicted_class else '#1f77b4' 
                        for i in range(len(prediction))]
        ))
        
        fig.update_layout(
            title="Predicted Probabilities by Class",
            xaxis_title="Probability",
            yaxis_title="Class",
            height=200 + len(prediction) * 25,
            showlegend=False
        )
        
        st.plotly_chart(fig, config={'responsive': True})
    
    # Create force plot
    st.markdown("### 📊 SHAP Force Plot")
    
    # Get base value based on model type and problem type
    model_type = st.session_state.builder.model.get("type", "unknown")
    
    if 'xgboost' in model_type or 'lightgbm' in model_type:
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            if problem_type == "multiclass_classification" and isinstance(prediction, np.ndarray):
                # For multiclass, use the expected value for the predicted class
                predicted_class = np.argmax(prediction)
                base_value = base_value[predicted_class] if predicted_class < len(base_value) else base_value[0]
            elif problem_type == "binary_classification":
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            else:
                # For regression, base_value might be a scalar (e.g., XGBoost TreeExplainer)
                if hasattr(base_value, '__len__'):  # Check if it has length before calling len()
                    base_value = base_value[0] if len(base_value) > 0 else 0.0
                # If it's already a scalar, use as-is
    else:
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            if problem_type == "multiclass_classification" and isinstance(prediction, np.ndarray):
                # For multiclass, use the expected value for the predicted class
                predicted_class = np.argmax(prediction)
                base_value = explainer.expected_value[predicted_class] if predicted_class < len(explainer.expected_value) else explainer.expected_value[0]
            elif problem_type == "binary_classification":
                base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
            else:
                base_value = explainer.expected_value[0] if len(explainer.expected_value) > 0 else 0.0
        else:
            base_value = explainer.expected_value
    
    # Validate dimensions before creating force plot
    # Display function processing SHAP values and features
    
    # Process SHAP values for force plot (similar to waterfall processing)
    force_plot_shap_values = shap_values
    if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
        if problem_type == "multiclass_classification":
            predicted_class = np.argmax(prediction) if isinstance(prediction, np.ndarray) else 0
            if shap_values.ndim == 3:
                # Shape: (1, n_features, n_classes) - get SHAP values for predicted class
                force_plot_shap_values = shap_values[0, :, predicted_class]
                # Force plot: 3D case - Using SHAP values for predicted class
            elif shap_values.ndim == 2:
                # Shape: (n_features, n_classes) - get SHAP values for predicted class
                force_plot_shap_values = shap_values[:, predicted_class]
                # Force plot: 2D case - Using SHAP values for predicted class
        elif shap_values.ndim == 2:
            # For binary classification or other 2D cases
            force_plot_shap_values = shap_values[0]
        else:
            # Fallback for other cases
            force_plot_shap_values = shap_values.flatten() if shap_values.shape[0] == 1 else shap_values[0]
    
    try:
        force_plot = create_force_plot(
            base_value,
                force_plot_shap_values,
            feature_names
        )
    except Exception as e:
        print(f"Error creating force plot: {str(e)}")
        # Skip force plot if there's an error
        force_plot = None
    col1, col2 = st.columns([3, 1])
    with col1:
        if force_plot is not None:
            try:
                # Try to display as matplotlib plot first
                st.pyplot(force_plot)
            except Exception as e:
                # If matplotlib display fails, try HTML display
                try:
                    import streamlit.components.v1 as components
                    if hasattr(force_plot, '_repr_html_'):
                        components.html(force_plot._repr_html_(), height=400)
                    else:
                        st.warning("Force plot could not be displayed. This may be due to SHAP version compatibility.")
                except Exception as e2:
                    st.warning(f"Could not display force plot: {str(e)}")
        else:
            st.warning("⚠️ Force plot could not be created due to dimension mismatch. The waterfall plot below shows the same information.")
    with col2:
        if problem_type == "multiclass_classification":
            predicted_class = np.argmax(prediction) if isinstance(prediction, np.ndarray) else 0
            
            # Get actual class names if available for display
            class_names = get_class_names_with_indices()
            if class_names and predicted_class < len(class_names):
                predicted_class_display = class_names[predicted_class]
            else:
                predicted_class_display = f"Class {predicted_class}"
            
            st.info(f"""
            **Reading the Force Plot (Multiclass):**
            - Red = increasing probability of {predicted_class_display}
            - Blue = decreasing probability of {predicted_class_display}
            - Width of bar = magnitude of impact
            - Shows contributions to the predicted class only
            - Base value = average probability for {predicted_class_display}
            """)
        elif problem_type == "binary_classification":
            st.info("""
                **Reading the Force Plot (Binary):**
                - Red = pushing toward positive class
                - Blue = pushing toward negative class
                - Width of bar = magnitude of impact
                - Base value = model's average positive class probability
                """)
        else:  # regression
            st.info("""
            **Reading the Force Plot (Regression):**
            - Red = pushing prediction higher
            - Blue = pushing prediction lower
            - Width of bar = magnitude of impact
            - Base value = model's average prediction
            """)
    
    if problem_type == "multiclass_classification":
        predicted_class = np.argmax(prediction) if isinstance(prediction, np.ndarray) else 0
        
        # Get actual class names if available for display
        class_names = get_class_names_with_indices()
        if class_names and predicted_class < len(class_names):
            predicted_class_display = class_names[predicted_class]
        else:
            predicted_class_display = f"Class {predicted_class}"
        
        st.info(f"""
            💡 **Base Value: {base_value:.4f}**
            
                This represents the model's average probability for {predicted_class_display}. 
                Feature impacts show how each feature moves the prediction away from this baseline 
                for the predicted class.
            """)
    elif problem_type == "binary_classification":
        st.info(f"""
                💡 **Base Value: {base_value:.4f}**
                
                This represents the model's average probability for the positive class. 
                Feature impacts show how each feature affects the likelihood of a positive prediction.
            """)
    else:  # regression
        st.info(f"""
                💡 **Base Value: {base_value:.4f}**
                
                This represents the model's average prediction. Feature impacts show how each feature 
                moves the prediction away from this baseline.
        """)

    # Create feature importance visualization
    st.markdown("### 🌊 SHAP Waterfall Plot")
    
    # Ensure SHAP values are 1D for DataFrame creation
    print(f"Waterfall plot - SHAP values shape: {shap_values.shape if hasattr(shap_values, 'shape') else type(shap_values)}")
    if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
        print(f"Processing SHAP values from {shap_values.shape} to 1D")
        if problem_type == "multiclass_classification":
            predicted_class = np.argmax(prediction) if isinstance(prediction, np.ndarray) else 0
            if shap_values.ndim == 3:
                # Shape: (1, n_features, n_classes) - get SHAP values for predicted class
                shap_values_1d = shap_values[0, :, predicted_class]
                print(f"Waterfall: 3D case - Extracted SHAP values for predicted class {predicted_class}: {shap_values_1d.shape}")
            elif shap_values.ndim == 2:
                # Shape: (n_features, n_classes) - get SHAP values for predicted class
                shap_values_1d = shap_values[:, predicted_class]
                print(f"Waterfall: 2D case - Extracted SHAP values for predicted class {predicted_class}: {shap_values_1d.shape}")
        elif shap_values.ndim == 2:
            # For binary classification or other 2D cases: shape is (1, n_features)
            shap_values_1d = shap_values[0]
        else:
            # Fallback for other cases
            shap_values_1d = shap_values.flatten() if shap_values.shape[0] == 1 else shap_values[0]
    else:
        shap_values_1d = shap_values
    
    print(f"Final SHAP values for waterfall: {shap_values_1d.shape if hasattr(shap_values_1d, 'shape') else type(shap_values_1d)}")
    
    # Validate that lengths match
    if len(shap_values_1d) != len(feature_names):
        print(f"Length mismatch in waterfall! SHAP: {len(shap_values_1d)}, Features: {len(feature_names)}")
        # Truncate to match
        min_length = min(len(shap_values_1d), len(feature_names))
        shap_values_1d = shap_values_1d[:min_length]
        feature_names_truncated = feature_names[:min_length]
        st.warning(f"⚠️ Feature dimensions adjusted for display. Showing {min_length} features.")
    else:
        feature_names_truncated = feature_names
    
    try:
        feature_importance = pd.DataFrame({
                'Feature': feature_names_truncated,
                'Value': [input_values[f] if f in input_values else 0 for f in feature_names_truncated],
                'SHAP': shap_values_1d,
                'Abs_SHAP': np.abs(shap_values_1d)
        }).sort_values('Abs_SHAP', ascending=True)
    except Exception as e:
        st.error(f"Error creating waterfall plot data: {str(e)}")
        print(f"DataFrame creation error: {str(e)}")
        # Create a simple fallback display
        st.markdown("**SHAP Values Summary:**")
        for i, (feat, val) in enumerate(zip(feature_names_truncated[:10], shap_values_1d[:10])):
            st.write(f"- **{feat}**: {val:.4f}")
        return
    
    try:
        # Create proper SHAP waterfall that starts from baseline and ends at final prediction
        
        # Calculate final prediction from baseline + sum of SHAP values
        final_prediction = float(base_value + np.sum(shap_values_1d))
        
        # Handle prediction value formatting for display
        if problem_type == "multiclass_classification" and isinstance(prediction, np.ndarray):
            # For multiclass, use the probability of the predicted class (same as SHAP values)
            predicted_class = np.argmax(prediction)
            prediction_value = float(prediction[predicted_class])
        else:
            prediction_value = float(prediction) if isinstance(prediction, (np.ndarray, np.float64)) else prediction
            
        # Validate SHAP explanation accuracy (small tolerance for floating point errors)
        shap_error = abs(final_prediction - prediction_value)
        if shap_error > 0.01:  # Tolerance of 0.01 for small numerical differences
            print(f"Warning: SHAP explanation error = {shap_error:.6f}")
            print(f"Base value: {base_value:.6f}, SHAP sum: {np.sum(shap_values_1d):.6f}, Final calc: {final_prediction:.6f}")
            print(f"Actual prediction: {prediction_value:.6f}")
        
        # Prepare data for waterfall: baseline + features + final prediction
        waterfall_features = ["Base Value"] + feature_importance['Feature'].tolist() + ["Final Prediction"]
        waterfall_values = [float(base_value)] + feature_importance['SHAP'].tolist() + [prediction_value]
        waterfall_measures = ["absolute"] + ["relative"] * len(feature_importance) + ["total"]
        
        # Create text for hover information
        waterfall_text = [f"Base Value: {base_value:.4f}"]
        waterfall_text.extend([f"Feature: {feat}<br>Value: {feature_importance.loc[feature_importance['Feature'] == feat, 'Value'].iloc[0]}<br>Impact: {shap_val:+.4f}" 
                               for feat, shap_val in zip(feature_importance['Feature'], feature_importance['SHAP'])])
        waterfall_text.append(f"Final: {prediction_value:.4f}")
        
        fig = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="h",
            measure=waterfall_measures,
            x=waterfall_values,
            y=waterfall_features,
            connector={"mode": "spanning", "line": {"width": 1, "color": "rgb(63, 63, 63)", "dash": "solid"}},
            decreasing={"marker": {"color": "rgba(50, 171, 96, 0.7)"}},
            increasing={"marker": {"color": "rgba(219, 64, 82, 0.7)"}},
            text=waterfall_text,
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>" +
                         "%{text}<br>" +
                         "<extra></extra>"
        ))
    
        fig.update_layout(
            title="SHAP Waterfall: From Base Value to Final Prediction",
            showlegend=False,
            height=max(500, (len(feature_importance) + 2) * 30),  # +2 for base value and final prediction
            margin=dict(t=50, b=50, l=50, r=50),
            yaxis=dict(title="Features & Values", autorange="reversed"),  # Reverse to show baseline at top
            xaxis=dict(title="Prediction Value")
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(fig, config={'responsive': True})
        with col2:
            if problem_type == "multiclass_classification":
                predicted_class = np.argmax(prediction) if isinstance(prediction, np.ndarray) else 0
                
                # Get actual class names if available for display
                class_names = get_class_names_with_indices()
                if class_names and predicted_class < len(class_names):
                    predicted_class_display = class_names[predicted_class]
                else:
                    predicted_class_display = f"Class {predicted_class}"
                
                st.info(f"""
                        **Reading the Waterfall (Multiclass):**
                        - **Base Value**: Model's average {predicted_class_display} probability
                        - **Green bars**: Features decreasing the probability
                        - **Red bars**: Features increasing the probability
                        - **Final Prediction**: Actual predicted probability
                        - Shows complete path from baseline to prediction
                    """)
            elif problem_type == "binary_classification":
                st.info("""
                        **Reading the Waterfall (Binary):**
                        - **Base Value**: Model's average positive class probability
                        - **Green bars**: Features decreasing positive probability  
                        - **Red bars**: Features increasing positive probability
                        - **Final Prediction**: Actual predicted probability
                        - Shows complete path from baseline to prediction
                    """)
            else:  # regression
                st.info("""
                        **Reading the Waterfall (Regression):**
                        - **Base Value**: Model's average prediction
                        - **Green bars**: Features pushing prediction lower
                        - **Red bars**: Features pushing prediction higher
                        - **Final Prediction**: Actual predicted value
                        - Shows complete path from baseline to prediction
                    """)
    except Exception as e:
        st.error(f"Error creating waterfall plot: {str(e)}")
        print(f"Plotly waterfall error: {str(e)}")
    
    # Add explanation about the waterfall format
    # Get prediction value for display (same logic as in waterfall creation)
    if problem_type == "multiclass_classification" and isinstance(prediction, np.ndarray):
        predicted_class = np.argmax(prediction)
        display_prediction_value = float(prediction[predicted_class])
    else:
        display_prediction_value = float(prediction) if isinstance(prediction, (np.ndarray, np.float64)) else prediction
        
    st.markdown(f"""
    💡 **Understanding the SHAP Waterfall:**
    
    This waterfall chart shows the complete journey from the model's baseline to your specific prediction:
    
    1. **Base Value ({base_value:.4f})**: The model's average prediction across all training data
    2. **Feature contributions**: Each feature either increases (red) or decreases (green) the prediction
    3. **Final Prediction ({display_prediction_value:.4f})**: The actual prediction for your input
    
    The mathematical relationship is: **Base Value + Sum of all SHAP values = Final Prediction**
    """)
    
    # Display feature impact table
    st.markdown("### 📋 Feature Impact Summary")
    try:
        impact_df = feature_importance[['Feature', 'Value', 'SHAP', 'Abs_SHAP']].copy()
        impact_df.columns = ['Feature', 'Input Value', 'Impact', 'Absolute Impact']
        impact_df = impact_df.sort_values('Absolute Impact', ascending=False)
    
        # Format the impact values
        impact_df['Impact'] = impact_df['Impact'].map('{:.4f}'.format)
        impact_df['Absolute Impact'] = impact_df['Absolute Impact'].map('{:.4f}'.format)
    
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(
                impact_df.style.background_gradient(
                    subset=['Impact'],
                    cmap='RdBu',
                    vmin=-abs(impact_df['Impact'].astype(float)).max(),
                    vmax=abs(impact_df['Impact'].astype(float)).max()
                ),
                width='stretch'
            )
        with col2:
            if problem_type == "multiclass_classification":
                predicted_class = np.argmax(prediction) if isinstance(prediction, np.ndarray) else 0
                
                # Get actual class names if available for display
                class_names = get_class_names_with_indices()
                if class_names and predicted_class < len(class_names):
                    predicted_class_display = class_names[predicted_class]
                else:
                    predicted_class_display = f"Class {predicted_class}"
                
                st.info(f"""
                        **Reading the Table (Multiclass):**
                        - Shows each feature's contribution to {predicted_class_display} probability
                        - Positive impact = increases {predicted_class_display} likelihood
                        - Negative impact = decreases {predicted_class_display} likelihood
                        - Sorted by absolute impact (strongest features first)
                    """)
            elif problem_type == "binary_classification":
                st.info("""
                        **Reading the Table (Binary):**
                        - Shows each feature's contribution to positive class probability
                        - Positive impact = increases positive class likelihood
                        - Negative impact = decreases positive class likelihood
                        - Sorted by absolute impact (strongest features first)
                    """)
            else:  # regression
                st.info("""
                        **Reading the Table (Regression):**
                        - Shows each feature's direct contribution to predicted value
                        - Positive impact = increases prediction
                        - Negative impact = decreases prediction
                        - Sorted by absolute impact (strongest features first)
                    """)
    except Exception as e:
        st.error(f"Error creating impact table: {str(e)}")
        print(f"Impact table error: {str(e)}")
        # Show simple text summary as fallback
        st.markdown("**Feature Impact Summary:**")
        try:
            for i, (feat, shap_val) in enumerate(zip(feature_names_truncated, shap_values_1d)):
                st.write(f"{i+1}. **{feat}**: {shap_val:.4f}")
        except Exception as e2:
            st.write("Unable to display feature impacts due to data format issues.")

def render_what_if_analysis():
    st.header("🔮 What If Analysis & Predictions")
                    
    st.markdown("""
    Explore how changes to feature values affect your model's predictions with this interactive tool.
    
    ### 📌 How to Use This "What If" Tool
    1. Enter values for features you want to test or load a random test sample
    2. Experiment by changing values to see how they impact predictions
    3. Compare different scenarios side by side
    4. Understand which changes have the most impact on your model's output
    
    ### 🎯 What You'll Get
    - Immediate prediction results as you modify values
    - Visual explanations showing how each feature influences the outcome
    - Ability to save and compare different scenarios
    - Insights into feature sensitivity and model behavior
    """)
    st.write("---")
    
    # Get problem type and model info
    problem_type = getattr(st.session_state, 'problem_type', st.session_state.builder.model.get("problem_type", "unknown"))
    model_type = st.session_state.builder.model.get("type", "unknown")
    
    # Display model information
    st.markdown(f"""
    ### 📊 Model Information
    - **Type**: {model_type}
    - **Problem Type**: {problem_type}
    """)
    
    # Create feature information
    feature_info = create_feature_info(st.session_state.builder)
    
    # Add option to load sample data from test set
    #st.subheader("📋 Choose a Starting Point")
    
    # Use session state to track if we've already loaded a random sample
    if 'random_sample_loaded' not in st.session_state:
        st.session_state.random_sample_loaded = False
        
    #starting_point_option = st.radio(
    #    "Select a starting point for your analysis:",
    #    ["Start from scratch", "Load a random test sample"],
    #    index=0,
    #    horizontal=True,
    #    key="starting_point_option"
    #)
    starting_point_option = "Start from scratch"
    
    initial_values = {}
    
    # Only load a new random sample if explicitly selected and not already loaded
    if starting_point_option == "Load a random test sample" and not st.session_state.random_sample_loaded:
        # Select a random sample from the test set
        if st.session_state.builder.X_test is not None:
            sample_index = np.random.randint(0, len(st.session_state.builder.X_test))
            sample_data = st.session_state.builder.X_test.iloc[sample_index]
            initial_values = sample_data.to_dict()
            
            # Set flag to prevent reloading on refresh
            st.session_state.random_sample_loaded = True
            # Save the sample in session state for reuse
            st.session_state.random_sample_values = initial_values
            
            actual_value = st.session_state.builder.y_test.iloc[sample_index]
            st.info(f"Loaded random test sample #{sample_index} with actual value: {actual_value:.4f}")
    
    # If we previously loaded a random sample, reuse those values
    elif starting_point_option == "Load a random test sample" and st.session_state.random_sample_loaded:
        if hasattr(st.session_state, 'random_sample_values'):
            initial_values = st.session_state.random_sample_values
            st.success("Using previously loaded random sample")
            
    else:
        # Reset the random sample flag if user selects "Start from scratch"
        st.session_state.random_sample_loaded = False
    
    # Initialize session state for scenarios if it doesn't exist
    if 'what_if_scenarios' not in st.session_state:
        st.session_state.what_if_scenarios = {}
    
    # Create two columns for the form and saved scenarios
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("💾 Saved Scenarios")
        
        # Display saved scenarios
        if not st.session_state.what_if_scenarios:
            st.write("No scenarios saved yet. Use the 'Save Scenario' button after making a prediction.")
        else:
            for scenario_name, scenario_data in st.session_state.what_if_scenarios.items():
                with st.expander(f"{scenario_name} - Prediction: {scenario_data['prediction']:.4f}"):
                    for feature, value in scenario_data['values'].items():
                        st.write(f"**{feature}**: {value}")
                    
                    if st.button(f"Load '{scenario_name}'", key=f"load_{scenario_name}"):
                        # Store in session state for next render
                        if 'initial_values' not in st.session_state:
                            st.session_state.initial_values = {}
                        st.session_state.initial_values = dict(scenario_data['values'])
                        
                        # Reset the random sample flag when loading a saved scenario
                        st.session_state.random_sample_loaded = False
                        
                        # Switch back to "Start from scratch" mode to avoid reloading samples
                        st.session_state.starting_point_option = "Start from scratch"
                        
                        st.success(f"Loaded scenario: {scenario_name}")
                        st.rerun()
    
    # Check if we have initial values in session state and use them
    if 'initial_values' in st.session_state and st.session_state.initial_values:
        initial_values = dict(st.session_state.initial_values)
        # Don't clear the values here to ensure they're used
    
    with col1:
        st.subheader("🔄 Enter Feature Values")
        
        # Always use real-time mode instead of having it as an option
        st.info("Values update predictions in real-time. Adjust any value to see the impact on predictions.")
        
        # Create input form with real-time predictions
        input_values = create_input_form(feature_info, initial_values)
        submitted = True  # Always consider as submitted in real-time mode
    
    # Now that the form is submitted, clear initial values to avoid reusing them on subsequent interactions
    if 'initial_values' in st.session_state and st.session_state.initial_values:
        st.session_state.initial_values = {}
    
    # Store results for comparison
    current_prediction_results = None
    current_input_values = None
    
    if submitted:
        try:
            # Get the original feature order from the training data
            original_features = st.session_state.builder.X_train.columns.tolist()
            
            # Create input DataFrame with features in the correct order
            model_input_values = input_values['model_values']
            ordered_input_data = {feature: model_input_values[feature] for feature in original_features}
            input_df = pd.DataFrame([ordered_input_data])
            
            # Make prediction
            model = st.session_state.builder.model["model"]
            if problem_type in ["binary_classification", "multiclass_classification", "classification"] and hasattr(model, "predict_proba"):
                if problem_type == "binary_classification":
                    prediction = model.predict_proba(input_df)[0][1]
                elif problem_type == "multiclass_classification":
                    prediction = model.predict_proba(input_df)[0]  # Return all class probabilities
                else:  # fallback for legacy "classification"
                    prediction = model.predict_proba(input_df)[0][1]
            else:
                prediction = model.predict(input_df)[0]
            
            # Create SHAP explainer
            # Create background data for the explainer with proper cache invalidation
            train_data_hash = str(pd.util.hash_pandas_object(st.session_state.builder.X_train).sum())
            background_data = get_background_data(st.session_state.builder, train_data_hash, 100)
            
            # Create explainer using utility function
            model_type = st.session_state.builder.model.get("type", "unknown")
            explainer = create_shap_explainer(model, background_data, problem_type, model_type)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(input_df)
            # Processing raw SHAP values
            if isinstance(shap_values, list):
                # Processing SHAP values list
                if len(shap_values) > 0:
                    # Processing first SHAP values
                    pass  # Continue processing below
                
                if problem_type == "binary_classification":
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                elif problem_type == "multiclass_classification":
                    # For multiclass, focus on the predicted class
                    if isinstance(prediction, np.ndarray) and len(prediction) > 1:
                        predicted_class = np.argmax(prediction)
                        # Using predicted class SHAP values
                        shap_values = shap_values[predicted_class] if predicted_class < len(shap_values) else shap_values[0]
                    else:
                        shap_values = shap_values[0]
                else:  # fallback for legacy "classification"
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                # Processing SHAP values array (not list)
                pass  # Continue processing below
            
            print(f"Final SHAP values after class selection: {shap_values.shape if hasattr(shap_values, 'shape') else type(shap_values)}")
            
            # Get base value based on model type and problem type
            if 'xgboost' in model_type or 'lightgbm' in model_type:
                base_value = explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    if problem_type == "multiclass_classification" and isinstance(prediction, np.ndarray):
                        # For multiclass, use the expected value for the predicted class
                        predicted_class = np.argmax(prediction)
                        base_value = base_value[predicted_class] if predicted_class < len(base_value) else base_value[0]
                    elif problem_type == "binary_classification":
                        base_value = base_value[1] if len(base_value) > 1 else base_value[0]
                    else:
                        # For regression, base_value might be a scalar (e.g., XGBoost TreeExplainer)
                        if hasattr(base_value, '__len__'):  # Check if it has length before calling len()
                            base_value = base_value[0] if len(base_value) > 0 else 0.0
                        # If it's already a scalar, use as-is
            else:
                if isinstance(explainer.expected_value, (list, np.ndarray)):
                    if problem_type == "multiclass_classification" and isinstance(prediction, np.ndarray):
                        # For multiclass, use the expected value for the predicted class
                        predicted_class = np.argmax(prediction)
                        base_value = explainer.expected_value[predicted_class] if predicted_class < len(explainer.expected_value) else explainer.expected_value[0]
                    elif problem_type == "binary_classification":
                        base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
                    else:
                        base_value = explainer.expected_value[0] if len(explainer.expected_value) > 0 else 0.0
                else:
                    base_value = explainer.expected_value
            
            # Ensure SHAP values are 1D for force plot (single sample)
            print(f"SHAP values shape before processing: {shap_values.shape if hasattr(shap_values, 'shape') else type(shap_values)}")
            if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
                shap_values_1d = shap_values[0]  # Take the first (and only) sample
                print(f"SHAP values shape after indexing: {shap_values_1d.shape}")
            else:
                shap_values_1d = shap_values
                print(f"SHAP values already 1D: {shap_values_1d.shape if hasattr(shap_values_1d, 'shape') else type(shap_values_1d)}")
            
            print(f"Base value: {base_value} (type: {type(base_value)})")
            print(f"Problem type: {problem_type}")
            print(f"Prediction type: {type(prediction)}, shape: {prediction.shape if hasattr(prediction, 'shape') else 'no shape'}")
            
            # Ensure feature names match the input DataFrame
            input_feature_names = list(input_df.columns)
            print(f"Input DataFrame columns: {len(input_feature_names)}")
            print(f"Original features list: {len(original_features)}")
            print(f"SHAP values length: {len(shap_values_1d) if hasattr(shap_values_1d, '__len__') else 'no length'}")
            
            # Use input DataFrame column names to ensure exact match
            feature_names_to_use = input_feature_names
            
            # Display results
            display_prediction_results(
                prediction,
                shap_values_1d,
                feature_names_to_use,  # Use input DataFrame column names for exact match
                ordered_input_data,  # Use ordered input data
                explainer,
                problem_type
            )
            
            # Log the prediction - handle multiclass format
            if problem_type == "multiclass_classification" and isinstance(prediction, np.ndarray):
                log_prediction = float(np.max(prediction))  # Use max probability for logging
            else:
                log_prediction = float(prediction) if isinstance(prediction, (np.ndarray, np.float64)) else prediction
            
            st.session_state.logger.log_calculation(
                "Model Prediction",
                {
                    "input_values": ordered_input_data,
                    "prediction": log_prediction,
                    "model_type": model_type,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Store results for comparison - handle multiclass predictions
            if problem_type == "multiclass_classification" and isinstance(prediction, np.ndarray):
                # For multiclass, store full probability array and summary metrics
                current_prediction_results = {
                    'probabilities': prediction.tolist(),
                    'predicted_class': int(np.argmax(prediction)),
                    'max_probability': float(np.max(prediction)),
                    'scalar_value': float(np.max(prediction))  # For backward compatibility with comparisons
                }
            else:
                # For binary classification and regression, store as before
                current_prediction_results = float(prediction) if isinstance(prediction, (np.ndarray, np.float64)) else prediction
            
            current_input_values = dict(input_values['all_values'])
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.session_state.logger.log_error(
                "Prediction Error",
                {
                    "error": str(e),
                    "input_values": model_input_values,
                    "timestamp": datetime.now().isoformat()
                }
            )

    # Add scenario saving and comparison after prediction
    if current_prediction_results is not None and current_input_values is not None:
        st.markdown("---")
        st.subheader("📊 Scenario Management")
        
        # Add help section for scenario management
        with st.expander("ℹ️ How to Use Scenario Management", expanded=False):
            st.markdown("""
            ## 📊 Scenario Management: Complete Guide
            
            The Scenario Management tool lets you save, compare, and analyse different feature configurations to understand how your model responds to changes in input values.
            
            ### 💾 Creating & Managing Scenarios
            
            #### Creating Scenarios
            1. Adjust feature values in the input form above
            2. Once you have a prediction result, enter a descriptive name (e.g., "Baseline", "High Income")
            3. Click "Save Current Scenario" to store this configuration
            4. The scenario will appear in the Saved Scenarios panel on the right
            
            #### Loading Scenarios
            - To retrieve a saved scenario, find it in the Saved Scenarios panel
            - Click "Load '[scenario name]'" to populate the input form with those values
            - This lets you start from a previous configuration instead of from scratch
            
            ### 🔍 Comparing Scenarios
            
            #### Setting Up Comparisons
            1. Save at least two different scenarios
            2. In the Compare Scenarios section, select two scenarios from the dropdown menus
            3. Click "Compare Selected Scenarios" to generate a detailed comparison
            
            #### Understanding the Comparison Results
            
            ##### Prediction Metrics
            - Shows the prediction value for each scenario and the absolute/percentage difference
            - Metrics display shows the values for each scenario and the difference between them
            
            ##### Feature Comparison Table
            - Displays only the features that differ between scenarios
            - Color-coded to show increases (green) and decreases (red)
            - Shows the exact differences in values
            
            ##### Visual Analysis Tools
            The comparison includes three visualization tabs:
            
            1. **Feature Changes**
                - Displays a detailed table of all features that differ between scenarios
                - Color-coded to highlight the direction of changes
                - Bar chart showing the magnitude and direction of changes for each feature
                - Helps identify which features changed the most between scenarios
            
            2. **Feature Importance**
                - Displays a radar chart showing the importance of each feature in explaining the prediction difference
                - Feature importance table showing which features contributed most to the prediction difference
                - Based on SHAP values that measure each feature's impact on model output
                - Sorted by magnitude of impact to highlight the most influential changes
            
            3. **Isolated Feature Effects**
                - Interactive tool to isolate the impact of individual features
                - Shows exactly how much a single feature contributes to the overall difference
                - Displays metrics showing the feature change and its prediction impact
                - Shows whether the feature supports or opposes the overall prediction difference
                - Provides a bar chart visualization of how the feature affects the prediction
                - Includes detailed insights based on the feature's impact strength
            
            ### 📥 Exporting Analysis Results
            
            The Export Report feature creates a comprehensive JSON file containing:
            
            - Complete metadata about both scenarios
            - Detailed feature-by-feature comparison
            - SHAP values showing feature importance
            - Isolated feature impact calculations
            - Model information and prediction explanations
            
            To export:
            1. After comparing scenarios, scroll to the Export Report section
            2. Click "Generate & Download Report"
            3. When processing completes, click the "Download Report" button to save the JSON file
            
            ### 💡 Pro Tips for Effective Analysis
            
            - **Baseline Comparison**: Always create a "Baseline" scenario with default or average values to compare against   
            - **Sensitivity Testing**: Create scenario sequences that incrementally change a feature to see threshold effects
            - **Feature Interactions**: Test combinations of feature changes to identify interaction effects in the model
            - **Outlier Analysis**: Create scenarios with extreme values to test model robustness
            - **Supporting vs Opposing**: Pay attention to whether features support or oppose the overall prediction change
            
            The more scenarios you create and compare, the deeper your understanding of model behavior will become!
            """)
        
        # Save current scenario
        col1, col2 = st.columns([1, 2])
        
        with col1:
            scenario_name = st.text_input("Scenario Name", value=f"Scenario {len(st.session_state.what_if_scenarios) + 1}")
            
            if st.button("💾 Save Current Scenario"):
                if scenario_name in st.session_state.what_if_scenarios:
                    st.warning(f"Scenario '{scenario_name}' already exists. It will be overwritten.")
                
                # Ensure we're storing a deep copy and proper types
                if isinstance(current_prediction_results, dict):
                    # Multiclass prediction with metadata
                    st.session_state.what_if_scenarios[scenario_name] = {
                        'values': dict(current_input_values),
                        'prediction': current_prediction_results['scalar_value'],  # Store scalar for compatibility
                        'prediction_metadata': current_prediction_results  # Store full metadata
                    }
                else:
                    # Binary classification or regression
                    st.session_state.what_if_scenarios[scenario_name] = {
                        'values': dict(current_input_values),
                        'prediction': float(current_prediction_results)
                    }
                
                # Provide feedback and log save action
                st.success(f"""
                    ✅ Scenario '{scenario_name}' saved successfully!
                    
                    You can now:
                    - Continue modifying values
                    - Create additional scenarios to compare
                    - Load this scenario later from the Saved Scenarios panel
                """)
                
                # Log the save action - handle multiclass format
                if isinstance(current_prediction_results, dict):
                    log_prediction = current_prediction_results['scalar_value']
                else:
                    log_prediction = float(current_prediction_results) if isinstance(current_prediction_results, (np.ndarray, np.float64)) else current_prediction_results
                
                st.session_state.logger.log_user_action(
                    "Scenario Saved",
                    {
                        "scenario_name": scenario_name,
                        "prediction": log_prediction,
                        "timestamp": datetime.now().isoformat()
                    }
                )
        
        with col2:
            if len(st.session_state.what_if_scenarios) >= 2:
                st.subheader("🔍 Compare Scenarios")
                
                # Select scenarios to compare
                col1, col2 = st.columns(2)
                
                with col1:
                    scenario1 = st.selectbox("Select first scenario", 
                                            options=list(st.session_state.what_if_scenarios.keys()),
                                            key="scenario1")
                
                with col2:
                    remaining_scenarios = [s for s in st.session_state.what_if_scenarios.keys() if s != scenario1]
                    scenario2 = st.selectbox("Select second scenario", 
                                            options=remaining_scenarios,
                                            key="scenario2")
                
                # Store the currently compared scenarios in session state for persistence
                if 'current_comparison' not in st.session_state:
                    st.session_state.current_comparison = {
                        'is_active': False,
                        'scenario1': None,
                        'scenario2': None,
                        'data1': None,
                        'data2': None,
                        'pred_diff': None,
                        'pred_diff_pct': None,
                        'diff_features': [],
                        'numeric_features': []
                    }
                
                # Use a button to trigger comparison or use stored values
                compare_button = st.button("Compare Selected Scenarios")
                
                # Clear comparison if scenarios change
                if (st.session_state.current_comparison['is_active'] and 
                    (st.session_state.current_comparison['scenario1'] != scenario1 or 
                        st.session_state.current_comparison['scenario2'] != scenario2)):
                    st.session_state.current_comparison['is_active'] = False
                
                # Perform comparison when button clicked or if we already have a stored comparison
                if compare_button or st.session_state.current_comparison['is_active']:
                    # If it's a new comparison, store the data
                    if compare_button or not st.session_state.current_comparison['is_active']:
                        # Get scenario data
                        data1 = st.session_state.what_if_scenarios[scenario1]
                        data2 = st.session_state.what_if_scenarios[scenario2]
                        
                        # Calculate prediction difference - handle multiclass
                        pred1 = data1['prediction']
                        pred2 = data2['prediction']
                        
                        # Handle multiclass predictions
                        if 'prediction_metadata' in data1 and 'prediction_metadata' in data2:
                            # Both are multiclass
                            meta1 = data1['prediction_metadata']
                            meta2 = data2['prediction_metadata']
                            pred_diff = pred2 - pred1  # Using scalar values for comparison
                            pred_diff_pct = (pred_diff / pred1) * 100 if pred1 != 0 else 0
                            
                            # Additional multiclass comparison info
                            class_change = meta2['predicted_class'] != meta1['predicted_class']
                            confidence_diff = meta2['max_probability'] - meta1['max_probability']
                        else:
                            # Binary classification or regression
                            pred_diff = pred2 - pred1
                            pred_diff_pct = (pred_diff / pred1) * 100 if pred1 != 0 else 0
                            class_change = False
                            confidence_diff = None
                            
                        # Find differing features
                        diff_features = []
                        for feature in data1['values'].keys():
                            if feature in data2['values'] and data1['values'][feature] != data2['values'][feature]:
                                try:
                                    # Try to compute numerical difference
                                    if isinstance(data1['values'][feature], (int, float)) and isinstance(data2['values'][feature], (int, float)):
                                        diff_value = f"{data2['values'][feature] - data1['values'][feature]:.4f}"
                                    else:
                                        diff_value = "N/A"
                                    
                                    diff_features.append({
                                        'Feature': feature,
                                        f'{scenario1}': data1['values'][feature],
                                        f'{scenario2}': data2['values'][feature],
                                        'Difference': diff_value
                                    })
                                except Exception as e:
                                    # Handle any comparison errors
                                    st.warning(f"Error comparing feature {feature}: {str(e)}")
                                    diff_features.append({
                                        'Feature': feature,
                                        f'{scenario1}': str(data1['values'][feature]),
                                        f'{scenario2}': str(data2['values'][feature]),
                                        'Difference': "Error"
                                    })
                                    
                            # Handle case where the two scenarios have different features
                            for feature in data1['values'].keys():
                                if feature not in data2['values']:
                                    diff_features.append({
                                        'Feature': feature,
                                        f'{scenario1}': data1['values'][feature],
                                        f'{scenario2}': "Missing",
                                        'Difference': "N/A"
                                    })
                            
                            for feature in data2['values'].keys():
                                if feature not in data1['values']:
                                    diff_features.append({
                                        'Feature': feature,
                                        f'{scenario1}': "Missing",
                                        f'{scenario2}': data2['values'][feature],
                                        'Difference': "N/A"
                                    })
                                    
                            # Find numeric features that can be compared
                            numeric_features = []
                            for feat in diff_features:
                                feature = feat['Feature']
                                if (feature in data1['values'] and feature in data2['values'] and
                                    isinstance(data1['values'][feature], (int, float)) and
                                    isinstance(data2['values'][feature], (int, float))):
                                    numeric_features.append(feature)
                            
                            # Store all data in session state
                            st.session_state.current_comparison = {
                                'is_active': True,
                                'scenario1': scenario1,
                                'scenario2': scenario2,
                                'data1': data1,
                                'data2': data2,
                                'pred_diff': pred_diff,
                                'pred_diff_pct': pred_diff_pct,
                                'diff_features': diff_features,
                                'numeric_features': numeric_features,
                                'class_change': class_change,
                                'confidence_diff': confidence_diff
                            }
                    else:
                        # Retrieve stored comparison data
                        scenario1 = st.session_state.current_comparison['scenario1']
                        scenario2 = st.session_state.current_comparison['scenario2']
                        data1 = st.session_state.current_comparison['data1']
                        data2 = st.session_state.current_comparison['data2']
                        pred_diff = st.session_state.current_comparison['pred_diff']
                        pred_diff_pct = st.session_state.current_comparison['pred_diff_pct']
                        diff_features = st.session_state.current_comparison['diff_features']
                        numeric_features = st.session_state.current_comparison['numeric_features']
                        class_change = st.session_state.current_comparison.get('class_change', False)
                        confidence_diff = st.session_state.current_comparison.get('confidence_diff', None)
                    
                    # Display prediction comparison
                    st.markdown(f"### Prediction Comparison")
                    
                    # Handle multiclass display
                    if 'prediction_metadata' in data1 and 'prediction_metadata' in data2:
                        # Multiclass comparison
                        meta1 = data1['prediction_metadata']
                        meta2 = data2['prediction_metadata']
                        
                        comparison_cols = st.columns(4)
                        
                        # Get actual class names if available
                        class_names = get_class_names_with_indices()
                        
                        with comparison_cols[0]:
                            class1_idx = meta1['predicted_class']
                            if class_names and class1_idx < len(class_names):
                                class1_display = class_names[class1_idx]
                            else:
                                class1_display = f"Class {class1_idx}"
                            st.metric(f"{scenario1} Class", class1_display)
                        
                        with comparison_cols[1]:
                            st.metric(f"{scenario1} Confidence", f"{meta1['max_probability']:.1%}")
                        
                        with comparison_cols[2]:
                            class2_idx = meta2['predicted_class']
                            if class_names and class2_idx < len(class_names):
                                class2_display = class_names[class2_idx]
                            else:
                                class2_display = f"Class {class2_idx}"
                            st.metric(f"{scenario2} Class", class2_display)
                            
                        with comparison_cols[3]:
                            st.metric(f"{scenario2} Confidence", f"{meta2['max_probability']:.1%}")
                        
                        # Show class change and confidence difference
                        if class_change:
                            st.warning(f"🔄 **Class Change**: {class1_display} → {class2_display}")
                        else:
                            st.info(f"✅ **No Class Change**: Both predict {class1_display}")
                        
                        if confidence_diff is not None:
                            conf_change_pct = confidence_diff * 100
                            st.metric("Confidence Change", 
                                    f"{conf_change_pct:+.1f}%",
                                    delta=f"{confidence_diff:+.3f}")
                    else:
                        # Binary classification or regression
                        comparison_cols = st.columns(3)
                        with comparison_cols[0]:
                            st.metric(f"{scenario1}", f"{data1['prediction']:.4f}")
                        
                        with comparison_cols[1]:
                            st.metric(f"{scenario2}", f"{data2['prediction']:.4f}")
                            
                        with comparison_cols[2]:
                            st.metric("Difference", 
                                    f"{pred_diff:.4f}", 
                                    delta=f"{pred_diff_pct:.2f}%")
                    
                    # Find differing features
                    diff_features = []
                    for feature in data1['values'].keys():
                        if feature in data2['values'] and data1['values'][feature] != data2['values'][feature]:
                            try:
                                # Try to compute numerical difference
                                if isinstance(data1['values'][feature], (int, float)) and isinstance(data2['values'][feature], (int, float)):
                                    diff_value = f"{data2['values'][feature] - data1['values'][feature]:.4f}"
                                else:
                                    diff_value = "N/A"
                                
                                diff_features.append({
                                    'Feature': feature,
                                    f'{scenario1}': data1['values'][feature],
                                    f'{scenario2}': data2['values'][feature],
                                    'Difference': diff_value
                                })
                            except Exception as e:
                                # Handle any comparison errors
                                st.warning(f"Error comparing feature {feature}: {str(e)}")
                                diff_features.append({
                                    'Feature': feature,
                                    f'{scenario1}': str(data1['values'][feature]),
                                    f'{scenario2}': str(data2['values'][feature]),
                                    'Difference': "Error"
                                })
                                
                    # Handle case where the two scenarios have different features
                    for feature in data1['values'].keys():
                        if feature not in data2['values']:
                            diff_features.append({
                                'Feature': feature,
                                f'{scenario1}': data1['values'][feature],
                                f'{scenario2}': "Missing",
                                'Difference': "N/A"
                            })
                    
                    for feature in data2['values'].keys():
                        if feature not in data1['values']:
                            diff_features.append({
                                'Feature': feature,
                                f'{scenario1}': "Missing",
                                f'{scenario2}': data2['values'][feature],
                                'Difference': "N/A"
                            })
                    
                    if diff_features:
                        diff_df = pd.DataFrame(diff_features)
                        
                        # Improve the styling of the dataframe with conditional formatting
                        def highlight_differences(s):
                            # Check if this is the Difference column
                            if s.name == 'Difference':
                                # Create an empty list to store styles for each cell
                                styles = []
                                # Process each value in the series individually
                                for val in s:
                                    if val not in ['N/A', 'Error']:
                                        try:
                                            num_val = float(val)
                                            if num_val < 0:
                                                styles.append('background-color: rgba(255,100,100,0.2)')
                                            else:
                                                styles.append('background-color: rgba(100,255,100,0.2)')
                                        except (ValueError, TypeError):
                                            styles.append('')
                                    else:
                                        styles.append('')
                                return styles
                            # For all other columns, return empty styles
                            return ['' for _ in range(len(s))]
                        
                        # Visualization for feature changes
                        numeric_features = []
                        
                        # Find numeric features that can be compared
                        for feat in diff_features:
                            feature = feat['Feature']
                            if (feature in data1['values'] and feature in data2['values'] and
                                isinstance(data1['values'][feature], (int, float)) and
                                isinstance(data2['values'][feature], (int, float))):
                                numeric_features.append(feature)
                        
                        if numeric_features:
                            # Create pills for different visualisations instead of tabs to preserve state
                            selected_viz = st.pills(
                                "Choose Analysis View:",
                                ["Feature Changes", "Feature Importance", "Isolated Feature Effects"],
                                default="Feature Changes",
                                key=f"viz_pills_{scenario1}_{scenario2}"
                            )
                            
                            if selected_viz == "Feature Changes":
                                # Log tab selection
                                st.session_state.logger.log_user_action("Visualization Tab Selected", {
                                    "tab": "Feature Changes",
                                    "comparison": f"{scenario1} vs {scenario2}"
                                })

                                # Create feature difference table
                                st.markdown("### Feature Comparison")
                                st.dataframe(diff_df.style.apply(highlight_differences, axis=0), width='stretch')
                                
                                # Create bar chart of changes
                                changes = [(data2['values'][f] - data1['values'][f]) for f in numeric_features]
                                
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=numeric_features,
                                    y=changes,
                                    marker_color=['rgba(219, 64, 82, 0.7)' if x < 0 else 'rgba(50, 171, 96, 0.7)' for x in changes]
                                ))
                                
                                fig.update_layout(
                                    title=f"Feature Changes from {scenario1} to {scenario2}",
                                    xaxis_title="Features",
                                    yaxis_title="Change in Value",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, config={'responsive': True})
                                
                                # Log the feature changes visualization
                                st.session_state.logger.log_calculation("Visualization", {
                                    "type": "Feature Changes Bar Chart",
                                    "feature_count": len(numeric_features),
                                    "scenario_comparison": f"{scenario1} vs {scenario2}"
                                })
                            
                            elif selected_viz == "Feature Importance":
                                # Log tab selection
                                st.session_state.logger.log_user_action("Visualization Tab Selected", {
                                    "tab": "Feature Importance",
                                    "comparison": f"{scenario1} vs {scenario2}"
                                })
                                
                                st.subheader("Feature Impact on Prediction Difference")
                                
                                # Calculate SHAP values for both scenarios to understand impact
                                try:
                                    # Cache key for SHAP values
                                    shap_cache_key = f"shap_values_{scenario1}_{scenario2}"
                                    
                                    # Check if we've already calculated SHAP values for these scenarios
                                    if shap_cache_key not in st.session_state:
                                        with st.spinner("Calculating feature impacts..."):
                                            # Create input DataFrames
                                            # Get the original feature order from the training data
                                            original_features = st.session_state.builder.X_train.columns.tolist()
                                            
                                            # Filter values to only include model features in the correct order
                                            filtered_values1 = {feature: data1['values'][feature] for feature in original_features if feature in data1['values']}
                                            filtered_values2 = {feature: data2['values'][feature] for feature in original_features if feature in data2['values']}
                                            
                                            # Create input DataFrames with only the model features in the correct order
                                            input_df1 = pd.DataFrame([filtered_values1])[original_features]
                                            input_df2 = pd.DataFrame([filtered_values2])[original_features]
                                            
                                            # Use the model and background data for SHAP analysis
                                            model = st.session_state.builder.model["model"]
                                            
                                            # Use cached background data with proper invalidation
                                            train_data_hash = str(pd.util.hash_pandas_object(st.session_state.builder.X_train).sum())
                                            background_data = get_background_data(st.session_state.builder, train_data_hash, 100)
                                            
                                            # Create explainer using utility function
                                            model_type = st.session_state.builder.model.get("type", "unknown")
                                            explainer = create_shap_explainer(model, background_data, problem_type, model_type)
                                            
                                            # Calculate SHAP values
                                            shap_values1 = explainer.shap_values(input_df1)
                                            shap_values2 = explainer.shap_values(input_df2)
                                            
                                            # Handle multiclass SHAP values - process both list and array formats
                                            def process_shap_values(shap_vals, data_dict, problem_type):
                                                if isinstance(shap_vals, list):
                                                    if problem_type == "binary_classification":
                                                        return shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                                                    elif problem_type == "multiclass_classification":
                                                        # For multiclass, use the predicted class for the scenario
                                                        if 'prediction_metadata' in data_dict:
                                                            pred_class = data_dict['prediction_metadata']['predicted_class']
                                                            return shap_vals[pred_class] if pred_class < len(shap_vals) else shap_vals[0]
                                                        else:
                                                            return shap_vals[0]
                                                    else:  # fallback
                                                        return shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                                                else:
                                                    # Handle array format (from TreeExplainer or KernelExplainer)
                                                    if problem_type == "multiclass_classification" and shap_vals.ndim > 1:
                                                        if shap_vals.ndim == 3:  # (1, n_features, n_classes)
                                                            if 'prediction_metadata' in data_dict:
                                                                pred_class = data_dict['prediction_metadata']['predicted_class']
                                                                return shap_vals[0, :, pred_class]
                                                            else:
                                                                # Default to first class if no metadata
                                                                return shap_vals[0, :, 0]
                                                        elif shap_vals.ndim == 2:  # (n_features, n_classes)
                                                            if 'prediction_metadata' in data_dict:
                                                                pred_class = data_dict['prediction_metadata']['predicted_class']
                                                                return shap_vals[:, pred_class]
                                                            else:
                                                                return shap_vals[:, 0]
                                                    else:
                                                        # For binary classification or regression
                                                        return shap_vals[0] if shap_vals.ndim > 1 else shap_vals
                                            
                                            # Process SHAP values for both scenarios
                                            shap_values1_processed = process_shap_values(shap_values1, data1, problem_type)
                                            shap_values2_processed = process_shap_values(shap_values2, data2, problem_type)
                                            
                                            print(f"Scenario comparison - Processed SHAP values shapes:")
                                            print(f"  Scenario 1: {shap_values1_processed.shape if hasattr(shap_values1_processed, 'shape') else type(shap_values1_processed)}")
                                            print(f"  Scenario 2: {shap_values2_processed.shape if hasattr(shap_values2_processed, 'shape') else type(shap_values2_processed)}")
                                            
                                            # Calculate the difference in SHAP values
                                            shap_diff = shap_values2_processed - shap_values1_processed
                                            shap_diff_abs = np.abs(shap_diff)
                                            
                                            # Get base value based on model type
                                            if 'xgboost' in model_type or 'lightgbm' in model_type:
                                                base_value = explainer.expected_value
                                                if isinstance(base_value, list):
                                                    base_value = base_value[1] if len(base_value) > 1 else base_value[0]
                                            else:
                                                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                                            
                                            # Store the results in session state
                                            st.session_state[shap_cache_key] = {
                                                'shap_values1': shap_values1_processed,
                                                'shap_values2': shap_values2_processed,
                                                'shap_diff': shap_diff,
                                                'shap_diff_abs': shap_diff_abs,
                                                'base_value': base_value
                                            }
                                    
                                    # Retrieve cached SHAP values
                                    shap_data = st.session_state[shap_cache_key]
                                    shap_values1 = shap_data['shap_values1']
                                    shap_values2 = shap_data['shap_values2']
                                    shap_diff = shap_data['shap_diff']
                                    shap_diff_abs = shap_data['shap_diff_abs']
                                    base_value = shap_data['base_value']
                                    
                                    # Create a dataframe for the SHAP differences
                                    original_features = st.session_state.builder.X_train.columns.tolist()
                                    shap_diff_df = pd.DataFrame({
                                        'Feature': original_features,
                                        'SHAP Impact': shap_diff,
                                        'Absolute Impact': np.abs(shap_diff)
                                    }).sort_values('Absolute Impact', ascending=False)
                                    
                                    # Get max SHAP difference for scaling
                                    max_shap_diff = np.max(np.abs(shap_diff))
                                    
                                    # Log the SHAP difference calculation
                                    st.session_state.logger.log_calculation("SHAP Difference Analysis", {
                                        "max_shap_diff": float(max_shap_diff),
                                        "feature_count": len(original_features),
                                        "top_feature": original_features[np.argmax(np.abs(shap_diff))],
                                        "top_impact": float(np.max(np.abs(shap_diff)))
                                    })
                                    
                                    try:
                                        # Normalize SHAP differences to 0-1 range for radar plot
                                        normalized_importance = shap_diff_abs / max_shap_diff if max_shap_diff > 0 else np.zeros_like(shap_diff_abs)
                                        
                                        # Prepare radar data
                                        radar_features = original_features.copy()
                                        # Close the loop for the radar chart
                                        radar_features.append(radar_features[0])
                                        radar_data = np.append(normalized_importance, normalized_importance[0])
                                        
                                        # Create radar chart
                                        radar_fig = go.Figure()
                                        radar_fig.add_trace(go.Scatterpolar(
                                            r=radar_data,
                                            theta=radar_features,
                                            fill='toself',
                                            name="Feature Importance",
                                            line_color='rgba(50, 171, 96, 0.7)',
                                            fillcolor='rgba(50, 171, 96, 0.2)'
                                        ))
                                        
                                        radar_fig.update_layout(
                                            polar=dict(
                                                radialaxis=dict(
                                                    visible=True,
                                                    range=[0, 1],
                                                    title="Importance"
                                                )
                                            ),
                                            title="Feature Importance in Prediction Difference",
                                            height=500,
                                            showlegend=True
                                        )
                                        
                                        st.plotly_chart(radar_fig, config={'responsive': True})
                                        
                                        # Log the radar chart creation
                                        st.session_state.logger.log_calculation("Visualization", {
                                            "type": "Radar Chart",
                                            "features": len(original_features),
                                            "scenario_comparison": f"{scenario1} vs {scenario2}"
                                        })
                                        
                                        st.info("""
                                            **How to read this radar chart:**
                                            - The values show the importance of each feature in explaining the prediction difference
                                            - Larger values indicate features that contributed more to the difference
                                            - The shape shows which features had the most impact on the prediction change
                                            - This helps identify which features were most influential in creating the difference between scenarios
                                        """)
                                    except Exception as e:
                                        st.error(f"Error creating radar chart: {str(e)}")
                                
                                    
                                    # Always show the importance table
                                    st.subheader("Feature Importance Table")
                                    
                                    # Create importance DataFrame with both raw and absolute values
                                    importance_df = pd.DataFrame({
                                        'Feature': original_features,
                                        'Importance': shap_diff_abs,
                                        'Raw Impact': shap_diff
                                    }).sort_values('Importance', ascending=False)
                                    
                                    # Display with styling
                                    st.dataframe(
                                        importance_df.style.background_gradient(
                                            subset=['Importance'],
                                            cmap='YlOrRd',
                                            vmin=0,
                                            vmax=max_shap_diff
                                        ),
                                        width='stretch'
                                    )
                                    
                                    st.info("""
                                        **Understanding this table:**
                                        - **Importance**: The absolute magnitude of each feature's contribution
                                        - **Raw Impact**: The actual impact (positive/negative) on the prediction
                                        - Features with higher importance have more influence on the prediction difference
                                    """)
                                    
                                except Exception as e:
                                    st.error(f"Error calculating feature impacts: {str(e)}")
                                    # Log the error
                                    st.session_state.logger.log_error("Feature Impact Calculation Error", {
                                        "error": str(e),
                                        "scenario_comparison": f"{scenario1} vs {scenario2}"
                                    })
                                    import traceback
                                    st.error(traceback.format_exc())
                            
                            elif selected_viz == "Isolated Feature Effects":
                                # Log tab selection
                                st.session_state.logger.log_user_action("Visualization Tab Selected", {
                                    "tab": "Isolated Feature Effects",
                                    "comparison": f"{scenario1} vs {scenario2}"
                                })
                                
                                st.subheader("Isolated Feature Effects")
                                
                                st.markdown("""
                                    This analysis shows how changing each feature individually would affect the prediction.
                                    Select a feature to see its isolated impact.
                                """)
                                
                                # Initialize the session state for feature analysis only once
                                if 'isolated_feature_analysis' not in st.session_state:
                                    st.session_state.isolated_feature_analysis = {}
                                
                                # Cache key for this specific comparison
                                analysis_cache_key = f"feature_analysis_{scenario1}_{scenario2}"
                                if analysis_cache_key not in st.session_state.isolated_feature_analysis:
                                    st.session_state.isolated_feature_analysis[analysis_cache_key] = {
                                        'feature': None,
                                        'hybrid_pred': None,
                                        'feature_impact': None,
                                        'importance_score': None,
                                        'same_direction': None
                                    }
                                
                                # Cached analysis for current comparison
                                current_analysis = st.session_state.isolated_feature_analysis[analysis_cache_key]
                                
                                # Show an expanded explanation
                                with st.expander("ℹ️ Understanding Isolated Feature Effects", expanded=False):
                                    st.markdown("""
                                    ## How Isolated Feature Effects Work
                                    
                                    This tool helps you understand how individual features influence your model's predictions by isolating their impact.
                                    
                                    ### What This Analysis Shows
                                    
                                    1. **Base Scenario**: The starting point (left bar in chart)
                                    2. **Feature-Only Change**: What happens when you change ONLY the selected feature (middle bar)
                                    3. **Target Scenario**: What happens when ALL features change (right bar)
                                    
                                    ### How to Read the Results
                                    
                                    - **Feature Change**: Shows how the selected feature value changed between scenarios
                                    - **Prediction Impact**: Shows the new prediction when only this feature changes
                                    - **Contribution**: The absolute impact this feature has on the prediction
                                        - A larger absolute value indicates a stronger effect on the prediction
                                        - The direction (Supporting/Opposing) tells you if this effect aligns with the overall change
                                        
                                    ### Direction Indicators
                                    
                                    - **Supporting** ✓: The feature change pushes the prediction in the same direction as the overall change
                                    - **Opposing** ✗: The feature change works against the overall prediction difference
                                    
                                    ### Example
                                    
                                    If comparing a "Low Risk" scenario to a "High Risk" scenario with a prediction increase of 0.20:
                                    - A feature with impact +0.15 (Supporting ✓) contributes strongly to the increased risk in the same direction
                                    - A feature with impact -0.10 (Opposing ✗) actually tries to reduce the risk, but is outweighed by other factors
                                    """)
                                
                                # Get the original feature order from the training data
                                original_features = st.session_state.builder.X_train.columns.tolist()
                                
                                # Filter to only show features that are in the model
                                model_numeric_features = [f for f in numeric_features if f in original_features]
                                
                                if not model_numeric_features:
                                    st.warning("No numeric model features found in the differences between scenarios.")
                                else:
                                    # Let user select a feature to analyse - no form needed, just direct selectbox
                                    selected_feature = st.selectbox(
                                        "Select a feature to analyse:",
                                        options=model_numeric_features,
                                        key=f"selected_feature_for_analysis_{scenario1}_{scenario2}"
                                    )
                                    
                                    # Automatically perform analysis when feature is selected
                                    if selected_feature:
                                        try:
                                            # Get the model
                                            model = st.session_state.builder.model["model"]
                                            
                                            # Only recalculate if we don't have cached results for this feature
                                            if current_analysis.get('feature') != selected_feature:
                                                with st.spinner(f"Analyzing impact of '{selected_feature}'..."):
                                                    # Log user action for feature analysis
                                                    st.session_state.logger.log_user_action("Feature Analysis", {
                                                        "feature": selected_feature,
                                                        "scenario1": scenario1,
                                                        "scenario2": scenario2
                                                    })
                                                    
                                                    # Create a hybrid scenario with all features from scenario1 except the selected one
                                                    hybrid_data = {}
                                                    
                                                    # First, fill with all required features from scenario1
                                                    for f in original_features:
                                                        if f in data1['values']:
                                                            hybrid_data[f] = data1['values'][f]
                                                        else:
                                                            # Use mean from training data for missing features
                                                            hybrid_data[f] = float(st.session_state.builder.X_train[f].mean())
                                                    
                                                    # Update only the selected feature from scenario2
                                                    if selected_feature in data2['values']:
                                                        hybrid_data[selected_feature] = data2['values'][selected_feature]
                                                    
                                                    # Create DataFrame with exact feature order
                                                    hybrid_df = pd.DataFrame([hybrid_data])[original_features]
                                                    
                                                    # Make prediction with hybrid scenario
                                                    if problem_type in ["binary_classification", "multiclass_classification", "classification"] and hasattr(model, "predict_proba"):
                                                        if problem_type == "binary_classification":
                                                            hybrid_pred = model.predict_proba(hybrid_df)[0][1]
                                                        elif problem_type == "multiclass_classification":
                                                            # For multiclass, use the probability of the predicted class or the max probability
                                                            probs = model.predict_proba(hybrid_df)[0]
                                                            hybrid_pred = np.max(probs)  # Use max probability as a scalar value
                                                        else:  # fallback for legacy "classification"
                                                            hybrid_pred = model.predict_proba(hybrid_df)[0][1]
                                                    else:
                                                        hybrid_pred = model.predict(hybrid_df)[0]
                                                    
                                                    # Calculate impact metrics
                                                    total_diff = data2['prediction'] - data1['prediction']
                                                    feature_impact = hybrid_pred - data1['prediction']
                                                    
                                                    # Determine if the impact is in the same direction as the total difference
                                                    same_direction = (feature_impact * total_diff) > 0 if abs(total_diff) > 0 else True
                                                    
                                                    # Calculate a relative importance score (0-100)
                                                    importance_score = min(100, (abs(feature_impact) / abs(total_diff)) * 100) if abs(total_diff) > 0 else 0
                                                    
                                                    # Log the calculation results
                                                    st.session_state.logger.log_calculation("Isolated Feature Impact", {
                                                        "feature": selected_feature,
                                                        "baseline": float(data1['prediction']),
                                                        "hybrid_pred": float(hybrid_pred),
                                                        "target": float(data2['prediction']),
                                                        "feature_impact": float(feature_impact),
                                                        "total_diff": float(total_diff),
                                                        "importance_score": float(importance_score),
                                                        "same_direction": same_direction
                                                    })
                                                    
                                                    # Store all values in cache
                                                    current_analysis.update({
                                                        'feature': selected_feature,
                                                        'hybrid_pred': float(hybrid_pred),
                                                        'feature_impact': float(feature_impact),
                                                        'total_diff': float(total_diff),
                                                        'importance_score': float(importance_score),
                                                        'same_direction': same_direction
                                                    })
                                            
                                            # Display summary card first
                                            st.markdown(f"""
                                            ### Impact Summary for {selected_feature}
                                            
                                            You're seeing what happens when **only** {selected_feature} is changed from {scenario1} to {scenario2}.
                                            """)
                                            
                                            # Display metrics using cached values
                                            impact_cols = st.columns(3)
                                            with impact_cols[0]:
                                                val1 = data1['values'][selected_feature]
                                                val2 = data2['values'][selected_feature]
                                                val_diff = float(val2) - float(val1) if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) else "N/A"
                                                
                                                st.metric(
                                                    "Feature Change",
                                                    f"{val1} → {val2}",
                                                    delta=f"{val_diff:+.4f}" if isinstance(val_diff, (int, float)) else "N/A"
                                                )
                                            
                                            with impact_cols[1]:
                                                st.metric(
                                                    "Prediction Impact",
                                                    f"{current_analysis['hybrid_pred']:.4f}",
                                                    delta=f"{current_analysis['feature_impact']:+.4f}"
                                                )
                                                
                                            with impact_cols[2]:
                                                # Show absolute contribution instead of percentage
                                                direction_icon = "✓" if current_analysis['same_direction'] else "✗"
                                                impact_direction = "Supporting" if current_analysis['same_direction'] else "Opposing"
                                                impact_color = "normal" if current_analysis['same_direction'] else "inverse"
                                                
                                                st.metric(
                                                    f"Contribution ({impact_direction} {direction_icon})",
                                                    f"{abs(current_analysis['feature_impact']):.4f}",
                                                    delta=f"{current_analysis['impact_direction'] if 'impact_direction' in current_analysis else impact_direction}",
                                                    delta_color=impact_color,
                                                    help="""This value represents the absolute impact this feature has on the prediction:
                                                    
- Calculation: Feature Impact = Prediction with only this feature changed - Baseline prediction
- Higher absolute value means stronger impact on the prediction
- 'Supporting' (✓) means the feature pushes the prediction in the same direction as the overall change
- 'Opposing' (✗) means the feature works against the overall change direction

The feature impact shows how much this one feature alone changes the prediction.
                                                    
This is calculated by changing only this feature's value from the first scenario to match the second scenario, while keeping all other features the same."""
                                                )
                                            
                                            # Create enhanced comparison visualization
                                            st.markdown("### Visualization of Feature Impact")
                                            
                                            # Prepare data for waterfall chart
                                            fig = go.Figure()
                                            
                                            # Set base scenario, hybrid scenario, and target scenario
                                            baseline = data1['prediction']
                                            hybrid = current_analysis['hybrid_pred']
                                            target = data2['prediction']
                                            feature_impact = hybrid - baseline
                                            remainder = target - hybrid
                                            
                                            # Create Waterfall with arrows
                                            direction_color = 'rgba(50, 171, 96, 0.7)' if current_analysis['same_direction'] else 'rgba(219, 64, 82, 0.7)'
                                            
                                            # Calculate importance for display (keep for internal reference)
                                            total_change = abs(target - baseline)
                                            raw_feature_pct = (abs(feature_impact) / total_change * 100) if total_change > 0 else 0
                                            # Store if impact exceeds 100% for explanations
                                            exceeds_100_pct = raw_feature_pct > 100
                                            
                                            # Add annotations for clearer explanations
                                            annotations = [
                                                # Baseline annotation
                                                dict(
                                                    x=0,
                                                    y=baseline,
                                                    text=f"{scenario1}<br>{baseline:.4f}",
                                                    showarrow=False,
                                                    font=dict(color="black"),
                                                    align="center",
                                                    bgcolor="white",
                                                    bordercolor="black",
                                                    borderwidth=1,
                                                    borderpad=4,
                                                    opacity=0.8
                                                ),
                                                # Feature impact annotation
                                                dict(
                                                    x=1,
                                                    y=hybrid,
                                                    text=f"Impact of {selected_feature}<br>{feature_impact:+.4f}",
                                                    showarrow=True,
                                                    arrowhead=3,
                                                    arrowcolor=direction_color,
                                                    arrowwidth=2,
                                                    arrowsize=1,
                                                    font=dict(color="black"),
                                                    align="center",
                                                    bgcolor="white",
                                                    bordercolor=direction_color,
                                                    borderwidth=1,
                                                    borderpad=4,
                                                    opacity=0.8
                                                ),
                                                # Target annotation
                                                dict(
                                                    x=2,
                                                    y=target,
                                                    text=f"{scenario2}<br>{target:.4f}",
                                                    showarrow=False,
                                                    font=dict(color="black"),
                                                    align="center",
                                                    bgcolor="white",
                                                    bordercolor="black",
                                                    borderwidth=1,
                                                    borderpad=4,
                                                    opacity=0.8
                                                )
                                            ]
                                                
                                            # Use bar chart with clearer labels
                                            scenarios = [scenario1, f"Only {selected_feature}<br>changed", scenario2]
                                            values = [baseline, hybrid, target]
                                            colors = ['rgba(55, 128, 191, 0.7)', direction_color, 'rgba(219, 64, 82, 0.7)']
                                            
                                            fig.add_trace(go.Bar(
                                                x=scenarios,
                                                y=values,
                                                marker_color=colors,
                                                text=[f"{x:.4f}" for x in values],
                                                textposition='auto'
                                            ))
                                            
                                            # Add line to connect points for better visual flow
                                            fig.add_trace(go.Scatter(
                                                x=scenarios,
                                                y=values,
                                                mode='lines',
                                                line=dict(color='black', width=1, dash='dot'),
                                                showlegend=False
                                            ))
                                            
                                            # Enhanced layout
                                            fig.update_layout(
                                                title=f"Impact of changing only '{selected_feature}'",
                                                xaxis_title="Scenarios",
                                                yaxis_title="Prediction Value",
                                                height=500,
                                                annotations=annotations,
                                                hovermode="x"
                                            )
                                            
                                            st.plotly_chart(fig, config={'responsive': True})
                                            
                                            # Add explanation text below chart
                                            same_direction = current_analysis['same_direction']
                                            if same_direction:
                                                st.markdown(f"""
                                                #### How to read this chart:
                                                
                                                1. **Left bar ({scenario1})**: The baseline prediction ({baseline:.4f})
                                                2. **Middle bar**: What happens when only '{selected_feature}' changes ({hybrid:.4f})
                                                3. **Right bar ({scenario2})**: The final prediction when all features change ({target:.4f})
                                                
                                                The middle bar shows that changing only '{selected_feature}' causes an impact of {feature_impact:+.4f} on the prediction.
                                                """)
                                            else:
                                                # Different message for opposing features
                                                if exceeds_100_pct:
                                                    st.markdown(f"""
                                                    #### How to read this chart:
                                                    
                                                    1. **Left bar ({scenario1})**: The baseline prediction ({baseline:.4f})
                                                    2. **Middle bar**: What happens when only '{selected_feature}' changes ({hybrid:.4f})
                                                    3. **Right bar ({scenario2})**: The final prediction when all features change ({target:.4f})
                                                    
                                                    The middle bar shows that changing only '{selected_feature}' has a very strong effect ({feature_impact:+.4f}) but **in the opposite direction** of the overall change.
                                                    
                                                    **Note:** This feature's impact is very strong, and other features must be counteracting it to produce the final result.
                                                    """)
                                                else:
                                                    st.markdown(f"""
                                                    #### How to read this chart:
                                                    
                                                    1. **Left bar ({scenario1})**: The baseline prediction ({baseline:.4f})
                                                    2. **Middle bar**: What happens when only '{selected_feature}' changes ({hybrid:.4f})
                                                    3. **Right bar ({scenario2})**: The final prediction when all features change ({target:.4f})
                                                    
                                                    The middle bar shows that changing only '{selected_feature}' moves the prediction by {feature_impact:+.4f}, but **in the opposite direction** of the overall change.
                                                    """)
                                            
                                            # Add insights based on calculated metrics
                                            importance_score = current_analysis['importance_score']
                                            
                                            # Add insight box with more detailed explanation
                                            if importance_score > 80:
                                                if same_direction:
                                                    insight_text = f"""
                                                    **Key Insight:** '{selected_feature}' has a strong impact of {feature_impact:+.4f} on the prediction.
                                                    
                                                    This feature alone explains the majority of the difference between {scenario1} and {scenario2}.
                                                    """
                                                    st.success(insight_text)
                                                else:
                                                    if exceeds_100_pct:
                                                        insight_text = f"""
                                                        **Critical Finding:** '{selected_feature}' has a very strong impact ({feature_impact:+.4f}) but in the opposite direction!
                                                        
                                                        This feature is actually working against the overall prediction change. Its impact is so strong that other features 
                                                        must be counteracting it to produce the final result. If other features weren't changed,
                                                        the prediction would move in the opposite direction.
                                                        """
                                                    else:
                                                        insight_text = f"""
                                                        **Critical Finding:** '{selected_feature}' has a strong impact ({feature_impact:+.4f}) but in the opposite direction!
                                                        
                                                        This feature is actually working against the overall prediction change. If other features weren't changed,
                                                        the prediction would move in the opposite direction.
                                                        """
                                                    st.warning(insight_text)
                                            elif importance_score > 50:
                                                if same_direction:
                                                    st.info(f"""
                                                    **Insight:** '{selected_feature}' has a substantial impact of {feature_impact:+.4f} on the prediction.
                                                    
                                                    This is a major contributor to the change between {scenario1} and {scenario2}.
                                                    """)
                                                else:
                                                    st.warning(f"""
                                                    **Insight:** '{selected_feature}' has a substantial impact ({feature_impact:+.4f}) but pushes the prediction in the opposite direction.
                                                    
                                                    This feature is working against the overall change direction. If this feature were the only one changed, 
                                                    the prediction would move in the opposite direction from the actual result.
                                                    """)
                                            elif importance_score < 10:
                                                st.info(f"""
                                                **Insight:** '{selected_feature}' has minimal impact on the prediction (only {feature_impact:+.4f}).
                                                
                                                Other features are much more influential in determining the difference between {scenario1} and {scenario2}.
                                                """)
                                        
                                        except Exception as e:
                                            st.error(f"Error analyzing feature impact: {str(e)}")
                                            # Log the error
                                            st.session_state.logger.log_error("Feature Analysis Error", {
                                                "feature": selected_feature,
                                                "error": str(e),
                                                "scenario_comparison": f"{scenario1} vs {scenario2}"
                                            })
                                            import traceback
                                            st.error(traceback.format_exc())
                                    else:
                                        st.info("No numeric features available for visualization.")
                        else:
                            st.info("No differences found between the selected scenarios.")
                        
                        # Add export option with automatic format-based download
                        st.markdown("### 📤 Export Report")
                        
                        # Format selection dropdown with automatic generation
                        available_formats = ["Text (.txt)", "JSON (.json)"]
                        if PDF_AVAILABLE:
                            available_formats.insert(0, "PDF (.pdf)")
                        
                        selected_format = st.selectbox(
                            "Select report format:",
                            available_formats,
                            key=f"report_format_{scenario1}_{scenario2}"
                        )
                        
                        # Automatically generate report when format is selected
                        report_key = f"generated_report_{scenario1}_{scenario2}_{selected_format.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
                        
                        # Generate report if not already cached for this format
                        if report_key not in st.session_state:
                            try:
                                with st.spinner(f"Generating {selected_format.split(' ')[0]} report..."):
                                    # Log user action for report generation
                                    st.session_state.logger.log_user_action("Report Generation", {
                                        "type": "Scenario Comparison",
                                        "format": selected_format,
                                        "scenario1": scenario1,
                                        "scenario2": scenario2
                                    })
                                    
                                    # Collect all necessary data for report generation
                                    # Use cached SHAP values if available
                                    shap_cache_key = f"shap_values_{scenario1}_{scenario2}"
                                    shap_values1 = None
                                    shap_values2 = None
                                    shap_diff = None
                                    
                                    if shap_cache_key in st.session_state:
                                        shap_data = st.session_state[shap_cache_key]
                                        shap_values1 = shap_data.get('shap_values1')
                                        shap_values2 = shap_data.get('shap_values2')
                                        shap_diff = shap_data.get('shap_diff')
                                    
                                    # Collect feature impact data
                                    analysis_cache_key = f"feature_analysis_{scenario1}_{scenario2}"
                                    all_feature_impacts = {}
                                    
                                    if hasattr(st.session_state, 'isolated_feature_analysis'):
                                        # Collect all cached feature impacts for this comparison
                                        for key, value in st.session_state.isolated_feature_analysis.items():
                                            if key.startswith(analysis_cache_key):
                                                # Extract feature name from the key
                                                feature_name = key.replace(analysis_cache_key, "").lstrip("_")
                                                if feature_name and feature_name != "":
                                                    all_feature_impacts[feature_name] = value
                                            # Also check the base cache key
                                            elif key == analysis_cache_key and value.get('feature'):
                                                feature_name = value.get('feature')
                                                all_feature_impacts[feature_name] = value
                                    
                                    # Get model information
                                    model_type = st.session_state.builder.model.get("type", "unknown")
                                    
                                    # Get model feature importances if available
                                    model_feature_importances = None
                                    if hasattr(st.session_state.builder.model["model"], "feature_importances_"):
                                        try:
                                            model_feature_importances = {
                                                name: float(imp) 
                                                for name, imp in zip(
                                                    st.session_state.builder.X_train.columns, 
                                                    st.session_state.builder.model["model"].feature_importances_
                                                )
                                            }
                                        except Exception:
                                            pass
                                    
                                    # Generate report using the new utility
                                    filename, mime_type, file_data = create_scenario_comparison_report(
                                        format_type=selected_format,
                                        scenario1=scenario1,
                                        scenario2=scenario2,
                                        data1=data1,
                                        data2=data2,
                                        pred_diff=pred_diff,
                                        pred_diff_pct=pred_diff_pct,
                                        diff_features=diff_features,
                                        all_feature_impacts=all_feature_impacts,
                                        model_type=model_type,
                                        problem_type=problem_type,
                                        shap_values1=shap_values1,
                                        shap_values2=shap_values2,
                                        shap_diff=shap_diff,
                                        model_feature_importances=model_feature_importances
                                    )
                                    
                                    # Store the generated report in session state
                                    st.session_state[report_key] = {
                                        'filename': filename,
                                        'mime_type': mime_type,
                                        'file_data': file_data,
                                        'format': selected_format
                                    }
                                    
                                    # Log successful report generation
                                    st.session_state.logger.log_calculation("Report Generation", {
                                        "status": "success",
                                        "format": selected_format,
                                        "filename": filename,
                                        "comparison": f"{scenario1} vs {scenario2}"
                                    })
                                    
                            except ImportError as e:
                                if "reportlab" in str(e):
                                    st.error("❌ PDF generation requires the 'reportlab' package. Please install it with: pip install reportlab")
                                else:
                                    st.error(f"❌ Import error: {str(e)}")
                            except Exception as e:
                                st.error(f"❌ Error generating report: {str(e)}")
                                # Log error in report generation
                                st.session_state.logger.log_error("Report Generation Error", {
                                    "error": str(e),
                                    "format": selected_format,
                                    "scenario_comparison": f"{scenario1} vs {scenario2}"
                                })
                                import traceback
                                st.error(traceback.format_exc())
                        
                        # Show download button if report has been generated successfully
                        if report_key in st.session_state:
                            report_data = st.session_state[report_key]
                            
                            st.success(f"✅ {selected_format.split(' ')[0]} report ready for download!")
                            
                            st.download_button(
                                label=f"📄 Download {report_data['format'].split(' ')[0]} Report",
                                data=report_data['file_data'],
                                file_name=report_data['filename'],
                                mime=report_data['mime_type'],
                                key=f"final_download_{report_key}",
                                width='stretch'
                            )
                        
                        # Add help information about report formats
                        with st.expander("ℹ️ About Report Formats", expanded=False):
                            st.markdown("""
                            ### 📊 Report Format Options
                            
                            **PDF (.pdf)** - Professional formatted report
                            - ✅ Best for presentations and sharing
                            - ✅ Professional layout with tables and formatting
                            - ✅ Ready to print or include in documents
                            - ⚠️ Requires 'reportlab' package installation
                            
                            **Text (.txt)** - Human-readable text format
                            - ✅ Easy to read and edit
                            - ✅ Can be opened in any text editor
                            - ✅ Great for quick review and analysis
                            - ✅ No additional dependencies required
                            
                            **JSON (.json)** - Structured data format
                            - ✅ Machine-readable format
                            - ✅ Can be imported into other analysis tools
                            - ✅ Contains all raw data and calculations
                            - ✅ Good for further programmatic analysis
                            
                            ### 📋 Report Contents
                            
                            All formats include:
                            - **Model Information**: Type, problem type, metadata
                            - **Scenario Comparison**: Predictions, differences, percentages
                            - **Feature Analysis**: Value changes, isolated impacts, importance scores
                            - **SHAP Values**: When available from previous calculations
                            - **Model Feature Importances**: When available from the model
                            """)
    
    else:
        st.info("""
        ### 🔄 Getting Started
        
        1. **Make a prediction** using the form above to see results
        2. **Save scenarios** with different feature combinations to compare them
        3. **Explore** how individual features affect your model's predictions
        
        Your predictions and scenarios will appear here once you start!
        """)