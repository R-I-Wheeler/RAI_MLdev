import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import auc
import base64

@st.cache_data(show_spinner=False)
def create_metric_trend_plot(train_values, val_values, title="Learning Progress", x_values=None):
    """Create a line plot showing both training and validation trends."""
    fig = go.Figure()

    # Use provided x_values if available, otherwise calculate based on training set size
    if x_values is not None:
        # Use the actual x-values from learning curve data
        x_axis_values = x_values
    elif hasattr(st.session_state.builder, 'X_train'):
        total_samples = len(st.session_state.builder.X_train)
        # Create x values as fractions of the total training set
        x_axis_values = [int((i + 1) * total_samples / len(train_values)) for i in range(len(train_values))]
    else:
        x_axis_values = list(range(1, len(train_values) + 1))
    
    # Add training score line
    fig.add_trace(go.Scatter(
        x=x_axis_values,
        y=train_values,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))

    # Add validation score line
    fig.add_trace(go.Scatter(
        x=x_axis_values,
        y=val_values,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=6)
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Number of Training Examples',
        yaxis_title='Score',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Format x-axis to show actual numbers without scientific notation
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        tickformat=',d'  # Format as integer with thousands separator
    )
    
    # Add grid for better readability
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig

@st.cache_data(show_spinner=False)
def create_threshold_gauge(value, threshold, title):
    """Create a gauge chart for metric visualization."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            },
            'steps': [
                {'range': [0, threshold], 'color': "#ff9999"},
                {'range': [threshold, 1], 'color': "#99ff99"}
            ]
        },
        number = {'font': {'size': 28}}
    ))
    
    fig.update_layout(
        height=300,  # Increased height
        margin=dict(l=30, r=30, t=50, b=30),  # Adjusted margins
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

@st.cache_data(show_spinner=False)
def create_metric_comparison_radar(metrics_dict, thresholds_dict):
    """Create a radar chart comparing multiple metrics against their thresholds."""
    fig = go.Figure()
    
    # Add threshold polygon
    fig.add_trace(go.Scatterpolar(
        r=list(thresholds_dict.values()),
        theta=list(thresholds_dict.keys()),
        fill='toself',
        name='Threshold',
        fillcolor='rgba(144, 238, 144, 0.2)',
        line=dict(color='green', dash='dash')
    ))
    
    # Add actual metrics
    fig.add_trace(go.Scatterpolar(
        r=list(metrics_dict.values()),
        theta=list(metrics_dict.keys()),
        fill='toself',
        name='Actual',
        fillcolor='rgba(30, 144, 255, 0.2)',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Metrics Overview",
        height=600
    )
    
    return fig

def display_model_improvements(result, problem_type):
    st.write("### 🚀 Model Health and Recommendations")
    
    # Create sections for different types of improvements
    improvements = []
    metrics_summary = []
    
    # Add model fit analysis section
    fit_metrics = []
    
    # Extract learning curve data for fit analysis
    if "learning_curve" in result:
        try:
            lc_fig = result["learning_curve"]
            if hasattr(lc_fig, 'data'):
                train_data = next((trace for trace in lc_fig.data if 'train' in trace.name.lower()), None)
                val_data = next((trace for trace in lc_fig.data if 'val' in trace.name.lower() or 'test' in trace.name.lower() or 'cross' in trace.name.lower()), None)
                
                if train_data is not None and val_data is not None:
                    try:
                        # Extract y values - handle both direct arrays and dict-wrapped data
                        train_y_data = train_data.y
                        val_y_data = val_data.y

                        # Debug: Print what we're getting
                        print(f"DEBUG: train_data.y type: {type(train_y_data)}")
                        print(f"DEBUG: train_data.y value: {train_y_data}")
                        print(f"DEBUG: val_data.y type: {type(val_y_data)}")
                        print(f"DEBUG: val_data.y value: {val_y_data}")

                        # If data is wrapped in dict, extract the actual values
                        if isinstance(train_y_data, dict):
                            # Handle Plotly's serialized binary data format
                            if 'bdata' in train_y_data and 'dtype' in train_y_data:
                                # Decode base64 binary data
                                binary_data = base64.b64decode(train_y_data['bdata'])
                                # Reconstruct numpy array from binary data
                                train_y_data = np.frombuffer(binary_data, dtype=train_y_data['dtype'])
                            else:
                                # Try common dict keys for Plotly data
                                train_y_data = train_y_data.get('data', train_y_data.get('y', train_y_data.get('values', [])))
                        
                        if isinstance(val_y_data, dict):
                            # Handle Plotly's serialized binary data format
                            if 'bdata' in val_y_data and 'dtype' in val_y_data:
                                # Decode base64 binary data
                                binary_data = base64.b64decode(val_y_data['bdata'])
                                # Reconstruct numpy array from binary data
                                val_y_data = np.frombuffer(binary_data, dtype=val_y_data['dtype'])
                            else:
                                # Try common dict keys for Plotly data
                                val_y_data = val_y_data.get('data', val_y_data.get('y', val_y_data.get('values', [])))

                        # Convert to numpy arrays to ensure compatibility with np.isnan()
                        # Handle potential non-numeric data by coercing to float
                        train_scores = np.array(train_y_data, dtype=float)
                        val_scores = np.array(val_y_data, dtype=float)

                        # Validate that we have non-empty arrays with at least SOME valid numeric data
                        # More lenient: check if we have ANY valid (non-NaN) values, not requiring ALL to be valid
                        has_any_valid_train = len(train_scores) > 0 and np.isfinite(train_scores).any()
                        has_any_valid_val = len(val_scores) > 0 and np.isfinite(val_scores).any()

                        if not (has_any_valid_train and has_any_valid_val):
                            # Data is empty or has no valid values
                            fit_metrics.append({
                                'metric': 'Model Fit Status',
                                'value': 'Unable to determine',
                                'details': f'Learning curve contains no valid numeric data (train: {has_any_valid_train}, val: {has_any_valid_val})',
                                'status': '⚠️ Data Issue'
                            })
                        else:
                            # Continue with analysis - data is valid
                            # Calculate fit metrics with NaN checking
                            final_train_score = train_scores[-1]
                            final_val_score = val_scores[-1]

                            # Check if final scores are valid numbers
                            if (np.isnan(final_train_score) or np.isnan(final_val_score) or
                                np.isinf(final_train_score) or np.isinf(final_val_score)):
                                # Handle NaN or infinite scores by skipping fit analysis
                                fit_metrics.append({
                                    'metric': 'Model Fit Status',
                                    'value': 'Unable to determine',
                                    'details': 'Invalid scores detected in learning curve data (NaN or infinite values)',
                                    'status': '⚠️ Data Issue'
                                })
                            else:
                                score_gap = final_train_score - final_val_score

                                # Determine model fit status with individual score thresholds
                                # Handle both binary and multiclass classification
                                if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                                    # Classification thresholds - handle perfect accuracy edge cases
                                    if final_train_score >= 1.0 and final_val_score >= 1.0:
                                        fit_status = "Perfect Fit (Rare)"
                                        status_icon = "🌟"
                                    elif final_train_score < 0.7 or final_val_score < 0.7:  # Either score below threshold
                                        fit_status = "Underfitting"
                                        status_icon = "⚠️"
                                    elif score_gap > 0.2:  # Overfitting threshold
                                        fit_status = "Overfitting"
                                        status_icon = "⚠️"
                                    else:
                                        fit_status = "Good Fit"
                                        status_icon = "✅"
                                else:
                                    # Regression thresholds - handle perfect R² edge cases
                                    if final_train_score >= 1.0 and final_val_score >= 1.0:
                                        fit_status = "Perfect Fit (Rare)"
                                        status_icon = "🌟"
                                    elif final_train_score < 0.5 or final_val_score < 0.5:  # Either score below threshold
                                        fit_status = "Underfitting"
                                        status_icon = "⚠️"
                                    elif score_gap > 0.3:  # Overfitting threshold
                                        fit_status = "Overfitting"
                                        status_icon = "⚠️"
                                    else:
                                        fit_status = "Good Fit"
                                        status_icon = "✅"

                                fit_metrics.append({
                                    'metric': 'Model Fit Status',
                                    'value': fit_status,
                                    'details': f"Train Score: {final_train_score:.3f}, Validation Score: {final_val_score:.3f}",
                                    'status': f"{status_icon} {fit_status}",
                                    'threshold': '0.7' if problem_type in ["classification", "binary_classification", "multiclass_classification"] else '0.5'
                                })

                                # Add train-validation gap metric with threshold
                                fit_metrics.append({
                                    'metric': 'Train-Validation Gap',
                                    'value': f"{score_gap:.3f}",
                                    'details': f"Difference between training and validation scores",
                                    'status': '✅ Good' if (
                                        (problem_type in ["classification", "binary_classification", "multiclass_classification"] and score_gap <= 0.2) or
                                        (problem_type == "regression" and score_gap <= 0.3)
                                    ) else '⚠️ Needs Attention',
                                    'threshold': '0.2' if problem_type in ["classification", "binary_classification", "multiclass_classification"] else '0.3'
                                })

                            # Add learning curve trend analysis with robust NaN handling
                            if len(train_scores) > 1 and len(val_scores) > 1:
                                # Arrays are already numpy arrays from earlier conversion
                                train_array = train_scores
                                val_array = val_scores

                                # Find indices of valid (non-NaN, non-infinite) values
                                train_valid = ~(np.isnan(train_array) | np.isinf(train_array))
                                val_valid = ~(np.isnan(val_array) | np.isinf(val_array))
                                both_valid = train_valid & val_valid

                                if np.sum(both_valid) >= 2:
                                    # Get valid indices and corresponding values
                                    valid_indices = np.where(both_valid)[0]
                                    valid_train_scores = train_array[both_valid]
                                    valid_val_scores = val_array[both_valid]

                                    # Use first and last valid points for trend
                                    train_start, train_end = valid_train_scores[0], valid_train_scores[-1]
                                    val_start, val_end = valid_val_scores[0], valid_val_scores[-1]

                                    train_trend = train_end - train_start
                                    val_trend = val_end - val_start

                                    # Create more descriptive value with context
                                    trend_description = "Improving" if val_trend > 0 else "Stagnant/Declining"
                                    trend_magnitude = abs(val_trend)

                                    if trend_magnitude >= 0.05:
                                        trend_strength = "Strong"
                                    elif trend_magnitude >= 0.02:
                                        trend_strength = "Moderate"
                                    else:
                                        trend_strength = "Weak"

                                    display_value = f"{trend_strength} {trend_description} ({val_trend:+.3f})"

                                    # Add note if some data points were invalid
                                    data_quality_note = ""
                                    if np.sum(both_valid) < len(train_scores):
                                        invalid_count = len(train_scores) - np.sum(both_valid)
                                        data_quality_note = f" (Note: {invalid_count} invalid points excluded)"

                                    fit_metrics.append({
                                        'metric': 'Learning Progress',
                                        'value': display_value,
                                        'details': f"Validation score changed by {val_trend:+.3f} from start to end of training{data_quality_note}",
                                        'status': '✅ Good' if val_trend > 0 else '⚠️ Needs Attention'
                                    })
                                else:
                                    # Count how many points were invalid to provide better feedback
                                    total_points = len(train_scores)
                                    valid_points = np.sum(both_valid)

                                    fit_metrics.append({
                                        'metric': 'Learning Progress',
                                        'value': 'Unable to determine',
                                        'details': f'Insufficient valid data points ({valid_points}/{total_points} usable)',
                                        'status': '⚠️ Data Issue'
                                    })
                            else:
                                fit_metrics.append({
                                    'metric': 'Learning Progress',
                                    'value': 'Unable to determine',
                                    'details': 'Insufficient data points for trend analysis',
                                    'status': '⚠️ Data Issue'
                                })
                    except (ValueError, TypeError) as e:
                        # Handle any data type or value errors with detailed information
                        error_details = f"Could not analyze learning curve data: {str(e)}"

                        # Add debug information if data structure is unexpected
                        try:
                            train_type = type(train_data.y).__name__ if train_data else "None"
                            val_type = type(val_data.y).__name__ if val_data else "None"
                            error_details += f" (Data types: train={train_type}, val={val_type})"
                        except:
                            pass

                        st.warning(error_details)

                        # Add a metric indicating the issue
                        fit_metrics.append({
                            'metric': 'Learning Curve Analysis',
                            'value': 'Failed',
                            'details': f'Data format error: {str(e)}',
                            'status': '⚠️ Data Issue'
                        })
        except Exception as e:
            st.warning(f"Could not analyze learning curve data: {str(e)}")
    
    # Add fit metrics to overall metrics summary
    metrics_summary.extend(fit_metrics)
    
    # Continue with existing metrics...
    # Handle both binary and multiclass classification
    if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
        # Classification metrics
        if "metrics" in result:
            metrics = result["metrics"]
            
            # Basic metrics
            for metric in ["accuracy", "precision", "recall", "f1"]:
                value = metrics.get(metric, 0)
                metrics_summary.append({
                    'metric': metric.title(),
                    'value': f"{value:.3f}",
                    'threshold': '0.7',
                    'status': '✅ Good' if value >= 0.7 else '⚠️ Needs Attention'
                })
            
            # Calculate ROC AUC from ROC curve if available
            if "roc_curve" in result and hasattr(result["roc_curve"], "data"):
                try:
                    roc_data = result["roc_curve"].data[0]
                    if hasattr(roc_data, "x") and hasattr(roc_data, "y"):
                        fpr = np.array(roc_data.x)
                        tpr = np.array(roc_data.y)

                        # Validate data before calculation
                        if len(fpr) > 0 and len(tpr) > 0 and not np.isnan(fpr).all() and not np.isnan(tpr).all():
                            roc_auc = auc(fpr, tpr)

                            # Handle edge cases
                            if np.isnan(roc_auc) or np.isinf(roc_auc):
                                roc_auc_value = "Invalid"
                                roc_auc_status = "⚠️ Data Issue"
                            elif roc_auc >= 1.0:
                                roc_auc_value = "1.000 (Perfect)"
                                roc_auc_status = "✅ Perfect"
                            else:
                                roc_auc_value = f"{roc_auc:.3f}"
                                roc_auc_status = '✅ Good' if roc_auc >= 0.8 else '⚠️ Needs Attention'

                            metrics_summary.append({
                                'metric': 'ROC AUC',
                                'value': roc_auc_value,
                                'threshold': '0.8',
                                'status': roc_auc_status
                            })
                except Exception:
                    pass
        
        # Class balance check - use actual training data distribution
        if hasattr(st.session_state.builder, 'y_train') and st.session_state.builder.y_train is not None:
            # Get current training data class distribution
            current_class_counts = pd.Series(st.session_state.builder.y_train).value_counts().to_dict()

            if current_class_counts and len(current_class_counts) > 1:
                max_support = max(current_class_counts.values())
                min_support = min(current_class_counts.values())

                # Avoid division by zero
                if min_support > 0:
                    imbalance_ratio = max_support / min_support

                    # Check if resampling was applied
                    resampling_applied = getattr(st.session_state, 'imbalance_handled', False)
                    original_data_kept = getattr(st.session_state, 'imbalance_skipped', False)

                    # Handle edge cases and create display value
                    if imbalance_ratio == 1.0:
                        ratio_display = "1.0:1 (Perfect Balance)"
                        if resampling_applied:
                            ratio_display += " ✨ Resampled"
                        status = "✅ Perfect"
                    elif imbalance_ratio > 1000:
                        ratio_display = f"{imbalance_ratio:.0f}:1 (Extreme Imbalance)"
                        if original_data_kept:
                            ratio_display += " ⚠️ Original Data"
                        status = "🔴 Critical"
                    else:
                        ratio_display = f"{imbalance_ratio:.1f}:1"
                        if resampling_applied and imbalance_ratio <= 3:
                            ratio_display += " ✨ Resampled"
                        elif original_data_kept and imbalance_ratio > 3:
                            ratio_display += " ⚠️ Original Data"
                        status = '✅ Good' if imbalance_ratio <= 3 else '⚠️ Needs Attention'

                    metrics_summary.append({
                        'metric': 'Class Balance Ratio',
                        'value': ratio_display,
                        'threshold': '3:1',
                        'status': status
                    })
        
        # Log journey with available data
        details = {"Model Type": st.session_state.builder.model['type']}

        # Add optional metrics if they exist
        fit_status_metric = next((m for m in fit_metrics if m['metric'] == 'Model Fit Status'), None)
        if fit_status_metric:
            details["Model fit status"] = fit_status_metric['value']

        gap_metric = next((m for m in fit_metrics if m['metric'] == 'Train-Validation Gap'), None)
        if gap_metric:
            details["Train Validation Gap"] = gap_metric['value']

        balance_metric = next((m for m in metrics_summary if m['metric'] == 'Class Balance Ratio'), None)
        if balance_metric:
            details["Class Balance Ratio"] = balance_metric['value']

        roc_metric = next((m for m in metrics_summary if m['metric'] == 'ROC AUC'), None)
        if roc_metric:
            details["ROC AUC"] = roc_metric['value']

        st.session_state.logger.log_journey_point(
            stage="MODEL_EVALUATION",
            decision_type="MODEL_EVALUATION",
            description="Model Health Metrics",
            details=details,
            parent_id=None
        )
    else:  # regression
        # Regression metrics
        if "metrics" in result:
            metrics = result["metrics"]
            
            # R² Score
            r2 = metrics.get("r2", 0)
            metrics_summary.append({
                'metric': 'R² Score',
                'value': f"{r2:.3f}",
                'threshold': '0.5',
                'status': '✅ Good' if r2 >= 0.5 else '⚠️ Needs Attention'
            })
            
            # MAE
            if "mae" in metrics:
                mae = metrics["mae"]
                if hasattr(st.session_state.builder, 'y_test'):
                    mean_target = np.mean(np.abs(st.session_state.builder.y_test))

                    # Validate mean_target to avoid division issues
                    if mean_target > 0 and not np.isnan(mean_target) and not np.isinf(mean_target):
                        mae_threshold = mean_target * 0.2

                        # Handle edge cases for MAE
                        if np.isnan(mae) or np.isinf(mae):
                            mae_value = "Invalid"
                            mae_status = "⚠️ Data Issue"
                        elif mae == 0.0:
                            mae_value = "0.000 (Perfect)"
                            mae_status = "🌟 Perfect"
                        else:
                            mae_value = f"{mae:.3f}"
                            mae_status = '✅ Good' if mae <= mae_threshold else '⚠️ Needs Attention'

                        metrics_summary.append({
                            'metric': 'Mean Absolute Error',
                            'value': mae_value,
                            'threshold': f"{mae_threshold:.3f}",
                            'status': mae_status
                        })
            
            # RMSE
            if "rmse" in metrics:
                rmse = metrics["rmse"]
                if hasattr(st.session_state.builder, 'y_test'):
                    std_target = np.std(st.session_state.builder.y_test)

                    # Validate std_target to avoid division issues
                    if std_target > 0 and not np.isnan(std_target) and not np.isinf(std_target):
                        rmse_threshold = std_target * 0.5

                        # Handle edge cases for RMSE
                        if np.isnan(rmse) or np.isinf(rmse):
                            rmse_value = "Invalid"
                            rmse_status = "⚠️ Data Issue"
                        elif rmse == 0.0:
                            rmse_value = "0.000 (Perfect)"
                            rmse_status = "🌟 Perfect"
                        else:
                            rmse_value = f"{rmse:.3f}"
                            rmse_status = '✅ Good' if rmse <= rmse_threshold else '⚠️ Needs Attention'

                        metrics_summary.append({
                            'metric': 'Root Mean Square Error',
                            'value': rmse_value,
                            'threshold': f"{rmse_threshold:.3f}",
                            'status': rmse_status
                        })
            # Log journey with available regression data
            details = {"Model Type": st.session_state.builder.model['type']}

            # Add optional metrics if they exist
            fit_status_metric = next((m for m in fit_metrics if m['metric'] == 'Model Fit Status'), None)
            if fit_status_metric:
                details["Model fit status"] = fit_status_metric['value']

            gap_metric = next((m for m in fit_metrics if m['metric'] == 'Train-Validation Gap'), None)
            if gap_metric:
                details["Train Validation Gap"] = gap_metric['value']

            st.session_state.logger.log_journey_point(
                        stage="MODEL_EVALUATION",
                        decision_type="MODEL_EVALUATION",
                        description="Model Health Metrics",
                        details=details,
                        parent_id=None
                    )
    
    # Display metrics summary
    if metrics_summary:
        st.write("#### Model Health Metrics")
        metrics_df = pd.DataFrame(metrics_summary)
        
        def style_status(row):
            return ['background-color: #c6efce' if '✅' in row['status'] else 'background-color: #ffc7ce' for _ in row]
        
        styled_df = metrics_df.style.apply(style_status, axis=1)
        st.dataframe(styled_df, width='stretch')
        
        with st.expander("ℹ️ Understanding the Metrics"):
            # Handle both binary and multiclass classification
            if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                st.markdown("""
                    ## 📊 Understanding Your Model's Health
                    
                    ### How Well Is Your Model Learning? 🎯
                    
                    **Model Fit Status**: Think of this like teaching a student:
                    - **Underfitting**: The model is learning too little (like memorizing only the chapter titles)
                      - *What it looks like*: Either training score < 0.7 OR validation score < 0.7
                      - *What to do*: Help the model learn more complex patterns
                    
                    - **Overfitting**: The model is memorizing instead of learning (like memorizing exact test answers)
                      - *What it looks like*: Big difference between training and test scores (> 0.2)
                      - *What to do*: Help the model focus on general patterns
                    
                    - **Good Fit**: The model has learned the right amount (like understanding the subject)
                      - *What it looks like*: Good training score and similar test performance

                    - **Perfect Fit**: Extremely rare scenario where model achieves 100% accuracy
                      - *What it looks like*: Both training and validation scores = 1.0
                      - *What to check*: Verify data quality and potential data leakage
                    
                    ### Learning Progress 📚

                    **Train-Validation Gap**
                    - *What it shows*: Difference between training and validation performance
                    - *Good range*: Less than 0.2 (or 20%)
                    - *Why it matters*: Shows if your model will work well on new data

                    **Learning Progress**
                    - *What it shows*: How much your model's validation performance changed from start to finish of training
                    - *How to read it*:
                      - Positive numbers (e.g., +0.023): Model improved during training ✅
                      - Negative numbers (e.g., -0.015): Model got worse during training ⚠️
                      - The magnitude shows how much: Strong (≥0.05), Moderate (0.02-0.05), Weak (<0.02)
                    - *Example*: "Strong Improving (+0.067)" means validation score increased by 0.067 points
                    - *Good sign*: Any positive trend, especially strong improvement
                    - *Bad sign*: Negative trend or very weak improvement
                            
                    ### Basic Performance Metrics 📈
                    
                    **Accuracy** (Target: ≥ 0.7 or 70%)
                    - *What it is*: Percentage of correct predictions
                    - *Example*: If accuracy is 0.75, the model correctly predicts 75 out of 100 cases
                    - ⚠️ *Watch out*: Can be misleading with unbalanced classes
                    
                    **Precision** (Target: ≥ 0.7 or 70%)
                    - *What it is*: How often the model is right when it makes a positive prediction
                    - *Example*: If precision is 0.8, when the model predicts "yes", it's right 80% of the time
                    - *Use when*: False positives are costly (like spam detection)
                    
                    **Recall** (Target: ≥ 0.7 or 70%)
                    - *What it is*: How many actual positives the model finds
                    - *Example*: If recall is 0.9, the model finds 90% of all actual positive cases
                    - *Use when*: Missing positives is costly (like disease detection)
                    
                    **F1 Score** (Target: ≥ 0.7 or 70%)
                    - *What it is*: Balance between precision and recall
                    - *Think of it as*: The model's overall effectiveness
                    - *Use when*: You need a single number to compare models
                    
                    ### Advanced Indicators 🔍
                    
                    **Class Balance Ratio** (Target: ≤ 3:1)
                    - *What it is*: Ratio between your most common and least common classes
                    - *Example*: 2:1 means your largest class has twice as many samples as your smallest
                    - *Why it matters*: Large imbalances can make your model biased
                    
                    **ROC AUC** (Target: ≥ 0.8 or 80%)
                    - *What it is*: How well your model distinguishes between classes
                    - *Scale*: 0.5 = random guessing, 1.0 = perfect separation
                    - *Use when*: You need a robust measure of model quality
                    
                    ---
                    
                    ### 🎯 How the Thresholds Were Calculated
                    
                    The performance thresholds used in this analysis are based on machine learning best practices and industry standards:
                    
                    #### **Model Fit Thresholds**
                    - **Underfitting threshold (< 0.7)**: Based on research showing that classification models with accuracy below 70% provide limited practical value for most business applications
                    - **Overfitting threshold (gap > 0.2)**: Industry standard indicating that a 20% performance drop from training to validation suggests the model won't generalize well
                    
                    #### **Basic Performance Metrics (≥ 0.7)**
                    - **Accuracy, Precision, Recall, F1**: The 70% threshold represents the minimum acceptable performance for most business applications
                    - **Rationale**: Below 70%, the model's predictions become unreliable for decision-making
                    - **Source**: Widely adopted in ML literature as the boundary between "fair" and "good" performance
                    
                    #### **Advanced Metrics**
                    - **ROC AUC (≥ 0.8)**: 
                      - 0.5 = Random guessing
                      - 0.7-0.8 = Acceptable discrimination
                      - 0.8+ = Good discrimination ability
                      - Based on medical diagnostic standards where good separation is critical
                    
                    - **Class Balance Ratio (≤ 3:1)**:
                      - Calculated as: (Largest class count) / (Smallest class count)
                      - 3:1 threshold: Research shows ratios above 3:1 can lead to biased predictions
                      - Beyond this ratio, minority classes become underrepresented and hard to predict
                    
                    #### **Why These Specific Numbers?**
                    
                    1. **0.7 (70%) for underfitting**:
                       - Below this, models perform worse than many business heuristics
                       - Represents minimum threshold for reliable business predictions
                    
                    2. **0.7 (70%) for basic metrics**:
                       - Industry benchmark for "acceptable" business performance
                       - Balances practical utility with achievable goals
                    
                    3. **0.2 (20%) gap for overfitting**:
                       - Based on empirical studies of model generalization
                       - Indicates model has memorized training-specific patterns
                    
                    4. **0.8 (80%) for ROC AUC**:
                       - Medical/diagnostic industry standard for "good" tests
                       - Ensures model can reliably distinguish between classes
                    
                    5. **3:1 for class balance**:
                       - Statistical threshold where minority class representation becomes problematic
                       - Based on sampling theory and ML fairness research
                    
                    ### 🔑 Key Things to Remember
                    
                    1. **Balance is Key**
                       - Good performance on training data
                       - Similar performance on validation data
                       - Consistent performance across all classes
                    
                    2. **Watch Out For**
                       - Very high training scores but low validation scores (overfitting)
                       - Low scores on both training and validation (underfitting)
                       - Big differences in performance between classes
                       - "Data Issue" warnings which indicate corrupted learning curve data
                    
                    3. **When You Have Unbalanced Classes**
                       - Look at per-class performance
                       - Focus on F1 score and ROC AUC
                       - Consider collecting more data for smaller classes
                """)
            else:
                st.markdown("""
                    ## 📊 Understanding Your Model's Health
                    
                    ### How Well Is Your Model Learning? 🎯
                    
                    **Model Fit Status**: Think of this like teaching a student:
                    - **Underfitting**: The model is learning too little (like only learning basic concepts)
                      - *What it looks like*: Either training R² < 0.5 OR validation R² < 0.5
                      - *What to do*: Help the model learn more complex patterns
                    
                    - **Overfitting**: The model is memorizing instead of learning (like memorizing specific examples)
                      - *What it looks like*: Big difference between training and test scores (> 0.3)
                      - *What to do*: Help the model focus on general patterns
                    
                    - **Good Fit**: The model has learned the right amount (like understanding the subject)
                      - *What it looks like*: Good R² score and similar test performance

                    - **Perfect Fit**: Extremely rare scenario where model achieves R² = 1.0
                      - *What it looks like*: Both training and validation R² = 1.0
                      - *What to check*: Verify data quality and potential data leakage
                    
                    ### Learning Progress 📚
                    
                    **Train-Validation Gap**
                    - *What it shows*: Difference between training and validation performance
                    - *Good range*: Less than 0.3 (or 30%)
                    - *Why it matters*: Shows if your model will work well on new data
                    
                    **Learning Progress**
                    - *What it shows*: How much your model's validation performance changed from start to finish of training
                    - *How to read it*:
                      - Positive numbers (e.g., +0.023): Model improved during training ✅
                      - Negative numbers (e.g., -0.015): Model got worse during training ⚠️
                      - The magnitude shows how much: Strong (≥0.05), Moderate (0.02-0.05), Weak (<0.02)
                    - *Example*: "Strong Improving (+0.067)" means validation R² increased by 0.067 points
                    - *Good sign*: Any positive trend, especially strong improvement
                    - *Bad sign*: Negative trend or very weak improvement
                            
                    ### Basic Performance Metrics 📈
                    
                    **R² Score** (Target: ≥ 0.5 or 50%)
                    - *What it is*: How much of the data variation your model explains
                    - *Scale*: 0 = poor fit, 1 = perfect fit
                    - *Example*: R² of 0.75 means the model explains 75% of the variation in your data
                    
                    **Mean Absolute Error (MAE)**
                    - *What it is*: Average size of prediction errors
                    - *Example*: If MAE = 5, predictions are off by 5 units on average
                    - *Good when*: All errors are equally important
                    
                    **Root Mean Square Error (RMSE)**
                    - *What it is*: Like MAE, but penalizes large errors more
                    - *Example*: If RMSE = 7, most errors are less than 7 units
                    - *Good when*: Large errors are especially bad
                    
                    ---
                    
                    ### 🎯 How the Thresholds Were Calculated
                    
                    The performance thresholds used in this analysis are tailored to your specific dataset and based on statistical principles:
                    
                    #### **Model Fit Thresholds**
                    - **Underfitting threshold (< 0.5)**: R² below 50% means the model explains less than half the data variation, indicating poor learning
                    - **Overfitting threshold (gap > 0.3)**: A 30% performance drop from training to validation suggests the model won't generalize well to new data
                    
                    #### **R² Score Threshold (≥ 0.5)**
                    - **Calculation**: Fixed threshold based on statistical significance
                    - **Meaning**: Model explains at least 50% of target variance
                    - **Rationale**: Below 0.5, predictions are not substantially better than using the mean value
                    - **Industry standard**: Widely accepted minimum for practical regression models
                    
                    #### **Error Thresholds (Data-Dependent)**
                    
                    **Mean Absolute Error (MAE)**:
                    - **Calculation**: `MAE threshold = Mean(|target values|) × 0.2`
                    - **Meaning**: Errors should be less than 20% of the typical target magnitude
                    - **Example**: If your target values average 100 units, MAE should be < 20 units
                    - **Why 20%**: Represents a reasonable prediction precision for most business applications
                    
                    **Root Mean Square Error (RMSE)**:
                    - **Calculation**: `RMSE threshold = Standard deviation of target × 0.5`
                    - **Meaning**: Errors should be less than half the natural variability in your data
                    - **Example**: If your target has std deviation of 50, RMSE should be < 25
                    - **Why 50%**: Ensures predictions are meaningfully better than just using variability estimates
                    
                    #### **Why These Specific Calculations?**
                    
                    1. **R² threshold (0.5)**:
                       - Statistical threshold where model performance becomes meaningful
                       - Below this, simple baselines often perform as well
                    
                    2. **MAE threshold (20% of mean)**:
                       - Percentage-based: adapts to your data's scale
                       - 20% represents "acceptable business error" across industries
                       - Ensures errors are proportional to typical values
                    
                    3. **RMSE threshold (50% of std deviation)**:
                       - Variability-based: compares errors to natural data spread
                       - 50% means predictions reduce uncertainty by at least half
                       - Accounts for the fact that some variation is inherently unpredictable
                    
                    4. **Train-validation gap (30%)**:
                       - Higher tolerance than classification due to regression's continuous nature
                       - Accounts for inherent noise in real-valued predictions
                       - Based on empirical studies of regression model generalization
                    
                    #### **Adaptive Nature of Thresholds**
                    
                    Unlike fixed thresholds, the error thresholds adapt to YOUR specific data:
                    
                    - **High-value targets** (e.g., house prices in millions): Higher absolute error tolerance
                    - **Low-value targets** (e.g., temperature): Lower absolute error tolerance
                    - **High-variability data**: More forgiving RMSE threshold
                    - **Low-variability data**: Stricter RMSE threshold
                    
                    This ensures the thresholds are always meaningful for your specific problem domain.
                    
                    ### 🔑 Key Things to Remember
                    
                    1. **Understanding Your Errors**
                       - MAE: Simple average error (easier to understand)
                       - RMSE: Weighted towards larger errors (more sensitive)
                       - Both are in your target variable's units (like dollars, temperature, etc.)
                    
                    2. **What Makes a Good Model**
                       - R² score above 0.5
                       - MAE less than 20% of your average target value
                       - RMSE less than 50% of your target's standard deviation
                       - Small difference between training and validation scores
                    
                    3. **Watch Out For**
                       - Very high training R² but low validation R² (overfitting)
                       - Low R² on both training and validation (underfitting)
                       - Unusually large errors in specific ranges
                       - "Data Issue" warnings which indicate corrupted learning curve data
                    
                    ### 💡 Tips for Improvement
                    
                    - If errors are too large: Consider feature engineering or more data
                    - If overfitting: Simplify model or add more training data
                    - If underfitting: Try more complex models or better features
                """)

        # Add visualizations section after the metrics table
        st.write("#### 📊 Performance Visualizations")
        
        # Score trends and gauge charts
        if "learning_curve" in result and hasattr(result["learning_curve"], 'data'):
            train_data = next((trace for trace in result["learning_curve"].data if 'train' in trace.name.lower()), None)
            val_data = next((trace for trace in result["learning_curve"].data if 'val' in trace.name.lower() or 'test' in trace.name.lower() or 'cross' in trace.name.lower()), None)
            
            if train_data is not None and val_data is not None:
                st.write("##### Learning Progress")
                
                # Learning Progress Visualization and Explanation
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(
                        create_metric_trend_plot(train_data.y, val_data.y, x_values=train_data.x),
                        use_container_width=True
                    )
                with col2:
                    st.markdown("""
                            This plot shows how your model's performance changes during training:
                            
                            **📊 Plot Elements:**
                            - 🔵 Blue line: Training Score
                            - 🟢 Green line: Validation Score
                            - X-axis: Training Progress (number of training examples)
                            - Y-axis: Model Performance Score (for classification: accuracy, for regression: R²)
                            """)
                    st.markdown("""---""")
                    with st.expander("ℹ️ Understanding the Learning Curve", expanded=False):
                        st.markdown("""
                            **🔍 Key Patterns to Look For:**
                            
                            1. **Ideal Learning Pattern:**
                               ```
                               Both lines rise and stabilize
                               Small gap between lines
                               ```
                               ✅ Model is learning well and generalizing
                            
                            2. **Overfitting Pattern:**
                               ```
                               Blue line keeps rising
                               Green line peaks and drops
                               Gap keeps growing
                               ```
                               ⚠️ Model is memorizing training data
                            
                            3. **Regularization Pattern:**
                               ```
                               Blue line slightly decreases
                               Green line increases
                               Gap narrows
                               ```
                               ✅ Model is generalizing better
                            
                            4. **Underfitting Pattern:**
                               ```
                               Both lines stay low
                               Small gap between lines
                               ```
                               ⚠️ Model needs more capacity
                            
                            5. **Unstable Learning:**
                               ```
                               Lines show erratic changes
                               No clear trend
                               ```
                               ⚠️ Learning rate may be too high
                            
                        """)
                
                # Add gauge charts only if we have valid fit metrics
                gap_metric = next((m for m in fit_metrics if m['metric'] == 'Train-Validation Gap'), None)
                if gap_metric and gap_metric['value'] != 'Unable to determine':
                    try:
                        score_gap = float(gap_metric['value'])
                        
                        # Extract and process learning curve data (same logic as earlier)
                        train_y_data = train_data.y
                        val_y_data = val_data.y
                        
                        # Handle Plotly's serialized binary data format (if cached)
                        if isinstance(train_y_data, dict):
                            if 'bdata' in train_y_data and 'dtype' in train_y_data:
                                binary_data = base64.b64decode(train_y_data['bdata'])
                                train_y_data = np.frombuffer(binary_data, dtype=train_y_data['dtype'])
                            else:
                                train_y_data = train_y_data.get('data', train_y_data.get('y', train_y_data.get('values', [])))
                        
                        if isinstance(val_y_data, dict):
                            if 'bdata' in val_y_data and 'dtype' in val_y_data:
                                binary_data = base64.b64decode(val_y_data['bdata'])
                                val_y_data = np.frombuffer(binary_data, dtype=val_y_data['dtype'])
                            else:
                                val_y_data = val_y_data.get('data', val_y_data.get('y', val_y_data.get('values', [])))
                        
                        # Convert to numpy arrays
                        train_scores_gauge = np.array(train_y_data, dtype=float)
                        val_scores_gauge = np.array(val_y_data, dtype=float)
                        
                        # Validate that we have non-empty arrays with valid data
                        if (len(train_scores_gauge) > 0 and len(val_scores_gauge) > 0 and
                            np.isfinite(train_scores_gauge[-1]) and np.isfinite(val_scores_gauge[-1])):

                            st.write("##### Key Learning Metrics Overview")

                            # Create columns for all gauges
                            gauge_cols = st.columns(3)

                            # Metrics data - using extracted scores
                            metrics_data = [
                                ("Training Score", train_scores_gauge[-1], 0.7 if problem_type in ["classification", "binary_classification", "multiclass_classification"] else 0.5),
                                ("Validation Score", val_scores_gauge[-1], 0.7 if problem_type in ["classification", "binary_classification", "multiclass_classification"] else 0.5),
                                ("Score Gap (Inverse)", 1.0 - score_gap, 0.8)
                            ]

                            # Display gauges side by side
                            for idx, (metric_name, value, threshold) in enumerate(metrics_data):
                                with gauge_cols[idx]:
                                    st.plotly_chart(
                                        create_threshold_gauge(value, threshold, metric_name),
                                        use_container_width=True,
                                        key=f"gauge_{metric_name.lower().replace(' ', '_')}"
                                    )
                        else:
                            st.info("Unable to display gauge charts: learning curve data is empty or invalid.")
                    except (ValueError, TypeError, KeyError, IndexError) as e:
                        st.info(f"Unable to display gauge charts due to invalid learning curve data: {type(e).__name__}")
                
                # Add explanations below the gauges
                st.markdown("##### Understanding the Metrics")
                explanation_cols = st.columns(3)
                
                with explanation_cols[0]:
                    st.markdown("**Training Score**")
                    st.markdown(f"""
                        - Shows final training performance
                        - Target: {'≥ 0.7 (70%)' if problem_type in ["classification", "binary_classification", "multiclass_classification"] else '≥ 0.5 (50%)'}
                        - 🟢 Green zone: Good performance
                        - 🔴 Red zone: Needs improvement
                    """)
                
                with explanation_cols[1]:
                    st.markdown("**Validation Score**")
                    st.markdown(f"""
                        - Shows performance on new data
                        - Target: {'≥ 0.7 (70%)' if problem_type in ["classification", "binary_classification", "multiclass_classification"] else '≥ 0.5 (50%)'}
                        - Should be close to training score
                        - Most important for real-world use
                    """)
                
                with explanation_cols[2]:
                    st.markdown("**Score Gap**")
                    st.markdown("""
                        - Shows training/validation similarity
                        - Target: ≥ 0.8 (gap ≤ 0.2)
                        - 🟢 High value: Good generalization
                        - 🔴 Low value: Possible overfitting
                    """)
                
                with st.expander("📖 Detailed Interpretation Guide"):
                    st.markdown("""
                    **How to Use These Metrics:**
                    
                    1. **Training Score**
                       - Indicates how well your model has learned from the training data
                       - High score needed but watch for overfitting
                       - If too low, model may need more capacity or training
                    
                    2. **Validation Score**
                       - Most important indicator of real-world performance
                       - Should be close to but slightly lower than training score
                       - If much lower than training, model is overfitting
                    
                    3. **Score Gap**
                       - Measures the difference between training and validation scores
                       - Small gap (high value) indicates good generalization
                       - Large gap (low value) suggests overfitting and need for regularization
                    
                    **Common Patterns:**
                    
                    ✅ **Healthy Model:**
                    - All gauges in green zones
                    - Small gap between training and validation
                    - Consistent performance across metrics
                    
                    ⚠️ **Potential Issues:**
                    - Training score in red: Underfitting
                    - Large score gap: Overfitting
                    - Both scores low: Fundamental model/data issues
                    
                    **Next Steps:**
                    - Red training score → Increase model complexity
                    - Large gap → Add regularization
                    - Both low → Review feature engineering and data quality
                    """)

                # Add radar chart for metric overview
                st.write("##### Overall Performance Overview")
                
                # Handle both binary and multiclass classification
                if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                    metrics_dict = {
                        'Accuracy': result['metrics'].get('accuracy', 0),
                        'Precision': result['metrics'].get('precision', 0),
                        'Recall': result['metrics'].get('recall', 0),
                        'F1': result['metrics'].get('f1', 0)
                    }
                    thresholds_dict = {
                        'Accuracy': 0.7,
                        'Precision': 0.7,
                        'Recall': 0.7,
                        'F1': 0.7
                    }
                    
                    # Add ROC AUC if available
                    if "roc_curve" in result and hasattr(result["roc_curve"], "data"):
                        try:
                            roc_data = result["roc_curve"].data[0]
                            if hasattr(roc_data, "x") and hasattr(roc_data, "y"):
                                # Extract and process ROC curve data (handle serialization)
                                fpr_data = roc_data.x
                                tpr_data = roc_data.y
                                
                                # Handle Plotly's serialized binary data format (if cached)
                                if isinstance(fpr_data, dict):
                                    if 'bdata' in fpr_data and 'dtype' in fpr_data:
                                        binary_data = base64.b64decode(fpr_data['bdata'])
                                        fpr_data = np.frombuffer(binary_data, dtype=fpr_data['dtype'])
                                    else:
                                        fpr_data = fpr_data.get('data', fpr_data.get('x', fpr_data.get('values', [])))
                                
                                if isinstance(tpr_data, dict):
                                    if 'bdata' in tpr_data and 'dtype' in tpr_data:
                                        binary_data = base64.b64decode(tpr_data['bdata'])
                                        tpr_data = np.frombuffer(binary_data, dtype=tpr_data['dtype'])
                                    else:
                                        tpr_data = tpr_data.get('data', tpr_data.get('y', tpr_data.get('values', [])))
                                
                                fpr = np.array(fpr_data, dtype=float)
                                tpr = np.array(tpr_data, dtype=float)
                                
                                # Calculate AUC only if data is valid
                                if len(fpr) > 0 and len(tpr) > 0:
                                    roc_auc = auc(fpr, tpr)
                                    metrics_dict['ROC AUC'] = roc_auc
                                    thresholds_dict['ROC AUC'] = 0.8
                        except Exception:
                            pass
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.plotly_chart(
                            create_metric_comparison_radar(metrics_dict, thresholds_dict),
                            use_container_width=True,
                            key="metric_comparison_radar"
                        )
                    with col2:
                        st.markdown("#### What am I looking at?")
                        st.markdown("""---""")
                        st.markdown("""
                            **Classification Performance Overview**
                            - Shows all key classification metrics
                            - Green area: Minimum targets
                            - Blue area: Your model's scores
                            
                            **Core Metrics:**
                            - Accuracy: Overall correct predictions (≥0.7)
                            - Precision: Positive prediction accuracy (≥0.7)
                            - Recall: Found positive cases (≥0.7)
                            - F1: Balance of precision & recall (≥0.7)
                            
                            **Advanced Metrics:**
                            - ROC AUC: Class separation ability (≥0.8)
                            
                        """)
                        with st.expander("📖 Detailed Interpretation"):
                            st.markdown("""---""")
                            st.markdown("**Key Points:**")
                            st.markdown("""
                            **Understanding Each Metric:**
                            
                            1. **Core Performance** (Target: ≥ 0.7)
                               - Accuracy: Percentage of correct predictions
                               - Precision: How many positive predictions were correct
                               - Recall: How many actual positives were found
                               - F1 Score: Harmonic mean of precision and recall
                            
                            2. **Advanced Performance** 
                               - **ROC AUC** (Target: ≥ 0.8): Class discrimination ability
                              
                            **What Good Looks Like:**
                            - Blue area covers or exceeds green area for ALL metrics
                            - Balanced performance across all dimensions
                            - No significant weak spots
                            
                            **Common Issues:**
                            - Low ROC AUC: Poor class separation
                            - Low Class Balance: Dataset imbalance issues
                            
                            **Improvement Tips:**
                            - Low core metrics → Model complexity/feature engineering
                            - Low ROC AUC → Feature quality/model selection
                            - Low balance → Data collection/class weighting
                            """)
                
                else:  # regression
                    metrics_dict = {
                        'R²': max(0, result['metrics'].get('r2', 0)),
                        'MAE Norm': max(0, 1 - (result['metrics'].get('mae', 0) / (0.2 * np.mean(np.abs(st.session_state.builder.y_test))))),
                        'RMSE Norm': max(0, 1 - (result['metrics'].get('rmse', 0) / (0.5 * np.std(st.session_state.builder.y_test))))
                    }
                    thresholds_dict = {
                        'R²': 0.5,
                        'MAE Norm': 0.5,  # Corresponds to MAE being less than 20% of mean target value
                        'RMSE Norm': 0.5   # Corresponds to RMSE being less than 50% of target std
                    }
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.plotly_chart(
                            create_metric_comparison_radar(metrics_dict, thresholds_dict),
                            use_container_width=True,
                            key="radar_chart"
                        )
                    with col2:
                        st.markdown("#### What am I looking at?")
                        st.markdown("""---""")
                        st.markdown("""
                            **Regression Performance Overview**
                            - Shows model performance metrics on a radar plot
                            - Green area: Target thresholds
                            - Blue area: Your model's performance
                            
                            **Core Metrics:**
                            - R²: Variance explained (≥0.5, higher is better)
                              - Should extend BEYOND green area
                            
                            - Error Metrics (should stay INSIDE green area):
                              - MAE Norm: Normalized Mean Absolute Error (≥0.5)
                              - RMSE Norm: Normalized Root Mean Square Error (≥0.5)
                            
                        """)
                        with st.expander("📖 Detailed Interpretation"):
                            st.markdown("""---""")
                            st.markdown("**Key Points:**")
                            st.markdown("""
                            **Understanding Each Metric:**
                            
                            1. **R² Score** (Target: ≥ 0.5)
                               - SHOULD EXTEND BEYOND green area
                               - 1.0 = Perfect prediction, 0.5 = Minimum acceptable
                               - 0.0 = No better than predicting mean
                            
                            2. **Normalized Error Metrics** (SHOULD STAY INSIDE green area)
                               - **MAE Norm**: Based on 20% of mean target value
                               - **RMSE Norm**: Based on 50% of target std deviation
                               - Higher values = better (less error)
                            
                            **What Good Performance Looks Like:**
                            
                            - R² (Blue) -----> Extends BEYOND green area
                            - Error Metrics -> Stay INSIDE green area  
                            
                            **Common Issues:**
                            - Low R²: Poor model fit
                            - Low Error Norms: Prediction errors too large
                            
                            """)
         
        # Generate improvements based on metrics
        improvements = []
        
        # Get model type from session state
        model_type = st.session_state.builder.model.get("type", "unknown")
        
        # Check for model fit issues first
        fit_status_metric = next((m for m in fit_metrics if m['metric'] == 'Model Fit Status'), None)
        if fit_status_metric:
            if "Underfitting" in fit_status_metric['status']:
                improvements.append({
                    'issue': 'Model Underfitting',
                    'category': 'Model Complexity',
                    'description': 'The model is not capturing the underlying patterns in the data.',
                    'impact': 'Poor performance on both training and validation data.',
                    'suggestions': [
                        'Increase model complexity:',
                        '  - For tree-based models: Increase max_depth or min_samples_split',
                        '  - For linear models: Add polynomial features or interaction terms',
                        '  - Consider using a more complex model type',
                        'Feature engineering improvements:',
                        '  - Create new features that better capture relationships',
                        '  - Review feature selection to ensure important features are included',
                        'Training process adjustments:',
                        '  - Increase the number of training iterations',
                        '  - Adjust learning rate if applicable',
                        '  - Review early stopping criteria'
                    ]
                })
            elif "Overfitting" in fit_status_metric['status']:
                improvements.append({
                    'issue': 'Model Overfitting',
                    'category': 'Model Complexity',
                    'description': 'The model is too closely fitted to the training data.',
                    'impact': 'Poor generalization to new data despite good training performance.',
                    'suggestions': [
                        'Reduce model complexity:',
                        '  - For tree-based models: Decrease max_depth or increase min_samples_split',
                        '  - For linear models: Add regularization (L1/L2)',
                        '  - Remove redundant or noisy features',
                        'Data improvements:',
                        '  - Collect more training data if possible',
                        '  - Review feature selection for potential noise',
                        '  - Consider cross-validation with more folds',
                        'Training adjustments:',
                        '  - Implement early stopping',
                        '  - Add dropout or other regularization techniques',
                        '  - Review validation strategy'
                    ]
                })
        
        # Check other performance issues
        performance_issues = []
        attention_metrics = [m for m in metrics_summary if '⚠️' in m['status']]
        
        # Check performance metrics based on problem type
        # Handle both binary and multiclass classification
        if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
            metrics_to_check = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            poor_metrics = [m for m in attention_metrics if m['metric'] in metrics_to_check]
            
            if poor_metrics:
                performance_issues.extend([f"low {m['metric'].lower()} ({m['value']})" for m in poor_metrics])
                
            # Check class balance
            class_balance_metric = next((m for m in metrics_summary if m['metric'] == 'Class Balance Ratio'), None)
            if class_balance_metric and '⚠️' in class_balance_metric['status']:
                performance_issues.append(f"class imbalance ({class_balance_metric['value']})")
        
        else:  # regression
            metrics_to_check = ['R² Score', 'Mean Absolute Error', 'Root Mean Square Error']
            poor_metrics = [m for m in attention_metrics if m['metric'] in metrics_to_check]
            
            if poor_metrics:
                performance_issues.extend([f"poor {m['metric'].lower()} ({m['value']})" for m in poor_metrics])
        
        if performance_issues:
            issues_description = ", ".join(performance_issues)
            
            base_suggestions = [
                'Increase the number of hyperparameter search iterations to explore the parameter space more thoroughly',
                'Increase the number of cross-validation folds in the Model Training stage for more robust model selection',
                'Review and refine the feature selection process',
                'Consider collecting additional training data if possible'
            ]
            
            data_suggestions = [
                'Review the data preprocessing steps:',
                '  - Check for and handle outliers',
                '  - Consider different feature scaling methods',
                '  - Look for opportunities to create interaction features',
                '  - Remove or combine highly correlated features'
            ]
            
            model_suggestions = [
                'Try a different model type:',
                '  - Linear/Logistic Regression for simpler, interpretable models',
                '  - Random Forest for handling non-linear relationships and feature interactions',
                '  - Gradient Boosting for potentially higher accuracy',
                '  - Neural Networks for complex pattern recognition'
            ]
            
            # Handle both binary and multiclass classification
            if problem_type in ["classification", "binary_classification", "multiclass_classification"] and any("class imbalance" in issue for issue in performance_issues):
                data_suggestions.extend([
                    'Address class imbalance:',
                    '  - Collect more data for minority classes',
                    '  - Consider adjusting sampling strategy in the Model Training stage',
                    '  - Review the sampling strategy used'
                ])
            
            all_suggestions = base_suggestions + data_suggestions + model_suggestions
            
            improvements.append({
                'issue': 'Model Performance Needs Improvement',
                'category': 'Model Performance and Data Quality',
                'description': f'The model shows multiple areas needing attention: {issues_description}.',
                'impact': 'These issues may affect model reliability and generalisation capability.',
                'suggestions': all_suggestions
            })
    else:
        st.warning("No metrics were collected. This might indicate missing data in the evaluation results.")
    
    # Display improvements if any
    if improvements:
        st.write("#### 📝 Suggested Improvements")
        # Log improvement suggestions
        st.session_state.logger.log_recommendation(
            "Model Improvements",
            {
                "problem_type": problem_type,
                "improvements": improvements,
                "metrics_summary": metrics_summary
            }
        )
        for improvement in improvements:
            with st.expander(f"⚠️ {improvement['issue']}", expanded=True):
                st.write(f"**Category:** {improvement['category']}")
                st.write(f"**Description:** {improvement['description']}")
                st.write(f"**Impact:** {improvement['impact']}")
                st.write("**Suggested Actions:**")
                for suggestion in improvement['suggestions']:
                    st.write(f"- {suggestion}")
    else:
        # Log good performance
        st.session_state.logger.log_recommendation(
            "No Improvements Needed",
            {
                "problem_type": problem_type,
                "metrics_summary": metrics_summary
            }
        )
        st.success("✅ No critical issues detected! The model appears to be performing well!")
