import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===== PERFORMANCE OPTIMIZATION: Import caching functions =====
from components.model_evaluation.visualisations import get_cached_predictions, get_cached_probabilities, get_data_hash

def add_probability_scores(predictions_df, model_instance, feature_sample, probabilities):
    """Add probability scores to predictions dataframe."""
    try:
        # Use pre-computed probabilities instead of calling predict_proba again
        
        if probabilities.shape[1] == 2:
            # Binary classification - show probability of positive class
            predictions_df.insert(2, 'Probability', probabilities[:, 1])
            predictions_df['Probability'] = predictions_df['Probability'].apply(
                lambda x: f"{x:.1%}"
            )
        else:
            # Multi-class classification - show maximum probability and predicted class confidence
            max_probs = np.max(probabilities, axis=1)
            predicted_classes = np.argmax(probabilities, axis=1)
            
            # Add max probability column
            predictions_df.insert(2, 'Max_Probability', max_probs)
            predictions_df['Max_Probability'] = predictions_df['Max_Probability'].apply(
                lambda x: f"{x:.1%}"
            )
            
            # Optionally, add the confidence in the predicted class
            pred_probs = [probabilities[i, pred] for i, pred in enumerate(predicted_classes)]
            predictions_df.insert(3, 'Pred_Class_Confidence', pred_probs)
            predictions_df['Pred_Class_Confidence'] = predictions_df['Pred_Class_Confidence'].apply(
                lambda x: f"{x:.1%}"
            )
    except Exception as e:
        # If probability prediction fails, just skip adding probabilities
        print(f"Error adding probability scores: {e}")
        pass

# Style the dataframe based on correct/incorrect predictions
def highlight_predictions(row):
    if ('Probability' in row.index or 'Max_Probability' in row.index):  # This indicates it's a classification model
        color = 'background-color: #c6efce' if row['Actual'] == row['Predicted'] else 'background-color: #ffc7ce'
    else:  # Regression model
        # Use relative tolerance of 5% for comparison
        actual = float(row['Actual'])
        predicted = float(row['Predicted'])
        # Handle division by zero or very small numbers
        if abs(actual) < 1e-10:
            relative_error = abs(actual - predicted)
            is_close = relative_error < 0.05  # Use absolute tolerance for values near zero
        else:
            relative_error = abs((actual - predicted) / actual)
            is_close = relative_error <= 0.05  # 5% relative tolerance
        
        if is_close:
            color = 'background-color: #c6efce'  # Green for good predictions
        elif relative_error <= 0.15:  # Between 5% and 15%
            color = 'background-color: #fff2cc'  # Yellow for moderate predictions
        else:
            color = 'background-color: #ffc7ce'  # Red for poor predictions
    return [color] * len(row)

def display_sample_predictions(problem_type):
    st.write("### 🔍 Sample Predictions")
    
    # Show threshold status if optimization is active
    if (hasattr(st.session_state, 'builder') and 
        hasattr(st.session_state.builder, 'model') and 
        st.session_state.builder.model.get("threshold_optimized", False) and
        problem_type in ["classification", "binary_classification", "multiclass_classification"]):
        
        optimal_threshold = st.session_state.builder.model.get("optimal_threshold", 0.5)
        is_binary = st.session_state.builder.model.get("threshold_is_binary", True)
        
        if is_binary:
            st.success(f"🎯 **Using Optimized Decision Threshold**: {optimal_threshold:.3f} for predictions")
        else:
            st.success(f"🎯 **Using Optimized Confidence Threshold**: {optimal_threshold:.3f} for predictions")
        
        st.info("💡 Sample predictions shown below use the optimized threshold.")
        st.markdown("---")
    
    with st.expander("ℹ️ Understanding Sample Predictions"):
        st.markdown("""
            This section shows real examples from your test data along with the model's predictions:
            
            - 🎯 **Actual**: The true value from your test data
            - 🤖 **Predicted**: What the model predicted
            - 📊 **Probability**: The model's confidence (classification only)
            
            **Color Coding:**
            
            For Classification Models:
            - 🟢 **Green**: Correct prediction (exact match)
            - 🔴 **Red**: Incorrect prediction
            
            For Regression Models:
            - 🟢 **Green**: Excellent prediction (within 5% of actual value)
            - 🟡 **Yellow**: Moderate prediction (between 5-15% of actual value)
            - 🔴 **Red**: Poor prediction (more than 15% off from actual value)
            
            **Benefits:**
            - See how your model performs on real examples
            - Identify patterns in correct/incorrect predictions
            - Understand prediction confidence levels
            - Visualize prediction accuracy through color coding
        """)
    col1, col2 = st.columns(2)
    with col1:
        model_instance = st.session_state.builder.model["model"]
        X_test = st.session_state.builder.X_test
        y_test = st.session_state.builder.y_test
        
        st.markdown("""
            ##### Select the number of samples to display
            """)
        # Add sample size control
        max_samples = len(X_test)  # Allow full dataset size
        sample_size = st.slider(
            "Number of samples to display",
            min_value=5,
            max_value=max_samples,
            value=min(50, max_samples),
            help="Select how many random samples to display from your test dataset"
        )
        
        # Log sample size selection
        st.session_state.logger.log_user_action(
            "Sample Size Selection",
            {
                "selected_size": sample_size,
                "max_available": max_samples,
                "problem_type": problem_type
            }
        )
    with col2:
        st.write(" ")

    # Add performance warning if sample size is large
    if sample_size > 50:
        st.warning("⚠️ Displaying a large number of samples may impact performance. Consider reducing the sample size if you experience slowdown.")
    
    # ===== PERFORMANCE: Cache sample indices to avoid recomputation on slider change =====
    if 'sample_indices' not in st.session_state or st.session_state.get('last_sample_size') != sample_size:
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        st.session_state.sample_indices = sample_indices
        st.session_state.last_sample_size = sample_size
    else:
        sample_indices = st.session_state.sample_indices

    feature_sample = X_test.iloc[sample_indices].copy()
    predictions_df = feature_sample.copy()
    predictions_df.insert(0, 'Actual', y_test.iloc[sample_indices])

    # ===== PERFORMANCE: Use cached predictions and probabilities =====
    X_test_hash = get_data_hash(X_test)
    has_proba = hasattr(model_instance, 'predict_proba')

    # Get cached full predictions
    y_pred_full = get_cached_predictions(model_instance, X_test, X_test_hash)
    y_prob_full = get_cached_probabilities(model_instance, X_test, X_test_hash, has_proba)

    # Extract predictions and probabilities for selected samples
    y_pred_sample = y_pred_full[sample_indices]
    y_prob_sample = y_prob_full[sample_indices] if y_prob_full is not None else None

    # Handle both binary and multiclass classification
    if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
        # Use optimal threshold if available, otherwise use default predictions
        if (hasattr(st.session_state, 'builder') and
            hasattr(st.session_state.builder, 'model') and
            st.session_state.builder.model.get("threshold_optimized", False)):

            optimal_threshold = st.session_state.builder.model.get("optimal_threshold", 0.5)
            is_binary = st.session_state.builder.model.get("threshold_is_binary", True)

            if is_binary and y_prob_sample is not None:
                # Binary classification with optimal threshold
                if len(y_prob_sample.shape) > 1 and y_prob_sample.shape[1] == 2:
                    y_prob_positive = y_prob_sample[:, 1]
                    predictions = (y_prob_positive >= optimal_threshold).astype(int)
                else:
                    predictions = y_pred_sample
            elif not is_binary and y_prob_sample is not None:
                # Multiclass classification with optimal confidence threshold
                max_probs = np.max(y_prob_sample, axis=1)
                predicted_classes = np.argmax(y_prob_sample, axis=1)

                # Only make predictions where confidence is above threshold
                confident_mask = max_probs >= optimal_threshold
                predictions = np.full(len(feature_sample), -1)  # -1 for uncertain predictions
                predictions[confident_mask] = predicted_classes[confident_mask]

                # Convert -1 (uncertain) to most likely class for display purposes
                predictions[predictions == -1] = predicted_classes[predictions == -1]
            else:
                predictions = y_pred_sample
        else:
            predictions = y_pred_sample

        predictions_df.insert(1, 'Predicted', predictions)

        # Only add probability scores if the model supports it
        if y_prob_sample is not None:
            try:
                add_probability_scores(predictions_df, model_instance, feature_sample, y_prob_sample)
            except Exception as e:
                print(f"Warning: Could not add probability scores: {e}")
    else:
        predictions = y_pred_sample
        predictions_df.insert(1, 'Predicted', np.round(predictions, 4))
        predictions_df['Actual'] = np.round(predictions_df['Actual'], 4)
    
    predictions_df = predictions_df.reset_index(drop=True)
    styled_df = predictions_df.style.apply(highlight_predictions, axis=1)
    st.dataframe(styled_df, width='stretch')
    
    # Calculate and display metrics for sample predictions
    # Handle both binary and multiclass classification
    if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
        sample_accuracy = np.mean(predictions_df['Actual'] == predictions_df['Predicted'])
        st.metric("Sample Accuracy", f"{sample_accuracy:.1%}")
        metrics = {
            "accuracy": float(sample_accuracy)
        }
    else:  # regression
        sample_mae = mean_absolute_error(predictions_df['Actual'], predictions_df['Predicted'])
        sample_rmse = np.sqrt(mean_squared_error(predictions_df['Actual'], predictions_df['Predicted']))
        sample_r2 = r2_score(predictions_df['Actual'], predictions_df['Predicted'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sample MAE", f"{sample_mae:.3f}")
        with col2:
            st.metric("Sample RMSE", f"{sample_rmse:.3f}")
        with col3:
            st.metric("Sample R²", f"{sample_r2:.3f}")
        
        metrics = {
            "mae": float(sample_mae),
            "rmse": float(sample_rmse),
            "r2": float(sample_r2)
        }
    
    # Log sample predictions
    st.session_state.logger.log_calculation(
        "Sample Predictions",
        {
            "num_samples": sample_size,
            "problem_type": problem_type,
            "metrics": metrics
        }
    )
    
    return {
        "predictions": predictions_df.to_dict('records'),
        "metrics": metrics,
        "num_samples": len(predictions_df)
    }