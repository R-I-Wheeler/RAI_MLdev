import streamlit as st
import pandas as pd
import numpy as np
from Builder import Builder, ModelStage
from app import create_sidebar_navigation
from utils.logging.log_viewer import render_log_viewer
from utils.logging.journey_viewer import render_journey_viewer
import json
import plotly.graph_objects as go


def get_current_problem_type():
    """Get the current problem type from session state with fallback to builder model."""
    if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type:
        return st.session_state.problem_type
    elif st.session_state.builder.model is not None:
        return st.session_state.builder.model.get("problem_type", "unknown")
    else:
        return "unknown"

def create_download_button(data, filename, mime_type, help_text=None):
    """Helper function to create a download button with tooltip."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.to_csv(index=False)
    elif isinstance(data, (dict, list)):
        data = json.dumps(data, indent=2)
    elif isinstance(data, (np.ndarray, np.generic)):
        data = np.array2string(data)
    
    data_bytes = data.encode() if isinstance(data, str) else data
    
    # Create the download button
    button = st.download_button(
        label=f"📥 Download {filename}",
        data=data_bytes,
        file_name=filename,
        mime=mime_type
    )
    
    # Add help text below the button if provided
    if help_text:
        st.markdown(f"<div style='font-size: 0.8em; color: #666; margin-top: -10px;'>{help_text}</div>", unsafe_allow_html=True)
    
    return button

def create_stage_progress(stage_completion):
    """Create a progress bar showing completion status of all stages."""
    stages = list(ModelStage)
    completed = sum(1 for stage in stages if stage_completion.get(stage, False))
    total = len(stages)
    
    # Create progress bar
    progress = completed / total
    st.progress(progress, text=f"Overall Progress: {completed}/{total} stages completed")
    
    # Create stage status indicators
    cols = st.columns(len(stages))
    for idx, stage in enumerate(stages):
        with cols[idx]:
            status = "✅" if stage_completion.get(stage, False) else "⏳"
            # Format stage name by replacing underscores with spaces and title casing
            stage_name = stage.value.replace('_', ' ').title()
            st.markdown(f"{status} {stage_name}")

def create_metrics_visualization(metrics):
    """Create a visual representation of model metrics."""
    if not metrics:
        return
    
    # Create a figure with subplots
    fig = go.Figure()
    
    # Add metrics as bars
    fig.add_trace(go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        text=[f"{v:.4f}" for v in metrics.values()],
        textposition='auto',
        marker_color='#1f77b4'
    ))
    
    # Update layout
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        showlegend=False,
        height=300,
        margin=dict(t=30, b=30, l=30, r=30)
    )
    
    return fig

def generate_model_recreation_code():
    """Generate code to recreate the model from scratch using the downloaded datasets."""
    
    if st.session_state.builder.model is None:
        return None
    
    # Get model information
    model_type = st.session_state.builder.model["type"]
    
    # Use session state for problem type detection
    if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type:
        problem_type = st.session_state.problem_type
    else:
        # Fallback to builder model for backward compatibility
        problem_type = st.session_state.builder.model.get("problem_type", "unknown")
    
    target_column = st.session_state.builder.target_column
    
    # Get model parameters
    model_instance = st.session_state.builder.model.get("model")
    active_params = st.session_state.builder.model.get('active_params', {})
    best_params = st.session_state.builder.model.get('best_params', {})
    
    # Check if model is calibrated
    is_calibrated = st.session_state.builder.model.get('is_calibrated', False)
    calibration_method = st.session_state.builder.model.get('calibration_method', 'isotonic')
    calibration_cv_folds = st.session_state.builder.model.get('calibration_cv_folds', 5)
    
    # Check if threshold optimization was applied
    threshold_optimized = st.session_state.builder.model.get('threshold_optimized', False)
    optimal_threshold = st.session_state.builder.model.get('optimal_threshold', 0.5)
    threshold_is_binary = st.session_state.builder.model.get('threshold_is_binary', True)
    threshold_criterion = st.session_state.builder.model.get('threshold_criterion', 'F1 Score')
    
    # Use active_params if available, otherwise use best_params
    model_params = active_params if active_params else best_params
    
    # Filter out None values from parameters
    if model_params:
        model_params = {k: v for k, v in model_params.items() if v is not None}
    
    # Get feature names
    features = list(st.session_state.builder.X_train.columns) if hasattr(st.session_state.builder, 'X_train') else []
    
    # Determine the specific classification type for model mappings
    is_classification = problem_type in ["binary_classification", "multiclass_classification"]
    
    # Map model types to actual sklearn/xgboost/lightgbm class names and imports
    model_mappings = {
        # Classification models
        "logistic_regression": {
            "import": "from sklearn.linear_model import LogisticRegression",
            "class_name": "LogisticRegression",
            "default_params": {"random_state": 42, "n_jobs": -1}
        },
        "naive_bayes": {
            "import": "from sklearn.naive_bayes import GaussianNB",
            "class_name": "GaussianNB",
            "default_params": {}
        },
        "decision_tree": {
            "import": "from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor",
            "class_name": "DecisionTreeClassifier" if is_classification else "DecisionTreeRegressor",
            "default_params": {"random_state": 42}
        },
        "random_forest": {
            "import": "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor",
            "class_name": "RandomForestClassifier" if is_classification else "RandomForestRegressor",
            "default_params": {"random_state": 42, "n_jobs": -1}
        },
        "mlp": {
            "import": "from sklearn.neural_network import MLPClassifier, MLPRegressor",
            "class_name": "MLPClassifier" if is_classification else "MLPRegressor",
            "default_params": {"max_iter": 1000, "random_state": 42}
        },
        "xgboost": {
            "import": "from xgboost import XGBClassifier, XGBRegressor",
            "class_name": "XGBClassifier" if is_classification else "XGBRegressor",
            "default_params": {
                "random_state": 42, 
                "use_label_encoder": False, 
                "eval_metric": 'logloss' if problem_type == "binary_classification" else ('mlogloss' if problem_type == "multiclass_classification" else 'rmse'),
                "nthread": -1
            }
        },
        "lightgbm": {
            "import": "from lightgbm import LGBMClassifier, LGBMRegressor",
            "class_name": "LGBMClassifier" if is_classification else "LGBMRegressor",
            "default_params": {
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
                "objective": 'binary' if problem_type == "binary_classification" else ('multiclass' if problem_type == "multiclass_classification" else 'regression')
            }
        },
        "hist_gradient_boosting": {
            "import": "from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor",
            "class_name": "HistGradientBoostingClassifier" if is_classification else "HistGradientBoostingRegressor",
            "default_params": {"random_state": 42}
        },
        "catboost": {
            "import": "from catboost import CatBoostClassifier, CatBoostRegressor",
            "class_name": "CatBoostClassifier" if is_classification else "CatBoostRegressor",
            "default_params": {
                "random_state": 42,
                "verbose": False,
                "iterations": 100
            }
        },
        # Regression models
        "linear_regression": {
            "import": "from sklearn.linear_model import LinearRegression",
            "class_name": "LinearRegression",
            "default_params": {"n_jobs": -1}
        },
        "ridge_regression": {
            "import": "from sklearn.linear_model import Ridge",
            "class_name": "Ridge",
            "default_params": {"random_state": 42}
        }
    }
    
    if model_type not in model_mappings:
        return None
    
    model_config = model_mappings[model_type]
    model_import = model_config["import"]
    class_name = model_config["class_name"]
    default_params = model_config["default_params"]
    
    # Combine default parameters with tuned parameters
    all_params = {**default_params, **model_params}
    
    # Generate calibration-specific code sections
    if is_calibrated:
        calibration_import = "from sklearn.calibration import CalibratedClassifierCV"
        calibration_metrics_import = """from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve"""
        calibration_code = "# Base model will be calibrated after training"
        
        calibration_analysis = """# Calibration Analysis (added because calibration was applied)
            if hasattr(model, "predict_proba"):
                try:
                    print("\\n" + "="*60)
                    print("CALIBRATION ANALYSIS")
                    print("="*60)
                    y_prob = model.predict_proba(X_test)
                    if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                        # Binary classification calibration
                        y_prob_pos = y_prob[:, 1]
                        brier_score = brier_score_loss(y_test, y_prob_pos)
                        print(f"Brier Score: {brier_score:.4f} (lower is better)")
                        # Calculate calibration curve
                        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob_pos, n_bins=10)
                        print("Calibration curve data:")
                        for i, (frac, pred) in enumerate(zip(fraction_of_positives, mean_predicted_value)):
                            print(f"  Bin {i+1}: Predicted={pred:.3f}, Actual={frac:.3f}")
                    else:
                        # Multiclass calibration metrics
                        from sklearn.preprocessing import LabelBinarizer
                        lb = LabelBinarizer()
                        y_test_onehot = lb.fit_transform(y_test)
                        if len(y_test_onehot.shape) == 1:
                            y_test_onehot = np.column_stack([1 - y_test_onehot, y_test_onehot])
                        brier_score = np.mean(np.sum((y_prob - y_test_onehot) ** 2, axis=1))
                        print(f"Multiclass Brier Score: {brier_score:.4f} (lower is better)")
                    print("\\nCalibration improves probability reliability for decision-making.")
                except Exception as e:
                    print(f"Could not perform calibration analysis: {e}")"""
        
        calibration_summary = f'print(f"Calibration Applied: Yes ({calibration_method} method, {calibration_cv_folds} CV folds)")'
        
        post_training_calibration = f"""# Apply calibration (this was used in development)
    print("\\nApplying calibration to trained model...")
    print(f"Calibration method: {calibration_method}")
    print(f"CV folds for calibration: {calibration_cv_folds}")
    
    try:
        # Create calibrated classifier with the trained base model
        calibrated_model = CalibratedClassifierCV(
            estimator=model,
            method="{calibration_method}",
            cv={calibration_cv_folds},
            n_jobs=-1
        )
        
        # Fit calibration on training data
        calibrated_model.fit(X_train, y_train)
        model = calibrated_model  # Replace base model with calibrated model
        print("Model calibration completed successfully!")
        
    except Exception as e:
        print(f"Error during calibration: {{e}}")
        print("Continuing with uncalibrated model...")"""
        
    else:
        calibration_import = ""
        calibration_metrics_import = ""
        calibration_code = "# No calibration was applied during development"
        calibration_analysis = ""
        calibration_summary = 'print("Calibration Applied: No")'
        post_training_calibration = ""
    
    # Generate threshold optimization specific code sections
    if threshold_optimized and is_classification:
        threshold_function = f"""
def apply_optimal_threshold(model, X_test, threshold={optimal_threshold}, is_binary={threshold_is_binary}):
    \"\"\"Apply the optimal threshold for predictions.\"\"\"
    if not hasattr(model, 'predict_proba'):
        print("Warning: Model doesn't support probability predictions. Using standard predict method.")
        return model.predict(X_test)
    
    y_prob = model.predict_proba(X_test)
    
    if is_binary:
        # Binary classification threshold
        if len(y_prob.shape) > 1:
            y_prob_positive = y_prob[:, 1]
        else:
            y_prob_positive = y_prob
        
        # Apply threshold
        y_pred_threshold = (y_prob_positive >= threshold).astype(int)
        print(f"Applied optimal threshold: {{threshold:.3f}} (instead of default 0.5)")
        return y_pred_threshold
    else:
        # Multiclass classification - confidence threshold
        max_probs = np.max(y_prob, axis=1)
        predicted_classes = np.argmax(y_prob, axis=1)
        
        # Only predict when confidence is above threshold
        confident_mask = max_probs >= threshold
        y_pred_threshold = np.full(len(X_test), -1)  # -1 indicates uncertain prediction
        y_pred_threshold[confident_mask] = predicted_classes[confident_mask]
        
        coverage = confident_mask.mean()
        print(f"Applied optimal confidence threshold: {{threshold:.3f}}")
        print(f"Coverage: {{coverage:.1%}} of predictions are confident")
        return y_pred_threshold"""
        
        if threshold_is_binary:
            threshold_prediction_code = f"""
    # Apply optimal threshold (this was used in development)
    print("\\nApplying optimal threshold for predictions...")
    y_pred_threshold = apply_optimal_threshold(model, X_test, threshold={optimal_threshold}, is_binary={threshold_is_binary})
    
    print("\\n" + "="*60)
    print("THRESHOLD OPTIMIZATION COMPARISON")
    print("="*60)
    
    # Standard predictions (default threshold 0.5)
    y_pred_standard = model.predict(X_test)
    
    # Binary classification comparison
    standard_accuracy = accuracy_score(y_test, y_pred_standard)
    standard_f1 = f1_score(y_test, y_pred_standard, zero_division=0)
    
    print(f"Standard predictions (0.5 threshold):")
    print(f"  Accuracy: {{standard_accuracy:.4f}}")
    print(f"  F1-Score: {{standard_f1:.4f}}")
    
    threshold_accuracy = accuracy_score(y_test, y_pred_threshold)
    threshold_f1 = f1_score(y_test, y_pred_threshold, zero_division=0)
    print(f'Optimized predictions ({optimal_threshold:.3f} threshold):')
    print(f'  Accuracy: {{threshold_accuracy:.4f}}')
    print(f'  F1-Score: {{threshold_f1:.4f}}')
    
    # Use threshold-optimized predictions for final evaluation
    y_pred = y_pred_threshold"""
        else:
            threshold_prediction_code = f"""
    # Apply optimal threshold (this was used in development)
    print("\\nApplying optimal confidence threshold for predictions...")
    y_pred_threshold = apply_optimal_threshold(model, X_test, threshold={optimal_threshold}, is_binary={threshold_is_binary})
    
    print("\\n" + "="*60)
    print("THRESHOLD OPTIMIZATION COMPARISON")
    print("="*60)
    
    # Standard predictions (default threshold 0.5)
    y_pred_standard = model.predict(X_test)
    
    # Multiclass classification comparison
    standard_accuracy = accuracy_score(y_test, y_pred_standard)
    standard_f1 = f1_score(y_test, y_pred_standard, average='weighted', zero_division=0)
    
    print(f"Standard predictions (0.5 threshold):")
    print(f"  Accuracy: {{standard_accuracy:.4f}}")
    print(f"  F1-Score: {{standard_f1:.4f}}")
    
    if y_pred_threshold is not None and len(y_pred_threshold) > 0:
        # Only evaluate confident predictions for multiclass
        confident_mask = y_pred_threshold != -1
        if confident_mask.sum() > 0:
            threshold_accuracy = accuracy_score(y_test[confident_mask], y_pred_threshold[confident_mask])
            threshold_f1 = f1_score(y_test[confident_mask], y_pred_threshold[confident_mask], average='weighted', zero_division=0)
            print(f'Optimized predictions ({optimal_threshold:.3f} confidence threshold):')
            print(f'  Accuracy: {{threshold_accuracy:.4f}}')
            print(f'  F1-Score: {{threshold_f1:.4f}}')
        else:
            print('No confident predictions with optimized threshold')
    
    # Use threshold-optimized predictions for final evaluation
    y_pred = y_pred_threshold"""
        
        threshold_summary = f'print(f"Threshold Optimization: Applied (threshold = {optimal_threshold}, criterion = {threshold_criterion})")'
        
    else:
        threshold_function = ""
        threshold_prediction_code = ""
        threshold_summary = 'print("Threshold Optimization: Not applied (using default 0.5)")'
    
    # Determine prediction code based on threshold optimization
    if threshold_optimized and is_classification:
        prediction_code_line = "# Threshold-optimized predictions already set above\n        pass  # y_pred already defined above"
    else:
        prediction_code_line = "y_pred = model.predict(X_test)"
    
    # Generate the code
    code = f'''"""
Model Recreation Script
This script recreates the exact model that was developed and trained.
Use this with the downloaded training_data.csv and test_data.csv files.

Model Details:
- Type: {model_type}
- Problem Type: {problem_type}
- Target Variable: {target_column}
- Features: {len(features)} features
{'- Calibration: ' + calibration_method + ' method with ' + str(calibration_cv_folds) + ' CV folds' if is_calibrated else '- Calibration: Not applied'}
{'- Threshold Optimization: Applied (threshold = ' + str(optimal_threshold) + ')' if threshold_optimized else '- Threshold Optimization: Not applied (using default 0.5)'}
"""

import pandas as pd
import numpy as np
{model_import}
{calibration_import}
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
{calibration_metrics_import}
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
{threshold_function}

def load_and_prepare_data():
    """Load the training and test datasets."""
    print("Loading datasets...")
    
    try:
        # Load the datasets
        train_data = pd.read_csv('training_data.csv')
        test_data = pd.read_csv('test_data.csv')
        
        print(f"Training data shape: {{train_data.shape}}")
        print(f"Test data shape: {{test_data.shape}}")
        
        # Define feature columns and target
        feature_columns = {features}
        target_column = '{target_column}'
        
        # Verify columns exist in the data
        missing_features = [col for col in feature_columns if col not in train_data.columns]
        if missing_features:
            print(f"Warning: Missing features in training data: {{missing_features}}")
        
        if target_column not in train_data.columns:
            print(f"Error: Target column '{{target_column}}' not found in training data")
            return None, None, None, None
        
        # Separate features and target
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]
        
        print(f"Training set: {{X_train.shape[0]}} samples, {{X_train.shape[1]}} features")
        print(f"Test set: {{X_test.shape[0]}} samples, {{X_test.shape[1]}} features")
        print(f"Target column: {{target_column}}")
        
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError as e:
        print(f"Error: {{e}}")
        print("Please ensure training_data.csv and test_data.csv are in the same directory as this script.")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading data: {{e}}")
        return None, None, None, None

def create_model():
    """Create the model with the exact same parameters used in development."""
    print("Creating model with optimized parameters...")
    
    # Model parameters (exactly as used in your development)
    model_params = {all_params}
    
    print(f"Model type: {class_name}")
    print(f"Parameters: {{model_params}}")
    
    # Create the base model instance
    base_model = {class_name}(**model_params)
    
    {calibration_code}
    model = base_model
    
    return model

def train_and_evaluate_model():
    """Train the model and evaluate its performance."""
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    if X_train is None:
        print("Failed to load data. Exiting.")
        return None, None, None, None
    
    # Create model
    model = create_model()
    
    # Train the model
    print("\\nTraining the model...")
    print("This may take a few moments depending on your data size and model complexity...")
    
    try:
        model.fit(X_train, y_train)
        print("Model training completed successfully!")
    except Exception as e:
        print(f"Error during training: {{e}}")
        return None, None, None, None
    
    {post_training_calibration}
    {threshold_prediction_code}
    
    # Make predictions (will use threshold-optimized predictions if available)
    print("Making predictions on test set...")
    try:
        {prediction_code_line}
    except Exception as e:
        print(f"Error during prediction: {{e}}")
        return model, X_test, y_test, None
    
    # Evaluate the model
    print("\\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    try:
        if '{problem_type}' in ['binary_classification', 'multiclass_classification']:
            # Classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Handle different classification types
            if '{problem_type}' == 'multiclass_classification':
                # Use macro averaging for multiclass
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                
                print(f"Accuracy:           {{accuracy:.4f}}")
                print(f"Precision (Macro):  {{precision:.4f}}")
                print(f"Recall (Macro):     {{recall:.4f}}")
                print(f"F1-Score (Macro):   {{f1:.4f}}")
                
                # Additional multiclass metrics
                precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                print(f"\\nWeighted Averages:")
                print(f"Precision (Weighted): {{precision_weighted:.4f}}")
                print(f"Recall (Weighted):    {{recall_weighted:.4f}}")
                print(f"F1-Score (Weighted):  {{f1_weighted:.4f}}")
                
            else:
                # Binary classification
                precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            
            print(f"Accuracy:  {{accuracy:.4f}}")
            print(f"Precision: {{precision:.4f}}")
            print(f"Recall:    {{recall:.4f}}")
            print(f"F1-Score:  {{f1:.4f}}")
            
            print("\\nDetailed Classification Report:")
            print(classification_report(y_test, y_pred))
            
            {calibration_analysis}
            
            # Confusion Matrix Visualization
            try:
                plt.figure(figsize=(10, 8))
                cm = confusion_matrix(y_test, y_pred)
                
                # Get unique classes for proper labeling
                unique_classes = np.unique(np.concatenate([y_test, y_pred]))
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=unique_classes, yticklabels=unique_classes)
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.show()
                print("Confusion matrix saved as 'confusion_matrix.png'")
                
                # Per-class analysis for multiclass
                if '{problem_type}' == 'multiclass_classification':
                    print("\\nPer-Class Performance:")
                    per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
                    per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
                    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
                    
                    for i, class_label in enumerate(unique_classes):
                        print(f"Class {{class_label}}: Precision={{per_class_precision[i]:.4f}}, Recall={{per_class_recall[i]:.4f}}, F1={{per_class_f1[i]:.4f}}")
                        
            except Exception as e:
                print(f"Could not create confusion matrix plot: {{e}}")
                
        else:
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            print(f"R² Score:                {{r2:.4f}}")
            print(f"Mean Absolute Error:     {{mae:.4f}}")
            print(f"Mean Squared Error:      {{mse:.4f}}")
            print(f"Root Mean Squared Error: {{rmse:.4f}}")
            
            # Prediction vs Actual plot
            try:
                plt.figure(figsize=(12, 5))
                
                # Subplot 1: Predicted vs Actual
                plt.subplot(1, 2, 1)
                plt.scatter(y_test, y_pred, alpha=0.6, s=50)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title('Predicted vs Actual Values')
                plt.grid(True, alpha=0.3)
                
                # Subplot 2: Residuals plot
                residuals = y_test - y_pred
                plt.subplot(1, 2, 2)
                plt.scatter(y_pred, residuals, alpha=0.6, s=50)
                plt.axhline(y=0, color='r', linestyle='--', lw=2)
                plt.xlabel('Predicted Values')
                plt.ylabel('Residuals (Actual - Predicted)')
                plt.title('Residual Plot')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('regression_plots.png', dpi=300, bbox_inches='tight')
                plt.show()
                print("Regression plots saved as 'regression_plots.png'")
            except Exception as e:
                print(f"Could not create regression plots: {{e}}")
        
        # Cross-validation
        print("\\nPerforming cross-validation...")
        try:
            if '{problem_type}' in ['binary_classification', 'multiclass_classification']:
                scoring_metric = 'accuracy'
            else:
                scoring_metric = 'r2'
                
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring_metric)
            print(f"Cross-validation scores: {{[f'{{score:.4f}}' for score in cv_scores]}}")
            print(f"Mean CV score: {{cv_scores.mean():.4f}} (+/- {{cv_scores.std() * 2:.4f}})")
            
            # Model stability assessment
            cv_std = cv_scores.std()
            if cv_std < 0.05:
                stability = "High"
            elif cv_std < 0.1:
                stability = "Moderate"
            else:
                stability = "Low"
            print(f"Model stability: {{stability}} (std = {{cv_std:.4f}})")
            
        except Exception as e:
            print(f"Could not perform cross-validation: {{e}}")
        
    except Exception as e:
        print(f"Error during evaluation: {{e}}")
    
    return model, X_test, y_test, y_pred

def generate_model_summary():
    """Generate a summary of the recreated model."""
    print("\\n" + "="*60)
    print("MODEL RECREATION SUMMARY")
    print("="*60)
    print(f"Model Type: {{MODEL_TYPE}} ({class_name})")
    print(f"Problem Type: {{PROBLEM_TYPE}}")
    print(f"Target Variable: {{TARGET_COLUMN}}")
    print(f"Number of Features: {len(features)}")
    print(f"Model Parameters: {len(all_params)} parameters configured")
    {calibration_summary}
    {threshold_summary}
    print("\\nFiles Generated:")
    print("- recreated_model.pkl (trained model)")
    if PROBLEM_TYPE in ['binary_classification', 'multiclass_classification']:
        print("- confusion_matrix.png (confusion matrix visualization)")
    else:
        print("- regression_plots.png (prediction vs actual and residual plots)")
    print("\\nNext Steps:")
    print("1. Use 'recreated_model.pkl' for making predictions on new data")
    print("2. Review the evaluation metrics to understand model performance")
    print("3. Examine the generated plots to assess model quality")

if __name__ == "__main__":
    import os
    
    # Model configuration (defined at script level)
    MODEL_TYPE = "{model_type}"
    PROBLEM_TYPE = "{problem_type}"
    TARGET_COLUMN = "{target_column}"
    
    print("="*60)
    print("ML MODEL RECREATION SCRIPT")
    print("="*60)
    print("This script recreates your exact trained model using the downloaded datasets.")
    print("\\nRequirements:")
    print("- training_data.csv (training dataset)")
    print("- test_data.csv (test dataset)")
    print("- Required Python packages: pandas, numpy, scikit-learn, matplotlib, seaborn")
    if MODEL_TYPE in ['xgboost']:
        print("- xgboost package")
    if MODEL_TYPE in ['lightgbm']:
        print("- lightgbm package")
    print("\\nStarting model recreation process...")
    print()
    
    try:
        # Train and evaluate the model
        model, X_test, y_test, y_pred = train_and_evaluate_model()
        
        if model is not None:
            # Generate summary
            generate_model_summary()
            
            print("\\n" + "="*60)
            print("MODEL RECREATION COMPLETED SUCCESSFULLY!")
            print("="*60)
        else:
            print("\\nModel recreation failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"\\nUnexpected error occurred: {{e}}")
        print("Please check that all required files are present and try again.")
'''
    
    return code

def main():
    st.title("🎉 ML Development Summary")
    
    # Add consistent navigation
    create_sidebar_navigation()
    
    # Initialize session state if needed
    if 'builder' not in st.session_state:
        st.session_state.builder = Builder()
        st.session_state.logger.log_stage_transition("START", "SUMMARY")
    
    # Set current stage to SUMMARY
    st.session_state.builder.current_stage = ModelStage.MODEL_EXPLANATION
    
    # Get and render stage info
    stage_info = st.session_state.builder.get_current_stage_info()
    
    # Log page state
    st.session_state.logger.log_page_state("Summary", {
        "model_exists": st.session_state.builder.model is not None,
        "data_loaded": st.session_state.builder.data is not None,
        "stages_completed": {stage.value: completed for stage, completed in st.session_state.builder.stage_completion.items()}
    })
    
    # Add stage progress indicator
    create_stage_progress(st.session_state.builder.stage_completion)
    
    # Add introduction section
    st.markdown("### Welcome to Your ML Development Summary")
    st.markdown("This page provides a comprehensive overview of your machine learning model development process. Here's what you'll find:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Model Information")
        st.markdown("""
        - Details about your trained model
        - Features used and target variable
        - Model performance metrics
        - Download option for the trained model
        """)
    
    with col2:
        st.markdown("#### Dataset Downloads")
        st.markdown("""
        - Training and test datasets
        - Preprocessed data ready for use
        - Complete feature set and target variable
        """)
    
    with col3:
        st.markdown("#### Development Logs")
        st.markdown("""
        - Complete history of your development process
        - Filterable by type and time range
        - Downloadable log history
        """)
    
    st.markdown("### Use this page to:")
    st.markdown("""
    - Review your model's performance and configuration
    - Download all necessary files for future use
    - Document your development process
    - Share your work with team members
    """)
    
    # Model Information
    if st.session_state.builder.model is not None:
        st.header("🤖 Model Information")
        
        # Create main columns for layout
        info_col, metrics_col = st.columns([1.2, 1])
        
        with info_col:
            # Create a styled container for model details
            with st.container():
                st.markdown("""
                <style>
                    [data-testid="stMarkdownContainer"] > div {
                        word-break: break-word;
                    }
                    .model-card {
                        background: var(--background-color, linear-gradient(135deg, rgba(245, 247, 250, 0.1), rgba(228, 232, 235, 0.1)));
                        border-radius: 15px;
                        padding: 25px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        margin-bottom: 50px;
                        border: 1px solid rgba(128, 128, 128, 0.2);
                    }
                    .model-header {
                        display: flex;
                        align-items: center;
                        margin-bottom: 15px;
                    }
                    .model-stats {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(50px, 1fr));
                        gap: 15px;
                        margin-top: 15px;
                    }
                    .stat-card {
                        background: var(--background-color, rgba(255, 255, 255, 0.05));
                        padding: 12px;
                        border-radius: 8px;
                        text-align: center;
                        border: 1px solid rgba(128, 128, 128, 0.2);
                    }
                    .metric-value {
                        font-size: 1.2em;
                        font-weight: bold;
                        margin-top: 5px;
                    }
                    .metric-label {
                        font-size: 0.9em;
                        opacity: 0.8;
                    }
                    .feature-list {
                        max-height: 50px;
                        overflow-y: auto;
                        padding: 10px;
                        background: var(--background-color, rgba(255, 255, 255, 0.05));
                        border-radius: 5px;
                        border: 1px solid rgba(128, 128, 128, 0.2);
                    }
                    /* Dark mode specific adjustments */
                    @media (prefers-color-scheme: dark) {
                        .model-card {
                            background: linear-gradient(135deg, rgba(45, 47, 50, 0.5), rgba(28, 32, 35, 0.5));
                        }
                        .stat-card {
                            background: rgba(45, 47, 50, 0.5);
                        }
                        .feature-list {
                            background: rgba(45, 47, 50, 0.5);
                        }
                    }
                </style>
                """, unsafe_allow_html=True)
                
                # Enhanced Model Overview Card
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                
                # Model Type and Problem Type
                col1, col2 = st.columns(2)
                with col1:
                    model_type = st.session_state.builder.model.get('type', 'Unknown').replace('_', ' ').title()
                    st.markdown(f"##### 📊 Model Type")
                    st.markdown(f"**{model_type}**")
                with col2:
                    # Use session state for problem type detection
                    if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type:
                        problem_type = st.session_state.problem_type
                    else:
                        # Fallback to builder model for backward compatibility
                        problem_type = st.session_state.builder.model.get('problem_type', 'Unknown')
                    
                    # Format problem type for display
                    problem_type_display = problem_type.replace('_', ' ').title()
                    st.markdown(f"##### 🎯 Problem Type")
                    st.markdown(f"**{problem_type_display}**")
                    
                    # Add additional info for multiclass
                    if hasattr(st.session_state, 'is_multiclass') and st.session_state.is_multiclass:
                        if hasattr(st.session_state.builder, 'y_train'):
                            n_classes = len(st.session_state.builder.y_train.unique())
                            st.markdown(f"*{n_classes} classes*")
                
                # Model Statistics
                st.markdown('<div class="model-stats">', unsafe_allow_html=True)
                
                # Training Data Stats
                if hasattr(st.session_state.builder, 'X_train'):
                    train_samples = len(st.session_state.builder.X_train)
                    train_features = st.session_state.builder.X_train.shape[1]
                    st.markdown(f"""
                        <div class="stat-card">
                            <div class="metric-label">Training Samples</div>
                            <div class="metric-value">{train_samples:,}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="stat-card">
                            <div class="metric-label">Features</div>
                            <div class="metric-value">{train_features}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Model Size (if available)
                import pickle
                model_size = len(pickle.dumps(st.session_state.builder.model.get('model'))) / (1024 * 1024)  # Size in MB
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="metric-label">Model Size</div>
                        <div class="metric-value">{model_size:.1f} MB</div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close model-stats
                st.markdown('</div>', unsafe_allow_html=True)  # Close model-card
                
                # Feature Information
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("#### 📋 Feature Information")
                
                # Check if builder exists and has the required data
                if not hasattr(st.session_state, 'builder'):
                    st.warning("No model builder available. Please complete the data preparation stage first.")
                elif not hasattr(st.session_state.builder, 'X_train') or st.session_state.builder.X_train is None:
                    st.warning("No training data available. Please complete the data preprocessing stage first.")
                else:
                    # Get feature information
                    features = list(st.session_state.builder.X_train.columns)
                    
                    # Feature list with scrollable area
                    st.markdown("##### Feature List:")
                    st.markdown('<div class="feature-list">', unsafe_allow_html=True)
                    for feature in features:
                        feature_type = "🔢"
                        st.markdown(f"{feature_type} **{feature}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close model-card
                
                # Target Variable Information
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("#### 🎯 Target Variable")
                target = st.session_state.builder.target_column
                
                if hasattr(st.session_state.builder, 'y_train'):
                    # Use session state to determine problem type
                    if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type:
                        current_problem_type = st.session_state.problem_type
                    else:
                        current_problem_type = problem_type
                    
                    if current_problem_type in ["binary_classification", "multiclass_classification"]:
                        # Show class distribution
                        class_dist = st.session_state.builder.y_train.value_counts().sort_index()
                        
                        # Create enhanced visualization for multiclass
                        if current_problem_type == "multiclass_classification":
                            fig = go.Figure()
                            
                            # Use different colors for multiclass
                            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#FFB6C1']
                            colors_extended = colors * (len(class_dist) // len(colors) + 1)
                            
                            fig.add_trace(go.Bar(
                                x=class_dist.index.astype(str),
                                y=class_dist.values,
                                text=class_dist.values,
                                textposition='auto',
                                    marker_color=colors_extended[:len(class_dist)],
                                    name="Class Count"
                                ))
                            
                            # Add percentage annotations
                            total_samples = class_dist.sum()
                            percentages = (class_dist.values / total_samples * 100).round(1)
                            
                            fig.update_layout(
                                title=f"Class Distribution for {target} ({len(class_dist)} classes)",
                                xaxis_title="Class",
                                yaxis_title="Count",
                                showlegend=False,
                                height=300,
                                margin=dict(t=30, b=30, l=30, r=30)
                            )
                            
                            # Add percentage labels
                            for i, (count, pct) in enumerate(zip(class_dist.values, percentages)):
                                fig.add_annotation(
                                    x=str(class_dist.index[i]),
                                    y=count + max(class_dist.values) * 0.05,
                                    text=f"{pct}%",
                                    showarrow=False,
                                    font=dict(size=10, color="gray")
                                )
                        else:
                            # Binary classification visualization
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=class_dist.index.astype(str),
                                y=class_dist.values,
                                text=class_dist.values,
                                textposition='auto',
                                marker_color=['#FF6B6B', '#4ECDC4'][:len(class_dist)]
                        ))
                        fig.update_layout(
                            title=f"Class Distribution for {target}",
                            xaxis_title="Class",
                            yaxis_title="Count",
                            showlegend=False,
                            height=250,
                            margin=dict(t=30, b=30, l=30, r=30)
                        )
                        
                        st.plotly_chart(fig, config={'responsive': True})
                        
                        # Add class balance analysis for classification
                        if len(class_dist) > 1:
                            min_class_count = class_dist.min()
                            max_class_count = class_dist.max()
                            balance_ratio = min_class_count / max_class_count
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Classes", len(class_dist))
                            with col2:
                                st.metric("Balance Ratio", f"{balance_ratio:.3f}")
                            with col3:
                                balance_status = "Balanced" if balance_ratio > 0.7 else ("Moderate" if balance_ratio > 0.3 else "Imbalanced")
                                st.metric("Balance Status", balance_status)
                    else:
                        # Show distribution for regression target
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=st.session_state.builder.y_train,
                            nbinsx=30,
                            marker_color='#2ecc71'
                        ))
                        fig.update_layout(
                            title=f"Distribution of {target}",
                            xaxis_title=target,
                            yaxis_title="Count",
                            showlegend=False,
                            height=250,
                            margin=dict(t=30, b=30, l=30, r=30)
                        )
                        st.plotly_chart(fig, config={'responsive': True})
                        
                        # Add regression target statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean", f"{st.session_state.builder.y_train.mean():.3f}")
                        with col2:
                            st.metric("Std Dev", f"{st.session_state.builder.y_train.std():.3f}")
                        with col3:
                            st.metric("Range", f"{st.session_state.builder.y_train.max() - st.session_state.builder.y_train.min():.3f}")
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close model-card
        
        with metrics_col:
            if st.session_state.builder.model is not None and "metrics" in st.session_state.builder.model:
                metrics = st.session_state.builder.model["metrics"]
                
                # Use session state for problem type detection
                if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type:
                    current_problem_type = st.session_state.problem_type
                else:
                    # Fallback to builder model for backward compatibility
                    current_problem_type = st.session_state.builder.model.get("problem_type", "Unknown")
                
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("#### 📈 Model Performance")
                
                # Format metrics based on problem type
                if current_problem_type in ["binary_classification", "multiclass_classification"]:
                    metrics_to_show = {
                        "Accuracy": metrics.get("accuracy", 0),
                        "Precision": metrics.get("precision", 0),
                        "Recall": metrics.get("recall", 0),
                        "F1 Score": metrics.get("f1", 0)
                    }
                else:  # regression
                    metrics_to_show = {
                        "R² Score": metrics.get("r2", 0),
                        "MAE": metrics.get("mae", 0),
                        "MSE": metrics.get("mse", 0),
                        "RMSE": metrics.get("rmse", 0)
                    }
                
                # Create interactive metrics visualization
                fig = go.Figure()
                
                if current_problem_type in ["binary_classification", "multiclass_classification"]:
                    # Enhanced gauge charts for classification metrics
                    for idx, (metric_name, value) in enumerate(metrics_to_show.items()):
                        fig.add_trace(go.Indicator(
                            mode="gauge+number",
                            value=float(value),
                            title={'text': metric_name, 'font': {'color': '#808080'}},
                            domain={'row': idx // 2, 'column': idx % 2},
                            gauge={
                                'axis': {'range': [0, 1], 'tickfont': {'color': '#808080'}},
                                'steps': [
                                    {'range': [0, 0.4], 'color': "rgba(211, 211, 211, 0.3)"},
                                    {'range': [0.4, 0.7], 'color': "rgba(255, 255, 0, 0.3)"},
                                    {'range': [0.7, 1], 'color': "rgba(144, 238, 144, 0.3)"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 0.7
                                },
                                'bar': {'color': "rgba(50,200,50,0.8)"}
                            },
                            number={'font': {'color': '#808080'}}
                        ))
                    
                    # Update layout for gauge charts
                    fig.update_layout(
                        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
                        height=400,
                        margin=dict(t=30, b=0, l=0, r=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#808080'}
                    )
                    
                    # Display the classification metrics chart
                    st.plotly_chart(fig, config={'responsive': True})
                    
                    # Add multiclass-specific additional metrics if available
                    if current_problem_type == "multiclass_classification":
                        if any(key in metrics for key in ['weighted_precision', 'weighted_recall', 'weighted_f1']):
                            st.markdown("##### Weighted Averages")
                            weighted_col1, weighted_col2, weighted_col3 = st.columns(3)
                            with weighted_col1:
                                if 'weighted_precision' in metrics:
                                    st.metric("Weighted Precision", f"{metrics['weighted_precision']:.4f}")
                            with weighted_col2:
                                if 'weighted_recall' in metrics:
                                    st.metric("Weighted Recall", f"{metrics['weighted_recall']:.4f}")
                            with weighted_col3:
                                if 'weighted_f1' in metrics:
                                    st.metric("Weighted F1", f"{metrics['weighted_f1']:.4f}")
                
                else:  # regression
                    # Create a more suitable visualization for regression metrics
                    # Use a combination of number indicators for R² (which is 0-1 bounded)
                    # and a bar chart for error metrics
                    
                    # First, show R² score as a gauge since it's bounded
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=float(metrics_to_show["R² Score"]),  # Convert to percentage
                        title={'text': "R² Score", 'font': {'color': '#808080'}},
                        domain={'row': 0, 'column': 0},
                        gauge={
                            'axis': {'range': [0, 1], 'tickfont': {'color': '#808080'}},  # Range is now 0-100
                            'steps': [
                                {'range': [0, 0.4], 'color': "rgba(211, 211, 211, 0.3)"},
                                {'range': [0.4, 0.7], 'color': "rgba(255, 255, 0, 0.3)"},
                                {'range': [0.7, 1], 'color': "rgba(144, 238, 144, 0.3)"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.7
                            },
                            'bar': {'color': "rgba(50,200,50,0.8)"}
                        },
                        number={'font': {'color': '#808080'}}
                    ))
                    
                    # Create bar chart for error metrics
                    error_metrics = {k: v for k, v in metrics_to_show.items() if k != "R² Score"}
                    
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=float(metrics_to_show["R² Score"]),  # Convert to percentage
                        title={'text': "R² Score", 'font': {'color': '#808080'}},
                        domain={'row': 0, 'column': 0},
                        gauge={
                            'axis': {'range': [0, 1], 'tickfont': {'color': '#808080'}},  # Range is now 0-100
                            'steps': [
                                {'range': [0, 0.4], 'color': "rgba(211, 211, 211, 0.3)"},
                                {'range': [0.4, 0.7], 'color': "rgba(255, 255, 0, 0.3)"},
                                {'range': [0.7, 1], 'color': "rgba(144, 238, 144, 0.3)"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.7
                            },
                            'bar': {'color': "rgba(50,200,50,0.8)"}
                        },
                        number={'font': {'color': '#808080'}}
                    ))
                    
                    # Update layout for combined visualization
                    fig.update_layout(
                        height=300,
                        margin=dict(t=30, b=30, l=30, r=30),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#808080'},
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, config={'responsive': True})
                    
                    # Display error metrics in a dashboard format
                    error_cols = st.columns(3)
                    for idx, (metric_name, value) in enumerate(error_metrics.items()):
                        with error_cols[idx]:
                            st.metric(
                                label=metric_name,
                                value=f"{value:.4f}",
                                help=f"Lower {metric_name} indicates better model performance"
                            )
                
                # Add interpretation guidelines
                if current_problem_type in ["binary_classification", "multiclass_classification"]:
                    if current_problem_type == "multiclass_classification":
                        st.info("""
                            **Multiclass Metric Interpretation:**
                            - **Accuracy**: Overall correct predictions across all classes
                            - **Precision**: Macro-averaged precision across classes (unweighted)
                            - **Recall**: Macro-averaged recall across classes (unweighted)
                            - **F1 Score**: Macro-averaged F1 score across classes (unweighted)
                            
                            **Note**: Macro averaging treats all classes equally, while weighted averaging accounts for class frequency.
                        """)
                    else:
                        st.info("""
                            **Binary Classification Metric Interpretation:**
                            - **Accuracy**: Overall correct predictions
                            - **Precision**: Accuracy of positive predictions
                            - **Recall**: Ability to find all positive cases
                            - **F1 Score**: Balance between precision and recall
                        """)
                else:
                    st.info("""
                        **Regression Metric Interpretation:**
                        - **R² Score**: Proportion of variance explained
                        - **MAE**: Average absolute prediction error
                        - **MSE**: Average squared prediction error
                        - **RMSE**: Root mean squared error (in target units)
                    """)
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close model-card
                
                # Model Stability Card
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown("#### 🎯 Model Stability")
                
                # Check if model and CV metrics are available
                if not hasattr(st.session_state, 'builder') or not hasattr(st.session_state.builder, 'model'):
                    st.warning("No model available. Please complete the model training stage first.")
                elif not st.session_state.builder.model or 'cv_metrics' not in st.session_state.builder.model:
                    st.warning("No cross-validation metrics available. Please complete the model training with cross-validation.")
                else:
                    cv_metrics = st.session_state.builder.model.get("cv_metrics", {})
                    if not cv_metrics:
                        st.warning("Cross-validation metrics are empty. Please retrain the model with cross-validation.")
                    else:
                        cv_std = cv_metrics.get("std_score", 0)
                        
                        # Create stability gauge with improved visibility
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=float((1 - cv_std) * 100),  # Convert stability score to percentage
                            title={'text': "Model Stability Score"},
                            delta={'reference': 90, 'increasing': {'color': "green"}},  # Updated reference to percentage
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(128,128,128,0.8)"},
                                'bar': {'color': "rgba(50,200,50,0.8)"},
                                'bgcolor': "rgba(128,128,128,0.1)",
                                'borderwidth': 2,
                                'bordercolor': "rgba(128,128,128,0.5)",
                                'steps': [
                                    {'range': [0, 70], 'color': 'rgba(255,0,0,0.2)'},
                                    {'range': [70, 90], 'color': 'rgba(255,255,0,0.2)'},
                                    {'range': [90, 100], 'color': 'rgba(0,255,0,0.2)'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            },
                            number={'suffix': "%", 'font': {'color': '#808080'}}  # Added percentage suffix
                        ))
                        
                        fig.update_layout(
                            height=250,
                            margin=dict(t=40, b=20, l=20, r=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(size=12, color='#808080')
                        )
                        st.plotly_chart(fig, config={'responsive': True})
                        
                        # Add more detailed stability metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            stability_level = "High" if cv_std < 0.05 else "Moderate" if cv_std < 0.1 else "Low"
                            st.metric(
                                "Stability Level",
                                stability_level,
                                delta="Good" if stability_level == "High" else ("Fair" if stability_level == "Moderate" else "Needs Improvement"),
                                delta_color="normal"
                            )
                        with col2:
                            st.metric(
                                "Cross-validation Std",
                                f"{cv_std:.3f}",
                                delta=f"{(1-cv_std)*100:.1f}% consistent",
                                delta_color="normal"
                            )
                        
                        # Add performance range if available
                        if all(key in cv_metrics for key in ["min_score", "mean_score", "max_score"]):
                            st.markdown("##### Performance Range")
                            range_col1, range_col2, range_col3 = st.columns(3)
                            with range_col1:
                                st.metric("Minimum Score", f"{cv_metrics['min_score']:.3f}")
                            with range_col2:
                                st.metric("Mean Score", f"{cv_metrics['mean_score']:.3f}")
                            with range_col3:
                                st.metric("Maximum Score", f"{cv_metrics['max_score']:.3f}")
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close model-card
        
        st.divider()
        # Dataset Information
        if st.session_state.builder.data is not None:
            st.header("📊 Download Training and Test Datasets")
            
            # Add explanation for the datasets
            st.markdown("### About the Datasets")
            
            # Get current problem type for contextual information
            current_problem_type = get_current_problem_type()
            
            if current_problem_type == "multiclass_classification":
                st.markdown("""
                The training and test datasets contain your preprocessed data, split for **multiclass classification** model development:
                """)
            elif current_problem_type == "binary_classification":
                st.markdown("""
                The training and test datasets contain your preprocessed data, split for **binary classification** model development:
                """)
            else:
                st.markdown("""
                The training and test datasets contain your preprocessed data, split for model development:
                """)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                    **Training Data:**
                    - Used to train the model
                    - Contains all preprocessing transformations applied during development
                    - Includes feature engineering, encoding, and binning operations
                    - Ready for immediate use in model training
                    
                    **Test Data:**
                    - Held out for model evaluation
                    - Never seen by the model during training (ensures unbiased evaluation)
                    - Processed to match the same format as the training data
                    - Provides realistic performance estimates
                    """)
            with col2:
                multiclass_info = ""
                if current_problem_type == "multiclass_classification" and hasattr(st.session_state.builder, 'y_train'):
                    n_classes = len(st.session_state.builder.y_train.unique())
                    multiclass_info = f"\n                    - **{n_classes} distinct classes** in the target variable"
                
                st.markdown(f"""  
                    **Each dataset includes:**
                    - All selected features after preprocessing
                    - The target variable for supervised learning{multiclass_info}
                    - Consistent data types and formatting
                    - No missing values (handled during preprocessing)
                    - Feature names exactly as used in model training
                    
                    **Download these datasets to:**
                    - Reproduce the exact model training process
                    - Share reproducible datasets with team members
                    - Create documentation for your ML workflow
                    - Train additional models with the same data split
                    """)
            st.write(" ")
            # Download training and test datasets
            if st.session_state.builder.X_train is not None and st.session_state.builder.X_test is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    create_download_button(
                        pd.concat([st.session_state.builder.X_train, st.session_state.builder.y_train], axis=1),
                        "training_data.csv",
                        "text/csv",
                        help_text="Download the training dataset used to train the model"
                    )
                with col2:
                    create_download_button(
                        pd.concat([st.session_state.builder.X_test, st.session_state.builder.y_test], axis=1),
                        "test_data.csv",
                        "text/csv",
                        help_text="Download the test dataset used to evaluate the model"
                    )
                with col3:
                    st.write(" ")

        st.write("---")
        
        st.markdown("### Model Recreation Script")
        st.markdown("""
        The following python script allows you to recreate the exact model from scratch using the downloaded datasets.
        It includes:
        
        1. **Exact Model Configuration:** Uses the same model type and optimized parameters
        2. **Data Loading:** Loads the training and test datasets you downloaded
        3. **Model Training:** Trains the model from scratch with the exact same settings
        4. **Performance Evaluation:** Evaluates the model with the same metrics used in development
        5. **Visualization:** Creates comprehensive plots and analysis charts
        6. **Error Handling:** Robust error handling and informative messages
        
        **Use this script to:**
        - Reproduce your exact model results
        - Understand the complete training process
        - Modify parameters for experimentation
        - Train on new data with the same configuration
        - Share reproducible ML workflows
        """)
        
        # Add environment setup instructions
        with st.expander("🔧 Python Environment Setup Guide",expanded=False):
            st.markdown("### Setting Up Your Python Environment")
            st.markdown("""
            To run the model recreation script successfully, you'll need to set up a proper Python environment. 
            Here are detailed instructions for different approaches:
            """)
            
            # Create tabs for different setup methods
            setup_tab1, setup_tab2, setup_tab3, setup_tab4 = st.tabs([
                "🐍 Using Conda", 
                "📦 Using pip + venv", 
                "🖥️ System Python", 
                "🔍 Troubleshooting"
            ])
            
            with setup_tab1:
                st.markdown("#### Recommended: Using Conda (Anaconda/Miniconda)")
                st.markdown("""
                Conda is the recommended approach as it handles both Python packages and system dependencies.
                
                **Step 1: Install Conda**
                - Download Anaconda: https://www.anaconda.com/products/distribution
                - Or Miniconda (lighter): https://docs.conda.io/en/latest/miniconda.html
                
                **Step 2: Create a New Environment**
                ```bash
                # Create a new environment with Python 3.9 or 3.10
                conda create -n ml_recreation python=3.10
                
                # Activate the environment
                conda activate ml_recreation
                ```
                
                **Step 3: Install Required Packages**
                ```bash
                # Install core packages
                conda install pandas numpy scikit-learn matplotlib seaborn
                
                # Install additional packages if needed for your model
                conda install -c conda-forge xgboost lightgbm  # If using XGBoost or LightGBM
                ```
                
                **Step 4: Run Your Script**
                ```bash
                # Navigate to your script directory
                cd /path/to/your/script
                
                # Run the recreation script
                python model_recreation_script.py
                ```
                """)
            
            with setup_tab2:
                st.markdown("#### Using pip with Virtual Environment")
                st.markdown("""
                This approach uses Python's built-in virtual environment capabilities.
                
                **Step 1: Check Python Installation**
                ```bash
                # Check if Python is installed (should be 3.8+)
                python --version
                # or
                python3 --version
                ```
                
                **Step 2: Create Virtual Environment**
                ```bash
                # Create virtual environment
                python -m venv ml_recreation_env
                
                # Activate environment (Windows)
                ml_recreation_env\\Scripts\\activate
                
                # Activate environment (Mac/Linux)
                source ml_recreation_env/bin/activate
                ```
                
                **Step 3: Install Packages**
                ```bash
                # Upgrade pip first
                pip install --upgrade pip
                
                # Install core packages
                pip install pandas numpy scikit-learn matplotlib seaborn
                
                # Install additional packages if needed
                pip install xgboost lightgbm  # If your model uses these
                ```
                
                **Step 4: Run Your Script**
                ```bash
                # Ensure environment is activated
                python model_recreation_script.py
                ```
                """)
            
            with setup_tab3:
                st.markdown("#### Using System Python (Not Recommended)")
                st.markdown("""
                ⚠️ **Warning**: Installing packages globally can cause conflicts with other projects.
                
                **Step 1: Install Packages Globally**
                ```bash
                # Install core packages
                pip install pandas numpy scikit-learn matplotlib seaborn
                
                # Install additional packages if needed
                pip install xgboost lightgbm  # If your model uses these
                ```
                
                **Step 2: Run Script**
                ```bash
                python model_recreation_script.py
                ```
                
                **Better Alternative**: Use `pip install --user` to install packages for your user only:
                ```bash
                pip install --user pandas numpy scikit-learn matplotlib seaborn
                ```
                """)
            
            with setup_tab4:
                st.markdown("#### Common Issues and Solutions")
                st.markdown("""
                **Problem: "Python not found" or "pip not found"**
                - **Solution**: Install Python from https://python.org or use a package manager
                - **Windows**: Add Python to PATH during installation
                - **Mac**: Use Homebrew: `brew install python`
                - **Linux**: Use package manager: `sudo apt install python3 python3-pip`
                
                **Problem: Permission denied when installing packages**
                - **Solution**: Use virtual environments (recommended) or `--user` flag
                - **Don't use**: `sudo pip install` (can break system packages)
                
                **Problem: Package version conflicts**
                - **Solution**: Use virtual environments to isolate dependencies
                - **Check versions**: `pip list` or `conda list`
                
                **Problem: "ModuleNotFoundError" when running script**
                - **Solution**: Ensure environment is activated and packages are installed
                - **Check**: `python -c "import pandas; print('OK')"`
                
                **Problem: XGBoost or LightGBM installation fails**
                - **Windows**: May need Visual Studio Build Tools
                - **Mac**: May need Xcode command line tools: `xcode-select --install`
                - **Alternative**: Try conda installation instead of pip
                
                **Problem: Plots not displaying**
                - **Solution**: Install GUI backend: `pip install tkinter` (usually included)
                - **Headless servers**: Use `matplotlib.use('Agg')` before importing pyplot
                
                **Problem: Script runs but produces different results**
                - **Check**: Package versions match recommendations
                - **Check**: Random seeds are set (should be handled in script)
                - **Check**: Data files are identical to downloaded versions
                """)
        
        # Generate and display the recreation code
        recreation_code = generate_model_recreation_code()
        
        if recreation_code:
            with st.expander("📜 View Model Recreation Script",expanded=False):
                st.code(recreation_code, language="python")
            
            # Add download button for the recreation script
            st.download_button(
                label="📥 Download Model Recreation Script",
                data=recreation_code,
                file_name="model_recreation_script.py",
                mime="text/x-python",
                help="Download the complete script to recreate your model from scratch"
            )
            
            st.info("""
            **📋 Quick Start Instructions:**
            1. **Set up Python environment** (see 🔧 **Python Environment Setup Guide** above for detailed instructions)
            2. Download this script (`model_recreation_script.py`)
            3. Download your training and test datasets (above)
            4. Place all files in the same directory
            5. Activate your Python environment
            6. Run: `python model_recreation_script.py`
            
            **💡 Need help with setup?** Expand the **Python Environment Setup Guide** above for step-by-step instructions
            including Conda, pip, virtual environments, and troubleshooting common issues.
            
            The script will recreate your exact model and provide the same evaluation results.
            """)
        else:
            st.error("Unable to generate recreation code. Model information may be incomplete.")
    
    st.divider()
    # Development Journey Map
    st.header("🗺️ Development Journey")
    
    # Add explanation for the journey section
    st.markdown("### About Your Development Journey")
    st.markdown("""
    The development journey map provides an interactive visualization of your ML model development process:
    
    **Journey Map Features:**
    - **Visual Timeline:** See your decisions and progress through each development stage
    - **Interactive Nodes:** Click on nodes to see detailed information about each step
    - **Stage Separation:** Clear visual distinction between different development phases
    - **Export Options:** Download your journey map for documentation and sharing
    
    **Development Story:**
    - **Narrative View:** Read your development process as a story
    - **Technical Details:** Access the technical information for each decision point
    - **Filter Options:** Focus on specific stages or time periods
    - **Export Story:** Download a complete narrative of your development process
    
    **Analytics Dashboard:**
    - **Progress Metrics:** Track your overall development progress
    - **Stage Analysis:** See time spent and decisions made in each stage
    - **Performance Insights:** Understand your development patterns
    
    **This journey map helps you:**
    - Document your development process for future reference
    - Share your methodology with team members or stakeholders
    - Learn from your decision-making patterns
    - Create reproducible development workflows
    - Demonstrate the rigor and thoughtfulness of your ML development process
    """)
    
    # Render the journey viewer component
    render_journey_viewer(expanded=True)
    
    st.divider()
    # Add log viewer
    render_log_viewer()
    
    # Add a back button at the bottom of the page
    st.write("---")  # Add a visual separator
    if st.button("⬅️ Back to Model Explanation",type="primary"):
        st.switch_page("pages/8_Model_Explanation.py")

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