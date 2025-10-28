import streamlit as st
import pandas as pd
import numpy as np
from Builder import Builder, ModelStage
from app import create_sidebar_navigation
from utils.logging.journey_viewer import render_journey_viewer
from utils.logging.log_viewer import render_log_viewer
from components.model_evaluation.performance_metrics import display_classification_report, display_regression_report
from components.model_evaluation.visualisations import display_classification_visualisations, display_regression_visualisations
from components.model_evaluation.sample_predictions import display_sample_predictions
from components.model_evaluation.model_health import display_model_improvements
from utils.data_exploration_component import DataExplorationComponent
from components.data_preprocessing.categorical_encoding import CategoricalEncodingComponent

def display_dataset_overview():
    st.header("📊 Dataset Overview")
    
    overview_container = st.container()
    with overview_container:
        col1, col2 = st.columns([2, 2], vertical_alignment="bottom")
        
        with col1:
            st.markdown("#### 🔵 Training Dataset")
            if hasattr(st.session_state.builder, 'X_train'):
                train_samples = len(st.session_state.builder.X_train)
                train_features = st.session_state.builder.X_train.shape[1]
                
                # Create metrics for training data
                st.metric("Number of samples", f"{train_samples:,}")
                st.metric("Number of features", train_features)
                
                # Show class distribution for classification
                if (st.session_state.builder.model and 
                    st.session_state.builder.model.get("problem_type") in ["classification", "binary_classification", "multiclass_classification"]):
                    train_dist = st.session_state.builder.y_train.value_counts()
                    st.write("**Class Distribution:**")
                    for cls, count in train_dist.items():
                        percentage = count/len(st.session_state.builder.y_train) * 100
                        st.progress(percentage/100)
                        st.write(f"{cls}: {count:,} ({percentage:.1f}%)")
        
        with col2:
            st.markdown("#### 🔴 Test Dataset")
            if hasattr(st.session_state.builder, 'X_test'):
                test_samples = len(st.session_state.builder.X_test)
                test_features = st.session_state.builder.X_test.shape[1]
                
                # Create metrics for test data
                st.metric("Number of samples", f"{test_samples:,}")
                st.metric("Number of features", test_features)
                
                # Show class distribution for classification
                if (st.session_state.builder.model and 
                    st.session_state.builder.model.get("problem_type") in ["classification", "binary_classification", "multiclass_classification"]):
                    test_dist = st.session_state.builder.y_test.value_counts()
                    st.write("**Class Distribution:**")
                    for cls, count in test_dist.items():
                        percentage = count/len(st.session_state.builder.y_test) * 100
                        st.progress(percentage/100)
                        st.write(f"{cls}: {count:,} ({percentage:.1f}%)")

def main():
        
    # Add consistent navigation
    create_sidebar_navigation()
    
    # Initialize session state if needed
    if 'builder' not in st.session_state:
        st.session_state.builder = Builder()
        st.session_state.logger.log_stage_transition("START", "MODEL_EVALUATION")
    
    # Set current stage to MODEL_EVALUATION
    st.session_state.builder.current_stage = ModelStage.MODEL_EVALUATION
    
    # Log page state
    page_state = {
        "stage": "MODEL_EVALUATION",
        "model_exists": st.session_state.builder.model is not None,
        "training_complete": st.session_state.builder.stage_completion[ModelStage.MODEL_TRAINING]
    }
    st.session_state.logger.log_page_state("Model_Evaluation", page_state)
    
    # Get and render stage info
    stage_info = st.session_state.builder.get_current_stage_info()
    
    st.header(stage_info["title"])
    st.write(stage_info["description"])
    
    with st.expander("Functionality"):
        for req in stage_info["requirements"]:
            if isinstance(req, dict):
                st.markdown(f"**{req['title']}**")
                for item in req['items']:
                    st.markdown(f"• {item}")
            else:
                if isinstance(req, str) and req.startswith("**"):
                    st.markdown(req)
                else:
                    st.markdown(f"• {req}")
            
    with st.expander("Ethical Considerations"):
        for consideration in stage_info["ethical_considerations"]:
            if isinstance(consideration, dict):
                st.markdown(f"**{consideration['title']}**")
                for item in consideration['items']:
                    st.markdown(f"• {item}")
            else:
                if isinstance(consideration, str) and consideration.startswith("**"):
                    st.markdown(consideration)
                else:
                    st.markdown(f"• {consideration}")
    
    # Display dataset overview
    display_dataset_overview()
    
    # Add selected model information section
    st.subheader("📋 Model Training Information")

    # Get comprehensive model information
    optimization_method = st.session_state.builder.model.get('optimisation_method', None)
    selection_type = st.session_state.builder.model.get('selection_type', 'mean_score')
    active_model = st.session_state.builder.model.get('active_model', None)
    active_params = st.session_state.builder.model.get('active_params', None)

    # Get problem type first
    problem_type = st.session_state.builder.model.get("problem_type", "unknown")

    # Get calibration information
    is_calibrated = st.session_state.builder.model.get('is_calibrated', False)
    calibration_method = st.session_state.builder.model.get('calibration_method', 'isotonic')
    calibration_cv_folds = st.session_state.builder.model.get('calibration_cv_folds', 5)

    # Get threshold optimization information
    threshold_optimized = st.session_state.builder.model.get('threshold_optimized', False)
    optimal_threshold = st.session_state.builder.model.get('optimal_threshold', 0.5)
    threshold_is_binary = st.session_state.builder.model.get('threshold_is_binary', True)
    threshold_criterion = st.session_state.builder.model.get('threshold_criterion', 'F1 Score')

    # Training Summary Dashboard
    st.markdown("### 📊 Training Summary")
    summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(5)

    with summary_col1:
        st.metric("Model Type", st.session_state.builder.model["type"])

    with summary_col2:
        opt_display = {
            'optuna': 'Optuna (Bayesian)',
            'random_search': 'Random Search',
            None: 'Standard'
        }.get(optimization_method, 'Unknown')
        st.metric("Optimization", opt_display)

    with summary_col3:
        if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
            enhancements = []
            if is_calibrated:
                enhancements.append("Calibrated")
            if threshold_optimized:
                enhancements.append("Threshold Opt.")
            enhancement_text = " + ".join(enhancements) if enhancements else "None"
            st.metric("Enhancements", enhancement_text)
        else:
            st.metric("Problem Type", "Regression")
    with summary_col4:
        if hasattr(st.session_state.builder, 'target_column'):
            target_col = st.session_state.builder.target_column
            if target_col in st.session_state.builder.data.columns:
                st.metric("Target Column", target_col)
            else:
                st.metric("Target Column", "N/A")
    with summary_col5:
        # Get CV score from training results first, then fallback to builder.model
        cv_score = None

        # Try to get from training results (preferred source)
        if hasattr(st.session_state, 'training_results') and st.session_state.training_results:
            training_info = st.session_state.training_results.get('info', {})
            if 'cv_metrics' in training_info:
                cv_score = training_info['cv_metrics'].get('mean_score', 0)
            elif selection_type == 'adjusted_score' and 'adjusted_cv_metrics' in training_info:
                cv_score = training_info['adjusted_cv_metrics'].get('mean_score', 0)

        # Fallback to builder.model if training_results not available
        if cv_score is None:
            if optimization_method == 'optuna' and 'cv_metrics' in st.session_state.builder.model:
                cv_score = st.session_state.builder.model['cv_metrics'].get('mean_score', 0)
            elif optimization_method == 'random_search':
                if selection_type == 'adjusted_score' and 'adjusted_cv_metrics' in st.session_state.builder.model:
                    cv_score = st.session_state.builder.model['adjusted_cv_metrics'].get('mean_score', 0)
                elif 'cv_metrics' in st.session_state.builder.model:
                    cv_score = st.session_state.builder.model['cv_metrics'].get('mean_score', 0)
            elif 'cv_metrics' in st.session_state.builder.model:
                cv_score = st.session_state.builder.model['cv_metrics'].get('mean_score', 0)

        if cv_score is not None:
            # Add context based on problem type - use correct training metrics
            if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                if problem_type == "multiclass_classification":
                    metric_name = "CV F1 (Macro)"
                else:
                    metric_name = "CV F1 Score"
                cv_display = f"{cv_score:.3f}" if cv_score <= 1.0 else f"{cv_score:.4f}"
            else:  # regression
                metric_name = "CV R² Score"
                cv_display = f"{cv_score:.4f}"
            st.metric(metric_name, cv_display)
        else:
            st.metric("CV Score", "N/A")

    # Display model optimization method and selection information
    if optimization_method == 'optuna':
        st.info(f"""
            **🎯 Model Selected: {st.session_state.builder.model["type"]} - Optimized with Optuna**

            This model was optimized using Optuna's advanced Bayesian optimisation techniques,
            which efficiently searched the parameter space to find the best configuration.
        """)

        # Move training metrics into expander
        with st.expander("📊 Model Training Metrics Details", expanded=False):
            # Add context based on problem type with post-training clarification
            if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                if problem_type == "multiclass_classification":
                    st.markdown("""
                        **📊 Model Training Metrics (Multiclass Classification)**
                        - Mean Score: Average F1 (Macro) score across all cross-validation folds
                        - Standard Deviation: Variability in F1 score (lower = more consistent)
                        - Best Score: The best F1 (Macro) score achieved during optimization

                        ℹ️ **Note**: These metrics reflect the original model's training performance before any post-training enhancements (calibration/threshold optimization).
                    """)
                else:
                    st.markdown("""
                        **📊 Model Training Metrics (Binary Classification)**
                        - Mean Score: Average F1 score across all cross-validation folds
                        - Standard Deviation: Variability in F1 score (lower = more consistent)
                        - Best Score: The best F1 score achieved during optimization

                        ℹ️ **Note**: These metrics reflect the original model's training performance before any post-training enhancements (calibration/threshold optimization).
                    """)
            else:
                st.markdown("""
                    **📊 Model Training Metrics (Regression)**
                    - Mean Score: Average R² score across all cross-validation folds
                    - Standard Deviation: Variability in R² score (lower = more consistent)
                    - Best Score: The best R² score achieved during optimization
                """)

            # Show metrics for the Optuna model - try training_results first
            metrics = None
            best_score = None

            # Get from training results (preferred)
            if hasattr(st.session_state, 'training_results') and st.session_state.training_results:
                training_info = st.session_state.training_results.get('info', {})
                if 'cv_metrics' in training_info:
                    metrics = training_info['cv_metrics']
                    best_score = training_info.get('best_score', metrics.get('mean_score', 0))

            # Fallback to builder.model
            if metrics is None and 'cv_metrics' in st.session_state.builder.model:
                metrics = st.session_state.builder.model['cv_metrics']
                best_score = st.session_state.builder.model.get('best_score', metrics.get('mean_score', 0))

            if metrics is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Score", f"{metrics.get('mean_score', 0):.4f}")
                with col2:
                    st.metric("Standard Deviation", f"{metrics.get('std_score', 0):.4f}")
                with col3:
                    st.metric("Best Score", f"{best_score:.4f}")
            else:
                st.warning("Training metrics not available.")
    
    elif optimization_method == 'random_search':
        if selection_type == 'adjusted_score':
            st.info(f"""
                **🌟 Model Selected: {st.session_state.builder.model["type"]} - Optimized for Stability (Random Search)**

                This model was selected because it provides the best balance between performance and stability
                using random search optimization. It may not have the highest average score, but it should
                perform more consistently across different data.
            """)

            # Move training metrics into expander
            with st.expander("📊 Model Training Metrics Details", expanded=False):
                # Add explanatory text
                if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                    if problem_type == "multiclass_classification":
                        st.markdown("""
                            **Model Training Metrics (Multiclass Classification)**
                            - Mean Score: Average F1 (Macro) score across all cross-validation folds
                            - Standard Deviation: Variability in F1 score (lower = more consistent)
                            - Adjusted Score: Mean score minus standard deviation (balances performance with consistency)

                            ℹ️ **Note**: These metrics reflect the original model's training performance before any post-training enhancements.
                        """)
                    else:
                        st.markdown("""
                            **Model Training Metrics (Binary Classification)**
                            - Mean Score: Average F1 score across all cross-validation folds
                            - Standard Deviation: Variability in F1 score (lower = more consistent)
                            - Adjusted Score: Mean score minus standard deviation (balances performance with consistency)

                            ℹ️ **Note**: These metrics reflect the original model's training performance before any post-training enhancements.
                        """)
                else:
                    st.markdown("""
                        **Model Training Metrics (Regression)**
                        - Mean Score: Average R² score across all cross-validation folds
                        - Standard Deviation: Variability in R² score (lower = more consistent)
                        - Adjusted Score: Mean score minus standard deviation (balances performance with consistency)
                    """)

                # Show metrics for this model if available - try training_results first
                metrics = None

                # Get from training results (preferred)
                if hasattr(st.session_state, 'training_results') and st.session_state.training_results:
                    training_info = st.session_state.training_results.get('info', {})
                    if 'adjusted_cv_metrics' in training_info:
                        metrics = training_info['adjusted_cv_metrics']

                # Fallback to builder.model
                if metrics is None and 'adjusted_cv_metrics' in st.session_state.builder.model:
                    metrics = st.session_state.builder.model['adjusted_cv_metrics']

                if metrics is not None:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        mean_score = metrics.get('mean_score', 0)
                        if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                            score_display = f"{mean_score:.3f}" if mean_score <= 1.0 else f"{mean_score:.4f}"
                            if problem_type == "multiclass_classification":
                                st.metric("Mean F1 (Macro)", score_display)
                            else:
                                st.metric("Mean F1 Score", score_display)
                        else:
                            st.metric("Mean R² Score", f"{mean_score:.4f}")
                    with col2:
                        std_score = metrics.get('std_score', 0)
                        st.metric("Standard Deviation", f"{std_score:.4f}")
                    with col3:
                        if 'adjusted_score' in metrics:
                            adj_score = metrics.get('adjusted_score', 0)
                            if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                                adj_display = f"{adj_score:.3f}" if adj_score <= 1.0 else f"{adj_score:.4f}"
                                st.metric("Adjusted Score", adj_display)
                            else:
                                st.metric("Adjusted Score", f"{adj_score:.4f}")
                        else:
                            # Calculate it if not already present
                            adjusted = metrics.get('mean_score', 0) - metrics.get('std_score', 0)
                            if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                                adj_display = f"{adjusted:.3f}" if adjusted <= 1.0 else f"{adjusted:.4f}"
                                st.metric("Adjusted Score", adj_display)
                            else:
                                st.metric("Adjusted Score", f"{adjusted:.4f}")
                else:
                    st.warning("Adjusted training metrics not available.")
        else:
            st.info(f"""
                **✅ Model Selected: {st.session_state.builder.model["type"]} - Optimized for Performance (Random Search)**

                This model was selected because it has the highest average performance score
                across all cross-validation folds using random search optimization.
            """)

            # Move training metrics into expander
            with st.expander("📊 Model Training Metrics Details", expanded=False):
                # Add explanatory text
                if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                    if problem_type == "multiclass_classification":
                        st.markdown("""
                            **Model Training Metrics (Multiclass Classification)**
                            - Mean Score: Average F1 (Macro) score across all cross-validation folds
                            - Standard Deviation: Variability in F1 score (lower = more consistent)
                            - Adjusted Score: Mean score minus standard deviation (balances performance with consistency)

                            ℹ️ **Note**: These metrics reflect the original model's training performance before any post-training enhancements.
                        """)
                    else:
                        st.markdown("""
                            **Model Training Metrics (Binary Classification)**
                            - Mean Score: Average F1 score across all cross-validation folds
                            - Standard Deviation: Variability in F1 score (lower = more consistent)
                            - Adjusted Score: Mean score minus standard deviation (balances performance with consistency)

                            ℹ️ **Note**: These metrics reflect the original model's training performance before any post-training enhancements.
                        """)
                else:
                    st.markdown("""
                        **Model Training Metrics (Regression)**
                        - Mean Score: Average R² score across all cross-validation folds
                        - Standard Deviation: Variability in R² score (lower = more consistent)
                        - Adjusted Score: Mean score minus standard deviation (balances performance with consistency)
                    """)

                # Show metrics for this model - try training_results first
                metrics = None

                # Get from training results (preferred)
                if hasattr(st.session_state, 'training_results') and st.session_state.training_results:
                    training_info = st.session_state.training_results.get('info', {})
                    if 'cv_metrics' in training_info:
                        metrics = training_info['cv_metrics']

                # Fallback to builder.model
                if metrics is None and 'cv_metrics' in st.session_state.builder.model:
                    metrics = st.session_state.builder.model['cv_metrics']

                if metrics is not None:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        mean_score = metrics.get('mean_score', 0)
                        if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                            score_display = f"{mean_score:.3f}" if mean_score <= 1.0 else f"{mean_score:.4f}"
                            if problem_type == "multiclass_classification":
                                st.metric("Mean F1 (Macro)", score_display)
                            else:
                                st.metric("Mean F1 Score", score_display)
                        else:
                            st.metric("Mean R² Score", f"{mean_score:.4f}")
                    with col2:
                        std_score = metrics.get('std_score', 0)
                        st.metric("Standard Deviation", f"{std_score:.4f}")
                    with col3:
                        # Calculate adjusted score
                        adjusted = metrics.get('mean_score', 0) - metrics.get('std_score', 0)
                        if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                            adj_display = f"{adjusted:.3f}" if adjusted <= 1.0 else f"{adjusted:.4f}"
                            st.metric("Adjusted Score", adj_display)
                        else:
                            st.metric("Adjusted Score", f"{adjusted:.4f}")
                else:
                    st.warning("Training metrics not available.")
    else:
        st.warning("""
            No optimization method information available. This model was likely trained
            before the optimization method tracking was implemented.
        """)

    if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
        # Post-Training Enhancements Section
        st.markdown("### 🚀 Post-Training Enhancements")

        enhancement_col1, enhancement_col2 = st.columns(2)
        with enhancement_col1:
            # Calibration Section
            st.markdown("#### 🎯 Model Calibration")
            if is_calibrated:
                st.success(f"""
                    **✅ Calibration Applied**

                    - **Method**: {calibration_method.title()}
                    - **CV Folds**: {calibration_cv_folds}
                    - **Purpose**: Improved probability reliability for decision-making
                """)

                if calibration_method == 'isotonic':
                    st.info("**Isotonic Regression** - More flexible, better for larger datasets")
                else:
                    st.info("**Platt Scaling** - Fits a sigmoid curve, good for small datasets")

            else:
                st.info("**❌ No Calibration Applied** - Model uses raw probability outputs")

        with enhancement_col2:
            # Threshold Optimization Section
            st.markdown("#### ⚖️ Threshold Optimization")
            if threshold_optimized:
                threshold_type = "Decision" if threshold_is_binary else "Confidence"
                st.success(f"""
                    **✅ Threshold Optimization Applied**

                    - **Type**: {threshold_type} Threshold
                    - **Optimal Value**: {optimal_threshold:.3f}
                    - **Criterion**: {threshold_criterion}
                """)

                if threshold_is_binary:
                    st.info("**Binary Classification** - Optimized decision threshold for classification boundary")
                else:
                    st.info("**Multiclass Classification** - Optimized confidence threshold for prediction reliability")

            else:
                st.info("**❌ No Threshold Optimization** - Using default 0.5 threshold")
    
    # Display selected hyperparameters
    with st.expander("🔍 Model Hyperparameters", expanded=False):
        st.write("Model Type: ", st.session_state.builder.model["type"])
        
        # Get the model instance to show all parameters
        model_instance = st.session_state.builder.model.get("model")
        
        if model_instance is not None:
            st.write("### Full Model Parameters")
            
            # Get all parameters from the model instance
            try:
                all_params = model_instance.get_params()
                
                # Filter out None/NULL parameters
                filtered_params = {k: v for k, v in all_params.items() if v is not None}
                
                # Get tuned parameters - try active_params first, then best_params
                tuned_param_values = {}
                if active_params is not None and active_params:
                    tuned_param_values = {k: v for k, v in active_params.items() if v is not None}
                elif 'best_params' in st.session_state.builder.model and st.session_state.builder.model['best_params']:
                    tuned_param_values = {k: v for k, v in st.session_state.builder.model['best_params'].items() if v is not None}

                # Get default parameters by creating a fresh instance of the same model type
                default_params = {}
                try:
                    model_type = type(model_instance)
                    # CatBoost models require verbose parameter during initialization
                    if 'CatBoost' in model_type.__name__:
                        fresh_model = model_type(verbose=False)
                    else:
                        fresh_model = model_type()
                    default_params = fresh_model.get_params()
                    # Filter out None values from defaults
                    default_params = {k: v for k, v in default_params.items() if v is not None}
                except Exception as e:
                    st.warning(f"Could not retrieve default parameters: {str(e)}")
                    default_params = {}

                # Create two columns to show tuned vs default parameters
                col1, col2 = st.columns(2)

                with col1:
                    st.write("#### 🎯 Tuned Parameters")
                    if tuned_param_values:
                        st.json(tuned_param_values)
                    else:
                        st.info("No parameters were tuned for this model.")

                with col2:
                    st.write("#### ⚙️ Default Parameters")
                    if default_params:
                        st.json(default_params)
                    else:
                        st.info("Could not retrieve default parameters.")
                
                # Show complete parameter list in a regular section (not nested expander)
                #st.write("---")
                #st.write("#### 📋 Complete Parameter List")
                #st.write("All model parameters (tuned + default, excluding None values):")
                #st.json(filtered_params)
                
                # Add parameter count summary
                st.write(f"**Total Parameters (non-None):** {len(filtered_params)}")
                st.write(f"**Tuned Parameters:** {len(tuned_param_values)}")
                st.write(f"**Default Parameters:** {len(default_params)}")
                if len(all_params) > len(filtered_params):
                    st.write(f"**Filtered out (None values):** {len(all_params) - len(filtered_params)}")
                    
            except Exception as e:
                st.error(f"Error retrieving model parameters: {str(e)}")
                
                # Fallback to showing just the tuned parameters
                st.write("### Tuned Parameters Only")
                if active_params is not None:
                    st.write("Current Active Parameters:")
                    # Filter out None values from active params too
                    filtered_active = {k: v for k, v in active_params.items() if v is not None}
                    st.json(filtered_active)
                elif 'best_params' in st.session_state.builder.model:
                    params = st.session_state.builder.model['best_params']
                    # Filter out None values from best params too
                    filtered_best = {k: v for k, v in params.items() if v is not None}
                    st.write("Best Parameters Found:")
                    st.json(filtered_best)
                else:
                    st.write("No hyperparameter information available.")
        else:
            # Fallback if model instance is not available
            st.write("### Tuned Parameters Only")
            if active_params is not None:
                st.write("Current Active Parameters:")
                # Filter out None values from active params
                filtered_active = {k: v for k, v in active_params.items() if v is not None}
                st.json(filtered_active)
            elif 'best_params' in st.session_state.builder.model:
                params = st.session_state.builder.model['best_params']
                # Filter out None values from best params
                filtered_best = {k: v for k, v in params.items() if v is not None}
                st.write("Best Parameters Found:")
                st.json(filtered_best)
            else:
                st.write("No hyperparameter information available.")
    
    if not st.session_state.builder.stage_completion[ModelStage.MODEL_TRAINING]:
        error_details = {
            "stage": "MODEL_EVALUATION",
            "error_type": "Prerequisites Not Met",
            "missing_stage": "MODEL_TRAINING",
            "stage_completion": st.session_state.builder.stage_completion,
            "timestamp": str(pd.Timestamp.now())
        }
        st.session_state.logger.log_error("Model Evaluation Prerequisites Not Met", error_details)
        st.error("⚠️ Please complete model training first")
        
        if st.button("Return to Model Training"):
            st.session_state.logger.log_user_action(
                "Navigation",
                {
                    "from": "MODEL_EVALUATION",
                    "to": "MODEL_TRAINING",
                    "reason": "Training incomplete",
                    "timestamp": str(pd.Timestamp.now())
                }
            )
            st.session_state.builder.current_stage = ModelStage.MODEL_TRAINING
            st.rerun()
        return
    
    if st.session_state.builder.model is None:
        error_details = {
            "stage": "MODEL_EVALUATION",
            "error_type": "No Model Found",
            "stage_status": st.session_state.builder.stage_completion,
            "timestamp": str(pd.Timestamp.now())
        }
        st.session_state.logger.log_error("Model Evaluation Failed - No Model", error_details)
        st.error("❌ No model found. Please train a model first.")
        return

    # ===== MODEL CHANGE DETECTION: Detect if model has changed and clear caches =====
    def get_model_signature():
        """Create a unique signature for the current model configuration."""
        model = st.session_state.builder.model
        return {
            'model_type': model.get('type'),
            'hyperparameters': str(model.get('hyperparameters', {})),
            'is_calibrated': model.get('is_calibrated', False),
            'calibration_method': model.get('calibration_method', ''),
            'threshold_optimized': model.get('threshold_optimized', False),
            'optimal_threshold': model.get('optimal_threshold', 0.5),
            'train_shape': st.session_state.builder.X_train.shape,
            'test_shape': st.session_state.builder.X_test.shape,
            'target_column': st.session_state.builder.target_column,
            # Use a sample of predictions as a fingerprint (faster than hashing entire model)
            'prediction_sample': str(model['model'].predict(st.session_state.builder.X_test.iloc[:5].values) if len(st.session_state.builder.X_test) >= 5 else None)
        }

    current_signature = get_model_signature()

    # Check if model has changed
    if 'last_model_signature' in st.session_state:
        if st.session_state.last_model_signature != current_signature:
            # Model has changed - clear all caches and reset state
            st.cache_data.clear()

            # Reset evaluation page state
            if 'viz_cache_warmed' in st.session_state:
                del st.session_state.viz_cache_warmed
            if 'selected_evaluation_tab' in st.session_state:
                st.session_state.selected_evaluation_tab = "📊 Performance Metrics"
            if 'sample_indices' in st.session_state:
                del st.session_state.sample_indices
            if 'last_sample_size' in st.session_state:
                del st.session_state.last_sample_size

            # Update signature
            st.session_state.last_model_signature = current_signature

            # Log the model change
            st.session_state.logger.log_user_action(
                "Model Change Detected",
                {
                    "action": "Cache Cleared",
                    "previous_model": st.session_state.get('last_model_signature', {}).get('model_type', 'Unknown'),
                    "new_model": current_signature['model_type'],
                    "timestamp": str(pd.Timestamp.now())
                }
            )

            # Show user notification
            st.info("🔄 Model change detected. Caches cleared and page reset.")
            st.rerun()
    else:
        # First time on this page - store the signature
        st.session_state.last_model_signature = current_signature

    # Log evaluation start with enhanced details
    st.session_state.logger.log_calculation(
        "Model Evaluation Started",
        {
            "model_type": st.session_state.builder.model["type"],
            "problem_type": problem_type,
            "train_samples": len(st.session_state.builder.X_train),
            "test_samples": len(st.session_state.builder.X_test),
            "num_features": st.session_state.builder.X_train.shape[1],
            "feature_names": list(st.session_state.builder.X_train.columns),
            "target_variable": st.session_state.builder.target_column
        }
    )
    
    with st.spinner("📊 Evaluating model performance..."):
        result = st.session_state.builder.evaluate_model()

        if result["success"]:
            # ===== PERFORMANCE: Pre-warm visualization cache for instant tab navigation =====
            if 'viz_cache_warmed' not in st.session_state:
                with st.spinner("🔥 Pre-loading visualizations for faster navigation..."):
                    try:
                        from components.model_evaluation.visualisations import get_data_hash, get_cached_predictions, get_cached_probabilities
                        from components.model_evaluation.evaluation_utils.eval_visualization_utils import (
                            create_classification_error_by_confidence_plot,
                            create_classification_confusion_by_features_plot
                        )

                        # Get data and model
                        X_test = st.session_state.builder.X_test
                        y_test = st.session_state.builder.y_test
                        model_instance = st.session_state.builder.model.get("active_model") or st.session_state.builder.model["model"]

                        # Get cached predictions (these are already cached from Phase 1)
                        X_test_hash = get_data_hash(X_test)
                        y_pred = get_cached_predictions(model_instance, X_test, X_test_hash)
                        y_prob = get_cached_probabilities(model_instance, X_test, X_test_hash,
                                                           hasattr(model_instance, 'predict_proba'))

                        # Pre-compute the 2 heaviest visualizations to populate cache
                        if y_prob is not None:
                            # These function calls trigger cache population
                            # Results are discarded here but will be instantly retrieved later
                            try:
                                _ = create_classification_error_by_confidence_plot(y_test, y_pred, y_prob)
                            except Exception as e:
                                print(f"Could not pre-warm error confidence plot: {e}")

                            try:
                                _ = create_classification_confusion_by_features_plot(y_test, y_pred, X_test, model_instance)
                            except Exception as e:
                                print(f"Could not pre-warm confusion by features plot: {e}")

                        st.session_state.viz_cache_warmed = True
                    except Exception as e:
                        print(f"Cache pre-warming encountered an error: {e}")
                        # Continue anyway - cache will populate on-demand
                        st.session_state.viz_cache_warmed = True

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                # ===== PERFORMANCE: Initialize selected tab in session state for lazy loading =====
                if 'selected_evaluation_tab' not in st.session_state:
                    st.session_state.selected_evaluation_tab = "📊 Performance Metrics"

                # Create main sections using pills
                selected_tab = st.pills(label="Select Analysis:", options=[
                    "📊 Performance Metrics",
                    "📈 Visualisations",
                    "📋 Sample Predictions",
                    "🚀 Model Health and Recommendations"
                ], default=st.session_state.selected_evaluation_tab, key="evaluation_tab_selector")

                # Update session state if tab changed (triggers rerun for true lazy loading)
                if selected_tab != st.session_state.selected_evaluation_tab:
                    st.session_state.selected_evaluation_tab = selected_tab
                    st.rerun()

                # Log user interface state
                st.session_state.logger.log_page_state(
                    "Model_Evaluation_Tabs",
                    {
                        "available_tabs": ["Performance Metrics", "Visualisations", "Sample Predictions", "Model Health and Recommendations"],
                        "problem_type": problem_type,
                        "current_tab": selected_tab
                    }
                )
                st.write(" ")

                def log_tab_switch(tab_name: str):
                    st.session_state.logger.log_user_action(
                        "Tab Switch",
                        {
                            "page": "Model_Evaluation",
                            "selected_tab": tab_name,
                            "timestamp": str(pd.Timestamp.now())
                        }
                    )
            with col2:
                st.markdown("""##### 📊 View Test Data Exploration""")
                @st.dialog(title="Testing Data Exploration", width="large")
                def data_explorer_dialog():
                    data_explorer = DataExplorationComponent(st.session_state.builder, st.session_state.logger, data=st.session_state.builder.testing_data, target_column=st.session_state.builder.target_column)
                    data_explorer.render()

                if st.button("Testing Data Exploration"):
                    data_explorer_dialog()

                with st.expander("Encoded and Binned Feature Details", expanded=False):
                    st.write("""
                        ### Encoded and Binned Feature Details
                        
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

            
            if selected_tab == "📊 Performance Metrics":
                log_tab_switch("Performance Metrics")
                
                # Handle both binary and multiclass classification
                if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                    display_classification_report(result)
                else:
                    display_regression_report(result)
            
            elif selected_tab == "📈 Visualisations":
                log_tab_switch("Visualisations")
                # Handle both binary and multiclass classification
                if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                    display_classification_visualisations(result)
                else:
                    display_regression_visualisations(result)
        
            elif selected_tab == "📋 Sample Predictions":
                log_tab_switch("Sample Predictions")
                sample_predictions = display_sample_predictions(problem_type)
            
            elif selected_tab == "🚀 Model Health and Recommendations":
                log_tab_switch("Model Health and Recommendations")
                display_model_improvements(result, problem_type)
            
            # Log completion status with comprehensive summary
            completion_log = {
                "stage": "MODEL_EVALUATION",
                "status": "completed",
                "problem_type": problem_type,
                "model_type": st.session_state.builder.model["type"],
                "performance_summary": result["metrics"],
                "evaluation_time": result.get("evaluation_time"),
                "warnings": result.get("warnings", []),
                "timestamp": str(pd.Timestamp.now())
            }
            st.session_state.logger.log_calculation("Evaluation Stage Complete", completion_log)
            
            # Navigation
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Back to Training", key="back_to_training_bottom", type="primary"):
                    st.session_state.logger.log_user_action("Navigation", {"direction": "back"})
                    st.session_state.current_stage = ModelStage.MODEL_TRAINING
                    st.switch_page("pages/6_Model_Training.py")
            with col2:
                st.success("✅ Model evaluation completed successfully!")
                st.session_state.builder.stage_completion[ModelStage.MODEL_EVALUATION] = True
            with col3:
                if st.button("⏭️ Continue to Model Explanation", 
                           key="bottom_next",
                           width='stretch',
                           type="primary"):
                    st.session_state.logger.log_user_action(
                        "Navigation",
                        {
                            "from": "MODEL_EVALUATION",
                            "to": "MODEL_EXPLANATION",
                            "status": "success",
                            "timestamp": str(pd.Timestamp.now())
                        }
                    )
                    next_page = "8_Model_Explanation"
                    st.switch_page(f"pages/{next_page}.py")
        else:
            error_details = {
                "error_message": result["message"],
                "stage": "MODEL_EVALUATION",
                "model_type": st.session_state.builder.model["type"],
                "problem_type": problem_type,
                "timestamp": str(pd.Timestamp.now())
            }
            st.error(f"❌ {result['message']}")
            st.session_state.logger.log_error("Model Evaluation Failed", error_details)

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