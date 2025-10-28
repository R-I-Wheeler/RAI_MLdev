import streamlit as st
import pandas as pd
from Builder import Builder, ModelStage
from app import create_sidebar_navigation
from utils.logging.journey_viewer import render_journey_viewer
from utils.logging.log_viewer import render_log_viewer
from components.model_training.imbalance_handling import detect_classification_imbalance, display_imbalance_analysis_preview, display_imbalance_handling_tools
from components.model_training.training_results import display_search_training_results, display_optuna_results
from components.model_training.calibration import display_calibration_section
from components.model_training.threshold_analysis import display_threshold_analysis_section
from components.model_training.utils.training_state_manager import TrainingStateManager

def display_training_results(results):
    """Display the training results analysis and visualisations."""
    
    # Early validation check for required model state
    if not hasattr(st.session_state, 'builder') or not st.session_state.builder.model:
        st.error("Model state not found. Please return to model training and restart the process.")
        return
    
    # Validate that essential training results are available
    required_keys = ['best_score', 'best_params']
    if not results.get('info'):
        st.error("Training results information not found. Please restart training.")
        return
    
    missing_keys = [key for key in required_keys if key not in results['info']]
    if missing_keys:
        st.error(f"Training results incomplete. Missing: {', '.join(missing_keys)}. Please restart training.")
        return
    
    # For display purposes, also check if we have training metrics  
    # (cv_metrics for standard training, or optimisation data for Optuna)
    has_cv_metrics = 'cv_metrics' in results['info']
    has_optuna_data = 'optimisation_plots' in results['info']
    
    if not has_cv_metrics and not has_optuna_data:
        st.warning("""
            Training metrics may be incomplete. Some visualizations might not be available.
            If this persists, please restart the training process.
        """)
        # Don't return here, just warn - some parts might still work
    
    # Check if this is a new training run (different from previous) and reset selection state if so
    if 'previous_training_id' not in st.session_state or st.session_state.previous_training_id != id(results):
        # Store the current training results ID
        st.session_state.previous_training_id = id(results)
        
        # Reset model selection state variables to force fresh selection
        if 'selected_model_type' in st.session_state:
            del st.session_state.selected_model_type
        if 'selected_model_stability' in st.session_state:
            del st.session_state.selected_model_stability
        if 'previous_model_selection' in st.session_state:
            del st.session_state.previous_model_selection
    
    st.header("Training Results Analysis")
    
    # Log the display of training results
    st.session_state.logger.log_user_action(
        "Viewing Training Results",
        {
            "best_score": results['info']['best_score'],
            "model_type": st.session_state.builder.model['type']
        }
    )
    
    # Check if this is Optuna optimisation
    is_optuna = 'optimisation_plots' in results['info']
    
    if is_optuna:
        display_optuna_results(results)
    
    else:
        display_search_training_results(results)


def main():
    st.title("Model Training")
    
    # Add consistent navigation
    create_sidebar_navigation()
    
    # Initialize session state if needed
    if 'builder' not in st.session_state:
        st.session_state.builder = Builder()
        st.session_state.logger.log_stage_transition("START", "MODEL_TRAINING")
    
    # Ensure stage_completion dict has all required keys (defensive programming for edge cases)
    if not hasattr(st.session_state.builder, 'stage_completion'):
        st.session_state.builder.stage_completion = {stage: False for stage in ModelStage}
    else:
        # Ensure MODEL_SELECTION key exists in the dictionary
        if ModelStage.MODEL_SELECTION not in st.session_state.builder.stage_completion:
            st.session_state.builder.stage_completion[ModelStage.MODEL_SELECTION] = False

    # Initialize session caches and perform cleanup if needed
    TrainingStateManager.init_session_caches()
    if TrainingStateManager.should_cleanup():
        optimization_stats = TrainingStateManager.optimize_session_state()
        if optimization_stats['memory_saved_mb'] > 0:
            st.info(f"🧹 Automatic cleanup freed {optimization_stats['memory_saved_mb']:.1f} MB of memory")
    
    # Set current stage to MODEL_TRAINING
    st.session_state.builder.current_stage = ModelStage.MODEL_TRAINING
    
    # Get and render stage info
    stage_info = st.session_state.builder.get_current_stage_info()
    
    # Use logger from session state
    st.session_state.logger.log_page_state("Data_Loading", {
        "data_loaded": bool(st.session_state.get('data')),
        "target_selected": bool(st.session_state.get('target_column'))
    })

    # Render stage info once
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

    # Add dataset overview section
    if st.session_state.builder.X_train is not None and st.session_state.builder.X_test is not None:
        # Log dataset statistics
        st.session_state.logger.log_calculation(
            "Training Data Overview",
            {
                "train_samples": len(st.session_state.builder.X_train),
                "test_samples": len(st.session_state.builder.X_test),
                "features": len(st.session_state.builder.X_train.columns),
                "feature_names": list(st.session_state.builder.X_train.columns)
            }
        )
        
        st.header("Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Dataset")
            st.write(f"• Number of samples: {len(st.session_state.builder.X_train)}")
            st.write(f"• Number of features: {len(st.session_state.builder.X_train.columns)}")
            st.write("• Features:", ", ".join(st.session_state.builder.X_train.columns))
            
        with col2:
            st.subheader("Test Dataset")
            st.write(f"• Number of samples: {len(st.session_state.builder.X_test)}")
            st.write(f"• Number of features: {len(st.session_state.builder.X_test.columns)}")
            test_percentage = (len(st.session_state.builder.X_test) / 
                             (len(st.session_state.builder.X_train) + len(st.session_state.builder.X_test)) * 100)
            st.write(f"• Test set size: {test_percentage:.1f}% of total data")
        
    st.divider()

    # Class imbalance detection and handling (before training)
    if (st.session_state.builder.X_train is not None and
        st.session_state.builder.y_train is not None and
        st.session_state.builder.stage_completion[ModelStage.MODEL_SELECTION]):

        # Initialize imbalance workflow session state
        if 'imbalance_handled' not in st.session_state:
            st.session_state.imbalance_handled = False
        if 'imbalance_skipped' not in st.session_state:
            st.session_state.imbalance_skipped = False

        # Detect class imbalance
        has_imbalance, imbalance_analysis = detect_classification_imbalance()

        if has_imbalance and not st.session_state.imbalance_handled:
            st.header("⚖️ Class Imbalance Detected")

            # Display preview of imbalance analysis
            display_imbalance_analysis_preview(imbalance_analysis)

            # Display the original working imbalance handling tools
            display_imbalance_handling_tools()

            st.divider()
        elif st.session_state.imbalance_handled:
            if st.session_state.imbalance_skipped:
                st.info("ℹ️ **Class imbalance was detected but not addressed.** Training will proceed with original data.")
            else:
                st.success("✅ **Class imbalance has been addressed.** You can proceed with model training on balanced data.")
            st.divider()

    # Rest of the model training logic
    if not st.session_state.builder.stage_completion[ModelStage.MODEL_SELECTION]:
        st.error("Please complete model selection first")
        st.session_state.logger.log_error(
            "Model Training Access Denied",
            {"reason": "Model selection not completed"}
        )
        if st.button("Return to Model Selection"):
            st.session_state.builder.current_stage = ModelStage.MODEL_SELECTION
            st.rerun()
        return

    if st.session_state.builder.model is not None:
        problem_type = st.session_state.builder.model.get("problem_type", "unknown")
        
        # Create a more user-friendly display of the problem type
        problem_type_display = {
            "binary_classification": "Binary Classification",
            "multiclass_classification": "Multiclass Classification", 
            "classification": "Classification",
            "regression": "Regression"
        }.get(problem_type, problem_type)
        
        st.info(f"Training {st.session_state.builder.model['type']} model for {problem_type_display} problem")
        
        # Log model configuration
        st.session_state.logger.log_calculation(
            "Model Configuration",
            {
                "model_type": st.session_state.builder.model['type'],
                "problem_type": problem_type
            }
        )

        # Parameter explanations are now handled by the ContentManager

        with st.expander("🎯 Model Parameters Being Tuned", expanded=True):
            st.markdown("""
                ### Parameters Being Tuned

                The parameter ranges are automatically adjusted based on your dataset characteristics:
                - Dataset size (small vs large)
                - Feature dimensionality (high vs normal)
                - Data sparsity
                - Target distribution (classification vs regression)
            """)
            # Use cached parameter explanations from ContentManager
            model_explanation = st.session_state.builder.content_manager.get_parameter_explanations(
                st.session_state.builder.model['type']
            )
            st.markdown(model_explanation)

        with st.expander("🎓 Understanding Cross-Validation", expanded=False):
            # Use cached CV explanation from ContentManager
            st.markdown(st.session_state.builder.content_manager.get_cv_explanation())

            # Add a visual explanation using emojis
            st.markdown("### Visual Example (5-fold)")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(st.session_state.builder.content_manager.get_cv_visual_example())
            with col2:
                st.markdown("""
                    🔵 = Test Data
                    ✨ = Train Data
                """)

        with st.expander("🎯 Understanding Parameter Search", expanded=False):
            # Use cached search explanation from ContentManager
            st.markdown(st.session_state.builder.content_manager.get_search_explanation())
        
        st.subheader("Model Training Settings")

        # Add optimisation method selection with dynamic help text based on model
        model_type = st.session_state.builder.model.get("type", "")

        # Check if model benefits significantly from Optuna
        optuna_recommended_models = ["xgboost", "lightgbm", "catboost", "mlp"]

        if model_type in optuna_recommended_models:
            help_text = f"""
            **Recommendation for {model_type.upper()}**: Use Optuna for best results

            - **Random Search**: Traditional random sampling (limited early stopping support)
            - **Optuna** ⭐: Advanced optimization with full early stopping support, 30-50% faster training
            """
        else:
            help_text = """
            - **Random Search**: Traditional random sampling of parameter space
            - **Optuna**: Advanced optimization using Bayesian optimization and sophisticated pruning
            """

        optimisation_method = st.radio(
            "Select Hyperparameter Optimisation Method",
            ["Random Search", "Optuna"],
            help=help_text,
            index=1 if model_type in optuna_recommended_models else 0  # Default to Optuna for recommended models
        )
        
        cv_folds = st.slider("Number of Cross-validation Folds", min_value=2, max_value=10, value=5)
        
        # Parameter configurations label based on method
        if optimisation_method=="Random Search":
            n_iter = st.slider(
                "Number of Parameter Configurations to Test",
                min_value=10,
                max_value=100,
                value=50
            )
        else:
            n_iter = st.slider(
                "Number of Optimisation Trials",
                min_value=10,
                max_value=100,
                value=50
            )

        # Log training parameters
        st.session_state.logger.log_user_action(
            "Training Parameters Set",
            {
                "cv_folds": cv_folds,
                "parameter_configurations": n_iter,
                "optimisation_method": optimisation_method
            }
        )
        
        if st.button("Start Automated Model Training", type="primary"):
            # Log training initiation
            st.session_state.logger.log_user_action(
                "Model Training Started",
                {
                    "model_type": st.session_state.builder.model['type'],
                    "cv_folds": cv_folds,
                    "parameter_configurations": n_iter,
                    "optimisation_method": optimisation_method
                }
            )
            
            # Create progress containers
            progress_container = st.container()
            status_container = st.container()

            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

            with status_container:
                if optimisation_method == "Random Search":
                    status_text.info("🔄 Initializing Random Search hyperparameter tuning...")
                    progress_bar.progress(10)

                    status_text.info("🎯 Testing parameter combinations...")
                    progress_bar.progress(30)

                    result = st.session_state.builder.auto_tune_hyperparameters(
                        cv_folds=cv_folds,
                        n_iter=n_iter
                    )

                    progress_bar.progress(90)
                    
                    if result["success"]:
                        # Don't overwrite the model state - the auto_tune_hyperparameters method
                        # already set up all the necessary model configurations including adjusted_model
                        # Just ensure the optimization method and selection type are set
                        st.session_state.builder.model.update({
                            "optimisation_method": "random_search",
                            "selection_type": "mean_score"
                        })
                        
                        # Set session state for model selection
                        st.session_state.selected_model_type = "mean_score"
                        st.session_state.previous_model_selection = "mean_score"
                        
                else:  # Optuna
                    status_text.info("🚀 Initializing Optuna hyperparameter optimization...")
                    progress_bar.progress(10)

                    status_text.info("🧠 Smart parameter exploration in progress...")
                    progress_bar.progress(30)

                    result = st.session_state.builder.auto_tune_hyperparameters_optuna(
                        cv_folds=cv_folds,
                        n_trials=n_iter
                    )

                    progress_bar.progress(90)
                
                if result["success"]:
                    status_text.success("✅ Training completed successfully!")
                    progress_bar.progress(100)

                    # Log detailed training results
                    st.session_state.logger.log_calculation(
                        "Model Training Completed",
                        {
                            "model_type": st.session_state.builder.model['type'],
                            "best_score": result["info"]["best_score"],
                            "best_params": result["info"]["best_params"],
                            "cv_metrics": result["info"]["cv_metrics"],
                            "training_time": result["info"].get("training_time", None)
                        }
                    )
                    
                    # Log model metrics
                    st.session_state.logger.log_model_metrics({
                        "best_cv_score": result["info"]["best_score"],
                        "cv_std": result["info"].get("cv_std", None),
                        "cv_metrics": result["info"]["cv_metrics"]
                    })
                    
                    # Log stability analysis
                    stability_level = result["info"]["stability_analysis"]["level"]
                    st.session_state.logger.log_calculation(
                        "Model Stability Analysis",
                        result["info"]["stability_analysis"]
                    )
                    
                    if stability_level != "High stability":
                        st.session_state.logger.log_recommendation(
                            "Stability Improvement Needed",
                            {
                                "stability_level": stability_level,
                                "recommendations": result["info"]["stability_analysis"]["recommendations"]
                            }
                        )
                    
                    st.success(result["message"])
                    st.session_state.builder.stage_completion[ModelStage.MODEL_TRAINING] = True
                    st.session_state.training_complete = True
                    st.session_state.training_results = result
                    
                    # Store model signature after training completes
                    # This allows us to detect if user goes back and selects a different model
                    # or modifies the dataset (e.g., feature selection changes)
                    model = st.session_state.builder.model
                    if model is not None:
                        st.session_state.last_training_model_signature = {
                            'model_type': model.get('type'),
                            'problem_type': model.get('problem_type'),
                            'n_features': st.session_state.builder.X_train.shape[1] if st.session_state.builder.X_train is not None else 0,
                            'n_train_samples': len(st.session_state.builder.X_train) if st.session_state.builder.X_train is not None else 0,
                            'n_test_samples': len(st.session_state.builder.X_test) if st.session_state.builder.X_test is not None else 0,
                            'feature_names': tuple(st.session_state.builder.X_train.columns) if st.session_state.builder.X_train is not None else ()
                        }
                    
                    # Reset navigation pill to default (Training Results) when training completes
                    if 'active_training_pill' in st.session_state:
                        del st.session_state.active_training_pill
                    
                    # Log stage completion
                    st.session_state.logger.log_stage_transition(
                        "MODEL_TRAINING",
                        "MODEL_EVALUATION"
                    )
                    st.rerun()  # Rerun to show results in a clean state
                else:
                    st.error(result["message"])
                    # Log training failure
                    st.session_state.logger.log_error(
                        "Model Training Failed",
                        {
                            "error": result["message"],
                            "model_type": st.session_state.builder.model['type'],
                            "cv_folds": cv_folds,
                            "n_iter": n_iter
                        }
                    )
    else:
        st.error("No model selected. Please return to model selection stage.")
        st.session_state.logger.log_error(
            "Training Failed",
            {"reason": "No model selected"}
        )

    # Show training results if they exist
    if hasattr(st.session_state, 'training_complete') and st.session_state.training_complete:
        # ===== MODEL CHANGE DETECTION: Detect if model or dataset has changed and clear results =====
        def get_model_signature():
            """Create a unique signature for the current model and dataset configuration."""
            model = st.session_state.builder.model
            if model is None:
                return None
            return {
                'model_type': model.get('type'),
                'problem_type': model.get('problem_type'),
                'n_features': st.session_state.builder.X_train.shape[1] if st.session_state.builder.X_train is not None else 0,
                'n_train_samples': len(st.session_state.builder.X_train) if st.session_state.builder.X_train is not None else 0,
                'n_test_samples': len(st.session_state.builder.X_test) if st.session_state.builder.X_test is not None else 0,
                'feature_names': tuple(st.session_state.builder.X_train.columns) if st.session_state.builder.X_train is not None else ()
            }
        
        # Check if model or dataset has changed since training was completed
        current_signature = get_model_signature()
        
        if current_signature is not None:
            if 'last_training_model_signature' in st.session_state:
                if st.session_state.last_training_model_signature != current_signature:
                    # Model or dataset has changed - clear all training results and reset state
                    
                    # Determine what changed for better user feedback
                    prev_sig = st.session_state.last_training_model_signature
                    change_details = []
                    
                    if prev_sig.get('model_type') != current_signature.get('model_type'):
                        change_details.append(f"Model type changed from {prev_sig.get('model_type')} to {current_signature.get('model_type')}")
                    
                    if prev_sig.get('n_features') != current_signature.get('n_features'):
                        change_details.append(f"Number of features changed from {prev_sig.get('n_features')} to {current_signature.get('n_features')}")
                    
                    if prev_sig.get('feature_names') != current_signature.get('feature_names'):
                        if prev_sig.get('n_features') == current_signature.get('n_features'):
                            change_details.append("Feature names have changed")
                        # If number changed, we already reported that
                    
                    if prev_sig.get('n_train_samples') != current_signature.get('n_train_samples'):
                        change_details.append(f"Training samples changed from {prev_sig.get('n_train_samples')} to {current_signature.get('n_train_samples')}")
                    
                    # Clear training results
                    if 'training_complete' in st.session_state:
                        del st.session_state.training_complete
                    if 'training_results' in st.session_state:
                        del st.session_state.training_results
                    
                    # Clear model selection state variables
                    if 'selected_model_type' in st.session_state:
                        del st.session_state.selected_model_type
                    if 'selected_model_stability' in st.session_state:
                        del st.session_state.selected_model_stability
                    if 'previous_model_selection' in st.session_state:
                        del st.session_state.previous_model_selection
                    if 'previous_training_id' in st.session_state:
                        del st.session_state.previous_training_id
                    
                    # Reset the navigation pill to default
                    if 'active_training_pill' in st.session_state:
                        del st.session_state.active_training_pill
                    
                    # Update signature
                    st.session_state.last_training_model_signature = current_signature
                    
                    # Log the change with details
                    st.session_state.logger.log_user_action(
                        "Model/Dataset Change Detected in Training",
                        {
                            "action": "Training Results Cleared",
                            "changes": change_details,
                            "previous_signature": prev_sig,
                            "current_signature": current_signature,
                            "timestamp": str(pd.Timestamp.now())
                        }
                    )
                    
                    # Show user notification with details
                    if change_details:
                        details_text = "\n- ".join([""] + change_details)
                        st.info(f"🔄 Configuration change detected:{details_text}\n\nTraining results have been cleared automatically.")
                    else:
                        st.info("🔄 Configuration change detected. Training results have been cleared automatically.")
                    st.rerun()
        
        # Add a clear results button at the top of the results section
        st.header("Training Results")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Training completed successfully! Review your results below.")
        with col2:
            if st.button("🗑️ Clear Results", type="secondary", help="Clear training results and start fresh"):
                # Use session state manager for comprehensive cleanup
                cleared_count = TrainingStateManager.clear_training_results()
                optimization_stats = TrainingStateManager.optimize_session_state()

                # Clear the training_complete flag
                if 'training_complete' in st.session_state:
                    del st.session_state.training_complete

                # Clear model selection state variables
                if 'selected_model_type' in st.session_state:
                    del st.session_state.selected_model_type
                if 'selected_model_stability' in st.session_state:
                    del st.session_state.selected_model_stability
                if 'previous_model_selection' in st.session_state:
                    del st.session_state.previous_model_selection
                if 'previous_training_id' in st.session_state:
                    del st.session_state.previous_training_id
                
                # Clear model signature to allow retraining
                if 'last_training_model_signature' in st.session_state:
                    del st.session_state.last_training_model_signature
                
                # Reset the navigation pill to default (Training Results)
                if 'active_training_pill' in st.session_state:
                    del st.session_state.active_training_pill
                
                # Reset model to its default/pre-trained state
                # Clear ONLY training-specific model attributes, preserve the base model structure
                # We need to keep 'type', 'problem_type', and 'model' (the base unfitted model)
                if st.session_state.builder.model:
                    model_keys_to_reset = [
                        'active_model',
                        'best_model',
                        'best_params',
                        'best_score',
                        'active_params',
                        'cv_metrics',
                        'cv_results',
                        'cv_std',
                        'stability_analysis',
                        'optimisation_method',
                        'selection_type',
                        'adjusted_model',
                        'adjusted_params',
                        'adjusted_cv_metrics',
                        'optimization_history',
                        'optimisation_plots',
                        'is_calibrated',
                        'calibrated_model',
                        'calibration_method',
                        'calibration_cv_folds',
                        'original_model',
                        'threshold_optimized',
                        'optimal_threshold',
                        'threshold_is_binary',
                        'threshold_criterion'
                    ]
                    
                    # Only delete training-specific keys, NOT 'model', 'type', or 'problem_type'
                    for key in model_keys_to_reset:
                        if key in st.session_state.builder.model:
                            del st.session_state.builder.model[key]
                    
                    # If the model has been fitted, we need to reset it to its original unfitted state
                    # Get a fresh instance of the base model with default parameters
                    if 'model' in st.session_state.builder.model and 'type' in st.session_state.builder.model:
                        model_type = st.session_state.builder.model['type']
                        problem_type = st.session_state.builder.model['problem_type']
                        
                        # Re-initialize the model with default parameters (matching Builder.select_model)
                        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
                        from sklearn.naive_bayes import GaussianNB
                        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                        from sklearn.neural_network import MLPClassifier, MLPRegressor
                        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
                        from xgboost import XGBClassifier, XGBRegressor
                        from lightgbm import LGBMClassifier, LGBMRegressor
                        from catboost import CatBoostClassifier, CatBoostRegressor
                        
                        model_configs = {
                            "classification": {
                                "logistic_regression": LogisticRegression(random_state=42, n_jobs=-1),
                                "naive_bayes": GaussianNB(),
                                "decision_tree": DecisionTreeClassifier(random_state=42),
                                "random_forest": RandomForestClassifier(random_state=42, n_jobs=-1),
                                "mlp": MLPClassifier(max_iter=1000, random_state=42),
                                "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
                                "catboost": CatBoostClassifier(random_state=42, verbose=False),
                                "xgboost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', nthread=-1),
                                "lightgbm": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
                            },
                            "regression": {
                                "linear_regression": LinearRegression(n_jobs=-1),
                                "ridge_regression": Ridge(random_state=42),
                                "decision_tree": DecisionTreeRegressor(random_state=42),
                                "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
                                "mlp": MLPRegressor(max_iter=1000, random_state=42),
                                "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
                                "catboost": CatBoostRegressor(random_state=42, verbose=False),
                                "xgboost": XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse', nthread=-1),
                                "lightgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
                            }
                        }
                        
                        config_key = "classification" if problem_type in ["binary_classification", "multiclass_classification", "classification"] else "regression"
                        
                        if model_type in model_configs[config_key]:
                            st.session_state.builder.model['model'] = model_configs[config_key][model_type]
                
                # Reset model training completion status
                st.session_state.builder.stage_completion[ModelStage.MODEL_TRAINING] = False
                
                # Reset calibration state using the state manager
                TrainingStateManager.reset_calibration_state()
                
                # Log the clear action
                st.session_state.logger.log_user_action(
                    "Training Results Cleared",
                    {"action": "clear_results", "reason": "user_requested"}
                )
                
                st.success(f"""
                    Training results cleared successfully!
                    - Cleared {cleared_count} training variables
                    - Freed {optimization_stats['memory_saved_mb']:.1f} MB of memory
                    - Cache items cleaned: {optimization_stats['cache_cleanup']['removed_items']}
                    - Model reset to default state
                """)
                st.rerun()
        
        st.divider()
        
        # Determine available pills based on problem type
        def get_available_pills():
            base_pills = ["📊 Training Results"]

            # Add classification-specific pills only for classification models
            if (st.session_state.builder.model and
                TrainingStateManager.is_classification_model()):
                base_pills.extend(["🎯 Model Calibration", "⚖️ Threshold Analysis"])

            return base_pills

        # Get available pills for current model type
        available_pills = get_available_pills()

        # Initialize session state for active pill tracking
        if 'active_training_pill' not in st.session_state:
            st.session_state.active_training_pill = available_pills[0]

        # Ensure current pill selection is still valid for current model type
        if st.session_state.active_training_pill not in available_pills:
            st.session_state.active_training_pill = available_pills[0]

        # Create pills navigation with conditional sections
        active_section = st.pills(
            "Training Analysis Sections",
            available_pills,
            default=st.session_state.active_training_pill,
            key="training_analysis_pills"
        )

        # Update session state with selected pill
        if active_section != st.session_state.active_training_pill:
            st.session_state.active_training_pill = active_section

        # Display content based on selected pill
        if active_section == "📊 Training Results":
            try:
                # Display training results with loading placeholder
                with st.container():
                    display_training_results(st.session_state.training_results)
            except KeyError as e:
                st.error(f"""
                    Error displaying training results: Missing data key '{e}'.

                    This might happen when navigating between pages or due to session state issues.

                    **To fix this:**
                    1. Return to the model training section
                    2. Restart the training process
                    3. Or use the navigation buttons below to continue
                """)
                st.session_state.logger.log_error(
                    "Display Training Results Error",
                    {
                        "error_type": "KeyError",
                        "missing_key": str(e),
                        "action": "navigation_suggested"
                    }
                )
            except Exception as e:
                st.error(f"""
                    Unexpected error displaying training results: {str(e)}

                    Please try refreshing the page or restarting the training process.
                """)
                st.session_state.logger.log_error(
                    "Display Training Results Error",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "action": "refresh_suggested"
                    }
                )

        elif active_section == "🎯 Model Calibration":
            # Lazy load calibration section only when pill is selected
            display_calibration_section()

        elif active_section == "⚖️ Threshold Analysis":
            # Lazy load threshold analysis only when pill is selected
            display_threshold_analysis_section()

        # Update the bottom proceed button as well
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Back to Model Selection", key="back_to_model_selection", type="primary"):
                st.session_state.logger.log_user_action("Navigation", {"direction": "back"})
                st.session_state.current_stage = ModelStage.MODEL_SELECTION
                st.switch_page("pages/5_Model_Selection.py")
        with col2:
            if hasattr(st.session_state, 'selected_model_type'):
                st.success("✅ Model training completed successfully!")
        with col3:
            # Only enable the button if a model has been selected
            button_disabled = not hasattr(st.session_state, 'selected_model_type')
            
            if st.button("✨ Continue to Model Evaluation", key="bottom_next", width='stretch', type="primary", disabled=button_disabled):
                st.session_state.logger.log_stage_transition(
                    "MODEL_TRAINING",
                    "MODEL_EVALUATION"
                )
                if 'cv_metrics' in st.session_state.builder.model:
                    metrics = st.session_state.builder.model['cv_metrics']
                    best_score = st.session_state.builder.model.get('best_score', metrics.get('mean_score', 0))

                # Get stability level from training results
                stability_level = "Unknown"
                if hasattr(st.session_state, 'training_results') and st.session_state.training_results:
                    stability_level = st.session_state.training_results.get("info", {}).get("stability_analysis", {}).get("level", "Unknown")


                st.session_state.logger.log_journey_point(
                        stage="MODEL_TRAINING",
                        decision_type="MODEL_TRAINING",
                        description="Model training completed",
                        details={"Model Type": st.session_state.builder.model['type'],
                                "Best Score": best_score,
                                "Optimisation Method": optimisation_method,
                                "Selection Type": st.session_state.builder.model['selection_type'],
                                "CV Folds": cv_folds,
                                "Parameter Configurations": n_iter,
                                "Stability Level": stability_level
                                },
                        parent_id=None
                    )
                next_page = "7_Model_Evaluation"
                st.switch_page(f"pages/{next_page}.py")
            
            if button_disabled:
                st.info("Please select a model to continue.")
    
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