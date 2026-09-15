"""
Automated Model Selection and Training Component

This module provides a fully automated component that combines the functionality
of Model Selection (Page 5) and Model Training (Page 6) into a single streamlined workflow.

The component follows the same architectural pattern as automated_preprocessing.py and provides:
- Automatic problem type detection and model recommendation
- Intelligent model selection based on dataset characteristics
- Automated hyperparameter optimization (Random Search or Optuna)
- Model training with cross-validation
- Classification-specific optimizations (calibration, threshold analysis)
- Comprehensive logging and tracking

Author: ML Builder Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
from typing import Dict, Any, Optional, Tuple
from copy import deepcopy

from content.stage_info import ModelStage
from components.model_training.utils.validation_utils import (
    resampling_estimator, validation_probabilities,
)


class AutomatedModelSelectionTraining:
    """
    Automated model selection and training component that replicates 100%
    of manual model selection and training functionality.

    This component automates the entire workflow from problem detection through
    final model optimization, including:
    - Problem type detection
    - Model recommendation
    - Model comparison and selection
    - Class imbalance handling
    - Hyperparameter optimization
    - Model calibration (classification)
    - Threshold optimization (binary classification)

    Attributes:
        builder: Builder instance containing data and model state
        logger: MLLogger instance for tracking operations
        optimization_method: 'random_search' or 'optuna' (default: 'optuna')
        cv_folds: Number of cross-validation folds (default: 5, adaptive)
        n_iter: Number of parameter configurations/trials (default: 50, adaptive)
        show_analysis: Whether to display analysis summaries (default: True)
        auto_handle_imbalance: Automatically handle class imbalance (default: True)
        model_override: Force selection of specific model (optional)
        selection_summary: Dictionary containing results from each step
    """

    def __init__(
        self,
        builder,
        logger,
        optimization_method: str = 'optuna',
        cv_folds: Optional[int] = None,
        n_iter: Optional[int] = None,
        show_analysis: bool = True,
        auto_handle_imbalance: bool = True,
        model_override: Optional[str] = None
    ):
        """
        Initialize the automated model selection and training component.

        Args:
            builder: Builder instance containing data and model state
            logger: MLLogger instance for tracking operations
            optimization_method: 'random_search' or 'optuna' (default: 'optuna')
            cv_folds: Number of cross-validation folds (None for adaptive, default: None)
            n_iter: Number of parameter configurations/trials (None for adaptive, default: None)
            show_analysis: Whether to display analysis summaries (default: True)
            auto_handle_imbalance: Automatically handle class imbalance (default: True)
            model_override: Force selection of specific model (optional)
        """
        self.builder = builder
        self.logger = logger
        self.optimization_method = optimization_method
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.show_analysis = show_analysis
        self.auto_handle_imbalance = auto_handle_imbalance
        self.model_override = model_override

        # Initialize selection summary to track all steps
        self.selection_summary = {
            'problem_type': None,
            'xgboost_compatible': None,
            'recommendation': None,
            'comparison_results': None,
            'selected_model': None,
            'selection_reason': None,
            'selected_from_comparison': False,
            'imbalance_handled': False,
            'imbalance_method': None,
            'imbalance_details': None,
            'training_complete': False,
            'training_results': None,
            'best_params': None,
            'best_score': None,
            'cv_metrics': None,
            'stability_analysis': None,
            'final_model_selected': False,
            'selection_type': None,
            'calibration_applied': False,
            'calibration_method': None,
            'calibration_comparison': None,
            'threshold_optimized': False,
            'optimal_threshold': None,
            'threshold_criterion': None
        }

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete automated model selection and training workflow.

        This method orchestrates all steps:
        1. Problem detection and model recommendation
        2. Model comparison and selection decision
        3. Model selection and initialization
        4. Class imbalance handling (classification only)
        5. Hyperparameter optimization and training
        6. Final model selection and validation
        7. Classification-specific optimizations

        Returns:
            Dict containing:
                - success: bool - Whether the process completed successfully
                - summary: str - High-level summary message
                - details: dict - Detailed results from each step
                - error: str - Error message if failed (optional)
        """
        try:
            if self.show_analysis:
                st.write("# 🤖 Automated Model Selection and Training")
                st.write("---")

            # Log start of automated process
            self.logger.log_stage_transition(
                "DATA_PREPROCESSING",
                "MODEL_SELECTION"
            )

            # Step 1: Problem Detection and Model Recommendation
            self._step_1_problem_detection_and_recommendation()

            # Step 2: Model Comparison and Selection Decision
            self._step_2_model_comparison_and_selection()

            # Step 3: Model Selection and Initialization
            self._step_3_model_selection()

            # Step 4: Class Imbalance Handling (classification only)
            self._step_4_class_imbalance_handling()

            # Step 5: Hyperparameter Optimization and Training
            self._step_5_hyperparameter_optimization()

            # Step 6: Final Model Selection and Validation
            self._step_6_final_model_selection()

            # Step 7: Classification-Specific Optimizations
            self._step_7_classification_optimizations()

            # Generate final summary
            summary = self._generate_summary()

            if self.show_analysis:
                st.write("---")
                st.success("✅ **Automated Model Selection and Training Complete!**")
                st.write(summary)

            return {
                'success': True,
                'summary': summary,
                'details': self.selection_summary
            }

        except Exception as e:
            error_msg = f"Automated model selection and training failed: {str(e)}"
            self.logger.log_error(
                "Automated Model Selection and Training Failed",
                {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "step_completed": self._get_last_completed_step()
                }
            )

            if self.show_analysis:
                st.error(f"❌ {error_msg}")
                st.error(f"**Error details:** {str(e)}")

            return {
                'success': False,
                'summary': error_msg,
                'details': self.selection_summary,
                'error': str(e)
            }

    def _step_1_problem_detection_and_recommendation(self):
        """
        Step 1: Problem Type Detection and Model Recommendation

        - Detects problem type (classification/regression)
        - Checks XGBoost compatibility
        - Analyzes dataset characteristics
        - Generates model recommendation with reasoning
        """
        if self.show_analysis:
            st.write("### Step 1: Problem Detection & Model Recommendation")

        # Detect problem type using Builder's built-in detection
        problem_type = self.builder.detect_problem_type()

        # Check XGBoost compatibility (class labeling requirements)
        xgboost_compatible = self.builder.check_xgboost_compatibility()

        # Get dataset characteristics
        from components.model_selection.model_recommendation_engine import get_dataset_characteristics

        characteristics = get_dataset_characteristics(
            self.builder.training_data,
            self.builder.target_column
        )

        # Generate model recommendation using recommendation engine
        from components.model_selection.model_recommendation_engine import get_model_recommendation

        recommendation = get_model_recommendation(
            self.builder.training_data,
            self.builder.target_column,
            problem_type
        )

        # Store recommendation details
        self.selection_summary['problem_type'] = problem_type
        self.selection_summary['xgboost_compatible'] = xgboost_compatible
        self.selection_summary['recommendation'] = recommendation

        # Log recommendation
        self.logger.log_calculation(
            "Automated Model Recommendation",
            {
                "problem_type": problem_type,
                "recommended_model": recommendation['recommended_model'],
                "reasons": recommendation['reasons'],
                "dataset_characteristics": characteristics
            }
        )

        if self.show_analysis:
            st.success(f"✅ Problem Type: **{problem_type}**")
            st.write(f"- **Recommended Model:** {recommendation['recommended_model']}")
            st.write(f"- **XGBoost Compatible:** {'Yes' if xgboost_compatible else 'No'}")
            with st.expander("View Recommendation Reasoning"):
                for reason in recommendation['reasons']:
                    st.write(f"• {reason}")

    def _step_2_model_comparison_and_selection(self):
        """
        Step 2: Model Comparison and Selection Decision

        - Runs quick comparison on all available models
        - Compares recommended model vs empirically best model
        - Uses defined test to make final selection decision
        - Always runs (not optional) for best results
        """
        if self.show_analysis:
            st.write("### Step 2: Model Comparison & Selection")

        # Perform quick comparison using Builder's built-in method
        results_df = self.builder.get_quick_model_comparison(
            sample_size=1000,
            exclude_xgboost=not self.selection_summary['xgboost_compatible']
        )

        # Find best performing model from comparison
        best_model, best_score, best_metric = self.builder.get_best_model_from_comparison(results_df)

        # Store comparison results
        self.selection_summary['comparison_results'] = {
            'results_df': results_df.to_dict('records'),
            'best_model': best_model,
            'best_score': best_score,
            'best_metric': best_metric
        }

        # Get recommended model from Step 1
        recommended_model = self.selection_summary['recommendation']['recommended_model']

        # Get metrics for recommended model from comparison
        recommended_model_row = results_df[results_df['Model'] == recommended_model]
        if not recommended_model_row.empty:
            recommended_score = recommended_model_row[best_metric].values[0]
        else:
            recommended_score = None

        # Make selection decision using defined test
        selected_model, selection_reason = self._make_model_selection_decision(
            recommended_model=recommended_model,
            recommended_score=recommended_score,
            best_model=best_model,
            best_score=best_score,
            best_metric=best_metric,
            results_df=results_df
        )

        # Store selection decision
        self.selection_summary['selected_model'] = selected_model
        self.selection_summary['selection_reason'] = selection_reason
        self.selection_summary['selected_from_comparison'] = True

        # Log comparison and selection decision
        self.logger.log_calculation(
            "Model Comparison and Selection",
            {
                "recommended_model": recommended_model,
                "recommended_score": float(recommended_score) if recommended_score is not None else None,
                "best_performing_model": best_model,
                "best_score": float(best_score),
                "best_metric": best_metric,
                "final_selection": selected_model,
                "selection_reason": selection_reason,
                "all_results": results_df.to_dict('records')
            }
        )

        if self.show_analysis:
            st.success(f"✅ Model Comparison complete")
            st.write(f"- **Recommended by Engine:** {recommended_model}")
            st.write(f"- **Best in Comparison:** {best_model} ({best_score:.4f} {best_metric})")
            st.write(f"- **Final Selection:** {selected_model}")
            st.write(f"- **Reason:** {selection_reason}")

    def _make_model_selection_decision(
        self,
        recommended_model: str,
        recommended_score: float,
        best_model: str,
        best_score: float,
        best_metric: str,
        results_df: pd.DataFrame
    ) -> Tuple[str, str]:
        """
        Make intelligent model selection decision between recommended and best-performing model.

        Decision Logic:
        1. If recommended == best, use it (unanimous decision)
        2. If recommended score is within 5% of best score, use recommended (trust analysis)
        3. If best score is significantly better (>5%), use best model with caveat
        4. If recommended model failed comparison, use best model

        Args:
            recommended_model: Model recommended by recommendation engine
            recommended_score: Score of recommended model in comparison
            best_model: Best performing model in comparison
            best_score: Score of best performing model
            best_metric: Metric used for comparison
            results_df: Full comparison results

        Returns:
            tuple: (selected_model, selection_reason)
        """
        # Case 1: Recommended model is also the best performer
        if recommended_model == best_model:
            return recommended_model, "Unanimous: Recommended by engine and best in empirical comparison"

        # Case 2: Recommended model not in comparison results
        if recommended_score is None:
            return best_model, f"Recommended model not available in comparison, using best performer ({best_model})"

        # Case 3: Check performance difference
        score_difference_pct = abs((best_score - recommended_score) / best_score) * 100

        if score_difference_pct <= 5.0:
            # Recommended model is within 5% of best - trust the sophisticated analysis
            return recommended_model, f"Recommended model within 5% of best ({score_difference_pct:.1f}% difference), trusting sophisticated analysis"

        elif score_difference_pct <= 10.0:
            # 5-10% difference - consider dataset characteristics from recommendation
            recommendation_reasons = self.selection_summary['recommendation'].get('reasons', [])
            has_strong_reasoning = any(
                keyword in ' '.join(recommendation_reasons).lower()
                for keyword in ['large', 'complex', 'high-dimensional', 'intricate']
            )

            if has_strong_reasoning:
                return recommended_model, f"Moderate difference ({score_difference_pct:.1f}%), but recommendation has strong reasoning for dataset characteristics"
            else:
                return best_model, f"Best performer has {score_difference_pct:.1f}% better score, selecting for empirical performance"

        else:
            # >10% difference - significant, use best model
            return best_model, f"Best performer significantly better ({score_difference_pct:.1f}% improvement over recommended)"

    def _step_3_model_selection(self):
        """
        Step 3: Model Selection and Initialization

        - Selects the recommended model (or user override)
        - Initializes model in Builder
        - Validates model compatibility
        - Marks model selection stage as complete
        """
        if self.show_analysis:
            st.write("### Step 3: Model Selection & Initialization")

        # Get selected model (from Step 2 or use override if specified)
        selected_model = (
            self.model_override
            if self.model_override
            else self.selection_summary['selected_model']
        )

        # Select model using Builder's selection method
        result = self.builder.select_model(selected_model)

        if not result['success']:
            raise Exception(f"Model selection failed: {result['message']}")

        # Update selection details if model override was used
        if self.model_override:
            self.selection_summary['selected_model'] = selected_model
            self.selection_summary['selection_reason'] = "User override"

        # Mark stage as complete
        self.builder.stage_completion[ModelStage.MODEL_SELECTION] = True

        # Log model selection
        self.logger.log_calculation(
            "Automated Model Selection",
            {
                "selected_model": selected_model,
                "selection_reason": self.selection_summary['selection_reason'],
                "problem_type": self.selection_summary['problem_type']
            }
        )

        self.logger.log_stage_transition(
            "MODEL_SELECTION",
            "MODEL_TRAINING"
        )

        if self.show_analysis:
            st.success(f"✅ Model initialized: **{selected_model}**")

    def _step_4_class_imbalance_handling(self):
        """
        Step 4: Class Imbalance Detection and Handling

        - Uses detect_classification_imbalance() from imbalance_handling.py
        - Gets recommendation from Builder (delegates to imbalance_utils.py)
        - Configures the recommended strategy for cross-validation training folds
        - Skips for regression problems
        """
        # A new automated run configures its own fold-local strategy.
        self.builder.model.pop('resampling_method', None)
        # Skip for regression problems
        if self.selection_summary['problem_type'] == 'regression' or not self.auto_handle_imbalance:
            if self.show_analysis:
                st.info("ℹ️ Automatic class imbalance handling is disabled or not applicable.")
            self.selection_summary['imbalance_handled'] = False
            return

        if self.show_analysis:
            st.write("### Step 4: Class Imbalance Handling")

        # Import from existing component
        from components.model_training.imbalance_handling import (
            detect_classification_imbalance,
        )

        # Detect class imbalance using existing function
        has_imbalance, imbalance_analysis = detect_classification_imbalance()

        if not has_imbalance:
            if self.show_analysis:
                st.info("✅ No significant class imbalance detected (ratio ≤ 3:1).")

            self.selection_summary['imbalance_handled'] = False
            self.logger.log_calculation(
                "Imbalance Detection",
                {"has_imbalance": False, "reason": "Imbalance ratio below threshold"}
            )
            return

        # Get recommendation from Builder (which uses imbalance_utils.py)
        recommendation = self.builder.get_imbalance_recommendation()

        if not recommendation['success']:
            if self.show_analysis:
                st.warning(f"⚠️ Could not get imbalance recommendation: {recommendation['message']}")
            self.selection_summary['imbalance_handled'] = False
            return

        recommended_method = recommendation['recommended_method']

        # Log the recommendation
        self.logger.log_calculation(
            "Imbalance Recommendation",
            {
                "recommended_method": recommended_method,
                "explanation": recommendation['explanation'],
                "metrics": recommendation['metrics']
            }
        )

        if self.show_analysis:
            st.info(f"🎯 Recommended method: **{recommended_method}**")
            st.write(recommendation['explanation'])

        # Skip if recommendation is "None (Original Data)"
        if recommended_method == "None (Original Data)":
            if self.show_analysis:
                st.info("ℹ️ Recommendation is to use original data. Skipping resampling.")
            self.selection_summary['imbalance_handled'] = False
            st.session_state.imbalance_handled = True
            st.session_state.imbalance_skipped = True
            return

        # Record a strategy; the tuner applies it inside each training fold.
        # Keep the canonical training data and validation class distribution intact.
        self.builder.model['resampling_method'] = recommended_method
        self.selection_summary['imbalance_handled'] = True
        self.selection_summary['imbalance_method'] = recommended_method
        original_dist = pd.Series(self.builder.y_train).value_counts()
        self.selection_summary['imbalance_details'] = {
            'original_distribution': original_dist.to_dict(),
            'original_samples': len(self.builder.y_train),
            'imbalance_ratio_before': float(imbalance_analysis['metrics']['imbalance_ratio']),
            'timing': 'Within each cross-validation training fold and final fit',
        }
        st.session_state.imbalance_handled = True
        st.session_state.imbalance_skipped = False
        self.logger.log_calculation("Automated Fold Resampling Configured", self.selection_summary['imbalance_details'])
        if self.show_analysis:
            st.success(f"✅ {recommended_method} will be applied within each training fold.")

    def _step_5_hyperparameter_optimization(self):
        """
        Step 5: Hyperparameter Optimization and Training

        - Executes automated hyperparameter tuning
        - Supports Random Search or Optuna optimization
        - Performs cross-validation
        - Generates stability analysis
        - Creates comprehensive training results
        """
        # Override optimization method to Optuna if CatBoost is selected
        # CatBoost is incompatible with Random Search due to architectural constraints
        original_optimization_method = self.optimization_method
        if self.selection_summary['selected_model'] == 'catboost' and self.optimization_method == 'random_search':
            self.optimization_method = 'optuna'
            if self.show_analysis:
                st.warning("⚠️ **Optimization Method Override:** CatBoost requires Optuna optimization. Switching from Random Search to Optuna.")

        if self.show_analysis:
            st.write(f"### Step 5: Hyperparameter Optimization ({self.optimization_method.title()})")

        # Calculate adaptive CV folds and iterations based on dataset characteristics
        cv_folds, n_iter = self._calculate_adaptive_training_params()

        if self.show_analysis:
            st.info(f"📊 Training Parameters:")
            folds_msg = f"{cv_folds} (user-specified)" if self.cv_folds is not None else f"{cv_folds} (adaptive based on dataset size)"
            iter_msg = f"{n_iter} (user-specified)" if self.n_iter is not None else f"{n_iter} (adaptive based on feature complexity)"
            st.write(f"- **CV Folds:** {folds_msg}")
            st.write(f"- **Iterations:** {iter_msg}")

        # Execute optimization based on selected method
        if self.optimization_method == 'random_search':
            result = self.builder.auto_tune_hyperparameters(
                cv_folds=cv_folds,
                n_iter=n_iter
            )
        else:  # optuna
            result = self.builder.auto_tune_hyperparameters_optuna(
                cv_folds=cv_folds,
                n_trials=n_iter
            )

        if not result['success']:
            raise Exception(f"Hyperparameter optimization failed: {result['message']}")

        # Store training results
        self.selection_summary['training_complete'] = True
        self.selection_summary['training_results'] = result
        self.selection_summary['best_params'] = result['info']['best_params']
        self.selection_summary['best_score'] = result['info']['best_score']
        self.selection_summary['cv_metrics'] = result['info']['cv_metrics']
        self.selection_summary['stability_analysis'] = result['info']['stability_analysis']
        self.selection_summary['optimization_method'] = self.optimization_method  # Store actual method used (may differ if overridden)

        # Store cv_std from cv_metrics (both random_search and optuna store it here)
        self.selection_summary['cv_std'] = result['info']['cv_metrics'].get('std_score', 0)

        # Store fold-level scores for visualization
        # Both Random Search and Optuna store fold_scores in cv_metrics['fold_scores']
        if 'fold_scores' in result['info']['cv_metrics']:
            fold_scores = result['info']['cv_metrics']['fold_scores']
            # Create cv_results structure for dashboard visualization
            # Format: {metric_name: [fold1_score, fold2_score, ...]}
            self.selection_summary['cv_results'] = {'test_score': fold_scores}
        else:
            self.selection_summary['cv_results'] = None

        # Mark training stage as complete
        self.builder.stage_completion[ModelStage.MODEL_TRAINING] = True

        # Store training results in session state (for downstream components)
        st.session_state.training_complete = True
        st.session_state.training_results = result

        # Log training results
        self.logger.log_calculation(
            "Automated Model Training",
            {
                "optimization_method": self.optimization_method,
                "cv_folds": cv_folds,
                "n_iter": n_iter,
                "best_score": result['info']['best_score'],
                "best_params": result['info']['best_params'],
                "cv_metrics": result['info']['cv_metrics']
            }
        )

        # Log model metrics
        self.logger.log_model_metrics({
            "best_cv_score": result['info']['best_score'],
            "cv_std": result['info'].get('cv_std', None),
            "cv_metrics": result['info']['cv_metrics']
        })

        # Log stability analysis
        stability_level = result['info']['stability_analysis']['level']
        self.logger.log_calculation(
            "Model Stability Analysis",
            result['info']['stability_analysis']
        )

        if stability_level != "High stability":
            self.logger.log_recommendation(
                "Stability Improvement Needed",
                {
                    "stability_level": stability_level,
                    "recommendations": result['info']['stability_analysis']['recommendations']
                }
            )

        # Display results summary if show_analysis is enabled
        if self.show_analysis:
            st.success(f"✅ Model training completed successfully!")
            st.write(f"- **Best Score:** {result['info']['best_score']:.4f}")
            st.write(f"- **Stability Level:** {stability_level}")
            st.write(f"- **Cross-Validation Folds:** {cv_folds}")
            st.write(f"- **Parameter Configurations Tested:** {n_iter}")

    def _calculate_adaptive_training_params(self) -> Tuple[int, int]:
        """
        Calculate adaptive CV folds and iterations based on dataset characteristics.

        If user has provided fixed values (self.cv_folds or self.n_iter are not None),
        those values are used directly. Otherwise, adaptive calculation is performed.

        Adaptive Logic:
        - CV Folds: Based on dataset size (5-10)
          - Small datasets (<1000): 5 folds
          - Medium datasets (1000-10000): 7 folds
          - Large datasets (>10000): 10 folds

        - Iterations: Based on feature complexity (50-100)
          - Low complexity (<10 features): 50 iterations
          - Medium complexity (10-50 features): 75 iterations
          - High complexity (>50 features): 100 iterations

        Returns:
            tuple: (cv_folds, n_iter)
        """
        # Get dataset characteristics
        n_samples = len(self.builder.training_data)
        n_features = len(self.builder.X_train.columns)

        # Determine CV folds (use user-provided value or calculate adaptive)
        if self.cv_folds is not None:
            cv_folds = self.cv_folds
            folds_source = "user-specified"
            folds_reason = f"User selected {cv_folds} folds"
        else:
            # Calculate adaptive CV folds based on dataset size
            if n_samples < 1000:
                cv_folds = 5  # Small dataset - use default
            elif n_samples < 10000:
                cv_folds = 7  # Medium dataset - use more folds
            else:
                cv_folds = 10  # Large dataset - use maximum folds
            folds_source = "adaptive"
            folds_reason = f"Adaptive: Dataset size {n_samples} samples"

        # Determine iterations (use user-provided value or calculate adaptive)
        if self.n_iter is not None:
            n_iter = self.n_iter
            iter_source = "user-specified"
            iter_reason = f"User selected {n_iter} iterations"
        else:
            # Calculate adaptive iterations based on feature complexity
            if n_features < 10:
                n_iter = 50  # Low complexity - use default
            elif n_features < 50:
                n_iter = 75  # Medium complexity - use more iterations
            else:
                n_iter = 100  # High complexity - use maximum iterations
            iter_source = "adaptive"
            iter_reason = f"Adaptive: Feature complexity {n_features} features"

        # Log parameters with source information
        self.logger.log_calculation(
            "Training Parameters Determination",
            {
                "n_samples": n_samples,
                "n_features": n_features,
                "cv_folds": cv_folds,
                "cv_folds_source": folds_source,
                "n_iter": n_iter,
                "n_iter_source": iter_source,
                "rationale": {
                    "folds_reason": folds_reason,
                    "iter_reason": iter_reason
                }
            }
        )

        return cv_folds, n_iter

    def _step_6_final_model_selection(self):
        """
        Step 6: Final Model Selection and Validation

        - Selects between mean score model and stability-adjusted model
        - Sets active model in Builder
        - Validates model is properly trained
        - Prepares model for evaluation stage
        """
        if self.show_analysis:
            st.write("### Step 6: Final Model Selection & Validation")

        # For automated mode, select the model with best mean CV score
        selection_type = "mean_score"

        # Update model configuration
        self.builder.model.update({
            "optimisation_method": self.optimization_method,
            "selection_type": selection_type,
            "active_model": self.builder.model['model'],
            "active_params": self.selection_summary['best_params']
        })

        # Set session state variables for downstream components
        st.session_state.selected_model_type = selection_type
        st.session_state.previous_model_selection = selection_type
        st.session_state.selected_model_stability = self.selection_summary['stability_analysis']

        # Validate model is properly configured
        self._validate_final_model()

        # Store final selection details
        self.selection_summary['final_model_selected'] = True
        self.selection_summary['selection_type'] = selection_type

        # Log final model selection
        self.logger.log_calculation(
            "Final Model Selection",
            {
                "selection_type": selection_type,
                "model_type": self.builder.model['type'],
                "problem_type": self.builder.model['problem_type'],
                "best_params": self.selection_summary['best_params'],
                "best_score": self.selection_summary['best_score']
            }
        )

        self.logger.log_stage_transition(
            "MODEL_TRAINING",
            "MODEL_EVALUATION"
        )

        if self.show_analysis:
            st.success(f"✅ Final model validated and ready for evaluation")

    def _validate_final_model(self):
        """
        Validate that the final model is properly configured and ready for evaluation.

        Checks:
        - Model exists in Builder
        - Model has been trained
        - Model has active_model set
        - Model has best_params set
        - Model has cv_metrics set
        """
        if not self.builder.model:
            raise Exception("Model not found in Builder after training")

        if 'model' not in self.builder.model:
            raise Exception("Model instance not found in model dictionary")

        if 'best_params' not in self.builder.model:
            raise Exception("Best parameters not found in model dictionary")

        if 'cv_metrics' not in self.builder.model:
            raise Exception("CV metrics not found in model dictionary")

        if 'active_model' not in self.builder.model:
            raise Exception("Active model not set in model dictionary")

        # Validate model has been fitted
        try:
            # Try to make a prediction with a single sample (will fail if not fitted)
            sample = self.builder.X_train.iloc[:1] if hasattr(self.builder.X_train, 'iloc') else self.builder.X_train[:1]
            _ = self.builder.model['active_model'].predict(sample)
        except Exception as e:
            raise Exception(f"Model does not appear to be properly fitted: {str(e)}")

    def _step_7_classification_optimizations(self):
        """
        Step 7: Classification-Specific Optimizations

        - Applies calibration for all classification models
        - Applies threshold optimization for binary classification only
        - Skips for regression problems
        - Uses methods from calibration.py and threshold_analysis.py
        """
        # Skip for regression
        if self.selection_summary['problem_type'] == 'regression':
            if self.show_analysis:
                st.info("ℹ️ Classification optimizations not applicable for regression.")
            self.selection_summary['calibration_applied'] = False
            self.selection_summary['threshold_optimized'] = False
            return

        if self.show_analysis:
            st.write("### Step 7: Classification-Specific Optimizations")

        problem_type = self.selection_summary['problem_type']
        is_binary = problem_type in ['binary_classification', 'classification']

        # Step 7a: Calibration (for all classification types)
        self._apply_calibration(is_binary)

        # Step 7b: Threshold optimization (binary classification only)
        if is_binary:
            self._apply_threshold_optimization()
        else:
            if self.show_analysis:
                st.info("ℹ️ Threshold optimization not applicable for multiclass classification.")
            self.selection_summary['threshold_optimized'] = False

    def _apply_calibration(self, is_binary: bool):
        """
        Apply calibration to the trained model using iterative approach.

        Tries all calibration methods (isotonic, sigmoid) and compares them to the
        original uncalibrated model. Selects the best performing option.

        Args:
            is_binary: Whether this is binary classification
        """
        try:
            if self.show_analysis:
                st.write("#### 7a. Model Calibration (Iterative Evaluation)")

            # Import calibration utilities
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.metrics import log_loss, brier_score_loss

            # Get the current (uncalibrated) model
            original_model = self.builder.model['active_model']

            # Analyze current calibration baseline
            if self.show_analysis:
                st.info("🔍 Evaluating calibration options...")

            # Compare calibration methods on out-of-fold training predictions.
            y_test = self.builder.y_train
            X_train = self.builder.X_train

            # Get baseline predictions
            y_pred_proba_original = validation_probabilities(
                original_model, X_train, y_test, self.builder.model.get('resampling_method')
            )

            # Calculate baseline metrics
            if is_binary:
                baseline_brier = brier_score_loss(y_test, y_pred_proba_original[:, 1])
                baseline_logloss = log_loss(y_test, y_pred_proba_original)
            else:
                baseline_brier = None
                baseline_logloss = log_loss(y_test, y_pred_proba_original)

            # Store baseline performance
            calibration_results = {
                'original': {
                    'method': 'original',
                    'log_loss': baseline_logloss,
                    'brier_score': baseline_brier,
                    'model': original_model
                }
            }

            if self.show_analysis:
                st.write(f"**Baseline (Uncalibrated):**")
                st.write(f"- Log Loss: {baseline_logloss:.4f}")
                if baseline_brier is not None:
                    st.write(f"- Brier Score: {baseline_brier:.4f}")

            # Try both calibration methods
            calibration_methods = ['isotonic', 'sigmoid']

            for method in calibration_methods:
                try:
                    # Create calibrated version
                    calibrated_model = CalibratedClassifierCV(
                        resampling_estimator(original_model, self.builder.model.get('resampling_method')),
                        method=method,
                        cv=5  # 5-fold cross-validation for calibration
                    )

                    # Outer validation folds exclude each scored row from both
                    # calibration and estimator fitting.
                    y_pred_proba_calibrated = validation_probabilities(
                        calibrated_model, X_train, y_test
                    )

                    # Calculate metrics
                    if is_binary:
                        calibrated_brier = brier_score_loss(y_test, y_pred_proba_calibrated[:, 1])
                        calibrated_logloss = log_loss(y_test, y_pred_proba_calibrated)
                    else:
                        calibrated_brier = None
                        calibrated_logloss = log_loss(y_test, y_pred_proba_calibrated)

                    # Store results
                    calibration_results[method] = {
                        'method': method,
                        'log_loss': calibrated_logloss,
                        'brier_score': calibrated_brier,
                        'model': calibrated_model
                    }

                    if self.show_analysis:
                        st.write(f"**{method.title()} Calibration:**")
                        st.write(f"- Log Loss: {calibrated_logloss:.4f}")
                        if calibrated_brier is not None:
                            st.write(f"- Brier Score: {calibrated_brier:.4f}")

                except Exception as e:
                    self.logger.log_error(
                        f"Calibration Method Failed: {method}",
                        {"error": str(e)}
                    )
                    if self.show_analysis:
                        st.warning(f"⚠️ {method.title()} calibration failed: {str(e)}")

            # Select the best performing option (lowest log loss is better)
            best_option = min(
                calibration_results.items(),
                key=lambda x: x[1]['log_loss']
            )

            best_method = best_option[0]
            best_metrics = best_option[1]

            # Calculate improvement over baseline
            improvement = (baseline_logloss - best_metrics['log_loss']) / baseline_logloss * 100

            # Log all calibration results
            self.logger.log_calculation(
                "Calibration Method Comparison",
                {
                    "is_binary": is_binary,
                    "baseline_log_loss": float(baseline_logloss),
                    "baseline_brier": float(baseline_brier) if baseline_brier is not None else None,
                    "isotonic_log_loss": float(calibration_results.get('isotonic', {}).get('log_loss', None)) if 'isotonic' in calibration_results else None,
                    "sigmoid_log_loss": float(calibration_results.get('sigmoid', {}).get('log_loss', None)) if 'sigmoid' in calibration_results else None,
                    "best_method": best_method,
                    "best_log_loss": float(best_metrics['log_loss']),
                    "improvement_pct": float(improvement)
                }
            )

            # Decision threshold: Only use calibration if improvement > 1%
            if best_method == 'original' or improvement < 1.0:
                if self.show_analysis:
                    st.success(f"✅ Keeping original uncalibrated model (best option)")
                    st.write(f"- **Decision:** Original model performs best or improvement is minimal ({improvement:.2f}%)")

                self.selection_summary['calibration_applied'] = False
                self.selection_summary['calibration_method'] = 'original'
                self.selection_summary['calibration_reason'] = f"Original model best (improvement would be {improvement:.2f}%)"
                self.selection_summary['calibration_comparison'] = {
                    'methods_tested': list(calibration_results.keys()),
                    'best_method': 'original',
                    'improvement': improvement
                }

                # Log decision
                self.logger.log_calculation(
                    "Calibration Decision",
                    {
                        "decision": "keep_original",
                        "reason": f"Original model best or improvement < 1% ({improvement:.2f}%)",
                        "best_method_tested": best_method
                    }
                )

            else:
                # Apply the best calibration method
                if self.show_analysis:
                    st.success(f"✅ Applying {best_method.title()} calibration")
                    st.write(f"- **Improvement:** {improvement:.2f}% better log loss")
                    st.write(f"- **Original Log Loss:** {baseline_logloss:.4f}")
                    st.write(f"- **Calibrated Log Loss:** {best_metrics['log_loss']:.4f}")

                # Fit the selected calibration configuration on training data only.
                best_metrics['model'].fit(X_train, y_test)
                # Update the model in builder with calibrated version
                self.builder.model['active_model'] = best_metrics['model']
                self.builder.model['is_calibrated'] = True
                self.builder.model['calibration_method'] = best_method
                self.builder.model['calibration_cv_folds'] = 5

                self.selection_summary['calibration_applied'] = True
                self.selection_summary['calibration_method'] = best_method
                self.selection_summary['calibration_improvement'] = improvement
                self.selection_summary['calibration_comparison'] = {
                    'methods_tested': list(calibration_results.keys()),
                    'best_method': best_method,
                    'improvement': improvement,
                    'baseline_log_loss': baseline_logloss,
                    'calibrated_log_loss': best_metrics['log_loss']
                }

                # Log calibration application
                self.logger.log_calculation(
                    "Automated Calibration Applied",
                    {
                        "method": best_method,
                        "is_binary": is_binary,
                        "improvement_pct": float(improvement),
                        "baseline_log_loss": float(baseline_logloss),
                        "calibrated_log_loss": float(best_metrics['log_loss']),
                        "methods_tested": list(calibration_results.keys())
                    }
                )

        except Exception as e:
            self.logger.log_error(
                "Automated Calibration Failed",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
            if self.show_analysis:
                st.error(f"❌ Calibration error: {str(e)}")
            self.selection_summary['calibration_applied'] = False

    def _apply_threshold_optimization(self):
        """
        Apply threshold optimization for binary classification using methods
        from threshold_analysis.py.

        Only applicable for binary classification models.
        """
        try:
            if self.show_analysis:
                st.write("#### 7b. Threshold Optimization")

            # Import threshold analysis utilities
            from components.model_training.threshold_analysis import (
                analyze_current_performance,
                recommend_optimization_criterion,
                find_optimal_threshold_binary
            )

            # Analyze current performance to get baseline metrics
            current_analysis = analyze_current_performance()

            if not current_analysis or not current_analysis.get('success'):
                if self.show_analysis:
                    st.warning("Could not analyze current performance - using default 0.5")
                self.selection_summary['threshold_optimized'] = False
                return

            # Get recommended optimization criterion based on data characteristics
            recommended_criterion = recommend_optimization_criterion(current_analysis, is_binary=True)

            # Map display name to internal criterion name
            criterion_mapping = {
                'F1 Score': 'f1',
                "Youden's J Statistic": 'youden',
                'Precision': 'precision',
                'Recall': 'recall',
                'Accuracy': 'accuracy'
            }

            criterion = criterion_mapping.get(recommended_criterion, 'f1')

            if self.show_analysis:
                st.info(f"🎯 Using recommended criterion: **{recommended_criterion}**")

            # Get predictions for threshold analysis
            y_test = current_analysis['y_true']
            y_prob = current_analysis['y_prob']

            # Get positive class probabilities for binary classification
            if len(y_prob.shape) > 1:
                y_prob_positive = y_prob[:, 1]
            else:
                y_prob_positive = y_prob

            # Find optimal threshold using recommended criterion
            optimal_threshold, threshold_results = find_optimal_threshold_binary(
                y_test,
                y_prob_positive,
                recommended_criterion
            )

            # Get baseline performance with default threshold (0.5)
            baseline_metrics = {
                'accuracy': current_analysis['accuracy'],
                'precision': current_analysis['precision'],
                'recall': current_analysis['recall'],
                'f1': current_analysis['f1']
            }

            # Get the metric value for the selected criterion
            baseline_value = baseline_metrics.get(criterion, baseline_metrics['f1'])

            # Get optimal threshold from results
            if criterion == 'f1':
                optimal_idx = np.argmax(threshold_results['f1'])
                optimal_value = threshold_results['f1'][optimal_idx]
            elif criterion == 'youden':
                optimal_idx = np.argmax(threshold_results['youden_j'])
                optimal_value = threshold_results['youden_j'][optimal_idx]
            elif criterion == 'precision':
                optimal_idx = np.argmax(threshold_results['precision'])
                optimal_value = threshold_results['precision'][optimal_idx]
            elif criterion == 'recall':
                optimal_idx = np.argmax(threshold_results['recall'])
                optimal_value = threshold_results['recall'][optimal_idx]
            elif criterion == 'accuracy':
                optimal_idx = np.argmax(threshold_results['accuracy'])
                optimal_value = threshold_results['accuracy'][optimal_idx]
            else:
                optimal_idx = np.argmax(threshold_results['f1'])
                optimal_value = threshold_results['f1'][optimal_idx]

            optimal_threshold = threshold_results['thresholds'][optimal_idx]

            # Get ALL metrics at the optimal threshold for comparison
            optimal_metrics_all = {
                'accuracy': threshold_results['accuracy'][optimal_idx],
                'precision': threshold_results['precision'][optimal_idx],
                'recall': threshold_results['recall'][optimal_idx],
                'f1': threshold_results['f1'][optimal_idx]
            }

            # Calculate improvement (for Youden's J, we can't directly compare to baseline)
            if criterion == 'youden':
                improvement = optimal_value - (baseline_metrics['recall'] + (1 - baseline_metrics['recall']) - 1)
                improvement = max(0, improvement)
            else:
                improvement = (optimal_value - baseline_value) / baseline_value if baseline_value > 0 else 0

            # Log threshold analysis
            self.logger.log_calculation(
                "Threshold Analysis",
                {
                    "optimal_threshold": float(optimal_threshold),
                    "default_threshold": 0.5,
                    "criterion": criterion,
                    "recommended_criterion": recommended_criterion,
                    "baseline_value": float(baseline_value),
                    "optimal_value": float(optimal_value),
                    "improvement": float(improvement),
                    "all_baseline_metrics": baseline_metrics
                }
            )

            # Check if optimization provides meaningful improvement (>2%)
            if improvement < 0.02:
                if self.show_analysis:
                    st.info(f"ℹ️ Optimal threshold ({optimal_threshold:.3f}) provides minimal improvement ({improvement*100:.1f}%) - keeping default 0.5")
                self.selection_summary['threshold_optimized'] = False
                self.selection_summary['threshold_reason'] = "Minimal improvement over default"
                return

            if self.show_analysis:
                st.info(f"🎯 Applying optimal threshold: **{optimal_threshold:.3f}** (improvement: {improvement*100:.1f}%)")

            # Apply threshold optimization by updating model state
            self.builder.model['threshold_optimized'] = True
            self.builder.model['optimal_threshold'] = float(optimal_threshold)
            self.builder.model['threshold_is_binary'] = True
            self.builder.model['threshold_criterion'] = recommended_criterion

            self.selection_summary['threshold_optimized'] = True
            self.selection_summary['optimal_threshold'] = float(optimal_threshold)
            self.selection_summary['threshold_criterion'] = recommended_criterion
            self.selection_summary['threshold_criterion_internal'] = criterion
            self.selection_summary['threshold_improvement'] = float(improvement)
            self.selection_summary['baseline_metrics'] = baseline_metrics
            self.selection_summary['optimal_metrics'] = optimal_metrics_all  # Store ALL metrics, not just the optimized one

            # Log threshold application
            self.logger.log_calculation(
                "Automated Threshold Optimization Applied",
                {
                    "optimal_threshold": float(optimal_threshold),
                    "criterion": recommended_criterion,
                    "criterion_internal": criterion,
                    "improvement": float(improvement),
                    "baseline_value": float(baseline_value),
                    "optimal_value": float(optimal_value)
                }
            )

            if self.show_analysis:
                st.success(f"✅ Threshold optimized to {optimal_threshold:.3f}")
                st.write(f"- **Criterion:** {recommended_criterion}")
                st.write(f"- **Baseline {criterion}:** {baseline_value:.4f}")
                st.write(f"- **Optimized {criterion}:** {optimal_value:.4f}")
                st.write(f"- **Improvement:** {improvement*100:.1f}%")

        except Exception as e:
            self.logger.log_error(
                "Automated Threshold Optimization Failed",
                {"error": str(e)}
            )
            if self.show_analysis:
                st.error(f"❌ Threshold optimization error: {str(e)}")
            self.selection_summary['threshold_optimized'] = False

    def _generate_summary(self) -> str:
        """
        Generate a comprehensive summary of the automated process.

        Returns:
            str: Markdown-formatted summary
        """
        summary_parts = []

        summary_parts.append("## 🎯 Model Selection & Training Summary\n")

        # Model Selection
        summary_parts.append(f"**Selected Model:** {self.selection_summary['selected_model']}")
        summary_parts.append(f"**Problem Type:** {self.selection_summary['problem_type']}")
        summary_parts.append(f"**Selection Reason:** {self.selection_summary['selection_reason']}\n")

        # Training Results
        summary_parts.append(f"**Best CV Score:** {self.selection_summary['best_score']:.4f}")
        summary_parts.append(f"**Optimization Method:** {self.optimization_method.title()}")
        summary_parts.append(f"**Stability:** {self.selection_summary['stability_analysis']['level']}\n")

        # Imbalance Handling
        if self.selection_summary['imbalance_handled']:
            summary_parts.append(f"**Imbalance Handling:** {self.selection_summary['imbalance_method']}")

        # Calibration
        if self.selection_summary['calibration_applied']:
            summary_parts.append(f"**Calibration:** {self.selection_summary['calibration_method'].title()} ({self.selection_summary['calibration_improvement']:.1f}% improvement)")

        # Threshold
        if self.selection_summary['threshold_optimized']:
            summary_parts.append(f"**Optimal Threshold:** {self.selection_summary['optimal_threshold']:.3f} ({self.selection_summary['threshold_criterion']})")

        return "\n".join(summary_parts)

    def _get_last_completed_step(self) -> str:
        """Get the last completed step for error reporting."""
        if self.selection_summary['threshold_optimized'] or self.selection_summary['calibration_applied']:
            return "Step 7: Classification Optimizations"
        elif self.selection_summary['final_model_selected']:
            return "Step 6: Final Model Selection"
        elif self.selection_summary['training_complete']:
            return "Step 5: Hyperparameter Optimization"
        elif self.selection_summary['imbalance_handled'] is not None:
            return "Step 4: Imbalance Handling"
        elif self.selection_summary['selected_model']:
            return "Step 3: Model Selection"
        elif self.selection_summary['comparison_results']:
            return "Step 2: Model Comparison"
        elif self.selection_summary['recommendation']:
            return "Step 1: Problem Detection"
        else:
            return "Initialization"
