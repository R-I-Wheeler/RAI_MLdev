"""Consolidated hyperparameter tuning manager for both Random Search and Optuna methods."""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from copy import deepcopy

from components.model_training.utils.parameter_ranges import AdaptiveParameterRanges
from components.model_training.utils.optuna_tuner import OptunaModelTuner
from components.model_training.utils.tuning_commons import StabilityAnalyzer, CVMetricsCalculator, PlotGenerator
from components.model_training.utils.validation_utils import selection_scoring, resampling_estimator, fitted_model


class HyperparameterTuner:
    """Unified hyperparameter tuning interface for both Random Search and Optuna methods."""

    def __init__(self):
        """Initialize the tuner."""
        self.stability_analyzer = StabilityAnalyzer()
        self.cv_calculator = CVMetricsCalculator()
        self.plot_generator = PlotGenerator()

    def tune_random_search(
        self,
        model_dict: Dict[str, Any],
        X_train,
        y_train,
        cv_folds: int = 5,
        n_iter: int = 20
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using RandomizedSearchCV.

        Args:
            model_dict: Model dictionary containing 'model', 'type', 'problem_type'
            X_train: Training features
            y_train: Training target
            cv_folds: Number of cross-validation folds
            n_iter: Number of parameter configurations to test

        Returns:
            Dictionary with tuning results in the expected format
        """
        try:
            problem_type = model_dict.get("problem_type", "unknown")
            if problem_type == "unknown":
                return {
                    "success": False,
                    "message": "Unable to determine problem type"
                }

            # Initialize adaptive parameter ranges with caching
            cache_key = f"{model_dict['type']}_{problem_type}_{X_train.shape}_{hash(str(X_train.dtypes.tolist()) if hasattr(X_train, 'dtypes') else 'array')}"

            # Check if parameter ranges are cached in session state
            if 'param_ranges_cache' not in st.session_state:
                st.session_state.param_ranges_cache = {}

            if cache_key not in st.session_state.param_ranges_cache:
                param_ranges = AdaptiveParameterRanges(X_train, y_train, problem_type)
                param_distributions = param_ranges.get_ranges(model_dict["type"], "random_search")
                st.session_state.param_ranges_cache[cache_key] = param_distributions
            else:
                param_distributions = st.session_state.param_ranges_cache[cache_key]

            scoring = selection_scoring(problem_type)
            resampling_method = model_dict.get("resampling_method")
            estimator = resampling_estimator(model_dict["model"], resampling_method)
            if resampling_method and resampling_method != "None (Original Data)":
                param_distributions = {f"model__{key}": value for key, value in param_distributions.items()}

            # Models that don't work well with RandomizedSearchCV
            model_type = model_dict.get("type", "")

            # CatBoost doesn't work well with sklearn's RandomizedSearchCV
            # Recommend using Optuna instead
            if model_type == "catboost":
                return {
                    "success": False,
                    "message": "CatBoost is not compatible with Random Search hyperparameter tuning. Please use Optuna optimization method instead, which works perfectly with CatBoost and often provides better results."
                }

            # For models with early stopping, we need special handling
            # XGBoost, LightGBM, and HistGradientBoosting support early stopping
            models_with_early_stopping = ["xgboost", "lightgbm", "hist_gradient_boosting"]

            if model_type in models_with_early_stopping:
                # NOTE: Random Search has limited early stopping support
                # For best results with XGBoost/LightGBM, use Optuna optimization instead

                # HistGradientBoosting handles early stopping natively through parameters
                if model_type == "hist_gradient_boosting":
                    # HistGradientBoosting's early_stopping parameter works with RandomizedSearchCV
                    # It uses internal validation split based on validation_fraction parameter
                    random_search = RandomizedSearchCV(
                        estimator=estimator,
                        param_distributions=param_distributions,
                        n_iter=n_iter,
                        cv=cv_folds,
                        scoring=scoring,
                        n_jobs=-1,  # Can use parallel for HistGradientBoosting
                        random_state=42,
                        return_train_score=True,
                        error_score=0
                    )
                    random_search.fit(X_train, y_train)
                else:
                    # XGBoost and LightGBM have limitations with RandomizedSearchCV
                    # early_stopping_rounds parameter is in param_distributions but
                    # RandomizedSearchCV can't pass it to fit() method dynamically
                    # Therefore, we remove it from the search and use default behavior

                    # Remove early stopping parameters that can't be used in Random Search
                    params_to_remove = ['early_stopping_rounds', 'model__early_stopping_rounds']
                    param_distributions_filtered = {
                        k: v for k, v in param_distributions.items()
                        if k not in params_to_remove
                    }

                    # Show informational message about limitation
                    if param_distributions_filtered != param_distributions:
                        try:
                            st.info(f"""
                                ℹ️ **Note**: Early stopping parameters are not fully supported in Random Search for {model_type.upper()}.

                                Random Search will tune other parameters, but for optimal early stopping support
                                and faster training, consider using **Optuna optimization** instead.
                            """)
                        except:
                            # If st.info fails (e.g., not in Streamlit context), just continue
                            pass

                    random_search = RandomizedSearchCV(
                        estimator=estimator,
                        param_distributions=param_distributions_filtered,
                        n_iter=n_iter,
                        cv=cv_folds,
                        scoring=scoring,
                        n_jobs=-1,
                        random_state=42,
                        return_train_score=True,
                        error_score=0
                    )
                    random_search.fit(X_train, y_train)
            else:
                # Standard random search for models without early stopping
                random_search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=param_distributions,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=-1,
                    random_state=42,
                    return_train_score=True,
                    error_score=0
                )

                # Perform the search
                random_search.fit(X_train, y_train)

            # Calculate CV metrics using actual fold scores
            cv_results = pd.DataFrame(random_search.cv_results_)
            # Expose the original estimator's parameter names to downstream callers.
            best_params = {key.removeprefix("model__"): value for key, value in random_search.best_params_.items()}
            cv_results['params'] = cv_results['params'].map(
                lambda params: {key.removeprefix("model__"): value for key, value in params.items()}
            )
            best_index = random_search.best_index_

            # Get all fold scores for the best parameters first
            fold_scores = []
            for i in range(cv_folds):
                col_name = f'split{i}_test_score'
                if col_name in cv_results.columns:
                    fold_scores.append(float(cv_results.loc[best_index, col_name]))

            # Calculate CV metrics using actual fold scores
            cv_metrics = self.cv_calculator.calculate_cv_metrics(fold_scores)

            # Calculate adjusted scores for all parameter combinations
            alpha = 1.0  # Coefficient for stability penalty
            adjusted_scores = []

            for i in range(len(cv_results)):
                # Get all fold scores for this configuration
                config_fold_scores = []
                for j in range(cv_folds):
                    col_name = f'split{j}_test_score'
                    if col_name in cv_results.columns:
                        config_fold_scores.append(float(cv_results.loc[i, col_name]))

                # Calculate mean and std for this configuration
                mean_score = np.mean(config_fold_scores)
                std_score = np.std(config_fold_scores)

                # Calculate adjusted score
                adjusted_score = mean_score - (alpha * std_score)
                adjusted_scores.append(adjusted_score)

            # Add adjusted scores to results
            cv_results['adjusted_score'] = adjusted_scores

            # Find best configuration by adjusted score
            best_adjusted_index = cv_results['adjusted_score'].idxmax()
            best_adjusted_params = cv_results.loc[best_adjusted_index, 'params']
            best_adjusted_score = cv_results.loc[best_adjusted_index, 'adjusted_score']

            # Get fold scores for best adjusted model
            adjusted_fold_scores = []
            for i in range(cv_folds):
                col_name = f'split{i}_test_score'
                if col_name in cv_results.columns:
                    adjusted_fold_scores.append(float(cv_results.loc[best_adjusted_index, col_name]))

            # Calculate metrics for best adjusted model
            adjusted_cv_metrics = self.cv_calculator.calculate_cv_metrics(
                adjusted_fold_scores,
                adjusted_score=float(best_adjusted_score)
            )

            # Check if best by mean and best by adjusted are the same
            same_model = (best_index == best_adjusted_index)

            # Create CV plots
            cv_plots = self.plot_generator.create_cv_distribution_plots(fold_scores, cv_metrics)

            # Create stability analysis
            stability_analysis = self.stability_analyzer.create_stability_analysis(
                cv_metrics, fold_scores
            )

            # Prepare results summary
            tuning_results = {
                "best_params": best_params,
                "scoring": scoring,
                "best_score": cv_metrics["mean_score"],
                "best_std": cv_metrics["std_score"],
                "all_results": {
                    "mean_test_scores": cv_results["mean_test_score"].tolist(),
                    "std_test_scores": cv_results["std_test_score"].tolist(),
                    "params_tested": cv_results["params"].tolist()
                },
                "cv_metrics": cv_metrics,
                "cv_plots": cv_plots,
                "stability_analysis": stability_analysis,
                "is_same_model": same_model,
                "adjusted_cv_metrics": adjusted_cv_metrics,
                "adjusted_params": best_adjusted_params
            }

            # Add regression-specific metrics if applicable
            if problem_type == "regression":
                # Fit the best model to calculate R2 scores
                best_model = deepcopy(model_dict["model"])
                best_model.set_params(**best_params)
                best_model.fit(X_train, y_train)

                r2_train = best_model.score(X_train, y_train)
                r2_val = cv_metrics["mean_score"]

                tuning_results.update({
                    "train_r2": float(r2_train),
                    "val_r2": float(r2_val)
                })

            return {
                "success": True,
                "message": "Hyperparameter tuning completed successfully",
                "info": tuning_results,
                "best_estimator": fitted_model(random_search.best_estimator_),
                "best_params": best_params,
                "optimisation_method": "random_search"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error during hyperparameter tuning: {str(e)}"
            }

    def tune_optuna(
        self,
        model_dict: Dict[str, Any],
        X_train,
        y_train,
        cv_folds: int = 5,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna optimization.

        Args:
            model_dict: Model dictionary containing 'model', 'type', 'problem_type'
            X_train: Training features
            y_train: Training target
            cv_folds: Number of cross-validation folds
            n_trials: Number of optimization trials

        Returns:
            Dictionary with tuning results in the expected format
        """
        try:
            # Create Optuna tuner
            tuner = OptunaModelTuner(
                X_train=X_train,
                y_train=y_train,
                model_type=model_dict["type"],
                problem_type=model_dict["problem_type"],
                cv_folds=cv_folds,
                n_trials=n_trials,
                resampling_method=model_dict.get("resampling_method"),
            )

            # Run optimisation
            result = tuner.optimize()

            if not result["success"]:
                return {
                    "success": False,
                    "message": f"Error during hyperparameter optimisation: {result['message']}"
                }

            # Get optimisation plots
            plot_result = tuner.get_optimisation_plots()

            # Calculate stability metrics
            stability_score = 1 - result["cv_std"]

            # Create stability analysis
            stability_analysis = self.stability_analyzer.create_optuna_stability_analysis(
                result["cv_results"], result["cv_mean"], result["cv_std"], stability_score
            )

            # Prepare final result with optimisation plots
            final_result = {
                "success": True,
                "message": "Hyperparameter optimisation completed successfully",
                "info": {
                    "scoring": selection_scoring(model_dict["problem_type"]),
                    "best_score": result["best_score"],
                    "best_params": result["best_params"],
                    "cv_metrics": {
                        "mean_score": result["cv_mean"],
                        "std_score": result["cv_std"],
                        "fold_scores": result["cv_results"].tolist()
                    },
                    "stability_analysis": stability_analysis,
                    "optimisation_plots": {
                        "history": plot_result["history"] if plot_result["success"] else None,
                        "param_importance": plot_result["param_importance"] if plot_result["success"] else None,
                        "timeline": plot_result["timeline"] if plot_result["success"] else None,
                        "param_importances_fig": plot_result["param_importances_fig"] if plot_result["success"] else None
                    },
                    "optimisation_history": {
                        "metrics": {
                            "n_complete_trials": len([t for t in result["optimisation_history"]["trials"] if t["state"] == "COMPLETE"]),
                            "n_pruned_trials": len([t for t in result["optimisation_history"]["trials"] if t["state"] == "PRUNED"]),
                            "study_duration": result["optimisation_history"]["study_duration"]
                        },
                        "values": result["optimisation_history"]["values"],
                        "params": result["optimisation_history"]["params"]
                    }
                },
                "best_model": result["model"],
                "best_params": result["best_params"],
                "optimisation_method": "optuna"
            }

            return final_result

        except Exception as e:
            return {
                "success": False,
                "message": f"Error during hyperparameter optimisation: {str(e)}"
            }
