"""Consolidated hyperparameter tuning manager for both Random Search and Optuna methods."""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from joblib import parallel_config
from threadpoolctl import threadpool_limits
import time
import warnings

from components.model_training.utils.parameter_ranges import AdaptiveParameterRanges
from components.model_training.utils.optuna_tuner import OptunaModelTuner
from components.model_training.utils.tuning_commons import StabilityAnalyzer, CVMetricsCalculator, PlotGenerator
from components.model_training.utils.validation_utils import selection_scoring, resampling_estimator, fitted_model, make_cv_splits, worker_budget
from components.model_training.utils.run_state import make_run_record


class HyperparameterTuner:
    """Unified hyperparameter tuning interface for both Random Search and Optuna methods."""

    def __init__(self):
        """Initialize the tuner."""
        self.stability_analyzer = StabilityAnalyzer()
        self.cv_calculator = CVMetricsCalculator()
        self.plot_generator = PlotGenerator()

    def tune_random_search(
        self, model_dict, X_train, y_train, cv_folds=5, n_iter=20,
        progress_callback=None, workers=None,
    ):
        """Evaluate complete candidates; failures never participate in ranking."""
        started = time.monotonic()
        try:
            problem_type = model_dict["problem_type"]
            scoring = selection_scoring(problem_type)
            method = model_dict.get("resampling_method")
            workers = worker_budget(workers)
            if n_iter < 1:
                raise ValueError("At least one parameter configuration is required.")
            if model_dict["type"] == "catboost":
                raise ValueError("Use Optuna for CatBoost parameter tuning.")
            splits = make_cv_splits(X_train, y_train, problem_type, cv_folds, method)
            base = model_dict.get("base_model", model_dict.get("best_model",
                   model_dict.get("original_model", model_dict["model"])))
            estimator = resampling_estimator(base, method)
            distributions = AdaptiveParameterRanges(X_train, y_train, problem_type).get_ranges(
                model_dict["type"], "random_search")
            candidates = list(ParameterSampler(distributions, n_iter=n_iter, random_state=42))
            rows, failures, run_warnings = [], [], []
            prefix = "model__" if method and method != "None (Original Data)" else ""
            def report(phase, completed):
                if progress_callback:
                    progress_callback(dict(phase=phase, completed=completed, total=len(candidates),
                        failed=len(failures), pruned=0, elapsed_seconds=time.monotonic()-started))
            report("search", 0)
            for number, params in enumerate(candidates):
                row = {"params": params}
                try:
                    search = RandomizedSearchCV(
                        estimator=estimator,
                        param_distributions={prefix+k: [v] for k, v in params.items()},
                        n_iter=1, cv=splits, scoring=scoring, n_jobs=workers,
                        random_state=42, return_train_score=False, error_score=np.nan, refit=False,
                    )
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        with parallel_config(backend="loky", inner_max_num_threads=1):
                            with threadpool_limits(limits=1):
                                search.fit(X_train, y_train)
                        run_warnings.extend(str(w.message) for w in caught)
                    scores = [float(search.cv_results_[f"split{i}_test_score"][0])
                              for i in range(cv_folds)]
                    if not np.isfinite(scores).all():
                        raise ValueError("At least one fold failed or produced a non-finite score.")
                    row.update({f"split{i}_test_score": score for i, score in enumerate(scores)})
                    row.update(mean_test_score=float(np.mean(scores)),
                               std_test_score=float(np.std(scores)),
                               adjusted_score=float(np.mean(scores)-np.std(scores)))
                    rows.append(row)
                except Exception as exc:
                    failures.append({"candidate": number + 1, "params": params, "reason": str(exc)})
                report("search", number + 1)
            if not rows:
                return {"success": False, "message": "No parameter configuration completed all folds successfully.",
                        "diagnostics": {"failed_trials": failures, "warnings": list(dict.fromkeys(run_warnings))}}
            cv_results = pd.DataFrame(rows)
            best_index = cv_results["mean_test_score"].idxmax()
            adjusted_index = cv_results["adjusted_score"].idxmax()
            def metrics_for(index):
                scores = [float(cv_results.loc[index, f"split{i}_test_score"]) for i in range(cv_folds)]
                return scores, self.cv_calculator.calculate_cv_metrics(
                    scores, adjusted_score=float(cv_results.loc[index, "adjusted_score"]))
            fold_scores, cv_metrics = metrics_for(best_index)
            adjusted_scores, adjusted_metrics = metrics_for(adjusted_index)
            best_params = cv_results.loc[best_index, "params"]
            adjusted_params = cv_results.loc[adjusted_index, "params"]
            report("refit", len(candidates))
            def fit_candidate(params):
                candidate = resampling_estimator(base, method, workers=workers)
                candidate.set_params(**{prefix+k: v for k, v in params.items()})
                with threadpool_limits(limits=workers):
                    candidate.fit(X_train, y_train)
                return fitted_model(candidate)
            best_model = fit_candidate(best_params)
            same_model = best_index == adjusted_index
            adjusted_model = best_model if same_model else fit_candidate(adjusted_params)
            info = {
                "best_params": best_params, "scoring": scoring,
                "best_score": cv_metrics["mean_score"], "best_std": cv_metrics["std_score"],
                "all_results": {
                    "mean_test_scores": cv_results["mean_test_score"].tolist(),
                    "std_test_scores": cv_results["std_test_score"].tolist(),
                    "params_tested": cv_results["params"].tolist(),
                },
                "cv_metrics": cv_metrics,
                "cv_plots": self.plot_generator.create_cv_distribution_plots(fold_scores, cv_metrics),
                "stability_analysis": self.stability_analyzer.create_stability_analysis(cv_metrics, fold_scores),
                "adjusted_stability_analysis": self.stability_analyzer.create_stability_analysis(adjusted_metrics, adjusted_scores),
                "is_same_model": same_model, "adjusted_cv_metrics": adjusted_metrics,
                "adjusted_params": adjusted_params,
                "diagnostics": {"completed": len(rows), "failed": len(failures), "pruned": 0,
                                "failed_trials": failures, "warnings": list(dict.fromkeys(run_warnings))},
                "training_time": time.monotonic() - started,
                "training_run": make_run_record(model_dict, X_train, y_train, "random_search",
                                                cv_folds, n_iter, workers),
            }
            info['training_run']['evaluated_trials'] = len(candidates)
            if problem_type == "regression":
                info.update(train_r2=float(best_model.score(X_train, y_train)),
                            val_r2=cv_metrics["mean_score"])
            report("complete", len(candidates))
            return {"success": True, "message": "Hyperparameter tuning completed successfully",
                    "info": info, "best_estimator": best_model, "adjusted_estimator": adjusted_model,
                    "best_params": best_params, "optimisation_method": "random_search"}
        except Exception as exc:
            return {"success": False, "message": f"Error during hyperparameter tuning: {exc}"}

    def tune_optuna(
        self,
        model_dict: Dict[str, Any],
        X_train,
        y_train,
        cv_folds: int = 5,
        n_trials: int = 50,
        progress_callback=None, workers=None,
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
                progress_callback=progress_callback, workers=workers,
                base_model=model_dict.get('base_model', model_dict.get('best_model', model_dict.get('original_model', model_dict['model']))),
            )

            # Run optimisation
            result = tuner.optimize()

            if not result["success"]:
                return {
                    "success": False,
                    "message": f"Error during hyperparameter optimisation: {result['message']}",
                    "diagnostics": result.get("diagnostics", {})
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
                    "diagnostics": result["diagnostics"],
                    "training_time": result["training_time"],
                    "training_run": make_run_record(model_dict, X_train, y_train, "optuna", cv_folds, n_trials, worker_budget(workers)),
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
                            "n_failed_trials": result['diagnostics']['failed'],
                            "study_duration": result["optimisation_history"]["study_duration"]
                        },
                        "values": result["optimisation_history"]["values"],
                        "params": result["optimisation_history"]["params"],
                        "trial_numbers": result["optimisation_history"]["trial_numbers"]
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
