import optuna
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from threadpoolctl import threadpool_limits
import time
import warnings
from sklearn.metrics import get_scorer
import numpy as np
from typing import Dict, Any, Callable
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, \
    HistGradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import plotly.graph_objects as go
import pandas as pd
from components.model_training.utils.parameter_ranges import AdaptiveParameterRanges
from components.model_training.utils.validation_utils import (
    selection_scoring, resampling_estimator, fitted_model, make_cv_splits, worker_budget,
)

class OptunaModelTuner:
    def __init__(self, X_train, y_train, model_type: str, problem_type: str, cv_folds: int = 5, n_trials: int = 50, resampling_method=None, progress_callback=None, workers=None, base_model=None):
        self.X_train = X_train
        self.y_train = y_train
        self.model_type = model_type
        self.problem_type = problem_type
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.resampling_method = resampling_method
        self.progress_callback = progress_callback
        self.workers = worker_budget(workers)
        self.base_model = base_model
        self.study = None
        self.best_model = None
        self.best_params = None
        self.cv_results = None
        
        # Initialize adaptive parameter ranges
        self.param_ranges = AdaptiveParameterRanges(X_train, y_train, problem_type)
        
        # Convert to numpy if pandas
        self.X_is_pandas = isinstance(X_train, pd.DataFrame)
        self.y_is_pandas = isinstance(y_train, pd.Series)
        self.X_train_values = X_train.values if self.X_is_pandas else X_train
        self.y_train_values = y_train.values if self.y_is_pandas else y_train
        
        # Calculate dataset characteristics for adaptive search space
        self.n_samples, self.n_features = self.X_train_values.shape
        self.class_distribution = None if problem_type == "regression" else pd.Series(self.y_train_values).value_counts().to_numpy()
        self.is_high_dimensional = self.n_features > 100
        self.is_small_dataset = self.n_samples < 1000
        self.feature_density = np.mean(np.abs(self.X_train_values) > 0)  # Measure of data sparsity

    def _get_objective_func(self) -> Callable:
        """Get the appropriate objective function based on model type."""
        
        def objective(trial) -> float:
            params = self._get_trial_params(trial)
            scorer = get_scorer(selection_scoring(self.problem_type))
            
            # Implement intermediate scoring for pruning
            scores = []
            for i, (train_idx, val_idx) in enumerate(self.cv_splits):
                # Handle both pandas and numpy data types
                if self.X_is_pandas:
                    X_fold_train = self.X_train.iloc[train_idx]
                    X_fold_val = self.X_train.iloc[val_idx]
                else:
                    X_fold_train = self.X_train_values[train_idx]
                    X_fold_val = self.X_train_values[val_idx]
                
                if self.y_is_pandas:
                    y_fold_train = self.y_train.iloc[train_idx]
                    y_fold_val = self.y_train.iloc[val_idx]
                else:
                    y_fold_train = self.y_train_values[train_idx]
                    y_fold_val = self.y_train_values[val_idx]
                
                # A fresh estimator and sampler see only this fold's training rows.
                model = resampling_estimator(self._create_model(params), self.resampling_method, workers=self.workers)
                with threadpool_limits(limits=self.workers):
                    model.fit(X_fold_train, y_fold_train)
                
                fold_score = scorer(model, X_fold_val, y_fold_val)
                if not np.isfinite(fold_score):
                    raise ValueError('A validation fold produced a non-finite score.')
                scores.append(float(fold_score))
                
                # Report intermediate value for pruning
                trial.report(float(np.mean(scores)), i + 1)
                
                # Handle pruning based on intermediate results
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            trial.set_user_attr('fold_scores', scores)
            return float(np.mean(scores))
        
        return objective

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define parameter search space for each model type with adaptive ranges."""
        ranges = self.param_ranges.get_ranges(self.model_type, "optuna")
        params = {}
        
        for param_name, range_info in ranges.items():
            if self.model_type == 'random_forest' and param_name == 'max_samples' and not params.get('bootstrap', True):
                continue
            param_type = range_info[0]
            if param_type == "int":
                _, low, high = range_info
                params[param_name] = trial.suggest_int(param_name, low, high)
            elif param_type == "float":
                _, low, high, is_log = range_info
                params[param_name] = trial.suggest_float(param_name, low, high, log=is_log)
            elif param_type == "categorical_tuple":
                # Handle tuple categories directly
                _, categories = range_info
                selected = trial.suggest_categorical(param_name, categories)
                params[param_name] = selected  # Already in tuple format
            elif param_type == "categorical":
                _, categories = range_info
                params[param_name] = trial.suggest_categorical(param_name, categories)
        
        return params

    def _create_model(self, params: Dict[str, Any]) -> Any:
        """Create a model instance with given parameters."""
        # Use all params directly - no early stopping parameters to filter anymore
        init_params = dict(params)
        if self.model_type == 'random_forest' and not init_params.get('bootstrap', True):
            init_params['max_samples'] = None
        if self.base_model is not None:
            return clone(self.base_model).set_params(**init_params)

        if self.model_type == "logistic_regression":
            return LogisticRegression(random_state=42, n_jobs=-1, **init_params)
        elif self.model_type == "naive_bayes":
            # GaussianNB for classification with continuous features
            return GaussianNB(**init_params)
        elif self.model_type == "linear_regression":
            return LinearRegression(n_jobs=-1)
        elif self.model_type == "ridge_regression":
            # Ridge regression with L2 regularization (regression only)
            return Ridge(random_state=42, **init_params)
        elif self.model_type == "decision_tree":
            # Handle both binary and multiclass classification
            if self.problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                return DecisionTreeClassifier(random_state=42, **init_params)
            else:
                return DecisionTreeRegressor(random_state=42, **init_params)
        elif self.model_type == "random_forest":
            # Handle both binary and multiclass classification
            if self.problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                return RandomForestClassifier(random_state=42, n_jobs=-1, **init_params)
            else:
                return RandomForestRegressor(random_state=42, n_jobs=-1, **init_params)
        elif self.model_type == "mlp":
            # hidden_layer_sizes is already a tuple from _get_trial_params
            # Handle both binary and multiclass classification
            if self.problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                return MLPClassifier(random_state=42, max_iter=1000, **init_params)
            else:
                return MLPRegressor(random_state=42, max_iter=1000, **init_params)
        elif self.model_type == "xgboost":
            # Handle both binary and multiclass classification
            if self.problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                return xgb.XGBClassifier(random_state=42, nthread=-1, **init_params)
            else:
                return xgb.XGBRegressor(random_state=42, nthread=-1, **init_params)
        elif self.model_type == "lightgbm":
            # Optimize data access pattern based on dataset shape
            # Row-wise is faster for tall datasets (many rows, few columns)
            force_row_wise = self.n_samples > (4 * self.n_features) if hasattr(self, 'n_features') else False

            # Handle both binary and multiclass classification
            if self.problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                return lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, force_row_wise=force_row_wise, **init_params)
            else:
                return lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, force_row_wise=force_row_wise, **init_params)
        elif self.model_type == "hist_gradient_boosting":
            # Handle both binary and multiclass classification
            if self.problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                return HistGradientBoostingClassifier(random_state=42, **init_params)
            else:
                return HistGradientBoostingRegressor(random_state=42, **init_params)
        elif self.model_type == "catboost":
            # Handle both binary and multiclass classification
            if self.problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                return CatBoostClassifier(random_state=42, verbose=False, **init_params)
            else:
                return CatBoostRegressor(random_state=42, verbose=False, **init_params)
        else:  # linear_regression
            return {}

    def optimize(self) -> Dict[str, Any]:
        """Run trials sequentially with a bounded estimator worker budget."""
        started = time.monotonic()
        failures, run_warnings = [], []
        try:
            if self.n_trials < 1:
                raise ValueError("At least one optimisation trial is required.")
            self.cv_splits = make_cv_splits(
                self.X_train, self.y_train, self.problem_type, self.cv_folds, self.resampling_method)
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=min(10, self.n_trials)),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=min(5, self.n_trials), n_warmup_steps=1),
            )
            self.study = study
            objective = self._get_objective_func()
            def report(phase):
                if self.progress_callback:
                    self.progress_callback(dict(
                        phase=phase, completed=len(study.trials), total=self.n_trials,
                        failed=sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
                        pruned=sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
                        elapsed_seconds=time.monotonic()-started))
            report("search")
            for _ in range(self.n_trials):
                trial = study.ask()
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        try:
                            value = objective(trial)
                        finally:
                            run_warnings.extend(str(w.message) for w in caught)
                    study.tell(trial, value)
                except optuna.TrialPruned:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                except Exception as exc:
                    trial.set_user_attr("failure_reason", str(exc))
                    failures.append({"candidate": trial.number + 1, "params": trial.params, "reason": str(exc)})
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                report("search")
            complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            diagnostics = {
                "completed": len(complete), "failed": len(failures),
                "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
                "failed_trials": failures, "warnings": list(dict.fromkeys(run_warnings)),
            }
            if not complete:
                return {"success": False, "message": "No optimisation trial completed all folds successfully.",
                        "diagnostics": diagnostics}
            self.best_params = dict(study.best_params)
            if self.model_type == "random_forest" and not self.best_params.get("bootstrap", True):
                self.best_params["max_samples"] = None
            report("refit")
            final_estimator = resampling_estimator(
                self._create_model(self.best_params), self.resampling_method, workers=self.workers)
            with threadpool_limits(limits=self.workers):
                final_estimator.fit(self.X_train, self.y_train)
            self.best_model = fitted_model(final_estimator)
            # Reuse the winning trial's scores instead of fitting every fold again.
            self.cv_results = np.asarray(study.best_trial.user_attrs["fold_scores"], dtype=float)
            duration = max(t.datetime_complete for t in study.trials) - min(t.datetime_start for t in study.trials)
            # Only fully evaluated trials are comparable in performance/parameter plots.
            history = {
                "values": [t.value for t in complete], "params": [t.params for t in complete],
                "trial_numbers": [t.number for t in complete],
                "trials": [{"state": t.state.name} for t in study.trials],
                "study_duration": duration,
                "metrics": {"n_complete_trials": len(complete), "n_pruned_trials": diagnostics["pruned"],
                            "n_failed_trials": len(failures), "study_duration": duration},
            }
            report("complete")
            return {
                "success": True, "message": "Optimisation completed successfully",
                "best_params": self.best_params, "best_score": float(study.best_value),
                "cv_mean": float(self.cv_results.mean()), "cv_std": float(self.cv_results.std()),
                "cv_results": self.cv_results, "optimisation_history": history, "model": self.best_model,
                "diagnostics": diagnostics, "training_time": time.monotonic()-started,
            }
        except Exception as exc:
            return {"success": False, "message": f"Error during optimisation: {exc}",
                    "diagnostics": {"failed_trials": failures, "warnings": list(dict.fromkeys(run_warnings))}}

    def get_optimisation_plots(self) -> Dict[str, Any]:
        """Get optimisation visualization plots."""
        try:
            if not hasattr(self, 'study') or self.study is None:
                print("Debug: No study available")
                return {
                    "success": False,
                    "message": "No optimisation study available"
                }

            # Get optimisation history
            df = self.study.trials_dataframe()
            df = df.loc[df['state'] == 'COMPLETE'].reset_index(drop=True)
            print(f"Debug: Number of trials in dataframe: {len(df)}")
            
            # Create history plot
            history_fig = go.Figure()
            
            # Find best trial
            best_trial_idx = int(df.loc[df['value'].idxmax(), 'number'])
            best_value = df['value'].max()
            
            # Add optimisation history line with highlighted best point
            history_fig.add_trace(go.Scatter(
                x=df['number'],
                y=df['value'],
                mode='markers+lines',
                name='Trial Score',
                marker=dict(
                    size=8,
                    color=df['value'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Score')
                ),
                line=dict(color='rgba(70, 130, 180, 0.3)'),
                hovertemplate="Trial: %{x}<br>Score: %{y:.4f}<extra></extra>"
            ))
            
            # Add best value line
            best_values = [max(df['value'][:i+1]) for i in range(len(df))]
            history_fig.add_trace(go.Scatter(
                x=df['number'],
                y=best_values,
                mode='lines',
                name='Best Score',
                line=dict(dash='dash', color='red', width=2)
            ))
            
            # Highlight the best point
            history_fig.add_trace(go.Scatter(
                x=[best_trial_idx],
                y=[best_value],
                mode='markers',
                name='Best Trial',
                marker=dict(
                    symbol='star',
                    size=20,
                    color='#2ecc71',
                    line=dict(color='black', width=2)
                ),
                hovertemplate="Best Trial<br>Score: %{y:.4f}<extra></extra>"
            ))
            
            # Add annotation for the best point
            history_fig.add_annotation(
                x=best_trial_idx,
                y=best_value,
                text=f"Best Score: {best_value:.4f} ★",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                font=dict(size=12, color='black', weight='bold'),
                bgcolor='#2ecc71',
                bordercolor='black',
                borderwidth=2,
                borderpad=4,
                opacity=0.8
            )
            
            history_fig.update_layout(
                title='Optimisation History',
                xaxis_title='Trial',
                yaxis_title='Score',
                showlegend=True,
                template='plotly_white',
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )

            # Create parallel coordinates plot for parameter importance
            param_cols = [col for col in df.columns if col.startswith('params_')]
            print(f"Debug: Parameter columns found: {param_cols}")
            
            if param_cols:
                dimensions = []
                
                # Get best parameters
                best_params = {col.replace('params_', ''): df.loc[best_trial_idx, col] 
                             for col in param_cols}
                
                for col in param_cols:
                    param_values = df[col].values
                    param_name = col.replace('params_', '')
                    best_param_value = best_params[param_name]
                    
                    try:
                        # Special handling for batch_size which can be int or 'auto'
                        if param_name == 'batch_size':
                            unique_values = sorted([str(v) for v in np.unique(param_values)])
                            value_map = {val: idx for idx, val in enumerate(unique_values)}
                            values = [value_map[str(val)] for val in param_values]
                            dimensions.append(dict(
                                range=[0, len(unique_values) - 1],
                                label=f"{param_name}<br>(Best: {best_param_value})",
                                values=values,
                                ticktext=unique_values,
                                tickvals=list(range(len(unique_values)))
                            ))
                            continue

                        # Try converting to numeric first
                        numeric_values = pd.to_numeric(param_values, errors='coerce')
                        if not pd.isnull(numeric_values).any():
                            # All values are numeric
                            min_val = float(numeric_values.min())
                            max_val = float(numeric_values.max())
                            tick_count = 6
                            step = (max_val - min_val) / (tick_count - 1)
                            tick_vals = [min_val + i * step for i in range(tick_count)]
                            tick_text = [f"{val:.3f}" for val in tick_vals]
                            
                            dimensions.append(dict(
                                range=[min_val, max_val],
                                label=f"{param_name}<br>(Best: {best_param_value:.3f})",
                                values=numeric_values.astype(float),
                                tickvals=tick_vals,
                                ticktext=tick_text
                            ))
                            continue

                        # If not all numeric, treat as categorical
                        unique_values = sorted([str(v) for v in np.unique(param_values)])
                        value_map = {val: idx for idx, val in enumerate(unique_values)}
                        values = [value_map[str(val)] for val in param_values]
                        dimensions.append(dict(
                            range=[0, len(unique_values) - 1],
                            label=f"{param_name}<br>(Best: {best_param_value})",
                            values=values,
                            ticktext=unique_values,
                            tickvals=list(range(len(unique_values)))
                        ))

                    except Exception as e:
                        print(f"Debug: Error processing parameter {param_name}: {str(e)}")
                        continue
                
                # Add score dimension with improved formatting
                score_min = float(df['value'].min())
                score_max = float(df['value'].max())
                score_tick_count = 6
                score_step = (score_max - score_min) / (score_tick_count - 1)
                score_tick_vals = [score_min + i * score_step for i in range(score_tick_count)]
                score_tick_text = [f"{val:.3f}" for val in score_tick_vals]
                
                dimensions.append(
                    dict(
                        range=[score_min, score_max],
                        label=f'Score<br>(Best: {best_value:.3f})',
                        values=df['value'].astype(float),
                        tickvals=score_tick_vals,
                        ticktext=score_tick_text
                    )
                )
                
                if dimensions:  # Only create plot if we have valid dimensions
                    param_fig = go.Figure(data=go.Parcoords(
                        line=dict(
                            color=df['value'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title='Score')
                        ),
                        dimensions=dimensions
                    ))
                    
                    param_fig.update_layout(
                        title='Parameter Importance',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=800,
                        margin=dict(l=120, r=120, t=80, b=80),
                        font=dict(size=10),
                        title_x=0.5,
                        title_y=0.95
                    )
                else:
                    print("Debug: No valid dimensions for parallel coordinates plot")
                    param_fig = None
            else:
                print("Debug: No parameter columns found")
                param_fig = None

            timeline_fig = optuna.visualization.plot_timeline(self.study)
            try:
                param_importances_fig = optuna.visualization.plot_param_importances(self.study)
            except (ValueError, RuntimeError):
                param_importances_fig = None
            # Prepare optimisation history data
            history_data = {
                'params': [t.params for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                'values': df['value'].tolist()
            }

            result = {
                "success": True,
                "history": history_fig,
                "param_importance": param_fig,
                "param_importances_fig": param_importances_fig,
                "timeline": timeline_fig,
                "history_data": history_data
            }
            print("Debug: Returning optimisation plots successfully")
            return result
            
        except Exception as e:
            print(f"Debug: Error in get_optimisation_plots: {str(e)}")
            return {
                "success": False,
                "message": f"Error creating optimisation plots: {str(e)}"
            }
