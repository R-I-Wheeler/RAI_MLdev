import optuna
from sklearn.model_selection import cross_val_score
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
    selection_scoring, resampling_estimator, fitted_model,
)

class OptunaModelTuner:
    def __init__(self, X_train, y_train, model_type: str, problem_type: str, cv_folds: int = 5, n_trials: int = 50, resampling_method=None):
        self.X_train = X_train
        self.y_train = y_train
        self.model_type = model_type
        self.problem_type = problem_type
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.resampling_method = resampling_method
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
        self.class_distribution = None if problem_type == "regression" else np.bincount(self.y_train_values)
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
                model = resampling_estimator(self._create_model(params), self.resampling_method)
                model.fit(X_fold_train, y_fold_train)
                
                fold_score = scorer(model, X_fold_val, y_fold_val)
                scores.append(fold_score)
                
                # Report intermediate value for pruning
                trial.report(fold_score, i)
                
                # Handle pruning based on intermediate results
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        return objective

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define parameter search space for each model type with adaptive ranges."""
        ranges = self.param_ranges.get_ranges(self.model_type, "optuna")
        params = {}
        
        for param_name, range_info in ranges.items():
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
        init_params = params

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
        """Run Optuna optimisation and return results."""
        try:
            # Configure pruning based on dataset characteristics
            # Use HyperbandPruner which works better with TPESampler (our default)
            n_startup_trials = max(10, self.n_trials // 5)  # Increased for better exploration

            # HyperbandPruner for better performance with TPESampler
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=1,  # Minimum number of CV folds before pruning
                max_resource=self.cv_folds,  # Maximum CV folds
                reduction_factor=3  # Aggressiveness of pruning
            )
            
            # Select appropriate sampler based on model type and parameter space
            if self.model_type in ["xgboost", "lightgbm", "hist_gradient_boosting", "catboost"]:
                # For tree-based models, use TPE with multivariate=True for better parameter relationships
                sampler = optuna.samplers.TPESampler(
                    multivariate=True,
                    n_startup_trials=n_startup_trials,
                    seed=42,
                    consider_endpoints=True
                )
            elif self.model_type in ["mlp"]:
                # For neural networks, try to use CmaEs if available, otherwise fall back to TPE
                try:
                    import cmaes
                    sampler = optuna.samplers.CmaEsSampler(
                        seed=42,
                        n_startup_trials=n_startup_trials
                    )
                except ImportError:
                    # Fall back to TPE sampler if cmaes is not installed
                    sampler = optuna.samplers.TPESampler(
                        multivariate=True,
                        n_startup_trials=n_startup_trials,
                        seed=42,
                        consider_endpoints=True
                    )
            else:
                # For simpler models, use standard TPE
                sampler = optuna.samplers.TPESampler(
                    seed=42,
                    n_startup_trials=n_startup_trials
                )
            
            # Create study with in-memory storage
            storage = optuna.storages.InMemoryStorage()
            study = optuna.create_study(
                direction="maximize",
                study_name=f"{self.model_type}_{self.problem_type}_optimisation",
                pruner=pruner,
                sampler=sampler,
                storage=storage  # Use in-memory storage
            )
            
            # Create cross-validation splits once
            from sklearn.model_selection import KFold, StratifiedKFold
            # Handle both binary and multiclass classification
            if self.problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            # Generate splits using numpy arrays
            self.cv_splits = list(cv.split(self.X_train_values, self.y_train_values))
            
            # Get objective function once
            objective_func = self._get_objective_func()
            
            # Run optimisation with catch trials
            n_catch_trials = max(3, self.n_trials // 10)  # 10% of trials as catch trials
            
            for _ in range(n_catch_trials):
                # Create catch trials with random sampling
                catch_trial = study.ask()
                catch_trial.set_user_attr("catch_trial", True)
                try:
                    result = objective_func(catch_trial)
                    study.tell(catch_trial, result)
                except Exception as e:
                    # Handle exceptions during catch trials
                    study.tell(catch_trial, state=optuna.trial.TrialState.FAIL)
            
            # Run main optimization
            study.optimize(
                objective_func,
                n_trials=self.n_trials - n_catch_trials,
                show_progress_bar=True,
                n_jobs=-1,  # Enable parallel optimization for all dataset sizes for maximum performance
                catch=(Exception,)  # Catch exceptions in individual trials without stopping the study
            )
            
            # Store study for later use
            self.study = study
            
            # Get best parameters and create best model
            self.best_params = study.best_params
            self.best_model = self._create_model(self.best_params)
            
            # Fit the best model with original data format
            final_estimator = resampling_estimator(self.best_model, self.resampling_method)
            final_estimator.fit(self.X_train, self.y_train)
            self.best_model = fitted_model(final_estimator)
            
            # Get cross-validation results for the best model
            scoring = selection_scoring(self.problem_type)
            
            self.cv_results = cross_val_score(
                resampling_estimator(self.best_model, self.resampling_method), self.X_train, self.y_train,
                cv=self.cv_splits, scoring=scoring, n_jobs=-1
            )
            
            # Calculate additional optimisation metrics
            optimisation_metrics = {
                "n_complete_trials": len(study.trials),
                "n_pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                "study_duration": study.trials[-1].datetime_complete - study.trials[0].datetime_start
            }
            
            # Prepare optimisation history data
            history = {
                "values": study.trials_dataframe()["value"].tolist(),
                "params": [t.params for t in study.trials],
                "metrics": optimisation_metrics,
                "trials": [{"state": t.state.name} for t in study.trials],
                "study_duration": study.trials[-1].datetime_complete - study.trials[0].datetime_start
            }
            
            return {
                "success": True,
                "message": "Optimisation completed successfully",
                "best_params": self.best_params,
                "best_score": float(study.best_value),
                "cv_mean": float(self.cv_results.mean()),
                "cv_std": float(self.cv_results.std()),
                "cv_results": self.cv_results,
                "optimisation_history": history,
                "model": self.best_model
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return {
                "success": False,
                "message": f"Error during optimisation: {str(e)}",
                "error_details": error_details
            }

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
            print(f"Debug: Number of trials in dataframe: {len(df)}")
            
            # Create history plot
            history_fig = go.Figure()
            
            # Find best trial
            best_trial_idx = df['value'].argmax()
            best_value = df['value'].max()
            
            # Add optimisation history line with highlighted best point
            history_fig.add_trace(go.Scatter(
                x=list(range(len(df))),
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
                x=list(range(len(df))),
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
            param_importances_fig = optuna.visualization.plot_param_importances(self.study)
            # Prepare optimisation history data
            history_data = {
                'params': [t.params for t in self.study.trials],
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
