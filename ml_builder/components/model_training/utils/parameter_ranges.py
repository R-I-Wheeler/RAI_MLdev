"""Utility class for adaptive parameter ranges based on dataset characteristics."""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import uniform, loguniform, randint
import streamlit as st

@st.cache_data(ttl=600, show_spinner=False)
def _calculate_dataset_characteristics(X_shape, y_shape, X_hash, y_hash, problem_type):
    """Cached calculation of dataset characteristics to avoid redundant computation"""
    # This function will be called with hashed versions of the data
    # The actual calculation will be done in the class
    return {
        'n_samples': X_shape[0],
        'n_features': X_shape[1],
        'is_high_dimensional': X_shape[1] > 100,
        'is_small_dataset': X_shape[0] < 1000,
        'problem_type': problem_type
    }

class AdaptiveParameterRanges:
    def __init__(self, X_train, y_train, problem_type: str):
        """Initialize with dataset characteristics."""
        self.X_train = X_train
        self.y_train = y_train
        self.problem_type = problem_type

        # Calculate dataset characteristics with caching
        self.X_is_pandas = isinstance(X_train, pd.DataFrame)
        self.y_is_pandas = isinstance(y_train, pd.Series)
        self.X_train_values = X_train.values if self.X_is_pandas else X_train
        self.y_train_values = y_train.values if self.y_is_pandas else y_train

        # Create cache keys
        X_hash = hash(str(self.X_train_values.shape) + str(self.X_train_values.dtype))
        y_hash = hash(str(self.y_train_values.shape) + str(self.y_train_values.dtype))

        # Use cached characteristics if available
        cached_chars = _calculate_dataset_characteristics(
            self.X_train_values.shape,
            self.y_train_values.shape,
            X_hash,
            y_hash,
            problem_type
        )

        self.n_samples = cached_chars['n_samples']
        self.n_features = cached_chars['n_features']
        self.is_high_dimensional = cached_chars['is_high_dimensional']
        self.is_small_dataset = cached_chars['is_small_dataset']

        # Calculate class distribution and feature density (these are less expensive)
        self.class_distribution = None if problem_type == "regression" else pd.Series(self.y_train_values).value_counts().to_numpy()
        self.feature_density = np.mean(np.abs(self.X_train_values) > 0)
        
    def get_ranges(self, model_type: str, tuning_method: str = "optuna") -> Dict[str, Any]:
        """Get parameter ranges based on model type and tuning method."""
        ranges = self._get_base_ranges(model_type)
        
        if tuning_method == "optuna":
            return ranges
        else:
            converted = self._convert_to_random_search(ranges)
            if model_type == 'random_forest':
                return [dict(converted, bootstrap=[True]),
                        dict(converted, bootstrap=[False], max_samples=[None])]
            return converted
    
    def _get_base_ranges(self, model_type: str) -> Dict[str, Any]:
        """Get base parameter ranges in Optuna format."""
        if model_type == "logistic_regression":
            c_min = 0.01 if self.is_high_dimensional else 0.1
            c_max = 1.0 if self.is_high_dimensional else 10.0
            return {
                "C": ("float", c_min, c_max, True),  # True for log scale
                "max_iter": ("int", 100, 500)
            }
        elif model_type == "naive_bayes":
            # Only for classification - GaussianNB for continuous features
            return {
                "var_smoothing": ("float", 1e-11, 1e-7, True),  # Log scale smoothing parameter
            }
        elif model_type == "ridge_regression":
            # Only for regression - L2 regularization
            alpha_min = 0.01 if self.is_high_dimensional else 0.1
            alpha_max = 10.0 if self.is_high_dimensional else 100.0
            return {
                "alpha": ("float", alpha_min, alpha_max, True),  # Log scale regularization strength
                "max_iter": ("int", 100, 1000)
            }
        elif model_type == "decision_tree":
            ranges = {
                "max_depth": ("int", 3, 20),
                "min_samples_split": ("int", 2, 50),
                "min_samples_leaf": ("int", 1, 20),
                "max_features": ("categorical", ['sqrt', 'log2', None]),
                "min_impurity_decrease": ("float", 0.0, 0.1, False),  # Minimum impurity decrease for split
                "ccp_alpha": ("float", 0.0, 0.05, False),  # Cost-complexity pruning
                "splitter": ("categorical", ["best", "random"])  # Split strategy
            }
            if self.problem_type == "regression":
                ranges["criterion"] = ("categorical", ["squared_error", "absolute_error", "friedman_mse"])
            else:
                ranges["criterion"] = ("categorical", ["gini", "entropy"])
            return ranges
        elif model_type == "random_forest":
            n_estimators_max = 200 if self.is_small_dataset else 300
            max_depth_max = 10 if self.is_high_dimensional else 15
            max_features_values = ['sqrt', 'log2'] if self.is_high_dimensional else ['sqrt', 'log2', None]

            ranges = {
                "n_estimators": ("int", 50, n_estimators_max),
                "max_depth": ("int", 3, max_depth_max),
                "min_samples_split": ("int", 2, 20),
                "min_samples_leaf": ("int", 1, 10),
                "max_features": ("categorical", max_features_values),
                "bootstrap": ("categorical", [True, False]),
                "min_impurity_decrease": ("float", 0.0, 0.1, False),  # Minimum impurity decrease for split
                "ccp_alpha": ("float", 0.0, 0.05, False),  # Cost-complexity pruning
                "max_samples": ("float", 0.5, 1.0, False)  # Max samples to use when building trees (if bootstrap=True)
            }
            if self.problem_type == "regression":
                ranges["criterion"] = ("categorical", ["squared_error", "absolute_error", "friedman_mse"])
            else:
                ranges["criterion"] = ("categorical", ["gini", "entropy"])
            return ranges
        elif model_type == "mlp":
            # Define architectures based on data characteristics
            if self.is_high_dimensional:
                hidden_sizes = [
                    (100,), (200,),  # Single layer
                    (100, 50), (200, 100),  # Two layers
                    (200, 100, 50)  # Three layers for complex patterns
                ]
            else:
                hidden_sizes = [
                    (50,), (100,),  # Single layer
                    (50, 25), (100, 50),  # Two layers
                    (100, 50, 25)  # Three layers for complex patterns
                ]
            
            # Adjust learning rates based on dataset size
            lr_min = 0.0001 if self.is_small_dataset else 0.001
            lr_max = 0.01 if self.is_small_dataset else 0.1
            
            # Adjust regularization based on data sparsity
            alpha_min = 0.0001 if self.feature_density > 0.5 else 0.001
            alpha_max = 0.01 if self.feature_density > 0.5 else 0.1
            
            return {
                "hidden_layer_sizes": ("categorical_tuple", hidden_sizes),
                "learning_rate_init": ("float", lr_min, lr_max, True),
                "alpha": ("float", alpha_min, alpha_max, True),
                "batch_size": ("categorical", [16, 32, 64, "auto"] if self.is_small_dataset else [32, 64, 128, "auto"]),
                "activation": ("categorical", ["relu", "tanh"]),
                "learning_rate": ("categorical", ["constant", "adaptive", "invscaling"]),
                "solver": ("categorical", ["adam", "sgd"]),  # Optimization algorithm
                "early_stopping": ("categorical", [True, False]),
                "validation_fraction": ("float", 0.1, 0.2, False),  # Fraction for validation with early stopping
                "n_iter_no_change": ("int", 5, 20),  # Patience for early stopping
                "momentum": ("float", 0.8, 0.99, False)  # Momentum for SGD (ignored for adam)
            }
        elif model_type == "xgboost":
            n_estimators_max = 300 if self.is_small_dataset else 500
            max_depth_max = 6 if self.is_high_dimensional else 10
            min_child_weight_max = 5 if self.is_high_dimensional else 7
            gamma_max = 1.0 if self.is_high_dimensional else 0.5

            return {
                "n_estimators": ("int", 100, n_estimators_max),
                "max_depth": ("int", 3, max_depth_max),
                "learning_rate": ("float", 0.01, 0.3, True),
                "subsample": ("float", 0.6, 1.0, False),
                "colsample_bytree": ("float", 0.6, 1.0, False),
                "min_child_weight": ("int", 1, min_child_weight_max),
                "gamma": ("float", 1e-8, gamma_max, True),
                "reg_alpha": ("float", 1e-8, 1.0, True),
                "reg_lambda": ("float", 1e-8, 1.0, True)
            }
        elif model_type == "lightgbm":
            num_leaves_max = 50 if self.is_high_dimensional else 100
            n_estimators_max = 300 if self.is_small_dataset else 500

            return {
                "n_estimators": ("int", 100, n_estimators_max),
                "num_leaves": ("int", 20, num_leaves_max),
                "learning_rate": ("float", 0.01, 0.3, True),
                "max_depth": ("int", 3, 10),
                "min_child_samples": ("int", 10, 50),
                "subsample": ("float", 0.6, 1.0, False),
                "colsample_bytree": ("float", 0.6, 1.0, False),
                "reg_alpha": ("float", 1e-8, 1.0, True),
                "reg_lambda": ("float", 1e-8, 1.0, True),
                "boosting_type": ("categorical", ["gbdt", "dart", "goss"])  # Different boosting strategies
            }
        elif model_type == "hist_gradient_boosting":
            n_estimators_max = 300 if self.is_small_dataset else 500  # Increased for early stopping
            max_depth_max = 10 if self.is_high_dimensional else 15

            return {
                "max_iter": ("int", 100, n_estimators_max),
                "max_depth": ("int", 3, max_depth_max),
                "learning_rate": ("float", 0.01, 0.3, True),
                "l2_regularization": ("float", 0.0, 1.0, False),
                "max_bins": ("int", 128, 255),
                "min_samples_leaf": ("int", 10, 50),
                "early_stopping": ("categorical", [True, "auto"]),  # Enable early stopping
                "validation_fraction": ("float", 0.1, 0.2, False),  # Fraction for validation
                "n_iter_no_change": ("int", 5, 20),  # Patience for early stopping
                "tol": ("float", 1e-7, 1e-3, True)  # Convergence tolerance
            }
        elif model_type == "catboost":
            n_estimators_max = 300 if self.is_small_dataset else 500
            depth_max = 6 if self.is_high_dimensional else 10

            return {
                "iterations": ("int", 100, n_estimators_max),
                "depth": ("int", 4, depth_max),
                "learning_rate": ("float", 0.01, 0.3, True),
                "l2_leaf_reg": ("float", 1.0, 10.0, True),
                "border_count": ("int", 32, 255),
                "bagging_temperature": ("float", 0.0, 1.0, False),
                "random_strength": ("float", 0.0, 10.0, False)
            }
        else:  # linear_regression
            return {}

    def _convert_to_random_search(self, optuna_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Optuna ranges to RandomizedSearchCV format."""
        random_search_ranges = {}
        
        for param, range_info in optuna_ranges.items():
            param_type = range_info[0]
            if param_type == "int":
                _, low, high = range_info
                random_search_ranges[param] = randint(low, high + 1)
            elif param_type == "float":
                _, low, high, is_log = range_info
                if is_log:
                    random_search_ranges[param] = loguniform(low, high)
                else:
                    random_search_ranges[param] = uniform(low, high - low)
            elif param_type == "categorical_tuple":
                # Handle tuple categories directly without modification
                _, categories = range_info
                random_search_ranges[param] = categories
            elif param_type == "categorical":
                _, categories = range_info
                random_search_ranges[param] = categories
        
        return random_search_ranges
