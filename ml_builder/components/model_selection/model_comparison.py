"""
Model comparison functionality for ML Builder.

Contains functions to perform quick comparisons of different machine learning models
using default parameters on data samples.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from components.model_training.utils.validation_utils import classification_average


def quick_model_comparison(training_data, testing_data, target_column, problem_type, sample_size=1000, exclude_xgboost=False):
    """
    Perform a quick comparison of all available models using a data sample.

    Args:
        training_data: DataFrame containing the training dataset
        testing_data: Retained for caller compatibility; never used for selection
        target_column: Name of the target column
        problem_type: Either "classification" or "regression"
        sample_size: Size of the sample to use
        exclude_xgboost: Whether to exclude XGBoost from the comparison

    Returns:
        DataFrame containing model performance metrics
    """
    # Validation is drawn only from training data. The final test set stays unseen.
    if len(training_data) > sample_size:
        sample = training_data.sample(n=sample_size, random_state=42)
    else:
        sample = training_data.copy()
    train_sample, test_sample = train_test_split(
        sample, test_size=0.2, random_state=42,
        stratify=sample[target_column] if problem_type != "regression" else None,
    )

    # Split features and target
    X_train = train_sample.drop(columns=[target_column])
    y_train = train_sample[target_column]
    X_test = test_sample.drop(columns=[target_column])
    y_test = test_sample[target_column]

    # Initialize models with default parameters
    # Handle both binary and multiclass classification
    if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
        # Set appropriate objective for LightGBM based on classification type
        if problem_type == "multiclass_classification":
            lgbm_objective = 'multiclass'
        else:
            lgbm_objective = 'binary'  # Works for both binary and general classification

        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
            "naive_bayes": GaussianNB(),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "mlp": MLPClassifier(max_iter=1000, random_state=42),
            "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42, max_iter=100),
            "catboost": CatBoostClassifier(random_state=42, iterations=100, verbose=False),
            "lightgbm": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, objective=lgbm_objective)
        }

        # Add XGBoost only if it's compatible
        if not exclude_xgboost:
            models["xgboost"] = XGBClassifier(random_state=42, eval_metric='logloss', nthread=-1)
    else:
        models = {
            "linear_regression": LinearRegression(n_jobs=-1),
            "ridge_regression": Ridge(random_state=42),
            "decision_tree": DecisionTreeRegressor(random_state=42),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "mlp": MLPRegressor(max_iter=1000, random_state=42),
            "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42, max_iter=100),
            "catboost": CatBoostRegressor(random_state=42, iterations=100, verbose=False),
            "lightgbm": LGBMRegressor(random_state=42, objective='regression', n_jobs=-1, verbose=-1)
        }

        # Add XGBoost only if it's compatible (for regression it's always compatible)
        if not exclude_xgboost:
            models["xgboost"] = XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse', nthread=-1)

    results = []

    # Train and evaluate each model
    for name, model in models.items():
        try:
            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            # Handle both binary and multiclass classification
            if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                avg_method = classification_average(problem_type)

                metrics = {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average=avg_method, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average=avg_method, zero_division=0),
                    "F1": f1_score(y_test, y_pred, average=avg_method, zero_division=0)
                }
            else:
                metrics = {
                    "Model": name,
                    "R²": r2_score(y_test, y_pred),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "MSE": mean_squared_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
                }
            results.append(metrics)
        except Exception as e:
            st.warning(f"Error evaluating {name}: {str(e)}")
            continue

    return pd.DataFrame(results)


def get_best_model_from_comparison(results_df, problem_type):
    """
    Get the best performing model from comparison results.

    Args:
        results_df: DataFrame with model comparison results
        problem_type: Type of ML problem

    Returns:
        tuple: (best_model_name, best_score, best_metric)
    """
    # Handle both binary and multiclass classification
    if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
        best_metric = "F1"
    else:
        best_metric = "R²"

    best_model = results_df.loc[results_df[best_metric].idxmax(), "Model"]
    best_score = results_df.loc[results_df[best_metric].idxmax(), best_metric]

    return best_model, best_score, best_metric


def style_comparison_results(results_df, best_metric):
    """
    Style the comparison results DataFrame to highlight the best model.

    Args:
        results_df: DataFrame with comparison results
        best_metric: The metric to use for highlighting

    Returns:
        Styled DataFrame
    """
    def highlight_best(s):
        if s.name == best_metric:
            return ['background-color: #90EE90' if v == s.max() else '' for v in s]
        return ['' for _ in s]

    # Format numeric columns to 3 decimal places
    numeric_cols = results_df.select_dtypes(include=['float64']).columns
    results_df[numeric_cols] = results_df[numeric_cols].round(3)

    return results_df.style.apply(highlight_best)
