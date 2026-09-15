"""
Core evaluation utilities for ML model evaluation.

This module contains the core logic for model evaluation including prediction extraction,
metrics calculation, and data preparation, separated from visualization logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, r2_score,
    mean_absolute_error, mean_squared_error
)
from sklearn.model_selection import learning_curve


def extract_model_predictions(model_dict: Dict[str, Any], X_test: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract predictions from model, handling threshold optimization and probability extraction.

    Args:
        model_dict: Dictionary containing model and optimization information
        X_test: Test features

    Returns:
        Tuple of (predictions, probabilities) where probabilities may be None
    """
    problem_type = model_dict.get("problem_type", "unknown")

    # Use the active model if available (user-selected), otherwise use the default model
    if "active_model" in model_dict:
        model_instance = model_dict["active_model"]
    else:
        model_instance = model_dict["model"]

    # Get prediction probabilities if available
    y_prob_matrix = None
    if hasattr(model_instance, 'predict_proba'):
        try:
            y_prob_matrix = model_instance.predict_proba(X_test)
        except:
            y_prob_matrix = None

    # Handle threshold optimization for classification
    if (problem_type in ["classification", "binary_classification", "multiclass_classification"] and
        model_dict.get("threshold_optimized", False) and
        model_dict.get("threshold_is_binary", True) and
        y_prob_matrix is not None):

        # Use optimal threshold for binary classification
        optimal_threshold = model_dict.get("optimal_threshold", 0.5)

        if len(y_prob_matrix.shape) > 1 and y_prob_matrix.shape[1] == 2:
            # Binary classification with optimal threshold
            y_prob_positive = y_prob_matrix[:, 1]
            y_pred = (y_prob_positive >= optimal_threshold).astype(int)
        else:
            # Fallback to default predict
            y_pred = model_instance.predict(X_test)

    elif (problem_type in ["classification", "binary_classification", "multiclass_classification"] and
          model_dict.get("threshold_optimized", False) and
          not model_dict.get("threshold_is_binary", True) and
          y_prob_matrix is not None):

        # Use optimal confidence threshold for multiclass classification
        optimal_threshold = model_dict.get("optimal_threshold", 0.5)
        max_probs = np.max(y_prob_matrix, axis=1)
        predicted_classes = np.argmax(y_prob_matrix, axis=1)

        # Only make predictions where confidence is above threshold
        confident_mask = max_probs >= optimal_threshold
        y_pred = np.full(len(X_test), -1)  # -1 for uncertain predictions
        y_pred[confident_mask] = predicted_classes[confident_mask]

        # Convert -1 (uncertain) to most likely class for metrics calculation
        y_pred[y_pred == -1] = predicted_classes[y_pred == -1]

    else:
        # Default prediction method
        y_pred = model_instance.predict(X_test)

    return y_pred, y_prob_matrix


def calculate_classification_metrics(y_test: Union[pd.Series, np.ndarray], y_pred: np.ndarray, problem_type=None) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_test: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of classification metrics
    """
    from components.model_training.utils.validation_utils import classification_average
    if problem_type is None:
        problem_type = "multiclass_classification" if len(np.unique(y_test)) > 2 else "binary_classification"
    average = classification_average(problem_type)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average=average, zero_division=0))
    }
    return metrics


def calculate_regression_metrics(y_test: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_test: True values
        y_pred: Predicted values

    Returns:
        Dictionary of regression metrics
    """
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred)))
    }
    return metrics


def get_classification_report(y_test: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Generate classification report.

    Args:
        y_test: True labels
        y_pred: Predicted labels

    Returns:
        Classification report as dictionary
    """
    return classification_report(y_test, y_pred, output_dict=True)


def prepare_evaluation_data(y_test: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> Tuple[np.ndarray, list]:
    """
    Prepare evaluation data including confusion matrix and class names.

    Args:
        y_test: True labels
        y_pred: Predicted labels

    Returns:
        Tuple of (confusion_matrix, class_names)
    """
    cm = confusion_matrix(y_test, y_pred)
    class_names = sorted(list(set(y_test)))
    return cm, class_names


def prepare_residuals_data(y_test: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> pd.DataFrame:
    """
    Prepare residuals data for regression analysis.

    Args:
        y_test: True values
        y_pred: Predicted values

    Returns:
        DataFrame with actual, predicted, and residuals columns
    """
    plot_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    plot_df['Residuals'] = plot_df['Actual'] - plot_df['Predicted']
    plot_df['Sqrt_Abs_Residuals'] = np.sqrt(np.abs(plot_df['Residuals']))
    return plot_df


def get_learning_curve_data(model_instance, X_train: pd.DataFrame, y_train: Union[pd.Series, np.ndarray],
                          problem_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate learning curve data.

    Args:
        model_instance: Trained model instance
        X_train: Training features
        y_train: Training targets
        problem_type: Type of problem (classification/regression)

    Returns:
        Tuple of (train_sizes, train_scores, val_scores)
    """
    scoring = 'accuracy' if problem_type in ["classification", "binary_classification", "multiclass_classification"] else 'r2'

    train_sizes, train_scores, val_scores = learning_curve(
        model_instance, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring=scoring
    )

    return train_sizes, train_scores, val_scores


def calculate_residuals_stats(residuals: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic residual statistics for logging.

    Args:
        residuals: Array of residual values

    Returns:
        Dictionary of residual statistics
    """
    return {
        'mean': float(residuals.mean()),
        'std': float(residuals.std()),
        'min': float(residuals.min()),
        'max': float(residuals.max())
    }
