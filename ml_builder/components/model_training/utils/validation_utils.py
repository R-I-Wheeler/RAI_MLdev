"""Shared training-only validation, scoring, and fold-local resampling."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def selection_scoring(problem_type):
    """Use the same objective for algorithm selection and parameter tuning."""
    if problem_type == "regression":
        return "r2"
    if problem_type == "multiclass_classification":
        return "f1_macro"
    return "f1"


def classification_average(problem_type):
    return "macro" if problem_type == "multiclass_classification" else "binary"


def resampling_estimator(model, method=None):
    """Wrap an estimator so only the training portion of each CV fold is sampled."""
    if not method or method == "None (Original Data)":
        return clone(model)
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler

    samplers = {
        "Random Oversampling": RandomOverSampler,
        "Random Undersampling": RandomUnderSampler,
        "SMOTE": SMOTE,
        "ADASYN": ADASYN,
    }
    if method not in samplers:
        raise ValueError(f"Unknown resampling method: {method}")
    return Pipeline([
        ("resampler", samplers[method](random_state=42)),
        ("model", clone(model)),
    ])


def fitted_model(estimator):
    """Keep the public fitted-model interface independent of the CV pipeline."""
    from imblearn.pipeline import Pipeline
    return estimator.named_steps["model"] if isinstance(estimator, Pipeline) else estimator


def validation_probabilities(model, X, y, resampling_method=None, cv_folds=5):
    """Predict each training row using a clone fitted without that row.

    Calibrated estimators already contain their resampling pipeline; wrapping them
    again would resample the calibration folds as well as the estimator folds.
    """
    from sklearn.calibration import CalibratedClassifierCV

    folds = min(cv_folds, int(pd.Series(y).value_counts().min()))
    if folds < 2:
        raise ValueError("Training-only validation requires at least two rows per class.")
    estimator = (clone(model) if isinstance(model, CalibratedClassifierCV)
                 else resampling_estimator(model, resampling_method))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")


def training_validation_predictions(builder):
    """Return labels and out-of-fold probabilities without accessing test data."""
    model = builder.model.get("active_model") or builder.model["model"]
    probabilities = validation_probabilities(
        model, builder.X_train, builder.y_train,
        builder.model.get("resampling_method"),
    )
    classes = np.unique(builder.y_train)
    return builder.y_train, classes[np.argmax(probabilities, axis=1)], probabilities
