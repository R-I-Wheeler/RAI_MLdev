"""Shared training-only validation, scoring, and fold-local resampling."""

import numpy as np
import pandas as pd
import hashlib
import os
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from threadpoolctl import threadpool_limits


def worker_budget(workers=None):
    return max(1, min(int(workers or 4), os.cpu_count() or 1))


def bounded_estimator(model, workers=1):
    estimator = clone(model)
    params = estimator.get_params(deep=True)
    limits = {key: workers for key in params
              if key.split('__')[-1] in {'n_jobs', 'nthread', 'thread_count'}}
    # CatBoost omits default parameters from get_params().
    if type(estimator).__module__.startswith('catboost'):
        limits['thread_count'] = workers
    for key, value in params.items():
        if type(value).__module__.startswith('catboost'):
            limits[f'{key}__thread_count'] = workers
    return estimator.set_params(**limits)


def make_cv_splits(X, y, problem_type, cv_folds=5, resampling_method=None):
    """Use identical, reproducible folds and reject infeasible runs before fitting."""
    if X is None or y is None or len(X) != len(y):
        raise ValueError('Training features and labels must contain the same rows.')
    if not isinstance(cv_folds, (int, np.integer)) or cv_folds < 2:
        raise ValueError('Cross-validation requires at least two folds.')
    if pd.Series(y).isna().any():
        raise ValueError('Training labels contain missing values.')
    if problem_type == 'regression':
        if len(y) < 2 * cv_folds:
            raise ValueError('R² validation requires at least two rows in every validation fold.')
        cv = KFold(cv_folds, shuffle=True, random_state=42)
    else:
        counts = pd.Series(y).value_counts()
        if len(counts) < 2 or counts.min() < cv_folds:
            raise ValueError(f'Each class needs at least {cv_folds} rows for {cv_folds}-fold validation. Reduce the fold count or add data.')
        cv = StratifiedKFold(cv_folds, shuffle=True, random_state=42)
    splits = list(cv.split(X, y))
    if resampling_method in {'SMOTE', 'ADASYN'}:
        for train, _ in splits:
            if pd.Series(np.asarray(y)[train]).value_counts().min() < 6:
                raise ValueError(f'{resampling_method} needs at least six examples per class in each training fold. Use random oversampling or add data.')
    return splits


def data_fingerprint(X, y):
    digest = hashlib.sha256()
    for value in (X, y):
        frame = pd.DataFrame(value)
        digest.update(repr((list(frame.columns), frame.dtypes.astype(str).tolist())).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def selection_scoring(problem_type):
    """Use the same objective for algorithm selection and parameter tuning."""
    if problem_type == "regression":
        return "r2"
    if problem_type == "multiclass_classification":
        return "f1_macro"
    return "f1"


def classification_average(problem_type):
    return "macro" if problem_type == "multiclass_classification" else "binary"


def resampling_estimator(model, method=None, workers=1):
    """Wrap an estimator so only the training portion of each CV fold is sampled."""
    if not method or method == "None (Original Data)":
        return bounded_estimator(model, workers)
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
        ("model", bounded_estimator(model, workers)),
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
    estimator = (bounded_estimator(model) if isinstance(model, CalibratedClassifierCV)
                 else resampling_estimator(model, resampling_method))
    cv = make_cv_splits(X, y, 'classification', folds,
                        None if isinstance(model, CalibratedClassifierCV) else resampling_method)
    if isinstance(model, CalibratedClassifierCV):
        inner_folds = (model.cv if isinstance(model.cv, int) else
                       model.cv.get_n_splits() if hasattr(model.cv, 'get_n_splits') else 5)
        for train, _ in cv:
            make_cv_splits(np.asarray(X)[train], np.asarray(y)[train],
                           'classification', inner_folds, resampling_method)
    with threadpool_limits(limits=1):
        return cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")


def training_validation_predictions(builder, model=None, cv_folds=None):
    """Return labels and out-of-fold probabilities without accessing test data."""
    import streamlit as st
    model = model if model is not None else (builder.model.get("active_model") or builder.model["model"])
    cv_folds = cv_folds or builder.model.get('training_run', {}).get('cv_folds', 5)
    key = (builder.model.get('training_run', {}).get('run_id'), id(model),
           repr(model.get_params(deep=True)), builder.model.get('resampling_method'),
           cv_folds, data_fingerprint(builder.X_train, builder.y_train))
    cache = st.session_state.setdefault('training_predictions_cache', {})
    if key in cache:
        return cache[key]
    probabilities = validation_probabilities(
        model, builder.X_train, builder.y_train,
        builder.model.get("resampling_method"),
        cv_folds=cv_folds,
    )
    classes = np.unique(builder.y_train)
    result = (builder.y_train.copy(), classes[np.argmax(probabilities, axis=1)], probabilities)
    # Keep a strong reference while cached so Python cannot reuse the model ID.
    refs = st.session_state.setdefault('training_prediction_models', {})
    while len(cache) >= 4:
        oldest = next(iter(cache))
        cache.pop(oldest)
        refs.pop(oldest, None)
    cache[key] = result
    refs[key] = model
    return result
