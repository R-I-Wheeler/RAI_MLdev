"""Provenance and invalidation for committed training runs."""

from datetime import datetime, timezone
from uuid import uuid4
from importlib.metadata import version
import streamlit as st
from content.stage_info import ModelStage
from components.model_training.utils.validation_utils import data_fingerprint, selection_scoring


def make_run_record(model_dict, X, y, method, folds, trials, workers):
    return {
        'run_id': uuid4().hex, 'created_at': datetime.now(timezone.utc).isoformat(),
        'model_type': model_dict['type'], 'problem_type': model_dict['problem_type'],
        'optimisation_method': method, 'cv_folds': folds, 'requested_trials': trials,
        'workers': workers, 'random_seed': 42, 'shuffle': True,
        'scoring': selection_scoring(model_dict['problem_type']),
        'resampling_method': model_dict.get('resampling_method', 'None (Original Data)'),
        'data_fingerprint': data_fingerprint(X, y), 'validation_version': 2,
        'library_versions': {name: version(name) for name in ('scikit-learn', 'optuna', 'numpy', 'imbalanced-learn')},
        'validation_scope': 'CV on the features supplied by preprocessing and feature selection',
    }


def clear_threshold(model):
    had_threshold = model.get('threshold_optimized', False)
    for key in ('threshold_optimized', 'optimal_threshold', 'threshold_is_binary', 'threshold_criterion'):
        model.pop(key, None)
    return had_threshold


def invalidate_predictions(builder, clear_validation=True):
    keys = ['training_metrics_cache', 'calibration_cache', 'threshold_analysis_cache', 'evaluation_results',
                'explanation_results', 'model_predictions', 'feature_importance_data',
                'viz_cache_warmed']
    if clear_validation:
        keys.extend(['training_predictions_cache', 'training_prediction_models'])
    for key in keys:
        st.session_state.pop(key, None)
    stages = list(ModelStage)
    for stage in stages[stages.index(ModelStage.MODEL_EVALUATION):]:
        if hasattr(builder, 'stage_completion'):
            builder.stage_completion[stage] = False


def commit_calibration(builder, model, *, original_model=None, method=None, cv_folds=None):
    """Activate an already fitted calibration candidate (or restore its original).

    Validation and fitting must succeed before calling this shared commit step.
    Cached out-of-fold probabilities remain valid; derived results do not.
    """
    builder.model.update(model=model, active_model=model, is_calibrated=method is not None)
    if method is None:
        for key in ('calibrated_model', 'calibration_method', 'calibration_cv_folds'):
            builder.model.pop(key, None)
    else:
        builder.model.update(original_model=original_model, calibrated_model=model,
                             calibration_method=method, calibration_cv_folds=cv_folds)
    if clear_threshold(builder.model):
        st.session_state.calibration_notice = (
            'Threshold reset to 0.5 because calibration changed. Re-run threshold analysis if needed.'
        )
    invalidate_predictions(builder, clear_validation=False)


def commit_threshold(builder, threshold, is_binary, criterion):
    """Keep manual and automated threshold changes consistent downstream."""
    builder.model.update(optimal_threshold=float(threshold), threshold_optimized=True,
                         threshold_is_binary=is_binary, threshold_criterion=criterion)
    invalidate_predictions(builder, clear_validation=False)
