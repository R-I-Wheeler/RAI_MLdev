"""Invalidate fitted results on committed model/data changes, not navigation."""

import hashlib
import pandas as pd
import streamlit as st
from content.stage_info import ModelStage


def selection_signature(builder):
    digest = hashlib.sha256()
    for name in ("training_data", "testing_data", "X_train", "y_train", "X_test", "y_test"):
        value = getattr(builder, name, None)
        digest.update(name.encode())
        if value is not None:
            frame = pd.DataFrame(value)
            digest.update(repr((list(frame.columns), frame.dtypes.astype(str).tolist())).encode())
            digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return (builder.target_column, builder.detect_problem_type(), digest.hexdigest())


def clear_model_results(builder, session_state=None):
    """Clear model-dependent state and completion flags as a single operation."""
    state = st.session_state if session_state is None else session_state
    keys = {
        "training_complete", "training_results", "cv_results", "optuna_results",
        "optuna_studies", "calibration_models", "model_predictions",
        "param_ranges_cache", "calibration_cache", "training_predictions_cache",
        "training_metrics_cache", "selected_model_type", "selected_model_stability",
        "previous_model_selection", "previous_training_id", "last_training_model_signature",
        "imbalance_handled", "imbalance_skipped", "active_training_pill",
        "automated_model_selection_training_completed", "automated_model_selection_training_result",
        "evaluation_results", "explanation_results", "feature_importance_data",
        "threshold_analysis_cache", "viz_cache_warmed", "sample_indices",
        "last_sample_size", "selected_evaluation_tab",
    }
    for key in keys:
        state.pop(key, None)
    stages = list(ModelStage)
    for stage in stages[stages.index(ModelStage.MODEL_SELECTION):]:
        builder.stage_completion[stage] = False
    builder.model = None


def invalidate_changed_data(builder, session_state=None):
    if not builder.model:
        return False
    signature = selection_signature(builder)
    previous = builder.model.get("selection_signature")
    if previous is not None and previous != signature:
        clear_model_results(builder, session_state)
        return True
    # Existing sessions acquire provenance on their first visit without losing work.
    builder.model["selection_signature"] = signature
    return False
