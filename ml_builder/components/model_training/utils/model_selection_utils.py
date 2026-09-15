"""Pure utility functions for model selection and training state management."""

import streamlit as st
from typing import Dict, Any


def select_final_model(model_dict: Dict[str, Any], selection_type: str = "mean_score") -> Dict[str, Any]:
    """Select which model to use for subsequent stages (mean score vs adjusted score).

    Args:
        model_dict: Dictionary containing model information and metrics
        selection_type: Either "mean_score" or "adjusted_score"

    Returns:
        Dictionary with operation status, details, and updated model_dict
    """
    if model_dict is None:
        return {
            "success": False,
            "message": "No model selected. Please select a model first.",
            "model_dict": model_dict
        }

    if "adjusted_model" not in model_dict:
        return {
            "success": False,
            "message": "No adjusted model available. Please run hyperparameter tuning first.",
            "model_dict": model_dict
        }

    try:
        # Create a copy to avoid modifying the original
        updated_model = model_dict.copy()

        if selection_type not in {'mean_score', 'adjusted_score'}:
            raise ValueError('Unknown model selection type.')
        # Keep both fitted candidates intact; model remains the active alias for
        # downstream consumers that predate active_model.
        mean_model = updated_model.get('best_model', updated_model.get('original_model', updated_model['model']))
        updated_model['best_model'] = mean_model

        if selection_type == "adjusted_score":
            # Use the model optimized for stability
            updated_model["active_model"] = updated_model["adjusted_model"]
            updated_model["model"] = updated_model["adjusted_model"]
            updated_model["active_params"] = updated_model["adjusted_params"]
            updated_model["active_cv_metrics"] = updated_model["adjusted_cv_metrics"]
            updated_model["selection_type"] = "adjusted_score"

            # Reset calibration state when new model is active
            updated_model = reset_model_training_state(updated_model)

            # Check if this is the same as the mean score model
            if updated_model["active_params"] == updated_model["best_params"]:
                same_model = True
            else:
                same_model = False

            return {
                "success": True,
                "message": "Selected model optimized for balanced performance and stability",
                "model_dict": updated_model,
                "info": {
                    "mean_score": updated_model["adjusted_cv_metrics"]["mean_score"],
                    "std_score": updated_model["adjusted_cv_metrics"]["std_score"],
                    "adjusted_score": updated_model["adjusted_cv_metrics"]["adjusted_score"],
                    "same_as_mean_score_model": same_model
                }
            }
        else:
            # Use the model optimized for mean score (default)
            updated_model["active_model"] = mean_model
            updated_model["model"] = mean_model
            updated_model["active_params"] = updated_model["best_params"]
            updated_model["active_cv_metrics"] = updated_model["cv_metrics"]
            updated_model["selection_type"] = "mean_score"

            # Reset calibration state when new model is active
            updated_model = reset_model_training_state(updated_model)

            # Calculate adjusted score for this model for comparison
            alpha = 1.0
            adjusted_score = updated_model["cv_metrics"]["mean_score"] - (
                        alpha * updated_model["cv_metrics"]["std_score"])

            return {
                "success": True,
                "message": "Selected model optimized for maximum average performance",
                "model_dict": updated_model,
                "info": {
                    "mean_score": updated_model["cv_metrics"]["mean_score"],
                    "std_score": updated_model["cv_metrics"]["std_score"],
                    "adjusted_score": adjusted_score
                }
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error selecting final model: {str(e)}",
            "model_dict": model_dict
        }


def reset_model_training_state(model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Reset all model-specific training state (calibration, threshold optimization) when a new model is trained.

    Args:
        model_dict: Dictionary containing model information

    Returns:
        Updated model dictionary with training state cleared
    """
    if not model_dict:
        return model_dict

    try:
        # Create a copy to avoid modifying the original
        updated_model = model_dict.copy()

        # Clear calibration flags and data
        calibration_keys_to_clear = [
            "is_calibrated",
            "calibrated_model",
            "calibration_method",
            "calibration_cv_folds",
            "original_model"
        ]

        # Clear threshold optimization flags and data
        threshold_keys_to_clear = [
            "threshold_optimized",
            "optimal_threshold",
            "threshold_is_binary",
            "threshold_criterion"
        ]

        # Combined list of all training state keys to clear
        all_keys_to_clear = calibration_keys_to_clear + threshold_keys_to_clear

        for key in all_keys_to_clear:
            if key in updated_model:
                del updated_model[key]

        # Clear related caches from session state if available
        try:
            cache_keys_to_clear = ['calibration_cache', 'threshold_analysis_cache',
                                   'training_predictions_cache', 'training_prediction_models', 'training_metrics_cache']
            for cache_key in cache_keys_to_clear:
                if (hasattr(st, 'session_state') and
                        hasattr(st.session_state, cache_key)):
                    getattr(st.session_state, cache_key).clear()

            # Clear any Streamlit cached functions for calibration and threshold analysis
            if hasattr(st, '_cached_data_cache'):
                cache_keys_to_remove = []
                for key in st._cached_data_cache.keys():
                    key_str = str(key).lower()
                    if any(term in key_str for term in ['calibration', 'threshold', 'roc', 'precision_recall']):
                        cache_keys_to_remove.append(key)
                for key in cache_keys_to_remove:
                    del st._cached_data_cache[key]
        except Exception:
            # Silently fail to avoid disrupting training process
            pass

        return updated_model

    except Exception:
        # Silently fail to avoid disrupting training process
        return model_dict
