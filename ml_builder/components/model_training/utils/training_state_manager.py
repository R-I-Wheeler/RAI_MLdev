"""Centralized state manager for model training components with session state optimization."""

import streamlit as st
import sys
from typing import Dict, Any, Optional, List
import hashlib


class TrainingStateManager:
    """Manages shared state and communication between training components with intelligent session state management."""

    # Define memory-heavy components to track
    MEMORY_HEAVY_KEYS = [
        'cv_results',
        'optuna_studies',
        'calibration_models',
        'large_visualizations',
        'model_predictions',
        'feature_importance_data',
        'training_predictions_cache', 'training_prediction_models', 'training_metrics_cache'
    ]

    # Maximum number of cached items to keep
    MAX_CACHE_ITEMS = 10

    # Memory threshold in MB for cleanup
    MEMORY_THRESHOLD_MB = 500

    @staticmethod
    def get_session_state_size() -> int:
        """Estimate session state memory usage in bytes."""
        total_size = 0
        for key, value in st.session_state.items():
            try:
                total_size += sys.getsizeof(value)
            except:
                # Some objects can't be sized, skip them
                continue
        return total_size

    @staticmethod
    def cleanup_old_caches() -> Dict[str, int]:
        """Clean up old cache entries to free memory."""
        cleanup_stats = {"removed_items": 0, "freed_memory": 0}

        # Clean up parameter ranges cache
        if 'param_ranges_cache' in st.session_state:
            cache = st.session_state.param_ranges_cache
            if len(cache) > TrainingStateManager.MAX_CACHE_ITEMS:
                # Keep only the most recent items
                items = list(cache.items())
                to_remove = len(items) - TrainingStateManager.MAX_CACHE_ITEMS
                for i in range(to_remove):
                    key, value = items[i]
                    cleanup_stats["freed_memory"] += sys.getsizeof(value)
                    del cache[key]
                    cleanup_stats["removed_items"] += 1

        # Clean up calibration cache
        if 'calibration_cache' in st.session_state:
            cache = st.session_state.calibration_cache
            if len(cache) > TrainingStateManager.MAX_CACHE_ITEMS:
                items = list(cache.items())
                to_remove = len(items) - TrainingStateManager.MAX_CACHE_ITEMS
                for i in range(to_remove):
                    key, value = items[i]
                    cleanup_stats["freed_memory"] += sys.getsizeof(value)
                    del cache[key]
                    cleanup_stats["removed_items"] += 1

        return cleanup_stats

    @staticmethod
    def cleanup_memory_heavy_items() -> Dict[str, int]:
        """Clean up memory-heavy items when memory usage is high."""
        cleanup_stats = {"removed_keys": 0, "freed_memory": 0}

        for key in TrainingStateManager.MEMORY_HEAVY_KEYS:
            if key in st.session_state:
                try:
                    size = sys.getsizeof(st.session_state[key])
                    del st.session_state[key]
                    cleanup_stats["removed_keys"] += 1
                    cleanup_stats["freed_memory"] += size
                except:
                    continue

        return cleanup_stats

    @staticmethod
    def optimize_session_state() -> Dict[str, Any]:
        """Perform comprehensive session state optimization."""
        initial_size = TrainingStateManager.get_session_state_size()
        initial_size_mb = initial_size / (1024 * 1024)

        optimization_stats = {
            "initial_size_mb": round(initial_size_mb, 2),
            "cache_cleanup": {},
            "memory_cleanup": {},
            "final_size_mb": 0,
            "memory_saved_mb": 0
        }

        # Always clean up old caches
        optimization_stats["cache_cleanup"] = TrainingStateManager.cleanup_old_caches()

        # If memory usage is high, clean up memory-heavy items
        if initial_size_mb > TrainingStateManager.MEMORY_THRESHOLD_MB:
            optimization_stats["memory_cleanup"] = TrainingStateManager.cleanup_memory_heavy_items()

        # Calculate final statistics
        final_size = TrainingStateManager.get_session_state_size()
        final_size_mb = final_size / (1024 * 1024)

        optimization_stats["final_size_mb"] = round(final_size_mb, 2)
        optimization_stats["memory_saved_mb"] = round(initial_size_mb - final_size_mb, 2)

        return optimization_stats

    @staticmethod
    def clear_training_results():
        """Clear all training-related results from session state."""
        keys_to_clear = [
            'training_results',
            'cv_results',
            'optuna_studies',
            'calibration_models',
            'model_predictions',
            'param_ranges_cache',
            'calibration_cache',
            'imbalance_handled',
            'imbalance_skipped',
            'training_predictions_cache',
            'training_prediction_models',
            'training_metrics_cache'
        ]

        cleared_count = 0
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
                cleared_count += 1

        return cleared_count

    @staticmethod
    def get_cache_info() -> Dict[str, Any]:
        """Get information about current cache usage."""
        cache_info = {
            "param_ranges_cache": 0,
            "calibration_cache": 0,
            "training_predictions_cache": 0,
            "training_metrics_cache": 0,
            "total_cache_items": 0
        }

        cache_keys = [
            'param_ranges_cache',
            'calibration_cache',
            'training_predictions_cache',
            'training_metrics_cache'
        ]

        for cache_key in cache_keys:
            if cache_key in st.session_state:
                cache_info[cache_key] = len(st.session_state[cache_key])
                cache_info["total_cache_items"] += cache_info[cache_key]

        return cache_info

    @staticmethod
    def init_session_caches():
        """Initialize session state caches if they don't exist."""
        cache_keys = [
            'param_ranges_cache',
            'calibration_cache',
            'training_predictions_cache',
            'training_metrics_cache'
        ]

        for cache_key in cache_keys:
            if cache_key not in st.session_state:
                st.session_state[cache_key] = {}

    @staticmethod
    def should_cleanup() -> bool:
        """Determine if cleanup is needed based on memory usage."""
        current_size_mb = TrainingStateManager.get_session_state_size() / (1024 * 1024)
        return current_size_mb > TrainingStateManager.MEMORY_THRESHOLD_MB

    @staticmethod
    def get_model_hash() -> str:
        """Generate a hash for the current model to use as cache key."""
        try:
            model = st.session_state.builder.model.get("active_model") or st.session_state.builder.model["model"]
            model_params = repr((st.session_state.builder.model.get('training_run', {}).get('run_id'),
                                 id(model), model.get_params() if hasattr(model, 'get_params') else str(model)))
            return hashlib.md5(model_params.encode()).hexdigest()[:8]
        except:
            return "unknown"

    @staticmethod
    def get_data_hash() -> str:
        """Generate a hash for the current dataset to use as cache key."""
        try:
            from components.model_training.utils.validation_utils import data_fingerprint
            builder = st.session_state.builder
            return data_fingerprint(builder.X_test, builder.y_test)
        except:
            return "unknown"

    @staticmethod
    def get_shared_cache_key(component: str, operation: str) -> str:
        """Generate a consistent cache key for cross-component sharing."""
        model_hash = TrainingStateManager.get_model_hash()
        data_hash = TrainingStateManager.get_data_hash()
        return f"{component}_{operation}_{model_hash}_{data_hash}"

    @staticmethod
    def cache_predictions(predictions: Dict[str, Any], key: str = "default"):
        """Cache model predictions for reuse across components."""
        TrainingStateManager.init_session_caches()
        cache_key = TrainingStateManager.get_shared_cache_key("predictions", key)
        st.session_state.training_predictions_cache[cache_key] = predictions

    @staticmethod
    def get_cached_predictions(key: str = "default") -> Optional[Dict[str, Any]]:
        """Retrieve cached model predictions."""
        if 'training_predictions_cache' not in st.session_state:
            return None

        cache_key = TrainingStateManager.get_shared_cache_key("predictions", key)
        return st.session_state.training_predictions_cache.get(cache_key)

    @staticmethod
    def cache_model_metrics(metrics: Dict[str, Any], key: str = "default"):
        """Cache model metrics for reuse across components."""
        TrainingStateManager.init_session_caches()
        cache_key = TrainingStateManager.get_shared_cache_key("metrics", key)
        st.session_state.training_metrics_cache[cache_key] = metrics

    @staticmethod
    def get_cached_metrics(key: str = "default") -> Optional[Dict[str, Any]]:
        """Retrieve cached model metrics."""
        if 'training_metrics_cache' not in st.session_state:
            return None

        cache_key = TrainingStateManager.get_shared_cache_key("metrics", key)
        return st.session_state.training_metrics_cache.get(cache_key)

    @staticmethod
    def get_model_info() -> Dict[str, Any]:
        """Get standardized model information for all components."""
        try:
            builder = st.session_state.builder
            model = builder.model.get("active_model") or builder.model["model"]

            return {
                "model": model,
                "model_type": builder.model["type"],
                "problem_type": builder.model["problem_type"],
                "X_train": builder.X_train,
                "y_train": builder.y_train,
                "X_test": builder.X_test,
                "y_test": builder.y_test,
                "has_training_results": hasattr(st.session_state, 'training_results'),
                "has_optuna_results": hasattr(st.session_state, 'optuna_results')
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def is_classification_model() -> bool:
        """Check if current model is for classification."""
        try:
            problem_type = st.session_state.builder.model["problem_type"]
            return problem_type in ["classification", "binary_classification", "multiclass_classification"]
        except:
            return False

    @staticmethod
    def is_regression_model() -> bool:
        """Check if current model is for regression."""
        try:
            problem_type = st.session_state.builder.model["problem_type"]
            return problem_type == "regression"
        except:
            return False

    @staticmethod
    def get_model_predictions() -> Optional[Dict[str, Any]]:
        """Get or generate model predictions for current test set."""
        # Check if predictions are already cached
        cached_predictions = TrainingStateManager.get_cached_predictions()
        if cached_predictions:
            return cached_predictions

        try:
            model_info = TrainingStateManager.get_model_info()
            if "error" in model_info:
                return None

            model = model_info["model"]
            X_test = model_info["X_test"]
            y_test = model_info["y_test"]

            predictions = {
                "y_pred": model.predict(X_test),
                "y_true": y_test
            }

            # Add probability predictions for classification
            if TrainingStateManager.is_classification_model() and hasattr(model, 'predict_proba'):
                predictions["y_prob"] = model.predict_proba(X_test)

            # Cache the predictions
            TrainingStateManager.cache_predictions(predictions)
            return predictions

        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            return None

    @staticmethod
    def clear_component_caches():
        """Clear all component-specific caches."""
        cache_keys = [
            'training_predictions_cache',
            'training_prediction_models',
            'training_metrics_cache',
            'param_ranges_cache',
            'calibration_cache'
        ]

        for key in cache_keys:
            if key in st.session_state:
                del st.session_state[key]

    @staticmethod
    def get_cache_stats() -> Dict[str, int]:
        """Get statistics about current cache usage."""
        stats = {}
        cache_keys = [
            'training_predictions_cache',
            'training_metrics_cache',
            'param_ranges_cache',
            'calibration_cache'
        ]

        for key in cache_keys:
            if key in st.session_state:
                stats[key] = len(st.session_state[key])
            else:
                stats[key] = 0

        stats['total_caches'] = sum(stats.values())
        return stats

    @staticmethod
    def reset_calibration_state():
        """Reset all model-specific training state (calibration, threshold optimization) when a new model is trained."""
        try:
            if hasattr(st.session_state, 'builder') and st.session_state.builder.model:
                model_dict = st.session_state.builder.model

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
                    if key in model_dict:
                        del model_dict[key]

                # Clear related caches from session state
                cache_keys_to_clear = ['calibration_cache', 'threshold_analysis_cache',
                                       'training_predictions_cache', 'training_prediction_models', 'training_metrics_cache']
                for cache_key in cache_keys_to_clear:
                    if cache_key in st.session_state:
                        st.session_state[cache_key].clear()

                # Clear any Streamlit cached functions for calibration and threshold analysis
                # This will force recalculation with the new model
                if hasattr(st, '_cached_data_cache'):
                    cache_keys_to_remove = []
                    for key in st._cached_data_cache.keys():
                        key_str = str(key).lower()
                        if any(term in key_str for term in ['calibration', 'threshold', 'roc', 'precision_recall']):
                            cache_keys_to_remove.append(key)
                    for key in cache_keys_to_remove:
                        del st._cached_data_cache[key]

        except Exception as e:
            # Log error but don't fail the training process
            if hasattr(st.session_state, 'logger'):
                st.session_state.logger.log_error(
                    "Calibration State Reset Failed",
                    {"error": str(e)}
                )

    @staticmethod
    def validate_training_state() -> Dict[str, bool]:
        """Validate that all required training state is available."""
        validation = {
            "builder_exists": hasattr(st.session_state, 'builder'),
            "model_trained": False,
            "test_data_available": False,
            "training_results_available": False
        }

        try:
            if validation["builder_exists"]:
                builder = st.session_state.builder
                validation["model_trained"] = builder.model is not None
                validation["test_data_available"] = (
                    hasattr(builder, 'X_test') and
                    hasattr(builder, 'y_test') and
                    builder.X_test is not None and
                    builder.y_test is not None
                )
                validation["training_results_available"] = hasattr(st.session_state, 'training_results')
        except:
            pass

        return validation


# Backward compatibility alias
SessionStateManager = TrainingStateManager
