# Copyright (c) 2025 Richard Wheeler
# Licensed under the Proprietary Evaluation License
# See LICENSE file for details
# For commercial licensing: richard.wheeler@priosym.com

from typing import Dict, List, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import traceback
from components.model_training.utils.imbalance_utils import (
    analyse_class_imbalance as analyse_class_imbalance_util,
    get_imbalance_recommendation as get_imbalance_recommendation_util,
    apply_resampling as apply_resampling_util
)
from components.model_training.utils.model_selection_utils import (
    select_final_model as select_final_model_util,
    reset_model_training_state
)
from utils.feature_importance_utils import analyze_feature_importance
from utils.logging.logger import MLLogger
from content.content_manager import ContentManager
from content.stage_info import ModelStage
import streamlit as st
from components.data_exploration.utils.missing_values_utils import analyze_missing_values
from components.data_exploration.utils.data_quality_utils import analyze_data_quality as analyze_data_quality_util
from components.data_exploration.utils.feature_relationship_utils import (
    get_feature_relationship_plots as get_feature_relationship_plots_util,
    analyze_feature_target_relationship as analyze_feature_target_relationship_util
)
from components.model_selection.model_recommendation_engine import get_model_recommendation
from components.model_selection.model_comparison import (
    quick_model_comparison,
    get_best_model_from_comparison,
    style_comparison_results
)
from components.model_selection.utils.compatibility_utils import check_xgboost_compatibility
from components.model_selection.utils.nonlinearity_detection import detect_nonlinearity
from components.model_selection.utils.model_options import get_model_options, get_problem_type_display

# ModelStage is now imported from content.stage_info

class Builder:
    def __init__(self):
        """Initialize the ML Builder with empty state."""
        self.data = None
        self.target_column = None
        self.training_data = None
        self.testing_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.feature_names = None
        self.encoders = {}
        self.encoding_mappings = {}  # Store encoding mappings for each column
        self.current_stage = ModelStage.DATA_LOADING
        self.stage_completion = {stage: False for stage in ModelStage}
        self.logger = MLLogger()
        self.original_dtypes = None  # Add this line to store original data types
        self.content_manager = ContentManager()  # Add content manager for text content
        
    def get_current_stage_info(self, stage=None) -> Dict[str, Any]:
        """Returns information about the current stage including description and requirements."""
        stage_to_use = stage if stage is not None else self.current_stage
        return self.content_manager.get_current_stage_info(stage_to_use)

    def get_calculation_explanation(self, calculation_type: str) -> Dict[str, str]:
        """Get explanation for a specific calculation type."""
        return self.content_manager.get_calculation_explanation(calculation_type)

    def get_statistical_explanation(self, test_type: str, values: Dict[str, float]) -> Dict[str, str]:
        """Get explanation for specific statistical test and its values."""
        return self.content_manager.get_statistical_explanation(test_type, values)

    def load_data(self, uploaded_file) -> Dict[str, Any]:
        """Load data with logging."""
        try:
            # Log only once at the start
            self.logger.log_user_action("Data Loading Started", {"file_name": uploaded_file.name})
            
            self.data = pd.read_csv(uploaded_file)
            self.original_dtypes = self.data.dtypes.copy()  # Store original dtypes right after loading
            
            basic_info = {
                "rows": len(self.data),
                "columns": len(self.data.columns),
                "column_names": list(self.data.columns),
                "missing_values": self.data.isnull().sum().to_dict()
            }
            
            self.stage_completion[ModelStage.DATA_LOADING] = True
            
            # Combine all stats into a single log message instead of multiple
            self.logger.log_calculation("Data Loading Complete", {
                "basic_info": basic_info,
                "dtypes": {col: str(dtype) for col, dtype in self.data.dtypes.to_dict().items()}
            })
            
            # Make sure to flush the logs
            self.logger.flush_logs()
            
            # Get basic info about the dataset
            #info = self.get_data_summary()
            
            return {
                "success": True,
                "message": "Data loaded successfully!",
                #"info": info
            }
        except Exception as e:
            self.logger.log_error("Data Loading Failed", {"error": str(e)})
            self.logger.flush_logs()
            return {
                "success": False,
                "message": f"Error loading data: {str(e)}"
            }

    def detect_problem_type(self) -> str:
        """Detect if this is a classification or regression problem using session state if available."""

        # Use session state problem type if available (set during data loading)
        if hasattr(st, 'session_state'):
            if hasattr(st.session_state, 'problem_type'):
                return st.session_state.problem_type

            # Fallback to individual flags if problem_type not available
            if hasattr(st.session_state, 'is_binary') and st.session_state.is_binary:
                return "binary_classification"
            if hasattr(st.session_state, 'is_multiclass') and st.session_state.is_multiclass:
                return "multiclass_classification"
            if hasattr(st.session_state, 'is_regression') and st.session_state.is_regression:
                return "regression"

        # Fallback to automatic detection if session state not available
        if self.y_train is None:
            return "unknown"

        # Check if target is categorical or numerical
        unique_values = len(np.unique(self.y_train))
        if unique_values == 2 or self.y_train.dtype == 'bool':
            return "binary_classification"
        elif 3 <= unique_values <= 20:
            return "multiclass_classification"
        return "regression"

    def analyse_feature_importance(self) -> Dict[str, Any]:
        """Analyse feature importance and return results."""
        try:
            # Get problem type for analysis
            problem_type = self.detect_problem_type()

            # Log the analysis start
            training_debug = {
                "X_train_shape": self.X_train.shape if self.X_train is not None else None,
                "y_train_shape": self.y_train.shape if self.y_train is not None else None,
                "problem_type": problem_type
            }
            self.logger.log_user_action("Feature Importance Analysis Started", training_debug)

            # Use the utility function for core analysis
            result = analyze_feature_importance(self.X_train, self.y_train, problem_type)

            # Log the analysis completion
            if result["success"]:
                analysis_debug = {
                    "total_features": len(result["feature_scores"]),
                    "protected_attributes_count": len(result["responsible_ai"]["protected_attributes"]),
                    "correlations_count": len(result["responsible_ai"]["correlations"]),
                    "quality_issues_count": len(result["responsible_ai"]["quality_issues"]),
                    "low_importance_features_count": len(result["responsible_ai"]["low_importance_features"])
                }
                self.logger.log_user_action("Feature Importance Analysis Completed", analysis_debug)
            else:
                self.logger.log_error("Feature importance analysis failed", {"error": result["message"]})

            return result

        except Exception as e:
            self.logger.log_error("Feature importance analysis failed", {"error": str(e)})
            return {
                "success": False,
                "message": f"Error analyzing feature importance: {str(e)}",
                "feature_scores": [],
                "visualization": None,
                "responsible_ai": {
                    "protected_attributes": [],
                    "correlations": [],
                    "quality_issues": [],
                    "low_importance_features": []
                }
            }


    """Data Exploration Methods"""

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics and initial data analysis with visualisations.

        Simplified version that only generates the data and visualizations actually
        used by the data exploration page and its components.
        """
        if self.data is None:
            return {"error": "No data loaded"}

        try:
            # Use the new utility function to generate the data summary
            from components.data_exploration.utils.data_summary_utils import generate_data_summary
            return generate_data_summary(self.data, self.target_column)
        except Exception as e:
            print(f"Error in get_data_summary: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Error generating data summary: {str(e)}"
            }

    def get_feature_relationship_plots(self, feature1: str, feature2: str, custom_data=None, grouping_feature=None) -> \
    Dict[str, Any]:
        """Generate plots and statistical analysis for the relationship between any two features.

        This method reuses the functionality of analyse_feature_target_relationship but adapts it
        for any pair of features rather than specifically feature vs target.

        Args:
            feature1: Name of the first feature to analyse
            feature2: Name of the second feature to analyse
            custom_data: Optional DataFrame to use instead of self.data
            grouping_feature: Optional feature to group data by for visualization

        Returns:
            Dictionary containing plots and statistical analysis results
        """
        # Use custom data if provided, otherwise use self.data
        data = custom_data if custom_data is not None else self.data
        return get_feature_relationship_plots_util(data, feature1, feature2, grouping_feature)

    def analyse_feature_target_relationship(self, feature: str, target: str, custom_data=None, grouping_feature=None) -> \
    Dict[str, Any]:
        """Analyse relationship between a feature and target variable.

        Args:
            feature: Name of the feature to analyse
            target: Name of the target variable
            custom_data: Optional DataFrame to use instead of self.data
            grouping_feature: Optional feature to group data by for visualization

        Returns:
            Dictionary containing analysis results and visualisations
        """
        # Use custom data if provided, otherwise use self.data
        data = custom_data if custom_data is not None else self.data
        return analyze_feature_target_relationship_util(data, feature, target, grouping_feature)

    def analyse_missing_values(self) -> Dict[str, Any]:
        """Analyse missing values and their patterns."""
        return analyze_missing_values(self.data)

    def analyse_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyse the quality of the provided dataset."""
        return analyze_data_quality_util(data)


    """Data Preprocessing Methods"""

    def suggest_encoding_strategies(self, data=None) -> Dict[str, Any]:
        """Suggest encoding strategies for categorical variables.

        Args:
            data: Optional data to use instead of self.data. If None, uses self.data.
        """
        from components.data_preprocessing.utils.categorical_encoding_utils import suggest_encoding_strategies

        if data is None:
            if self.data is None:
                return {"success": False, "message": "No data loaded"}
            data_to_use = self.data
        else:
            data_to_use = data

        return suggest_encoding_strategies(data_to_use, self.target_column)

    def handle_categorical_data(self, handling_dict: Dict[str, Dict], data=None, is_training=True) -> Dict[str, Any]:
        """Handle categorical variables with encoding or dropping options.

        Args:
            handling_dict: Dictionary with column names as keys and encoding strategies as values
            data: Optional data to use instead of self.data. If None, uses self.data.
            is_training: Whether this is being called on training data (to fit encoders) or test data (to apply existing encodings)
        """
        from components.data_preprocessing.utils.categorical_encoding_utils import handle_categorical_data

        if data is None:
            if self.data is None:
                return {"success": False, "message": "No data loaded"}
            data_to_use = self.data
        else:
            data_to_use = data

        # Call the utility function
        result = handle_categorical_data(
            data_to_use, handling_dict, self.target_column, self.encoding_mappings, is_training
        )

        if result["success"]:
            # Update Builder's state with the result
            if "data" in result:
                if data is None:
                    self.data = result["data"]
                elif is_training:
                    self.training_data = result["data"]
                else:
                    self.testing_data = result["data"]

            # Update encoding mappings
            if "encoding_mappings" in result:
                self.encoding_mappings = result["encoding_mappings"]

        return result

    def handle_outliers(self, column, strategy):
        """
        Handle outliers in a specified column using the chosen strategy.

        Args:
            column (str): The column to handle outliers in
            strategy (str): The strategy to use ('Remove', 'Remove Extreme', 'Cap', 'Isolation Forest', or 'None')

        Returns:
            dict: Result of the operation with success status and message
        """
        from components.data_preprocessing.utils.outlier_detection_utils import handle_outliers

        if self.training_data is None:
            return {
                "success": False,
                "message": "No training data available"
            }

        # Call the utility function
        result = handle_outliers(self.training_data, column, strategy)

        if result["success"] and "data" in result:
            # Update Builder's training data with the result
            self.training_data = result["data"]

        return result

    def suggest_outlier_strategies(self) -> Dict[str, Any]:
        """
        Suggest strategies for handling outliers in numerical columns.

        Returns:
            dict: Success status, message, and outlier handling suggestions for each numeric column
        """
        from components.data_preprocessing.utils.outlier_detection_utils import suggest_outlier_strategies

        if self.training_data is not None:
            data_to_use = self.training_data
        elif self.data is not None:
            data_to_use = self.data
        else:
            return {
                "success": False,
                "message": "No data available for analysis"
            }

        return suggest_outlier_strategies(data_to_use, self.target_column)

    def analyse_zero_values(self) -> Dict[str, Any]:
        """Analyse zero values in numerical columns, excluding binary features."""
        from components.data_preprocessing.utils.zero_values_utils import analyse_zero_values

        if self.data is None:
            return {
                "success": False,
                "message": "No data loaded"
            }

        return analyse_zero_values(self.data)

    def handle_zero_values(self, strategy_dict: Dict[str, str]) -> Dict[str, Any]:
        """Handle zero values according to specified strategies."""
        from components.data_preprocessing.utils.zero_values_utils import handle_zero_values

        if self.data is None:
            return {
                "success": False,
                "message": "No data loaded"
            }

        # Call the utility function
        result = handle_zero_values(self.data, strategy_dict)

        if result["success"] and "data" in result:
            # Update Builder's data with the result
            self.data = result["data"]

        return result

    def suggest_binning_strategies(self, use_training_data=False) -> Dict[str, Any]:
        """Suggest binning strategies for numerical and categorical variables."""
        from components.data_preprocessing.utils.binning_utils import suggest_binning_strategies

        if use_training_data and self.training_data is not None:
            data_to_use = self.training_data
        elif self.data is not None:
            data_to_use = self.data
        else:
            return {"success": False, "message": "No data available"}

        return suggest_binning_strategies(data_to_use, self.target_column)

    def apply_binning(self, strategy_dict: Dict[str, Dict[str, Any]], use_training_data=False) -> Dict[str, Any]:
        """Apply binning strategies to specified columns."""
        from components.data_preprocessing.utils.binning_utils import apply_binning

        # Determine which data to use
        if use_training_data and self.training_data is not None:
            training_data_to_use = self.training_data
            testing_data_to_use = self.testing_data  # Could be None
        elif self.data is not None:
            training_data_to_use = self.data
            testing_data_to_use = None  # No testing data available
        else:
            return {"success": False, "message": "No data available"}

        # Call the utility function
        result = apply_binning(
            training_data=training_data_to_use,
            testing_data=testing_data_to_use,
            strategy_dict=strategy_dict,
            target_column=self.target_column
        )

        # Update Builder's internal state if successful
        if result["success"]:
            if use_training_data:
                self.training_data = result["training_data"]
                if result.get("testing_data") is not None:
                    self.testing_data = result["testing_data"]
            else:
                self.data = result["training_data"]  # In this case, it's actually the main data

        return result


    """Model Selection Methods"""

    def select_model(self, model_type: str) -> Dict[str, Any]:
        """Select and configure a machine learning model."""
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from xgboost import XGBClassifier, XGBRegressor
        from lightgbm import LGBMClassifier, LGBMRegressor
        from catboost import CatBoostClassifier, CatBoostRegressor

        try:
            problem_type = self.detect_problem_type()
            if problem_type == "unknown":
                return {
                    "success": False,
                    "message": "Unable to determine problem type. Please ensure data is properly loaded."
                }

            # Simplified model configurations - only model instances needed
            # Parameter ranges are now handled by AdaptiveParameterRanges utility
            model_configs = {
                "classification": {
                    "logistic_regression": LogisticRegression(random_state=42, n_jobs=-1),
                    "naive_bayes": GaussianNB(),
                    "decision_tree": DecisionTreeClassifier(random_state=42),
                    "random_forest": RandomForestClassifier(random_state=42, n_jobs=-1),
                    "mlp": MLPClassifier(max_iter=1000, random_state=42),
                    "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
                    "catboost": CatBoostClassifier(random_state=42, verbose=False),
                    "xgboost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', nthread=-1),
                    "lightgbm": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
                },
                "regression": {
                    "linear_regression": LinearRegression(n_jobs=-1),
                    "ridge_regression": Ridge(random_state=42),
                    "decision_tree": DecisionTreeRegressor(random_state=42),
                    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
                    "mlp": MLPRegressor(max_iter=1000, random_state=42),
                    "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
                    "catboost": CatBoostRegressor(random_state=42, verbose=False),
                    "xgboost": XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse', nthread=-1),
                    "lightgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
                }
            }

            # Map problem types to model configuration keys
            config_key = "classification" if problem_type in ["binary_classification", "multiclass_classification", "classification"] else "regression"

            # Get the appropriate model configuration based on problem type
            if model_type not in model_configs[config_key]:
                return {
                    "success": False,
                    "message": f"Model type '{model_type}' not available for {problem_type} problems"
                }

            # Store model configuration and type
            self.model = {
                "model": model_configs[config_key][model_type],
                "type": model_type,
                "problem_type": problem_type
            }

            # Set quantile loss for gradient boosting if target is skewed
            if problem_type == "regression" and model_type == "gradient_boosting" and self.data is not None:
                target_col = self.target_column
                if target_col:
                    skewness = self.data[target_col].skew()
                    if abs(skewness) > 1.0:  # Same threshold as in model selection
                        self.model["model"].set_params(loss="quantile", alpha=0.5)

            self.stage_completion[ModelStage.MODEL_SELECTION] = True

            return {
                "success": True,
                "message": f"Selected {model_type} model for {problem_type}",
                "info": {
                    "problem_type": problem_type
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error selecting model: {str(e)}"
            }


    """Model Training Methods"""

    def auto_tune_hyperparameters(self, cv_folds: int = 5, n_iter: int = 20) -> Dict[str, Any]:
        """Perform automated hyperparameter tuning using RandomizedSearchCV."""
        if self.model is None:
            return {
                "success": False,
                "message": "No model selected. Please select a model first."
            }

        try:
            # Use the new hyperparameter tuner
            from components.model_training.utils.hyperparameter_tuner import HyperparameterTuner

            tuner = HyperparameterTuner()
            result = tuner.tune_random_search(
                model_dict=self.model,
                X_train=self.X_train,
                y_train=self.y_train,
                cv_folds=cv_folds,
                n_iter=n_iter
            )

            if not result["success"]:
                return result

            # Update the model with best parameters and state
            self.model["model"] = result["best_estimator"]
            self.model["best_params"] = result["best_params"]
            self.model["optimisation_method"] = result["optimisation_method"]
            self.model["cv_metrics"] = result["info"]["cv_metrics"]

            # Store adjusted model information if different models exist
            if not result["info"]["is_same_model"]:
                # Create adjusted model
                from copy import deepcopy
                adjusted_model = deepcopy(self.model["model"])
                adjusted_model.set_params(**result["info"]["adjusted_params"])

                self.model["adjusted_model"] = adjusted_model
                self.model["adjusted_params"] = result["info"]["adjusted_params"]
                self.model["adjusted_cv_metrics"] = result["info"]["adjusted_cv_metrics"]
            else:
                # If they're the same, just use references to the main model
                self.model["adjusted_model"] = self.model["model"]
                self.model["adjusted_params"] = self.model["best_params"]
                self.model["adjusted_cv_metrics"] = result["info"]["cv_metrics"]

            # Set active model and parameters
            self.model["active_model"] = self.model["model"]
            self.model["active_params"] = self.model["best_params"]

            # Reset calibration state when new model is active
            self._reset_calibration_state()

            return {
                "success": True,
                "message": result["message"],
                "info": result["info"]
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error during hyperparameter tuning: {str(e)}"
            }

    def auto_tune_hyperparameters_optuna(self, cv_folds: int = 5, n_trials: int = 50) -> Dict[str, Any]:
        """Tune hyperparameters using Optuna optimisation."""
        if self.model is None or self.X_train is None or self.y_train is None:
            return {
                "success": False,
                "message": "Model or training data not available"
            }

        try:
            # Use the new hyperparameter tuner
            from components.model_training.utils.hyperparameter_tuner import HyperparameterTuner

            tuner = HyperparameterTuner()
            result = tuner.tune_optuna(
                model_dict=self.model,
                X_train=self.X_train,
                y_train=self.y_train,
                cv_folds=cv_folds,
                n_trials=n_trials
            )

            if not result["success"]:
                return result

            # Store the best model and optimisation method
            # Note: The best_model from Optuna is already fitted on the full training data
            self.model["model"] = result["best_model"]
            self.model["best_model"] = result["best_model"]
            self.model["best_params"] = result["best_params"]
            self.model["optimisation_method"] = result["optimisation_method"]
            self.model["cv_metrics"] = result["info"]["cv_metrics"]
            self.model["selection_type"] = "optuna"  # Optuna doesn't have mean/adjusted distinction

            # No need to refit - Optuna already returns a fitted model
            # (Attempting to refit CatBoost models causes errors)

            # Store active model and parameters for consistency
            self.model["active_model"] = self.model["model"]
            self.model["active_params"] = self.model["best_params"]

            # Reset calibration state when new model is active
            self._reset_calibration_state()

            return {
                "success": True,
                "message": result["message"],
                "info": result["info"]
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error during hyperparameter optimisation: {str(e)}"
            }

    def analyse_class_imbalance(self) -> Dict[str, Any]:
        """
        Analyse class imbalance in the target variable and provide recommendations.

        Returns:
            Dict containing analysis results, metrics, and recommendations
        """
        return analyse_class_imbalance_util(self.X_train, self.y_train)

    def get_imbalance_recommendation(self) -> Dict[str, Any]:
        """
        Analyse dataset characteristics to recommend the best resampling method.

        Returns:
            Dict containing recommended method and explanation
        """
        return get_imbalance_recommendation_util(self.X_train, self.y_train)

    def apply_resampling(self, method: str) -> Dict[str, Any]:
        """
        Apply the specified resampling method to handle class imbalance.

        Args:
            method: The resampling method to use ('oversampling', 'undersampling', or 'smote')

        Returns:
            Dict containing success status and resampling results
        """
        result = apply_resampling_util(self.X_train, self.y_train, method)

        # Update the training data if resampling was successful
        if result.get("success") and "X_resampled" in result and "y_resampled" in result:
            self.X_train = result["X_resampled"]
            self.y_train = result["y_resampled"]

            # Remove the raw resampled data from the result to keep the interface clean
            result_clean = result.copy()
            del result_clean["X_resampled"]
            del result_clean["y_resampled"]
            return result_clean

        return result

    def select_final_model(self, selection_type: str = "mean_score") -> Dict[str, Any]:
        """Select which model to use for subsequent stages (mean score vs adjusted score).

        Args:
            selection_type: Either "mean_score" or "adjusted_score"

        Returns:
            Dictionary with operation status and details
        """
        result = select_final_model_util(self.model, selection_type)

        # Update the model state if selection was successful
        if result.get("success") and "model_dict" in result:
            self.model = result["model_dict"]

            # Remove the model_dict from the result to keep the interface clean
            result_clean = result.copy()
            del result_clean["model_dict"]
            return result_clean

        return result

    def _reset_calibration_state(self):
        """Reset all model-specific training state (calibration, threshold optimization) when a new model is trained."""
        if self.model:
            self.model = reset_model_training_state(self.model)



    """Model Evaluation Methods"""

    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate the trained model's performance with comprehensive visualisations."""
        if self.model is None or self.X_test is None or self.y_test is None:
            return {
                "success": False,
                "message": "Model or test data not available"
            }

        try:
            # Import utility functions
            from components.model_evaluation.evaluation_utils.eval_core_utils import (
                extract_model_predictions, calculate_classification_metrics,
                calculate_regression_metrics, get_classification_report,
                prepare_evaluation_data, prepare_residuals_data,
                calculate_residuals_stats
            )
            from components.model_evaluation.evaluation_utils.eval_visualization_utils import (
                create_confusion_matrix, create_roc_curve, create_classification_learning_curve,
                create_actual_vs_predicted_plot, create_residuals_analysis_plot,
                create_regression_learning_curve
            )

            problem_type = self.model.get("problem_type", "unknown")

            # Use the active model if available (user-selected), otherwise use the default model
            if "active_model" in self.model:
                model_instance = self.model["active_model"]
            else:
                model_instance = self.model["model"]

            # Extract predictions using utility function
            y_pred, y_prob_matrix = extract_model_predictions(self.model, self.X_test)

            if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
                # Calculate metrics
                metrics = calculate_classification_metrics(self.y_test, y_pred)

                # Store metrics in model dictionary
                self.model["metrics"] = metrics

                # Prepare evaluation data
                cm, class_names = prepare_evaluation_data(self.y_test, y_pred)

                # Create threshold info for confusion matrix
                threshold_info = {
                    "threshold_optimized": self.model.get("threshold_optimized", False),
                    "optimal_threshold": self.model.get("optimal_threshold", 0.5),
                    "threshold_is_binary": self.model.get("threshold_is_binary", True)
                }

                # Create visualizations using utility functions
                confusion_fig = create_confusion_matrix(
                    self.y_test, y_pred, class_names, threshold_info=threshold_info
                )

                roc_fig = create_roc_curve(self.y_test, y_prob_matrix, class_names)

                learning_fig = create_classification_learning_curve(
                    model_instance, self.X_train, self.y_train
                )

                # Get classification report
                clf_report = get_classification_report(self.y_test, y_pred)

                return {
                    "success": True,
                    "message": "Model evaluation completed successfully",
                    "metrics": metrics,
                    "confusion_matrix": confusion_fig,
                    "roc_curve": roc_fig,
                    "learning_curve": learning_fig,
                    "classification_report": clf_report
                }

            else:  # regression
                # Calculate metrics
                metrics = calculate_regression_metrics(self.y_test, y_pred)

                # Store metrics in model dictionary
                self.model["metrics"] = metrics

                # Prepare residuals data for logging
                residuals_df = prepare_residuals_data(self.y_test, y_pred)
                residuals_stats = calculate_residuals_stats(residuals_df['Residuals'].values)

                # Log residuals statistics for debugging
                print(f"Original Residuals - Mean: {residuals_stats['mean']:.4f}, Std: {residuals_stats['std']:.4f}")
                print(f"Original Residuals range - Min: {residuals_stats['min']:.4f}, Max: {residuals_stats['max']:.4f}")

                # Create visualizations using utility functions
                pred_fig = create_actual_vs_predicted_plot(self.y_test, y_pred)

                residuals_fig = create_residuals_analysis_plot(self.y_test, y_pred)

                learning_fig = create_regression_learning_curve(
                    model_instance, self.X_train, self.y_train
                )

                return {
                    "success": True,
                    "message": "Model evaluation completed successfully",
                    "metrics": metrics,
                    "prediction_plot": pred_fig,
                    "residuals_plot": residuals_fig,
                    "learning_curve": learning_fig
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error evaluating model: {str(e)}"
        }


    """Feature Selection Methods"""

    def update_features(self, features_to_remove: List[str]) -> Dict[str, Any]:
        """Update feature set by removing specified features.

        This method delegates to the feature selection utility component.
        """
        from components.feature_selection.utils.feature_utils import update_features as update_features_impl
        return update_features_impl(self, features_to_remove)

    def run_boruta_selection(self) -> Dict[str, Any]:
        """
        Run Boruta feature selection algorithm.

        This method is deprecated. The Boruta selection functionality has been moved
        to the AutomatedFeatureSelectionComponent for better modularity.

        Returns:
            Dict containing deprecation message
        """
        return {
            "success": False,
            "message": "This method has been deprecated. Please use the AutomatedFeatureSelectionComponent directly."
        }


    """Model Explanations Methods"""

    def get_protected_attributes(self) -> List[str]:
        """
        Automatically detect potential protected attributes in the dataset.

        Returns:
            List[str]: List of column names that might be protected attributes
        """
        from components.model_explanation.explanation_utils.protected_attributes_utils import get_protected_attributes

        if self.X_test is None:
            return []

        return get_protected_attributes(self.X_test)

    def explain_model(self) -> Dict[str, Any]:
        """Generate model explanations using SHAP values."""
        from components.model_explanation.explanation_utils.model_explanation_core import explain_model

        result = explain_model(
            model_dict=self.model,
            X_train=self.X_train,
            X_test=self.X_test
        )

        # Update stage completion if successful
        if result.get("success", False):
            self.stage_completion[ModelStage.MODEL_EXPLANATION] = True

        return result

    def explain_prediction(self, row_index: int) -> Dict[str, Any]:
        """Generate explanation for a specific prediction."""
        from components.model_explanation.explanation_utils.individual_explanation_utils import explain_prediction

        return explain_prediction(
            model_dict=self.model,
            X_train=self.X_train,
            X_test=self.X_test,
            y_test=self.y_test,
            row_index=row_index
        )

    def analyse_model_limitations(self) -> Dict[str, Any]:
        """Analyse and identify potential limitations and considerations of the current model."""
        from components.model_evaluation.evaluation_utils.model_limitations_utils import analyze_model_limitations

        # Get problem type
        problem_type = self.model.get("problem_type", "unknown") if self.model else "unknown"

        return analyze_model_limitations(
            model_dict=self.model,
            X_train=self.X_train,
            y_train=self.y_train,
            problem_type=problem_type
        )

    def generate_ale(self, feature_name: str, num_bins: int = 50, sample_size: int = 5000) -> go.Figure:
        """Generate Accumulated Local Effects Plot for a feature with optimisations for large datasets.

        Args:
            feature_name: Name of the feature to analyse
            num_bins: Number of bins to use for continuous features
            sample_size: Maximum number of samples to use for calculation
        """
        from components.model_explanation.explanation_utils.ale_utils import generate_ale

        return generate_ale(
            model_dict=self.model,
            X_train=self.X_train,
            X_test=self.X_test,
            feature_name=feature_name,
            num_bins=num_bins,
            sample_size=sample_size
        )

    # Model Selection delegation methods
    def get_model_recommendation(self) -> Dict[str, Any]:
        """Get model recommendation based on dataset characteristics."""
        return get_model_recommendation(
            training_data=self.training_data,
            target_column=self.target_column,
            problem_type=self.detect_problem_type()
        )

    def get_quick_model_comparison(self, sample_size=1000, exclude_xgboost=False) -> pd.DataFrame:
        """Perform quick model comparison using data samples."""
        return quick_model_comparison(
            training_data=self.training_data,
            testing_data=self.testing_data,
            target_column=self.target_column,
            problem_type=self.detect_problem_type(),
            sample_size=sample_size,
            exclude_xgboost=exclude_xgboost
        )

    def get_best_model_from_comparison(self, results_df) -> tuple:
        """Get the best performing model from comparison results."""
        return get_best_model_from_comparison(
            results_df=results_df,
            problem_type=self.detect_problem_type()
        )

    def style_comparison_results(self, results_df, best_metric) -> Any:
        """Style comparison results DataFrame to highlight best model."""
        return style_comparison_results(
            results_df=results_df,
            best_metric=best_metric
        )

    def check_xgboost_compatibility(self) -> bool:
        """Check if XGBoost is compatible with current dataset."""
        return check_xgboost_compatibility(
            data=self.training_data,
            target_column=self.target_column,
            problem_type=self.detect_problem_type()
        )

    def detect_data_nonlinearity(self) -> bool:
        """Detect if there are non-linear relationships in the data."""
        return detect_nonlinearity(
            data=self.training_data,
            target_column=self.target_column,
            problem_type=self.detect_problem_type()
        )

    def get_model_options(self, xgboost_compatible=True) -> Dict[str, str]:
        """Get available model options based on problem type."""
        return get_model_options(
            problem_type=self.detect_problem_type(),
            xgboost_compatible=xgboost_compatible
        )

    def get_problem_type_display(self) -> str:
        """Get user-friendly display name for current problem type."""
        return get_problem_type_display(self.detect_problem_type())

    def auto_select_and_train_model(
        self,
        optimization_method='optuna',
        cv_folds=5,
        n_iter=50,
        auto_handle_imbalance=True,
        show_analysis=True,
        model_override=None
    ) -> Dict[str, Any]:
        """
        Convenience method to run automated model selection and training.

        This method provides a single entry point to the automated model selection
        and training component, which replicates 100% of manual model selection
        (Page 5) and model training (Page 6) functionality.

        The automated workflow includes:
        1. Problem type detection and model recommendation
        2. Model comparison and intelligent selection
        3. Class imbalance handling (classification only)
        4. Hyperparameter optimization (Random Search or Optuna)
        5. Model training with cross-validation
        6. Classification-specific optimizations (calibration, threshold analysis)

        Args:
            optimization_method: 'random_search' or 'optuna' (default: 'optuna')
            cv_folds: Number of cross-validation folds (default: 5, adaptive)
            n_iter: Number of parameter configurations/trials (default: 50, adaptive)
            auto_handle_imbalance: Automatically handle class imbalance (default: True)
            show_analysis: Show step-by-step analysis (default: True)
            model_override: Force selection of specific model (optional)

        Returns:
            Dict containing:
                - success: bool - Whether the process completed successfully
                - summary: str - High-level summary message
                - details: dict - Detailed results from each step
                - error: str - Error message if failed (optional)

        Example:
            >>> result = builder.auto_select_and_train_model(
            ...     optimization_method='optuna',
            ...     show_analysis=True
            ... )
            >>> if result['success']:
            ...     print(f"Model trained: {result['details']['selected_model']}")
            ...     print(f"Best score: {result['details']['best_score']}")
        """
        from utils.automated_model_selection_training import AutomatedModelSelectionTraining

        auto_trainer = AutomatedModelSelectionTraining(
            builder=self,
            logger=st.session_state.logger,
            optimization_method=optimization_method,
            cv_folds=cv_folds,
            n_iter=n_iter,
            auto_handle_imbalance=auto_handle_imbalance,
            show_analysis=show_analysis,
            model_override=model_override
        )

        return auto_trainer.run()

