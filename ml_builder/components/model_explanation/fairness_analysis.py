from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
    equalized_odds_difference,
    demographic_parity_difference,
    count
)

# Constants for fairness analysis
FAIRNESS_THRESHOLD = 0.8
EXCELLENT_FAIRNESS_THRESHOLD = 0.9
SEVERE_BIAS_THRESHOLD = 0.6
PERFECT_FAIRNESS_SCORE = 1.0
WORST_FAIRNESS_SCORE = 0.0
SIGNIFICANT_DIFFERENCE_THRESHOLD = 0.2
MAX_UNIQUE_VALUES_FOR_ANALYSIS = 20
MIN_SAMPLE_SIZE_WARNING = 10

# UI and analysis constants
MAX_FEATURES_BEFORE_WARNING = 15
FAST_ANALYSIS_FEATURE_LIMIT = 10
CATEGORICAL_THRESHOLD = 10
HIGH_CARDINALITY_WARNING_LIMIT = 15
MULTICLASS_THRESHOLD = 2
UI_COLUMNS_STANDARD = 3
UI_COLUMNS_METRICS = 4

# Impact calculation constants
EXAMPLE_MONTHLY_APPLICATIONS = 1000
EXCELLENT_BIAS_IMPACT_FACTOR = 0.1
GOOD_BIAS_IMPACT_FACTOR = 0.3
CONCERNING_BIAS_IMPACT_FACTOR = 0.5
DISCRIMINATION_FACTOR = 0.6

# === UTILITY FUNCTIONS ===

def rmse_score(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def safe_auc_score(y_true, y_pred):
    """Calculate AUC score with error handling for multiclass"""
    try:
        # For multiclass, use macro average
        if len(np.unique(y_true)) > MULTICLASS_THRESHOLD:
            # AUC requires probabilities for multiclass, but we have class labels
            # Skip AUC for multiclass when we only have predictions
            return np.nan
        else:
            return roc_auc_score(y_true, y_pred)
    except (ValueError, AttributeError, TypeError) as e:
        # Handle specific sklearn errors that can occur with AUC calculation
        return np.nan

def safe_precision_score(y_true, y_pred):
    """Calculate precision score with zero division and multiclass handling"""
    try:
        # For multiclass, use macro average
        if len(np.unique(y_true)) > MULTICLASS_THRESHOLD:
            return precision_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            return precision_score(y_true, y_pred, zero_division=0)
    except (ValueError, AttributeError, TypeError) as e:
        # Handle specific sklearn errors that can occur with precision calculation
        return np.nan

def safe_f1_score(y_true, y_pred):
    """Calculate F1 score with zero division and multiclass handling"""
    try:
        # For multiclass, use macro average
        if len(np.unique(y_true)) > MULTICLASS_THRESHOLD:
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            return f1_score(y_true, y_pred, zero_division=0)
    except (ValueError, AttributeError, TypeError) as e:
        # Handle specific sklearn errors that can occur with F1 calculation
        return np.nan

def safe_recall_score(y_true, y_pred):
    """Calculate recall score with zero division and multiclass handling"""
    try:
        # For multiclass, use macro average
        if len(np.unique(y_true)) > MULTICLASS_THRESHOLD:
            return recall_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            return recall_score(y_true, y_pred, zero_division=0)
    except (ValueError, AttributeError, TypeError) as e:
        # Handle specific sklearn errors that can occur with recall calculation
        return np.nan

def true_positive_rate(y_true, y_pred):
    """Calculate the true positive rate (sensitivity/recall)."""
    if np.sum(y_true == 1) == 0:
        return 0.0
    return np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)

def is_metric_column(col_name):
    """
    Determine if a column name represents a metric/score rather than a feature.
    
    This function filters out columns that appear to contain model evaluation metrics,
    scores, or calculated values that shouldn't be used for fairness analysis.
    Such columns are typically outputs or derivatives of model predictions.
    
    Args:
        col_name (str): The column name to evaluate
    
    Returns:
        bool: True if the column appears to be a metric/score column, False otherwise
    
    Detection Logic:
        Checks if column name (case-insensitive) contains any of these keywords:
        - accuracy, precision, recall, f1
        - roc, auc (ROC curve related)  
        - error, loss, mse, mae, rmse (error metrics)
        - r2 (R-squared)
        - score, metric (generic score indicators)
    
    Examples:
        >>> is_metric_column('age')              # False - regular feature
        >>> is_metric_column('model_accuracy')   # True - contains 'accuracy'
        >>> is_metric_column('F1_Score')         # True - contains 'f1' and 'score'
        >>> is_metric_column('precision_weighted')  # True - contains 'precision'
        >>> is_metric_column('customer_score')   # True - contains 'score'
    
    Notes:
        - Used to automatically exclude metric columns from fairness analysis
        - Prevents circular analysis (using model outputs to evaluate model fairness)
        - Conservative approach: flags any column that might be a metric
        - Case-insensitive matching for robustness across naming conventions
    """
    metric_keywords = ['accuracy', 'precision', 'recall', 'f1', 'roc', 'auc', 'error',
                     'loss', 'mse', 'mae', 'rmse', 'r2', 'score', 'metric']
    
    col_lower = col_name.lower()
    
    for keyword in metric_keywords:
        if keyword in col_lower:
            return True
    
    return False

def get_accessible_color_palette():
    """Return colorblind-friendly color palette."""
    return {
        'good': '#2E8B57',     # Sea Green
        'warning': '#FF8C00',  # Dark Orange  
        'bad': '#DC143C',      # Crimson
        'neutral': '#4682B4'   # Steel Blue
    }

# === FEATURE SELECTION INTERFACE ===

def create_clean_feature_selection_interface(valid_columns, protected_cols, filtered_columns):
    """
    Create an intuitive Streamlit interface for selecting features to analyze for bias.
    
    This function renders a comprehensive UI that helps users select which features
    (protected attributes) to analyze for fairness. It provides smart defaults,
    explains feature characteristics, and warns about potential analysis issues.
    
    Args:
        valid_columns (list): List of column names eligible for fairness analysis
                             (excludes metric columns and other unsuitable features)
        protected_cols (list): Auto-detected protected attribute column names
                              (e.g., age, gender, race) found using pattern matching
        filtered_columns (list): Column names that were filtered out (e.g., metric columns)
                                to show users what was excluded and why
    
    Returns:
        list: Selected column names to analyze for fairness, or empty list if none selected
    
    UI Features:
        Smart Selection (Recommended):
            - Uses auto-detected protected attributes as defaults
            - Shows explanation of why each feature was selected
            - Displays feature characteristics and metadata
            
        Custom Selection:
            - For datasets ≤20 features: Visual checkbox interface with feature metadata
            - For larger datasets: Traditional multiselect with smart defaults
            - Groups features by type (categorical vs numeric)
            - Shows unique value counts and analysis suitability warnings
            
        Analyze All Features:
            - Selects all valid columns for comprehensive analysis
            - Warns about performance implications for large feature sets
    
    Smart Warnings and Recommendations:
        - Performance warnings for >15 features with option to limit
        - High-cardinality warnings for features with >20 unique values
        - Small group size warnings for groups with <10 samples  
        - Actionable recommendations for each type of issue detected
        
    UI Organization:
        1. Auto-detection results and excluded columns summary
        2. Selection method choice (Smart/Custom/All)
        3. Feature selection interface (varies by method)
        4. Analysis warnings and recommendations
        5. Final selection summary and validation
    
    Notes:
        - Accesses st.session_state.builder.X_test for feature analysis
        - All UI text includes helpful explanations and context
        - Provides educational content about fairness analysis
        - Gracefully handles edge cases (no features, all filtered, etc.)
        
    Example:
        >>> selected_features = create_clean_feature_selection_interface(
        ...     valid_columns=['age', 'income', 'location'],
        ...     protected_cols=['age'],
        ...     filtered_columns=['score_column']
        ... )
        >>> print(f"Selected: {selected_features}")
    """
    
    st.markdown("## 🔍 Feature Selection for Fairness Analysis")
    
    # === 1. SHOW WHAT WE FOUND ===
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if protected_cols:
            st.success(f"🎯 **Auto-detected protected attributes:** {', '.join(protected_cols)}")
        else:
            st.info("🔍 No protected attributes auto-detected. You can manually select features below.")
    
    with col2:
        if filtered_columns:
            with st.expander("📊 Excluded Columns", expanded=False):
                st.caption("These columns were excluded as they appear to be metrics:")
                for col in filtered_columns:
                    st.caption(f"• {col}")

    # === 2. SELECTION METHOD ===
    st.markdown("### 🎛️ Choose Analysis Scope")
    
    selection_method = st.radio(
        "Select your approach:",
        [
            "🎯 Smart Selection (Recommended)",
            "🎨 Custom Selection", 
            "🔄 Analyze All Features"
        ],
        key="selection_method",
        horizontal=True
    )

    # === 3. CONFIGURATION BASED ON SELECTION ===
    columns_to_analyse = []
    
    if selection_method == "🎯 Smart Selection (Recommended)":
        if protected_cols:
            columns_to_analyse = [col for col in protected_cols if col in valid_columns]
            
            st.info(f"""
            ✅ **Using {len(columns_to_analyse)} auto-detected features:** {', '.join(columns_to_analyse)}
            
            These were automatically identified as potential protected attributes based on:
            - Low number of unique values (≤{MAX_UNIQUE_VALUES_FOR_ANALYSIS})
            - Common protected attribute names (age, gender, race, etc.)
            - Data type patterns
            """)
        else:
            st.warning("⚠️ **No Protected Attributes Automatically Detected**\n\n"
                       "We couldn't find features with common protected attribute names "
                       "(like 'age', 'gender', 'race', 'income') or features with suitable "
                       "characteristics for fairness analysis.\n\n"
                       "**What This Means:** Your dataset may not contain obvious demographic features, "
                       "or they may be named differently than expected.\n\n"
                       "**Next Steps:** Use 'Custom Selection' below to manually choose features "
                       "that represent demographic groups or protected classes in your use case.")
            return []
    
    elif selection_method == "🎨 Custom Selection":
        st.markdown("**Select specific features to analyze:**")
        
        # Create a more visual selection interface
        if len(valid_columns) <= 20:
            # For smaller datasets, show columns with metadata
            selected_columns = []
            
            # Group columns by type for better organization
            categorical_cols = []
            numeric_cols = []
            
            for col in valid_columns:
                sample_data = st.session_state.builder.X_test[col]
                if sample_data.nunique() <= CATEGORICAL_THRESHOLD:
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)
            
            if categorical_cols:
                st.markdown("**📊 Categorical Features (Good for fairness analysis):**")
                cat_cols = st.columns(min(UI_COLUMNS_STANDARD, len(categorical_cols)))
                for i, col in enumerate(categorical_cols):
                    with cat_cols[i % UI_COLUMNS_STANDARD]:
                        sample_data = st.session_state.builder.X_test[col]
                        unique_count = sample_data.nunique()
                        is_selected = st.checkbox(
                            f"{col}",
                            value=col in (protected_cols or []),
                            key=f"cat_{col}"
                        )
                        st.caption(f"{unique_count} unique values")
                        if is_selected:
                            selected_columns.append(col)
            
            if numeric_cols:
                st.markdown("**🔢 Numeric Features (May need binning):**")
                num_cols = st.columns(min(UI_COLUMNS_STANDARD, len(numeric_cols)))
                for i, col in enumerate(numeric_cols):
                    with num_cols[i % UI_COLUMNS_STANDARD]:
                        sample_data = st.session_state.builder.X_test[col]
                        unique_count = sample_data.nunique()
                        is_selected = st.checkbox(
                            f"{col}",
                            value=False,
                            key=f"num_{col}"
                        )
                        st.caption(f"{unique_count} unique values")
                        if unique_count > CATEGORICAL_THRESHOLD:
                            st.caption("⚠️ Many values - may need grouping")
                        if is_selected:
                            selected_columns.append(col)
            
            columns_to_analyse = selected_columns
            
        else:
            # For larger datasets, use traditional multiselect but with better defaults
            default_selection = protected_cols if protected_cols else []
            
            columns_to_analyse = st.multiselect(
                "Choose features to analyze:",
                options=valid_columns,
                default=default_selection,
                help="💡 Tip: Start with demographic features like age, gender, location, etc."
            )
        
        if columns_to_analyse:
            st.success(f"✅ Selected {len(columns_to_analyse)} features: {', '.join(columns_to_analyse)}")
        else:
            st.info("👆 **Feature Selection Required**\n\n"
                    "Please select at least one feature from the options above to perform fairness analysis. "
                    "We recommend starting with demographic features like age, location, or other attributes "
                    "that might create different groups in your data.")
    
    else:  # "🔄 Analyze All Features"
        columns_to_analyse = valid_columns
        
        st.info(f"""
        📋 **Analyzing all {len(valid_columns)} available features.**
        
        This will analyze every feature in your dataset for potential bias.
        """)

    # === 4. SMART WARNINGS AND RECOMMENDATIONS ===
    if columns_to_analyse:
        # Check for potential issues
        warnings = []
        
        # Check for too many features
        if len(columns_to_analyse) > MAX_FEATURES_BEFORE_WARNING:
            warnings.append(f"⏱️ Analyzing {len(columns_to_analyse)} features may take longer")
            
            # Offer to limit
            limit_analysis = st.checkbox(
                f"Limit to first {FAST_ANALYSIS_FEATURE_LIMIT} features for faster analysis?", 
                value=False,
                key="limit_features"
            )
            if limit_analysis:
                columns_to_analyse = columns_to_analyse[:FAST_ANALYSIS_FEATURE_LIMIT]
                st.info(f"Limited to first {FAST_ANALYSIS_FEATURE_LIMIT} features: {', '.join(columns_to_analyse)}")
        
        # Check for high-cardinality features
        high_cardinality = []
        for col in columns_to_analyse:
            if col in valid_columns:
                unique_count = st.session_state.builder.X_test[col].nunique()
                if unique_count > MAX_UNIQUE_VALUES_FOR_ANALYSIS:
                    high_cardinality.append((col, unique_count))
        
        if high_cardinality:
            warnings.append("🔢 Some features have many unique values")
        
        # Check for small groups
        small_group_features = []
        for col in columns_to_analyse:
            if col in valid_columns:
                min_group_size = st.session_state.builder.X_test[col].value_counts().min()
                if min_group_size < MIN_SAMPLE_SIZE_WARNING:
                    small_group_features.append((col, min_group_size))
        
        if small_group_features:
            warnings.append("👥 Some features have very small groups")
            
        # Show summary
        if warnings:
            st.markdown("### ⚠️ Analysis Notes")
            for warning in warnings:
                st.caption(warning)
            if high_cardinality:
                with st.expander("⚠️ High-Cardinality Features", expanded=False):
                    st.warning("These features have many unique values, which may affect analysis quality:")
                    for col, count in high_cardinality:
                        st.markdown(f"• **{col}**: {count} unique values")
                    
                    st.markdown("""
                    **Recommendations:**
                    - Consider grouping values into ranges (e.g., age groups: 18-25, 26-35, etc.)
                    - Focus on features with fewer categories for more reliable results
                    - Remove features that are essentially unique identifiers
                    """)

            if small_group_features:
                with st.expander("👥 Small Group Sizes", expanded=False):
                    st.warning("These features have very small groups, which may give unreliable results:")
                    for col, min_size in small_group_features:
                        st.markdown(f"• **{col}**: smallest group has {min_size} samples")
                    
                    st.markdown("""
                    **Consider:**
                    - Collecting more data for underrepresented groups
                    - Combining small categories where appropriate
                    - Interpreting results for small groups with caution
                    """)
                    
    return columns_to_analyse

# === CORE ANALYSIS FUNCTIONS ===

def analyse_model_fairness(X_test, y_test, predictions, columns_to_analyse, problem_type):
    """
    Analyze model fairness across protected attributes and demographic groups.
    
    This function performs comprehensive fairness analysis by calculating fairness metrics
    for each selected feature/attribute. It supports binary classification, multiclass 
    classification, and regression problems with different fairness measures for each.
    
    Args:
        X_test (pd.DataFrame): Test dataset features containing the protected attributes
        y_test (pd.Series or np.array): True target values for test dataset  
        predictions (np.array): Model predictions on test dataset
        columns_to_analyse (list): List of column names to analyze for bias/fairness
        problem_type (str): Type of ML problem - 'binary_classification', 
                           'multiclass_classification', or 'regression'
    
    Returns:
        dict: Comprehensive fairness analysis results containing:
            - success (bool): Whether analysis completed successfully
            - message (str): Error message if analysis failed
            - fairness_score (float): Overall fairness score (min across all features)
            - column_scores (dict): Individual fairness scores by feature
            - bias_detected (bool): Whether bias was detected in any feature
            - bias_types (list): List of bias descriptions for detected issues
            - recommendations (list): Actionable recommendations (empty in this function)
            - metric_frames (dict): Fairlearn MetricFrames for detailed analysis
            - feature_summaries (dict): Summary statistics for each analyzed feature
    
    Fairness Metrics by Problem Type:
        Binary Classification:
            - Demographic Parity: Equal positive prediction rates across groups
            - Equalized Odds: Equal true/false positive rates across groups
            - Overall Score: Minimum of the two above (stricter fairness standard)
            
        Multiclass Classification:
            - Demographic Parity: Equal prediction distributions across all classes
            - Accuracy Consistency: Similar accuracy across all groups
            - Overall Score: Minimum of the two above
            
        Regression:
            - Error Consistency: Similar Mean Absolute Error across groups
            - Overall Score: Ratio of best to worst group performance
    
    Notes:
        - Fairness scores range from 0 (completely unfair) to 1 (perfectly fair)
        - Scores >= 0.8 are generally considered acceptable
        - Scores < 0.6 indicate severe bias requiring immediate attention
        - Small sample sizes may produce unreliable results
        - Progress indicators are displayed during analysis for user feedback
    
    Raises:
        Does not raise exceptions - returns error information in result dict instead
        
    Example:
        >>> fairness_results = analyse_model_fairness(
        ...     X_test=df_test[['age', 'gender']], 
        ...     y_test=df_test['approved'],
        ...     predictions=model.predict(X_test),
        ...     columns_to_analyse=['age', 'gender'],
        ...     problem_type='binary_classification'
        ... )
        >>> print(f"Overall fairness: {fairness_results['fairness_score']:.3f}")
    """
    
    # Enhanced input validation
    if X_test is None or y_test is None or predictions is None:
        return _create_empty_fairness_results("Missing required data")
    
    if len(X_test) != len(y_test) or len(X_test) != len(predictions):
        return _create_empty_fairness_results("Inconsistent data lengths")
    
    if len(X_test) < MIN_SAMPLE_SIZE_WARNING:
        st.warning(f"⚠️ **Small Sample Size Warning**\n\n"
                   f"Your test dataset has only **{len(X_test)} samples**, which may not provide "
                   f"statistically reliable fairness results.\n\n"
                   f"**Recommended minimum:** At least 100 samples total, with at least 10 samples "
                   f"per demographic group.\n\n"
                   f"**Impact:** Results may be unstable and conclusions about bias may not be trustworthy. "
                   f"Consider collecting more data or using a larger test set if possible.")
    
    fairness_results = {
        "success": True,
        "message": "",
        "fairness_score": None,
        "column_scores": {},
        "bias_detected": False,
        "bias_types": [],
        "recommendations": [],
        "metric_frames": {},
        "feature_summaries": {}
    }
    
    try:
        # Define metrics based on problem type - handle both binary and multiclass classification
        if problem_type in ["binary_classification", "multiclass_classification", "classification"]:
            performance_metrics = {
                'accuracy': accuracy_score,
                'precision': safe_precision_score,
                'recall': safe_recall_score,
                'f1': safe_f1_score,
                'auc': safe_auc_score,
                'count': count
            }
            
            # For multiclass, use different fairness metrics that work with multiclass
            if problem_type == "multiclass_classification":
                fairness_metrics = {
                    'selection_rate': selection_rate,  # This works for multiclass
                    'accuracy': accuracy_score,  # Simple fairness check - accuracy across groups
                    'precision': safe_precision_score,  # Macro-averaged precision
                    'recall': safe_recall_score  # Macro-averaged recall
                }
            else:
                # Binary classification fairness metrics
                fairness_metrics = {
                    'selection_rate': selection_rate,
                    'false_positive_rate': false_positive_rate,
                    'false_negative_rate': false_negative_rate,
                    'true_positive_rate': true_positive_rate
                }
        else:  # regression
            performance_metrics = {
                'r2': r2_score,
                'mse': mean_squared_error,
                'rmse': rmse_score,
                'mae': mean_absolute_error,
                'count': count
            }
            
            fairness_metrics = {
                'mean_prediction': lambda y_true, y_pred: np.mean(y_pred),
                'mae': lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
                'mse': lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
                'r2': lambda y_true, y_pred: r2_score(y_true, y_pred)
            }
        
        # Add progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Analyse each protected attribute
        for i, feature_col in enumerate(columns_to_analyse):
            try:
                # Update progress
                progress = (i + 1) / len(columns_to_analyse)
                status_text.text(f"Analyzing feature {i+1}/{len(columns_to_analyse)}: {feature_col}")
                progress_bar.progress(progress)
                
                # Create MetricFrame for performance metrics
                perf_metric_frame = MetricFrame(
                    metrics=performance_metrics,
                    y_true=y_test,
                    y_pred=predictions,
                    sensitive_features=X_test[feature_col]
                )
                
                # Create MetricFrame for fairness metrics
                fairness_metric_frame = MetricFrame(
                    metrics=fairness_metrics,
                    y_true=y_test,
                    y_pred=predictions,
                    sensitive_features=X_test[feature_col]
                )
                
                # Store both MetricFrames
                fairness_results["metric_frames"][feature_col] = {
                    'performance': perf_metric_frame,
                    'fairness': fairness_metric_frame
                }
                
                # Calculate fairness scores - handle all classification types
                if problem_type in ["binary_classification", "multiclass_classification", "classification"]:
                    if problem_type == "multiclass_classification":
                        # For multiclass, use simpler fairness measures
                        try:
                            # Calculate demographic parity difference (works for multiclass)
                            dp_diff = demographic_parity_difference(
                                y_true=y_test,
                                y_pred=predictions,
                                sensitive_features=X_test[feature_col]
                            )
                            dp_score = PERFECT_FAIRNESS_SCORE - abs(dp_diff)
                            
                            # For multiclass, use accuracy consistency across groups as fairness measure
                            accuracy_frame = fairness_metric_frame.by_group['accuracy']
                            accuracy_min = accuracy_frame.min()
                            accuracy_max = accuracy_frame.max()
                            accuracy_fairness = accuracy_min / accuracy_max if accuracy_max > 0 else PERFECT_FAIRNESS_SCORE
                            
                            # Use the worse of the two scores
                            fairness_score = min(dp_score, accuracy_fairness)
                            
                            # Store individual scores for display
                            individual_scores = {
                                'demographic_parity': dp_score,
                                'accuracy_consistency': accuracy_fairness,
                                'overall': fairness_score
                            }
                        except Exception as e:
                            # Log error and continue with None values for this feature
                            fairness_score = None
                            individual_scores = {
                                'demographic_parity': None,
                                'accuracy_consistency': None,
                                'overall': None
                            }
                            # Could add logging here if needed: st.session_state.logger.log_error(...)
                    else:
                        # Binary classification - use traditional fairness metrics
                        try:
                            # Calculate demographic parity difference
                            dp_diff = demographic_parity_difference(
                                y_true=y_test,
                                y_pred=predictions,
                                sensitive_features=X_test[feature_col]
                            )
                            
                            # Calculate equalized odds difference
                            eo_diff = equalized_odds_difference(
                                y_true=y_test,
                                y_pred=predictions,
                                sensitive_features=X_test[feature_col]
                            )
                            
                            # Convert differences to fairness scores
                            dp_score = PERFECT_FAIRNESS_SCORE - abs(dp_diff)
                            eo_score = PERFECT_FAIRNESS_SCORE - abs(eo_diff)
                            
                            # Use the worse of the two scores
                            fairness_score = min(dp_score, eo_score)
                            
                            # Store individual scores for display
                            individual_scores = {
                                'demographic_parity': dp_score,
                                'equalized_odds': eo_score,
                                'overall': fairness_score
                            }
                        except Exception as e:
                            # Log error and continue with None values for this feature  
                            fairness_score = None
                            individual_scores = {
                                'demographic_parity': None,
                                'equalized_odds': None,
                                'overall': None
                            }
                            # Could add logging here if needed: st.session_state.logger.log_error(...)
                    
                else:  # regression
                    try:
                        # Calculate ratio of group MAEs
                        group_maes = fairness_metric_frame.by_group['mae']
                        max_mae = group_maes.max()
                        min_mae = group_maes.min()
                        
                        if max_mae > 0:
                            mae_ratio = min_mae / max_mae
                        else:
                            mae_ratio = PERFECT_FAIRNESS_SCORE
                            
                        fairness_score = mae_ratio
                        
                        # For regression, we only have one main fairness measure
                        individual_scores = {
                            'error_consistency': fairness_score,
                            'overall': fairness_score
                        }
                    except Exception as e:
                        fairness_score = None
                        individual_scores = {
                            'error_consistency': None,
                            'overall': None
                        }
                
                fairness_results["column_scores"][feature_col] = fairness_score
                
                # Generate enhanced feature summary
                fairness_results["feature_summaries"][feature_col] = {
                    "score": fairness_score,
                    "individual_scores": individual_scores,
                    "groups": X_test[feature_col].nunique(),
                    "smallest_group": X_test[feature_col].value_counts().min(),
                    "largest_group": X_test[feature_col].value_counts().max(),
                    "is_numeric": pd.api.types.is_numeric_dtype(X_test[feature_col])
                }
                
                # Check for bias
                if fairness_score is not None and fairness_score < FAIRNESS_THRESHOLD:
                    fairness_results["bias_detected"] = True
                    fairness_results["bias_types"].append(
                        f"Potential bias detected in {feature_col} (score: {fairness_score:.3f})"
                    )
                
            except Exception as e:
                st.error(f"❌ **Failed to analyze feature '{feature_col}'**\n\n"
                         f"**Error Details:** {str(e)}\n\n"
                         f"**Possible Causes:**\n"
                         f"- Insufficient data for this feature (too few samples)\n"
                         f"- All groups have identical values (no variance)\n"
                         f"- Data type incompatibility with fairness metrics\n\n"
                         f"**Recommendation:** Try selecting different features or check data quality for '{feature_col}'")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Calculate overall fairness score
        valid_scores = [score for score in fairness_results["column_scores"].values() if score is not None]
        if valid_scores:
            fairness_results["fairness_score"] = min(valid_scores)
        else:
            # If no valid scores, set to None and add explanation
            fairness_results["fairness_score"] = None
            fairness_results["message"] = "Could not calculate fairness scores for any selected features. This may be due to insufficient data variance, incompatible data types, or multiclass prediction format issues."
        
        return fairness_results
        
    except Exception as e:
        fairness_results["success"] = False
        fairness_results["message"] = (f"Fairness analysis failed: {str(e)}. "
                                       f"This may be due to: (1) Incompatible data types, "
                                       f"(2) Insufficient data variance, (3) All features having identical values, "
                                       f"or (4) Prediction format issues for multiclass models. "
                                       f"Please verify your data quality and feature selection.")
        return fairness_results

def _create_empty_fairness_results(message):
    """Create empty fairness results structure with error message."""
    return {
        "success": False,
        "message": message,
        "fairness_score": None,
        "column_scores": {},
        "bias_detected": False,
        "bias_types": [],
        "recommendations": [],
        "metric_frames": {},
        "feature_summaries": {}
    }

def generate_specific_recommendations(fairness_results, X_test, problem_type):
    """
    Generate specific, actionable recommendations for addressing detected bias.
    
    This function analyzes fairness results and generates targeted recommendations
    based on the type of bias detected, the problem type, and feature characteristics.
    Recommendations are prioritized by severity and include effort estimates.
    
    Args:
        fairness_results (dict): Results from analyse_model_fairness() containing
                               fairness scores and feature summaries
        X_test (pd.DataFrame): Test dataset features for analyzing group characteristics
        problem_type (str): ML problem type ('binary_classification', 
                           'multiclass_classification', or 'regression')
    
    Returns:
        list: List of recommendation dictionaries, each containing:
            - feature (str): Name of the biased feature
            - issue (str): Description of the fairness issue detected
            - action (str): Specific recommended action to address the issue
            - priority (str): Priority level ('Critical', 'High', 'Medium', 'Low')
            - effort (str): Estimated effort required ('Low', 'Medium', 'High')
    
    Recommendation Categories:
        Critical Priority:
            - Severe bias (score < 0.6) requiring immediate attention
            - Protected attribute violations with legal implications
            - Major demographic parity or equalized odds violations
            
        High Priority:
            - Moderate bias (score 0.6-0.8) that should be addressed
            - Small group sizes affecting statistical reliability
            - Protected attribute concerns with moderate impact
            
        Medium Priority:
            - High cardinality features needing data preprocessing
            - General fairness improvements for multiclass models
            - Feature engineering suggestions for better fairness
    
    Effort Estimates:
        Low: Data preprocessing, feature grouping, documentation
        Medium: Model retraining, fairness constraints, threshold tuning
        High: Data collection, new model architectures, major system changes
    
    Notes:
        - Only generates recommendations for features with scores < FAIRNESS_THRESHOLD
        - Takes into account problem-specific fairness challenges
        - Considers feature characteristics (cardinality, group sizes, data types)
        - Provides business context and actionable next steps
        
    Example:
        >>> recommendations = generate_specific_recommendations(
        ...     fairness_results={'column_scores': {'age': 0.5, 'gender': 0.9}},
        ...     X_test=test_data,
        ...     problem_type='binary_classification'
        ... )
        >>> for rec in recommendations:
        ...     print(f"{rec['priority']}: {rec['action']}")
    """
    
    recommendations = []
    
    for feature, score in fairness_results["column_scores"].items():
        if score is None or score >= FAIRNESS_THRESHOLD:
            continue
            
        # Analyze the feature characteristics
        feature_data = X_test[feature]
        n_groups = feature_data.nunique()
        group_sizes = feature_data.value_counts()
        summary = fairness_results.get("feature_summaries", {}).get(feature, {})
        individual_scores = summary.get("individual_scores", {})
        
        # Problem-type specific recommendations
        if problem_type == "multiclass_classification":
            # Multiclass-specific recommendations
            if score < SEVERE_BIAS_THRESHOLD:
                if 'demographic_parity' in individual_scores and individual_scores['demographic_parity'] and individual_scores['demographic_parity'] < SEVERE_BIAS_THRESHOLD:
                    recommendations.append({
                        "feature": feature,
                        "issue": "Severe demographic parity violation across classes",
                        "action": f"Review class prediction distributions for '{feature}' groups. Consider class-specific fairness constraints or rebalancing training data",
                        "priority": "Critical",
                        "effort": "High"
                    })
                
                if 'accuracy_consistency' in individual_scores and individual_scores['accuracy_consistency'] and individual_scores['accuracy_consistency'] < SEVERE_BIAS_THRESHOLD:
                    recommendations.append({
                        "feature": feature,
                        "issue": "Large accuracy gaps between groups",
                        "action": f"Investigate why some '{feature}' groups have much lower accuracy. Consider group-specific model tuning or feature engineering",
                        "priority": "Critical",
                        "effort": "High"
                    })
                    
            elif score < 0.8:
                recommendations.append({
                    "feature": feature,
                    "issue": "Moderate bias in multiclass predictions",
                    "action": f"Monitor '{feature}' groups for class-specific disparities. Consider ensemble methods that account for group membership",
                    "priority": "High", 
                    "effort": "Medium"
                })
                
            # Multiclass-specific guidance
            if n_groups > HIGH_CARDINALITY_WARNING_LIMIT:
                recommendations.append({
                    "feature": feature,
                    "issue": "Too many groups for multiclass fairness analysis",
                    "action": f"Group '{feature}' into 3-10 meaningful categories to better understand class-specific bias patterns",
                    "priority": "Medium",
                    "effort": "Low"
                })
                
        else:
            # Binary classification and regression recommendations
            if n_groups > 20:
                recommendations.append({
                    "feature": feature,
                    "issue": "Too many groups for effective analysis",
                    "action": f"Consider binning '{feature}' into fewer categories",
                    "priority": "Medium",
                    "effort": "Low"
                })
        
            if score < SEVERE_BIAS_THRESHOLD:
                recommendations.append({
                    "feature": feature,
                    "issue": "Severe fairness violations",
                    "action": f"Apply fairness constraints during model training",
                    "priority": "Critical",
                    "effort": "Medium"
                })
        
        # Common recommendations for all problem types
        if group_sizes.min() < MIN_SAMPLE_SIZE_WARNING:
            recommendations.append({
                "feature": feature,
                "issue": "Small group sizes affecting reliability",
                "action": f"Collect more data for underrepresented groups in '{feature}'",
                "priority": "High",
                "effort": "High"
            })
        
        # Protected attribute recommendations
        if feature.lower() in ['age', 'income', 'salary', 'gender', 'race', 'ethnicity']:
            if problem_type == "multiclass_classification":
                recommendations.append({
                    "feature": feature,
                        "issue": "Potential indirect discrimination across multiple classes",
                        "action": f"Apply fairness-aware preprocessing and consider class-specific bias mitigation for '{feature}'",
                        "priority": "High",
                    "effort": "Medium"
                })
            else:
                recommendations.append({
                    "feature": feature,
                    "issue": "Potential indirect discrimination",
                        "action": f"Consider fairness-aware preprocessing techniques for '{feature}'",
                    "priority": "High",
                    "effort": "Low"
                    })
    
    # Add general multiclass recommendations if applicable
    if problem_type == "multiclass_classification" and any(score < FAIRNESS_THRESHOLD for score in fairness_results["column_scores"].values() if score is not None):
        recommendations.append({
            "feature": "All features",
            "issue": "Multiclass fairness challenges detected",
            "action": "Consider using fairness-aware multiclass algorithms (e.g., fair multiclass classification with equalized odds)",
            "priority": "Medium",
            "effort": "High"
            })
    
    return recommendations

# === COMPREHENSIVE DASHBOARD ===

def create_comprehensive_fairness_dashboard(fairness_results, problem_type, columns_to_analyse):
    """Create a comprehensive single-page fairness dashboard."""
    
    fairness_score = fairness_results.get("fairness_score", PERFECT_FAIRNESS_SCORE)
    
    # === 1. EXECUTIVE SUMMARY ===
    st.markdown("## 📊 Fairness Analysis Dashboard")
    
    # Quick metrics row
    col1, col2, col3, col4 = st.columns(UI_COLUMNS_METRICS)
    
    with col1:
        if fairness_score is None:
            st.metric("Overall Fairness", "N/A", "❌ Analysis Failed")
        elif fairness_score >= EXCELLENT_FAIRNESS_THRESHOLD:
            st.metric("Overall Fairness", f"{fairness_score:.3f}", "🏆 Excellent")
        elif fairness_score >= FAIRNESS_THRESHOLD:
            st.metric("Overall Fairness", f"{fairness_score:.3f}", "✅ Good")
        else:
            st.metric("Overall Fairness", f"{fairness_score:.3f}", "⚠️ Needs Attention")
    
    with col2:
        st.metric("Features Analyzed", len(fairness_results.get("column_scores", {})))
    
    with col3:
        issues = len(fairness_results.get("bias_types", []))
        st.metric("Issues Found", issues)
    
    with col4:
        fair_count = sum(1 for score in fairness_results.get("column_scores", {}).values() 
                        if score and score >= FAIRNESS_THRESHOLD)
        st.metric("Fair Features", f"{fair_count}/{len(fairness_results.get('column_scores', {}))}")

    # === 4. ALL FEATURES SUMMARY ===
    st.markdown("### 📋 All Features Analysis")
    
    if fairness_results.get("column_scores"):
        features_data = []
        for feature, score in fairness_results["column_scores"].items():
            if score is None:
                continue
            
            summary = fairness_results.get("feature_summaries", {}).get(feature, {})
            
            if score >= FAIRNESS_THRESHOLD:
                status = "✅ Fair"
                risk = "Low"
            else:
                status = "⚠️ Bias Risk"
                risk = "High" if score < SEVERE_BIAS_THRESHOLD else "Medium"
            
            features_data.append({
                "Feature": feature,
                "Score": f"{score:.3f}",
                "Status": status,
                "Groups": summary.get('groups', 'N/A'),
                "Risk": risk
            })
        
        if features_data:
            st.dataframe(pd.DataFrame(features_data), width='stretch', hide_index=True)
            
            # Add expandable detailed scores
            with st.expander("🔍 Detailed Component Scores", expanded=False):
                st.markdown("**Fairness Component Breakdown by Feature:**")
                
                for feature, score in fairness_results["column_scores"].items():
                    if score is None:
                        continue
                    
                    summary = fairness_results.get("feature_summaries", {}).get(feature, {})
                    individual_scores = summary.get("individual_scores", {})
                    
                    if individual_scores:
                        st.markdown(f"**{feature}:**")
                        
                        if problem_type in ["binary_classification", "multiclass_classification", "classification"]:
                            if problem_type == "multiclass_classification":
                                # Multiclass display
                                if 'demographic_parity' in individual_scores and 'accuracy_consistency' in individual_scores:
                                    dp_score = individual_scores['demographic_parity']
                                    ac_score = individual_scores['accuracy_consistency']
                                    
                                    comp_detail_col1, comp_detail_col2, comp_detail_col3 = st.columns(3)
                                    
                                    with comp_detail_col1:
                                        if dp_score is not None:
                                            dp_status = "✅" if dp_score >= FAIRNESS_THRESHOLD else "⚠️"
                                            st.caption(f"Demographic Parity: {dp_score:.3f} {dp_status}")
                                    
                                    with comp_detail_col2:
                                        if ac_score is not None:
                                            ac_status = "✅" if ac_score >= FAIRNESS_THRESHOLD else "⚠️"
                                            st.caption(f"Accuracy Consistency: {ac_score:.3f} {ac_status}")
                                    
                                    with comp_detail_col3:
                                        if score is not None:
                                            overall_status = "✅" if score >= FAIRNESS_THRESHOLD else "⚠️"
                                            st.caption(f"Overall: {score:.3f} {overall_status}")
                            else:
                                # Binary classification display
                                if 'demographic_parity' in individual_scores and 'equalized_odds' in individual_scores:
                                    dp_score = individual_scores['demographic_parity']
                                    eo_score = individual_scores['equalized_odds']
                                    
                                    comp_detail_col1, comp_detail_col2, comp_detail_col3 = st.columns(3)
                                    
                                    with comp_detail_col1:
                                        if dp_score is not None:
                                            dp_status = "✅" if dp_score >= FAIRNESS_THRESHOLD else "⚠️"
                                            st.caption(f"Demographic Parity: {dp_score:.3f} {dp_status}")
                                    
                                    with comp_detail_col2:
                                        if eo_score is not None:
                                            eo_status = "✅" if eo_score >= FAIRNESS_THRESHOLD else "⚠️"
                                            st.caption(f"Equalized Odds: {eo_score:.3f} {eo_status}")
                                    
                                    with comp_detail_col3:
                                        if score is not None:
                                            overall_status = "✅" if score >= FAIRNESS_THRESHOLD else "⚠️"
                                            st.caption(f"Overall: {score:.3f} {overall_status}")
                        
                        else:  # regression
                            if 'error_consistency' in individual_scores:
                                error_score = individual_scores['error_consistency']
                                if error_score is not None:
                                    error_status = "✅" if error_score >= FAIRNESS_THRESHOLD else "⚠️"
                                    st.caption(f"Error Consistency: {error_score:.3f} {error_status}")
                        
                        st.markdown("---")
    # === 2. INTERPRETATION ===
    st.markdown("### 🎯 What This Means in Plain English")
    
    if fairness_score is None:
        st.error("""
        ❌ **Analysis could not be completed.** The fairness analysis failed, likely due to:
        
        **Possible causes:**
        - Insufficient data for reliable analysis
        - All groups have identical predictions (no variance)
        - Data format incompatibility with fairness metrics
        - Selected features may not be suitable for fairness analysis
        
        **What to try:** Select different features or check your data quality.
        """)
    elif fairness_score >= EXCELLENT_FAIRNESS_THRESHOLD:
        if problem_type == "multiclass_classification":
            st.success("""
            🏆 **Excellent!** Your multiclass model treats all groups very fairly across all classes. There's minimal risk of discrimination.
            
            **Real-world impact:** 
            - All groups have similar prediction distributions across all classes
            - All groups receive similar prediction accuracy
            - Very low risk of systematic bias in any class assignment
            """)
        else:
            st.success("""
            🏆 **Excellent!** Your model treats all groups very fairly. There's minimal risk of discrimination.
            
            **Real-world impact:** Your model's decisions are likely to be fair and equitable across all groups.
            """)
    elif fairness_score >= FAIRNESS_THRESHOLD:
        if problem_type == "multiclass_classification":
            st.info("""
            ✅ **Good fairness.** Your multiclass model meets industry standards with only minor differences between groups.
            
            **Real-world impact:** 
            - Low risk of discrimination across classes
            - Some minor variations in class predictions or accuracy exist but are within acceptable ranges
            - Consider monitoring specific classes that may show larger disparities
            """)
        else:
            st.info("""
            ✅ **Good fairness.** Your model meets industry standards with only minor differences between groups.
            
            **Real-world impact:** Low risk of discrimination. Some minor variations exist but are within acceptable ranges.
            """)
    else:
        if problem_type == "multiclass_classification":
            st.warning("""
            ⚠️ **Fairness concerns detected.** Your multiclass model shows concerning differences between groups.
            
            **Real-world impact:** 
            - Some groups may be unfairly disadvantaged in certain class predictions
            - Prediction accuracy may vary significantly between groups
            - Risk of systematic bias in class assignments
            - **Action needed:** Review class-specific disparities and consider model adjustments
            """)
        else:
            st.warning("""
            ⚠️ **Fairness concerns detected.** Your model shows concerning differences between groups.
            
            **Real-world impact:** Some groups may be unfairly disadvantaged by your model's decisions.
            """)

    # === 3. REAL-WORLD IMPACT CALCULATOR ===
    st.markdown("### 🎲 Real-World Impact Example")
    
    if fairness_score is not None:
        st.markdown(f"**Scenario: Loan Application Decisions ({EXAMPLE_MONTHLY_APPLICATIONS:,} applications/month)**")
    
        # Calculate impact based on fairness score
        bias_severity = 1 - fairness_score
    else:
        st.markdown("**Impact analysis not available due to fairness calculation failure.**")
        return  # Exit early if no fairness score
    
    if bias_severity <= EXCELLENT_BIAS_IMPACT_FACTOR:  # Very fair (score >= 0.9)
        st.success(f"""
        **📊 Impact Analysis:**
        - **Fairness Score:** {fairness_score:.3f} (Excellent)
        - **Bias Severity:** {bias_severity:.1%}
        - **Estimated Impact:** ~{int(EXAMPLE_MONTHLY_APPLICATIONS * bias_severity * EXCELLENT_BIAS_IMPACT_FACTOR)} applications/month might experience unfair treatment
        - **Risk Level:** Minimal - Your model meets the highest fairness standards
        """)
    elif bias_severity <= SIGNIFICANT_DIFFERENCE_THRESHOLD:  # Good fairness (score >= 0.8)
        affected_apps = int(EXAMPLE_MONTHLY_APPLICATIONS * bias_severity * GOOD_BIAS_IMPACT_FACTOR)
        st.info(f"""
        **📊 Impact Analysis:**
        - **Fairness Score:** {fairness_score:.3f} (Good)
        - **Bias Severity:** {bias_severity:.1%}
        - **Estimated Impact:** ~{affected_apps} applications/month might experience unfair treatment
        - **Risk Level:** Low - Within acceptable industry standards
        
        **Example:** If Group A has 80% approval rate and Group B has 75% approval rate, 
        that's a {bias_severity:.1%} difference affecting roughly {affected_apps} qualified applicants.
        """)
    else:  # Concerning bias (score < 0.8)
        affected_apps = int(EXAMPLE_MONTHLY_APPLICATIONS * bias_severity * CONCERNING_BIAS_IMPACT_FACTOR)
        potential_discrimination = int(affected_apps * DISCRIMINATION_FACTOR)
        st.warning(f"""
        **📊 Impact Analysis:**
        - **Fairness Score:** {fairness_score:.3f} (Needs Attention)
        - **Bias Severity:** {bias_severity:.1%}
        - **Estimated Impact:** ~{affected_apps} applications/month might experience unfair treatment
        - **Potential Discrimination:** ~{potential_discrimination} qualified applicants might be wrongly rejected
        - **Risk Level:** High - Requires immediate attention
        
        **Example:** If Group A has 80% approval rate and Group B has {80 - (bias_severity * 100):.0f}% approval rate, 
        this {bias_severity:.1%} gap could affect {affected_apps} people monthly, with {potential_discrimination} potentially 
        qualified applicants being unfairly denied loans.
        """)
    
    # Add calculation explanation
    with st.expander("🧮 How We Calculate Impact", expanded=False):
        st.markdown(f"""
        **Calculation Method:**
        
        1. **Bias Severity:** 1 - Fairness Score = 1 - {fairness_score:.3f} = {bias_severity:.3f}
        2. **Impact Factor:** Based on severity level:
           - Excellent (≥0.9): 10% of bias severity
           - Good (≥0.8): 30% of bias severity  
           - Concerning (<0.8): 50% of bias severity
        3. **Affected Applications:** 1,000 × {bias_severity:.3f} × Impact Factor
        
        **Assumptions:**
        - Monthly application volume: 1,000
        - Not all bias translates to actual harm (hence the impact factor)
        - Real impact depends on business context and decision thresholds
        
        **Note:** These are estimates for illustration. Actual impact depends on your specific 
        business context, decision thresholds, and the nature of the bias detected.
        """)

    st.markdown("---")

    # === 5. FEATURE SELECTOR FOR DETAILED ANALYSIS ===
    st.markdown("### 🔍 Select Feature for Detailed Analysis")
    selected_feature = st.selectbox(
        "Choose a feature to analyze in detail:",
        columns_to_analyse,
        key="feature_selector",
        help="Select which feature you want to see detailed fairness analysis for."
    )

    # === 6. SELECTED FEATURE DEEP DIVE ===
    st.markdown(f"### 🔍 Detailed Analysis: {selected_feature}")
    
    feature_score = fairness_results["column_scores"].get(selected_feature, PERFECT_FAIRNESS_SCORE)
    feature_info = fairness_results.get("feature_summaries", {}).get(selected_feature, {})
    individual_scores = feature_info.get("individual_scores", {})
    
    # Show overall score prominently
    detail_col1, detail_col2, detail_col3 = st.columns(UI_COLUMNS_STANDARD)
    
    with detail_col1:
        st.metric("Overall Fairness", f"{feature_score:.3f}")
    
    with detail_col2:
        st.metric("Groups", feature_info.get('groups', 'N/A'))
    
    with detail_col3:
        smallest = feature_info.get('smallest_group', 'N/A')
        st.metric("Smallest Group", smallest)
        if isinstance(smallest, int) and smallest < MIN_SAMPLE_SIZE_WARNING:
            st.caption("⚠️ Small sample size")

    # Show individual fairness component scores
    if individual_scores:
        st.markdown("#### 🔍 Fairness Component Breakdown")
        
        if problem_type in ["binary_classification", "multiclass_classification", "classification"]:
            if 'demographic_parity' in individual_scores and 'equalized_odds' in individual_scores:
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    dp_score = individual_scores['demographic_parity']
                    if dp_score is not None:
                        dp_color = "🟢" if dp_score >= FAIRNESS_THRESHOLD else "🟡" if dp_score >= 0.6 else "🔴"
                        st.metric(
                            "Demographic Parity", 
                            f"{dp_score:.3f}",
                            help="Do all groups get positive predictions at similar rates?"
                        )
                        st.caption(f"{dp_color} {'Fair' if dp_score >= FAIRNESS_THRESHOLD else 'Biased'}")
                
                with comp_col2:
                    eo_score = individual_scores['equalized_odds']
                    if eo_score is not None:
                        eo_color = "🟢" if eo_score >= FAIRNESS_THRESHOLD else "🟡" if eo_score >= 0.6 else "🔴"
                        st.metric(
                            "Equalized Odds", 
                            f"{eo_score:.3f}",
                            help="Do all groups have similar error rates?"
                        )
                        st.caption(f"{eo_color} {'Fair' if eo_score >= FAIRNESS_THRESHOLD else 'Biased'}")
                
                # Explanation
                st.info(f"""
                💡 **Overall Score Explanation:** The overall fairness score ({feature_score:.3f}) is the **worst** of these two components, 
                because true fairness requires both conditions to be met.
                """)
        
        else:  # regression
            if 'error_consistency' in individual_scores:
                error_score = individual_scores['error_consistency']
                if error_score is not None:
                    error_color = "🟢" if error_score >= FAIRNESS_THRESHOLD else "🟡" if error_score >= 0.6 else "🔴"
                    st.metric(
                        "Error Consistency", 
                        f"{error_score:.3f}",
                        help="Do all groups get similarly accurate predictions?"
                    )
                    st.caption(f"{error_color} {'Fair' if error_score >= FAIRNESS_THRESHOLD else 'Biased'}")
                    
                    st.info("""
                    💡 **Score Explanation:** This measures how consistently accurate the model is across different groups. 
                    A score of 1.0 means all groups have identical prediction errors.
                    """)

        # === 9. METHODOLOGY EXPLANATION ===
        with st.expander("🔬 How We Calculated These Scores", expanded=False):
            if problem_type in ["binary_classification", "multiclass_classification", "classification"]:
                st.markdown("""
                **For Classification Models:**
                
                1. **Demographic Parity:** Do all groups get positive predictions at similar rates?
                - Perfect score (1.0) = All groups have identical positive prediction rates
                - Poor score (<0.8) = Some groups get positive predictions much more/less often
                
                2. **Equalized Odds:** Do all groups have similar error rates?
                - Perfect score (1.0) = All groups have identical true/false positive rates
                - Poor score (<0.8) = Some groups have higher error rates
                
                3. **Final Score:** We take the worst of these two scores because fairness requires both conditions.
                
                **Example:** If Group A gets loans 80% of the time and Group B gets loans 60% of the time, 
                that's a demographic parity difference of 20%, giving a score of 0.8.
                """)
            else:
                st.markdown("""
                **For Regression Models:**
                
                1. **Error Consistency:** Do all groups get similarly accurate predictions?
                - We calculate Mean Absolute Error (MAE) for each group
                - Perfect score (1.0) = All groups have identical prediction errors
                - Poor score (<0.8) = Some groups get much less accurate predictions
                
                2. **Final Score:** Ratio of best error rate to worst error rate across groups.
                
                **Example:** If Group A has average error of $1,000 and Group B has average error of $2,000,
                the fairness score would be 1,000/2,000 = 0.5 (concerning bias).
                """)

    # === 7. PERFORMANCE BY GROUP ===
    if selected_feature in fairness_results.get("metric_frames", {}):
        st.markdown("#### 📊 Model Performance by Group")
        
        # === METRIC EXPLAINER SECTION ===
        with st.expander("📚 Understanding Performance Metrics", expanded=False):
            st.markdown("### 🎯 What Each Metric Means")
            
            if problem_type in ["binary_classification", "multiclass_classification", "classification"]:
                st.markdown("""
                **Classification Metrics Explained:**
                
                🎯 **Accuracy** = (Correct Predictions) / (Total Predictions)
                - What it means: Overall percentage of correct predictions
                - Good for: Balanced datasets where all classes are equally important
                - Can be misleading when: Dataset is imbalanced (e.g., 95% negative, 5% positive cases)
                
                🎯 **Precision** = (True Positives) / (True Positives + False Positives) 
                - What it means: Of all positive predictions, how many were actually correct?
                - Good for: When false positives are costly (e.g., spam detection, fraud alerts)
                - High precision = Few false alarms
                
                🎯 **Recall (Sensitivity)** = (True Positives) / (True Positives + False Negatives)
                - What it means: Of all actual positives, how many did we correctly identify?
                - Good for: When missing positives is costly (e.g., disease diagnosis, fraud detection)
                - High recall = Few missed cases
                
                🎯 **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
                - What it means: Harmonic mean of precision and recall
                - Good for: Balancing precision and recall, especially with imbalanced data
                - High F1 = Good balance of precision and recall
                
                🎯 **AUC (Area Under Curve)** = Area under the ROC curve
                - What it means: Model's ability to distinguish between classes at all thresholds
                - Range: 0.5 (random) to 1.0 (perfect)
                - Good for: Overall model discrimination ability
                """)
                
                st.markdown("---")
                st.markdown("### 🤔 Why Metrics Can Seem Contradictory")
                
                st.warning("""
                **Common Scenario: High Accuracy, Low Recall/F1**
                
                **Example:** A loan approval model shows:
                - Group A: Accuracy = 85.9%, Recall = 29%, F1 = 43.5%
                - Group B: Accuracy = 87.2%, Recall = 78%, F1 = 82.1%
                
                **What's happening?** 
                
                🔍 **The Hidden Story:**
                - Group A might have very few people who actually qualify for loans (class imbalance)
                - If only 10% of Group A applicants should get loans, the model can achieve 85.9% accuracy by being very conservative
                - But this conservativeness means it only identifies 29% of qualified applicants (low recall)
                - The F1 score (43.5%) reveals this isn't actually good performance
                
                **Real-world impact:**
                - Group A: 71% of qualified applicants get wrongly rejected
                - Group B: Only 22% of qualified applicants get wrongly rejected
                - This suggests potential bias against Group A
                """)
                
                st.info("""
                **Key Insight:** High accuracy can hide bias!
                
                When groups have different base rates (% of positive cases), accuracy alone can be misleading:
                - A conservative model gets high accuracy on low-positive-rate groups
                - The same model might have lower accuracy but better recall on high-positive-rate groups
                - Always look at precision, recall, and F1 together for the full picture
                """)
                
                st.markdown("### 📊 How to Interpret Cross-Group Differences")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success("""
                    **✅ Good Fairness Indicators:**
                    - Similar F1 scores across groups
                    - Similar recall rates (important for opportunity)
                    - Precision differences ≤ 10%
                    - AUC scores within 5% of each other
                    """)
                
                with col2:
                    st.error("""
                    **⚠️ Potential Bias Indicators:**
                    - Large recall differences (>20%)
                    - F1 scores varying significantly
                    - One group has high accuracy but low recall
                    - AUC differences > 10%
                    """)
                
                st.markdown("""
                **🎯 Practical Guidelines:**
                
                1. **For High-Stakes Decisions (loans, hiring, medical):**
                   - Focus on recall and F1 score differences
                   - Accuracy alone is insufficient
                   - Ensure no group has systematically lower recall
                
                2. **For Screening Applications (fraud, spam):**
                   - Precision is often more important
                   - But still check for fairness in recall
                   - False positive rates should be similar across groups
                
                3. **Red Flags to Watch:**
                   - Group A: High accuracy, low recall → Model is too conservative for this group
                   - Group B: Lower accuracy, high recall → Model might be too liberal for this group
                   - Large F1 differences → Unequal treatment quality
                """)
                
            else:  # regression
                st.markdown("""
                **Regression Metrics Explained:**
                
                🎯 **R² (R-squared)** = 1 - (Sum of Squared Residuals) / (Total Sum of Squares)
                - What it means: Percentage of variance in target explained by the model
                - Range: 0% to 100% (higher is better)
                - Good for: Understanding overall model fit
                
                🎯 **MSE (Mean Squared Error)** = Average of (Actual - Predicted)²
                - What it means: Average squared difference between actual and predicted values
                - Units: Squared units of your target variable
                - Good for: Penalizing large errors more heavily
                
                🎯 **RMSE (Root Mean Squared Error)** = √(MSE)
                - What it means: Average error in the same units as your target
                - Units: Same as your target variable
                - Good for: Interpretable error magnitude
                
                🎯 **MAE (Mean Absolute Error)** = Average of |Actual - Predicted|
                - What it means: Average absolute difference between actual and predicted
                - Units: Same as your target variable  
                - Good for: Understanding typical error size (less sensitive to outliers)
                """)
                
                st.markdown("### 🤔 Why Regression Metrics Can Vary Across Groups")
                
                st.warning("""
                **Common Scenario: Different Error Patterns**
                
                **Example:** A salary prediction model shows:
                - Group A: R² = 0.85, RMSE = $8,000, MAE = $5,000
                - Group B: R² = 0.72, RMSE = $15,000, MAE = $12,000
                
                **What's happening?**
                - The model explains variance much better for Group A (85% vs 72%)
                - Group B has much larger prediction errors on average
                - This suggests the model learned patterns that work well for Group A but not Group B
                
                **Potential causes:**
                - Training data had more examples from Group A
                - Different salary structures or career patterns between groups
                - Model features work better for predicting Group A salaries
                """)
                
                st.info("""
                **Key Insight:** Similar R² doesn't guarantee fair errors!
                
                Even if R² scores are similar, check:
                - Are RMSE/MAE values similar? (error magnitude)
                - Are errors consistently over/under-predicting for any group?
                - Do some groups have more variable errors than others?
                """)
            
            st.markdown("---")
            st.markdown("### 🎯 Action Items Based on Metric Patterns")
            
            if problem_type == "classification":
                st.markdown("""
                **If you see high accuracy but low recall/F1 for a group:**
                1. Check class balance - is this group's positive rate much lower?
                2. Consider adjusting decision threshold for this group
                3. Collect more positive examples for this group
                4. Use fairness-aware training methods
                
                **If you see consistent metric differences across groups:**
                1. Investigate root causes (data quality, representation, feature relevance)
                2. Consider group-specific models or post-processing adjustments
                3. Document and monitor these differences
                4. Consult with domain experts about acceptable trade-offs
                """)
            else:
                st.markdown("""
                **If you see large RMSE/MAE differences:**
                1. Check if groups have different target distributions
                2. Consider group-specific feature engineering
                3. Investigate if model assumptions hold for all groups
                4. Consider ensemble methods or group-specific models
                
                **If you see systematic over/under-prediction:**
                1. Check for bias in training data representation
                2. Consider calibration techniques
                3. Investigate feature interactions that might favor certain groups
                4. Add group-aware regularization during training
                """)
        
        metric_frame = fairness_results["metric_frames"][selected_feature]
        if 'performance' in metric_frame:
            perf_data = metric_frame['performance'].by_group
            
            # Show key metric chart
            if problem_type in ["binary_classification", "multiclass_classification", "classification"] and 'accuracy' in perf_data.columns:
                fig = px.bar(
                    x=perf_data.index, 
                    y=perf_data['accuracy'],
                    title=f"Accuracy by {selected_feature}",
                    labels={'x': selected_feature, 'y': 'Accuracy'},
                    color=perf_data['accuracy'],
                    color_continuous_scale='RdYlGn'
                )
                fig.add_hline(y=perf_data['accuracy'].mean(), line_dash="dash", annotation_text="Average")
                st.plotly_chart(fig, config={'responsive': True})
            
            elif problem_type == "regression":
                # Choose the best available metric for visualization
                available_metrics = perf_data.columns
                
                # Priority order for regression metrics (most interpretable first)
                metric_priority = ['r2', 'mae', 'rmse', 'mse']
                selected_metric = None
                
                for metric in metric_priority:
                    if metric in available_metrics:
                        selected_metric = metric
                        break
                
                if selected_metric:
                    # Create appropriate chart based on metric type
                    metric_display_names = {
                        'r2': 'R² Score',
                        'mae': 'Mean Absolute Error',
                        'rmse': 'Root Mean Squared Error', 
                        'mse': 'Mean Squared Error'
                    }
                    
                    metric_name = metric_display_names.get(selected_metric, selected_metric.upper())
                    
                    # For R², higher is better (use green color scale)
                    # For error metrics, lower is better (use reversed color scale)
                    if selected_metric == 'r2':
                        color_scale = 'RdYlGn'  # Red to Green
                        color_values = perf_data[selected_metric]
                    else:
                        color_scale = 'RdYlGn_r'  # Green to Red (reversed)
                        color_values = perf_data[selected_metric]
                    
                    fig = px.bar(
                        x=perf_data.index,
                        y=perf_data[selected_metric],
                        title=f"{metric_name} by {selected_feature}",
                        labels={'x': selected_feature, 'y': metric_name},
                        color=color_values,
                        color_continuous_scale=color_scale
                    )
                    
                    # Add average line
                    fig.add_hline(
                        y=perf_data[selected_metric].mean(), 
                        line_dash="dash", 
                        annotation_text="Average"
                    )
                    
                    st.plotly_chart(fig, config={'responsive': True})
                    
                    # Add interpretation helper
                    if selected_metric == 'r2':
                        st.caption("📊 Higher R² scores indicate better model fit for that group")
                    else:
                        st.caption("📊 Lower error values indicate better model performance for that group")
                
                # If we have multiple metrics, offer to show additional charts
                if len([m for m in metric_priority if m in available_metrics]) > 1:
                    with st.expander("📈 Show Additional Metric Charts", expanded=False):
                        additional_metrics = [m for m in metric_priority if m in available_metrics and m != selected_metric]
                        
                        for metric in additional_metrics:
                            metric_name = metric_display_names.get(metric, metric.upper())
                            
                            if metric == 'r2':
                                color_scale = 'RdYlGn'
                            else:
                                color_scale = 'RdYlGn_r'
                            
                            fig = px.bar(
                                x=perf_data.index,
                                y=perf_data[metric],
                                title=f"{metric_name} by {selected_feature}",
                                labels={'x': selected_feature, 'y': metric_name},
                                color=perf_data[metric],
                                color_continuous_scale=color_scale
                            )
                            
                            fig.add_hline(
                                y=perf_data[metric].mean(),
                                line_dash="dash",
                                annotation_text="Average"
                            )
                            
                            st.plotly_chart(fig, config={'responsive': True})
            
            # Performance table (for both classification and regression)
            display_perf = perf_data.copy()
            if 'count' in display_perf.columns:
                display_perf = display_perf.drop('count', axis=1)
            st.dataframe(display_perf.style.format("{:.3f}"), width='stretch')
            
            # Show group distribution (for both classification and regression)
            if 'count' in perf_data.columns:
                st.markdown("**Sample Distribution:**")
                total_samples = perf_data['count'].sum()
                dist_data = []
                for group, count in perf_data['count'].items():
                    percentage = (count / total_samples) * 100
                    dist_data.append({
                        "Group": str(group),
                        "Count": f"{count:,}",
                        "Percentage": f"{percentage:.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(dist_data), width='stretch', hide_index=True)

    st.markdown("---")

    # === 8. ACTIONABLE RECOMMENDATIONS ===
    st.markdown("### 💡 What You Should Do")
    
    recs = generate_specific_recommendations(fairness_results, st.session_state.builder.X_test, problem_type)
    
    if recs:
        # Group by priority and show clearly
        critical = [r for r in recs if r.get("priority") == "Critical"]
        high = [r for r in recs if r.get("priority") == "High"]
        medium = [r for r in recs if r.get("priority") == "Medium"]
        
        if critical:
            st.error("🚨 **Critical Actions Needed:**")
            for i, rec in enumerate(critical, 1):
                st.markdown(f"{i}. **{rec['action']}** (Feature: {rec['feature']})")
        
        if high:
            st.warning("⚠️ **High Priority Actions:**")
            for i, rec in enumerate(high, 1):
                st.markdown(f"{i}. **{rec['action']}** (Feature: {rec['feature']})")
        
        if medium:
            st.info("ℹ️ **Consider These Improvements:**")
            for i, rec in enumerate(medium, 1):
                st.markdown(f"{i}. **{rec['action']}** (Feature: {rec['feature']})")
    else:
        st.success("🎉 No specific actions needed - your model shows good fairness!")
        
        # Show best practices even for fair models
        st.markdown("""
        #### 📚 Best Practices for Maintaining Fairness
        
        - **Regular Monitoring:** Check fairness metrics monthly with new data
        - **Documentation:** Keep records of fairness assessments and decisions  
        - **Stakeholder Review:** Have domain experts validate your fairness conclusions
        - **Continuous Learning:** Stay updated on fairness research and regulations
        """)

# === MAIN RENDER FUNCTION ===

def render_fairness_analysis():
    """Main function to render the fairness analysis interface."""
    
    st.markdown("# 🎯 Model Fairness Analysis")
    
    # Get problem type to customize introduction
    if hasattr(st.session_state, 'problem_type'):
        problem_type = st.session_state.problem_type
    else:
        problem_type = st.session_state.builder.model.get("problem_type", "unknown")
    
    # Introduction with problem-type specific content
    if problem_type == "multiclass_classification":
        st.markdown("""
            ### 🤔 What is Model Fairness for Multiclass Classification?
            
            Model fairness ensures your model treats all groups fairly across **all classes** without discrimination. 
            For multiclass models, fairness is more complex because we need to ensure:
            
            - **Equal representation**: All groups have similar prediction distributions across all classes
            - **Consistent accuracy**: All groups receive predictions of similar quality
            - **No systematic bias**: No group is consistently favored or disadvantaged for any particular class
            
            **Example**: In a medical diagnosis model predicting 3 conditions (Healthy, Condition A, Condition B), 
            fairness means that patients from different demographic groups have:
            - Similar chances of being diagnosed with each condition (when appropriate)
            - Similar diagnostic accuracy across all conditions
            
            This analysis helps you identify potential bias and provides actionable recommendations.
        """)
    elif problem_type == "binary_classification":
        st.markdown("""
            ### 🤔 What is Model Fairness for Binary Classification?
            
            Model fairness ensures your model treats all groups fairly without discrimination. 
            For binary classification, this means ensuring:
            
            - **Equal opportunity**: Qualified individuals from all groups have similar chances of positive outcomes
            - **Equal treatment**: Similar false positive and false negative rates across groups
            - **Demographic parity**: Overall positive prediction rates are similar across groups
            
            This analysis helps you identify potential bias and provides actionable recommendations.
        """)
    else:
        st.markdown("""
            ### 🤔 What is Model Fairness?
            Model fairness ensures your model treats all groups fairly without discrimination. 
            This analysis helps you identify potential bias and provides actionable recommendations.
        """)
    
    # Get data and validate
    if not hasattr(st.session_state.builder, 'X_test') or st.session_state.builder.X_test is None:
        st.error("❌ **No Test Data Available for Fairness Analysis**\n\n"
                 "**What This Means:** The fairness analysis requires test data (X_test) to evaluate "
                 "how your model performs across different demographic groups.\n\n"
                 "**To Fix This Issue:**\n"
                 "1. Complete the data preprocessing step to generate train/test splits\n"
                 "2. Train a model in the Model Training section\n"
                 "3. Ensure your model has been evaluated on test data\n\n"
                 "**Next Steps:** Go back to the Data Preprocessing page to split your data, "
                 "then proceed through Model Selection and Training before returning here.")
        return
    
    # Process columns
    all_available_columns = st.session_state.builder.X_test.columns.tolist()
    protected_cols = st.session_state.builder.get_protected_attributes()
    
    valid_columns = []
    filtered_columns = []
    for col in all_available_columns:
        if not is_metric_column(col):
            valid_columns.append(col)
        else:
            filtered_columns.append(col)
    
    # === CLEAN FEATURE SELECTION INTERFACE ===
    columns_to_analyse = create_clean_feature_selection_interface(
        valid_columns, protected_cols, filtered_columns
    )
    
    if not columns_to_analyse:
        st.info("👆 Please select features to analyze above.")
        st.stop()
    
    # Add educational content about fairness metrics
    with st.expander("📚 Understanding Fairness Metrics for Your Model Type", expanded=False):
        if problem_type == "multiclass_classification":
            st.markdown("""
            ## Fairness Metrics for Multiclass Classification
            
            Since your model predicts multiple classes, we use specialized fairness measures:
            
            ### 🎯 **Demographic Parity**
            - **What it measures**: Whether all groups have similar prediction distributions across all classes
            - **Why it matters**: Ensures no group is systematically over- or under-represented in any class
            - **Example**: In a hiring model with outcomes (Reject, Interview, Hire), demographic parity ensures that the percentage of each outcome is similar across demographic groups
            
            ### 📊 **Accuracy Consistency** 
            - **What it measures**: Whether all groups receive predictions of similar quality
            - **Why it matters**: Ensures no group gets systematically worse predictions
            - **Calculation**: Ratio of minimum group accuracy to maximum group accuracy
            - **Example**: If Group A has 85% accuracy and Group B has 90% accuracy, consistency = 0.94
            
            ### 🎯 **Overall Fairness Score**
            - Takes the minimum of Demographic Parity and Accuracy Consistency
            - **Score ≥ 0.8**: Generally considered fair
            - **Score < 0.8**: May indicate bias requiring attention
            
            ### ⚠️ **Why we don't use traditional binary fairness metrics**:
            - **False Positive/Negative Rates**: Don't directly apply to multiclass (which "positive" class?)
            - **Equalized Odds**: Designed for binary outcomes
            - **ROC-AUC**: Requires probability scores, complex for multiclass
            """)
        elif problem_type == "binary_classification":
            st.markdown("""
            ## Fairness Metrics for Binary Classification
            
            For binary classification, we use well-established fairness measures:
            
            ### 🎯 **Demographic Parity**
            - **What it measures**: Whether all groups have similar positive prediction rates
            - **Why it matters**: Ensures equal representation in positive outcomes
            
            ### ⚖️ **Equalized Odds**
            - **What it measures**: Whether true positive and false positive rates are similar across groups
            - **Why it matters**: Ensures equal treatment quality across groups
            
            ### 🎯 **Overall Fairness Score**
            - Takes the minimum of Demographic Parity and Equalized Odds scores
            - **Score ≥ 0.8**: Generally considered fair
            - **Score < 0.8**: May indicate bias requiring attention
            """)
        else:
            st.markdown("""
            ## Fairness Metrics for Regression
            
            For regression models, fairness focuses on prediction quality:
            
            ### 📊 **Error Consistency**
            - **What it measures**: Whether prediction errors are similar across groups
            - **Why it matters**: Ensures no group receives systematically worse predictions
            """)
        
        st.markdown("""
        ---
        ### 💡 **General Interpretation Guidelines**
        
        - **Scores near 1.0**: Excellent fairness
        - **Scores 0.8-0.9**: Good fairness, acceptable for most applications
        - **Scores 0.6-0.8**: Moderate bias, should be investigated
        - **Scores below 0.6**: Significant bias, requires immediate attention
        
        Remember: Perfect fairness (score = 1.0) is rare in real-world data. The key is understanding 
        whether observed differences are acceptable for your specific use case and legal/ethical requirements.
        """)
    
    # Get problem type from session state if available
    if hasattr(st.session_state, 'problem_type'):
        problem_type = st.session_state.problem_type
    else:
        # Fallback to model's problem type
        problem_type = st.session_state.builder.model.get("problem_type", "unknown")
    
    # === RUN ANALYSIS ===
    st.markdown("---")
    
    with st.spinner("🔍 Analyzing fairness across selected features..."):
        # Get predictions in the right format
        model = st.session_state.builder.model["model"]
        X_test = st.session_state.builder.X_test
        predictions = model.predict(X_test)
        

        
        fairness_results = analyse_model_fairness(
            X_test, 
            st.session_state.builder.y_test, 
            predictions, 
            columns_to_analyse, 
            problem_type
        )
    
    if not fairness_results["success"]:
        st.error(f"❌ **Fairness Analysis Failed**\n\n"
                 f"**Error Details:** {fairness_results['message']}\n\n"
                 f"**What You Can Try:**\n"
                 f"1. Select different features with more data variation\n"
                 f"2. Check that your selected features have multiple distinct groups\n"
                 f"3. Ensure your model predictions are in the correct format\n"
                 f"4. Verify that test data contains the selected features\n\n"
                 f"**Need Help?** Use the Custom Selection method to choose features with "
                 f"fewer unique values and larger group sizes.")
        return
    
    # === COMPREHENSIVE DASHBOARD ===
    create_comprehensive_fairness_dashboard(
        fairness_results, problem_type, columns_to_analyse
    )
    
    # Log the analysis
    if fairness_results.get("column_scores"):
        score_data = []
        for feature, score in fairness_results["column_scores"].items():
            if score is not None:
                score_data.append({
                    "Feature": feature,
                    "Score": score,
                    "Status": "Fair" if score >= FAIRNESS_THRESHOLD else "Bias Risk"
                })
        
        if hasattr(st.session_state, 'logger'):
            st.session_state.logger.log_journey_point(
                stage="MODEL_EXPLANATION",
                decision_type="MODEL_EXPLANATION",
                description="Fairness Analysis Completed",
                details={"Fairness Results": score_data},
                parent_id=None
            ) 