import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
from scipy import stats

from components.data_exploration.feature_analysis import get_visualisation_info
from utils.dataframe_comparison import DataframeComparisonComponent
from utils.data_exploration_component import DataExplorationComponent


@st.cache_data(show_spinner=False)
def get_cached_binning_suggestions(data_hash: str, target_column: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Get cached binning suggestions for the given data.
    
    Args:
        data_hash: Hash of the dataframe to ensure cache invalidation when data changes
        target_column: Name of the target column
        data: The dataframe to analyse
        
    Returns:
        Dict containing binning suggestions
    """
    try:
        # Skip processing if target column is not in the dataset
        if target_column not in data.columns:
            return {"success": False, "message": "Target column not found in dataset"}
            
        suggestions = {}
        
        # Handle numeric columns
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Handle categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column from both lists if present
        for col_list in [numeric_cols, categorical_cols]:
            if target_column in col_list:
                col_list.remove(target_column)
            
        # Identify binary features more efficiently (both numeric and categorical)
        nunique_series = data.nunique()
        binary_features = nunique_series[nunique_series == 2].index.tolist()
        
        # Remove binary features from numeric and categorical columns
        numeric_cols = [col for col in numeric_cols if col not in binary_features]
        categorical_cols = [col for col in categorical_cols if col not in binary_features]
        
        # Add binary features to suggestions with "None" strategy
        for col in binary_features:
            suggestions[col] = {
                "strategy": "None",
                "reason": "This is a binary feature and does not need binning.",
                "needs_binning": False
            }
            
        # Use session state to determine problem type if available, otherwise fall back to heuristics
        import streamlit as st
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'problem_type'):
            problem_type = st.session_state.problem_type
            if problem_type == "binary_classification":
                target_type = "binary"
            elif problem_type == "multiclass_classification":
                target_type = "multiclass"
            else:  # regression
                target_type = "continuous"
        else:
            # Fallback to original heuristic
            target_type = "continuous" if pd.api.types.is_numeric_dtype(data[target_column]) else "binary"
        
        # Precompute target values once to avoid repetition
        target_values = data[target_column]
        if not pd.api.types.is_numeric_dtype(target_values):
            target_values = pd.Categorical(target_values).codes

        def check_non_linear_relationship(feature_values, target_values):
            """
            Check if there's a non-linear relationship between feature and target.
            Returns a tuple of (is_non_linear, explanation)
            """
            try:
                # Remove NaN values
                mask = ~(np.isnan(feature_values) | np.isnan(target_values))
                feature_values = feature_values[mask]
                target_values = target_values[mask]
                
                if len(feature_values) < 10:  # Need sufficient data points
                    return False, "Insufficient data points for non-linearity test"
                
                # Calculate linear correlation
                linear_corr = stats.pearsonr(feature_values, target_values)[0]
                
                # Calculate rank correlation (captures monotonic relationships)
                rank_corr = stats.spearmanr(feature_values, target_values)[0]
                
                # If rank correlation is significantly stronger than linear correlation,
                # this suggests a non-linear relationship
                if abs(rank_corr) > abs(linear_corr) + 0.1:  # Threshold of 0.1 difference
                    return True, f"Non-linear relationship detected (Pearson={linear_corr:.2f}, Spearman={rank_corr:.2f})"
                
                # Check for U-shaped relationship using quadratic regression
                x = feature_values.values.reshape(-1, 1)
                y = target_values
                
                # Fit linear model
                from sklearn.linear_model import LinearRegression
                linear_model = LinearRegression()
                linear_score = linear_model.fit(x, y).score(x, y)
                
                # Fit quadratic model
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=2)
                x_poly = poly.fit_transform(x)
                quad_model = LinearRegression()
                quad_score = quad_model.fit(x_poly, y).score(x_poly, y)
                
                # If quadratic model performs significantly better
                if quad_score > linear_score + 0.1:  # 10% improvement threshold
                    return True, f"U-shaped relationship detected (Linear R²={linear_score:.2f}, Quadratic R²={quad_score:.2f})"
                
                return False, "Linear relationship detected"
                
            except Exception as e:
                return False, f"Error in non-linearity test: {str(e)}"

        # Process numeric columns
        for column in numeric_cols:
            try:
                # Analyse distribution
                unique_count = data[column].nunique()
                skewness = data[column].skew()
                n_samples = len(data)
                
                # Check for non-linear relationship with target
                is_non_linear, non_linear_explanation = check_non_linear_relationship(
                    data[column].values,
                    target_values
                )
                
                # Check for skewed numerical features with low cardinality
                if abs(skewness) > 2 and unique_count < 10:
                    suggestions[column] = {
                        "strategy": "Optimal",
                        "reason": f"""This feature is highly skewed (skewness = {skewness:.2f}) and has low cardinality ({unique_count} unique values).
                        We recommend optimal binning because:
                        - The skewness suggests non-linear patterns
                        - The low number of unique values makes it more suitable as a categorical feature
                        - Binning will help capture value groups that are meaningful for prediction""",
                        "needs_binning": True,
                        "convert_to_categorical": True
                    }
                    continue
                
                # If non-linear relationship is detected, suggest binning regardless of skewness
                if is_non_linear:
                    suggestions[column] = {
                        "strategy": "Optimal",
                        "reason": f"""This feature shows a non-linear relationship with the target variable.
                        {non_linear_explanation}
                        We recommend optimal binning because:
                        - It will help capture the non-linear patterns
                        - Binning can improve model performance with non-linear relationships
                        - It will create meaningful value groups based on the target relationship""",
                        "needs_binning": True,
                        "n_bins": min(20, max(3, unique_count // 5))
                    }
                    continue
                
                # Only suggest binning for highly skewed features
                if abs(skewness) <= 2:
                    suggestions[column] = {
                        "strategy": "None",
                        "reason": f"""This feature has a relatively normal distribution (skewness = {skewness:.2f}). 
                        Binning is most useful for features that are highly skewed or have outliers. 
                        This feature looks good as it is!""",
                        "needs_binning": False
                    }
                    continue
                
                if unique_count < 10:
                    suggestions[column] = {
                        "strategy": "None",
                        "reason": f"""This feature only has {unique_count} unique values, which is too few for meaningful binning. 
                        Binning works best with continuous variables that have many different values.""",
                        "needs_binning": False
                    }
                else:
                    # Try optimal binning for numeric columns
                    min_n_bins = 3
                    max_n_bins = max(min_n_bins, min(20, unique_count // 5))
                    min_bin_size = max(0.05, 1/np.sqrt(n_samples))
                    
                    suggestions[column] = {
                        "strategy": "Optimal",
                        "reason": f"""This feature is highly skewed (skewness = {skewness:.2f}). 
                        Optimal binning will help:
                        - Make the distribution more balanced
                        - Capture non-linear patterns with the target
                        - Handle outliers naturally
                        - Create statistically significant bins
                        - Maximize predictive power""",
                        "needs_binning": True,
                        "n_bins": max_n_bins
                    }
            except Exception as e:
                suggestions[column] = {
                    "strategy": "None",
                    "reason": f"Error analyzing column: {str(e)}",
                    "needs_binning": False
                }
        
        # Process categorical columns
        for column in categorical_cols:
            try:
                unique_count = data[column].nunique()
                n_samples = len(data)
                
                if unique_count <= 5:
                    suggestions[column] = {
                        "strategy": "None",
                        "reason": f"This categorical feature only has {unique_count} unique values, which is few enough to use directly.",
                        "needs_binning": False
                    }
                    continue
                
                # For high cardinality categorical variables
                min_n_bins = 3
                max_n_bins = max(min_n_bins, min(10, unique_count // 3))
                
                suggestions[column] = {
                    "strategy": "Optimal",
                    "reason": f"""This categorical feature has {unique_count} unique values, which is relatively high.
                    Optimal binning will:
                    - Group similar categories based on target relationship
                    - Create statistically significant bins
                    - Handle rare categories effectively
                    - Maximize predictive power""",
                    "needs_binning": True,
                    "n_bins": max_n_bins
                }
            except Exception as e:
                suggestions[column] = {
                    "strategy": "None",
                    "reason": f"Error analyzing column: {str(e)}",
                    "needs_binning": False
                }

        return {"success": True, "suggestions": suggestions}
    except Exception as e:
        return {"success": False, "message": f"Error suggesting binning strategies: {str(e)}"}

class FeatureBinningComponent:
    """
    Component for feature binning, which helps handle both numerical and categorical features 
    by grouping them into fewer categories.
    """
    
    def __init__(self, builder, logger):
        """
        Initialize the Feature Binning component.
        
        Args:
            builder: The ML Builder instance containing the data
            logger: Logger instance for tracking user actions and calculations
        """
        self.builder = builder
        self.logger = logger
    
        # Initialize undo functionality with single backup (memory optimized)
        if "feature_binning_ops_applied" not in st.session_state:
            st.session_state.feature_binning_ops_applied = []
            
        # Store initial state for undo functionality (single backup for both datasets)
        if "feature_binning_entry_data" not in st.session_state:
            st.session_state.feature_binning_entry_data = {
                'training_data': self.builder.training_data.copy(),
                'testing_data': self.builder.testing_data.copy()
            }

    def render(self):
        """Render the feature binning interface"""
        st.write("---")
        st.write("Using the data exploration component may cause the page to reload, any changes that you have applied will still be in effect. you can use the undo button to reset the data to it's original state when you first entered the page")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            @st.dialog(title="Data Exploration", width="large")
            def data_explorer_dialog():
                data_explorer = DataExplorationComponent(self.builder, st.session_state.logger, data=st.session_state.builder.training_data, target_column=st.session_state.builder.target_column)
                data_explorer.render()

            if st.button("Training Data Exploration",on_click=st.rerun):
                data_explorer_dialog()
        with col2:
            st.write("")
        with col3:
            if st.button("Undo Feature Binning", type="primary", width='stretch'):
                if st.session_state.feature_binning_ops_applied:
                    # Restore data to entry state
                    entry_data = st.session_state.feature_binning_entry_data
                    self.builder.training_data = entry_data['training_data'].copy()
                    self.builder.testing_data = entry_data['testing_data'].copy()
                    
                    # Clear operations tracking
                    ops_count = len(st.session_state.feature_binning_ops_applied)
                    st.session_state.feature_binning_ops_applied = []
                    
                    st.success(f"✅ Undid {ops_count} feature binning operation(s). Training and testing data restored to entry state.")
                    st.rerun()
                else:
                    st.info("No feature binning operations to undo.")

        st.write("### Feature Binning")
        st.write("""
            Binning helps handle both numerical and categorical features by grouping them into fewer categories. 
            This is particularly useful when:
            - A numerical feature has a very uneven distribution (highly skewed)
            - A categorical feature has too many unique values (high cardinality)
            - There are outliers that might affect the model
            - You want to capture non-linear relationships
            - You want to make the model more robust
            
            We'll analyse your features and suggest binning for those that would benefit from it.
        """)

        self._render_understanding_binning_expander()
        self._process_features_for_binning()
    
    def _render_understanding_binning_expander(self):
        """Render the expander with detailed explanation about binning strategies"""
        with st.expander("📚 Understanding Binning Strategies"):
            st.write("""
                ### Feature Binning Strategies
                
                Feature binning, also known as discretization, is a technique that transforms continuous or high-cardinality categorical variables into a smaller set of discrete bins. Here's a detailed look at the available strategy:
                
                #### Optimal Binning
                - **What it does:** Creates bins based on the relationship between the feature and target variable, optimizing for predictive power
                - **How it works:**
                    - Analyses the relationship between each feature and the target variable
                    - Automatically determines the optimal number of bins
                    - Creates cut points that maximize the predictive power
                    - Handles both numerical and categorical variables
                - **Best when:**
                    - You have a clear target variable
                    - The relationship between feature and target is non-linear
                    - You want to maximize predictive power
                    - You have high-cardinality categorical variables
                - **Advantages:**
                    - Maximizes information value and predictive power
                    - Creates meaningful splits based on target relationship
                    - Automatically determines optimal number of bins
                    - Handles both numerical and categorical data
                    - Reduces overfitting through smart bin boundary selection
                - **Key Features:**
                    - Adaptive bin size based on data distribution
                    - Handles outliers intelligently
                    - Preserves important patterns in the data
                    - Maintains statistical significance between bins
                
                ### Impact of Binning
                
                1. **Data Distribution**
                   - Reduces impact of outliers
                   - Makes distributions more balanced
                   - Reveals non-linear patterns
                   - Preserves important relationships with target
                
                2. **Model Performance**
                   - Improves model stability
                   - Captures non-linear relationships effectively
                   - Reduces noise in the data
                   - Enhances feature predictive power
                
                3. **Memory and Computation**
                   - Reduces memory usage for high-cardinality features
                   - Speeds up model training
                   - Makes feature engineering more efficient
                
                ### Best Practices
                
                1. **When to Apply Binning**
                   - For numerical features with non-linear relationships
                   - For categorical features with high cardinality
                   - When feature distributions are highly skewed
                   - When you want to reduce the impact of outliers
                
                2. **Validation**
                   - Review the generated bin boundaries
                   - Check sample distribution across bins
                   - Verify relationship with target variable
                   - Monitor impact on model performance
                
                3. **Documentation**
                   - Keep track of binning decisions
                   - Document bin boundaries and their meaning
                   - Record the impact on feature distributions
                   - Note any special cases or exceptions
            """)
    
    def _process_features_for_binning(self):
        """Process features that may need binning and provide interface for configuration"""
        original_data = self.builder.training_data.copy()
        
        # Get numeric and categorical columns excluding target
        numeric_cols = self.builder.training_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        categorical_cols = self.builder.training_data.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        # Remove target column if present
        if self.builder.target_column in numeric_cols:
            numeric_cols.remove(self.builder.target_column)
        if self.builder.target_column in categorical_cols:
            categorical_cols.remove(self.builder.target_column)
        
        # Remove already binned columns
        numeric_cols = [col for col in numeric_cols if not col.endswith('_binned')]
        categorical_cols = [col for col in categorical_cols if not col.endswith('_binned')]

        if numeric_cols or categorical_cols:
            # Get binning suggestions using cached function
            data_hash = pd.util.hash_pandas_object(self.builder.training_data).sum()
            suggestions = get_cached_binning_suggestions(
                data_hash=str(data_hash),
                target_column=self.builder.target_column,
                data=self.builder.training_data
            )
            
            if suggestions["success"]:
                # Log binning suggestions
                self.logger.log_calculation(
                    "Binning Analysis",
                    {
                        "total_features_analysed": len(numeric_cols) + len(categorical_cols),
                        "numeric_features": len(numeric_cols),
                        "categorical_features": len(categorical_cols),
                        "features_needing_binning": len([col for col in suggestions["suggestions"] if suggestions["suggestions"][col].get("needs_binning", False)]),
                        "suggestions": {col: {
                            "needs_binning": suggestions["suggestions"][col].get("needs_binning", False),
                            "strategy": suggestions["suggestions"][col].get("strategy", "None"),
                            "reason": suggestions["suggestions"][col].get("reason", "")
                        } for col in suggestions["suggestions"]}
                    }
                )
                
                # First show summary of which features need binning
                features_to_bin = [col for col in numeric_cols + categorical_cols 
                                 if suggestions["suggestions"][col].get("needs_binning", False)]
                if features_to_bin:
                    st.write(f"**{len(features_to_bin)} features** would benefit from binning:")
                    st.write(", ".join(features_to_bin))
                    
                    # Add handling controls only for features that need binning
                    handling_dict = {}
                    for col in features_to_bin:
                        self._render_feature_binning_configuration(col, suggestions, handling_dict, categorical_cols)
                else:
                    # Initialize empty handling dict even when no features are recommended
                    handling_dict = {}
                    st.info("No features were found that would benefit from binning.")
                
                # Add manual feature selection section
                st.write("---")
                st.write("### 🎯 Manual Feature Selection")
                st.write("""
                    You can also manually select additional features to bin, even if they weren't recommended. 
                    This gives you full control over your feature engineering process.
                """)
                
                # Get all available features (excluding already binned ones and target)
                available_features = numeric_cols + categorical_cols
                # Remove features that were already recommended for binning
                features_already_recommended = [col for col in available_features 
                                              if suggestions["suggestions"][col].get("needs_binning", False)]
                available_for_manual = [col for col in available_features 
                                      if col not in features_already_recommended]
                
                # Apply safety rules to filter out inappropriate features
                # 1. Remove binary features (2 unique values)
                nunique_series = self.builder.training_data.nunique()
                binary_features = nunique_series[nunique_series == 2].index.tolist()
                available_for_manual = [col for col in available_for_manual if col not in binary_features]
                
                # 2. Remove target column (double-check)
                if self.builder.target_column in available_for_manual:
                    available_for_manual.remove(self.builder.target_column)
                
                # 3. Remove already binned features (double-check)
                available_for_manual = [col for col in available_for_manual if not col.endswith('_binned')]
                
                # 4. Remove low cardinality features
                low_cardinality_features = []
                features_to_remove = []
                for col in available_for_manual:
                    unique_count = self.builder.training_data[col].nunique()
                    is_categorical = col in categorical_cols
                    
                    # Apply same cardinality rules as recommendations
                    if is_categorical and unique_count <= 5:
                        low_cardinality_features.append(f"{col} ({unique_count} categories)")
                        features_to_remove.append(col)
                    elif not is_categorical and unique_count < 10:
                        low_cardinality_features.append(f"{col} ({unique_count} values)")
                        features_to_remove.append(col)
                
                available_for_manual = [col for col in available_for_manual if col not in features_to_remove]
                
                if available_for_manual:
                    # Let user select additional features
                    manual_features = st.multiselect(
                        "Select additional features to bin:",
                        options=available_for_manual,
                        help="Choose any viable feature to apply binning to, regardless of the recommendations."
                    )
                    
                    if manual_features:
                        st.write(f"**Configuring binning for {len(manual_features)} manually selected features:**")
                        
                        # Add manual selections to the same handling dict
                        for col in manual_features:
                            self._render_manual_feature_binning_configuration(
                                col, handling_dict, categorical_cols
                            )
                else:
                    st.info("All available features have already been recommended for binning or are not suitable for binning.")
                    
                    # Show what was filtered out and why
                    filtered_out_reasons = []
                    if binary_features:
                        filtered_out_reasons.append(f"**Binary features** ({len(binary_features)}): {', '.join(binary_features[:3])}{'...' if len(binary_features) > 3 else ''}")
                    
                    # Check for features that were already recommended
                    if features_already_recommended:
                        filtered_out_reasons.append(f"**Already recommended** ({len(features_already_recommended)}): {', '.join(features_already_recommended[:3])}{'...' if len(features_already_recommended) > 3 else ''}")
                    
                    # Check for already binned features
                    already_binned = [col for col in self.builder.training_data.columns if col.endswith('_binned')]
                    if already_binned:
                        filtered_out_reasons.append(f"**Already binned** ({len(already_binned)}): {', '.join(already_binned[:3])}{'...' if len(already_binned) > 3 else ''}")
                    
                    # Show low cardinality features that were filtered out
                    if low_cardinality_features:
                        filtered_out_reasons.append(f"**Low cardinality features** ({len(low_cardinality_features)}): {', '.join([f.split(' (')[0] for f in low_cardinality_features[:3]])}{'...' if len(low_cardinality_features) > 3 else ''}")
                    
                    if filtered_out_reasons:
                        st.write("**Features excluded from manual selection:**")
                        for reason in filtered_out_reasons:
                            st.write(f"- {reason}")
                        st.write("*Binary features, target column, already-binned features, and low cardinality features are automatically excluded for safety and quality.*")
                
                # Single button to apply all binning strategies (both recommended and manual)
                if handling_dict:
                    st.write("---")
                    total_features = len(handling_dict)
                    recommended_count = len([col for col in handling_dict if col in features_to_bin])
                    manual_count = total_features - recommended_count
                    
                    if manual_count > 0 and recommended_count > 0:
                        summary_text = f"Apply binning to {total_features} features ({recommended_count} recommended + {manual_count} manually selected)"
                    elif manual_count > 0:
                        summary_text = f"Apply binning to {manual_count} manually selected features"
                    else:
                        summary_text = f"Apply binning to {recommended_count} recommended features"
                    
                    if st.button(summary_text):
                        # Store current state before applying changes
                        st.session_state.feature_binning_ops_applied.append("binning_operation")

                        self._apply_binning_strategies(handling_dict)
            else:
                st.error(suggestions["message"])
        else:
            st.info("No numeric or categorical columns available for binning!")
    
    def _render_feature_binning_configuration(self, col, suggestions, handling_dict, categorical_cols):
        """Render configuration options for a feature that needs binning"""
        with st.expander(f"Configure binning for {col}", expanded=True):
            # Show current distribution
            fig = px.histogram(
                self.builder.training_data,
                x=col,
                title=f"Current Distribution of {col}",
                marginal="box"
            )
            st.plotly_chart(fig)
            
            # Display explanation
            suggestion = suggestions["suggestions"][col]
            st.info(suggestion["reason"])
            
            # Let user choose binning strategy
            strategy = st.selectbox(
                "Choose binning strategy",
                ["None", "Optimal"],
                index=["None", "Optimal"].index(suggestion["strategy"]),
                key=f"binning_strategy_{col}",
                help="""
                - None: Keep variable as is
                - Optimal: Use OptBinning to find optimal cut points based on target relationship
                """
            )
            
            # Log user's strategy selection
            if strategy != "None":
                self.logger.log_user_action(
                    "Binning Strategy Selected",
                    {
                        "column": col,
                        "selected_strategy": strategy,
                        "recommended_strategy": suggestion["strategy"],
                        "followed_recommendation": strategy == suggestion["strategy"],
                        "is_categorical": col in categorical_cols,
                        "unique_values": self.builder.training_data[col].nunique(),
                        "has_nulls": self.builder.training_data[col].isnull().any()
                    }
                )
            
            if strategy != "None":
                # Set default n_bins for all strategies
                n_bins = suggestion.get("n_bins", 10)
                
                # Only show n_bins input for non-Optimal strategies
                if strategy != "Optimal":
                    n_bins = st.number_input(
                        "Number of bins",
                        min_value=2,
                        max_value=20,
                        value=n_bins,
                        key=f"n_bins_{col}",
                        help="More bins = more granular but might lead to sparse bins"
                    )
                else:
                    st.info("Number of bins will be automatically determined based on the relationship with the target variable.")
                
                # Always include n_bins in the handling dict
                handling_dict[col] = {"strategy": strategy, "n_bins": n_bins}
    
    def _render_manual_feature_binning_configuration(self, col, handling_dict, categorical_cols):
        """Render configuration options for a manually selected feature"""
        with st.expander(f"Configure binning for {col} (Manual Selection)", expanded=True):
            # Show current distribution
            fig = px.histogram(
                self.builder.training_data,
                x=col,
                title=f"Current Distribution of {col}",
                marginal="box"
            )
            st.plotly_chart(fig)
            
            # Show basic statistics and analyze feature characteristics
            col_data = self.builder.training_data[col]
            is_categorical = col in categorical_cols
            
            # Analyze feature characteristics for warnings
            unique_count = col_data.nunique()
            warnings = []
            info_messages = []
            
            if not is_categorical:
                skewness = col_data.skew()
                
                # Check for low skewness (typically not recommended for binning)
                if abs(skewness) <= 2:
                    warnings.append({
                        "type": "info",
                        "message": f"⚠️ **Low Skewness Warning**: This feature has relatively normal distribution (skewness = {skewness:.2f}). Binning is typically most beneficial for highly skewed features."
                    })
                
                # Check for non-linear relationship (positive indicator)
                try:
                    # Get target values for non-linearity check
                    target_values = self.builder.training_data[self.builder.target_column]
                    if not pd.api.types.is_numeric_dtype(target_values):
                        target_values = pd.Categorical(target_values).codes
                    
                    # Remove NaN values
                    feature_values = col_data.values
                    mask = ~(np.isnan(feature_values) | np.isnan(target_values))
                    feature_values = feature_values[mask]
                    target_values = target_values[mask]
                    
                    if len(feature_values) >= 10:
                        linear_corr = stats.pearsonr(feature_values, target_values)[0]
                        rank_corr = stats.spearmanr(feature_values, target_values)[0]
                        
                        if abs(rank_corr) > abs(linear_corr) + 0.1:
                            info_messages.append({
                                "type": "success",
                                "message": f"✅ **Non-linear Relationship Detected**: This feature shows a non-linear relationship with the target (Pearson={linear_corr:.2f}, Spearman={rank_corr:.2f}). Binning could be beneficial!"
                            })
                except Exception:
                    pass  # Skip non-linearity check if it fails
                    
            else:  # Categorical feature
                if unique_count > 20:
                    info_messages.append({
                        "type": "success", 
                        "message": f"✅ **High Cardinality Detected**: This categorical feature has {unique_count} unique values. Binning could help reduce complexity and improve model performance."
                    })
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Feature Statistics:**")
                stats_info = f"""
                - Data type: {'Categorical' if is_categorical else 'Numerical'}
                - Unique values: {col_data.nunique():,}
                """
                if not is_categorical:
                    stats_info += f"""
                - Mean: {col_data.mean():.2f}
                - Std: {col_data.std():.2f}
                - Skewness: {col_data.skew():.2f}
                """
                st.markdown(stats_info)
            
            with col2:
                st.write("**General Binning Benefits:**")
                reason_info = """
                Binning can help with:
                - Handling outliers naturally
                - Capturing non-linear relationships
                - Reducing memory usage
                - Improving model interpretability
                - Creating meaningful value groups
                """
                st.markdown(reason_info)
            
            # Display warnings and info messages
            for warning in warnings:
                if warning["type"] == "warning":
                    st.warning(warning["message"])
                elif warning["type"] == "info":
                    st.info(warning["message"])
            
            for info in info_messages:
                if info["type"] == "success":
                    st.success(info["message"])
            
            # Let user choose binning strategy
            strategy = st.selectbox(
                "Choose binning strategy",
                ["None", "Optimal"],
                index=1,  # Default to Optimal for manual selections
                key=f"manual_binning_strategy_{col}",
                help="""
                - None: Keep variable as is
                - Optimal: Use OptBinning to find optimal cut points based on target relationship
                """
            )
            
            # Log user's manual strategy selection
            if strategy != "None":
                self.logger.log_user_action(
                    "Manual Binning Strategy Selected",
                    {
                        "column": col,
                        "selected_strategy": strategy,
                        "manual_selection": True,
                        "is_categorical": col in categorical_cols,
                        "unique_values": self.builder.training_data[col].nunique(),
                        "has_nulls": self.builder.training_data[col].isnull().any(),
                        "skewness": self.builder.training_data[col].skew() if not is_categorical else None,
                        "warnings_shown": [w["message"] for w in warnings],
                        "proceeded_despite_warnings": len(warnings) > 0
                    }
                )
            
            if strategy != "None":
                # Set default n_bins based on feature characteristics
                default_n_bins = min(10, max(3, col_data.nunique() // 5)) if is_categorical else 10
                
                # Only show n_bins input for non-Optimal strategies
                if strategy != "Optimal":
                    n_bins = st.number_input(
                        "Number of bins",
                        min_value=2,
                        max_value=20,
                        value=default_n_bins,
                        key=f"manual_n_bins_{col}",
                        help="More bins = more granular but might lead to sparse bins"
                    )
                else:
                    st.info("Number of bins will be automatically determined based on the relationship with the target variable.")
                    n_bins = default_n_bins
                
                # Include in the handling dict
                handling_dict[col] = {"strategy": strategy, "n_bins": n_bins}
    
    def _apply_binning_strategies(self, handling_dict):
        """Apply the selected binning strategies to the data"""
        before_data = self.builder.training_data.copy()
        result = self.builder.apply_binning(handling_dict, use_training_data=True)
        
        if result["success"]:
            # Store fitted binning objects for visualization
            if not hasattr(st.session_state, 'fitted_binning_objects'):
                st.session_state.fitted_binning_objects = {}
            
            # Validate that the expected number of features were actually binned
            expected_features = [col for col, params in handling_dict.items() 
                               if params.get("strategy") != "None"]
            actually_modified = result.get("modified_columns", [])
            actually_unchanged = result.get("unchanged_columns", [])
            
            # Check for discrepancies and determine specific failure reasons
            failed_features = []
            failure_details = {}
            
            for expected_col in expected_features:
                if expected_col in actually_unchanged:
                    failed_features.append(expected_col)
                    
                    # Determine specific failure reason by analyzing the feature characteristics
                    failure_reason = self._determine_binning_failure_reason(
                        before_data, expected_col, handling_dict[expected_col]
                    )
                    failure_details[expected_col] = failure_reason
            
            # Enhanced logging for successful binning (including failures)
            self.logger.log_calculation(
                "Binning Results",
                {
                    "strategies_applied": handling_dict,
                    "expected_to_bin": expected_features,
                    "actually_binned": [col for col in expected_features if col not in failed_features],
                    "failed_to_bin": failed_features,
                    "failure_details": failure_details,
                    "modified_columns": result.get("modified_columns", []),
                    "dropped_columns": result.get("dropped_columns", []),
                    "unchanged_columns": result.get("unchanged_columns", []),
                    "bin_ranges": result.get("bin_ranges", {}),
                    "impact_summary": {
                        orig_col: {
                            "original_unique_values": before_data[orig_col].nunique(),
                            "binned_unique_values": self.builder.training_data[binned_col].nunique(),
                            "reduction_ratio": 1 - (self.builder.training_data[binned_col].nunique() / before_data[orig_col].nunique()),
                            "memory_impact": (self.builder.training_data[binned_col].memory_usage() - before_data[orig_col].memory_usage()) / 1024
                        } for orig_col, binned_col in zip(result.get("dropped_columns", []), result.get("modified_columns", []))
                    }
                }
            )
            
            self.logger.log_user_action(
                "Variable Binning",
                {
                    "strategies": handling_dict,
                    "expected_features": expected_features,
                    "successful_features": [col for col in expected_features if col not in failed_features],
                    "failed_features": failed_features,
                    "failure_details": failure_details,
                    "modified_columns": result.get("modified_columns", []),
                    "dropped_columns": result.get("dropped_columns", []),
                    "unchanged_columns": result.get("unchanged_columns", []),
                    "bin_ranges": result.get("bin_ranges", {})
                }
            )
            
            self.logger.log_journey_point(
                stage="DATA_PREPROCESSING",
                decision_type="FEATURE_BINNING",
                description="Feature binning completed",
                details={'New Binned Columns': result.get("modified_columns", []),
                         'Dropped Columns': result.get("dropped_columns", []),
                         'Unchanged Columns': result.get("unchanged_columns", []),
                         'Failed Features': failed_features,
                         'Failure Details': failure_details,
                        },
                parent_id=None
            )
            
            # Store binning information in session state
            if not hasattr(st.session_state, 'binning_info'):
                st.session_state.binning_info = {}
            
            # For each binned column, store its original name and bin ranges
            for orig_col, binned_col in zip(result["dropped_columns"], result["modified_columns"]):
                if binned_col in result.get("bin_ranges", {}):
                    st.session_state.binning_info[binned_col] = {
                        "original_feature": orig_col,
                        "bin_ranges": result["bin_ranges"][binned_col],
                        "is_categorical": orig_col in self.builder.training_data.select_dtypes(include=['object', 'category']).columns
                    }
            
            # Provide more accurate success messaging with specific failure reasons
            if failed_features:
                if len(failed_features) == len(expected_features):
                    st.error(f"❌ **Binning Failed**: None of the {len(expected_features)} selected features could be binned.")
                    st.write("**Specific failure reasons:**")
                    for feature, reason in failure_details.items():
                        st.write(f"- **{feature}**: {reason}")
                else:
                    successfully_binned = len(expected_features) - len(failed_features)
                    st.warning(f"⚠️ **Partial Success**: {successfully_binned} out of {len(expected_features)} features were successfully binned.")
                    
                    st.write("**Specific failure reasons:**")
                    for feature, reason in failure_details.items():
                        st.write(f"- **{feature}**: {reason}")
                    
                    if successfully_binned > 0:
                        st.success(f"✅ Successfully binned: **{', '.join([col for col in expected_features if col not in failed_features])}**")
            else:
                st.success(f"✅ **Binning Successful**: All {len(expected_features)} selected features were successfully binned!")
            
            # Convert binned features to appropriate data types
            for orig_col, binned_col in zip(result["dropped_columns"], result["modified_columns"]):
                if orig_col in self.builder.training_data.select_dtypes(include=['object', 'category']).columns:
                    # For originally categorical features, convert to categorical
                    self.builder.training_data[binned_col] = self.builder.training_data[binned_col].astype('category')
                else:
                    # For originally numeric features, convert to integer
                    self.builder.training_data[binned_col] = self.builder.training_data[binned_col].astype('int32')
            
            # Show impact of changes only if there were successful binnings
            if len(actually_modified) > 0:
                self._display_binning_impact(result, before_data)
            else:
                st.info("No features were modified, so no impact analysis is available.")

            # Create a DataframeComparisonComponent instance
            comparison_component = DataframeComparisonComponent(
                original_df=before_data,
                modified_df=self.builder.training_data,
                target_column=self.builder.target_column)
            comparison_component.render()
        else:
            st.error(result["message"])
    
    def _determine_binning_failure_reason(self, data, column, params):
        """
        Determine the specific reason why binning failed for a column by analyzing its characteristics.
        """
        try:
            strategy = params.get("strategy", "Optimal")
            is_categorical = data[column].dtype == 'object' or data[column].dtype.name == 'category'
            
            # Basic feature characteristics
            unique_count = data[column].nunique()
            n_samples = len(data)
            has_nulls = data[column].isnull().any()
            null_percentage = (data[column].isnull().sum() / len(data)) * 100
            
            # Target analysis
            target_column = self.builder.target_column
            target_values = data[target_column]
            if not pd.api.types.is_numeric_dtype(target_values):
                target_values = pd.Categorical(target_values).codes
            
            # Use session state to determine problem type if available, otherwise fall back to heuristics
            import streamlit as st
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'problem_type'):
                problem_type = st.session_state.problem_type
                if problem_type == "binary_classification":
                    target_type = "binary"
                elif problem_type == "multiclass_classification":
                    target_type = "multiclass"
                else:  # regression
                    target_type = "continuous"
            else:
                # Fallback to original heuristic
                target_unique_count = data[target_column].nunique()
                if pd.api.types.is_numeric_dtype(data[target_column]) and target_unique_count <= 2:
                    target_type = "binary"
                elif target_unique_count <= 2:
                    target_type = "binary" 
                else:
                    target_type = "continuous"
            
            # Try to simulate what OptimalBinning would do to understand the failure
            try:
                if strategy == "Optimal":
                    from optbinning import OptimalBinning, ContinuousOptimalBinning
                    
                    # Configure binning
                    min_n_bins = 3
                    max_n_bins = max(min_n_bins, min(20 if not is_categorical else 10, 
                                                    unique_count // (5 if not is_categorical else 3)))
                    min_bin_size = max(0.05, 1/np.sqrt(n_samples))
                    
                    # Select appropriate binning class based on problem type
                    if target_type == "binary":
                        binning_class = OptimalBinning
                        additional_params = {"cat_cutoff": 0.05, "max_pvalue": 0.05}
                    else:  # multiclass or continuous
                        binning_class = ContinuousOptimalBinning
                        additional_params = {}
                    
                    binning = binning_class(
                        name=column,
                        dtype="categorical" if is_categorical else "numerical",
                        min_n_bins=min_n_bins,
                        max_n_bins=max_n_bins,
                        min_bin_size=min_bin_size,
                        monotonic_trend="auto" if not is_categorical else None,
                        min_prebin_size=0.01 if not is_categorical else None,
                        max_n_prebins=50 if not is_categorical else None,
                        **additional_params
                    )
                    
                    # Fit and check results
                    binning.fit(data[column], target_values)
                    
                    if is_categorical:
                        # For categorical features, check if cardinality was reduced
                        binned_data = binning.transform(data[column], metric="indices")
                        binned_unique_count = pd.Series(binned_data).nunique()
                        
                        if binned_unique_count >= unique_count:
                            return f"Optimal binning couldn't reduce cardinality. Original: {unique_count} categories, After binning: {binned_unique_count} categories. The algorithm determined that merging categories would not improve predictive power."
                        
                    else:
                        # For numerical features, check splits
                        splits = binning.splits
                        if len(splits) == 0:
                            # Analyze why no splits were found
                            if unique_count < 10:
                                return f"Feature has too few unique values ({unique_count}) for meaningful binning. Optimal binning requires sufficient variation to create statistically significant splits."
                            
                            # Check correlation with target
                            from scipy import stats
                            try:
                                correlation = stats.pearsonr(data[column].fillna(data[column].median()), target_values)[0]
                                if abs(correlation) < 0.01:
                                    return f"Very weak correlation with target variable (r={correlation:.4f}). Optimal binning couldn't find splits that improve predictive power over the original continuous feature."
                            except:
                                pass
                            
                            # Check for monotonic relationship
                            try:
                                spearman_corr = stats.spearmanr(data[column].fillna(data[column].median()), target_values)[0]
                                if abs(spearman_corr) < 0.05:
                                    return f"No significant monotonic relationship with target (Spearman r={spearman_corr:.4f}). Optimal binning requires some relationship with the target to create meaningful bins."
                            except:
                                pass
                            
                            return f"OptimalBinning algorithm couldn't find statistically significant split points. This usually means the feature doesn't have a clear relationship with the target variable that would benefit from binning."
                    
                    return "Unknown reason - binning should have succeeded based on feature characteristics."
                    
            except Exception as binning_error:
                return f"OptimalBinning algorithm error: {str(binning_error)}"
                
        except Exception as e:
            return f"Error analyzing failure reason: {str(e)}"
        
        # Fallback analysis based on feature characteristics
        if has_nulls and null_percentage > 50:
            return f"Feature has {null_percentage:.1f}% missing values, which may prevent effective binning."
        
        if unique_count < 3:
            return f"Feature has only {unique_count} unique values, insufficient for binning (minimum 3 required)."
        
        if is_categorical and unique_count <= 5:
            return f"Categorical feature has only {unique_count} categories, which is already low cardinality and doesn't need binning."
        
        return "Feature characteristics don't support optimal binning, but specific reason couldn't be determined."
    
    def _display_binning_impact(self, result, before_data):
        import streamlit as st

        """Display the impact of binning on the features"""
        st.write("### Impact of Binning")
        st.write("""
            Below is a detailed analysis of how binning has affected your features. 
            We'll look at distribution changes, information preservation, and potential impact on model performance.
        """)
        
        # Calculate impact summary
        impact_summary = {}
        for orig_col, binned_col in zip(result["dropped_columns"], result["modified_columns"]):
            impact_summary[orig_col] = {
                "original_unique_values": before_data[orig_col].nunique(),
                "binned_unique_values": self.builder.training_data[binned_col].nunique(),
                "reduction_ratio": 1 - (self.builder.training_data[binned_col].nunique() / before_data[orig_col].nunique()),
                "memory_impact": (self.builder.training_data[binned_col].memory_usage() - before_data[orig_col].memory_usage()) / 1024
            }
        
        # Overall summary
        st.write("#### 📊 Overall Summary")
        total_memory_impact = sum(
            impact["memory_impact"] 
            for impact in impact_summary.values()
        )
        avg_reduction = np.mean([
            impact["reduction_ratio"] 
            for impact in impact_summary.values()
        ]) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Features Modified",
                len(result["modified_columns"]),
                help="Number of features that were successfully binned"
            )
        with col2:
            st.metric(
                "Avg. Cardinality Reduction",
                f"{avg_reduction:.1f}%",
                help="Average reduction in number of unique values across all binned features"
            )
        with col3:
            st.metric(
                "Memory Impact",
                f"{total_memory_impact:.2f} KB",
                help="Total change in memory usage after binning (negative means reduction)"
            )

        # Keep track of columns we've already shown
        shown_columns = set()
        
        for idx, (orig_col, binned_col) in enumerate(zip(result["dropped_columns"], result["modified_columns"])):
            # Skip if we've already shown this column
            if orig_col in shown_columns:
                continue
            shown_columns.add(orig_col)
            
            st.write(f"#### 📈 Analysis for {orig_col}")
            
            # Get impact metrics
            impact = impact_summary[orig_col]
            is_categorical = orig_col in before_data.select_dtypes(include=['object', 'category']).columns
            
            # Display event rate visualization using actual bin ranges
            if binned_col in result.get("bin_ranges", {}):
                self._display_event_rate_from_actual_bins(
                    orig_col, binned_col, result["bin_ranges"][binned_col], before_data, is_categorical
                )
            
            # Bin information with enhanced explanations
            if binned_col in result.get("bin_ranges", {}):
                self._display_bin_details(orig_col, binned_col, result, before_data, is_categorical)
    
    def _display_bin_details(self, orig_col, binned_col, result, before_data, is_categorical):
        import streamlit as st
        """Display detailed information about the bins created for a feature"""
        st.write("**📋 Bin Details**")
        bin_ranges = result["bin_ranges"][binned_col]
        
        # Get event rate data if available
        event_rate_data = None
        if hasattr(st.session_state, 'event_rate_data') and orig_col in st.session_state.event_rate_data:
            event_rate_data = st.session_state.event_rate_data[orig_col]
        
        # Use session state to determine problem type for appropriate bin details display
        import streamlit as st
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'problem_type'):
            problem_type = st.session_state.problem_type
            is_binary_target = problem_type == "binary_classification"
            is_multiclass_target = problem_type == "multiclass_classification"
        else:
            # Fallback to heuristic
            target_unique_count = before_data[self.builder.target_column].nunique()
            is_binary_target = target_unique_count == 2
            is_multiclass_target = target_unique_count > 2 and target_unique_count <= 20
        
        # Check if we're dealing with a categorical binning result where bin_ranges is a dict
        # mapping bin IDs to category lists
        if isinstance(bin_ranges, dict) and all(isinstance(v, list) for v in bin_ranges.values()):
            # This is a categorical binning format - bin_id -> list of category values
            table_data = []
            
            for bin_id, categories in sorted(bin_ranges.items()):
                bin_size = sum(before_data[orig_col].isin(categories))
                bin_pct = (bin_size / len(before_data)) * 100
                
                row = {
                    "Bin": f"Bin {bin_id}",
                    "Categories": ', '.join(str(cat) for cat in categories[:3]),
                    "Samples": f"{bin_size:,}",
                    "Percentage": f"{bin_pct:.1f}%"
                }
                
                if len(categories) > 3:
                    row["Categories"] += f" ... (+{len(categories)-3} more)"
                
                # Add target information based on problem type
                if event_rate_data and is_binary_target:
                    matching_stat = next((stat for stat in event_rate_data if stat['bin_id'] == f"Bin {bin_id}"), None)
                    if matching_stat:
                        row["Events"] = matching_stat['event_count']
                        row["Non-Events"] = matching_stat['non_event_count']
                        row["Event Rate"] = f"{matching_stat['event_rate']:.3f}"
                elif event_rate_data and is_multiclass_target:
                    # For multi-class targets, show dominant class information
                    matching_stat = next((stat for stat in event_rate_data if stat['bin_id'] == f"Bin {bin_id}"), None)
                    if matching_stat:
                        row["Dominant Class"] = str(matching_stat['dominant_class'])
                        row["Class Proportion"] = f"{matching_stat['dominant_class_proportion']:.3f}"
                elif event_rate_data and not is_binary_target and not is_multiclass_target:
                    # For continuous targets, use the pre-calculated target statistics from event_rate_data
                    matching_stat = next((stat for stat in event_rate_data if stat['bin_id'] == f"Bin {bin_id}"), None)
                    if matching_stat:
                        row["Target Mean"] = f"{matching_stat['mean_target']:.3f}"
                        row["Target Std"] = f"{matching_stat['std_target']:.3f}"
                
                table_data.append(row)
            
            if table_data:
                df = pd.DataFrame(table_data)
                st.dataframe(df, width='stretch')
        
        # Handle numeric binning result which is a list of (lower, upper) tuples
        elif isinstance(bin_ranges, list) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in bin_ranges):
            # Create a DataFrame for better visualization of numeric bins
            bin_details = []
            
            # Handle numerical bins (list of tuples)
            for i, (lower, upper) in enumerate(bin_ranges):
                # Format range string
                if np.isinf(lower) and np.isinf(upper):
                    range_str = "All values"
                elif np.isinf(lower):
                    range_str = f"≤ {upper:.2f}"
                elif np.isinf(upper):
                    range_str = f"> {lower:.2f}"
                else:
                    range_str = f"{lower:.2f} to {upper:.2f}"
                
                # Calculate bin statistics
                mask = (before_data[orig_col] > lower) & (before_data[orig_col] <= upper) if not np.isinf(upper) \
                      else (before_data[orig_col] > lower) if np.isinf(upper) \
                      else (before_data[orig_col] <= upper)
                
                bin_data = before_data[orig_col][mask]
                bin_size = len(bin_data)
                bin_pct = (bin_size / len(before_data)) * 100
                
                row = {
                    "Bin": f"Bin {i}",
                    "Range": range_str,
                    "Samples": f"{bin_size:,}",
                    "Percentage": f"{bin_pct:.1f}%",
                    "Mean": f"{bin_data.mean():.2f}" if bin_size > 0 else "N/A",
                    "Std": f"{bin_data.std():.2f}" if bin_size > 0 else "N/A"
                }
                
                # Add target information based on problem type
                if event_rate_data and is_binary_target:
                    matching_stat = next((stat for stat in event_rate_data if stat['bin_id'] == f"Bin {i}"), None)
                    if matching_stat:
                        row["Events"] = matching_stat['event_count']
                        row["Non-Events"] = matching_stat['non_event_count']
                        row["Event Rate"] = f"{matching_stat['event_rate']:.3f}"
                elif event_rate_data and is_multiclass_target:
                    # For multi-class targets, show dominant class information
                    matching_stat = next((stat for stat in event_rate_data if stat['bin_id'] == f"Bin {i}"), None)
                    if matching_stat:
                        row["Dominant Class"] = str(matching_stat['dominant_class'])
                        row["Class Proportion"] = f"{matching_stat['dominant_class_proportion']:.3f}"
                elif event_rate_data and not is_binary_target and not is_multiclass_target:
                    # For continuous targets, use the pre-calculated target statistics from event_rate_data
                    matching_stat = next((stat for stat in event_rate_data if stat['bin_id'] == f"Bin {i}"), None)
                    if matching_stat:
                        row["Target Mean"] = f"{matching_stat['mean_target']:.3f}"
                        row["Target Std"] = f"{matching_stat['std_target']:.3f}"
                
                bin_details.append(row)
            
            # Convert to DataFrame and display as a styled table
            if bin_details:
                bin_df = pd.DataFrame(bin_details)
                
                # Create a more visual representation with a styled table
                st.write("Below is a comprehensive summary of each bin's characteristics:")
                
                # Add table interpretation
                with st.expander("📖 Table Column Explanations"):
                    st.write("**Basic Bin Information:**")
                    st.write("- **Bin**: Bin identifier")
                    st.write("- **Range/Categories**: Value range or category list for the bin")
                    st.write("- **Samples**: Number of observations in the bin")
                    st.write("- **Percentage**: Percentage of total data in the bin")
                    
                    if is_binary_target and event_rate_data:
                        st.write("\n**Event Rate Information (Binary Classification):**")
                        st.write("- **Events**: Number of positive outcomes (1s)")
                        st.write("- **Non-Events**: Number of negative outcomes (0s)")
                        st.write("- **Event Rate**: Proportion of positive outcomes")
                    elif is_multiclass_target and event_rate_data:
                        st.write("\n**Class Distribution Information (Multi-class Classification):**")
                        st.write("- **Dominant Class**: The most frequent class in the bin")
                        st.write("- **Class Proportion**: Proportion of the dominant class in the bin")
                    elif not is_binary_target and not is_multiclass_target and event_rate_data:
                        st.write("\n**Target Statistics (Regression):**")
                        st.write("- **Target Mean**: Average target value in the bin")
                        st.write("- **Target Std**: Standard deviation of target in the bin")
                    
                    st.write("\n**Feature Statistics (Numerical Features):**")
                    st.write("- **Mean**: Average feature value in the bin")
                    st.write("- **Std**: Standard deviation of feature values in the bin")
                
                # Style the dataframe
                def highlight_bins(x):
                    return ['background-color: #f0f2f6'] * len(x) if x.name % 2 == 0 else ['background-color: white'] * len(x)
                
                # Apply styling and display
                styled_df = bin_df.style\
                    .apply(highlight_bins, axis=1)\
                    .set_properties(**{
                        'text-align': 'center',
                        'font-size': '14px',
                        'padding': '5px'
                    })\
                    .set_table_styles([
                        {'selector': 'th',
                         'props': [('background-color', '#e6e9ef'),
                                  ('color', 'black'),
                                  ('font-weight', 'bold'),
                                  ('text-align', 'center'),
                                  ('padding', '5px')]},
                        {'selector': 'td',
                         'props': [('padding', '5px')]}
                    ])
                
                st.dataframe(styled_df, hide_index=True)
        
        # Special case for OptimalBinning categorical output where bin_ranges is just indices
        elif isinstance(bin_ranges, dict) and all(isinstance(k, str) and k.isdigit() for k in bin_ranges.keys()):
            st.write("This categorical feature was successfully binned into groups:")
            
            # Check if this is just bin indices
            has_simple_indices = True
            for bin_id, value in bin_ranges.items():
                if not (isinstance(value, (int, str)) or (isinstance(value, list) and len(value) == 1)):
                    has_simple_indices = False
                    break
            
            if has_simple_indices:
                # Get the unique bins in the training data
                unique_bins = sorted(self.builder.training_data[binned_col].unique())
                
                # Create a table showing the distribution of data across bins
                bin_counts = self.builder.training_data[binned_col].value_counts().sort_index()
                bin_pcts = (bin_counts / len(self.builder.training_data) * 100).round(1)
                
                table_data = []
                for bin_value in unique_bins:
                    count = bin_counts.get(bin_value, 0)
                    pct = bin_pcts.get(bin_value, 0)
                    
                    row = {
                        "Bin": f"Bin {bin_value}",
                        "Samples": f"{count:,}",
                        "Percentage": f"{pct:.1f}%"
                    }
                    
                    # Add target information based on problem type
                    if event_rate_data and is_binary_target:
                        matching_stat = next((stat for stat in event_rate_data if stat['bin_id'] == f"Bin {bin_value}"), None)
                        if matching_stat:
                            row["Events"] = matching_stat['event_count']
                            row["Non-Events"] = matching_stat['non_event_count']
                            row["Event Rate"] = f"{matching_stat['event_rate']:.3f}"
                    elif event_rate_data and is_multiclass_target:
                        # For multi-class targets, show dominant class information
                        matching_stat = next((stat for stat in event_rate_data if stat['bin_id'] == f"Bin {bin_value}"), None)
                        if matching_stat:
                            row["Dominant Class"] = str(matching_stat['dominant_class'])
                            row["Class Proportion"] = f"{matching_stat['dominant_class_proportion']:.3f}"
                    elif event_rate_data and not is_binary_target and not is_multiclass_target:
                        # For continuous targets, use the pre-calculated target statistics from event_rate_data
                        matching_stat = next((stat for stat in event_rate_data if stat['bin_id'] == f"Bin {bin_value}"), None)
                        if matching_stat:
                            row["Target Mean"] = f"{matching_stat['mean_target']:.3f}"
                            row["Target Std"] = f"{matching_stat['std_target']:.3f}"
                    
                    table_data.append(row)
                
                if table_data:
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, width='stretch')
            else:
                # Fall back to showing bin_ranges directly if structure is unexpected
                st.write(f"Bin mapping: {bin_ranges}")
        
        # Unknown format - just show the raw data
        else:
            st.info(f"Binning was applied to {binned_col}, but detailed bin information is not available in the expected format.")
            st.write("Summary of binned values:")
            
            if binned_col in self.builder.training_data.columns:
                bin_values = self.builder.training_data[binned_col].value_counts().sort_index()
                table_data = []
                
                for bin_val, count in bin_values.items():
                    pct = (count / len(self.builder.training_data)) * 100
                    
                    row = {
                        "Bin Value": f"Value {bin_val}",
                        "Samples": f"{count:,}",
                        "Percentage": f"{pct:.1f}%"
                    }
                    
                    # Add target information based on problem type
                    if event_rate_data and is_binary_target:
                        matching_stat = next((stat for stat in event_rate_data if stat['bin_id'] == f"Bin {bin_val}"), None)
                        if matching_stat:
                            row["Events"] = matching_stat['event_count']
                            row["Non-Events"] = matching_stat['non_event_count']
                            row["Event Rate"] = f"{matching_stat['event_rate']:.3f}"
                    elif event_rate_data and is_multiclass_target:
                        # For multi-class targets, show dominant class information
                        matching_stat = next((stat for stat in event_rate_data if stat['bin_id'] == f"Bin {bin_val}"), None)
                        if matching_stat:
                            row["Dominant Class"] = str(matching_stat['dominant_class'])
                            row["Class Proportion"] = f"{matching_stat['dominant_class_proportion']:.3f}"
                    elif event_rate_data and not is_binary_target and not is_multiclass_target:
                        # For continuous targets, use the pre-calculated target statistics from event_rate_data
                        matching_stat = next((stat for stat in event_rate_data if stat['bin_id'] == f"Bin {bin_val}"), None)
                        if matching_stat:
                            row["Target Mean"] = f"{matching_stat['mean_target']:.3f}"
                            row["Target Std"] = f"{matching_stat['std_target']:.3f}"
                    
                    table_data.append(row)
                
                if table_data:
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, width='stretch')
            else:
                st.write("(Binned column not found in training data)")
    
    def _display_event_rate_from_actual_bins(self, orig_col, binned_col, bin_ranges, before_data, is_categorical):
        """Display event rate visualization from actual bin ranges."""
        try:
            import streamlit as st
            st.write(f"**📊 Event Rate Analysis for {orig_col}**")
            
            # Use session state to determine problem type for appropriate visualization
            import streamlit as st
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'problem_type'):
                problem_type = st.session_state.problem_type
                is_binary_classification = problem_type == "binary_classification"
                is_multiclass_classification = problem_type == "multiclass_classification"
            else:
                # Fallback to heuristic
                target_unique_count = before_data[self.builder.target_column].nunique()
                is_binary_classification = target_unique_count == 2
                is_multiclass_classification = target_unique_count > 2 and target_unique_count <= 20
            
            if is_multiclass_classification:
                st.info("For multi-class classification problems, we'll show the distribution of target classes across bins instead of event rates.")
                self._display_multiclass_distribution_from_actual_bins(orig_col, bin_ranges, before_data, is_categorical)
                return
            elif not is_binary_classification:
                st.info("For regression problems, we'll show mean target values across bins instead of event rates.")
                self._display_mean_value_from_actual_bins(orig_col, bin_ranges, before_data, is_categorical)
                return
            
            # Calculate event rates for each bin using actual bin ranges
            bin_stats = []
            
            if isinstance(bin_ranges, dict) and all(isinstance(v, list) for v in bin_ranges.values()):
                # Categorical binning: bin_ranges is {bin_id: [category_list]}
                for bin_id, categories in sorted(bin_ranges.items()):
                    mask = before_data[orig_col].isin(categories)
                    bin_data = before_data[mask]
                    
                    if len(bin_data) > 0:
                        total_count = len(bin_data)
                        event_count = bin_data[self.builder.target_column].sum()
                        non_event_count = total_count - event_count
                        event_rate = event_count / total_count if total_count > 0 else 0
                        
                        bin_stats.append({
                            'bin_id': f"Bin {bin_id}",
                            'total_count': total_count,
                            'event_count': event_count,
                            'non_event_count': non_event_count,
                            'event_rate': event_rate,
                            'categories': categories
                        })
                        
            elif isinstance(bin_ranges, list) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in bin_ranges):
                # Numerical binning: bin_ranges is [(lower, upper), ...]
                for i, (lower, upper) in enumerate(bin_ranges):
                    # Create mask based on bin boundaries
                    if np.isinf(lower) and np.isinf(upper):
                        mask = pd.Series([True] * len(before_data), index=before_data.index)
                    elif np.isinf(lower):
                        mask = before_data[orig_col] <= upper
                    elif np.isinf(upper):
                        mask = before_data[orig_col] > lower
                    else:
                        mask = (before_data[orig_col] > lower) & (before_data[orig_col] <= upper)
                    
                    bin_data = before_data[mask]
                    
                    if len(bin_data) > 0:
                        total_count = len(bin_data)
                        event_count = bin_data[self.builder.target_column].sum()
                        non_event_count = total_count - event_count
                        event_rate = event_count / total_count if total_count > 0 else 0
                        
                        # Format range string
                        if np.isinf(lower) and np.isinf(upper):
                            range_str = "All values"
                        elif np.isinf(lower):
                            range_str = f"≤ {upper:.2f}"
                        elif np.isinf(upper):
                            range_str = f"> {lower:.2f}"
                        else:
                            range_str = f"({lower:.2f}, {upper:.2f}]"
                        
                        bin_stats.append({
                            'bin_id': f"Bin {i}",
                            'total_count': total_count,
                            'event_count': event_count,
                            'non_event_count': non_event_count,
                            'event_rate': event_rate,
                            'range': range_str
                        })
            
            if not bin_stats:
                st.warning("Could not calculate event rates from bin ranges.")
                return
            
            # Create the event rate plot
            fig = go.Figure()
            
            # Prepare data for plotting
            bin_labels = [stat['bin_id'] for stat in bin_stats]
            event_rates = [stat['event_rate'] for stat in bin_stats]
            
            # Prepare custom data for hover
            hover_data = []
            for stat in bin_stats:
                hover_data.append([
                    stat['total_count'],
                    stat['non_event_count'],
                    stat['event_count']
                ])
            
            # Add event rate bars
            fig.add_trace(go.Bar(
                x=bin_labels,
                y=event_rates,
                name='Event Rate',
                marker_color='lightblue',
                hovertemplate='<b>%{x}</b><br>' +
                             '<b>Event Rate</b>: %{y:.3f}<br>' +
                             '<b>Total Count</b>: %{customdata[0]}<br>' +
                             '<b>Non-Events</b>: %{customdata[1]}<br>' +
                             '<b>Events</b>: %{customdata[2]}<br>' +
                             '<extra></extra>',
                customdata=hover_data
            ))
            
            # Add overall event rate line
            overall_event_rate = before_data[self.builder.target_column].mean()
            fig.add_hline(
                y=overall_event_rate,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Overall Event Rate: {overall_event_rate:.3f}",
                annotation_position="top right"
            )
            
            # Update layout
            fig.update_layout(
                title=f"Event Rate by Bin - {orig_col}",
                xaxis_title="Bins",
                yaxis_title="Event Rate",
                height=400,
                showlegend=True,
                hovermode='closest'
            )
            
            # Add annotation explaining what we're seeing
            st.info("📈 This chart shows how the event rate (proportion of positive outcomes) varies across different bins of the feature. Bins with significantly different event rates from the overall average indicate strong predictive power.")
            
            st.plotly_chart(fig, config={'responsive': True})
            
            # Store event rate data for the bin details table
            if not hasattr(st.session_state, 'event_rate_data'):
                st.session_state.event_rate_data = {}
            st.session_state.event_rate_data[orig_col] = bin_stats
            
        except Exception as e:
            st.warning(f"Error creating event rate visualization: {str(e)}")
            # Debug information
            st.write(f"Debug info - bin_ranges type: {type(bin_ranges)}")
            st.write(f"Debug info - bin_ranges: {bin_ranges}")
    
    def _display_multiclass_distribution_from_actual_bins(self, orig_col, bin_ranges, before_data, is_categorical):
        """Display multi-class target distribution by bin using actual bin ranges."""
        try:
            import streamlit as st
            # Calculate class distributions for each bin using actual bin ranges
            bin_stats = []
            target_classes = sorted(before_data[self.builder.target_column].unique())
            
            if isinstance(bin_ranges, dict) and all(isinstance(v, list) for v in bin_ranges.values()):
                # Categorical binning: bin_ranges is {bin_id: [category_list]}
                for bin_id, categories in sorted(bin_ranges.items()):
                    mask = before_data[orig_col].isin(categories)
                    bin_data = before_data[mask]
                    
                    if len(bin_data) > 0:
                        total_count = len(bin_data)
                        class_counts = {}
                        class_proportions = {}
                        
                        for cls in target_classes:
                            count = (bin_data[self.builder.target_column] == cls).sum()
                            proportion = count / total_count if total_count > 0 else 0
                            class_counts[cls] = count
                            class_proportions[cls] = proportion
                        
                        bin_stats.append({
                            'bin_id': f"Bin {bin_id}",
                            'total_count': total_count,
                            'class_counts': class_counts,
                            'class_proportions': class_proportions,
                            'categories': categories
                        })
                        
            elif isinstance(bin_ranges, list) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in bin_ranges):
                # Numerical binning: bin_ranges is [(lower, upper), ...]
                for i, (lower, upper) in enumerate(bin_ranges):
                    # Create mask based on bin boundaries
                    if np.isinf(lower) and np.isinf(upper):
                        mask = pd.Series([True] * len(before_data), index=before_data.index)
                    elif np.isinf(lower):
                        mask = before_data[orig_col] <= upper
                    elif np.isinf(upper):
                        mask = before_data[orig_col] > lower
                    else:
                        mask = (before_data[orig_col] > lower) & (before_data[orig_col] <= upper)
                    
                    bin_data = before_data[mask]
                    
                    if len(bin_data) > 0:
                        total_count = len(bin_data)
                        class_counts = {}
                        class_proportions = {}
                        
                        for cls in target_classes:
                            count = (bin_data[self.builder.target_column] == cls).sum()
                            proportion = count / total_count if total_count > 0 else 0
                            class_counts[cls] = count
                            class_proportions[cls] = proportion
                        
                        # Format range string
                        if np.isinf(lower) and np.isinf(upper):
                            range_str = "All values"
                        elif np.isinf(lower):
                            range_str = f"≤ {upper:.2f}"
                        elif np.isinf(upper):
                            range_str = f"> {lower:.2f}"
                        else:
                            range_str = f"({lower:.2f}, {upper:.2f}]"
                        
                        bin_stats.append({
                            'bin_id': f"Bin {i}",
                            'total_count': total_count,
                            'class_counts': class_counts,
                            'class_proportions': class_proportions,
                            'range': range_str
                        })
            
            if not bin_stats:
                st.warning("Could not calculate class distributions from bin ranges.")
                return
            
            # Create stacked bar chart showing class distributions
            fig = go.Figure()
            
            # Prepare data for plotting
            bin_labels = [stat['bin_id'] for stat in bin_stats]
            
            # Add a trace for each class
            colors = px.colors.qualitative.Set3[:len(target_classes)]
            
            for idx, cls in enumerate(target_classes):
                proportions = [stat['class_proportions'][cls] for stat in bin_stats]
                counts = [stat['class_counts'][cls] for stat in bin_stats]
                
                fig.add_trace(go.Bar(
                    name=f'Class {cls}',
                    x=bin_labels,
                    y=proportions,
                    marker_color=colors[idx % len(colors)],
                    hovertemplate='<b>%{x}</b><br>' +
                                 f'<b>Class {cls}</b><br>' +
                                 '<b>Proportion</b>: %{y:.3f}<br>' +
                                 '<b>Count</b>: %{customdata}<br>' +
                                 '<extra></extra>',
                    customdata=counts
                ))
            
            # Update layout for stacked bars
            fig.update_layout(
                title=f"Class Distribution by Bin - {orig_col}",
                xaxis_title="Bins",
                yaxis_title="Class Proportion",
                barmode='stack',
                height=400,
                showlegend=True,
                hovermode='closest'
            )
            
            # Add annotation explaining what we're seeing
            st.info("📊 This stacked bar chart shows how the distribution of target classes varies across different bins. Bins with significantly different class distributions indicate strong predictive power for multi-class classification.")
            
            st.plotly_chart(fig, config={'responsive': True})
            
            # Store class distribution data for the bin details table
            if not hasattr(st.session_state, 'event_rate_data'):
                st.session_state.event_rate_data = {}
            
            # Convert to format compatible with bin details table
            converted_stats = []
            for stat in bin_stats:
                # Find the dominant class
                dominant_class = max(stat['class_proportions'], key=stat['class_proportions'].get)
                dominant_proportion = stat['class_proportions'][dominant_class]
                
                converted_stats.append({
                    'bin_id': stat['bin_id'],
                    'total_count': stat['total_count'],
                    'dominant_class': dominant_class,
                    'dominant_class_proportion': dominant_proportion,
                    'class_counts': stat['class_counts'],
                    'class_proportions': stat['class_proportions']
                })
            
            st.session_state.event_rate_data[orig_col] = converted_stats
            
        except Exception as e:
            st.warning(f"Error creating multi-class distribution visualization: {str(e)}")
            # Debug information
            st.write(f"Debug info - bin_ranges type: {type(bin_ranges)}")
            st.write(f"Debug info - bin_ranges: {bin_ranges}")
    
    def _display_mean_value_from_actual_bins(self, orig_col, bin_ranges, before_data, is_categorical):
        """Display mean target value by bin for continuous targets using actual bin ranges."""
        try:
            import streamlit as st
            # Calculate mean target values for each bin using actual bin ranges
            bin_stats = []
            
            if isinstance(bin_ranges, dict) and all(isinstance(v, list) for v in bin_ranges.values()):
                # Categorical binning: bin_ranges is {bin_id: [category_list]}
                for bin_id, categories in sorted(bin_ranges.items()):
                    mask = before_data[orig_col].isin(categories)
                    bin_data = before_data[mask]
                    
                    if len(bin_data) > 0:
                        total_count = len(bin_data)
                        mean_target = bin_data[self.builder.target_column].mean()
                        std_target = bin_data[self.builder.target_column].std()
                        
                        bin_stats.append({
                            'bin_id': f"Bin {bin_id}",
                            'total_count': total_count,
                            'mean_target': mean_target,
                            'std_target': std_target,
                            'categories': categories
                        })
                        
            elif isinstance(bin_ranges, list) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in bin_ranges):
                # Numerical binning: bin_ranges is [(lower, upper), ...]
                for i, (lower, upper) in enumerate(bin_ranges):
                    # Create mask based on bin boundaries
                    if np.isinf(lower) and np.isinf(upper):
                        mask = pd.Series([True] * len(before_data), index=before_data.index)
                    elif np.isinf(lower):
                        mask = before_data[orig_col] <= upper
                    elif np.isinf(upper):
                        mask = before_data[orig_col] > lower
                    else:
                        mask = (before_data[orig_col] > lower) & (before_data[orig_col] <= upper)
                    
                    bin_data = before_data[mask]
                    
                    if len(bin_data) > 0:
                        total_count = len(bin_data)
                        mean_target = bin_data[self.builder.target_column].mean()
                        std_target = bin_data[self.builder.target_column].std()
                        
                        # Format range string
                        if np.isinf(lower) and np.isinf(upper):
                            range_str = "All values"
                        elif np.isinf(lower):
                            range_str = f"≤ {upper:.2f}"
                        elif np.isinf(upper):
                            range_str = f"> {lower:.2f}"
                        else:
                            range_str = f"({lower:.2f}, {upper:.2f}]"
                        
                        bin_stats.append({
                            'bin_id': f"Bin {i}",
                            'total_count': total_count,
                            'mean_target': mean_target,
                            'std_target': std_target,
                            'range': range_str
                        })
            
            if not bin_stats:
                st.warning("Could not calculate mean target values from bin ranges.")
                return
            
            # Create the plot
            fig = go.Figure()
            
            # Prepare data for plotting
            bin_labels = [stat['bin_id'] for stat in bin_stats]
            mean_values = [stat['mean_target'] for stat in bin_stats]
            
            # Prepare custom data for hover
            hover_data = []
            for stat in bin_stats:
                hover_data.append([
                    stat['total_count'],
                    stat['std_target']
                ])
            
            fig.add_trace(go.Bar(
                x=bin_labels,
                y=mean_values,
                name='Mean Target Value',
                marker_color='lightgreen',
                hovertemplate='<b>%{x}</b><br>' +
                             '<b>Mean Target</b>: %{y:.3f}<br>' +
                             '<b>Count</b>: %{customdata[0]}<br>' +
                             '<b>Std Dev</b>: %{customdata[1]:.3f}<br>' +
                             '<extra></extra>',
                customdata=hover_data
            ))
            
            # Add overall mean line
            overall_mean = before_data[self.builder.target_column].mean()
            fig.add_hline(
                y=overall_mean,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Overall Mean: {overall_mean:.3f}",
                annotation_position="top right"
            )
            
            fig.update_layout(
                title=f"Mean Target Value by Bin - {orig_col}",
                xaxis_title="Bins",
                yaxis_title="Mean Target Value",
                height=400,
                showlegend=True,
                hovermode='closest'
            )
            
            st.info("📈 This chart shows how the mean target value varies across different bins. Bins with significantly different means indicate predictive power.")
            st.plotly_chart(fig, config={'responsive': True})
            
            # Store event rate data for the bin details table (even for continuous targets)
            if not hasattr(st.session_state, 'event_rate_data'):
                st.session_state.event_rate_data = {}
            st.session_state.event_rate_data[orig_col] = bin_stats
                
        except Exception as e:
            st.warning(f"Error creating mean value visualization: {str(e)}")
    
    def _display_test_dataset_summary(self):
        """Display summary information about the test dataset after binning"""
        st.write("---")
        st.subheader("Test Dataset")

        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)

        # Log key metrics about the final dataset
        metrics = {
            "total_rows": int(len(self.builder.testing_data)),
            "total_columns": int(len(self.builder.testing_data.columns)),
            "memory_usage_mb": float(self.builder.testing_data.memory_usage().sum() / 1024 / 1024),
            "missing_values": int(self.builder.testing_data.isnull().sum().sum())
        }
        self.logger.log_calculation("Testing Dataset Metrics", metrics)

        with col1:
            st.metric(
                "Training Rows",
                f"{metrics['total_rows']:,}",
                delta=None
            )
        with col2:
            st.metric(
                "Total Columns",
                f"{metrics['total_columns']:,}",
                delta=None
            )
        with col3:
            memory_usage = float(self.builder.training_data.memory_usage().sum() / 1024 / 1024)
            st.metric(
                "Memory Usage",
                f"{memory_usage:.2f} MB",
                delta=None
            )
        with col4:
            missing_values = int(self.builder.training_data.isnull().sum().sum())
            st.metric(
                "Missing Values",
                f"{missing_values:,}",
                delta=None
            )

        st.dataframe(self.builder.testing_data.style.background_gradient(cmap='Blues'), width='stretch')

    # ... existing code ... 