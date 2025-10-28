import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from typing import Dict, Any, Optional

def get_visualisation_info():
    """Returns information about visualisations and metrics used in the feature analysis."""
    with st.expander("ℹ️ Understanding the visualisations and Metrics", expanded=False):
        st.markdown("""
        ### 📊 Visualization Types
        
        #### Feature Distribution (Left Panel)
        - **Numeric Features**: 
            - Histogram showing value distribution
            - KDE (Kernel Density Estimation) curve showing smoothed distribution
        - **Categorical Features**: 
            - Bar chart showing frequency of each category
        
        #### What is KDE (Kernel Density Estimation)?
        - KDE is used to create a smooth curve that represents the distribution of data points. It's particularly useful when you want to understand the underlying distribution of data without assuming a specific model.           
        - **How it Works**:
            - Imagine you have a set of data points plotted on a number line.
            - For each data point, KDE places a small, smooth "bump" or "kernel" (usually a Gaussian or bell-shaped curve) centered at that point.
            - These bumps are then added together to create a smooth curve that represents the overall distribution of the data.
        
        #### Feature-Target Relationship (Right Panel)
        
        For Classification Problems (Binary and Multi-class):
        - **Numeric Feature**: 
            - Box plot showing distribution across target classes
            - Violin plot showing density distribution
        - **Categorical Feature**: 
            - Stacked bar chart showing class distribution within categories
        
        For Regression Problems:
        - **Numeric Feature**: 
            - Scatter plot with trend line
        - **Categorical Feature**: 
            - Box plot showing target distribution per category
        
        ### 📈 Statistical Metrics
        
        #### Distribution Statistics
        - **Mean**: Average value (numeric features)
        - **Median**: Middle value (numeric features)
        - **Standard Deviation**: Measure of spread (numeric features)
        - **Unique Values**: Number of distinct values
        - **Missing Values**: Count of null/NA values
        
        #### Relationship Metrics
        For Classification (Binary and Multi-class):
        - **Numeric Features**: ANOVA F-test (compares means across classes)
        - **Categorical Features**: Chi-square test (tests independence between variables)
        
        For Regression:
        - **Numeric Features**: Correlation coefficient
        - **Categorical Features**: F-statistic
        
        #### Multi-class Classification Notes
        - **ANOVA F-test**: Tests if feature means differ significantly across multiple classes (3+ classes)
        - **Chi-square test**: Tests if categorical features are independent of class membership across all classes
        - **Visualizations**: Box plots and stacked bar charts automatically adapt to show all classes with different colors
        - **Class separation**: Look for features that clearly separate different classes in the visualizations
        - **Large number of classes**: If you have >20 classes, consider grouping similar classes for better visualization
        - **Macro averaging**: Metrics are calculated as averages across all classes to give equal weight to each class
        """)

def analyse_feature_distribution(
    data: pd.DataFrame,
    selected_feature: str,
    logger: Optional[Any] = None
) -> tuple[Dict[str, Any], go.Figure]:
    """
    Analyse and visualize the distribution of a selected feature.
    
    Args:
        data: DataFrame containing the feature
        selected_feature: Name of the feature to analyse
        logger: Optional logger object for logging calculations
        
    Returns:
        Tuple of (statistics dictionary, plotly figure)
    """
    # Calculate distribution statistics
    stats = {
        "mean": data[selected_feature].mean() if pd.api.types.is_numeric_dtype(data[selected_feature]) else None,
        "median": data[selected_feature].median() if pd.api.types.is_numeric_dtype(data[selected_feature]) else None,
        "std": data[selected_feature].std() if pd.api.types.is_numeric_dtype(data[selected_feature]) else None,
        "unique_values": data[selected_feature].nunique(),
        "missing_values": data[selected_feature].isna().sum()
    }
    
    # Log distribution analysis if logger provided
    if logger:
        logger.log_calculation(
            f"Feature Distribution Analysis - {selected_feature}",
            stats
        )
    
    # Create distribution visualization
    dist_fig = go.Figure()
    if pd.api.types.is_numeric_dtype(data[selected_feature]):
        # Add histogram for numeric features
        dist_fig.add_trace(go.Histogram(
            x=data[selected_feature],
            name=selected_feature,
            opacity=0.7,
            nbinsx=30,
            showlegend=False
        ))
        # Add KDE curve
        try:
            kde_x = np.linspace(data[selected_feature].min(), 
                              data[selected_feature].max(), 100)
            kde = gaussian_kde(data[selected_feature].dropna())
            kde_y = kde(kde_x)
            
            # Scale KDE to match histogram height
            hist, bins = np.histogram(data[selected_feature].dropna(), bins=30)
            bin_width = bins[1] - bins[0]
            kde_scale = len(data[selected_feature].dropna()) * bin_width
            
            dist_fig.add_trace(go.Scatter(
                x=kde_x,
                y=kde_y * kde_scale,
                name='KDE',
                line=dict(color='red', width=2),
                mode='lines'
            ))
            
            # Add mean line
            if stats["mean"] is not None:
                mean_value = stats["mean"]
                dist_fig.add_vline(
                    x=mean_value,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Mean: {mean_value:.2f}",
                    annotation_position="top right"
                )
            
            # Add median line
            if stats["median"] is not None:
                median_value = stats["median"]
                dist_fig.add_vline(
                    x=median_value,
                    line_dash="dash",
                    line_color="blue",
                    annotation_text=f"Median: {median_value:.2f}",
                    annotation_position="top left"
                )
                
        except Exception as e:
            st.warning(f"Could not add KDE curve: {str(e)}")
    else:
        # Add bar chart for categorical features
        value_counts = data[selected_feature].value_counts()
        dist_fig.add_trace(go.Bar(
            x=value_counts.index,
            y=value_counts.values,
            name=selected_feature,
            opacity=0.7,
            showlegend=False
        ))
    
    dist_fig.update_layout(
        title=f'Distribution of {selected_feature}',
        xaxis_title=selected_feature,
        yaxis_title='Count',
        height=400
    )
    
    return stats, dist_fig

def analyse_feature_target_relationship(
    data: pd.DataFrame,
    selected_feature: str,
    target_column: str,
    is_classification: bool,
    logger: Optional[Any] = None
) -> tuple[Dict[str, Any], go.Figure]:
    """
    Analyse and visualize the relationship between a feature and the target variable.
    
    Args:
        data: DataFrame containing the feature and target
        selected_feature: Name of the feature to analyse
        target_column: Name of the target column
        is_classification: Whether the problem is classification
        logger: Optional logger object for logging calculations
        
    Returns:
        Tuple of (relationship metrics dictionary, plotly figure)
    """
    relationship_metrics = {}
    
    # Drop any rows where either the feature or target is null
    valid_data = data[[selected_feature, target_column]].dropna()
    
    if pd.api.types.is_numeric_dtype(data[selected_feature]):
        if is_classification:
            # Box plot for numeric feature vs categorical target
            # Handle cases where target might be numeric (encoded classes)
            target_display = valid_data[target_column].astype(str) if pd.api.types.is_numeric_dtype(valid_data[target_column]) else valid_data[target_column]
            
            target_fig = px.box(
                valid_data.assign(**{f'{target_column}_str': target_display}),
                x=f'{target_column}_str',
                y=selected_feature,
                title=f'{selected_feature} Distribution by {target_column}'
            )
            
            # Add violin plot
            target_fig.add_traces(
                px.violin(
                    valid_data.assign(**{f'{target_column}_str': target_display}),
                    x=f'{target_column}_str',
                    y=selected_feature
                ).data
            )
            
            # ANOVA F-test for numeric feature vs categorical target
            try:
                # Group data and ensure we have at least 2 groups with more than 1 sample each
                groups = [group for name, group in valid_data[selected_feature].groupby(valid_data[target_column])
                         if len(group) > 1]
                
                if len(groups) >= 2:  # Need at least 2 groups for ANOVA
                    f_stat, p_value = stats.f_oneway(*groups)
                    if not np.isnan(f_stat) and not np.isnan(p_value):
                        relationship_metrics["ANOVA F-statistic"] = f"{f_stat:.3f}"
                        relationship_metrics["p-value"] = f"{p_value:.3f}"
                        if p_value < 0.05:
                            relationship_metrics["Significance"] = "Significant (p < 0.05)"
                        else:
                            relationship_metrics["Significance"] = "Not significant (p ≥ 0.05)"
                            
                        # Get statistical explanation from builder
                        if 'builder' in st.session_state:
                            explanation = st.session_state.builder.get_statistical_explanation(
                                "anova",
                                {"statistic": f_stat, "p_value": p_value}
                            )
                            relationship_metrics["explanation"] = explanation
                else:
                    relationship_metrics["Note"] = "Insufficient data for ANOVA test"
            except Exception as e:
                relationship_metrics["Note"] = f"Could not perform ANOVA test: {str(e)}"
        else:
            # Create scatter plot for numeric feature vs numeric target
            target_fig = go.Figure()
            target_fig.add_trace(go.Scatter(
                x=valid_data[selected_feature],
                y=valid_data[target_column],
                mode='markers',
                name='Data points'
            ))
            
            # Add trend line if we have enough points
            if len(valid_data) >= 2:
                try:
                    from scipy import stats as scipy_stats
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                        valid_data[selected_feature], 
                        valid_data[target_column]
                    )
                    x_range = np.linspace(valid_data[selected_feature].min(), valid_data[selected_feature].max(), 100)
                    y_range = slope * x_range + intercept
                    target_fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_range,
                        mode='lines',
                        name='Trend line',
                        line=dict(color='red')
                    ))
                    
                    # Add correlation statistics
                    correlation = r_value
                    if not np.isnan(correlation):
                        relationship_metrics["Correlation"] = f"{correlation:.3f}"
                        relationship_metrics["R-squared"] = f"{r_value**2:.3f}"
                        relationship_metrics["p-value"] = f"{p_value:.3f}"
                        if abs(correlation) > 0.7:
                            relationship_metrics["Strength"] = "Strong"
                        elif abs(correlation) > 0.3:
                            relationship_metrics["Strength"] = "Moderate"
                        else:
                            relationship_metrics["Strength"] = "Weak"
                        if p_value < 0.05:
                            relationship_metrics["Significance"] = "Significant (p < 0.05)"
                        else:
                            relationship_metrics["Significance"] = "Not significant (p ≥ 0.05)"
                            
                        # Get statistical explanation from builder
                        if 'builder' in st.session_state:
                            explanation = st.session_state.builder.get_statistical_explanation(
                                "pearson",
                                {"statistic": correlation, "p_value": p_value}
                            )
                            relationship_metrics["explanation"] = explanation
                except Exception as e:
                    relationship_metrics["Note"] = f"Could not calculate trend line: {str(e)}"
            else:
                relationship_metrics["Note"] = "Insufficient data for correlation"
            
            target_fig.update_layout(
                title=f'{selected_feature} vs {target_column}',
                xaxis_title=selected_feature,
                yaxis_title=target_column
            )
    else:
        if is_classification:
            # Stacked bar chart for categorical feature vs categorical target
            # Handle cases where target might be numeric (encoded classes)
            target_display = valid_data[target_column].astype(str) if pd.api.types.is_numeric_dtype(valid_data[target_column]) else valid_data[target_column]
            
            target_fig = px.histogram(
                valid_data.assign(**{f'{target_column}_str': target_display}),
                x=selected_feature,
                color=f'{target_column}_str',
                title=f'{selected_feature} Distribution by {target_column}',
                barmode='stack'
            )
            
            # Chi-square test for categorical feature vs categorical target
            try:
                # Use string-converted target for consistency
                target_for_test = valid_data[target_column].astype(str) if pd.api.types.is_numeric_dtype(valid_data[target_column]) else valid_data[target_column]
                contingency_table = pd.crosstab(
                    valid_data[selected_feature],
                    target_for_test
                )
                
                # Check if we have enough data for chi-square test
                if (contingency_table.shape[0] > 1 and 
                    contingency_table.shape[1] > 1 and 
                    (contingency_table > 5).all().all()):  # Expected frequencies should be > 5
                    
                    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
                    if not np.isnan(chi2) and not np.isnan(p_value):
                        relationship_metrics["Chi-square statistic"] = f"{chi2:.3f}"
                        relationship_metrics["p-value"] = f"{p_value:.3f}"
                        if p_value < 0.05:
                            relationship_metrics["Significance"] = "Significant (p < 0.05)"
                        else:
                            relationship_metrics["Significance"] = "Not significant (p ≥ 0.05)"
                            
                        # Get statistical explanation from builder
                        if 'builder' in st.session_state:
                            explanation = st.session_state.builder.get_statistical_explanation(
                                "chi2",
                                {"chi2": chi2, "p_value": p_value}
                            )
                            relationship_metrics["explanation"] = explanation
                else:
                    relationship_metrics["Note"] = "Insufficient data for chi-square test (need >5 samples per category)"
            except Exception as e:
                relationship_metrics["Note"] = f"Could not perform chi-square test: {str(e)}"
        else:
            # Box plot for categorical feature vs numeric target
            target_fig = px.box(
                valid_data,
                x=selected_feature,
                y=target_column,
                title=f'{target_column} Distribution by {selected_feature}'
            )
            
            # F-test for categorical feature vs numeric target
            try:
                # Group data and ensure we have at least 2 groups with more than 1 sample each
                groups = [group for name, group in valid_data[target_column].groupby(valid_data[selected_feature])
                         if len(group) > 1]
                
                if len(groups) >= 2:  # Need at least 2 groups for F-test
                    f_stat, p_value = stats.f_oneway(*groups)
                    if not np.isnan(f_stat) and not np.isnan(p_value):
                        relationship_metrics["F-statistic"] = f"{f_stat:.3f}"
                        relationship_metrics["p-value"] = f"{p_value:.3f}"
                        if p_value < 0.05:
                            relationship_metrics["Significance"] = "Significant (p < 0.05)"
                        else:
                            relationship_metrics["Significance"] = "Not significant (p ≥ 0.05)"
                            
                        # Get statistical explanation from builder
                        if 'builder' in st.session_state:
                            explanation = st.session_state.builder.get_statistical_explanation(
                                "anova",  # F-test is a type of ANOVA
                                {"statistic": f_stat, "p_value": p_value}
                            )
                            relationship_metrics["explanation"] = explanation
                else:
                    relationship_metrics["Note"] = "Insufficient data for F-test"
            except Exception as e:
                relationship_metrics["Note"] = f"Could not perform F-test: {str(e)}"
    
    target_fig.update_layout(height=400)
    
    # Log relationship analysis if logger provided
    if logger:
        logger.log_calculation(
            f"Feature-Target Relationship Analysis - {selected_feature}",
            relationship_metrics
        )
    
    return relationship_metrics, target_fig

def show_feature_analysis(
    data: pd.DataFrame,
    target_column: str,
    selected_feature: str = None,
    logger: Optional[Any] = None,
    feature_analysis_key: str = "default_feature_analysis"
) -> None:
    """
    Display comprehensive feature analysis including distribution and target relationship.
    
    Args:
        data: DataFrame containing the features and target
        target_column: Name of the target column
        selected_feature: Optional pre-selected feature to analyse
        logger: Optional logger object for logging calculations
        feature_analysis_key: Optional key for the feature analysis UI elements
    """
    # Use session state variables for problem type detection
    if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type:
        problem_type = st.session_state.problem_type
        is_binary = getattr(st.session_state, 'is_binary', False)
        is_multiclass = getattr(st.session_state, 'is_multiclass', False)
        is_regression = getattr(st.session_state, 'is_regression', False)
    else:
        # Fallback to heuristic detection for backward compatibility
        y = data[target_column]
        is_target_numeric = pd.api.types.is_numeric_dtype(y)
        
        # Check for binary classification (2 unique values)
        is_binary = y.nunique() <= 2
        
        # Check for multi-class classification
        is_multiclass = False
        if 'encoding_mappings' in st.session_state and target_column in st.session_state.encoding_mappings:
            # If we have encoding mapping, it was treated as classification in data loading
            is_multiclass = y.nunique() > 2
        elif is_target_numeric and 3 <= y.nunique() <= 20:
            # Check if this might be multiclass by looking for integer-like values
            try:
                y_no_nan = y.dropna()
                if len(y_no_nan) > 0:
                    is_integer_like = np.allclose(y_no_nan, np.round(y_no_nan), atol=1e-10)
                    consecutive_integers = (y_no_nan.min() >= 0 and 
                                          set(y_no_nan.unique()) == set(range(int(y_no_nan.min()), int(y_no_nan.max()) + 1)))
                    is_multiclass = is_integer_like and consecutive_integers
            except:
                is_multiclass = False
        
        is_regression = not (is_binary or is_multiclass)
        
        # Set problem_type for consistency
        if is_binary:
            problem_type = "binary_classification"
        elif is_multiclass:
            problem_type = "multiclass_classification"
        else:
            problem_type = "regression"
    
    # Determine if this is any kind of classification problem
    is_classification = is_binary or is_multiclass
    
    # Get number of classes for information display
    num_classes = data[target_column].nunique()
    
    # Add information about the problem type detected
    if is_multiclass and num_classes > 20:
        st.warning(f"⚠️ Your target variable has {num_classes} unique classes (multiclass classification). Visualizations may be difficult to interpret with this many classes. Consider grouping similar classes or using dimensionality reduction techniques.")
    elif is_multiclass:
        st.info(f"📊 **Multiclass Classification** detected: {num_classes} classes in target variable '{target_column}'")
    elif is_binary:
        st.info(f"📊 **Binary Classification** detected: 2 classes in target variable '{target_column}'")
    else:
        st.info(f"📊 **Regression** detected: Continuous target variable '{target_column}'")
    
    if selected_feature is None:
        # Select a feature to visualize
        selected_feature = st.selectbox(
            "Select a feature to analyse:",
            options=list(data.columns.drop(target_column)),
            help="Choose a feature to see its distribution and relationship with the target variable",
            key=f"{feature_analysis_key}_selectbox"
        )
     
    if selected_feature:
        # Create two columns for the visualisations
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            st.write("📊 Feature Distribution")
            stats, dist_fig = analyse_feature_distribution(data, selected_feature, logger)
            st.plotly_chart(dist_fig, config={'responsive': True})
            
            # Display statistics
            st.write("📈 Statistics:")
            stats_df = pd.DataFrame([stats]).T
            stats_df.columns = ['Value']
            stats_df = stats_df[stats_df['Value'].notna()]
            st.dataframe(stats_df, width='stretch')
        
        with dist_col2:
            st.write("🎯 Relationship with Target")
            metrics, target_fig = analyse_feature_target_relationship(
                data, selected_feature, target_column, is_classification, logger
            )
            st.plotly_chart(target_fig, config={'responsive': True})
            
            # Display relationship metrics
            st.write("📊 Relationship Metrics:")
            # Create a dictionary of metrics excluding the explanation
            display_metrics = {k: [v] for k, v in metrics.items() if k != 'explanation'}
            if display_metrics:  # Only create DataFrame if there are metrics to display
                metrics_df = pd.DataFrame(display_metrics).T
                metrics_df.columns = ['Value']
                st.dataframe(metrics_df, width='stretch')
            
            # Display statistical test explanation if available
            if 'explanation' in metrics:
                with st.expander("ℹ️ Understanding the Statistical Test"):
                    st.markdown(metrics['explanation']['method'])
                    st.markdown("**Interpretation of Results:**")
                    st.markdown(metrics['explanation']['interpretation'])
        
        # Log the problem type detection for debugging
        if logger:
            logger.log_calculation(
                "Problem Type Detection in Feature Analysis",
                {
                    "selected_feature": selected_feature,
                    "problem_type": problem_type,
                    "is_binary": is_binary,
                    "is_multiclass": is_multiclass,
                    "is_regression": is_regression,
                    "is_classification": is_classification,
                    "num_classes": num_classes,
                    "used_session_state": hasattr(st.session_state, 'problem_type') and st.session_state.problem_type is not None
                }
            )