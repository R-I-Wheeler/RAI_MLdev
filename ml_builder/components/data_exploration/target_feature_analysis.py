import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, pointbiserialr, f_oneway, chi2_contingency
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import plotly.express as px
import plotly.graph_objects as go

def display_target_distribution(data: pd.DataFrame, target_column: str) -> None:
    """
    Display the distribution of the target variable.
    
    Parameters:
    -----------
    data : pandas DataFrame
        The dataset containing the target
    target_column : str
        The name of the target column
    """
    if target_column not in data.columns:
        st.error(f"Target column '{target_column}' not found in the dataset.")
        return
    
    st.write("### Target Distribution Analysis")
    
    # Get target data
    y = data[target_column]
    
    # Always determine if target is numeric (needed for visualization logic)
    is_target_numeric = pd.api.types.is_numeric_dtype(y)
    
    # Use session state variables for problem type detection
    if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type:
        problem_type = st.session_state.problem_type
        is_binary_classification = getattr(st.session_state, 'is_binary', False)
        is_multiclass_classification = getattr(st.session_state, 'is_multiclass', False)
        is_regression = getattr(st.session_state, 'is_regression', False)
    else:
        # Fallback to heuristic detection for backward compatibility
        # Determine if target is binary (has only 2 unique values)
        is_binary_classification = y.nunique() <= 2
        
        # Determine if target is multi-class (has 3+ unique values but should be treated as classification)
        # We can detect this by checking if we have encoding mapping information or if target has reasonable number of classes
        is_multiclass_classification = False
        if 'encoding_mappings' in st.session_state and target_column in st.session_state.encoding_mappings:
            # If we have encoding mapping, it was treated as classification in data loading
            is_multiclass_classification = y.nunique() > 2
        elif is_target_numeric and 3 <= y.nunique() <= 20:
            # Check if this might be multiclass by looking for integer-like values
            try:
                # If all values are close to integers, it might be encoded multiclass
                y_no_nan = y.dropna()
                if len(y_no_nan) > 0:
                    is_integer_like = np.allclose(y_no_nan, np.round(y_no_nan), atol=1e-10)
                    consecutive_integers = (y_no_nan.min() >= 0 and 
                                          set(y_no_nan.unique()) == set(range(int(y_no_nan.min()), int(y_no_nan.max()) + 1)))
                    is_multiclass_classification = is_integer_like and consecutive_integers
            except:
                is_multiclass_classification = False
        
        is_regression = not (is_binary_classification or is_multiclass_classification)
    
    # Display appropriate visualization based on target type
    if is_target_numeric and not is_binary_classification and not is_multiclass_classification:
        # Numeric target (regression case)
        fig = px.histogram(
            data, 
            x=target_column,
            nbins=min(50, int(len(data) / 10)),
            marginal="box",
            title=f"Distribution of {target_column} (Regression Target)",
            opacity=0.7,
            color_discrete_sequence=['#3366CC']
        )
        fig.update_layout(
            xaxis_title=target_column,
            yaxis_title="Count",
            bargap=0.05,
            height=500
        )
        # Add mean and median lines
        mean_val = y.mean()
        median_val = y.median()
        fig.add_shape(
            type="line",
            x0=mean_val, x1=mean_val,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
        )
        fig.add_shape(
            type="line",
            x0=median_val, x1=median_val,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="green", width=2, dash="dot"),
        )
        # Add annotations for mean and median
        fig.add_annotation(
            x=mean_val,
            y=1,
            yref="paper",
            text=f"Mean: {mean_val:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=40,
            ay=-40,
            font=dict(color="red"),
        )
        fig.add_annotation(
            x=median_val,
            y=0.9,
            yref="paper",
            text=f"Median: {median_val:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=-40,
            ay=-40,
            font=dict(color="green"),
        )
        
        st.plotly_chart(fig, config={'responsive': True}, key="target_regression_plot")

        # Add explanation for the visualization
        with st.expander("📊 How to Interpret This Visualization", expanded=False):
            st.markdown("""
            #### Understanding the Regression Target Distribution
            
            This histogram shows how your target variable values are distributed across your dataset:
            
            - **Histogram bars**: Represent the frequency (count) of observations within each value range
            - **Red dashed line**: Mean value (average of all observations)
            - **Green dotted line**: Median value (middle point when all values are sorted)
            - **Box plot** (top): Shows the quartiles and identifies potential outliers
                - Middle line: Median
                - Box edges: 25th and 75th percentiles
                - Whiskers: Typically extend to 1.5x the interquartile range
                - Points beyond whiskers: Potential outliers
            
            #### What to Look For
            
            1. **Shape**: Is the distribution normal (bell-shaped), skewed, or multi-modal (multiple peaks)?
            2. **Skewness**: If mean ≠ median, the distribution is skewed
                - Mean > Median: Right-skewed (positive skew)
                - Mean < Median: Left-skewed (negative skew)
            3. **Spread**: Wide distributions indicate high variability in your target
            4. **Outliers**: Extreme values that may influence model performance
            
            #### Why This Matters for Modeling
    
            - **Outliers** might need special handling (capping, removal, etc.)
            - **Multi-modal distributions** could indicate subpopulations that might benefit from separate models
            - **Understanding central tendency** helps set realistic expectations for model performance
            """)
        
        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{mean_val:.2f}")
        with col2:
            st.metric("Median", f"{median_val:.2f}")
        with col3:
            st.metric("Min", f"{y.min():.2f}")
        with col4:
            st.metric("Max", f"{y.max():.2f}")
        
        # Add insights about the distribution
        skewness = y.skew()
        kurtosis = y.kurtosis()
        
        insights = []
        if abs(skewness) > 1:
            direction = "right" if skewness > 0 else "left"
            insights.append(f"• The distribution is **highly skewed** to the {direction} (skewness = {skewness:.2f}).")
            
        elif abs(skewness) > 0.5:
            direction = "right" if skewness > 0 else "left"
            insights.append(f"• The distribution is **moderately skewed** to the {direction} (skewness = {skewness:.2f}).")
        else:
            insights.append(f"• The distribution is approximately **symmetric** (skewness = {skewness:.2f}).")
        
        if abs(mean_val - median_val) > (y.max() - y.min()) * 0.1:
            insights.append(f"• There is a **notable difference** between mean and median, further indicating skewness.")
        
        if kurtosis > 1:
            insights.append(f"• The distribution has **heavy tails** (kurtosis = {kurtosis:.2f}), indicating potential outliers.")
        elif kurtosis < -1:
            insights.append(f"• The distribution has **light tails** (kurtosis = {kurtosis:.2f}), indicating a narrow spread of values.")
        
        if insights:
            st.write("#### Distribution Insights:")
            st.markdown("\n".join(insights))
            
    elif is_binary_classification:
        # Binary classification case
        value_counts = y.value_counts().sort_index()
        class_names = value_counts.index.tolist()
        class_counts = value_counts.values
        
        # Create a bar chart for binary classification
        fig = px.bar(
            x=class_names,
            y=class_counts,
            text=class_counts,
            title=f"Distribution of {target_column} (Binary Classification Target)",
            color=class_names,
            color_discrete_sequence=['#3366CC', '#DC3912'],
            opacity=0.8
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(
            xaxis_title=target_column,
            yaxis_title="Count",
            bargap=0.3,
            height=500,
            uniformtext_minsize=10,
            uniformtext_mode='hide'
        )
        
        st.plotly_chart(fig, config={'responsive': True}, key="target_binary_plot")
        
        # Add explanation for binary classification visualization
        with st.expander("📊 How to Interpret This Visualization", expanded=False):
            st.markdown("""
            #### Understanding the Binary Classification Target Distribution
            
            This bar chart shows the distribution of your target classes:
            
            - **Bars**: Represent the count of observations in each class
            - **Numbers at top**: Exact count of observations in each class
            
            #### What to Look For
            
            1. **Class Balance**: Are the classes evenly distributed or imbalanced?
            2. **Imbalance Ratio**: The ratio between the majority and minority class
            3. **Class Prevalence**: The proportion of the positive class in the dataset
            
            #### Why This Matters for Modeling
            
            - **Balanced classes** (ratio near 1:1): Standard classification metrics and algorithms work well
            - **Moderate imbalance** (ratio 3:1 to 10:1): Consider using class weights or stratified sampling
            - **Severe imbalance** (ratio > 10:1): Consider:
                - Resampling techniques (SMOTE, under/oversampling)
                - Anomaly detection approaches instead of classification
                - Specialized metrics (F1, precision-recall AUC instead of accuracy)
                - Adjusted probability thresholds
            
            - **Domain knowledge** is critical - sometimes the rare class is the most important one!
            """)
        
        # Display class balance metrics
        total = sum(class_counts)
        class_percentages = [count / total * 100 for count in class_counts]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Class Balance Ratio", f"1:{class_counts[1]/class_counts[0]:.2f}" if class_counts[0] > 0 else "∞")
        with col2:
            imbalance_ratio = max(class_counts) / min(class_counts) if min(class_counts) > 0 else float('inf')
            st.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}:1")
        
        # Add insights about class imbalance
        insights = []
        if imbalance_ratio > 10:
            insights.append(f"• The classes are **highly imbalanced** (ratio = {imbalance_ratio:.2f}:1).")
            insights.append(f"• Consider using techniques like SMOTE, class weights, or stratified sampling to handle this imbalance.")
        elif imbalance_ratio > 3:
            insights.append(f"• The classes are **moderately imbalanced** (ratio = {imbalance_ratio:.2f}:1).")
            insights.append(f"• Consider using class weights or stratified sampling during model training.")
        else:
            insights.append(f"• The classes are **relatively balanced** (ratio = {imbalance_ratio:.2f}:1).")
            
        if insights:
            st.write("#### Class Balance Insights:")
            st.markdown("\n".join(insights))
            
    elif is_multiclass_classification:
        # Multi-class classification case
        value_counts = y.value_counts().sort_index()
        class_names = value_counts.index.tolist()
        class_counts = value_counts.values
        
        # Get original class names if encoding mapping exists
        display_names = class_names
        if 'encoding_mappings' in st.session_state and target_column in st.session_state.encoding_mappings:
            encoding_info = st.session_state.encoding_mappings[target_column]
            if 'mapping' in encoding_info:
                # Create reverse mapping from encoded values to original names
                reverse_mapping = {v: k for k, v in encoding_info['mapping'].items()}
                display_names = [reverse_mapping.get(name, name) for name in class_names]
        
        # Create a bar chart for multi-class classification
        fig = px.bar(
            x=display_names,
            y=class_counts,
            text=class_counts,
            title=f"Distribution of {target_column} (Multi-class Classification Target)",
            color=display_names,
            color_discrete_sequence=px.colors.qualitative.Set3[:len(class_names)],
            opacity=0.8
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(
            xaxis_title=target_column,
            yaxis_title="Count",
            bargap=0.3,
            height=500,
            uniformtext_minsize=10,
            uniformtext_mode='hide'
        )
        
        st.plotly_chart(fig, config={'responsive': True}, key="target_multiclass_plot")
        
        # Add explanation for multi-class classification visualization
        with st.expander("📊 How to Interpret This Visualization", expanded=False):
            st.markdown("""
            #### Understanding the Multi-class Classification Target Distribution
            
            This bar chart shows the distribution of your target classes:
            
            - **Bars**: Represent the count of observations in each class
            - **Numbers at top**: Exact count of observations in each class
            - **Colors**: Different colors help distinguish between the classes
            
            #### What to Look For
            
            1. **Class Balance**: Are all classes evenly represented or are some dominant?
            2. **Class Distribution**: Which classes are most/least common in your dataset?
            3. **Minority Classes**: Very small classes might be challenging to predict accurately
            
            #### Why This Matters for Modeling
            
            - **Balanced classes**: All classes well-represented - standard algorithms work well
            - **Moderate imbalance**: Consider class weights or stratified sampling
            - **Severe imbalance**: Consider:
                - Resampling techniques (SMOTE, stratified sampling)
                - Cost-sensitive learning with class weights
                - Ensemble methods that handle imbalance well
                - Specialized multi-class metrics (macro/micro F1, balanced accuracy)
            
            - **Very small classes** (< 5% of data): May need special attention or could be combined with similar classes
            - **Domain knowledge** is crucial for understanding which classes are most important to predict correctly
            """)
        
        # Display class distribution metrics
        total = sum(class_counts)
        class_percentages = [count / total * 100 for count in class_counts]
        
        # Create a summary table of class distribution
        class_summary = pd.DataFrame({
            'Class': display_names,
            'Count': class_counts,
            'Percentage': [f"{pct:.1f}%" for pct in class_percentages]
        })
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Classes", len(class_names))
        with col2:
            largest_class_pct = max(class_percentages)
            st.metric("Largest Class", f"{largest_class_pct:.1f}%")
        with col3:
            smallest_class_pct = min(class_percentages)
            st.metric("Smallest Class", f"{smallest_class_pct:.1f}%")
        
        # Show class distribution table
        st.write("#### Class Distribution Summary:")
        st.dataframe(class_summary, width='stretch', hide_index=True)
        
        # Add insights about class distribution
        insights = []
        imbalance_ratio = max(class_counts) / min(class_counts) if min(class_counts) > 0 else float('inf')
        
        if imbalance_ratio > 10:
            insights.append(f"• The classes are **highly imbalanced** (ratio = {imbalance_ratio:.1f}:1 between largest and smallest class).")
            insights.append(f"• Consider using techniques like class weights, SMOTE, or stratified sampling.")
        elif imbalance_ratio > 3:
            insights.append(f"• The classes are **moderately imbalanced** (ratio = {imbalance_ratio:.1f}:1 between largest and smallest class).")
            insights.append(f"• Consider using class weights or stratified sampling during model training.")
        else:
            insights.append(f"• The classes are **relatively balanced** (ratio = {imbalance_ratio:.1f}:1 between largest and smallest class).")
        
        # Check for very small classes
        small_class_threshold = 5.0  # 5% threshold
        small_classes = [display_names[i] for i, pct in enumerate(class_percentages) if pct < small_class_threshold]
        if small_classes:
            insights.append(f"• **Small classes detected**: {', '.join(small_classes)} have less than {small_class_threshold}% of the data.")
            insights.append(f"• Small classes may be harder to predict accurately. Consider combining with similar classes if appropriate.")
        
        if insights:
            st.write("#### Class Distribution Insights:")
            st.markdown("\n".join(insights))
    
    # If none of the above cases apply, show a generic message
    else:
        st.warning("⚠️ Unable to determine the appropriate visualization for this target variable. Please verify your target variable selection.")

def analyse_feature_relationships(data: pd.DataFrame, target_column: str) -> None:
    """
    Analyse and display feature-target relationships in the data.
    
    Parameters:
    -----------
    data : pandas DataFrame
        The dataset containing both features and target
    target_column : str
        The name of the target column
    """
    
    # Run relationship type detection
    with st.spinner("Analyzing feature-target relationships..."):
        try:
            relationship_df = detect_relationship_type(data, target_column)
            
            if relationship_df.empty:
                st.info("No numerical features found for relationship analysis.")
            else:
                # Create a styled dataframe with metric columns
                st.write("### Feature-Target Relationships")
                
                # Add user experience level selector
                st.write("#### Display Options")
                show_advanced_metrics = st.toggle(
                    "Show Advanced Statistical Metrics",
                    value=False,
                    help="Toggle on to see detailed statistical metrics and technical explanations. Toggle off for simplified view with essential information only."
                )
                
                # Set user level based on toggle
                user_level = "Advanced" if show_advanced_metrics else "Beginner"
                
                # Add user-level specific instructions
                if user_level == "Beginner":
                    st.info("💡 **Simplified Mode**: Showing essential information only. Toggle on 'Show Advanced Statistical Metrics' above to see detailed analysis.")
                else:
                    st.info("🔬 **Advanced Mode**: Showing all statistical metrics. Toggle off 'Show Advanced Statistical Metrics' above for a simplified view.")

                # Add explanation of the metrics
                with st.expander("ℹ️ Understanding Metrics and Relationships", expanded=False):
                    st.markdown("""
                    ### Understanding Metrics and Relationships
                    """)
                    
                    # Get actual columns that will be in the table
                    actual_columns = []
                    if relationship_df is not None and isinstance(relationship_df, pd.DataFrame) and not relationship_df.empty:
                        try:
                            # Get column names and renames
                            column_renames = {
                                'Pearson_Correlation': 'Pearson r',
                                'Pearson_P_Value': 'Pearson p-value',
                                'Spearman_Correlation': 'Spearman ρ',
                                'Spearman_P_Value': 'Spearman p-value',
                                'Mutual_Information': 'Mutual Info',
                                'ANOVA_F': 'ANOVA F',
                                'ANOVA_P_Value': 'ANOVA p-value',
                                'Normalized_MI': 'Norm. MI',
                                'Linear_Relationship': 'Linear?',
                                'Non_Linear_Relationship': 'Non-Linear?',
                                'Chi2': 'Chi-square',
                                'DOF': 'Degrees of Freedom',
                                'Chi2_P_Value': 'Chi-square p-value', 
                                'Eta_Squared': 'Eta Squared',
                                'Cramers_V': "Cramer's V",
                                'PointBiserial_Correlation': 'Point-Biserial r',
                                'PointBiserial_P_Value': 'Point-Biserial p-value',
                                'Significant_Difference': 'Significant Groups?',
                                'Significant_Association': 'Significant Association?'
                            }
                            
                            # Create a map of original column names to display names
                            rename_dict = {k: v for k, v in column_renames.items() if k in relationship_df.columns}
                            
                            # Track columns that actually exist in the table
                            for original_col, renamed_col in rename_dict.items():
                                if original_col in relationship_df.columns:
                                    actual_columns.append(renamed_col)
                            
                            # Always include these key columns
                            actual_columns.extend(['Feature', 'Relationship_Type', 'Strength'])
                            actual_columns = list(set(actual_columns))  # Remove duplicates
                        except:
                            # If we can't get column names, fall back to default descriptions
                            actual_columns = []
                    
                    if user_level == "Advanced":
                        # Show full metric explanations for advanced users
                        # Metric explanations
                        st.markdown("#### Metric Explanations:")
                        
                        metrics_desc = []
                        
                        # Only show descriptions for metrics that are actually in the table
                        if not actual_columns or 'Pearson r' in actual_columns:
                            metrics_desc.append("- **Pearson r**: Measures linear relationship strength (-1 to 1); values near -1 or 1 indicate strong linear relationships (for numeric features with numeric target)")
                        
                        if not actual_columns or 'Pearson p-value' in actual_columns:
                            metrics_desc.append("- **Pearson p-value**: Statistical significance of the Pearson correlation; values < 0.05 indicate statistically significant linear relationships")
                        
                        if not actual_columns or 'Spearman ρ' in actual_columns:
                            metrics_desc.append("- **Spearman ρ**: Measures monotonic relationship strength (-1 to 1); effective for finding non-linear but ordered relationships (for numeric features with numeric target)")
                        
                        if not actual_columns or 'Spearman p-value' in actual_columns:
                            metrics_desc.append("- **Spearman p-value**: Statistical significance of the Spearman correlation; values < 0.05 indicate statistically significant monotonic relationships")
                        
                        if not actual_columns or 'Point-Biserial r' in actual_columns or 'PointBiserial_Correlation' in relationship_df.columns:
                            metrics_desc.append("- **Point-Biserial r**: Special correlation coefficient between a numeric feature and a binary target (-1 to 1); used specifically for binary classification problems")
                        
                        if not actual_columns or 'Point-Biserial p-value' in actual_columns or 'PointBiserial_P_Value' in relationship_df.columns:
                            metrics_desc.append("- **Point-Biserial p-value**: Statistical significance of the Point-Biserial correlation; values < 0.05 indicate statistically significant relationships for binary targets")
                        
                        if not actual_columns or 'Mutual Info' in actual_columns or 'Norm. MI' in actual_columns:
                            metrics_desc.append("- **Mutual Info/Norm. MI**: Measures general dependence (0 to 1); can detect complex non-linear relationships for both regression and classification problems")
                        
                        if not actual_columns or 'ANOVA F' in actual_columns:
                            metrics_desc.append("- **ANOVA F**: Higher values indicate stronger differences between groups; important for classification problems to identify features that separate classes well")
                        
                        if not actual_columns or 'ANOVA p-value' in actual_columns:
                            metrics_desc.append("- **ANOVA p-value**: Statistical significance of the ANOVA test; values < 0.05 indicate significant differences between group means")
                        
                        if not actual_columns or 'Eta Squared' in actual_columns or 'Eta_Squared' in relationship_df.columns:
                            metrics_desc.append("- **Eta Squared**: Effect size measure (0 to 1); indicates how much variance in the target is explained by the feature; values above 0.13 suggest strong predictive power for classification")
                        
                        if not actual_columns or 'Chi-square' in actual_columns:
                            metrics_desc.append("- **Chi-square**: Tests relationship between categorical variables; higher values indicate stronger associations (only shown for categorical features with categorical target)")
                        
                        if not actual_columns or 'Chi-square p-value' in actual_columns:
                            metrics_desc.append("- **Chi-square p-value**: Statistical significance of the Chi-square test; values < 0.05 indicate significant association between categorical variables")
                        
                        if not actual_columns or 'DOF' in actual_columns or 'Degrees of Freedom' in actual_columns:
                            metrics_desc.append("- **Degrees of Freedom (DOF)**: The number of independent values that can vary in the analysis; for categorical variables, calculated as (rows-1) × (columns-1) in the contingency table; higher values indicate more complex relationships")
                        
                        if not actual_columns or "Cramer's V" in actual_columns:
                            metrics_desc.append("- **Cramer's V**: Standardized measure (0 to 1) of association between categorical variables (only shown for categorical features with categorical target)")
                        
                        if not actual_columns or 'Linear?' in actual_columns:
                            metrics_desc.append("- **Linear?**: Indicates if a significant linear relationship exists (Yes/No)")
                        
                        if not actual_columns or 'Non-Linear?' in actual_columns:
                            metrics_desc.append("- **Non-Linear?**: Indicates if a significant non-linear relationship exists (Yes/No)")
                        
                        # Add specific note about classification problems
                        metrics_desc.append("")
                        metrics_desc.append("*Classification Problems Note:*")
                        metrics_desc.append("* *For classification problems (binary or multi-class), specialized metrics are used:*")
                        metrics_desc.append("* *For **Numeric features with classification target**: Point-Biserial Correlation (binary only), ANOVA F, Eta Squared, and Normalized MI*")
                        metrics_desc.append("* *For **Categorical features with classification target**: Chi-square and Cramer's V*")
                        metrics_desc.append("* *The **Class Separation** relationship type indicates features that effectively separate your classes*")
                        metrics_desc.append("* *Multi-class problems use ANOVA F and Eta Squared to measure how well features distinguish between multiple classes*")
                        
                        # Display metric descriptions
                        if metrics_desc:
                            st.markdown("\n".join(metrics_desc))
                    else:
                        # Simplified explanation for beginners
                        st.markdown("""
                        #### Quick Guide for Beginners
                        
                        This analysis shows how well each feature can predict your target variable. Here's what you need to know:
                        
                        **What the Table Shows:**
                        - **Feature**: The name of each column in your dataset
                        - **Relationship Type**: How the feature relates to your target (Linear, Non-linear, etc.)
                        - **Strength**: How useful this feature is for prediction
                        - **Type**: Whether the feature contains numbers (Numerical) or categories (Categorical)
                        - **Linear?**: Indicates if a significant linear relationship exists (Yes/No)
                        
                        **Focus On:**
                        - 🟢 **Strong** features: These are your best predictors - definitely include them!
                        - 🟡 **Moderate** features: Good predictors that should be included
                        - 🟠 **Weak** features: May still be useful, especially combined with others
                        - 🔴 **Very Weak** features: Consider removing if you have many other strong features
                        """)
                    
                    # Always show these standard sections
                    st.markdown("""
                    #### Relationship Types:
                    - **Linear**: Simple straight-line relationship that can be positive or negative
                    - **Non-linear**: Curved or complex pattern that cannot be described by a straight line
                    - **Complex**: Contains both linear and non-linear components
                    - **Category Effect**: Categorical variable with significant impact on the target
                    - **Class Separation**: Feature effectively distinguishes between classes in a binary classification problem
                    
                    #### Strength Interpretation:
                    - **Strong**: Feature has substantial predictive power; prioritize in modeling
                    - **Moderate**: Good predictor that should be included in models
                    - **Weak**: Limited predictive power; may still be useful in combination with other features
                    - **Very Weak**: Minimal relationship; consider removing unless domain knowledge suggests otherwise
                    """)
                    
                    if user_level == "Advanced":
                        st.markdown("""
                        #### How Feature Strength is Calculated:
                        Feature strength classifications vary by problem type:
                        
                        **For Regression (numeric target):**
                        - Based on correlation coefficients (Pearson/Spearman):
                          - **Strong**: |correlation| ≥ 0.7
                          - **Moderate**: |correlation| ≥ 0.4
                          - **Weak**: |correlation| ≥ 0.2
                          - **Very Weak**: |correlation| < 0.2
                        - *Note: While Mutual Information is calculated for regression problems, it's primarily used to detect non-linear relationships rather than determine strength. For regression, correlation coefficients provide a more interpretable measure of linear relationship strength.*
                        
                        **For Classification (Binary and Multi-class):**
                        - For numeric features: Based on Eta Squared (when available):
                          - **Strong**: Eta Squared ≥ 0.26
                          - **Moderate**: Eta Squared ≥ 0.13
                          - **Weak**: Eta Squared ≥ 0.02
                          - **Very Weak**: Eta Squared < 0.02
                        - *If Eta Squared isn't available, Point-Biserial correlation (binary only) or Normalized MI may be used:*
                          - **For Normalized MI**: Strong (≥0.5), Moderate (≥0.3), Weak (≥0.1), Very Weak (<0.1)
                        
                        - For categorical features: Based on Cramer's V:
                          - **Strong**: Cramer's V ≥ 0.5
                          - **Moderate**: Cramer's V ≥ 0.3
                          - **Weak**: Cramer's V ≥ 0.1
                          - **Very Weak**: Cramer's V < 0.1
                        
                        *Note: Multi-class classification uses the same thresholds as binary classification, but Eta Squared and ANOVA F are particularly important for measuring how well features separate multiple classes.*
                        """) 

                st.write("The following table shows how each feature relates to your target variable:")
                
                try:
                    # Check for basic validity of the relationship_df
                    if relationship_df is None or not isinstance(relationship_df, pd.DataFrame):
                        st.error("The relationship DataFrame is not valid.")
                    elif relationship_df.empty:
                        st.info("No feature relationships were found.")
                    else:
                        # Show how many features we found
                        st.info(f"Found relationships for {len(relationship_df)} features.")
                        
                        # Count relationship types for summary
                        linear_count = len(relationship_df[relationship_df['Relationship_Type'] == 'Linear'])
                        nonlinear_count = len(relationship_df[relationship_df['Relationship_Type'].isin(['Non-linear', 'Non-linear/Complex'])])
                        complex_count = len(relationship_df[relationship_df['Relationship_Type'] == 'Complex (Linear + Non-linear)'])
                        class_separation_count = len(relationship_df[relationship_df['Relationship_Type'] == 'Class Separation'])
                        strong_count = len(relationship_df[relationship_df['Strength'] == 'Strong'])
                        moderate_count = len(relationship_df[relationship_df['Strength'] == 'Moderate'])
                        
                        # Create a summary card
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Linear Relationships", linear_count)
                        with col2:
                            st.metric("Non-linear/Complex", nonlinear_count + complex_count)
                        with col3:
                            st.metric("Class Separation", class_separation_count)
                        with col4:
                            st.metric("Strong/Moderate Predictors", strong_count + moderate_count)
                        
                        # Add a quick summary of findings
                        if strong_count > 0:
                            st.success(f"✅ Found {strong_count} strong predictors that may be highly valuable for your model.")
                        elif moderate_count > 0:
                            st.info("ℹ️ Found moderate predictors, but no strong ones. Your model may benefit from feature engineering.")
                        
                        if nonlinear_count + complex_count > 0:
                            st.warning(f"⚠️ Found {nonlinear_count + complex_count} non-linear relationships. Consider using tree-based models or transformations.")
                        
                        if class_separation_count > 0:
                            st.success(f"✅ Found {class_separation_count} features with good class separation for classification (binary or multi-class).")
                        
                        # Create a simplified version of the dataframe with all metrics
                        complete_table = relationship_df.copy()
                        
                        # Round numeric columns to 3 decimal places for readability
                        numeric_columns = complete_table.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
                        for col in numeric_columns:
                            if col != 'Feature':
                                complete_table[col] = complete_table[col].round(3)
                        
                        # Rename columns to be more user-friendly
                        column_renames = {
                            'Pearson_Correlation': 'Pearson r',
                            'Pearson_P_Value': 'Pearson p-value',
                            'Spearman_Correlation': 'Spearman ρ',
                            'Spearman_P_Value': 'Spearman p-value',
                            'Mutual_Information': 'Mutual Info',
                            'ANOVA_F': 'ANOVA F',
                            'ANOVA_P_Value': 'ANOVA p-value',
                            'Normalized_MI': 'Norm. MI',
                            'Linear_Relationship': 'Linear?',
                            'Non_Linear_Relationship': 'Non-Linear?',
                            'Chi2': 'Chi-square',
                            'DOF': 'Degrees of Freedom',
                            'Chi2_P_Value': 'Chi-square p-value', 
                            'Eta_Squared': 'Eta Squared',
                            'Cramers_V': "Cramer's V",
                            'PointBiserial_Correlation': 'Point-Biserial r',
                            'PointBiserial_P_Value': 'Point-Biserial p-value',
                            'Significant_Difference': 'Significant Groups?',
                            'Significant_Association': 'Significant Association?'
                        }
                        
                        # Only rename columns that exist
                        rename_dict = {k: v for k, v in column_renames.items() if k in complete_table.columns}
                        complete_table = complete_table.rename(columns=rename_dict)
                        
                        # Determine what type of metrics we have
                        has_pearson = 'Pearson r' in complete_table.columns
                        has_anova = 'ANOVA F' in complete_table.columns
                        
                        # Create display columns based on user level
                        if user_level == "Beginner":
                            # Simplified view for beginners - only essential columns
                            essential_cols = ['Feature', 'Relationship_Type', 'Strength']
                            
                            # Add feature type column for beginners (rename for simplicity)
                            if 'Feature_Type' in complete_table.columns:
                                complete_table['Type'] = complete_table['Feature_Type']
                                essential_cols.append('Type')
                            
                            # Add a simplified Linear indicator column for beginners
                            if 'Linear_Relationship' in complete_table.columns:
                                complete_table['Linear?'] = complete_table['Linear_Relationship']
                                essential_cols.append('Linear?')
                            elif 'Linear?' in complete_table.columns:
                                essential_cols.append('Linear?')
                            
                            # Add one key metric based on the problem type, but be selective
                            # Avoid Point-Biserial r as it's too technical for beginners
                            #if 'Pearson r' in complete_table.columns:
                            #    essential_cols.append('Pearson r')
                            #elif 'Eta Squared' in complete_table.columns:
                            #    essential_cols.append('Eta Squared')
                            #elif "Cramer's V" in complete_table.columns:
                            #    essential_cols.append("Cramer's V")
                            # Only include Point-Biserial as last resort and rename it
                            #elif 'Point-Biserial r' in complete_table.columns:
                            #    complete_table['Correlation'] = complete_table['Point-Biserial r']
                            #    essential_cols.append('Correlation')
                            
                            display_cols = [col for col in essential_cols if col in complete_table.columns]
                        else:
                            # Advanced view - all metrics
                            # Set up display columns - prioritize key columns first
                            display_cols = ['Feature', 'Relationship_Type', 'Strength']
                            
                            # Get all columns except for the key columns we've already added
                            metric_cols = [col for col in complete_table.columns 
                                         if col not in display_cols 
                                         and col not in ['Feature_Type']]  # Exclude Feature_Type
                            
                            # Sort the metric columns logically
                            pearson_metrics = [col for col in metric_cols if 'Pearson' in col]
                            spearman_metrics = [col for col in metric_cols if 'Spearman' in col]
                            anova_metrics = [col for col in metric_cols if 'ANOVA' in col]
                            # Make sure we catch both original and renamed versions of Eta Squared
                            eta_metrics = [col for col in metric_cols if 'Eta' in col or col == 'Eta_Squared' or col == 'Eta Squared']
                            # Also ensure Eta_Squared is included from complete_table if it exists but wasn't in metric_cols
                            if 'Eta_Squared' in complete_table.columns and 'Eta_Squared' not in eta_metrics:
                                eta_metrics.append('Eta_Squared')
                            if 'Eta Squared' in complete_table.columns and 'Eta Squared' not in eta_metrics:
                                eta_metrics.append('Eta Squared')
                            
                            chi2_metrics = [col for col in metric_cols if 'Chi' in col]
                            cramers_metrics = [col for col in metric_cols if 'Cramer' in col]
                            mi_metrics = [col for col in metric_cols if 'MI' in col or 'Mutual' in col]
                            other_metrics = [col for col in metric_cols if col not in pearson_metrics + spearman_metrics + 
                                             anova_metrics + eta_metrics + chi2_metrics + cramers_metrics + mi_metrics]
                            
                            # Combine all metrics in a logical order
                            ordered_metrics = pearson_metrics + spearman_metrics + anova_metrics + eta_metrics + chi2_metrics + cramers_metrics + mi_metrics + other_metrics
                            
                            # Make sure Eta_Squared is included if it exists in the dataframe
                            if 'Eta_Squared' in complete_table.columns and 'Eta_Squared' not in ordered_metrics:
                                ordered_metrics.append('Eta_Squared')
                            if 'Eta Squared' in complete_table.columns and 'Eta Squared' not in ordered_metrics:
                                ordered_metrics.append('Eta Squared')
                            
                            # Add all metric columns to display columns
                            display_cols.extend(ordered_metrics)
                        
                        # Create the final display table with selected metrics
                        display_table = complete_table[display_cols]
                        
                        # Remove any blank rows (NaN values in Feature column) and empty rows
                        display_table = display_table.dropna(subset=['Feature']).reset_index(drop=True)
                        
                        # Add journey point for feature relationships that contains relationship type and strength for each feature
                        st.session_state.logger.log_journey_point(
                            stage="DATA_EXPLORATION",
                            decision_type="FEATURE_RELATIONSHIPS",
                            description="Feature-Target Relationships",
                            details={'Relationships': display_table[['Feature','Relationship_Type','Strength']].to_dict(orient='records')},
                            parent_id=None
                        )
                        

                        # Split tables by feature type based on user level
                        if 'Feature_Type' in complete_table.columns and user_level == "Advanced":
                            # For advanced users, show separate tables for numeric and categorical features
                            numeric_features = complete_table[complete_table['Feature_Type'] == 'Numerical']['Feature'].tolist()
                            categorical_features = complete_table[complete_table['Feature_Type'] == 'Categorical']['Feature'].tolist()
                            
                            numeric_table = complete_table[complete_table['Feature'].isin(numeric_features)][display_cols]
                            categorical_table = complete_table[complete_table['Feature'].isin(categorical_features)][display_cols]
                            
                            # Display separate tables
                            st.subheader("Feature-Target Relationship Analysis")
                            
                            # Add user-level specific instructions
                            st.info("🔬 **Advanced Mode**: Showing all statistical metrics. Toggle off 'Show Advanced Statistical Metrics' above for a simplified view.")
                            
                            # Display tabs for the tables
                            tab1, tab2 = st.tabs(["Numeric Features", "Categorical Features"])
                            
                            with tab1:
                                numeric_count = len(numeric_table)
                                if numeric_count > 0:
                                    style_and_display_table(numeric_table, "Numerical Features", display_cols, user_level)
                                    st.info(f"💡 Showing {numeric_count} numerical features. For regression problems, look for strong linear or non-linear relationships. For classification, focus on features with high class separation.")
                                else:
                                    st.info("No numerical features found in the analysis.")
                            
                            with tab2:
                                categorical_count = len(categorical_table)
                                if categorical_count > 0:
                                    style_and_display_table(categorical_table, "Categorical Features", display_cols, user_level)
                                    st.info(f"💡 Showing {categorical_count} categorical features. For classification problems (binary or multi-class), look for features with high Chi-square values and Cramer's V. For regression, focus on features with high Eta Squared.")
                                else:
                                    st.info("No categorical features found in the analysis.")

                            # Add explanation of color encoding
                            with st.expander("🎨 Understanding Table Colour Coding", expanded=False):
                                st.markdown("""
                                ### Table Colour Legend
                                
                                The tables below use color coding to help you quickly identify important relationships:
                                
                                #### Strength Column Colour Coding:
                                - 🟢 **Light Green**: Strong relationships (high predictive value)
                                - 🟡 **Light Yellow**: Moderate relationships (good predictive value)
                                - 🟠 **Light Orange**: Weak relationships (limited predictive value)
                                - 🔴 **Light Red**: Very weak relationships (minimal predictive value)
                                """)
                                
                                st.markdown("""
                                #### Numeric Metric Columns (Gradient):
                                - **Red-Yellow-Green Gradient**: Applied to correlation coefficients, p-values, and other numeric metrics
                                - 🟢 **Green shades**: Higher values (stronger relationships, more significant)
                                - 🟡 **Yellow shades**: Moderate values 
                                - 🔴 **Red shades**: Lower values (weaker relationships, less significant)
                                """)
                                
                                st.markdown("""
                                #### Quick Visual Scanning Tips:
                                - **Look for green**: Features with strong predictive power
                                - **Focus on yellow**: Features with moderate predictive power worth including
                                - **Consider orange carefully**: Weak features that might still be useful in combination
                                - **Question red**: Very weak features that may be candidates for removal
                                
                                *Note: The gradient colours help you quickly spot patterns across multiple metrics, while the strength colours provide a clear assessment of each feature's overall predictive value.*
                                """)
                        else:
                            # For beginners, show a single combined table or when Feature_Type is not available
                            st.subheader("Feature-Target Relationship Analysis")
                            
                            # Style the DataFrame to make it more readable
                            styled_table = display_table.style.format(precision=3)
                            
                            # Replace NaN values with a more readable format
                            styled_table = styled_table.format(na_rep="—")
                            
                            # Apply more styling to highlight important information
                            # Create a custom styler function to color the Strength column based on values
                            def highlight_strength(val):
                                if val == 'Strong':
                                    return 'background-color: #a8f0a8'
                                elif val == 'Moderate':
                                    return 'background-color: #f0f0a8'
                                elif val == 'Weak':
                                    return 'background-color: #f0d0a8'
                                elif val == 'Very Weak':
                                    return 'background-color: #f0b0a8'
                                return ''
                            
                            # Apply highlighting to only the Strength column
                            styled_table = styled_table.applymap(highlight_strength, subset=['Strength'])
                            
                            # Apply gradient to numeric columns if any (only for advanced users)
                            if user_level == "Advanced":
                                numeric_cols = display_table.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
                                if len(numeric_cols) > 0:
                                    # Apply gradient only to numeric columns
                                    try:
                                        styled_table = styled_table.background_gradient(
                                            subset=numeric_cols, 
                                            cmap='RdYlGn', 
                                            low=0.3, 
                                            high=0.9
                                        )
                                    except:
                                        # If gradient fails, continue without it
                                        st.warning("Could not apply gradient styling to numeric columns.")
                            
                            # Display table title based on user level
                            if user_level == "Beginner":
                                st.write("#### All Features - Simplified View")
                                st.caption("*Showing essential information for all features. Features are sorted by predictive strength.*")
                            else:
                                st.write("#### All Features - Complete Analysis")
                            
                            # Use st.dataframe with horizontal scrolling enabled
                            st.dataframe(
                                styled_table,
                                width='stretch',
                                height=max(200, min(500, 80 + 35 * len(display_table)))  # Better height adjustment
                            )
                            
                            # Add beginner-friendly summary below the table
                            if user_level == "Beginner" and len(display_table) > 0:
                                # Count features by strength
                                strong_features = len(display_table[display_table['Strength'] == 'Strong'])
                                moderate_features = len(display_table[display_table['Strength'] == 'Moderate'])
                                weak_features = len(display_table[display_table['Strength'] == 'Weak'])
                                very_weak_features = len(display_table[display_table['Strength'] == 'Very Weak'])
                                
                                # Provide actionable insights for beginners
                                with st.expander("💡 Quick Feature Selection Guide", expanded=True):
                                    if strong_features > 0:
                                        strong_list = display_table[display_table['Strength'] == 'Strong']['Feature'].tolist()
                                        st.success(f"🟢 **{strong_features} Strong features**: {', '.join(strong_list[:5])}{'...' if len(strong_list) > 5 else ''} → **Definitely include these in your model!**")
                                        #st.markdown("→ **Definitely include these in your model!**")
                                    
                                    if moderate_features > 0:
                                        moderate_list = display_table[display_table['Strength'] == 'Moderate']['Feature'].tolist()
                                        st.info(f"🟡 **{moderate_features} Moderate features**: {', '.join(moderate_list[:5])}{'...' if len(moderate_list) > 5 else ''} → **Good to include - they add predictive value**")
                                        #st.markdown("→ **Good to include - they add predictive value**")
                                    
                                    if weak_features > 0:
                                        st.warning(f"🟠 **{weak_features} Weak features**: May still be useful in combination with stronger features")
                                    
                                    if very_weak_features > 0:
                                        st.error(f"🔴 **{very_weak_features} Very Weak features**: Consider removing unless domain knowledge suggests they're important")
                                    
                                    # Overall guidance
                                    st.markdown("---")
                                    if strong_features + moderate_features == 0:
                                        st.warning("⚠️ **No strong or moderate features found.** You may need to:\n- Consider feature engineering\n- Check for data quality issues\n- Verify your target variable is appropriate")
                                    elif strong_features + moderate_features >= 3:
                                        st.success("✅ **You have good predictive features!** Focus on the green and yellow highlighted features for your model.")
                                    else:
                                        st.info("ℹ️ **You have some useful features**, but consider feature engineering to create more predictive variables.")

                except Exception as e:
                    # Show detailed error for the inner try block
                    st.error(f"Error analyzing feature relationship details: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                
                # Log relationship analysis
                if 'logger' in st.session_state:
                    st.session_state.logger.log_calculation(
                        "Feature-Target Relationship Analysis",
                        {
                            "feature_count": len(relationship_df),
                            "linear_count": len(relationship_df[relationship_df['Relationship_Type'] == 'Linear']),
                            "nonlinear_count": len(relationship_df[relationship_df['Relationship_Type'].isin(['Non-linear', 'Non-linear/Complex'])]),
                            "complex_count": len(relationship_df[relationship_df['Relationship_Type'] == 'Complex (Linear + Non-linear)']),
                            "class_separation_count": len(relationship_df[relationship_df['Relationship_Type'] == 'Class Separation']),
                            "strong_features": list(relationship_df[relationship_df['Strength'] == 'Strong']['Feature']),
                            "moderate_features": list(relationship_df[relationship_df['Strength'] == 'Moderate']['Feature']),
                            "user_level": user_level
                        }
                    )
        
        except Exception as e:
            st.error(f"Error analyzing feature relationships: {str(e)}")
            # Log the error
            if 'logger' in st.session_state:
                st.session_state.logger.log_error(
                    "Feature Relationship Analysis Failed",
                    {"error": str(e)}
                )

def detect_relationship_type(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Detect linear and non-linear relationships between features and target.
    
    Returns a DataFrame with relationship information for each numerical feature.
    
    Parameters:
    -----------
    data : pandas DataFrame
        The dataset containing both features and target
    target_column : str
        The name of the target column
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with relationship information for each feature
    """
    # Initialize results dictionary
    results = []
    
    # Get target data
    y = data[target_column]
    
    # Always determine if target is numeric (needed for analysis logic)
    is_target_numeric = pd.api.types.is_numeric_dtype(y)
    
    # Use session state variables for problem type detection
    if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type:
        problem_type = st.session_state.problem_type
        is_binary_classification = getattr(st.session_state, 'is_binary', False)
        is_multiclass_classification = getattr(st.session_state, 'is_multiclass', False)
        is_regression = getattr(st.session_state, 'is_regression', False)
    else:
        # Fallback to heuristic detection for backward compatibility
        # Determine if target is binary (has only 2 unique values)
        is_binary_classification = y.nunique() <= 2
        
        # Determine if target is multi-class classification
        is_multiclass_classification = False
        if 'encoding_mappings' in st.session_state and target_column in st.session_state.encoding_mappings:
            # If we have encoding mapping, it was treated as classification in data loading
            is_multiclass_classification = y.nunique() > 2
        elif is_target_numeric and 3 <= y.nunique() <= 20:
            # Check if this might be multiclass by looking for integer-like values
            try:
                # If all values are close to integers, it might be encoded multiclass
                y_no_nan = y.dropna()
                if len(y_no_nan) > 0:
                    is_integer_like = np.allclose(y_no_nan, np.round(y_no_nan), atol=1e-10)
                    consecutive_integers = (y_no_nan.min() >= 0 and 
                                          set(y_no_nan.unique()) == set(range(int(y_no_nan.min()), int(y_no_nan.max()) + 1)))
                    is_multiclass_classification = is_integer_like and consecutive_integers
            except:
                is_multiclass_classification = False
        
        is_regression = not (is_binary_classification or is_multiclass_classification)
    
    # Get numerical features (exclude target)
    numerical_features = [col for col in data.columns 
                        if col != target_column and pd.api.types.is_numeric_dtype(data[col])]
    
    for feature in numerical_features:
        # Get clean feature data (no NaN, inf)
        X = data[feature].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Get valid indices where both feature and target are valid
        if is_target_numeric:
            y_clean = y.replace([np.inf, -np.inf], np.nan).dropna()
        else:
            y_clean = y.dropna()
            
        valid_indices = X.index.intersection(y_clean.index)
        X_clean = X[valid_indices]
        y_clean = y_clean[valid_indices]
        
        # Skip if not enough data
        if len(X_clean) < 5:
            continue
            
        result = {"Feature": feature}
        
        if is_target_numeric and not is_binary_classification and not is_multiclass_classification:
            # REGRESSION CASE: Continuous numeric target
            # Pearson correlation (linear)
            pearson_corr, pearson_p = pearsonr(X_clean, y_clean)
            result["Pearson_Correlation"] = pearson_corr
            result["Pearson_P_Value"] = pearson_p
            result["Linear_Relationship"] = "Yes" if abs(pearson_corr) >= 0.3 and pearson_p < 0.05 else "No"
            
            # Spearman rank correlation (monotonic)
            spearman_corr, spearman_p = spearmanr(X_clean, y_clean)
            result["Spearman_Correlation"] = spearman_corr
            result["Spearman_P_Value"] = spearman_p
            
            # Detect non-linear relationship
            # If Spearman is significant but Pearson is not as strong,
            # it suggests a monotonic but non-linear relationship
            non_linear = (abs(spearman_corr) >= 0.3 and spearman_p < 0.05 and 
                        (abs(spearman_corr) - abs(pearson_corr) >= 0.1))
            
            # Compute mutual information for additional non-linear detection
            if len(X_clean) >= 4:  # Need enough samples for binning
                try:
                    # Discretize X and y for mutual information
                    x_discrete = KBinsDiscretizer(n_bins=max(2, min(5, len(X_clean) // 10)), 
                                                encode='ordinal', strategy='uniform').fit_transform(X_clean.values.reshape(-1, 1)).flatten()
                    y_discrete = KBinsDiscretizer(n_bins=max(2, min(5, len(y_clean) // 10)), 
                                                encode='ordinal', strategy='uniform').fit_transform(y_clean.values.reshape(-1, 1)).flatten()
                    
                    # Calculate mutual information
                    mi = mutual_info_score(x_discrete, y_discrete)
                    x_entropy = mutual_info_score(x_discrete, x_discrete)
                    y_entropy = mutual_info_score(y_discrete, y_discrete)
                    if x_entropy > 0 and y_entropy > 0:
                        normalized_mi = mi / np.sqrt(x_entropy * y_entropy)
                    else:
                        normalized_mi = 0
                    result["Mutual_Information"] = normalized_mi
                    
                    # If MI suggests relationship but Pearson correlation is weak, it's likely non-linear
                    mi_suggests_nonlinear = normalized_mi >= 0.2 and abs(pearson_corr) < 0.3
                    non_linear = non_linear or mi_suggests_nonlinear
                    
                except Exception as mi_error:
                    # Skip MI if it fails but log the error for debugging
                    result["MI_Error"] = str(mi_error)
                    pass
            
            # Store non-linear relationship detection internally but don't include in final output
            has_non_linear = "Yes" if non_linear else "No"
            
            # Determine relationship type and strength
            if result["Linear_Relationship"] == "Yes":
                if has_non_linear == "Yes":
                    result["Relationship_Type"] = "Complex (Linear + Non-linear)"
                else:
                    result["Relationship_Type"] = "Linear"
            elif has_non_linear == "Yes":
                result["Relationship_Type"] = "Non-linear"
            else:
                result["Relationship_Type"] = "Weak/None"
                
            # Determine strength
            correlation = max(abs(pearson_corr), abs(spearman_corr))
            if correlation >= 0.7:
                result["Strength"] = "Strong"
            elif correlation >= 0.4:
                result["Strength"] = "Moderate"
            elif correlation >= 0.2:
                result["Strength"] = "Weak"
            else:
                result["Strength"] = "Very Weak"
                
        elif is_binary_classification:
            # BINARY CLASSIFICATION CASE
            # We'll use specific binary classification metrics
            try:
                # Convert binary target to numeric if not already
                if not is_target_numeric:
                    # Create a mapping of the two unique values to 0 and 1
                    unique_vals = y_clean.unique()
                    y_mapping = {val: i for i, val in enumerate(unique_vals)}
                    y_numeric = y_clean.map(y_mapping)
                else:
                    y_numeric = y_clean
                
                # Point-Biserial Correlation (special correlation for binary target)
                try:
                    pointb_corr, pointb_p = pointbiserialr(X_clean, y_numeric)
                    result["PointBiserial_Correlation"] = pointb_corr
                    result["PointBiserial_P_Value"] = pointb_p
                    result["Linear_Relationship"] = "Yes" if abs(pointb_corr) >= 0.3 and pointb_p < 0.05 else "No"
                except Exception as e:
                    # If point-biserial fails, fall back to ANOVA
                    result["Linear_Relationship"] = "Undetermined"
                    
                # Group by target categories for ANOVA
                groups = [X_clean[y_numeric == val] for val in y_numeric.unique()]
                groups = [g for g in groups if len(g) > 1]
                
                if len(groups) >= 2:
                    # ANOVA analysis for group differences
                    f_stat, p_value = f_oneway(*groups)
                    result["ANOVA_F"] = f_stat
                    result["ANOVA_P_Value"] = p_value
                    result["Significant_Difference"] = "Yes" if p_value < 0.05 else "No"
                    
                    # Compute effect size (simplified Eta squared)
                    y_values = y_numeric.unique()
                    group_means = [X_clean[y_numeric == v].mean() for v in y_values]
                    grand_mean = X_clean.mean()
                    group_sizes = [len(X_clean[y_numeric == v]) for v in y_values]
                    
                    # Between groups sum of squares
                    ss_between = sum(size * ((mean - grand_mean) ** 2) for size, mean in zip(group_sizes, group_means))
                    # Total sum of squares
                    ss_total = sum((x - grand_mean) ** 2 for x in X_clean)
                    
                    if ss_total > 0:
                        eta_squared = ss_between / ss_total
                        result["Eta_Squared"] = eta_squared
                    else:
                        eta_squared = 0
                
                # Compute normalized mutual information (more robust for binary targets)
                try:
                    # For binary targets, discretization of y is unnecessary
                    x_discrete = KBinsDiscretizer(n_bins=max(2, min(5, len(X_clean) // 10)), 
                                                encode='ordinal', strategy='uniform').fit_transform(X_clean.values.reshape(-1, 1)).flatten()
                    
                    # Calculate mutual information
                    mi = mutual_info_score(x_discrete, y_numeric)
                    entropy_x = mutual_info_score(x_discrete, x_discrete)
                    if entropy_x > 0:
                        normalized_mi = mi / entropy_x
                        result["Normalized_MI"] = normalized_mi
                    
                        # Determine if there's a non-linear component
                        # Compare MI to the Point-Biserial correlation
                        if "PointBiserial_Correlation" in result:
                            pointb_strength = abs(result["PointBiserial_Correlation"])
                            # If MI is substantially higher than what linear correlation would suggest
                            non_linear = normalized_mi >= 0.2 and (normalized_mi - pointb_strength >= 0.15)
                            # Store non-linear relationship internally but don't include in final output
                            has_non_linear = "Yes" if non_linear else "No"
                        else:
                            # If Point-Biserial failed, use MI threshold
                            has_non_linear = "Yes" if normalized_mi >= 0.2 else "No"
                    
                except Exception as mi_error:
                    result["MI_Error"] = str(mi_error)
                    has_non_linear = "Undetermined"
                
                # Determine feature strength for binary classification
                if "Eta_Squared" in result and eta_squared > 0:
                    # Use Eta Squared for strength assessment
                    if eta_squared >= 0.26:
                        result["Strength"] = "Strong"
                    elif eta_squared >= 0.13:
                        result["Strength"] = "Moderate"
                    elif eta_squared >= 0.02:
                        result["Strength"] = "Weak"
                    else:
                        result["Strength"] = "Very Weak"
                elif "PointBiserial_Correlation" in result:
                    # Use Point-Biserial correlation for strength
                    pb_corr = abs(result["PointBiserial_Correlation"])
                    if pb_corr >= 0.7:
                        result["Strength"] = "Strong"
                    elif pb_corr >= 0.4:
                        result["Strength"] = "Moderate"
                    elif pb_corr >= 0.2:
                        result["Strength"] = "Weak"
                    else:
                        result["Strength"] = "Very Weak"
                elif "Normalized_MI" in result:
                    # Use MI if other metrics aren't available
                    if result["Normalized_MI"] >= 0.5:
                        result["Strength"] = "Strong"
                    elif result["Normalized_MI"] >= 0.3:
                        result["Strength"] = "Moderate"
                    elif result["Normalized_MI"] >= 0.1:
                        result["Strength"] = "Weak"
                    else:
                        result["Strength"] = "Very Weak"
                else:
                    result["Strength"] = "Undetermined"
                    
                # Determine relationship type
                if result["Significant_Difference"] == "Yes":
                    # For binary classification, use class separation term to match binary classification metrics
                    if is_binary_classification:
                        result["Relationship_Type"] = "Class Separation"
                    else:
                        result["Relationship_Type"] = "Category Association"
                else:
                    result["Relationship_Type"] = "Weak/None"
                
            except Exception as e:
                result["Relationship_Type"] = "Error in Analysis"
                result["Strength"] = "Undetermined"
        
        elif is_multiclass_classification:
            # MULTI-CLASS CLASSIFICATION CASE
            try:
                # Convert multi-class target to numeric if not already
                if not is_target_numeric:
                    # Create a mapping of the unique values to 0, 1, 2, etc.
                    unique_vals = y_clean.unique()
                    y_mapping = {val: i for i, val in enumerate(unique_vals)}
                    y_numeric = y_clean.map(y_mapping)
                else:
                    y_numeric = y_clean
                
                # Group by target categories for ANOVA
                groups = [X_clean[y_numeric == val] for val in y_numeric.unique()]
                groups = [g for g in groups if len(g) > 1]
                
                if len(groups) >= 2:
                    # ANOVA analysis for group differences
                    f_stat, p_value = f_oneway(*groups)
                    result["ANOVA_F"] = f_stat
                    result["ANOVA_P_Value"] = p_value
                    result["Significant_Difference"] = "Yes" if p_value < 0.05 else "No"
                    
                    # Compute effect size (Eta squared)
                    y_values = y_numeric.unique()
                    group_means = [X_clean[y_numeric == v].mean() for v in y_values]
                    grand_mean = X_clean.mean()
                    group_sizes = [len(X_clean[y_numeric == v]) for v in y_values]
                    
                    # Between groups sum of squares
                    ss_between = sum(size * ((mean - grand_mean) ** 2) for size, mean in zip(group_sizes, group_means))
                    # Total sum of squares
                    ss_total = sum((x - grand_mean) ** 2 for x in X_clean)
                    
                    if ss_total > 0:
                        eta_squared = ss_between / ss_total
                        result["Eta_Squared"] = eta_squared
                    else:
                        eta_squared = 0
                
                # Compute normalized mutual information for multi-class
                try:
                    # Discretize X for mutual information
                    x_discrete = KBinsDiscretizer(n_bins=max(2, min(5, len(X_clean) // 10)), 
                                                encode='ordinal', strategy='uniform').fit_transform(X_clean.values.reshape(-1, 1)).flatten()
                    
                    # Calculate mutual information
                    mi = mutual_info_score(x_discrete, y_numeric)
                    entropy_x = mutual_info_score(x_discrete, x_discrete)
                    if entropy_x > 0:
                        normalized_mi = mi / entropy_x
                        result["Normalized_MI"] = normalized_mi
                    
                except Exception as mi_error:
                    result["MI_Error"] = str(mi_error)
                
                # Determine feature strength for multi-class classification
                if "Eta_Squared" in result and eta_squared > 0:
                    # Use Eta Squared for strength assessment
                    if eta_squared >= 0.26:
                        result["Strength"] = "Strong"
                    elif eta_squared >= 0.13:
                        result["Strength"] = "Moderate"
                    elif eta_squared >= 0.02:
                        result["Strength"] = "Weak"
                    else:
                        result["Strength"] = "Very Weak"
                elif "Normalized_MI" in result:
                    # Use MI if Eta Squared isn't available
                    if result["Normalized_MI"] >= 0.5:
                        result["Strength"] = "Strong"
                    elif result["Normalized_MI"] >= 0.3:
                        result["Strength"] = "Moderate"
                    elif result["Normalized_MI"] >= 0.1:
                        result["Strength"] = "Weak"
                    else:
                        result["Strength"] = "Very Weak"
                else:
                    result["Strength"] = "Undetermined"
                    
                # Determine relationship type
                if result["Significant_Difference"] == "Yes":
                    result["Relationship_Type"] = "Class Separation"
                else:
                    result["Relationship_Type"] = "Weak/None"
                
            except Exception as e:
                result["Relationship_Type"] = "Error in Analysis"
                result["Strength"] = "Undetermined"
        
        else:
            # OTHER CLASSIFICATION CASES or NON-NUMERIC TARGET
            # Use ANOVA F-statistic for category separation
            try:
                # Group by target categories
                groups = [X_clean[y_clean == cat] for cat in y_clean.unique()]
                groups = [g for g in groups if len(g) > 1]  # Need at least 2 samples per group
                
                if len(groups) >= 2:
                    f_stat, p_value = f_oneway(*groups)
                    result["ANOVA_F"] = f_stat
                    result["ANOVA_P_Value"] = p_value
                    result["Significant_Difference"] = "Yes" if p_value < 0.05 else "No"
                    
                    # Check for non-linear pattern using entropy-based measures
                    # Use Mutual Information
                    try:
                        # Discretize X for mutual information
                        x_discrete = KBinsDiscretizer(n_bins=max(2, min(5, len(X_clean) // 10)), 
                                                    encode='ordinal', strategy='uniform').fit_transform(X_clean.values.reshape(-1, 1)).flatten()
                        
                        # Calculate mutual information
                        mi = mutual_info_score(x_discrete, y_clean)
                        entropy_x = mutual_info_score(x_discrete, x_discrete)
                        if entropy_x > 0:
                            normalized_mi = mi / entropy_x
                            result["Normalized_MI"] = normalized_mi
                            
                            # Determine relationship type
                            if result["Significant_Difference"] == "Yes":
                                if normalized_mi >= 0.3:
                                    result["Relationship_Type"] = "Non-linear/Complex"
                                else:
                                    result["Relationship_Type"] = "Group Differences"
                            else:
                                result["Relationship_Type"] = "Weak/None"
                                
                            # Determine strength based on MI
                            if normalized_mi >= 0.5:
                                result["Strength"] = "Strong"
                            elif normalized_mi >= 0.3:
                                result["Strength"] = "Moderate"
                            elif normalized_mi >= 0.1:
                                result["Strength"] = "Weak"
                            else:
                                result["Strength"] = "Very Weak"
                        else:
                            result["Relationship_Type"] = "Undetermined"
                            result["Strength"] = "Undetermined"
                    except Exception as mi_error:
                        result["Relationship_Type"] = "Undetermined"
                        result["Strength"] = "Undetermined"
                else:
                    result["Relationship_Type"] = "Insufficient Data"
                    result["Strength"] = "Undetermined"
            except Exception as mi_error:
                result["Relationship_Type"] = "Error in Analysis"
                result["Strength"] = "Undetermined"
        
        results.append(result)
    
    # Get categorical features (exclude target)
    categorical_features = [col for col in data.columns 
                          if col != target_column and not pd.api.types.is_numeric_dtype(data[col])]
    
    for feature in categorical_features:
        # Get clean feature data (no NaN)
        X = data[feature].dropna()
        
        # Get valid indices where both feature and target are valid
        if is_target_numeric:
            y_clean = y.replace([np.inf, -np.inf], np.nan).dropna()
        else:
            y_clean = y.dropna()
            
        valid_indices = X.index.intersection(y_clean.index)
        X_clean = X[valid_indices]
        y_clean = y_clean[valid_indices]
        
        # Skip if not enough data
        if len(X_clean) < 5:
            continue
            
        result = {"Feature": feature}
        result["Feature_Type"] = "Categorical"
        
        # Get number of unique categories and their counts
        unique_cats = X_clean.value_counts()
        result["Categories_Count"] = len(unique_cats)
        
        # Handle case with too many categories (> 100)
        if len(unique_cats) > 100:
            result["Relationship_Type"] = "Too Many Categories"
            result["Strength"] = "Undetermined"
            result["Note"] = f"Feature has {len(unique_cats)} categories, which is too many for reliable analysis"
            results.append(result)
            continue
        
        # For numeric target, use ANOVA or similar test
        if is_target_numeric and not is_binary_classification:
            try:
                # Group by feature categories
                category_groups = []
                for cat in X_clean.unique():
                    group = y_clean[X_clean == cat]
                    if len(group) > 1:  # Need at least 2 samples per group
                        category_groups.append(group)
                
                if len(category_groups) >= 2:
                    # Calculate ANOVA
                    f_stat, p_value = f_oneway(*category_groups)
                    result["ANOVA_F"] = f_stat
                    result["ANOVA_P_Value"] = p_value
                    result["Significant_Difference"] = "Yes" if p_value < 0.05 else "No"
                    
                    # Calculate categorical R-squared (eta squared)
                    # Sum of squares between groups / total sum of squares
                    group_means = [group.mean() for group in category_groups]
                    group_sizes = [len(group) for group in category_groups]
                    grand_mean = y_clean.mean()
                    
                    # Between groups sum of squares
                    ss_between = sum(size * ((mean - grand_mean) ** 2) for size, mean in zip(group_sizes, group_means))
                    
                    # Total sum of squares
                    ss_total = sum((y - grand_mean) ** 2 for y in y_clean)
                    
                    # Calculate eta squared if ss_total is not zero
                    if ss_total > 0:
                        eta_squared = ss_between / ss_total
                        result["Eta_Squared"] = eta_squared
                        
                        # Determine strength based on eta squared
                        if eta_squared >= 0.26:
                            result["Strength"] = "Strong"
                        elif eta_squared >= 0.13:
                            result["Strength"] = "Moderate"
                        elif eta_squared >= 0.02:
                            result["Strength"] = "Weak"
                        else:
                            result["Strength"] = "Very Weak"
                        
                        # Determine relationship type
                        if result["Significant_Difference"] == "Yes":
                            result["Relationship_Type"] = "Category Effect"
                        else:
                            result["Relationship_Type"] = "Weak/None"
                    else:
                        result["Relationship_Type"] = "Undetermined"
                        result["Strength"] = "Undetermined"
                else:
                    result["Relationship_Type"] = "Insufficient Data"
                    result["Strength"] = "Undetermined"
            except Exception as e:
                result["Relationship_Type"] = "Error in Analysis"
                result["Strength"] = "Undetermined"
                result["Error"] = str(e)
        
        # For classification (binary or multi-class) or categorical target, use Chi-Square test
        elif (is_binary_classification or is_multiclass_classification) or not is_target_numeric:
            try:
                from scipy.stats import chi2_contingency
                
                # Handle binary/multi-class classification case specifically
                if (is_binary_classification or is_multiclass_classification) and is_target_numeric:
                    # Convert numeric target to categorical for crosstab
                    unique_vals = y_clean.unique()
                    y_clean = y_clean.astype(str)  # Convert to string for crosstab
                
                # Create contingency table
                contingency = pd.crosstab(X_clean, y_clean)
                
                # Check if contingency table is valid
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    result["Relationship_Type"] = "Insufficient Categories"
                    result["Strength"] = "Undetermined"
                    results.append(result)
                    continue
                
                # Run chi-square test
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                
                result["Chi2"] = chi2
                result["Chi2_P_Value"] = p_value
                result["DOF"] = dof
                result["Significant_Association"] = "Yes" if p_value < 0.05 else "No"
                
                # Calculate Cramer's V for effect size
                n = contingency.sum().sum()
                phi2 = chi2 / n
                r, k = contingency.shape
                phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                rcorr = r - ((r-1)**2)/(n-1)
                kcorr = k - ((k-1)**2)/(n-1)
                
                if min(kcorr-1, rcorr-1) == 0:
                    cramers_v = 0
                else:
                    cramers_v = np.sqrt(phi2corr / min(kcorr-1, rcorr-1))
                
                result["Cramers_V"] = cramers_v
                
                # Flag to track if Eta Squared was successfully calculated
                has_eta_squared = False
                
                # Calculate Eta Squared for classification if target is numeric
                # This provides an additional effect size measure
                if (is_binary_classification or is_multiclass_classification) and is_target_numeric:
                    try:
                        # Convert back to numeric for ANOVA-based calculation
                        y_numeric = y_clean.astype(float) if y_clean.dtype == 'object' else y_clean
                        
                        # Group by feature categories
                        category_groups = []
                        for cat in X_clean.unique():
                            group = y_numeric[X_clean == cat]
                            if len(group) > 1:  # Need at least 2 samples per group
                                category_groups.append(group)
                        
                        if len(category_groups) >= 2:
                            # Calculate categorical R-squared (eta squared)
                            # Sum of squares between groups / total sum of squares
                            group_means = [group.mean() for group in category_groups]
                            group_sizes = [len(group) for group in category_groups]
                            grand_mean = y_numeric.mean()
                            
                            # Between groups sum of squares
                            ss_between = sum(size * ((mean - grand_mean) ** 2) for size, mean in zip(group_sizes, group_means))
                            
                            # Total sum of squares
                            ss_total = sum((y - grand_mean) ** 2 for y in y_numeric)
                            
                            # Calculate eta squared if ss_total is not zero
                            if ss_total > 0:
                                eta_squared = ss_between / ss_total
                                result["Eta_Squared"] = eta_squared
                                has_eta_squared = True
                    except Exception as e:
                        # If Eta Squared calculation fails, just continue without it
                        pass
                
                # For non-numeric classification target, convert to numeric and calculate Eta Squared
                elif (is_binary_classification or is_multiclass_classification) and not is_target_numeric:
                    try:
                        # Convert categorical target to numeric (0, 1)
                        unique_vals = y_clean.unique()
                        y_mapping = {val: i for i, val in enumerate(unique_vals)}
                        y_numeric = y_clean.map(y_mapping)
                        
                        # Group by feature categories
                        category_groups = []
                        for cat in X_clean.unique():
                            group = y_numeric[X_clean == cat]
                            if len(group) > 1:  # Need at least 2 samples per group
                                category_groups.append(group)
                        
                        if len(category_groups) >= 2:
                            # Calculate categorical R-squared (eta squared)
                            # Sum of squares between groups / total sum of squares
                            group_means = [group.mean() for group in category_groups]
                            group_sizes = [len(group) for group in category_groups]
                            grand_mean = y_numeric.mean()
                            
                            # Between groups sum of squares
                            ss_between = sum(size * ((mean - grand_mean) ** 2) for size, mean in zip(group_sizes, group_means))
                            
                            # Total sum of squares
                            ss_total = sum((y - grand_mean) ** 2 for y in y_numeric)
                            
                            # Calculate eta squared if ss_total is not zero
                            if ss_total > 0:
                                eta_squared = ss_between / ss_total
                                result["Eta_Squared"] = eta_squared
                                has_eta_squared = True
                            else:
                                eta_squared = 0
                    except Exception as e:
                        # If Eta Squared calculation fails, just continue without it
                        pass
                
                # Determine strength based on available metrics for classification
                if (is_binary_classification or is_multiclass_classification) and has_eta_squared:
                    # Use Eta Squared for strength assessment for binary classification
                    eta_squared = result["Eta_Squared"]
                    if eta_squared >= 0.26:
                        result["Strength"] = "Strong"
                    elif eta_squared >= 0.13:
                        result["Strength"] = "Moderate"
                    elif eta_squared >= 0.02:
                        result["Strength"] = "Weak"
                    else:
                        result["Strength"] = "Very Weak"
                else:
                    # Use Cramer's V for all other categorical cases
                    if cramers_v >= 0.5:
                        result["Strength"] = "Strong"
                    elif cramers_v >= 0.3:
                        result["Strength"] = "Moderate"
                    elif cramers_v >= 0.1:
                        result["Strength"] = "Weak"
                    else:
                        result["Strength"] = "Very Weak"
                
                # Determine relationship type
                if result["Significant_Association"] == "Yes":
                    # For classification problems, use class separation term
                    if is_binary_classification or is_multiclass_classification:
                        result["Relationship_Type"] = "Class Separation"
                    else:
                        result["Relationship_Type"] = "Category Association"
                else:
                    result["Relationship_Type"] = "Weak/None"
                
            except Exception as e:
                result["Relationship_Type"] = "Error in Analysis"
                result["Strength"] = "Undetermined"
                result["Error"] = str(e)
        else:
            # Handle any remaining cases that don't fit the above categories
            result["Relationship_Type"] = "Undetermined"
            result["Strength"] = "Undetermined"
        
        results.append(result)
    
    # Create DataFrame from results
    if not results:
        return pd.DataFrame(columns=["Feature", "Relationship_Type", "Strength"])
    
    result_df = pd.DataFrame(results)
    
    # Add feature type column for all features
    if "Feature_Type" not in result_df.columns:
        result_df["Feature_Type"] = "Numerical"
    
    # Set feature type for numerical features
    result_df.loc[~result_df["Feature_Type"].isin(["Categorical"]), "Feature_Type"] = "Numerical"
    
    # Sort by strength (most important first)
    strength_order = {"Strong": 0, "Moderate": 1, "Weak": 2, "Very Weak": 3, "Undetermined": 4}
    if "Strength" in result_df.columns:
        result_df["Strength_Order"] = result_df["Strength"].map(strength_order)
        result_df = result_df.sort_values(["Strength_Order", "Relationship_Type"])
        if "Strength_Order" in result_df.columns:
            result_df = result_df.drop(columns=["Strength_Order"])
    
    # Remove Non_Linear_Relationship column if it exists
    if "Non_Linear_Relationship" in result_df.columns:
        result_df = result_df.drop(columns=["Non_Linear_Relationship"])
    
    # Return all columns in the result DataFrame to ensure metrics are available
    return result_df 

# Function to style and display a table
def style_and_display_table(table, title, display_cols=None, user_level="Beginner"):
    if table.empty:
        st.info(f"No {title.lower()} features found in the analysis.")
        return
    
    # Remove columns with all NaN values to clean up the display
    table_cleaned = table.dropna(axis=1, how='all')
    
    # Always keep these key columns regardless
    key_columns = ['Feature', 'Relationship_Type', 'Strength']
    
    # For categorical features, always keep Eta Squared if it exists in original table
    if title == "Categorical Features":
        if 'Eta_Squared' in table.columns:
            key_columns.append('Eta_Squared')
        if 'Eta Squared' in table.columns:
            key_columns.append('Eta Squared')
    
    for col in key_columns:
        if col in table.columns and col not in table_cleaned.columns:
            table_cleaned[col] = table[col]
    
    # Count removed columns for user feedback
    removed_columns_count = len(table.columns) - len(table_cleaned.columns)
    
    # Ensure columns are in a logical order
    if display_cols is not None:
        available_cols = [col for col in display_cols if col in table_cleaned.columns]
        table_cleaned = table_cleaned[available_cols]
        
    # Style the DataFrame to make it more readable
    styled_table = table_cleaned.style.format(precision=3)
    
    # Replace NaN values with a more readable format
    styled_table = styled_table.format(na_rep="—")
    
    # Apply highlighting to only the Strength column
    styled_table = styled_table.applymap(highlight_strength, subset=['Strength'])
    
    # Apply gradient to numeric columns only for advanced users
    if user_level == "Advanced":
        numeric_cols = table_cleaned.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        if len(numeric_cols) > 0:
            # Apply gradient only to numeric columns
            try:
                styled_table = styled_table.background_gradient(
                    subset=numeric_cols, 
                    cmap='RdYlGn', 
                    low=0.3, 
                    high=0.9
                )
            except:
                # If gradient fails, continue without it
                pass
    
    # Display table with appropriate height and user-specific context
    if user_level == "Beginner":
        st.write(f"#### {title} ({len(table_cleaned)} features) - Simplified View")
        if removed_columns_count > 0:
            st.caption(f"*Simplified view showing essential columns only. Switch to 'Advanced' mode to see all {removed_columns_count + len(table_cleaned.columns)} columns.*")
        else:
            st.caption("*Simplified view showing essential columns only.*")
    else:
        st.write(f"#### {title} ({len(table_cleaned)} features) - Complete Analysis")
        if removed_columns_count > 0:
            st.caption(f"*{removed_columns_count} empty columns were hidden for clarity*")
    
    st.dataframe(
        styled_table,
        width='stretch',
        height=max(200, min(400, 80 + 35 * len(table_cleaned)))  # Better height adjustment
    )
    
    # Add beginner-friendly interpretation help
    if user_level == "Beginner" and len(table_cleaned) > 0:
        # Count features by strength
        strong_features = len(table_cleaned[table_cleaned['Strength'] == 'Strong'])
        moderate_features = len(table_cleaned[table_cleaned['Strength'] == 'Moderate'])
        weak_features = len(table_cleaned[table_cleaned['Strength'] == 'Weak'])
        very_weak_features = len(table_cleaned[table_cleaned['Strength'] == 'Very Weak'])
        
        # Provide actionable insights for beginners
        insights = []
        if strong_features > 0:
            strong_list = table_cleaned[table_cleaned['Strength'] == 'Strong']['Feature'].tolist()
            insights.append(f"🟢 **{strong_features} Strong features**: {', '.join(strong_list[:3])}{'...' if len(strong_list) > 3 else ''} - Definitely include these!")
        
        if moderate_features > 0:
            moderate_list = table_cleaned[table_cleaned['Strength'] == 'Moderate']['Feature'].tolist()
            insights.append(f"🟡 **{moderate_features} Moderate features**: {', '.join(moderate_list[:3])}{'...' if len(moderate_list) > 3 else ''} - Good to include")
        
        if weak_features > 0:
            insights.append(f"🟠 **{weak_features} Weak features**: May still be useful in combination")
        
        if very_weak_features > 0:
            insights.append(f"🔴 **{very_weak_features} Very Weak features**: Consider removing unless you have domain knowledge suggesting they're important")
        
        if insights:
            with st.expander(f"💡 Quick Insights for {title}", expanded=False):
                for insight in insights:
                    st.markdown(f"- {insight}")
                
                if strong_features + moderate_features == 0:
                    st.warning("⚠️ No strong or moderate features found. You may need to:\n- Consider feature engineering\n- Look for data quality issues\n- Verify that your target variable is appropriate")
                elif strong_features + moderate_features >= 3:
                    st.success("✅ You have good predictive features! Focus on the strong and moderate ones for your model.")
                else:
                    st.info("ℹ️ You have some useful features, but consider feature engineering to create more predictive variables.")

# Create a custom styler function to color the Strength column based on values
def highlight_strength(val):
    if val == 'Strong':
        return 'background-color: #a8f0a8'
    elif val == 'Moderate':
        return 'background-color: #f0f0a8'
    elif val == 'Weak':
        return 'background-color: #f0d0a8'
    elif val == 'Very Weak':
        return 'background-color: #f0b0a8'
    return ''