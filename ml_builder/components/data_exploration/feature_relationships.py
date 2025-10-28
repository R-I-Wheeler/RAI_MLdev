import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency, f_oneway, pointbiserialr

# Define correlation thresholds locally to avoid circular import
# These match the centralized values in components.feature_selection.utils.correlation_utils
CORRELATION_DETECTION_THRESHOLD = 0.7   # For identifying correlated pairs
CORRELATION_STRONG_THRESHOLD = 0.8      # For "strong" correlation
CORRELATION_VERY_STRONG_THRESHOLD = 0.9 # For "very strong" correlation

class FeatureRelationshipsComponent:
    """
    Component for analyzing and visualizing relationships between features and with target variable.
    Contains methods for feature association analysis and detailed feature relationship analysis.
    
    This component can be used with data from the Builder class or with custom data.
    
    Example usage with data from Builder:
    ```python
    # Using with Builder data
    feature_relationships = FeatureRelationshipsComponent(builder, logger)
    
    # Get data summary from Builder (can be reused for multiple visualisations)
    summary = builder.get_data_summary()
    
    # Display feature associations using the summary
    feature_relationships.display_feature_associations_analysis(summary)
    
    # Display detailed feature relationship analysis using the summary
    feature_relationships.display_detailed_feature_relationship_analysis(summary)
    ```
    
    Example usage with custom data:
    ```python
    # Using with custom data
    custom_data = pd.DataFrame(...)  # Your custom dataframe
    custom_target = "target_column_name"  # Your custom target column name
    
    custom_feature_relationships = FeatureRelationshipsComponent(
        builder, 
        logger,
        data=custom_data,
        target_column=custom_target
    )
    
    # Auto-generate summary and display feature associations
    custom_feature_relationships.display_feature_associations_analysis()
    
    # Display detailed feature relationship analysis (no summary needed)
    custom_feature_relationships.display_detailed_feature_relationship_analysis()
    ```
    
    Both methods can be used without a summary parameter. In this case:
    - display_feature_associations_analysis will generate its own summary based on the data
    - display_detailed_feature_relationship_analysis will create dynamic visualisations on demand
    
    Key capabilities:
    1. Generate and display feature association heatmaps and matrices
    2. Analyse relationships between any two features with appropriate visualisations
    3. Provide statistical analysis of feature relationships with interpretations
    4. Work with both pre-computed summaries or generate visualisations on-demand
    5. Support both the Builder's data or any custom DataFrame
    """
    
    def __init__(self, builder, logger, data=None, target_column=None):
        """
        Initialize the component with builder and logger instances.
        Optionally accepts custom data and target column.
        
        Args:
            builder: The Builder instance with data and model building methods
            logger: The Logger instance for tracking user actions and errors
            data: Optional pandas DataFrame to use instead of builder.data
            target_column: Optional target column name to use instead of builder.target_column
        """
        self.builder = builder
        self.logger = logger
        self.data = data if data is not None else builder.data
        self.target_column = target_column if target_column is not None else builder.target_column
    
    def display_feature_associations_analysis(self, summary=None):
        """
        Display Feature Associations Analysis section including heatmap and matrix.
        
        Args:
            summary: Dictionary containing data summary and visualisations. If None, summary will be generated
                    from self.data and self.target_column.
        """
        # Generate summary if not provided
        if summary is None:
            # Log that we're generating a summary
            self.logger.log_calculation(
                "Generating Feature Associations Summary",
                {
                    "data_shape": self.data.shape,
                    "custom_data": self.data is not self.builder.data,
                    "timestamp": datetime.now().isoformat()
                }
            )
            summary = self.builder.get_data_summary() if self.data is self.builder.data else self._generate_data_summary()
        
        # Feature Associations section with improved layout
        st.subheader("Feature Correlation/Association")
        
        # Associations heatmap with improved visual layout
        if "associations" in summary.get("visualisations", {}):
            st.markdown("""
            <div style="background-color:#f5fff0; padding:15px; border-radius:10px; margin:10px 0px 15px 0px">
            <h4 style="margin-top:0">📊 Feature Correlation/Association Heatmap</h4>
            <ul>
            <li>Shows relationships between all features</li>
            <li>Works with both numerical and categorical data</li>
            <li>Darker colors indicate stronger relationships</li>
            <li>Useful for understanding feature interactions</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("ℹ️ Understanding Feature Correlation/Association Heatmap", expanded=False):
                explanation = self.builder.get_calculation_explanation("feature_associations")
                
                # Use columns for better organization
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Method:**")
                    st.markdown(explanation["method"])
                with col2:
                    st.markdown(explanation["interpretation"])

            # Make plot container larger and more prominent
            st.plotly_chart(summary["visualisations"]["associations"], config={'responsive': True})
            
            # Add null check for associations_matrix
            #associations_matrix = summary.get("associations_matrix")
            #if associations_matrix and "data" in associations_matrix:
            #    st.write("### Detailed Associations Matrix")
                # Display the regular DataFrame with improved styling
            #    st.dataframe(
            #        associations_matrix["data"],
            #        width='stretch',
            #        height=400
            #    )
        else:
            st.warning("""
                Could not generate associations visualization. This might happen with very large datasets 
                or when there are issues with the data types.
                
                Common issues:
                - Mixed data types in columns
                - Too many unique values in categorical columns
                - Missing values
                - Memory constraints with large datasets
                
                Check the console for detailed error messages.
            """)
    
    def _generate_data_summary(self):
        """
        Generate a data summary similar to Builder.get_data_summary but using self.data and self.target_column.
        This allows the component to work with custom data without modifying the Builder instance.
        
        Returns:
            Dictionary containing summary statistics and visualisations
        """
        try:
            import numpy as np
            import plotly.express as px
            from dython.nominal import identify_nominal_columns, associations
            
            # Exclude target variable from all analyses
            feature_columns = [col for col in self.data.columns if col != self.target_column]
            
            # Initialize summary dictionary
            summary = {}
            
            # Create visualisations dictionary
            figs = {}
            
            # Feature Associations Analysis using dython
            try:
                # Use feature columns (excluding target)
                data_for_associations = self.data[feature_columns].copy()
                
                # Identify categorical columns (excluding target)
                categorical_features = identify_nominal_columns(data_for_associations)
                
                # Create a copy of the data for associations
                data_for_assoc = data_for_associations.copy()
                
                # Convert all object columns to category
                for col in data_for_assoc.select_dtypes(['object']):
                    data_for_assoc[col] = data_for_assoc[col].astype('category')
                
                # Handle any missing values - use mode for each column
                for col in data_for_assoc.columns:
                    if data_for_assoc[col].isnull().any():
                        col_mode = data_for_assoc[col].mode()
                        if not col_mode.empty:
                            data_for_assoc[col] = data_for_assoc[col].fillna(col_mode.iloc[0])
                
                # Calculate associations
                complete_correlation = associations(
                    data_for_assoc,
                    nominal_columns=categorical_features,
                    plot=False,
                    compute_only=True
                )
                
                # Get correlation matrix from the dictionary
                df_complete_corr = pd.DataFrame(
                    complete_correlation['corr'],
                    index=data_for_assoc.columns,
                    columns=data_for_assoc.columns
                )
                
                # Convert to numpy array and handle any non-finite values
                corr_values = df_complete_corr.values
                corr_values = np.nan_to_num(corr_values, nan=0)
                
                # Create heatmap using plotly
                associations_fig = px.imshow(
                    corr_values,
                    x=df_complete_corr.columns,
                    y=df_complete_corr.index,
                    color_continuous_scale='RdBu',
                    aspect='auto',
                    title='Feature Associations Heatmap',
                    zmin=-1,
                    zmax=1
                )
                
                # Update layout
                associations_fig.update_layout(
                    height=800,
                    width=800,
                    xaxis_tickangle=-45
                )
                
                # Add text annotations
                associations_fig.update_traces(
                    text=np.round(corr_values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10, "color": "white"}
                )
                
                # Add to visualisations
                figs["associations"] = associations_fig
                
                # Convert the correlation matrix to a regular DataFrame with rounded values
                df_display = pd.DataFrame(
                    np.round(corr_values, 2),
                    index=df_complete_corr.index,
                    columns=df_complete_corr.columns
                )
                
                # Store both the styled and regular versions
                summary["associations_matrix"] = {
                    "data": df_display,
                    "styled": df_complete_corr.style.background_gradient(
                        cmap='coolwarm', 
                        axis=None
                    ).set_precision(2)
                }
                
            except Exception as e:
                self.logger.log_error(
                    "Feature Associations Analysis Failed",
                    {"error": str(e)}
                )
                # Don't add to figs if there's an error
            
            return {
                "summary": self.data[feature_columns].describe().to_dict(),
                "dtypes": {col: str(dtype) for col, dtype in self.data[feature_columns].dtypes.to_dict().items()},
                "missing_values": self.data[feature_columns].isnull().sum().to_dict(),
                "visualisations": figs,
                "associations_matrix": summary.get("associations_matrix")
            }
        except Exception as e:
            self.logger.log_error(
                "Data Summary Generation Failed",
                {"error": str(e)}
            )
            return {
                "error": f"Error generating data summary: {str(e)}"
            }
    
    def display_correlation_group_analysis(self, summary=None, correlation_threshold=None):
        """
        Display correlation group analysis that identifies groups of highly correlated features
        and provides recommendations for feature removal.

        Args:
            summary: Dictionary containing data summary and visualisations. If None, will generate correlation matrix
            correlation_threshold: Threshold for identifying highly correlated features (default: CORRELATION_STRONG_THRESHOLD)
        """
        # Use centralized threshold if not specified
        if correlation_threshold is None:
            correlation_threshold = CORRELATION_STRONG_THRESHOLD
        import numpy as np
        import plotly.graph_objects as go
        import plotly.express as px
        import networkx as nx
        from sklearn.cluster import AgglomerativeClustering
        
        # Generate summary if not provided
        if summary is None:
            self.logger.log_calculation(
                "Generating Correlation Analysis Summary",
                {
                    "data_shape": self.data.shape,
                    "custom_data": self.data is not self.builder.data,
                    "correlation_threshold": correlation_threshold,
                    "timestamp": datetime.now().isoformat()
                }
            )
            summary = self.builder.get_data_summary() if self.data is self.builder.data else self._generate_data_summary()
        
        
        
        # Get correlation matrix from summary, with fallback options
        correlation_matrix = None
        method_used = "unknown"
        
        # Debug: Check what's actually in the summary
        debug_info = {
            "summary_keys": list(summary.keys()) if summary else [],
            "has_associations_matrix": "associations_matrix" in summary if summary else False,
            "has_visualizations": "visualisations" in summary if summary else False,
            "associations_in_viz": "associations" in summary.get("visualisations", {}) if summary else False,
        }
        
        # Add detailed analysis of associations_matrix structure
        if summary and "associations_matrix" in summary:
            associations_matrix = summary["associations_matrix"]
            debug_info["associations_matrix_structure"] = {
                "type": str(type(associations_matrix)),
                "is_dict": isinstance(associations_matrix, dict),
                "is_none": associations_matrix is None,
                "content": str(associations_matrix)[:200] if associations_matrix else "None"
            }
            
            if isinstance(associations_matrix, dict):
                debug_info["associations_matrix_keys"] = list(associations_matrix.keys())
                debug_info["has_data_key"] = "data" in associations_matrix
                if "data" in associations_matrix:
                    data_obj = associations_matrix["data"]
                    debug_info["data_structure"] = {
                        "type": str(type(data_obj)),
                        "shape": data_obj.shape if hasattr(data_obj, 'shape') else "No shape attribute",
                        "is_dataframe": isinstance(data_obj, pd.DataFrame)
                    }
        else:
            debug_info["associations_matrix_structure"] = "Not present in summary"
            
        self.logger.log_calculation("Summary Content Debug", debug_info)
        
        # Check if the associations were successfully generated for the heatmap
        has_dython_associations = (summary and 
                                 "visualisations" in summary and 
                                 "associations" in summary.get("visualisations", {}))
        
        # First, try to get from summary
        if summary and "associations_matrix" in summary and summary["associations_matrix"]:
            associations_matrix = summary["associations_matrix"]
            if isinstance(associations_matrix, dict) and "data" in associations_matrix:
                correlation_matrix = associations_matrix["data"]
                method_used = "associations_matrix (dython)"
                self.logger.log_calculation(
                    "Associations Matrix Found",
                    {
                        "matrix_shape": correlation_matrix.shape,
                        "matrix_type": str(type(correlation_matrix)),
                        "method_used": method_used
                    }
                )
            else:
                self.logger.log_error(
                    "Associations Matrix Invalid Structure",
                    {
                        "associations_matrix_type": str(type(associations_matrix)),
                        "associations_matrix_content": str(associations_matrix) if associations_matrix else "None"
                    }
                )
        
        # If that fails, try to generate a simple correlation matrix for numeric features only
        if correlation_matrix is None:
            # Provide more specific messaging based on whether dython was attempted
            if has_dython_associations:
                st.info("🔄 Dython associations were generated for the heatmap but correlation matrix data is not accessible. Generating fallback correlation matrix...")
            else:
                st.info("🔄 Feature associations matrix not available (dython associations may have failed). Generating correlation matrix for numeric features...")
            
            correlation_matrix, method_used = self._generate_simple_correlation_matrix()
        
        if correlation_matrix is None:
            st.error("""
            ❌ **Could not generate correlation matrix**
            
            This could be due to several reasons:
            - No numeric features in the dataset
            - Too many missing values
            - Data type issues
            - Insufficient data for correlation analysis
            
            **Troubleshooting steps:**
            1. Ensure your dataset has at least 2 numeric features
            2. Check for and handle missing values
            3. Run the Feature Associations Analysis first
            4. Verify your data types are appropriate
            """)
            return
        
        # Display information about the correlation method used
        #st.info(f"📊 Using correlation matrix from: **{method_used}** | Matrix size: **{correlation_matrix.shape[0]} × {correlation_matrix.shape[1]}** features")
        
        st.divider()
        # Always run low information quality analysis regardless of correlation groups
        st.markdown("### 🔍 Low Information Quality Features Analysis")
        
        with st.expander("ℹ️ Understanding Low Information Quality Analysis", expanded=False):
            st.markdown("""
            ### 🎯 Purpose
            Beyond removing redundant features, we should also consider removing features that provide little predictive value.
            
            ### 📊 What This Analysis Identifies
            - **Low Target Correlation**: Features with very weak relationships to the target variable
            - **Low Variance**: Features with little variation (mostly constant values)
            - **High Missing Values**: Features with excessive missing data
            - **Weak Overall Relationships**: Features that don't correlate strongly with anything
            
            ### 💡 Why Remove These Features
            - **Reduce Noise**: Low-quality features can add noise without signal
            - **Improve Performance**: Fewer irrelevant features often lead to better model performance
            - **Faster Training**: Removing uninformative features speeds up model training
            - **Better Generalization**: Models with fewer, higher-quality features often generalize better
            
            ### 🔍 How Quality Metrics Are Calculated
            
            Each feature is evaluated using multiple quality factors that are combined into an overall quality score:
            
            #### **Quality Score (0.0 - 1.0)**
            - **Calculation**: Weighted combination of all quality factors below
            - **Interpretation**: 
              - 0.7+ = Good quality (Monitor)
              - 0.5-0.7 = Moderate issues (High priority)
              - 0.3-0.5 = Poor quality (High priority)
              - <0.3 = Very poor quality (Critical - recommended for removal)
            - **Purpose**: Overall assessment of feature's information value
            
            #### **Target Correlation/Association (0.0 - 1.0)**
            - **For Numeric Features**: Pearson correlation coefficient with target
            - **For Categorical Features**: Association strength using Cramér's V or similar measure
            - **Interpretation**:
              - 0.0-0.1 = Very weak relationship
              - 0.1-0.3 = Weak relationship
              - 0.3-0.5 = Moderate relationship
              - 0.5+ = Strong relationship
            - **Weight in Quality Score**: 40% (most important factor)
            
            #### **Average Correlation with Other Features (0.0 - 1.0)**
            - **Calculation**: Mean absolute correlation/association with all other features
            - **Interpretation**:
              - Low values indicate the feature is isolated/unique
              - High values may indicate redundancy with other features
              - Very low values (<0.05) suggest the feature doesn't relate to anything
            - **Weight in Quality Score**: 20%
            
            #### **Missing % (0.0 - 100.0)**
            - **Calculation**: Percentage of rows with missing/null values
            - **Interpretation**:
              - 0-10%: Acceptable missing data
              - 10-30%: Moderate missing data (may impact model)
              - 30-50%: High missing data (significant concern)
              - 50%+: Very high missing data (major issue)
            - **Weight in Quality Score**: 15% (penalty for missing data)
            
            #### **Variance/Diversity Score (0.0 - 1.0)**
            - **For Numeric Features**: Normalized variance (higher = more variation)
            - **For Categorical Features**: Entropy-based diversity measure
            - **Interpretation**:
              - Near 0: Feature has very little variation (mostly constant)
              - 0.1-0.3: Low variation (limited information)
              - 0.3+: Good variation (informative)
            - **Weight in Quality Score**: 25%
            
            #### **Severity Classification**
            - **🚨 Critical (Quality Score < 0.3)**: Strong removal candidates
            - **⚠️ High (Quality Score 0.3-0.5)**: Consider removal or improvement
            - **⚡ Monitor (Quality Score 0.5-0.7)**: Watch for potential issues
            
            #### **Primary Issue**
            - **Identifies**: The factor contributing most to the low quality score
            - **Common Issues**:
              - "Weak target association": Poor predictive value
              - "Low variance/diversity": Feature is mostly constant
              - "High missing values": Too much missing data
              - "Weak feature associations": Doesn't relate to other features
            
            ### 🎯 How to Use These Metrics
            
            1. **Focus on Critical features first** (🚨) - these are the safest to remove
            2. **Check Primary Issue** to understand why a feature scores poorly
            3. **Consider Target Correlation** - features with very low target correlation (<0.05) add little predictive value
            4. **Evaluate Missing %** - features with >50% missing data are problematic
            5. **Look at Variance** - features with very low variance (<0.1) may be mostly constant
            
            ### ⚖️ Quality Score Weighting Logic
            
            The overall quality score prioritizes:
            1. **Target Relationship (40%)** - Most important for prediction
            2. **Variance/Diversity (25%)** - Need variation to be informative  
            3. **Overall Associations (20%)** - Should relate to something in the dataset
            4. **Missing Data Penalty (15%)** - Too much missing data is problematic
            
            This weighting ensures features that can't predict the target or have no variation get low scores.
            """)
        
        low_quality_recommendations = self._analyze_low_information_features(correlation_matrix)
        
        if low_quality_recommendations:
            st.markdown("#### 🔍 Low Information Quality Features Analysis Results")
            
            # Add clarification about removal recommendations
            st.info("📋 **Analysis includes all features with quality issues for monitoring. Only Critical features (🚨) are recommended for removal in the combined strategy.**")
            
            # Create a comprehensive table
            low_quality_df = pd.DataFrame(low_quality_recommendations)
            if not low_quality_df.empty:
                # Sort by priority (lower score = higher priority for removal)
                low_quality_df = low_quality_df.sort_values('quality_score', ascending=True)
                
                # Format for display
                display_df = low_quality_df.copy()
                display_df['Quality Score'] = display_df['quality_score'].round(3)
                display_df['Target Correlation'] = display_df['target_correlation'].round(3)
                display_df['Avg Correlation'] = display_df['avg_correlation'].round(3)
                display_df['Missing %'] = (display_df['missing_percentage'] * 100).round(1)
                display_df['Variance'] = display_df['variance'].round(4)
                
                # Add severity level for color coding
                display_df['Severity'] = display_df['quality_score'].apply(
                    lambda x: '🚨 Critical' if x < 0.3 else '⚠️ High' if x < 0.5 else '⚡ Monitor'
                )
                
                # Create recommendation text - make it more concise
                display_df['Recommendation'] = display_df.apply(
                    lambda row: f"{row['recommendation'][:50]}..." if len(row['recommendation']) > 50 else row['recommendation'], 
                    axis=1
                )
                
                # Select columns for display
                final_df = display_df[[
                    'feature', 'Severity', 'Quality Score', 'primary_issue', 
                    'Target Correlation', 'Avg Correlation', 'Missing %', 'Variance', 'Recommendation'
                ]].copy()
                
                final_df.columns = [
                    'Feature', 'Severity', 'Quality Score', 'Primary Issue', 
                    'Target Correlation', 'Avg Correlation', 'Missing %', 'Variance', 'Recommendation'
                ]
                
                # Apply color coding using styling
                def color_severity(val):
                    if '🚨' in str(val):
                        return 'background-color: #ffebee; color: #c62828; font-weight: bold'
                    elif '⚠️' in str(val):
                        return 'background-color: #fff3e0; color: #ef6c00; font-weight: bold'
                    elif '⚡' in str(val):
                        return 'background-color: #e8f5e8; color: #2e7d32; font-weight: bold'
                    return ''
                
                def color_quality_score(val):
                    try:
                        score = float(val)
                        if score < 0.3:
                            return 'background-color: #ffcdd2; color: #d32f2f; font-weight: bold'
                        elif score < 0.5:
                            return 'background-color: #ffe0b2; color: #f57c00; font-weight: bold'
                        elif score < 0.7:
                            return 'background-color: #dcedc8; color: #689f38'
                        return ''
                    except:
                        return ''
                
                # Style the dataframe with better text wrapping
                styled_df = final_df.style.applymap(
                    color_severity, subset=['Severity']
                ).applymap(
                    color_quality_score, subset=['Quality Score']
                ).set_properties(**{
                    'text-align': 'left',
                    'white-space': 'pre-wrap',
                    'word-wrap': 'break-word',
                    'max-width': '200px'
                }).set_properties(subset=['Recommendation'], **{
                    'max-width': '300px',
                    'min-width': '200px'
                })
                
                # Display the styled table with increased height for better readability
                st.dataframe(styled_df, width='stretch', height=500)
                
                # Add detailed recommendations in an expandable section
                with st.expander("📋 View Full Recommendations and Issues Details", expanded=False):
                    st.markdown("**Complete recommendations and identified issues for each feature:**")
                    for _, row in display_df.iterrows():
                        severity_emoji = "🚨" if row['quality_score'] < 0.3 else "⚠️" if row['quality_score'] < 0.5 else "⚡"
                        st.markdown(f"""
                        **{severity_emoji} {row['feature']}** (Quality Score: {row['quality_score']:.3f})
                        - **Recommendation:** {row['recommendation']}
                        - **Issues:** {', '.join(row['issues']) if row['issues'] else 'None identified'}
                        """)
                        st.markdown("---")
                
                # Provide interpretation
                st.markdown("""
                **How to interpret this table:**
                - **Severity**: 🚨 Critical (recommended for removal), ⚠️ High (monitor closely), ⚡ Monitor (consider improvement)
                - **Quality Score**: 0-1 scale, lower = worse quality (color coded: red=critical, orange=high, green=monitor)
                - **Primary Issue**: Main reason for low quality score
                - **Target/Avg Correlation**: Relationship strengths (higher = more informative)
                - **Missing %**: Percentage of missing values (lower = better)
                - **Variance**: Measure of feature variation (higher = more informative)
                - **Recommendation**: Specific action and key issues identified
                
                **Note:** Only Critical features (🚨) are included in the final removal recommendations below.
                """)
                
                # Summary statistics
                critical_count = len(display_df[display_df['quality_score'] < 0.3])
                high_count = len(display_df[(display_df['quality_score'] >= 0.3) & (display_df['quality_score'] < 0.5)])
                monitor_count = len(display_df[display_df['quality_score'] >= 0.5])
                
                st.markdown("### 📊 Quality Issues Summary")
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("🚨 Critical Issues", critical_count, help="Features with quality score < 0.3")
                with summary_col2:
                    st.metric("⚠️ High Priority Issues", high_count, help="Features with quality score 0.3-0.5")
                with summary_col3:
                    st.metric("⚡ Monitor", monitor_count, help="Features with quality score 0.5-0.7")
        else:
            st.success("✅ No features identified as having low information quality!")
        
        st.divider()
        st.markdown("### 📊 Feature Correlation/Association Groups Analysis")
        st.markdown("This analysis identifies groups of highly correlated features that may contain redundant information.")
        
        # Add explanation
        with st.expander("ℹ️ Understanding Correlation Groups Analysis", expanded=False):
            st.markdown("""
            ### 🎯 Purpose
            This analysis identifies groups of highly correlated features that may contain redundant information.
            
            ### 📊 What This Analysis Shows
            - **Correlation Groups**: Clusters of features that are highly correlated with each other
            - **Network Visualization**: Shows relationships between features as a network graph
            - **Feature Importance**: Ranks features within each group by their relationship to the target
            - **Removal Recommendations**: Suggests which features to consider removing to reduce redundancy
            
            ### 🔍 Why This Matters
            - **Reduce Overfitting**: Too many correlated features can cause models to overfit
            - **Improve Performance**: Removing redundant features can improve model speed and accuracy
            - **Better Interpretability**: Fewer, more distinct features are easier to understand
            - **Avoid Multicollinearity**: Highly correlated features can cause instability in some models
            
            ### 💡 How to Use Results
            - **Keep one representative** from each correlation group
            - **Prioritize features** with stronger target relationships
            - **Consider domain knowledge** when deciding which features to remove
            """)
        # Configure correlation threshold
        st.markdown("#### 🎛️ Configuration")
        col1, col2 = st.columns(2)
        with col1:
            correlation_threshold = st.slider(
                "Correlation Threshold",
                min_value=0.5,
                max_value=0.95,
                value=correlation_threshold,
                step=0.05,
                help=f"Features with correlation above this threshold will be grouped together (Default: {CORRELATION_STRONG_THRESHOLD})"
            )
        with col2:
            min_group_size = st.slider(
                "Minimum Group Size",
                min_value=2,
                max_value=5,
                value=2,
                help="Minimum number of features required to form a correlation group"
            )
        
        # Find correlation groups
        groups, group_info = self._find_correlation_groups(
            correlation_matrix, 
            correlation_threshold, 
            min_group_size
        )
        
        # Safety check: Remove target variable from any groups (should not happen but being extra safe)
        if self.target_column:
            groups = [[feature for feature in group if feature != self.target_column] for group in groups]
            # Filter out any groups that became too small after removing target
            groups = [group for group in groups if len(group) >= min_group_size]

        # Now handle correlation groups analysis results
        if not groups:
            st.markdown("#### 📊 Correlation Groups Analysis")
            st.info(f"No correlation groups found with threshold {correlation_threshold:.2f}. Try lowering the threshold to find weaker correlations.")
            # Set empty lists for correlation-based recommendations
            removal_recommendations = []
            features_to_keep = []
            features_to_remove = []
        else:
            # Display correlation groups results
            st.markdown("#### 📊 Correlation Groups Found")
            st.write(f"Found **{len(groups)}** correlation groups with {sum(len(group) for group in groups)} features total.")
            
            # Create network visualization
            network_fig = self._create_correlation_network(correlation_matrix, correlation_threshold, groups)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(network_fig, config={'responsive': True})
            
            with col2:
                # Create correlation strength heatmap for identified groups
                group_heatmap = self._create_group_correlation_heatmap(correlation_matrix, groups)
                st.plotly_chart(group_heatmap, config={'responsive': True})
            
            # Analyze each group
            st.markdown("#### 🔍 Detailed Group Analysis")
            
            # Add methodology explanation before the group analysis (avoiding nested expanders)
            with st.expander("📖 Understanding the Recommendation Methodology", expanded=False):
                st.markdown("""
                **🎯 How Recommendations Are Calculated:**
                
                This analysis uses a **sophisticated two-factor approach** to determine which features to keep/remove:
                
                **1. PRIMARY FACTOR: Redundancy Score (70% weight)**
                - Average absolute correlation with other features in this group
                - Higher score = more redundant = more likely to remove
                - Range: 0.0 (unique) to 1.0 (completely redundant)
                
                **2. SECONDARY FACTOR: Target Correlation (30% weight)**
                - Absolute correlation with target variable
                - Higher score = more predictive = less likely to remove
                - Used as tie-breaker when redundancy scores are similar
                
                **3. COMBINED REMOVAL SCORE:**
                ```
                Removal Score = (0.7 × Redundancy) - (0.3 × Target Correlation)
                ```
                - Higher removal score = higher priority for removal
                - The feature with the **lowest removal score** is kept
                
                **💡 Why this approach is optimal:**
                - **Preserves Information**: Keeps the feature that shares least information with others
                - **Minimizes Redundancy**: Removes features that overlap most with group members
                - **Considers Predictive Power**: Still accounts for target relationships as secondary factor
                - **Prevents Information Loss**: Avoids removing features that contain unique information
                
                **📊 How to interpret the scores:**
                - **Low Redundancy Score** (< 0.5): Feature has unique information
                - **High Redundancy Score** (> 0.8): Feature is highly overlapped with others
                - **Removal Score**: Negative values favor keeping, positive values favor removing
                
                **🔬 Information Theory Basis:**
                Features with high redundancy scores provide overlapping information. Keeping the least redundant feature maximizes information preservation while minimizing multicollinearity.
                """)
            
            removal_recommendations = []
            
            for i, group in enumerate(groups):
                with st.expander(f"📁 Group {i+1}: {len(group)} features", expanded=True):
                    st.write(f"**Features in this group:** {', '.join(group)}")
                    
                    # Calculate group statistics
                    group_stats = self._analyze_group_relationships(group, correlation_matrix)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📈 Correlation Statistics:**")
                        stats_df = pd.DataFrame([group_stats]).T
                        stats_df.columns = ['Value']
                        st.dataframe(stats_df, width='stretch')
                    
                    with col2:
                        # Feature-target correlations for ranking
                        target_correlations = self._get_target_correlations(group)
                        if target_correlations:
                            st.write("**🎯 Target Correlations:**")
                            target_df = pd.DataFrame(list(target_correlations.items()), 
                                                   columns=['Feature', 'Target Correlation'])
                            target_df['Abs Correlation'] = target_df['Target Correlation'].abs()
                            target_df = target_df.sort_values('Abs Correlation', ascending=False)
                            st.dataframe(target_df[['Feature', 'Target Correlation']], width='stretch')
                        else:
                            st.info("Target correlations not available (target may not be numeric)")
                    
                    # Generate removal recommendations for this group
                    group_recommendations = self._generate_group_recommendations(
                        group, correlation_matrix, target_correlations
                    )
                    removal_recommendations.extend(group_recommendations)
                    
                    # Display detailed recommendation analysis table
                    if group_recommendations:
                        st.write("**🧮 Detailed Recommendation Analysis:**")
                        
                        # Create a comprehensive analysis table
                        rec_analysis = []
                        for rec in group_recommendations:
                            rec_analysis.append({
                                'Feature': rec['feature'],
                                'Action': '✅ Keep' if rec['action'] == 'keep' else '⚠️ Remove',
                                'Redundancy Score': f"{rec.get('redundancy_score', 0):.3f}",
                                'Target Correlation': f"{rec.get('target_correlation', 0):.3f}" if rec.get('target_correlation', 0) > 0 else 'N/A',
                                'Removal Score': f"{rec.get('removal_score', 0):.3f}",
                                'Primary Reason': rec['reason']
                            })
                        
                        rec_df = pd.DataFrame(rec_analysis)
                        st.dataframe(rec_df, width='stretch')
                        
                        # Add a simple info note about the methodology (no nested expander)
                        st.info("💡 **Tip:** Refer to the 'Understanding the Recommendation Methodology' section above for details on how these scores are calculated.")
                    
                    # Display recommendations for this group with enhanced styling
                    if group_recommendations:
                        st.write("**💡 Recommendations for this group:**")
                        for rec in group_recommendations:
                            if rec['action'] == 'keep':
                                st.success(f"✅ **Keep** {rec['feature']}: {rec['reason']}")
                            else:
                                st.warning(f"⚠️ **Consider removing** {rec['feature']}: {rec['reason']}")
            
            features_to_keep = [rec['feature'] for rec in removal_recommendations if rec['action'] == 'keep']
            features_to_remove = [rec['feature'] for rec in removal_recommendations if rec['action'] == 'remove']
        
        # Extract features_to_keep and features_to_remove for the case where no groups were found
        if not groups:
            features_to_keep = []
            features_to_remove = []
        
        st.divider()
        # Combined summary
        st.markdown("### 📊 Combined Feature Removal Strategy")
        
        # Get all unique features recommended for removal
        correlation_removals = set(features_to_remove)
        quality_removals = set([rec['feature'] for rec in low_quality_recommendations if rec['quality_score'] < 0.3])
        
        # Safety check: Remove target variable from removal recommendations (should not happen but being extra safe)
        if self.target_column:
            correlation_removals.discard(self.target_column)
            quality_removals.discard(self.target_column)
        
        all_removals = correlation_removals.union(quality_removals)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.warning(f"**⚠️ Remove due to CORRELATION ({len(correlation_removals)} features):**")
            if correlation_removals:
                for feature in correlation_removals:
                    st.write(f"• {feature}")
            else:
                st.write("• No correlation-based removals identified")
                
        with col2:
            st.error(f"**🚨 Remove due to LOW QUALITY ({len(quality_removals)} features):**")
            if quality_removals:
                for feature in quality_removals:
                    st.write(f"• {feature}")
            else:
                st.write("• No low-quality features identified")
        
        # Overall impact analysis
        st.markdown("### 📈 Overall Feature Reduction Impact")
        
        total_features = len(correlation_matrix.columns)
        features_in_groups = sum(len(group) for group in groups) if groups else 0
        correlated_to_remove = len(correlation_removals)
        total_potential_reduction = len(all_removals)
        
        impact_col1, impact_col2, impact_col3, impact_col4 = st.columns(4)
        
        with impact_col1:
            st.metric("Total Features", total_features)
        with impact_col2:
            st.metric("Correlated Features", correlated_to_remove)
        with impact_col3:
            st.metric("Low Quality Features", len(quality_removals))
        with impact_col4:
            reduction_pct = (total_potential_reduction / total_features) * 100 if total_features > 0 else 0
            st.metric("Total Potential Reduction", f"{total_potential_reduction} ({reduction_pct:.1f}%)")
        
        if total_potential_reduction > 0:
            # create journey point for feature removal recommendations
            self.logger.log_journey_point(
                    stage="DATA_EXPLORATION",
                    decision_type="FEATURE_REMOVAL_RECOMMENDATIONS",
                    description="Feature removal recommendations",
                    details={'Correlation Removals': list(correlation_removals), 
                            'Quality Removals': list(quality_removals),
                            'Total Potential Reduction': total_potential_reduction,
                            'Reduction Percentage': reduction_pct,
                            'Correlation Method': method_used},
                    parent_id=None
                )
        
        # Log the analysis results
        self.logger.log_calculation(
            "Correlation Groups Analysis Results",
            {
                "correlation_threshold": correlation_threshold,
                "groups_found": len(groups),
                "total_features": total_features,
                "features_in_groups": features_in_groups,
                "features_to_keep": len(features_to_keep),
                "correlation_removals": len(correlation_removals),
                "quality_removals": len(quality_removals),
                "total_potential_reduction": total_potential_reduction,
                "reduction_percentage": reduction_pct,
                "correlation_method": method_used,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _generate_simple_correlation_matrix(self):
        """
        Generate a correlation matrix with basic support for mixed data types.
        This is a fallback when the full associations matrix is not available.
        
        Returns:
            Tuple of (correlation_matrix, method_used) or (None, "failed")
        """
        try:
            import numpy as np
            from sklearn.preprocessing import LabelEncoder
            
            # Get all columns except target for analysis
            analysis_columns = [col for col in self.data.columns if col != self.target_column]
            
            if len(analysis_columns) < 2:
                self.logger.log_error(
                    "Simple Correlation Matrix Generation Failed",
                    {"error": "Less than 2 features available for analysis", "columns": len(analysis_columns)}
                )
                return None, "failed - insufficient features"
            
            # Separate numeric and categorical columns
            numeric_columns = self.data[analysis_columns].select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = self.data[analysis_columns].select_dtypes(include=['object', 'category']).columns.tolist()
            
            self.logger.log_calculation(
                "Feature Type Analysis",
                {
                    "total_features": len(analysis_columns),
                    "numeric_features": len(numeric_columns),
                    "categorical_features": len(categorical_columns)
                }
            )
            
            # Strategy 1: Try mixed-type correlation if we have both types
            if len(numeric_columns) >= 1 and len(categorical_columns) >= 1:
                try:
                    correlation_matrix = self._generate_mixed_type_correlation(analysis_columns, numeric_columns, categorical_columns)
                    if correlation_matrix is not None:
                        return correlation_matrix, "mixed-type correlation (label encoded)"
                except Exception as e:
                    self.logger.log_error("Mixed-type correlation failed", {"error": str(e)})
            
            # Strategy 2: Numeric-only correlation (original fallback)
            if len(numeric_columns) >= 2:
                # Remove columns with all NaN or constant values
                valid_numeric_columns = []
                for col in numeric_columns:
                    col_data = self.data[col].dropna()
                    if len(col_data) > 1 and col_data.std() > 0:  # Has variation
                        valid_numeric_columns.append(col)
                
                if len(valid_numeric_columns) >= 2:
                    # Calculate correlation matrix for numeric features
                    correlation_matrix = self.data[valid_numeric_columns].corr()
                    correlation_matrix = correlation_matrix.fillna(0)
                    
                    self.logger.log_calculation(
                        "Numeric-Only Correlation Matrix Generated",
                        {
                            "original_numeric_columns": len(numeric_columns),
                            "valid_columns": len(valid_numeric_columns),
                            "matrix_shape": correlation_matrix.shape
                        }
                    )
                    
                    return correlation_matrix, "numeric-only correlation (pandas.corr())"
            
            # Strategy 3: Categorical-only using Cramér's V
            if len(categorical_columns) >= 2:
                try:
                    correlation_matrix = self._generate_categorical_correlation(categorical_columns)
                    if correlation_matrix is not None:
                        return correlation_matrix, "categorical-only correlation (Cramér's V)"
                except Exception as e:
                    self.logger.log_error("Categorical correlation failed", {"error": str(e)})
            
            # If all strategies fail
            self.logger.log_error(
                "All Correlation Strategies Failed",
                {
                    "numeric_columns": len(numeric_columns),
                    "categorical_columns": len(categorical_columns),
                    "total_columns": len(analysis_columns)
                }
            )
            return None, "failed - no suitable correlation method"
            
        except Exception as e:
            self.logger.log_error(
                "Simple Correlation Matrix Generation Failed",
                {"error": str(e)}
            )
            return None, "failed - error during generation"

    def _generate_mixed_type_correlation(self, all_columns, numeric_columns, categorical_columns):
        """Generate correlation matrix for mixed data types using label encoding."""
        try:
            # Create a copy of the data for processing
            processed_data = self.data[all_columns].copy()
            
            # Label encode categorical features
            label_encoders = {}
            for col in categorical_columns:
                # Only process columns with reasonable number of categories
                n_unique = processed_data[col].nunique()
                if n_unique > 50:  # Too many categories, skip
                    continue
                    
                le = LabelEncoder()
                # Handle missing values by treating them as a separate category
                processed_data[col] = processed_data[col].fillna('__MISSING__')
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
                label_encoders[col] = le
            
            # Remove columns that couldn't be processed
            valid_columns = []
            for col in all_columns:
                if col in numeric_columns:
                    col_data = processed_data[col].dropna()
                    if len(col_data) > 1 and col_data.std() > 0:
                        valid_columns.append(col)
                elif col in label_encoders:
                    valid_columns.append(col)
            
            if len(valid_columns) < 2:
                return None
            
            # Calculate correlation matrix
            correlation_matrix = processed_data[valid_columns].corr()
            correlation_matrix = correlation_matrix.fillna(0)
            
            self.logger.log_calculation(
                "Mixed-Type Correlation Generated",
                {
                    "original_columns": len(all_columns),
                    "valid_columns": len(valid_columns),
                    "encoded_categorical": len(label_encoders),
                    "matrix_shape": correlation_matrix.shape
                }
            )
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.log_error("Mixed-type correlation generation failed", {"error": str(e)})
            return None

    def _generate_categorical_correlation(self, categorical_columns):
        """Generate correlation matrix for categorical features using Cramér's V."""
        try:
            from scipy.stats import chi2_contingency
            import numpy as np
            
            # Filter columns with reasonable number of categories
            valid_columns = []
            for col in categorical_columns:
                n_unique = self.data[col].nunique()
                if 2 <= n_unique <= 20:  # Reasonable range for categorical analysis
                    valid_columns.append(col)
            
            if len(valid_columns) < 2:
                return None
            
            # Calculate Cramér's V matrix
            n_cols = len(valid_columns)
            correlation_matrix = np.ones((n_cols, n_cols))
            
            for i, col1 in enumerate(valid_columns):
                for j, col2 in enumerate(valid_columns):
                    if i != j:
                        try:
                            # Create contingency table
                            contingency = pd.crosstab(self.data[col1].fillna('Missing'), 
                                                    self.data[col2].fillna('Missing'))
                            
                            # Calculate Cramér's V
                            chi2, _, _, _ = chi2_contingency(contingency)
                            n = contingency.sum().sum()
                            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                            correlation_matrix[i, j] = min(cramers_v, 1.0)  # Cap at 1.0
                            
                        except:
                            correlation_matrix[i, j] = 0.0
            
            # Convert to DataFrame
            correlation_df = pd.DataFrame(
                correlation_matrix,
                index=valid_columns,
                columns=valid_columns
            )
            
            self.logger.log_calculation(
                "Categorical Correlation Generated",
                {
                    "original_columns": len(categorical_columns),
                    "valid_columns": len(valid_columns),
                    "matrix_shape": correlation_df.shape,
                    "method": "Cramér's V"
                }
            )
            
            return correlation_df
            
        except Exception as e:
            self.logger.log_error("Categorical correlation generation failed", {"error": str(e)})
            return None

    def _find_correlation_groups(self, correlation_matrix, threshold, min_group_size):
        """Find groups of highly correlated features."""
        import networkx as nx
        
        # Create a graph of correlations above threshold
        G = nx.Graph()
        
        # Add all features as nodes (excluding target variable)
        features = [col for col in correlation_matrix.columns.tolist() if col != self.target_column]
        G.add_nodes_from(features)
        
        # Add edges for correlations above threshold
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i < j:  # Avoid duplicates
                    corr_value = abs(correlation_matrix.iloc[i, j])
                    if corr_value >= threshold:
                        G.add_edge(feat1, feat2, weight=corr_value)
        
        # Find connected components (correlation groups)
        groups = []
        group_info = []
        
        for component in nx.connected_components(G):
            if len(component) >= min_group_size:
                group = list(component)
                groups.append(group)
                
                # Calculate group statistics
                group_correlations = []
                for i, feat1 in enumerate(group):
                    for j, feat2 in enumerate(group):
                        if i < j:
                            corr_val = abs(correlation_matrix.loc[feat1, feat2])
                            group_correlations.append(corr_val)
                
                info = {
                    'size': len(group),
                    'avg_correlation': np.mean(group_correlations) if group_correlations else 0,
                    'max_correlation': np.max(group_correlations) if group_correlations else 0,
                    'features': group
                }
                group_info.append(info)
        
        return groups, group_info

    def _create_correlation_network(self, correlation_matrix, threshold, groups):
        """Create a network visualization of correlation groups."""
        import plotly.graph_objects as go
        import networkx as nx
        import numpy as np
        
        # Create graph
        G = nx.Graph()
        features = [col for col in correlation_matrix.columns.tolist() if col != self.target_column]
        
        # Add nodes with group colors
        node_colors = {}
        group_colors = px.colors.qualitative.Set3
        
        for i, group in enumerate(groups):
            color = group_colors[i % len(group_colors)]
            for feature in group:
                node_colors[feature] = color
        
        # Features not in any group get gray color
        for feature in features:
            if feature not in node_colors:
                node_colors[feature] = 'gray'
        
        # Add edges for significant correlations
        edge_trace = []
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i < j:
                    corr_value = abs(correlation_matrix.iloc[i, j])
                    if corr_value >= threshold:
                        G.add_edge(feat1, feat2, weight=corr_value)
        
        # Get positions using spring layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        
        # Create the plot
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='lightgray'),
            hoverinfo='none',
            mode='lines',
            name='Correlations'
        ))
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_color.append(node_colors[node])
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=30,
                color=node_color,
                line=dict(width=2, color='black')
            ),
            name='Features',
            hovertemplate='Feature: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Feature Correlation Network (threshold: {threshold:.2f})",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        
        return fig

    def _create_group_correlation_heatmap(self, correlation_matrix, groups):
        """Create a heatmap focusing on correlation groups."""
        import plotly.express as px
        
        # Get all features in groups
        group_features = []
        for group in groups:
            group_features.extend(group)
        
        if not group_features:
            # Return empty figure if no groups
            fig = go.Figure()
            fig.update_layout(title="No correlation groups found")
            return fig
        
        # Create subset of correlation matrix
        group_corr_matrix = correlation_matrix.loc[group_features, group_features]
        
        # Create heatmap
        fig = px.imshow(
            group_corr_matrix,
            title="Correlation Heatmap - Grouped Features Only",
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1,
            aspect='auto'
        )
        
        # Add text annotations to show correlation values
        fig.update_traces(
            text=np.round(group_corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        )
        
        fig.update_layout(height=500)
        return fig

    def _analyze_group_relationships(self, group, correlation_matrix):
        """Analyze relationships within a correlation group."""
        if len(group) < 2:
            return {}
        
        correlations = []
        for i, feat1 in enumerate(group):
            for j, feat2 in enumerate(group):
                if i < j:
                    corr_val = abs(correlation_matrix.loc[feat1, feat2])
                    correlations.append(corr_val)
        
        return {
            "Group Size": len(group),
            "Average Correlation": f"{np.mean(correlations):.3f}",
            "Max Correlation": f"{np.max(correlations):.3f}",
            "Min Correlation": f"{np.min(correlations):.3f}",
            "Correlation Std": f"{np.std(correlations):.3f}"
        }

    def _get_target_correlations(self, group):
        """Get correlations between group features and target variable."""
        target_correlations = {}
        
        for feature in group:
            if feature != self.target_column:
                try:
                    # Calculate correlation with target
                    if (pd.api.types.is_numeric_dtype(self.data[feature]) and 
                        pd.api.types.is_numeric_dtype(self.data[self.target_column])):
                        corr = self.data[feature].corr(self.data[self.target_column])
                        if not np.isnan(corr):
                            target_correlations[feature] = corr
                except:
                    pass
        
        return target_correlations

    def _generate_group_recommendations(self, group, correlation_matrix, target_correlations):
        """
        Generate recommendations for which features to keep/remove in a group.
        
        Uses a sophisticated approach that prioritizes:
        1. PRIMARY: Remove features with highest average correlation to other features in group (most redundant)
        2. SECONDARY: Among features with similar redundancy, prefer keeping those with stronger target correlation
        
        This approach maximizes information preservation while minimizing redundancy.
        """
        recommendations = []
        
        if len(group) <= 1:
            return recommendations
        
        # Calculate redundancy scores (average correlation with other features in the group)
        redundancy_scores = {}
        target_correlation_scores = {}
        
        for feature in group:
            # Calculate average absolute correlation with other features in the group
            other_features = [f for f in group if f != feature]
            if other_features:
                correlations_with_others = []
                for other_feature in other_features:
                    try:
                        corr_value = abs(correlation_matrix.loc[feature, other_feature])
                        if not np.isnan(corr_value):
                            correlations_with_others.append(corr_value)
                    except:
                        pass
                
                if correlations_with_others:
                    redundancy_scores[feature] = np.mean(correlations_with_others)
                else:
                    redundancy_scores[feature] = 0.0
            else:
                redundancy_scores[feature] = 0.0
            
            # Get target correlation (for secondary criterion)
            if target_correlations and feature in target_correlations:
                target_correlation_scores[feature] = abs(target_correlations[feature])
            else:
                target_correlation_scores[feature] = 0.0
        
        # Create a DataFrame for easier analysis and sorting
        analysis_df = pd.DataFrame({
            'feature': group,
            'redundancy_score': [redundancy_scores.get(f, 0) for f in group],
            'target_correlation': [target_correlation_scores.get(f, 0) for f in group]
        })
        
        # Calculate a combined score that prioritizes redundancy but considers target correlation
        # Higher redundancy score = more likely to remove
        # Higher target correlation = less likely to remove
        # Weighted combination: redundancy is primary factor (weight=0.7), target correlation is secondary (weight=0.3)
        redundancy_weight = 0.7
        target_weight = 0.3
        
        # Normalize scores to 0-1 range for fair weighting
        if analysis_df['redundancy_score'].std() > 0:
            analysis_df['redundancy_normalized'] = (analysis_df['redundancy_score'] - analysis_df['redundancy_score'].min()) / (analysis_df['redundancy_score'].max() - analysis_df['redundancy_score'].min())
        else:
            analysis_df['redundancy_normalized'] = 0.0
        
        if analysis_df['target_correlation'].std() > 0:
            analysis_df['target_normalized'] = (analysis_df['target_correlation'] - analysis_df['target_correlation'].min()) / (analysis_df['target_correlation'].max() - analysis_df['target_correlation'].min())
        else:
            analysis_df['target_normalized'] = 0.0
        
        # Combined removal score: high redundancy increases removal likelihood, high target correlation decreases it
        analysis_df['removal_score'] = (redundancy_weight * analysis_df['redundancy_normalized']) - (target_weight * analysis_df['target_normalized'])
        
        # Sort by removal score (highest first = most likely to remove)
        analysis_df = analysis_df.sort_values('removal_score', ascending=False)
        
        # Log the detailed analysis for transparency
        self.logger.log_calculation(
            f"Group Recommendation Analysis - {len(group)} features",
            {
                "group_features": group,
                "redundancy_scores": redundancy_scores,
                "target_correlations": target_correlation_scores,
                "redundancy_weight": redundancy_weight,
                "target_weight": target_weight,
                "sorted_by_removal_score": analysis_df[['feature', 'redundancy_score', 'target_correlation', 'removal_score']].to_dict('records')
            }
        )
        
        # Generate recommendations based on the analysis
        for idx, row in analysis_df.iterrows():
            feature = row['feature']
            redundancy = row['redundancy_score']
            target_corr = row['target_correlation']
            removal_score = row['removal_score']
            
            if idx == len(analysis_df) - 1:  # Keep the feature with lowest removal score
                reason_parts = [f"Lowest redundancy (avg corr: {redundancy:.3f})"]
                if target_corr > 0:
                    reason_parts.append(f"good target relationship ({target_corr:.3f})")
                reason = " and ".join(reason_parts)
                
                recommendations.append({
                    'feature': feature,
                    'action': 'keep',
                    'reason': reason,
                    'redundancy_score': redundancy,
                    'target_correlation': target_corr,
                    'removal_score': removal_score
                })
            else:  # Remove features with higher removal scores
                # Determine primary reason for removal
                if redundancy > analysis_df['redundancy_score'].median():
                    primary_reason = f"High redundancy (avg corr: {redundancy:.3f})"
                else:
                    primary_reason = f"Moderate redundancy (avg corr: {redundancy:.3f})"
                
                # Add context about target correlation if available
                if target_corr > 0:
                    if target_corr < analysis_df['target_correlation'].median():
                        secondary_reason = f"weaker target relationship ({target_corr:.3f})"
                    else:
                        secondary_reason = f"despite good target relationship ({target_corr:.3f}), redundancy is primary concern"
                    reason = f"{primary_reason}, {secondary_reason}"
                else:
                    reason = f"{primary_reason}, target relationship not available"
                
                recommendations.append({
                    'feature': feature,
                    'action': 'remove',
                    'reason': reason,
                    'redundancy_score': redundancy,
                    'target_correlation': target_corr,
                    'removal_score': removal_score
                })
        
        return recommendations

    def display_detailed_feature_relationship_analysis(self, summary=None):
        """
        Display Detailed Feature Relationship Analysis section including feature selection and visualisations.
        
        Args:
            summary: Dictionary containing data summary and visualisations. If None, a minimal summary
                    structure will be created and the method will:
                    1. Generate feature-target relationships on-the-fly when the user selects a feature
                    2. Use the builder's analyse_feature_target_relationship method with custom data
                    3. Use get_feature_relationship_plots for comparing non-target features
                    
        This method is designed to work efficiently both with a pre-computed summary (faster)
        or without one (more flexible). When no summary is provided, it dynamically generates
        the necessary plots and statistics as features are selected by the user.
        
        The method now supports an optional third feature for grouping, which can provide an
        additional dimension to the analysis when applicable.
        """
        # Generate summary if not provided - we only need a minimal structure for this method
        if summary is None:
            # Create a minimal summary structure - the feature relationships will be generated on-demand
            summary = {
                "feature_target_relationships": {},
                "visualisations": {}
            }
            
            # Log that we're using a dynamically generated approach
            self.logger.log_calculation(
                "Feature Relationship Analysis",
                {
                    "using_dynamic_generation": True,
                    "data_shape": self.data.shape,
                    "custom_data": self.data is not self.builder.data,
                    "timestamp": datetime.now().isoformat()
                }
            )

        # Create feature selection UI
        feature_for_analysis, comparison_feature, grouping_feature = self._create_feature_selection_ui()

        # Early return if UI creation failed
        if not feature_for_analysis or not comparison_feature:
            return

        # Log user feature selection
        if feature_for_analysis and comparison_feature:
            log_data = {
                "primary_feature": feature_for_analysis,
                "comparison_feature": comparison_feature,
                "is_target_comparison": comparison_feature == self.target_column,
                "primary_feature_type": str(self.data[feature_for_analysis].dtype),
                "comparison_feature_type": str(self.data[comparison_feature].dtype),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add grouping feature to log if it's selected
            if grouping_feature:
                log_data.update({
                    "grouping_feature": grouping_feature,
                    "grouping_feature_type": str(self.data[grouping_feature].dtype)
                })
                
            self.logger.log_user_action(
                "Feature Selection for Relationship Analysis",
                log_data
            )
        
        if feature_for_analysis:
            try:
                # Prepare and process the data for analysis
                analysis_data, grouping_feature, is_continuous_numeric, original_grouping_feature, binning_method, num_bins = self._prepare_analysis_data(grouping_feature)

                # Determine data types and classification status
                is_comparison_numeric, is_feature_numeric, is_classification, classification_type = self._determine_feature_types_and_classification(
                    feature_for_analysis, comparison_feature
                )

                # Generate feature plots using the extracted method
                feature_plots = self._generate_feature_plots(
                    feature_for_analysis, comparison_feature, summary, grouping_feature,
                    is_continuous_numeric, original_grouping_feature, binning_method, num_bins,
                    analysis_data, is_classification, classification_type
                )

                # Handle case where Builder methods return None
                if feature_plots is None:
                    st.warning("⚠️ Builder methods returned None. This might indicate an issue with the Builder's classification detection.")
                    feature_plots = {}  # Initialize empty dict so we can add fallback plots

                # Override local classification detection based on what Builder method actually returned
                # If Builder returned classification plots, trust that it detected classification correctly
                builder_returned_classification_plots = any(key in feature_plots for key in ["density", "violin", "box", "stacked_bar", "mosaic", "heatmap"])
                builder_returned_regression_plots = any(key in feature_plots for key in ["scatter", "hexbin", "bar"])

                if builder_returned_classification_plots and not builder_returned_regression_plots:
                    # Builder detected classification, override our local detection
                    if not is_classification:
                        st.info("🔄 Builder method detected classification - adjusting visualization selection...")
                        is_classification = True
                        self.logger.log_calculation(
                            "Classification Detection Override",
                            {
                                "local_detection": "regression",
                                "builder_detection": "classification",
                                "builder_plot_keys": list(feature_plots.keys()),
                                "override_applied": True
                            }
                        )

                if feature_plots is not None:
                    # Display analysis results in two-column layout
                    self._display_analysis_layout(
                        feature_for_analysis, comparison_feature, feature_plots,
                        grouping_feature, is_continuous_numeric, original_grouping_feature,
                        analysis_data, binning_method, num_bins, is_feature_numeric, is_classification
                    )

            except Exception as e:
                st.error(f"❌ An error occurred during feature relationship analysis: {str(e)}")
                self.logger.log_calculation(
                    "Feature Relationship Analysis Error",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "feature_for_analysis": feature_for_analysis,
                        "comparison_feature": comparison_feature,
                        "grouping_feature": grouping_feature
                    }
                )
                st.info("Please try selecting different features or check your data for any issues.")
    
    def get_plot_description(self, plot_type: str) -> str:
        """Return the description for each plot type."""
        descriptions = {
            "scatter": """📊 **Scatter Plot with Trend Line**  
            
- Shows relationship between numerical feature and target
- Trend line shows the general pattern
- Upward trend indicates positive correlation
- Downward trend indicates negative correlation
- Scattered points suggest weak relationship

**How to Interpret:**
- **Strong positive relationship:** Points trend upward from left to right
- **Strong negative relationship:** Points trend downward from left to right
- **No relationship:** Points randomly scattered with flat trend line
- **Clustered points:** May indicate distinct groups in your data
- **Outliers:** Points far from the trend line may need investigation""",

            "hexbin": """📊 **Density Heatmap**  
            
- Shows concentration of points in scatter plot
- Darker colors indicate more observations
- Useful for large datasets with overlapping points
- Helps identify common value ranges
- Shows overall relationship pattern

**How to Interpret:**
- **Dark areas:** High concentration of data points in that region
- **Light areas:** Fewer observations in those regions
- **Multiple dark regions:** May indicate multiple clusters or modes
- **Diagonal patterns:** Suggest correlation between variables
- **Horizontal/vertical bands:** May indicate skewed distributions""",

            "box": """📊 **Box Plot**  
            
- Shows distribution of values for each category
- Box shows 25th to 75th percentiles
- Line in box is median
- Whiskers show range of non-outlier points
- Points are outliers
- Compare distributions across categories

**How to Interpret:**
- **Box size:** Larger box means more variability within that category
- **Median position:** Compare central tendencies across categories
- **Outliers:** Individual points outside whiskers need investigation
- **Box position:** Higher/lower boxes indicate category effect on the value
- **Whisker length:** Longer whiskers indicate wider distribution tails""",

            "violin": """📊 **Violin Plot**  
            
- Combines box plot with density distribution
- Width shows frequency of values
- Shows full distribution shape
- Easier to compare distributions than box plots
- Good for seeing multimodal distributions

**How to Interpret:**
- **Width at any point:** Represents frequency of values at that level
- **Multiple bulges:** Indicate multimodal distribution (multiple peaks)
- **Shape comparison:** Compare distribution patterns across categories
- **Narrow violins:** Concentrated values with less variation
- **Wide violins:** More variability across the range of values""",

            "stacked_bar": """📊 **Stacked Bar Chart**  
            
- Shows class distribution for each category
- Height shows total count
- Colors show proportion of each class
- Good for seeing class imbalance
- Compare class distributions across categories

**How to Interpret:**
- **Color proportions:** Compare class distribution across categories
- **Bar height:** Indicates total count for each category
- **Dominant colors:** Show which classes are most common in each category
- **Color patterns:** Look for categories with unique class distributions
- **Similar proportions:** Categories with similar impacts on target""",

            "mosaic": """📊 **Mosaic Plot**  
            
- Shows relationship between categorical variables
- Area represents frequency
- Width shows marginal frequency
- Height shows conditional probability
- Good for showing independence/dependence

**How to Interpret:**
- **Rectangle size:** Larger area means more observations
- **Rectangle alignment:** Perfect alignment suggests independence
- **Misalignment:** Suggests a relationship between variables
- **Color intensity:** Often shows deviation from expected frequencies
- **Row/column patterns:** Reveal how categories interact""",

            "heatmap": """📊 **Association Heatmap**  
            
- Shows strength of association between categories
- Darker colors indicate stronger association
- Good for seeing patterns in categorical data
- Helps identify important category combinations
- Shows over/under-representation

**How to Interpret:**
- **Dark cells:** Strong association between categories
- **Light cells:** Weak or negative association
- **Diagonal patterns:** May indicate related category groups
- **Isolated dark cells:** Specific category interactions
- **Color clusters:** Groups of categories with similar relationships""",

            "density": """📈 **Distribution by Class**  
            
- Shows value distribution for each class
- Overlapping indicates poor class separation
- Separated peaks suggest good predictive power
- Shows where classes are most distinct
- Helps identify optimal decision boundaries

**How to Interpret:**
- **Separated curves:** Feature effectively separates classes
- **Overlapping curves:** Classes harder to distinguish using this feature
- **Multiple peaks:** Potential subgroups within classes
- **Curve width:** Wider curves indicate more variability within class
- **Skewness:** Asymmetrical curves suggest uneven distribution""",

            "bar": """📊 **Mean Value by Category**  
            
- Shows average target value per category
- Error bars show confidence intervals
- Compare target values across categories
- Identify categories with extreme values
- Good for seeing overall patterns

**How to Interpret:**
- **Bar height:** Average target value for each category
- **Error bar length:** Confidence in the mean estimate
- **Overlapping error bars:** Categories may not be significantly different
- **Non-overlapping bars:** Significant difference between categories
- **Outlier categories:** Categories with much higher/lower values"""
        }
        return descriptions.get(plot_type, "No description available.")

    def _create_feature_selection_ui(self):
        """
        Create the feature selection UI with three columns for primary feature, comparison feature, and grouping feature.

        Returns:
            tuple: (feature_for_analysis, comparison_feature, grouping_feature)

        Raises:
            ValueError: If no valid features are available for selection
        """
        if self.data is None or self.data.empty:
            st.error("❌ No data available for feature selection")
            return None, None, None

        # Validate that we have features available for analysis
        available_features = [col for col in self.data.columns if col != self.target_column]
        if not available_features:
            st.error("❌ No features available for analysis (all columns are target variables)")
            return None, None, None

        st.markdown("Select features to analyse their relationship in detail:")

        # Create three columns for feature selection
        col1, col2, col3 = st.columns(3)

        with col1:
            feature_for_analysis = st.selectbox(
                "Select primary feature to analyse",
                available_features,
                help="This is the main feature you want to understand"
            )

        with col2:
            # Ensure target column exists and create comparison options
            comparison_options = []
            if self.target_column and self.target_column in self.data.columns:
                comparison_options.append(self.target_column)

            # Add other features excluding the selected primary feature
            other_features = [
                col for col in self.data.columns
                if col != feature_for_analysis and col != self.target_column
            ]
            comparison_options.extend(other_features)

            if not comparison_options:
                st.error("❌ No comparison features available")
                return feature_for_analysis, None, None

            comparison_feature = st.selectbox(
                "Select feature to compare against (default: target)",
                comparison_options,
                index=0,  # Default to target if available
                help="Compare against target variable or another feature"
            )

        with col3:
            # Add the new grouping feature selection with "None" option
            grouping_options = ["None"] + [
                col for col in self.data.columns
                if col != feature_for_analysis and col != comparison_feature
            ]
            grouping_feature = st.selectbox(
                "Optional: Group by feature",
                grouping_options,
                index=0,  # Default to None
                help="Add another dimension to your analysis by grouping the data by this feature"
            )

            # Convert "None" to actual None for easier handling
            grouping_feature = None if grouping_feature == "None" else grouping_feature

        return feature_for_analysis, comparison_feature, grouping_feature

    def _determine_feature_types_and_classification(self, feature_for_analysis, comparison_feature):
        """
        Determine data types and classification status for the given features.

        Args:
            feature_for_analysis: The primary feature being analyzed
            comparison_feature: The feature to compare against

        Returns:
            tuple: (is_comparison_numeric, is_feature_numeric, is_classification, classification_type)
        """
        # Determine data types
        is_comparison_numeric = pd.api.types.is_numeric_dtype(self.data[comparison_feature])
        is_feature_numeric = pd.api.types.is_numeric_dtype(self.data[feature_for_analysis])

        # Use session state variables for problem type detection when comparing against target
        if comparison_feature == self.target_column:
            # Always get the target data (needed for debugging and analysis)
            y = self.data[comparison_feature]
            is_target_numeric = pd.api.types.is_numeric_dtype(y)

            # Use session state variables if available
            if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type:
                problem_type = st.session_state.problem_type
                is_binary = getattr(st.session_state, 'is_binary', False)
                is_multiclass = getattr(st.session_state, 'is_multiclass', False)
                is_regression = getattr(st.session_state, 'is_regression', False)

                is_classification = is_binary or is_multiclass
                if is_binary:
                    classification_type = "binary"
                elif is_multiclass:
                    classification_type = "multi-class"
                else:
                    classification_type = "regression"
            else:
                # Fallback to heuristic detection for backward compatibility

                # Check if this target was encoded during data loading (indicates classification)
                if 'encoding_mappings' in st.session_state and comparison_feature in st.session_state.encoding_mappings:
                    # If we have encoding mapping, it was treated as classification in data loading
                    is_classification = True
                    classification_type = "multi-class" if y.nunique() > 2 else "binary"
                elif y.nunique() <= 2:
                    # Binary classification (2 unique values)
                    is_classification = True
                    classification_type = "binary"
                elif is_target_numeric and 3 <= y.nunique() <= 20:
                    # Check if this might be encoded multiclass by looking for integer-like values
                    try:
                        y_no_nan = y.dropna()
                        if len(y_no_nan) > 0:
                            is_integer_like = np.allclose(y_no_nan, np.round(y_no_nan), atol=1e-10)
                            consecutive_integers = (y_no_nan.min() >= 0 and
                                                  set(y_no_nan.unique()) == set(range(int(y_no_nan.min()), int(y_no_nan.max()) + 1)))
                            is_classification = is_integer_like and consecutive_integers
                            classification_type = "multi-class" if is_classification else "regression"
                        else:
                            is_classification = False
                            classification_type = "regression"
                    except:
                        is_classification = False
                        classification_type = "regression"
                else:
                    # Too many unique values for classification
                    is_classification = False
                    classification_type = "regression"

            # Debug: Add more detailed logging to understand why classification detection failed
            encoding_mappings = st.session_state.get('encoding_mappings', {})
            has_encoding = comparison_feature in encoding_mappings
            unique_count = y.nunique()

            # Additional debug checks
            debug_info = {
                "comparison_feature": comparison_feature,
                "is_target": True,
                "is_classification": is_classification,
                "classification_type": classification_type,
                "unique_values": unique_count,
                "has_encoding_mapping": has_encoding,
                "is_target_numeric": is_target_numeric,
                "session_state_has_encoding_mappings": 'encoding_mappings' in st.session_state,
                "encoding_mappings_keys": list(encoding_mappings.keys()) if encoding_mappings else [],
                "y_dtype": str(y.dtype),
                "y_sample_values": y.dropna().head(10).tolist() if len(y.dropna()) > 0 else []
            }

            if is_target_numeric and 3 <= unique_count <= 20:
                try:
                    y_no_nan = y.dropna()
                    if len(y_no_nan) > 0:
                        is_integer_like = np.allclose(y_no_nan, np.round(y_no_nan), atol=1e-10)
                        consecutive_integers = (y_no_nan.min() >= 0 and
                                              set(y_no_nan.unique()) == set(range(int(y_no_nan.min()), int(y_no_nan.max()) + 1)))
                        debug_info.update({
                            "integer_like_check": is_integer_like,
                            "consecutive_integers_check": consecutive_integers,
                            "y_min": float(y_no_nan.min()),
                            "y_max": float(y_no_nan.max()),
                            "y_unique_values": sorted(y_no_nan.unique().tolist())
                        })
                except Exception as e:
                    debug_info["integer_check_error"] = str(e)

            # Log the classification decision for debugging
            self.logger.log_calculation(
                "Classification Type Detection Debug",
                debug_info
            )

            # Log problem type detection for debugging
            self.logger.log_calculation(
                "Problem Type Detection in Feature Relationships",
                {
                    "primary_feature": feature_for_analysis,
                    "comparison_feature": comparison_feature,
                    "is_target_comparison": True,
                    "classification_type": classification_type,
                    "is_classification": is_classification,
                    "used_session_state": hasattr(st.session_state, 'problem_type') and st.session_state.problem_type is not None,
                    "session_state_problem_type": getattr(st.session_state, 'problem_type', 'Not available'),
                    "session_state_is_binary": getattr(st.session_state, 'is_binary', 'Not available'),
                    "session_state_is_multiclass": getattr(st.session_state, 'is_multiclass', 'Not available'),
                    "session_state_is_regression": getattr(st.session_state, 'is_regression', 'Not available')
                }
            )
        else:
            # For non-target comparisons, use the original heuristic approach
            comparison_unique_values = self.data[comparison_feature].nunique()
            is_classification = not is_comparison_numeric or comparison_unique_values <= 10
            classification_type = "classification" if is_classification else "regression"

        return is_comparison_numeric, is_feature_numeric, is_classification, classification_type

    def _generate_feature_plots(self, feature_for_analysis, comparison_feature, summary, grouping_feature,
                              is_continuous_numeric, original_grouping_feature, binning_method, num_bins,
                              analysis_data, is_classification, classification_type):
        """
        Generate feature plots using Builder methods with appropriate parameters.

        Args:
            feature_for_analysis: The primary feature being analyzed
            comparison_feature: The feature to compare against
            summary: Summary data structure from the Builder
            grouping_feature: Optional grouping feature
            is_continuous_numeric: Whether grouping feature is continuous numeric
            original_grouping_feature: Original name of grouping feature before binning
            binning_method: Method used for binning (if applicable)
            num_bins: Number of bins used (if applicable)
            analysis_data: The processed data for analysis
            is_classification: Whether this is a classification problem
            classification_type: Type of classification (binary, multi-class, regression)

        Returns:
            dict: Feature plots dictionary from Builder methods
        """
        # Use target relationships if comparing against target
        if comparison_feature == self.target_column:
            # If no grouping feature is selected, proceed as before
            if grouping_feature is None:
                feature_relationships = summary.get("feature_target_relationships", {})
                feature_plots = feature_relationships.get(feature_for_analysis)

                # If feature_plots is None or empty (when no summary is provided),
                # generate the plots on the fly for target relationship
                if feature_plots is None:
                    st.info("Generating relationship plots dynamically...")
                    feature_plots = self.builder.analyse_feature_target_relationship(
                        feature_for_analysis,
                        self.target_column,
                        custom_data=analysis_data
                    )
                    # Log dynamic generation
                    self.logger.log_calculation(
                        "Dynamic Feature-Target Analysis",
                        {
                            "feature": feature_for_analysis,
                            "target": self.target_column,
                            "data_shape": analysis_data.shape
                        }
                    )
            else:
                # If grouping feature is selected, always generate plots dynamically
                # with the grouping feature included
                if is_continuous_numeric:
                    grouping_description = f"binned '{original_grouping_feature}'"
                else:
                    grouping_description = f"'{grouping_feature}'"

                st.info(f"Generating relationship plots dynamically with grouping by {grouping_description}...")

                # Use the analyse_feature_target_relationship method with grouping
                feature_plots = self.builder.analyse_feature_target_relationship(
                    feature_for_analysis,
                    self.target_column,
                    custom_data=analysis_data,
                    grouping_feature=grouping_feature
                )

                # Log dynamic generation with grouping
                log_data = {
                    "feature": feature_for_analysis,
                    "target": self.target_column,
                    "grouping_feature": original_grouping_feature if is_continuous_numeric else grouping_feature,
                    "is_binned": is_continuous_numeric,
                    "binning_method": binning_method if is_continuous_numeric else None,
                    "num_bins": num_bins if is_continuous_numeric else None,
                    "data_shape": analysis_data.shape
                }

                self.logger.log_calculation(
                    "Dynamic Feature-Target Analysis with Grouping",
                    log_data
                )
        else:
            # For non-target comparisons, generate plots on the fly
            # Include grouping feature if selected
            if is_continuous_numeric:
                grouping_description = f"binned '{original_grouping_feature}'"
            else:
                grouping_description = f"'{grouping_feature}'" if grouping_feature else "none"

            st.info(f"Generating feature relationship plots with grouping by {grouping_description}...")

            feature_plots = self.builder.get_feature_relationship_plots(
                feature_for_analysis,
                comparison_feature,
                custom_data=analysis_data,  # Use binned data if available
                grouping_feature=grouping_feature
            )

            # Log generation with grouping if applicable
            log_data = {
                "feature1": feature_for_analysis,
                "feature2": comparison_feature,
                "data_shape": analysis_data.shape
            }

            if grouping_feature:
                log_data.update({
                    "grouping_feature": original_grouping_feature if is_continuous_numeric else grouping_feature,
                    "is_binned": is_continuous_numeric,
                    "binning_method": binning_method if is_continuous_numeric else None,
                    "num_bins": num_bins if is_continuous_numeric else None,
                })

            self.logger.log_calculation(
                "Dynamic Feature-Feature Analysis" + (" with Grouping" if grouping_feature else ""),
                log_data
            )

        # Debug: Log what we got back from Builder methods
        self.logger.log_calculation(
            "Builder Method Results Debug",
            {
                "feature_plots_is_none": feature_plots is None,
                "feature_plots_keys": list(feature_plots.keys()) if feature_plots is not None else [],
                "comparison_feature": comparison_feature,
                "is_classification_detected": is_classification,
                "classification_type": classification_type,
                "is_target_comparison": comparison_feature == self.target_column,
                "builder_method_used": "analyse_feature_target_relationship" if comparison_feature == self.target_column else "get_feature_relationship_plots"
            }
        )

        return feature_plots

    def _display_analysis_layout(self, feature_for_analysis, comparison_feature, feature_plots,
                               grouping_feature, is_continuous_numeric, original_grouping_feature,
                               analysis_data, binning_method, num_bins, is_feature_numeric, is_classification):
        """
        Display the main analysis layout with two columns for statistical analysis and visualizations.

        Args:
            feature_for_analysis: The primary feature being analyzed
            comparison_feature: The feature to compare against
            feature_plots: Dictionary containing plots from Builder methods
            grouping_feature: Optional grouping feature
            is_continuous_numeric: Whether grouping feature is continuous numeric
            original_grouping_feature: Original name of grouping feature before binning
            analysis_data: The processed data for analysis
            binning_method: Method used for binning (if applicable)
            num_bins: Number of bins used (if applicable)
            is_feature_numeric: Whether the primary feature is numeric
            is_classification: Whether this is a classification problem
        """
        # Create two-column layout for results (similar to Feature Analysis tab)
        analysis_col1, analysis_col2 = st.columns(2)

        # Left column: Statistical Analysis
        with analysis_col1:
            st.write("📊 Statistical Analysis")

            # Perform statistical analysis using the extracted method
            group_stats, stats_errors = self._perform_statistical_analysis(
                feature_for_analysis, comparison_feature, feature_plots,
                grouping_feature, is_continuous_numeric, original_grouping_feature,
                analysis_data, binning_method, num_bins
            )

        # Right column: Visualizations
        with analysis_col2:
            # Generate and display visualizations using the extracted method
            self._generate_and_display_visualizations(
                feature_for_analysis, comparison_feature, feature_plots,
                grouping_feature, is_continuous_numeric, original_grouping_feature,
                analysis_data, is_feature_numeric, is_classification
            )

    def _prepare_analysis_data(self, grouping_feature):
        """
        Prepare and process data for analysis, including sorting and binning if needed.

        Args:
            grouping_feature: The selected grouping feature or None

        Returns:
            Tuple of (analysis_data, grouping_feature, is_continuous_numeric, original_grouping_feature, binning_method, num_bins)

        Raises:
            ValueError: If grouping feature is not found in the data
            Exception: For unexpected errors during data preparation
        """
        try:
            # Validate grouping feature exists in data
            if grouping_feature and grouping_feature not in self.data.columns:
                raise ValueError(f"Grouping feature '{grouping_feature}' not found in data columns: {list(self.data.columns)}")

            # Process grouping feature if it's numeric to ensure sorted display
            sorted_data = None
            if grouping_feature and pd.api.types.is_numeric_dtype(self.data[grouping_feature]):
                # Create a sorted copy of the data for visualization
                st.info(f"Sorting data by '{grouping_feature}' values for clearer visualization")
                sorted_data = self.data.sort_values(by=grouping_feature).copy()
                self.logger.log_calculation(
                    "Sorting Grouping Feature",
                    {
                        "grouping_feature": grouping_feature,
                        "is_numeric": True,
                        "sorted_values_count": len(sorted_data)
                    }
                )

            # Use the original or sorted data based on whether we sorted the grouping feature
            analysis_data = sorted_data if sorted_data is not None else self.data

        except Exception as e:
            # Log the error and re-raise with more context
            self.logger.log_calculation(
                "Data Preparation Error",
                {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "grouping_feature": grouping_feature,
                    "data_shape": self.data.shape if hasattr(self, 'data') else "Unknown"
                }
            )
            raise

        # Check if grouping feature is continuous numeric with many unique values - for potential binning
        is_continuous_numeric = grouping_feature and pd.api.types.is_numeric_dtype(analysis_data[grouping_feature]) and analysis_data[grouping_feature].nunique() > 10
        original_grouping_feature = grouping_feature

        # Apply binning if needed
        binning_method = None
        num_bins = None
        if is_continuous_numeric:
            analysis_data, grouping_feature, binning_method, num_bins = self._apply_binning_to_data(analysis_data, grouping_feature)

        return analysis_data, grouping_feature, is_continuous_numeric, original_grouping_feature, binning_method, num_bins

    def _apply_binning_to_data(self, analysis_data, grouping_feature):
        """
        Apply binning to continuous numeric grouping feature.

        Args:
            analysis_data: The data to process
            grouping_feature: The grouping feature to bin

        Returns:
            Tuple of (binned_data, new_grouping_feature_name, binning_method, num_bins)
        """
        # Create a section for binning configuration
        st.markdown(f"""
        <div style="background-color:#f0f8ff; padding:10px; border-radius:10px; margin:15px 0px">
        <h4 style="margin-top:0">Configure Binning for '{grouping_feature}'</h4>
        </div>
        """, unsafe_allow_html=True)

        st.info(f"'{grouping_feature}' is a continuous numeric feature. Creating bins for consistent analysis and visualization.")

        # Let user choose binning method and number of bins
        binning_col1, binning_col2 = st.columns(2)
        with binning_col1:
            binning_method = st.selectbox(
                "Binning method",
                ["Quantile (equal count)", "Equal width"],
                help="Quantile bins have equal number of samples. Equal width bins have equal value ranges."
            )
        with binning_col2:
            num_bins = st.slider(
                "Number of bins",
                min_value=3,
                max_value=10,
                value=5,
                help="More bins provide finer granularity but may reduce statistical power"
            )

        # Create a copy of the data to avoid modifying the original
        binned_data = analysis_data.copy()

        # Apply binning based on selected method
        if binning_method == "Quantile (equal count)":
            # Quantile binning to have approximately equal numbers of samples in each bin
            bins = pd.qcut(
                binned_data[grouping_feature],
                q=num_bins,
                duplicates='drop',
                labels=False
            )
            # Create labels for the bins
            bin_edges = pd.qcut(binned_data[grouping_feature], q=num_bins, duplicates='drop').cat.categories
            bin_labels = [f"Q{i+1}: {str(bin_edge.left)[:6]}-{str(bin_edge.right)[:6]}" for i, bin_edge in enumerate(bin_edges)]
        else:  # Equal width
            # Equal width binning
            bins = pd.cut(
                binned_data[grouping_feature],
                bins=num_bins,
                labels=False
            )
            # Create labels for the bins
            bin_edges = pd.cut(binned_data[grouping_feature], bins=num_bins).cat.categories
            bin_labels = [f"Bin {i+1}: {str(bin_edge.left)[:6]}-{str(bin_edge.right)[:6]}" for i, bin_edge in enumerate(bin_edges)]

        # Map numeric bin indices to descriptive labels
        bin_mapping = {i: label for i, label in enumerate(bin_labels) if i in bins.unique()}

        # Create a new column with the bin labels
        binned_data["binned_group"] = bins
        binned_data["bin_label"] = binned_data["binned_group"].map(bin_mapping)

        # Show bin information
        bin_counts = binned_data["bin_label"].value_counts().sort_index()
        st.write("**Sample distribution across bins:**")
        bin_count_chart = px.bar(
            x=bin_counts.index,
            y=bin_counts.values,
            labels={"x": f"Bins of {grouping_feature}", "y": "Number of samples"},
            title=f"Distribution of samples across {grouping_feature} bins"
        )
        bin_count_chart.update_layout(height=300)
        st.plotly_chart(bin_count_chart, config={'responsive': True})

        # Return the binned data and the new grouping feature name
        return binned_data, "bin_label", binning_method, num_bins

    def _perform_statistical_analysis(self, feature_for_analysis, comparison_feature, feature_plots,
                                    grouping_feature, is_continuous_numeric, original_grouping_feature,
                                    analysis_data, binning_method, num_bins):
        """
        Perform statistical analysis for feature relationships.

        Args:
            feature_for_analysis: The primary feature being analyzed
            comparison_feature: The feature to compare against
            feature_plots: Dictionary containing statistical results
            grouping_feature: Optional grouping feature
            is_continuous_numeric: Whether grouping feature is continuous numeric
            original_grouping_feature: Original name of grouping feature before binning
            analysis_data: The processed data for analysis
            binning_method: Method used for binning (if applicable)
            num_bins: Number of bins used (if applicable)

        Returns:
            Tuple of (group_stats, stats_errors)
        """
        # Initialize group analysis variables (needed for all code paths)
        group_stats = []
        stats_errors = []

        # Display statistical test results
        if "stats" in feature_plots:
            stats = feature_plots["stats"]
            if "error" in stats:
                st.warning(f"⚠️ {stats['error']}")
                if "reason" in stats:
                    st.info(f"Reason: {stats['reason']}")
                self.logger.log_error(
                    "Statistical Test Failed",
                    {
                        "primary_feature": feature_for_analysis,
                        "comparison_feature": comparison_feature,
                        "grouping_feature": grouping_feature,
                        "error": stats['error'],
                        "reason": stats.get('reason', 'Unknown')
                    }
                )
            else:
                st.write(f"**Relationship:** {feature_for_analysis} vs {comparison_feature}")
                test_type = stats.get("test", "Unknown test")

                # Log statistical analysis results
                log_data = {
                    "primary_feature": feature_for_analysis,
                    "comparison_feature": comparison_feature,
                    "test_type": test_type,
                    "results": stats
                }

                if grouping_feature:
                    log_data["grouping_feature"] = grouping_feature

                self.logger.log_calculation(
                    "Statistical Analysis",
                    log_data
                )

                # Create metrics table and display significance
                self._display_statistical_metrics(stats, test_type, feature_for_analysis, comparison_feature, grouping_feature)

                # Get and display the explanation
                self._display_statistical_explanation(test_type, stats)

                # Perform group analysis if needed
                if grouping_feature and ("statistic" in stats or "chi2" in stats):
                    group_stats, stats_errors = self._analyze_group_statistics(
                        feature_for_analysis, comparison_feature, stats, test_type,
                        grouping_feature, is_continuous_numeric, original_grouping_feature,
                        analysis_data, binning_method, num_bins
                    )

        return group_stats, stats_errors

    def _display_statistical_metrics(self, stats, test_type, feature_for_analysis=None, comparison_feature=None, grouping_feature=None):
        """Display statistical metrics in a formatted table."""
        st.write("📊 Statistical Metrics:")
        metrics_data = {"Test Type": [test_type]}

        # Add different statistics based on test type
        if "statistic" in stats:
            # Label the statistic differently based on test type
            statistic_label = "F-Statistic" if "ANOVA" in test_type else (
                "T-Statistic" if "T-Test" in test_type else (
                    "Correlation Coefficient" if "Pearson" in test_type else "Test Statistic"))
            metrics_data[statistic_label] = [f"{stats['statistic']:.4f}"]
        elif "chi2" in stats:
            metrics_data["Chi-square Value"] = [f"{stats['chi2']:.4f}"]

        if "p_value" in stats:
            metrics_data["P-value"] = [f"{stats['p_value']:.4f}"]

            # Add significance interpretation
            if stats['p_value'] < 0.05:
                metrics_data["Significance"] = ["Significant (p < 0.05)"]
                st.success(f"✓ Statistically significant relationship")
                # Log significant relationship
                self.logger.log_recommendation(
                    "Significant Feature Relationship",
                    {
                        "primary_feature": feature_for_analysis,
                        "comparison_feature": comparison_feature,
                        "p_value": stats['p_value'],
                        "test_type": test_type
                    }
                )
            else:
                metrics_data["Significance"] = ["Not significant (p ≥ 0.05)"]
                st.info(f"ℹ️ No statistically significant relationship")

        if "dof" in stats:
            metrics_data["Degrees of Freedom"] = [str(stats['dof'])]

        # Display metrics table
        metrics_df = pd.DataFrame(metrics_data).T
        metrics_df.columns = ['Value']
        st.dataframe(metrics_df, width='stretch')

    def _display_statistical_explanation(self, test_type, stats):
        """Display explanation of the statistical test."""
        explanation = self.builder.get_statistical_explanation(test_type, stats)
        with st.expander(f"ℹ️ Understanding the Statistical Test", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**What this test measures:**")
                st.markdown(explanation["method"])
            with col2:
                st.markdown("**How to interpret results:**")

                # Add specific explanation for the displayed statistics
                if "ANOVA" in test_type:
                    st.markdown("""
                    **F-Statistic:**
                    - Larger values indicate stronger differences between groups
                    - The value shown is the F-statistic from the ANOVA test
                    - Values further from 1 suggest stronger relationships

                    **P-value:**
                    - Values below 0.05 indicate statistically significant differences
                    - Small p-values suggest the relationship is unlikely due to chance
                    """)
                elif "Pearson" in test_type:
                    st.markdown("""
                    **Correlation Coefficient:**
                    - Values range from -1 to +1
                    - Closer to +1: Strong positive correlation (as one increases, the other increases)
                    - Closer to -1: Strong negative correlation (as one increases, the other decreases)
                    - Near 0: Weak or no linear relationship

                    **P-value:**
                    - Values below 0.05 indicate statistically significant correlation
                    - Small p-values suggest the correlation is unlikely due to chance
                    """)
                elif "Chi-square" in test_type:
                    st.markdown("""
                    **Chi-square Value:**
                    - Larger values indicate stronger associations between categories
                    - The value measures how far the observed data is from expected values

                    **P-value:**
                    - Values below 0.05 indicate statistically significant association
                    - Small p-values suggest the relationship is unlikely due to chance
                    """)
                elif "T-Test" in test_type:
                    st.markdown("""
                    **T-Statistic:**
                    - Values further from 0 indicate stronger differences between groups
                    - Values above 2 or below -2 often indicate significant differences

                    **P-value:**
                    - Values below 0.05 indicate statistically significant differences
                    - Small p-values suggest the differences are unlikely due to chance
                    """)
                else:
                    # Fall back to the general explanation
                    st.markdown(explanation["interpretation"])

        # Log statistical explanation
        self.logger.log_calculation(
            "Statistical Test Explanation",
            {
                "test_type": test_type,
                "explanation": explanation
            }
        )

    def _analyze_group_statistics(self, feature_for_analysis, comparison_feature, stats, test_type,
                                grouping_feature, is_continuous_numeric, original_grouping_feature,
                                analysis_data, binning_method, num_bins):
        """
        Analyze statistical relationships within each group.

        Returns:
            Tuple of (group_stats, stats_errors)
        """
        import numpy as np
        from scipy.stats import f_oneway, ttest_ind, pearsonr, chi2_contingency

        group_stats = []
        stats_errors = []

        st.write("**Group Analysis:**")
        st.info(f"Analyzing relationship within each group of '{grouping_feature}'")
        st.info(f"This analysis shows how the relationship between '{feature_for_analysis}' and '{comparison_feature}' varies across different groups of '{original_grouping_feature if is_continuous_numeric else grouping_feature}'.")

        # Group the data appropriately
        if is_continuous_numeric:
            # We're already using the binned data from earlier
            grouped_data = analysis_data.groupby(grouping_feature)  # grouping_feature is now "bin_label"
        else:
            # Non-continuous or already categorical - use direct grouping
            grouped_data = analysis_data.groupby(grouping_feature)

        # Set maximum groups to display to avoid overwhelming the UI
        max_groups_to_display = 12

        # Process each group
        for group_name, group_df in grouped_data:
            # Skip groups with too few samples
            if len(group_df) < 5:  # Minimum sample size for meaningful statistics
                stats_errors.append(f"Group '{group_name}' has fewer than 5 samples")
                continue

            # Calculate statistics for this group
            try:
                stat_value, p_value = self._calculate_group_statistic(
                    group_df, feature_for_analysis, comparison_feature, test_type, group_name, stats_errors
                )

                if stat_value is not None and p_value is not None:
                    # Add to results
                    is_significant = p_value < 0.05
                    group_stats.append({
                        "Group": group_name,
                        "Samples": len(group_df[feature_for_analysis].dropna()),
                        "Statistic": stat_value,
                        "P-value": p_value,
                        "Significant": is_significant
                    })
            except Exception as e:
                # Log error but continue with other groups
                stats_errors.append(f"Error analyzing group '{group_name}': {str(e)}")

        # Display group statistics if we have any
        if group_stats:
            self._display_group_statistics_table(
                group_stats, is_continuous_numeric, original_grouping_feature,
                grouping_feature, feature_for_analysis, comparison_feature,
                stats, test_type, binning_method, num_bins, max_groups_to_display, analysis_data
            )
        else:
            # Display error messages if we have them
            if stats_errors:
                st.warning(f"Could not calculate per-group statistics. Common issues:")
                with st.expander("See details"):
                    for i, error in enumerate(stats_errors[:5], 1):
                        st.markdown(f"**Issue {i}:** {error}")
                    if len(stats_errors) > 5:
                        st.markdown(f"...and {len(stats_errors) - 5} more issues")
            else:
                st.info(f"Could not calculate per-group statistics. Each group needs sufficient data points and variation for statistical analysis.")

        return group_stats, stats_errors

    def _calculate_group_statistic(self, group_df, feature_for_analysis, comparison_feature, test_type, group_name, stats_errors):
        """
        Calculate the appropriate statistical test for a single group.

        Returns:
            Tuple of (stat_value, p_value) or (None, None) if calculation fails
        """
        import numpy as np
        from scipy.stats import f_oneway, ttest_ind, pearsonr, chi2_contingency

        # Prepare data
        feature_data = group_df[feature_for_analysis].replace([np.inf, -np.inf], np.nan).dropna()
        comparison_data = group_df[comparison_feature].replace([np.inf, -np.inf], np.nan).dropna()

        # Get valid indices
        valid_indices = feature_data.index.intersection(comparison_data.index)
        feature_data = feature_data[valid_indices]
        comparison_data = comparison_data[valid_indices]

        # Skip if not enough data
        if len(feature_data) < 5:
            stats_errors.append(f"Group '{group_name}' has fewer than 5 valid samples after cleaning")
            return None, None

        # Calculate the appropriate statistic based on the test type
        if "ANOVA" in test_type:
            # For ANOVA, we need to group the feature data by comparison categories
            groups = [group for name, group in feature_data.groupby(comparison_data) if len(group) > 1]
            if len(groups) >= 2:
                f_stat, p_value = f_oneway(*groups)
                return f_stat, p_value
            else:
                stats_errors.append(f"Group '{group_name}' doesn't have enough categories for ANOVA")
                return None, None

        elif "Pearson" in test_type:
            # For Pearson correlation, both feature and comparison must be numeric
            if pd.api.types.is_numeric_dtype(feature_data) and pd.api.types.is_numeric_dtype(comparison_data):
                corr, p_value = pearsonr(feature_data, comparison_data)
                return corr, p_value
            else:
                stats_errors.append(f"Group '{group_name}' can't use Pearson correlation with non-numeric data")
                return None, None

        elif "Chi-square" in test_type:
            # For Chi-square, create a contingency table
            contingency = pd.crosstab(feature_data, comparison_data)
            if contingency.size > 1 and not (contingency < 5).any().any():
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                return chi2, p_value
            else:
                stats_errors.append(f"Group '{group_name}' has sparse data for Chi-square test")
                return None, None

        elif "T-Test" in test_type:
            # For T-test, comparison must have exactly 2 categories
            categories = comparison_data.unique()
            if len(categories) == 2:
                group1 = feature_data[comparison_data == categories[0]]
                group2 = feature_data[comparison_data == categories[1]]
                if len(group1) > 1 and len(group2) > 1:
                    t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
                    return t_stat, p_value
                else:
                    stats_errors.append(f"Group '{group_name}' doesn't have enough samples in each category for T-test")
                    return None, None
            else:
                stats_errors.append(f"Group '{group_name}' needs exactly 2 categories for T-test")
                return None, None
        else:
            # Skip for unknown test types
            stats_errors.append(f"Unknown test type '{test_type}' for group '{group_name}'")
            return None, None

    def _display_group_statistics_table(self, group_stats, is_continuous_numeric, original_grouping_feature,
                                       grouping_feature, feature_for_analysis, comparison_feature,
                                       stats, test_type, binning_method, num_bins, max_groups_to_display, analysis_data):
        """Display the group statistics table with proper formatting and interpretation."""
        # Sort by group value if numeric bins (already ordered) or by significance/p-value for categorical
        if is_continuous_numeric:
            # For binned data, preserve the bin order (already ordered)
            group_stats_df = pd.DataFrame(group_stats)
        else:
            # For non-binned data, sort as before
            if pd.api.types.is_numeric_dtype(analysis_data[grouping_feature]):
                group_stats_df = pd.DataFrame(group_stats).sort_values("Group")
            else:
                group_stats_df = pd.DataFrame(group_stats).sort_values(["Significant", "P-value"], ascending=[False, True])

        # Limit the number of groups displayed
        if len(group_stats_df) > max_groups_to_display:
            st.info(f"Showing statistics for {max_groups_to_display} out of {len(group_stats_df)} groups with sufficient data.")
            group_stats_df = group_stats_df.head(max_groups_to_display)

        # Format the DataFrame for display
        display_df = group_stats_df.copy()
        display_df["Statistic"] = display_df["Statistic"].round(4)
        display_df["P-value"] = display_df["P-value"].round(4)
        display_df["Significant"] = display_df["Significant"].map({True: "✓", False: "✗"})

        # Rename columns to be more descriptive about the relationship
        statistic_label = "F-Statistic" if "ANOVA" in test_type else (
            "T-Statistic" if "T-Test" in test_type else (
                "Correlation Coefficient" if "Pearson" in test_type else "Test Statistic"))
        test_label = "Chi-square" if "chi2" in stats else statistic_label

        display_df = display_df.rename(columns={
            "Group": f"{original_grouping_feature if is_continuous_numeric else grouping_feature} Value",
            "Statistic": f"{test_label} ({feature_for_analysis} vs {comparison_feature})",
            "P-value": "P-value",
            "Significant": "Significant Relationship"
        })

        # Display the per-group statistics
        st.dataframe(display_df, width='stretch')

        # Add interpretation help
        self._display_group_statistics_interpretation(
            is_continuous_numeric, original_grouping_feature, grouping_feature,
            feature_for_analysis, comparison_feature, test_label, test_type,
            binning_method, num_bins
        )

        # Log the group statistics
        self.logger.log_calculation(
            "Group-level Statistical Analysis",
            {
                "primary_feature": feature_for_analysis,
                "comparison_feature": comparison_feature,
                "grouping_feature": original_grouping_feature if is_continuous_numeric else grouping_feature,
                "is_binned": is_continuous_numeric,
                "binning_method": binning_method if is_continuous_numeric else "None",
                "num_bins": num_bins if is_continuous_numeric else 0,
                "test_type": test_type,
                "group_count": len(group_stats_df),
                "significant_groups": sum(group_stats_df["Significant"] == "✓")
            }
        )

    def _display_group_statistics_interpretation(self, is_continuous_numeric, original_grouping_feature,
                                               grouping_feature, feature_for_analysis, comparison_feature,
                                               test_label, test_type, binning_method, num_bins):
        """Display interpretation help for group statistics."""
        with st.expander(f"ℹ️ How to interpret group statistics{' (binned)' if is_continuous_numeric else ''}", expanded=False):
            st.markdown(f"""
            This table shows how the relationship between '{feature_for_analysis}' and '{comparison_feature}'
            varies across different groups of '{original_grouping_feature if is_continuous_numeric else grouping_feature}'{' (binned)' if is_continuous_numeric else ''}:

            - **Group**: The value{" range" if is_continuous_numeric else ""} of '{original_grouping_feature if is_continuous_numeric else grouping_feature}' for this subset of data
            - **Samples**: Number of data points in this group (after removing missing values)
            - **Statistic**: The {test_label} for the '{feature_for_analysis}' vs '{comparison_feature}' relationship within this group only
            - **P-value**: Statistical significance of the relationship between '{feature_for_analysis}' and '{comparison_feature}' within this group
            - **Significant**: ✓ means a statistically significant relationship exists within this group (p < 0.05), ✗ means no significant relationship

            **What this tells you:**
            For each group of '{original_grouping_feature if is_continuous_numeric else grouping_feature}', we're rerunning the same '{test_type}' test
            that was applied to the entire dataset, but only using data points from that specific group.
            """)

            if is_continuous_numeric:
                st.markdown(f"""
                **About the binning:**
                - Continuous values have been grouped into bins to enable meaningful analysis
                - Each bin represents a range of values for the grouping feature
                - Bin labels show the approximate range of values in each bin
                - {binning_method} was used with {num_bins} bins
                """)

            st.markdown(f"""
            **Looking at these group-level statistics helps you:**
            - Identify if the relationship between '{feature_for_analysis}' and '{comparison_feature}' is consistent across all values of '{original_grouping_feature if is_continuous_numeric else grouping_feature}'
            - Find specific values of '{original_grouping_feature if is_continuous_numeric else grouping_feature}' where the relationship is stronger or weaker
            - Discover potential interactions where '{original_grouping_feature if is_continuous_numeric else grouping_feature}' influences how '{feature_for_analysis}' relates to '{comparison_feature}'
            - Better understand when and where a relationship holds true in your data
            """)

    def _generate_and_display_visualizations(self, feature_for_analysis, comparison_feature, feature_plots,
                                           grouping_feature, is_continuous_numeric, original_grouping_feature,
                                           analysis_data, is_feature_numeric, is_classification):
        """
        Generate and display visualizations for feature relationships.

        Args:
            feature_for_analysis: The primary feature being analyzed
            comparison_feature: The feature to compare against
            feature_plots: Dictionary containing plots from Builder methods
            grouping_feature: Optional grouping feature
            is_continuous_numeric: Whether grouping feature is continuous numeric
            original_grouping_feature: Original name of grouping feature before binning
            analysis_data: The processed data for analysis
            is_feature_numeric: Whether the primary feature is numeric
            is_classification: Whether this is a classification problem
        """
        st.write("🎯 Visualizations")

        # Classification logic was already determined above
        # If a grouping feature is used, display a helper message
        if grouping_feature:
            if is_continuous_numeric:
                st.info(f"Visualizations are grouped by binned '{original_grouping_feature}'. The colors or facets in the plots represent binned values.")
            else:
                st.info(f"Visualizations are grouped by '{grouping_feature}'. The colors or facets in the plots represent different values of this feature.")

        # Show all applicable visualizations in tabs
        viz_tabs = []
        tab_titles = []

        # Check if we need to create fallback plots
        needs_fallback, fallback_reason = self._check_if_fallback_needed(
            grouping_feature, is_classification, comparison_feature, feature_plots, is_feature_numeric
        )

        if needs_fallback:
            st.info(f"🔄 Generating fallback visualizations for {fallback_reason}...")
            feature_plots = self._create_fallback_plots(
                feature_plots, grouping_feature, analysis_data, feature_for_analysis,
                comparison_feature, is_feature_numeric, is_classification
            )

        # Populate visualization tabs based on feature types
        self._populate_visualization_tabs(
            viz_tabs, tab_titles, feature_plots, is_feature_numeric, is_classification
        )

        # Display visualizations
        self._display_visualization_tabs(
            viz_tabs, tab_titles, feature_for_analysis, comparison_feature, grouping_feature,
            feature_plots, is_feature_numeric, is_classification
        )

    def _check_if_fallback_needed(self, grouping_feature, is_classification, comparison_feature, feature_plots, is_feature_numeric):
        """Check if fallback plots are needed."""
        needs_fallback = False
        fallback_reason = ""

        if grouping_feature:
            # Check if we need to create fallback plots if no grouped plots were returned
            needs_fallback = not any(key for key in feature_plots.keys() if key not in ["stats", "error"])
            fallback_reason = "grouping feature"
        elif is_classification and comparison_feature == self.target_column:
            # Check if Builder methods returned classification plots when we know it's classification
            expected_plot_types = ["density", "violin", "box"] if is_feature_numeric else ["stacked_bar", "mosaic", "heatmap"]
            has_expected_plots = any(plot_type in feature_plots for plot_type in expected_plot_types)
            needs_fallback = not has_expected_plots
            fallback_reason = "multi-class classification not detected by Builder methods"

        return needs_fallback, fallback_reason

    def _create_fallback_plots(self, feature_plots, grouping_feature, analysis_data, feature_for_analysis,
                             comparison_feature, is_feature_numeric, is_classification):
        """Create fallback plots when Builder methods don't return expected visualizations."""
        # Create a copy of the data for visualization
        if grouping_feature:
            viz_df = pd.DataFrame({
                'feature': analysis_data[feature_for_analysis],
                'target': analysis_data[comparison_feature],
                'group': analysis_data[grouping_feature]
            })
        else:
            viz_df = pd.DataFrame({
                'feature': analysis_data[feature_for_analysis],
                'target': analysis_data[comparison_feature]
            })

        # Generate basic plots based on data types
        if is_feature_numeric:
            if is_classification:
                # For numeric feature vs categorical comparison
                feature_plots = self._create_numeric_classification_plots(
                    feature_plots, viz_df, grouping_feature, feature_for_analysis, comparison_feature
                )
            else:
                # For numeric feature vs numeric comparison
                feature_plots = self._create_numeric_regression_plots(
                    feature_plots, viz_df, grouping_feature, feature_for_analysis, comparison_feature
                )
        else:
            # For categorical feature
            feature_plots = self._create_categorical_plots(
                feature_plots, viz_df, grouping_feature, feature_for_analysis, comparison_feature
            )

        return feature_plots

    def _create_numeric_classification_plots(self, feature_plots, viz_df, grouping_feature, feature_for_analysis, comparison_feature):
        """Create plots for numeric features vs categorical targets."""
        try:
            # Ensure target is treated as categorical
            viz_df['target_str'] = viz_df['target'].astype(str)

            if grouping_feature:
                # Box plot with grouping
                fig_box = px.box(
                    viz_df,
                    x='target_str',
                    y='feature',
                    color='group',
                    title=f"Distribution of {feature_for_analysis} by {comparison_feature}, grouped by {grouping_feature}",
                    labels={"feature": feature_for_analysis, "target_str": comparison_feature, "group": grouping_feature}
                )
                feature_plots["box"] = fig_box

                # Violin plot with grouping
                fig_violin = px.violin(
                    viz_df,
                    x='target_str',
                    y='feature',
                    color='group',
                    box=True,
                    title=f"Distribution of {feature_for_analysis} by {comparison_feature}, grouped by {grouping_feature}"
                )
                feature_plots["violin"] = fig_violin
            else:
                # Box plot without grouping (for multi-class fallback)
                fig_box = px.box(
                    viz_df,
                    x='target_str',
                    y='feature',
                    title=f"Distribution of {feature_for_analysis} by {comparison_feature}",
                    labels={"feature": feature_for_analysis, "target_str": comparison_feature}
                )
                feature_plots["box"] = fig_box

                # Violin plot without grouping
                fig_violin = px.violin(
                    viz_df,
                    x='target_str',
                    y='feature',
                    box=True,
                    title=f"Distribution of {feature_for_analysis} by {comparison_feature}"
                )
                feature_plots["violin"] = fig_violin

                # Density plot (histogram overlay) for classification
                fig_density = px.histogram(
                    viz_df,
                    x='feature',
                    color='target_str',
                    marginal='rug',
                    opacity=0.7,
                    title=f"Distribution of {feature_for_analysis} by {comparison_feature}",
                    labels={"feature": feature_for_analysis, "target_str": comparison_feature}
                )
                feature_plots["density"] = fig_density
        except Exception as e:
            pass

        return feature_plots

    def _create_numeric_regression_plots(self, feature_plots, viz_df, grouping_feature, feature_for_analysis, comparison_feature):
        """Create plots for numeric features vs numeric targets."""
        try:
            # Scatter plot
            fig_scatter = px.scatter(
                viz_df,
                x='feature',
                y='target',
                color='group' if grouping_feature else None,
                title=f"{feature_for_analysis} vs {comparison_feature}" + (f", grouped by {grouping_feature}" if grouping_feature else ""),
                labels={"feature": feature_for_analysis, "target": comparison_feature}
            )
            feature_plots["scatter"] = fig_scatter
        except Exception as e:
            pass

        return feature_plots

    def _create_categorical_plots(self, feature_plots, viz_df, grouping_feature, feature_for_analysis, comparison_feature):
        """Create plots for categorical features."""
        try:
            # Ensure target is treated as categorical
            viz_df['target_str'] = viz_df['target'].astype(str)

            if grouping_feature:
                # Stacked bar chart with grouping
                fig_bar = px.histogram(
                    viz_df,
                    x='feature',
                    color='target_str',
                    barmode='stack',
                    facet_row='group',
                    title=f"Distribution of {comparison_feature} by {feature_for_analysis}, grouped by {grouping_feature}",
                    labels={"target_str": comparison_feature}
                )
                feature_plots["stacked_bar"] = fig_bar
            else:
                # Stacked bar chart without grouping (for multi-class fallback)
                fig_bar = px.histogram(
                    viz_df,
                    x='feature',
                    color='target_str',
                    barmode='stack',
                    title=f"Distribution of {comparison_feature} by {feature_for_analysis}",
                    labels={"target_str": comparison_feature}
                )
                feature_plots["stacked_bar"] = fig_bar

                # Also create a proportion chart for better multi-class visualization
                try:
                    # Calculate proportions
                    prop_data = viz_df.groupby(['feature', 'target_str']).size().reset_index(name='count')
                    prop_data['proportion'] = prop_data.groupby('feature')['count'].transform(lambda x: x / x.sum())

                    fig_prop = px.bar(
                        prop_data,
                        x='feature',
                        y='proportion',
                        color='target_str',
                        title=f"Class Proportions by {feature_for_analysis}",
                        labels={"target_str": comparison_feature, "proportion": "Proportion"}
                    )
                    feature_plots["heatmap"] = fig_prop  # Use heatmap slot for proportions
                except Exception as e:
                    pass
        except Exception as e:
            pass

        return feature_plots

    def _populate_visualization_tabs(self, viz_tabs, tab_titles, feature_plots, is_feature_numeric, is_classification):
        """Populate the visualization tabs based on available plots."""
        if is_feature_numeric:
            if is_classification:
                # Numerical feature vs Categorical target (Binary/Multi-class Classification)
                if "density" in feature_plots:
                    viz_tabs.append(("density", feature_plots["density"]))
                    tab_titles.append("Distribution by Class")
                if "violin" in feature_plots:
                    viz_tabs.append(("violin", feature_plots["violin"]))
                    tab_titles.append("Violin Plot")
                if "box" in feature_plots:
                    viz_tabs.append(("box", feature_plots["box"]))
                    tab_titles.append("Box Plot")
            else:
                # Numerical feature vs Numerical target (Regression)
                if "scatter" in feature_plots:
                    viz_tabs.append(("scatter", feature_plots["scatter"]))
                    tab_titles.append("Scatter Plot")
                if "hexbin" in feature_plots:
                    viz_tabs.append(("hexbin", feature_plots["hexbin"]))
                    tab_titles.append("Density Heatmap")
        else:
            if is_classification:
                # Categorical feature vs Categorical target (Binary/Multi-class Classification)
                if "stacked_bar" in feature_plots:
                    viz_tabs.append(("stacked_bar", feature_plots["stacked_bar"]))
                    tab_titles.append("Stacked Bar")
                if "mosaic" in feature_plots:
                    viz_tabs.append(("mosaic", feature_plots["mosaic"]))
                    tab_titles.append("Mosaic Plot")
                if "heatmap" in feature_plots:
                    viz_tabs.append(("heatmap", feature_plots["heatmap"]))
                    tab_titles.append("Association Heatmap")
            else:
                # Categorical feature vs Numerical target (Regression)
                if "box" in feature_plots:
                    viz_tabs.append(("box", feature_plots["box"]))
                    tab_titles.append("Box Plot")
                if "violin" in feature_plots:
                    viz_tabs.append(("violin", feature_plots["violin"]))
                    tab_titles.append("Violin Plot")
                if "bar" in feature_plots:
                    viz_tabs.append(("bar", feature_plots["bar"]))
                    tab_titles.append("Mean Value by Category")

    def _display_visualization_tabs(self, viz_tabs, tab_titles, feature_for_analysis, comparison_feature, grouping_feature, feature_plots=None, is_feature_numeric=False, is_classification=False):
        """Display the visualization tabs."""
        # Display visualizations in tabs if we have multiple
        if len(viz_tabs) > 0:
            if len(viz_tabs) > 1:
                plot_tabs = st.tabs(tab_titles)
                for (plot_type, plot), tab in zip(viz_tabs, plot_tabs):
                    with tab:
                        # Display the plot description in a styled streamlit container
                        with st.expander("Plot Description and Interpretation Guide"):
                            # Use regular markdown for proper rendering of bold text and emojis
                            st.markdown(self.get_plot_description(plot_type))

                        st.plotly_chart(plot, config={'responsive': True})
                    # Log visualization generation
                    log_data = {
                        "primary_feature": feature_for_analysis,
                        "comparison_feature": comparison_feature,
                        "plot_type": plot_type,
                        "generated": True
                    }

                    if grouping_feature:
                        log_data["grouping_feature"] = grouping_feature

                    self.logger.log_calculation(
                        f"Feature Analysis Plot Generation",
                        log_data
                    )
            else:
                plot_type, plot = viz_tabs[0]
                # Display the plot description in a styled streamlit container
                with st.expander("Plot Description and Interpretation Guide"):
                    # Use regular markdown for proper rendering of bold text and emojis
                    st.markdown(self.get_plot_description(plot_type))

                st.plotly_chart(plot, config={'responsive': True})
                # Log visualization generation
                log_data = {
                    "primary_feature": feature_for_analysis,
                    "comparison_feature": comparison_feature,
                    "plot_type": plot_type,
                    "generated": True
                }

                if grouping_feature:
                    log_data["grouping_feature"] = grouping_feature

                self.logger.log_calculation(
                    f"Feature Analysis Plot Generation",
                    log_data
                )
        else:
            # Debug: Log detailed information about why no visualizations were found
            self.logger.log_error(
                "No Suitable Visualizations Found - Debug Info",
                {
                    "primary_feature": feature_for_analysis,
                    "comparison_feature": comparison_feature,
                    "is_feature_numeric": is_feature_numeric,
                    "is_classification": is_classification,
                    "grouping_feature": grouping_feature,
                    "feature_plots_keys": list(feature_plots.keys()) if feature_plots else [],
                    "feature_plots_has_stats": "stats" in feature_plots if feature_plots else False,
                    "expected_plot_types_for_numeric_classification": ["density", "violin", "box"] if is_feature_numeric and is_classification else "N/A",
                    "expected_plot_types_for_categorical_classification": ["stacked_bar", "mosaic", "heatmap"] if not is_feature_numeric and is_classification else "N/A",
                    "viz_tabs_count": len(viz_tabs)
                }
            )

            st.warning("No suitable visualizations available for this combination of features.")
            st.info("🔍 **Debug Information:**")
            st.write(f"- Feature type: {'Numeric' if is_feature_numeric else 'Categorical'}")
            st.write(f"- Comparison type: {'Classification' if is_classification else 'Regression'}")
            st.write(f"- Available plot keys: {list(feature_plots.keys()) if feature_plots else 'None'}")
            if is_classification:
                expected_plots = ["density", "violin", "box"] if is_feature_numeric else ["stacked_bar", "mosaic", "heatmap"]
                st.write(f"- Expected plot types: {expected_plots}")

            log_data = {
                "primary_feature": feature_for_analysis,
                "comparison_feature": comparison_feature,
                "reason": "No suitable visualisations available"
            }

            if grouping_feature:
                log_data["grouping_feature"] = grouping_feature

            self.logger.log_error(
                "Visualization Generation Failed",
                log_data
            )

    def _analyze_low_information_features(self, correlation_matrix):
        """
        Analyze features for low information quality and recommend removal.
        Uses the associations matrix (which includes categorical associations) when available,
        falling back to correlation-only analysis when needed.
        
        Args:
            correlation_matrix: DataFrame containing feature correlations/associations
            
        Returns:
            List of dictionaries containing low quality feature recommendations
        """
        low_quality_recommendations = []
        
        # Get all feature names (excluding target if it's in the matrix)
        feature_names = [col for col in correlation_matrix.columns if col != self.target_column]
        
        # Check if we have the target in the matrix for better analysis
        has_target_in_matrix = self.target_column in correlation_matrix.columns
        
        for feature in feature_names:
            issues = []
            quality_factors = []
            
            # 1. Target correlation/association analysis
            target_correlation = 0.0
            if has_target_in_matrix:
                # Use the associations matrix value if available (handles both numeric and categorical)
                try:
                    target_correlation = abs(correlation_matrix.loc[feature, self.target_column])
                    if pd.isna(target_correlation):
                        target_correlation = 0.0
                except:
                    target_correlation = 0.0
            else:
                # Fallback to direct correlation calculation
                if self.target_column in self.data.columns:
                    try:
                        # Check if this is a classification problem and target should be treated as categorical
                        is_classification_target = (hasattr(st.session_state, 'problem_type') and 
                                                   st.session_state.problem_type in ['binary_classification', 'multiclass_classification'])
                        
                        if is_classification_target:
                            # For classification, use appropriate association measure
                            is_binary = (hasattr(st.session_state, 'is_binary') and st.session_state.is_binary)
                            
                            if pd.api.types.is_numeric_dtype(self.data[feature]):
                                if is_binary:
                                    # Numeric feature vs binary target - use point-biserial correlation (more standard)
                                    from scipy.stats import pointbiserialr
                                    feature_data = self.data[feature].dropna()
                                    target_data = self.data[self.target_column].dropna()
                                    
                                    # Get common indices
                                    common_idx = feature_data.index.intersection(target_data.index)
                                    if len(common_idx) > 5:  # Need sufficient data
                                        feature_clean = feature_data[common_idx]
                                        target_clean = target_data[common_idx]
                                        
                                        # Convert target to binary numeric (0, 1) if needed
                                        if not pd.api.types.is_numeric_dtype(target_clean):
                                            unique_vals = target_clean.unique()
                                            if len(unique_vals) == 2:
                                                target_numeric = (target_clean == unique_vals[1]).astype(int)
                                            else:
                                                target_numeric = target_clean
                                        else:
                                            target_numeric = target_clean
                                        
                                        if len(target_numeric.unique()) == 2:  # Verify it's actually binary
                                            correlation, p_value = pointbiserialr(target_numeric, feature_clean)
                                            target_correlation = abs(correlation)
                                            # Log the method used
                                            self.logger.log_calculation(
                                                "Target Correlation Method",
                                                {"feature": feature, "method": "point_biserial", "correlation": target_correlation}
                                            )
                                else:
                                    # Numeric feature vs multiclass target - use ANOVA-based eta squared
                                    from scipy.stats import f_oneway
                                    feature_data = self.data[feature].dropna()
                                    target_data = self.data[self.target_column].dropna()
                                    
                                    # Get common indices
                                    common_idx = feature_data.index.intersection(target_data.index)
                                    if len(common_idx) > 5:  # Need sufficient data
                                        feature_clean = feature_data[common_idx]
                                        target_clean = target_data[common_idx]
                                        
                                        # Group by target classes
                                        groups = [feature_clean[target_clean == cls] for cls in target_clean.unique()]
                                        groups = [g for g in groups if len(g) > 1]  # Need >1 sample per group
                                        
                                        if len(groups) >= 2:
                                            f_stat, p_value = f_oneway(*groups)
                                            # Convert F-statistic to eta squared approximation
                                            n_total = len(feature_clean)
                                            n_groups = len(groups)
                                            if n_total > n_groups:
                                                eta_squared = (f_stat * (n_groups - 1)) / (f_stat * (n_groups - 1) + n_total - n_groups)
                                                target_correlation = min(eta_squared, 1.0)  # Cap at 1
                                                # Log the method used
                                                self.logger.log_calculation(
                                                    "Target Correlation Method",
                                                    {"feature": feature, "method": "eta_squared", "correlation": target_correlation}
                                                )
                            else:
                                # Categorical feature vs categorical target
                                from scipy.stats import chi2_contingency
                                contingency = pd.crosstab(self.data[feature], self.data[self.target_column])
                                if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                                    n = contingency.sum().sum()
                                    
                                    if is_binary and self.data[feature].nunique() == 2:
                                        # Both feature and target are binary - use Phi coefficient (simpler)
                                        target_correlation = np.sqrt(chi2 / n)
                                        # Log the method used
                                        self.logger.log_calculation(
                                            "Target Correlation Method",
                                            {"feature": feature, "method": "phi_coefficient", "correlation": target_correlation}
                                        )
                                    else:
                                        # Use Cramér's V for general case (multiclass or mixed)
                                        phi2 = chi2 / n
                                        r, k = contingency.shape
                                        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                                        rcorr = r - ((r-1)**2)/(n-1)
                                        kcorr = k - ((k-1)**2)/(n-1)
                                        if min(kcorr-1, rcorr-1) > 0:
                                            target_correlation = np.sqrt(phi2corr / min(kcorr-1, rcorr-1))
                                            # Log the method used
                                            self.logger.log_calculation(
                                                "Target Correlation Method",
                                                {"feature": feature, "method": "cramers_v", "correlation": target_correlation}
                                            )
                        else:
                            # For regression, use standard correlation
                            if (pd.api.types.is_numeric_dtype(self.data[feature]) and 
                                pd.api.types.is_numeric_dtype(self.data[self.target_column])):
                                target_correlation = abs(self.data[feature].corr(self.data[self.target_column]))
                                if pd.isna(target_correlation):
                                    target_correlation = 0.0
                    except Exception as e:
                        # Log the error for debugging
                        self.logger.log_error(
                            "Target Correlation Calculation Failed", 
                            {"feature": feature, "error": str(e)}
                        )
                        target_correlation = 0.0
            
            # Factor 1: Target correlation/association (weight: 0.4)
            target_score = min(target_correlation * 2, 1.0)  # Scale 0-0.5 to 0-1
            quality_factors.append(('target_correlation', target_score, 0.4))
            
            if target_correlation < 0.05:
                issues.append(f"Very weak target association ({target_correlation:.3f})")
            elif target_correlation < 0.1:
                issues.append(f"Weak target association ({target_correlation:.3f})")
            
            # 2. Variance analysis (adapted for categorical features)
            variance_score = 0.0
            try:
                if pd.api.types.is_numeric_dtype(self.data[feature]):
                    feature_data = self.data[feature].dropna()
                    if len(feature_data) > 1:
                        # Calculate coefficient of variation for better normalization
                        mean_val = feature_data.mean()
                        std_val = feature_data.std()
                        
                        if std_val > 0 and abs(mean_val) > 1e-10:
                            # Use coefficient of variation (CV) normalized to 0-1 scale
                            cv = std_val / abs(mean_val)
                            # Cap CV at reasonable values and normalize to 0-1
                            variance_score = min(cv / 2.0, 1.0)  # CV of 2 = max score of 1
                        elif std_val > 0:
                            # For cases where mean is near zero, use normalized standard deviation
                            # Normalize by the range of the data
                            data_range = feature_data.max() - feature_data.min()
                            if data_range > 0:
                                variance_score = min(std_val / data_range, 1.0)
                            else:
                                variance_score = 0.0
                        else:
                            variance_score = 0.0
                    else:
                        variance_score = 0.0
                else:
                    # For categorical features, use entropy-like measure
                    feature_data = self.data[feature].dropna()
                    if len(feature_data) > 0:
                        value_counts = feature_data.value_counts()
                        probabilities = value_counts / len(feature_data)
                        # Calculate entropy-based diversity score
                        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
                        max_entropy = np.log2(len(value_counts)) if len(value_counts) > 1 else 1
                        variance_score = entropy / max_entropy if max_entropy > 0 else 0
                    else:
                        variance_score = 0.0
            except:
                variance_score = 0.0
            
            # Factor 2: Variance/Diversity (weight: 0.25)
            quality_factors.append(('variance', variance_score, 0.25))
            
            if variance_score < 0.1:
                issues.append(f"Very low variance/diversity ({variance_score:.3f})")
            elif variance_score < 0.2:
                issues.append(f"Low variance/diversity ({variance_score:.3f})")
            
            # 3. Missing values analysis
            missing_percentage = self.data[feature].isnull().mean()
            missing_score = max(1 - missing_percentage * 2, 0)  # Penalty for missing values
            
            # Factor 3: Missing values (weight: 0.15)
            quality_factors.append(('missing_values', missing_score, 0.15))
            
            if missing_percentage > 0.5:
                issues.append(f"Very high missing values ({missing_percentage*100:.1f}%)")
            elif missing_percentage > 0.3:
                issues.append(f"High missing values ({missing_percentage*100:.1f}%)")
            elif missing_percentage > 0.1:
                issues.append(f"Moderate missing values ({missing_percentage*100:.1f}%)")
            
            # 4. Overall associations analysis (uses the full associations matrix)
            avg_correlation = 0.0
            try:
                if feature in correlation_matrix.columns:
                    # Get associations with all other features (this now includes categorical associations)
                    feature_correlations = correlation_matrix[feature].drop([feature, self.target_column], errors='ignore')
                    avg_correlation = abs(feature_correlations).mean()
                    if pd.isna(avg_correlation):
                        avg_correlation = 0.0
            except:
                avg_correlation = 0.0
            
            # Factor 4: Overall relationships (weight: 0.2)
            overall_score = min(avg_correlation * 3, 1.0)  # Scale 0-0.33 to 0-1
            quality_factors.append(('overall_correlation', overall_score, 0.2))
            
            if avg_correlation < 0.05:
                issues.append(f"Very weak relationships with other features ({avg_correlation:.3f})")
            elif avg_correlation < 0.1:
                issues.append(f"Weak relationships with other features ({avg_correlation:.3f})")
            
            # Calculate weighted quality score
            quality_score = sum(score * weight for _, score, weight in quality_factors)
            
            # Determine primary issue
            factor_scores = {factor: score for factor, score, _ in quality_factors}
            primary_issue_factor = min(factor_scores.keys(), key=lambda k: factor_scores[k])
            
            primary_issue_map = {
                'target_correlation': f"Weak target association ({target_correlation:.3f})",
                'variance': f"Low variance/diversity ({variance_score:.3f})",
                'missing_values': f"High missing values ({missing_percentage*100:.1f}%)",
                'overall_correlation': f"Weak feature associations ({avg_correlation:.3f})"
            }
            primary_issue = primary_issue_map[primary_issue_factor]
            
            # Generate recommendation
            if quality_score < 0.3:
                recommendation = "Strong candidate for removal - provides minimal information value"
            elif quality_score < 0.5:
                recommendation = "Consider removing - low information content may not justify inclusion"
            elif quality_score < 0.7:
                recommendation = "Monitor - may benefit from feature engineering or transformation"
            else:
                recommendation = "Feature appears to have acceptable information quality"
            
            # Include all features with quality issues in the analysis (< 0.7) for display purposes
            if quality_score < 0.7:
                low_quality_recommendations.append({
                    'feature': feature,
                    'quality_score': quality_score,
                    'target_correlation': target_correlation,
                    'variance': variance_score,
                    'missing_percentage': missing_percentage,
                    'avg_correlation': avg_correlation,
                    'primary_issue': primary_issue,
                    'issues': issues,
                    'recommendation': recommendation,
                    'quality_factors': quality_factors
                })
        
        # Log the low quality analysis
        self.logger.log_calculation(
            "Low Information Quality Analysis",
            {
                "features_analyzed": len(feature_names),
                "low_quality_features": len(low_quality_recommendations),
                "severe_issues": len([r for r in low_quality_recommendations if r['quality_score'] < 0.3]),
                "moderate_issues": len([r for r in low_quality_recommendations if 0.3 <= r['quality_score'] < 0.5]),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return low_quality_recommendations