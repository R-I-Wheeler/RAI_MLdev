import streamlit as st
import pandas as pd
import plotly.graph_objects as go

class DataQualityAnalysis:
    def __init__(self, builder):
        self.builder = builder
        
    def render_missing_values_analysis(self):
        """
        Renders the missing values analysis section.
        """
        st.write("""
        ### Understanding Missing Values Patterns
        
        Analyzing missing values helps you:
        - Identify patterns in data completeness
        - Understand potential biases in your dataset
        - Make informed decisions about imputation strategies
        - Assess the impact of missing data on your model
        - Determine if certain features should be dropped
        """)
        st.subheader("Missing Values Analysis")
        with st.expander("ℹ️ Understanding Missing Values Analysis"):
            explanation = self.builder.get_calculation_explanation("missing_values")
            st.write("**Method:**")
            st.markdown(explanation["method"])
            st.write("**How to Interpret Results:**")
            st.markdown(explanation["interpretation"])
        
        st.write("### Missing Values Analysis")
        
        # Get missing value statistics
        missing_analysis = self.builder.analyse_missing_values()
        if missing_analysis["success"]:
            # Overview statistics
            col1, col2, col3 = st.columns(3)
            total_missing = missing_analysis["stats"]["total_missing"]
            total_cells = len(self.builder.data) * len(self.builder.data.columns)
            
            with col1:
                st.metric(
                    "Total Missing Values",
                    total_missing,
                    f"{(total_missing/total_cells)*100:.2f}% of all data"
                )
            with col2:
                st.metric(
                    "Columns with Missing Values",
                    sum(1 for x in missing_analysis["stats"]["missing_by_column"].values() if x > 0)
                )
            with col3:
                st.metric(
                    "Rows with Any Missing",
                    self.builder.data.isnull().any(axis=1).sum(),
                    f"{(self.builder.data.isnull().any(axis=1).sum()/len(self.builder.data))*100:.2f}%"
                )
            
            # Detailed missing values table
            st.write("#### Missing Values by Column")
            missing_df = pd.DataFrame({
                'Column': list(missing_analysis["stats"]["missing_by_column"].keys()),
                'Missing Count': list(missing_analysis["stats"]["missing_by_column"].values()),
                'Missing Percentage': [
                    f"{missing_analysis['stats']['missing_percentage'][col]:.2f}%"
                    for col in missing_analysis["stats"]["missing_by_column"].keys()
                ]
            }).sort_values('Missing Count', ascending=False)
            
            # Only show columns with missing values
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            if not missing_df.empty:
                st.dataframe(missing_df, width='stretch')
        
                col1, col2 = st.columns([2,1])
                with col1:
                    # Missing values heatmap
                    st.write("#### Missing Values Heatmap")

                    # Check if summary exists and has visualisations
                    if hasattr(self.builder, 'summary') and "visualisations" in self.builder.summary:
                        summary = self.builder.summary
                        if "missing_heatmap" in summary["visualisations"]:
                            st.write("### Missing Values Heatmap")
                            st.info("""
                                📊 **Missing Values Heatmap**
                                - Dark spots show missing values
                                - Look for patterns in missing data
                                - Vertical patterns suggest feature-specific issues
                                - Horizontal patterns suggest row/observation issues
                            """)

                    st.plotly_chart(
                        missing_analysis["visualisations"]["pattern"],
                        width='stretch'
                    )
                with col2:
                    # Missing value correlations
                    st.write("#### Missing Value Correlations")
                    if hasattr(self.builder, 'summary') and "visualisations" in self.builder.summary:
                        summary = self.builder.summary
                        if "missing_correlation" in summary["visualisations"]:
                            st.info("""
                                📊 **Missing Value Correlations**
                                - Shows relationships between missing values
                                - Strong correlations suggest systematic missing patterns
                                - Can help identify if values are Missing Not At Random (MNAR)
                                - Useful for choosing imputation strategies
                            """)

                    st.plotly_chart(
                        missing_analysis["visualisations"]["correlations"],
                        width='stretch'
                    )
                
                # Add recommendations
                st.write("#### Recommendations")
                for col in missing_df['Column']:
                    missing_pct = missing_analysis["stats"]["missing_percentage"][col]
                    #with st.expander(f"Recommendations for {col}"):
                    if missing_pct > 50:
                        st.warning(
                            f"⚠️ **{col}**: High number of missing values ({missing_pct:.1f}%). "
                            "Consider dropping this column or investigating why so much data is missing."
                        )
                    elif missing_pct > 25:
                        st.warning(
                            f"⚠️ **{col}**: Significant missing values ({missing_pct:.1f}%). "
                            "Consider advanced imputation methods like KNN or investigate the missing data pattern."
                        )
                    else:
                        st.info(
                            f"ℹ️ **{col}**: Moderate missing values ({missing_pct:.1f}%). "
                            "Could be handled with standard imputation methods in the missing values section of Data Preprocessing."
                        )
            else:
                st.success("🎉 No missing values found in the dataset!")
    
    def render_data_quality_analysis(self):
        """
        Renders the data quality analysis section.
        """
        # Get model limitations analysis results first
        limitations_result = self.builder.analyse_data_quality(self.builder.data)
        if not limitations_result["success"]:
            error_msg = limitations_result.get('message', 'Unknown error')
            st.error(f"Error analyzing data quality: {error_msg}")
            self.builder.logger.log_error(
                "Data Quality Analysis Failed",
                {"error": error_msg}
            )
            return
        
        st.subheader("📋 Data Quality Analysis")
        with st.expander("ℹ️ Understanding Data Quality Metrics"):
            st.markdown("""
                ### 📊 Understanding Data Quality Metrics
                
                Our comprehensive data quality assessment considers three key dimensions, with the final Quality Score being a weighted combination of these dimensions (40% Completeness, 30% Consistency, 30% Validity):
                
                #### 1. Completeness (0-100)
                - **What it measures**: Data presence and missing value patterns
                - **How it's calculated**: 
                    * Direct calculation: 100 - (percentage of missing values in column)
                    * Example: If a column has 5% missing values, its completeness score is 95
                - **Scoring criteria**:
                    * 90-100: Minimal missing data (<10%)
                    * 70-89: Some missing data (11-30%)
                    * 50-69: Significant gaps (31-50%)
                    * <50: Critical missing data (>50%)
                
                #### 2. Consistency (0-100)
                - **What it measures**: Data reliability and uniformity
                - **How it's calculated**: Average of two components
                    * Type Consistency: Percentage of values matching the expected data type
                    * Range Consistency:
                        - For numeric columns: Percentage of values within [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
                        - For non-numeric columns: Always 100%
                - **Scoring criteria**:
                    * 90-100: Highly consistent data types and ranges
                    * 70-89: Minor type or range inconsistencies
                    * 50-69: Notable inconsistencies
                    * <50: Major consistency issues
                
                #### 3. Validity (0-100)
                - **What it measures**: Data accuracy and correctness
                - **How it's calculated**:
                    * For numeric columns: Percentage of values that can be converted to numbers
                    * For non-numeric columns: Percentage of non-empty strings
                - **Scoring criteria**:
                    * 90-100: All or nearly all values are valid
                    * 70-89: Most values are valid
                    * 50-69: Significant invalid data
                    * <50: Major validity issues
                
                #### 🎯 Final Quality Score Calculation
                The overall quality score for each column is calculated as:
                ```
                Quality Score = (0.4 × Completeness) + (0.3 × Consistency) + (0.3 × Validity)
                ```
                
                #### 🔍 Additional Metrics Tracked
                - **Missing Values (%)**: Direct percentage of missing values
                - **Unique Values (%)**: Percentage of unique values relative to total rows
                
                #### 💡 How to Use These Scores
                - Focus first on columns with low completeness scores as they have the highest weight
                - Investigate type mismatches in columns with low consistency scores
                - Check for data entry issues in columns with low validity scores
                - Consider dropping or imputing columns with very low overall quality scores
                - Pay special attention to columns where multiple metrics are poor
                """)

        if "data_quality" in limitations_result:
            # Display quality scores with detailed breakdown
            quality_df = pd.DataFrame(limitations_result["data_quality"]).T
            
            # Create two columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Quality Scores by Feature")
                styled_df = quality_df.style.background_gradient(
                    subset=['Quality Score'],
                    cmap='RdYlGn',
                    vmin=0,
                    vmax=100
                )
                st.dataframe(styled_df)
            
            with col2:
                st.info("""
                    **Reading the Scores:**
                    
                    🟢 Green = High quality
                    🟡 Yellow = Moderate quality
                    🔴 Red = Low quality
                    
                    Hover over scores for details!
                """)

            # Add quality insights
            st.subheader("📊 Quality Insights")
            
            # Calculate overall statistics
            avg_score = quality_df['Quality Score'].mean()
            lowest_feature = quality_df['Quality Score'].idxmin()
            lowest_score = quality_df['Quality Score'].min()
            
            # Log quality metrics
            self.builder.logger.log_calculation(
                "Data Quality Metrics",
                {
                    "average_quality_score": avg_score,
                    "lowest_quality_feature": lowest_feature,
                    "lowest_quality_score": lowest_score,
                    "quality_scores": quality_df.to_dict('records')
                }
            )
            
            # Display insights
            cols = st.columns(3)
            with cols[0]:
                st.metric("Average Quality Score", f"{avg_score:.1f}")
            with cols[1]:
                st.metric("Lowest Quality Feature", lowest_feature)
            with cols[2]:
                st.metric("Lowest Quality Score", f"{lowest_score:.1f}")
            
            # Add recommendations based on scores
            st.subheader("💡 Quality Improvement Suggestions")
            
            # Low quality features
            low_quality = quality_df[quality_df['Quality Score'] < 70]
            if not low_quality.empty:
                st.warning("Features Needing Attention:")
                for feature in low_quality.index:
                    score = low_quality.loc[feature, 'Quality Score']
                    st.markdown(f"""
                        **{feature}** (Score: {score:.1f})
                        - Consider additional data validation
                        - Check for missing or incorrect values
                        - Verify data collection process
                    """)
                    # Log recommendation for low quality feature
                    self.builder.logger.log_recommendation(
                        "Low Quality Feature Detected",
                        {
                            "feature": feature,
                            "quality_score": score,
                            "recommendation": "Consider additional data validation and cleaning"
                        }
                    )
            else:
                st.success("All features have acceptable quality scores! 🎉")
                self.builder.logger.log_calculation(
                    "Data Quality Check",
                    {"status": "all_features_acceptable"}
                )
            
            # General recommendations
            st.markdown("""
                ### General Quality Tips:
                
                1. **Regular Monitoring**
                    - Track quality scores over time
                    - Set up alerts for score drops
                    - Review data collection processes
                
                2. **Data Validation**
                    - Implement input validation
                    - Set up automated quality checks
                    - Document acceptable value ranges
                
                3. **Documentation**
                    - Keep track of quality issues
                    - Document cleaning procedures
                    - Maintain data dictionaries
            """) 