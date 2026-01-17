import streamlit as st
from Builder import Builder, ModelStage
from app import create_sidebar_navigation
from utils.logging.journey_viewer import render_journey_viewer
from utils.logging.log_viewer import render_log_viewer
from datetime import datetime
from components.data_exploration.feature_analysis import show_feature_analysis, get_visualisation_info
from components.data_exploration.target_feature_analysis import analyse_feature_relationships, display_target_distribution
from components.data_exploration.feature_relationships import FeatureRelationshipsComponent
from components.data_exploration.data_quality import DataQualityAnalysis
from components.data_exploration.render_advanced_automated_preprocessing import render_advanced_automated_preprocessing
from utils.dataset_overview import DatasetOverviewComponent
from streamlit_scroll_to_top import scroll_to_here

def scroll():  
    st.session_state.scroll_to_top = True

def main():
    st.title("Data Exploration")

    # Initialize session state variables if they don't exist
    if 'scroll_to_top' not in st.session_state:
        st.session_state.scroll_to_top = False
    
    # Add consistent navigation
    create_sidebar_navigation()
    
    # Initialize session state if needed
    if 'builder' not in st.session_state:
        st.session_state.builder = Builder()
        st.session_state.logger.log_stage_transition("START", "DATA_EXPLORATION")
    
    # Set current stage to DATA_EXPLORATION
    st.session_state.builder.current_stage = ModelStage.DATA_EXPLORATION
    
    # Get and render stage info
    stage_info = st.session_state.builder.get_current_stage_info()
    
    # Edge case handling: occasionally stage_info may not be ready on first call (e.g., on refresh)
    if not stage_info:
        if 'logger' in st.session_state:
            st.session_state.logger.log_error(
                "Stage Info Fetch - Retry",
                {
                    "stage": "DATA_EXPLORATION",
                    "reason": "stage_info was None or empty on first attempt",
                    "timestamp": datetime.now().isoformat()
                }
            )
        # Retry fetching once
        stage_info = st.session_state.builder.get_current_stage_info()

    # As a final safeguard, ensure stage_info has the expected structure
    if not stage_info:
        stage_info = {
            "title": "Data Exploration",
            "description": "Explore your dataset to understand its structure, quality, and key relationships before preprocessing and modeling.",
            "requirements": [],
            "ethical_considerations": []
        }
    
    # Enhanced logging for page initialization
    if 'logger' in st.session_state:
        data_status = {
            "data_loaded": bool(st.session_state.builder.data is not None),
            "data_shape": None if st.session_state.builder.data is None else st.session_state.builder.data.shape,
            "target_selected": bool(st.session_state.builder.target_column),
            "target_column": st.session_state.builder.target_column,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log page state with enhanced information
        st.session_state.logger.log_page_state("Data_Exploration", data_status)
        
        # Log data exploration page visit
        st.session_state.logger.log_user_action(
            "Page Navigation", 
            {
                "page": "Data Exploration",
                "data_available": bool(st.session_state.builder.data is not None),
                "target_selected": bool(st.session_state.builder.target_column)
            }
        )
    
    st.header(stage_info["title"])
    st.write(stage_info["description"])
    
    # Create two columns for requirements and ethical considerations
    req_col, eth_col = st.columns(2)
    
    with st.expander("Functionality", expanded=False):
        for req in stage_info["requirements"]:
            st.markdown(req)

    with st.expander("Ethical Considerations", expanded=False):
        for consideration in stage_info["ethical_considerations"]:
            st.markdown(consideration)

    st.write("---")
    st.subheader("Duplicate Check and Removal")
    st.write("Removing duplicates ensures datasets are clean, efficient, and representative, leading to more reliable analyses and effective machine learning models.")
    st.write("We will check for both exact duplicates and partial duplicates (same feature values but different target values).")
    
    if st.session_state.builder.data is not None and st.session_state.builder.target_column:
        # Check for duplicates before proceeding with data summary
        data = st.session_state.builder.data
        initial_row_count = len(data)
        
        # Check for exact duplicates
        exact_duplicates = data.duplicated().sum()
        if exact_duplicates > 0:
            st.warning(f"Found {exact_duplicates} exact duplicate rows. The first occurrence will be kept.")
            # Remove exact duplicates
            data = data.drop_duplicates(keep='first')
            rows_after_exact = len(data)
            st.session_state.builder.data = data
            st.session_state.logger.log_calculation(
                "Duplicate Removal",
                {
                    "exact_duplicates_removed": int(exact_duplicates),
                    "initial_rows": initial_row_count,
                    "rows_after_exact": rows_after_exact
                }
            )
        else:
            st.success("No exact duplicates found in the dataset")
            rows_after_exact = len(data)
            
        # Check for partial duplicates (same values in all columns except target)
        non_target_cols = [col for col in data.columns if col != st.session_state.builder.target_column]
        partial_duplicates = data.duplicated(subset=non_target_cols).sum()
        if partial_duplicates > 0:
            st.warning(f"Found {partial_duplicates} partial duplicate rows (same feature values but different target values).")
            # Remove partial duplicates
            data = data.drop_duplicates(subset=non_target_cols)
            final_row_count = len(data)
            st.session_state.builder.data = data
            st.session_state.logger.log_calculation(
                "Duplicate Removal",
                {
                    "partial_duplicates_removed": int(partial_duplicates),
                    "rows_after_exact": rows_after_exact,
                    "final_rows": final_row_count
                }
            )
        else:
            st.success("No partial duplicates found in the dataset")
            final_row_count = len(data)

        # Show total reduction summary if any duplicates were removed
        if exact_duplicates > 0 or partial_duplicates > 0:
            total_reduction = initial_row_count - final_row_count
            reduction_percentage = (total_reduction / initial_row_count) * 100
           
            st.write("### Summary of Duplicate Removal")
            st.write(f"- Initial dataset size: {initial_row_count:,} rows")
            st.write(f"- Final dataset size: {final_row_count:,} rows")
            st.write(f"- Total rows removed: {total_reduction:,} ({reduction_percentage:.2f}%)")
            st.session_state.logger.log_calculation(
                "Duplicate Removal Summary",
                {
                    "initial_rows": initial_row_count,
                    "final_rows": final_row_count,
                    "total_reduction": total_reduction,
                    "reduction_percentage": reduction_percentage
                }
            )

        summary = st.session_state.builder.get_data_summary()
        
        if "error" in summary:
            st.error(summary["error"])
            st.session_state.logger.log_error(
                "Data Summary Failed",
                {"error": summary["error"]}
            )
            return
            
        # Display problem type information from session state
        if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type:
            problem_type = st.session_state.problem_type
            num_classes = st.session_state.builder.data[st.session_state.builder.target_column].nunique()
            
            if problem_type == "binary_classification":
                st.info(f"🎯 **Problem Type**: Binary Classification (2 classes)")
            elif problem_type == "multiclass_classification":
                st.info(f"🎯 **Problem Type**: Multiclass Classification ({num_classes} classes)")
            elif problem_type == "regression":
                st.info(f"🎯 **Problem Type**: Regression (continuous target)")
            else:
                st.info(f"🎯 **Problem Type**: {problem_type}")
            
            # Log problem type detection for debugging
            st.session_state.logger.log_calculation(
                "Problem Type Detection in Data Exploration",
                {
                    "problem_type": problem_type,
                    "is_binary": getattr(st.session_state, 'is_binary', False),
                    "is_multiclass": getattr(st.session_state, 'is_multiclass', False),
                    "is_regression": getattr(st.session_state, 'is_regression', False),
                    "num_classes": num_classes,
                    "target_column": st.session_state.builder.target_column,
                    "used_session_state": True
                }
            )
        else:
            st.warning("⚠️ Problem type not detected from session state. Using heuristic detection in components.")
            # Log that session state was not available
            st.session_state.logger.log_calculation(
                "Problem Type Detection in Data Exploration - Fallback",
                {
                    "used_session_state": False,
                    "fallback_reason": "Session state problem_type not available",
                    "target_column": st.session_state.builder.target_column
                }
            )
            
        # Log data quality metrics
        st.session_state.logger.log_calculation(
            "Data Quality Metrics",
            {
                "missing_values": summary["missing_values"],
                "dtypes": {k: str(v) for k, v in summary["dtypes"].items()}
            }
        )
        
        # Basic Statistics
        if "summary" in summary:
            st.write("---")
            
            # Add custom CSS for better styled cards
            st.markdown("""
            <style>
            div.data-card {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
                border-left: 4px solid #4e8df5;
            }
            div.data-card h4 {
                color: #1e3a8a;
                margin-top: 0;
                margin-bottom: 10px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Use the DatasetOverviewComponent to display dataset overview
            dataset_overview = DatasetOverviewComponent(st.session_state.builder.data, st.session_state.logger)
            dataset_overview.display_overview()
            
            # Set position to scroll to the top of the page
            if st.session_state.scroll_to_top:
                scroll_to_here(0, key='top')  # Scroll to the top of the page, 0 means instantly, but you can add a delay (im milliseconds)
                st.session_state.scroll_to_top = False  # Reset the state after scrolling

            # Continue with the rest of the data exploration page
            # Quality Score
            st.write("---")
            st.subheader("Data Analysis")

            # Create pills for different types of analysis
            selected_section = st.pills(
                "Choose Analysis Section",
                ["Target Feature Analysis", 
                "Feature Analysis",
                "Correlation/Association Analysis",
                "Feature Relationships",
                "Data Quality Analysis"],
                key="analysis_section",
                default="Target Feature Analysis"
            )
            
            # Log user selection of analysis section
            st.session_state.logger.log_user_action(
                "Analysis Section Selected",
                {
                    "selected_section": selected_section,
                    "timestamp": datetime.now().isoformat()
                }
            )

            # Target Distribution Section
            if selected_section == "Target Feature Analysis":
                st.write("""
                ### Target Feature Analysis

                This section helps you understand your target variable and how it relates to other features in your dataset. It provides two key insights:

                1. **Target Distribution Analysis**
                - Visualizes how your target variable is distributed
                - **Binary Classification**: Shows the count of samples in each of the 2 classes
                - **Multiclass Classification**: Shows the count of samples across all classes (3+ classes)
                - **Regression**: Shows the spread of continuous numerical values
                - Helps identify class imbalance (classification) or data distribution patterns (regression)

                2. **Feature-Target Relationships**
                - Analyses how each feature relates to your target variable
                - Automatically detects whether relationships are linear, non-linear, or complex
                - Provides strength assessments (Strong, Moderate, Weak, Very Weak) for each relationship
                - Shows relevant statistical metrics based on your problem type:
                    - **Regression**: Pearson correlation, Spearman correlation, Mutual Information
                    - **Binary Classification**: Point-Biserial correlation, ANOVA F-test, Chi-square tests, Cramer's V
                    - **Multiclass Classification**: ANOVA F-test (3+ classes), Chi-square tests, Cramer's V, Eta Squared
                - Features are organized into separate tabs for numerical and categorical variables

                💡 **How to use this analysis:**
                - Look at the target distribution first to understand if you need to address any class imbalance
                - Focus on features marked as "Strong" or "Moderate" - these are your most predictive variables
                - For classification tasks, pay attention to features with good "Class Separation"
                - For regression tasks, look for strong linear or non-linear relationships
                - Use the detailed metrics and explanations in the expandable sections to dive deeper into specific relationships
                
                """)
                display_target_distribution(st.session_state.builder.data, st.session_state.builder.target_column)
                analyse_feature_relationships(st.session_state.builder.data, st.session_state.builder.target_column)
            
            # Numerical Distributions Section
            elif selected_section == "Feature Analysis":
                st.subheader("Feature Analysis")
                st.write("""
                ### Understanding Your Dataset's Features
                
                The Feature Analysis section provides interactive visualisations and statistical insights to help you:
                
                **For Numerical Features:**
                - Visualize distributions through histograms and density plots
                - Identify outliers through box plots and statistical measures
                - Examine relationships with the target variable through scatter plots
                - Assess feature importance and predictive power
                - Detect potential scaling or transformation needs
                
                **For Categorical Features:**
                - View category distributions and frequencies
                - Identify imbalanced or rare categories
                - Analyse relationships with the target variable
                - Evaluate encoding requirements
                - Detect potential data quality issues
                
                Select a feature below to explore its characteristics and relationship with the target variable.
                """)
                
                get_visualisation_info()
                show_feature_analysis(st.session_state.builder.data, st.session_state.builder.target_column)
            
            # Correlation/Association Analysis Section
            elif selected_section == "Correlation/Association Analysis":
                
                st.write("""
                ### Understanding Correlation/Association Analysis

                This section provides comprehensive analysis of relationships between features in your dataset, helping you understand feature dependencies and identify redundant information.

                #### 🔍 **What You'll Find Here:**
                
                **1. Feature Correlation/Association**
                - **Mixed-type correlation matrix**: Works with both numeric and categorical features
                - **Interactive heatmap**: Visual representation with correlation values displayed on each cell
                - **Comprehensive relationships**: Uses advanced methods (Pearson, Cramér's V, correlation ratios) for different data types
                - **Easy interpretation**: Color-coded with exact numerical values for precision
                
                **2. Low Information Quality Features**
                - **An analysis of features that provide little predictive value with recommendations on which features to keep or remove
                - **Low Target Correlation**: Features with very weak relationships to the target variable
                - **Low Variance**: Features with little variation (mostly constant values)
                - **High Missing Values**: Features with excessive missing data
                - **Weak Overall Relationships**: Features that don't correlate strongly with any other features
                
                **3. Correlation Groups Analysis**  
                - **Intelligent grouping**: Identifies clusters of highly correlated features using network analysis
                - **Redundancy detection**: Finds features that provide overlapping information
                - **Smart recommendations**: Advanced algorithm that prioritizes redundancy removal while preserving predictive power
                - **Feature selection guidance**: Data-driven suggestions on which features to keep or remove
                - **Impact analysis**: Shows potential feature reduction and its benefits
                
                #### 💡 **Why This Matters:**
                - **Reduce overfitting** by removing redundant features
                - **Improve model performance** through cleaner feature sets  
                - **Speed up training** with fewer, more distinct features
                - **Enhance interpretability** by focusing on unique information
                - **Prevent multicollinearity** issues in linear models
                
                #### 🎯 **Perfect For:**
                - Feature selection and engineering decisions
                - Understanding your data's structure and relationships
                - Preparing optimal feature sets for model training
                - Identifying data collection redundancies
                - Creating more interpretable and robust models
                
                """)
                # Feature Relationships Sections - Use new component
                feature_relationships = FeatureRelationshipsComponent(st.session_state.builder, st.session_state.logger)
                
                # Display Feature Associations Analysis
                feature_relationships.display_feature_associations_analysis(summary)

                # Display Correlation Groups Analysis - use the same summary that contains dython associations
                feature_relationships.display_correlation_group_analysis(summary)

            # Feature Relationships Section
            elif selected_section == "Feature Relationships":
                st.subheader("Feature Relationship Analysis")

                # Add explanation guide for this section
                st.markdown("""
                        ##### How to Use This Section

                        This tool helps you analyse the relationship between any two features in your dataset:

                        1. **Select features to analyse:** Choose a primary feature and a comparison feature (default is your target variable)
                        2. **Optionally select a grouping feature:** Add a third dimension to your analysis by grouping the data
                        3. **Review statistical analysis:** Examine the statistical significance of the relationship
                        4. **Explore visualisations:** Different chart types will automatically be selected based on data types

                        The appropriate visualisations and statistical tests are automatically chosen based on whether your 
                        features are categorical or numerical. This helps you understand how features relate to each other 
                        and to your target variable, which is crucial for feature selection and engineering.
                        """)

                # Feature Relationships Sections - Use new component
                feature_relationships = FeatureRelationshipsComponent(st.session_state.builder, st.session_state.logger)
                                
                # Display Detailed Feature Relationship Analysis
                feature_relationships.display_detailed_feature_relationship_analysis(summary)

            # Data Quality Analysis Section
            elif selected_section == "Data Quality Analysis":
                st.write("""
                ### Understanding Data Quality Analysis
                
                Data quality analysis helps you:
                - Assess the overall health of your dataset
                - Identify potential data integrity issues
                - Understand data completeness and consistency
                - Spot patterns in data validity
                - Make informed decisions about data cleaning steps
                """)
                # Log data quality analysis initiation
                st.session_state.logger.log_calculation(
                    "Data Quality Analysis Started",
                    {"timestamp": datetime.now().isoformat()}
                )
                
                # Initialize and use our data quality component
                data_quality = DataQualityAnalysis(st.session_state.builder)
                
                # Render missing values analysis
                data_quality.render_missing_values_analysis()
                
                # Render data quality analysis
                data_quality.render_data_quality_analysis()

            # Mark data exploration as complete before showing the proceed button
            st.session_state.builder.stage_completion[ModelStage.DATA_EXPLORATION] = True

            st.write(" ")
            st.write(" ")
            nav_cols = st.columns([1, 2, 1])
            with nav_cols[2]:
                st.button("Scroll to Top", on_click=scroll)

    # Render advanced automated preprocessing - moved outside the data check block
    # so it can display completion status even after preprocessing is done
    render_advanced_automated_preprocessing()

    # Only show "Proceed to Data Preprocessing" button if advanced preprocessing has not been completed
    if not st.session_state.get('advanced_preprocessing_completed', False):
        st.write("---")
        if st.button("Proceed to Data Preprocessing",type="primary", key="proceed_to_dp_button"):
            st.session_state.builder.stage_completion[ModelStage.DATA_EXPLORATION] = True
            
            # Enhanced logging for transition to Data Preprocessing
            st.session_state.logger.log_calculation(
                "Data Exploration Results",
                {
                    "data_shape": st.session_state.builder.data.shape,
                    "preprocessing_completed": False,
                    "auto_preprocessing_used": False,
                    "feature_count": len(st.session_state.builder.data.columns) - 1  # Excluding target
                }
            )
            
            st.session_state.logger.log_journey_point(
                stage="DATA_EXPLORATION",
                decision_type="DATA_EXPLORATION_COMPLETED",
                description="Data Exploration completed",
                details={
                    "Data Shape": st.session_state.builder.data.shape,
                    "Feature Count": len(st.session_state.builder.data.columns) - 1,  # Excluding target
                    "Duplicate Removal Completed": True,
                    "Auto Preprocessing Used": False
                },
                parent_id=None
            )
            st.session_state.logger.log_stage_transition(
                "DATA_EXPLORATION",
                "DATA_PREPROCESSING",
                {
                    "auto_preprocessing_skipped": True,
                    "rows": len(st.session_state.builder.data),
                    "columns": len(st.session_state.builder.data.columns)
                }
            )
            
            st.session_state.logger.log_user_action(
                "Navigation",
                {"direction": "forward", "to_stage": "DATA_PREPROCESSING"}
            )
            next_page = "3_Data_Preprocessing"
            st.switch_page(f"pages/{next_page}.py")
    

    # At the end of each page's script
    if 'logger' in st.session_state:
        st.session_state.logger.flush_logs()
    
    render_journey_viewer(expanded=True)
    st.write("---")
   
    # Add log viewer
    render_log_viewer()

     # Bottom footer with version and copyright
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666666; padding: 10px;'>
        <small>Version 1.0.0 | Copyright © 2025, Richard Wheeler. All rights reserved.</small><br>
        <small>ML Model Development Guide</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 