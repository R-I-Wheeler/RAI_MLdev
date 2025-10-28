import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from components.data_exploration.feature_relationships import FeatureRelationshipsComponent
from components.data_exploration.target_feature_analysis import analyse_feature_relationships
from utils.dataset_overview import DatasetOverviewComponent
from utils.dataframe_comparison import DataframeComparisonComponent
from utils.data_exploration_component import DataExplorationComponent

class FinalDataReviewComponent:
    """
    Component for displaying the final data review dashboard at the end of the preprocessing stage.
    Shows overall metrics, data quality warnings, duplicate detection, and dataset previews.
    
    This component provides a comprehensive view of the preprocessed data before moving to feature selection,
    including data quality warnings, metrics, and visualisations.
    
    Example usage:
    ```python
    # Initialize the component
    final_review = FinalDataReviewComponent(builder, logger)
    
    # Display the final review dashboard
    final_review.display_final_review()
    ```
    """
    
    def __init__(self, builder, logger):
        """
        Initialize the component with builder and logger instances.
        
        Args:
            builder: The Builder instance with data and model building methods
            logger: The Logger instance for tracking user actions and errors
        """
        self.builder = builder
        self.logger = logger
        
    def display_final_review(self):
        """
        Display the complete final data review dashboard including:
        - Duplicate check and removal
        - Data metrics dashboard
        - Data quality warnings
        - Dataset previews and summaries
        """
        
        st.write("### Final Review")
        st.write("""
            Review your preprocessed dataset before moving to feature selection:
            - Check the final dataset preview
            - Verify the number of rows and columns
            - Confirm all preprocessing steps were applied
            - Verify data type consistency
            
            Make sure:
            - No missing values remain
            - All categorical variables are encoded
            - Outliers have been handled appropriately
            - Data types are appropriate for each feature
        """)
        
        st.write("---")
        self._check_and_remove_duplicates()
        #self._display_training_data_dashboard()

        st.subheader("Final Datasets")
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
        show_dataset_overview = st.toggle(label='Show Training/Testing Data Overview - Toggle off for Testing data', value=True, key="show_dataset_overview")
        if show_dataset_overview:
            st.subheader("Final Training Dataset")
            # Use the DatasetOverviewComponent to display dataset overview
            dataset_overview = DatasetOverviewComponent(st.session_state.builder.training_data, st.session_state.logger)
            dataset_overview.display_overview()
        else:
            st.subheader("Final Testing Dataset")
            dataset_overview = DatasetOverviewComponent(st.session_state.builder.testing_data, st.session_state.logger, keyidentifier="testing_data")
            dataset_overview.display_overview()
        st.write("---")
        st.markdown("""
                    ### Training Data Exploration
                    
                    Access the training data exploration dialog to view and analyse the training data in detail.
                    """)
        col1, col2, col3 = st.columns(3)
        with col1:
            @st.dialog(title="Training Data Exploration", width="large")
            def data_explorer_dialog():
                data_explorer = DataExplorationComponent(self.builder, st.session_state.logger, data=st.session_state.builder.training_data, target_column=st.session_state.builder.target_column)
                data_explorer.render()

            if st.button("Training Data Exploration",on_click=st.rerun):
                data_explorer_dialog()
        with col2:
            st.write("")
        with col3:
            st.write("")

        st.write("---")
        st.markdown("""
                    The following is a comparison of the original training dataset from the train/test split stage and the final training dataset after all preprocessing steps were applied.
                    """)
        comparison_component = DataframeComparisonComponent(
                original_df=st.session_state.tts_original_data,
                modified_df=self.builder.training_data,
                target_column=self.builder.target_column)
        comparison_component.render()
        # Create pills for different types of analysis
        #selected_section = st.pills(
        #    "Choose Analysis Section",
        #    ["Feature-Target Relationships", 
        #    "Feature Associations",
        #    "Feature Relationships"],
        #    key="analysis_section",
        #    default="Feature-Target Relationships"
        #)
        
        # Feature Relationships Sections - Use new component
        #feature_relationships = FeatureRelationshipsComponent(st.session_state.builder, st.session_state.logger, st.session_state.builder.training_data, st.session_state.builder.target_column)
        
        #if selected_section == "Feature-Target Relationships":
        #    analyse_feature_relationships(st.session_state.builder.training_data, st.session_state.builder.target_column)

        #if selected_section == "Feature Associations":
            # Display Feature Associations Analysis
        #    feature_relationships.display_feature_associations_analysis()

        #if selected_section == "Feature Relationships":
            # Display Detailed Feature Relationship Analysis
        #    feature_relationships.display_detailed_feature_relationship_analysis()
        
        st.write("---")
        
        

    def _check_and_remove_duplicates(self):
        """
        Check for and remove duplicate entries in both training and testing datasets.
        Handles both exact duplicates and partial duplicates (same features but different targets).
        """
        st.subheader("Duplicate Check and Removal")
        st.write("Eliminating duplicates ensures that datasets remain clean, efficient, and representative, which leads to more reliable analyses and effective machine learning models.")
        st.write("We are re-evaluating for duplicates at this stage because changes made during data preprocessing might have introduced new duplicate entries.")
        st.write("We will examine both exact duplicates and partial duplicates, where feature values are identical but target values differ.")
            
        if (self.builder.training_data is not None and 
            self.builder.testing_data is not None and 
            self.builder.target_column):
            
            #Check training and testing data for missing values
            missing_values_train = st.session_state.builder.training_data.isnull().sum()
            missing_values_test = st.session_state.builder.testing_data.isnull().sum()
            
            if (missing_values_train > 0).any():
                st.session_state.builder.training_data = st.session_state.builder.training_data.dropna()
                st.warning(f"Training data rows with missing values removed: {missing_values_train.sum()}")

                #log missing values removed
                st.session_state.logger.log_calculation("Missing Values Removed", {
                    "missing_values_train": missing_values_train.to_dict()
                })
            if (missing_values_test > 0).any():
                st.session_state.builder.testing_data = st.session_state.builder.testing_data.dropna()
                st.warning(f"Testing data rows with missing values removed: {missing_values_test.sum()}")

                #log missing values removed
                st.session_state.logger.log_calculation("Missing Values Removed", {
                    "missing_values_test": missing_values_test.to_dict()
                })
            
            # Process training data
            st.write("### Training Data Duplicate Check")
            training_data = self.builder.training_data
            initial_train_row_count = len(training_data)
            
            # Check for exact duplicates in training data
            exact_duplicates_train = training_data.duplicated().sum()
            if exact_duplicates_train > 0:
                st.warning(f"Found {exact_duplicates_train} exact duplicate rows in training data. The first occurrence will be kept.")
                # Remove exact duplicates
                training_data = training_data.drop_duplicates(keep='first')
                rows_after_exact_train = len(training_data)
                self.builder.training_data = training_data
                self.logger.log_calculation(
                    "Training Data Duplicate Removal",
                    {
                        "exact_duplicates_removed": int(exact_duplicates_train),
                        "initial_rows": initial_train_row_count,
                        "rows_after_exact": rows_after_exact_train
                    }
                )
            else:
                st.success("No exact duplicates found in the training dataset")
                rows_after_exact_train = len(training_data)
                
            # Check for partial duplicates in training data (same values in all columns except target)
            non_target_cols = [col for col in training_data.columns if col != self.builder.target_column]
            partial_duplicates_train = training_data.duplicated(subset=non_target_cols).sum()
            if partial_duplicates_train > 0:
                st.warning(f"Found {partial_duplicates_train} partial duplicate rows in training data (same feature values but different target values).")
                # Remove partial duplicates
                training_data = training_data.drop_duplicates(subset=non_target_cols)
                final_train_row_count = len(training_data)
                self.builder.training_data = training_data
                self.logger.log_calculation(
                    "Training Data Duplicate Removal",
                    {
                        "partial_duplicates_removed": int(partial_duplicates_train),
                        "rows_after_exact": rows_after_exact_train,
                        "final_rows": final_train_row_count
                    }
                )
            else:
                st.success("No partial duplicates found in the training dataset")
                final_train_row_count = len(training_data)

            # Show total reduction summary for training data if any duplicates were removed
            if exact_duplicates_train > 0 or partial_duplicates_train > 0:
                total_reduction_train = initial_train_row_count - final_train_row_count
                reduction_percentage_train = (total_reduction_train / initial_train_row_count) * 100
               
                st.write("### Summary of Training Data Duplicate Removal")
                st.write(f"- Initial training dataset size: {initial_train_row_count:,} rows")
                st.write(f"- Final training dataset size: {final_train_row_count:,} rows")
                st.write(f"- Total rows removed: {total_reduction_train:,} ({reduction_percentage_train:.2f}%)")
                self.logger.log_calculation(
                    "Training Data Duplicate Removal Summary",
                    {
                        "initial_rows": initial_train_row_count,
                        "final_rows": final_train_row_count,
                        "total_reduction": total_reduction_train,
                        "reduction_percentage": reduction_percentage_train
                    }
                )
            
            # Process testing data
            st.write("### Testing Data Duplicate Check")
            testing_data = self.builder.testing_data
            initial_test_row_count = len(testing_data)
            
            # Check for exact duplicates in testing data
            exact_duplicates_test = testing_data.duplicated().sum()
            if exact_duplicates_test > 0:
                st.warning(f"Found {exact_duplicates_test} exact duplicate rows in testing data. The first occurrence will be kept.")
                # Remove exact duplicates
                testing_data = testing_data.drop_duplicates(keep='first')
                rows_after_exact_test = len(testing_data)
                self.builder.testing_data = testing_data
                self.logger.log_calculation(
                    "Testing Data Duplicate Removal",
                    {
                        "exact_duplicates_removed": int(exact_duplicates_test),
                        "initial_rows": initial_test_row_count,
                        "rows_after_exact": rows_after_exact_test
                    }
                )
            else:
                st.success("No exact duplicates found in the testing dataset")
                rows_after_exact_test = len(testing_data)
                
            # Check for partial duplicates in testing data (same values in all columns except target)
            non_target_cols = [col for col in testing_data.columns if col != self.builder.target_column]
            partial_duplicates_test = testing_data.duplicated(subset=non_target_cols).sum()
            if partial_duplicates_test > 0:
                st.warning(f"Found {partial_duplicates_test} partial duplicate rows in testing data (same feature values but different target values).")
                # Remove partial duplicates
                testing_data = testing_data.drop_duplicates(subset=non_target_cols)
                final_test_row_count = len(testing_data)
                self.builder.testing_data = testing_data
                self.logger.log_calculation(
                    "Testing Data Duplicate Removal",
                    {
                        "partial_duplicates_removed": int(partial_duplicates_test),
                        "rows_after_exact": rows_after_exact_test,
                        "final_rows": final_test_row_count
                    }
                )
            else:
                st.success("No partial duplicates found in the testing dataset")
                final_test_row_count = len(testing_data)

            # Show total reduction summary for testing data if any duplicates were removed
            if exact_duplicates_test > 0 or partial_duplicates_test > 0:
                total_reduction_test = initial_test_row_count - final_test_row_count
                reduction_percentage_test = (total_reduction_test / initial_test_row_count) * 100
               
                st.write("### Summary of Testing Data Duplicate Removal")
                st.write(f"- Initial testing dataset size: {initial_test_row_count:,} rows")
                st.write(f"- Final testing dataset size: {final_test_row_count:,} rows")
                st.write(f"- Total rows removed: {total_reduction_test:,} ({reduction_percentage_test:.2f}%)")
                self.logger.log_calculation(
                    "Testing Data Duplicate Removal Summary",
                    {
                        "initial_rows": initial_test_row_count,
                        "final_rows": final_test_row_count,
                        "total_reduction": total_reduction_test,
                        "reduction_percentage": reduction_percentage_test
                    }
                )
                self.logger.log_journey_point(
                    stage="DATA_PREPROCESSING",
                    decision_type="DUPLICATE_CHECK",
                    description="Final Review Duplicate check and removal",
                    details={
                        'Total Rows Removed (Training)': total_reduction_train,
                        'Total Rows Removed (Testing)': total_reduction_test,
                        "Training Data Shape": self.builder.training_data.shape,
                        "Testing Data Shape": self.builder.testing_data.shape
                    },
                    parent_id=None
                )

    def _display_training_data_dashboard(self):
        """
        Display a comprehensive dashboard for the training data, including metrics,
        data preview, statistics, data types, and column information.
        """
        # Create a dashboard-like layout for the final preview
        st.write("---")
        st.write("## Final Dataset Dashboard")
        
        # Key Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Log key metrics about the final dataset
        metrics = {
            "total_rows": int(len(self.builder.training_data)),
            "total_columns": int(len(self.builder.training_data.columns)),
            "memory_usage_mb": float(self.builder.training_data.memory_usage().sum() / 1024 / 1024),
            "missing_values": int(self.builder.training_data.isnull().sum().sum())
        }
        self.logger.log_calculation("Final Training Dataset Metrics", metrics)
        
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
        with col5:
            st.metric(
                "Target Column",
                self.builder.target_column,
            )

        # Data Overview Section
        st.write("### Data Overview")
        tabs = st.tabs(["📊 Preview", "📈 Statistics", "🔍 Data Types", "📋 Column Info"])
        
        # Log data overview information
        self.logger.log_calculation("Data Types Summary", {
            "type_distribution": self.builder.training_data.dtypes.astype(str).value_counts().to_dict()
        })
        
        with tabs[0]:
            # Enhanced preview with styling
            st.dataframe(
                self.builder.training_data.head(10).style.background_gradient(cmap='Blues'),
                width='stretch'
            )
            
        with tabs[1]:
            # Statistical summary
            numeric_stats = self.builder.training_data.describe()
            st.dataframe(
                numeric_stats.style.background_gradient(cmap='Greens'),
                width='stretch'
            )
            
        with tabs[2]:
            # Data types analysis
            dtype_counts = self.builder.training_data.dtypes.astype(str).value_counts()
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.write("**Data Type Distribution:**")
                st.dataframe(
                    pd.DataFrame({
                        'Data Type': dtype_counts.index,
                        'Count': dtype_counts.values
                    })
                )
            
            with col2:
                # Create a pie chart of data types with string values
                fig = px.pie(
                    values=dtype_counts.values,
                    names=dtype_counts.index.astype(str),  # Ensure names are strings
                    title="Distribution of Data Types"
                )
                st.plotly_chart(fig, config={'responsive': True})
            
        with tabs[3]:
            # Detailed column information
            column_info = pd.DataFrame({
                'Column': self.builder.training_data.columns,
                'Type': self.builder.training_data.dtypes.astype(str),
                'Non-Null Count': self.builder.training_data.count(),
                'Null Count': self.builder.training_data.isnull().sum(),
                'Unique Values': [self.builder.training_data[col].nunique() 
                                for col in self.builder.training_data.columns],
                'Memory Usage (KB)': [self.builder.training_data[col].memory_usage() / 1024 
                                    for col in self.builder.training_data.columns]
            })
            st.dataframe(
                column_info.style.background_gradient(subset=['Memory Usage (KB)'], cmap='YlOrRd'),
                width='stretch'
            )

        # Data Quality Metrics
        st.write("### Data Quality Metrics")
        
        # Calculate and log completeness metrics
        completeness = (1 - self.builder.training_data.isnull().sum() / len(self.builder.training_data)) * 100
        self.logger.log_calculation("Data Completeness", {
            "column_completeness": completeness.to_dict()
        })
        
        # Calculate and log unique value distributions
        unique_counts = self.builder.training_data.nunique() / len(self.builder.training_data) * 100
        self.logger.log_calculation("Unique Values Distribution", {
            "column_unique_percentages": unique_counts.to_dict()
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Completeness metrics
            completeness = (1 - self.builder.training_data.isnull().sum() / len(self.builder.training_data)) * 100
            fig = px.bar(
                x=completeness.index,
                y=completeness.values,
                title="Data Completeness by Column (%)",
                labels={'x': 'Column', 'y': 'Completeness (%)'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, config={'responsive': True})
        
        with col2:
            # Unique value distribution
            unique_counts = self.builder.training_data.nunique() / len(self.builder.training_data) * 100
            fig = px.bar(
                x=unique_counts.index,
                y=unique_counts.values,
                title="Unique Values Distribution (%)",
                labels={'x': 'Column', 'y': 'Unique Values (%)'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, config={'responsive': True})
            
    def _display_data_quality_warnings(self):
        """
        Identify and display data quality warnings including:
        - Binary features
        - High cardinality features
        - Highly skewed numerical distributions
        - Near-constant columns
        
        Returns:
            list: List of warning messages that were displayed
        """
        st.write("### Data Quality Warnings")

        # Add explanations in an expander
        with st.expander("ℹ️ Understanding Data Quality Issues"):
            st.write("""
            We check for several types of data quality issues that could affect your model's performance:
            
            **1. Binary Features**
            - These are columns that have exactly two unique values (e.g., Yes/No, 0/1, True/False)
            - While not an issue, it's important to be aware of these as they might need special handling
            
            **2. High Cardinality**
            - This occurs when a categorical column has too many unique values
            - For example, if a column has unique values for more than 50% of the rows
            - High cardinality can make it difficult for models to learn patterns and may require special encoding techniques
            
            **3. High Skewness**
            - Skewness measures how asymmetrical the data distribution is
            - A high skew (>2 or <-2) means the data is not evenly distributed
            - For example, if most values are low with a few very high values, the data is positively skewed
            - Highly skewed data might need transformation (like log transformation) for better model performance
            
            **4. Near-Constant Values**
            - These are columns where almost all rows have the same value
            - For example, if 99% of the values in a column are identical
            - Near-constant columns provide little to no useful information for modeling and might need to be removed
            """)
            
        warnings = []
        data_quality_issues = {
            "high_cardinality": [],
            "high_skewness": [],
            "near_constant": [],
            "binary_features": []
        }
        
        # First identify binary features
        binary_features = [
            col for col in self.builder.training_data.columns 
            if self.builder.training_data[col].nunique() == 2
        ]
        data_quality_issues["binary_features"] = binary_features
        
        # Check for high cardinality in categorical columns
        cat_cols = self.builder.training_data.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if col not in binary_features:
                unique_ratio = self.builder.training_data[col].nunique() / len(self.builder.training_data)
                if unique_ratio > 0.5:
                    warning = f"⚠️ High cardinality in '{col}' ({unique_ratio:.1%} unique values)"
                    warnings.append(warning)
                    data_quality_issues["high_cardinality"].append({
                        "column": col,
                        "unique_ratio": float(unique_ratio)
                    })
        
        # Check for skewed numerical distributions
        num_cols = self.builder.training_data.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col not in binary_features:
                skew = self.builder.training_data[col].skew()
                if abs(skew) > 2:
                    warning = f"⚠️ High skewness in '{col}' (skew={skew:.2f})"
                    warnings.append(warning)
                    data_quality_issues["high_skewness"].append({
                        "column": col,
                        "skewness": float(skew)
                    })
        
        # Check for constant or near-constant columns
        for col in self.builder.training_data.columns:
            if col not in binary_features and not col.endswith('_binned'):
                unique_ratio = self.builder.training_data[col].nunique() / len(self.builder.training_data)
                if unique_ratio < 0.01:
                    warning = f"⚠️ Near-constant values in '{col}' ({unique_ratio:.1%} unique values)"
                    warnings.append(warning)
                    data_quality_issues["near_constant"].append({
                        "column": col,
                        "unique_ratio": float(unique_ratio)
                    })
        
        # Log all data quality issues
        self.logger.log_calculation("Data Quality Analysis", data_quality_issues)
        
        if binary_features:
            st.info(f"📊 Found {len(binary_features)} binary features: {', '.join(binary_features)}")
        
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("✅ No major data quality issues detected")
            
        return warnings
            
    def _display_testing_data_dashboard(self):
        """
        Display a dashboard for the testing data with key metrics and preview.
        """
        # Create a dashboard-like layout for the test dataset
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
                "Testing Rows",
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
            memory_usage = float(self.builder.testing_data.memory_usage().sum() / 1024 / 1024)
            st.metric(
                "Memory Usage",
                f"{memory_usage:.2f} MB",
                delta=None
            )
        with col4:
            missing_values = int(self.builder.testing_data.isnull().sum().sum())
            st.metric(
                "Missing Values",
                f"{missing_values:,}",
                delta=None
            )

        st.dataframe(self.builder.testing_data.style.background_gradient(cmap='Blues'), width='stretch') 