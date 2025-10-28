import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.dataframe_comparison import DataframeComparisonComponent
from utils.data_exploration_component import DataExplorationComponent
from utils.logging.logger import MLLogger

class MissingValuesAnalysis:
    def __init__(self, builder):
        """
        Initialize the Missing Values Analysis component.
        
        Args:
            builder: The Builder instance containing the training and testing data.
        """
        self.builder = builder
        self.logger = MLLogger()

        # Initialize undo functionality with single backup (memory optimized)
        if 'missing_values_ops_applied' not in st.session_state:
            st.session_state.missing_values_ops_applied = []
        
        # Store initial state for undo functionality (single backup for both datasets)
        if 'missing_values_entry_data' not in st.session_state:
            st.session_state.missing_values_entry_data = {
                'training_data': self.builder.training_data.copy(),
                'testing_data': self.builder.testing_data.copy()
            }
        
    def render_missing_values_analysis(self):
        """
        Renders the missing values analysis section.
        """
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
            if st.button("Undo Missing Values", type="primary", width='stretch'):
                if st.session_state.missing_values_ops_applied:
                    # Restore data to entry state
                    entry_data = st.session_state.missing_values_entry_data
                    self.builder.training_data = entry_data['training_data'].copy()
                    self.builder.testing_data = entry_data['testing_data'].copy()
                    
                    # Clear operations tracking
                    ops_count = len(st.session_state.missing_values_ops_applied)
                    st.session_state.missing_values_ops_applied = []
                    
                    # Reset completion flag
                    st.session_state.missing_values_complete = False
                    
                    st.success(f"✅ Undid {ops_count} missing values operation(s). Training and testing data restored to entry state.")
                    st.rerun()
                else:
                    st.info("No missing values operations to undo.")

        st.write("### Missing Values")
        st.write("""
            Missing values can affect model performance. Here you can:
            - Visualize missing value patterns
            - See which columns have missing values
            - Get recommendations for handling missing values
            - Apply different strategies based on data type and amount missing
            
            Common strategies:
            - Drop rows: Use when few rows have missing values
            - Drop column: Use when too many values are missing
            - KNN Imputation: Use when there are strong correlations between features
              (Now supports both numerical and categorical data!)
            - Mean/Median: Use for numerical data
            - Mode: Use for categorical data
        """)
        
        # Check if training data exists
        if self.builder.training_data is None:
            st.error("Training data not available. Please perform train-test split first.")
            return
            
        # Add missing values heatmap
        st.write("**Missing Values Heatmap (Training Data):**")
        
        # Check if there are any missing values
        if self.builder.training_data.isnull().sum().sum() > 0:
            # Create missing values matrix
            df = self.builder.training_data
            missing_matrix = df.isnull()
            
            # Visualize missing values as a heatmap using Plotly instead of matplotlib
            fig = px.imshow(
                df.isnull(),
                color_continuous_scale='viridis',
                labels=dict(color="Missing"),
                title='Missing Values Heatmap'
            )
            fig.update_layout(
                height=600,
                width=900,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, config={'responsive': True})
            
        else:
            st.success("No missing values found in the training data! 🎉")
          
        # Custom missing values analysis for training data
        training_data = self.builder.training_data
        missing_stats = {
            'total_missing': training_data.isnull().sum().sum(),
            'missing_by_column': training_data.isnull().sum().to_dict(),
            'missing_percentage': (training_data.isnull().sum() / len(training_data) * 100).to_dict()
        }
        
        # Log the missing values analysis results
        st.session_state.logger.log_calculation(
            "Missing Values Analysis (Training Data)",
            {
                "total_missing": missing_stats["total_missing"],
                "missing_by_column": missing_stats["missing_by_column"],
                "missing_percentage": missing_stats["missing_percentage"],
                "total_rows": len(training_data),
                "total_columns_with_missing": sum(1 for count in missing_stats["missing_by_column"].values() if count > 0)
            }
        )

        cols_with_missing = [
            col for col, count in missing_stats["missing_by_column"].items() 
            if count > 0
        ]
        
        if not cols_with_missing:
            st.success("No missing values found in the training dataset!")
            # Set the completion flag
            st.session_state.missing_values_complete = True
            return
            
        st.write("**Columns with Missing Values in Training Data:**")
        st.dataframe(pd.DataFrame({
            'Column': cols_with_missing,
            'Missing Count': [missing_stats["missing_by_column"][col] for col in cols_with_missing],
            'Missing Percentage': [
                f"{(missing_stats['missing_by_column'][col] / len(training_data)) * 100:.2f}%"
                for col in cols_with_missing
            ]
        }))
        
        with st.expander("📚 Understanding Missing Value Strategies"):
            st.write("""
                ### Missing Value Handling Strategies
                
                #### 1. Drop Rows (`drop_rows`)
                - **What it does:** Removes entire rows that contain any missing values
                - **Best when:**
                    - Few rows have missing values (<5% of data)
                    - Missing values are randomly distributed
                    - You have enough data to afford losing some rows
                - **Advantages:**
                    - Simple and clean solution
                    - No data imputation bias
                - **Disadvantages:**
                    - Loss of potentially valuable data
                    - Can significantly reduce dataset size if many rows have missing values
                
                #### 2. Drop Column (`drop_column`)
                - **What it does:** Removes entire columns that contain missing values
                - **Best when:**
                    - A column has many missing values (>50%)
                    - The column isn't crucial for prediction
                - **Advantages:**
                    - Eliminates uncertainty from missing data
                    - Reduces dimensionality
                - **Disadvantages:**
                    - Loss of potentially important features
                    - May remove important predictive signals
                
                #### 3. Mean Imputation (`mean`)
                - **What it does:** Replaces missing values with the column's mean value
                - **Best when:**
                    - Data is normally distributed
                    - Missing values are random
                    - Working with numerical data
                - **Advantages:**
                    - Simple and fast
                    - Preserves mean of the variable
                - **Disadvantages:**
                    - Reduces variance
                    - Can distort relationships between variables
                    - Not suitable for skewed distributions
                
                #### 4. Median Imputation (`median`)
                - **What it does:** Replaces missing values with the column's median value
                - **Best when:**
                    - Data is skewed or has outliers
                    - Working with numerical data
                    - Missing values are random
                - **Advantages:**
                    - Robust to outliers
                    - Better than mean for skewed data
                - **Disadvantages:**
                    - Can still distort relationships
                    - May not capture complex patterns
                
                #### 5. Mode Imputation (`mode`)
                - **What it does:** Replaces missing values with the most frequent value
                - **Best when:**
                    - Working with categorical data
                    - Data has clear dominant categories
                - **Advantages:**
                    - Natural choice for categorical data
                    - Preserves category distributions
                - **Disadvantages:**
                    - May oversimplify patterns
                    - Can strengthen majority class bias
                
                #### 6. KNN Imputation (`knn`)
                - **What it does:** Imputes values based on K nearest neighbors
                - **Best when:**
                    - Data has strong correlations between features
                    - Complex relationships exist
                    - Both numerical and categorical data present
                - **Advantages:**
                    - Considers relationships between features
                    - Can capture complex patterns
                    - Works well with both numerical and categorical data
                    - Preserves multivariate relationships in the data
                - **Disadvantages:**
                    - Computationally intensive
                    - Sensitive to K value choice
                    - May not work well with high-dimensional data
                - **Implementation Notes:**
                    - Our implementation automatically handles both numerical and categorical features
                    - Numerical features are used directly without scaling to preserve original values
                    - Categorical features are ordinally encoded to preserve relationships
                    - For test data, rows with missing values will be dropped instead of imputed
                
                ### Best Practices
                1. **Analyse Missing Patterns**
                    - Check if values are Missing Completely at Random (MCAR)
                    - Look for patterns in missingness
                
                2. **Consider Data Type**
                    - Numerical: Mean/Median for normal/skewed data, KNN when relationships matter
                    - Categorical: Mode for simple cases, KNN for preserving feature relationships
                
                3. **Evaluate Impact**
                    - Test different strategies
                    - Validate results
                    - Consider domain knowledge
                
                4. **Document Choices**
                    - Record which strategy was used for each feature
                    - Note the reasoning behind each choice
            """)

        # Custom strategy suggestions for training data
        suggestions = {}
        for column in cols_with_missing:
            missing_count = missing_stats["missing_by_column"][column]
            missing_percentage = (missing_count / len(training_data)) * 100
            is_numeric = pd.api.types.is_numeric_dtype(training_data[column])
            
            if missing_percentage > 50:
                suggestions[column] = {
                    "strategy": "drop_column",
                    "reason": f"High percentage of missing values ({missing_percentage:.1f}%). Consider removing this column."
                }
            elif missing_percentage < 5:
                suggestions[column] = {
                    "strategy": "drop_rows",
                    "reason": f"Low percentage of missing values ({missing_percentage:.1f}%). Safe to remove these rows."
                }
            elif is_numeric:
                # Check for correlations with other numeric columns
                numeric_cols = training_data.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 1:
                    suggestions[column] = {
                        "strategy": "knn",
                        "reason": f"Moderate missing values ({missing_percentage:.1f}%) in numeric column. KNN imputation can preserve relationships between features."
                    }
                else:
                    suggestions[column] = {
                        "strategy": "median",
                        "reason": f"Moderate missing values ({missing_percentage:.1f}%) in numeric column. Median imputation preserves distribution."
                    }
            else:
                # For categorical columns, check if KNN might be appropriate
                categorical_cols = training_data.select_dtypes(include=['object', 'category']).columns
                numeric_cols = training_data.select_dtypes(include=['int64', 'float64']).columns
                # If we have multiple categorical columns or both numeric and categorical data,
                # KNN might be a better choice than mode
                if len(categorical_cols) > 1 or len(numeric_cols) > 0:
                    suggestions[column] = {
                        "strategy": "knn",
                        "reason": f"Moderate missing values ({missing_percentage:.1f}%) in categorical column. KNN imputation can preserve relationships with other features."
                    }
                else:
                    suggestions[column] = {
                        "strategy": "mode",
                        "reason": f"Moderate missing values ({missing_percentage:.1f}%) in categorical column. Mode imputation maintains most common category."
                    }
        
        # Log the suggested strategies
        st.session_state.logger.log_recommendation(
            "Missing Values Strategies (Training Data)",
            {
                "column_suggestions": {
                    col: {
                        "recommended_strategy": suggestions[col]["strategy"],
                        "reason": suggestions[col]["reason"],
                        "missing_count": missing_stats["missing_by_column"][col],
                        "missing_percentage": missing_stats["missing_percentage"][col]
                    }
                    for col in cols_with_missing
                }
            }
        )

        strategy_dict = {}
        for col in cols_with_missing:
            with st.expander(f"Handle missing values in: {col}", expanded=True):
                suggestion = suggestions[col]
                st.info(f"**Recommended Strategy:** {suggestion['strategy']}\n\n"
                       f"**Reason:** {suggestion['reason']}")
                
                # Determine available strategies based on data type
                is_numeric = pd.api.types.is_numeric_dtype(self.builder.training_data[col])
                available_strategies = ["drop_rows", "drop_column", "knn"]  # These are always available
                
                if is_numeric:
                    available_strategies.extend(["mean", "median"])
                else:
                    available_strategies.append("mode")
                
                # Find the index of the suggested strategy in the available strategies
                try:
                    default_index = available_strategies.index(suggestion['strategy'])
                except ValueError:
                    default_index = 0  # Default to first strategy if suggested one isn't available
                
                strategy = st.selectbox(
                    f"Choose strategy for {col}",
                    available_strategies,
                    index=default_index,
                    help="""
                    - drop_rows: Remove rows with missing values
                    - drop_column: Remove the entire column
                    - mean: Fill with column mean (numeric only)
                    - median: Fill with column median (numeric only)
                    - mode: Fill with most common value (categorical only)
                    - knn: Use K-Nearest Neighbors to impute values based on similar data points.
                           Works with both numeric and categorical features.
                    """,
                    key=f"missing_strategy_{col}"
                )
                strategy_dict[col] = strategy
            
            # Log each strategy selection
            st.session_state.logger.log_user_action(
                "Missing Value Strategy Selected (Training Data)",
                {
                    "column": col,
                    "selected_strategy": strategy,
                    "recommended_strategy": suggestions[col]["strategy"],
                    "followed_recommendation": strategy == suggestions[col]["strategy"],
                    "missing_count": missing_stats["missing_by_column"][col]
                }
            )
        
        if st.button("Apply Missing Value Strategies"):
            self._apply_missing_value_strategies(strategy_dict)

    def _apply_missing_value_strategies(self, strategy_dict):
        """
        Apply the selected missing value strategies to both training and testing data.
        
        Args:
            strategy_dict: Dictionary mapping column names to strategies.
        """
        # Track the operation for memory optimization 
        if 'missing_values_ops_applied' not in st.session_state:
            st.session_state.missing_values_ops_applied = []
        
        st.session_state.missing_values_ops_applied.append({
            'type': 'missing_values_handling',
            'strategies': strategy_dict.copy()
        })
        
        # Store data before processing for comparison (temporary for this operation)
        before_data_train = self.builder.training_data.copy()
        before_data_test = self.builder.testing_data.copy()
        
        # Handle missing values in training and testing data
        modified_columns_train = []
        modified_columns_test = []
        
        for column, strategy in strategy_dict.items():
            if strategy == "drop_rows":
                # Drop rows with missing values
                self.builder.training_data = self.builder.training_data.dropna(subset=[column])
                self.builder.testing_data = self.builder.testing_data.dropna(subset=[column])
                modified_columns_train.append(column)
                modified_columns_test.append(column)
            elif strategy == "drop_column":
                # Drop entire column
                self.builder.training_data = self.builder.training_data.drop(columns=[column])
                self.builder.testing_data = self.builder.testing_data.drop(columns=[column])
                modified_columns_train.append(column)
                modified_columns_test.append(column)
            elif strategy == "mean":
                # Calculate mean from training data only
                mean_value = self.builder.training_data[column].mean()
                # Apply to both datasets
                self.builder.training_data[column] = self.builder.training_data[column].fillna(mean_value)
                self.builder.testing_data = self.builder.testing_data.dropna(subset=[column])
                modified_columns_train.append(column)
                modified_columns_test.append(column)
            elif strategy == "median":
                # Calculate median from training data only
                median_value = self.builder.training_data[column].median()
                # Apply to both datasets
                self.builder.training_data[column] = self.builder.training_data[column].fillna(median_value)
                self.builder.testing_data = self.builder.testing_data.dropna(subset=[column])
                modified_columns_train.append(column)
                modified_columns_test.append(column)
            elif strategy == "mode":
                # Calculate mode from training data only
                mode_value = self.builder.training_data[column].mode()[0]
                # Apply to both datasets
                self.builder.training_data[column] = self.builder.training_data[column].fillna(mode_value)
                self.builder.testing_data = self.builder.testing_data.dropna(subset=[column])
                modified_columns_train.append(column)
                modified_columns_test.append(column)
            elif strategy == "knn":
                # For KNN, handle training data with KNN imputation
                self._apply_knn_imputation(column, modified_columns_train, modified_columns_test)
        
        # Log the actions taken
        st.session_state.logger.log_user_action(
            "Missing Values Handled (Training Data)",
            {
                "modified_columns": modified_columns_train,
                "strategies": strategy_dict
            }
        )
        
        # Log test data changes
        st.session_state.logger.log_user_action(
            "Missing Values Handled (Test Data)",
            {
                "modified_columns": modified_columns_test,
                "strategies": strategy_dict,
                "note": "For KNN strategy, rows with missing values were dropped from test data"
            }
        )

        st.session_state.logger.log_journey_point(
            stage="DATA_PREPROCESSING",
            decision_type="MISSING_VALUES_HANDLING",
            description="Missing values handling completed",
            details={'Modified Columns (Training)': modified_columns_train, 
                      'Strategies': strategy_dict,
                      'Training Data Shape': self.builder.training_data.shape,
                      'Testing Data Shape': self.builder.testing_data.shape},
            parent_id=None
        )

        # Set the completion flag
        st.session_state.missing_values_complete = True 

        st.success("✅ Missing values handled successfully in both training and test data!")
        
        # Show immediate impact summary
        train_shape_change = f"{before_data_train.shape[0]} → {self.builder.training_data.shape[0]} rows"
        test_shape_change = f"{before_data_test.shape[0]} → {self.builder.testing_data.shape[0]} rows"
        st.info(f"📊 Training data: {train_shape_change} | Testing data: {test_shape_change}")
        
        # Show comparison analysis automatically
        comparison_component = DataframeComparisonComponent(
            original_df=before_data_train,
            modified_df=self.builder.training_data,
            target_column=self.builder.target_column)
        comparison_component.render()
        
        # Clear temporary data to save memory
        del before_data_train, before_data_test
    
    def _display_impact_analysis(self, before_data_train, before_data_test, strategy_dict):
        """
        Display the impact analysis of applying missing value strategies.
        
        Args:
            before_data_train: Training data before applying strategies.
            before_data_test: Testing data before applying strategies.
            strategy_dict: Dictionary of applied strategies.
        """
        # Create a dashboard-like layout for the impact analysis
        st.write("---")
        st.subheader("Impact Analysis of Missing Value Strategies")
        
        # Training data impact analysis
        st.write("### Training Data Impact")
        
        # Key metrics comparison for training data
        cols_train = st.columns(4)
        with cols_train[0]:
            st.metric(
                "Rows Before", 
                f"{len(before_data_train):,}",
                delta=f"{len(self.builder.training_data) - len(before_data_train):,}",
                delta_color="inverse" 
            )
        with cols_train[1]:
            st.metric(
                "Columns Before", 
                f"{len(before_data_train.columns):,}",
                delta=f"{len(self.builder.training_data.columns) - len(before_data_train.columns):,}",
                delta_color="inverse"
            )
        with cols_train[2]:
            before_missing = before_data_train.isnull().sum().sum()
            after_missing = self.builder.training_data.isnull().sum().sum()
            st.metric(
                "Missing Values Before", 
                f"{before_missing:,}",
                delta=f"{after_missing - before_missing:,}",
                delta_color="inverse"
            )
        with cols_train[3]:
            memory_before = float(before_data_train.memory_usage().sum() / 1024 / 1024)
            memory_after = float(self.builder.training_data.memory_usage().sum() / 1024 / 1024)
            st.metric(
                "Memory Usage Before", 
                f"{memory_before:.2f} MB",
                delta=f"{memory_after - memory_before:.2f} MB",
                delta_color="inverse"
            )
        
        # Show strategies applied
        st.write("**Strategies Applied:**")
        strategy_df = pd.DataFrame({
            'Column': list(strategy_dict.keys()),
            'Strategy': list(strategy_dict.values()),
            'Missing Before': [before_data_train[col].isnull().sum() for col in strategy_dict.keys() if col in before_data_train.columns],
            'Missing After': [self.builder.training_data[col].isnull().sum() if col in self.builder.training_data.columns else 0 for col in strategy_dict.keys()]
        })
        st.dataframe(strategy_df, width='stretch')
        
        # Column distribution changes for non-dropped columns
        for col in strategy_dict:
            if strategy_dict[col] != "drop_column" and col in self.builder.training_data.columns:
                with st.expander(f"Distribution changes for {col} (Training)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Before Imputation:**")
                        if pd.api.types.is_numeric_dtype(before_data_train[col]):
                            # For numeric columns, show histograms with Plotly
                            fig = px.histogram(
                                before_data_train, 
                                x=col,
                                opacity=0.7,
                                title=f"{col} Before Imputation",
                                nbins=20
                            )
                            fig.update_layout(
                                xaxis_title=col,
                                yaxis_title="Count"
                            )
                            st.plotly_chart(fig, config={'responsive': True})
                            
                            # Basic stats
                            st.write(f"Mean: {before_data_train[col].mean():.2f}")
                            st.write(f"Median: {before_data_train[col].median():.2f}")
                            st.write(f"Std Dev: {before_data_train[col].std():.2f}")
                        else:
                            # For categorical columns, show value counts with Plotly
                            value_counts = before_data_train[col].value_counts().reset_index()
                            value_counts.columns = [col, 'Count']
                            fig = px.bar(
                                value_counts.head(10), 
                                x=col, 
                                y='Count',
                            title=f"{col} Value Counts Before Imputation"
                            )
                            st.plotly_chart(fig, config={'responsive': True})
                    
                    with col2:
                        st.write("**After Imputation:**")
                        if pd.api.types.is_numeric_dtype(self.builder.training_data[col]):
                            # For numeric columns, show histograms with Plotly
                            fig = px.histogram(
                                self.builder.training_data, 
                                x=col,
                                opacity=0.7,
                                title=f"{col} After Imputation",
                                nbins=20
                            )
                            fig.update_layout(
                                xaxis_title=col,
                                yaxis_title="Count"
                            )
                            st.plotly_chart(fig, config={'responsive': True})
                            
                            # Basic stats
                            st.write(f"Mean: {self.builder.training_data[col].mean():.2f}")
                            st.write(f"Median: {self.builder.training_data[col].median():.2f}")
                            st.write(f"Std Dev: {self.builder.training_data[col].std():.2f}")
                        else:
                            # For categorical columns, show value counts with Plotly
                            value_counts = self.builder.training_data[col].value_counts().reset_index()
                            value_counts.columns = [col, 'Count']
                            fig = px.bar(
                                value_counts.head(10), 
                                x=col, 
                                y='Count',
                            title=f"{col} Value Counts After Imputation"
                            )
                            st.plotly_chart(fig, config={'responsive': True})
        
        # Testing data impact analysis
        st.write("---")
        st.write("### Testing Data Impact")
        
        # Key metrics comparison for test data
        cols_test = st.columns(4)
        with cols_test[0]:
            st.metric(
                "Rows Before", 
                f"{len(before_data_test):,}",
                delta=f"{len(self.builder.testing_data) - len(before_data_test):,}",
                delta_color="inverse"
            )
        with cols_test[1]:
            st.metric(
                "Columns Before", 
                f"{len(before_data_test.columns):,}",
                delta=f"{len(self.builder.testing_data.columns) - len(before_data_test.columns):,}",
                delta_color="inverse"
            )
        with cols_test[2]:
            before_missing_test = before_data_test.isnull().sum().sum()
            after_missing_test = self.builder.testing_data.isnull().sum().sum()
            st.metric(
                "Missing Values Before", 
                f"{before_missing_test:,}",
                delta=f"{after_missing_test - before_missing_test:,}",
                delta_color="inverse"
            )
        with cols_test[3]:
            memory_before_test = float(before_data_test.memory_usage().sum() / 1024 / 1024)
            memory_after_test = float(self.builder.testing_data.memory_usage().sum() / 1024 / 1024)
            st.metric(
                "Memory Usage Before", 
                f"{memory_before_test:.2f} MB",
                delta=f"{memory_after_test - memory_before_test:.2f} MB",
                delta_color="inverse"
            )
            
        # Show the actual data
        st.write("### Final Training Data Sample")
        st.dataframe(self.builder.training_data.head().style.background_gradient(cmap='Blues'), width='stretch')
        
        st.write("### Final Testing Data Sample")
        st.dataframe(self.builder.testing_data.head().style.background_gradient(cmap='Blues'), width='stretch')
        
        # Log the impact analysis
        st.session_state.logger.log_calculation(
            "Missing Values Impact Analysis",
            {
                "training_data": {
                    "rows_before": int(len(before_data_train)),
                    "rows_after": int(len(self.builder.training_data)),
                    "columns_before": int(len(before_data_train.columns)),
                    "columns_after": int(len(self.builder.training_data.columns)),
                    "missing_values_before": int(before_missing),
                    "missing_values_after": int(after_missing),
                    "memory_before_mb": float(memory_before),
                    "memory_after_mb": float(memory_after)
                },
                "testing_data": {
                    "rows_before": int(len(before_data_test)),
                    "rows_after": int(len(self.builder.testing_data)),
                    "columns_before": int(len(before_data_test.columns)),
                    "columns_after": int(len(self.builder.testing_data.columns)),
                    "missing_values_before": int(before_missing_test),
                    "missing_values_after": int(after_missing_test),
                    "memory_before_mb": float(memory_before_test),
                    "memory_after_mb": float(memory_after_test)
                }
            }
        )

    def _apply_knn_imputation(self, column, modified_columns_train, modified_columns_test):
        """
        Apply KNN imputation to a specific column.
        
        Args:
            column: The column to apply KNN imputation to.
            modified_columns_train: List to track modified columns in training data.
            modified_columns_test: List to track modified columns in test data.
        """
        train_data_for_impute = self.builder.training_data.copy()
        
        # Only proceed if we have enough data for KNN
        if len(train_data_for_impute) <= 5:  # Not enough samples for KNN
            self._apply_fallback_strategy(column, modified_columns_train, modified_columns_test)
            return
            
        # Identify column types
        numeric_cols = train_data_for_impute.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = train_data_for_impute.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Prepare preprocessing for both types of data
        transformers = []
        
        # Only add numeric preprocessing if we have numeric columns
        if numeric_cols:
            # Just pass numeric columns through without scaling to preserve original values
            transformers.append(('num', 'passthrough', numeric_cols))
        
        # Only add categorical preprocessing if we have categorical columns
        if categorical_cols:
            # Use OrdinalEncoder for categorical data to preserve distances
            categorical_transformer = Pipeline(steps=[
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            transformers.append(('cat', categorical_transformer, categorical_cols))
        
        # Create and apply the column transformer
        if not transformers:  # No transformers to apply
            self._apply_fallback_strategy(column, modified_columns_train, modified_columns_test)
            return
            
        try:
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'
            )
            
            # Fit and transform the data
            train_processed = preprocessor.fit_transform(train_data_for_impute)
            
            # Apply KNN imputation
            imputer = KNNImputer(n_neighbors=min(5, len(train_data_for_impute)-1))
            imputed_data = imputer.fit_transform(train_processed)
            
            # If we're imputing the selected column
            if column in numeric_cols:
                # Find the position of the column in the processed data
                col_pos = numeric_cols.index(column)
                # Replace the column with imputed values in training data
                self.builder.training_data[column] = imputed_data[:, col_pos]
                modified_columns_train.append(column)
            elif column in categorical_cols:
                # Find the position of the column
                col_pos = len(numeric_cols) + categorical_cols.index(column)
                
                # Get the encoder for this categorical column
                encoder = preprocessor.transformers_[1][1].named_steps['ordinal']
                
                # Get the imputed values and inverse transform them
                imputed_cat_values = imputed_data[:, col_pos].reshape(-1, 1)
                
                # Round the values to nearest integer (since ordinal encoding produces integers)
                imputed_cat_values = np.round(imputed_cat_values).astype(int)
                
                # Convert back to original categories
                original_categories = encoder.inverse_transform(imputed_cat_values)
                
                # Replace the column
                self.builder.training_data[column] = original_categories.flatten()
                modified_columns_train.append(column)
                
            # For test data, drop rows with missing values
            if column in self.builder.testing_data.columns:
                self.builder.testing_data = self.builder.testing_data.dropna(subset=[column])
                modified_columns_test.append(column)
                
        except Exception as e:
            # Log the exception for debugging
            st.session_state.logger.log_error(
                "KNN Imputation Error",
                {
                    "error": str(e),
                    "column": column,
                    "fallback": "Using mode/median imputation instead"
                }
            )
            self._apply_fallback_strategy(column, modified_columns_train, modified_columns_test)
    
    def _apply_fallback_strategy(self, column, modified_columns_train, modified_columns_test):
        """
        Apply a fallback strategy when KNN imputation fails.
        
        Args:
            column: The column to apply the fallback strategy to.
            modified_columns_train: List to track modified columns in training data.
            modified_columns_test: List to track modified columns in test data.
        """
        if pd.api.types.is_numeric_dtype(self.builder.training_data[column]):
            median_value = self.builder.training_data[column].median()
            self.builder.training_data[column] = self.builder.training_data[column].fillna(median_value)
            self.builder.testing_data[column] = self.builder.testing_data[column].fillna(median_value)
        else:
            mode_value = self.builder.training_data[column].mode()[0]
            self.builder.training_data[column] = self.builder.training_data[column].fillna(mode_value)
            self.builder.testing_data[column] = self.builder.testing_data[column].fillna(mode_value)
        modified_columns_train.append(column)
        modified_columns_test.append(column) 