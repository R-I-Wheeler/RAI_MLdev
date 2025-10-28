import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Optional

from components.data_exploration.feature_analysis import get_visualisation_info, show_feature_analysis
from utils.dataframe_comparison import DataframeComparisonComponent
from utils.data_exploration_component import DataExplorationComponent

class FeatureManagementComponent:
    """
    Component for managing dataset features, including:
    - Data type conversion
    - Column removal
    - Feature exploration and analysis
    """
    
    def __init__(self, builder, logger):
        """
        Initialize the Feature Management component.
        
        Args:
            builder: The ML Builder instance containing the data
            logger: Logger instance for tracking user actions and calculations
        """
        self.builder = builder
        self.logger = logger
        self.data = builder.data
        self.target_column = builder.target_column
        
        # Initialize session state variables for this component
        if 'show_conversion_analysis' not in st.session_state:
            st.session_state.show_conversion_analysis = None

        if 'show_removal_analysis' not in st.session_state:
            st.session_state.show_removal_analysis = None

        # Initialize undo functionality with single backup (memory optimized)
        if "feature_management_ops_applied" not in st.session_state:
            st.session_state.feature_management_ops_applied = []
            
        # Store initial state for undo functionality (single backup)
        if "feature_management_entry_data" not in st.session_state:
            st.session_state.feature_management_entry_data = self.builder.data.copy()
    
    def render(self):
        """Render the feature management interface"""
        st.write("---")
        st.write("Using the data exploration component may cause the page to reload, any changes that you have applied will still be in effect. you can use the undo button to reset the data to it's original state when you first entered the page")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            @st.dialog(title="Data Exploration", width="large")
            def data_explorer_dialog():
                data_explorer = DataExplorationComponent(self.builder, st.session_state.logger, data=st.session_state.builder.data, target_column=st.session_state.builder.target_column)
                data_explorer.render()
            if st.button("Original Data Exploration",on_click=st.rerun):
                data_explorer_dialog()
        with col2:
            st.write("")
        with col3:
            if st.button("Undo Feature Management", type="primary", width='stretch'):
                if st.session_state.feature_management_ops_applied:
                    # Restore data to entry state
                    self.builder.data = st.session_state.feature_management_entry_data.copy()
                    
                    # Clear operations tracking
                    ops_count = len(st.session_state.feature_management_ops_applied)
                    st.session_state.feature_management_ops_applied = []
                    
                    # Clear analysis states
                    st.session_state.show_conversion_analysis = None
                    st.session_state.show_removal_analysis = None
                    
                    st.success(f"✅ Undid {ops_count} feature management operation(s). Data restored to entry state.")
                    st.rerun()
                else:
                    st.info("No feature management operations to undo.")
        
        st.write("### Feature Management")
        st.write("""
            This step helps you manage the features of your dataset.
            You can:
            - Convert columns to the correct data type
            - Remove columns that are not needed
                 
            Use your findings from the exploration step to help you decide which features to convert or remove.
        """)
        
        # Display basic dataset information
        self._show_dataset_overview()

        # Column management
        st.write("**Column Management:**")
        
        column_info = pd.DataFrame({
            'Column': self.data.columns,
            'Type': self.data.dtypes.astype(str),
            'Non-Null Count': self.data.count(),
            'Null Count': self.data.isnull().sum(),
            'Unique Values': [self.data[col].nunique() for col in self.data.columns]
        })
        st.dataframe(column_info, width='stretch')
        
        # Type conversion section
        self._render_type_conversion_section()
        
        # Column removal section
        self._render_column_removal_section()
    
    def _show_dataset_overview(self):
        """Display basic dataset information in a compact form"""
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Rows", len(self.data))
        with col2:
            st.metric("Number of Columns", len(self.data.columns))
        with col3:
            st.metric("Missing Values", self.data.isnull().sum().sum())
        
        # Log dataset overview
        self.logger.log_calculation(
            "Data Overview",
            {
                "total_rows": len(self.data),
                "total_columns": len(self.data.columns),
                "total_missing": self.data.isnull().sum().sum(),
                "memory_usage_mb": self.data.memory_usage().sum() / 1024 / 1024,
                "numeric_columns": len(self.data.select_dtypes(include=[np.number]).columns),
                "categorical_columns": len(self.data.select_dtypes(include=['object', 'category']).columns),
                "data_types": self.data.dtypes.astype(str).to_dict()
            }
        )
        
        self.logger.log_page_state(
            "Data Preprocessing Overview",
            {
                "current_tab": "overview",
                "dataset_shape": self.data.shape,
                "memory_usage": f"{self.data.memory_usage().sum() / 1024 / 1024:.2f} MB"
            }
        )
    
    def _render_type_conversion_section(self):
        """Render the data type conversion interface"""
        st.subheader("Data Type Modification")
        st.write("""
        Review your features' data types and modify them if needed:
        
        - **Numerical to Categorical**: Convert when numbers represent categories rather than quantities for example, Rating scales (1,2,3,4,5)
        - **Categorical to Numerical**: Uses label encoding to convert categorical variables to numerical values
        
        """)
        with st.expander("What is label encoding?"):
            st.write("""
                     ### 1. Label Encoding
                    - **What it does:** Assigns a unique integer to each category (0, 1, 2, ...)
                    - **Best when:**
                        - Categories have a natural order (e.g., Small, Medium, Large)
                        - Working with tree-based models
                        - Dealing with binary categories
                    - **Advantages:**
                        - Simple and memory-efficient
                        - Preserves ordinality if present
                        - No dimensionality increase
                    - **Disadvantages:**
                        - Creates arbitrary numerical relationships
                        - May mislead non-tree models
                        - Doesn't capture category similarities
                     """)
            
        st.write("**Convert Column Types**")
        col1, col2 = st.columns(2)
        with col1:
            # Filter out the target column from the options
            column_options = [col for col in self.data.columns 
                             if col != self.target_column]
            col_to_convert = st.selectbox(
                "Select column to convert",
                column_options,
                help="Select a column to change its data type"
            )
        
        with col2:
            current_type = self.data[col_to_convert].dtype
            if pd.api.types.is_numeric_dtype(current_type):
                new_type = st.selectbox(
                    "Convert to type",
                    ["category", "object", "Keep current"],
                    help="Convert numeric column to categorical"
                )
            else:
                new_type = st.selectbox(
                    "Convert to type",
                    ["int64", "float64", "Keep current"],
                    help="Convert categorical column to numeric"
                )
        
        if new_type != "Keep current":
            if st.button(f"Convert {col_to_convert} to {new_type}"):
                try:
                    # Track the operation for memory optimization
                    st.session_state.feature_management_ops_applied.append({
                        'type': 'conversion',
                        'column': col_to_convert,
                        'from_type': str(self.data[col_to_convert].dtype),
                        'to_type': new_type
                    })
                    # Store the data before conversion for comparison (temporary for this operation)
                    before_data = self.data.copy()
                    old_type = str(self.data[col_to_convert].dtype)
                    
                    if new_type in ["category", "object"]:
                        self.data[col_to_convert] = self.data[col_to_convert].astype(new_type)
                    else:  # numeric conversion
                        # If converting from categorical/object to numeric, use label encoding
                        if old_type in ['category', 'object']:
                            le = LabelEncoder()
                            # Fit and transform the data
                            self.data[col_to_convert] = le.fit_transform(self.data[col_to_convert])
                            
                            # Save encoding mappings to session state
                            if "encoding_mappings" not in st.session_state:
                                st.session_state.encoding_mappings = {}
                            
                            # Create mapping dictionary for this column
                            original_values = le.classes_
                            mapping = {str(orig): int(encoded) for orig, encoded in zip(original_values, range(len(original_values)))}
                            
                            # Update the encoding mappings by adding new mapping while preserving existing ones
                            st.session_state.encoding_mappings[col_to_convert] = {
                                "method": "Label Encoding",
                                "mapping": mapping,
                                "original_values": original_values.tolist()
                            }
                            
                            # Display the encoding mappings
                            st.write("### 🔄 Label Encoding Mappings")
                            with st.expander(f"📊 {col_to_convert} - Label Encoding", expanded=True):
                                mapping_df = pd.DataFrame({
                                    "Original Value": original_values,
                                    "Encoded Value": range(len(original_values))
                                })
                                st.dataframe(
                                    mapping_df.style.background_gradient(cmap='Blues', axis=0),
                                    width='stretch'
                                )
                        else:
                            self.data[col_to_convert] = pd.to_numeric(self.data[col_to_convert])
                        self.data[col_to_convert] = self.data[col_to_convert].astype(new_type)
                    
                    # Update the builder's data
                    self.builder.data = self.data
                    
                    # Enhanced logging for type conversion
                    self.logger.log_calculation(
                        "Data Type Conversion",
                        {
                            "column": col_to_convert,
                            "old_type": old_type,
                            "new_type": new_type,
                            "unique_values_before": before_data[col_to_convert].nunique(),
                            "unique_values_after": self.data[col_to_convert].nunique(),
                            "memory_impact_kb": (
                                self.data[col_to_convert].memory_usage() -
                                before_data[col_to_convert].memory_usage()
                            ) / 1024,
                            "null_values_before": before_data[col_to_convert].isnull().sum(),
                            "null_values_after": self.data[col_to_convert].isnull().sum()
                        }
                    )
                    
                    self.logger.log_user_action(
                        "Column Type Conversion",
                        {
                            "column": col_to_convert,
                            "from_type": old_type,
                            "to_type": new_type,
                            "success": True
                        }
                    )
                    
                    self.logger.log_journey_point(
                        stage="DATA_PREPROCESSING",
                        decision_type="COLUMN_TYPE_CONVERSION",
                        description="Column type conversion completed",
                        details={'Column': col_to_convert, 'From Type': old_type, 'To Type': new_type, 'Success': True},
                        parent_id=None
                    )

                    # Show immediate success message
                    st.success(f"✅ Successfully converted {col_to_convert} from {old_type} to {new_type}")
                    
                    # Show comparison analysis automatically
                    comparison = DataframeComparisonComponent(
                        original_df=before_data,
                        modified_df=self.builder.data,
                        target_column=self.target_column
                    )
                    comparison.render()
                    
                    # Clear the temporary before_data to save memory
                    del before_data

                    #st.rerun()
                except Exception as e:
                    error_msg = f"Error converting column: {str(e)}"
                    st.error(error_msg)
                    self.logger.log_error(
                        "Column Type Conversion Failed",
                        {
                            "column": col_to_convert,
                            "attempted_type": new_type,
                            "error_message": str(e),
                            "current_type": str(self.data[col_to_convert].dtype)
                        }
                    )
        
        # Show conversion impact analysis if available
        #self._show_conversion_analysis()
    
    def _show_conversion_analysis(self):
        """Show the impact analysis of a column type conversion"""
        if st.session_state.show_conversion_analysis:
            col = st.session_state.show_conversion_analysis['col']
            before_data = st.session_state.show_conversion_analysis['before_data']
            new_type = st.session_state.show_conversion_analysis['new_type']
            
            st.write("### Analysis of Feature after Type Conversion")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Before Conversion:**")
                st.write(f"- Type: {before_data[col].dtype}")
                st.write(f"- Unique values: {before_data[col].nunique()}")
                st.write(f"- Memory usage: {before_data[col].memory_usage() / 1024:.2f} KB")
            with col2:
                st.write("**After Conversion:**")
                st.write(f"- Type: {self.data[col].dtype}")
                st.write(f"- Unique values: {self.data[col].nunique()}")
                st.write(f"- Memory usage: {self.data[col].memory_usage() / 1024:.2f} KB")
            
            get_visualisation_info()
            show_feature_analysis(self.data, target_column=self.target_column, selected_feature=col, logger=self.logger)
            
            # Show potential implications
            st.write("**Implications of Conversion:**")
            if new_type in ["category", "object"]:
                st.info("""
                    Converting to categorical type:
                    - Reduces memory usage for columns with few unique values
                    - Enables categorical operations and encoding
                    - May require encoding before model training
                    - Useful for grouping and aggregation operations
                """)
            else:
                st.info("""
                    Converting to numeric type:
                    - Enables mathematical operations and statistical analysis
                    - Required for most machine learning models
                    - May introduce NaN values for non-convertible entries
                    - Consider checking for data loss or unexpected values
                """)
            
            # Add a button to clear the analysis
            if st.button("Clear Impact Analysis"):
                st.session_state.show_conversion_analysis = None
                st.rerun()
    
    def _render_column_removal_section(self):
        """Render the column removal interface"""
        st.subheader("**Remove Columns**")
        st.write("""
            Review your features and remove any that are not needed:
            These may include:
            - Features that are not useful for the analysis, for example, IDs or timestamps
            - Personal information features, for example, names, emails, phone numbers, etc.
            - Features containing information that is not relevant to the target variable
            """)
        
        with st.expander("When deciding which columns to remove, consider:"):
            st.write("""
            1. **Relevance to Target**
            - Does the column help predict your target variable?
            - Is there a logical relationship between this feature and what you're trying to predict?
            
            2. **Data Quality**
            - High percentage of missing values (>50% might be candidates for removal)
            - Columns with constant or near-constant values
            - Columns with too many unique values (like IDs or timestamps)
            
            3. **Redundancy**
            - Highly correlated features that provide similar information
            - Derived or calculated fields where you have the source data
            - Multiple columns representing the same information
            
            4. **Business Context**
            - Features that wouldn't be available in production
            - Sensitive or protected attributes that shouldn't influence predictions
            - Information that would cause data leakage
            
            5. **Technical Considerations**
            - Columns that are difficult to preprocess or encode
            - Text fields that require complex NLP processing
            - Features that would significantly increase model complexity
            """)
            
        # Filter out the target column from the options
        column_options = [col for col in self.data.columns 
                         if col != self.target_column]
        cols_to_remove = st.multiselect(
            "Select columns to remove",
            column_options,
            help="Select any columns you want to remove from the dataset"
        )
        
        if cols_to_remove:
            if st.button("Remove Selected Columns"):
                # Track the operation for memory optimization
                st.session_state.feature_management_ops_applied.append({
                    'type': 'removal',
                    'columns': cols_to_remove.copy()
                })
                # Store data before removal for comparison (temporary for this operation)
                before_data = self.data.copy()
                before_cols = list(self.data.columns)
                before_shape = before_data.shape
                before_memory = before_data.memory_usage(deep=True).sum() / (1024 * 1024)  # Convert to MB
                
                modified_data = self.data.drop(columns=cols_to_remove)
                self.data = modified_data
                
                # Update the builder's data
                self.builder.data = self.data
                
                # Enhanced logging for column removal
                self.logger.log_calculation(
                    "Column Removal",
                    {
                        "columns_removed": cols_to_remove,
                        "rows_affected": len(self.data),
                        "memory_reduction_mb": before_memory - modified_data.memory_usage(deep=True).sum() / (1024 * 1024),
                        "remaining_columns": list(modified_data.columns),
                        "shape_before": before_shape,
                        "shape_after": modified_data.shape
                    }
                )
                
                self.logger.log_user_action(
                    "Remove Columns",
                    {
                        "columns_removed": cols_to_remove,
                        "total_columns_removed": len(cols_to_remove),
                        "remaining_column_count": len(modified_data.columns)
                    }
                )
                
                # Show immediate success message
                st.success(f"✅ Successfully removed {len(cols_to_remove)} columns: {', '.join(cols_to_remove)}")
                
                # Memory impact summary
                memory_saved = before_memory - modified_data.memory_usage(deep=True).sum() / (1024 * 1024)
                st.info(f"💾 Memory saved: {memory_saved:.2f} MB")

                self.logger.log_journey_point(
                    stage="DATA_PREPROCESSING",
                    decision_type="COLUMN_REMOVAL",
                    description="Column removal completed",
                    details={'Columns Removed': cols_to_remove, 'Success': True},
                    parent_id=None
                )
                
                # Show comparison analysis automatically
                comparison = DataframeComparisonComponent(
                    original_df=before_data,
                    modified_df=self.builder.data,
                    target_column=self.target_column
                )
                comparison.render()
                
                # Clear the temporary before_data to save memory
                del before_data
        
        # Show removal impact analysis if available
        #self._show_removal_analysis()
    
    def _show_removal_analysis(self):
        """Show the impact analysis of column removal"""
        if st.session_state.show_removal_analysis:
            before_data = st.session_state.show_removal_analysis['before_data']
            before_shape = st.session_state.show_removal_analysis['before_shape']
            before_memory = st.session_state.show_removal_analysis['before_memory']
            cols_removed = st.session_state.show_removal_analysis['cols_removed']
            modified_data = st.session_state.show_removal_analysis['modified_data']
            
            st.write("### Impact Analysis of Column Removal")
            
            # Basic statistics comparison
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Before Removal:**")
                st.write(f"- Number of columns: {before_shape[1]}")
                st.write(f"- Memory usage: {before_memory:.2f} MB")
                
                # Show correlation heatmap before
                if len(before_data.select_dtypes(include=[np.number]).columns) > 1:
                    before_corr = before_data.select_dtypes(include=[np.number]).corr()
                    fig = px.imshow(
                        before_corr,
                        title="Correlation Matrix Before Removal"
                    )
            st.plotly_chart(fig, config={'responsive': True})
            
            with col2:
                st.write("**After Removal:**")
                st.write(f"- Number of columns: {modified_data.shape[1]}")
                st.write(f"- Memory usage: {modified_data.memory_usage().sum() / 1024 / 1024:.2f} MB")
                
                # Show correlation heatmap after
                if len(modified_data.select_dtypes(include=[np.number]).columns) > 1:
                    after_corr = modified_data.select_dtypes(include=[np.number]).corr()
                    fig = px.imshow(
                        after_corr,
                        title="Correlation Matrix After Removal"
                    )
            st.plotly_chart(scatter_fig, config={'responsive': True})
            
            # Detailed analysis of removed columns
            st.write("**Details of Removed Columns:**")
            for col in cols_removed:
                with st.expander(f"Analysis of removed column: {col}"):
                    st.write(f"- Data type: {before_data[col].dtype}")
                    st.write(f"- Unique values: {before_data[col].nunique()}")
                    st.write(f"- Missing values: {before_data[col].isnull().sum()} ({before_data[col].isnull().mean()*100:.1f}%)")
                    
                    # Visualization of removed column
                    if pd.api.types.is_numeric_dtype(before_data[col].dtype):
                        fig = px.histogram(before_data, x=col, title=f"Distribution of {col}")
                        st.plotly_chart(fig, config={'responsive': True})
                    else:
                        fig = px.bar(before_data[col].value_counts(), title=f"Value Counts of {col}")
                        st.plotly_chart(fig, config={'responsive': True})
            
            # Show implications
            st.write("**Implications of Column Removal:**")
            st.info(f"""
                Impact of removing {len(cols_removed)} column(s):
                - Reduced dataset dimensionality by {len(cols_removed)} features
                - Memory usage reduced by {(before_memory - modified_data.memory_usage().sum() / 1024 / 1024):.2f} MB
                - Consider if removed columns contained important information for your analysis
                - Check if remaining columns are sufficient for your modeling goals
            """)
            
            # Add a button to clear the analysis
            if st.button("Clear Removal Analysis"):
                st.session_state.show_removal_analysis = None
                st.rerun() 