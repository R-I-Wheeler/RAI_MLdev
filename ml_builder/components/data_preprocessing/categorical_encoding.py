import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, Any, List, Tuple, Optional
from utils.dataframe_comparison import DataframeComparisonComponent
from utils.data_exploration_component import DataExplorationComponent


class CategoricalEncodingComponent:
    def __init__(self, builder, logger):
        self.builder = builder
        self.logger = logger
        
        # Initialize undo functionality with single backup (memory optimized)
        if "categorical_encoding_ops_applied" not in st.session_state:
            st.session_state.categorical_encoding_ops_applied = []
            
        # Store initial state for undo functionality (single backup for both datasets)
        if "categorical_encoding_entry_data" not in st.session_state:
            st.session_state.categorical_encoding_entry_data = {
                'training_data': self.builder.training_data.copy(),
                'testing_data': self.builder.testing_data.copy()
            }

    def render(self):
        """Render the categorical encoding component interface."""
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
            if st.button("Undo Categorical Encoding", type="primary", width='stretch'):
                if st.session_state.categorical_encoding_ops_applied:
                    # Restore data to entry state
                    entry_data = st.session_state.categorical_encoding_entry_data
                    self.builder.training_data = entry_data['training_data'].copy()
                    self.builder.testing_data = entry_data['testing_data'].copy()
                    
                    # Clear operations tracking
                    ops_count = len(st.session_state.categorical_encoding_ops_applied)
                    st.session_state.categorical_encoding_ops_applied = []
                    
                    # Clear categorical encoding related session state variables
                    cleanup_keys = [
                        "categorical_encoding_complete", "encoding_result", "before_train_data",
                        "before_test_data", "modified_columns", "encoding_mappings"
                    ]
                    for key in cleanup_keys:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.success(f"✅ Undid {ops_count} categorical encoding operation(s). Training and testing data restored to entry state.")
                    st.rerun()
                else:
                    st.info("No categorical encoding operations to undo.")

        st.write("### Categorical Features")
        st.write("""
            Categorical features need to be converted to numbers. Here you can:
            - Visualize category distributions
            - Get recommendations for encoding methods
            - Apply different encoding strategies
            
            Available encoding methods:
            - Label Encoding: Convert categories to numbers (0,1,2...)
            - One-Hot Encoding: Create binary columns for each category
            - Target Encoding: Convert based on target variable relationship
            
            Choose based on:
            - Number of unique categories
            - Relationship with target variable
            - Whether categories have a natural order
        """)

        with st.expander("📚 Understanding Categorical Encoding Strategies"):
            self._render_encoding_explanation()

        # Check if impact analysis should be shown (after encoding was applied)
        if "categorical_encoding_complete" in st.session_state and st.session_state.categorical_encoding_complete:
            if "encoding_result" in st.session_state and "before_train_data" in st.session_state and "before_test_data" in st.session_state:
                # Show the impact analysis using stored data from session state
                self._show_impact_analysis(
                    st.session_state.modified_columns,
                    st.session_state.before_train_data, 
                    st.session_state.before_test_data, 
                    st.session_state.encoding_result
                )
                return
        
        # Check if training data exists
        if hasattr(self.builder, 'training_data') and self.builder.training_data is not None:
            # Update categorical column detection to include both object and category dtypes
            categorical_cols = self.builder.training_data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                selected_cat_col = st.selectbox("Select categorical column to visualize", categorical_cols)
                fig = px.bar(self.builder.training_data[selected_cat_col].value_counts(), 
                            title=f"Distribution of {selected_cat_col}")
                st.plotly_chart(fig)

                # Get encoding suggestions - based on training data
                suggestions = self.builder.suggest_encoding_strategies(data=self.builder.training_data)
                if suggestions["success"] and suggestions["suggestions"]:
                    self._render_encoding_options(suggestions)
            else:
                st.success("No categorical variables found in the training data!")
                st.session_state.categorical_encoding_complete = True
        else:
            st.warning("Training data not found. Please split your data first before proceeding with categorical encoding.")

    def _render_encoding_explanation(self):
        """Render the detailed explanation of encoding strategies."""
        st.write("""
            ### Categorical Feature Encoding Strategies
            
            Categorical encoding transforms non-numeric categorical data into a format suitable for machine learning models. Here's a detailed look at different encoding strategies:
            
            ### Encoding Methods
            
            #### 1. Label Encoding
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
            
            #### 2. One-Hot Encoding
            - **What it does:** Creates binary columns for each category
            - **Best when:**
                - Categories have no natural order
                - Working with linear models
                - Relatively few unique categories (<10)
            - **Advantages:**
                - No assumed ordering
                - Works well with most models
                - Preserves all category information
            - **Disadvantages:**
                - Increases dimensionality
                - Memory intensive for high cardinality
                - Can create sparse matrices
            
            #### 3. Target Encoding
            - **What it does:** Replaces categories with target variable statistics
            - **Best when:**
                - High cardinality features
                - Categories correlate with target
                - Sufficient data per category
            - **Advantages:**
                - Handles high cardinality well
                - Captures category-target relationship
                - No dimensionality increase
            - **Disadvantages:**
                - Risk of overfitting
                - Requires target variable
                - May leak target information
            
            ### Impact on Data
            
            1. **Dimensionality**
               - Label Encoding: No change
               - One-Hot: Increases by (n-1) columns per feature
               - Target: No change
            
            2. **Information Preservation**
               - Label: May lose category relationships
               - One-Hot: Preserves all information
               - Target: Preserves target relationship
            
            3. **Memory Usage**
               - Label: Most efficient
               - One-Hot: Least efficient
               - Target: Moderately efficient
            
            ### Special Cases
            
            #### 1. High Cardinality
            - **Challenge:** Many unique categories
            - **Solutions:**
                - Group rare categories
                - Use target encoding
                - Consider feature hashing
            
            #### 2. Rare Categories
            - **Challenge:** Categories with few samples
            - **Solutions:**
                - Group into "Other" category
                - Use smoothed target encoding
                - Remove or combine rare categories
            
            #### 3. New Categories
            - **Challenge:** Handling unseen categories in test data
            - **Solutions:**
                - Create "Unknown" category
                - Use fallback encoding
                - Apply smoothing techniques
            
            ### Best Practices
            
            1. **Selection Criteria**
               - Consider model type
               - Evaluate cardinality
               - Check for natural ordering
               - Assess memory constraints
            
            2. **Validation**
               - Cross-validate target encoding
               - Check for information leakage
               - Monitor model performance
            
            3. **Documentation**
               - Record encoding decisions
               - Document category mappings
               - Note handling of special cases
            
            4. **Implementation Tips**
               - Handle missing values first
               - Consider feature interactions
               - Test multiple strategies
               - Use cross-validation when target encoding
        """)

    def _render_encoding_options(self, suggestions):
        """Render encoding options for categorical columns."""
        categorical_columns = list(suggestions["suggestions"].keys())
        
        if categorical_columns:
            st.write("**Categorical Columns Found:**")
            handling_dict = {}
            
            for col in categorical_columns:
                with st.expander(f"Handle categorical variable: {col}", expanded=True):
                    suggestion = suggestions["suggestions"][col]
                    
                    # Check if this is a binned column
                    is_binned = col.endswith('_binned')
                    
                    if is_binned:
                        # Get original column name (remove '_binned' suffix)
                        original_col = col[:-7]  # Remove '_binned'
                        
                        # Check if original column was numeric
                        if original_col in self.builder.original_dtypes and \
                        pd.api.types.is_numeric_dtype(self.builder.original_dtypes[original_col]):
                            st.info("This is a binned numeric feature - it will automatically use label encoding.")
                            handling_dict[col] = {"method": "label"}
                            continue
                        else:
                            st.info("This is a binned categorical feature - please select an encoding method.")
                    
                    st.info(f"**Recommended Strategy:** {suggestion['strategy']}\n\n"
                          f"**Reason:** {suggestion['reason']}")
                    
                    encoding_options = ["label", "onehot", "target", "drop_column"]
                    default_index = 0
                    
                    strategy_mapping = {
                        "label": "label",
                        "onehot": "onehot",
                        "target": "target",
                        "drop": "drop_column"
                    }
                    
                    suggested_strategy = strategy_mapping.get(suggestion['strategy'], 'label')
                    if suggested_strategy in encoding_options:
                        default_index = encoding_options.index(suggested_strategy)
                    
                    method = st.selectbox(
                        f"Choose encoding method for {col}",
                        options=encoding_options,
                        index=default_index,
                        help="Select how to handle this categorical variable",
                        key=f"cat_method_{col}"
                    )
                    handling_dict[col] = {"method": method}
            
            if st.button("Apply Categorical Encoding"):
                self.save_encoding_state()
                self._apply_encoding(handling_dict)

    def _apply_encoding(self, handling_dict):
        """Apply the selected encoding methods to categorical columns."""
        # Store copies of original data for comparison
        before_train_data = self.builder.training_data.copy()
        before_test_data = self.builder.testing_data.copy() if hasattr(self.builder, 'testing_data') and self.builder.testing_data is not None else None
        
        # Store in session state for later use
        st.session_state.before_train_data = before_train_data
        st.session_state.before_test_data = before_test_data
        
        # Log the encoding attempt with details
        self.logger.log_user_action(
            "Starting Categorical Encoding",
            {
                "encoding_methods": handling_dict,
                "categorical_columns": list(handling_dict.keys()),
                "train_dataset_shape": self.builder.training_data.shape,
                "test_dataset_shape": self.builder.testing_data.shape if before_test_data is not None else None
            }
        )
        
        # First apply encoding to training data
        result = self.builder.handle_categorical_data(
            handling_dict, 
            data=self.builder.training_data,
            is_training=True
        )
        
        # If successful and test data exists, apply the same transformations to test data
        if result["success"] and before_test_data is not None:
            test_result = self.builder.handle_categorical_data(
                handling_dict,
                data=self.builder.testing_data,
                is_training=False
            )
            
            if not test_result["success"]:
                st.warning(f"Warning when encoding test data: {test_result.get('message', 'Unknown error')}")
                if "traceback" in test_result:
                    with st.expander("Show error details"):
                        st.code(test_result["traceback"], language="python")
        
        if result["success"]:
            # Store result in session state for later use
            st.session_state.encoding_result = result
            
            # Store encoding mappings directly in session state if available
            if result.get("encoding_mappings"):
                # Initialize encoding_mappings if it doesn't exist
                if "encoding_mappings" not in st.session_state:
                    st.session_state.encoding_mappings = {}
                
                # Merge new mappings with existing ones
                for col, mapping in result["encoding_mappings"].items():
                    st.session_state.encoding_mappings[col] = mapping
            
            # Get the list of modified columns
            modified_columns = result.get("modified_columns", [])
            if not modified_columns:  # If modified_columns is empty, use the keys from handling_dict
                modified_columns = list(result.get("encoding_methods", {}).keys())
            
            # Store modified columns in session state
            st.session_state.modified_columns = modified_columns
            
            self.logger.log_journey_point(
                stage="DATA_PREPROCESSING",
                decision_type="CATEGORICAL_ENCODING",
                description="Categorical encoding completed",
                details={'Modified Columns': modified_columns,
                         'Encoding Methods': handling_dict,
                         'Training Data Shape': self.builder.training_data.shape,
                         'Testing Data Shape': self.builder.testing_data.shape},
                parent_id=None
            )
            
            self._show_encoding_success(result, before_train_data, before_test_data)
        else:
            st.error(f"Error encoding categorical variables: {result.get('message', 'Unknown error')}")

    def _show_encoding_success(self, result, before_train_data, before_test_data):
        """Show success message and impact analysis after encoding."""
        # Log successful encoding with detailed metrics
        memory_metrics = {
            "train_before_mb": before_train_data.memory_usage(deep=True).sum() / 1024 / 1024,
            "train_after_mb": self.builder.training_data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        if before_test_data is not None:
            memory_metrics.update({
                "test_before_mb": before_test_data.memory_usage(deep=True).sum() / 1024 / 1024,
                "test_after_mb": self.builder.testing_data.memory_usage(deep=True).sum() / 1024 / 1024
            })
        
        self.logger.log_calculation(
            "Categorical Encoding",
            {
                "modified_columns": result.get("modified_columns", []),
                "dropped_columns": result.get("dropped_columns", []),
                "encoding_methods": result.get("encoding_methods", {}),
                "rows_processed_train": len(self.builder.training_data),
                "rows_processed_test": len(self.builder.testing_data) if before_test_data is not None else 0,
                "memory_impact": memory_metrics
            }
        )
        
        # Set the completion flag
        st.session_state.categorical_encoding_complete = True
        
        # Log any recommendations based on the encoding results
        if result.get("bin_ranges"):
            self.logger.log_recommendation(
                "Consider reviewing bin ranges for encoded categorical variables",
                {"bin_ranges": result.get("bin_ranges")}
            )
        
        # Log the state transition
        self.logger.log_stage_transition(
            "categorical_preprocessing_start",
            "categorical_preprocessing_complete"
        )
        
        # Log the page state
        self.logger.log_page_state(
            "Data_Preprocessing",
            {
                "step": "categorical",
                "status": "complete",
                "encoded_columns": result.get("modified_columns", []),
                "remaining_categorical": list(set(self.builder.training_data.select_dtypes(include=['object', 'category']).columns) - set(result.get("modified_columns", [])))
            }
        )
        
        st.success("Categorical variables encoded successfully for both training and test data!")
        
        # Use modified_columns from session state
        self._show_impact_analysis(st.session_state.modified_columns, before_train_data, before_test_data, result)

    def _show_impact_analysis(self, modified_columns, before_train_data, before_test_data, result):
        """Show detailed impact analysis after encoding."""
        # Display encoding mappings for label and target encoding
        from components.data_exploration.feature_analysis import show_feature_analysis, get_visualisation_info
        
        st.write("### 📊 Impact Analysis of Categorical Encoding")
        st.write("""
            Below is a detailed analysis of how encoding has affected your categorical variables. 
            We'll look at cardinality changes, memory impact, and information preservation for both training and test datasets.
        """)
        
        # Calculate overall impact metrics
        # For one-hot encoding, we need to compare the original column with all new encoded columns
        valid_columns = []
        encoded_cols_dict = {}
        for col in modified_columns:
            if col in before_train_data.columns:
                if col in self.builder.training_data.columns:
                    # Direct column match (for label/target encoding)
                    valid_columns.append(col)
                    encoded_cols_dict[col] = [col]
                else:
                    # Check for one-hot encoded columns
                    encoded_cols = [c for c in self.builder.training_data.columns if c.startswith(f"{col}_")]
                    if encoded_cols:
                        valid_columns.extend(encoded_cols)
                        encoded_cols_dict[col] = encoded_cols
        
        if encoded_cols_dict:
            # Calculate memory usage for training data - original columns
            train_memory_before = sum(
                before_train_data[col].memory_usage(deep=True) 
                for col in modified_columns 
                if col in before_train_data.columns
            ) / 1024 / 1024  # MB
            
            # Calculate memory usage for training data - encoded columns
            train_memory_after = sum(
                self.builder.training_data[col].memory_usage(deep=True)
                for col in valid_columns
            ) / 1024 / 1024  # MB
            
            train_memory_change = train_memory_after - train_memory_before
            
            #with col1:
            st.write("### 📈 Training Dataset Impact")
            
            # Training data metrics
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric(
                    "Features Encoded",
                    len(modified_columns),
                    help="Number of categorical features that were encoded"
                )
            with metrics_cols[1]:
                st.metric(
                    "Memory Usage",
                    f"{train_memory_after:.2f} MB",
                    f"{train_memory_change:+.2f} MB",
                    help="Total memory usage after encoding and the change in memory usage"
                )
            with metrics_cols[2]:
                # Calculate dimensionality change
                dim_before = len(before_train_data.columns)
                dim_after = len(self.builder.training_data.columns)
                dim_change = dim_after - dim_before
                st.metric(
                    "Dimensionality",
                    f"{dim_after}",
                    f"{dim_change:+d}",
                    help="Number of columns after encoding and the change in dimensionality"
                )
            
            self._show_encoding_mappings(result)
            # Create a DataframeComparisonComponent instance
            comparison_component = DataframeComparisonComponent(
                original_df=before_train_data,
                modified_df=self.builder.training_data,
                target_column=self.builder.target_column)
            comparison_component.render()

    def _show_encoding_mappings(self, result):
        """Show encoding mappings for label and target encoding."""
        if result.get("encoding_mappings"):
            # Save encoding mappings to session state for use elsewhere
            if "encoding_mappings" not in st.session_state:
                st.session_state.encoding_mappings = {}
            
            # Update session state with the latest mappings while preserving existing ones
            for col, mapping in result["encoding_mappings"].items():
                st.session_state.encoding_mappings[col] = mapping
            
            st.write("### 🔄 Encoding Mappings")
            st.write("Below are the mappings used for label and target encoding:")
            
            for column, mapping_info in result["encoding_mappings"].items():
                with st.expander(f"📊 {column} - {mapping_info['method']}", expanded=True):
                    if mapping_info["method"] in ["Label Encoding", "Target Encoding"]:
                        try:
                            # Create a DataFrame to display the mappings
                            if mapping_info["method"] == "Label Encoding":
                                # For Label Encoding, create a clean display of original→encoded
                                original_values = mapping_info["original_values"]
                                
                                # Get the encoded values directly from the mapping
                                encoded_values = []
                                for val in original_values:
                                    encoded_values.append(mapping_info["mapping"].get(str(val), "N/A"))
                                
                                mapping_df = pd.DataFrame({
                                    "Original Value": original_values,
                                    "Encoded Value": encoded_values
                                })
                            else:
                                # For Target Encoding, use the existing approach
                                mapping_df = pd.DataFrame({
                                    "Original Value": mapping_info["original_values"],
                                    "Encoded Value": [mapping_info["mapping"].get(str(val), "N/A") 
                                                    for val in mapping_info["original_values"]]
                                })
                            
                            # Display the mapping table with styling
                            st.dataframe(
                                mapping_df.style.background_gradient(cmap='Blues', axis=0),
                                width='stretch'
                            )
                        except Exception as e:
                            st.error(f"Error displaying mapping: {str(e)}")
                            st.json(mapping_info)
                    elif mapping_info["method"] == "One-Hot Encoding":
                        st.write("**Original Values:**", ", ".join(map(str, mapping_info["original_values"])))
                        st.write("**New Columns Created:**", ", ".join(mapping_info["new_columns"]))

    @staticmethod
    def get_encoding_mappings():
        """
        Get the encoding mappings from session state.
        Can be called by other components to access encoding mappings.
        
        Returns:
            dict: Dictionary containing encoding mappings or empty dict if none exist
        """
        # Only return mappings if encoding is complete, otherwise return empty dict
        if "encoding_mappings" in st.session_state and "categorical_encoding_complete" in st.session_state:
            return st.session_state.encoding_mappings
        return {}

    def save_encoding_state(self):
        """
        Save the current state to the undo stack.
        This should be called before applying new encodings.
        """
        # Store current state before applying changes
        current_state = {
            'training_data': self.builder.training_data.copy(),
            'testing_data': self.builder.testing_data.copy()
        }
        
        # Add to operations applied
        st.session_state.categorical_encoding_ops_applied.append("encoding_operation") 