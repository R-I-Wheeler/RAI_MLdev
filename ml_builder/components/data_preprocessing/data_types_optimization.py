from datetime import datetime
import streamlit as st
import pandas as pd
import traceback

class DataTypesOptimisationComponent:
    def __init__(self, builder, logger):
        self.builder = builder
        self.logger = logger
        # Initialize any required state
        if 'optimisation_results' not in st.session_state:
            st.session_state.optimisation_results = None
    
    def render(self):
        """Render the data types optimisation component"""
        self._render_data_types_review()
    
    def _render_data_types_review(self):
        """Render the data types review and optimisation section"""
        st.write("### Data Types Review")
        st.write("""
            Review and optimise data types for all columns in the training data to ensure they are appropriate for modeling and memory efficient:
            - Verify numeric columns are using the correct data type (int, float)
            - Check categorical columns are properly encoded
            - Optimise memory usage by downcasting when possible
            - Ensure datetime fields are properly formatted
            
            The table below shows:
            - Current data type of each column
            - Memory usage per column
            - Number of unique values
            - Sample data from each column
            - Suggested optimisation action
        """)
        
        # Initial Synchronization of data types
        self._synchronize_data_types()
        
        # Display current data types and memory usage
        st.write("#### 📊 Column Data Types and Optimisation Suggestions")
        
        # Create a comprehensive DataFrame with all information based on training data
        dtype_df = pd.DataFrame({
            'Column': self.builder.training_data.columns,
            'Current Type': self.builder.training_data.dtypes.astype(str),
            'Memory (KB)': [f"{self.builder.training_data[col].memory_usage() / 1024:.1f}" 
                          for col in self.builder.training_data.columns],
            'Unique Values': [f"{self.builder.training_data[col].nunique():,} ({self.builder.training_data[col].nunique() / len(self.builder.training_data):.1%})"
                            for col in self.builder.training_data.columns],
            'Sample Values': [str(self.builder.training_data[col].head(3).tolist())[:50] + '...' 
                            for col in self.builder.training_data.columns]
        })
        
        # Add suggested optimisations column
        suggestions, optimized_types = self._get_optimisation_suggestions(self.builder.training_data)
        
        dtype_df['Suggested Action'] = suggestions
        
        # Display the DataFrame
        st.dataframe(dtype_df, width='stretch')
        
        # Log the data types analysis
        self.logger.log_calculation(
            "Data Types Analysis",
            {
                "current_types": dict(self.builder.training_data.dtypes.astype(str)),
                "memory_usage_kb": {col: float(mem.replace('KB', '')) for col, mem in dtype_df['Memory (KB)'].items()},
                "unique_value_counts": {col: int(count.split()[0].replace(',', '')) for col, count in dtype_df['Unique Values'].items()},
                "suggested_actions": dict(zip(dtype_df['Column'], dtype_df['Suggested Action']))
            }
        )
        
        # Display memory usage metrics
        self._display_memory_metrics()
        
        # Add a button to apply all optimisations
        self._render_optimisation_button(optimized_types, dtype_df)
        
        # Display optimisation results if they exist
        self._display_optimisation_results()
    
    def _get_optimisation_suggestions(self, data):
        """Generate optimisation suggestions for each column in the data"""
        suggestions = []
        optimized_types = {}
        
        for col in data.columns:
            col_type = data[col].dtype
            col_memory = data[col].memory_usage() / 1024  # KB
            unique_values = set(data[col].dropna().unique())
            
            # Check for binary features (containing only 0 and 1)
            if unique_values.issubset({0, 1}):
                suggestions.append("Convert to int8 (binary)")
                optimized_types[col] = 'int8'
            elif col_type == 'int64':
                if data[col].min() >= -32768 and data[col].max() <= 32767:
                    suggestions.append("Convert to int16")
                    optimized_types[col] = 'int16'
                else:
                    suggestions.append("Keep current")
                    optimized_types[col] = 'int64'
            elif col_type == 'float64':
                if data[col].min() >= -3.4e38 and data[col].max() <= 3.4e38:
                    suggestions.append("Convert to float32")
                    optimized_types[col] = 'float32'
                else:
                    suggestions.append("Keep current")
                    optimized_types[col] = 'float64'
            elif col_type == 'object':
                # Check if object column contains only 0 and 1 as strings
                str_values = set(str(x).strip() for x in data[col].dropna().unique())
                if str_values.issubset({'0', '1'}):
                    suggestions.append("Convert to int8 (binary)")
                    optimized_types[col] = 'int8'
                elif data[col].nunique() / len(data) < 0.5:
                    suggestions.append("Convert to category")
                    optimized_types[col] = 'category'
                else:
                    try:
                        pd.to_datetime(data[col].iloc[0])
                        suggestions.append("Convert to datetime")
                        optimized_types[col] = 'datetime'
                    except:
                        suggestions.append("Keep as object")
                        optimized_types[col] = 'object'
            else:
                suggestions.append("Keep current")
                optimized_types[col] = str(col_type)
        
        return suggestions, optimized_types
    
    def _display_memory_metrics(self):
        """Display memory usage metrics for training and testing data"""
        train_memory = self.builder.training_data.memory_usage().sum() / 1024 / 1024  # Convert to MB
        test_memory = self.builder.testing_data.memory_usage().sum() / 1024 / 1024  # Convert to MB
        total_memory = train_memory + test_memory
        
        st.metric("💾 Total Dataset Memory Usage", f"{total_memory:.2f} MB", 
                help=f"Training data: {train_memory:.2f} MB, Testing data: {test_memory:.2f} MB")
    
    def _render_optimisation_button(self, optimized_types, dtype_df):
        """Render the button to apply optimisations and handle the optimisation process"""
        st.write("#### 🔄 Apply Optimisations")
        st.write("Click the button below to automatically apply all suggested type conversions to both training and testing data:")
        if st.button("🚀 Optimise All Data Types", type="secondary"):
            self._apply_optimisations(optimized_types, dtype_df)
    
    def _apply_optimisations(self, optimized_types, dtype_df):
        """Apply the optimisations to training and testing data"""
        try:
            # Initialize lists to store successful conversions
            successful_conversions = []
            
            # Calculate initial memory usage for comparison
            train_memory = self.builder.training_data.memory_usage().sum() / 1024 / 1024
            test_memory = self.builder.testing_data.memory_usage().sum() / 1024 / 1024
            total_memory = train_memory + test_memory
            
            # Log the start of optimisation
            self.logger.log_user_action(
                "Data Type Optimisation Started",
                {"total_columns": len(optimized_types)}
            )
            
            for col, new_type in optimized_types.items():
                if new_type != str(self.builder.training_data[col].dtype):
                    try:
                        old_type = str(self.builder.training_data[col].dtype)
                        
                        # Apply to training data
                        if new_type == 'datetime':
                            self.builder.training_data[col] = pd.to_datetime(self.builder.training_data[col])
                        elif new_type == 'int8' and self.builder.training_data[col].dtype == 'object':
                            # Convert string '0'/'1' to integers first
                            self.builder.training_data[col] = self.builder.training_data[col].map({'0': 0, '1': 1}).astype('int8')
                        else:
                            self.builder.training_data[col] = self.builder.training_data[col].astype(new_type)
                        
                        # Apply to testing data (if column exists)
                        if col in self.builder.testing_data.columns:
                            # Special handling for '_binned' features that may have been changed to category
                            if '_binned' in col:
                                # Binned features might still be categorical but need to be converted to numeric
                                current_test_dtype = self.builder.testing_data[col].dtype
                                try:
                                    if pd.api.types.is_categorical_dtype(current_test_dtype) and pd.api.types.is_numeric_dtype(new_type):
                                        # Convert categorical binned feature to numeric
                                        # First try to convert category codes to the target numeric type
                                        try:
                                            self.builder.testing_data[col] = pd.to_numeric(
                                                self.builder.testing_data[col].cat.codes, 
                                                errors='coerce'
                                            ).astype(new_type)
                                        except (AttributeError, ValueError):
                                            # If that fails, try converting the category values directly
                                            try:
                                                self.builder.testing_data[col] = pd.to_numeric(
                                                    self.builder.testing_data[col].astype(str), 
                                                    errors='coerce'
                                                ).astype(new_type)
                                            except:
                                                # Last resort: map categorical values to numeric indices
                                                categories = self.builder.testing_data[col].cat.categories
                                                mapping = {cat: i for i, cat in enumerate(categories)}
                                                self.builder.testing_data[col] = (
                                                    self.builder.testing_data[col]
                                                    .map(mapping)
                                                    .astype(new_type)
                                                )
                                    elif new_type == 'datetime':
                                        self.builder.testing_data[col] = pd.to_datetime(self.builder.testing_data[col])
                                    elif new_type == 'int8' and current_test_dtype == 'object':
                                        # Convert string '0'/'1' to integers first
                                        self.builder.testing_data[col] = self.builder.testing_data[col].map({'0': 0, '1': 1}).astype('int8')
                                    else:
                                        # Standard conversion for other binned features
                                        self.builder.testing_data[col] = self.builder.testing_data[col].astype(new_type)
                                except Exception as bin_err:
                                    st.warning(f"Could not convert binned feature {col} from {current_test_dtype} to {new_type}: {str(bin_err)}")
                                    self.logger.log_error(
                                        "Binned Feature Conversion Failed",
                                        {
                                            "column": col, 
                                            "current_test_dtype": str(current_test_dtype),
                                            "target_type": new_type,
                                            "error": str(bin_err)
                                        }
                                    )
                                    # Skip this conversion and continue
                                    continue
                            elif new_type == 'datetime':
                                self.builder.testing_data[col] = pd.to_datetime(self.builder.testing_data[col])
                            elif new_type == 'int8' and self.builder.testing_data[col].dtype == 'object':
                                # Convert string '0'/'1' to integers first
                                self.builder.testing_data[col] = self.builder.testing_data[col].map({'0': 0, '1': 1}).astype('int8')
                            else:
                                self.builder.testing_data[col] = self.builder.testing_data[col].astype(new_type)
                        

                        successful_conversions.append((col, old_type, new_type))
                        
                        # Log each successful conversion
                        self.logger.log_calculation(
                            "Column Type Conversion",
                            {
                                "column": col,
                                "old_type": old_type,
                                "new_type": new_type,
                                "memory_before": float(dtype_df.loc[dtype_df['Column'] == col, 'Memory (KB)'].iloc[0].replace('KB', '')),
                                "memory_after_train": self.builder.training_data[col].memory_usage() / 1024,
                                "memory_after_test": self.builder.testing_data[col].memory_usage() / 1024 if col in self.builder.testing_data.columns else 0
                            }
                        )
                    except Exception as e:
                        st.warning(f"Could not convert {col} to {new_type}: {str(e)}")
                        # Log conversion failure
                        self.logger.log_error(
                            "Column Type Conversion Failed",
                            {
                                "column": col,
                                "intended_type": new_type,
                                "error": str(e)
                            }
                        )
                        continue
            
            # Calculate memory savings
            new_train_memory = self.builder.training_data.memory_usage().sum() / 1024 / 1024
            new_test_memory = self.builder.testing_data.memory_usage().sum() / 1024 / 1024
            new_total_memory = new_train_memory + new_test_memory
            memory_saved = total_memory - new_total_memory
            
            # Store optimisation results in session state
            st.session_state.optimisation_results = {
                'successful_conversions': successful_conversions,
                'old_memory': total_memory,
                'new_memory': new_total_memory,
                'memory_saved': memory_saved,
                'train_memory_before': train_memory,
                'train_memory_after': new_train_memory,
                'test_memory_before': test_memory,
                'test_memory_after': new_test_memory
            }
            
            # Log optimisation completion
            self.logger.log_calculation(
                "Data Type Optimisation Complete",
                {
                    "successful_conversions": len(successful_conversions),
                    "total_columns_attempted": len(optimized_types),
                    "memory_before_mb": total_memory,
                    "memory_after_mb": new_total_memory,
                    "memory_saved_mb": memory_saved,
                    "memory_reduction_percent": (memory_saved / total_memory) * 100,
                    "train_memory_before": train_memory,
                    "train_memory_after": new_train_memory,
                    "test_memory_before": test_memory,
                    "test_memory_after": new_test_memory
                }
            )
            self.logger.log_journey_point(
                stage="DATA_PREPROCESSING",
                decision_type="DATA_TYPES_OPTIMIZATION",
                description="Data types optimisation completed",
                details={
                        'Successful Conversions': len(successful_conversions),
                        'Total Columns Attempted': len(optimized_types),
                        'Memory Saved': memory_saved,
                        'Memory Reduction Percentage': (memory_saved / total_memory) * 100,
                        },
                parent_id=None
            )

            # Force a rerun to update the display
            st.rerun()
            
        except Exception as e:
            error_msg = f"❌ Error applying optimisations: {str(e)}"
            st.error(error_msg)
            # Log the overall optimisation failure
            self.logger.log_error(
                "Data Type Optimisation Failed",
                {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def _display_optimisation_results(self):
        """Display optimisation results if they exist"""
        if hasattr(st.session_state, 'optimisation_results') and st.session_state.optimisation_results:
            results = st.session_state.optimisation_results
            
            # Success message with memory savings
            success_msg = f"""
            ✨ Successfully optimised data types!
            💾 Memory usage reduced from {results['old_memory']:.2f} MB to {results['new_memory']:.2f} MB
            📊 Total memory saved: {results['memory_saved']:.2f} MB ({(results['memory_saved']/results['old_memory']*100):.1f}%)
            
            Training data: {results['train_memory_before']:.2f} MB → {results['train_memory_after']:.2f} MB
            Testing data: {results['test_memory_before']:.2f} MB → {results['test_memory_after']:.2f} MB
            """
            st.success(success_msg)
            
            # Summary of changes
            if results['successful_conversions']:
                st.write("#### 📋 Summary of Changes")
                st.write("The following columns were optimised in both training and testing data:")
                for col, old_type, new_type in results['successful_conversions']:
                    st.write(f"- `{col}`: {old_type} → {new_type}")
            
            # Add a button to clear the optimisation results
            if st.button("Clear Summary"):
                # Log clearing of results
                self.logger.log_user_action(
                    "Clear Optimisation Results",
                    {"optimisation_summary": "cleared"}
                )
                del st.session_state.optimisation_results
                st.rerun() 

    def _synchronize_data_types(self):
        """
        Synchronize data types in testing dataset to match exactly with training dataset.
        This ensures that any optimized/transformed features in the training data
        are matched in the testing data.
        """
        if self.builder.training_data is None or self.builder.testing_data is None:
            return
            
        # Get all features that exist in both datasets
        common_features = [col for col in self.builder.training_data.columns 
                         if col in self.builder.testing_data.columns]
        
        # Track which features needed synchronization
        synchronized_features = []
        
        # For each common feature, make testing data match training data type
        for feature in common_features:
            train_dtype = self.builder.training_data[feature].dtype
            test_dtype = self.builder.testing_data[feature].dtype
            
            if train_dtype != test_dtype:
                # Convert testing data to match training data type
                try:
                    # Special handling for binned features that may still be categorical
                    if '_binned' in feature:
                        if (pd.api.types.is_categorical_dtype(test_dtype) and 
                            pd.api.types.is_numeric_dtype(train_dtype)):
                            # Convert categorical binned feature to numeric
                            # First try to convert the categories to numeric
                            try:
                                # Convert category codes to the target numeric type
                                self.builder.testing_data[feature] = pd.to_numeric(
                                    self.builder.testing_data[feature].cat.codes, 
                                    errors='coerce'
                                ).astype(train_dtype)
                            except (AttributeError, ValueError):
                                # If that fails, try converting the category values directly
                                try:
                                    self.builder.testing_data[feature] = pd.to_numeric(
                                        self.builder.testing_data[feature].astype(str), 
                                        errors='coerce'
                                    ).astype(train_dtype)
                                except:
                                    # Last resort: map categorical values to numeric indices
                                    categories = self.builder.testing_data[feature].cat.categories
                                    mapping = {cat: i for i, cat in enumerate(categories)}
                                    self.builder.testing_data[feature] = (
                                        self.builder.testing_data[feature]
                                        .map(mapping)
                                        .astype(train_dtype)
                                    )
                        else:
                            # Standard conversion for other binned features
                            self.builder.testing_data[feature] = self.builder.testing_data[feature].astype(train_dtype)
                    else:
                        # Standard conversion for non-binned features
                        self.builder.testing_data[feature] = self.builder.testing_data[feature].astype(train_dtype)
                    
                    synchronized_features.append({
                        "feature": feature,
                        "original_type": str(test_dtype),
                        "new_type": str(train_dtype),
                        "is_binned": '_binned' in feature
                    })
                except Exception as e:
                    # If conversion fails, provide more detailed warning for binned features
                    if '_binned' in feature:
                        st.warning(f"⚠️ Could not synchronize binned feature '{feature}' to type {train_dtype}: {str(e)}. "
                                 f"This may cause issues during model training. Original type: {test_dtype}")
                        self.logger.log_error(
                            "Binned Feature Synchronization Failed",
                            {
                                "feature": feature,
                                "train_dtype": str(train_dtype),
                                "test_dtype": str(test_dtype),
                                "error": str(e)
                            }
                        )
                    else:
                        st.warning(f"⚠️ Could not convert '{feature}' to type {train_dtype}: {str(e)}")
                        self.logger.log_error(
                            "Feature Synchronization Failed",
                            {
                                "feature": feature,
                                "train_dtype": str(train_dtype),
                                "test_dtype": str(test_dtype),
                                "error": str(e)
                            }
                        )
        
        # Log the synchronization
        if synchronized_features:
            self.logger.log_calculation(
                "Data Type Synchronization",
                {
                    "synchronized_features": synchronized_features,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Display information about synchronized features
            st.info("✅ Synchronized data types for testing data to match training data:")
            for sync in synchronized_features:
                binned_indicator = " (binned feature)" if sync.get('is_binned', False) else ""
                st.write(f"- '{sync['feature']}': {sync['original_type']} → {sync['new_type']}{binned_indicator}") 