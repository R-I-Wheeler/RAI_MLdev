import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components.data_exploration.feature_analysis import get_visualisation_info
from utils.dataframe_comparison import DataframeComparisonComponent
from utils.data_exploration_component import DataExplorationComponent
from utils.logging.logger import MLLogger

class ZeroValuesAnalysis:
    def __init__(self, builder):
        self.builder = builder
        self.logger = MLLogger()

        # Initialize undo functionality with single backup (memory optimized)
        if "zero_values_ops_applied" not in st.session_state:
            st.session_state.zero_values_ops_applied = []
            
        # Store initial state for undo functionality (single backup)
        if "zero_values_entry_data" not in st.session_state:
            st.session_state.zero_values_entry_data = self.builder.data.copy()

    def render_zero_values_analysis(self):
        """
        Renders the zero values analysis section.
        """
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
            if st.button("Undo Zero Values", type="primary", width='stretch'):
                if st.session_state.zero_values_ops_applied:
                    # Restore data to entry state
                    self.builder.data = st.session_state.zero_values_entry_data.copy()
                    
                    # Clear operations tracking
                    ops_count = len(st.session_state.zero_values_ops_applied)
                    st.session_state.zero_values_ops_applied = []
                    
                    st.success(f"✅ Undid {ops_count} zero values operation(s). Data restored to entry state.")
                    st.rerun()
                else:
                    st.info("No zero values operations to undo.")

        st.write("### Zero Values Analysis")
        st.write("""
            Some features might contain zero values that could represent:
            - True zero measurements
            - Missing data encoded as zeros
            - Measurement errors
            
            Review each feature with zero values and decide how to handle them:
            - Remove rows with zeros if they're invalid measurements
            - Convert zeros to null values if they represent missing data
            - Keep zeros if they're valid measurements
        """)
        
        # Analyse zero values
        zero_analysis = self.builder.analyse_zero_values()
        if zero_analysis["success"]:
            if zero_analysis["stats"]:
                cols_with_zeros = list(zero_analysis["stats"].keys())
                
                # Exclude target column from zero values recommendations
                if self.builder.target_column and self.builder.target_column in cols_with_zeros:
                    cols_with_zeros.remove(self.builder.target_column)
                    st.info(f"ℹ️ The target column '{self.builder.target_column}' contains zero values but is excluded from modification recommendations.")
                
                if cols_with_zeros:
                    # Log calculation results for zero analysis
                    st.session_state.logger.log_calculation(
                        "Zero Values Analysis",
                        {
                            "columns_with_zeros": {
                                col: {
                                    "count": zero_analysis["stats"][col]['count'],
                                    "percentage": zero_analysis["stats"][col]['percentage']
                                }
                                for col in cols_with_zeros
                            },
                            "total_columns_affected": len(cols_with_zeros)
                        }
                    )

                    st.write("**Columns with Zero Values:**")
                    zero_summary = pd.DataFrame({
                        'Column': cols_with_zeros,
                        'Number of Zeros': [zero_analysis["stats"][col]['count'] for col in cols_with_zeros],
                        'Percentage': [f"{zero_analysis['stats'][col]['percentage']:.2f}%" for col in cols_with_zeros]
                    })
                    st.dataframe(zero_summary)
                    
                    strategy_dict = {}
                    for col in cols_with_zeros:
                        st.write(f"**Handle zeros in {col}:**")
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            strategy = st.radio(
                                f"Choose strategy for {col}",
                                ["keep", "remove", "null"],
                                format_func=lambda x: {
                                    "keep": "Keep zeros (valid measurements)",
                                    "remove": "Remove rows with zeros",
                                    "null": "Convert zeros to missing values"
                                }[x],
                                key=f"zero_strategy_{col}"
                            )
                        with col2:
                            # Show small distribution plot
                            fig = go.Figure()
                            fig.add_trace(go.Box(y=st.session_state.builder.data[col], name=col))
                            fig.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0))
                            st.plotly_chart(fig, config={'responsive': True})
                        
                        if strategy != "keep":
                            strategy_dict[col] = strategy
                            # Log user's strategy selection
                            st.session_state.logger.log_user_action(
                                "Zero Handling Strategy Selected",
                                {
                                    "column": col,
                                    "strategy": strategy,
                                    "zeros_count": zero_analysis["stats"][col]['count'],
                                    "zeros_percentage": zero_analysis["stats"][col]['percentage']
                                }
                            )
                    
                    if strategy_dict:
                        if st.button("Apply Zero Handling Strategies"):
                            # Track the operation for memory optimization
                            st.session_state.zero_values_ops_applied.append({
                                'type': 'zero_handling',
                                'strategies': strategy_dict.copy()
                            })
                            # Store data before processing for comparison (temporary for this operation)
                            before_data = st.session_state.builder.data.copy()
                            result = self.builder.handle_zero_values(strategy_dict)
                            if result["success"]:
                                # Enhanced logging for successful zero handling
                                st.session_state.logger.log_calculation(
                                    "Zero Values Handling",
                                    {
                                        "modified_columns": result.get("modified_columns", []),
                                        "attempted_columns": result.get("attempted_columns", []),
                                        "strategies_applied": strategy_dict,
                                        "rows_before": len(before_data),
                                        "rows_after": len(st.session_state.builder.data),
                                        "rows_affected": len(before_data) - len(st.session_state.builder.data),
                                        "columns_modified": {
                                            col: {
                                                "zeros_before": before_data[col].eq(0).sum(),
                                                "zeros_after": st.session_state.builder.data[col].eq(0).sum()
                                            }
                                            for col in result.get("modified_columns", [])
                                        }
                                    }
                                )
                                
                                st.success("Zero values handled successfully!")
                                
                                self.logger.log_journey_point(
                                    stage="DATA_PREPROCESSING",
                                    decision_type="ZERO_VALUES_HANDLING",
                                    description="Zero values handling completed",
                                    details={'Columns Modified': result.get("modified_columns", []),
                                              'Strategies Applied': strategy_dict,
                                              'Data Shape': self.builder.data.shape,
                                              'No. of Rows Removed': len(before_data) - len(st.session_state.builder.data)},
                                    parent_id=None
                                )

                                # Update visualization info
                                get_visualisation_info()
                                
                                # Add enhanced impact assessment
                                st.write("### 📊 Impact Analysis of Zero Value Strategies")
                                st.write("""
                                    Below is a detailed analysis of how handling zero values has affected your dataset.
                                    This analysis helps you understand the changes in data quality and structure.
                                """)
                                
                                # Calculate impact metrics
                                rows_affected = len(before_data) - len(st.session_state.builder.data)
                                memory_before = before_data.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
                                memory_after = st.session_state.builder.data.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
                                memory_change = memory_after - memory_before
                                
                                # Overall impact summary
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(
                                        "Rows Before", 
                                        f"{len(before_data):,}",
                                        delta=f"{len(st.session_state.builder.data) - len(before_data):,}",
                                        delta_color="inverse" 
                                    )
                                with col2:
                                    attempted = len(result.get("attempted_columns", []))
                                    modified = len(result.get("modified_columns", []))
                                    st.metric(
                                        "Columns Modified", 
                                        f"{modified}/{attempted}",
                                        help=f"Successfully modified {modified} columns out of {attempted} columns with strategies"
                                    )
                                with col3:
                                    total_zeros_before = sum(before_data[col].eq(0).sum() for col in result.get("attempted_columns", []) if col in before_data.columns)
                                    total_zeros_after = sum(st.session_state.builder.data[col].eq(0).sum() for col in result.get("attempted_columns", []) if col in st.session_state.builder.data.columns)
                                    st.metric(
                                        "Total Zeros", 
                                        f"{total_zeros_before:,}",
                                        delta=f"{total_zeros_after - total_zeros_before:,}",
                                        delta_color="inverse"
                                    )
                                with col4:
                                    st.metric(
                                        "Memory Usage", 
                                        f"{memory_before:.2f} MB",
                                        delta=f"{memory_change:.2f} MB",
                                        delta_color="inverse" if memory_change > 0 else "normal"
                                    )
                                
                                # Detailed impact for each column with a strategy
                                st.write("#### Column-Level Impact")
                                
                                # Create detailed impact data for all attempted columns
                                impact_data = []
                                column_results = result.get("column_results", {})
                                
                                for col in result.get("attempted_columns", []):
                                    if col in column_results:
                                        col_result = column_results[col]
                                        strategy = col_result.get("strategy", "")
                                        modified = col_result.get("modified", False)
                                        reason = col_result.get("reason", "")
                                        zeros_before = before_data[col].eq(0).sum() if col in before_data.columns else 0
                                        zeros_after = st.session_state.builder.data[col].eq(0).sum() if col in st.session_state.builder.data.columns else 0
                                        nulls_before = before_data[col].isnull().sum() if col in before_data.columns else 0
                                        nulls_after = st.session_state.builder.data[col].isnull().sum() if col in st.session_state.builder.data.columns else 0
                                        
                                        impact_row = {
                                            "Column": col,
                                            "Strategy": strategy,
                                            "Applied": "✅" if modified else "❌",
                                            "Zeros Before": zeros_before,
                                            "Zeros After": zeros_after,
                                            "Nulls Before": nulls_before,
                                            "Nulls After": nulls_after
                                        }
                                        
                                        # Add specific metrics based on strategy
                                        if strategy == "remove":
                                            impact_row["Rows Removed"] = col_result.get("rows_removed", 0)
                                        elif strategy == "null":
                                            impact_row["Zeros Converted"] = col_result.get("zeros_converted", 0)
                                            
                                        if not modified:
                                            impact_row["Note"] = reason
                                            
                                        impact_data.append(impact_row)
                                
                                if impact_data:
                                    impact_df = pd.DataFrame(impact_data)
                                    st.dataframe(impact_df, width='stretch')
                                    
                                    if len(result.get("modified_columns", [])) < len(result.get("attempted_columns", [])):
                                        st.info("⚠️ Some columns were not modified. This could be because zeros were not found in those columns during processing, despite being detected during analysis. This can happen if other strategies changed the dataset structure first.")
                                
                                # Log detailed impact analysis
                                st.session_state.logger.log_calculation(
                                    "Zero Values Impact Analysis",
                                    {
                                        "overall": {
                                            "rows_before": len(before_data),
                                            "rows_after": len(st.session_state.builder.data),
                                            "rows_affected": rows_affected,
                                            "attempted_columns": len(result.get("attempted_columns", [])),
                                            "modified_columns": len(result.get("modified_columns", [])),
                                            "memory_before_mb": float(memory_before),
                                            "memory_after_mb": float(memory_after),
                                            "memory_change_mb": float(memory_change)
                                        },
                                        "columns_impact": {
                                            col: column_results.get(col, {})
                                            for col in result.get("attempted_columns", [])
                                        }
                                    }
                                )
                                
                                # Show comparison analysis automatically
                                comparison = DataframeComparisonComponent(
                                    original_df=before_data,
                                    modified_df=st.session_state.builder.data,
                                    target_column=st.session_state.builder.target_column
                                )
                                comparison.render()
                                
                                # Clear the temporary before_data to save memory
                                del before_data
                            else:
                                # Log error if zero handling fails
                                st.session_state.logger.log_error(
                                    "Zero Values Handling Failed",
                                    {
                                        "error_message": result["message"],
                                        "attempted_strategies": strategy_dict,
                                        "affected_columns": list(strategy_dict.keys())
                                    }
                                )
                                st.error(result["message"])
                else:
                    # Log when no zero values are found
                    st.session_state.logger.log_calculation(
                        "Zero Values Analysis",
                        {
                            "result": "no_zeros_found",
                            "numerical_columns_checked": len(st.session_state.builder.data.select_dtypes(include=[np.number]).columns)
                        }
                    )
                    st.success("No zero values found in numerical columns!")
            else:
                # Log when no zero values are found
                st.session_state.logger.log_calculation(
                    "Zero Values Analysis",
                    {
                        "result": "no_zeros_found",
                        "numerical_columns_checked": len(st.session_state.builder.data.select_dtypes(include=[np.number]).columns)
                    }
                )
                st.success("No zero values found in numerical columns!")
    
    def _get_visualisation_info(self):
        """Helper method to update visualization info in session state"""
        if hasattr(st.session_state, 'get_visualisation_info'):
            st.session_state.get_visualisation_info()
    
    def _show_changes_visualization(self, before_data, after_data, process_name, columns=None, directly_modified_columns=None):
        """
        Show visualization of data changes if the function is available.
        
        This is a compatibility wrapper around the show_changes_visualization function.
        """
        # Use the function from session state
        if hasattr(st.session_state, 'show_changes_visualization'):
            st.session_state.show_changes_visualization(
                before_data,
                after_data,
                process_name,
                columns,
                directly_modified_columns
            )
        else:
            st.warning("Visualization function not available. Skipping change visualization.") 