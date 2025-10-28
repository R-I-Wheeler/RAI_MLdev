import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from components.data_exploration.feature_analysis import get_visualisation_info
from utils.dataframe_comparison import DataframeComparisonComponent
from utils.data_exploration_component import DataExplorationComponent

class OutlierDetectionComponent:
    def __init__(self, builder, logger):
        """
        Initialize the Outlier Detection component.
        
        Args:
            builder: The Builder instance containing the training and testing data.
            logger: The Logger instance for tracking actions and calculations.
        """
        self.builder = builder
        self.logger = logger
        
        # Initialize undo functionality with single backup (memory optimized)
        if "outlier_detection_ops_applied" not in st.session_state:
            st.session_state.outlier_detection_ops_applied = []
            
        # Store initial state for undo functionality (single backup for both datasets)
        if "outlier_detection_entry_data" not in st.session_state:
            # Collect current strategy selections at entry
            current_strategies = {}
            for key in st.session_state.keys():
                if key.startswith("outlier_strategy_"):
                    current_strategies[key] = st.session_state[key]
                    
            st.session_state.outlier_detection_entry_data = {
                'training_data': self.builder.training_data.copy(),
                'testing_data': self.builder.testing_data.copy(),
                'strategies': current_strategies
            }
    def render(self):
        """
        Renders the outlier detection and handling section.
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
            if st.button("Undo Outlier Handling", type="primary", width='stretch'):
                if st.session_state.outlier_detection_ops_applied:
                    # Restore data to entry state
                    entry_data = st.session_state.outlier_detection_entry_data
                    self.builder.training_data = entry_data['training_data'].copy()
                    self.builder.testing_data = entry_data['testing_data'].copy()
                    
                    # Restore strategy selections to entry state
                    if 'strategies' in entry_data:
                        for key, value in entry_data['strategies'].items():
                            st.session_state[key] = value
                    
                    # Clear operations tracking
                    ops_count = len(st.session_state.outlier_detection_ops_applied)
                    st.session_state.outlier_detection_ops_applied = []
                    
                    st.success(f"✅ Undid {ops_count} outlier detection operation(s). Training and testing data restored to entry state.")
                    st.rerun()
                else:
                    st.info("No outlier detection operations to undo.")

        st.write("### Outlier Detection and Handling")
        st.write("""
            Outliers are unusual values that might affect your model. Here you can:
            - Identify outliers using statistical methods
            - Visualize outlier patterns
            - Get statistics about outliers in each column
            - Choose how to handle outliers
            
            Available strategies:
            - Remove: Delete rows with outliers (using 1.5*IQR method)
            - Remove Extreme: Delete rows with extreme outliers only (using 3*IQR method for truly aberrant values)
            - Cap: Limit values to a reasonable range
            - Isolation Forest: Use machine learning to detect and remove outliers
            - None: Keep outliers as they are
            
            Consider domain knowledge when deciding how to handle outliers - some might be valid data points!
        """)
        
        with st.expander("📚 Understanding Outlier Handling Strategies"):
            st.write("""
                ### Outlier Detection and Handling Strategies
                
                Outliers are data points that significantly deviate from the overall pattern of your data. Here's a detailed look at how to detect and handle them:
                
                ### Detection Methods
                
                #### 1. Statistical Methods (IQR Method)
                - **How it works:** Uses the Interquartile Range (IQR = Q3 - Q1)
                - **Standard IQR (1.5*IQR):** Identifies mild outliers
                    - Lower outliers: < Q1 - 1.5 * IQR
                    - Upper outliers: > Q3 + 1.5 * IQR
                - **Extended IQR (3*IQR):** Identifies extreme outliers (Tukey's far outliers)
                    - Lower extreme outliers: < Q1 - 3 * IQR
                    - Upper extreme outliers: > Q3 + 3 * IQR
                - **Advantages:**
                    - Simple and widely used
                    - Works well for approximately normal distributions
                    - Robust to extreme values
                    - Extended method catches only truly aberrant values
                - **Limitations:**
                    - May not suit all distributions
                    - Assumes data symmetry
                
                #### 2. Isolation Forest Method
                - **How it works:** 
                    - Builds decision trees to isolate data points
                    - Points that need fewer splits to isolate are likely outliers
                    - Uses "contamination" parameter to decide outlier threshold
                - **Typically flags:**
                    - Points that can be easily isolated from the rest of the data
                    - Points in sparse regions of the feature space
                - **Advantages:**
                    - Works well with high-dimensional data
                    - Doesn't make distribution assumptions
                    - Effective for complex patterns
                - **Limitations:**
                    - More computationally intensive
                    - Requires sufficient sample size (30+ samples)
                    - Less interpretable than statistical methods
                
                ### How Recommendations Are Made
                
                The system analyses each column and recommends different strategies based on:
                
                1. **Data Characteristics:**
                   - **Small outlier percentage (<1%):** Remove recommended, as these are likely true anomalies
                   - **Moderate outliers (1-10%):** Isolation Forest or Cap based on distribution shape
                   - **High outlier percentage (10-25%):** Cap recommended to preserve data structure
                   - **Very high outliers (>25%):** No action recommended, as this might be a natural distribution
                
                2. **Column Importance:**
                   - **Target column:** Always recommends "None" to preserve target distribution
                   - **Input features:** Analysed based on skewness and outlier percentage
                   - **ID-like columns:** Identifies and skips columns with high cardinality (many unique values)
                
                3. **Data Distribution:**
                   - **Highly skewed data (skewness > 3):** Isolation Forest recommended if sufficient samples
                   - **Moderately skewed data (skewness > 2):** Distribution-aware recommendation
                   - **Approximately normal data:** Cap recommended to preserve overall distribution
                
                4. **Sample Size:**
                   - **Small samples (<30):** Traditional capping methods preferred for stability
                   - **Larger samples:** Machine learning methods like Isolation Forest considered
                
                The system prioritizes data preservation while balancing the need to handle extreme values that could negatively impact model performance.
                
                ### Handling Strategies
                
                #### 1. Remove Outliers
                - **What it does:** Deletes rows containing outlier values (using 1.5*IQR)
                - **Best when:**
                    - Outliers are clearly errors
                    - You have sufficient data
                    - Values are impossible or invalid
                - **Advantages:**
                    - Clean, straightforward solution
                    - Removes clearly invalid data
                - **Disadvantages:**
                    - Loses potentially valuable information
                    - Reduces dataset size
                    - May introduce bias
                
                #### 2. Remove Extreme Outliers
                - **What it does:** Deletes rows containing only extreme outlier values (using 3*IQR - Tukey's far outliers)
                - **Best when:**
                    - You want to remove only truly aberrant values (like 50,000 sq ft houses when normal range is 1,000-10,000)
                    - Data contains both valid unusual values and clear errors
                    - You want a conservative approach to outlier removal
                - **Advantages:**
                    - More conservative than standard removal
                    - Targets only clearly erroneous values
                    - Preserves more borderline cases
                    - Statistically well-established (Tukey's method)
                - **Disadvantages:**
                    - May miss some problematic outliers
                    - Still reduces dataset size
                    - Requires domain knowledge to validate effectiveness
                
                #### 3. Capping (Winsorization)
                - **What it does:** Limits outliers to a specified threshold
                - **Best when:**
                    - You want to preserve data points
                    - Outliers are extreme but possible
                    - Distribution is skewed
                - **Advantages:**
                    - Preserves data quantity
                    - Reduces impact of extreme values
                    - Maintains data relationships
                - **Disadvantages:**
                    - May distort true relationships
                    - Artificial boundary creation
                    - Loss of genuine extreme values
                
                #### 4. Isolation Forest
                - **What it does:** Uses machine learning to identify outliers and removes those rows
                - **Best when:**
                    - Data has complex patterns
                    - Outliers don't follow simple statistical rules
                    - Multiple variables interact
                - **Advantages:**
                    - More accurate detection in complex data
                    - Doesn't assume specific distribution
                    - Can find subtle anomalies
                - **Disadvantages:**
                    - Reduces dataset size
                    - May remove valid but unusual data points
                    - Requires sufficient data volume
                
                #### 5. Keep Outliers (No Action)
                - **Best when:**
                    - Outliers are valid measurements
                    - They represent rare but important events
                    - You're unsure about their validity
                - **Advantages:**
                    - Preserves all information
                    - No risk of removing valid data
                    - Maintains real patterns
                - **Disadvantages:**
                    - May affect model performance
                    - Can skew statistics
                    - Might require robust modeling techniques
                    
                ### Impact on Analysis
                
                1. **Statistical Measures**
                   - Mean: Highly affected by outliers
                   - Median: More robust to outliers
                   - Standard deviation: Can be inflated
                
                2. **Model Performance**
                   - Can affect model accuracy
                   - May lead to overfitting
                   - Might require robust algorithms
                
                3. **Visualization**
                   - Can distort plots and graphs
                   - May hide patterns in main data
                   - Affects scale of visualisations
                
                ### Best Practices
                
                1. **Investigation**
                   - Always visualize outliers first
                   - Understand the source of outliers
                   - Consider domain knowledge
                
                2. **Documentation**
                   - Record outlier handling decisions
                   - Note thresholds and criteria used
                   - Document rationale for choices
                
                3. **Validation**
                   - Check impact on model performance
                   - Validate with domain experts
                   - Consider multiple approaches
            """)
        
        self._process_outliers()
    
    def _process_outliers(self):
        """
        Processes outliers in the dataset.
        """
        # Get numerical columns
        numeric_cols = self.builder.training_data.select_dtypes(include=[np.number]).columns
        
        # Exclude binary columns and binned features
        binary_cols = [col for col in numeric_cols if self.builder.training_data[col].nunique() == 2]
        binned_cols = [col for col in numeric_cols if col.endswith('_binned')]
        numeric_cols = [col for col in numeric_cols if col not in binary_cols and col not in binned_cols]
        
        if binary_cols:
            st.info(f"The following binary columns will be excluded from outlier analysis: {', '.join(binary_cols)}")
        if binned_cols:
            st.info(f"The following binned columns will be excluded from outlier analysis: {', '.join(binned_cols)}")
        
        # Get outlier handling suggestions
        suggestions = self.builder.suggest_outlier_strategies()
        if suggestions["success"]:
            # Display target analysis if available
            if suggestions.get("target_analysis"):
                target_info = suggestions["target_analysis"]
                st.write("### Target Variable Analysis")
                
                # Display message with appropriate styling based on severity
                if target_info["severity"] == "warning":
                    st.warning(f"⚠️ **Target Variable Alert:** {target_info['message']}")
                elif target_info["severity"] == "info":
                    st.info(f"ℹ️ **Target Variable Info:** {target_info['message']}")
                elif target_info["severity"] == "success":
                    st.success(f"✅ **Target Variable Status:** {target_info['message']}")
                
                # Show detailed target statistics in an expander
                with st.expander(f"📊 Detailed Target Analysis - {target_info['column']}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Target Variable Statistics:**")
                        stats = target_info["stats"]
                        st.write(f"- Standard outliers (1.5*IQR): {stats['outlier_count']} ({stats['outlier_percentage']:.2f}%)")
                        st.write(f"- Extreme outliers (3*IQR): {stats['extreme_outlier_count']} ({stats['extreme_outlier_percentage']:.2f}%)")
                        st.write(f"- Skewness: {stats['skewness']:.2f}")
                        
                        # Provide guidance
                        st.write("**💡 Guidance:**")
                        st.write("- Target outliers often represent valuable rare cases")
                        st.write("- Consider domain knowledge before removing target outliers")
                        st.write("- Extreme target outliers might indicate data quality issues")
                        
                    with col2:
                        # Create a box plot for the target variable
                        fig = px.box(self.builder.training_data, y=target_info['column'],
                                   title=f"Target Variable Distribution - {target_info['column']}")
                        st.plotly_chart(fig)
                
                st.write("---")  # Separator between target analysis and feature recommendations
            
            handling_dict = {}  # Store strategies for all columns
            
            # Add section header for feature recommendations
            st.write("### Feature Outlier Recommendations")
            st.write("The following recommendations are for input features only (target variable excluded):")
            
            # First, calculate outliers for all columns to show summary
            cols_with_outliers = []
            outlier_counts = {}
            
            # Calculate outliers using traditional IQR method for display purposes
            for col in numeric_cols:
                # Skip target column from feature analysis (it's handled separately above)
                if col == self.builder.target_column:
                    continue
                    
                Q1 = self.builder.training_data[col].quantile(0.25)
                Q3 = self.builder.training_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.builder.training_data[
                    (self.builder.training_data[col] < lower_bound) | 
                    (self.builder.training_data[col] > upper_bound)
                ]
                if len(outliers) > 0:
                    cols_with_outliers.append(col)
                    outlier_counts[col] = len(outliers)
                    # Log outlier detection statistics
                    self.logger.log_calculation(
                        "Outlier Detection",
                        {
                            "column": col,
                            "outlier_count": len(outliers),
                            "percentage": (len(outliers) / len(self.builder.training_data) * 100),
                            "bounds": {"lower": lower_bound, "upper": upper_bound},
                            "IQR": IQR,
                            "Q1": Q1,
                            "Q3": Q3
                        }
                    )
            
            if cols_with_outliers:
                st.write("**Columns with Outliers:**")
                outlier_summary = pd.DataFrame({
                    'Column': cols_with_outliers,
                    'Number of Outliers': [outlier_counts[col] for col in cols_with_outliers],
                    'Percentage': [
                        f"{(outlier_counts[col] / len(self.builder.training_data) * 100):.2f}%"
                        for col in cols_with_outliers
                    ]
                })
                st.dataframe(outlier_summary)
                
                # Log overall outlier summary
                self.logger.log_calculation(
                    "Outlier Summary",
                    {
                        "total_columns_with_outliers": len(cols_with_outliers),
                        "columns": cols_with_outliers,
                        "outlier_counts": outlier_counts
                    }
                )
                
                for col in cols_with_outliers:  # Only loop through columns with outliers
                    with st.expander(f"Analyse outliers in {col}", expanded=True):
                        # Calculate statistics
                        Q1 = self.builder.training_data[col].quantile(0.25)
                        Q3 = self.builder.training_data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = self.builder.training_data[
                            (self.builder.training_data[col] < lower_bound) | 
                            (self.builder.training_data[col] > upper_bound)
                        ]
                        
                        # Calculate extreme outliers using 3*IQR
                        extreme_lower_bound = Q1 - 3 * IQR
                        extreme_upper_bound = Q3 + 3 * IQR
                        extreme_outliers = self.builder.training_data[
                            (self.builder.training_data[col] < extreme_lower_bound) | 
                            (self.builder.training_data[col] > extreme_upper_bound)
                        ]
                        
                        # Display suggestion if available
                        if col in suggestions["suggestions"]:
                            suggestion = suggestions["suggestions"][col]
                            st.info(f"**Recommended Strategy:** {suggestion['strategy']}\n\n"
                                   f"**Reason:** {suggestion['reason']}")
                            # Log recommendation
                            self.logger.log_recommendation(
                                f"Outlier handling strategy for {col}",
                                {
                                    "column": col,
                                    "recommended_strategy": suggestion['strategy'],
                                    "reason": suggestion['reason']
                                }
                            )
                        
                        # Display statistics and visualization
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Statistical Summary:**")

                            # Get actual whisker values (min/max within bounds)
                            data_within_bounds = self.builder.training_data[col][
                                (self.builder.training_data[col] >= lower_bound) &
                                (self.builder.training_data[col] <= upper_bound)
                            ]
                            lower_whisker = data_within_bounds.min() if len(data_within_bounds) > 0 else lower_bound
                            upper_whisker = data_within_bounds.max() if len(data_within_bounds) > 0 else upper_bound

                            percentage = (len(outliers) / len(self.builder.training_data)) * 100
                            extreme_percentage = (len(extreme_outliers) / len(self.builder.training_data)) * 100
                            skewness = self.builder.training_data[col].skew()

                            # Create statistics dataframe
                            stats_data = {
                                'Metric': [
                                    'Mean', 'Median', 'Std Dev', 'Min', 'Max',
                                    'Q1 (25%)', 'Q3 (75%)', 'IQR',
                                    'Lower Whisker (Box Plot)', 'Upper Whisker (Box Plot)',
                                    'Outliers (1.5×IQR)', 'Outlier %',
                                    'Extreme Outliers (3×IQR)', 'Extreme Outlier %',
                                    'Skewness'
                                ],
                                'Value': [
                                    f"{self.builder.training_data[col].mean():.2f}",
                                    f"{self.builder.training_data[col].median():.2f}",
                                    f"{self.builder.training_data[col].std():.2f}",
                                    f"{self.builder.training_data[col].min():.2f}",
                                    f"{self.builder.training_data[col].max():.2f}",
                                    f"{Q1:.2f}",
                                    f"{Q3:.2f}",
                                    f"{IQR:.2f}",
                                    f"{lower_whisker:.4f}",
                                    f"{upper_whisker:.4f}",
                                    f"{len(outliers)}",
                                    f"{percentage:.2f}%",
                                    f"{len(extreme_outliers)}",
                                    f"{extreme_percentage:.2f}%",
                                    f"{skewness:.2f}"
                                ]
                            }
                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df, width='stretch', hide_index=True)

                            # Log detailed column statistics
                            self.logger.log_calculation(
                                "Column Statistics",
                                {
                                    "column": col,
                                    "statistics": {
                                        "lower_bound": float(lower_bound),
                                        "upper_bound": float(upper_bound),
                                        "outlier_count": int(len(outliers)),
                                        "outlier_percentage": float(percentage),
                                        "skewness": float(skewness)
                                    }
                                }
                            )

                        with col2:
                            fig = px.box(self.builder.training_data, y=col,
                                       title=f"Box Plot with Outliers - {col}")
                            st.plotly_chart(fig)
                        
                        # Outlier handling strategy selection
                        strategy = st.selectbox(
                            "Choose outlier handling strategy",
                            ["None", "Remove", "Remove Extreme", "Cap", "Isolation Forest"],
                            index=["None", "Remove", "Remove Extreme", "Cap", "Isolation Forest"].index("None"),
                            help="""
                            - None: Keep outliers as they are
                            - Remove: Delete rows with outliers
                            - Remove Extreme: Delete rows with extreme outliers only
                            - Cap: Cap outliers at the bounds
                            - Isolation Forest: Use machine learning to detect and remove outliers
                            """,
                            key=f"outlier_strategy_{col}"
                        )
                        
                        # Log strategy selection
                        if strategy != "None":
                            handling_dict[col] = strategy
                            self.logger.log_user_action(
                                "Strategy Selection",
                                {
                                    "column": col,
                                    "selected_strategy": strategy
                                }
                            )
                            
                            # If this is the first strategy selection after loading the page,
                            # save the current state so we can undo back to it
                            if len(st.session_state.outlier_detection_ops_applied) == 0:
                                # Collect current strategy selections
                                current_strategies = {}
                                for key in st.session_state.keys():
                                    if key.startswith("outlier_strategy_"):
                                        current_strategies[key] = st.session_state[key]
                                
                                # Store the first operation
                                st.session_state.outlier_detection_ops_applied.append("strategy_selection")

            if handling_dict:  # Only show button if there are features to process
                if st.button("Apply All Outlier Handling Strategies"):
                    before_data = self.builder.training_data.copy()
                    
                    # Create a dictionary to store current strategy selections
                    current_strategies = {}
                    for key in st.session_state.keys():
                        if key.startswith("outlier_strategy_"):
                            current_strategies[key] = st.session_state[key]
                    
                    # Store current state before applying changes
                    st.session_state.outlier_detection_ops_applied.append("batch_outlier_processing")
                    
                    # Log start of batch processing
                    self.logger.log_user_action(
                        "Start Batch Outlier Processing",
                        {
                            "columns": list(handling_dict.keys()),
                            "strategies": handling_dict
                        }
                    )
                    #get_visualisation_info()
                    for col, strategy in handling_dict.items():
                        result = self.builder.handle_outliers(col, strategy)
                        if result["success"]:
                            self.logger.log_user_action(
                                "Outlier Handling",
                                {
                                    "column": col,
                                    "method": strategy,
                                    "modified": result.get("modified", False),
                                    "status": "success"
                                }
                            )

                            st.success(f"Outliers in {col} handled successfully!")
                        else:
                            self.logger.log_error(
                                f"Outlier handling failed for {col}",
                                {
                                    "column": col,
                                    "method": strategy,
                                    "error_message": result["message"]
                                }
                            )
                            st.error(f"Error handling outliers in {col}: {result['message']}")
                    
                        #st.session_state.show_changes_visualization(before_data, self.builder.training_data, "outliers", col, directly_modified_columns=[col])
                    
                    self.logger.log_journey_point(
                        stage="DATA_PREPROCESSING",
                        decision_type="OUTLIER_HANDLING",
                        description="Outlier handling completed",
                        details={'Columns Modified': list(handling_dict.keys()),
                                 'Strategies': handling_dict,
                                 'Training Data Shape': self.builder.training_data.shape,
                                 'No. of Rows Removed': len(before_data) - len(self.builder.training_data),
                                },
                        parent_id=None
                    )

                    # Display boxplots for features that had strategies applied
                    st.write("### Features with Applied Strategies")
                    for col in handling_dict.keys():
                        if col in self.builder.training_data.columns:
                            with st.expander(f"📊 {col} - After {handling_dict[col]} Strategy", expanded=False):
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.write("**Statistical Summary:**")
                                    # Calculate current statistics
                                    Q1 = self.builder.training_data[col].quantile(0.25)
                                    Q3 = self.builder.training_data[col].quantile(0.75)
                                    IQR = Q3 - Q1
                                    lower_bound = Q1 - 1.5 * IQR
                                    upper_bound = Q3 + 1.5 * IQR

                                    # Get actual whisker values (min/max within bounds)
                                    data_within_bounds = self.builder.training_data[col][
                                        (self.builder.training_data[col] >= lower_bound) &
                                        (self.builder.training_data[col] <= upper_bound)
                                    ]
                                    lower_whisker = data_within_bounds.min() if len(data_within_bounds) > 0 else lower_bound
                                    upper_whisker = data_within_bounds.max() if len(data_within_bounds) > 0 else upper_bound

                                    # Count current outliers
                                    current_outliers = self.builder.training_data[
                                        (self.builder.training_data[col] < lower_bound) |
                                        (self.builder.training_data[col] > upper_bound)
                                    ]

                                    # Extreme outliers
                                    extreme_lower_bound = Q1 - 3 * IQR
                                    extreme_upper_bound = Q3 + 3 * IQR
                                    extreme_outliers = self.builder.training_data[
                                        (self.builder.training_data[col] < extreme_lower_bound) |
                                        (self.builder.training_data[col] > extreme_upper_bound)
                                    ]

                                    # Create statistics dataframe
                                    stats_data = {
                                        'Metric': [
                                            'Mean', 'Median', 'Std Dev', 'Min', 'Max',
                                            'Q1 (25%)', 'Q3 (75%)', 'IQR',
                                            'Lower Whisker (Box Plot)', 'Upper Whisker (Box Plot)',
                                            'Remaining Outliers', 'Outlier %',
                                            'Extreme Outliers', 'Extreme Outlier %',
                                            'Skewness'
                                        ],
                                        'Value': [
                                            f"{self.builder.training_data[col].mean():.2f}",
                                            f"{self.builder.training_data[col].median():.2f}",
                                            f"{self.builder.training_data[col].std():.2f}",
                                            f"{self.builder.training_data[col].min():.2f}",
                                            f"{self.builder.training_data[col].max():.2f}",
                                            f"{Q1:.2f}",
                                            f"{Q3:.2f}",
                                            f"{IQR:.2f}",
                                            f"{lower_whisker:.4f}",
                                            f"{upper_whisker:.4f}",
                                            f"{len(current_outliers)}",
                                            f"{len(current_outliers)/len(self.builder.training_data)*100:.2f}%",
                                            f"{len(extreme_outliers)}",
                                            f"{len(extreme_outliers)/len(self.builder.training_data)*100:.2f}%",
                                            f"{self.builder.training_data[col].skew():.2f}"
                                        ]
                                    }
                                    stats_df = pd.DataFrame(stats_data)
                                    st.dataframe(stats_df, width='stretch', hide_index=True)

                                with col2:
                                    fig = px.box(self.builder.training_data, y=col,
                                               title=f"Box Plot After {handling_dict[col]} - {col}")
                                    st.plotly_chart(fig, config={'responsive': True})

                    # Create a DataframeComparisonComponent instance
                    comparison_component = DataframeComparisonComponent(
                        original_df=before_data,
                        modified_df=self.builder.training_data,
                        target_column=self.builder.target_column)
                    comparison_component.render()

                    # Log completion of batch processing
                    self.logger.log_user_action(
                        "Complete Batch Outlier Processing",
                        {
                            "processed_columns": list(handling_dict.keys()),
                            "total_columns_processed": len(handling_dict)
                        }
                    )
            else:
                st.info("No outliers detected or no features selected for outlier handling.")
        else:
            st.error(f"Error analyzing outliers: {suggestions.get('message', 'Unknown error')}") 