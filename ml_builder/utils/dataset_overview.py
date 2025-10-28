import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
try:
    from dython.nominal import associations
except ImportError:
    associations = None

class DatasetOverviewComponent:
    """
    Component for displaying a comprehensive dataset overview.
    
    This component provides a reusable interface for visualizing dataset statistics,
    including data types, missing values, numerical summaries, and categorical analysis.
    
    Example usage:
    ```python
    # Initialize the component with a dataframe
    dataset_overview = DatasetOverviewComponent(df, logger)
    
    # Display the overview
    dataset_overview.display_overview()
    ```
    
    The component can also be used with a custom title:
    ```python
    dataset_overview.display_overview(title="My Custom Dataset Overview")
    ```
    """
    
    def __init__(self, data: pd.DataFrame, logger: Optional[Any] = None, keyidentifier: Optional[str] = None):
        """
        Initialize the Dataset Overview component.
        
        Args:
            data: The pandas DataFrame to analyse
            logger: Optional logger instance for tracking user actions and calculations
        """
        self.data = data
        self.logger = logger
        self.keyidentifier = keyidentifier
    def _get_data_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the dataset.
        
        Returns:
            Dictionary with summary statistics, dtypes, and missing values information
        """
        # Calculate basic statistics for numerical columns
        numerical_cols = self.data.select_dtypes(include=['number']).columns
        summary_stats = self.data[numerical_cols].describe().to_dict()
        
        # Get data types
        dtypes = self.data.dtypes.apply(lambda x: str(x)).to_dict()
        
        # Calculate missing values
        missing_values = self.data.isnull().sum().to_dict()
        
        return {
            "summary": summary_stats,
            "dtypes": dtypes,
            "missing_values": missing_values
        }
        
    def display_overview(self, title: str = "📊 Dataset Overview"):
        """
        Display a comprehensive overview of the dataset.
        
        Args:
            title: Optional custom title for the overview section
        """
        # Log the analysis if logger is provided
        if self.logger:
            self.logger.log_calculation(
                "Dataset Overview",
                {"rows": len(self.data), "columns": len(self.data.columns)}
            )
        
        # Generate data summary
        summary = self._get_data_summary()
        
        # Display title with custom styling
        st.markdown(f"""
        <div class="data-card">
        <h4>{title}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Display basic dataset metrics
        cols = st.columns(3)
        with cols[0]:
            st.metric("Number of Rows", f"{len(self.data):,}", delta_color="off")
        with cols[1]:
            st.metric("Number of Columns", len(self.data.columns), delta_color="off")
        with cols[2]:
            missing_count = self.data.isnull().sum().sum()
            missing_percent = round(missing_count / (len(self.data) * len(self.data.columns)) * 100, 2)
            st.metric("Missing Values", f"{missing_count:,}", delta=f"{missing_percent}%", delta_color="inverse")
        
        # Create tabs for different data views - use pills to maintain active tab state
        tab_names = ["📋 Dataset Preview", "📈 Numerical Analysis",
                     "🔤 Categorical Analysis", "🧩 Data Types", "❓ Missing Values", "🔗 Correlation/Association"]

        # Initialize active tab in session state
        active_tab_key = f"active_tab_{self.keyidentifier}"
        if active_tab_key not in st.session_state:
            st.session_state[active_tab_key] = "📋 Dataset Preview"

        # Create pills for tab selection that persists state
        selected_tab = st.pills(
            "Select view:",
            tab_names,
            key=active_tab_key,
            label_visibility="collapsed"
        )

        # Display content based on selected tab
        if selected_tab == "📋 Dataset Preview":
            self._display_dataset_preview()
        elif selected_tab == "📈 Numerical Analysis":
            self._display_numerical_analysis(summary)
        elif selected_tab == "🔤 Categorical Analysis":
            self._display_categorical_analysis()
        elif selected_tab == "🧩 Data Types":
            self._display_data_types(summary)
        elif selected_tab == "❓ Missing Values":
            self._display_missing_values(summary)
        elif selected_tab == "🔗 Correlation/Association":
            self._display_association_heatmap()
            
    def _display_dataset_preview(self):
        """Display dataset preview with basic information."""
        st.write("### Dataset Preview")
        with st.expander("Show dataset details", expanded=True):
            # Add dataset info like shape, memory usage
            mem_usage = self.data.memory_usage(deep=True).sum()
            mem_display = f"{mem_usage / 1024**2:.2f} MB" if mem_usage > 1024**2 else f"{mem_usage / 1024:.2f} KB"
            
            st.markdown(f"""
            **Shape**: {self.data.shape[0]} rows × {self.data.shape[1]} columns  
            **Memory Usage**: {mem_display}  
            **Column Count**: {len(self.data.columns)}
            """)
        
        # Use a more modern dataframe display with highlighted first row
        st.dataframe(
            self.data,
            width='stretch',
            height=300
        )
        
    def _display_numerical_analysis(self, summary: Dict[str, Any]):
        """Display numerical features analysis."""
        st.write("### Numerical Features Analysis")
        
        # Create a more attractive, interactive numerical summary
        if summary["summary"]:
            summary_df = pd.DataFrame(summary["summary"]).reset_index()
            summary_df = summary_df.rename(columns={'index': 'Feature'})
            
            # Add an expander for explanation
            with st.expander("Understanding the numerical summary"):
                st.markdown("""
                This table provides key statistics for each numerical feature:
                - **count**: Number of non-null values
                - **mean**: Average value 
                - **std**: Standard deviation (measure of dispersion)
                - **min/max**: Minimum and maximum values
                - **25%, 50% (median), 75%**: Percentile values showing data distribution
                """)
            
            # Display the summary with better styling
            st.dataframe(
                summary_df,
                width='stretch',
                height=400,
                hide_index=True
            )
            
            # Log basic statistics
            if self.logger:
                self.logger.log_calculation(
                    "Basic Statistics",
                    {"summary_statistics": summary["summary"]}
                )
        else:
            st.info("No numerical features found in the dataset")
            
    def _display_categorical_analysis(self):
        """Display categorical features analysis."""
        st.write("### Categorical Features Analysis")
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            # Add explanation about categorical features
            with st.expander("Understanding categorical features"):
                st.markdown("""
                **Categorical features** contain values from a fixed set of categories.
                - **Unique Values**: Number of different categories in the feature
                - **Most Common Value**: The category that appears most frequently
                - **Most Common Count**: How many times the most common category appears
                """)
                
            cat_analysis = []
            for col in categorical_cols:
                value_counts = self.data[col].value_counts()
                unique_values = self.data[col].nunique()
                
                try:
                    most_common = value_counts.index[0]
                    # Truncate very long values for display
                    if isinstance(most_common, str) and len(most_common) > 50:
                        most_common = most_common[:47] + "..."
                except:
                    most_common = "N/A"
                    
                cat_analysis.append({
                    "Feature": col,
                    "Unique Values": unique_values,
                    "Most Common Value": most_common,
                    "Most Common Count": value_counts.iloc[0],
                    "Most Common %": f"{value_counts.iloc[0] / len(self.data) * 100:.1f}%"
                })
                
            cat_df = pd.DataFrame(cat_analysis)
            
            # Display with better formatting
            st.dataframe(
                cat_df, 
                width='stretch',
                height=400,
                hide_index=True
            )
            
            # Add interactive top values for selected categorical feature
            if len(categorical_cols) > 0:
                st.write("### Explore Categorical Value Distribution")

                # Initialize session state for this component's selections
                cat_select_key = f"cat_select_{self.keyidentifier}"
                slider_key = f"cat_slider_{self.keyidentifier}"

                if cat_select_key not in st.session_state:
                    st.session_state[cat_select_key] = categorical_cols[0]
                if slider_key not in st.session_state:
                    st.session_state[slider_key] = 5

                selected_cat = st.selectbox(
                    "Select a categorical feature:",
                    categorical_cols,
                    key=cat_select_key
                )

                col1, col2 = st.columns(2)
                with col1:
                    top_n = st.slider(
                        "Show top N values:",
                        3, 20, 5,
                        key=slider_key
                    )

                # Get value counts and calculate percentages
                value_counts = self.data[selected_cat].value_counts().head(top_n)
                percentages = (value_counts / value_counts.sum() * 100).round(1)

                # Create a dataframe with both counts and percentages
                dist_df = pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': percentages.values
                })

                # Display as a horizontal bar chart
                fig = px.bar(
                    dist_df,
                    x='Count',
                    y='Value',
                    text='Percentage',
                    labels={'Count': 'Frequency', 'Value': selected_cat},
                    color='Count',
                    color_continuous_scale='blues',
                    text_auto='.1f%',
                    height=400
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, config={'responsive': True}, key=f"categorical_analysis_{self.keyidentifier}")
            
            # Log categorical analysis
            if self.logger:
                self.logger.log_calculation(
                    "Categorical Features Analysis",
                    {"categorical_features": cat_analysis}
                )
        else:
            st.info("No categorical features found in the dataset")
            
    def _display_data_types(self, summary: Dict[str, Any]):
        """Display data types analysis with visualization."""
        st.write("### Data Types")
        
        # Add explanation about data types
        with st.expander("Understanding Data Types"):
            st.markdown("""
            **Common Data Types in pandas:**
            
            **Numeric Types:**
            *Integer Types (signed):*
            - `int8`: 8-bit integer (-128 to 127)
            - `int16`: 16-bit integer (-32,768 to 32,767)
            - `int32`: 32-bit integer (-2³¹ to 2³¹-1)
            - `int64`: 64-bit integer (-2⁶³ to 2⁶³-1) - Default integer type
            
            *Integer Types (unsigned):*
            - `uint8`: 8-bit unsigned (0 to 255)
            - `uint16`: 16-bit unsigned (0 to 65,535)
            - `uint32`: 32-bit unsigned (0 to 2³²-1)
            - `uint64`: 64-bit unsigned (0 to 2⁶⁴-1)
            
            *Float Types:*
            - `float16`: Half precision (3 decimal places)
            - `float32`: Single precision (6-7 decimal places)
            - `float64`: Double precision (15-17 decimal places) - Default float type
            
            💡 *Choose smaller datatypes to save memory when you know your data's range*
            
            **Text Types:**
            - `object`: Usually represents text/string data
            - `string`: Dedicated string data type (more efficient than object)
            
            **Date/Time Types:**
            - `datetime64`: Date and time values
            - `timedelta64`: Time differences/durations
            
            **Categorical Types:**
            - `category`: Efficient storage for repeated string values
            - `bool`: True/False values
            
            **Other Types:**
            - `complex128`: Complex numbers
            - `Int64`: Nullable integer type (can contain missing values)
            - `Float64`: Nullable float type
            """)
        
        dtypes_series = pd.Series(summary["dtypes"]).astype(str)
        dtypes_df = pd.DataFrame({
            'Column': dtypes_series.index,
            'Type': dtypes_series.values
        })
        
        # Count occurrences of each data type
        type_counts = dtypes_df['Type'].value_counts().reset_index()
        type_counts.columns = ['Data Type', 'Count']
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Display data types table
            st.dataframe(dtypes_df, width='stretch', hide_index=True)
        
        with col2:
            # Add a pie chart to visualize data type distribution
            fig = px.pie(
                type_counts, 
                values='Count', 
                names='Data Type',
                title='Data Type Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig, config={'responsive': True}, key=f"data_types_{self.keyidentifier}")
        
        # Log data types analysis
        if self.logger:
            self.logger.log_calculation(
                "Data Types",
                {"column_types": dict(zip(dtypes_series.index, dtypes_series.values))}
            )
            
    def _display_missing_values(self, summary: Dict[str, Any]):
        """Display missing values analysis with visualization."""
        st.write("### Missing Values")
        
        missing_df = pd.DataFrame({
            'Column': summary["missing_values"].keys(),
            'Missing Count': summary["missing_values"].values()
        })
        missing_df['Missing Percentage'] = (missing_df['Missing Count'] / len(self.data) * 100).round(2)
        missing_df = missing_df.sort_values('Missing Percentage', ascending=False)
        
        # Filter to show only columns with missing values
        missing_cols = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_cols) > 0:
            st.write(f"**Found {len(missing_cols)} columns with missing values**")
            
            # Display missing values table
            st.dataframe(
                missing_cols,
                width='stretch',
                hide_index=True
            )
            
            # Add a horizontal bar chart of missing percentages
            if len(missing_cols) > 0:
                fig = px.bar(
                    missing_cols.head(15),  # Show top 15 missing columns
                    x='Missing Percentage',
                    y='Column',
                    orientation='h',
                    title='Columns with Highest Percentage of Missing Values',
                    labels={'Missing Percentage': 'Missing (%)', 'Column': ''},
                    text='Missing Percentage',
                    color='Missing Percentage',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig, config={'responsive': True}, key=f"missing_values_{self.keyidentifier}")
        else:
            st.success("✅ No missing values found in the dataset")
        
        # Log missing values analysis
        if self.logger:
            self.logger.log_calculation(
                "Missing Values Analysis",
                {"missing_values_summary": missing_df.to_dict('records')}
            )
            
    def _display_association_heatmap(self):
        """Display association heatmap using Dython library."""
        st.write("### Correlation/Association Heatmap")
        
        # Add explanation about association analysis
        with st.expander("Understanding Correlation/Association Analysis"):
            st.markdown("""
            **Correlation/Association Analysis** shows relationships between all variables in your dataset:
            
            - **For Numerical Variables**: Uses Pearson correlation coefficient
            - **For Categorical Variables**: Uses Cramér's V statistic  
            - **For Mixed Variables**: Uses correlation ratio (eta)
                              
            **Correlation Values:**
            - Range from -1 to 1 for numerical relationships
            - Range from 0 to 1 for categorical relationships
            - Darker colors indicate stronger relationships
            
            **Interpretation Guidelines:**

            **Numerical vs Numerical:**
            - **1**: Perfect positive correlation
            - **-1**: Perfect negative correlation
            - **0**: No linear relationship
            
            **Categorical vs Categorical (Cramer's V):**
            - **0**: No association
            - **1**: Perfect association
            - **>0.3**: Strong association
            
            **Numerical vs Categorical (Correlation Ratio):**
            - **0**: No association
            - **1**: Perfect association
            - **>0.4**: Strong association
            
            **What to Look For:**
            - **Strong associations between features (may indicate redundancy)**
            - **Unexpected relationships that merit investigation**
            - **Groups of related features**
            - **Potential feature selection insights**
                        
            **Special Handling for Classification Problems:**
            - **Multiclass Classification**: Target variables are automatically treated as categorical even if encoded as numbers
            - **Binary Classification**: Target variables are handled appropriately based on their encoding
            - **This ensures proper association measures** (Cramér's V instead of Pearson correlation) for classification targets
                        
            This analysis helps identify:
            - Which features are correlated with each other
            - Potential multicollinearity issues
            - Features that might be redundant
            - Hidden patterns in your data
            """)
        
        if associations is None:
            st.error("""
            **Dython library not installed!**
            
            To use the Association Heatmap feature, please install dython:
            ```bash
            pip install dython
            ```
            
            Dython provides advanced association measures for mixed data types.
            """)
            return
        
        try:
            # Sample data if too large to avoid performance issues
            #sample_size = min(2000, len(self.data))
            #if len(self.data) > sample_size:
            #    st.info(f"📊 Using a sample of {sample_size:,} rows for performance (original: {len(self.data):,} rows)")
            #    sample_data = self.data.sample(n=sample_size, random_state=42)
            #else:
            #    sample_data = self.data.copy()
            sample_data = self.data.copy()
            
            # Remove columns with too many unique values (likely IDs)
            max_categories = 50
            cols_to_analyze = []
            
            for col in sample_data.columns:
                if sample_data[col].dtype in ['object', 'category']:
                    if sample_data[col].nunique() <= max_categories:
                        cols_to_analyze.append(col)
                else:
                    cols_to_analyze.append(col)
            
            if len(cols_to_analyze) < 2:
                st.warning("Need at least 2 suitable columns for association analysis. Columns with too many unique categorical values are excluded.")
                return
            
            # Filter data to analyzed columns
            analysis_data = sample_data[cols_to_analyze].copy()
            
            # Handle missing values
            # if feature is categorical, fill with mode, if numeric, fill with median
            if analysis_data.isnull().sum().sum() > 0:
                st.info("📝 Missing values detected. If categorical, fill with mode, if numeric, fill with median.")
                for col in analysis_data.columns:
                    if analysis_data[col].dtype in ['object', 'category']:
                        analysis_data[col] = analysis_data[col].fillna(analysis_data[col].mode().iloc[0])
                    else:
                        analysis_data[col] = analysis_data[col].fillna(analysis_data[col].median())
            #if analysis_data.isnull().sum().sum() > 0:
            #    st.info("📝 Missing values detected. Using the last valid value to forward fill.")
            #    analysis_data = analysis_data.fillna(method='ffill').fillna(method='bfill')
            
            #st.info(f"🔍 Analyzing associations between {len(cols_to_analyze)} features...")
            
            # Determine which columns should be treated as categorical (nominal)
            # Start with automatically detected categorical columns
            categorical_columns = list(analysis_data.select_dtypes(include=['object', 'category']).columns)
            
            # Check if we have problem type information from session state
            if hasattr(st.session_state, 'problem_type') and hasattr(st.session_state, 'builder'):
                problem_type = getattr(st.session_state, 'problem_type', None)
                target_column = getattr(st.session_state.builder, 'target_column', None)
                is_multiclass = getattr(st.session_state, 'is_multiclass', False)
                
                # If this is multiclass classification and target is in our analysis data,
                # treat the target as categorical even if it's encoded as numeric
                if (problem_type == 'multiclass_classification' or is_multiclass) and target_column:
                    if target_column in analysis_data.columns and target_column not in categorical_columns:
                        categorical_columns.append(target_column)
                        st.info(f"🎯 Treating target '{target_column}' as categorical for multiclass classification analysis")
                        
                        # Log this decision
                        if self.logger:
                            self.logger.log_calculation(
                                "Association Analysis - Target Treatment",
                                {
                                    "problem_type": problem_type,
                                    "target_column": target_column,
                                    "treated_as_categorical": True,
                                    "reason": "multiclass_classification"
                                }
                            )
            
            # Compute associations using dython
            with st.spinner("Computing associations..."):
                try:
                    assoc_result = associations(
                        analysis_data,
                        nominal_columns=categorical_columns if categorical_columns else 'auto',
                        plot=False
                    )
                    
                    # Debug: Show what dython returned
                    #st.write("Debug info:", type(assoc_result))
                    #if isinstance(assoc_result, dict):
                    #    st.write("Dictionary keys:", list(assoc_result.keys()))
                    
                    # Extract the correlation matrix from the result
                    # Dython returns a dictionary, we need to get the matrix
                    if isinstance(assoc_result, dict):
                        # Check for common keys that might contain the correlation matrix
                        if 'associations' in assoc_result:
                            assoc_matrix = assoc_result['associations']
                        elif 'corr' in assoc_result:
                            assoc_matrix = assoc_result['corr']
                        else:
                            # If it's a dict with other structure, try to find the DataFrame
                            for key, value in assoc_result.items():
                                if hasattr(value, 'columns') and hasattr(value, 'index'):
                                    assoc_matrix = value
                                    break
                            else:
                                st.error("Unable to extract correlation matrix from dython result")
                                # Fallback to pandas correlation for numerical data only
                                st.info("Falling back to pandas correlation for numerical features only...")
                                numerical_data = analysis_data.select_dtypes(include=[np.number])
                                if len(numerical_data.columns) >= 2:
                                    assoc_matrix = numerical_data.corr()
                                else:
                                    st.warning("Not enough numerical features for correlation analysis")
                                    return
                    else:
                        assoc_matrix = assoc_result
                    
                except Exception as e:
                    st.warning(f"Dython associations failed: {str(e)}")
                    st.info("Falling back to pandas correlation for numerical features only...")
                    numerical_data = analysis_data.select_dtypes(include=[np.number])
                    if len(numerical_data.columns) >= 2:
                        assoc_matrix = numerical_data.corr()
                    else:
                        st.warning("Not enough numerical features for correlation analysis")
                        return
            
            # Ensure we have a valid DataFrame
            if not hasattr(assoc_matrix, 'columns'):
                st.error("Invalid correlation matrix format returned by dython")
                return
            
            # Create heatmap using plotly
            fig = go.Figure(data=go.Heatmap(
                z=assoc_matrix.values,
                x=assoc_matrix.columns,
                y=assoc_matrix.index,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Association Strength"),
                text=np.round(assoc_matrix.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Feature Association Heatmap",
                xaxis_title="Features",
                yaxis_title="Features",
                height=max(700, len(cols_to_analyze) * 30),
                width=max(700, len(cols_to_analyze) * 30)
            )
            
            # Rotate x-axis labels for better readability
            fig.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig, config={'responsive': True}, key=f"association_heatmap_{self.keyidentifier}")
            
            
        except Exception as e:
            st.error(f"Error computing associations: {str(e)}")
            st.info("This might be due to data type incompatibilities or insufficient data variation.") 