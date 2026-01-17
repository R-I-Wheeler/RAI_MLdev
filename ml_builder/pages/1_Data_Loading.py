import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from Builder import Builder, ModelStage
from app import create_sidebar_navigation
from utils.logging.log_viewer import render_log_viewer
import os
from utils.dataset_overview import DatasetOverviewComponent
import magic  # for mime type checking
from datetime import datetime

def validate_csv_file(file):
    """
    Validate uploaded CSV file for security
    Returns (is_valid, message)
    """
    try:
        # Check file size (limit to 100MB)
        MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB in bytes
        file_size = len(file.getvalue())
        if file_size > MAX_FILE_SIZE:
            return False, "File size exceeds 100MB limit"
            
        # Check actual file content type using python-magic
        file_content = file.getvalue()
        mime = magic.Magic(mime=True)
        content_type = mime.from_buffer(file_content)
        
        valid_types = ['text/csv', 'text/plain', 'application/csv', 'application/vnd.ms-excel']
        if content_type not in valid_types:
            return False, f"Invalid file type detected: {content_type}"
        
        # Try reading first few rows to validate CSV structure
        df = pd.read_csv(BytesIO(file_content), nrows=5)
        
        # Basic content validation
        if len(df.columns) < 2:  # Ensure at least 2 columns
            return False, "CSV must contain at least 2 columns"
            
        # Check for potential code injection in column names
        dangerous_patterns = ['=', '+', '-', '@', '|', '<script']
        for col in df.columns:
            if any(pattern in str(col) for pattern in dangerous_patterns):
                return False, "Invalid characters detected in column names"
                
        # Check for reasonable number of columns (adjust limit as needed)
        if len(df.columns) > 1000:
            return False, "Too many columns (>1000) detected"
            
        return True, "File validation successful"
        
    except pd.errors.EmptyDataError:
        return False, "The file appears to be empty"
    except pd.errors.ParserError:
        return False, "Unable to parse CSV file - invalid format"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def main():
    st.title("Data Loading")
    
    # Add consistent navigation
    create_sidebar_navigation()
    
    # Initialize session state if needed
    if 'builder' not in st.session_state:
        st.session_state.builder = Builder()
        st.session_state.current_stage = ModelStage.DATA_LOADING
        st.session_state.logger.log_stage_transition("START", "DATA_LOADING")
    
    # Get and render stage info
    stage_info = st.session_state.builder.get_current_stage_info()
    
    # Edge case handling: occasionally stage_info may not be ready on first call (e.g., on refresh)
    if not stage_info:
        if 'logger' in st.session_state:
            st.session_state.logger.log_error(
                "Stage Info Fetch - Retry",
                {
                    "stage": "DATA_LOADING",
                    "reason": "stage_info was None or empty on first attempt",
                    "timestamp": datetime.now().isoformat()
                }
            )
        # Retry fetching once
        stage_info = st.session_state.builder.get_current_stage_info()

    # As a final safeguard, ensure stage_info has the expected structure
    if not stage_info:
        stage_info = {
            "title": "Data Loading",
            "description": "Securely load your dataset, validate its structure, and select the target variable to begin the ML workflow.",
            "requirements": [],
            "ethical_considerations": []
        }
    
    st.header(stage_info["title"])
    st.write(stage_info["description"])
    
    with st.expander("Functionality"):
        for req in stage_info["requirements"]:
            st.write(f"• {req}")
            
    with st.expander("Ethical Considerations"):
        for consideration in stage_info["ethical_considerations"]:
            st.write(f"• {consideration}")
     
    # Add Sample Data Section
    st.write("---")
    st.subheader("Load Sample Datasets")
    st.write("You can try out the application with these sample datasets:")
    
    # Store the file uploader in session state to preserve it across reruns
    if 'file_uploader' not in st.session_state:
        st.session_state.file_uploader = None
    
    # Define the base directory for sample data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level to project root
    sample_data_dir = os.path.join(base_dir, "sample_data")
    
    # Debug logging
    #st.write(f"Base directory: {base_dir}")
    #st.write(f"Sample data directory: {sample_data_dir}")
    #if os.path.exists(sample_data_dir):
    #    st.write(f"Files in sample_data directory: {os.listdir(sample_data_dir)}")
    #else:
    #    st.error(f"Sample data directory not found at: {sample_data_dir}")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Load Titanic Dataset"):
            try:
                titanic_path = os.path.join(sample_data_dir, "Titanic_Dataset_simple.csv")
                with open(titanic_path, "rb") as file:
                    file_content = file.read()
                file_obj = BytesIO(file_content)
                file_obj.name = "Titanic_Dataset_simple.csv"
                # Store both the content and name in session state
                st.session_state.sample_file = {
                    'content': file_content,
                    'name': 'Titanic_Dataset_simple.csv'
                }
                st.session_state.logger.log_user_action("Sample Data Loading", {"dataset": "titanic"})
                st.rerun()
            except FileNotFoundError:
                st.error("Titanic dataset file not found in sample_data directory")
    with col2:
        with st.popover("Titanic Dataset Info"):
            st.markdown("""
            **Titanic Dataset Variables:**
            This is a classification dataset with the following variables:
            - **survival**: Survival (0 = No, 1 = Yes) - the target variable
            - **pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
            - **sex**: Sex
            - **age**: Age in years
            - **sibsp**: # of siblings/spouses aboard
            - **parch**: # of parents/children aboard
            - **fare**: Passenger fare
            - **embarked**: Port of Embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)
            
            *Notes: 
            - pclass: Is a proxy for socio-economic status (1st=Upper, 2nd=Middle, 3rd=Lower)
            - age: If the age is estimated, is it in the form of xx.5
            - sipsp: Sibling = brother, sister, stepbrother, stepsister. Spouse = husband, wife (mistresses and fiancés were ignored)
            - parch: Parent = mother, father, Child = son, daughter.
            """)
    
    col3, col4 = st.columns([1, 3])
    with col3:
        if st.button("Load Miami Housing Dataset"):
            try:
                miami_path = os.path.join(sample_data_dir, "miami-housing_xs.csv")
                with open(miami_path, "rb") as file:
                    file_content = file.read()
                file_obj = BytesIO(file_content)
                file_obj.name = "miami-housing_xs.csv"
                # Store both the content and name in session state
                st.session_state.sample_file = {
                    'content': file_content,
                    'name': 'miami-housing_xs.csv'
                }
                st.session_state.logger.log_user_action("Sample Data Loading", {"dataset": "miami"})
                st.rerun()
            except FileNotFoundError:
                st.error("Miami Housing dataset file not found in sample_data directory")
    with col4:
        with st.popover("Miami Housing Dataset Info"):
            st.markdown("""
            **Miami Housing Dataset Variables:**
            This is a regression dataset with the following variables:
            - **PARCELNO**: unique identifier for each property. About 1% appear multiple times.
            - **SALE_PRC**: Sale price ($) - the target variable
            - **LND_SQFOOT**: Land area (square feet)
            - **TOT_LVG_AREA**: Floor area (square feet)
            - **SPEC_FEAT_VAL**: Value of special features ($)
            - **RAIL_DIST**: Distance to nearest rail line (feet)
            - **OCEAN_DIST**: Distance to ocean (feet)
            - **WATER_DIST**: Distance to nearest water body (feet)
            - **CNTR_DIST**: Distance to Miami CBD (feet)
            - **HWY_DIST**: Distance to nearest highway (feet)
            - **age**: Age of structure
            - **avno60plus**: Airplane noise exceeding acceptable level
            - **structure_quality**: Quality of structure
            - **month_sold**: Sale month in 2016 (1=Jan)
            - **LATITUDE/LONGITUDE**: Location coordinates
            """)
    
    # Upload Your Own Dataset
    st.write("---")
    st.subheader("Upload Your Own Dataset")
    file_upload = st.file_uploader("Choose a CSV file", type="csv")
    st.write("---")

    # Create a fresh BytesIO object from stored content if we have a sample file
    if 'sample_file' in st.session_state and not file_upload:
        file_obj = BytesIO(st.session_state.sample_file['content'])
        file_obj.name = st.session_state.sample_file['name']
        uploaded_file = file_obj
    else:
        uploaded_file = file_upload

    # Use logger from session state
    st.session_state.logger.log_page_state("Data_Loading", {
        "data_loaded": bool(st.session_state.get('data')),
        "target_selected": bool(st.session_state.get('target_column'))
    })
    
    if uploaded_file is not None:
        # Validate file before processing
        is_valid, validation_message = validate_csv_file(uploaded_file)
        
        if not is_valid:
            st.error(f"File validation failed: {validation_message}")
            st.session_state.logger.log_error("File Validation Failed", {"error": validation_message})
            st.stop()
            
        # Reset file pointer after validation
        uploaded_file.seek(0)
        
        # Use logger from session state
        st.session_state.logger.log_user_action("Data Loading Started", {
            "file_name": uploaded_file.name,
            "validation_status": "passed"
        })
        
        result = st.session_state.builder.load_data(uploaded_file)
        
        if result["success"]:
            # Store original data types right after successful data loading
            st.session_state.builder.original_dtypes = st.session_state.builder.data.dtypes.copy()
            
            st.success(result["message"])
            st.session_state.logger.log_calculation(
                "Data Summary",
                #result["info"]
            )
            
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
            
            # Target variable selection
            st.subheader("Select Target Variable")
            target_column = st.selectbox(
                "Choose the target variable (what you want to predict)",
                options=list(st.session_state.builder.data.columns),
                help="This is the variable your model will learn to predict"
            )
            
            if target_column:
                # Analyse target variable
                target_data = st.session_state.builder.data[target_column]

                # Check for missing values in target column
                missing_count = target_data.isnull().sum()
                total_rows = len(target_data)

                if missing_count > 0:
                    missing_percentage = (missing_count / total_rows) * 100

                    # Handle edge case: all values are missing
                    if missing_count == total_rows:
                        st.error(
                            f"❌ **All Target Values Missing**: "
                            f"Column '{target_column}' contains only missing values and cannot be used as a target. "
                            f"Please select a different target variable."
                        )
                        st.session_state.logger.log_error(
                            "Target Column All Missing",
                            {
                                "target_column": target_column,
                                "total_rows": int(total_rows),
                                "missing_count": int(missing_count)
                            }
                        )
                        st.stop()

                    # Warn user about missing values
                    if missing_percentage > 50:
                        st.error(
                            f"⚠️ **High Missing Target Values**: "
                            f"Found {missing_count} missing values ({missing_percentage:.1f}%) in target column '{target_column}'. "
                            f"This is more than 50% of your data. Consider reviewing your dataset quality. "
                            f"These rows will be automatically removed as they cannot be used for model training."
                        )
                    else:
                        st.warning(
                            f"⚠️ **Missing Target Values Detected**: "
                            f"Found {missing_count} missing values ({missing_percentage:.1f}%) in target column '{target_column}'. "
                            f"These rows will be automatically removed as they cannot be used for model training."
                        )

                    # Remove rows with missing target values
                    valid_mask = target_data.notna()
                    st.session_state.builder.data = st.session_state.builder.data[valid_mask].reset_index(drop=True)

                    # Update target_data after removal
                    target_data = st.session_state.builder.data[target_column]
                    rows_after = len(st.session_state.builder.data)

                    # Log the removal action
                    st.session_state.logger.log_user_action(
                        "Missing Target Values Removed",
                        {
                            "target_column": target_column,
                            "missing_count": int(missing_count),
                            "missing_percentage": float(missing_percentage),
                            "rows_before": int(total_rows),
                            "rows_after": int(rows_after)
                        }
                    )

                    # Check if remaining dataset is very small
                    if rows_after < 50:
                        st.warning(
                            f"⚠️ **Small Dataset Warning**: "
                            f"After removing missing values, only {rows_after} rows remain. "
                            f"This may not be sufficient for reliable model training. "
                            f"Consider collecting more data or reviewing data quality."
                        )

                    # Show updated statistics
                    st.info(f"✅ **Dataset Updated**: Removed {missing_count} rows with missing target values. New dataset size: {rows_after} rows")

                # Check if numeric
                is_numeric = np.issubdtype(target_data.dtype, np.number)
                unique_count = target_data.nunique()
                
                # Initialize classification/regression flags
                is_binary = False
                is_multiclass = False
                is_regression = False
                encoding_mapping = None
                can_proceed = False
                
                if is_numeric:
                    if unique_count == 2:
                        is_binary = True
                        can_proceed = True
                    elif 3 <= unique_count <= 10:
                        # Ask user to confirm if this should be multiclass or regression
                        use_multiclass = st.radio(
                            f"'{target_column}' is numeric with {unique_count} unique values. How would you like to treat it?",
                            options=['Regression (continuous values)', 'Multiclass Classification (discrete classes)'],
                            index=0,
                            key=f"numeric_multiclass_{target_column}"
                        )
                        if use_multiclass == 'Multiclass Classification (discrete classes)':
                            # Check if 0 is in the unique values
                            unique_vals = target_data.dropna().unique()
                            has_zero = 0 in unique_vals or 0.0 in unique_vals
                            
                            if not has_zero:
                                # Convert to categorical and apply label encoding like categorical features
                                unique_cats = sorted([str(val) for val in unique_vals])
                                encoding_mapping = {cat: i for i, cat in enumerate(unique_cats)}
                                # Encode the target - convert to string first to match mapping keys
                                st.session_state.builder.data[target_column] = target_data.astype(str).map(encoding_mapping)
                                st.info(f"ℹ️ Since '{target_column}' doesn't include 0, it's being treated as categorical and label encoded.")
                                
                                # Store the encoding mapping for later use (same as categorical variables)
                                if "encoding_mappings" not in st.session_state:
                                    st.session_state.encoding_mappings = {}
                                if not hasattr(st.session_state.builder, 'encoding_mappings'):
                                    st.session_state.builder.encoding_mappings = {}
                                
                                # Store in both session state and builder
                                mapping_info = {
                                    "method": "Target Label Encoding (Numeric to Categorical)",
                                    "mapping": encoding_mapping,
                                    "original_values": list(encoding_mapping.keys()),
                                    "encoded_values": list(encoding_mapping.values()),
                                    "is_target_variable": True,
                                    "was_numeric": True
                                }
                                st.session_state.encoding_mappings[target_column] = mapping_info
                                st.session_state.builder.encoding_mappings[target_column] = mapping_info
                            else:
                                # Convert float values to int for multiclass classification
                                # This prevents errors later in the application when float classes are used
                                if target_data.dtype in [np.float32, np.float64, float]:
                                    st.session_state.builder.data[target_column] = target_data.astype(int)
                                    st.info(f"ℹ️ Converted float target values to integers for multiclass classification.")
                            
                            is_multiclass = True
                            can_proceed = True
                        else:
                            is_regression = True
                            can_proceed = True
                    else:
                        is_regression = True
                        can_proceed = True
                else:
                    # Categorical variable
                    if unique_count == 2:
                        use_categorical = st.radio(
                            f"'{target_column}' is a categorical variable with 2 unique values. Would you like to use it as a binary classification target?",
                            options=['No', 'Yes'],
                            index=0,
                            key=f"binary_{target_column}"
                        )
                        if use_categorical == 'Yes':
                            # Create binary mapping - handle missing values and mixed types
                            unique_vals = target_data.dropna().unique()  # Remove NaN values
                            unique_cats = sorted([str(val) for val in unique_vals])  # Convert to string for consistent sorting
                            encoding_mapping = {unique_cats[0]: 0, unique_cats[1]: 1}
                            # Encode the target - convert to string first to match mapping keys
                            st.session_state.builder.data[target_column] = target_data.astype(str).map(encoding_mapping)
                            is_numeric = True
                            is_binary = True
                            can_proceed = True
                    elif 3 <= unique_count <= 10:
                        use_multiclass = st.radio(
                            f"'{target_column}' is a categorical variable with {unique_count} unique values. Would you like to use it as a multiclass classification target?",
                            options=['No', 'Yes'],
                            index=0,
                            key=f"multiclass_{target_column}"
                        )
                        if use_multiclass == 'Yes':
                            # Create label encoding mapping - handle missing values and mixed types
                            unique_vals = target_data.dropna().unique()  # Remove NaN values
                            unique_cats = sorted([str(val) for val in unique_vals])  # Convert to string for consistent sorting
                            encoding_mapping = {cat: i for i, cat in enumerate(unique_cats)}
                            # Encode the target - convert to string first to match mapping keys
                            st.session_state.builder.data[target_column] = target_data.astype(str).map(encoding_mapping)
                            is_numeric = True
                            is_multiclass = True
                            can_proceed = True
                    elif unique_count > 10:
                        st.warning(f"⚠️ '{target_column}' has {unique_count} unique categories. This is too many for practical classification.")
                        can_proceed = False
                
                # Display target variable analysis
                st.write("Target Variable Analysis:")
                if not can_proceed:
                    if not is_numeric and encoding_mapping is None:
                        st.error("❌ The selected target variable must be numeric or have a reasonable number of categories for classification.")
                elif is_binary:
                    st.success("✅ Valid target for binary classification")
                    if encoding_mapping:
                        st.write("**Target Binary Encoding Mapping:**")
                        # Create a nice table display
                        mapping_df = pd.DataFrame({
                            "Original Class": list(encoding_mapping.keys()),
                            "Encoded Value": list(encoding_mapping.values())
                        })
                        st.dataframe(
                            mapping_df.style.background_gradient(cmap='Blues', axis=0),
                            width='stretch',
                            hide_index=True
                        )
                    else:
                        # Handle mixed types and missing values safely
                        unique_vals = target_data.dropna().unique()
                        try:
                            unique_vals = sorted(unique_vals)
                        except TypeError:
                            # If sorting fails due to mixed types, convert to string first
                            unique_vals = sorted([str(val) for val in unique_vals])
                        
                        # Convert numpy types to regular Python types for cleaner display
                        clean_vals = [int(val) if isinstance(val, (np.integer, np.int64)) else 
                                    float(val) if isinstance(val, (np.floating, np.float64)) else 
                                    str(val) for val in unique_vals]
                        st.write(f"Binary classes: {clean_vals[0]} and {clean_vals[1]}")
                elif is_multiclass:
                    st.success("✅ Valid target for multiclass classification")
                    if encoding_mapping:
                        st.write("**Target Label Encoding Mapping:**")
                        # Create a nice table display
                        mapping_df = pd.DataFrame({
                            "Original Class": list(encoding_mapping.keys()),
                            "Encoded Value": list(encoding_mapping.values())
                        })
                        st.dataframe(
                            mapping_df.style.background_gradient(cmap='Blues', axis=0),
                            width='stretch',
                            hide_index=True
                        )
                    else:
                        # Handle mixed types and missing values safely
                        unique_vals = target_data.dropna().unique()
                        try:
                            unique_vals = sorted(unique_vals)
                        except TypeError:
                            # If sorting fails due to mixed types, convert to string first
                            unique_vals = sorted([str(val) for val in unique_vals])
                        
                        # Convert numpy types to regular Python types for cleaner display
                        clean_vals = [int(val) if isinstance(val, (np.integer, np.int64)) else 
                                    float(val) if isinstance(val, (np.floating, np.float64)) else 
                                    str(val) for val in unique_vals]
                        st.write(f"Classes: {clean_vals}")
                    st.write(f"Number of classes: {unique_count}")
                elif is_regression:
                    st.success("✅ Valid target for regression")
                    st.write(f"Range: {target_data.min():.2f} to {target_data.max():.2f}")
                    st.write(f"Number of unique values: {unique_count}")
            
            proceed_button = st.button(
                "Set Target and Proceed to Data Exploration", type="primary",
                disabled=not can_proceed if 'can_proceed' in locals() else True
            )
            
            if proceed_button:
                # Store target variable and log the action
                st.session_state.builder.target_column = target_column
                
                # Store target encoding mapping if categorical encoding was applied
                # Only store if the mapping doesn't already exist (to avoid overwriting numeric-to-categorical mappings)
                if encoding_mapping is not None and (target_column not in st.session_state.get('encoding_mappings', {})):
                    # Initialize encoding_mappings in session state if it doesn't exist
                    if "encoding_mappings" not in st.session_state:
                        st.session_state.encoding_mappings = {}
                    
                    # Store the target encoding mapping
                    st.session_state.encoding_mappings[target_column] = {
                        "method": "Target Label Encoding",
                        "mapping": encoding_mapping,
                        "original_values": list(encoding_mapping.keys()),
                        "encoded_values": list(encoding_mapping.values()),
                        "is_target_variable": True
                    }
                    
                    # Also store in builder's encoding_mappings
                    if not hasattr(st.session_state.builder, 'encoding_mappings'):
                        st.session_state.builder.encoding_mappings = {}
                    st.session_state.builder.encoding_mappings[target_column] = {
                        "method": "Target Label Encoding", 
                        "mapping": encoding_mapping,
                        "original_values": list(encoding_mapping.keys()),
                        "encoded_values": list(encoding_mapping.values()),
                        "is_target_variable": True
                    }
                
                # Determine problem type for logging and store in session state
                problem_type = "regression"
                if is_binary:
                    problem_type = "binary_classification"
                elif is_multiclass:
                    problem_type = "multiclass_classification"
                
                # Store problem type and classification flags in session state
                st.session_state.problem_type = problem_type
                st.session_state.is_binary = is_binary
                st.session_state.is_multiclass = is_multiclass
                st.session_state.is_regression = is_regression
                
                st.session_state.logger.log_user_action(
                    "Target Selection",
                    {
                        "target_column": target_column,
                        "problem_type": problem_type,
                        "is_binary": is_binary,
                        "is_multiclass": is_multiclass,
                        "is_numeric": is_numeric,
                        "unique_count": unique_count,
                        "encoding_applied": encoding_mapping is not None
                    }
                )
                # Log the journey point
                st.session_state.logger.log_journey_point(
                    stage="DATA_LOADING",
                    decision_type="DATA_LOADED",
                    description="Dataset loaded successfully: " + uploaded_file.name,
                    details={'Number of Rows': len(st.session_state.builder.data), 'Number of Columns' : len(st.session_state.builder.data.columns)},
                    parent_id=None  # This will be the root node
                )
                
                # Log the journey point - parent_id will be automatically set to previous node
                st.session_state.logger.log_journey_point(
                    stage="DATA_LOADING",
                    decision_type="TARGET_SELECTION",
                    description="Target variable selected",
                    details={
                        'target_column': target_column, 
                        'problem_type': problem_type,
                        'is_binary': is_binary,
                        'is_multiclass': is_multiclass,
                        'is_numeric': is_numeric,
                        'unique_count': unique_count
                    },
                    parent_id=None  # Will automatically use the last node as parent
                )
                
                st.session_state.builder.stage_completion[ModelStage.DATA_LOADING] = True
                st.session_state.logger.log_stage_transition(
                    "DATA_LOADING",
                    "DATA_EXPLORATION"
                )
                
                next_page = "2_Data_Exploration"
                st.switch_page(f"pages/{next_page}.py")
        else:
            st.error(result["message"])
            st.session_state.logger.log_error(
                "Data Loading Failed",
                {"error": result["message"]}
            )
    
    # Flush logs before rendering the log viewer
    if 'logger' in st.session_state:
        st.session_state.logger.flush_logs()
    
    st.divider()
    # Add log viewer at the bottom of the page
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