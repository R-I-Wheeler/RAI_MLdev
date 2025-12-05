import streamlit as st
from Builder import Builder, ModelStage
from app import create_sidebar_navigation
from utils.logging.journey_viewer import render_journey_viewer
from utils.logging.log_viewer import render_log_viewer
import time
from streamlit_scroll_to_top import scroll_to_here
import numpy as np


# Import custom components
from components.data_preprocessing.feature_management import FeatureManagementComponent
from components.data_preprocessing.zero_values_analysis import ZeroValuesAnalysis
from components.data_preprocessing.train_test_split import TrainTestSplitComponent
from components.data_preprocessing.missing_values_analysis import MissingValuesAnalysis
from components.data_preprocessing.feature_creation import FeatureCreationComponent
from components.data_preprocessing.feature_binning import FeatureBinningComponent
from components.data_preprocessing.outlier_detection import OutlierDetectionComponent
from components.data_preprocessing.categorical_encoding import CategoricalEncodingComponent
from components.data_preprocessing.data_types_optimization import DataTypesOptimisationComponent
from components.data_preprocessing.final_data_review import FinalDataReviewComponent

def scroll():
    
    st.session_state.scroll_to_top = True
    
    st.rerun()

def fc_scroll():
    st.session_state.scroll_to_top = True

def main():
    
    if 'scroll_to_top' not in st.session_state:
        st.session_state.scroll_to_top = False
    
    if 'preprocessing_step' not in st.session_state:
        st.session_state.preprocessing_step = 'feature_management'
    
    if 'categorical_encoding_complete' not in st.session_state:
        st.session_state.categorical_encoding_complete = False
        
    # Reset categorical encoding completion when moving back to overview
    if st.session_state.preprocessing_step == 'feature_management':
        st.session_state.categorical_encoding_complete = False

    if 'missing_values_complete' not in st.session_state:
        st.session_state.missing_values_complete = False
        
    # Reset categorical encoding completion when moving back to overview
    if st.session_state.preprocessing_step == 'feature_management':
        st.session_state.missing_values_complete = False

    if 'final_preprocessing_training_data' not in st.session_state:
        st.session_state.final_preprocessing_training_data = None

    
    st.title("Data Preprocessing")
    
    # Add consistent navigation
    create_sidebar_navigation()
    
    # Initialize session state if needed
    if 'builder' not in st.session_state:
        st.session_state.builder = Builder()
        st.session_state.logger.log_stage_transition("START", "DATA_PREPROCESSING")
    
    # Set current stage to DATA_PREPROCESSING
    st.session_state.builder.current_stage = ModelStage.DATA_PREPROCESSING
    
    # Initialize preprocessing step if not exists
    if 'preprocessing_step' not in st.session_state:
        st.session_state.preprocessing_step = 'feature_management'
        st.session_state.logger.log_user_action(
            "Preprocessing Started",
            {"initial_step": "feature_management"}
        )
    
    # Get and render stage info
    stage_info = st.session_state.builder.get_current_stage_info()
    
    st.header(stage_info["title"])
    st.write(stage_info["description"])
    
    with st.expander("Functionality"):
        for req in stage_info["requirements"]:
            st.markdown(req)
            
    with st.expander("Ethical Considerations"):
        for consideration in stage_info["ethical_considerations"]:
            st.markdown(consideration)

    if st.session_state.builder.data is not None:
        # Log initial data state
        st.session_state.logger.log_calculation(
            "Initial Data State",
            {
                "rows": len(st.session_state.builder.data),
                "columns": len(st.session_state.builder.data.columns),
                "missing_values": st.session_state.builder.data.isnull().sum().sum()
            }
        )
        
        st.write("### Data Preprocessing Pipeline")
        
        st.info("""
            Follow these steps to preprocess your data:
            1. Review dataset and remove unwanted columns
            2. Check for zero values that may represent missing data
            3. Split the data into training and testing sets
            4. Handle missing values
            5. Apply variable binning strategies
            6. Handle outliers
            7. Handle categorical variables
            8. Create new features from existing numeric features (optional)
            9. Optimise feature datatypes for memory efficiency
            10. Review your dataset
            
            Progress through each step using the 'Continue' button at the bottom.
        """)
        # Note about normalization
        st.markdown("""
        **Note**: This application intentionally excludes normalization/standardization and advanced transformations 
        to maintain model explainability. For features with very different scales or non-linear relationships:
        - Use **Variable Binning** to handle outliers and non-linear patterns
        - Apply **Outlier Handling** to cap extreme values
        - Focus on proper **Missing Value** handling and **Categorical Encoding**
        
        These approaches provided in the preprocessing stage help prepare your data while maintaining interpretability.
        """)
        st.write("---")
        # Store minimal metadata for preprocessing tracking (memory optimized)
        if 'preprocessing_entry_stage' not in st.session_state:
            st.session_state.preprocessing_entry_stage = 'feature_management'
            
        # Store original data backup for reset functionality (only once when entering preprocessing)
        if 'preprocessing_original_data' not in st.session_state:
            st.session_state.preprocessing_original_data = st.session_state.builder.data.copy()
        
        # Add simple undo button at the top - recreate original data on demand
        if 'show_reset_confirmation' not in st.session_state:
            st.session_state.show_reset_confirmation = False
            
        if st.button("↺ Undo All Changes", type="primary"):
            st.session_state.show_reset_confirmation = True
            st.rerun()
            
        if st.session_state.show_reset_confirmation:
            # Show warning and confirmation in an expander
            with st.expander("⚠️ Reset Confirmation", expanded=True):
                st.warning("This will reset all preprocessing changes and restore the dataset to its original state from when you first entered preprocessing.")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("✅ Confirm Reset", type="secondary", width='stretch'):
                        # Restore original data from backup
                        if 'preprocessing_original_data' in st.session_state:
                            st.session_state.builder.data = st.session_state.preprocessing_original_data.copy()
                        
                        # Clear preprocessing-related session state
                        keys_to_remove = [key for key in st.session_state.keys() 
                                        if any(prefix in key for prefix in [
                                            'handling_', 'binning_', 'encoding_', 'outlier_', 'cat_method_', 
                                            'preprocessing_', 'feature_management_', 'zero_values_', 'tts_',
                                            'missing_values_', 'feature_binning_', 'feature_creation_',
                                            'outlier_detection_', 'categorical_encoding_', 'show_reset_confirmation'
                                        ])]
                        for key in keys_to_remove:
                            del st.session_state[key]
                        
                        # Reset preprocessing step
                        st.session_state.preprocessing_step = 'feature_management'
                        
                        # Clear train/test split data to force recomputation
                        st.session_state.builder.training_data = None
                        st.session_state.builder.testing_data = None
                        if hasattr(st.session_state.builder, 'X_train'):
                            st.session_state.builder.X_train = None
                        if hasattr(st.session_state.builder, 'X_test'):
                            st.session_state.builder.X_test = None
                        if hasattr(st.session_state.builder, 'y_train'):
                            st.session_state.builder.y_train = None
                        if hasattr(st.session_state.builder, 'y_test'):
                            st.session_state.builder.y_test = None
                        
                        st.session_state.logger.log_user_action(
                            "Dataset Reset to Original",
                            {
                                "action": "reset_to_original_data",
                                "original_shape": st.session_state.builder.data.shape
                            }
                        )
                        
                        st.success("All changes undone! Dataset restored to original state.")
                        time.sleep(1)
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", width='stretch'):
                        st.session_state.show_reset_confirmation = False
                        st.rerun()

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.session_state.get('original_data') is not None:
                changes = []
                if len(st.session_state.builder.data) != len(st.session_state.original_data):
                    changes.append(f"Rows: {len(st.session_state.original_data)} → {len(st.session_state.builder.data)}")
                if len(st.session_state.builder.data.columns) != len(st.session_state.original_data.columns):
                    changes.append(f"Columns: {len(st.session_state.original_data.columns)} → {len(st.session_state.builder.data.columns)}")
                if changes:
                    st.info("Changes from original: " + ", ".join(changes))
        
        # Add progress indicator
        current_step = st.session_state.preprocessing_step
        steps = ['feature_management',  'zero_values', 'split_data', 'missing_values', 'binning', 'outliers',  'categorical', 'feature_creation', 'data_types', 'final']
        step_labels = ['Feature Management', 'Zero Values', 'Train Test Split','Missing Values', 'Feature Binning', 'Outliers', 'Categorical Features', 'Feature Creation','Data Types', 'Final Review']
        current_step_idx = steps.index(current_step)
        
        # Define the step sequence and labels
        step_sequence = {
            'feature_management': {'prev': None, 'next': 'zero_values', 'label': 'Feature Management'},
            'zero_values': {'prev': 'feature_management', 'next': 'split_data', 'label': 'Zero Values'},
            'split_data': {'prev': 'zero_values', 'next': 'missing_values', 'label': 'Train Test Split'},
            'missing_values': {'prev': 'split_data', 'next': 'binning', 'label': 'Missing Values'},
            'binning': {'prev': 'missing_values', 'next': 'outliers', 'label': 'Feature Binning'},
            'outliers': {'prev': 'binning', 'next': 'categorical', 'label': 'Outliers'},
            'categorical': {'prev': 'outliers', 'next': 'feature_creation', 'label': 'Categorical Features'},
            'feature_creation': {'prev': 'categorical', 'next': 'data_types', 'label': 'Feature Creation'},
            'data_types': {'prev': 'feature_creation', 'next': 'final', 'label': 'Data Types'},
            'final': {'prev': 'data_types', 'next': None, 'label': 'Final Review'}
        }
        
        # Get current step info
        step_info = step_sequence[current_step]
        
        # Set position to scroll to the top of the page
        if st.session_state.scroll_to_top:
            scroll_to_here(0, key='top')  # Scroll to the top of the page, 0 means instantly, but you can add a delay (im milliseconds)
            st.session_state.scroll_to_top = False  # Reset the state after scrolling


        progress_html = "<div style='display: flex; justify-content: space-between; margin-bottom: 1rem;'>"
        for i, (step, label) in enumerate(zip(steps, step_labels)):
            if i == current_step_idx:
                style = "color: #FF4B4B; font-weight: bold"  # Current step
            elif i < current_step_idx:
                style = "color: #00CC00"  # Completed step
            else:
                style = "color: #808080"  # Future step
            progress_html += f"<div style='{style}'>{label}</div>"
        progress_html += "</div>"
        st.markdown(progress_html, unsafe_allow_html=True)
        
        if st.session_state.preprocessing_step == 'feature_management':
            # Use the new FeatureManagementComponent
            feature_manager = FeatureManagementComponent(st.session_state.builder, st.session_state.logger)
            feature_manager.render()

            # Add navigation buttons
            st.markdown("---")  # Add a visual separator
            nav_cols = st.columns([1, 2, 1])
            
            with nav_cols[1]:
                st.markdown(
                    """
                    <style>
                    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
                        width: 100%;
                        background-color: #f0f2f6;
                        border: 1px solid #e0e0e0;
                    }
                    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
                        width: 100%;
                        background-color: #FF4B4B;
                        color: white;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                nav_button_cols = st.columns([1, 1])
                with nav_button_cols[0]:
                    if step_info['prev']:
                        if st.button(f"← Back to {step_sequence[step_info['prev']]['label']}", width='stretch'):
                            st.session_state.preprocessing_step = step_info['prev']
                            st.rerun()
                
                with nav_button_cols[1]:
                    if step_info['next']:
                        if st.button(f"Continue to {step_sequence[step_info['next']]['label']} →", width='stretch', on_click=scroll):
                            # Log stage transition
                            st.session_state.logger.log_stage_transition(
                                "Feature Management",
                                step_sequence[step_info['next']]['label']
                            )
                            st.session_state.preprocessing_step = step_info['next']
                            st.session_state.scroll_to_top = True
                            st.rerun()
        elif st.session_state.preprocessing_step == 'zero_values':
            # Log page state when entering zero values analysis
            st.session_state.logger.log_page_state(
                "Data Preprocessing - Zero Values",
                {
                    "current_step": "zero_values",
                    "dataset_shape": st.session_state.builder.data.shape
                }
            )

            # Render zero values analysis component
            zero_values_component = ZeroValuesAnalysis(st.session_state.builder)
            zero_values_component.render_zero_values_analysis()
            
            # Navigation buttons
            st.markdown("---")
            nav_cols = st.columns([1, 2, 1])
            
            with nav_cols[1]:
                st.markdown(
                    """
                    <style>
                    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
                        width: 100%;
                        background-color: #f0f2f6;
                        border: 1px solid #e0e0e0;
                    }
                    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
                        width: 100%;
                        background-color: #FF4B4B;
                        color: white;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                nav_button_cols = st.columns([1, 1])
                with nav_button_cols[0]:
                    if step_info['prev']:
                        if st.button(f"← Back to {step_sequence[step_info['prev']]['label']}", width='stretch'):
                            st.session_state.preprocessing_step = step_info['prev']
                            st.rerun()
                
                with nav_button_cols[1]:
                    if step_info['next']:
                        if st.button(f"Continue to {step_sequence[step_info['next']]['label']} →", width='stretch', on_click=scroll):
                            # Log stage transition
                            st.session_state.logger.log_stage_transition(
                                step_sequence[current_step]['label'],
                                step_sequence[step_info['next']]['label']
                            )
                            st.session_state.preprocessing_step = step_info['next']
                            st.session_state.scroll_to_top = True
                            st.rerun()

        elif st.session_state.preprocessing_step == 'split_data':
            # Log page state when entering train test split analysis
            st.session_state.logger.log_page_state(
                "Data Preprocessing - Train Test Split",
                {
                    "current_step": "split_data",
                    "dataset_shape": st.session_state.builder.data.shape
                }
            )
            
            # Render train test split component
            train_test_split_component = TrainTestSplitComponent(st.session_state.builder)
            train_test_split_component.render_train_test_split()
            
            # Add navigation buttons
            st.markdown("---")  # Add a visual separator
            nav_cols = st.columns([1, 2, 1])
            
            with nav_cols[1]:
                st.markdown(
                    """
                    <style>
                    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
                        width: 100%;
                        background-color: #f0f2f6;
                        border: 1px solid #e0e0e0;
                    }
                    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
                        width: 100%;
                        background-color: #FF4B4B;
                        color: white;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                nav_button_cols = st.columns([1, 1])
                with nav_button_cols[0]:
                    if step_info['prev']:
                        if st.button(f"← Back to {step_sequence[step_info['prev']]['label']}", width='stretch'):
                            st.session_state.preprocessing_step = step_info['prev']
                            st.rerun()
                
                with nav_button_cols[1]:
                    if step_info['next']:
                        if st.button(f"Continue to {step_sequence[step_info['next']]['label']} →", width='stretch', on_click=scroll):
                            # Log stage transition
                            st.session_state.logger.log_stage_transition(
                                step_sequence[current_step]['label'],
                                step_sequence[step_info['next']]['label']
                            )
                            st.session_state.preprocessing_step = step_info['next']
                            st.session_state.scroll_to_top = True
                            st.rerun()
        
        elif st.session_state.preprocessing_step == 'missing_values':
            # Log page state when entering missing values analysis
            st.session_state.logger.log_page_state(
                "Data Preprocessing - Missing Values",
                {
                    "current_step": "missing_values",
                    "dataset_shape": st.session_state.builder.training_data.shape,
                    "total_missing": st.session_state.builder.training_data.isnull().sum().sum()
                }
            )

            # Use the MissingValuesAnalysis component
            missing_values_component = MissingValuesAnalysis(st.session_state.builder)
            missing_values_component.render_missing_values_analysis()

            # Add navigation buttons
            st.markdown("---")  # Add a visual separator
            nav_cols = st.columns([1, 2, 1])
            
            with nav_cols[1]:
                st.markdown(
                    """
                    <style>
                    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
                        width: 100%;
                        background-color: #f0f2f6;
                        border: 1px solid #e0e0e0;
                    }
                    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
                        width: 100%;
                        background-color: #FF4B4B;
                        color: white;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                nav_button_cols = st.columns([1, 1])
                with nav_button_cols[0]:
                    if step_info['prev']:
                        if st.button(f"← Back to {step_sequence[step_info['prev']]['label']}", width='stretch'):
                            st.session_state.preprocessing_step = step_info['prev']
                            st.rerun()
                
                with nav_button_cols[1]:
                    if step_info['next']:
                        # Only enable the continue button if missing values is complete
                        if st.session_state.missing_values_complete:
                            if st.button(f"Continue to {step_sequence[step_info['next']]['label']} →", width='stretch', on_click=scroll):
                                # Log stage transition
                                st.session_state.logger.log_stage_transition(
                                    step_sequence[current_step]['label'],
                                    step_sequence[step_info['next']]['label']
                                )
                                st.session_state.preprocessing_step = step_info['next']
                                st.session_state.scroll_to_top = True
                                st.rerun()
                        else:
                            st.button(f"Continue to {step_sequence[step_info['next']]['label']} →", disabled=True, help="Please apply missing values before continuing", width='stretch')
        
        elif st.session_state.preprocessing_step == 'binning':
            # Log page state when entering binning analysis
            st.session_state.logger.log_page_state(
                "Data Preprocessing - Binning",
                {
                    "current_step": "binning",
                    "dataset_shape": st.session_state.builder.training_data.shape,
                    "numeric_columns_count": len([col for col in st.session_state.builder.training_data.select_dtypes(include=[np.number]).columns if col != st.session_state.builder.target_column]),
                    "categorical_columns_count": len([col for col in st.session_state.builder.training_data.select_dtypes(include=['object', 'category']).columns if col != st.session_state.builder.target_column])
                }
            )
            
            # Use the FeatureBinningComponent to render the feature binning interface
            feature_binning = FeatureBinningComponent(st.session_state.builder, st.session_state.logger)
            feature_binning.render()
            
            # Add navigation buttons
            st.markdown("---")  # Add a visual separator
            nav_cols = st.columns([1, 2, 1])
            
            with nav_cols[1]:
                st.markdown(
                    """
                    <style>
                    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
                        width: 100%;
                        background-color: #f0f2f6;
                        border: 1px solid #e0e0e0;
                    }
                    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
                        width: 100%;
                        background-color: #FF4B4B;
                        color: white;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                nav_button_cols = st.columns([1, 1])
                with nav_button_cols[0]:
                    if step_info['prev']:
                        if st.button(f"← Back to {step_sequence[step_info['prev']]['label']}", width='stretch'):
                            st.session_state.preprocessing_step = step_info['prev']
                            st.rerun()
                
                with nav_button_cols[1]:
                    if step_info['next']:
                        if st.button(f"Continue to {step_sequence[step_info['next']]['label']} →", width='stretch', on_click=scroll):
                            # Log stage transition
                            st.session_state.logger.log_stage_transition(
                                step_sequence[current_step]['label'],
                                step_sequence[step_info['next']]['label']
                            )
                            st.session_state.preprocessing_step = step_info['next']
                            st.session_state.scroll_to_top = True
                            st.rerun()
        
        elif st.session_state.preprocessing_step == 'outliers':
            st.session_state.logger.log_page_state(
                "Data Preprocessing - Outliers",
                {
                    "current_step": "outliers",
                    "dataset_shape": st.session_state.builder.training_data.shape,
                    "numeric_columns_count": len([col for col in st.session_state.builder.training_data.select_dtypes(include=[np.number]).columns if col != st.session_state.builder.target_column]),
                    "categorical_columns_count": len([col for col in st.session_state.builder.training_data.select_dtypes(include=['object', 'category']).columns if col != st.session_state.builder.target_column])
                }
            )
            
            # Use the OutlierDetectionComponent to render the outlier detection interface
            outlier_detection = OutlierDetectionComponent(st.session_state.builder, st.session_state.logger)
            outlier_detection.render()
            
            # Add navigation buttons
            st.markdown("---")  # Add a visual separator
            nav_cols = st.columns([1, 2, 1])
            
            with nav_cols[1]:
                st.markdown(
                    """
                    <style>
                    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
                        width: 100%;
                        background-color: #f0f2f6;
                        border: 1px solid #e0e0e0;
                    }
                    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
                        width: 100%;
                        background-color: #FF4B4B;
                        color: white;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                nav_button_cols = st.columns([1, 1])
                with nav_button_cols[0]:
                    if step_info['prev']:
                        if st.button(f"← Back to {step_sequence[step_info['prev']]['label']}", width='stretch'):
                            st.session_state.preprocessing_step = step_info['prev']
                            st.rerun()
                
                with nav_button_cols[1]:
                    if step_info['next']:
                        if st.button(f"Continue to {step_sequence[step_info['next']]['label']} →", width='stretch', on_click=scroll):
                            # Log stage transition
                            st.session_state.logger.log_stage_transition(
                                step_sequence[current_step]['label'],
                                step_sequence[step_info['next']]['label']
                            )
                            st.session_state.preprocessing_step = step_info['next']
                            st.session_state.scroll_to_top = True
                            st.rerun()
                            
        elif st.session_state.preprocessing_step == 'categorical':
            st.session_state.logger.log_page_state(
                "Data Preprocessing - Categorical Variables",
                {
                    "current_step": "categorical",
                    "dataset_shape": st.session_state.builder.training_data.shape,
                    "numeric_columns_count": len([col for col in st.session_state.builder.training_data.select_dtypes(include=[np.number]).columns if col != st.session_state.builder.target_column]),
                    "categorical_columns_count": len([col for col in st.session_state.builder.training_data.select_dtypes(include=['object', 'category']).columns if col != st.session_state.builder.target_column])
                }
            )
            
            # Create and render the categorical encoding component
            categorical_component = CategoricalEncodingComponent(st.session_state.builder, st.session_state.logger)
            categorical_component.render()
            
            # Check if all categorical features have been encoded or if there are no categorical features
            categorical_cols = st.session_state.builder.training_data.select_dtypes(include=['object', 'category']).columns
            categorical_cols_test = st.session_state.builder.testing_data.select_dtypes(include=['object', 'category']).columns if hasattr(st.session_state.builder, 'testing_data') and st.session_state.builder.testing_data is not None else []
            
            encoding_complete = getattr(st.session_state, 'categorical_encoding_complete', False)
            no_categorical_features = len(categorical_cols) == 0 and len(categorical_cols_test) == 0
            
            can_proceed = encoding_complete or no_categorical_features
            
            if can_proceed == False:
                st.warning("Categorical encoding is not complete. Please apply categorical encoding before continuing. Categorical columns found: ")
            # Add navigation buttons
            st.markdown("---")  # Add a visual separator
            nav_cols = st.columns([1, 2, 1])
            
            with nav_cols[1]:
                st.markdown(
                    """
                    <style>
                    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
                        width: 100%;
                        background-color: #f0f2f6;
                        border: 1px solid #e0e0e0;
                    }
                    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
                        width: 100%;
                        background-color: #f0f2f6;
                        border: 1px solid #e0e0e0;
                    }
                    div[data-testid="stHorizontalBlock"] > div:nth-child(3) button {
                        width: 100%;
                        background-color: #FF4B4B;
                        color: white;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                nav_button_cols = st.columns([1, 1, 1])
                with nav_button_cols[0]:
                    if step_info['prev']:
                        if st.button(f"← Back to {step_sequence[step_info['prev']]['label']}", width='stretch'):
                            st.session_state.preprocessing_step = step_info['prev']
                            st.rerun()
                
                with nav_button_cols[1]:
                     if st.button("Skip Feature Creation ↠", width='stretch', help="Skip feature creation and go directly to data types"):
                        # Log stage transition
                        st.session_state.logger.log_stage_transition(
                            step_sequence[current_step]['label'],
                            "Data Types"
                        )
                        st.session_state.skipped_feature_creation = True
                        st.session_state.preprocessing_step = 'data_types'
                        st.session_state.scroll_to_top = True
                        st.rerun()

                with nav_button_cols[2]:
                    if step_info['next']:
                        # Only enable the continue button if missing values is complete
                        if can_proceed:
                            if st.button(f"Continue to {step_sequence[step_info['next']]['label']} →", width='stretch', on_click=scroll):
                                # Log stage transition
                                st.session_state.logger.log_stage_transition(
                                    step_sequence[current_step]['label'],
                                    step_sequence[step_info['next']]['label']
                                )
                                st.session_state.skipped_feature_creation = False
                                st.session_state.preprocessing_step = step_info['next']
                                st.session_state.scroll_to_top = True
                                st.rerun()
                        else:
                            st.button(f"Continue to {step_sequence[step_info['next']]['label']} →", disabled=True, help="Please apply categorical encoding before continuing", width='stretch')

        elif st.session_state.preprocessing_step == 'feature_creation':
            # Log page state when entering feature creation
            st.session_state.logger.log_page_state(
                "Data Preprocessing - Feature Creation",
                {
                    "current_step": "feature_creation",
                    "training_data_shape": st.session_state.builder.training_data.shape if hasattr(
                        st.session_state.builder, 'training_data') else None,
                    "testing_data_shape": st.session_state.builder.testing_data.shape if hasattr(
                        st.session_state.builder, 'testing_data') else None
                }
            )

            # Check if training and testing data are available
            if not hasattr(st.session_state.builder, 'training_data') or not hasattr(st.session_state.builder,
                                                                                     'testing_data'):
                st.error(
                    "Training and testing data must be created before feature creation. Please go back to the Train Test Split step.")
            else:
                # Create and render the FeatureCreationComponent
                feature_creation = FeatureCreationComponent(
                    st.session_state.builder,
                    st.session_state.logger,
                    st.session_state.builder.training_data,
                    st.session_state.builder.testing_data,
                    st.session_state.builder.target_column
                )
                feature_creation.render()

            # Add navigation buttons
            st.markdown("---")  # Add a visual separator
            nav_cols = st.columns([1, 2, 1])

            with nav_cols[1]:
                st.markdown(
                    """
                    <style>
                    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
                        width: 100%;
                        background-color: #f0f2f6;
                        border: 1px solid #e0e0e0;
                    }
                    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
                        width: 100%;
                        background-color: #FF4B4B;
                        color: white;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                nav_button_cols = st.columns([1, 1])
                with nav_button_cols[0]:
                    if step_info['prev']:
                        if st.button(f"← Back to {step_sequence[step_info['prev']]['label']}",
                                     width='stretch'):
                            st.session_state.preprocessing_step = step_info['prev']
                            st.rerun()

                with nav_button_cols[1]:
                    if step_info['next']:
                        if st.button(f"Continue to {step_sequence[step_info['next']]['label']} →",
                                     width='stretch', on_click=fc_scroll):
                            # Log stage transition
                            st.session_state.logger.log_stage_transition(
                                step_sequence[current_step]['label'],
                                step_sequence[step_info['next']]['label']
                            )
                            st.session_state.skipped_feature_creation = False
                            st.session_state.preprocessing_step = step_info['next']
                            st.session_state.scroll_to_top = True
                            st.rerun()

        elif st.session_state.preprocessing_step == 'data_types':
            st.session_state.logger.log_page_state(
                "Data Preprocessing - Data Types Review",
                {
                    "current_step": "data_types",
                    "dataset_shape": st.session_state.builder.training_data.shape,
                    "numeric_columns_count": len([col for col in st.session_state.builder.training_data.select_dtypes(include=[np.number]).columns if col != st.session_state.builder.target_column]),
                    "categorical_columns_count": len([col for col in st.session_state.builder.training_data.select_dtypes(include=['object', 'category']).columns if col != st.session_state.builder.target_column])
                }
            )
            
            # Create and render the data types optimisation component
            data_types_component = DataTypesOptimisationComponent(st.session_state.builder, st.session_state.logger)
            data_types_component.render()
            
            # Add navigation buttons
            st.markdown("---")  # Add a visual separator
            nav_cols = st.columns([1, 2, 1])
            
            with nav_cols[1]:
                st.markdown(
                    """
                    <style>
                    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
                        width: 100%;
                        background-color: #f0f2f6;
                        border: 1px solid #e0e0e0;
                    }
                    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
                        width: 100%;
                        background-color: #FF4B4B;
                        color: white;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                nav_button_cols = st.columns([1, 1])
                with nav_button_cols[0]:
                    # Determine back target based on skip flag
                    back_target = 'feature_creation'
                    back_label = 'Feature Creation'
                    
                    if st.session_state.get('skipped_feature_creation', False):
                        back_target = 'categorical'
                        back_label = 'Categorical Features'
                        
                    if st.button(f"← Back to {back_label}", width='stretch'):
                        st.session_state.preprocessing_step = back_target
                        st.rerun()
                
                with nav_button_cols[1]:
                    if step_info['next']:
                        if st.button(f"Continue to {step_sequence[step_info['next']]['label']} →", width='stretch', on_click=scroll):
                            # Log stage transition
                            st.session_state.logger.log_stage_transition(
                                step_sequence[current_step]['label'],
                                step_sequence[step_info['next']]['label']
                            )
                            st.session_state.preprocessing_step = step_info['next']
                            st.session_state.scroll_to_top = True
                            st.rerun()
        
        elif st.session_state.preprocessing_step == 'final':
            st.session_state.logger.log_page_state(
                "Data Preprocessing - Final Dataset Review",
                {
                    "current_step": "final",
                    "dataset_shape": st.session_state.builder.training_data.shape,
                    "numeric_columns_count": len([col for col in st.session_state.builder.training_data.select_dtypes(include=[np.number]).columns if col != st.session_state.builder.target_column]),
                    "categorical_columns_count": len([col for col in st.session_state.builder.training_data.select_dtypes(include=['object', 'category']).columns if col != st.session_state.builder.target_column])
                }
            )
            
            # Use the new FinalDataReviewComponent
            final_review = FinalDataReviewComponent(st.session_state.builder, st.session_state.logger)
            final_review.display_final_review()
            
            # Navigation buttons
            st.markdown("---")
            nav_cols = st.columns([1, 2, 1])
            
            with nav_cols[1]:
                st.markdown(
                    """
                    <style>
                    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
                        width: 100%;
                        background-color: #f0f2f6;
                        border: 1px solid #e0e0e0;
                    }
                    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
                        width: 100%;
                        background-color: #FF4B4B;
                        color: white;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                nav_button_cols = st.columns([1, 1])
                with nav_button_cols[0]:
                    if step_info['prev']:
                        if st.button(f"← Back to {step_sequence[step_info['prev']]['label']}", width='stretch'):
                            st.session_state.preprocessing_step = step_info['prev']
                            st.rerun()
                
                with nav_button_cols[1]:
                    if st.button("Complete Preprocessing →", type="primary", width='stretch'):
                        # Get data quality warnings first for logging
                        warnings = final_review._display_data_quality_warnings()
                        
                        # Log final preprocessing state and metrics
                        final_state = {
                            "total_rows": len(st.session_state.builder.training_data),
                            "total_columns": len(st.session_state.builder.training_data.columns),
                            "data_types": st.session_state.builder.training_data.dtypes.astype(str).to_dict(),
                            "missing_values": int(st.session_state.builder.training_data.isnull().sum().sum()),
                            "warnings_count": len(warnings)
                        }
                        st.session_state.logger.log_stage_transition(
                            "DATA_PREPROCESSING",
                            "FEATURE_SELECTION"
                        )
                        st.session_state.logger.log_calculation("Final Preprocessing State", final_state)
                        
                        st.session_state.builder.stage_completion[ModelStage.DATA_PREPROCESSING] = True
                        
                        st.session_state.final_preprocessing_training_data = st.session_state.builder.training_data.copy()
                        
                        # Clean up session state and temporary backups (memory optimization)
                        cleanup_keys = [
                            'preprocessing_step', 'tts_original_data', 'tts_split_metadata',
                            'preprocessing_original_data', 'preprocessing_entry_stage',
                            'feature_management_ops_applied', 'feature_management_entry_data',
                            'zero_values_ops_applied', 'zero_values_entry_data',
                            'missing_values_ops_applied', 'missing_values_entry_data',
                            'feature_binning_ops_applied', 'feature_binning_entry_data',
                            'feature_creation_ops_applied', 'feature_creation_entry_data',
                            'outlier_detection_ops_applied', 'outlier_detection_entry_data',
                            'categorical_encoding_ops_applied', 'categorical_encoding_entry_data'
                        ]
                        for key in cleanup_keys:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        next_page = "4_Feature_Selection"
                        st.switch_page(f"pages/{next_page}.py")

            st.markdown("---")

    else:
        st.error("Please load data first.")
        st.session_state.logger.log_error(
            "Preprocessing Failed",
            {"reason": "No data loaded"}
        )

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

