"""
Advanced Automated Preprocessing Component

This component provides a sophisticated automated preprocessing pipeline that uses
the AutomatedPreprocessingComponent from utils/automated_preprocessing.py.

Key differences from the basic auto preprocessing:
- 10 preprocessing steps (vs 5 in basic version)
- KNN imputation support for missing values
- Optimal binning with optbinning library
- Complete feature creation with 7-step filtering
- Duplicate removal before and after preprocessing
- Final validation ensuring train/test consistency

Features:
- Initial duplicate removal (before split)
- Train-test split with stratification
- Missing values (KNN imputation with fallback)
- Feature binning (optimal binning)
- Outlier handling (training data only)
- Feature creation (with complete filtering)
- Categorical encoding (with mapping storage)
- Data types optimization (with synchronization)
- Final duplicate removal (after preprocessing)
- Final data validation
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from Builder import ModelStage
from utils.automated_preprocessing import AutomatedPreprocessingComponent
from utils.automated_feature_selection import AutomatedFeatureSelectionComponent
from utils.dataset_overview import DatasetOverviewComponent


def render_advanced_automated_preprocessing():
    """
    Render the advanced automated preprocessing interface.

    This provides a more sophisticated preprocessing pipeline than the basic
    auto preprocessing, using the complete AutomatedPreprocessingComponent.
    """
    st.markdown("---")
    st.write("### 🤖 Advanced Auto Data Preprocessing")

    # Check if preprocessing has already been done - do this FIRST before any UI
    # This ensures the navigation button always shows when preprocessing is complete
    if 'advanced_preprocessing_completed' in st.session_state and st.session_state.advanced_preprocessing_completed:
        # Check if feature selection was also completed
        fs_completed = st.session_state.get('advanced_feature_selection_completed', False)

        if fs_completed:
            st.info("✨ Advanced auto preprocessing AND feature selection have been completed. Ready for model selection!")
        else:
            st.info("✨ Advanced auto preprocessing has already been completed. The data is ready for feature selection.")

        # Show comprehensive dashboard
        if 'advanced_preprocessing_result' in st.session_state:
            result = st.session_state.advanced_preprocessing_result
            render_advanced_preprocessing_dashboard(
                builder=st.session_state.builder,
                include_feature_selection=fs_completed
            )

        # Dynamic navigation button based on feature selection completion
        st.write("---")

        # Determine next stage
        if fs_completed:
            next_stage_name = "Model Selection"
            next_page = "5_Model_Selection"
            next_stage_key = "MODEL_SELECTION"
        else:
            next_stage_name = "Feature Selection"
            next_page = "4_Feature_Selection"
            next_stage_key = "FEATURE_SELECTION"

        if st.button(f"Proceed to {next_stage_name}", type="primary", key="advanced_proceed_to_next_stage"):
            # Enhanced logging for transition
            st.session_state.logger.log_calculation(
                "Data Exploration Results",
                {
                    "final_training_shape": st.session_state.builder.training_data.shape,
                    "final_testing_shape": st.session_state.builder.testing_data.shape,
                    "advanced_preprocessing_completed": True,
                    "advanced_feature_selection_completed": fs_completed,
                    "feature_count": len(st.session_state.builder.training_data.columns) - 1
                }
            )

            # Get detailed information from preprocessing and feature selection results
            preprocessing_details = {}
            feature_selection_details = {}

            if 'advanced_preprocessing_result' in st.session_state:
                prep_result = st.session_state.advanced_preprocessing_result
                prep_details = prep_result.get('details', {})

                # Extract feature creation details
                for step in prep_details.get('steps_completed', []):
                    if step['step'] == 'Feature Creation':
                        step_details = step.get('details', {})
                        added_features = step_details.get('added_features_list', [])
                        preprocessing_details['features_created'] = added_features
                        preprocessing_details['features_created_count'] = len(added_features)

                    elif step['step'] == 'Missing Values Handling':
                        step_details = step.get('details', {})
                        preprocessing_details['missing_values_strategies'] = step_details.get('strategies_applied', {})

                    elif step['step'] == 'Categorical Encoding':
                        step_details = step.get('details', {})
                        preprocessing_details['encoding_methods'] = step_details.get('encoding_methods', {})

                    elif step['step'] == 'Outlier Handling':
                        step_details = step.get('details', {})
                        preprocessing_details['outlier_methods'] = {
                            oh['feature']: oh['strategy']
                            for oh in step_details.get('outliers_handled', [])
                        }

                    elif step['step'] == 'Feature Binning':
                        step_details = step.get('details', {})
                        preprocessing_details['binned_features'] = step_details.get('modified_columns', [])

            if fs_completed and 'advanced_auto_feature_selection_result' in st.session_state:
                fs_result = st.session_state.advanced_auto_feature_selection_result
                fs_details = fs_result.get('details', {})
                feature_selection_details['features_removed'] = fs_details.get('features_removed', [])
                feature_selection_details['features_removed_count'] = len(fs_details.get('features_removed', []))
                feature_selection_details['low_importance_removed'] = fs_details.get('low_importance_removed', [])
                feature_selection_details['correlation_removed'] = fs_details.get('correlation_removed', {})
                feature_selection_details['boruta_applied'] = fs_details.get('boruta_applied', False)

            st.session_state.logger.log_journey_point(
                stage="DATA_EXPLORATION",
                decision_type=f"ADVANCED_AUTO_PIPELINE_TO_{next_stage_key}",
                description=f"Advanced auto pipeline completed, proceeding to {next_stage_name}",
                details={
                    'Final Training Shape': st.session_state.builder.training_data.shape,
                    'Final Testing Shape': st.session_state.builder.testing_data.shape,
                    'Preprocessing Completed': True,
                    'Feature Selection Completed': fs_completed,
                    'Feature Count': len(st.session_state.builder.training_data.columns) - 1,
                    'Preprocessing Details': preprocessing_details,
                    'Feature Selection Details': feature_selection_details
                },
                parent_id=None
            )

            st.session_state.logger.log_stage_transition(
                "DATA_EXPLORATION",
                next_stage_key,
                {
                    "advanced_preprocessing_completed": True,
                    "advanced_feature_selection_completed": fs_completed,
                    "training_rows": len(st.session_state.builder.training_data),
                    "testing_rows": len(st.session_state.builder.testing_data),
                    "features_count": len(st.session_state.builder.training_data.columns) - 1
                }
            )

            st.session_state.logger.log_user_action(
                "Navigation",
                {"direction": "forward", "to_stage": next_stage_key}
            )
            st.switch_page(f"pages/{next_page}.py")
        return

    # Add explanation expander
    with st.expander("ℹ️ Understanding Advanced Auto Preprocessing", expanded=False):
        # Get explanation from content manager
        from content.content_manager import ContentManager
        content_manager = ContentManager()
        st.markdown(content_manager.get_advanced_auto_preprocessing_explanation())

    # Configuration options
    st.write("#### Configuration")

    include_automated_feature_selection = False
    auto_select_top_features = 10
    show_analysis = False
    use_boruta = False
    boruta_threshold = 10

    # Add the toggle switches
    advanced_auto_preprocess_enabled = st.toggle(
        "Enable Advanced Auto Preprocessing",
        help="Automatically preprocess your dataset using the comprehensive 10-step pipeline",
        key="advanced_auto_preprocess_toggle"
    )

    if advanced_auto_preprocess_enabled:
        include_automated_feature_selection = st.toggle(
            "Include Automated Feature Selection",
            value=False,
            help="Automatically run feature selection after preprocessing (removes low importance and correlated features)",
            key="advanced_auto_include_feature_selection"
        )

    col1, col2 = st.columns(2)

    with col1:
        if advanced_auto_preprocess_enabled:
            use_feature_creation = st.checkbox(
                "Enable Feature Creation",
                value=False,
                help="Use feature creation to create new features from existing data",
                key="advanced_auto_use_feature_creation"
            )
            if use_feature_creation:
                auto_select_top_features = st.slider(
                    "Number of engineered features to create",
                    min_value=0,
                    max_value=20,
                    value=10,
                    help="Number of top-scoring engineered features to automatically add (0 to skip feature creation)",
                    key="advanced_auto_select_top_features"
                )
            else:
                auto_select_top_features = 0

            show_analysis = st.checkbox(
                "Show detailed step-by-step analysis",
                value=False,
                help="Display detailed information about each preprocessing step",
                key="advanced_auto_show_analysis"
            )

    with col2:
        if include_automated_feature_selection:
            use_boruta = st.checkbox(
                "Enable Boruta Algorithm",
                value=False,
                help="Use Boruta algorithm for advanced feature selection (conservative approach)",
                key="advanced_auto_use_boruta"
            )

            if use_boruta:
                boruta_threshold = st.number_input(
                    "Boruta Threshold (min features)",
                    min_value=5,
                    max_value=50,
                    value=10,
                    help="Minimum number of features required to run Boruta. Boruta will be skipped if fewer features remain.",
                    key="advanced_auto_boruta_threshold"
                )

    st.session_state.advanced_auto_preprocess_enabled = advanced_auto_preprocess_enabled

    # Log user preference
    st.session_state.logger.log_user_action(
        "Advanced Auto Preprocessing Preference",
        {
            "advanced_auto_preprocess_enabled": bool(advanced_auto_preprocess_enabled),
            "auto_select_top_features": auto_select_top_features,
            "show_analysis": show_analysis,
            "include_automated_feature_selection": include_automated_feature_selection,
            "use_boruta": use_boruta,
            "boruta_threshold": boruta_threshold,
            "timestamp": datetime.now().isoformat()
        }
    )

    if advanced_auto_preprocess_enabled:
        st.session_state.logger.log_journey_point(
            stage="DATA_EXPLORATION",
            decision_type="ADVANCED_AUTO_PREPROCESSING_ENABLED",
            description="Advanced auto preprocessing enabled",
            details={
                'advanced_auto_preprocess_enabled': advanced_auto_preprocess_enabled,
                'auto_select_top_features': auto_select_top_features
            },
            parent_id=None
        )

        # Validate initial state
        if st.session_state.builder.data is None:
            st.error("❌ No data loaded. Please load data first.")
            return

        if st.session_state.builder.target_column is None:
            st.error("❌ No target column set. Please set target column first.")
            return

        # ID/Index Column Detection and Removal
        st.write("---")
        st.write("### 🔍 ID/Index Column Detection")
        st.info("""
        Before preprocessing, please identify any ID or index columns that should be removed.
        These might be:
        - Row ID columns
        - Record identifiers
        - Timestamp columns (if not relevant for prediction)
        - Other unique identifiers

        You can also remove any columns that are not relevant for prediction.
        """)

        # Get current data
        original_data = st.session_state.builder.data.copy()

        # Get all columns except target
        available_columns = [col for col in original_data.columns
                          if col != st.session_state.builder.target_column]

        # Try to automatically identify potential ID columns
        potential_id_cols = []
        for col in available_columns:
            # Check if column name contains common ID indicators
            if any(id_term in col.lower() for id_term in ['id', 'index', 'key', 'num', 'code', 'uuid']):
                potential_id_cols.append(col)
            # Check if column has unique values
            elif original_data[col].nunique() == len(original_data):
                potential_id_cols.append(col)

        # Create multiselect with potential ID columns pre-selected
        selected_id_columns = st.multiselect(
            "Select columns to remove (ID/index columns)",
            options=available_columns,
            default=potential_id_cols,
            help="Select any columns that are identifiers and should not be used for modeling",
            key="advanced_auto_id_columns"
        )

        # Log user selection of ID columns
        if selected_id_columns:
            st.session_state.logger.log_user_action(
                "Advanced Auto - ID Columns Selected",
                {
                    "selected_id_columns": selected_id_columns,
                    "count": len(selected_id_columns),
                    "auto_detected_columns": potential_id_cols,
                    "timestamp": datetime.now().isoformat()
                }
            )

        # Show what will be removed
        if selected_id_columns:
            st.write(f"🗑️ Will remove {len(selected_id_columns)} ID/index columns:")
            for col in selected_id_columns:
                st.write(f"- {col}")

        # Add confirmation button
        st.write("---")
        st.write("#### Ready to Start?")
        if selected_id_columns:
            st.info(f"""
            Click the button below to start the advanced automated preprocessing pipeline.
            This will first remove {len(selected_id_columns)} ID/index columns, then execute all preprocessing steps automatically.
            """)
        else:
            st.info("""
            Click the button below to start the advanced automated preprocessing pipeline.
            This will execute all preprocessing steps automatically.
            """)

        if st.button("🚀 Run Advanced Auto Preprocessing", type="primary"):
            try:
                with st.spinner("🔄 Running advanced automated preprocessing pipeline..."):
                    # Remove selected ID columns before preprocessing
                    if selected_id_columns:
                        st.write(f"🗑️ Removing {len(selected_id_columns)} ID/index columns...")
                        st.session_state.builder.data = st.session_state.builder.data.drop(columns=selected_id_columns)

                        # Log column removal
                        st.session_state.logger.log_calculation(
                            "Advanced Auto - ID Columns Removed",
                            {
                                "columns_removed": selected_id_columns,
                                "count": len(selected_id_columns),
                                "remaining_columns": len(st.session_state.builder.data.columns),
                                "timestamp": datetime.now().isoformat()
                            }
                        )

                        st.success(f"✅ Removed {len(selected_id_columns)} ID/index columns")

                    # Create the automated preprocessing component
                    auto_prep = AutomatedPreprocessingComponent(
                        builder=st.session_state.builder,
                        logger=st.session_state.logger,
                        auto_select_top_features=auto_select_top_features if auto_select_top_features > 0 else 0,
                        show_analysis=show_analysis,
                        skip_feature_creation=(auto_select_top_features == 0)
                    )

                    # Run the preprocessing pipeline
                    result = auto_prep.run()

                    # Store result in session state
                    st.session_state.advanced_preprocessing_result = result

                    if result['success']:
                        # Mark preprocessing as complete
                        st.session_state.advanced_preprocessing_completed = True

                        # Mark stages as complete
                        st.session_state.builder.stage_completion[ModelStage.DATA_PREPROCESSING] = True
                        st.session_state.builder.stage_completion[ModelStage.DATA_EXPLORATION] = True

                        # Mark component completions
                        st.session_state.missing_values_complete = True
                        st.session_state.categorical_encoding_complete = True

                        # CRITICAL: Store final preprocessing data for What-If Analysis
                        # This ensures auxiliary features can be found for calculated fields
                        st.session_state.final_preprocessing_training_data = st.session_state.builder.training_data.copy()

                        st.success("✅ Advanced automated preprocessing completed successfully!")

                        # Log completion
                        details = result.get('details', {})
                        st.session_state.logger.log_calculation(
                            "Advanced Auto Preprocessing Completed",
                            {
                                "training_shape": st.session_state.builder.training_data.shape,
                                "testing_shape": st.session_state.builder.testing_data.shape,
                                "steps_completed": len(details.get('steps_completed', [])),
                                "steps_failed": len(details.get('steps_failed', [])),
                                "timestamp": datetime.now().isoformat()
                            }
                        )

                        # Extract detailed information for journey log
                        preprocessing_summary = {
                            'training_shape': st.session_state.builder.training_data.shape,
                            'testing_shape': st.session_state.builder.testing_data.shape,
                            'steps_completed': len(details.get('steps_completed', []))
                        }

                        # Add details from each step
                        for step in details.get('steps_completed', []):
                            step_name = step['step']
                            step_details = step.get('details', {})

                            if step_name == 'Feature Creation':
                                added_features = step_details.get('added_features_list', [])
                                preprocessing_summary['features_created'] = added_features
                                preprocessing_summary['features_created_count'] = len(added_features)

                            elif step_name == 'Missing Values Handling':
                                preprocessing_summary['missing_values_strategies'] = step_details.get('strategies_applied', {})
                                preprocessing_summary['missing_values_columns_processed'] = step_details.get('columns_processed', 0)

                            elif step_name == 'Categorical Encoding':
                                encoding_methods = step_details.get('encoding_methods', {})
                                preprocessing_summary['categorical_encoding'] = {
                                    col: info['method'] for col, info in encoding_methods.items()
                                }
                                preprocessing_summary['categorical_encoding_count'] = len(encoding_methods)

                            elif step_name == 'Outlier Handling':
                                outlier_details = {
                                    oh['feature']: {
                                        'strategy': oh['strategy'],
                                        'outliers_removed': oh['outliers_count']
                                    }
                                    for oh in step_details.get('outliers_handled', [])
                                }
                                preprocessing_summary['outlier_handling'] = outlier_details
                                preprocessing_summary['total_outliers_removed'] = step_details.get('total_outliers_processed', 0)

                            elif step_name == 'Feature Binning':
                                preprocessing_summary['binned_features'] = step_details.get('modified_columns', [])
                                preprocessing_summary['binned_features_count'] = len(step_details.get('modified_columns', []))

                            elif step_name == 'Initial Duplicate Removal':
                                preprocessing_summary['initial_duplicates_removed'] = step_details.get('rows_removed', 0)

                            elif step_name == 'Final Duplicate Removal':
                                preprocessing_summary['final_duplicates_removed'] = step_details.get('total_rows_removed', 0)

                        st.session_state.logger.log_journey_point(
                            stage="DATA_EXPLORATION",
                            decision_type="ADVANCED_AUTO_PREPROCESSING_COMPLETED",
                            description="Advanced auto preprocessing completed successfully",
                            details=preprocessing_summary,
                            parent_id=None
                        )

                        # Run automated feature selection if enabled
                        if include_automated_feature_selection:
                            try:
                                st.write("---")
                                st.write("### 🎯 Running Automated Feature Selection")

                                with st.spinner("🔄 Running automated feature selection pipeline..."):
                                    # Create the automated feature selection component
                                    auto_fs = AutomatedFeatureSelectionComponent(
                                        builder=st.session_state.builder,
                                        logger=st.session_state.logger,
                                        show_analysis=show_analysis,
                                        use_boruta=use_boruta,
                                        boruta_threshold=boruta_threshold
                                    )

                                    # Run feature selection
                                    fs_result = auto_fs.run()

                                    # Store result in session state
                                    st.session_state.advanced_auto_feature_selection_result = fs_result

                                    if fs_result['success']:
                                        # Mark feature selection as complete
                                        st.session_state.advanced_feature_selection_completed = True

                                        # Mark feature selection stage as complete
                                        st.session_state.builder.stage_completion[ModelStage.FEATURE_SELECTION] = True

                                        st.success("✅ Automated feature selection completed successfully!")

                                        # Log feature selection completion
                                        fs_details = fs_result.get('details', {})
                                        st.session_state.logger.log_calculation(
                                            "Advanced Auto Feature Selection Completed",
                                            {
                                                "initial_features": fs_details.get('initial_features', 0),
                                                "final_features": fs_details.get('final_features', 0),
                                                "features_removed": len(fs_details.get('features_removed', [])),
                                                "boruta_applied": fs_details.get('boruta_applied', False),
                                                "timestamp": datetime.now().isoformat()
                                            }
                                        )

                                        # Extract detailed information for journey log
                                        feature_selection_summary = {
                                            'initial_features': fs_details.get('initial_features', 0),
                                            'final_features': fs_details.get('final_features', 0),
                                            'features_removed': fs_details.get('features_removed', []),
                                            'features_removed_count': len(fs_details.get('features_removed', []))
                                        }

                                        # Add removal details
                                        if 'low_importance_removed' in fs_details:
                                            feature_selection_summary['low_importance_removed'] = fs_details['low_importance_removed']
                                            feature_selection_summary['low_importance_removed_count'] = len(fs_details['low_importance_removed'])

                                        if 'correlation_removed' in fs_details:
                                            correlation_removed = fs_details['correlation_removed']
                                            # Flatten correlation groups
                                            all_corr_removed = []
                                            for group_features in correlation_removed.values():
                                                all_corr_removed.extend(group_features)
                                            feature_selection_summary['correlation_removed'] = all_corr_removed
                                            feature_selection_summary['correlation_removed_count'] = len(all_corr_removed)
                                            feature_selection_summary['correlation_groups'] = correlation_removed

                                        if fs_details.get('boruta_applied', False):
                                            feature_selection_summary['boruta_applied'] = True
                                            # Extract Boruta details
                                            for step in fs_details.get('steps_completed', []):
                                                if 'Boruta' in step.get('step', ''):
                                                    boruta_details = step.get('details', {})
                                                    feature_selection_summary['boruta_removed'] = boruta_details.get('removed_features_list', [])
                                                    feature_selection_summary['boruta_removed_count'] = boruta_details.get('features_removed', 0)
                                                    feature_selection_summary['boruta_confirmed'] = boruta_details.get('confirmed_features', [])
                                                    break

                                        if 'protected_attributes' in fs_details:
                                            feature_selection_summary['protected_attributes'] = fs_details['protected_attributes']

                                        if 'removed_features_stats' in fs_details:
                                            feature_selection_summary['removed_features_stats'] = fs_details['removed_features_stats']

                                        st.session_state.logger.log_journey_point(
                                            stage="DATA_EXPLORATION",
                                            decision_type="ADVANCED_AUTO_FEATURE_SELECTION_COMPLETED",
                                            description="Advanced auto feature selection completed successfully",
                                            details=feature_selection_summary,
                                            parent_id=None
                                        )

                                    else:
                                        # Feature selection failed
                                        st.error(f"❌ Automated feature selection failed: {fs_result.get('summary', 'Unknown error')}")
                                        st.warning("⚠️ You can proceed to feature selection page to complete this step manually.")

                                        # Mark feature selection as not completed
                                        st.session_state.advanced_feature_selection_completed = False

                                        # Show errors
                                        fs_errors = fs_result.get('errors', [])
                                        if fs_errors:
                                            with st.expander("View Error Details"):
                                                for error in fs_errors:
                                                    if isinstance(error, dict):
                                                        st.code(error.get('traceback', error.get('error', str(error))))
                                                    else:
                                                        st.code(str(error))

                                        # Log failure
                                        st.session_state.logger.log_error(
                                            "Advanced Auto Feature Selection Failed",
                                            {
                                                "error": fs_result.get('summary', 'Unknown error'),
                                                "errors": fs_errors,
                                                "timestamp": datetime.now().isoformat()
                                            }
                                        )

                            except Exception as e:
                                st.error(f"❌ Error during automated feature selection: {str(e)}")
                                st.warning("⚠️ Preprocessing was successful. You can proceed to feature selection page to complete this step manually.")

                                # Mark feature selection as not completed
                                st.session_state.advanced_feature_selection_completed = False

                                # Show detailed error
                                import traceback
                                with st.expander("View Error Details"):
                                    st.code(traceback.format_exc())

                                # Log error
                                st.session_state.logger.log_error(
                                    "Advanced Auto Feature Selection Error",
                                    {
                                        "error": str(e),
                                        "error_type": type(e).__name__,
                                        "traceback": traceback.format_exc(),
                                        "timestamp": datetime.now().isoformat()
                                    }
                                )

                        # Rerun to display the appropriate "Proceed to..." button at the top
                        st.rerun()

                    else:
                        # Preprocessing failed
                        st.error(f"❌ Advanced automated preprocessing failed: {result.get('summary', 'Unknown error')}")

                        # Show errors
                        errors = result.get('errors', [])
                        if errors:
                            with st.expander("View Error Details"):
                                for error in errors:
                                    if isinstance(error, dict):
                                        st.code(error.get('traceback', error.get('error', str(error))))
                                    else:
                                        st.code(str(error))

                        # Log failure
                        st.session_state.logger.log_error(
                            "Advanced Auto Preprocessing Failed",
                            {
                                "error": result.get('summary', 'Unknown error'),
                                "errors": errors,
                                "timestamp": datetime.now().isoformat()
                            }
                        )

            except Exception as e:
                st.error(f"❌ Error during advanced auto preprocessing: {str(e)}")

                # Show detailed error
                import traceback
                with st.expander("View Error Details"):
                    st.code(traceback.format_exc())

                # Log error
                st.session_state.logger.log_error(
                    "Advanced Auto Preprocessing Error",
                    {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now().isoformat()
                    }
                )


def render_advanced_preprocessing_dashboard(builder, include_feature_selection=False):
    """
    Render a comprehensive preprocessing dashboard showing all key results.

    This dashboard displays:
    - Overall summary metrics
    - Processing duration and steps completed
    - Data transformation statistics
    - Key preprocessing actions (binning, encoding, outliers, etc.)
    - Feature selection results (if applicable)
    - Download buttons for detailed reports

    Args:
        builder: The Builder instance
        include_feature_selection: Whether to display feature selection results
    """
    st.write("## 📊 Pipeline Dashboard")

    # Get the preprocessing result from session state
    if 'advanced_preprocessing_result' not in st.session_state:
        st.error("Preprocessing results not available.")
        return

    result = st.session_state.advanced_preprocessing_result
    details = result.get('details', {})

    # Calculate key metrics
    duration = (details['end_time'] - details['start_time']).total_seconds()
    steps_completed = len(details.get('steps_completed', []))
    steps_failed = len(details.get('steps_failed', []))
    initial_shape = details.get('initial_shape', (0, 0))
    final_shape = details.get('final_shape', (0, 0))

    # Top-level metrics in columns
    st.markdown("### 📈 Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Processing Time",
            value=f"{duration:.1f}s",
            help="Total time taken for all preprocessing steps"
        )

    with col2:
        st.metric(
            label="Steps Completed",
            value=f"{steps_completed}/{steps_completed + steps_failed}",
            help="Number of preprocessing steps successfully completed"
        )

    with col3:
        st.metric(
            label="Final Features",
            value=final_shape[1] - 1 if final_shape else 0,
            delta=f"{(final_shape[1] - 1 if final_shape else 0) - (initial_shape[1] - 1 if initial_shape else 0)}",
            help="Number of features after preprocessing (excluding target)"
        )

    with col4:
        st.metric(
            label="Final Training Rows",
            value=final_shape[0] if final_shape else 0,
            delta=f"{(final_shape[0] if final_shape else 0) - (initial_shape[0] if initial_shape else 0)}",
            help="Number of rows in training set after preprocessing"
        )

    # Data transformation overview
    st.markdown("---")
    st.markdown("### 🔄 Data Transformations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Shape Changes")
        st.write(f"**Initial Shape:** {initial_shape[0]:,} rows × {initial_shape[1]:,} columns")
        st.write(f"**Final Training Shape:** {final_shape[0]:,} rows × {final_shape[1]:,} columns")

        # Get testing shape from preprocessing result to show post-preprocessing state
        # (not current builder state which may reflect post-feature-selection)
        test_shape_info = details.get('steps_completed', [])
        test_final_shape = None
        for step in test_shape_info:
            if step['step'] == 'Final Data Validation':
                test_final_shape = step['details'].get('test_shape', None)
                break

        if test_final_shape:
            st.write(f"**Final Testing Shape:** {test_final_shape[0]:,} rows × {test_final_shape[1]:,} columns")
        elif builder.testing_data is not None:
            # Fallback to builder.testing_data if validation step not found
            st.write(f"**Final Testing Shape:** {builder.testing_data.shape[0]:,} rows × {builder.testing_data.shape[1]:,} columns")

    with col2:
        st.markdown("#### Key Outputs")
        st.write(f"**Encoding Mappings Stored:** {len(details.get('encoding_mappings', {}))}")
        st.write(f"**Features Binned:** {len(details.get('bin_ranges', {}))}")
        st.write(f"**Outlier Strategies Applied:** {len(details.get('outlier_strategies', {}))}")

    # Detailed step breakdown in expanders
    st.markdown("---")
    st.markdown("### 📋 Preprocessing Steps Breakdown")

    # Group steps by category for better organization
    for step_info in details.get('steps_completed', []):
        step_name = step_info['step']
        step_details = step_info.get('details', {})

        with st.expander(f"✅ {step_name}", expanded=False):
            # Display step-specific information
            if step_name == "Train-Test Split":
                st.write(f"- **Training samples:** {step_details.get('training_shape', (0,))[0]:,}")
                st.write(f"- **Testing samples:** {step_details.get('testing_shape', (0,))[0]:,}")
                st.write(f"- **Split method:** {step_details.get('split_method', 'N/A').title()}")
                st.write(f"- **Test size:** {step_details.get('test_size', 0):.1%}")

            elif step_name == "Missing Values Handling":
                # Calculate initial missing values from strategies_applied dict
                strategies = step_details.get('strategies_applied', {})
                initial_missing_cols = len(strategies)

                st.write(f"- **Columns with missing values:** {initial_missing_cols}")
                st.write(f"- **Columns processed:** {step_details.get('columns_processed', 0)}")
                st.write(f"- **Remaining missing values:** {step_details.get('remaining_missing', 0)}")

                # Show strategies applied to each feature
                if strategies:
                    st.write("- **Strategies applied by feature:**")
                    strategy_name_map = {
                        'drop_column': 'Drop Column (>50% missing)',
                        'drop_rows': 'Drop Rows (<5% missing)',
                        'knn': 'KNN Imputation',
                        'median': 'Median Imputation',
                        'mode': 'Mode Imputation',
                        'mean': 'Mean Imputation'
                    }
                    for col, strategy in sorted(strategies.items()):
                        strategy_display = strategy_name_map.get(strategy, strategy)
                        st.write(f"  - **`{col}`**: {strategy_display}")

            elif step_name == "Feature Binning":
                st.write(f"- **Features analyzed:** {step_details.get('features_analyzed', 0)}")
                st.write(f"- **Features binned:** {step_details.get('features_binned', 0)}")

                modified_cols = step_details.get('modified_columns', [])
                bin_ranges = step_details.get('bin_ranges', {})

                if modified_cols:
                    st.write("- **Binned features with ranges:**")
                    for col in modified_cols:
                        st.write(f"  - **`{col}`**:")
                        if col in bin_ranges:
                            ranges = bin_ranges[col]
                            if isinstance(ranges, list):
                                # Numerical binning - format as clean ranges
                                for i, bin_range in enumerate(ranges):
                                    from utils.automated_preprocessing import _format_bin_range
                                    formatted_range = _format_bin_range(bin_range)
                                    st.write(f"    - Bin {i}: {formatted_range}")
                            elif isinstance(ranges, dict):
                                # Categorical binning - show category mappings
                                for bin_id, categories in ranges.items():
                                    cat_str = ', '.join(map(str, categories))
                                    st.write(f"    - {bin_id}: {cat_str}")
                            else:
                                st.write(f"    - {ranges}")

            elif step_name == "Outlier Handling":
                st.write(f"- **Features analyzed:** {step_details.get('features_analyzed', 0)}")
                st.write(f"- **Features with outliers handled:** {step_details.get('features_handled', 0)}")
                st.write(f"- **Total outliers processed:** {step_details.get('total_outliers_processed', 0)}")

                # Only show features where outliers were actually handled (count > 0)
                outliers_handled = step_details.get('outliers_handled', [])
                # Filter to only show features with actual outliers removed
                outliers_with_action = [oh for oh in outliers_handled if oh.get('outliers_count', 0) > 0]

                if outliers_with_action:
                    st.write("- **Outlier handling applied:**")
                    strategy_desc_map = {
                        'Remove': 'IQR method (1.5×IQR)',
                        'Remove Extreme': 'Extended IQR method (3×IQR)',
                        'Cap': 'Winsorization (capping)',
                        'Isolation Forest': 'ML-based detection'
                    }
                    for oh in outliers_with_action:
                        strategy_desc = strategy_desc_map.get(oh['strategy'], oh['strategy'])
                        st.write(f"  - **`{oh['feature']}`**: {strategy_desc}")
                        st.write(f"    - Outliers removed/capped: {oh['outliers_count']}")
                elif outliers_handled:
                    st.info("No outliers were found that required removal/capping")

            elif step_name == "Feature Creation":
                st.write(f"- **Combinations generated:** {step_details.get('features_generated', 0)}")
                st.write(f"- **Features after filtering:** {step_details.get('features_analyzed', 0)}")
                st.write(f"- **Features added:** {step_details.get('features_added', 0)}")

                added_features = step_details.get('added_features_list', [])
                if added_features:
                    st.write(f"- **All engineered features added ({len(added_features)}):**")
                    # Show all features, not just top 5
                    for feat in added_features:
                        st.write(f"  - `{feat}`")

            elif step_name == "Categorical Encoding":
                st.write(f"- **Columns encoded:** {step_details.get('columns_encoded', 0)}")
                st.write(f"- **Encoding mappings stored:** {step_details.get('mappings_stored', 0)}")

                encoding_methods = step_details.get('encoding_methods', {})
                if encoding_methods:
                    st.write("- **Encoding applied by feature:**")
                    method_name_map = {
                        'label': 'Label Encoding (ordinal)',
                        'onehot': 'One-Hot Encoding',
                        'target': 'Target Encoding (mean encoding)',
                        'drop_column': 'Drop Column'
                    }

                    # Show each feature with its encoding method and mapping
                    for col, col_info in sorted(encoding_methods.items()):
                        method = col_info['method']
                        method_display = method_name_map.get(method, method)
                        st.write(f"  - **`{col}`**: {method_display}")

                        # Show clean mapping if available
                        if col in details.get('encoding_mappings', {}):
                            mapping_info = details['encoding_mappings'][col]
                            if isinstance(mapping_info, dict):
                                # Extract clean mapping based on encoding type
                                if mapping_info.get('method') == 'Label Encoding':
                                    # Show label encoding mapping
                                    if 'mapping' in mapping_info:
                                        clean_mapping = {}
                                        for key, val in mapping_info['mapping'].items():
                                            # Convert numpy types to Python types
                                            clean_val = int(val) if hasattr(val, 'item') else val
                                            clean_mapping[key] = clean_val
                                        st.write(f"    - Mapping: {clean_mapping}")
                                elif mapping_info.get('method') == 'One-Hot Encoding':
                                    # Show one-hot encoding columns created
                                    if 'new_columns' in mapping_info:
                                        st.write(f"    - New columns: {', '.join(mapping_info['new_columns'])}")
                                elif mapping_info.get('method') == 'Target Encoding':
                                    # Show target encoding info
                                    if 'mapping' in mapping_info and len(mapping_info['mapping']) <= 20:
                                        clean_mapping = {}
                                        for key, val in mapping_info['mapping'].items():
                                            clean_val = float(val) if hasattr(val, 'item') else val
                                            clean_mapping[key] = f"{clean_val:.4f}"
                                        st.write(f"    - Target means: {clean_mapping}")

            elif step_name == "Data Types Optimization":
                st.write(f"- **Type conversions:** {step_details.get('conversions_applied', 0)}")
                memory_saved = step_details.get('memory_saved_mb', 0)
                memory_reduction_pct = step_details.get('memory_reduction_pct', 0)
                st.write(f"- **Memory saved:** {memory_saved:.2f} MB ({memory_reduction_pct:.1f}%)")

            elif "Duplicate Removal" in step_name:
                action = step_details.get('action', '')
                if action == 'skipped':
                    st.write("- No duplicates found")
                else:
                    # Show separate stats for training and testing if available
                    if 'train_exact_duplicates' in step_details:
                        # Final duplicate removal with separate train/test stats
                        st.write("**Training Data:**")
                        train_exact = step_details.get('train_exact_duplicates', 0)
                        train_partial = step_details.get('train_partial_duplicates', 0)
                        train_removed = step_details.get('train_rows_removed', 0)

                        if train_removed > 0:
                            st.write(f"  - Exact duplicates: {train_exact}")
                            st.write(f"  - Partial duplicate groups: {train_partial}")
                            st.write(f"  - Rows removed: {train_removed}")
                        else:
                            st.write(f"  - No duplicates found")

                        st.write("")
                        st.write("**Testing Data:**")
                        test_exact = step_details.get('test_exact_duplicates', 0)
                        test_partial = step_details.get('test_partial_duplicates', 0)
                        test_removed = step_details.get('test_rows_removed', 0)

                        if test_removed > 0:
                            st.write(f"  - Exact duplicates: {test_exact}")
                            st.write(f"  - Partial duplicate groups: {test_partial}")
                            st.write(f"  - Rows removed: {test_removed}")
                        else:
                            st.write(f"  - No duplicates found")

                        st.write("")
                        total_removed = step_details.get('total_rows_removed', 0)
                        st.write(f"**Total rows removed: {total_removed}**")
                    else:
                        # Initial duplicate removal (before split)
                        exact_dups = step_details.get('exact_duplicates_found', 0)
                        partial_dups = step_details.get('partial_duplicate_groups_found', 0)
                        rows_removed = step_details.get('rows_removed', 0)

                        st.write(f"- **Exact duplicates found:** {exact_dups}")
                        st.write(f"- **Partial duplicate groups:** {partial_dups}")
                        st.write(f"- **Total rows removed:** {rows_removed}")

            elif step_name == "Final Data Validation":
                st.write(f"- **Training shape:** {step_details.get('train_shape', 'N/A')}")
                st.write(f"- **Testing shape:** {step_details.get('test_shape', 'N/A')}")
                st.write(f"- **Number of features:** {step_details.get('num_features', 'N/A')}")
                st.write("- **All validation checks passed:** ✓")

            else:
                # Generic display for other steps
                for key, value in step_details.items():
                    if not isinstance(value, (dict, list)) or len(str(value)) < 100:
                        st.write(f"- **{key}:** {value}")

    # Feature Selection Results section (if applicable)
    if include_feature_selection and 'advanced_auto_feature_selection_result' in st.session_state:
        st.markdown("---")
        st.markdown("### 🎯 Feature Selection Results")

        fs_result = st.session_state.advanced_auto_feature_selection_result
        fs_details = fs_result.get('details', {})

        if fs_result.get('success', False):
            # Feature selection summary metrics
            fs_col1, fs_col2, fs_col3, fs_col4 = st.columns(4)

            initial_features = fs_details.get('initial_features', 0)
            final_features = fs_details.get('final_features', 0)
            features_removed = len(fs_details.get('features_removed', []))
            reduction_pct = (features_removed / initial_features * 100) if initial_features > 0 else 0

            with fs_col1:
                st.metric(
                    label="Initial Features",
                    value=initial_features,
                    help="Number of features before feature selection"
                )

            with fs_col2:
                st.metric(
                    label="Final Features",
                    value=final_features,
                    delta=f"-{features_removed}",
                    delta_color="normal",
                    help="Number of features after feature selection"
                )

            with fs_col3:
                st.metric(
                    label="Reduction",
                    value=f"{reduction_pct:.1f}%",
                    help="Percentage of features removed"
                )

            with fs_col4:
                boruta_applied = fs_details.get('boruta_applied', False)
                st.metric(
                    label="Boruta Applied",
                    value="Yes" if boruta_applied else "No",
                    help="Whether Boruta algorithm was used"
                )

            # Feature selection steps breakdown
            st.markdown("#### Steps Completed")
            steps_completed = fs_details.get('steps_completed', [])

            for step_info in steps_completed:
                step_name = step_info.get('step', 'Unknown Step')
                step_details = step_info.get('details', {})

                with st.expander(f"✅ {step_name}", expanded=False):
                    # Display step-specific information
                    if "Feature Importance Analysis" in step_name:
                        st.write(f"- **Total features:** {step_details.get('total_features', 0)}")
                        st.write(f"- **Low importance features:** {step_details.get('low_importance_count', 0)}")
                        st.write(f"- **Correlation pairs:** {step_details.get('correlation_pairs', 0)}")
                        st.write(f"- **Protected attributes:** {step_details.get('protected_attributes_count', 0)}")

                    elif "Low Importance" in step_name:
                        removed_features = step_details.get('removed_features_list', [])
                        if removed_features:
                            st.write(f"- **Features removed:** {len(removed_features)}")
                            st.write(f"- **Remaining features:** {step_details.get('remaining_features', 0)}")

                            # Show detailed removal information if available
                            fs_result = st.session_state.get('advanced_auto_feature_selection_result', {})
                            fs_summary = fs_result.get('details', {})
                            low_imp_details = fs_summary.get('low_importance_details', {})

                            if low_imp_details:
                                st.write("- **Detailed Removal Information:**")
                                # Create a table with feature, importance, and reason
                                removal_data = []
                                for feat in removed_features:
                                    if feat in low_imp_details:
                                        details = low_imp_details[feat]
                                        # Values may be already formatted strings or floats
                                        importance = details.get('importance', 0)

                                        # Format only if numeric, otherwise use as-is
                                        if isinstance(importance, (int, float)):
                                            importance_str = f"{importance:.6f}"
                                        else:
                                            importance_str = str(importance)

                                        removal_data.append({
                                            'Feature': feat,
                                            'Importance': importance_str,
                                            'Category': details.get('category', 'Unknown'),
                                            'Reason': details.get('reason', 'Low importance')
                                        })

                                if removal_data:
                                    st.dataframe(pd.DataFrame(removal_data), width='stretch', hide_index=True)
                            else:
                                st.write("- **Removed features:**")
                                for feat in removed_features:
                                    st.write(f"  - `{feat}`")
                        else:
                            st.write("- No features removed")

                    elif "Correlation" in step_name:
                        removed_by_group = step_details.get('removed_by_group', {})
                        if removed_by_group:
                            iterations = step_details.get('iterations', 1)
                            st.write(f"- **Iterations:** {iterations}")
                            st.write(f"- **Features removed:** {step_details.get('features_removed', 0)}")
                            st.write(f"- **Remaining features:** {step_details.get('remaining_features', 0)}")

                            # Show detailed removal information if available
                            fs_result = st.session_state.get('advanced_auto_feature_selection_result', {})
                            fs_summary = fs_result.get('details', {})
                            corr_details = fs_summary.get('correlation_details', {})

                            if corr_details:
                                st.write("- **Detailed Removal Information by Group:**")
                                for group_id, features in removed_by_group.items():
                                    st.write(f"  **{group_id}:**")
                                    # Create a table for this group
                                    removal_data = []
                                    for feat in features:
                                        if feat in corr_details:
                                            details = corr_details[feat]
                                            # Values may be already formatted strings or floats
                                            importance = details.get('importance', 0)
                                            total_corr = details.get('total_correlation', 0)

                                            # Format only if numeric, otherwise use as-is
                                            if isinstance(importance, (int, float)):
                                                importance_str = f"{importance:.6f}"
                                            else:
                                                importance_str = str(importance)

                                            if isinstance(total_corr, (int, float)):
                                                total_corr_str = f"{total_corr:.3f}"
                                            else:
                                                total_corr_str = str(total_corr)

                                            removal_data.append({
                                                'Feature': feat,
                                                'Importance': importance_str,
                                                'Total Correlation': total_corr_str,
                                                'Reason': details.get('reason', 'High correlation')
                                            })

                                    if removal_data:
                                        st.dataframe(pd.DataFrame(removal_data), width='stretch', hide_index=True)
                            else:
                                st.write("- **Removed by group:**")
                                for group_id, features in removed_by_group.items():
                                    st.write(f"  - **{group_id}:** {', '.join(features)}")
                        else:
                            st.write("- No correlation-based removal")

                    elif "Protected Attributes" in step_name:
                        protected_attrs = step_details.get('protected_attributes_list', [])
                        if protected_attrs:
                            st.warning(f"⚠️ {len(protected_attrs)} protected attribute(s) detected:")
                            for attr in protected_attrs:
                                st.write(f"  - `{attr}`")
                        else:
                            st.write("- No protected attributes detected")

                    elif "Duplicate Removal" in step_name:
                        action = step_details.get('action', '')
                        if action == 'skipped':
                            st.write("- No duplicates found")
                        else:
                            total_removed = step_details.get('total_removed', 0)
                            st.write(f"- **Total rows removed:** {total_removed}")

                            if 'training_stats' in step_details:
                                train_stats = step_details['training_stats']
                                test_stats = step_details.get('testing_stats', {})

                                st.write("**Training:**")
                                st.write(f"  - Exact: {train_stats.get('exact_duplicates_found', 0)}")
                                st.write(f"  - Partial: {train_stats.get('partial_duplicates_found', 0)}")

                                st.write("**Testing:**")
                                st.write(f"  - Exact: {test_stats.get('exact_duplicates_found', 0)}")
                                st.write(f"  - Partial: {test_stats.get('partial_duplicates_found', 0)}")

                    elif "Boruta" in step_name:
                        action = step_details.get('action', '')
                        if action == 'skipped':
                            st.write(f"- Skipped: {step_details.get('reason', 'N/A')}")
                        else:
                            features_removed = step_details.get('features_removed', 0)
                            st.write(f"- **Features removed:** {features_removed}")
                            st.write(f"- **Confirmed features:** {len(step_details.get('confirmed_features', []))}")
                            st.write(f"- **Remaining features:** {step_details.get('remaining_features', 0)}")

                            if 'boruta_stats' in step_details:
                                boruta_stats = step_details['boruta_stats']
                                st.write(f"- **Total analyzed:** {boruta_stats.get('total_features', 0)}")
                                st.write(f"- **Confirmed:** {boruta_stats.get('confirmed_features', 0)}")
                                st.write(f"- **Tentative:** {boruta_stats.get('tentative_features', 0)}")
                                st.write(f"- **Rejected:** {boruta_stats.get('rejected_features', 0)}")

                            # Show detailed removal information if available
                            fs_result = st.session_state.get('advanced_auto_feature_selection_result', {})
                            fs_summary = fs_result.get('details', {})
                            boruta_removed_details = fs_summary.get('boruta_removed_details', {})
                            removed_features_list = step_details.get('removed_features_list', [])

                            if boruta_removed_details and removed_features_list:
                                st.write("- **Detailed Removal Information:**")
                                removal_data = []
                                for feat in removed_features_list:
                                    if feat in boruta_removed_details:
                                        details = boruta_removed_details[feat]
                                        # Values may be already formatted strings or floats
                                        importance = details.get('importance', 0)

                                        # Format only if numeric, otherwise use as-is
                                        if isinstance(importance, (int, float)):
                                            importance_str = f"{importance:.6f}"
                                        else:
                                            importance_str = str(importance)

                                        removal_data.append({
                                            'Feature': feat,
                                            'Importance': importance_str,
                                            'Boruta Status': details.get('status', 'Unknown'),
                                            'Rank': details.get('rank', 'N/A'),
                                            'Reason': details.get('reason', 'Boruta removal')
                                        })

                                if removal_data:
                                    st.dataframe(pd.DataFrame(removal_data), width='stretch', hide_index=True)

                    elif "Validation" in step_name:
                        st.write(f"- **Training shape:** {step_details.get('final_training_shape', 'N/A')}")
                        st.write(f"- **Testing shape:** {step_details.get('final_testing_shape', 'N/A')}")
                        st.write(f"- **Final features:** {step_details.get('final_features', 0)}")
                        st.write("- **All validation checks passed:** ✓")

                    elif "Stage Completion" in step_name:
                        st.write(f"- **Initial features:** {step_details.get('initial_features', 0)}")
                        st.write(f"- **Final features:** {step_details.get('final_features', 0)}")
                        st.write(f"- **Features removed:** {step_details.get('features_removed', 0)}")
                        st.write(f"- **Reduction:** {step_details.get('reduction_percentage', 0):.1f}%")

                    else:
                        # Generic display for other steps
                        for key, value in step_details.items():
                            if not isinstance(value, (dict, list)) or len(str(value)) < 100:
                                st.write(f"- **{key}:** {value}")

        else:
            st.error("❌ Feature selection did not complete successfully.")
            st.write(fs_result.get('summary', 'Unknown error'))

    # Dataset overview section
    st.markdown("---")
    st.write("## 📋 Dataset Review")

    if builder.X_train is None:
        st.error("Training data not available for review.")
        return

    # Add toggle for switching between train and test sets
    # Initialize session state if needed (don't use value= when using session state)
    if 'advanced_auto_preview_set_toggle' not in st.session_state:
        st.session_state.advanced_auto_preview_set_toggle = True

    st.toggle(
        "Switch off for Test Dataset Overview",
        help="Toggle between training and test dataset overviews, default is Training",
        key="advanced_auto_preview_set_toggle"
    )

    preview_set = "Training" if st.session_state.advanced_auto_preview_set_toggle else "Test"

    # Select the appropriate dataset based on toggle
    if preview_set == "Training":
        preview_X = builder.X_train
        preview_y = builder.y_train
    else:
        preview_X = builder.X_test
        preview_y = builder.y_test

    # Create preview data
    preview_data = preview_X.copy()
    preview_data[builder.target_column] = preview_y

    # Use the DatasetOverviewComponent to display dataset overview
    dataset_overview = DatasetOverviewComponent(
        preview_data, st.session_state.logger, keyidentifier="aap"
    )
    dataset_overview.display_overview()

    # Download detailed report
    st.markdown("---")
    st.markdown("### 📥 Download Reports")

    # Generate detailed markdown report
    report_content = _generate_detailed_report(result, builder, include_feature_selection)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📄 Download Complete Pipeline Report",
            data=report_content,
            file_name=f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            help="Download a detailed report of all preprocessing" + (" and feature selection" if include_feature_selection else "") + " steps"
        )

    with col2:
        # Separate feature selection report (if applicable)
        #if include_feature_selection and 'advanced_auto_feature_selection_result' in st.session_state:
        #    fs_result = st.session_state.advanced_auto_feature_selection_result
        #    fs_report = _generate_feature_selection_report(fs_result)

        #    st.download_button(
        #        label="📄 Download Feature Selection Report",
        #        data=fs_report,
        #        file_name=f"feature_selection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        #        mime="text/markdown",
        #        help="Download a detailed report of feature selection steps and results"
        #    )
        st.write(" ")

def _generate_detailed_report(result, builder, include_feature_selection=False) -> str:
    """
    Generate a comprehensive markdown report of all preprocessing operations.

    Args:
        result: The preprocessing result dictionary
        builder: The Builder instance
        include_feature_selection: Whether to include feature selection results

    Returns:
        Detailed markdown report as a string
    """
    details = result.get('details', {})

    report = "# Automated Preprocessing Report\n\n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Executive Summary
    report += "## Executive Summary\n\n"
    duration = (details['end_time'] - details['start_time']).total_seconds()
    steps_completed = len(details.get('steps_completed', []))
    steps_failed = len(details.get('steps_failed', []))
    initial_shape = details.get('initial_shape', (0, 0))
    final_shape = details.get('final_shape', (0, 0))

    # Get testing shape from preprocessing result to show post-preprocessing state
    test_final_shape = None
    for step in details.get('steps_completed', []):
        if step['step'] == 'Final Data Validation':
            test_final_shape = step['details'].get('test_shape', None)
            break

    report += f"- **Processing Duration:** {duration:.1f} seconds\n"
    report += f"- **Steps Completed:** {steps_completed}/{steps_completed + steps_failed}\n"
    report += f"- **Initial Dataset Shape:** {initial_shape[0]:,} rows × {initial_shape[1]:,} columns\n"
    report += f"- **Final Training Shape:** {final_shape[0]:,} rows × {final_shape[1]:,} columns\n"
    if test_final_shape:
        report += f"- **Final Testing Shape:** {test_final_shape[0]:,} rows × {test_final_shape[1]:,} columns\n"
    elif builder.testing_data is not None:
        report += f"- **Final Testing Shape:** {builder.testing_data.shape[0]:,} rows × {builder.testing_data.shape[1]:,} columns\n"
    report += f"- **Features Added:** {(final_shape[1] - initial_shape[1]):+,}\n"
    report += f"- **Rows Removed:** {(final_shape[0] - initial_shape[0]):+,}\n"
    report += "\n"

    # Key Outputs
    report += "## Key Outputs\n\n"
    report += f"- **Encoding Mappings Stored:** {len(details.get('encoding_mappings', {}))}\n"
    report += f"- **Features Binned:** {len(details.get('bin_ranges', {}))}\n"
    report += f"- **Outlier Strategies Applied:** {len(details.get('outlier_strategies', {}))}\n"
    report += "\n"

    # Detailed Step Results
    report += "## Detailed Step Results\n\n"

    for step_info in details.get('steps_completed', []):
        step_name = step_info['step']
        step_details = step_info.get('details', {})

        report += f"### {step_name}\n\n"

        # Custom formatting for each step type
        if step_name == "Train-Test Split":
            report += f"- Training samples: {step_details.get('training_shape', (0,))[0]:,}\n"
            report += f"- Testing samples: {step_details.get('testing_shape', (0,))[0]:,}\n"
            report += f"- Split method: {step_details.get('split_method', 'N/A').title()}\n"
            report += f"- Test size: {step_details.get('test_size', 0):.1%}\n"

        elif step_name == "Missing Values Handling":
            strategies = step_details.get('strategies_applied', {})
            initial_missing_cols = len(strategies)

            report += f"- Columns with missing values: {initial_missing_cols}\n"
            report += f"- Columns processed: {step_details.get('columns_processed', 0)}\n"
            report += f"- Remaining missing values: {step_details.get('remaining_missing', 0)}\n"

            if strategies:
                report += "\n**Strategies Applied by Feature:**\n\n"
                strategy_name_map = {
                    'drop_column': 'Drop Column (>50% missing)',
                    'drop_rows': 'Drop Rows (<5% missing)',
                    'knn': 'KNN Imputation',
                    'median': 'Median Imputation',
                    'mode': 'Mode Imputation',
                    'mean': 'Mean Imputation'
                }
                for col, strategy in sorted(strategies.items()):
                    strategy_display = strategy_name_map.get(strategy, strategy)
                    report += f"- `{col}`: {strategy_display}\n"

        elif step_name == "Feature Binning":
            report += f"- Features analyzed: {step_details.get('features_analyzed', 0)}\n"
            report += f"- Features binned: {step_details.get('features_binned', 0)}\n"

            modified_cols = step_details.get('modified_columns', [])
            bin_ranges = step_details.get('bin_ranges', {})

            if modified_cols:
                report += "\n**Binned Features with Ranges:**\n\n"
                for col in modified_cols:
                    report += f"**`{col}`:**\n"
                    if col in bin_ranges:
                        ranges = bin_ranges[col]
                        if isinstance(ranges, list):
                            # Numerical binning
                            from utils.automated_preprocessing import _format_bin_range
                            for i, bin_range in enumerate(ranges):
                                formatted_range = _format_bin_range(bin_range)
                                report += f"  - Bin {i}: {formatted_range}\n"
                        elif isinstance(ranges, dict):
                            # Categorical binning
                            for bin_id, categories in ranges.items():
                                cat_str = ', '.join(map(str, categories))
                                report += f"  - {bin_id}: {cat_str}\n"
                        else:
                            report += f"  - {ranges}\n"
                    report += "\n"

        elif step_name == "Outlier Handling":
            report += f"- Features analyzed: {step_details.get('features_analyzed', 0)}\n"
            report += f"- Features with outliers handled: {step_details.get('features_handled', 0)}\n"
            report += f"- Total outliers processed: {step_details.get('total_outliers_processed', 0)}\n"

            # Only show features where outliers were actually handled (count > 0)
            outliers_handled = step_details.get('outliers_handled', [])
            outliers_with_action = [oh for oh in outliers_handled if oh.get('outliers_count', 0) > 0]

            if outliers_with_action:
                report += "\n**Outlier Handling Applied:**\n\n"
                strategy_desc_map = {
                    'Remove': 'IQR method (1.5×IQR)',
                    'Remove Extreme': 'Extended IQR method (3×IQR)',
                    'Cap': 'Winsorization (capping)',
                    'Isolation Forest': 'ML-based detection'
                }
                for oh in outliers_with_action:
                    strategy_desc = strategy_desc_map.get(oh['strategy'], oh['strategy'])
                    report += f"- `{oh['feature']}`: {strategy_desc}\n"
                    report += f"  - Outliers removed/capped: {oh['outliers_count']}\n"
            elif outliers_handled:
                report += "\nNo outliers were found that required removal/capping\n"

        elif step_name == "Feature Creation":
            report += f"- Combinations generated: {step_details.get('features_generated', 0)}\n"
            report += f"- Features after filtering: {step_details.get('features_analyzed', 0)}\n"
            report += f"- Features successfully added: {step_details.get('features_added', 0)}\n"

            added_features = step_details.get('added_features_list', [])
            if added_features:
                report += f"\n**All Engineered Features Added ({len(added_features)}):**\n\n"
                for feat in added_features:
                    report += f"- `{feat}`\n"

        elif step_name == "Categorical Encoding":
            report += f"- Columns encoded: {step_details.get('columns_encoded', 0)}\n"
            report += f"- Encoding mappings stored: {step_details.get('mappings_stored', 0)}\n"

            encoding_methods = step_details.get('encoding_methods', {})
            if encoding_methods:
                report += "\n**Encoding Applied by Feature:**\n\n"
                method_name_map = {
                    'label': 'Label Encoding (ordinal)',
                    'onehot': 'One-Hot Encoding',
                    'target': 'Target Encoding (mean encoding)',
                    'drop_column': 'Drop Column'
                }

                for col, col_info in sorted(encoding_methods.items()):
                    method = col_info['method']
                    method_display = method_name_map.get(method, method)
                    report += f"- `{col}`: {method_display}\n"

                    # Show clean mapping if available
                    if col in details.get('encoding_mappings', {}):
                        mapping_info = details['encoding_mappings'][col]
                        if isinstance(mapping_info, dict):
                            # Extract clean mapping based on encoding type
                            if mapping_info.get('method') == 'Label Encoding':
                                # Show label encoding mapping
                                if 'mapping' in mapping_info:
                                    clean_mapping = {}
                                    for key, val in mapping_info['mapping'].items():
                                        # Convert numpy types to Python types
                                        clean_val = int(val) if hasattr(val, 'item') else val
                                        clean_mapping[key] = clean_val
                                    report += f"  - Mapping: {clean_mapping}\n"
                            elif mapping_info.get('method') == 'One-Hot Encoding':
                                # Show one-hot encoding columns created
                                if 'new_columns' in mapping_info:
                                    report += f"  - New columns: {', '.join(mapping_info['new_columns'])}\n"
                            elif mapping_info.get('method') == 'Target Encoding':
                                # Show target encoding info
                                if 'mapping' in mapping_info and len(mapping_info['mapping']) <= 20:
                                    clean_mapping = {}
                                    for key, val in mapping_info['mapping'].items():
                                        clean_val = float(val) if hasattr(val, 'item') else val
                                        clean_mapping[key] = f"{clean_val:.4f}"
                                    report += f"  - Target means: {clean_mapping}\n"

        elif step_name == "Data Types Optimization":
            report += f"- Type conversions applied: {step_details.get('conversions_applied', 0)}\n"
            memory_saved = step_details.get('memory_saved_mb', 0)
            memory_reduction_pct = step_details.get('memory_reduction_pct', 0)
            report += f"- Memory saved: {memory_saved:.2f} MB ({memory_reduction_pct:.1f}%)\n"

        elif "Duplicate Removal" in step_name:
            action = step_details.get('action', '')
            if action == 'skipped':
                report += "- No duplicates found\n"
            else:
                # Show separate stats for training and testing if available
                if 'train_exact_duplicates' in step_details:
                    # Final duplicate removal with separate train/test stats
                    report += "\n**Training Data:**\n"
                    train_exact = step_details.get('train_exact_duplicates', 0)
                    train_partial = step_details.get('train_partial_duplicates', 0)
                    train_removed = step_details.get('train_rows_removed', 0)

                    if train_removed > 0:
                        report += f"- Exact duplicates: {train_exact}\n"
                        report += f"- Partial duplicate groups: {train_partial}\n"
                        report += f"- Rows removed: {train_removed}\n"
                    else:
                        report += "- No duplicates found\n"

                    report += "\n**Testing Data:**\n"
                    test_exact = step_details.get('test_exact_duplicates', 0)
                    test_partial = step_details.get('test_partial_duplicates', 0)
                    test_removed = step_details.get('test_rows_removed', 0)

                    if test_removed > 0:
                        report += f"- Exact duplicates: {test_exact}\n"
                        report += f"- Partial duplicate groups: {test_partial}\n"
                        report += f"- Rows removed: {test_removed}\n"
                    else:
                        report += "- No duplicates found\n"

                    total_removed = step_details.get('total_rows_removed', 0)
                    report += f"\n**Total rows removed: {total_removed}**\n"
                else:
                    # Initial duplicate removal (before split)
                    exact_dups = step_details.get('exact_duplicates_found', 0)
                    partial_dups = step_details.get('partial_duplicate_groups_found', 0)
                    rows_removed = step_details.get('rows_removed', 0)

                    report += f"- Exact duplicates found: {exact_dups}\n"
                    report += f"- Partial duplicate groups: {partial_dups}\n"
                    report += f"- Total rows removed: {rows_removed}\n"

        elif step_name == "Final Data Validation":
            report += f"- Training shape: {step_details.get('train_shape', 'N/A')}\n"
            report += f"- Testing shape: {step_details.get('test_shape', 'N/A')}\n"
            report += f"- Number of features: {step_details.get('num_features', 'N/A')}\n"
            report += "- All validation checks: PASSED ✓\n"

        else:
            # Generic display for other steps
            for key, value in step_details.items():
                if not isinstance(value, (dict, list)) or len(str(value)) < 100:
                    report += f"- {key}: {value}\n"

        report += "\n"

    # Failed steps (if any)
    if details.get('steps_failed'):
        report += "## Failed Steps\n\n"
        for step_info in details['steps_failed']:
            report += f"### ❌ {step_info['step']}\n\n"
            report += f"**Error:** {step_info['error']}\n\n"

    # Feature Selection Results (if applicable)
    if include_feature_selection and 'advanced_auto_feature_selection_result' in st.session_state:
        report += "## Feature Selection Results\n\n"

        fs_result = st.session_state.advanced_auto_feature_selection_result
        fs_details = fs_result.get('details', {})

        if fs_result.get('success', False):
            # Summary metrics
            initial_features = fs_details.get('initial_features', 0)
            final_features = fs_details.get('final_features', 0)
            features_removed = len(fs_details.get('features_removed', []))
            reduction_pct = (features_removed / initial_features * 100) if initial_features > 0 else 0

            report += f"- **Initial Features:** {initial_features}\n"
            report += f"- **Final Features:** {final_features}\n"
            report += f"- **Features Removed:** {features_removed} ({reduction_pct:.1f}%)\n"
            report += f"- **Boruta Applied:** {'Yes' if fs_details.get('boruta_applied', False) else 'No'}\n"
            report += "\n"

            # Steps completed
            report += "### Feature Selection Steps\n\n"
            steps_completed = fs_details.get('steps_completed', [])

            for step_info in steps_completed:
                step_name = step_info.get('step', 'Unknown Step')
                step_details = step_info.get('details', {})

                report += f"#### {step_name}\n\n"

                if "Feature Importance Analysis" in step_name:
                    report += f"- Total features: {step_details.get('total_features', 0)}\n"
                    report += f"- Low importance features: {step_details.get('low_importance_count', 0)}\n"
                    report += f"- Correlation pairs: {step_details.get('correlation_pairs', 0)}\n"
                    report += f"- Protected attributes: {step_details.get('protected_attributes_count', 0)}\n"

                elif "Low Importance" in step_name:
                    removed_features = step_details.get('removed_features_list', [])
                    if removed_features:
                        report += f"- Features removed: {len(removed_features)}\n"
                        report += f"- Remaining features: {step_details.get('remaining_features', 0)}\n"
                        report += "\n**Removed Features:**\n\n"
                        for feat in removed_features:
                            report += f"- `{feat}`\n"
                    else:
                        report += "- No features removed\n"

                elif "Correlation" in step_name:
                    removed_by_group = step_details.get('removed_by_group', {})
                    if removed_by_group:
                        report += f"- Correlation groups: {step_details.get('correlation_groups', 0)}\n"
                        report += f"- Features removed: {step_details.get('features_removed', 0)}\n"
                        report += f"- Remaining features: {step_details.get('remaining_features', 0)}\n"
                        report += "\n**Removed by Group:**\n\n"
                        for group_id, features in removed_by_group.items():
                            report += f"- **{group_id}:** {', '.join(features)}\n"
                    else:
                        report += "- No correlation-based removal\n"

                elif "Protected Attributes" in step_name:
                    protected_attrs = step_details.get('protected_attributes_list', [])
                    if protected_attrs:
                        report += f"- Protected attributes detected: {len(protected_attrs)}\n"
                        report += "\n**Protected Attributes:**\n\n"
                        for attr in protected_attrs:
                            report += f"- `{attr}`\n"
                    else:
                        report += "- No protected attributes detected\n"

                elif "Boruta" in step_name:
                    action = step_details.get('action', '')
                    if action == 'skipped':
                        report += f"- Skipped: {step_details.get('reason', 'N/A')}\n"
                    else:
                        features_removed = step_details.get('features_removed', 0)
                        report += f"- Features removed: {features_removed}\n"
                        report += f"- Confirmed features: {len(step_details.get('confirmed_features', []))}\n"
                        report += f"- Remaining features: {step_details.get('remaining_features', 0)}\n"

                        if 'boruta_stats' in step_details:
                            boruta_stats = step_details['boruta_stats']
                            report += f"\n**Boruta Statistics:**\n\n"
                            report += f"- Total analyzed: {boruta_stats.get('total_features', 0)}\n"
                            report += f"- Confirmed: {boruta_stats.get('confirmed_features', 0)}\n"
                            report += f"- Tentative: {boruta_stats.get('tentative_features', 0)}\n"
                            report += f"- Rejected: {boruta_stats.get('rejected_features', 0)}\n"

                else:
                    # Generic display for other steps
                    for key, value in step_details.items():
                        if not isinstance(value, (dict, list)) or len(str(value)) < 100:
                            report += f"- {key}: {value}\n"

                report += "\n"

        else:
            report += "❌ Feature selection did not complete successfully.\n\n"
            report += f"**Error:** {fs_result.get('summary', 'Unknown error')}\n\n"

    # Final dataset information
    report += "## Final Dataset Information\n\n"
    report += f"- Target Column: `{builder.target_column}`\n"
    report += f"- Feature Count: {len(builder.feature_names)}\n"
    report += f"- Training Samples: {len(builder.y_train):,}\n"
    report += f"- Testing Samples: {len(builder.y_test):,}\n"
    report += "\n**Feature Names:**\n\n"
    for feat in builder.feature_names:
        report += f"- `{feat}`\n"

    report += "\n---\n\n"
    report += "*Report generated by ML Builder - Advanced Automated " + ("Pipeline" if include_feature_selection else "Preprocessing") + "*\n"

    return report


def _generate_feature_selection_report(fs_result) -> str:
    """
    Generate a detailed markdown report for feature selection results.

    Args:
        fs_result: The feature selection result dictionary

    Returns:
        Detailed markdown report as a string
    """
    fs_details = fs_result.get('details', {})

    report = "# Automated Feature Selection Report\n\n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Executive Summary
    report += "## Executive Summary\n\n"

    if fs_result.get('success', False):
        initial_features = fs_details.get('initial_features', 0)
        final_features = fs_details.get('final_features', 0)
        features_removed = len(fs_details.get('features_removed', []))
        reduction_pct = (features_removed / initial_features * 100) if initial_features > 0 else 0

        duration_seconds = 0
        if fs_details.get('start_time') and fs_details.get('end_time'):
            duration_seconds = (fs_details['end_time'] - fs_details['start_time']).total_seconds()

        report += f"- **Processing Duration:** {duration_seconds:.1f} seconds\n"
        report += f"- **Initial Features:** {initial_features}\n"
        report += f"- **Final Features:** {final_features}\n"
        report += f"- **Features Removed:** {features_removed} ({reduction_pct:.1f}%)\n"
        report += f"- **Boruta Algorithm Applied:** {'Yes' if fs_details.get('boruta_applied', False) else 'No'}\n"
        report += f"- **Steps Completed:** {len(fs_details.get('steps_completed', []))}\n"
        report += "\n"

        # Key Actions
        report += "## Key Actions\n\n"
        report += f"- **Low Importance Features Removed:** {len(fs_details.get('low_importance_removed', []))}\n"
        report += f"- **Correlation Groups Addressed:** {len(fs_details.get('correlation_removed', {}))}\n"
        report += f"- **Protected Attributes Identified:** {len(fs_details.get('protected_attributes', []))}\n"

        duplicates_stats = fs_details.get('duplicates_stats', {})
        train_removed = duplicates_stats.get('training', {}).get('total_reduction', 0)
        test_removed = duplicates_stats.get('testing', {}).get('total_reduction', 0)
        report += f"- **Duplicate Rows Removed:** {train_removed + test_removed} (Train: {train_removed}, Test: {test_removed})\n"
        report += "\n"

        # Detailed Step Results
        report += "## Detailed Step Results\n\n"
        steps_completed = fs_details.get('steps_completed', [])

        for step_info in steps_completed:
            step_name = step_info.get('step', 'Unknown Step')
            step_details = step_info.get('details', {})

            report += f"### {step_name}\n\n"

            if "Feature Importance Analysis" in step_name:
                report += f"- Total features: {step_details.get('total_features', 0)}\n"
                report += f"- Low importance features: {step_details.get('low_importance_count', 0)}\n"
                report += f"- Correlation pairs: {step_details.get('correlation_pairs', 0)}\n"
                report += f"- Protected attributes: {step_details.get('protected_attributes_count', 0)}\n"

            elif "Low Importance" in step_name:
                removed_features = step_details.get('removed_features_list', [])
                if removed_features:
                    report += f"- Features removed: {len(removed_features)}\n"
                    report += f"- Remaining features: {step_details.get('remaining_features', 0)}\n"

                    # Add detailed removal information if available
                    low_imp_details = fs_details.get('low_importance_details', {})
                    if low_imp_details:
                        report += "\n**Detailed Removal Information:**\n\n"
                        report += "| Feature | Importance Score | Category | Detailed Reason |\n"
                        report += "|---------|-----------------|----------|----------------|\n"
                        for feat in removed_features:
                            if feat in low_imp_details:
                                details = low_imp_details[feat]
                                importance = details.get('importance', 0)
                                category = details.get('category', 'Unknown')
                                reason = details.get('reason', 'Low importance')
                                if isinstance(importance, (int, float)):
                                    importance_str = f"{importance:.6f}"
                                else:
                                    importance_str = str(importance)
                                report += f"| {feat} | {importance_str} | {category} | {reason} |\n"
                    else:
                        report += "\n**Removed Features:**\n\n"
                        for feat in removed_features:
                            report += f"- `{feat}`\n"
                else:
                    report += "- No features removed\n"

            elif "Correlation" in step_name:
                removed_by_group = step_details.get('removed_by_group', {})
                if removed_by_group:
                    report += f"- Correlation groups: {step_details.get('correlation_groups', 0)}\n"
                    report += f"- Features removed: {step_details.get('features_removed', 0)}\n"
                    report += f"- Remaining features: {step_details.get('remaining_features', 0)}\n"

                    # Add detailed removal information if available
                    corr_details = fs_details.get('correlation_details', {})
                    if corr_details:
                        report += "\n**Detailed Removal Information by Group:**\n\n"
                        for group_id, features in removed_by_group.items():
                            report += f"**{group_id}:**\n\n"
                            report += "| Feature | Importance Score | Total Correlation | Detailed Reason |\n"
                            report += "|---------|-----------------|-------------------|----------------|\n"
                            for feat in features:
                                if feat in corr_details:
                                    details = corr_details[feat]
                                    importance = details.get('importance', 0)
                                    total_corr = details.get('total_correlation', 0)
                                    reason = details.get('reason', 'High correlation')

                                    if isinstance(importance, (int, float)):
                                        importance_str = f"{importance:.6f}"
                                    else:
                                        importance_str = str(importance)

                                    if isinstance(total_corr, (int, float)):
                                        total_corr_str = f"{total_corr:.3f}"
                                    else:
                                        total_corr_str = str(total_corr)

                                    report += f"| {feat} | {importance_str} | {total_corr_str} | {reason} |\n"
                            report += "\n"
                    else:
                        report += "\n**Removed by Group:**\n\n"
                        for group_id, features in removed_by_group.items():
                            report += f"- **{group_id}:** {', '.join(features)}\n"
                else:
                    report += "- No correlation-based removal\n"

            elif "Protected Attributes" in step_name:
                protected_attrs = step_details.get('protected_attributes_list', [])
                if protected_attrs:
                    report += f"- Protected attributes detected: {len(protected_attrs)}\n"
                    report += "\n**Protected Attributes:**\n\n"
                    for attr in protected_attrs:
                        report += f"- `{attr}`\n"
                else:
                    report += "- No protected attributes detected\n"

            elif "Duplicate Removal" in step_name:
                action = step_details.get('action', '')
                if action == 'skipped':
                    report += "- No duplicates found\n"
                else:
                    total_removed = step_details.get('total_removed', 0)
                    report += f"- Total rows removed: {total_removed}\n"

                    if 'training_stats' in step_details:
                        train_stats = step_details['training_stats']
                        test_stats = step_details.get('testing_stats', {})

                        report += "\n**Training Data:**\n\n"
                        report += f"- Exact duplicates: {train_stats.get('exact_duplicates_found', 0)}\n"
                        report += f"- Partial duplicates: {train_stats.get('partial_duplicates_found', 0)}\n"
                        report += f"- Rows removed: {train_stats.get('total_reduction', 0)}\n"

                        report += "\n**Testing Data:**\n\n"
                        report += f"- Exact duplicates: {test_stats.get('exact_duplicates_found', 0)}\n"
                        report += f"- Partial duplicates: {test_stats.get('partial_duplicates_found', 0)}\n"
                        report += f"- Rows removed: {test_stats.get('total_reduction', 0)}\n"

            elif "Boruta" in step_name:
                action = step_details.get('action', '')
                if action == 'skipped':
                    report += f"- Skipped: {step_details.get('reason', 'N/A')}\n"
                else:
                    features_removed = step_details.get('features_removed', 0)
                    report += f"- Features removed: {features_removed}\n"
                    report += f"- Confirmed features: {len(step_details.get('confirmed_features', []))}\n"
                    report += f"- Remaining features: {step_details.get('remaining_features', 0)}\n"

                    if 'boruta_stats' in step_details:
                        boruta_stats = step_details['boruta_stats']
                        report += f"\n**Boruta Statistics:**\n\n"
                        report += f"- Total analyzed: {boruta_stats.get('total_features', 0)}\n"
                        report += f"- Confirmed: {boruta_stats.get('confirmed_features', 0)}\n"
                        report += f"- Tentative: {boruta_stats.get('tentative_features', 0)}\n"
                        report += f"- Rejected: {boruta_stats.get('rejected_features', 0)}\n"

                    # Show removed features with detailed information
                    removed_features = step_details.get('removed_features_list', [])
                    if removed_features:
                        boruta_removed_details = fs_details.get('boruta_removed_details', {})
                        if boruta_removed_details:
                            report += f"\n**Detailed Removal Information:**\n\n"
                            report += "| Feature | Importance Score | Boruta Status | Rank | Detailed Reason |\n"
                            report += "|---------|-----------------|---------------|------|----------------|\n"
                            for feat in removed_features:
                                if feat in boruta_removed_details:
                                    details = boruta_removed_details[feat]
                                    importance = details.get('importance', 0)
                                    status = details.get('status', 'Unknown')
                                    rank = details.get('rank', 'N/A')
                                    reason = details.get('reason', 'Boruta removal')

                                    if isinstance(importance, (int, float)):
                                        importance_str = f"{importance:.6f}"
                                    else:
                                        importance_str = str(importance)

                                    report += f"| {feat} | {importance_str} | {status} | {rank} | {reason} |\n"
                        else:
                            report += f"\n**Removed Features ({len(removed_features)}):**\n\n"
                            for feat in removed_features:
                                report += f"- `{feat}`\n"

            elif "Validation" in step_name or "Stage Completion" in step_name:
                report += f"- Initial features: {step_details.get('initial_features', 0)}\n"
                report += f"- Final features: {step_details.get('final_features', 0)}\n"
                report += f"- Features removed: {step_details.get('features_removed', 0)}\n"
                report += f"- Reduction: {step_details.get('reduction_percentage', 0):.1f}%\n"

            else:
                # Generic display for other steps
                for key, value in step_details.items():
                    if not isinstance(value, (dict, list)) or len(str(value)) < 100:
                        report += f"- {key}: {value}\n"

            report += "\n"

        # Comprehensive Feature Removal Summary
        all_removed = fs_details.get('features_removed', [])
        if all_removed:
            report += "## Complete Feature Removal Summary\n\n"
            report += f"**Total Features Removed:** {len(all_removed)}\n\n"

            # Get detailed information
            low_imp_details = fs_details.get('low_importance_details', {})
            corr_details = fs_details.get('correlation_details', {})
            boruta_details = fs_details.get('boruta_removed_details', {})

            # Create comprehensive table
            report += "| # | Feature | Removal Method | Importance | Additional Info |\n"
            report += "|---|---------|----------------|------------|----------------|\n"

            for idx, feat in enumerate(all_removed, 1):
                # Determine removal method and get details
                if feat in low_imp_details:
                    method = "Low Importance"
                    details = low_imp_details[feat]
                    importance = f"{details.get('importance', 0):.6f}" if isinstance(details.get('importance'), (int, float)) else 'N/A'
                    info = details.get('category', 'Unknown')
                elif feat in corr_details:
                    method = "Correlation"
                    details = corr_details[feat]
                    importance = f"{details.get('importance', 0):.6f}" if isinstance(details.get('importance'), (int, float)) else 'N/A'
                    total_corr = details.get('total_correlation', 0)
                    if isinstance(total_corr, (int, float)):
                        info = f"Total Corr: {total_corr:.3f}"
                    else:
                        info = f"Total Corr: {total_corr}"
                elif feat in boruta_details:
                    method = "Boruta"
                    details = boruta_details[feat]
                    importance = f"{details.get('importance', 0):.6f}" if isinstance(details.get('importance'), (int, float)) else 'N/A'
                    info = details.get('status', 'Unknown')
                else:
                    method = "Other"
                    importance = 'N/A'
                    info = '-'

                report += f"| {idx} | {feat} | {method} | {importance} | {info} |\n"

            report += "\n"

    else:
        report += "❌ Feature selection did not complete successfully.\n\n"
        report += f"**Error:** {fs_result.get('summary', 'Unknown error')}\n\n"

        errors = fs_result.get('errors', [])
        if errors:
            report += "## Errors\n\n"
            for error in errors:
                if isinstance(error, dict):
                    report += f"- **Step:** {error.get('step', 'Unknown')}\n"
                    report += f"- **Error:** {error.get('error', 'Unknown')}\n\n"
                else:
                    report += f"- {str(error)}\n\n"

    report += "\n---\n\n"
    report += "*Report generated by ML Builder - Advanced Automated Feature Selection*\n"

    return report

