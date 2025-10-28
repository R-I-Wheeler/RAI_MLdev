"""
Automated Model Selection and Training Render Component

This component provides a user interface for the automated model selection and training
pipeline, following the same pattern as render_advanced_automated_preprocessing.py.

Features:
- Toggle switch to enable/disable automation
- Configuration options for optimization method, CV folds, and iterations
- Progress tracking through 3 phases: Selection, Training, Optimization
- Comprehensive dashboard with expandable sections
- Error handling with manual fallback options
- Session state management for downstream compatibility
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from Builder import ModelStage
from utils.automated_model_selection_training import AutomatedModelSelectionTraining


def render_automated_model_selection_training():
    """
    Render the automated model selection and training interface.

    This provides a streamlined workflow that combines model selection (Page 5)
    and model training (Page 6) into a single automated process.
    """
    st.markdown("---")
    st.write("### 🤖 Automated Model Selection & Training")

    # Check if automation has already been completed
    if 'automated_model_selection_training_completed' in st.session_state and st.session_state.automated_model_selection_training_completed:
        # Add header with clear button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("✨ Automated model selection and training has been completed. Ready for model evaluation!")
        with col2:
            if st.button("🗑️ Clear Results", type="secondary", help="Clear automation results and start fresh"):
                # Clear automation state
                if 'automated_model_selection_training_completed' in st.session_state:
                    del st.session_state.automated_model_selection_training_completed
                if 'automated_model_selection_training_result' in st.session_state:
                    del st.session_state.automated_model_selection_training_result

                # Reset model selection and training stages
                st.session_state.builder.stage_completion[ModelStage.MODEL_SELECTION] = False
                st.session_state.builder.stage_completion[ModelStage.MODEL_TRAINING] = False

                # Clear training state
                if 'training_complete' in st.session_state:
                    del st.session_state.training_complete
                if 'training_results' in st.session_state:
                    del st.session_state.training_results
                if 'selected_model_type' in st.session_state:
                    del st.session_state.selected_model_type
                if 'selected_model_stability' in st.session_state:
                    del st.session_state.selected_model_stability
                if 'previous_model_selection' in st.session_state:
                    del st.session_state.previous_model_selection

                # Clear builder model state
                st.session_state.builder.model = None

                # Log the clear action
                st.session_state.logger.log_user_action(
                    "Automated Model Selection & Training Results Cleared",
                    {"action": "clear_results", "reason": "user_requested"}
                )

                st.success("Automation results cleared successfully! You can now select a different model or run automation again.")
                st.rerun()

        st.divider()

        # Show comprehensive dashboard
        if 'automated_model_selection_training_result' in st.session_state:
            result = st.session_state.automated_model_selection_training_result
            render_automation_dashboard(result)

        # Navigation button
        st.write("---")
        if st.button("Proceed to Model Evaluation", type="primary", key="automated_proceed_to_evaluation"):
            # Extract detailed metrics from stored result
            details = result.get('details', {})
            best_score = details.get('best_score', 0)
            cv_metrics = details.get('cv_metrics', {})
            stability_analysis = details.get('stability_analysis', {})

            # Enhanced logging for transition
            st.session_state.logger.log_calculation(
                "Model Selection and Training Results",
                {
                    "final_model": st.session_state.builder.model['type'],
                    "problem_type": details.get('problem_type', 'Unknown'),
                    "best_score": float(best_score) if best_score else 0,
                    "cv_metrics": cv_metrics,
                    "stability_level": stability_analysis.get('level', 'Unknown'),
                    "automated_workflow": True
                }
            )

            # Create comprehensive journey point for navigation
            journey_details = {
                'Model Type': st.session_state.builder.model['type'],
                'Problem Type': details.get('problem_type', 'Unknown').replace('_', ' ').title(),
                'Best Score': f"{best_score:.4f}" if best_score else 'N/A',
                'Stability': stability_analysis.get('level', 'Unknown'),
                'Automated Workflow': 'Yes'
            }

            # Add CV metrics
            if cv_metrics:
                for metric, value in cv_metrics.items():
                    if isinstance(value, (int, float)):
                        journey_details[f'CV {metric}'] = f"{value:.4f}"

            # Add optimization details
            if details.get('calibration_applied'):
                journey_details['Calibration'] = details.get('calibration_method', 'Unknown').title()
            if details.get('threshold_optimized'):
                journey_details['Optimal Threshold'] = f"{details.get('optimal_threshold', 0.5):.3f}"

            st.session_state.logger.log_journey_point(
                stage="MODEL_SELECTION",
                decision_type="AUTOMATED_MODEL_SELECTION_TRAINING_TO_EVALUATION",
                description="Automated model selection and training completed, proceeding to evaluation",
                details=journey_details,
                parent_id=None
            )

            st.session_state.logger.log_stage_transition(
                "MODEL_TRAINING",
                "MODEL_EVALUATION",
                {
                    "automated_workflow": True,
                    "model_type": st.session_state.builder.model['type'],
                    "best_score": float(best_score) if best_score else 0
                }
            )

            st.session_state.logger.log_user_action(
                "Navigation",
                {"direction": "forward", "to_stage": "MODEL_EVALUATION", "automated": True}
            )
            st.switch_page("pages/7_Model_Evaluation.py")
        return

    # Add explanation expander
    with st.expander("ℹ️ Understanding Automated Model Selection & Training", expanded=False):
        st.markdown("""
        ### What This Does

        This automated workflow combines **Model Selection** and **Model Training** into a single streamlined process:

        **Phase 1: Selection** (Steps 1-3)
        - Detects problem type (classification/regression)
        - Analyzes dataset characteristics
        - Recommends optimal model
        - Runs quick comparison across all models
        - Selects best model based on intelligent decision logic

        **Phase 2: Training** (Steps 4-5)
        - Handles class imbalance (classification only)
        - Performs hyperparameter optimization (Random Search or Optuna)
        - Trains model with cross-validation
        - Analyzes model stability

        **Phase 3: Optimization** (Steps 6-7)
        - Selects final model configuration
        - Applies calibration (classification)
        - Optimizes decision threshold (binary classification)

        ### When to Use This

        ✅ **Use Automation When:**
        - You want a quick, intelligent model selection and training
        - You trust the recommendation engine
        - You want to compare multiple models efficiently
        - You're exploring the dataset for the first time

        ⚠️ **Use Manual When:**
        - You know exactly which model you want
        - You need fine-grained control over hyperparameters
        - You want to experiment with specific configurations
        - You're working with domain-specific requirements

        ### What Gets Set

        Upon successful completion, this workflow:
        - Sets the same session state as manual training
        - Marks both MODEL_SELECTION and MODEL_TRAINING stages complete
        - Allows you to skip directly to Model Evaluation
        - Provides detailed results in a comprehensive dashboard
        """)

    # Configuration options
    st.write("#### Configuration")

    # Add the toggle switch
    automated_enabled = st.toggle(
        "Enable Automated Model Selection & Training",
        help="Automatically select and train the best model using intelligent automation",
        key="automated_model_selection_training_toggle"
    )

    col1, col2 = st.columns(2)

    with col1:
        optimization_method = st.radio(
            "Optimization Method",
            ["Random Search", "Optuna"],
            help="""
            - Random Search: Traditional random sampling of parameter space
            - Optuna: Advanced optimization using Bayesian optimization (recommended)
            """,
            key="automated_optimization_method",
            index=1  # Default to Optuna
        )

        show_analysis = st.checkbox(
            "Show detailed step-by-step analysis",
            value=False,
            help="Display detailed information about each automation step",
            key="automated_show_analysis"
        )

    with col2:
        cv_folds_option = st.selectbox(
            "Cross-Validation Folds",
            ["Adaptive (recommended)", "5", "7", "10"],
            help="""
                    - Adaptive: Automatically adjusts based on dataset size
                    - Fixed values: Use specific number of folds
                    """,
            key="automated_cv_folds"
        )

        n_iter_option = st.selectbox(
            "Parameter Configurations/Trials",
            ["Adaptive (recommended)", "50", "75", "100"],
            help="""
            - Adaptive: Automatically adjusts based on feature complexity
            - Fixed values: Use specific number of iterations
            """,
            key="automated_n_iter"
        )

    st.session_state.automated_enabled = automated_enabled

    # Log user preference
    st.session_state.logger.log_user_action(
        "Automated Model Selection & Training Preference",
        {
            "automated_enabled": bool(automated_enabled),
            "optimization_method": optimization_method,
            "cv_folds": cv_folds_option,
            "n_iter": n_iter_option,
            "show_analysis": show_analysis,
            "timestamp": datetime.now().isoformat()
        }
    )

    if automated_enabled:
        st.session_state.logger.log_journey_point(
            stage="MODEL_SELECTION",
            decision_type="AUTOMATED_MODEL_SELECTION_TRAINING_ENABLED",
            description="Automated model selection and training enabled",
            details={
                'optimization_method': optimization_method,
                'cv_folds': cv_folds_option,
                'n_iter': n_iter_option
            },
            parent_id=None
        )

        # Validate initial state
        if st.session_state.builder.training_data is None:
            st.error("❌ No data loaded. Please complete preprocessing first.")
            return

        if st.session_state.builder.target_column is None:
            st.error("❌ No target column set. Please set target column first.")
            return

        # Add confirmation button
        st.write("---")
        st.write("#### Ready to Start?")
        st.info("""
        Click the button below to start the automated model selection and training pipeline.
        This will execute all steps automatically: problem detection, model comparison,
        selection, training, and optimization.
        """)

        if st.button("🚀 Run Automated Model Selection & Training", type="primary"):
            try:
                # Parse configuration
                cv_folds = None if cv_folds_option == "Adaptive (recommended)" else int(cv_folds_option)
                n_iter = None if n_iter_option == "Adaptive (recommended)" else int(n_iter_option)
                optimization_method_internal = "optuna" if optimization_method == "Optuna" else "random_search"

                # Create progress containers
                progress_container = st.container()
                status_container = st.container()

                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    phase_text = st.empty()

                with status_container:
                    # Phase 1: Selection
                    phase_text.info("📊 **Phase 1: Selection** - Analyzing dataset and selecting optimal model...")
                    status_text.info("🔄 Step 1/7: Problem detection and model recommendation...")
                    progress_bar.progress(5)

                    # Create the automated component
                    # Pass cv_folds and n_iter (None for adaptive, or specific values)
                    auto_pipeline = AutomatedModelSelectionTraining(
                        builder=st.session_state.builder,
                        logger=st.session_state.logger,
                        optimization_method=optimization_method_internal,
                        cv_folds=cv_folds,
                        n_iter=n_iter,
                        show_analysis=show_analysis,
                        auto_handle_imbalance=True
                    )

                    status_text.info("🔄 Step 2/7: Model comparison across all algorithms...")
                    progress_bar.progress(15)

                    status_text.info("🔄 Step 3/7: Final model selection...")
                    progress_bar.progress(25)

                    # Phase 2: Training
                    phase_text.info("🎯 **Phase 2: Training** - Optimizing hyperparameters and training model...")
                    status_text.info("🔄 Step 4/7: Class imbalance handling...")
                    progress_bar.progress(35)

                    status_text.info("🔄 Step 5/7: Hyperparameter optimization and model training...")
                    progress_bar.progress(50)

                    # Phase 3: Optimization
                    phase_text.info("✨ **Phase 3: Optimization** - Fine-tuning model configuration...")
                    status_text.info("🔄 Step 6/7: Final model validation...")
                    progress_bar.progress(75)

                    status_text.info("🔄 Step 7/7: Classification-specific optimizations...")
                    progress_bar.progress(85)

                    # Run the pipeline
                    result = auto_pipeline.run()

                    progress_bar.progress(100)

                    # Store result in session state
                    st.session_state.automated_model_selection_training_result = result

                    if result['success']:
                        phase_text.success("✅ **All Phases Complete!**")
                        status_text.success("✅ Automated model selection and training completed successfully!")

                        # Mark automation as complete
                        st.session_state.automated_model_selection_training_completed = True

                        # Extract detailed metrics from result
                        details = result.get('details', {})
                        best_score = details.get('best_score', 0)
                        cv_metrics = details.get('cv_metrics', {})
                        stability_analysis = details.get('stability_analysis', {})

                        # Get actual CV folds and iterations used (could be adaptive)
                        actual_cv_folds = auto_pipeline.cv_folds if hasattr(auto_pipeline, 'cv_folds') else cv_folds
                        actual_n_iter = auto_pipeline.n_iter if hasattr(auto_pipeline, 'n_iter') else n_iter

                        # Log completion with detailed metrics
                        st.session_state.logger.log_calculation(
                            "Automated Model Selection & Training Completed",
                            {
                                "model_type": st.session_state.builder.model['type'],
                                "problem_type": details.get('problem_type', 'Unknown'),
                                "best_score": float(best_score) if best_score else 0,
                                "cv_metrics": cv_metrics,
                                "stability_level": stability_analysis.get('level', 'Unknown'),
                                "optimization_method": optimization_method,
                                "cv_folds": actual_cv_folds,
                                "n_iter": actual_n_iter,
                                "imbalance_handled": details.get('imbalance_handled', False),
                                "calibration_applied": details.get('calibration_applied', False),
                                "threshold_optimized": details.get('threshold_optimized', False),
                                "timestamp": datetime.now().isoformat()
                            }
                        )

                        # Create comprehensive journey point with all key metrics
                        journey_details = {
                            'Model Type': st.session_state.builder.model['type'],
                            'Problem Type': details.get('problem_type', 'Unknown').replace('_', ' ').title(),
                            'Best Score': f"{best_score:.4f}" if best_score else 'N/A',
                            'Stability': stability_analysis.get('level', 'Unknown'),
                            'Optimization Method': optimization_method,
                            'CV Folds': str(actual_cv_folds) if actual_cv_folds else 'Adaptive',
                            'Iterations': str(actual_n_iter) if actual_n_iter else 'Adaptive',
                            'Selection Reason': details.get('selection_reason', 'N/A')
                        }

                        # Add CV metrics to details
                        if cv_metrics:
                            for metric, value in cv_metrics.items():
                                if isinstance(value, (int, float)):
                                    journey_details[f'CV {metric}'] = f"{value:.4f}"

                        # Add optimization flags
                        if details.get('imbalance_handled'):
                            journey_details['Imbalance Handling'] = details.get('imbalance_method', 'Applied')
                        if details.get('calibration_applied'):
                            journey_details['Calibration'] = f"{details.get('calibration_method', 'Unknown').title()} ({details.get('calibration_improvement', 0):.1f}% improvement)"
                        if details.get('threshold_optimized'):
                            journey_details['Optimal Threshold'] = f"{details.get('optimal_threshold', 0.5):.3f}"

                        st.session_state.logger.log_journey_point(
                            stage="MODEL_SELECTION",
                            decision_type="AUTOMATED_MODEL_SELECTION_TRAINING_COMPLETED",
                            description="Automated model selection and training completed successfully",
                            details=journey_details,
                            parent_id=None
                        )

                        st.success("✅ Automated model selection and training completed successfully!")
                        st.rerun()

                    else:
                        # Automation failed
                        phase_text.error("❌ **Automation Failed**")
                        status_text.error(f"❌ {result.get('summary', 'Unknown error')}")

                        # Show which step failed
                        last_step = result.get('details', {}).get('last_completed_step', 'Unknown')
                        st.error(f"**Failed at:** {last_step}")

                        # Show error details
                        error_msg = result.get('error', 'Unknown error')
                        st.error(f"**Error:** {error_msg}")

                        # Provide manual fallback option
                        st.warning("⚠️ You can proceed to manual model selection and training to complete this step.")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("↩️ Return to Manual Model Selection", key="return_to_manual"):
                                st.rerun()
                        with col2:
                            if st.button("➡️ Proceed to Manual Training", key="proceed_to_manual_training"):
                                st.switch_page("pages/6_Model_Training.py")

                        # Log failure
                        st.session_state.logger.log_error(
                            "Automated Model Selection & Training Failed",
                            {
                                "error": error_msg,
                                "last_step": last_step,
                                "timestamp": datetime.now().isoformat()
                            }
                        )

            except Exception as e:
                st.error(f"❌ Error during automated model selection and training: {str(e)}")

                # Show detailed error
                import traceback
                with st.expander("View Error Details"):
                    st.code(traceback.format_exc())

                # Provide manual fallback option
                st.warning("⚠️ You can proceed to manual model selection and training to complete this step.")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("↩️ Return to Manual Model Selection", key="return_to_manual_error"):
                        st.rerun()
                with col2:
                    if st.button("➡️ Proceed to Manual Training", key="proceed_to_manual_training_error"):
                        st.switch_page("pages/6_Model_Training.py")

                # Log error
                st.session_state.logger.log_error(
                    "Automated Model Selection & Training Error",
                    {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now().isoformat()
                    }
                )


def render_automation_dashboard(result):
    """
    Render a comprehensive dashboard showing all automation results.

    This dashboard displays:
    - Summary metrics (top-level cards)
    - Phase 1: Selection (expandable)
    - Phase 2: Training (expandable)
    - Phase 3: Optimization (expandable)
    - Download report button

    Args:
        result: The automation result dictionary
    """
    st.write("## 📊 Automation Dashboard")

    details = result.get('details', {})

    # Top-level summary metrics
    st.markdown("### 📈 Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Model Selected",
            value=details.get('selected_model', 'N/A'),
            help="Final model selected by automation"
        )

    with col2:
        best_score = details.get('best_score', 0)
        st.metric(
            label="Training Score",
            value=f"{best_score:.4f}" if best_score else "N/A",
            help="Best cross-validation score achieved"
        )

    with col3:
        stability = details.get('stability_analysis', {}).get('level', 'Unknown')
        st.metric(
            label="Model Stability",
            value=stability,
            help="Model stability assessment"
        )

    with col4:
        problem_type = details.get('problem_type', 'Unknown')
        st.metric(
            label="Problem Type",
            value=problem_type.replace('_', ' ').title(),
            help="Detected problem type"
        )

    # Phase 1: Selection
    st.markdown("---")
    st.markdown("### 📋 Automation Phases")

    with st.expander("📊 **Phase 1: Selection** - Model Selection Process", expanded=False):
        st.markdown("#### Problem Detection")
        st.write(f"- **Problem Type:** {details.get('problem_type', 'N/A')}")
        st.write(f"- **XGBoost Compatible:** {'Yes' if details.get('xgboost_compatible') else 'No'}")

        st.markdown("#### Model Recommendation")
        recommendation = details.get('recommendation', {})
        st.write(f"- **Recommended Model:** {recommendation.get('recommended_model', 'N/A')}")
        if recommendation.get('reasons'):
            st.write("- **Reasoning:**")
            for reason in recommendation['reasons']:
                st.write(f"  - {reason}")

        st.markdown("#### Model Comparison")
        comparison = details.get('comparison_results', {})
        if comparison:
            st.write(f"- **Best Performer:** {comparison.get('best_model', 'N/A')} ({comparison.get('best_score', 0):.4f} {comparison.get('best_metric', '')})")

            # Show comparison results table if available
            if 'results_df' in comparison:
                results_data = comparison['results_df']
                if results_data:
                    st.dataframe(pd.DataFrame(results_data), width='stretch')

        st.markdown("#### Final Selection")
        st.write(f"- **Selected Model:** {details.get('selected_model', 'N/A')}")
        st.write(f"- **Selection Reason:** {details.get('selection_reason', 'N/A')}")

    # Phase 2: Training
    with st.expander("🎯 **Phase 2: Training** - Model Training Process", expanded=False):
        st.markdown("#### Imbalance Handling")
        if details.get('imbalance_handled'):
            imbalance_details = details.get('imbalance_details', {})
            st.write(f"- **Method Applied:** {details.get('imbalance_method', 'N/A')}")
            st.write(f"- **Original Samples:** {imbalance_details.get('original_samples', 'N/A')}")
            st.write(f"- **Resampled Samples:** {imbalance_details.get('new_samples', 'N/A')}")
            st.write(f"- **Imbalance Ratio Before:** {imbalance_details.get('imbalance_ratio_before', 0):.2f}:1")
            st.write(f"- **Imbalance Ratio After:** {imbalance_details.get('imbalance_ratio_after', 0):.2f}:1")
        else:
            st.info("No imbalance handling applied")

        st.markdown("#### Hyperparameter Optimization")
        training_results = details.get('training_results', {})
        if training_results:
            info = training_results.get('info', {})
            st.write(f"- **Best Score:** {details.get('best_score', 0):.4f}")

            # Best Parameters in expandable section
            with st.expander("View Best Hyperparameters", expanded=False):
                best_params = details.get('best_params', {})
                if best_params:
                    params_df = pd.DataFrame([
                        {'Parameter': k, 'Value': str(v)}
                        for k, v in best_params.items()
                    ])
                    st.dataframe(params_df, width='stretch', hide_index=True)
                else:
                    st.write("No parameters available")

            # CV Metrics in table with visualization
            cv_metrics = details.get('cv_metrics', {})
            cv_results = details.get('cv_results', {})

            if cv_metrics:
                st.markdown("##### Cross-Validation Metrics")

                # Create metrics table
                metrics_data = []
                for metric, value in cv_metrics.items():
                    if isinstance(value, (int, float)):
                        metrics_data.append({
                            'Metric': metric.replace('_', ' ').title(),
                            'Mean': f"{value:.4f}",
                            'Score': float(value)
                        })

                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)

                    # Display as table
                    st.dataframe(
                        metrics_df[['Metric', 'Mean']],
                        width='stretch',
                        hide_index=True
                    )

                    # Add fold-by-fold visualization if cv_results available
                    import plotly.graph_objects as go
                    import numpy as np

                    if cv_results and isinstance(cv_results, dict):
                        # Create fold-by-fold visualization
                        # cv_results structure: {'test_metric_name': [fold1, fold2, ...]}
                        st.markdown("##### Performance Across CV Folds")

                        fig = go.Figure()

                        # Extract fold scores for each metric
                        for metric_name, scores in cv_results.items():
                            if isinstance(scores, (list, np.ndarray)) and len(scores) > 0:
                                # Clean up metric name
                                display_name = metric_name.replace('test_', '').replace('_', ' ').title()
                                fold_numbers = list(range(1, len(scores) + 1))

                                fig.add_trace(go.Scatter(
                                    x=fold_numbers,
                                    y=scores,
                                    mode='lines+markers',
                                    name=display_name,
                                    marker=dict(size=8),
                                    line=dict(width=2)
                                ))

                        fig.update_layout(
                            title="Metric Performance Across CV Folds",
                            xaxis_title="Fold Number",
                            yaxis_title="Score",
                            yaxis_range=[0, 1.1],
                            height=400,
                            hovermode='x unified',
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=0.99,
                                xanchor="right",
                                x=0.99
                            )
                        )

                        st.plotly_chart(fig, config={'responsive': True})
                    else:
                        # Fallback to simple bar chart of mean metrics
                        fig = go.Figure(data=[
                            go.Bar(
                                x=metrics_df['Metric'],
                                y=metrics_df['Score'],
                                marker_color='lightblue',
                                text=metrics_df['Mean'],
                                textposition='outside'
                            )
                        ])

                        fig.update_layout(
                            title="Cross-Validation Mean Metrics",
                            xaxis_title="Metric",
                            yaxis_title="Score",
                            yaxis_range=[0, 1.1],
                            height=400,
                            showlegend=False
                        )

                        st.plotly_chart(fig, config={'responsive': True})

        st.markdown("#### Stability Analysis")
        stability = details.get('stability_analysis', {})
        cv_std = details.get('cv_std', stability.get('cv_std', 0)) if stability else 0
        if stability:
            st.write(f"- **Stability Level:** {stability.get('level', 'Unknown')}")
            st.write(f"- **CV Std:** {cv_std:.4f}")
            if stability.get('recommendations'):
                st.write("- **Recommendations:**")
                for rec in stability['recommendations']:
                    st.write(f"  - {rec}")

    # Phase 3: Optimization
    with st.expander("✨ **Phase 3: Optimization** - Model Fine-Tuning", expanded=False):
        st.markdown("#### Calibration")
        if details.get('calibration_applied'):
            calibration_comparison = details.get('calibration_comparison', {})
            st.write(f"- **Method Applied:** {details.get('calibration_method', 'N/A').title()}")
            st.write(f"- **Improvement:** {details.get('calibration_improvement', 0):.2f}%")

            if calibration_comparison:
                st.write(f"- **Methods Tested:** {', '.join(calibration_comparison.get('methods_tested', []))}")
                st.write(f"- **Best Method:** {calibration_comparison.get('best_method', 'N/A')}")
        else:
            reason = details.get('calibration_reason', 'Not applied')
            st.info(f"Calibration not applied: {reason}")

        st.markdown("#### Threshold Optimization")
        if details.get('threshold_optimized'):
            st.write(f"- **Optimal Threshold:** {details.get('optimal_threshold', 0.5):.3f}")
            st.write(f"- **Optimization Criterion:** {details.get('threshold_criterion', 'N/A')}")
            st.write(f"- **Improvement:** {details.get('threshold_improvement', 0) * 100:.1f}%")

            # Show baseline vs optimal metrics
            baseline_metrics = details.get('baseline_metrics', {})
            optimal_metrics = details.get('optimal_metrics', {})

            if baseline_metrics and optimal_metrics:
                st.markdown("##### Performance Comparison")

                # Get the optimized criterion
                threshold_criterion_internal = details.get('threshold_criterion_internal', 'f1')

                # Build comparison table with proper optimal values
                comparison_data = []
                for metric_name, baseline_value in baseline_metrics.items():
                    # Get optimal value from optimal_metrics (now includes all metrics)
                    optimal_value = optimal_metrics.get(metric_name, baseline_value)

                    difference = optimal_value - baseline_value
                    difference_pct = (difference / baseline_value * 100) if baseline_value > 0 else 0

                    comparison_data.append({
                        'Metric': metric_name.replace('_', ' ').title(),
                        'Baseline (0.5)': f"{baseline_value:.4f}",
                        'Optimal ({:.3f})'.format(details.get('optimal_threshold', 0.5)): f"{optimal_value:.4f}",
                        'Difference': f"{difference:+.4f}",
                        'Change %': f"{difference_pct:+.2f}%"
                    })

                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, width='stretch', hide_index=True)

                # Add note about optimization
                st.caption(f"⚡ Optimized for: **{details.get('threshold_criterion', 'N/A')}**")

                # Visualization of the comparison
                import plotly.graph_objects as go

                metrics_list = [item['Metric'] for item in comparison_data]
                baseline_vals = [float(item['Baseline (0.5)']) for item in comparison_data]
                optimal_vals = [float(item['Optimal ({:.3f})'.format(details.get('optimal_threshold', 0.5))]) for item in comparison_data]

                fig = go.Figure(data=[
                    go.Bar(
                        name='Baseline (0.5)',
                        x=metrics_list,
                        y=baseline_vals,
                        marker_color='lightcoral',
                        text=[f"{v:.4f}" for v in baseline_vals],
                        textposition='outside'
                    ),
                    go.Bar(
                        name=f'Optimal ({details.get("optimal_threshold", 0.5):.3f})',
                        x=metrics_list,
                        y=optimal_vals,
                        marker_color='lightgreen',
                        text=[f"{v:.4f}" for v in optimal_vals],
                        textposition='outside'
                    )
                ])

                fig.update_layout(
                    title="Threshold Optimization - Performance Comparison",
                    xaxis_title="Metric",
                    yaxis_title="Score",
                    yaxis_range=[0, 1.1],
                    barmode='group',
                    height=400,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                st.plotly_chart(fig, config={'responsive': True})
            else:
                st.info("No performance comparison data available")
        else:
            reason = details.get('threshold_reason', 'Not applicable or minimal improvement')
            st.info(f"Threshold optimization: {reason}")

    # Download report
    st.markdown("---")
    st.markdown("### 📥 Download Report")

    report_content = _generate_automation_report(result, st.session_state.builder)

    st.download_button(
        label="📄 Download Model Selection and Training Report",
        data=report_content,
        file_name=f"model_selection_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        help="Download a detailed report of all automation steps and results"
    )


def _generate_automation_report(result, builder) -> str:
    """
    Generate a comprehensive markdown report of the automation process.

    Args:
        result: The automation result dictionary
        builder: The Builder instance

    Returns:
        Detailed markdown report as a string
    """
    details = result.get('details', {})

    report = "# Automated Model Selection & Training Report\n\n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Executive Summary
    report += "## Executive Summary\n\n"
    report += f"- **Selected Model:** {details.get('selected_model', 'N/A')}\n"
    report += f"- **Problem Type:** {details.get('problem_type', 'N/A').replace('_', ' ').title()}\n"
    report += f"- **Best Training Score:** {details.get('best_score', 0):.4f}\n"
    report += f"- **Model Stability:** {details.get('stability_analysis', {}).get('level', 'Unknown')}\n"
    cv_std = details.get('cv_std', 0)
    report += f"- **CV Standard Deviation:** {cv_std:.4f}\n"
    report += f"- **Selection Reason:** {details.get('selection_reason', 'N/A')}\n"
    report += f"- **Optimization Method:** {details.get('optimization_method', 'N/A').replace('_', ' ').title()}\n"

    # Add optimization flags
    optimization_flags = []
    if details.get('imbalance_handled'):
        optimization_flags.append(f"Imbalance Handling ({details.get('imbalance_method', 'N/A')})")
    if details.get('calibration_applied'):
        optimization_flags.append(f"Calibration ({details.get('calibration_method', 'N/A').title()})")
    if details.get('threshold_optimized'):
        optimization_flags.append(f"Threshold Optimization ({details.get('optimal_threshold', 0.5):.3f})")

    if optimization_flags:
        report += f"- **Optimizations Applied:** {', '.join(optimization_flags)}\n"

    report += "\n"

    # Phase 1: Selection
    report += "## Phase 1: Selection\n\n"

    report += "### Problem Detection\n\n"
    report += f"- Problem Type: {details.get('problem_type', 'N/A')}\n"
    report += f"- XGBoost Compatible: {'Yes' if details.get('xgboost_compatible') else 'No'}\n\n"

    report += "### Model Recommendation\n\n"
    recommendation = details.get('recommendation', {})
    report += f"- Recommended Model: {recommendation.get('recommended_model', 'N/A')}\n"
    if recommendation.get('reasons'):
        report += "- Reasoning:\n"
        for reason in recommendation['reasons']:
            report += f"  - {reason}\n"
    report += "\n"

    report += "### Model Comparison\n\n"
    comparison = details.get('comparison_results', {})
    if comparison:
        report += f"- Best Performer: {comparison.get('best_model', 'N/A')}\n"
        report += f"- Best Score: {comparison.get('best_score', 0):.4f}\n"
        report += f"- Metric: {comparison.get('best_metric', 'N/A')}\n\n"

    report += "### Final Selection\n\n"
    report += f"- Selected Model: {details.get('selected_model', 'N/A')}\n"
    report += f"- Selection Reason: {details.get('selection_reason', 'N/A')}\n\n"

    # Phase 2: Training
    report += "## Phase 2: Training\n\n"

    report += "### Imbalance Handling\n\n"
    if details.get('imbalance_handled'):
        imbalance_details = details.get('imbalance_details', {})
        report += f"- Method Applied: {details.get('imbalance_method', 'N/A')}\n"
        report += f"- Original Samples: {imbalance_details.get('original_samples', 'N/A')}\n"
        report += f"- Resampled Samples: {imbalance_details.get('new_samples', 'N/A')}\n"
        report += f"- Imbalance Ratio Before: {imbalance_details.get('imbalance_ratio_before', 0):.2f}:1\n"
        report += f"- Imbalance Ratio After: {imbalance_details.get('imbalance_ratio_after', 0):.2f}:1\n\n"
    else:
        report += "- No imbalance handling applied\n\n"

    report += "### Hyperparameter Optimization\n\n"
    report += f"- Best Score: {details.get('best_score', 0):.4f}\n"
    report += f"- Optimization Method: {details.get('optimization_method', 'N/A').replace('_', ' ').title()}\n"

    # Best Parameters
    best_params = details.get('best_params', {})
    if best_params:
        report += "- Best Parameters:\n"
        for param, value in best_params.items():
            report += f"  - {param}: {value}\n"
    report += "\n"

    cv_metrics = details.get('cv_metrics', {})
    cv_results = details.get('cv_results', {})

    if cv_metrics:
        report += "**Cross-Validation Metrics:**\n\n"
        # Separate fold_scores from other metrics
        for metric, value in cv_metrics.items():
            if metric == 'fold_scores' and isinstance(value, list):
                # Show fold-by-fold scores
                report += f"- Fold Scores: {[f'{v:.4f}' for v in value]}\n"
            elif isinstance(value, (int, float)):
                report += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
            elif metric != 'fold_scores':  # Skip fold_scores if not a list
                report += f"- {metric.replace('_', ' ').title()}: {value}\n"
        report += "\n"

    # Add fold-by-fold analysis if available
    if cv_results and isinstance(cv_results, dict):
        report += "**Fold-by-Fold Performance:**\n\n"
        for metric_name, scores in cv_results.items():
            if isinstance(scores, (list, np.ndarray)) and len(scores) > 0:
                display_name = metric_name.replace('test_', '').replace('_', ' ').title()
                report += f"- {display_name}:\n"
                for i, score in enumerate(scores, 1):
                    report += f"  - Fold {i}: {score:.4f}\n"
        report += "\n"

    report += "### Stability Analysis\n\n"
    stability = details.get('stability_analysis', {})
    cv_std = details.get('cv_std', stability.get('cv_std', 0)) if stability else 0
    if stability:
        report += f"- Stability Level: {stability.get('level', 'Unknown')}\n"
        report += f"- CV Standard Deviation: {cv_std:.4f}\n"
        report += f"- Stability Score: {stability.get('score', 0):.4f}\n"
        if stability.get('recommendations'):
            report += "- Recommendations:\n"
            for rec in stability['recommendations']:
                report += f"  - {rec}\n"
        report += "\n"

    # Phase 3: Optimization
    report += "## Phase 3: Optimization\n\n"

    report += "### Calibration\n\n"
    if details.get('calibration_applied'):
        report += f"- Method Applied: {details.get('calibration_method', 'N/A').title()}\n"
        report += f"- Improvement: {details.get('calibration_improvement', 0):.2f}%\n"

        calibration_comparison = details.get('calibration_comparison', {})
        if calibration_comparison:
            report += f"- Methods Tested: {', '.join(calibration_comparison.get('methods_tested', []))}\n"
            report += f"- Best Method: {calibration_comparison.get('best_method', 'N/A')}\n"
        report += "\n"
    else:
        reason = details.get('calibration_reason', 'Not applied')
        report += f"- Status: Not applied ({reason})\n\n"

    report += "### Threshold Optimization\n\n"
    if details.get('threshold_optimized'):
        report += f"- Optimal Threshold: {details.get('optimal_threshold', 0.5):.3f}\n"
        report += f"- Optimization Criterion: {details.get('threshold_criterion', 'N/A')}\n"
        report += f"- Criterion (Internal): {details.get('threshold_criterion_internal', 'N/A')}\n"
        report += f"- Overall Improvement: {details.get('threshold_improvement', 0) * 100:.1f}%\n\n"

        # Add detailed performance comparison
        baseline_metrics = details.get('baseline_metrics', {})
        optimal_metrics = details.get('optimal_metrics', {})

        if baseline_metrics and optimal_metrics:
            report += "**Performance Comparison:**\n\n"
            report += "| Metric | Baseline (0.5) | Optimal ({:.3f}) | Difference | Change % |\n".format(details.get('optimal_threshold', 0.5))
            report += "|--------|---------------|-----------------|------------|----------|\n"

            for metric_name, baseline_value in baseline_metrics.items():
                optimal_value = optimal_metrics.get(metric_name, baseline_value)
                difference = optimal_value - baseline_value
                difference_pct = (difference / baseline_value * 100) if baseline_value > 0 else 0

                report += f"| {metric_name.replace('_', ' ').title()} | {baseline_value:.4f} | {optimal_value:.4f} | {difference:+.4f} | {difference_pct:+.2f}% |\n"

            report += "\n"
            report += f"*Note: Optimized for {details.get('threshold_criterion', 'N/A')}*\n\n"
    else:
        reason = details.get('threshold_reason', 'Not applicable')
        report += f"- Status: Not applied ({reason})\n\n"

    # Final Model Information
    report += "## Final Model Information\n\n"
    report += f"- Model Type: {builder.model['type']}\n"
    report += f"- Problem Type: {builder.model['problem_type']}\n"
    report += f"- Training Samples: {len(builder.y_train):,}\n"
    report += f"- Testing Samples: {len(builder.y_test):,}\n"
    
    # Handle feature_names being None - fallback to X_train shape
    if builder.feature_names is not None:
        report += f"- Features: {len(builder.feature_names)}\n"
    elif builder.X_train is not None:
        report += f"- Features: {builder.X_train.shape[1]}\n"
    else:
        report += "- Features: N/A\n"
    
    report += "\n"

    report += "---\n\n"
    report += "*Report generated by ML Builder - Automated Model Selection & Training*\n"

    return report
