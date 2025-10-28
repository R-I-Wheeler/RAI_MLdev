import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import export_text, plot_tree
import io
import base64
from components.model_explanation.explanation_utils.shap_visualization_utils import create_shap_summary_plot, create_interactive_shap_summary_plot

def extract_base_model(model):
    """Extract the base estimator from wrapped models (calibrated, threshold-adjusted, etc.)."""
    # Check if it's a calibrated model first - extract the base estimator
    if hasattr(model, 'calibrated_classifiers_') or 'Calibrated' in model.__class__.__name__:
        base_model = None

        # Try different ways to access the base estimator
        if hasattr(model, 'base_estimator'):
            base_model = model.base_estimator
        elif hasattr(model, 'estimator'):
            base_model = model.estimator
        elif hasattr(model, 'calibrated_classifiers_'):
            try:
                if len(model.calibrated_classifiers_) > 0:
                    first_calibrated = model.calibrated_classifiers_[0]
                    if hasattr(first_calibrated, 'base_estimator'):
                        base_model = first_calibrated.base_estimator
                    elif hasattr(first_calibrated, 'estimator'):
                        base_model = first_calibrated.estimator
            except:
                pass

        if base_model is not None:
            return extract_base_model(base_model)  # Recursively unwrap

    # Check for other wrapper types (e.g., threshold-adjusted classifiers)
    if hasattr(model, 'estimator') and not hasattr(model, 'tree_') and not hasattr(model, 'estimators_'):
        return extract_base_model(model.estimator)

    return model

def supports_decision_tree_visualization(model):
    """Check if model supports decision tree visualization."""
    # Extract base model from any wrappers
    base_model = extract_base_model(model)

    # Single decision tree models
    if hasattr(base_model, 'tree_') or 'DecisionTree' in base_model.__class__.__name__:
        return True, "single"

    # Check for XGBoost and LightGBM models - these don't support reliable visualization
    model_name = base_model.__class__.__name__
    excluded_models = ['XGB', 'LightGBM', 'LGBM', 'HistGradientBoosting', 'CatBoost']
    for excluded in excluded_models:
        if excluded in model_name:
            return False, None

    # Ensemble models with individual trees (excluding XGBoost and LightGBM)
    ensemble_indicators = [
        ('RandomForest', 'ensemble'),
        ('ExtraTree', 'ensemble'),
        ('GradientBoosting', 'ensemble')
    ]

    for indicator, tree_type in ensemble_indicators:
        if indicator in model_name:
            return True, tree_type

    return False, None

def create_decision_tree_plot(model, feature_names, max_depth=3, tree_index=0, display_mode="Visual Tree"):
    """Create decision tree visualization plot and/or text representation."""
    try:
        # Determine tree type and get the actual tree
        is_tree_compatible, tree_type = supports_decision_tree_visualization(model)
        
        if not is_tree_compatible:
            return None, "Model does not support decision tree visualization", None
        
        # Get the actual tree to visualize
        tree_model = None
        ensemble_info = ""
        tree_fig = None
        tree_text = None
        class_names = None
        
        # Get class names for classification models - ensure we get them from the original model
        original_model = model  # Keep reference to original model
        class_names = None
        
        if hasattr(model, "classes_"):
            class_names = [f"Class_{i} ({cls})" for i, cls in enumerate(model.classes_)]
            print(f"DEBUG: Found class names from model: {class_names}")  # Debug print
        else:
            print(f"DEBUG: No classes_ attribute found on model {model.__class__.__name__}")  # Debug print
        
        # First check if this is a calibrated model and extract the base estimator
        if hasattr(model, 'calibrated_classifiers_') or 'Calibrated' in model.__class__.__name__:
            # This is a calibrated model - extract the underlying base estimator
            base_model = None
            
            if hasattr(model, 'base_estimator'):
                base_model = model.base_estimator
            elif hasattr(model, 'estimator'):
                base_model = model.estimator
            elif hasattr(model, 'calibrated_classifiers_'):
                try:
                    if len(model.calibrated_classifiers_) > 0:
                        first_calibrated = model.calibrated_classifiers_[0]
                        if hasattr(first_calibrated, 'base_estimator'):
                            base_model = first_calibrated.base_estimator
                        elif hasattr(first_calibrated, 'estimator'):
                            base_model = first_calibrated.estimator
                except:
                    pass
            
            if base_model is not None:
                # Recursively call this function with the base model, but preserve class names from original
                if not hasattr(base_model, "classes_") and hasattr(model, "classes_"):
                    # Transfer class names to base model if it doesn't have them
                    base_model.classes_ = model.classes_
                return create_decision_tree_plot(base_model, feature_names, max_depth, tree_index, display_mode)
            else:
                return None, "Could not extract base estimator from calibrated model", None
        
        if tree_type == "single":
            tree_model = model
            ensemble_info = ""
        else:  # ensemble
            model_name = model.__class__.__name__
            
            # Handle different ensemble types
            if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                # RandomForest, ExtraTree, GradientBoosting
                print(f"DEBUG: Processing ensemble with {len(model.estimators_)} estimators")
                print(f"DEBUG: Estimator shape/type: {type(model.estimators_[0])}")
                
                # Special handling for GradientBoosting multiclass - they store trees as [n_boosting_rounds, n_classes]
                if 'GradientBoosting' in model_name and hasattr(model, 'classes_') and len(model.classes_) > 2:
                    # For multiclass GradientBoosting, estimators_ is shaped (n_estimators, n_classes)
                    n_classes = len(model.classes_)
                    boosting_round = tree_index // n_classes
                    class_idx = tree_index % n_classes
                    
                    if boosting_round < len(model.estimators_):
                        tree_model = model.estimators_[boosting_round][class_idx]
                        ensemble_info = f"Tree {tree_index + 1} (Round {boosting_round + 1}, Class {class_names[class_idx] if class_names else class_idx}) from {model_name}"
                    else:
                        return None, f"Tree index {tree_index} out of range for {model_name}", None
                else:
                    # Regular ensemble handling (RandomForest, etc.)
                    tree_model = model.estimators_[tree_index]
                    if hasattr(tree_model, '__len__'):  # Some models store as [tree, ]
                        tree_model = tree_model[0] if len(tree_model) > 0 else tree_model
                    ensemble_info = f"Tree {tree_index + 1} of {len(model.estimators_)} from {model_name}"
                
            elif 'XGB' in model_name:
                # XGBoost - use text representation since tree structure is different
                try:
                    # Get booster and dump tree
                    booster = model.get_booster()
                    tree_dump = booster.get_dump(dump_format='text')
                    if tree_index < len(tree_dump):
                        tree_text = tree_dump[tree_index]
                        # Add class labeling information for multiclass XGBoost
                        if class_names and len(class_names) > 2:
                            tree_text = f"# XGBoost Tree for Multiclass Classification\n# Classes: {', '.join(class_names)}\n# Note: This tree contributes to the prediction for one of the classes\n\n{tree_text}"
                        
                        # Create a simple text visualization for XGBoost
                        if display_mode in ["Visual Tree", "Both"]:
                            fig = go.Figure()
                            fig.add_annotation(
                                text=f"<pre>{tree_text[:2000]}{'...' if len(tree_text) > 2000 else ''}</pre>",
                                xref="paper", yref="paper",
                                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                                showarrow=False,
                                font=dict(family="monospace", size=10)
                            )
                            fig.update_layout(
                                title=f"XGBoost Decision Tree {tree_index + 1}",
                                height=600,
                                xaxis=dict(visible=False),
                                yaxis=dict(visible=False)
                            )
                            return fig, f"Tree {tree_index + 1} from XGBoost ensemble", tree_text
                        else:
                            return None, f"Tree {tree_index + 1} from XGBoost ensemble", tree_text
                except Exception as e:
                    return None, f"XGBoost tree visualization not available: {str(e)}", None
                    
            elif 'LightGBM' in model_name or 'LGBM' in model_name:
                # LightGBM - use text representation
                try:
                    # Get tree as text
                    tree_text = model.booster_.dump_model()['tree_info'][tree_index]['tree_structure']
                    tree_str = str(tree_text)[:2000]
                    
                    # Add class labeling information for multiclass LightGBM
                    if class_names and len(class_names) > 2:
                        tree_str = f"# LightGBM Tree for Multiclass Classification\n# Classes: {', '.join(class_names)}\n# Note: This tree contributes to the prediction for one of the classes\n\n{tree_str}"
                    
                    if display_mode in ["Visual Tree", "Both"]:
                        fig = go.Figure()
                        fig.add_annotation(
                            text=f"<pre>{tree_str}{'...' if len(str(tree_text)) > 2000 else ''}</pre>",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, xanchor='center', yanchor='middle',
                            showarrow=False,
                            font=dict(family="monospace", size=10)
                        )
                        fig.update_layout(
                            title=f"LightGBM Decision Tree {tree_index + 1}",
                            height=600,
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False)
                        )
                        return fig, f"Tree {tree_index + 1} from LightGBM ensemble", tree_str
                    else:
                        return None, f"Tree {tree_index + 1} from LightGBM ensemble", tree_str
                except Exception as e:
                    return None, f"LightGBM tree visualization not available: {str(e)}", None
            else:
                return None, f"Tree visualization not implemented for {model_name}", None
        
        # For single trees and sklearn ensemble trees, use sklearn's plot_tree
        if tree_model is not None and hasattr(tree_model, 'tree_'):
            # Determine actual depth to use
            if max_depth is None:
                actual_max_depth = tree_model.tree_.max_depth  # Show full depth
            else:
                actual_max_depth = min(max_depth, tree_model.tree_.max_depth)
            
            # Generate text representation with class names if available
            try:
                print(f"DEBUG: Calling export_text with class_names: {class_names}")  # Debug print
                if class_names:
                    # For ensemble trees, sklearn might not accept class names directly
                    # Try with class names first, fall back to without if it fails
                    try:
                        tree_text = export_text(tree_model, feature_names=feature_names, max_depth=actual_max_depth, class_names=class_names)
                    except Exception as class_name_error:
                        print(f"DEBUG: export_text failed with class_names, trying without: {class_name_error}")
                        tree_text = export_text(tree_model, feature_names=feature_names, max_depth=actual_max_depth)
                        # Add class information manually at the top
                        if class_names and len(class_names) > 2:
                            class_info = f"# Class Labels: {', '.join(class_names)}\n# Note: Individual ensemble trees may show class indices instead of names\n\n"
                            tree_text = class_info + tree_text
                else:
                    tree_text = export_text(tree_model, feature_names=feature_names, max_depth=actual_max_depth)
            except Exception as e:
                print(f"DEBUG: export_text failed completely: {str(e)}")
                tree_text = f"Text representation not available: {str(e)}"
            
            # Generate visual representation if requested
            tree_fig = None
            if display_mode in ["Visual Tree", "Both"]:
                try:
                    # Create matplotlib figure - larger size for full depth trees
                    if max_depth is None:
                        plt.figure(figsize=(30, 40))  # Larger figure for full trees
                    else:
                        plt.figure(figsize=(20, 12))
                    
                    print(f"DEBUG: Calling plot_tree with class_names: {class_names}")  # Debug print
                    if class_names:
                        try:
                            plot_tree(
                                tree_model,
                                feature_names=feature_names,
                                class_names=class_names,  # Add class names to visual plot
                                filled=True,
                                rounded=True,
                                fontsize=10,
                                max_depth=actual_max_depth
                            )
                        except Exception as class_name_error:
                            print(f"DEBUG: plot_tree failed with class_names, trying without: {class_name_error}")
                            plot_tree(
                                tree_model,
                                feature_names=feature_names,
                                filled=True,
                                rounded=True,
                                fontsize=10,
                                max_depth=actual_max_depth
                            )
                    else:
                        plot_tree(
                            tree_model,
                            feature_names=feature_names,
                            filled=True,
                            rounded=True,
                            fontsize=10,
                            max_depth=actual_max_depth
                        )
                    
                    depth_info = f" (Full Depth: {actual_max_depth})" if max_depth is None else f" (Max Depth: {actual_max_depth})"
                    plt.title(f"Decision Tree Visualization{depth_info}{' - ' + ensemble_info if ensemble_info else ''}")
                    
                    # Convert matplotlib plot to plotly-compatible format
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    plt.close()
                    
                    # Encode image to base64 for display
                    img_base64 = base64.b64encode(buf.read()).decode()
                    
                    # Create plotly figure to display the image
                    tree_fig = go.Figure()
                    
                    # Add the image
                    tree_fig.add_layout_image(
                        dict(
                            source=f"data:image/png;base64,{img_base64}",
                            xref="paper", yref="paper",
                            x=0, y=1, sizex=1, sizey=1,
                            xanchor="left", yanchor="top"
                        )
                    )
                    
                    depth_title_info = f" (Full Depth: {actual_max_depth})" if max_depth is None else f" (Max Depth: {actual_max_depth})"
                    # Adjust height based on tree depth
                    plot_height = 1200 if max_depth is None else 800
                    tree_fig.update_layout(
                        title=f"Decision Tree Visualization{depth_title_info}{' - ' + ensemble_info if ensemble_info else ''}",
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        height=plot_height,
                        margin=dict(l=0, r=0, t=50, b=0)
                    )
                except Exception as e:
                    tree_fig = None
                    if display_mode == "Visual Tree":
                        tree_text = f"Visual tree not available: {str(e)}\n\nFalling back to text representation:\n{tree_text}"
            
            return tree_fig, ensemble_info, tree_text
        
        # Fallback: create text representation only
        if tree_model is not None:
            try:
                # Handle full depth option for text export
                text_max_depth = max_depth if max_depth is not None else tree_model.tree_.max_depth
                if class_names:
                    try:
                        tree_text = export_text(tree_model, feature_names=feature_names, max_depth=text_max_depth, class_names=class_names)
                    except Exception as class_name_error:
                        print(f"DEBUG: fallback export_text failed with class_names: {class_name_error}")
                        tree_text = export_text(tree_model, feature_names=feature_names, max_depth=text_max_depth)
                        if class_names and len(class_names) > 2:
                            class_info = f"# Class Labels: {', '.join(class_names)}\n\n"
                            tree_text = class_info + tree_text
                else:
                    tree_text = export_text(tree_model, feature_names=feature_names, max_depth=text_max_depth)
                
                # Only create visual fallback if visual mode is requested
                tree_fig = None
                if display_mode in ["Visual Tree", "Both"]:
                    tree_fig = go.Figure()
                    tree_fig.add_annotation(
                        text=f"<pre>{tree_text}</pre>",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, xanchor='center', yanchor='middle',
                        showarrow=False,
                        font=dict(family="monospace", size=10)
                    )
                    tree_fig.update_layout(
                        title=f"Decision Tree Rules{' - ' + ensemble_info if ensemble_info else ''}",
                        height=600,
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False)
                    )
                
                return tree_fig, ensemble_info, tree_text
            except Exception as e:
                return None, f"Unable to create tree visualization: {str(e)}", None
        
        return None, "No tree structure found", None
        
    except Exception as e:
        return None, f"Error creating decision tree plot: {str(e)}", None

def render_feature_analysis(builder, result, importance_df=None, limitations_result=None):
    """Render the feature analysis tab"""
    # Use session state problem type if available
    if hasattr(st.session_state, 'problem_type'):
        problem_type = st.session_state.problem_type
    else:
        # Fallback to model's problem type
        problem_type = builder.model.get("problem_type", "unknown")
    
    st.header("Feature Analysis")
    
    if result and result.get("success"):
        st.success("Model explanation generated successfully")
        
        if 'feature_importances' in result:
            st.subheader("Feature Importance")
            importance_data = result['feature_importances']
            
            # Create DataFrame from importance data
            importance_df = pd.DataFrame(importance_data.items(), columns=['Feature', 'Importance'])
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance Rankings"
            )
            
            # Display appropriate interpretation based on problem type
            if problem_type in ["binary_classification", "multiclass_classification", "classification"]:
                st.info("For classification problems, higher importance values indicate features that are more influential in determining the predicted class.")
            elif problem_type == "regression":
                st.info("For regression problems, higher importance values indicate features that are more influential in determining the predicted value.")
            else:
                st.info("Feature importance shows which features have the most influence on model predictions.")
            
            fig.update_layout(height=max(400, len(importance_df) * 30))
            st.plotly_chart(fig, config={'responsive': True})
            
            # Display the data table
            st.subheader("Feature Importance Values")
            st.dataframe(importance_df, width='stretch')
    else:
        st.error(f"Feature analysis failed: {result.get('message', 'Unknown error') if result else 'No result available'}")
    
    st.markdown("""
        ### 🎯 Understanding Your Model's Decision Making
        
        This section shows you **how your model thinks** when making predictions. We'll break down:
        1. Which features matter most
        2. How each feature affects predictions
        3. The exact decision rules your model follows
    """)

    # Determine which tabs to show based on model capabilities
    available_tabs = ["Feature Importance", "Accumulated Local Effects"]
    
    # Check if model supports decision tree visualization
    if hasattr(builder, "model") and builder.model is not None:
        model_instance = builder.model.get("active_model") or builder.model["model"]
        has_decision_tree, tree_type = supports_decision_tree_visualization(model_instance)
        
        if has_decision_tree:
            available_tabs.append("Decision Tree Visualization")
    else:
        has_decision_tree = False
    
    # Create tabs with session state to maintain active tab across widget changes
    if 'feature_analysis_active_tab' not in st.session_state:
        st.session_state.feature_analysis_active_tab = 0
    
    # Ensure the active tab index is valid
    if st.session_state.feature_analysis_active_tab >= len(available_tabs):
        st.session_state.feature_analysis_active_tab = 0
    
    # Use pills instead of tabs for better state persistence
    selected_tab = st.pills(
        "Analysis Type:",
        available_tabs,
        default=available_tabs[st.session_state.feature_analysis_active_tab],
        key="feature_analysis_tab_selector"
    )
    
    # Update session state when tab changes
    if selected_tab != available_tabs[st.session_state.feature_analysis_active_tab]:
        st.session_state.feature_analysis_active_tab = available_tabs.index(selected_tab)
        # Reset decision tree tab tracking when switching tabs
        if 'dt_viz_tab_entered' in st.session_state:
            st.session_state.dt_viz_tab_entered = False
    
    # Tab 0: Feature Importance
    if selected_tab == "Feature Importance":
        st.subheader("📊 Feature Importance")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if result["shap_values"]["plot"] is not None:
                st.pyplot(result["shap_values"]["plot"])
                
                st.session_state.logger.log_journey_point(
                    stage="MODEL_EXPLANATION",
                    decision_type="MODEL_EXPLANATION",
                    description="SHAP Feature Importance",
                    details={"Feature Importances": result["shap_values"]["feature_importance"]},
                    parent_id=None
                )
        with col2:
            if problem_type == "regression":
                st.markdown("""
                    ### 📖 How to Read This Plot
                    
                    This bar chart shows feature importance based on SHAP values:

                    - The Y-axis lists features ranked by importance.
                    - The X-axis shows the mean absolute SHAP value for each feature.
                    
                    **Interpretation:**
                    - Longer bars indicate features with stronger impact on predictions
                    - Features at the top have the greatest influence on model outcomes
                    - The value represents how much a feature changes the prediction on average
                      (taking the absolute value of positive and negative impacts)
                """)
            else:  # classification
                st.markdown("""
                    ### 📖 How to Read This Plot
                    
                    This bar chart shows feature importance based on SHAP values:

                    - The Y-axis lists features ranked by importance.
                    - The X-axis shows the mean absolute SHAP value for each feature.
                    
                    **Interpretation:**
                    - Longer bars indicate features with stronger impact on predictions
                    - Features at the top have the greatest influence on model outputs
                    - The value represents how much a feature changes the prediction on average
                      (taking the absolute value of positive and negative contributions)
                """)
        
        # Add SHAP Summary Plot section
        st.markdown("---")  # Visual separator
        st.subheader("🎨 Interactive SHAP Summary Plot (Feature Impact Distribution)")
        
        # Check if we have the raw SHAP values and feature data for summary plot
        if ("raw_shap_values" in result["shap_values"] and 
            "feature_data" in result["shap_values"] and 
            "feature_names" in result["shap_values"]):
            
            # Customization controls - always visible, no expander
            st.markdown("**⚙️ Customize Plot:**")
            
            # Determine if we need a slider or can just show all features
            num_features = len(result["shap_values"]["feature_names"])
            
            if num_features <= 5:
                # Not enough features for a slider, just show all
                control_cols = st.columns([3, 3, 1])
                max_features = num_features
                with control_cols[0]:
                    st.info(f"📊 Showing all {num_features} features")
            else:
                # Enough features for customization
                control_cols = st.columns([2, 2, 2, 1])
                
                with control_cols[0]:
                    # Control for number of features to display
                    max_features = st.slider(
                        "Number of features",
                        min_value=min(5, num_features),
                        max_value=min(30, num_features),
                        value=min(20, num_features),
                        step=1,
                        help="Control how many top features are shown in the plot",
                        key="shap_max_features_slider"
                    )
            
            # Determine column indices based on whether slider is shown
            checkbox_col = 0 if num_features <= 5 else 1
            download_col = 1 if num_features <= 5 else 2
            
            with control_cols[checkbox_col]:
                # Option to select specific features
                show_feature_selector = st.checkbox(
                    "Select specific features",
                    value=False,
                    help="Choose exactly which features to display",
                    key="shap_feature_selector_checkbox"
                )
            
            with control_cols[download_col]:
                # Download data option - direct download button
                # Prepare data for download (pd already imported at top)
                shap_df = pd.DataFrame(
                    result["shap_values"]["raw_shap_values"],
                    columns=result["shap_values"]["feature_names"]
                )
                shap_df.insert(0, 'Sample_Index', range(len(shap_df)))
                
                csv = shap_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data",
                    data=csv,
                    file_name="shap_values.csv",
                    mime="text/csv",
                    help="Download SHAP values as CSV",
                    key="shap_download_button"
                )
            
            # Feature selector if enabled
            selected_features = None
            if show_feature_selector:
                # Only use the spacer column if we have enough columns (>5 features)
                if num_features > 5:
                    with control_cols[3]:
                        st.write("")  # Spacer
                st.markdown("")  # Small gap
                selected_features = st.multiselect(
                    "Select features to display:",
                    options=result["shap_values"]["feature_names"],
                    default=None,
                    help="Choose which features to show. Leave empty to show top features by importance.",
                    key="shap_feature_multiselect"
                )
                if not selected_features:
                    selected_features = None  # Fall back to top features
            
            st.markdown("")  # Small gap before plot
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                try:
                    # Generate the interactive SHAP summary plot
                    interactive_summary_plot = create_interactive_shap_summary_plot(
                        shap_values=result["shap_values"]["raw_shap_values"],
                        feature_data=result["shap_values"]["feature_data"],
                        feature_names=result["shap_values"]["feature_names"],
                        problem_type=problem_type,
                        max_display=max_features,
                        selected_features=selected_features
                    )
                    
                    st.plotly_chart(interactive_summary_plot, config={'responsive': True})
                    
                    # Add helpful tips below the plot
                    st.caption("""
                        💡 **Interactive Tips:** 
                        • Hover over points for detailed information 
                        • Zoom by dragging a box 
                        • Double-click to reset view 
                        • Click legend items to toggle features
                    """)
                    
                    #st.session_state.logger.log_journey_point(
                    #    stage="MODEL_EXPLANATION",
                    #    decision_type="MODEL_EXPLANATION",
                    #    description="Interactive SHAP Summary Plot",
                    #    details={
                    #        "plot_type": "interactive_summary",
                    #        "features_shown": max_features if not selected_features else len(selected_features),
                    #        "custom_features": bool(selected_features)
                    #    },
                    #    parent_id=None
                    #)
                except Exception as e:
                    st.warning(f"Could not generate interactive SHAP summary plot: {str(e)}")
                    # Fallback to static plot
                    try:
                        st.info("Falling back to static plot...")
                        summary_plot = create_shap_summary_plot(
                            shap_values=result["shap_values"]["raw_shap_values"],
                            feature_data=result["shap_values"]["feature_data"],
                            feature_names=result["shap_values"]["feature_names"],
                            problem_type=problem_type
                        )
                        st.pyplot(summary_plot)
                    except Exception as fallback_e:
                        st.error(f"Static plot also failed: {str(fallback_e)}")
            
            with col2:
                st.markdown("""
                    ### 📖 How to Read This Interactive Plot
                    
                    This plot shows **how each feature affects predictions** across all samples:

                    **Understanding the Plot:**
                    - **Y-axis**: Features ranked by importance (most important at top)
                    - **X-axis**: SHAP value (impact on prediction)
                      - Positive values → Push prediction higher
                      - Negative values → Push prediction lower
                    - **Each dot**: Represents one sample from your data
                    - **Color**: Shows the feature's actual value
                      - 🔴 Red/Pink = High feature values
                      - 🔵 Blue = Low feature values
                    
                    **🔍 Interactive Features:**
                    - **Hover**: See detailed info (feature value, SHAP value, sample #)
                    - **Zoom**: Drag to select an area to zoom in
                    - **Pan**: Click and drag to move around
                    - **Reset**: Double-click to reset the view
                    - **Controls**: Use the ⚙️ Customize Plot section to:
                      - Change number of features shown
                      - Select specific features
                      - Download raw SHAP data
                    
                    **What to Look For:**
                    
                    1. **Feature Importance**: Features at the top matter most
                    
                    2. **Impact Direction**: 
                       - Dots mostly on the right = feature increases predictions
                       - Dots mostly on the left = feature decreases predictions
                       - Dots on both sides = feature effect varies by context
                    
                    3. **Value-Impact Relationship**:
                       - Red dots on right + Blue dots on left = Higher values increase prediction
                       - Red dots on left + Blue dots on right = Higher values decrease prediction
                       - Mixed colors = Complex non-linear relationship
                    
                    4. **Spread**: Wide spread = feature has varying effects across samples
                """)
                
                if problem_type == "regression":
                    st.markdown("""
                    **Regression Example:**
                    Predicting house prices:
                    - **Square footage** (top feature)
                      - Red dots (large houses) on right → Higher prices
                      - Blue dots (small houses) on left → Lower prices
                    - This shows larger homes increase predicted price!
                    """)
                else:  # classification
                    st.markdown("""
                    **Classification Example:**
                    Predicting loan approval:
                    - **Credit score** (top feature)
                      - Red dots (high scores) on right → More likely approved
                      - Blue dots (low scores) on left → Less likely approved
                    - This shows high credit scores increase approval probability!
                    """)
        else:
            st.info("SHAP Summary plot data not available. This may occur with certain model types or configurations.")

        # Try to get feature importance if not already available
        if importance_df is None:
            try:
                importance_result = builder.analyse_feature_importance()
                if importance_result and importance_result["success"]:
                    importance_df = pd.DataFrame(importance_result["feature_scores"])
            except Exception as e:
                st.warning(f"Could not process feature importance analysis: {str(e)}")
        
        with st.expander("📊 Understanding SHAP Values and Feature Importance", expanded=False):
            st.write("""
                #### What are SHAP Values?
                SHAP (SHapley Additive exPlanations) values help us understand how each feature in your data influences your model's predictions. Think of them as a way to "score" each feature's contribution to a prediction.

                Key points about SHAP values:
                - They are based on game theory concepts
                - They provide both local (individual prediction) and global (overall model) explanations
                - They are consistent and mathematically fair in attributing feature importance
                - The values can be positive or negative, showing whether a feature increases or decreases predictions

                #### Feature Importance Based on SHAP Values
                When we talk about feature importance using SHAP values:

                1. **Mean Absolute SHAP Value**
                - We take the absolute value of SHAP values for each feature
                - Average these values across all predictions
                - Higher values = More important features

                2. **Interpretation**
                - Features with larger SHAP values have stronger impact on predictions
                - The ranking shows which features are most influential in your model
                - Both positive and negative impacts are considered equally important

                #### 💡 Real-World Example
                Imagine predicting house prices:
                - A SHAP value of 50,000 for 'square_footage' means this feature alone pushes the prediction up by $50,000
                - A SHAP value of -20,000 for 'age' means this feature reduces the predicted price by $20,000
                - The absolute importance doesn't care about direction, just magnitude of impact

                #### Why SHAP Values Matter
                - They provide transparent, interpretable model explanations
                - Help identify which features drive your model's decisions
                - Useful for model debugging and improvement
                - Essential for explaining predictions to stakeholders
                
            """)
    
    # Tab 1: Accumulated Local Effects
    elif selected_tab == "Accumulated Local Effects":
        st.subheader("Accumulated Local Effects (ALE)")
        st.markdown("""
                    ALE plots help you understand how each feature affects your model's predictions. Think of them like a 
                    "what-if" analysis that shows how changing a feature's value impacts the prediction.
                    
                    🎯 **What Makes ALE Special:**
                    - Shows how predictions change at different feature values
                    - Better at handling complex relationships
                    
                    📊 **How to Read These Plots:**
                    - **X-axis**: The values of your feature
                    - **Y-axis**: How much the prediction changes
                    - **Line**: Shows the effect at each value
                        - Going up = feature increases the prediction
                        - Going down = feature decreases the prediction
                        - Flat = feature has little effect in that range
                    
                    🔍 **Real-World Example:**
                    Imagine predicting house prices based on size:
                    - Upward slope: Bigger houses → Higher prices
                    - Flat line: Size stops mattering after a certain point
                    - Downward slope: Too big might lower price in some areas
                    """)
        
        with st.expander("ℹ️ ALE Plot Settings Help", expanded=False):
            st.write("""
                There are two settings to help you balance speed and accuracy:
                
                1. **Sample Size:**
                - What it does: Controls how many data points to use
                - Higher numbers = More precise but slower
                - Lower numbers = Faster but less precise
                - Recommended: Start with 5000 and adjust if needed
                
                2. **Number of Bins:**
                - What it does: Groups similar values together
                - More bins = More detailed but might be noisier
                - Fewer bins = Smoother but might miss details
                - Recommended: Start with 50 bins
                
                💡 **Tips for Best Results:**
                - For quick exploration: Use smaller sample size
                - For final analysis: Use larger sample size
                - If plot looks too noisy: Reduce number of bins
                - If plot looks too smooth: Increase number of bins
            """)
        
        # Add configuration options
        with st.expander("⚙️ ALE Plot Settings", expanded=False):
            # Calculate recommended values based on dataset size
            dataset_size = len(builder.X_train)
            recommended_sample_size = min(dataset_size, 5000)  # Cap at 5000 by default
            recommended_bins = min(50, max(10, dataset_size // 100))  # Scale bins with dataset size
            
            col1, col2 = st.columns(2)
            with col1:
                sample_size = st.number_input(
                    "Maximum Sample Size",
                    min_value=min(1000, dataset_size),
                    max_value=dataset_size,
                    value=recommended_sample_size,
                    step=min(1000, max(100, dataset_size // 10)),
                    help=f"""Larger values give more precise results but take longer to compute.
                            Dataset size: {dataset_size:,}
                            Recommended: {recommended_sample_size:,}"""
                )
            with col2:
                num_bins = st.number_input(
                    "Number of Bins",
                    min_value=5,
                    max_value=min(200, dataset_size // 10),
                    value=recommended_bins,
                    step=5,
                    help=f"""More bins show finer detail but may increase noise.
                            Recommended for your dataset: {recommended_bins}
                            Reduce if plot looks too noisy, increase if too smooth."""
                )
            
            # Log ALE configuration settings
            st.session_state.logger.log_user_action("ALE Configuration", {
                "sample_size": sample_size,
                "num_bins": num_bins,
                "dataset_size": dataset_size,
                "recommended_sample_size": recommended_sample_size,
                "recommended_bins": recommended_bins
            })
            
            # Add info about current settings
            st.info(f"""
                **Current Settings:**
                - Using {sample_size:,} samples ({(sample_size/dataset_size*100):.1f}% of dataset)
                - {num_bins} bins for feature value ranges
                
                These settings are optimized for your dataset size of {dataset_size:,} samples.
            """)
        
        # Add checkbox for showing all features
        show_all_features = st.checkbox("Show ALE plots for all features", key="ale_all_features")
        
        # Log user choice for feature display
        st.session_state.logger.log_user_action("ALE Feature Display", {
            "show_all_features": show_all_features
        })
        
        # Check if we have the model and data
        if hasattr(builder, "model") and builder.model is not None:
            if hasattr(builder.model["model"], "predict"):
                # Get feature importance
                importance_result = builder.analyse_feature_importance()
                if importance_result and importance_result["success"]:
                    importance_df = pd.DataFrame(importance_result["feature_scores"])
                    
                    # Get features to display
                    if show_all_features:
                        features_to_show = importance_df['feature'].tolist()
                    else:
                        features_to_show = importance_df.head(3)['feature'].tolist()
                    
                    # Show selected features count
                    st.info(f"📊 Showing ALE plots for {'all' if show_all_features else 'top 3'} features")
                    
                    # Generate plots
                    for feature in features_to_show:
                        with st.spinner(f"Generating ALE plot for {feature}..."):
                            # Log the ALE plot generation attempt
                            st.session_state.logger.log_calculation("ALE Plot Generation", {
                                "feature": feature,
                                "num_bins": num_bins,
                                "sample_size": sample_size
                            })
                            
                            ale_fig = builder.generate_ale(
                                feature,
                                num_bins=num_bins,
                                sample_size=sample_size
                            )
                            if ale_fig is not None:
                                st.plotly_chart(ale_fig, config={'responsive': True})
                                
                                # Log successful ALE plot generation
                                st.session_state.logger.log_calculation("Visualization", {
                                    "type": "ALE Plot",
                                    "feature": feature,
                                    "status": "success"
                                })
                            else:
                                st.warning(f"Could not generate ALE plot for {feature}")
                                # Log failed ALE plot generation
                                st.session_state.logger.log_error("ALE Plot Generation Error", {
                                    "feature": feature,
                                    "error": "Could not generate ALE plot"
                                })
                else:
                    st.warning("Feature importance data is not available. This might be due to model type limitations or analysis errors.")
                    if limitations_result and limitations_result.get("message"):
                        st.info(f"Analysis message: {limitations_result['message']}")
            else:
                st.warning("Current model does not support ALE generation")
        else:
            st.warning("No model available. Please train a model first.")
    
    # Tab 2: Decision Tree Visualization (only if available)
    elif selected_tab == "Decision Tree Visualization" and has_decision_tree:
            # Initialize session state for tracking tree visualization
            if 'dt_viz_last_logged_tree' not in st.session_state:
                st.session_state.dt_viz_last_logged_tree = None
            if 'dt_viz_tab_entered' not in st.session_state:
                st.session_state.dt_viz_tab_entered = False
            
            st.subheader("🌳 Decision Tree Visualization")

            st.markdown("""
                **Decision trees** show you exactly how your model makes decisions by following a series of
                yes/no questions about your features. This is the most interpretable view of your model's logic.

                🎯 **What you'll see:**
                - The exact rules your model follows
                - Which features are most important for decisions
                - How different feature values lead to different predictions

                📊 **Perfect for:**
                - Understanding individual predictions
                - Creating business rules
                - Explaining model decisions to stakeholders
                - Debugging model behavior
                """)

            # Get model instance - keep wrapped model for create_decision_tree_plot
            # which handles unwrapping internally and preserves class labels
            model_instance = builder.model.get("active_model") or builder.model["model"]

            # Extract base model to check ensemble properties (for selectbox)
            base_model = extract_base_model(model_instance)
            _, tree_type = supports_decision_tree_visualization(model_instance)

            # Add depth and display controls
            col_control1, col_control2 = st.columns(2)

            with col_control1:
                show_full_tree = st.checkbox(
                    "Show full tree depth",
                    value=False,
                    help="Display the complete tree structure. Warning: Very deep trees may be difficult to read and can take longer to render.",
                    key="dt_viz_show_full_tree"
                )

                if not show_full_tree:
                    max_depth = st.slider(
                        "Maximum tree depth to display:",
                        min_value=1,
                        max_value=10,
                        value=3,
                        help="Limiting depth makes trees more readable. Full trees can be very complex.",
                        key="dt_viz_tree_depth"
                    )
                else:
                    max_depth = None  # This will be handled in the plotting function
                    st.warning("⚠️ **Full tree depth enabled**: Very deep trees may be difficult to read and can take longer to render. Consider using the text view for better readability.")

            with col_control2:
                display_mode = st.radio(
                    "Display mode:",
                    ["Visual Tree", "Text Rules", "Both"],
                    index=0,
                    help="Visual trees can become cluttered for deep trees. Text rules provide a cleaner alternative.",
                    key="dt_viz_display_mode"
                )

            # For ensemble models, add tree selector
            tree_index = 0
            if tree_type == "ensemble":
                model_name = base_model.__class__.__name__
                if hasattr(base_model, 'estimators_') and len(base_model.estimators_) > 0:
                    max_trees = len(base_model.estimators_)
                    st.info(f"ℹ️ **Ensemble Model Detected**: {model_name} contains {max_trees} trees. Showing a representative tree from the ensemble.")
                    tree_index = st.selectbox(
                        "Select tree to visualize:", 
                        range(min(10, max_trees)),  # Limit to first 10 trees for performance
                        index=0,
                        format_func=lambda x: f"Tree {x + 1}",
                        help="For ensemble models, you can view individual trees. The first few trees are often most representative.",
                        key="dt_viz_tree_selector"
                    )
                elif 'XGB' in model_name or 'LightGBM' in model_name or 'LGBM' in model_name:
                    st.info(f"ℹ️ **Ensemble Model Detected**: {model_name} ensemble. Showing a representative tree structure.")
                    tree_index = st.selectbox(
                        "Select tree to visualize:", 
                        range(10),  # Default to first 10 boosting rounds
                        index=0,
                        format_func=lambda x: f"Boosting Round {x + 1}",
                        help="For boosting models, each round adds a new tree. Early rounds are often most interpretable.",
                        key="dt_viz_boosting_tree_selector"
                    )
                else:
                    st.info(f"ℹ️ **Ensemble Model Detected**: {model_name}. Showing a representative tree.")
            else:
                st.info("🌳 **Single Decision Tree**: Showing the complete decision tree structure.")
            
            # Generate and display the tree
            col1, col2 = st.columns([2, 1])
            
            with col1:
                try:
                    # Get feature names from the training data
                    if hasattr(builder, 'X_train') and hasattr(builder.X_train, 'columns'):
                        feature_names = list(builder.X_train.columns)
                    else:
                        feature_names = [f"Feature_{i}" for i in range(len(builder.X_train.shape[1]) if hasattr(builder.X_train, 'shape') else 10)]
                    
                    tree_fig, ensemble_info, tree_text = create_decision_tree_plot(
                        model_instance, 
                        feature_names, 
                        max_depth=max_depth, 
                        tree_index=tree_index,
                        display_mode=display_mode
                    )
                    
                    # Display based on selected mode
                    if display_mode in ["Visual Tree", "Both"] and tree_fig is not None:
                        st.plotly_chart(tree_fig, config={'responsive': True}, key="dt_viz_decision_tree_plot")
                    
                    if display_mode in ["Text Rules", "Both"] and tree_text is not None:
                        st.subheader("📝 Decision Tree Rules")
                        st.code(tree_text, language="text")
                    
                    if tree_fig is None and tree_text is None:
                        st.error("Unable to generate decision tree visualization")
                        st.write("This might be due to model complexity or incompatibility with tree visualization libraries.")
                    else:
                        if ensemble_info:
                            st.caption(f"Displaying: {ensemble_info}")
                        
                        # Only log journey point when first entering the tab or when tree selection changes
                        should_log = False
                        if not st.session_state.dt_viz_tab_entered:
                            # First time entering this tab
                            should_log = True
                            st.session_state.dt_viz_tab_entered = True
                            st.session_state.dt_viz_last_logged_tree = tree_index
                        elif st.session_state.dt_viz_last_logged_tree != tree_index:
                            # Tree selection changed
                            should_log = True
                            st.session_state.dt_viz_last_logged_tree = tree_index
                        
                        if should_log:
                            # Log the tree visualization
                            st.session_state.logger.log_journey_point(
                                stage="MODEL_EXPLANATION",
                                decision_type="MODEL_EXPLANATION",
                                description="Decision Tree Visualization",
                                details={
                                    "Model Type": model_name if tree_type == "ensemble" else "Single Decision Tree",
                                    "Display Mode": display_mode,
                                    "Max Depth": max_depth,
                                    "Tree Index": tree_index if tree_type == "ensemble" else None,
                                    "Ensemble Info": ensemble_info,
                                    "Tree Text": tree_text
                                },
                                parent_id=None
                            )
                        
                        # Add text view explanation
                        if display_mode in ["Text Rules", "Both"]:
                            with st.expander("ℹ️ Understanding Text Rules"):
                                problem_type = builder.model.get("problem_type", "classification")
                                model_name = model_instance.__class__.__name__
                                
                                if problem_type == "regression":
                                    st.markdown("""
                                    **How to read regression tree rules:**
                                    - Each line represents a decision rule
                                    - **Indentation** shows the tree structure (deeper = further down tree)
                                    - **|--- feature <= threshold** means "if feature value is less than or equal to threshold"
                                    - **value: [number]** shows the predicted numerical value
                                    - **squared_error** shows the variance/error at that node
                                    
                                    **Benefits of text view:**
                                    - More readable for deep trees
                                    - Shows exact threshold and prediction values
                                    - Can be converted to business rules for scoring
                                    - No visual clustering issues
                                    - Easy to trace prediction paths
                                    """)
                                else:
                                    if tree_type == "single":
                                        # Single decision tree explanation
                                        st.markdown("""
                                        **How to read single decision tree classification rules:**
                                        - Each line represents a decision rule
                                        - **Indentation** shows the tree structure (deeper = further down tree)
                                        - **|--- feature <= threshold** means "if feature value is less than or equal to threshold"
                                        - **class: [class_name]** shows the predicted class for samples reaching this node
                                        - **value: [array]** shows class counts at this node
                                          - For binary classification: [count_class_0, count_class_1]
                                          - For multiclass: [count_class_0, count_class_1, count_class_2, ...]
                                          - Example: [12,0] = 12 samples of class 0, 0 samples of class 1
                                          - Example: [7,1,2] = 7 samples of class 0, 1 sample of class 1, 2 samples of class 2
                                        
                                        **Benefits of text view:**
                                        - More readable for deep trees
                                        - Shows exact threshold values and class names
                                        - Can be copied and used as business rules
                                        - No visual clustering issues
                                        """)
                                    else:
                                        # Ensemble tree explanation
                                        if problem_type == "multiclass_classification":
                                            st.markdown(f"""
                                            **How to read ensemble tree classification rules ({model_name}):**
                                            - Each line represents a decision rule from ONE tree in the ensemble
                                            - **Indentation** shows the tree structure (deeper = further down tree)
                                            - **|--- feature <= threshold** means "if feature value is less than or equal to threshold"
                                            - **class: [class_name]** shows this tree's contribution to that class
                                            - **value: [number or array]** shows this tree's output
                                              - **Important**: For multiclass ensembles, each tree typically contributes to predicting ONE specific class
                                              - The final prediction combines outputs from ALL trees in the ensemble
                                              - Class names are shown as "Class_0 (actual_name)", "Class_1 (actual_name)", etc.
                                            
                                            **🔍 Understanding Ensemble Context:**
                                            - This is ONE tree out of many (e.g., 100+ trees in Random Forest)
                                            - Each tree sees a different subset of features/samples
                                            - Final prediction = majority vote or average of all trees
                                            - Different trees may focus on different classes or decision patterns
                                            
                                            **Benefits of text view:**
                                            - Shows exact decision logic for this specific tree
                                            - More readable than complex visual trees
                                            - Can trace decision paths easily
                                            - Includes class label information for clarity
                                            """)
                                        else:
                                            st.markdown(f"""
                                            **How to read ensemble tree classification rules ({model_name}):**
                                            - Each line represents a decision rule from ONE tree in the ensemble
                                            - **Indentation** shows the tree structure (deeper = further down tree)
                                            - **|--- feature <= threshold** means "if feature value is less than or equal to threshold"
                                            - **class: [class_name]** shows this tree's prediction
                                            - **value: [array]** shows class counts or probabilities for this tree
                                              - For binary classification: [count_class_0, count_class_1] or [prob_class_0, prob_class_1]
                                              - Example: [12,0] = 12 samples favor class 0, 0 favor class 1
                                            
                                            **🔍 Understanding Ensemble Context:**
                                            - This is ONE tree out of many (e.g., 100+ trees in Random Forest)
                                            - Each tree sees a different subset of features/samples
                                            - Final prediction = majority vote or average of all trees
                                            - Different trees may make different predictions for the same input
                                            
                                            **Benefits of text view:**
                                            - Shows exact decision logic for this specific tree
                                            - More readable than complex visual trees
                                            - Can trace decision paths easily
                                            - Reveals individual tree patterns
                                            """)
                        
                        # Add strategic understanding guide for all tree types
                        with st.expander("🔍 Understanding This Tree in Context"):
                            problem_type = builder.model.get("problem_type", "classification")
                            
                            if tree_type == "ensemble":
                                st.markdown(f"""
                                **🎯 Using This Tree to Understand Your {model_name} Model:**
                                
                                **What This Tree Reveals About Your Model:**
                                - **Feature Preferences**: Which features your model finds most useful for splitting data
                                - **Decision Boundaries**: The specific thresholds that matter for predictions
                                - **Data Patterns**: How your model has learned to group similar examples
                                - **Prediction Logic**: The step-by-step reasoning process for classifications
                                
                                **🔄 Ensemble Model Insights:**
                                - **This is just ONE perspective** - your model combines many such trees
                                - **Look for patterns** when viewing multiple trees - consistent features are most important
                                - **Different specializations** - some trees may be better at certain types of predictions
                                - **Collective wisdom** - the ensemble is stronger than any individual tree
                                
                                **💡 Strategic Questions to Ask:**
                                - *Which features appear most often at the top of trees?* (These are your most important features)
                                - *Do the decision thresholds make business sense?* (Validate against domain knowledge)
                                - *Are there unexpected feature combinations?* (Discover new insights about your data)
                                - *How deep do trees typically go?* (Indicates complexity of your decision problem)
                                
                                **🚀 Actionable Insights:**
                                - **Feature Engineering**: Focus on improving the most frequently used features
                                - **Data Quality**: Pay special attention to threshold values that seem unusual
                                - **Business Rules**: Extract simple rules from shallow branches for manual decision-making
                                - **Model Trust**: Verify that important splits align with your domain expertise
                                """)
                            else:
                                st.markdown("""
                                **🎯 Using This Tree to Understand Your Decision Tree Model:**
                                
                                **What This Tree Reveals About Your Model:**
                                - **Complete Decision Logic**: This IS your model's entire reasoning process
                                - **Feature Hierarchy**: Features at the top are most important for your problem
                                - **Decision Boundaries**: Every threshold shown is used in real predictions
                                - **Prediction Confidence**: Sample counts show how certain each prediction is
                                
                                **🔍 Key Strategic Insights:**
                                - **Simplicity vs Accuracy**: Deeper trees capture more nuance but may overfit
                                - **Feature Dependencies**: See which features work together in combinations
                                - **Edge Cases**: Deep branches reveal unusual data patterns your model handles
                                - **Business Alignment**: Verify that splits make sense for your domain
                                
                                **💡 Strategic Questions to Ask:**
                                - *Are the most important features (top of tree) what you expected?*
                                - *Do the threshold values align with business knowledge?*
                                - *Are there branches that seem overly complex?* (Possible overfitting)
                                - *Can you explain any prediction path to a stakeholder?* (Model transparency test)
                                
                                **🚀 Actionable Insights:**
                                - **Rule Extraction**: Convert tree paths to business rules for operational use
                                - **Feature Focus**: Invest in improving data quality for top-level features  
                                - **Threshold Validation**: Check that important cut-points make domain sense
                                - **Complexity Management**: Consider pruning if tree is too deep for your use case
                                """)
                        
                except Exception as e:
                    st.error(f"Decision tree visualization failed: {str(e)}")
                    st.write("Try reducing the maximum depth or check if your model supports tree visualization.")
            
            with col2:
                st.markdown("#### What am I looking at?")
                st.markdown("""---""")
                problem_type = builder.model.get("problem_type", "classification")
                if problem_type == "regression":
                    st.markdown("""
                    **Decision Tree Visualization** shows how your regression model makes predictions through a series of feature-based decisions.
                    
                    - **Root (top)**: Starting point for all predictions
                    - **Branches**: Decision rules based on feature thresholds
                    - **Leaves (bottom)**: Numerical prediction values
                    - **Colors**: Often indicate prediction values or sample density
                    """)
                else:
                    st.markdown("""
                    **Decision Tree Visualization** shows how your model makes decisions through a series of yes/no questions about your features.
                    
                    - **Root (top)**: Starting point for all predictions
                    - **Branches**: Decision rules based on feature values
                    - **Leaves (bottom)**: Final prediction outcomes
                    - **Colors**: Usually indicate prediction strength or class
                    """)
                
                with st.expander("📖 Detailed Interpretation Guide"):
                    st.markdown("**How to Read the Visual Elements:**")
                    if problem_type == "regression":
                        st.markdown("""
                        **Visual Tree Elements:**
                        - **📦 Boxes (Nodes)**: Each rectangle shows a decision or prediction point
                        - **🔀 Split Text**: "feature_name <= 12.5" means "if feature value is 12.5 or less, go left; otherwise go right"
                        - **👥 samples = X**: Number of training data points that reached this box
                        - **📊 mse = X.XX**: Mean Squared Error - how much predictions vary at this point (lower = more certain)
                        - **🎯 value = X.XX**: Shows different information based on tree type:
                          - **Single trees**: Average target value of samples at this node
                          - **Random Forest trees**: Average target value of samples in this tree's subset
                          - **Gradient Boosting**: Residual or contribution value for this tree
                        - **🎨 Colors**: Usually darker = higher predicted values, lighter = lower predicted values
                        
                        **Text Rules Format:**
                        - **Indentation**: More spaces = deeper in the tree
                        - **|--- feature <= threshold**: The decision rule
                        - **value: [number]**: Prediction value (varies by model type)
                        - **squared_error**: How much uncertainty remains at this node
                        """)
                    else:
                        st.markdown("""
                        **Visual Tree Elements:**
                        - **📦 Boxes (Nodes)**: Each rectangle shows a decision or prediction point
                        - **🔀 Split Text**: "feature_name <= 12.5" means "if feature value is 12.5 or less, go left; otherwise go right"
                        - **👥 samples = X**: Number of training data points that reached this box
                        - **📊 value = [X, Y, ...]**: Shows different information based on tree type and problem
                          - **Single trees**: [50, 10] = 50 samples of first class, 10 of second class
                          - **Random Forest trees**: [50, 10] = 50 samples of first class, 10 of second class (from subset)
                          - **Gradient Boosting (multiclass)**: Single value = probability contribution for assigned class
                          - **Gradient Boosting (binary/regression)**: Single value = prediction contribution
                        - **🏷️ class = ClassName**: The winning class at this point (most samples)
                        - **🎨 Colors**: Each class gets a different color, intensity shows confidence
                        
                        **Text Rules Format:**
                        - **Indentation**: More spaces = deeper in the tree  
                        - **|--- feature <= threshold**: The if-then decision rule
                        - **class: ClassName**: What class this path predicts
                        - **value: [array or single]**: Depends on model type:
                          - **Single trees & Random Forest**: Sample counts per class [X, Y, Z]
                          - **Gradient Boosting**: Single contribution value for tree's purpose
                        """)
                        
                        if problem_type == "multiclass_classification":
                            st.markdown("""
                            **🎯 Multiclass Specifics:**
                            - **Class Labels**: Shows "Class_0 (ActualName)" for clarity
                            - **Value Arrays**: [X, Y, Z] for 3 classes showing sample counts
                            - **Winner Takes All**: The class with most samples becomes the prediction
                            """)
                        
                        if tree_type == "ensemble":
                            st.markdown("""
                            **🌲 Ensemble Tree Notes:**
                            - **Individual View**: You're seeing ONE tree from many (e.g., 1 of 100)
                            - **Values Shown**: Depends on the ensemble type:
                              - **Random Forest**: Shows sample counts or averages (like single trees, but from data subset)
                              - **Gradient Boosting (multiclass)**: Single probability value for assigned class
                                - **Class Assignment**: Look at tree title - shows which class this tree predicts
                                - **Example**: "Tree 5 (Round 2, Class_1 (Positive))" = predicts for "Positive" class
                              - **Gradient Boosting (binary/regression)**: Single contribution/residual value
                            - **Tree Contribution**: Each tree's output gets combined for final prediction
                            - **Data Subsets**: Random Forest trees see random subsets; Gradient Boosting trees see full data but focus on errors
                            """)
    
    # Show info about decision tree availability if not available and that tab was selected
    if not has_decision_tree and selected_tab == "Decision Tree Visualization":
        if hasattr(builder, "model") and builder.model is not None:
            st.info("💡 **Note**: Decision Tree Visualization is not available for this model type. Available for: Decision Tree, Random Forest, Gradient Boosting, XGBoost, and LightGBM models.")
            st.markdown("""
                **Tree-based models that support visualization:**
                - **Single Decision Trees**: Show the complete decision logic
                - **Random Forest**: View individual trees from the ensemble
                - **Gradient Boosting**: Explore sequential tree improvements
                - **XGBoost/LightGBM**: Examine boosted tree structures
                
                **Why decision trees are valuable for feature analysis:**
                - Show exact decision boundaries
                - Reveal feature interaction patterns
                - Provide interpretable business rules
                - Help identify the most important decision points
                """)
        else:
            st.warning("No model available. Please train a model first.")