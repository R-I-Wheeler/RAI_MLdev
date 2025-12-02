import streamlit as st
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go

def get_class_names_with_indices(model=None):
    """Get class names in the format 'Class_0 (ActualName)' for consistent display."""
    try:
        if model is None and hasattr(st.session_state, 'builder') and hasattr(st.session_state.builder, 'model'):
            model = st.session_state.builder.model.get("model") or st.session_state.builder.model.get("active_model")
        
        if model and hasattr(model, "classes_"):
            return [f"Class_{i} ({cls})" for i, cls in enumerate(model.classes_)]
        else:
            return None
    except Exception as e:
        print(f"DEBUG: Error getting class names in individual_prediction: {e}")
        return None

def render_individual_prediction_analysis():
    """Render the individual prediction analysis section."""
    st.header("Individual Prediction Analysis")
    
    with st.expander("ℹ️ How to Use This Section", expanded=True):
        problem_type = st.session_state.get('problem_type', 'unknown')
        if problem_type == "multiclass_classification":
            st.markdown("""
                Analyse individual predictions to understand specific multiclass classification cases:
                
                1. **Select a Sample**: Choose any test case to analyse
                2. **Compare Classifications**: See predicted vs actual classes and probabilities
                3. **Feature Impact**: Understand what drove the class prediction
                4. **Class Probabilities**: See confidence across all classes
                5. **Detailed Breakdown**: See contribution of each feature to predicted class
                
                This helps you:
                - Validate class predictions and confidence levels
                - Debug misclassified samples
                - Explain classification decisions to stakeholders
                - Identify features that distinguish between classes
                - Understand model uncertainty across classes
            """)
        elif problem_type == "binary_classification":
            st.markdown("""
                Analyse individual predictions to understand specific binary classification cases:
                
                1. **Select a Sample**: Choose any test case to analyse
                2. **Compare Classifications**: See predicted vs actual classes and probabilities
                3. **Feature Impact**: Understand what drove the positive/negative prediction
                4. **Classification Decision**: See probability threshold and confidence
                5. **Detailed Breakdown**: See contribution of each feature to classification
                
                This helps you:
                - Validate classification predictions and confidence
                - Debug false positives and false negatives
                - Explain classification decisions to stakeholders
                - Identify key features for positive/negative classification
                - Understand decision boundaries and thresholds
            """)
        else:  # regression or unknown
            st.markdown("""
                Analyse individual predictions to understand specific regression cases:
                
                1. **Select a Sample**: Choose any test case to analyse
                2. **Compare Predictions**: See predicted vs actual values and error
                3. **Feature Impact**: Understand what drove the predicted value
                4. **Value Breakdown**: See how features increase/decrease prediction
                5. **Detailed Analysis**: See contribution of each feature to final value
                
                This helps you:
                - Validate predicted values and understand errors
                - Debug over/under-predictions
                - Explain value predictions to stakeholders
                - Identify features that drive higher/lower values
                - Understand feature interactions and their effects
            """)
    
    if st.session_state.builder.X_test is not None:
        n_samples = len(st.session_state.builder.X_test)
        
        # Add sample filtering options
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_index = st.selectbox(
                "Select a sample to explain:",
                options=list(range(n_samples)),
                format_func=lambda x: f"Sample {x + 1}"
            )
        with col2:
            st.info("💡 Choose any sample to see detailed explanation")
        
        if selected_index is not None:
            # Enhanced logging for sample selection
            st.session_state.logger.log_user_action(
                "Sample Selected for Explanation",
                {
                    "sample_index": selected_index,
                    "sample_features": st.session_state.builder.X_test.iloc[selected_index].to_dict(),
                    "actual_value": str(st.session_state.builder.y_test.iloc[selected_index])
                }
            )
            
            explanation = st.session_state.builder.explain_prediction(selected_index)
            
            if explanation["success"]:
                # Enhanced logging for prediction explanation
                st.session_state.logger.log_calculation(
                    "Individual Prediction Explanation",
                    {
                        "sample_index": selected_index,
                        "prediction": explanation["individual_explanation"]["prediction"],
                        "confidence": explanation["individual_explanation"].get("confidence", None),
                        "feature_contributions": explanation["individual_explanation"]["contributions"],
                        "explanation_method": explanation["individual_explanation"].get("method", "SHAP")
                    }
                )
                
                # Prediction metrics
                metrics_container = st.container()
                col1, col2, col3 = metrics_container.columns([1, 1, 1])
                
                problem_type = getattr(st.session_state, 'problem_type', st.session_state.builder.model.get("problem_type", "unknown"))
                prediction_data = explanation['individual_explanation']['prediction']
                
                with col1:
                    if problem_type == "multiclass_classification":
                        # Show max class probability and predicted class
                        predicted_class = prediction_data.get('predicted_class', 0)
                        max_probability = prediction_data.get('predicted', 0)
                        
                        # Get actual class names if available
                        class_names = get_class_names_with_indices()
                        if class_names and predicted_class < len(class_names):
                            predicted_class_display = class_names[predicted_class]
                        else:
                            predicted_class_display = f"Class {predicted_class}"
                        
                        st.metric(
                            "Predicted Class",
                            predicted_class_display,
                            help="The class with highest predicted probability"
                        )
                    elif problem_type == "binary_classification":
                        # Show positive class probability
                        predicted_value = prediction_data['predicted']
                        st.metric(
                            "Positive Class Probability",
                            f"{predicted_value * 100:.2f}%"
                        )
                    else:
                        # For regression, show the predicted value
                        predicted_value = prediction_data['predicted']
                        actual_value = prediction_data['actual']
                        difference = predicted_value - actual_value
                        st.metric(
                            "Predicted",
                            f"{predicted_value:.4f}",
                            delta=f"{difference:+.4f}",
                            delta_color="inverse"
                        )
            
                with col2:    
                    if problem_type == "multiclass_classification":
                        # Show confidence (max probability)
                        max_probability = prediction_data.get('predicted', 0)
                        st.metric(
                            "Confidence",
                            f"{max_probability * 100:.2f}%",
                            help="Probability of the predicted class"
                        )
                    elif problem_type == "binary_classification":
                        # Show predicted class using optimal threshold if available
                        if (hasattr(st.session_state, 'builder') and 
                            hasattr(st.session_state.builder, 'model') and 
                            st.session_state.builder.model.get("threshold_optimized", False) and
                            st.session_state.builder.model.get("threshold_is_binary", True)):
                            optimal_threshold = st.session_state.builder.model.get("optimal_threshold", 0.5)
                            predicted_class = prediction_data.get('predicted_class', 
                                            1 if prediction_data['predicted'] >= optimal_threshold else 0)
                        else:
                            predicted_class = prediction_data.get('predicted_class', 
                                            1 if prediction_data['predicted'] >= 0.5 else 0)
                        st.metric(
                            "Predicted Class",
                            f"{predicted_class} ({('Positive' if predicted_class == 1 else 'Negative')})"
                        )
                    else:
                        # For regression, show actual value
                        actual_value = prediction_data['actual']
                        st.metric(
                            "Actual",
                            f"{actual_value:.4f}",
                            help="Difference shows: Predicted - Actual. Positive means over-prediction, negative means under-prediction."
                        )
            
                with col3:
                    if problem_type in ["binary_classification", "multiclass_classification"]:
                        actual_value = prediction_data['actual']
                        if problem_type == "multiclass_classification":
                            actual_class_idx = int(actual_value)
                            
                            # Get actual class names if available
                            class_names = get_class_names_with_indices()
                            if class_names and actual_class_idx < len(class_names):
                                actual_class_display = class_names[actual_class_idx]
                            else:
                                actual_class_display = f"Class {actual_class_idx}"
                            
                            st.metric(
                                "Actual Class",
                                actual_class_display
                            )
                        else:  # binary_classification
                            # The actual value should match the true class, not be based on threshold
                            # But for consistency, we'll keep the existing logic for display purposes
                            actual_numeric = 1 if float(actual_value) >= 0.5 else 0
                            st.metric(
                                "Actual",
                                f"{actual_numeric} ({('Positive' if actual_numeric == 1 else 'Negative')})"
                            )
                                         # For regression, actual is already shown in col2
                
                # Show class probabilities for multiclass
                if problem_type == "multiclass_classification" and 'class_probabilities' in prediction_data:
                    st.subheader("Class Probabilities")
                    probs = prediction_data['class_probabilities']
                    
                    # Get actual class names if available
                    class_names = get_class_names_with_indices()
                    if class_names and len(class_names) == len(probs):
                        class_labels = class_names
                    else:
                        # Fallback to generic names
                        class_labels = [f'Class {i}' for i in range(len(probs))]
                    
                    prob_df = pd.DataFrame({
                        'Class': class_labels,
                        'Probability': probs
                    })
                    prob_df['Percentage'] = prob_df['Probability'] * 100
                    
                    # Create a horizontal bar chart
                    fig = go.Figure(go.Bar(
                        x=prob_df['Probability'],
                        y=prob_df['Class'],
                        orientation='h',
                        text=[f'{p:.1f}%' for p in prob_df['Percentage']],
                        textposition='inside',
                        marker_color=['#ff7f0e' if i == prediction_data.get('predicted_class', 0) else '#1f77b4' 
                                    for i in range(len(probs))]
                    ))
                    
                    fig.update_layout(
                        title="Predicted Probabilities by Class",
                        xaxis_title="Probability",
                        yaxis_title="Class",
                        height=200 + len(probs) * 25,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, config={'responsive': True})
                
                # Add explanation of SHAP vs LIME
                with st.expander("ℹ️ Understanding SHAP vs LIME Explanations", expanded=False):
                    st.markdown("""
                        ### 🔄 Understanding Model Explanations: SHAP vs LIME
                        
                        Both SHAP and LIME help us understand why the model made a specific prediction, but they use different approaches. Let's break them down with examples:
                        
                        #### SHAP (SHapley Additive exPlanations) 🎮
                        
                        **Simple Explanation:**
                        Imagine you're playing a team game and want to know how much each player contributed to winning. SHAP is like looking at all possible combinations of players and seeing how the team performs with and without each player.
                        
                        **Example:**
                        Let's say we're predicting house prices:
                        - Base price (average): £300,000
                        - SHAP tells us:
                            - Large size: +£50,000
                            - Good location: +£30,000
                            - Old age: -£20,000
                            - Final prediction: £360,000
                        
                        **How it Works:**
                        1. Takes your prediction
                        2. Starts from the average prediction for all houses
                        3. Shows how each feature moves the prediction up or down
                        4. Considers how features work together (e.g., large size AND good location might be worth more than just adding their individual effects)
                        
                        **When to Use:**
                        - When you need very precise contribution values
                        - When you want to understand how features work together
                        - When you need consistent explanations
                        
                        #### LIME (Local Interpretable Model-agnostic Explanations) 🔍
                        
                        **Simple Explanation:**
                        LIME is like explaining a complex drawing by making a simple sketch that captures the main ideas. It creates a simple, understandable model that behaves similarly to the complex model for the specific prediction we're interested in.
                        
                        **Example:**
                        For the same house price prediction:
                        - LIME might say:
                            - "Houses between 2000-2500 sq ft → Higher price"
                            - "Houses 5-10 miles from downtown → Lower price"
                            - "Houses built 1990-2000 → Higher price"
                        
                        **How it Works:**
                        1. Takes your specific case (e.g., a house)
                        2. Creates similar examples by slightly changing features
                        3. Sees how the model's predictions change
                        4. Creates simple rules that explain what's important
                        
                        **When to Use:**
                        - When you want easy-to-understand rules
                        - When you need to explain predictions to non-technical people
                        - When you want to understand what ranges of values matter
                        
                        #### 🤔 Key Differences with Examples
                        
                        **1. Precision vs Interpretability**
                        - SHAP: "Size adds exactly £50,000 to the price"
                        - LIME: "Houses between 2000-2500 sq ft increase the price"
                        
                        **2. Feature Interactions**
                        - SHAP: Considers that a large house in a good location might be worth more than just adding individual effects
                        - LIME: Looks at each feature more independently
                        
                        **3. Consistency**
                        - SHAP: Always gives the same explanation for the same prediction
                        - LIME: Might give slightly different explanations each time (because it creates random similar examples)
                        
                        #### 🎯 Best Practices
                        
                        1. **Use Both When Possible**
                           - SHAP for precise numbers
                           - LIME for understandable ranges
                        
                        2. **Look for Agreement**
                           - If both methods say a feature is important, it probably is!
                        
                        3. **Consider Your Audience**
                           - Technical audience → Start with SHAP
                           - Non-technical audience → Start with LIME
                    """)
                tab1, tab2, tab3 = st.tabs(["SHAP Contributions", "LIME Contributions", "SHAP vs LIME Comparison"])
                with tab1:

                    # Waterfall plot with explanation
                    st.subheader("SHAP Force Plot")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if "contributions" in explanation["individual_explanation"]:
                            # Extract feature names and values for the force plot
                            feature_names = [contrib['Feature'] for contrib in explanation["individual_explanation"]["contributions"]]
                            feature_values = [contrib['Impact'] for contrib in explanation["individual_explanation"]["contributions"]]
                            
                            # Convert to numpy array for SHAP
                            shap_values = np.array(feature_values)
                            
                            # Create force plot with SHAP v0.20+ compatibility (following what-if analysis approach)
                            try:
                                # Try new SHAP v0.20+ API first
                                try:
                                    force_plot = shap.plots.force(
                                        explanation["individual_explanation"]["base_value"],
                                        shap_values,
                                        feature_names=feature_names,
                                        matplotlib=True,
                                        show=False
                                    )
                                except (ValueError, TypeError) as e:
                                    if "multiple samples" in str(e).lower():
                                        # If matplotlib fails due to multiple samples, use matplotlib=False
                                        force_plot = shap.plots.force(
                                            explanation["individual_explanation"]["base_value"],
                                            shap_values,
                                            feature_names=feature_names,
                                            matplotlib=False,
                                            show=False
                                        )
                                    else:
                                        raise e
                            except AttributeError:
                                # Fallback to old API for older SHAP versions
                                try:
                                    force_plot = shap.force_plot(
                                        explanation["individual_explanation"]["base_value"],
                                        shap_values,
                                        feature_names=feature_names,
                                        matplotlib=True,
                                        show=False
                                    )
                                except (ValueError, TypeError) as e:
                                    if "multiple samples" in str(e).lower():
                                        # If matplotlib fails due to multiple samples, use matplotlib=False
                                        force_plot = shap.force_plot(
                                            explanation["individual_explanation"]["base_value"],
                                            shap_values,
                                            feature_names=feature_names,
                                            matplotlib=False,
                                            show=False
                                        )
                                    else:
                                        raise e
                            st.pyplot(force_plot)
                        else:
                            st.warning("Force plot not available for this model type")
                    
                    with col2:
                        problem_type = st.session_state.get('problem_type', 'unknown')
                        if problem_type == "multiclass_classification":
                            st.info("""
                                **Reading the Force Plot (Multiclass):**
                                - Red = increasing predicted class probability
                                - Blue = decreasing predicted class probability
                                - Width = magnitude of impact on predicted class
                                - Base value = average probability for predicted class
                                - Shows contributions for the predicted class only
                            """)
                        elif problem_type == "binary_classification":
                            st.info("""
                                **Reading the Force Plot (Binary):**
                                - Red = increasing positive class probability
                                - Blue = decreasing positive class probability
                                - Width = magnitude of impact on positive class
                                - Base value = average positive class probability
                                - Final prediction = probability of positive class
                            """)
                        else:  # regression or unknown
                            st.info("""
                                **Reading the Force Plot (Regression):**
                                - Red = pushing prediction higher
                                - Blue = pushing prediction lower
                                - Width = magnitude of direct impact on value
                                - Base value = model's average prediction
                                - Final prediction = actual predicted value
                            """)

                    # SHAP Waterfall plot
                    st.subheader("SHAP Feature Contribution Waterfall")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if "contributions" in explanation["individual_explanation"]:
                            # Create dataframe for the waterfall chart
                            waterfall_df = pd.DataFrame(explanation["individual_explanation"]["contributions"])
                            waterfall_df = waterfall_df.sort_values('Impact', key=abs, ascending=True)
                            
                            # Get base value for proper waterfall
                            base_value = explanation["individual_explanation"]["base_value"]
                            
                            # Get prediction value for display
                            prediction_data = explanation["individual_explanation"]["prediction"]
                            prediction_value = prediction_data["predicted"]
                            
                            # Create proper SHAP waterfall that starts from baseline and ends at final prediction
                            waterfall_features = ["Base Value"] + waterfall_df['Feature'].tolist() + ["Final Prediction"]
                            waterfall_values = [float(base_value)] + waterfall_df['Impact'].tolist() + [prediction_value]
                            waterfall_measures = ["absolute"] + ["relative"] * len(waterfall_df) + ["total"]
                            
                            # Create text for hover information
                            waterfall_text = [f"Base Value: {base_value:.4f}"]
                            waterfall_text.extend([f"Feature: {feat}<br>Value: {val}<br>Impact: {imp:+.4f}" 
                                                   for feat, val, imp in zip(waterfall_df['Feature'], waterfall_df['Value'], waterfall_df['Impact'])])
                            waterfall_text.append(f"Final: {prediction_value:.4f}")
                            
                            # Create waterfall chart
                            fig = go.Figure(go.Waterfall(
                                name="SHAP",
                                orientation="h",
                                measure=waterfall_measures,
                                x=waterfall_values,
                                y=waterfall_features,
                                connector={"mode": "spanning", "line": {"width": 1, "color": "rgb(63, 63, 63)", "dash": "solid"}},
                                decreasing={"marker": {"color": "rgba(50, 171, 96, 0.7)"}},
                                increasing={"marker": {"color": "rgba(219, 64, 82, 0.7)"}},
                                text=waterfall_text,
                                textposition="auto",
                                hovertemplate="<b>%{y}</b><br>" +
                                             "%{text}<br>" +
                                             "<extra></extra>"
                            ))
                            
                            fig.update_layout(
                                title="SHAP Waterfall: From Base Value to Final Prediction",
                                showlegend=False,
                                height=max(500, (len(waterfall_df) + 2) * 30),  # +2 for base value and final prediction
                                margin=dict(t=50, b=50, l=50, r=50),
                                yaxis=dict(title="Features & Values", autorange="reversed"),  # Reverse to show baseline at top
                                xaxis=dict(title="Prediction Value")
                            )
                            
                            st.plotly_chart(fig, config={'responsive': True})
                        else:
                            st.warning("Waterfall plot not available for this model type")
                    
                    with col2:
                        problem_type = st.session_state.get('problem_type', 'unknown')
                        if problem_type == "multiclass_classification":
                            st.info("""
                                **Reading the Waterfall (Multiclass):**
                                - **Base Value**: Model's average predicted class probability
                                - **Green bars**: Features decreasing the probability
                                - **Red bars**: Features increasing the probability
                                - **Final Prediction**: Actual predicted probability
                                - Shows complete path from baseline to prediction
                            """)
                        elif problem_type == "binary_classification":
                            st.info("""
                                **Reading the Waterfall (Binary):**
                                - **Base Value**: Model's average positive class probability
                                - **Green bars**: Features decreasing positive probability
                                - **Red bars**: Features increasing positive probability
                                - **Final Prediction**: Actual predicted probability
                                - Shows complete path from baseline to prediction
                            """)
                        else:  # regression or unknown
                            st.info("""
                                **Reading the Waterfall (Regression):**
                                - **Base Value**: Model's average prediction
                                - **Green bars**: Features pushing prediction lower
                                - **Red bars**: Features pushing prediction higher
                                - **Final Prediction**: Actual predicted value
                                - Shows complete path from baseline to prediction
                            """)

                    contributions_df = pd.DataFrame(explanation["individual_explanation"]["contributions"])
                    
                    
                    # Format the dataframes
                    contributions_df['Impact'] = contributions_df['Impact'].map('{:.4f}'.format)
                    contributions_df['Value'] = contributions_df.apply(
                        lambda x: f"{x['Value']:.4f}" if isinstance(x['Value'], (float, np.float64))
                        else str(x['Value']),
                        axis=1
                    )

                    st.dataframe(
                        contributions_df.style.background_gradient(
                            subset=['Impact'],
                            cmap='RdBu',
                            vmin=-abs(contributions_df['Impact'].astype(float)).max(),
                            vmax=abs(contributions_df['Impact'].astype(float)).max()
                        ))
                    
                    problem_type = st.session_state.get('problem_type', 'unknown')
                    base_value = explanation['individual_explanation']['base_value']
                    prediction_value = explanation['individual_explanation']['prediction']['predicted']
                    
                    st.info(f"""
                        💡 **Understanding the SHAP Waterfall:**
                        
                        This waterfall chart shows the complete journey from the model's baseline to your specific prediction:
                        
                        1. **Base Value ({base_value:.4f})**: The model's average prediction across all training data
                        2. **Feature contributions**: Each feature either increases (red) or decreases (green) the prediction
                        3. **Final Prediction ({prediction_value:.4f})**: The actual prediction for this sample
                        
                        The mathematical relationship is: **Base Value + Sum of all SHAP values = Final Prediction**
                    """)
                    
                    st.write("---")

                with tab2:

                    # Add LIME plot
                    st.subheader("LIME Explanation Plot")
                    
                    # First add an explanation of how LIME works
                    with st.expander("ℹ️ Understanding LIME Feature Ranges", expanded=False):
                        st.markdown("""
                            ### How LIME Creates Feature Ranges
                            
                            LIME explains predictions by:
                            1. Taking your sample point (e.g., age = 35)
                            2. Creating many similar samples by varying feature values
                            3. Finding decision boundaries that best explain the model's behavior
                            
                            The ranges you see represent these decision boundaries:
                            
                            **For Numeric Features** (e.g., Age: 29-39):
                            - Shows the range where the model behaves similarly
                            - Centered around your actual value
                            - Wider ranges = model is less sensitive to exact values
                            - Narrower ranges = model is more sensitive to changes
                            
                            **For Binary Features** (e.g., Sex: 0-1):
                            - Shows the impact of the feature changing from 0 to 1
                            - Range always shown as 0-1 for binary features
                            - Actual value is either 0 or 1
                            
                            **Impact Values:**
                            - Positive impact = feature values in this range push prediction higher
                            - Negative impact = feature values in this range push prediction lower
                            - Magnitude shows how strongly the feature affects the prediction
                        """)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if "lime_contributions" in explanation["individual_explanation"]:
                            # Create our own LIME plot instead of using the one from the explanation
                            lime_df = pd.DataFrame(explanation["individual_explanation"]["lime_contributions"])
                            
                            # Sort by absolute impact
                            lime_df['Impact'] = pd.to_numeric(lime_df['Impact'])
                            lime_df = lime_df.sort_values('Impact', key=abs, ascending=True)
                            
                            # Create the waterfall plot
                            # Create the bar chart (better for LIME than waterfall as there's no base value)
                            fig = go.Figure(go.Bar(
                                name="LIME",
                                orientation="h",
                                x=lime_df['Impact'],
                                y=lime_df['Feature'],
                                text=lime_df['Value'],  # Use the Value column directly
                                textposition="auto",
                                marker_color=['rgba(219, 64, 82, 0.7)' if x > 0 else 'rgba(50, 171, 96, 0.7)' for x in lime_df['Impact']],
                                hovertemplate="<b>%{y}</b><br>" +
                                            "Impact: %{x:.4f}<br>" +
                                            "Value: %{text}<br>" +
                                            "<extra></extra>"
                            ))
                            
                            fig.update_layout(
                                title="LIME Feature Contributions",
                                showlegend=False,
                                height=max(400, len(lime_df) * 25),
                                margin=dict(t=50, b=50, l=50, r=50),
                                yaxis=dict(
                                    title="Features",
                                    autorange="reversed"  # Put strongest features at the top
                                ),
                                xaxis=dict(
                                    title="Impact on Prediction",
                                    zeroline=True,
                                    zerolinewidth=2,
                                    zerolinecolor='rgba(0,0,0,0.2)'
                                )
                            )
                            
                            st.plotly_chart(fig, config={'responsive': True})
                            
                            # Add table showing LIME contributions
                            lime_contributions_df = pd.DataFrame(explanation["individual_explanation"]["lime_contributions"])
                            lime_contributions_df['Impact'] = lime_contributions_df['Impact'].map('{:.4f}'.format)

                            # Create display DataFrame without the Original_Description column
                            lime_display_df = lime_contributions_df[['Feature', 'Value', 'Impact']].copy()
                            
                            st.dataframe(
                                lime_display_df.style.background_gradient(
                                    subset=['Impact'],
                                    cmap='RdBu',
                                    vmin=-abs(lime_display_df['Impact'].astype(float)).max(),
                                    vmax=abs(lime_display_df['Impact'].astype(float)).max()
                                ))
                            
                            problem_type = st.session_state.get('problem_type', 'unknown')
                            if problem_type == "multiclass_classification":
                                st.info("""
                                    💡 **LIME Contributions (Multiclass)**
                                    
                                    These values show how each feature contributes to the 
                                    probability of the predicted class based on LIME's 
                                    local approximation.
                                    
                                    **Value Format:**
                                    - Range: Decision boundaries for the predicted class
                                    - Value: Thresholds that affect class prediction
                                    - Actual: The actual feature value
                                    - Impact: Effect on predicted class probability
                                """)
                            elif problem_type == "binary_classification":
                                st.info("""
                                    💡 **LIME Contributions (Binary)**
                                    
                                    These values show how each feature contributes to the 
                                    probability of the positive class based on LIME's 
                                    local approximation.
                                    
                                    **Value Format:**
                                    - Range: Decision boundaries for classification
                                    - Value: Thresholds that affect positive/negative prediction
                                    - Actual: The actual feature value
                                    - Impact: Effect on positive class probability
                                """)
                            else:  # regression or unknown
                                st.info("""
                                    💡 **LIME Contributions (Regression)**
                                    
                                    These values show how each feature contributes to the 
                                    predicted value based on LIME's local approximation 
                                    of the model.
                                    
                                    **Value Format:**
                                    - Range: Value ranges for numeric features
                                    - Value: Specific thresholds that affect prediction
                                    - Actual: The actual feature value
                                    - Impact: Direct effect on predicted value
                                """)
                        else:
                            st.warning("LIME plot not available for this model type")
                    
                    with col2:
                        problem_type = st.session_state.get('problem_type', 'unknown')
                        if problem_type == "multiclass_classification":
                            st.info("""
                                **Reading the LIME Plot (Multiclass):**
                                - Red = increasing predicted class probability
                                - Green = decreasing predicted class probability
                                - Length = strength of impact on predicted class
                                - Shows contributions for predicted class only
                                
                                **Feature Ranges:**
                                - Show decision boundaries for predicted class
                                - Centered around actual values
                                - Binary features show 0-1 range
                                - Wider ranges = more stable class predictions
                                
                                **Example:**
                                If Age=35 shows range 29-39:
                                - Model treats ages 29-39 similarly for this class
                                - Changes within range have less impact on class prediction
                                - Changes outside range may change predicted class
                            """)
                        elif problem_type == "binary_classification":
                            st.info("""
                                **Reading the LIME Plot (Binary):**
                                - Red = increasing positive class probability
                                - Green = decreasing positive class probability
                                - Length = strength of impact on classification
                                - Shows likelihood of positive class
                                
                                **Feature Ranges:**
                                - Show decision boundaries for classification
                                - Centered around actual values
                                - Binary features show 0-1 range
                                - Wider ranges = more stable classification
                                
                                **Example:**
                                If Age=35 shows range 29-39:
                                - Model treats ages 29-39 similarly for classification
                                - Changes within range less likely to flip prediction
                                - Changes outside range may change classification
                            """)
                        else:  # regression or unknown
                            st.info("""
                                **Reading the LIME Plot (Regression):**
                                - Red = increasing predicted value
                                - Green = decreasing predicted value
                                - Length = strength of impact on value
                                - Shows direct effect on predicted number
                                
                                **Feature Ranges:**
                                - Show where model behavior changes
                                - Centered around actual values
                                - Binary features show 0-1 range
                                - Wider ranges = more stable predictions
                                
                                **Example:**
                                If Age=35 shows range 29-39:
                                - Model treats ages 29-39 similarly
                                - Changes within range have less impact on value
                                - Changes outside range may have different effects
                            """)

                    st.write("---")

                with tab3:
                    # SHAP vs LIME Comparison
                    st.subheader("SHAP vs LIME Feature Contribution Comparison")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if "contributions" in explanation["individual_explanation"] and "lime_contributions" in explanation["individual_explanation"]:
                            # Create combined dataframe for comparison
                            shap_df = pd.DataFrame(explanation["individual_explanation"]["contributions"])
                            lime_df = pd.DataFrame(explanation["individual_explanation"]["lime_contributions"])
                            
                            # Ensure both SHAP and LIME values are numeric
                            shap_df['Impact'] = pd.to_numeric(shap_df['Impact'], errors='coerce')
                            lime_df['Impact'] = pd.to_numeric(lime_df['Impact'], errors='coerce')
                            
                            # Normalize both SHAP and LIME values to make them comparable
                            # Use min-max scaling to preserve sign and relative magnitude
                            def normalize_impacts(values):
                                max_abs = max(abs(values.min()), abs(values.max()))
                                return values / max_abs if max_abs != 0 else values
                            
                            shap_df['Impact_Normalized'] = normalize_impacts(shap_df['Impact'])
                            lime_df['Impact_Normalized'] = normalize_impacts(lime_df['Impact'])
                            
                            # Create comparison plot
                            fig = go.Figure()
                            
                            # Add SHAP values
                            fig.add_trace(go.Bar(
                                name='SHAP',
                                y=shap_df['Feature'],
                                x=shap_df['Impact_Normalized'],
                                orientation='h',
                                marker_color='rgba(219, 64, 82, 0.7)',
                                text=shap_df.apply(lambda x: f"Value: {x['Value']}<br>Impact: {x['Impact']:.4f}", axis=1),
                                textposition='auto',
                                hovertemplate="<b>%{y}</b><br>" +
                                            "%{text}<br>" +
                                            "Normalized Impact: %{x:.4f}<br>" +
                                            "<extra></extra>"
                            ))
                            
                            # Add LIME values
                            fig.add_trace(go.Bar(
                                name='LIME',
                                y=lime_df['Feature'],
                                x=lime_df['Impact_Normalized'],
                                orientation='h',
                                marker_color='rgba(50, 171, 96, 0.7)',
                                text=lime_df.apply(lambda x: f"Value: {x['Value']}<br>Impact: {x['Impact']:.4f}", axis=1),
                                textposition='auto',
                                hovertemplate="<b>%{y}</b><br>" +
                                            "%{text}<br>" +
                                            "Normalized Impact: %{x:.4f}<br>" +
                                            "<extra></extra>"
                            ))
                            
                            fig.update_layout(
                                title={
                                    'text': "SHAP vs LIME Feature Contributions (Normalized)",
                                    'x': 0.5,
                                    'xanchor': 'center'
                                },
                                barmode='group',
                                height=max(400, len(shap_df) * 25),
                                margin=dict(t=50, b=50, l=50, r=50),
                                yaxis=dict(
                                    title="Features",
                                    autorange="reversed"  # Put strongest features at the top
                                ),
                                xaxis=dict(
                                    title="Normalized Impact on Prediction",
                                    zeroline=True,
                                    zerolinewidth=2,
                                    zerolinecolor='rgba(0,0,0,0.2)'
                                )
                            )
                            
                            st.plotly_chart(fig, config={'responsive': True})
                            
                        else:
                            st.warning("Comparison plot not available")
                    
                    with col2:
                        problem_type = st.session_state.get('problem_type', 'unknown')
                        if problem_type == "multiclass_classification":
                            st.info("""
                                **Reading the Comparison (Multiclass):**
                                - Red bars = SHAP values for predicted class
                                - Green bars = LIME values for predicted class
                                - Values normalized to [-1, 1] for comparison
                                - Hover shows both normalized and original values
                                - Longer bars = stronger impact on class probability
                                
                                **Normalization:**
                                - Both methods scaled to same range
                                - Preserves relative importance and direction
                                - Original class probability impacts in hover
                                
                                **Interpretation:**
                                - Agreement = reliable class prediction factors
                                - Differences = examine class decision boundaries
                                - Both methods focus on predicted class only
                            """)
                        elif problem_type == "binary_classification":
                            st.info("""
                                **Reading the Comparison (Binary):**
                                - Red bars = SHAP values for positive class
                                - Green bars = LIME values for positive class
                                - Values normalized to [-1, 1] for comparison
                                - Hover shows both normalized and original values
                                - Longer bars = stronger impact on classification
                                
                                **Normalization:**
                                - Both methods scaled to same range
                                - Preserves relative importance and direction
                                - Original probability impacts in hover
                                
                                **Interpretation:**
                                - Agreement = reliable classification factors
                                - Differences = examine decision boundaries
                                - Both methods focus on positive class likelihood
                            """)
                        else:  # regression or unknown
                            st.info("""
                                **Reading the Comparison (Regression):**
                                - Red bars = SHAP values
                                - Green bars = LIME values
                                - Values normalized to [-1, 1] for comparison
                                - Hover shows both normalized and original values
                                - Longer bars = stronger impact on predicted value
                                
                                **Normalization:**
                                - Both methods scaled to same range
                                - Preserves relative importance and direction
                                - Original value impacts shown in hover
                                
                                **Interpretation:**
                                - Agreement = reliable prediction factors
                                - Differences = examine local vs global effects
                                - Both methods show direct value impacts
                            """)
                    st.write("---")
                    
            else:
                st.error(explanation["message"])
    else:
        st.error("No test data available for explanation.") 