"""
Model selection interface components for ML Builder.

Contains UI components for model selection including model explanations,
comparison displays, and selection interfaces.
"""

import streamlit as st
import pandas as pd
from content.content_manager import ContentManager


def render_model_explainer_section(content_manager: ContentManager):
    """
    Render the comprehensive model explainer section.

    Args:
        content_manager: ContentManager instance for accessing content
    """
    with st.expander("📚 Understanding Available Models - Detailed Comparison"):
        # Introduction
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: #262730;'>🤖 Machine Learning Model Guide</h3>
            <p>Understanding the strengths and trade-offs of different machine learning models is crucial for making the right choice for your specific use case.</p>
        </div>
        """, unsafe_allow_html=True)

        # Create tabs for different model categories
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Linear Models", "🌳 Tree-Based Models", "🧠 Neural Networks", "🚀 Gradient Boosting"])

        with tab1:
            linear_tab1, linear_tab2, linear_tab3 = st.tabs(["Linear/Logistic Regression", "Ridge Regression", "Naive Bayes"])

            with linear_tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(content_manager.get_model_detailed_explanation("linear_models"))

            with linear_tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(content_manager.get_model_detailed_explanation("ridge_regression"))

            with linear_tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(content_manager.get_model_detailed_explanation("naive_bayes"))

        with tab2:
            tree_tab1, tree_tab2 = st.tabs(["Decision Trees", "Random Forest"])

            with tree_tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(content_manager.get_model_detailed_explanation("decision_trees"))

            with tree_tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(content_manager.get_model_detailed_explanation("random_forest"))

        with tab3:
            st.markdown(content_manager.get_model_detailed_explanation("mlp"))

        with tab4:
            boost_tab1, boost_tab2, boost_tab3, boost_tab4 = st.tabs([
                "Hist Gradient Boosting",
                "XGBoost",
                "LightGBM",
                "CatBoost"
            ])

            with boost_tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(content_manager.get_model_detailed_explanation("hist_gradient_boosting"))

            with boost_tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(content_manager.get_model_detailed_explanation("xgboost"))

            with boost_tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(content_manager.get_model_detailed_explanation("lightgbm"))

            with boost_tab4:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(content_manager.get_model_detailed_explanation("catboost"))

        # Comparison Chart Section
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: #262730;'>📊 Model Comparison Chart</h3>
            <p>Quick reference guide for comparing model characteristics</p>
        </div>
        """, unsafe_allow_html=True)

        # Convert comparison chart to DataFrame for better display
        chart_data = content_manager.get_model_comparison_chart()
        df = pd.DataFrame(chart_data)

        # Display with custom styling for better readability
        st.dataframe(
            df,
            width='stretch',
            hide_index=True,
            height=400
        )

        # Quick Selection Guide
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: #262730;'>🎯 Quick Selection Guide</h3>
            <p>Choose the right model based on your specific needs</p>
        </div>
        """, unsafe_allow_html=True)

        quick_guide_col1, quick_guide_col2 = st.columns(2)

        with quick_guide_col1:
            st.markdown(content_manager.get_quick_selection_guide("linear_models"))
            st.markdown(content_manager.get_quick_selection_guide("ridge_regression"))
            st.markdown(content_manager.get_quick_selection_guide("naive_bayes"))
            st.markdown(content_manager.get_quick_selection_guide("decision_trees"))
            st.markdown(content_manager.get_quick_selection_guide("random_forest"))

        with quick_guide_col2:
            st.markdown(content_manager.get_quick_selection_guide("mlp"))
            st.markdown(content_manager.get_quick_selection_guide("hist_gradient_boosting"))
            st.markdown(content_manager.get_quick_selection_guide("xgboost"))
            st.markdown(content_manager.get_quick_selection_guide("lightgbm"))
            st.markdown(content_manager.get_quick_selection_guide("catboost"))


def render_performance_metrics_explainer(content_manager: ContentManager, problem_type):
    """
    Render the performance metrics explainer section.

    Args:
        content_manager: ContentManager instance for accessing content
        problem_type: Type of ML problem (classification or regression)
    """
    with st.expander("📊 Understanding Model Performance Metrics - Detailed Guide"):
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: #262730;'>📈 Model Evaluation Metrics Guide</h3>
            <p>Understanding what each metric measures and when to prioritize different metrics for your specific use case.</p>
        </div>
        """, unsafe_allow_html=True)

        # Handle both binary and multiclass classification
        if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
            st.markdown("### Classification Metrics")

            # Create tabs for different classification metrics
            acc_tab, prec_tab, rec_tab, f1_tab = st.tabs(["🎯 Accuracy", "🔍 Precision", "📡 Recall", "⚖️ F1-Score"])

            with acc_tab:
                col1, col2 = st.columns([2, 1])
                with col1:
                    accuracy_info = content_manager.get_performance_metrics_explanation("classification", "accuracy")
                    st.markdown(f"""
                    #### What is Accuracy?
                    {accuracy_info.get('what_is', '')}

                    **Formula:** {accuracy_info.get('formula', '')}

                    **Range:** {accuracy_info.get('range', '')}
                    """)

                with col2:
                    best_used = accuracy_info.get('best_used_when', [])
                    avoid_when = accuracy_info.get('avoid_when', [])
                    st.info(f"""
                    **Best Used When:**
                    {''.join([f'✅ {item}' for item in best_used])}

                    **Avoid When:**
                    {''.join([f'❌ {item}' for item in avoid_when])}
                    """)

                if 'warning' in accuracy_info:
                    st.warning(accuracy_info['warning'])

            with prec_tab:
                col1, col2 = st.columns([2, 1])
                with col1:
                    precision_info = content_manager.get_performance_metrics_explanation("classification", "precision")
                    st.markdown(f"""
                    #### What is Precision?
                    {precision_info.get('what_is', '')}

                    **Formula:** {precision_info.get('formula', '')}

                    **Range:** {precision_info.get('range', '')}
                    """)

                with col2:
                    prioritize_when = precision_info.get('prioritize_when', [])
                    examples = precision_info.get('examples', [])
                    st.info(f"""
                    **Prioritize When:**
                    {chr(10).join([f'✅ {item}' for item in prioritize_when])}. 
                    **Examples:**
                    {chr(10).join(examples)}
                    """)

                st.markdown("""
                **Real-World Example:**
                - **Email Spam Filter:** High precision means fewer legitimate emails go to spam folder
                - **Medical Test:** High precision means fewer healthy people are told they're sick
                - **Security System:** High precision means fewer false security alerts
                """)

            with rec_tab:
                col1, col2 = st.columns([2, 1])
                with col1:
                    recall_info = content_manager.get_performance_metrics_explanation("classification", "recall")
                    st.markdown(f"""
                    #### What is Recall?
                    {recall_info.get('what_is', '')}

                    **Formula:** {recall_info.get('formula', '')}

                    **Range:** {recall_info.get('range', '')}
                    """)

                with col2:
                    prioritize_when = recall_info.get('prioritize_when', [])
                    examples = recall_info.get('examples', [])
                    st.info(f"""
                    **Prioritize When:**
                    {chr(10).join([f'✅ {item}' for item in prioritize_when])}. 
                    **Examples:**
                    {chr(10).join(examples)}
                    """)

                st.markdown("""
                **Real-World Example:**
                - **Cancer Screening:** High recall means fewer cancer cases are missed
                - **Fire Detection:** High recall means fewer real fires go undetected
                - **Search Engine:** High recall means fewer relevant results are missed
                """)

            with f1_tab:
                col1, col2 = st.columns([2, 1])
                with col1:
                    f1_info = content_manager.get_performance_metrics_explanation("classification", "f1_score")
                    st.markdown(f"""
                    #### What is F1-Score?
                    {f1_info.get('what_is', '')}

                    **Formula:** {f1_info.get('formula', '')}

                    **Range:** {f1_info.get('range', '')}
                    """)

                with col2:
                    best_used_when = f1_info.get('best_used_when', [])
                    st.info(f"""
                    **Best Used When:**
                    {chr(10).join([f'✅ {item}' for item in best_used_when])}. 
                    **The Gold Standard:**
                    🏆 Most commonly used metric
                    """)

                st.markdown(f"""
                **Why Harmonic Mean?**
                {f1_info.get('why_harmonic_mean', 'The harmonic mean penalizes extreme values more than arithmetic mean - both precision AND recall must be high for a high F1-score')}

                **Example:**
                - Precision = 1.0, Recall = 0.1 → F1 = 0.18 (punishes the low recall)
                - Precision = 0.8, Recall = 0.8 → F1 = 0.80 (rewards balance)
                """)

            # Precision-Recall Trade-off explanation
            st.markdown("---")
            st.markdown("""
            ### 🎯 The Precision-Recall Trade-off

            **Key Insight:** There's almost always a trade-off between precision and recall!

            | Scenario | High Precision | High Recall | Result |
            |----------|---------------|-------------|---------|
            | Conservative Model | ✅ Few false alarms | ❌ Misses many cases | Very strict criteria |
            | Liberal Model | ❌ Many false alarms | ✅ Catches most cases | Very loose criteria |
            | Balanced Model | ⚖️ Moderate false alarms | ⚖️ Moderate missed cases | **F1-Score optimized** |

            **💡 This is why F1-Score is so valuable** - it helps you find the sweet spot!
            """)

        else:  # regression
            st.markdown("### Regression Metrics")

            # Create tabs for different regression metrics
            r2_tab, mae_tab, mse_tab, rmse_tab = st.tabs(["📊 R² Score", "📏 MAE", "📐 MSE", "📈 RMSE"])

            with r2_tab:
                col1, col2 = st.columns([2, 1])
                with col1:
                    r2_info = content_manager.get_performance_metrics_explanation("regression", "r2_score")
                    st.markdown(f"""
                    #### What is R² (R-Squared)?
                    {r2_info.get('what_is', '')}

                    **Formula:** {r2_info.get('formula', '')}

                    **Range:** {r2_info.get('range', '')}
                    """)

                with col2:
                    guidelines = r2_info.get('guidelines', {})
                    st.info(f"""
                    **Interpretation:**
                    🎯 **{guidelines.get('excellent', '')}** = Excellent
                    ✅ **{guidelines.get('good', '')}** = Good
                    ⚠️ **{guidelines.get('moderate', '')}** = Moderate
                    ❌ **{guidelines.get('poor', '')}** = Poor
                    """)

            with mae_tab:
                col1, col2 = st.columns([2, 1])
                with col1:
                    mae_info = content_manager.get_performance_metrics_explanation("regression", "mae")
                    st.markdown(f"""
                    #### What is MAE (Mean Absolute Error)?
                    {mae_info.get('what_is', '')}

                    **Formula:** {mae_info.get('formula', '')}

                    **Range:** {mae_info.get('range', '')}

                    **Units:** {mae_info.get('units', '')}
                    """)

                with col2:
                    advantages = mae_info.get('advantages', [])
                    best_used_when = mae_info.get('best_used_when', [])
                    st.info(f"""
                    **Key Advantages:**
                    {chr(10).join([f'✅ {item}' for item in advantages])}. 
                    **Best Used When:**
                    {chr(10).join([f'📊 {item}' for item in best_used_when])}
                    """)

                st.markdown("""
                **Real-World Example:**
                - **House Prices:** MAE = $15,000 means predictions are off by $15,000 on average
                - **Temperature:** MAE = 2.5°F means predictions are off by 2.5 degrees on average
                - **Sales:** MAE = 100 units means predictions are off by 100 units on average
                """)

            with mse_tab:
                col1, col2 = st.columns([2, 1])
                with col1:
                    mse_info = content_manager.get_performance_metrics_explanation("regression", "mse")
                    st.markdown(f"""
                    #### What is MSE (Mean Squared Error)?
                    {mse_info.get('what_is', '')}

                    **Formula:** {mse_info.get('formula', '')}

                    **Range:** {mse_info.get('range', '')}

                    **Units:** {mse_info.get('units', '')}
                    """)

                with col2:
                    characteristics = mse_info.get('characteristics', [])
                    best_used_when = mse_info.get('best_used_when', [])
                    st.info(f"""
                    **Key Characteristics:**
                    {chr(10).join([f'⚠️ {item}' for item in characteristics])}. 
                    **Best Used When:**
                    {chr(10).join([f'💥 {item}' for item in best_used_when])}
                    """)

                st.markdown("""
                **Why Squared Errors?**
                - **Magnifies large errors:** Error of 10 contributes 100 to MSE, but error of 1 contributes only 1
                - **Smooth for optimization:** Makes it easier for algorithms to find the best model
                - **Mathematical convenience:** Works well with many statistical methods

                **Trade-off:** More sensitive to outliers than MAE
                """)

            with rmse_tab:
                col1, col2 = st.columns([2, 1])
                with col1:
                    rmse_info = content_manager.get_performance_metrics_explanation("regression", "rmse")
                    st.markdown(f"""
                    #### What is RMSE (Root Mean Squared Error)?
                    {rmse_info.get('what_is', '')}

                    **Formula:** {rmse_info.get('formula', '')}

                    **Range:** {rmse_info.get('range', '')}

                    **Units:** {rmse_info.get('units', '')}
                    """)

                with col2:
                    benefits = rmse_info.get('benefits', [])
                    st.info(f"""
                    **Best of Both Worlds:**
                    {chr(10).join([f'✅ {item}' for item in benefits])}. 
                    **The Gold Standard:**
                    🏆 Most popular regression metric
                    """)

                comparison = rmse_info.get('comparison', {})
                st.markdown(f"""
                **RMSE vs MAE:**
                - **RMSE ≈ MAE:** {comparison.get('rmse_approx_mae', 'Errors are consistent in size')}
                - **RMSE >> MAE:** {comparison.get('rmse_much_greater_mae', 'You have some large errors (outliers)')}
                - **RMSE = MAE:** {comparison.get('rmse_equals_mae', 'All errors are exactly the same size (rare!)')}

                **Real-World Example:**
                - **House Prices:** RMSE = $20,000 means typical prediction error is $20,000
                - **Temperature:** RMSE = 3.2°F means typical error is about 3.2 degrees
                """)

            # Metric comparison for regression
            st.markdown("---")
            st.markdown("""
            ### 📊 Choosing the Right Regression Metric

            | Metric | Interpretation | Best Used When | Sensitivity to Outliers |
            |--------|---------------|----------------|------------------------|
            | **R²** | Variance explained | Understanding model fit | Medium |
            | **MAE** | Average error magnitude | Simple interpretation needed | Low |
            | **MSE** | Squared error average | Training/optimization | High |
            | **RMSE** | Typical error size | General model comparison | High |

            **💡 Pro Tip:**
            - Use **R²** to understand overall model performance
            - Use **RMSE** for comparing different models
            - Use **MAE** when you want simple, robust error measurement
            - Compare **RMSE vs MAE** to detect outliers in your predictions
            """)

        # General guidance section
        st.markdown("---")
        st.markdown("""
        ### 🎯 Quick Decision Guide

        **For Model Selection:**
        1. **Look at the primary metric first** (F1 for classification, R² for regression)
        2. **Check for balance** - avoid models that excel in one metric but fail in others
        3. **Consider your specific use case** - do you care more about false positives or false negatives?
        4. **Remember**: These are sample results with default parameters!

        **🚨 Important Reminders:**
        - These metrics are calculated on a **small sample** with **default parameters**
        - Final performance will likely be different with the **full dataset** and **proper tuning**
        - The ranking of models may change significantly in the final evaluation
        """)


def display_model_information(content_manager: ContentManager, selected_model):
    """
    Display detailed information about the selected model.

    Args:
        content_manager: ContentManager instance for accessing content
        selected_model: The selected model key
    """
    st.write("### Model Information")

    model_char = content_manager.get_model_characteristics(selected_model)
    if model_char:
        st.write(model_char.get("description", "No description available."))

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Best Used For:**")
            for use_case in model_char.get("best_for", []):
                st.write(f"• {use_case}")

        with col2:
            st.write("**Limitations:**")
            for limitation in model_char.get("limitations", []):
                st.write(f"• {limitation}")
    else:
        st.write("Model information not available.")