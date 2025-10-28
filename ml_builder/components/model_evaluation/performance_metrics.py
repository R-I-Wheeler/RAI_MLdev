import streamlit as st
import pandas as pd
import numpy as np

def display_classification_report(result):
    st.write("### 📋 Detailed Classification Report")
    
    # Check if optimal threshold has been applied and display status
    if (hasattr(st.session_state, 'builder') and 
        hasattr(st.session_state.builder, 'model') and 
        st.session_state.builder.model.get("threshold_optimized", False)):
        
        optimal_threshold = st.session_state.builder.model.get("optimal_threshold", 0.5)
        is_binary = st.session_state.builder.model.get("threshold_is_binary", True)
        
        if is_binary:
            st.success(f"🎯 **Using Optimized Decision Threshold**: {optimal_threshold:.3f} (instead of default 0.5)")
        else:
            st.success(f"🎯 **Using Optimized Confidence Threshold**: {optimal_threshold:.3f} (for multiclass predictions)")
        
        st.info("💡 These performance metrics reflect predictions made with the optimized threshold.")
        st.markdown("---")
    else:
        # Only show for classification models
        problem_type = getattr(st.session_state.builder, 'model', {}).get('problem_type', '')
        if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
            st.info("📊 These metrics use the default prediction threshold (0.5 for binary classification).")
            
            # Check if model supports threshold optimization but it hasn't been applied
            model_instance = st.session_state.builder.model.get("active_model") or st.session_state.builder.model.get("model", None)
            if model_instance and hasattr(model_instance, 'predict_proba'):
                with st.expander("💡 Want to optimize the decision threshold?", expanded=False):
                    st.markdown("""
                    **Your model supports threshold optimization!**
                    
                    To potentially improve performance:
                    1. Go back to **Model Training** page
                    2. Scroll down to **"🎯 Probability Threshold Optimization"** section  
                    3. Run threshold analysis to find the optimal threshold
                    4. **Important**: Click **"🎯 Apply Optimal Threshold to Model"** to save it
                    5. Return here to see the updated metrics
                    
                    **Why optimize thresholds?**
                    - Default 0.5 threshold may not be optimal for your specific use case
                    - Can significantly improve precision, recall, or F1-score
                    - Especially helpful for imbalanced datasets
                    """)
            
            st.markdown("---")
    
    # Create a more visually appealing classification report
    report_dict = result["classification_report"]
    
    # Separate class-specific metrics from aggregates
    class_metrics = {k: v for k, v in report_dict.items() 
                    if k not in ['accuracy', 'macro avg', 'weighted avg']}
    
    # Create DataFrame for class-specific metrics
    class_df = pd.DataFrame.from_dict(class_metrics, orient='index')
    class_df = class_df.round(3)
    
    # Get aggregate metrics
    agg_metrics = {k: v for k, v in report_dict.items() 
                  if k in ['macro avg', 'weighted avg']}
    
    # Display overall accuracy prominently
    if 'accuracy' in report_dict:
        st.write("#### 🎯 Overall Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        accuracy = report_dict['accuracy']
        accuracy_color = "green" if accuracy >= 0.8 else "orange" if accuracy >= 0.6 else "red"
        accuracy_status = "Excellent" if accuracy >= 0.8 else "Good" if accuracy >= 0.6 else "Needs Improvement"
        
        with col1:
            st.metric("Overall Accuracy", f"{accuracy:.3f}", help="Percentage of correct predictions")
            st.markdown(f"<span style='color: {accuracy_color}'>{accuracy_status}</span>", unsafe_allow_html=True)
        
        # Add macro and weighted averages if available
        if 'macro avg' in agg_metrics:
            macro_f1 = agg_metrics['macro avg']['f1-score']
            with col2:
                st.metric("Macro F1-Score", f"{macro_f1:.3f}", help="Average F1-score across all classes")
                macro_color = "green" if macro_f1 >= 0.7 else "orange" if macro_f1 >= 0.5 else "red"
                st.markdown(f"<span style='color: {macro_color}'>{'Excellent' if macro_f1 >= 0.7 else 'Good' if macro_f1 >= 0.5 else 'Needs Work'}</span>", unsafe_allow_html=True)
        
        if 'weighted avg' in agg_metrics:
            weighted_f1 = agg_metrics['weighted avg']['f1-score']
            with col3:
                st.metric("Weighted F1-Score", f"{weighted_f1:.3f}", help="F1-score weighted by class support")
                weighted_color = "green" if weighted_f1 >= 0.7 else "orange" if weighted_f1 >= 0.5 else "red"
                st.markdown(f"<span style='color: {weighted_color}'>{'Excellent' if weighted_f1 >= 0.7 else 'Good' if weighted_f1 >= 0.5 else 'Needs Work'}</span>", unsafe_allow_html=True)
        
        # Calculate number of classes
        with col4:
            n_classes = len(class_metrics)
            st.metric("Number of Classes", f"{n_classes}", help="Total classes in the dataset")
            class_balance = "Balanced" if n_classes <= 5 else "Multi-class"
            st.markdown(f"<span style='color: blue'>{class_balance}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced class-specific metrics display
    st.write("#### 📊 Class-by-Class Performance")
    
    # Add support column if available
    if 'support' in class_df.columns:
        class_df_display = class_df[['precision', 'recall', 'f1-score', 'support']].copy()
        # Convert support to integers
        class_df_display['support'] = class_df_display['support'].astype(int)
    else:
        class_df_display = class_df[['precision', 'recall', 'f1-score']].copy()
    
    # Enhanced styling function with color coding
    def style_classification_dataframe(df):
        def highlight_performance(val):
            if pd.isna(val):
                return ''
            
            try:
                val = float(val)
                if val >= 0.8:
                    return 'background-color: #d4edda; color: #155724'  # Green
                elif val >= 0.6:
                    return 'background-color: #fff3cd; color: #856404'  # Yellow
                elif val >= 0.4:
                    return 'background-color: #ffeaa7; color: #6c757d'  # Light orange
                else:
                    return 'background-color: #f8d7da; color: #721c24'  # Red
            except:
                return ''  # No styling for non-numeric values (like support column)
        
        # Format numbers first
        format_dict = {col: "{:.3f}" for col in ['precision', 'recall', 'f1-score'] if col in df.columns}
        if 'support' in df.columns:
            format_dict['support'] = "{:d}"
        
        # Apply styling to performance metrics columns only
        style_cols = [col for col in ['precision', 'recall', 'f1-score'] if col in df.columns]
        
        return df.style.format(format_dict).applymap(highlight_performance, subset=style_cols)
    
    # Display the styled dataframe
    st.dataframe(style_classification_dataframe(class_df_display), width='stretch')
    
    # Performance legend
    col1, col2 = st.columns([3, 1])
    with col2:
        st.write("**Performance Scale:**")
        st.markdown("🟢 **Excellent** (≥ 0.8)")
        st.markdown("🟡 **Good** (0.6 - 0.8)")
        st.markdown("🟠 **Fair** (0.4 - 0.6)")
        st.markdown("🔴 **Poor** (< 0.4)")
    
    with col1:
        # Add insights based on performance
        st.write("**📈 Performance Insights:**")
        
        # Identify best and worst performing classes
        f1_scores = class_df_display['f1-score']
        best_class = f1_scores.idxmax()
        worst_class = f1_scores.idxmin()
        
        avg_f1 = f1_scores.mean()
        
        insights = []
        
        if avg_f1 >= 0.8:
            insights.append("✅ **Excellent overall performance** across all classes")
        elif avg_f1 >= 0.6:
            insights.append("👍 **Good overall performance** with room for improvement")
        else:
            insights.append("⚠️ **Performance needs attention** - consider model improvements")
        
        insights.append(f"🏆 **Best performing class**: {best_class} (F1: {f1_scores[best_class]:.3f})")
        
        if f1_scores[worst_class] < 0.5:
            insights.append(f"🎯 **Focus on improving**: {worst_class} (F1: {f1_scores[worst_class]:.3f})")
        
        # Check for class imbalance issues
        if 'support' in class_df_display.columns:
            support_vals = class_df_display['support']
            if support_vals.max() / support_vals.min() > 10:
                insights.append("⚖️ **Class imbalance detected** - consider balancing techniques")
        
        # Performance variability
        f1_std = f1_scores.std()
        if f1_std > 0.2:
            insights.append("📊 **High performance variability** between classes")
        elif f1_std < 0.1:
            insights.append("📊 **Consistent performance** across all classes")
        
        for insight in insights:
            st.markdown(insight)
    
    st.markdown("---")
    
    # Aggregate metrics in a more visual way
    if agg_metrics:
        st.write("#### 📈 Summary Statistics")
        
        agg_df = pd.DataFrame.from_dict(agg_metrics, orient='index')
        agg_df = agg_df.round(3)
        
        # Create visual metrics for macro and weighted averages
        col1, col2 = st.columns(2)
        
        with col1:
            if 'macro avg' in agg_metrics:
                st.write("**Macro Average** (unweighted mean)")
                macro_data = agg_metrics['macro avg']
                
                macro_col1, macro_col2, macro_col3 = st.columns(3)
                with macro_col1:
                    st.metric("Precision", f"{macro_data['precision']:.3f}")
                with macro_col2:
                    st.metric("Recall", f"{macro_data['recall']:.3f}")
                with macro_col3:
                    st.metric("F1-Score", f"{macro_data['f1-score']:.3f}")
                
                st.caption("Treats all classes equally regardless of size")
        
        with col2:
            if 'weighted avg' in agg_metrics:
                st.write("**Weighted Average** (by class size)")
                weighted_data = agg_metrics['weighted avg']
                
                weight_col1, weight_col2, weight_col3 = st.columns(3)
                with weight_col1:
                    st.metric("Precision", f"{weighted_data['precision']:.3f}")
                with weight_col2:
                    st.metric("Recall", f"{weighted_data['recall']:.3f}")
                with weight_col3:
                    st.metric("F1-Score", f"{weighted_data['f1-score']:.3f}")
                
                st.caption("Accounts for class size differences")
    
    # Enhanced educational content
    with st.expander("📚 Understanding Classification Metrics"):
        st.markdown("""
        ### 🎯 **Core Metrics Explained**
        
        **Precision**:
        - *"When I predict this class, how often am I right?"*
        - High precision = Few false positives
        - Important when false positives are costly
        - **Thresholds**: 🟢 ≥0.8 | 🟡 0.6-0.8 | 🟠 0.4-0.6 | 🔴 <0.4
        
        **Recall (Sensitivity)**:
        - *"Of all actual cases of this class, how many did I find?"*
        - High recall = Few false negatives  
        - Important when missing cases is costly
        - **Thresholds**: 🟢 ≥0.8 | 🟡 0.6-0.8 | 🟠 0.4-0.6 | 🔴 <0.4
        
        **F1-Score**:
        - *"Balanced measure combining precision and recall"*
        - Harmonic mean of precision and recall
        - Good single metric for overall performance
        - **Thresholds**: 🟢 ≥0.8 | 🟡 0.6-0.8 | 🟠 0.4-0.6 | 🔴 <0.4
        
        **Support**:
        - *"How many samples of this class are in the test set"*
        - Indicates class representation in test data
        - Affects reliability of per-class metrics
        - Higher support = more reliable metrics
        
        ### 📊 **Averaging Methods**
        
        **Macro Average**:
        - Simple average across all classes
        - Treats all classes equally regardless of size
        - Good for balanced datasets
        - Sensitive to minority class performance
        
        **Weighted Average**:
        - Average weighted by class size (support)
        - Accounts for class imbalance
        - Dominated by majority classes
        - More representative for imbalanced datasets
        
        ### 🎯 **Performance Guidelines**
        
        **Overall Performance Scale**:
        - **0.9+**: Exceptional performance
        - **0.8-0.9**: Excellent performance
        - **0.6-0.8**: Good performance
        - **0.4-0.6**: Fair performance (needs improvement)
        - **Below 0.4**: Poor performance (significant issues)
        
        **Class Balance Considerations**:
        - Large differences in support indicate class imbalance
        - Focus on macro avg for equal treatment of all classes
        - Focus on weighted avg for overall dataset performance
        """)
    
    with st.expander("🔧 Improvement Recommendations"):
        # Generate specific recommendations based on the results
        recommendations = []
        
        if avg_f1 < 0.6:
            recommendations.extend([
                "🔄 **Try different algorithms**: Random Forest, Gradient Boosting, or Neural Networks",
                "🎯 **Feature engineering**: Create new features or transform existing ones",
                "📊 **Data quality**: Check for missing values, outliers, or data leakage"
            ])
        
        if 'support' in class_df_display.columns:
            support_vals = class_df_display['support']
            if support_vals.max() / support_vals.min() > 5:
                recommendations.extend([
                    "⚖️ **Address class imbalance**: Use SMOTE, class weights, or stratified sampling",
                    "📈 **Collect more data**: Especially for underrepresented classes",
                    "🎯 **Adjust thresholds**: Use precision-recall curve to find optimal cutoffs"
                ])
        
        # Performance variability recommendations
        if f1_scores.std() > 0.2:
            recommendations.extend([
                "🎯 **Class-specific tuning**: Optimize hyperparameters for worst-performing classes",
                "🔍 **Feature analysis**: Identify which features help distinguish problem classes",
                "📊 **Ensemble methods**: Combine multiple models for better consistency"
            ])
        
        # Precision vs Recall trade-offs
        precision_scores = class_df_display['precision']
        recall_scores = class_df_display['recall']
        
        if (precision_scores - recall_scores).abs().mean() > 0.2:
            recommendations.append("⚖️ **Balance precision and recall**: Adjust decision thresholds or try different algorithms")
        
        if not recommendations:
            recommendations = ["🎉 **Great job!** Your model shows excellent performance across all metrics."]
        
        for rec in recommendations:
            st.markdown(rec)

    st.session_state.logger.log_journey_point(
        stage="MODEL_EVALUATION",
        decision_type="MODEL_EVALUATION",
        description="Classification Model Performance Metrics",
        details={"Model Type": st.session_state.builder.model['type'],
                "Overall Accuracy": f"{accuracy:.3f}",
                "Precision": f"{weighted_data['precision']:.3f}",
                "Recall": f"{weighted_data['recall']:.3f}",
                "F1-Score": f"{weighted_data['f1-score']:.3f}"
                },
        parent_id=None
    )

def display_regression_report(result):
    st.write("### 📊 Detailed Regression Report")
    
    # Get the regression metrics
    metrics = result["metrics"]
    
    # Display overall performance prominently
    st.write("#### 🎯 Overall Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    # R² Score
    r2 = metrics.get('r2', 0)
    r2_color = "green" if r2 >= 0.8 else "orange" if r2 >= 0.6 else "red"
    r2_status = "Excellent" if r2 >= 0.8 else "Good" if r2 >= 0.6 else "Needs Improvement"
    
    with col1:
        st.metric("R² Score", f"{r2:.4f}", help="Coefficient of determination (1.0 = perfect, 0.0 = no better than mean)")
        st.markdown(f"<span style='color: {r2_color}'>{r2_status}</span>", unsafe_allow_html=True)
    
    # RMSE (normalized by target range)
    rmse = metrics.get('rmse', 0)
    with col2:
        st.metric("RMSE", f"{rmse:.4f}", help="Root Mean Square Error (lower is better)")
        # Get target range for normalization context
        if hasattr(st.session_state.builder, 'y_test'):
            y_range = st.session_state.builder.y_test.max() - st.session_state.builder.y_test.min()
            rmse_pct = (rmse / y_range) * 100 if y_range > 0 else 0
            rmse_color = "green" if rmse_pct < 5 else "orange" if rmse_pct < 15 else "red"
            st.markdown(f"<span style='color: {rmse_color}'>{rmse_pct:.1f}% of range</span>", unsafe_allow_html=True)
    
    # MAE
    mae = metrics.get('mae', 0)
    with col3:
        st.metric("MAE", f"{mae:.4f}", help="Mean Absolute Error (lower is better)")
        if hasattr(st.session_state.builder, 'y_test'):
            y_range = st.session_state.builder.y_test.max() - st.session_state.builder.y_test.min()
            mae_pct = (mae / y_range) * 100 if y_range > 0 else 0
            mae_color = "green" if mae_pct < 3 else "orange" if mae_pct < 10 else "red"
            st.markdown(f"<span style='color: {mae_color}'>{mae_pct:.1f}% of range</span>", unsafe_allow_html=True)
    
    # Explained Variance
    with col4:
        explained_var = r2 * 100
        st.metric("Explained Variance", f"{explained_var:.1f}%", help="Percentage of target variance explained by the model")
        var_color = "green" if explained_var >= 80 else "orange" if explained_var >= 60 else "red"
        st.markdown(f"<span style='color: {var_color}'>{'Strong' if explained_var >= 80 else 'Moderate' if explained_var >= 60 else 'Weak'}</span>", unsafe_allow_html=True)
    
    # Add a secondary row showing Error % of Range metrics prominently
    if hasattr(st.session_state.builder, 'y_test'):
        st.write("#### 📏 Error as Percentage of Target Range")
        col1_err, col2_err, col3_err, col4_err = st.columns(4)
        
        y_range = st.session_state.builder.y_test.max() - st.session_state.builder.y_test.min()
        
        if y_range > 0:
            rmse_pct = (rmse / y_range) * 100
            mae_pct = (mae / y_range) * 100
            
            with col1_err:
                st.metric("RMSE % of Range", f"{rmse_pct:.1f}%", help="Root Mean Square Error as percentage of target variable range")
                rmse_status = "Low" if rmse_pct < 5 else "Moderate" if rmse_pct < 15 else "High"
                rmse_color = "green" if rmse_pct < 5 else "orange" if rmse_pct < 15 else "red"
                st.markdown(f"<span style='color: {rmse_color}'>{rmse_status}</span>", unsafe_allow_html=True)
            
            with col2_err:
                st.metric("MAE % of Range", f"{mae_pct:.1f}%", help="Mean Absolute Error as percentage of target variable range")
                mae_status = "Low" if mae_pct < 3 else "Moderate" if mae_pct < 10 else "High"
                mae_color = "green" if mae_pct < 3 else "orange" if mae_pct < 10 else "red"
                st.markdown(f"<span style='color: {mae_color}'>{mae_status}</span>", unsafe_allow_html=True)
            
            with col3_err:
                # Calculate target range info
                st.metric("Target Range", f"{y_range:.3f}", help="Difference between maximum and minimum target values")
                st.markdown("<span style='color: #666666'>Reference</span>", unsafe_allow_html=True)
            
            with col4_err:
                # Show average error percentage
                avg_error_pct = (rmse_pct + mae_pct) / 2
                st.metric("Average Error %", f"{avg_error_pct:.1f}%", help="Average of RMSE and MAE as percentage of range")
                avg_status = "Low" if avg_error_pct < 5 else "Moderate" if avg_error_pct < 12 else "High"
                avg_color = "green" if avg_error_pct < 5 else "orange" if avg_error_pct < 12 else "red"
                st.markdown(f"<span style='color: {avg_color}'>{avg_status}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Performance insights and error analysis
    st.write("#### 📈 Performance Analysis")
    
    
    st.write("**📊 Model Performance Insights:**")
    
    insights = []
    
    # R² interpretation
    if r2 >= 0.9:
        insights.append("🌟 **Exceptional fit** - Model explains >90% of variance")
    elif r2 >= 0.8:
        insights.append("✅ **Excellent fit** - Strong predictive performance")
    elif r2 >= 0.6:
        insights.append("👍 **Good fit** - Decent predictive performance with room for improvement")
    elif r2 >= 0.3:
        insights.append("⚠️ **Moderate fit** - Model captures some patterns but significant improvement needed")
    else:
        insights.append("❌ **Poor fit** - Model performs poorly, major improvements required")
    
    # RMSE vs MAE comparison
    if rmse > 0 and mae > 0:
        rmse_mae_ratio = rmse / mae
        if rmse_mae_ratio > 1.5:
            insights.append("📊 **Large errors present** - RMSE much higher than MAE indicates outliers")
        elif rmse_mae_ratio < 1.2:
            insights.append("📊 **Consistent errors** - RMSE close to MAE indicates uniform error distribution")
        else:
            insights.append("📊 **Normal error distribution** - Typical RMSE to MAE ratio")
    
    # Calculate additional insights if we have access to predictions
    if hasattr(st.session_state.builder, 'y_test'):
        y_test = st.session_state.builder.y_test
        model_instance = st.session_state.builder.model.get("active_model") or st.session_state.builder.model["model"]
        y_pred = model_instance.predict(st.session_state.builder.X_test)
        
        # Bias analysis
        mean_error = np.mean(y_test - y_pred)
        if abs(mean_error) < mae * 0.1:
            insights.append("🎯 **Well-calibrated** - No systematic bias in predictions")
        elif mean_error > mae * 0.1:
            insights.append("📈 **Slight underestimation** - Model tends to predict lower values")
        else:
            insights.append("📉 **Slight overestimation** - Model tends to predict higher values")
        
        # Prediction range analysis
        pred_range = y_pred.max() - y_pred.min()
        actual_range = y_test.max() - y_test.min()
        range_ratio = pred_range / actual_range if actual_range > 0 else 0
        
        if range_ratio < 0.8:
            insights.append("🔒 **Conservative predictions** - Model predictions have limited range")
        elif range_ratio > 1.2:
            insights.append("🎢 **Wide predictions** - Model predictions span broader range than training data")
        else:
            insights.append("📏 **Appropriate range** - Prediction range matches target distribution")
    
    for insight in insights:
        st.markdown(insight)
    
    
    st.markdown("---")
    
    # Detailed metrics breakdown
    st.write("#### 📋 Detailed Metrics")
    
    # Create a comprehensive metrics table
    metrics_data = []
    
    if hasattr(st.session_state.builder, 'y_test'):
        y_test = st.session_state.builder.y_test
        model_instance = st.session_state.builder.model.get("active_model") or st.session_state.builder.model["model"]
        y_pred = model_instance.predict(st.session_state.builder.X_test)
        
        # Calculate additional metrics
        residuals = y_test - y_pred
        
        # Calculate MAPE safely
        try:
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            mape_str = f"{mape:.2f}%"
        except:
            mape_str = "N/A"
        
        # Calculate Adjusted R²
        n = len(y_test)
        p = len(st.session_state.builder.X_test.columns)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        
        # Calculate Error % of Range metrics
        y_range = y_test.max() - y_test.min()
        rmse_pct = (rmse / y_range) * 100 if y_range > 0 else 0
        mae_pct = (mae / y_range) * 100 if y_range > 0 else 0
        
        metrics_data = [
            {
                'Metric': 'R² Score',
                'Value': f"{r2:.4f}",
                'Description': 'Proportion of variance explained',
                'Interpretation': 'Higher is better (max 1.0)'
            },
            {
                'Metric': 'Adjusted R²',
                'Value': f"{adj_r2:.4f}",
                'Description': 'R² adjusted for number of predictors',
                'Interpretation': 'Penalizes model complexity'
            },
            {
                'Metric': 'RMSE',
                'Value': f"{rmse:.4f}",
                'Description': 'Root Mean Square Error',
                'Interpretation': 'Penalizes large errors more'
            },
            {
                'Metric': 'MAE',
                'Value': f"{mae:.4f}",
                'Description': 'Mean Absolute Error',
                'Interpretation': 'Average absolute prediction error'
            },
            {
                'Metric': 'MAPE',
                'Value': mape_str,
                'Description': 'Mean Absolute Percentage Error',
                'Interpretation': 'Average percentage error'
            },
            {
                'Metric': 'RMSE % of Range',
                'Value': f"{rmse_pct:.2f}%",
                'Description': 'RMSE as percentage of target range',
                'Interpretation': 'Lower percentages indicate better precision'
            },
            {
                'Metric': 'MAE % of Range',
                'Value': f"{mae_pct:.2f}%",
                'Description': 'MAE as percentage of target range',
                'Interpretation': 'Lower percentages indicate better precision'
            },
            {
                'Metric': 'Mean Error',
                'Value': f"{np.mean(residuals):.4f}",
                'Description': 'Average prediction bias',
                'Interpretation': 'Should be close to zero'
            },
            {
                'Metric': 'Std of Residuals',
                'Value': f"{np.std(residuals):.4f}",
                'Description': 'Standard deviation of errors',
                'Interpretation': 'Lower indicates more consistent errors'
            }
        ]
    else:
        # Fallback to basic metrics if detailed data not available
        metrics_data = [
            {
                'Metric': 'R² Score',
                'Value': f"{r2:.4f}",
                'Description': 'Proportion of variance explained',
                'Interpretation': 'Higher is better (max 1.0)'
            },
            {
                'Metric': 'RMSE',
                'Value': f"{rmse:.4f}",
                'Description': 'Root Mean Square Error',
                'Interpretation': 'Penalizes large errors more'
            },
            {
                'Metric': 'MAE',
                'Value': f"{mae:.4f}",
                'Description': 'Mean Absolute Error',
                'Interpretation': 'Average absolute prediction error'
            }
        ]
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Style the metrics table with comprehensive color coding
    def style_regression_metrics(df):
        # Get target range for contextual thresholds
        target_range = None
        if hasattr(st.session_state.builder, 'y_test'):
            y_test = st.session_state.builder.y_test
            target_range = y_test.max() - y_test.min()
        
        def get_color_for_metric(metric_name, val_str):
            """Get color coding based on metric type and value."""
            try:
                # Handle percentage values
                if '%' in val_str:
                    val_float = float(val_str.replace('%', ''))
                else:
                    val_float = float(val_str)
                
                # R² Score and Adjusted R² (higher is better)
                if metric_name in ['R² Score', 'Adjusted R²']:
                    if val_float >= 0.8:
                        return 'background-color: #d4edda; color: #155724'  # Green - Excellent
                    elif val_float >= 0.6:
                        return 'background-color: #fff3cd; color: #856404'  # Yellow - Good
                    elif val_float >= 0.3:
                        return 'background-color: #ffeaa7; color: #6c757d'  # Light orange - Fair
                    else:
                        return 'background-color: #f8d7da; color: #721c24'  # Red - Poor
                
                # Error % of Range metrics (lower is better)
                elif 'Range' in metric_name:
                    if 'RMSE' in metric_name:
                        # RMSE % of Range thresholds
                        if val_float < 5:
                            return 'background-color: #d4edda; color: #155724'  # Green - Excellent
                        elif val_float < 15:
                            return 'background-color: #fff3cd; color: #856404'  # Yellow - Good
                        elif val_float < 25:
                            return 'background-color: #ffeaa7; color: #6c757d'  # Orange - Fair
                        else:
                            return 'background-color: #f8d7da; color: #721c24'  # Red - Poor
                    else:
                        # MAE % of Range thresholds
                        if val_float < 3:
                            return 'background-color: #d4edda; color: #155724'  # Green - Excellent
                        elif val_float < 10:
                            return 'background-color: #fff3cd; color: #856404'  # Yellow - Good
                        elif val_float < 20:
                            return 'background-color: #ffeaa7; color: #6c757d'  # Orange - Fair
                        else:
                            return 'background-color: #f8d7da; color: #721c24'  # Red - Poor
                
                # MAPE (Mean Absolute Percentage Error) - lower is better
                elif metric_name == 'MAPE':
                    if val_float < 10:
                        return 'background-color: #d4edda; color: #155724'  # Green - Excellent
                    elif val_float < 20:
                        return 'background-color: #fff3cd; color: #856404'  # Yellow - Good
                    elif val_float < 50:
                        return 'background-color: #ffeaa7; color: #6c757d'  # Orange - Fair
                    else:
                        return 'background-color: #f8d7da; color: #721c24'  # Red - Poor
                
                # Mean Error (Bias) - closer to zero is better
                elif metric_name == 'Mean Error':
                    abs_val = abs(val_float)
                    # Use target range for context if available
                    if target_range and target_range > 0:
                        bias_pct = (abs_val / target_range) * 100
                        if bias_pct < 1:
                            return 'background-color: #d4edda; color: #155724'  # Green - Minimal bias
                        elif bias_pct < 5:
                            return 'background-color: #fff3cd; color: #856404'  # Yellow - Slight bias
                        elif bias_pct < 10:
                            return 'background-color: #ffeaa7; color: #6c757d'  # Orange - Moderate bias
                        else:
                            return 'background-color: #f8d7da; color: #721c24'  # Red - Significant bias
                    else:
                        # Fallback thresholds without context
                        if abs_val < 0.1:
                            return 'background-color: #d4edda; color: #155724'  # Green
                        elif abs_val < 0.5:
                            return 'background-color: #fff3cd; color: #856404'  # Yellow
                        else:
                            return 'background-color: #f8d7da; color: #721c24'  # Red
                
                # Absolute error metrics (RMSE, MAE) - use target range for context
                elif metric_name in ['RMSE', 'MAE']:
                    if target_range and target_range > 0:
                        error_pct = (val_float / target_range) * 100
                        # Adjusted thresholds for RMSE vs MAE
                        if metric_name == 'RMSE':
                            if error_pct < 5:
                                return 'background-color: #d4edda; color: #155724'  # Green - Excellent
                            elif error_pct < 15:
                                return 'background-color: #fff3cd; color: #856404'  # Yellow - Good
                            elif error_pct < 25:
                                return 'background-color: #ffeaa7; color: #6c757d'  # Orange - Fair
                            else:
                                return 'background-color: #f8d7da; color: #721c24'  # Red - Poor
                        else:  # MAE
                            if error_pct < 3:
                                return 'background-color: #d4edda; color: #155724'  # Green - Excellent
                            elif error_pct < 10:
                                return 'background-color: #fff3cd; color: #856404'  # Yellow - Good
                            elif error_pct < 20:
                                return 'background-color: #ffeaa7; color: #6c757d'  # Orange - Fair
                            else:
                                return 'background-color: #f8d7da; color: #721c24'  # Red - Poor
                
                # Standard deviation of residuals
                elif metric_name == 'Std of Residuals':
                    if target_range and target_range > 0:
                        std_pct = (val_float / target_range) * 100
                        if std_pct < 5:
                            return 'background-color: #d4edda; color: #155724'  # Green - Low variance
                        elif std_pct < 15:
                            return 'background-color: #fff3cd; color: #856404'  # Yellow - Moderate variance
                        elif std_pct < 25:
                            return 'background-color: #ffeaa7; color: #6c757d'  # Orange - High variance
                        else:
                            return 'background-color: #f8d7da; color: #721c24'  # Red - Very high variance
                
                else:
                    return ''  # No coloring for unknown metrics
                    
            except (ValueError, TypeError):
                return ''  # No coloring if value can't be parsed
        
        def highlight_row(row):
            colors = [''] * len(row)
            # Apply coloring to the Value column (index 1)
            colors[1] = get_color_for_metric(row['Metric'], row['Value'])
            return colors
        
        return df.style.apply(highlight_row, axis=1)
    
    st.dataframe(style_regression_metrics(metrics_df), width='stretch', hide_index=True)
    
    # Add color coding legend
    st.markdown("""
    **📊 Color Coding Legend:**
    - 🟢 **Green**: Excellent performance (top tier)
    - 🟡 **Yellow**: Good performance (acceptable range)  
    - 🟠 **Orange**: Fair performance (needs attention)
    - 🔴 **Red**: Poor performance (requires improvement)
    
    *Note: Thresholds are automatically adjusted based on your data's scale for contextual accuracy.*
    """)
    
    st.markdown("---")
    
    # Enhanced educational content
    with st.expander("📚 Understanding Regression Metrics"):
        st.markdown("""
        ### 🎯 **Core Metrics Explained**
        
        *All metrics in the detailed table above are color-coded based on performance thresholds to help you quickly assess model quality.*
        
        **R² Score (Coefficient of Determination)**:
        - *"What percentage of the target variance does my model explain?"*
        - Range: -∞ to 1.0 (1.0 = perfect fit, 0 = no better than mean)
        - Most important single metric for regression
        - **Thresholds**: 🟢 ≥0.8 | 🟡 0.6-0.8 | 🟠 0.3-0.6 | 🔴 <0.3
        
        **Adjusted R²**:
        - *"R² adjusted for model complexity"*
        - Penalizes adding features that don't improve fit
        - Better for comparing models with different numbers of features
        - **Thresholds**: Same as R² Score (🟢 ≥0.8 | 🟡 0.6-0.8 | 🟠 0.3-0.6 | 🔴 <0.3)
        
        **RMSE (Root Mean Square Error)**:
        - *"Typical prediction error, emphasizing large mistakes"*
        - Same units as target variable
        - Penalizes large errors more than small ones
        - **Thresholds**: Based on % of target range (🟢 <5% | 🟡 5-15% | 🟠 15-25% | 🔴 >25%)
        
        **MAE (Mean Absolute Error)**:
        - *"Average absolute prediction error"*
        - Same units as target variable
        - Less sensitive to outliers than RMSE
        - **Thresholds**: Based on % of target range (🟢 <3% | 🟡 3-10% | 🟠 10-20% | 🔴 >20%)
        
        **MAPE (Mean Absolute Percentage Error)**:
        - *"Average percentage error"*
        - Scale-independent, good for comparing across datasets
        - Can be problematic when actual values are near zero
        - **Thresholds**: 🟢 <10% | 🟡 10-20% | 🟠 20-50% | 🔴 >50%
        
        **Mean Error (Bias)**:
        - *"Average signed prediction error (shows systematic bias)"*
        - Positive values = model tends to underestimate
        - Negative values = model tends to overestimate
        - **Thresholds**: Based on % of target range (🟢 <1% | 🟡 1-5% | 🟠 5-10% | 🔴 >10%)
        
        **Explained Variance**:
        - *"Percentage of target variance captured by the model"*
        - Simply R² score expressed as a percentage (R² × 100)
        - Easier to interpret: "My model explains 75% of the variation in the data"
        - Range: 0% to 100% (higher is better)
        
        **Error % of Range**:
        - *"How large are prediction errors relative to the target variable's range?"*
        - Calculated as: (Error / Target Range) × 100
        - Makes errors interpretable across different scales and domains
        - Example: 5% means errors are typically 5% of the full data range
        - Lower percentages indicate better precision relative to data scale
        - **RMSE Thresholds**: 🟢 <5% | 🟡 5-15% | 🟠 15-25% | 🔴 >25%
        - **MAE Thresholds**: 🟢 <3% | 🟡 3-10% | 🟠 10-20% | 🔴 >20%
        
        ### 📊 **Interpreting Error Relationships**
        
        **RMSE vs MAE Ratio**:
        - **Close to 1.0**: Errors are consistent in size
        - **Much higher**: Large outlier errors present
        - **Typical range**: 1.1 to 1.4 for most datasets
        
        **Mean Error (Bias)**:
        - **Near zero**: Well-calibrated model
        - **Positive**: Model tends to underestimate
        - **Negative**: Model tends to overestimate
        
        ### 🎯 **Performance Benchmarks**
        
        **R² Score Guidelines**:
        - **0.9+**: Exceptional (rare in real-world data)
        - **0.8-0.9**: Excellent
        - **0.6-0.8**: Good
        - **0.3-0.6**: Moderate (may be acceptable depending on domain)
        - **Below 0.3**: Poor (needs significant improvement)
        
        **Error as % of Target Range**:
        - **< 5%**: Excellent precision
        - **5-15%**: Good precision
        - **15-25%**: Moderate precision
        - **> 25%**: Poor precision
        """)
    
    with st.expander("🔧 Improvement Recommendations"):
        recommendations = []
        
        # R² based recommendations
        if r2 < 0.3:
            recommendations.extend([
                "🔄 **Try different algorithms**",
                "🔍 **Feature selection**: Remove irrelevant or noisy features"
            ])
        elif r2 < 0.6:
            recommendations.extend([
                "🎛️ **Hyperparameter tuning**: Optimize model parameters",
                "🔄 **Try different algorithms**",
                "🔍 **Feature selection**: Remove irrelevant or noisy features"
            ])
        
        # Error-based recommendations
        if hasattr(st.session_state.builder, 'y_test'):
            y_test = st.session_state.builder.y_test
            model_instance = st.session_state.builder.model.get("active_model") or st.session_state.builder.model["model"]
            y_pred = model_instance.predict(st.session_state.builder.X_test)
            
            # Check for bias
            mean_error = np.mean(y_test - y_pred)
            if abs(mean_error) > mae * 0.2:
                recommendations.append("⚖️ **Address systematic bias**: Model consistently over/under-predicts")
            
            # Check RMSE vs MAE ratio
            if rmse > 0 and mae > 0:
                rmse_mae_ratio = rmse / mae
                if rmse_mae_ratio > 1.5:
                    recommendations.extend([
                        "🎯 **Handle outliers**: Remove or handle extreme values",
                        "📊 **Outlier analysis**: Investigate high-error predictions"
                    ])
            
            # Check error percentage
            y_range = y_test.max() - y_test.min()
            if y_range > 0:
                error_pct = (rmse / y_range) * 100
                if error_pct > 20:
                    recommendations.extend([
                        "📈 **More training data**: Collect additional samples",
                        "🔍 **Feature importance**: Focus on most predictive features",
                        "🎛️ **Model complexity**: Try more sophisticated algorithms"
                    ])
        
        # General recommendations
        if len(recommendations) == 0:
            if r2 >= 0.8:
                recommendations = [
                    "🎉 **Excellent performance!** Consider:",
                    "✅ **Model validation**: Test on new data to ensure robustness",
                    "📊 **Residual analysis**: Check for any remaining patterns",
                    "🚀 **Production readiness**: Model appears ready for deployment"
                ]
            else:
                recommendations = [
                    "👍 **Good performance!** Consider:",
                    "🔍 **Feature engineering**: Add domain-specific features",
                    "⚖️ **Cross-validation**: Ensure robust performance",
                    "📈 **Incremental improvements**: Fine-tune hyperparameters"
                ]
        
        for rec in recommendations:
            st.markdown(rec)

    st.session_state.logger.log_journey_point(
        stage="MODEL_EVALUATION",
        decision_type="MODEL_EVALUATION",
        description="Regression Model Performance Metrics",
        details={"Model Type": st.session_state.builder.model['type'],
                "R² Score": f"{r2:.4f}",
                "RMSE": f"{rmse:.4f}",
                "MAE": f"{mae:.4f}",
                "Average Error %": f"{avg_error_pct:.1f}%"
                },
        parent_id=None
    )