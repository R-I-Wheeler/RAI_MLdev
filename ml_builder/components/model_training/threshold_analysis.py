import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, precision_score,
    recall_score, accuracy_score
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

@st.cache_data(ttl=300, show_spinner=False)
def _cached_threshold_analysis(model_params_hash, X_test_shape, y_test_hash, problem_type):
    """Cache threshold analysis to avoid recalculation on every page load"""
    # Placeholder for cached analysis - will be implemented in main function
    return None

@st.cache_data(ttl=300, show_spinner=False)
def _cached_roc_pr_curves(y_test_hash, y_prob_hash):
    """Cache ROC and PR curve calculations"""
    # Placeholder for cached curve calculations
    return None

@st.cache_data(ttl=300, show_spinner=False)
def _cached_threshold_metrics(y_test_hash, y_prob_hash, thresholds):
    """Cache threshold metrics calculation"""
    # Placeholder for cached metrics calculation
    return None

def display_threshold_analysis_section():
    """Display the probability threshold optimization section in the training page."""

    # Import state manager for better component communication
    from components.model_training.utils.training_state_manager import TrainingStateManager

    # Validate training state and check for classification model
    validation = TrainingStateManager.validate_training_state()
    if not all(validation.values()):
        st.warning("Please complete model training first to access threshold analysis.")
        return

    if not TrainingStateManager.is_classification_model():
        return
    
    # Check if model supports probability prediction
    model = st.session_state.builder.model.get("active_model") or st.session_state.builder.model["model"]
    if not hasattr(model, 'predict_proba'):
        st.warning("⚠️ Current model doesn't support probability predictions - threshold analysis not available")
        return
    
    # Check if this is multiclass and warn about limitations
    problem_type = st.session_state.builder.model["problem_type"]
    is_binary = problem_type in ["classification", "binary_classification"]
    
    if not is_binary:
        st.warning("""
        ⚠️ **Multiclass Classification Detected**: 
        
        Threshold optimization is **not recommended** for multiclass problems because:
        - It can severely impact model performance (accuracy drops from 90% to <1%)
        - Unlike binary classification, there's no single meaningful threshold
        - The concept of "confidence threshold" doesn't translate well to final predictions
        
        **Recommendation**: Use the default model predictions without threshold optimization.
        """)
        return
    
    st.header("🎯 Probability Threshold Optimization (Optional)")
    
    # Check if threshold optimization has already been applied
    if (hasattr(st.session_state, 'builder') and 
        hasattr(st.session_state.builder, 'model') and 
        st.session_state.builder.model.get("threshold_optimized", False)):
        
        optimal_threshold = st.session_state.builder.model.get("optimal_threshold", 0.5)
        is_binary = st.session_state.builder.model.get("threshold_is_binary", True)
        criterion = st.session_state.builder.model.get("threshold_criterion", "Unknown")
        
        st.success(f"""
        ✅ **Threshold Optimization Active**
        
        - **Current Threshold**: {optimal_threshold:.3f} (instead of default 0.5)
        - **Optimization Criterion**: {criterion}
        - **Type**: {'Binary Classification' if is_binary else 'Multiclass Classification'}
        """)
        
        if st.button("🔄 Revert to Default Threshold (0.5)", type="secondary",key="revert_threshold_button"):
            st.session_state.builder.model["threshold_optimized"] = False
            if "optimal_threshold" in st.session_state.builder.model:
                del st.session_state.builder.model["optimal_threshold"]
            if "threshold_is_binary" in st.session_state.builder.model:
                del st.session_state.builder.model["threshold_is_binary"]
            if "threshold_criterion" in st.session_state.builder.model:
                del st.session_state.builder.model["threshold_criterion"]
            
            st.session_state.logger.log_user_action(
                "Threshold Reverted",
                {
                    "reverted_from": optimal_threshold,
                    "reverted_to": 0.5,
                    "is_binary": is_binary
                }
            )
            
            st.success("✅ Reverted to default threshold (0.5)")
            st.rerun()
        
        st.markdown("---")
    
    # Add threshold analysis explanation
    with st.expander("🔍 What is Threshold Optimization? (Simple Explanation)", expanded=False):
        st.markdown("""
        ### 🎯 Think of Your Model Like a Security Guard
        
        Imagine your ML model is a security guard deciding who gets into a VIP event. The guard has to make a **yes/no decision** for each person.
        
        #### 🤔 How Does the Guard Decide?
        Your model doesn't just say "yes" or "no" - it gives a **confidence score** from 0% to 100%:
        - 20% confident = "Probably shouldn't let them in"
        - 80% confident = "Pretty sure they should get in"
        - 95% confident = "Definitely let them in!"
        
        #### 🚪 The "Threshold" is the Minimum Confidence Level
        - **Threshold = 50%**: Let in anyone the model is 50%+ confident about
        - **Threshold = 80%**: Only let in people the model is 80%+ confident about
        - **Threshold = 20%**: Let in people the model is only 20%+ confident about
        
        ---
        
        ### 🔧 Why Change the Threshold?
        
        **The default threshold is 50%**, but this might not be best for your situation:
        
        #### 📧 Example: Email Spam Detection
        - **High threshold (80%)**: Only block emails you're very sure are spam
          - ✅ Good: Fewer important emails accidentally blocked
          - ❌ Bad: Some spam might slip through
        
        - **Low threshold (20%)**: Block emails even if only slightly suspicious
          - ✅ Good: Catches almost all spam
          - ❌ Bad: Might block some important emails by mistake
        
        #### 🏥 Example: Medical Diagnosis
        - **Low threshold (30%)**: Flag patients even with mild symptoms
          - ✅ Good: Don't miss anyone who might be sick
          - ❌ Bad: Many false alarms, unnecessary tests
        
        - **High threshold (70%)**: Only flag patients with strong symptoms
          - ✅ Good: Fewer false alarms
          - ❌ Bad: Might miss some sick patients
        
        ---
        
        ### 🎯 When Should You Optimize Your Threshold?
        
        #### ✅ **Definitely optimize when:**
        - You have **unbalanced data** (e.g., 90% normal emails, 10% spam)
        - **Mistakes have different costs** (missing cancer vs false alarm)
        - **Your model seems "too cautious" or "too aggressive"**
        - **Default 50% doesn't match your business needs**
        
        #### 🤷 **Maybe not needed when:**
        - Your data is perfectly balanced (50/50 split)
        - You only care about ranking things, not making decisions
        - The default 50% is already working great
        
        ---
        
        ### 📊 What Do the Different Goals Mean?
        
        **Choose based on what matters most to you:**
        
        #### 🎯 **F1 Score** (Most Popular Choice)
        - **What it does**: Balances catching positives vs avoiding false alarms
        - **When to use**: When both types of mistakes matter equally
        - **Example**: General email classification, customer segmentation
        
        #### 🎪 **Precision** (Avoid False Alarms)
        - **What it does**: Minimizes wrongly flagging things as positive
        - **When to use**: When false alarms are expensive/annoying
        - **Example**: Fraud detection (don't want to block real purchases)
        
        #### 🕸️ **Recall** (Catch Everything Important)
        - **What it does**: Minimizes missing important cases
        - **When to use**: When missing something is dangerous/costly
        - **Example**: Disease screening (don't miss any sick patients)
        
        #### 📏 **Accuracy** (Overall Correctness)
        - **What it does**: Maximizes the total number of correct predictions
        - **When to use**: When all mistakes are equally bad and data is balanced
        - **Example**: General classification where false positives = false negatives
        
        #### ⚖️ **Youden's J** (Medical/Scientific)
        - **What it does**: Balances true positives and true negatives equally
        - **When to use**: Medical research, scientific studies
        - **Example**: Diagnostic tests, clinical trials
        
        ---
        
        ### 🚀 Quick Start Guide
        
        1. **🔍 Start here**: Look at your current results below
        2. **🎯 Pick a goal**: Choose what matters most (F1 Score is usually good)
        3. **▶️ Click optimize**: Let the system find the best threshold
        4. **📈 Check improvement**: See if it made things better
        5. **✅ Done**: The new threshold is automatically applied!
        
        **💡 Pro tip**: You can always revert back to 50% if the optimization doesn't help!
        """)
    
    # Current threshold analysis (default 0.5)
    st.subheader("📊 Current Performance Analysis (Threshold = 0.5)")
    
    with st.spinner("Analyzing current model performance..."):
        current_analysis = analyze_current_performance()
        
        if current_analysis["success"]:
            display_current_performance(current_analysis)
            
            # Threshold optimization section
            st.subheader("🔧 Find Optimal Threshold")
            
            # Get problem type for appropriate optimization options
            problem_type = st.session_state.builder.model["problem_type"]
            is_binary = problem_type in ["classification", "binary_classification"]
            
            # Since we now only allow binary classification, simplify the UI
            col1, col2 = st.columns(2)
            
            with col1:
                # Optimization criterion selection for binary classification
                criterion_options = [
                    "F1 Score", 
                    "Youden's J Statistic", 
                    "Precision", 
                    "Recall",
                    "Accuracy"
                ]
                
                # Get recommendation and set as default
                recommended_criterion = recommend_optimization_criterion(current_analysis, is_binary)
                
                # Find the index of the recommended criterion
                try:
                    recommended_index = criterion_options.index(recommended_criterion)
                except ValueError:
                    recommended_index = 0  # Default to first option if not found
                
                optimization_criterion = st.selectbox(
                    "Optimization Goal",
                    criterion_options,
                    index=recommended_index,  # Set recommended as default
                    help="Choose the metric you want to optimize"
                )
                
                # Show recommendation if it's different from selection
                if recommended_criterion != optimization_criterion:
                    st.info(f"💡 **Recommended**: {recommended_criterion} based on your data characteristics")
            
            with col2:
                # Additional options
                show_curves = st.checkbox(
                    "Show ROC and Precision-Recall Curves", 
                    value=True,
                    help="Display detailed performance curves"
                )
            
            # Analysis button
            if st.button("🎯 Analyze & Optimize Threshold", type="primary"):
                optimize_threshold(
                    optimization_criterion, 
                    show_curves, 
                    is_binary
                )

        else:
            st.error(f"Could not analyze current performance: {current_analysis['message']}")

def analyze_current_performance() -> Dict[str, Any]:
    """Analyze the current model performance with default threshold."""
    try:
        # Get model and data
        model = st.session_state.builder.model.get("active_model") or st.session_state.builder.model["model"]
        X_test = st.session_state.builder.X_test
        y_test = st.session_state.builder.y_test
        problem_type = st.session_state.builder.model["problem_type"]
        
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Handle binary vs multiclass
        is_binary = problem_type in ["classification", "binary_classification"]
        
        if is_binary:
            # For binary classification
            if len(y_prob.shape) > 1:
                y_prob_positive = y_prob[:, 1]
            else:
                y_prob_positive = y_prob
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Get confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            return {
                "success": True,
                "is_binary": True,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm,
                "y_true": y_test,
                "y_prob": y_prob_positive,
                "y_pred": y_pred,
                "class_balance": np.bincount(y_test) / len(y_test)
            }
        else:
            # For multiclass classification
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Get confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Get max probabilities for confidence analysis
            max_probs = np.max(y_prob, axis=1)
            predicted_classes = np.argmax(y_prob, axis=1)
            
            return {
                "success": True,
                "is_binary": False,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm,
                "y_true": y_test,
                "y_prob": y_prob,
                "max_probs": max_probs,
                "predicted_classes": predicted_classes,
                "y_pred": y_pred,
                "n_classes": len(np.unique(y_test))
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }

def display_current_performance(analysis: Dict[str, Any]):
    """Display current model performance metrics."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{analysis['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{analysis['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{analysis['recall']:.3f}")
    with col4:
        st.metric("F1 Score", f"{analysis['f1']:.3f}")
    
    # Show confusion matrix
    st.write("**Confusion Matrix (Current Threshold):**")
    cm_df = pd.DataFrame(
        analysis['confusion_matrix'],
        index=[f"Actual {i}" for i in range(len(analysis['confusion_matrix']))],
        columns=[f"Predicted {i}" for i in range(len(analysis['confusion_matrix'][0]))]
    )
    st.dataframe(cm_df, width='stretch')
    
    # Class balance information
    if analysis['is_binary']:
        balance = analysis['class_balance']
        st.write(f"**Class Balance**: {balance[0]:.1%} (Class 0) vs {balance[1]:.1%} (Class 1)")
        
        if abs(balance[0] - balance[1]) > 0.2:  # More than 20% difference
            st.warning("⚠️ **Imbalanced Dataset Detected**: Consider threshold optimization to improve performance on minority class")

def recommend_optimization_criterion(analysis: Dict[str, Any], is_binary: bool) -> str:
    """Recommend the best optimization criterion based on data characteristics."""
    
    if is_binary:
        balance = analysis['class_balance']
        precision = analysis['precision']
        recall = analysis['recall']
        
        # Check for class imbalance
        imbalance_ratio = min(balance) / max(balance)
        
        if imbalance_ratio < 0.3:  # Severely imbalanced
            if precision < 0.6:
                return "Precision"  # Focus on reducing false positives
            elif recall < 0.6:
                return "Recall"  # Focus on catching minority class
            else:
                return "F1 Score"  # Balance both
        elif imbalance_ratio < 0.7:  # Moderately imbalanced
            return "F1 Score"  # Good balance for moderate imbalance
        else:
            # Well balanced - consider current performance
            if abs(precision - recall) > 0.2:
                return "F1 Score"  # Balance precision and recall
            else:
                return "Youden's J Statistic"  # Optimize sensitivity and specificity
    else:
        # For multiclass, F1 weighted is usually best
        return "F1 Score (Weighted)"

def optimize_threshold(criterion: str, show_curves: bool, is_binary: bool):
    """Perform comprehensive threshold optimization analysis."""
    
    try:
        with st.spinner("Optimizing threshold..."):
            # Get model and data
            model = st.session_state.builder.model.get("active_model") or st.session_state.builder.model["model"]
            X_test = st.session_state.builder.X_test
            y_test = st.session_state.builder.y_test
            
            # Get probabilities
            y_prob = model.predict_proba(X_test)
            
            # Binary classification threshold optimization only
            if len(y_prob.shape) > 1:
                y_prob_positive = y_prob[:, 1]
            else:
                y_prob_positive = y_prob
            
            # Find optimal threshold
            optimal_threshold, threshold_results = find_optimal_threshold_binary(
                y_test, y_prob_positive, criterion
            )
            
            # Automatically apply the optimal threshold
            apply_optimal_threshold(optimal_threshold, True, criterion)
            
            # Display results
            display_binary_threshold_results(
                optimal_threshold, threshold_results, y_test, y_prob_positive, show_curves
            )
            
            # Log the threshold optimization
            st.session_state.logger.log_user_action(
                "Threshold Optimization",
                {
                    "criterion": criterion,
                    "optimal_threshold": optimal_threshold,
                    "is_binary": is_binary,
                    "show_curves": show_curves,
                    "automatically_applied": True
                }
            )
            
    except Exception as e:
        st.error(f"Error during threshold optimization: {str(e)}")
        st.session_state.logger.log_error("Threshold Optimization Failed", {"error": str(e)})

def find_optimal_threshold_binary(y_true, y_prob, criterion: str) -> Tuple[float, Dict]:
    """Find optimal threshold for binary classification."""
    
    # Generate threshold range
    thresholds = np.linspace(0.01, 0.99, 99)
    
    results = {
        'thresholds': thresholds,
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'specificity': [],
        'youden_j': [],
        'custom_cost': []
    }
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Youden's J statistic
        youden_j = recall + specificity - 1
        
        # Custom cost (if specified)
        custom_cost = 0
        
        # Store results
        results['accuracy'].append(accuracy)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)
        results['specificity'].append(specificity)
        results['youden_j'].append(youden_j)
        results['custom_cost'].append(custom_cost)
    
    # Find optimal threshold based on criterion
    if criterion == "F1 Score":
        optimal_idx = np.argmax(results['f1'])
    elif criterion == "Youden's J Statistic":
        optimal_idx = np.argmax(results['youden_j'])
    elif criterion == "Precision":
        optimal_idx = np.argmax(results['precision'])
    elif criterion == "Recall":
        optimal_idx = np.argmax(results['recall'])
    elif criterion == "Accuracy":
        optimal_idx = np.argmax(results['accuracy'])
    else:
        optimal_idx = np.argmax(results['f1'])  # Default to F1
    
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, results

def display_binary_threshold_results(optimal_threshold: float, results: Dict, y_true, y_prob, show_curves: bool):
    """Display results for binary classification threshold optimization."""
    
    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    y_pred_default = (y_prob >= 0.5).astype(int)
    
    # Metrics comparison
    st.subheader("🎯 Optimization Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Optimal Threshold:**")
        st.metric("Threshold", f"{optimal_threshold:.3f}")
        st.metric("Accuracy", f"{accuracy_score(y_true, y_pred_optimal):.3f}")
        st.metric("Precision", f"{precision_score(y_true, y_pred_optimal, zero_division=0):.3f}")
        st.metric("Recall", f"{recall_score(y_true, y_pred_optimal, zero_division=0):.3f}")
        st.metric("F1 Score", f"{f1_score(y_true, y_pred_optimal, zero_division=0):.3f}")
    
    with col2:
        st.write("**Default Threshold (0.5):**")
        st.metric("Threshold", "0.500")
        st.metric("Accuracy", f"{accuracy_score(y_true, y_pred_default):.3f}")
        st.metric("Precision", f"{precision_score(y_true, y_pred_default, zero_division=0):.3f}")
        st.metric("Recall", f"{recall_score(y_true, y_pred_default, zero_division=0):.3f}")
        st.metric("F1 Score", f"{f1_score(y_true, y_pred_default, zero_division=0):.3f}")
    
    # Improvement summary
    f1_improvement = f1_score(y_true, y_pred_optimal, zero_division=0) - f1_score(y_true, y_pred_default, zero_division=0)
    
    if f1_improvement > 0.01:
        st.success(f"✅ **Improvement Found**: F1 Score improved by {f1_improvement:.3f} with optimized threshold!")
    elif f1_improvement > 0:
        st.info(f"📈 **Marginal Improvement**: F1 Score improved by {f1_improvement:.3f}")
    else:
        st.warning("⚠️ **No Significant Improvement**: Default threshold (0.5) appears to be already optimal")
    
    # Threshold curve plot
    st.subheader("📈 Threshold Analysis Curves")
    
    fig = create_threshold_curves_binary(results, optimal_threshold)
    st.plotly_chart(fig, config={'responsive': True}, key="binary_threshold_curves")
    
    # ROC and PR curves if requested
    if show_curves:
        st.subheader("📊 ROC and Precision-Recall Curves")
        
        roc_pr_fig = create_roc_pr_curves_binary(y_true, y_prob, optimal_threshold)
        st.plotly_chart(roc_pr_fig, config={'responsive': True}, key="binary_roc_pr_curves")

def create_threshold_curves_binary(results: Dict, optimal_threshold: float):
    """Create threshold analysis curves for binary classification."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Precision vs Recall", "F1 Score", "Accuracy", "ROC Metrics"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    thresholds = results['thresholds']
    
    # Precision vs Recall
    fig.add_trace(
        go.Scatter(x=thresholds, y=results['precision'], name="Precision", line=dict(color="blue")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=thresholds, y=results['recall'], name="Recall", line=dict(color="red")),
        row=1, col=1
    )
    
    # F1 Score
    fig.add_trace(
        go.Scatter(x=thresholds, y=results['f1'], name="F1 Score", line=dict(color="green")),
        row=1, col=2
    )
    
    # Accuracy
    fig.add_trace(
        go.Scatter(x=thresholds, y=results['accuracy'], name="Accuracy", line=dict(color="purple")),
        row=2, col=1
    )
    
    # ROC Metrics
    fig.add_trace(
        go.Scatter(x=thresholds, y=results['recall'], name="Sensitivity (Recall)", line=dict(color="orange")),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=thresholds, y=results['specificity'], name="Specificity", line=dict(color="brown")),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=thresholds, y=results['youden_j'], name="Youden's J", line=dict(color="pink")),
        row=2, col=2
    )
    
    # Add optimal threshold line
    for row in range(1, 3):
        for col in range(1, 3):
            fig.add_vline(x=optimal_threshold, line_dash="dash", line_color="red", 
                         annotation_text=f"Optimal: {optimal_threshold:.3f}", row=row, col=col)
    
    fig.update_layout(
        height=600,
        title_text="Threshold Analysis for Binary Classification",
        showlegend=True
    )
    
    return fig

def create_roc_pr_curves_binary(y_true, y_prob, optimal_threshold):
    """Create ROC and Precision-Recall curves for binary classification."""
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Calculate PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"ROC Curve (AUC = {roc_auc:.3f})", f"Precision-Recall Curve (AP = {avg_precision:.3f})")
    )
    
    # ROC Curve
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, name=f"ROC Curve (AUC = {roc_auc:.3f})", line=dict(color="blue")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="gray")),
        row=1, col=1
    )
    
    # PR Curve
    fig.add_trace(
        go.Scatter(x=recall, y=precision, name=f"PR Curve (AP = {avg_precision:.3f})", line=dict(color="red")),
        row=1, col=2
    )
    
    # Add baseline for PR curve
    baseline = np.sum(y_true) / len(y_true)
    fig.add_hline(y=baseline, line_dash="dash", line_color="gray", 
                 annotation_text=f"Baseline = {baseline:.3f}", row=1, col=2)
    
    fig.update_layout(
        height=400,
        title_text="ROC and Precision-Recall Curves",
        showlegend=True
    )
    
    return fig

def apply_optimal_threshold(optimal_threshold: float, is_binary: bool, criterion: str = None):
    """Apply the optimal threshold to the model for future predictions."""
    
    try:
        # Verify session state and model exist
        if not hasattr(st.session_state, 'builder'):
            st.error("❌ Session state builder not found")
            return
            
        if not hasattr(st.session_state.builder, 'model') or st.session_state.builder.model is None:
            st.error("❌ Model not found in session state")
            return
        
        # Store the optimal threshold in the model configuration
        st.session_state.builder.model["optimal_threshold"] = optimal_threshold
        st.session_state.builder.model["threshold_optimized"] = True
        st.session_state.builder.model["threshold_is_binary"] = is_binary
        st.session_state.builder.model["threshold_criterion"] = criterion
        
        # Log the threshold application
        st.session_state.logger.log_user_action(
            "Optimal Threshold Applied",
            {
                "threshold": optimal_threshold,
                "is_binary": is_binary,
                "criterion": criterion
            }
        )
        
        st.session_state.logger.log_journey_point(
            stage="MODEL_TRAINING",
            decision_type="MODEL_TRAINING", 
            description="Optimal threshold applied",
            details={
                "Threshold": optimal_threshold,
                "Binary Classification": is_binary,
                "Optimization Criterion": criterion
            },
            parent_id=None
        )
        
        if is_binary:
            st.success(f"✅ Optimal threshold ({optimal_threshold:.3f}) applied automatically using {criterion}!")
        else:
            st.success(f"✅ Optimal confidence threshold ({optimal_threshold:.3f}) applied automatically using {criterion}!")
        
        # Add revert option
        if st.button("🔄 Revert to Default Threshold (0.5)"):
            st.session_state.builder.model["threshold_optimized"] = False
            if "optimal_threshold" in st.session_state.builder.model:
                del st.session_state.builder.model["optimal_threshold"]
            if "threshold_is_binary" in st.session_state.builder.model:
                del st.session_state.builder.model["threshold_is_binary"]
            if "threshold_criterion" in st.session_state.builder.model:
                del st.session_state.builder.model["threshold_criterion"]
            
            st.session_state.logger.log_user_action(
                "Threshold Reverted",
                {
                    "reverted_from": optimal_threshold,
                    "reverted_to": 0.5,
                    "is_binary": is_binary
                }
            )
            
            st.success("✅ Reverted to default threshold (0.5)")
            st.rerun()
        
    except Exception as e:
        st.error(f"Error applying optimal threshold: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        if hasattr(st.session_state, 'logger'):
            st.session_state.logger.log_error("Apply Optimal Threshold Failed", {"error": str(e), "traceback": traceback.format_exc()}) 