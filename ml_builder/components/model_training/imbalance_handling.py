"""Comprehensive class imbalance analysis and handling with preview functionality.

Note: This component currently uses imbalance handling methods through the Builder class,
which now delegate to utilities in components.model_training.utils.imbalance_utils.
For direct access to the utility functions, you can import from:
- components.model_training.utils.imbalance_utils
- components.model_training.utils.model_selection_utils
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


@st.cache_data(ttl=300, show_spinner=False)
def _cached_class_imbalance_analysis(X_train_hash, y_train_hash, problem_type):
    """Cached class imbalance analysis to avoid recalculation on every page load"""
    # This will be called from the main function with hashed data
    return st.session_state.builder.analyse_class_imbalance()


@st.cache_data(ttl=300, show_spinner=False)
def _cached_imbalance_recommendation(X_train_hash, y_train_hash, imbalance_ratio):
    """Cached imbalance recommendation to avoid recalculation"""
    return st.session_state.builder.get_imbalance_recommendation()


@st.cache_data(ttl=300, show_spinner=False)
def _create_class_distribution_chart(class_distribution):
    """Cache the class distribution chart creation"""
    fig = px.bar(
        x=list(class_distribution.keys()),
        y=list(class_distribution.values()),
        title="Class Distribution",
        labels={"x": "Class", "y": "Count"}
    )
    return fig


@st.cache_data(ttl=300, show_spinner=False)
def _create_class_distribution_preview_chart(class_distribution):
    """Cache the class distribution chart creation for preview"""
    fig = px.bar(
        x=list(class_distribution.keys()),
        y=list(class_distribution.values()),
        title="Class Distribution",
        labels={"x": "Class", "y": "Count"}
    )
    fig.update_layout(height=300)  # Smaller height for preview
    return fig


@st.cache_data(ttl=300, show_spinner=False)
def _create_feature_composition_chart(numerical_features, categorical_features):
    """Cache the feature composition chart creation"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Numerical', 'Categorical'],
            y=[numerical_features, categorical_features],
            text=[numerical_features, categorical_features],
            textposition='auto',
        )
    ])
    fig.update_layout(title='Feature Types Distribution')
    return fig


def detect_classification_imbalance():
    """
    Detect if there's significant class imbalance that should be addressed before training.
    Returns tuple: (has_imbalance, analysis_results)
    """
    try:
        # Check if we have the necessary data
        if (not hasattr(st.session_state, 'builder') or
            st.session_state.builder.X_train is None or
            st.session_state.builder.y_train is None):
            return False, None

        # Check if this is a classification problem
        problem_type = st.session_state.builder.model.get("problem_type", "unknown")
        if problem_type not in ["classification", "binary_classification", "multiclass_classification"]:
            return False, None

        # Perform imbalance analysis
        imbalance_analysis = st.session_state.builder.analyse_class_imbalance()

        if imbalance_analysis["success"]:
            metrics = imbalance_analysis["metrics"]
            # Show imbalance handling if ratio > 3 (moderate to severe imbalance)
            has_significant_imbalance = metrics["imbalance_ratio"] > 3
            return has_significant_imbalance, imbalance_analysis

        return False, None

    except Exception as e:
        st.error(f"Error detecting class imbalance: {str(e)}")
        return False, None


def display_imbalance_analysis_preview(imbalance_analysis):
    """Display a preview of class imbalance analysis before training."""

    if not imbalance_analysis or not imbalance_analysis.get("success"):
        return

    metrics = imbalance_analysis["metrics"]

    with st.expander("📚 Learn About Class Imbalance", expanded=False):
        st.markdown("""
        ### What is Class Imbalance?

        Class imbalance occurs when some classes in your dataset have significantly more examples than others.

        ### Why is it a Problem?

        **Example: Email Spam Detection**
        - 95% Normal emails, 5% Spam emails
        - A model that always predicts "Normal" gets 95% accuracy!
        - But it never catches any spam (0% recall for spam)

        ### Common Scenarios with Class Imbalance:
        - 🏥 **Medical diagnosis**: 99% healthy, 1% disease cases
        - 🔒 **Fraud detection**: 99.9% legitimate, 0.1% fraudulent transactions
        - 📧 **Spam detection**: 90% normal, 10% spam emails
        - 🏭 **Quality control**: 95% good products, 5% defective

        ### Resampling Techniques:

        **Oversampling (Add minority samples):**
        - **Random Oversampling**: Duplicate existing minority samples
        - **SMOTE**: Create synthetic samples using nearest neighbors
        - **ADASYN**: Focus on hard-to-learn minority samples

        **Undersampling (Remove majority samples):**
        - **Random Undersampling**: Randomly remove majority samples
        - Less common due to information loss

        ### When to Use Each Method:
        - **Small datasets** (< 1,000 samples): Random Oversampling or SMOTE
        - **Large datasets** (> 10,000 samples): Any method works well
        - **Complex patterns**: ADASYN or SMOTE
        - **Simple patterns**: Random methods often sufficient
        """)

    # Display warning based on severity
    if metrics["severity"] == "Severe":
        st.error(f"""
        🚨 **Severe Class Imbalance Detected** (Ratio: {metrics['imbalance_ratio']:.1f}:1)

        Your dataset has significant class imbalance that will likely lead to biased model predictions.
        **Strongly recommended** to address this before training.
        """)
    elif metrics["severity"] == "Moderate":
        st.warning(f"""
        ⚠️ **Moderate Class Imbalance Detected** (Ratio: {metrics['imbalance_ratio']:.1f}:1)

        Your dataset has noticeable class imbalance that may affect model performance.
        **Recommended** to consider addressing this before training.
        """)

    # Show quick metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Imbalance Ratio",
            f"{metrics['imbalance_ratio']:.1f}:1",
            help="Ratio between majority and minority classes"
        )

    with col2:
        st.metric(
            "Majority Class",
            f"{metrics['majority_class']['count']:,}",
            f"Class: {metrics['majority_class']['label']}"
        )

    with col3:
        st.metric(
            "Minority Class",
            f"{metrics['minority_class']['count']:,}",
            f"Class: {metrics['minority_class']['label']}"
        )

    # Show small distribution chart
    fig = _create_class_distribution_preview_chart(metrics["class_distribution"])
    st.plotly_chart(fig, config={'responsive': True})

    # Explanation of impact
    st.info("""
    **Why does this matter?**
    - Models trained on imbalanced data often become biased toward the majority class
    - Poor performance on minority class (the cases you might care about most)
    - Misleadingly high accuracy scores that hide poor minority class prediction

    **Choose how to proceed:**
    - **Select a resampling method** to balance your data
    - **Choose "None"** to proceed with training on original data (not recommended for severe imbalance)
    """)


def display_imbalance_handling_tools():
    """Display the full imbalance handling tools."""

    # Validate we have the necessary data
    if (not hasattr(st.session_state, 'builder') or
        st.session_state.builder.X_train is None or
        st.session_state.builder.y_train is None):
        st.error("Training data not available for imbalance handling.")
        return

    # Get fresh analysis
    imbalance_analysis = st.session_state.builder.analyse_class_imbalance()

    if not imbalance_analysis["success"]:
        st.error(f"Error analyzing imbalance: {imbalance_analysis['message']}")
        return

    metrics = imbalance_analysis["metrics"]

    st.subheader("🔧 Class Imbalance Handling Tools")

    # Create hashes for caching
    X_train_hash = hash(str(st.session_state.builder.X_train.shape) + str(st.session_state.builder.X_train.dtypes.tolist()))
    y_train_hash = hash(str(pd.Series(st.session_state.builder.y_train).value_counts().to_dict()))

    # Get resampling recommendation
    recommendation = _cached_imbalance_recommendation(
        X_train_hash, y_train_hash, metrics["imbalance_ratio"]
    )

    if recommendation["success"]:
        with st.expander("💡 Recommended Approach", expanded=True):
            st.write(f"**Recommended Method: {recommendation['recommended_method']}**")
            st.write(recommendation["explanation"])

            if recommendation["considerations"]:
                st.write("**Important Considerations:**")
                for consideration in recommendation["considerations"]:
                    st.write(f"• {consideration}")

    # Resampling method selection
    st.write("### Choose Resampling Method")

    method_options = [
        "None (Original Data)",
        "Random Oversampling",
        "Random Undersampling",
        "SMOTE",
        "ADASYN"
    ]

    # Put recommended method first if available
    if recommendation["success"]:
        recommended_method = recommendation["recommended_method"]
        if recommended_method in method_options:
            method_options.remove(recommended_method)
            method_options.insert(0, recommended_method)

    selected_method = st.selectbox(
        "Select Method",
        method_options,
        help="""
        - **None**: Use original imbalanced data
        - **Random Oversampling**: Duplicate minority class samples
        - **Random Undersampling**: Remove majority class samples
        - **SMOTE**: Generate synthetic minority samples
        - **ADASYN**: Advanced synthetic sampling
        """
    )

    # Method explanation
    method_explanations = {
        "None (Original Data)": {
            "description": "Use the original dataset without any resampling.",
            "pros": ["No data modification", "Maintains original data distribution"],
            "cons": ["Model may be biased towards majority class", "Poor minority class prediction"]
        },
        "Random Oversampling": {
            "description": "Randomly duplicate minority class samples until classes are balanced.",
            "pros": ["Simple to implement", "No data loss"],
            "cons": ["May lead to overfitting", "Duplicate samples"]
        },
        "Random Undersampling": {
            "description": "Randomly remove majority class samples until classes are balanced.",
            "pros": ["Simple to implement", "Reduces training time"],
            "cons": ["Loss of potentially important information", "Reduced dataset size"]
        },
        "SMOTE": {
            "description": "Generate synthetic minority samples using nearest neighbors.",
            "pros": ["Creates synthetic samples", "Better generalisation than oversampling"],
            "cons": ["Computationally intensive", "May generate noisy samples"]
        },
        "ADASYN": {
            "description": "Advanced synthetic sampling focusing on difficult-to-learn samples.",
            "pros": ["Focuses on hard-to-learn examples", "Better handling of complex patterns"],
            "cons": ["Most computationally intensive", "May not work well with very small datasets"]
        }
    }

    with st.expander("📋 Method Details", expanded=True):
        method_info = method_explanations[selected_method]
        st.write(method_info["description"])

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Advantages:**")
            for pro in method_info["pros"]:
                st.write(f"✓ {pro}")
        with col2:
            st.write("**Disadvantages:**")
            for con in method_info["cons"]:
                st.write(f"⚠️ {con}")

    # Action buttons
    col1 = st.columns(1)[0]

    with col1:
        if selected_method != "None (Original Data)":
            if st.button("🔧 Apply Resampling", type="primary"):
                apply_resampling_method(selected_method)
        else:
            if st.button("✅ Keep Original Data", type="primary"):
                st.session_state.imbalance_handled = True
                st.session_state.imbalance_skipped = True
                st.success("✅ Keeping original data - you can proceed with training.")
                st.rerun()


def apply_resampling_method(method_name):
    """Apply the selected resampling method to the training data."""

    try:
        with st.spinner(f"Applying {method_name}..."):
            # Get original data distribution
            original_dist = pd.Series(st.session_state.builder.y_train).value_counts()

            # Apply resampling based on selected method
            if method_name == "Random Oversampling":
                from imblearn.over_sampling import RandomOverSampler
                resampler = RandomOverSampler(random_state=42)
            elif method_name == "Random Undersampling":
                from imblearn.under_sampling import RandomUnderSampler
                resampler = RandomUnderSampler(random_state=42)
            elif method_name == "SMOTE":
                from imblearn.over_sampling import SMOTE
                resampler = SMOTE(random_state=42)
            elif method_name == "ADASYN":
                from imblearn.over_sampling import ADASYN
                resampler = ADASYN(random_state=42)
            else:
                st.error("Unknown resampling method")
                return

            # Resample the data
            X_resampled, y_resampled = resampler.fit_resample(
                st.session_state.builder.X_train,
                st.session_state.builder.y_train
            )

            # Update the builder with resampled data
            st.session_state.builder.X_train = X_resampled
            st.session_state.builder.y_train = y_resampled

            # Get new distribution
            new_dist = pd.Series(y_resampled).value_counts()

            # Log the resampling
            if hasattr(st.session_state, 'logger'):
                st.session_state.logger.log_calculation(
                    "Pre-Training Resampling Applied",
                    {
                        "method": method_name,
                        "original_samples": len(original_dist),
                        "resampled_samples": len(y_resampled),
                        "original_distribution": original_dist.to_dict(),
                        "new_distribution": new_dist.to_dict()
                    }
                )

                st.session_state.logger.log_journey_point(
                    stage="MODEL_TRAINING",
                    decision_type="RESAMPLING",
                    description="Pre-training resampling applied",
                    details={
                        "Method": method_name,
                        "Original Samples": len(original_dist),
                        "Resampled Samples": len(y_resampled),
                        "Timing": "Before Training"
                    },
                    parent_id=None
                )

            # Show comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Original',
                x=original_dist.index.astype(str),
                y=original_dist.values,
                text=original_dist.values,
                textposition='auto',
            ))
            fig.add_trace(go.Bar(
                name='After Resampling',
                x=new_dist.index.astype(str),
                y=new_dist.values,
                text=new_dist.values,
                textposition='auto',
            ))
            fig.update_layout(
                title='Class Distribution: Before vs After Resampling',
                xaxis_title='Class',
                yaxis_title='Count',
                barmode='group',
                height=400
            )

            st.plotly_chart(fig, config={'responsive': True})

            st.success(f"""
            ✅ **{method_name} Applied Successfully!**

            - Original samples: {len(original_dist):,}
            - After resampling: {len(y_resampled):,}
            - Classes are now balanced

            You can proceed with model training on the balanced dataset.
            """)

            # Update session state
            st.session_state.imbalance_handled = True
            st.session_state.show_imbalance_tools = False

    except Exception as e:
        st.error(f"Error applying resampling: {str(e)}")
        if hasattr(st.session_state, 'logger'):
            st.session_state.logger.log_error(
                "Pre-Training Resampling Failed",
                {
                    "method": method_name,
                    "error": str(e)
                }
            )


def display_imbalance_handling():
    """Display class imbalance handling with improved component communication."""

    # Import state manager for better component communication
    from components.model_training.utils.training_state_manager import TrainingStateManager

    # Validate training state
    validation = TrainingStateManager.validate_training_state()
    if not all(validation.values()):
        st.warning("Please complete model training first to access imbalance analysis.")
        return

    # Add class imbalance analysis for classification problems
    if st.session_state.builder.model is not None:
        problem_type = st.session_state.builder.model.get("problem_type", "unknown")

        # Handle both binary and multiclass classification
        if problem_type in ["classification", "binary_classification", "multiclass_classification"]:
            st.header("Class Imbalance Analysis")

            # Get imbalance analysis with caching
            # Create hashes for caching key
            X_train_hash = hash(str(st.session_state.builder.X_train.shape) + str(st.session_state.builder.X_train.dtypes.tolist()))
            y_train_hash = hash(str(pd.Series(st.session_state.builder.y_train).value_counts().to_dict()))

            imbalance_analysis = _cached_class_imbalance_analysis(X_train_hash, y_train_hash, problem_type)

            if imbalance_analysis["success"]:
                metrics = imbalance_analysis["metrics"]

                # Log the imbalance analysis
                st.session_state.logger.log_calculation(
                    "Class Imbalance Analysis",
                    {
                        "imbalance_ratio": metrics["imbalance_ratio"],
                        "severity": metrics["severity"],
                        "majority_class": metrics["majority_class"],
                        "minority_class": metrics["minority_class"],
                        "class_distribution": metrics["class_distribution"]
                    }
                )

                # Display class distribution using cached chart
                fig = _create_class_distribution_chart(metrics["class_distribution"])
                st.plotly_chart(fig)

                # Display imbalance metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Imbalance Ratio", f"{metrics['imbalance_ratio']:.2f}")
                    st.write(f"Imbalance Severity: **{metrics['severity']}**")

                with col2:
                    st.metric(
                        "Majority Class",
                        f"{metrics['majority_class']['count']} samples",
                        f"Class: {metrics['majority_class']['label']}"
                    )
                    st.metric(
                        "Minority Class",
                        f"{metrics['minority_class']['count']} samples",
                        f"Class: {metrics['minority_class']['label']}"
                    )

                # Show recommendations and resampling options
                if metrics['severity'] != "Mild or None":
                    st.warning("""
                        **Class Imbalance Detected**
                        Your dataset shows significant class imbalance, which can lead to biased model predictions.
                        Let's analyse the best resampling approach for your specific dataset.
                    """)

                    # Log imbalance warning and recommendations
                    st.session_state.logger.log_recommendation(
                        "Class Imbalance Mitigation",
                        {
                            "severity": metrics["severity"],
                            "imbalance_ratio": metrics["imbalance_ratio"],
                            "recommended_action": "Consider resampling techniques"
                        }
                    )

                    # Get resampling recommendation with caching
                    recommendation = _cached_imbalance_recommendation(
                        X_train_hash, y_train_hash, metrics["imbalance_ratio"]
                    )

                    if recommendation["success"]:
                        with st.expander("📊 Recommended Resampling Approach", expanded=True):
                            st.subheader(f"Recommended Method: {recommendation['recommended_method']}")

                            # Show explanation with metrics
                            st.write("### Why This Method?")
                            st.write(recommendation["explanation"])

                            # Show dataset characteristics
                            st.write("### Dataset Characteristics")
                            metrics = recommendation["metrics"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Imbalance Ratio", f"{metrics['imbalance_ratio']:.2f}")
                            with col2:
                                st.metric("Total Samples", f"{metrics['total_samples']:,}")
                            with col3:
                                st.metric("Min Class Size", f"{metrics['min_class_size']:,}")

                            # Show feature composition using cached chart
                            st.write("### Feature Composition")
                            fig = _create_feature_composition_chart(
                                metrics['numerical_features'],
                                metrics['categorical_features']
                            )
                            st.plotly_chart(fig)

                            # Show considerations
                            if recommendation["considerations"]:
                                st.write("### Important Considerations")
                                for consideration in recommendation["considerations"]:
                                    st.write(f"• {consideration}")

                    # Continue with resampling method selection...
                    resampling_method = st.selectbox(
                        "Select Resampling Method",
                        options=[
                            recommendation["recommended_method"],  # Put recommended method first
                            *[m for m in [
                                "None (Original Data)",
                                "Random Oversampling",
                                "Random Undersampling",
                                "SMOTE",
                                "ADASYN"
                            ] if m != recommendation["recommended_method"]]
                        ],
                        help="""
                        - **None**: Use original imbalanced data
                        - **Random Oversampling**: Duplicate minority class samples
                        - **Random Undersampling**: Remove majority class samples
                        - **SMOTE**: Generate synthetic minority samples
                        - **ADASYN**: Advanced synthetic sampling
                        """
                    )

                    # Log user selection
                    st.session_state.logger.log_user_action(
                        "Resampling Method Selected",
                        {"method": resampling_method}
                    )

                    # Show method explanation
                    method_explanations = {
                        "None (Original Data)": {
                            "description": "Use the original dataset without any resampling.",
                            "pros": ["No data modification", "Maintains original data distribution"],
                            "cons": ["Model may be biased towards majority class", "Poor minority class prediction"]
                        },
                        "Random Oversampling": {
                            "description": "Randomly duplicate minority class samples until classes are balanced.",
                            "pros": ["Simple to implement", "No data loss"],
                            "cons": ["May lead to overfitting", "Duplicate samples"]
                        },
                        "Random Undersampling": {
                            "description": "Randomly remove majority class samples until classes are balanced.",
                            "pros": ["Simple to implement", "Reduces training time"],
                            "cons": ["Loss of potentially important information", "Reduced dataset size"]
                        },
                        "SMOTE": {
                            "description": "Generate synthetic minority samples using nearest neighbors.",
                            "pros": ["Creates synthetic samples", "Better generalisation than oversampling"],
                            "cons": ["Computationally intensive", "May generate noisy samples"]
                        },
                        "ADASYN": {
                            "description": "Advanced synthetic sampling focusing on difficult-to-learn samples.",
                            "pros": ["Focuses on hard-to-learn examples", "Better handling of complex patterns"],
                            "cons": ["Most computationally intensive", "May not work well with very small datasets"]
                        }
                    }

                    with st.expander("Method Details", expanded=True):
                        method_info = method_explanations[resampling_method]
                        st.write(method_info["description"])
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Advantages:**")
                            for pro in method_info["pros"]:
                                st.write(f"✓ {pro}")
                        with col2:
                            st.write("**Disadvantages:**")
                            for con in method_info["cons"]:
                                st.write(f"⚠ {con}")

                    # Apply resampling if user chooses a method
                    if resampling_method != "None (Original Data)":
                        if st.button("Apply Resampling"):
                            # Log resampling attempt
                            st.session_state.logger.log_user_action(
                                "Resampling Initiated",
                                {"method": resampling_method}
                            )

                            with st.spinner("Applying resampling technique..."):
                                # Get original data distribution
                                original_dist = pd.Series(st.session_state.builder.y_train).value_counts()

                                # Apply resampling
                                try:
                                    if resampling_method == "Random Oversampling":
                                        from imblearn.over_sampling import RandomOverSampler
                                        resampler = RandomOverSampler(random_state=42)
                                    elif resampling_method == "Random Undersampling":
                                        from imblearn.under_sampling import RandomUnderSampler
                                        resampler = RandomUnderSampler(random_state=42)
                                    elif resampling_method == "SMOTE":
                                        from imblearn.over_sampling import SMOTE
                                        resampler = SMOTE(random_state=42)
                                    else:  # ADASYN
                                        from imblearn.over_sampling import ADASYN
                                        resampler = ADASYN(random_state=42)

                                    # Resample the data
                                    X_resampled, y_resampled = resampler.fit_resample(
                                        st.session_state.builder.X_train,
                                        st.session_state.builder.y_train
                                    )

                                    # Update the builder with resampled data
                                    st.session_state.builder.X_train = X_resampled
                                    st.session_state.builder.y_train = y_resampled

                                    # Get new distribution
                                    new_dist = pd.Series(y_resampled).value_counts()

                                    # Log resampling results
                                    st.session_state.logger.log_calculation(
                                        "Resampling Results",
                                        {
                                            "method": resampling_method,
                                            "original_samples": len(st.session_state.builder.y_train),
                                            "resampled_samples": len(y_resampled),
                                            "original_distribution": original_dist.to_dict(),
                                            "new_distribution": new_dist.to_dict()
                                        }
                                    )
                                    st.session_state.logger.log_journey_point(
                                        stage="MODEL_TRAINING",
                                        decision_type="RESAMPLING",
                                        description="Resampling applied",
                                        details={"Method": resampling_method,
                                                "Original Samples": len(st.session_state.builder.y_train),
                                                "Resampled Samples": len(y_resampled),
                                                "Original Distribution": original_dist.to_dict()},
                                        parent_id=None
                                    )
                                    # Create comparison plot
                                    fig = go.Figure()
                                    fig.add_trace(go.Bar(
                                        name='Original',
                                        x=original_dist.index.astype(str),
                                        y=original_dist.values,
                                        text=original_dist.values,
                                        textposition='auto',
                                    ))
                                    fig.add_trace(go.Bar(
                                        name='After Resampling',
                                        x=new_dist.index.astype(str),
                                        y=new_dist.values,
                                        text=new_dist.values,
                                        textposition='auto',
                                    ))
                                    fig.update_layout(
                                        title='Class Distribution Comparison',
                                        xaxis_title='Class',
                                        yaxis_title='Count',
                                        barmode='group'
                                    )

                                    st.plotly_chart(fig)

                                    st.success(f"""
                                        Successfully applied {resampling_method}!
                                        - Original samples: {len(st.session_state.builder.y_train)}
                                        - After resampling: {len(y_resampled)}
                                    """)

                                except Exception as e:
                                    error_msg = str(e)
                                    st.error(f"Error applying resampling: {error_msg}")
                                    st.session_state.logger.log_error(
                                        "Resampling Failed",
                                        {
                                            "method": resampling_method,
                                            "error": error_msg,
                                            "original_samples": len(st.session_state.builder.y_train)
                                        }
                                    )
                else:
                    st.success("""
                        Your dataset shows good class balance. No resampling is necessary.
                        You can proceed with model training using the original data.
                    """)
            else:
                st.error(imbalance_analysis["message"])
                st.session_state.logger.log_error(
                    "Class Imbalance Analysis Failed",
                    {"error": imbalance_analysis["message"]}
                )