import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

class AutomatedFeatureSelectionComponent:
    def __init__(self, builder, logger):
        """
        Initialize the AutomatedFeatureSelectionComponent.
        
        Args:
            builder: The Builder instance containing the dataset and methods
            logger: The logger instance for tracking actions and calculations
        """
        self.builder = builder
        self.logger = logger
        
        # Initialize session state variables if not present
        if 'auto_selection_active' not in st.session_state:
            st.session_state.auto_selection_active = False
        if 'boruta_result' not in st.session_state:
            st.session_state.boruta_result = None
        if 'boruta_success' not in st.session_state:
            st.session_state.boruta_success = None

    def render(self):
        """Render the automated feature selection interface."""
        # Add explanation expander
        with st.expander("ℹ️ Understanding Automated Feature Selection", expanded=False):
            st.markdown("""
            ### What is Automated Feature Selection?
            
            Think of automated feature selection like having a smart assistant that helps you choose the most important ingredients for a recipe. Just like how not every ingredient is crucial for a dish, not every feature (piece of information) in your data is important for making predictions.
            
            ### How Does It Work?
            
            The Boruta algorithm (our automated assistant) works by:
            1. 🔄 Creating "shadow" copies of your features by mixing up their values
            2. 🌳 Using Random Forest (a powerful machine learning method) to compare real features with shadows
            3. ⚖️ Keeping only features that consistently perform better than their shadows
            4. 📊 Giving you a clear report of which features to keep or remove
            
            ### Why Use Automated Selection?
            
            **Benefits:**
            - ⏱️ **Saves Time**: Instead of manually reviewing each feature, the algorithm does it for you
            - 📈 **More Accurate**: Uses statistical methods to make unbiased decisions
            - 🎯 **Consistent**: Applies the same criteria to all features
            - 🧪 **Scientifically Proven**: Based on well-tested mathematical principles
            
            ### When to Use It?
            
            Automated selection is great when:
            - You have many features (10+ columns)
            - You're not sure which features are important
            - You want an unbiased, data-driven approach
            - You need a quick, reliable starting point
            
            ### What to Expect?
            
            After running automated selection:
            1. Features will be categorized as:
               - ✅ **Confirmed**: Definitely keep these
               - ⚠️ **Tentative**: Maybe keep these (needs your judgment)
               - ❌ **Rejected**: Safe to remove
            2. You'll see clear visualisations of the results
            3. You can still make the final decision on what to keep
            
            > 💡 **Tip**: Even with automation, your domain knowledge is valuable! Use the automated results as a guide, but feel free to keep features you know are important for your specific case.
            """)
        
        # Add automated selection toggle
        auto_selection = st.toggle(
            "Use Automated Feature Selection",
            help="Enable to use Boruta algorithm for automated feature selection",
            #value=st.session_state.auto_selection_active,
            value=False,
            key="auto_selection_toggle"
        )
        
        # Update the active state
        st.session_state.auto_selection_active = auto_selection
        
        if auto_selection:
            self._render_automated_selection_interface()

    def _render_automated_selection_interface(self):
        """Render the interface for automated feature selection."""
        st.warning("""
        ⚠️ **Automated Feature Selection**

        This will use the Boruta algorithm to automatically select relevant features. The process may take several minutes
        depending on your dataset size and complexity. Boruta works by:

        1. Creating shadow features by shuffling your original features
        2. Training a Random Forest model multiple times
        3. Comparing original features with shadow features to determine importance
        4. Iteratively selecting statistically significant features

        Please be patient while the algorithm runs.
        """)

        # Check if we have a success message to display
        if 'boruta_success' in st.session_state:
            if st.session_state.boruta_success is not None:
                st.success(st.session_state.boruta_success)
                # Clear the success message after displaying it
                del st.session_state.boruta_success

        if st.button("Start Automated Selection", type="primary"):
            # Log the start of automated selection
            if 'logger' in st.session_state:
                st.session_state.logger.log_user_action(
                    "Automated Feature Selection Started",
                    {
                        "method": "Boruta Algorithm",
                        "initial_features": list(self.builder.X_train.columns),
                        "initial_feature_count": len(self.builder.X_train.columns),
                        "training_samples": len(self.builder.X_train),
                        "testing_samples": len(self.builder.X_test)
                    }
                )

            with st.spinner("Running Boruta feature selection..."):
                # Store current state in history before making any changes
                st.session_state.feature_history.append({
                    'X_train': self.builder.X_train.copy(),
                    'X_test': self.builder.X_test.copy(),
                    'y_train': self.builder.y_train.copy(),
                    'y_test': self.builder.y_test.copy(),
                    'step': st.session_state.get('feature_selection_step', 1),
                    'metrics': st.session_state.get('dedup_metrics', None)
                })

                # Run Boruta selection and store results in session state
                st.session_state.boruta_result = self._run_boruta_selection()

                # Log the completion of Boruta analysis
                if 'logger' in st.session_state and st.session_state.boruta_result.get("success"):
                    result_info = st.session_state.boruta_result.get("info", {})
                    stats = result_info.get("statistics", {})
                    st.session_state.logger.log_calculation(
                        "Boruta Feature Analysis Completed",
                        {
                            "total_features": stats.get("total_features", 0),
                            "confirmed_features": stats.get("confirmed_features", 0),
                            "tentative_features": stats.get("tentative_features", 0),
                            "rejected_features": stats.get("rejected_features", 0),
                            "selection_ratio": stats.get("selection_ratio", 0),
                            "confirmed_feature_list": result_info.get("confirmed_features", []),
                            "tentative_feature_list": result_info.get("tentative_features", []),
                            "rejected_feature_list": result_info.get("rejected_features", [])
                        }
                    )

                st.rerun()
        
        # Check if we have Boruta results in session state and they are valid
        boruta_result = st.session_state.get('boruta_result')
        if boruta_result is not None and isinstance(boruta_result, dict) and boruta_result.get("success"):
            self._render_boruta_results()

    def _render_boruta_results(self):
        """Render the results of the Boruta feature selection."""
        result = st.session_state.boruta_result
        st.success("✅ Automated feature selection completed!")
        
        # Display statistics
        stats = result["info"]["statistics"]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Features", stats["total_features"])
        with col2:
            st.metric(
                "Confirmed Features",
                stats["confirmed_features"],
                f"{stats['selection_ratio']:.1%}"
            )
        with col3:
            st.metric("Tentative Features", stats["tentative_features"])
        with col4:
            st.metric("Rejected Features", stats["rejected_features"])
        
        # Create feature importance plot
        self._create_feature_importance_plot(result["info"]["feature_ranking"])
        
        # Show feature details in expandable sections
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self._render_feature_status_section(result["info"]["feature_ranking"])
        
        with col2:
            self._render_selection_summary(result)

    def _create_feature_importance_plot(self, feature_ranking: List[Dict[str, Any]]):
        """Create and display the feature importance plot."""
        fig = go.Figure()
        
        # Add bars for each feature status
        colors = {
            "Confirmed": "rgba(46, 204, 113, 0.8)",
            "Tentative": "rgba(241, 196, 15, 0.8)",
            "Rejected": "rgba(231, 76, 60, 0.8)"
        }
        
        for status in ["Confirmed", "Tentative", "Rejected"]:
            features = [f for f in feature_ranking if f["status"] == status]
            if features:
                fig.add_trace(go.Bar(
                    name=status,
                    x=[f["feature"] for f in features],
                    y=[f["importance"] for f in features],
                    marker_color=colors[status],
                    hovertemplate="<b>%{x}</b><br>" +
                                "Importance: %{y:.3f}<br>" +
                                f"Status: {status}<extra></extra>"
                ))
        
        fig.update_layout(
            title="Feature Importance by Status",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            barmode='group',
            height=500,
            xaxis={'tickangle': 45},
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        st.plotly_chart(fig, config={'responsive': True}, key="boruta_feature_importance_plot")

    def _render_feature_status_section(self, feature_ranking: List[Dict[str, Any]]):
        """Render the feature status section with expandable lists."""
        st.write("### Feature Status")
        for status in ["Confirmed", "Tentative", "Rejected"]:
            with st.expander(f"{status} Features"):
                features = [f["feature"] for f in feature_ranking if f["status"] == status]
                for feat in features:
                    st.markdown(f"• {feat}")

    def _render_selection_summary(self, result: Dict[str, Any]):
        """Render the selection summary and options."""
        st.write("### Selection Summary")
        st.write("""
        #### Understanding the Results
        
        - **Confirmed Features**: Consistently important features that should be kept
        - **Tentative Features**: Features that might be important but need further investigation
        - **Rejected Features**: Features that can be safely removed
        
        #### Next Steps
        """)
        
        # Add options for feature selection
        selection_option = st.radio(
            "Choose which features to keep:",
            ["Confirmed Only", "Confirmed + Tentative", "Custom Selection"],
            help="Select which feature sets to keep in your dataset",
            key="boruta_selection_option"
        )
        
        selected_features = self._get_selected_features(selection_option, result)
        
        if selected_features:
            if st.button("Apply Selection", type="primary", key="boruta_apply_selection"):
                self._apply_feature_selection(selected_features)
        else:
            st.error(f"Error in automated feature selection: {result['message']}")

    def _get_selected_features(self, selection_option: str, result: Dict[str, Any]) -> List[str]:
        """Get the list of selected features based on the selection option."""
        if selection_option == "Custom Selection":
            return st.multiselect(
                "Select features to remove:",
                options=list(self.builder.X_train.columns),
                default=result["info"]["rejected_features"],
                help="Choose features to remove from your dataset",
                key="boruta_custom_selection"
            )
        elif selection_option == "Confirmed Only":
            return result["info"]["tentative_features"] + result["info"]["rejected_features"]
        else:  # Confirmed + Tentative
            return result["info"]["rejected_features"]

    def _run_boruta_selection(self) -> Dict[str, Any]:
        """
        Run Boruta feature selection algorithm.

        Returns:
            Dict containing:
            - success: bool indicating if the operation was successful
            - message: str explaining the result
            - info: Dict containing detailed results if successful
        """
        try:
            from boruta import BorutaPy
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            # Determine problem type and create appropriate estimator
            problem_type = self.builder.detect_problem_type()
            is_classification = problem_type in ["binary_classification", "multiclass_classification", "classification"]
            if is_classification:
                estimator = RandomForestClassifier(n_jobs=-1, max_depth=5)
            else:  # regression
                estimator = RandomForestRegressor(n_jobs=-1, max_depth=5)

            # Initialize Boruta
            boruta = BorutaPy(
                estimator=estimator,
                n_estimators='auto',
                max_iter=100,  # Increased from default for more thorough selection
                verbose=2,  # Enable verbose output for debugging
                random_state=42
            )

            # Create processed copies of the data
            X_train_processed = self.builder.X_train.copy()
            y_train_processed = self.builder.y_train.copy()

            # Process each column based on its content
            for col in X_train_processed.columns:
                column_data = X_train_processed[col]

                # Try to convert to numeric first
                try:
                    numeric_data = pd.to_numeric(column_data, errors='raise')
                    X_train_processed[col] = numeric_data
                    continue
                except (ValueError, TypeError):
                    pass

                # If not numeric, handle categorical data
                unique_values = column_data.nunique()
                if unique_values <= 10:  # Threshold for categorical
                    X_train_processed[col] = pd.Categorical(column_data).codes
                else:
                    # For high cardinality strings, use hash encoding
                    X_train_processed[col] = pd.util.hash_array(column_data.fillna(''), num_items=100)

                X_train_processed[col] = X_train_processed[col].astype(float)

            # Handle missing values before Boruta
            X_train_processed = X_train_processed.fillna(X_train_processed.mean())

            # Fit Boruta
            boruta.fit(X_train_processed.values, y_train_processed.values)

            # Get feature ranking and support masks
            feature_ranks = boruta.ranking_
            confirmed_mask = boruta.support_
            tentative_mask = boruta.support_weak_
            rejected_mask = ~(confirmed_mask | tentative_mask)

            # Create feature importance ranking
            feature_ranking = []
            for idx, (feature, rank) in enumerate(zip(self.builder.X_train.columns, feature_ranks)):
                status = "Confirmed" if confirmed_mask[idx] else ("Tentative" if tentative_mask[idx] else "Rejected")
                feature_ranking.append({
                    "feature": feature,
                    "rank": int(rank),
                    "status": status,
                    "importance": float(
                        boruta.importance_history_[:, idx].mean() if boruta.importance_history_ is not None else 0)
                })

            # Sort by importance
            feature_ranking.sort(
                key=lambda x: (-1 if x["status"] == "Confirmed" else (0 if x["status"] == "Tentative" else 1),
                               x["importance"]),
                reverse=False)

            # Calculate statistics
            stats = {
                "total_features": len(self.builder.X_train.columns),
                "confirmed_features": int(confirmed_mask.sum()),
                "tentative_features": int(tentative_mask.sum()),
                "rejected_features": int(rejected_mask.sum()),
                "selection_ratio": float(confirmed_mask.sum() / len(self.builder.X_train.columns)),
                "iterations": 100,  # Using max_iter value since n_iter_ is not available
            }

            # Get lists of features by status
            confirmed_features = [f["feature"] for f in feature_ranking if f["status"] == "Confirmed"]
            tentative_features = [f["feature"] for f in feature_ranking if f["status"] == "Tentative"]
            rejected_features = [f["feature"] for f in feature_ranking if f["status"] == "Rejected"]

            # Create importance history if available
            importance_history = None
            if hasattr(boruta, 'importance_history_') and boruta.importance_history_ is not None:
                importance_history = boruta.importance_history_.tolist()

            return {
                "success": True,
                "message": "Boruta feature selection completed successfully",
                "info": {
                    "feature_ranking": feature_ranking,
                    "statistics": stats,
                    "confirmed_features": confirmed_features,
                    "tentative_features": tentative_features,
                    "rejected_features": rejected_features,
                    "importance_history": importance_history
                }
            }
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return {
                "success": False,
                "message": f"Error running Boruta feature selection: {str(e)}\n\nDetails:\n{error_details}"
            }

    def _apply_feature_selection(self, selected_features: List[str]):
        """Apply the selected feature changes."""
        # Import the update_features function from our new utility module
        from components.feature_selection.utils.feature_utils import update_features

        # Store current state in history before making any changes
        st.session_state.feature_history.append({
            'X_train': self.builder.X_train.copy(),
            'X_test': self.builder.X_test.copy(),
            'y_train': self.builder.y_train.copy(),
            'y_test': self.builder.y_test.copy(),
            'step': st.session_state.get('feature_selection_step', 1),
            'metrics': st.session_state.get('dedup_metrics', None)
        })

        # Get the selection option to determine what type of removal this was
        selection_option = st.session_state.get('boruta_selection_option', 'Confirmed Only')

        # Use the utility function instead of the builder method
        update_result = update_features(self.builder, selected_features)
        if update_result["success"]:
            # Track the automated feature removal using the tracking function
            if hasattr(st.session_state, 'track_automated_feature_removal'):
                # Determine if this addresses low importance features
                addresses_low_importance = True  # Boruta inherently removes low importance features

                # Get more specific method name based on selection option
                method_name = f"Automated Selection (Boruta - {selection_option})"

                # Call the tracking function
                st.session_state.track_automated_feature_removal(
                    removed_features=selected_features,
                    method_name=method_name,
                    addresses_low_importance=addresses_low_importance,
                    addresses_correlation=False  # Boruta doesn't specifically address correlation
                )

                # Log the automated selection details
                if 'logger' in st.session_state:
                    boruta_result = st.session_state.get('boruta_result', {})
                    st.session_state.logger.log_user_action(
                        "Automated Feature Selection Applied",
                        {
                            "method": "Boruta Algorithm",
                            "selection_option": selection_option,
                            "removed_features": selected_features,
                            "removed_count": len(selected_features),
                            "remaining_features": list(self.builder.X_train.columns),
                            "remaining_count": len(self.builder.X_train.columns),
                            "boruta_statistics": boruta_result.get("info", {}).get("statistics", {}),
                            "confirmed_features": boruta_result.get("info", {}).get("confirmed_features", []),
                            "tentative_features": boruta_result.get("info", {}).get("tentative_features", []),
                            "rejected_features": boruta_result.get("info", {}).get("rejected_features", [])
                        }
                    )

            # Reset Boruta state and update step
            if 'boruta_result' in st.session_state:
                del st.session_state.boruta_result
            if 'auto_selection_active' not in st.session_state:
                st.session_state.auto_selection_active = False
            st.session_state.feature_selection_step = 2
            st.session_state.boruta_success = "✅ Feature selection applied successfully! Selected features have been updated."
            st.rerun()
        else:
            st.error(update_result["message"]) 