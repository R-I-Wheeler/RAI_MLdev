import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components.model_training.utils.training_state_manager import TrainingStateManager


def safe_set_params_and_fit(model, params, X_train, y_train):
    """
    Safely set parameters and fit a model, handling CatBoost's inability to change params after fitting.

    Args:
        model: The model instance
        params: Parameters to set
        X_train: Training features
        y_train: Training target

    Returns:
        Fitted model instance (either the same model or a new one for CatBoost)
    """
    model_type = type(model).__name__

    # CatBoost models cannot have their parameters changed after fitting
    # We need to create a new instance instead
    if 'CatBoost' in model_type:
        # Create a new model instance with the desired parameters
        model_class = type(model)
        new_model = model_class(**params)
        new_model.fit(X_train, y_train)
        return new_model
    else:
        # For other models, we can use set_params
        model.set_params(**params)
        model.fit(X_train, y_train)
        return model

@st.cache_data(ttl=300, show_spinner=False)
def _cached_architecture_analysis(param_values_hash, scores_hash, best_value_hash):
    """Cache architecture analysis to avoid recalculation"""
    return None

@st.cache_data(ttl=300, show_spinner=False)
def _cached_parameter_visualization(param_name, param_values, scores, param_type):
    """Cache parameter visualization creation"""
    return None

@st.cache_data(ttl=600, show_spinner=False)
def _cached_results_processing(cv_results_hash, problem_type):
    """Cache training results processing"""
    return None

def normalize_arch(arch):
    # Normalize architecture values to a consistent string format
    if isinstance(arch, (tuple, list)):
        return ','.join(str(x) for x in arch)
    return str(arch).strip('()[]').rstrip(',')

def analyse_network_architecture(param_values, scores, best_value):
    """Create visualization for neural network architecture parameter."""
   
    # Create normalized architecture scores dictionary
    arch_scores = {}
    for val, score in zip(param_values, scores):
        norm_val = normalize_arch(val)
        if norm_val not in arch_scores:
            arch_scores[norm_val] = []
        arch_scores[norm_val].append(score)
    
    # Create tabs for different visualizations
    arch_tab1, arch_tab2 = st.tabs(["Performance Comparison", "Network Structure"])
    
    with arch_tab1:
        # Performance comparison
        fig = go.Figure()
        
        # Find the best performing architecture
        best_arch_score = float('-inf')
        best_performing_arch = None
        for arch, scores_list in arch_scores.items():
            mean_score = np.mean(scores_list)
            if mean_score > best_arch_score:
                best_arch_score = mean_score
                best_performing_arch = arch
            
            # Add box plot with highlighting for best value
            is_best = normalize_arch(best_value) == arch
            fig.add_trace(go.Box(
                y=scores_list,
                name=arch,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                marker=dict(
                    color='#2ecc71' if is_best else 'lightblue',
                    line=dict(color='black', width=2) if is_best else None
                ),
                hovertemplate="Architecture: %{x}<br>Score: %{y:.4f}<extra></extra>"
            ))
        
        # Add annotation for the best architecture
        norm_best = normalize_arch(best_value)
        if norm_best in arch_scores:
            fig.add_annotation(
                x=norm_best,
                y=np.mean(arch_scores[norm_best]),
                text="Best Architecture ★",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                font=dict(size=12, color='black', weight='bold'),
                bgcolor='#2ecc71',
                bordercolor='black',
                borderwidth=2,
                borderpad=4,
                opacity=0.8
            )
        
        fig.update_layout(
            title="Architecture Performance Comparison",
            yaxis_title='Score',
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, config={'responsive': True})
    
    with arch_tab2:
        try:
            # Get the normalized best value for lookup
            norm_best = normalize_arch(best_value)
            
            st.info(f"""
                **Visualizing Best Architecture**: {best_value}
                
                This network structure represents the best performing architecture found during optimisation.
                The visualization shows how information flows through the layers.
            """)
            
            # Parse the architecture string to get layer sizes
            try:
                if isinstance(best_value, str):
                    # Handle string format
                    clean_arch = best_value.strip('()[]')  # Remove brackets/parentheses
                    clean_arch = clean_arch.rstrip(',')  # Remove trailing comma if present
                    # Split and convert to integers, handling empty strings
                    layers = [int(x.strip()) for x in clean_arch.split(",") if x.strip()]
                elif isinstance(best_value, tuple):
                    # Handle tuple format directly
                    layers = list(best_value)
                else:
                    # Try converting to string and parsing
                    clean_arch = str(best_value).strip('()[]').rstrip(',')
                    layers = [int(x.strip()) for x in clean_arch.split(",") if x.strip()]
                
                # Validate layers
                if not layers:
                    raise ValueError("No valid layer sizes found in architecture specification")
                
                # Convert any numpy integers to Python integers
                layers = [int(layer) for layer in layers]
                
                # Calculate positions
                input_size = len(st.session_state.builder.X_train.columns)
                all_layers = [input_size] + layers + [1]  # Add input and output layers
                max_neurons = max(all_layers)
                layer_positions = {}  # Store positions for each layer
                
                # Create network visualization
                fig = go.Figure()
                
                # First pass: Create all node positions
                for i, layer_size in enumerate(all_layers):
                    x_pos = i / (len(all_layers) - 1)  # Normalize x position
                    y_positions = np.linspace(-0.9, 0.9, layer_size)
                    layer_positions[i] = {'x': x_pos, 'y': y_positions}
                
                # Second pass: Create edges
                edge_traces = []
                for i in range(len(all_layers) - 1):
                    current_layer = layer_positions[i]
                    next_layer = layer_positions[i + 1]
                    
                    edge_x = []
                    edge_y = []
                    
                    # Create edges between current and next layer
                    for y1 in current_layer['y']:
                        for y2 in next_layer['y']:
                            edge_x.extend([current_layer['x'], next_layer['x'], None])
                            edge_y.extend([y1, y2, None])
                    
                    # Add edges trace
                    edge_traces.append(go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        mode='lines',
                        line=dict(
                            color='rgba(70, 130, 180, 0.3)',
                            width=1
                        ),
                        hoverinfo='none',
                        showlegend=False
                    ))
                
                # Add all edge traces to figure
                for trace in edge_traces:
                    fig.add_trace(trace)
                
                # Third pass: Create nodes
                for i, layer_size in enumerate(all_layers):
                    # Determine node properties based on layer type
                    if i == 0:
                        node_color = 'lightblue'
                        node_text = "Input"
                    elif i == len(all_layers) - 1:
                        node_color = 'lightgreen'
                        node_text = "Output"
                    else:
                        node_color = '#2ecc71'  # Highlight hidden layers
                        node_text = f"Hidden {i}"
                    
                    # Add nodes trace
                    fig.add_trace(go.Scatter(
                        x=[layer_positions[i]['x']] * layer_size,
                        y=layer_positions[i]['y'],
                        mode='markers+text',
                        marker=dict(
                            size=20,
                            color=node_color,
                            line=dict(color='black', width=1)
                        ),
                        text=[f"{node_text} ({layer_size})"] * layer_size,
                        textposition="top center",
                        hovertemplate=f"Layer: {node_text}<br>Neurons: {layer_size}<extra></extra>",
                        showlegend=False
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"Network Architecture: {best_value}",
                    showlegend=False,
                    plot_bgcolor='white',
                    height=500,  # Increased height
                    margin=dict(l=40, r=40, t=60, b=40),  # Increased margins
                    xaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False,
                        range=[-0.1, 1.1]  # Add some padding
                    ),
                    yaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False,
                        range=[-1.1, 1.1]  # Add some padding
                    )
                )
                
                st.plotly_chart(fig, config={'responsive': True})
                
                # Add architecture details with normalized score lookup
                st.info(f"""
                    📊 **Architecture Details**:
                    - Input Layer: {input_size} neurons
                    - Hidden Layers: {' → '.join(str(x) for x in layers)} neurons
                    - Output Layer: 1 neuron
                    - Total Parameters: {sum(l1 * l2 + l2 for l1, l2 in zip(all_layers[:-1], all_layers[1:]))}
                    - Performance Score: {np.mean(arch_scores[norm_best]):.4f}
                """)
                
            except Exception as e:
                st.error(f"""
                    Could not visualize network architecture: {str(e)}
                    
                    This might happen if the architecture format is unexpected or invalid.
                    Expected format examples: '50', '100,50', '50,25,10'
                """)
                st.warning(f"Raw architecture value: {best_value}")
        
        except Exception as e:
            st.error(f"""
                Failed to process network architecture: {str(e)}
                
                Please ensure the architecture values are in the correct format:
                - Single layer: '50' or '(50,)'
                - Multiple layers: '100,50' or '(100,50)'
            """)
    
    # Return the number of unique architectures and average performance
    return len(arch_scores), np.mean([np.mean(scores_list) for scores_list in arch_scores.values()])

def analyse_mixed_parameter(param_values, scores, best_value):
    """Create visualization for parameters with both numeric and 'auto' values."""
    auto_scores = []
    numeric_scores = {}
    
    for val, score in zip(param_values, scores):
        if str(val).lower() == 'auto':
            auto_scores.append(score)
        else:
            try:
                num_val = float(val)
                if num_val not in numeric_scores:
                    numeric_scores[num_val] = []
                numeric_scores[num_val].append(score)
            except (ValueError, TypeError):
                continue
    
    fig = go.Figure()
    
    # Find the best performing value
    best_score = float('-inf')
    best_performing_value = None
    
    if auto_scores:
        auto_mean = np.mean(auto_scores)
        if auto_mean > best_score:
            best_score = auto_mean
            best_performing_value = 'auto'
        fig.add_trace(go.Box(
            y=auto_scores,
            name='auto',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(
                color='lightblue' if best_value != 'auto' else '#2ecc71',
                line=dict(color='black', width=2) if best_value == 'auto' else None
            ),
            hovertemplate="Value: auto<br>Score: %{y:.4f}<extra></extra>"
        ))
    
    for val in sorted(numeric_scores.keys()):
        scores_list = numeric_scores[val]
        mean_score = np.mean(scores_list)
        if mean_score > best_score:
            best_score = mean_score
            best_performing_value = val
        
        is_best = val == best_value
        fig.add_trace(go.Box(
            y=scores_list,
            name=str(int(val) if float(val).is_integer() else val),
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(
                color='#2ecc71' if is_best else 'lightblue',
                line=dict(color='black', width=2) if is_best else None
            ),
            hovertemplate=f"Value: {val}<br>Score: %{{y:.4f}}<extra></extra>"
        ))
    
    # Add a more prominent indicator for the best value
    fig.add_annotation(
        x=str(best_value),
        y=best_score,
        text="Best Value ★",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40,
        font=dict(size=12, color='black', weight='bold'),
        bgcolor='#2ecc71',
        bordercolor='black',
        borderwidth=2,
        borderpad=4,
        opacity=0.8
    )
    
    fig.update_layout(
        title="Parameter Value Performance",
        yaxis_title='Score',
        showlegend=False,
        height=400
    )
    
    auto_mean = np.mean(auto_scores) if auto_scores else None
    numeric_mean = np.mean([np.mean(s) for s in numeric_scores.values()]) if numeric_scores else None
    total_configs = len(auto_scores) + sum(len(s) for s in numeric_scores.values())
    
    return fig, auto_mean, numeric_mean, total_configs

def display_optuna_results(results):
    with st.expander("How Optuna Works in ML Builder", expanded=False):
        st.markdown("""
            ### How Optuna Works in ML Builder

            Optuna is an advanced hyperparameter optimisation library that intelligently finds the best parameters for your model. In ML Builder, we've implemented a sophisticated version that:

            #### 1. Analyses Your Dataset 📊
            Before starting optimisation, ML Builder analyses your data to determine:
            - Dataset size (small < 1000 samples)
            - Feature dimensionality (high dimensional > 100 features)
            - Data sparsity (density of non-zero values)
            - Class distribution (for classification problems)

            #### 2. Adapts Parameter Search Ranges 🎯
            Based on the dataset analysis, ML Builder automatically adjusts parameter ranges:

            - **For Large Datasets:**
                - Allows larger model sizes
                - Tests higher learning rates
                - Explores more complex architectures

            - **For High-Dimensional Data:**
                - Restricts tree depths
                - Adjusts regularization ranges
                - Modifies network architectures

            - **For Small Datasets:**
                - Reduces model complexity
                - Tests smaller batch sizes
                - Focuses on preventing overfitting

            #### 3. Model-Specific Optimisation 🔄
            Different strategies for each model type:

            - **Neural Networks (MLP):**
                - Adapts hidden layer sizes to feature count
                - Adjusts learning rates based on dataset size
                - Tests appropriate batch sizes

            - **Tree-Based Models (Random Forest, XGBoost, LightGBM):**
                - Scales tree depth with feature count
                - Adjusts number of estimators based on dataset size
                - Optimizes leaf node parameters

            - **Linear Models:**
                - Adapts regularization strength to data dimensionality
                - Adjusts convergence parameters

            #### 4. Smart Trial Management ⚡
            - Implements early pruning of unpromising trials
            - Uses Bayesian optimisation to learn from previous trials
            - Tracks and analyses parameter importance

            #### 5. Cross-Validation Integration 🔄
            - Evaluates each parameter set using cross-validation
            - Maintains consistent splits across trials
            - Uses appropriate metrics for your problem type:
                - Classification: F1 Score
                - Regression: R² Score

            This sophisticated approach ensures that:
            - Parameter search is optimized for your specific case
            - Computational resources are used efficiently
            - The best parameters are found more quickly
            - Results are more reliable and robust
        """)
    
    #Display the parameters used in the Optuna optimisation
    st.header("🎯 Final Model Configuration")
    
    # Display the best parameters and scores
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Best Parameters")
        st.json(results['info']['best_params'])
    
    with col2:
        st.subheader("Model Performance")
        # Validate cv_metrics exists before accessing
        if 'cv_metrics' not in results['info']:
            st.error("Cross-validation metrics not available in training results.")
            return
        
        metrics_df = pd.DataFrame({
            'Metric': ['Mean CV Score', 'Standard Deviation', 'Best Score'],
            'Value': [
                f"{results['info']['cv_metrics']['mean_score']:.4f}",
                f"{results['info']['cv_metrics']['std_score']:.4f}",
                f"{results['info']['best_score']:.4f}"
            ]
        })
        st.table(metrics_df)
    
    #Save best_score to model info
    st.session_state.builder.model['best_score'] = results['info']['best_score']
    
    # Display Optuna-specific visualizations
    #with st.expander("📈 Optimisation History", expanded=True):
        
    
    # Add new optimisation analysis sections
    with st.expander("📊 Optimisation Analysis", expanded=True):
        st.subheader("Detailed Optimisation Metrics")
        
        # Extract optimisation metrics
        if ('optimisation_history' in results['info'] and 
            'metrics' in results['info']['optimisation_history'] and 
            results['info']['optimisation_history']['metrics'] is not None):
            metrics = results['info']['optimisation_history']['metrics']
            
            # Create columns for metrics display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Trials",
                    metrics.get('n_complete_trials', 'N/A'),
                    f"{metrics.get('n_pruned_trials', 0)} pruned"
                )
            
            with col2:
                # Calculate optimisation efficiency if data available
                if ('n_complete_trials' in metrics and 'n_pruned_trials' in metrics and 
                    metrics['n_complete_trials'] > 0):  # Check for zero
                    efficiency = (metrics['n_complete_trials'] - metrics['n_pruned_trials']) / metrics['n_complete_trials'] * 100
                    st.metric(
                        "Optimisation Efficiency",
                        f"{efficiency:.1f}%",
                        "of trials completed"
                    )
                else:
                    st.metric(
                        "Optimisation Efficiency",
                        "N/A",
                        "No completed trials"
                    )
            
            with col3:
                # Format duration if available
                if 'study_duration' in metrics and metrics['study_duration'] is not None:
                    duration = metrics['study_duration']
                    # Convert duration into minutes and seconds
                    minutes = int(duration.total_seconds() // 60)
                    seconds = int(duration.total_seconds() % 60)

                    duration_str = f"{minutes} minutes {seconds} seconds"
                    st.metric("Total Duration", duration_str)
                else:
                    st.metric("Total Duration", "N/A")
            
            if results['info']['optimisation_plots']['timeline'] is not None:
                st.plotly_chart(results['info']['optimisation_plots']['timeline'], config={'responsive': True})

            st.subheader("Optimisation Progress")
            if results['info']['optimisation_plots']['history'] is not None:
                st.plotly_chart(results['info']['optimisation_plots']['history'], config={'responsive': True})
                st.markdown("""
                    ### Understanding the Optimisation History Plot
                    
                    This plot shows how the model's performance improved during the optimisation process:
                    
                    - **Blue Line & Points**: Individual trial scores
                    - **Red Dashed Line**: Best score achieved so far
                    
                    The ideal pattern shows:
                    - Increasing trend in the red line (finding better solutions)
                    - Convergence towards the end (stabilizing on good parameters)
                    - Higher density of points near the best scores (focusing on promising regions)
                """)
            else:
                st.info("Optimisation history plot not available")
            
            # Add parameter analysis section
            st.subheader("Parameter Analysis")
            
            st.markdown(""" 
                        #### Parameter Importance
                        
                        Visualises the importance of each parameter in the model in the prediction of the target metric
                        """)

            if results['info']['optimisation_plots']['param_importances_fig'] is not None:
                st.plotly_chart(results['info']['optimisation_plots']['param_importances_fig'], config={'responsive': True})

            # Get parameter data
            if ('params' in results['info']['optimisation_history'] and 
                results['info']['optimisation_history']['params'] and
                'values' in results['info']['optimisation_history']):
                
                params = results['info']['optimisation_history']['params']
                scores = results['info']['optimisation_history']['values']
                best_params = results['info']['best_params']
                trial_numbers = list(range(len(scores)))
                
                # Add performance filter
                st.markdown("### 🎯 Filter Trials by Performance")
                min_score = min(scores)
                max_score = max(scores)
                score_range = st.slider(
                    "Select score range to analyse:",
                    min_value=float(min_score),
                    max_value=float(max_score),
                    value=(float(min_score), float(max_score)),
                    format="%.4f"
                )
                
                # Filter trials based on score range
                filtered_indices = [i for i, score in enumerate(scores) 
                                    if score_range[0] <= score <= score_range[1]]
                filtered_params = [params[i] for i in filtered_indices]
                filtered_scores = [scores[i] for i in filtered_indices]
                filtered_trials = [trial_numbers[i] for i in filtered_indices]
                
                st.markdown("""
                    This analysis shows how different parameter values affected model performance during optimisation.
                    - Use the score range slider above to focus on specific performance levels
                    - Hover over points for detailed information
                    - Click on parameters in the legend to show/hide them
                """)
                
                # Create tabs for each parameter
                param_tabs = st.tabs([f"📊 {param}" for param in best_params.keys()])
                
                # Process each parameter in its own tab
                for tab, (param_name, best_value) in zip(param_tabs, best_params.items()):
                    with tab:
                        param_values = [p.get(param_name) for p in filtered_params]
                        
                        if param_values:
                            # Create columns for metrics and visualization
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.markdown(f"### {param_name}")
                                # Show parameter statistics
                                if isinstance(best_value, (int, float)):
                                    st.metric("Best Value", f"{best_value:.4f}" if isinstance(best_value, float) else str(best_value))
                                else:
                                    st.metric("Best Value", str(best_value))
                                
                                st.metric("Values Tested", len(param_values))
                                
                                # Try converting to numeric for correlation analysis
                                try:
                                    numeric_values = [float(v) for v in param_values]
                                    correlation = np.corrcoef(numeric_values, filtered_scores)[0, 1]
                                    
                                    # Determine correlation strength and direction
                                    correlation_strength = abs(correlation)
                                    if correlation_strength < 0.2:
                                        strength_desc = "very weak"
                                        correlation_color = "gray"
                                    elif correlation_strength < 0.4:
                                        strength_desc = "weak"
                                        correlation_color = "blue"
                                    elif correlation_strength < 0.6:
                                        strength_desc = "moderate"
                                        correlation_color = "orange"
                                    else:
                                        strength_desc = "strong"
                                        correlation_color = "red"
                                    
                                    direction = "positive" if correlation > 0 else "negative"
                                    
                                    # Create correlation explanation
                                    correlation_text = f"""
                                        **Correlation with Performance**: :{correlation_color}[{correlation:.3f}]
                                        
                                        This indicates a {strength_desc} {direction} relationship between this parameter and model performance:
                                        
                                        - **Direction**: {direction.capitalize()}
                                            - {'Higher parameter values tend to improve performance' if correlation > 0 else 'Lower parameter values tend to improve performance'}
                                        
                                        - **Strength**: {strength_desc.capitalize()}
                                            - {'This parameter has a significant impact on performance' if correlation_strength >= 0.4 else 'This parameter has limited impact on performance'}
                                        
                                        *Note: Correlation ranges from -1 to 1, where:*
                                        - 1.0 = Perfect positive correlation
                                        - 0.0 = No correlation
                                        - -1.0 = Perfect negative correlation
                                    """
                                    st.markdown(correlation_text)
                                    
                                    # Add trend analysis
                                    if len(numeric_values) > 1:
                                        trend = np.polyfit(filtered_trials, numeric_values, 1)[0]
                                        trend_direction = "increasing" if trend > 0 else "decreasing"
                                        st.markdown(f"""
                                            **Value Trend**: Parameter values were {trend_direction} over trials
                                            - This suggests the optimisation process {'favored higher' if trend > 0 else 'favored lower'} values
                                        """)
                                except (ValueError, TypeError):
                                    st.markdown("Correlation analysis not applicable for non-numeric values")
                            
                            with col2:
                                # Special handling for different parameter types
                                if param_name == "hidden_layer_sizes":
                                    num_archs, avg_perf = analyse_network_architecture(param_values, filtered_scores, best_value)
                                elif any(isinstance(v, (int, float)) and str(v).lower() != 'auto' for v in param_values):
                                    fig, auto_mean, num_mean, total_configs = analyse_mixed_parameter(
                                        param_values, filtered_scores, best_value
                                    )
                                    st.plotly_chart(fig, config={'responsive': True})
                                    
                                    if auto_mean is not None or num_mean is not None:
                                        auto_info = f"- 'auto' setting mean score: {auto_mean:.4f}" if auto_mean is not None else ""
                                        num_info = f"- Numeric values mean score: {num_mean:.4f}" if num_mean is not None else ""
                                        
                                        st.info(f"""
                                            🔄 **Mixed Parameter Analysis**:
                                            {auto_info}
                                            {num_info}
                                            - Best value: **{best_value}**
                                            - Total configurations tested: {total_configs}
                                        """)
                                else:
                                    # Try numeric visualization first
                                    try:
                                        numeric_values = [float(v) for v in param_values]
                                        is_numeric = True
                                        unique_values = len(set(numeric_values))
                                        
                                        # Show parameter evolution for numeric parameters
                                        st.markdown("##### Parameter Evolution")
                                        evolution_fig = go.Figure()
                                        
                                        # Find the trial with the best score
                                        best_score_idx = np.argmax(filtered_scores)
                                        best_trial_value = numeric_values[best_score_idx]
                                        
                                        # Add value trace with highlighted best point
                                        evolution_fig.add_trace(go.Scatter(
                                            x=filtered_trials,
                                            y=numeric_values,
                                            mode='markers',
                                            name='Value',
                                            marker=dict(
                                                size=10,
                                                color=filtered_scores,
                                                colorscale='Viridis',
                                                showscale=True,
                                                colorbar=dict(title='Score')
                                            ),
                                            hovertemplate="Trial: %{x}<br>Value: %{y}<br>Score: %{marker.color:.4f}<extra></extra>"
                                        ))
                                        
                                        # Add trend line
                                        z = np.polyfit(filtered_trials, numeric_values, 1)
                                        p = np.poly1d(z)
                                        evolution_fig.add_trace(go.Scatter(
                                            x=filtered_trials,
                                            y=p(filtered_trials),
                                            mode='lines',
                                            name='Trend',
                                            line=dict(color='red', dash='dash')
                                        ))
                                        
                                        # Highlight the best point
                                        evolution_fig.add_trace(go.Scatter(
                                            x=[filtered_trials[best_score_idx]],
                                            y=[best_trial_value],
                                            mode='markers',
                                            name='Best Score',
                                            marker=dict(
                                                symbol='star',
                                                size=20,
                                                color='#2ecc71',
                                                line=dict(color='black', width=2)
                                            ),
                                            hovertemplate="Best Trial<br>Value: %{y}<br>Score: " + f"{filtered_scores[best_score_idx]:.4f}" + "<extra></extra>"
                                        ))
                                        
                                        # Add annotation for the best point
                                        evolution_fig.add_annotation(
                                            x=filtered_trials[best_score_idx],
                                            y=best_trial_value,
                                            text="Best Score ★",
                                            showarrow=True,
                                            arrowhead=1,
                                            ax=0,
                                            ay=-40,
                                            font=dict(size=12, color='black', weight='bold'),
                                            bgcolor='#2ecc71',
                                            bordercolor='black',
                                            borderwidth=2,
                                            borderpad=4,
                                            opacity=0.8
                                        )
                                        
                                        evolution_fig.update_layout(
                                            title="Parameter Value Evolution",
                                            xaxis_title="Trial Number",
                                            yaxis_title="Parameter Value",
                                            showlegend=True,
                                            height=300
                                        )
                                        
                                        st.plotly_chart(evolution_fig, config={'responsive': True})
                                        
                                    except (ValueError, TypeError):
                                        # Fallback to categorical visualization
                                        is_numeric = False
                                        unique_values = len(set(str(v) for v in param_values))
                                        
                                        # Create categorical visualization
                                        fig = go.Figure()
                                        
                                        # Calculate performance stats for each category
                                        value_scores = {}
                                        for val, score in zip(param_values, filtered_scores):
                                            if val not in value_scores:
                                                value_scores[val] = []
                                            value_scores[val].append(score)
                                        
                                        # Find the best performing category
                                        best_cat_score = float('-inf')
                                        best_category = None
                                        for val, scores_list in value_scores.items():
                                            mean_score = np.mean(scores_list)
                                            if mean_score > best_cat_score:
                                                best_cat_score = mean_score
                                                best_category = val
                                        
                                        # Create box plot with highlighted best category
                                        for val in value_scores:
                                            is_best = val == best_value
                                            fig.add_trace(go.Box(
                                                y=value_scores[val],
                                                name=str(val),
                                                boxpoints='all',
                                                jitter=0.3,
                                                pointpos=-1.8,
                                                marker=dict(
                                                    color='#2ecc71' if is_best else 'lightblue',
                                                    line=dict(color='black', width=2) if is_best else None
                                                ),
                                                hovertemplate="Value: %{x}<br>Score: %{y:.4f}<extra></extra>"
                                            ))
                                        
                                        # Add annotation for the best category
                                        fig.add_annotation(
                                            x=str(best_value),
                                            y=np.mean(value_scores[best_value]),
                                            text="Best Value ★",
                                            showarrow=True,
                                            arrowhead=1,
                                            ax=0,
                                            ay=-40,
                                            font=dict(size=12, color='black', weight='bold'),
                                            bgcolor='#2ecc71',
                                            bordercolor='black',
                                            borderwidth=2,
                                            borderpad=4,
                                            opacity=0.8
                                        )
                                        
                                        fig.update_layout(
                                            title=f'{param_name} Performance by Value',
                                            yaxis_title='Score',
                                            showlegend=False,
                                            height=400
                                        )
                                        
                                        st.plotly_chart(fig, config={'responsive': True})
            else:
                st.info("Detailed optimisation metrics not available for this run.")
        
    with st.expander("Stability Analysis", expanded=True):
        st.subheader("Stability Analysis")
        st.markdown("""
            Visualises the stability of the model during the optimisation process.
            """)
        # Add stability analysis from Optuna results
        if 'stability_analysis' in results['info']:
            stability = results['info']['stability_analysis']
            
            # Display stability level and score
            st.markdown(f"### Model Stability Level: {stability['level']}")
            
            # Display stability score gauge
            st.plotly_chart(stability['plots']['gauge'])
            
            # Display performance variation and fold comparison
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(stability['plots']['variation'])
                st.info("""
                    ### Performance Variation
                    - Blue line shows min, mean, and max scores
                    - Red point with error bars shows standard deviation
                    - Shorter error bars indicate more stable performance
                """)
            
            with col2:
                st.plotly_chart(stability['plots']['fold_comparison'])
                st.info("""
                    ### Fold Performance
                    - Each bar represents one cross-validation fold
                    - Red dashed line shows mean score
                    - More consistent bar heights indicate stable performance
                """)
            
            # Show recommendations if any
            if 'recommendations' in stability:
                st.markdown("### 💡 Stability Recommendations")
                for rec in stability['recommendations']:
                    st.markdown(f"- {rec}")
        
        # Continue with the rest of the visualizations...
    
    # Add optimisation insights
    with st.expander("💡 Optimisation Insights", expanded=False):
            st.markdown("""
            ### Key Findings from the Optimisation Process
            
            #### 🎯 Best Parameter Values
            The optimisation process identified these key patterns:
            
            1. **High-Impact Parameters**
                - Parameters with strongest influence on model performance
                - Look for clear patterns in parameter plots
                - Focus on parameters with consistent optimal ranges
            
            2. **Parameter Interactions**
                - Some parameters work best in combination
                - Look for clusters of high-performing points
                - Pay attention to parameter ranges with consistently good scores
            
            3. **Stability vs Performance**
                - Best values balance high performance with stability
                - Look for clusters of good performance (dark regions)
                - Avoid isolated high scores that might be unstable
            
            #### 📊 How to Read the Visualizations
            
            1. **Parameter Evolution Plots**
                - **Colored Points**: Each trial, colored by performance (darker = better)
                - **★ Star Marker**: Best found value
                - **Score Scale**: Shows performance range from worst to best
            
            2. **Convergence Plot**
                - **Red Dashed Line**: Best score found so far
                - **Colored Points**: Individual trial scores
                - **Color Gradient**: Performance improvement over time
            
            #### 🚀 Optimisation Progress
            
            1. **Search Patterns**
                - Early trials: Wide exploration of parameter space
                - Middle trials: Focusing on promising regions
                - Later trials: Fine-tuning best parameters
            
            2. **Confidence Indicators**
                - **High Confidence**: Dense cluster of good scores around best value
                - **Medium Confidence**: Scattered good scores across range
                - **Low Confidence**: Isolated good scores or boundary values
            
            #### 💡 Recommendations
            
            1. **For Current Results**
                - Use parameters from regions with consistently good scores
                - Prefer values from stable, high-performing clusters
                - Be cautious of isolated high-performing points
            
            2. **For Future Optimisation**
                - Expand ranges for parameters hitting boundaries
                - Add more trials if performance is still improving
                - Focus on ranges showing promising clusters
        """)
    
    # Update selected model type to the Optuna best model type
    if not hasattr(st.session_state, 'selected_model_type'):
        # Ensure the model state is properly set for Optuna results
        st.session_state.builder.model.update({
            "best_params": results['info']['best_params'],
            "cv_metrics": results['info']['cv_metrics'],
            "active_params": results['info']['best_params'],  # Set active params to best params
            "best_model": st.session_state.builder.model['model'],  # Use the existing model
            "active_model": st.session_state.builder.model['model'],  # Use the existing model
            "optimisation_method": "optuna"
        })
        
        with st.spinner("Selecting optimized model..."):
            # Set the model state directly since we already have the best model
            st.session_state.selected_model_type = "mean_score"
            st.session_state.previous_model_selection = "mean_score"
            st.session_state.selected_model_stability = results['info']['stability_analysis']
            
            # CRITICAL FIX: Ensure the Optuna model is properly configured
            if "best_params" in st.session_state.builder.model:
                best_params = st.session_state.builder.model["best_params"]
                # Use safe method to handle CatBoost models
                fitted_model = safe_set_params_and_fit(
                    st.session_state.builder.model["model"],
                    best_params,
                    st.session_state.builder.X_train,
                    st.session_state.builder.y_train
                )
                st.session_state.builder.model["model"] = fitted_model
                # CRITICAL: Set the active_model reference for evaluation
                st.session_state.builder.model["active_model"] = fitted_model
                st.session_state.builder.model["active_params"] = best_params
                st.session_state.builder.model["selection_type"] = "mean_score"
                # CRITICAL: Reset calibration state when new model is active
                TrainingStateManager.reset_calibration_state()
            
            st.success("✅ Optimized model selected successfully!")

# Define callback function to handle model selection
def on_model_change(results, model_options):
    if 'selected_model_option' in st.session_state:
        selected_type = st.session_state.selected_model_option
        # Only trigger if selection actually changed
        if not hasattr(st.session_state, 'previous_model_selection') or st.session_state.previous_model_selection != selected_type:
            # Store the previous selection for comparison
            st.session_state.previous_model_selection = selected_type
            
            with st.spinner(f"Selecting {model_options[selected_type]}..."):
                result = st.session_state.builder.select_final_model(selected_type)
                if result["success"]:
                    # Check if this is a different model than previously selected
                    is_new_selection = not hasattr(st.session_state, 'selected_model_type') or st.session_state.selected_model_type != selected_type
                    st.session_state.selected_model_type = selected_type
                    
                    # CRITICAL FIX: Actually retrain the main model with the selected parameters
                    if selected_type == "adjusted_score" and "adjusted_params" in st.session_state.builder.model:
                        # Set the main model to use adjusted parameters
                        adjusted_params = st.session_state.builder.model["adjusted_params"]
                        # Use safe method to handle CatBoost models
                        fitted_model = safe_set_params_and_fit(
                            st.session_state.builder.model["model"],
                            adjusted_params,
                            st.session_state.builder.X_train,
                            st.session_state.builder.y_train
                        )
                        st.session_state.builder.model["model"] = fitted_model
                        # CRITICAL: Set the active_model reference for evaluation
                        st.session_state.builder.model["active_model"] = fitted_model
                        st.session_state.builder.model["active_params"] = adjusted_params
                        st.session_state.builder.model["selection_type"] = "adjusted_score"
                        # CRITICAL: Reset calibration state when new model is active
                        TrainingStateManager.reset_calibration_state()

                    elif selected_type == "mean_score" and "best_params" in st.session_state.builder.model:
                        # Set the main model to use best parameters
                        best_params = st.session_state.builder.model["best_params"]
                        # Use safe method to handle CatBoost models
                        fitted_model = safe_set_params_and_fit(
                            st.session_state.builder.model["model"],
                            best_params,
                            st.session_state.builder.X_train,
                            st.session_state.builder.y_train
                        )
                        st.session_state.builder.model["model"] = fitted_model
                        # CRITICAL: Set the active_model reference for evaluation
                        st.session_state.builder.model["active_model"] = fitted_model
                        st.session_state.builder.model["active_params"] = best_params
                        st.session_state.builder.model["selection_type"] = "mean_score"
                    
                    # Store the appropriate stability analysis
                    if selected_type == "mean_score":
                        st.session_state.selected_model_stability = results['info']['stability_analysis']
                    else:  # adjusted_score
                        if 'adjusted_stability_analysis' in results['info']:
                            stability = results['info']['adjusted_stability_analysis']
                            st.session_state.selected_model_stability = stability
                        else:
                            # Create adjusted stability analysis if it doesn't exist
                            if 'adjusted_cv_metrics' in st.session_state.builder.model:
                                adjusted_metrics = st.session_state.builder.model['adjusted_cv_metrics']
                                
                                # Create a copy of the original stability analysis
                                import copy
                                stability = copy.deepcopy(results['info']['stability_analysis'])
                                
                                # Update stability level based on adjusted metrics
                                std_score = adjusted_metrics.get('std_score', 0)
                                if std_score > 0.1:
                                    stability['level'] = "High variability"
                                elif std_score > 0.05:
                                    stability['level'] = "Moderate stability"
                                else:
                                    stability['level'] = "High stability"
                                
                                # Update stability score
                                stability['score'] = 1 - std_score
                                
                                # Create updated plots for stability analysis
                                if 'fold_scores' in adjusted_metrics:
                                    fold_scores = adjusted_metrics['fold_scores']
                                    
                                    # Create new variation plot
                                    import plotly.graph_objects as go
                                    
                                    # Performance variation figure
                                    variation_fig = go.Figure()
                                    
                                    # Add min-mean-max line
                                    variation_fig.add_trace(go.Scatter(
                                        x=['Min', 'Mean', 'Max'],
                                        y=[adjusted_metrics['min_score'], adjusted_metrics['mean_score'], adjusted_metrics['max_score']],
                                        mode='lines+markers',
                                        name='Score Range',
                                        marker=dict(size=10)
                                    ))
                                    
                                    # Add error bars using actual std
                                    variation_fig.add_trace(go.Scatter(
                                        x=['Mean'],
                                        y=[adjusted_metrics['mean_score']],
                                        error_y=dict(
                                            type='data',
                                            array=[adjusted_metrics['std_score']],
                                            visible=True
                                        ),
                                        mode='markers',
                                        name='Standard Deviation',
                                        marker=dict(size=12, color='red')
                                    ))
                                    
                                    variation_fig.update_layout(
                                        title='Performance Variation',
                                        yaxis_title='Score',
                                        showlegend=True,
                                        height=300
                                    )
                                    
                                    # Create fold comparison chart
                                    fold_fig = go.Figure(data=go.Bar(
                                        x=[f'Fold {i+1}' for i in range(len(fold_scores))],
                                        y=fold_scores,
                                        marker_color='lightblue'
                                    ))
                                    
                                    fold_fig.add_hline(
                                        y=adjusted_metrics['mean_score'],
                                        line_dash="dash",
                                        line_color="red",
                                        annotation_text=f"Mean Score: {adjusted_metrics['mean_score']:.3f}"
                                    )
                                    
                                    fold_fig.update_layout(
                                        title='Performance Across Folds',
                                        xaxis_title='Fold',
                                        yaxis_title='Score',
                                        showlegend=False,
                                        height=400
                                    )
                                    
                                    # Create gauge chart with standardized size
                                    gauge_fig = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=1 - adjusted_metrics["std_score"],
                                        title={'text': "Model Stability Score"},
                                        gauge={
                                            'axis': {'range': [0, 1]},
                                            'steps': [
                                                {'range': [0, 0.9], 'color': "lightgray"},
                                                {'range': [0.9, 0.95], 'color': "yellow"},
                                                {'range': [0.95, 1], 'color': "lightgreen"}
                                            ],
                                            'threshold': {
                                                'line': {'color': "red", 'width': 4},
                                                'thickness': 0.75,
                                                'value': 1 - adjusted_metrics["std_score"]
                                            }
                                        }
                                    ))
                                    
                                    # Standardize gauge size
                                    gauge_fig.update_layout(
                                        height=350,
                                        margin=dict(l=30, r=30, t=50, b=30)
                                    )
                                    
                                    # Update the plots in the stability analysis
                                    stability['plots'] = {
                                        'gauge': gauge_fig,
                                        'variation': variation_fig,
                                        'fold_comparison': fold_fig
                                    }
                                
                                # Store the adjusted stability analysis
                                results['info']['adjusted_stability_analysis'] = stability
                                st.session_state.selected_model_stability = stability
                            else:
                                # Fallback to original if no adjusted metrics available
                                stability = results['info']['stability_analysis']
                                st.session_state.selected_model_stability = stability
                    
                    # Get the appropriate score value based on selected model type
                    score_value = result["info"]["mean_score"] if selected_type == "mean_score" else result["info"]["adjusted_score"]
                    score_type = "Mean Score" if selected_type == "mean_score" else "Adjusted Score"
                    
                    # Display notification if model was changed
                    if is_new_selection:
                        st.success(f"✨ **Model changed!** Now using model optimized for {selected_type.replace('_', ' ')} ({score_type}: {score_value:.4f})")
                    
                    # Log selection
                    st.session_state.logger.log_user_action(
                        "Model Selection",
                        {
                            "selection_type": selected_type,
                            "mean_score": result["info"]["mean_score"],
                            "std_score": result["info"]["std_score"],
                            "adjusted_score": result["info"]["adjusted_score"],
                            "is_new_selection": is_new_selection
                        }
                    )
                    
                    # Force a rerun to update all visualisations
                    st.rerun()
                else:
                    st.error(result["message"])

def display_search_training_results(results):
    import plotly.graph_objects as go
    # Display all parameter combinations and their performances
    with st.expander("📊 All Model Configurations", expanded=True):
        st.subheader("Parameter Combinations Tested")
        
        # Get the data from results
        all_params = results['info']['all_results']['params_tested']
        mean_scores = results['info']['all_results']['mean_test_scores']
        std_scores = results['info']['all_results']['std_test_scores']
        
        # Create a DataFrame with all tested configurations
        data = []
        best_params = results['info']['best_params']
        
        # Define alpha coefficient for adjusted score (balance between performance and stability)
        alpha = 1.0
        
        # Calculate adjusted scores and track the best one
        best_adjusted_score = -float('inf')
        best_adjusted_index = None
        
        for i, (params, mean_score, std_score) in enumerate(zip(all_params, mean_scores, std_scores)):
            # Check if this is the best model by standard metrics
            is_best = True
            for key, value in best_params.items():
                if key in params and params[key] != value:
                    is_best = False
                    break
            
            # Calculate adjusted score that balances performance and stability
            adjusted_score = mean_score - (alpha * std_score)
            
            # Track best adjusted score
            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_adjusted_index = i
                
            # Create a row with all parameters and scores
            row = {
                'Iteration': i+1, 
                'Mean CV Score': mean_score, 
                'Std Dev': std_score, 
                'Adjusted Score': adjusted_score,
                'Is Best': is_best,
                'Is Best Adjusted': False  # Will be updated later
            }
            
            for key, value in params.items():
                # Convert tuple or complex objects to strings for display
                if isinstance(value, tuple):
                    row[key] = str(value)
                else:
                    row[key] = value
            data.append(row)
        
        # Mark the model with best adjusted score
        if best_adjusted_index is not None:
            data[best_adjusted_index]['Is Best Adjusted'] = True
        
        # Create DataFrame and allow sorting by different metrics
        df = pd.DataFrame(data)
        
        # Add a widget to select sorting criterion
        sort_by = st.radio(
            "Sort configurations by:",
            ["Mean CV Score", "Adjusted Score", "Std Dev"],
            horizontal=True
        )
        
        # Sort the DataFrame based on user selection
        if sort_by == "Std Dev":
            df = df.sort_values('Std Dev', ascending=True)  # Lower std dev is better
        else:
            df = df.sort_values(sort_by, ascending=False)  # Higher scores are better
        
        # Format the score columns for better readability
        df['Mean CV Score'] = df['Mean CV Score'].map('{:.4f}'.format)
        df['Std Dev'] = df['Std Dev'].map('{:.4f}'.format)
        df['Adjusted Score'] = df['Adjusted Score'].map('{:.4f}'.format)
        
        # Add column for model status that shows both types of "best" models
        df['Model Status'] = ''
        for idx, row in df.iterrows():
            status = []
            if row['Is Best']:
                status.append('✅ Best by Mean Score')
            if row['Is Best Adjusted']:
                status.append('🌟 Best by Adjusted Score')
            
            # If both are true, create a special combined status
            if row['Is Best'] and row['Is Best Adjusted']:
                df.at[idx, 'Model Status'] = '✅🌟 Best Overall (Mean & Adjusted)'
            else:
                df.at[idx, 'Model Status'] = ' | '.join(status)
        
        # Store the DataFrame in session state for model difference detection
        if 'data' not in st.session_state:
            st.session_state.data = {}
        st.session_state.data['model_config_table'] = df
        
        # Remove the flag columns
        display_df = df.drop(columns=['Is Best', 'Is Best Adjusted'])
        
        # Show the table with all configurations
        st.write("The table below shows all parameter combinations tested during the tuning process.")
        st.dataframe(display_df, width='stretch')
        
        st.info("""
            **Understanding the Results Table:**
            - **Mean CV Score**: The average performance across all cross-validation folds
            - **Std Dev**: Standard deviation of scores (lower means more consistent performance)
            - **Adjusted Score**: Balance of performance and stability (Mean Score - Standard Deviation)
            - **Model Status**: 
                - ✅ Best by Mean Score: Model with highest average performance
                - 🌟 Best by Adjusted Score: Model with best balance of performance and stability
                - ✅🌟 Best Overall: Model is best by both metrics (optimal performance and stability)
        """)

    # Explain the meaning of adjusted score outside of expanders to avoid nesting issues
    with st.expander("📚 Understanding the Adjusted Score", expanded=False):
        st.markdown("""
        ### Balancing Performance and Stability
        
        The **Adjusted Score** helps you identify models that balance high performance with good stability using this formula:
        
        ```
        Adjusted Score = Mean CV Score - Standard Deviation)
        ```
        
        Where:
        - **Mean CV Score**: Average performance across folds (higher is better)
        - **Standard Deviation**: Measure of performance variability (lower is better)
        
        
        ### Why Use Adjusted Score?
        
        1. **More reliable model selection**: Prevents choosing models that perform well on average but are highly inconsistent
        2. **Better real-world performance**: Models with lower variability generally perform more consistently on new data
        3. **Risk mitigation**: Reduces the chance of catastrophic performance drops in production
        
        ### Example Comparison
        
        | Model | Mean Score | Std Dev | Adjusted Score | Better Choice? |
        |-------|------------|---------|---------------|----------------|
        | A     | 0.90       | 0.15    | 0.75          | No             |
        | B     | 0.85       | 0.03    | 0.82          | Yes ✓          |
        
        While Model A has a higher mean score, Model B's much lower standard deviation makes it more reliable overall, resulting in a higher adjusted score.
        
        ### When to Use Each Metric
        
        - **Mean CV Score**: When maximum average performance is the only priority
        - **Adjusted Score**: When you want a balance of performance and stability (recommended for most cases)
        """)

    # Add model selection section - MOVED HERE
    st.header("🧩 Select Model to Use")
    
    # Initialize stability data to use the default stability analysis
    stability_data = results['info']['stability_analysis']
    
    # Check if we have different models
    results_for_model_selection = results
    
    # Initialize variables that might be used in debug section
    params_are_different = False
    models_are_different = False 
    metrics_are_different = False
    significant_param_difference = False
    different_params = []
    
    # DIRECT SOLUTION: Check if we have separate models based on the flag from the Builder
    # The most reliable way is to check if best_index and best_adjusted_index are different
    has_different_models = False
    
    if 'info' in results_for_model_selection and 'all_results' in results_for_model_selection['info']:
        all_results = results_for_model_selection['info']['all_results']
        
        # Check if the 'same_model' flag is directly available
        if 'is_same_model' in results_for_model_selection['info']:
            # If the flag exists, use it directly
            has_different_models = not results_for_model_selection['info']['is_same_model']
        
        # If 'best_model_index' and 'best_adjusted_model_index' are available, use those
        elif 'best_model_index' in all_results and 'best_adjusted_model_index' in all_results:
            best_idx = all_results['best_model_index']
            adj_idx = all_results['best_adjusted_model_index']
            
            # If they have different indices, they are different models
            has_different_models = (best_idx != adj_idx)
        
        # FALLBACK: Direct parameter comparison if necessary
        elif "best_params" in st.session_state.builder.model and "adjusted_params" in st.session_state.builder.model:
            # Examine the parameter dictionaries
            best_params = st.session_state.builder.model["best_params"]
            adj_params = st.session_state.builder.model["adjusted_params"]
            
            # Use string conversion to reliably compare the parameter dictionaries
            params_are_different = str(best_params) != str(adj_params)
            has_different_models = params_are_different
            
            if params_are_different:
                # Find which parameters differ for debugging
                for key in set(best_params.keys()) | set(adj_params.keys()):
                    if key not in best_params or key not in adj_params:
                        different_params.append(key)
                    elif str(best_params[key]) != str(adj_params[key]):
                        different_params.append(key)
    
    # FINAL CHECK: Look for markers in the training data table itself
    if not has_different_models and 'info' in results_for_model_selection and 'all_results' in results_for_model_selection['info']:
        # Check if we have a DataFrame with model flags
        if 'data' in st.session_state and 'model_config_table' in st.session_state['data']:
            model_table = st.session_state['data']['model_config_table']
            
            # Check if there are rows marked as best and best adjusted
            best_mean_rows = model_table[model_table['Is Best'] == True]
            best_adjusted_rows = model_table[model_table['Is Best Adjusted'] == True]
            
            # If there are different numbers of rows, they must be different models
            if len(best_mean_rows) > 0 and len(best_adjusted_rows) > 0:
                # Check if any row is both best and best adjusted
                both_best = model_table[(model_table['Is Best'] == True) & 
                                    (model_table['Is Best Adjusted'] == True)]
                
                # If no row is both, then they are different models
                has_different_models = len(both_best) == 0
    
    if has_different_models:
        st.info("""
            **Two different optimal models were identified:**
            
            1. **Best by Mean Score**: Highest average performance across folds
            2. **Best by Adjusted Score**: Best balance of performance and stability
            
            You need to select which one to use for the rest of your ML pipeline.
        """)
        
        # Show comparison table of the two models - with error handling
        if "cv_metrics" not in st.session_state.builder.model or "adjusted_cv_metrics" not in st.session_state.builder.model:
            st.error("""
                Training metrics are not available. This might happen when navigating between pages.
                Please restart the training process or return to the previous step.
            """)
            return
        
        mean_metrics = st.session_state.builder.model["cv_metrics"]
        adjusted_metrics = st.session_state.builder.model["adjusted_cv_metrics"]
        
        # Add the adjusted score to mean metrics
        alpha = 1.0
        mean_adjusted_score = mean_metrics["mean_score"] - (alpha * mean_metrics["std_score"])
        
        # Create comparison table
        comparison_data = {
            "Model": ["Best by Mean Score", "Best by Adjusted Score"],
            "Mean Score": [f"{mean_metrics['mean_score']:.4f}", f"{adjusted_metrics['mean_score']:.4f}"],
            "Std Dev": [f"{mean_metrics['std_score']:.4f}", f"{adjusted_metrics['std_score']:.4f}"],
            "Adjusted Score": [f"{mean_adjusted_score:.4f}", f"{adjusted_metrics['adjusted_score']:.4f}"]
        }
        
        # Add parameter differences
        if (st.session_state.builder.model.get("best_params") and 
            st.session_state.builder.model.get("adjusted_params") and
            st.session_state.builder.model["best_params"] != st.session_state.builder.model["adjusted_params"]):
            st.write("#### Key Hyperparameter Differences")
            
            # Create columns for each model
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Best by Mean Score:**")
                for param, value in st.session_state.builder.model["best_params"].items():
                    adj_value = st.session_state.builder.model["adjusted_params"].get(param)
                    if value != adj_value:
                        st.write(f"• {param}: **{value}**")
            
            with col2:
                st.write("**Best by Adjusted Score:**")
                for param, value in st.session_state.builder.model["adjusted_params"].items():
                    mean_value = st.session_state.builder.model["best_params"].get(param)
                    if value != mean_value:
                        st.write(f"• {param}: **{value}**")
        
        # Convert to DataFrame for display
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        # Let user select model with dropdown instead of buttons
        st.write("### Select Model to Use")
        
        # Create options for the dropdown
        model_options = {
            "mean_score": "Best Mean Score Model (Maximum Performance)",
            "adjusted_score": "Best Adjusted Score Model (Balance of Performance & Stability)"
        }
        
        # Check if there's already a selected model
        default_option = st.session_state.selected_model_type if hasattr(st.session_state, 'selected_model_type') else "mean_score"
                            
        # Initialize model selection on first load if needed
        if not hasattr(st.session_state, 'selected_model_type'):
            # Auto-select the default model on first load
            with st.spinner(f"Selecting {model_options['mean_score']}..."):
                result = st.session_state.builder.select_final_model("mean_score")
                if result["success"]:
                    st.session_state.selected_model_type = "mean_score"
                    st.session_state.previous_model_selection = "mean_score"
                    st.session_state.selected_model_stability = results['info']['stability_analysis']
                    
                    # CRITICAL FIX: Ensure the default model is properly configured
                    if "best_params" in st.session_state.builder.model:
                        best_params = st.session_state.builder.model["best_params"]
                        # Use safe method to handle CatBoost models
                        fitted_model = safe_set_params_and_fit(
                            st.session_state.builder.model["model"],
                            best_params,
                            st.session_state.builder.X_train,
                            st.session_state.builder.y_train
                        )
                        st.session_state.builder.model["model"] = fitted_model
                        # CRITICAL: Set the active_model reference for evaluation
                        st.session_state.builder.model["active_model"] = fitted_model
                        st.session_state.builder.model["active_params"] = best_params
                        st.session_state.builder.model["selection_type"] = "mean_score"
        
        # Create the selectbox with callback
        st.selectbox(
            "Choose which model to use:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=list(model_options.keys()).index(default_option),
            key="selected_model_option",
            on_change=on_model_change(results, model_options)
        )
        
        # Add help text explaining the automatic selection
        st.caption("The model will be applied automatically when you change your selection.")

    else:
        # Models are the same
        st.success("""
            ✅🌟 **Great news!** The model with the highest mean score is also the model with the best adjusted score.
            
            This means you've found an optimal model that provides both:
            - The highest average performance
            - The best balance of performance and stability
            
            This model will be used for the rest of your ML pipeline.
        """)
        
        # Automatically select the model since there's only one option
        if not hasattr(st.session_state, 'selected_model_type') or not hasattr(st.session_state, 'selected_model_stability'):
            with st.spinner("Selecting optimal model..."):
                result = st.session_state.builder.select_final_model("mean_score")
                if result["success"]:
                    # Update all necessary session state variables
                    st.session_state.selected_model_type = "mean_score"
                    st.session_state.previous_model_selection = "mean_score"
                    st.session_state.selected_model_stability = results['info']['stability_analysis']
                    
                    # CRITICAL FIX: Ensure the single optimal model is properly configured
                    if "best_params" in st.session_state.builder.model:
                        best_params = st.session_state.builder.model["best_params"]
                        # Use safe method to handle CatBoost models
                        fitted_model = safe_set_params_and_fit(
                            st.session_state.builder.model["model"],
                            best_params,
                            st.session_state.builder.X_train,
                            st.session_state.builder.y_train
                        )
                        st.session_state.builder.model["model"] = fitted_model
                        # CRITICAL: Set the active_model reference for evaluation
                        st.session_state.builder.model["active_model"] = fitted_model
                        st.session_state.builder.model["active_params"] = best_params
                        st.session_state.builder.model["selection_type"] = "mean_score"
                    
                    # Log selection
                    st.session_state.logger.log_user_action(
                        "Model Selection",
                        {
                            "selection_type": "mean_score",
                            "mean_score": result["info"]["mean_score"],
                            "std_score": result["info"]["std_score"],
                            "adjusted_score": result["info"]["adjusted_score"],
                            "note": "Only one optimal model found"
                        }
                    )
                    
                    # Force a rerun to update all visualisations
                    st.rerun()
                else:
                    st.error(result["message"])

    # Display CV analysis plots with explanations
    #with st.expander("📈 Cross-validation Analysis", expanded=True):
    #    st.plotly_chart(results['info']['cv_plots']['distribution'])
    #    st.info("""
    #        **Score Distribution Plot:**
    #        - Shows how model performance scores are distributed across CV folds
    #        - Bell-shaped curve indicates consistent performance
    #        - Wide spread suggests high variability
    #    """)
        
        

    # Display stability analysis
    #with st.expander("🔍 Model Stability Analysis", expanded=True):
    # Determine which stability data to use - prioritize session state data first
    if hasattr(st.session_state, 'selected_model_stability'):
        # Use the pre-selected stability analysis stored in session state
        stability = st.session_state.selected_model_stability
        
        # Determine which model is being displayed
        if hasattr(st.session_state, 'selected_model_type'):
            if st.session_state.selected_model_type == "mean_score":
                model_display_name = "Best Mean Score Model (Maximum Performance)"
            else:  # adjusted_score
                model_display_name = "Best Adjusted Score Model (Balance of Performance & Stability)"
        else:
            model_display_name = "Default Model (No Selection Made)"
    # Fallback to determine which stability data to use if not in session state
    elif hasattr(st.session_state, 'selected_model_type'):
        if st.session_state.selected_model_type == "mean_score":
            # For mean score model, use the original stability analysis
            stability = results['info']['stability_analysis']
            # Store it in session state for persistence
            st.session_state.selected_model_stability = stability
            model_display_name = "Best Mean Score Model (Maximum Performance)"
        elif st.session_state.selected_model_type == "adjusted_score":
            # For adjusted score model, use adjusted stability analysis if available
            if 'adjusted_stability_analysis' in results['info']:
                stability = results['info']['adjusted_stability_analysis']
                # Store it in session state for persistence
                st.session_state.selected_model_stability = stability
                model_display_name = "Best Adjusted Score Model (Balance of Performance & Stability)"
            else:
                # If adjusted stability analysis doesn't exist, create one using the adjusted model's metrics
                if 'adjusted_cv_metrics' in st.session_state.builder.model:
                    adjusted_metrics = st.session_state.builder.model['adjusted_cv_metrics']
                    
                    # Create a copy of the original stability analysis
                    import copy
                    stability = copy.deepcopy(results['info']['stability_analysis'])
                    
                    # Update stability level based on adjusted metrics
                    std_score = adjusted_metrics.get('std_score', 0)
                    if std_score > 0.1:
                        stability['level'] = "High variability"
                    elif std_score > 0.05:
                        stability['level'] = "Moderate stability"
                    else:
                        stability['level'] = "High stability"
                    
                    # Update stability score
                    stability['score'] = 1 - std_score
                    
                    # Update stability plots if the adjusted model has fold scores
                    if 'fold_scores' in adjusted_metrics:
                        fold_scores = adjusted_metrics['fold_scores']
                        
                        # Create new variation plot
                        import plotly.graph_objects as go
                        
                        # Performance variation figure
                        variation_fig = go.Figure()
                        
                        # Add min-mean-max line
                        variation_fig.add_trace(go.Scatter(
                            x=['Min', 'Mean', 'Max'],
                            y=[adjusted_metrics['min_score'], adjusted_metrics['mean_score'], adjusted_metrics['max_score']],
                            mode='lines+markers',
                            name='Score Range',
                            marker=dict(size=10)
                        ))
                        
                        # Add error bars using actual std
                        variation_fig.add_trace(go.Scatter(
                            x=['Mean'],
                            y=[adjusted_metrics['mean_score']],
                            error_y=dict(
                                type='data',
                                array=[adjusted_metrics['std_score']],
                                visible=True
                            ),
                            mode='markers',
                            name='Standard Deviation',
                            marker=dict(size=12, color='red')
                        ))
                        
                        variation_fig.update_layout(
                            title='Performance Variation',
                            yaxis_title='Score',
                            showlegend=True,
                            height=300
                        )
                        
                        # Create fold comparison chart
                        fold_fig = go.Figure(data=go.Bar(
                            x=[f'Fold {i+1}' for i in range(len(fold_scores))],
                            y=fold_scores,
                            marker_color='lightblue'
                        ))
                        
                        fold_fig.add_hline(
                            y=adjusted_metrics['mean_score'],
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Mean Score: {adjusted_metrics['mean_score']:.3f}"
                        )
                        
                        fold_fig.update_layout(
                            title='Performance Across Folds',
                            xaxis_title='Fold',
                            yaxis_title='Score',
                            showlegend=False,
                            height=400
                        )
                        
                        # Create gauge chart with standardized size
                        gauge_fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=1 - adjusted_metrics["std_score"],  # Convert std to stability score
                            title={'text': "Model Stability Score"},
                            gauge={
                                'axis': {'range': [0, 1]},
                                'steps': [
                                    {'range': [0, 0.9], 'color': "lightgray"},
                                    {'range': [0.9, 0.95], 'color': "yellow"},
                                    {'range': [0.95, 1], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 1 - adjusted_metrics["std_score"]
                                }
                            }
                        ))
                        
                        # Standardize gauge size
                        gauge_fig.update_layout(
                            height=350,
                            margin=dict(l=30, r=30, t=50, b=30)
                        )
                        
                        # Update the plots in the stability analysis
                        stability['plots'] = {
                            'gauge': gauge_fig,
                            'variation': variation_fig,
                            'fold_comparison': fold_fig
                        }
                    
                    # Store the adjusted stability analysis for future use
                    results['info']['adjusted_stability_analysis'] = stability
                    
                    # Also store it in session state for persistence
                    st.session_state.selected_model_stability = stability
                    model_display_name = "Best Adjusted Score Model (Balance of Performance & Stability)"
                else:
                    # Fallback to original if no adjusted metrics available
                    stability = results['info']['stability_analysis']
                    st.session_state.selected_model_stability = stability
                    model_display_name = "Default Model"
    else:
        # No model selected yet, use default
        stability = results['info']['stability_analysis']
        model_display_name = "Default Model (No Selection Made)"
    
    # Display which model is being analysed 
    st.markdown(f"## 🧠 Analyzing: {model_display_name}")
    
    # Check if there's a gauge in the stability plots and standardize its size
    if 'plots' in stability and 'gauge' in stability['plots']:
        # Update the gauge size to be consistent
        stability['plots']['gauge'].update_layout(
            height=350,
            margin=dict(l=30, r=30, t=50, b=30)
        )
    
    # Add an introduction to model stability with improved explanation
    with st.expander("📚 What is Model Stability?", expanded=False):
        st.markdown("""
        ### 📚 What is Model Stability?
        
        **Model stability** refers to how *consistently* your model performs across different subsets of your data. 
        Think of it like this:
        
        - ✅ **Stable model**: Performs similarly no matter which part of your data it sees
        - ❌ **Unstable model**: Performance varies widely depending on which data it's given
        
        **Why stability matters:**
        - More reliable predictions in real-world use
        - Less chance of unexpected behavior with new data
        - Higher confidence in model decisions
                    
        #### Real-World Example: Medical Diagnosis
        
        Imagine you're building a model to detect a disease from patient data:
        
        - **Unstable model**: 95% accurate with patients from Hospital A, but only 70% accurate with patients from Hospital B
        - **Stable model**: 85% accurate with patients from both Hospital A and Hospital B
        
        **Which would you trust more?**
        
        While the unstable model has a higher peak performance (95%), its inconsistency makes it risky. 
        The stable model (85%) would be more reliable across different patient populations.
        
        **This is why we care about stability!** In real-world applications, consistent performance is often 
        more important than having high performance on only certain subsets of data.
        """)
        
        # Add a visual explanation of what cross-validation folds are
        st.markdown("""
        ### How We Measure Stability
        
        We use **cross-validation** to test your model on different subsets of your data:
        
        ```
        Dataset split into 5 folds:
        [Fold 1][Fold 2][Fold 3][Fold 4][Fold 5]
        
        Round 1: Test on [Fold 1], Train on [Fold 2][Fold 3][Fold 4][Fold 5] → Score A
        Round 2: Test on [Fold 2], Train on [Fold 1][Fold 3][Fold 4][Fold 5] → Score B
        Round 3: Test on [Fold 3], Train on [Fold 1][Fold 2][Fold 4][Fold 5] → Score C
        Round 4: Test on [Fold 4], Train on [Fold 1][Fold 2][Fold 3][Fold 5] → Score D
        Round 5: Test on [Fold 5], Train on [Fold 1][Fold 2][Fold 3][Fold 4] → Score E
        ```
        
        **If Scores A-E are similar**: Your model is stable ✅  
        **If Scores A-E vary widely**: Your model is unstable ❌
        """)
    
    # Display the stability level with improved visual cues
    stability_emoji = {
        "High stability": "🌟",
        "Moderate stability": "⚠️",
        "High variability": "⚠️⚠️"
    }
    
    emoji = stability_emoji.get(stability['level'], "")
    st.markdown(f"### Your Model's Stability: {emoji} **{stability['level']}**")
    
    # Show which model's stability is being displayed
    if hasattr(st.session_state, 'selected_model_type'):
        model_type = "Best Mean Score" if st.session_state.selected_model_type == "mean_score" else "Best Adjusted Score"
        st.info(f"""
            **Showing stability analysis for the {model_type} model that you selected.**
            
            This analysis is specifically for the model configuration you've chosen to use.
            """)
    else:
        st.info("Select a model above to see its stability analysis.")
    
    # Log stability analysis results
    st.session_state.logger.log_calculation(
        "Model Stability Analysis",
        {
            "stability_level": stability['level'],
            "stability_score": stability.get('score', None),
            "model_type": getattr(st.session_state, 'selected_model_type', 'unknown')
        }
    )

    # Show stability score gauge with enhanced explanation
    st.markdown("### Stability Score")
    st.plotly_chart(stability['plots']['gauge'])
    st.info("""
        ### Understanding the Stability Score
        
        This gauge shows how consistently your model performs across different data splits:
        
        - **Score > 0.95**: 🌟 **High stability** — Great! Your model performs very consistently.
        - *What it means*: Your model will likely give reliable predictions with new data.
        - *Action needed*: None! Your model is stable.
        
        - **Score 0.90-0.95**: ⚠️ **Moderate stability** — Your model shows some inconsistency.
        - *What it means*: Your model's performance may vary somewhat with different data.
        - *Action needed*: Consider small adjustments to improve consistency.
        
        - **Score < 0.90**: ⚠️⚠️ **High variability** — Your model's performance changes significantly with different data.
        - *What it means*: Your model might be overfitting or struggling with certain data patterns.
        - *Action needed*: Review features, adjust parameters, or try different algorithms.
    """)

    # Log stability recommendations if needed
    if stability['level'] != "High stability":
        st.session_state.logger.log_recommendation(
            "Model Stability Improvements",
            {
                "current_level": stability['level'],
                "recommendations": stability.get('recommendations', []),
                "model_type": getattr(st.session_state, 'selected_model_type', 'unknown')
            }
        )
    
    # Show performance variation with enhanced explanations
    st.subheader("Performance Variation Analysis")
    col1, col2 = st.columns(2)
    with col1:
        # Use the variation plot from the selected stability data
        variation_plot = stability['plots']['variation']
        st.plotly_chart(variation_plot)
        st.info("""
            ### Performance Variation Chart
            
            This chart shows your model's performance range and variation:
            
            - **Blue line with points**: Shows the minimum, mean, and maximum scores across all folds
            - **Red dot with error bars**: Shows the mean score with standard deviation bars
            
            **How to interpret this chart:**
            - **Distance between Min and Max points**: Indicates overall performance range
            - **Error bars length**: Shows standard deviation (shorter = more stable)
            - **Ideal pattern**: Short error bars and minimal distance between Min and Max
            
            **Tip for beginners**: Look at the Standard Deviation and Score Range - smaller values indicate a more stable model that performs consistently across different data splits.
        """)
    with col2:
        # Use the fold comparison plot from the selected stability data
        fold_comparison_plot = stability['plots']['fold_comparison']
        st.plotly_chart(fold_comparison_plot)
        st.info("""
            ### Fold Comparison Chart
            
            This chart compares model performance across each cross-validation fold:
            
            - **Consistent height bars**: Your model performs similarly on all data splits
            - **Varying height bars**: Your model performs differently depending on which data it sees
            
            **For beginners**: More even bar heights indicate a more stable model.
        """)
        
    # Show detailed fold metrics for the selected model
    st.markdown("### Detailed Cross-Validation Metrics")
    
    # Get the fold metrics for the currently selected model
    model_metrics = None
    if hasattr(st.session_state, 'selected_model_type'):
        if st.session_state.selected_model_type == "mean_score" and "cv_metrics" in st.session_state.builder.model:
            model_metrics = st.session_state.builder.model["cv_metrics"]
            title = "Mean Score Model Fold Performance"
        elif st.session_state.selected_model_type == "adjusted_score" and "adjusted_cv_metrics" in st.session_state.builder.model:
            model_metrics = st.session_state.builder.model["adjusted_cv_metrics"]
            title = "Adjusted Score Model Fold Performance"
    
    # Display fold performance if metrics are available
    if model_metrics and 'fold_scores' in model_metrics:
        st.markdown(f"### {title}")
        fold_scores = model_metrics['fold_scores']
        
        # Create fold comparison chart
        fold_fig = go.Figure(data=go.Bar(
            x=[f'Fold {i+1}' for i in range(len(fold_scores))],
            y=fold_scores,
            marker_color='lightblue'
        ))
        
        fold_fig.add_hline(
            y=model_metrics['mean_score'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean Score: {model_metrics['mean_score']:.3f}"
        )
        
        fold_fig.update_layout(
            title=f'Selected Model: {title}',
            xaxis_title='Fold',
            yaxis_title='Score',
            showlegend=False,
            height=400
        )
        
        #st.plotly_chart(fold_fig)
        
        # Display metrics in table format
        metrics_df = pd.DataFrame({
            'Metric': ['Mean Score', 'Standard Deviation', 'Minimum Score', 'Maximum Score', 'Score Range'],
            'Value': [
                f"{model_metrics['mean_score']:.4f}",
                f"{model_metrics['std_score']:.4f}",
                f"{model_metrics['min_score']:.4f}",
                f"{model_metrics['max_score']:.4f}",
                f"{model_metrics['max_score'] - model_metrics['min_score']:.4f}"
            ]
        })
        st.table(metrics_df)
    
        st.info("""
            ### Reading the Metrics Table
            
            This table summarizes your model's performance across all cross-validation folds:
            
            - **Mean Score**: Average performance score across all CV folds
            - **Standard Deviation**: How much scores vary between folds (lower = more stable)
            - **Minimum Score**: Lowest score in any fold (worst-case performance)
            - **Maximum Score**: Highest score in any fold (best-case performance)
            - **Score Range**: Difference between highest and lowest scores (smaller = more stable)
            
            **Tip for beginners**: Look at the Standard Deviation and Score Range - smaller values indicate a more stable model that performs consistently across different data splits.
        """)