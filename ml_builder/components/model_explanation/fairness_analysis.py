from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
    equalized_odds_difference,
    demographic_parity_difference,
    count
)

# Constants for fairness analysis
FAIRNESS_THRESHOLD = 0.8
EXCELLENT_FAIRNESS_THRESHOLD = 0.9
SEVERE_BIAS_THRESHOLD = 0.6
PERFECT_FAIRNESS_SCORE = 1.0
WORST_FAIRNESS_SCORE = 0.0
SIGNIFICANT_DIFFERENCE_THRESHOLD = 0.2
MAX_UNIQUE_VALUES_FOR_ANALYSIS = 20
MIN_SAMPLE_SIZE_WARNING = 10

# UI and analysis constants
MAX_FEATURES_BEFORE_WARNING = 15
FAST_ANALYSIS_FEATURE_LIMIT = 10
CATEGORICAL_THRESHOLD = 10
HIGH_CARDINALITY_WARNING_LIMIT = 15
MULTICLASS_THRESHOLD = 2
UI_COLUMNS_STANDARD = 3
UI_COLUMNS_METRICS = 4

# Impact calculation constants
EXAMPLE_MONTHLY_APPLICATIONS = 1000
EXCELLENT_BIAS_IMPACT_FACTOR = 0.1
GOOD_BIAS_IMPACT_FACTOR = 0.3
CONCERNING_BIAS_IMPACT_FACTOR = 0.5
DISCRIMINATION_FACTOR = 0.6

# === UTILITY FUNCTIONS ===

def rmse_score(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def safe_auc_score(y_true, y_pred):
    """Calculate AUC score with error handling for multiclass"""
    try:
        # For multiclass, use macro average
        if len(np.unique(y_true)) > MULTICLASS_THRESHOLD:
            # AUC requires probabilities for multiclass, but we have class labels
            # Skip AUC for multiclass when we only have predictions
            return np.nan
        else:
            return roc_auc_score(y_true, y_pred)
    except (ValueError, AttributeError, TypeError) as e:
        # Handle specific sklearn errors that can occur with AUC calculation
        return np.nan

def safe_precision_score(y_true, y_pred):
    """Calculate precision score with zero division and multiclass handling"""
    try:
        # For multiclass, use macro average
        if len(np.unique(y_true)) > MULTICLASS_THRESHOLD:
            return precision_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            return precision_score(y_true, y_pred, zero_division=0)
    except (ValueError, AttributeError, TypeError) as e:
        # Handle specific sklearn errors that can occur with precision calculation
        return np.nan

def safe_f1_score(y_true, y_pred):
    """Calculate F1 score with zero division and multiclass handling"""
    try:
        # For multiclass, use macro average
        if len(np.unique(y_true)) > MULTICLASS_THRESHOLD:
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            return f1_score(y_true, y_pred, zero_division=0)
    except (ValueError, AttributeError, TypeError) as e:
        # Handle specific sklearn errors that can occur with F1 calculation
        return np.nan

def safe_recall_score(y_true, y_pred):
    """Calculate recall score with zero division and multiclass handling"""
    try:
        # For multiclass, use macro average
        if len(np.unique(y_true)) > MULTICLASS_THRESHOLD:
            return recall_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            return recall_score(y_true, y_pred, zero_division=0)
    except (ValueError, AttributeError, TypeError) as e:
        # Handle specific sklearn errors that can occur with recall calculation
        return np.nan

def true_positive_rate(y_true, y_pred):
    """Calculate the true positive rate (sensitivity/recall)."""
    if np.sum(y_true == 1) == 0:
        return 0.0
    return np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)

def is_metric_column(col_name):
    """
    Determine if a column name represents a metric/score rather than a feature.
    
    This function filters out columns that appear to contain model evaluation metrics,
    scores, or calculated values that shouldn't be used for fairness analysis.
    Such columns are typically outputs or derivatives of model predictions.
    
    Args:
        col_name (str): The column name to evaluate
    
    Returns:
        bool: True if the column appears to be a metric/score column, False otherwise
    
    Detection Logic:
        Checks if column name (case-insensitive) contains any of these keywords:
        - accuracy, precision, recall, f1
        - roc, auc (ROC curve related)  
        - error, loss, mse, mae, rmse (error metrics)
        - r2 (R-squared)
        - score, metric (generic score indicators)
    
    Examples:
        >>> is_metric_column('age')              # False - regular feature
        >>> is_metric_column('model_accuracy')   # True - contains 'accuracy'
        >>> is_metric_column('F1_Score')         # True - contains 'f1' and 'score'
        >>> is_metric_column('precision_weighted')  # True - contains 'precision'
        >>> is_metric_column('customer_score')   # True - contains 'score'
    
    Notes:
        - Used to automatically exclude metric columns from fairness analysis
        - Prevents circular analysis (using model outputs to evaluate model fairness)
        - Conservative approach: flags any column that might be a metric
        - Case-insensitive matching for robustness across naming conventions
    """
    metric_keywords = ['accuracy', 'precision', 'recall', 'f1', 'roc', 'auc', 'error',
                     'loss', 'mse', 'mae', 'rmse', 'r2', 'score', 'metric']
    
    col_lower = col_name.lower()
    
    for keyword in metric_keywords:
        if keyword in col_lower:
            return True
    
    return False

def get_accessible_color_palette():
    """Return colorblind-friendly color palette."""
    return {
        'good': '#2E8B57',     # Sea Green
        'warning': '#FF8C00',  # Dark Orange  
        'bad': '#DC143C',      # Crimson
        'neutral': '#4682B4'   # Steel Blue
    }

def auto_bin_high_cardinality_feature(feature_data, feature_name, n_bins=6, min_group_size=10):
    """
    Automatically bin a high-cardinality feature into meaningful groups.
    
    This function intelligently groups features with many unique values into fewer
    categories for better fairness analysis interpretability and statistical reliability.
    
    Args:
        feature_data (pd.Series): The feature column to bin
        feature_name (str): Name of the feature
        n_bins (int): Target number of bins (default 6)
        min_group_size (int): Minimum samples per group (default 10)
    
    Returns:
        tuple: (binned_series, bin_info_dict)
            - binned_series: pd.Series with binned values
            - bin_info_dict: Dictionary with binning metadata
    """
    
    # Determine if feature is numeric or categorical
    is_numeric = pd.api.types.is_numeric_dtype(feature_data)
    n_unique = feature_data.nunique()
    
    bin_info = {
        'original_feature': feature_name,
        'binning_method': None,
        'n_original_groups': n_unique,
        'n_final_groups': 0,
        'bin_labels': {},
        'was_binned': False
    }
    
    # If already has few groups, don't bin
    if n_unique <= 15:
        bin_info['was_binned'] = False
        return feature_data, bin_info
    
    binned_data = feature_data.copy()
    
    try:
        if is_numeric:
            # Numeric feature: use quantile-based binning
            bin_info['binning_method'] = 'quantile'
            
            # Special handling for age-like features
            # Special handling for age-like features
            # Check if it looks like age (positive values, reasonable range for human age)
            min_val = feature_data.min()
            max_val = feature_data.max()
            
            if 'age' in feature_name.lower() and min_val >= -1 and max_val <= 125:
                # Use domain-appropriate age ranges
                bins = [0, 18, 25, 35, 45, 55, 65, 100]
                labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
                try:
                    binned_data = pd.cut(feature_data, bins=bins, labels=labels, include_lowest=True)
                    
                    # Convert to strings and handle any NaNs (out of range values)
                    # We convert categorical to object, fill NaNs, and ensure all are strings
                    binned_data = binned_data.astype(object)
                    binned_data = binned_data.fillna('Unknown')
                    binned_data = binned_data.astype(str)
                    
                    bin_info['binning_method'] = 'age_ranges'
                    bin_info['bin_labels'] = {label: f"{feature_name}: {label}" for label in labels}
                    bin_info['bin_labels']['Unknown'] = f"{feature_name}: Unknown"
                except Exception:
                    # Fall back to quantile if age binning fails
                    pass
            
            # If age binning didn't work or not an age feature, use quantile binning
            if bin_info['binning_method'] == 'quantile':
                # Use quantile-based binning for roughly equal sample sizes
                try:
                    binned_data, bin_edges = pd.qcut(
                        feature_data, 
                        q=n_bins, 
                        labels=False, 
                        duplicates='drop',
                        retbins=True
                    )
                    
                    # Create readable labels
                    labels = []
                    for i in range(len(bin_edges) - 1):
                        lower = bin_edges[i]
                        upper = bin_edges[i + 1]
                        label = f"{lower:.1f}-{upper:.1f}"
                        labels.append(label)
                        bin_info['bin_labels'][i] = f"{feature_name}: {label}"
                    
                    # Map numeric bins to labels and ensure they're strings
                    binned_data = binned_data.map(lambda x: str(labels[int(x)]) if pd.notna(x) else 'Unknown')
                    
                except Exception as e:
                    # If quantile binning fails, fall back to equal-width
                    try:
                        binned_data = pd.cut(feature_data, bins=n_bins, labels=False, include_lowest=True)
                        binned_data = binned_data.map(lambda x: f"Bin {int(x)+1}" if pd.notna(x) else 'Unknown')
                        bin_info['binning_method'] = 'equal_width'
                    except Exception:
                        # If all binning fails, return original data
                        bin_info['was_binned'] = False
                        return feature_data, bin_info
        
        else:
            # Categorical feature: group by frequency
            bin_info['binning_method'] = 'frequency'
            
            # Get value counts
            value_counts = feature_data.value_counts()
            
            # Keep top (n_bins - 1) categories, group rest as 'Other'
            top_categories = value_counts.head(n_bins - 1).index.tolist()
            
            # Create binned version and ensure all values are strings for consistency
            binned_data = feature_data.apply(
                lambda x: str(x) if x in top_categories else 'Other'
            )
            
            # Create labels
            for cat in top_categories:
                bin_info['bin_labels'][cat] = f"{feature_name}: {cat}"
            bin_info['bin_labels']['Other'] = f"{feature_name}: Other"
        
        # Update bin info
        bin_info['n_final_groups'] = binned_data.nunique()
        bin_info['was_binned'] = True
        
        # Check if any groups are too small
        group_sizes = binned_data.value_counts()
        if group_sizes.min() < min_group_size:
            bin_info['warning'] = f"Some groups have fewer than {min_group_size} samples"
        
        return binned_data, bin_info
    
    except Exception as e:
        # If binning fails, return original data
        bin_info['was_binned'] = False
        bin_info['error'] = str(e)
        return feature_data, bin_info

def suggest_binning_strategy(feature_data, feature_name):
    """
    Analyze a feature and suggest optimal binning strategy with preview.
    
    This educational function helps users understand how to best group
    high-cardinality features for fairness analysis.
    
    Args:
        feature_data (pd.Series): The feature column to analyze
        feature_name (str): Name of the feature
    
    Returns:
        dict: Dictionary with binning recommendations and preview
    """
    
    is_numeric = pd.api.types.is_numeric_dtype(feature_data)
    n_unique = feature_data.nunique()
    
    recommendations = {
        'feature_name': feature_name,
        'n_unique': n_unique,
        'is_numeric': is_numeric,
        'suggested_method': None,
        'suggested_bins': None,
        'reasoning': None,
        'preview': None
    }
    
    if n_unique <= 15:
        recommendations['suggested_method'] = 'none'
        recommendations['reasoning'] = f"{feature_name} has {n_unique} groups, which is manageable for analysis."
        return recommendations
    
    if is_numeric:
        # Analyze numeric distribution
        min_val = feature_data.min()
        max_val = feature_data.max()
        median_val = feature_data.median()
        
        # Check if it looks like age
        if 'age' in feature_name.lower() or (min_val >= 0 and max_val <= 120):
            recommendations['suggested_method'] = 'age_ranges'
            recommendations['suggested_bins'] = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            recommendations['reasoning'] = (
                f"{feature_name} appears to be age-related. Using standard age brackets "
                "provides interpretable groups aligned with life stages."
            )
        else:
            recommendations['suggested_method'] = 'quantile'
            recommendations['suggested_bins'] = 6
            recommendations['reasoning'] = (
                f"{feature_name} is numeric with {n_unique} unique values. "
                "Quantile-based binning creates groups with roughly equal sample sizes, "
                "which is good for statistical reliability."
            )
        
        # Create preview
        binned_preview, _ = auto_bin_high_cardinality_feature(feature_data, feature_name)
        preview_counts = binned_preview.value_counts().sort_index()
        recommendations['preview'] = preview_counts.to_dict()
    
    else:
        # Categorical feature
        value_counts = feature_data.value_counts()
        top_5 = value_counts.head(5)
        
        recommendations['suggested_method'] = 'frequency'
        recommendations['suggested_bins'] = f"Top 5 categories + Other"
        recommendations['reasoning'] = (
            f"{feature_name} is categorical with {n_unique} unique values. "
            "Keeping the most frequent categories and grouping rare ones as 'Other' "
            "maintains interpretability while ensuring adequate sample sizes."
        )
        
        # Create preview
        recommendations['preview'] = {
            'top_categories': top_5.to_dict(),
            'other_count': value_counts.iloc[5:].sum() if len(value_counts) > 5 else 0
        }
    
    return recommendations

def export_fairness_report(fairness_results, problem_type, columns_to_analyse, format='json'):
    """
    Export fairness analysis results in various formats.
    
    Args:
        fairness_results (dict): Results from analyse_model_fairness()
        problem_type (str): Type of ML problem
        columns_to_analyse (list): List of analyzed features
        format (str): Export format - 'json', 'html', or 'markdown'
    
    Returns:
        str: Formatted export content
    """
    from datetime import datetime
    
    fairness_score = fairness_results.get("fairness_score", None)
    
    # Use active threshold from session if available
    try:
        threshold_active = getattr(st.session_state, 'fairness_threshold', FAIRNESS_THRESHOLD)
    except Exception:
        threshold_active = FAIRNESS_THRESHOLD
    
    if format == 'json':
        import json
        
        # Helper function to convert numpy types to native Python types
        def convert_to_native(obj):
            """Recursively convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        # Create clean JSON export
        export_data = {
            'export_metadata': {
                'timestamp': datetime.now().isoformat(),
                'problem_type': problem_type,
                'analysis_version': '1.0'
            },
            'overall_fairness': {
                'score': float(fairness_score) if fairness_score is not None else None,
                'threshold': float(threshold_active),
                'status': 'fair' if fairness_score and fairness_score >= threshold_active else 'biased'
            },
            'features_analyzed': columns_to_analyse,
            'feature_scores': {
                feat: {
                    'overall_score': float(score) if score is not None else None,
                    'summary': convert_to_native(fairness_results.get("feature_summaries", {}).get(feat, {}))
                }
                for feat, score in fairness_results.get("column_scores", {}).items()
                if score is not None
            },
            'bias_detected': bool(fairness_results.get("bias_detected", False)),
            'bias_types': fairness_results.get("bias_types", [])
        }
        
        # Convert entire structure to ensure all numpy types are handled
        export_data = convert_to_native(export_data)
        
        return json.dumps(export_data, indent=2)
    
    elif format == 'html':
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fairness Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .metric {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .good {{ background: #d4edda; border-left: 4px solid #28a745; }}
                .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
                .danger {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>🎯 Fairness Analysis Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Problem Type:</strong> {problem_type}</p>
            
            <h2>Executive Summary</h2>
            <div class="metric {'good' if fairness_score and fairness_score >= threshold_active else 'danger'}">
                <p><strong>Overall Fairness Score:</strong> {f"{fairness_score:.3f}" if fairness_score else 'N/A'}</p>
                <p><strong>Status:</strong> {'Fair' if fairness_score and fairness_score >= threshold_active else 'Bias Detected'}</p>
                <p><strong>Features Analyzed:</strong> {len(columns_to_analyse)}</p>
            </div>
            
            <h2>Feature-Level Analysis</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Fairness Score</th>
                    <th>Groups</th>
                    <th>Status</th>
                </tr>
        """
        
        for feat, score in fairness_results.get("column_scores", {}).items():
            if score is None:
                continue
            summary = fairness_results.get("feature_summaries", {}).get(feat, {})
            status = "Fair" if score >= threshold_active else "Biased"
            html_content += f"""
                <tr>
                    <td>{feat}</td>
                    <td>{score:.3f}</td>
                    <td>{summary.get('groups', 'N/A')}</td>
                    <td>{'✅ ' if score >= FAIRNESS_THRESHOLD else '⚠️ '}{status}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Recommendations</h2>
            <div class="metric warning">
                <p>Review features with scores below the threshold ({FAIRNESS_THRESHOLD:.2f})</p>
                <p>Consider fairness-aware preprocessing and model training techniques</p>
                <p>Monitor fairness metrics regularly with new data</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    else:  # markdown
        # Create Markdown report
        md_content = f"""# Fairness Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Problem Type:** {problem_type}

## Executive Summary

- **Overall Fairness Score:** {f'{fairness_score:.3f}' if fairness_score else 'N/A'}
- **Status:** {'✅ Fair' if fairness_score and fairness_score >= threshold_active else '⚠️ Bias Detected'}
- **Features Analyzed:** {len(columns_to_analyse)}
- **Threshold:** {threshold_active:.2f}

## Feature-Level Analysis

| Feature | Fairness Score | Groups | Status |
|---------|----------------|--------|--------|
"""
        
        for feat, score in fairness_results.get("column_scores", {}).items():
            if score is None:
                continue
            summary = fairness_results.get("feature_summaries", {}).get(feat, {})
            status = "✅ Fair" if score >= threshold_active else "⚠️ Biased"
            md_content += f"| {feat} | {score:.3f} | {summary.get('groups', 'N/A')} | {status} |\n"
        
        md_content += f"""

## Recommendations

- Review features with scores below the threshold ({FAIRNESS_THRESHOLD:.2f})
- Consider fairness-aware preprocessing and model training techniques
- Monitor fairness metrics regularly with new data
- Document any accepted fairness trade-offs for stakeholder review
"""
        
        return md_content

def create_interactive_impact_visualization(fairness_score, fairness_results, problem_type, monthly_volume=1000):
    """
    Create interactive visualization showing real-world impact of bias.
    
    Args:
        fairness_score (float): Overall fairness score
        fairness_results (dict): Results from fairness analysis
        problem_type (str): Type of ML problem
        monthly_volume (int): Monthly application volume
    
    Returns:
        plotly.graph_objects.Figure: Interactive impact visualization
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    if fairness_score is None:
        return None
    
    colors = get_accessible_color_palette()
    
    # Calculate impact
    bias_severity = 1 - fairness_score
    
    # Determine impact level and factors
    if bias_severity <= EXCELLENT_BIAS_IMPACT_FACTOR:  # Excellent
        impact_factor = EXCELLENT_BIAS_IMPACT_FACTOR
        risk_level = "Minimal"
        risk_color = colors['good']
    elif bias_severity <= SIGNIFICANT_DIFFERENCE_THRESHOLD:  # Good
        impact_factor = GOOD_BIAS_IMPACT_FACTOR
        risk_level = "Low"
        risk_color = colors['neutral']
    else:  # Concerning
        impact_factor = CONCERNING_BIAS_IMPACT_FACTOR
        risk_level = "High"
        risk_color = colors['bad']
    
    # Calculate affected applications
    affected_apps = int(monthly_volume * bias_severity * impact_factor)
    affected_apps_yearly = affected_apps * 12
    
    # Calculate potential discrimination (60% of affected)
    potential_discrimination = int(affected_apps * DISCRIMINATION_FACTOR)
    
    # Create visualization with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Monthly Impact',
            'Risk Level',
            'Annual Projection',
            'Bias Breakdown by Feature'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'indicator'}],
            [{'type': 'bar'}, {'type': 'bar'}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Monthly Impact (horizontal bar)
    fig.add_trace(
        go.Bar(
            y=['Potentially Affected', 'Likely Discriminated'],
            x=[affected_apps, potential_discrimination],
            orientation='h',
            marker_color=[risk_color, colors['bad']],
            text=[f"{affected_apps:,}", f"{potential_discrimination:,}"],
            textposition='auto',
            hovertemplate='%{y}: %{x:,} applications<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. Risk Level Gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=fairness_score,
            title={'text': f"Fairness Score"},
            delta={'reference': FAIRNESS_THRESHOLD, 'increasing': {'color': colors['good']}},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, SEVERE_BIAS_THRESHOLD], 'color': '#ffe6e6'},
                    {'range': [SEVERE_BIAS_THRESHOLD, FAIRNESS_THRESHOLD], 'color': '#fff4e6'},
                    {'range': [FAIRNESS_THRESHOLD, 1], 'color': '#e6f7e6'}
                ],
                'threshold': {
                    'line': {'color': colors['warning'], 'width': 4},
                    'thickness': 0.75,
                    'value': FAIRNESS_THRESHOLD
                }
            }
        ),
        row=1, col=2
    )

    # 3. Annual Projection
    fig.add_trace(
        go.Bar(
            x=['Monthly', 'Yearly'],
            y=[affected_apps, affected_apps_yearly],
            marker_color=[risk_color, risk_color],
            text=[f"{affected_apps:,}", f"{affected_apps_yearly:,}"],
            textposition='auto',
            hovertemplate='%{x}: %{y:,} affected<extra></extra>'
        ),
        row=2, col=1
    )

    # 4. Bias by Feature
    if fairness_results.get("column_scores"):
        # Get features with bias (score < 0.8)
        biased_features = [
            (feat, score) for feat, score in fairness_results["column_scores"].items()
            if score is not None and score < FAIRNESS_THRESHOLD
        ]

        if biased_features:
            # Sort by score (worst first)
            biased_features.sort(key=lambda x: x[1])
            feat_names = [f[0] for f in biased_features[:5]]  # Top 5
            feat_scores = [f[1] for f in biased_features[:5]]
            feat_colors = [colors['bad'] if s < SEVERE_BIAS_THRESHOLD else colors['warning'] for s in feat_scores]

            fig.add_trace(
                go.Bar(
                    x=feat_names,
                    y=feat_scores,
                    marker_color=feat_colors,
                    text=[f"{s:.3f}" for s in feat_scores],
                    textposition='auto',
                    hovertemplate='%{x}: %{y:.3f}<extra></extra>'
                ),
                row=2, col=2
            )
        else:
            # No biased features
            fig.add_trace(
                go.Bar(
                    x=['No Issues'],
                    y=[1.0],
                    marker_color=[colors['good']],
                    text=['All Fair'],
                    textposition='auto'
                ),
                row=2, col=2
            )

    # Update layout
    fig.update_xaxes(title_text="Applications", row=1, col=1)
    fig.update_xaxes(title_text="Period", row=2, col=1)
    fig.update_xaxes(title_text="Features", row=2, col=2)
    fig.update_yaxes(title_text="Fairness Score", row=2, col=2)
    
    fig.update_layout(
        title_text=f"🎯 Real-World Impact Analysis (Based on {monthly_volume:,} monthly applications)",
        showlegend=False,
        height=700
    )
    
    return fig

def create_enhanced_group_comparison(perf_data, selected_feature, problem_type, fairness_threshold=0.8):
    """
    Create enhanced group comparison visualizations adapted to the number of groups.
    
    Args:
        perf_data (pd.DataFrame): Performance metrics by group
        selected_feature (str): Name of the feature being analyzed
        problem_type (str): Type of ML problem
        fairness_threshold (float): Threshold for fairness (default 0.8)
    
    Returns:
        list: List of Plotly figures to display
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    n_groups = len(perf_data)
    figures = []
    colors = get_accessible_color_palette()
    
    # Remove 'count' from display if present
    display_data = perf_data.copy()
    if 'count' in display_data.columns:
        group_counts = display_data['count']
        display_data = display_data.drop('count', axis=1)
    else:
        group_counts = None
    
    if problem_type in ["binary_classification", "multiclass_classification", "classification"]:
        # Classification visualizations
        key_metrics = ['accuracy', 'precision', 'recall', 'f1']
        available_metrics = [m for m in key_metrics if m in display_data.columns]
        
        if n_groups <= 10 and len(available_metrics) > 1:
            # For few groups: grouped bar chart with multiple metrics
            fig = go.Figure()
            
            for metric in available_metrics:
                fig.add_trace(go.Bar(
                    name=metric.capitalize(),
                    x=display_data.index,
                    y=display_data[metric],
                    text=[f"{v:.3f}" for v in display_data[metric]],
                    textposition='auto',
                ))
            
            # Add fairness threshold line
            fig.add_hline(
                y=fairness_threshold, 
                line_dash="dash", 
                line_color=colors['warning'],
                annotation_text=f"Fairness Threshold ({fairness_threshold})",
                annotation_position="right"
            )
            
            fig.update_layout(
                title=f"Performance Metrics by {selected_feature}",
                xaxis_title=selected_feature,
                yaxis_title="Score",
                barmode='group',
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            figures.append(fig)
        
        else:
            # For many groups: separate charts for each key metric
            for metric in available_metrics[:4]:  # Show top 4 metrics
                # Determine color based on values relative to mean
                mean_val = display_data[metric].mean()
                colors_list = [colors['good'] if v >= mean_val else colors['warning'] 
                             for v in display_data[metric]]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=display_data.index,
                    y=display_data[metric],
                    marker_color=colors_list,
                    text=[f"{v:.3f}" for v in display_data[metric]],
                    textposition='auto',
                ))
                
                # Add mean line
                fig.add_hline(
                    y=mean_val,
                    line_dash="dash",
                    annotation_text="Average",
                    annotation_position="right"
                )
                
                fig.update_layout(
                    title=f"{metric.capitalize()} by {selected_feature}",
                    xaxis_title=selected_feature,
                    yaxis_title=metric.capitalize(),
                    height=350,
                    showlegend=False
                )
                
                figures.append(fig)
    
    else:
        # Regression visualizations
        reg_metrics = ['r2', 'mae', 'rmse']
        available_metrics = [m for m in reg_metrics if m in display_data.columns]
        
        for metric in available_metrics[:2]:  # Show top 2 metrics
            # For error metrics, lower is better; for R², higher is better
            is_error_metric = metric in ['mae', 'rmse', 'mse']
            
            if is_error_metric:
                # Lower is better - color accordingly
                min_val = display_data[metric].min()
                colors_list = [colors['good'] if v <= min_val * 1.2 else colors['warning']
                             for v in display_data[metric]]
            else:
                # Higher is better (R²)
                max_val = display_data[metric].max()
                colors_list = [colors['good'] if v >= max_val * 0.9 else colors['warning']
                             for v in display_data[metric]]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=display_data.index,
                y=display_data[metric],
                marker_color=colors_list,
                text=[f"{v:.3f}" for v in display_data[metric]],
                textposition='auto',
            ))
            
            # Add mean line
            mean_val = display_data[metric].mean()
            fig.add_hline(
                y=mean_val,
                line_dash="dash",
                annotation_text="Average",
                annotation_position="right"
            )
            
            metric_names = {
                'r2': 'R² Score',
                'mae': 'Mean Absolute Error',
                'rmse': 'Root Mean Squared Error',
                'mse': 'Mean Squared Error'
            }
            
            fig.update_layout(
                title=f"{metric_names.get(metric, metric.upper())} by {selected_feature}",
                xaxis_title=selected_feature,
                yaxis_title=metric_names.get(metric, metric.upper()),
                height=350,
                showlegend=False
            )
            
            figures.append(fig)
    
    # Add sample distribution subplot if we have group counts
    if group_counts is not None and len(figures) > 0:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            x=group_counts.index,
            y=group_counts.values,
            marker_color=colors['neutral'],
            text=[f"{int(v):,}" for v in group_counts.values],
            textposition='auto',
        ))
        
        fig_dist.update_layout(
            title=f"Sample Distribution Across {selected_feature} Groups",
            xaxis_title=selected_feature,
            yaxis_title="Number of Samples",
            height=300,
            showlegend=False
        )
        
        figures.append(fig_dist)
    
    return figures

def create_fairness_heatmap(fairness_results, problem_type):
    """
    Create an interactive heatmap showing fairness scores across all features and components.
    
    This provides an at-a-glance overview of fairness landscape, making it easy to identify
    which features and which fairness components need attention.
    
    Args:
        fairness_results (dict): Results from analyse_model_fairness()
        problem_type (str): Type of ML problem
    
    Returns:
        plotly.graph_objects.Figure: Interactive heatmap figure
    """
    import plotly.graph_objects as go
    
    if not fairness_results.get("column_scores"):
        return None
    
    # Get color palette
    colors = get_accessible_color_palette()
    
    # Prepare data for heatmap
    features = []
    scores_matrix = []
    component_names = []
    
    # Determine component names based on problem type
    if problem_type == "multiclass_classification":
        component_names = ['Demographic Parity', 'Accuracy Consistency']
    elif problem_type in ["binary_classification", "classification"]:
        component_names = ['Demographic Parity', 'Equalized Odds']
    else:  # regression
        component_names = ['Error Consistency']
    
    # Sort features by overall score (worst first)
    sorted_features = sorted(
        fairness_results["column_scores"].items(),
        key=lambda x: x[1] if x[1] is not None else 1.0
    )
    
    for feature, overall_score in sorted_features:
        if overall_score is None:
            continue
            
        features.append(feature)
        summary = fairness_results.get("feature_summaries", {}).get(feature, {})
        individual_scores = summary.get("individual_scores", {})
        
        # Build row based on problem type
        if problem_type == "multiclass_classification":
            row = [
                individual_scores.get('demographic_parity', 0),
                individual_scores.get('accuracy_consistency', 0),
            ]
        elif problem_type in ["binary_classification", "classification"]:
            row = [
                individual_scores.get('demographic_parity', 0),
                individual_scores.get('equalized_odds', 0),
            ]
        else:  # regression
            row = [
                individual_scores.get('error_consistency', 0),
            ]
        
        scores_matrix.append(row)
    
    if not scores_matrix:
        return None
    
    # Create heatmap with custom colorscale
    # Red (0-0.6), Orange (0.6-0.8), Green (0.8-1.0)
    colorscale = [
        [0.0, colors['bad']],      # Severe bias
        [0.6, colors['warning']],  # Moderate bias
        [0.8, colors['good']],     # Good fairness
        [1.0, colors['good']]      # Excellent fairness
    ]
    
    # Create hover text with interpretations
    hover_text = []
    for i, feature in enumerate(features):
        row_hover = []
        for j, component in enumerate(component_names):
            score = scores_matrix[i][j]
            if score >= 0.8:
                status = "✅ Fair"
            elif score >= 0.6:
                status = "⚠️ Review Needed"
            else:
                status = "🔴 Critical"
            
            summary = fairness_results.get("feature_summaries", {}).get(feature, {})
            groups = summary.get('groups', 'N/A')
            
            hover_text_cell = (
                f"<b>{feature}</b><br>"
                f"Component: {component}<br>"
                f"Score: {score:.3f}<br>"
                f"Status: {status}<br>"
                f"Groups: {groups}"
            )
            row_hover.append(hover_text_cell)
        hover_text.append(row_hover)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=scores_matrix,
        x=component_names,
        y=features,
        colorscale=colorscale,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(
            title="Fairness<br>Score",
            ticktext=['Severe<br>(<0.6)', 'Review<br>(0.6-0.8)', 'Fair<br>(≥0.8)'],
            tickvals=[0.3, 0.7, 0.9],
            tickmode='array'
        ),
        zmid=0.7,  # Center the colorscale at 0.7
        zmin=0.0,
        zmax=1.0
    ))
    
    fig.update_layout(
        title={
            'text': "🗺️ Fairness Landscape Overview<br><sub>Darker red = more bias concern | Green = fair</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Fairness Components",
        yaxis_title="Features (Sorted by Overall Score)",
        height=max(300, len(features) * 40),  # Dynamic height based on number of features
        margin=dict(l=150, r=100, t=80, b=50),
        font=dict(size=11)
    )
    
    return fig

# === FEATURE SELECTION INTERFACE ===

def create_clean_feature_selection_interface(valid_columns, protected_cols, filtered_columns):
    """
    Create an intuitive Streamlit interface for selecting features to analyze for bias.
    
    This function renders a comprehensive UI that helps users select which features
    (protected attributes) to analyze for fairness. It provides smart defaults,
    explains feature characteristics, and warns about potential analysis issues.
    
    Args:
        valid_columns (list): List of column names eligible for fairness analysis
                             (excludes metric columns and other unsuitable features)
        protected_cols (list): Auto-detected protected attribute column names
                              (e.g., age, gender, race) found using pattern matching
        filtered_columns (list): Column names that were filtered out (e.g., metric columns)
                                to show users what was excluded and why
    
    Returns:
        list: Selected column names to analyze for fairness, or empty list if none selected
    
    UI Features:
        Smart Selection (Recommended):
            - Uses auto-detected protected attributes as defaults
            - Shows explanation of why each feature was selected
            - Displays feature characteristics and metadata
            
        Custom Selection:
            - For datasets ≤20 features: Visual checkbox interface with feature metadata
            - For larger datasets: Traditional multiselect with smart defaults
            - Groups features by type (categorical vs numeric)
            - Shows unique value counts and analysis suitability warnings
            
        Analyze All Features:
            - Selects all valid columns for comprehensive analysis
            - Warns about performance implications for large feature sets
    
    Smart Warnings and Recommendations:
        - Performance warnings for >15 features with option to limit
        - High-cardinality warnings for features with >20 unique values
        - Small group size warnings for groups with <10 samples  
        - Actionable recommendations for each type of issue detected
        
    UI Organization:
        1. Auto-detection results and excluded columns summary
        2. Selection method choice (Smart/Custom/All)
        3. Feature selection interface (varies by method)
        4. Analysis warnings and recommendations
        5. Final selection summary and validation
    
    Notes:
        - Accesses st.session_state.builder.X_test for feature analysis
        - All UI text includes helpful explanations and context
        - Provides educational content about fairness analysis
        - Gracefully handles edge cases (no features, all filtered, etc.)
        
    Example:
        >>> selected_features = create_clean_feature_selection_interface(
        ...     valid_columns=['age', 'income', 'location'],
        ...     protected_cols=['age'],
        ...     filtered_columns=['score_column']
        ... )
        >>> print(f"Selected: {selected_features}")
    """
    
    st.markdown("## 🔍 Feature Selection for Fairness Analysis")
    
    # === 1. SHOW WHAT WE FOUND ===
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if protected_cols:
            st.success(f"🎯 **Auto-detected protected attributes:** {', '.join(protected_cols)}")
        else:
            st.info("🔍 No protected attributes auto-detected. You can manually select features below.")
    
    with col2:
        if filtered_columns:
            with st.expander("📊 Excluded Columns", expanded=False):
                st.caption("These columns were excluded as they appear to be metrics:")
                for col in filtered_columns:
                    st.caption(f"• {col}")

    # === 2. SELECTION METHOD ===
    st.markdown("### 🎛️ Choose Analysis Scope")
    
    selection_method = st.radio(
        "Select your approach:",
        [
            "🎯 Smart Selection (Recommended)",
            "🎨 Custom Selection", 
            "🔄 Analyze All Features"
        ],
        key="selection_method",
        horizontal=True
    )

    # === 3. CONFIGURATION BASED ON SELECTION ===
    columns_to_analyse = []
    
    if selection_method == "🎯 Smart Selection (Recommended)":
        if protected_cols:
            columns_to_analyse = [col for col in protected_cols if col in valid_columns]
            
            st.info(f"""
            ✅ **Using {len(columns_to_analyse)} auto-detected features:** {', '.join(columns_to_analyse)}
            
            These were automatically identified as potential protected attributes based on:
            - Low number of unique values (≤{MAX_UNIQUE_VALUES_FOR_ANALYSIS})
            - Common protected attribute names (age, gender, race, etc.)
            - Data type patterns
            """)
        else:
            st.warning("⚠️ **No Protected Attributes Automatically Detected**\n\n"
                       "We couldn't find features with common protected attribute names "
                       "(like 'age', 'gender', 'race', 'income') or features with suitable "
                       "characteristics for fairness analysis.\n\n"
                       "**What This Means:** Your dataset may not contain obvious demographic features, "
                       "or they may be named differently than expected.\n\n"
                       "**Next Steps:** Use 'Custom Selection' below to manually choose features "
                       "that represent demographic groups or protected classes in your use case.")
            return []
    
    elif selection_method == "🎨 Custom Selection":
        st.markdown("**Select specific features to analyze:**")
        
        # Create a more visual selection interface
        if len(valid_columns) <= 20:
            # For smaller datasets, show columns with metadata
            selected_columns = []
            
            # Group columns by type for better organization
            categorical_cols = []
            numeric_cols = []
            
            for col in valid_columns:
                sample_data = st.session_state.builder.X_test[col]
                if sample_data.nunique() <= CATEGORICAL_THRESHOLD:
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)
            
            if categorical_cols:
                st.markdown("**📊 Categorical Features (Good for fairness analysis):**")
                cat_cols = st.columns(min(UI_COLUMNS_STANDARD, len(categorical_cols)))
                for i, col in enumerate(categorical_cols):
                    with cat_cols[i % UI_COLUMNS_STANDARD]:
                        sample_data = st.session_state.builder.X_test[col]
                        unique_count = sample_data.nunique()
                        is_selected = st.checkbox(
                            f"{col}",
                            value=col in (protected_cols or []),
                            key=f"cat_{col}"
                        )
                        st.caption(f"{unique_count} unique values")
                        if is_selected:
                            selected_columns.append(col)
            
            if numeric_cols:
                st.markdown("**🔢 Numeric Features (May need binning):**")
                num_cols = st.columns(min(UI_COLUMNS_STANDARD, len(numeric_cols)))
                for i, col in enumerate(numeric_cols):
                    with num_cols[i % UI_COLUMNS_STANDARD]:
                        sample_data = st.session_state.builder.X_test[col]
                        unique_count = sample_data.nunique()
                        is_selected = st.checkbox(
                            f"{col}",
                            value=False,
                            key=f"num_{col}"
                        )
                        st.caption(f"{unique_count} unique values")
                        if unique_count > CATEGORICAL_THRESHOLD:
                            st.caption("⚠️ Many values - may need grouping")
                        if is_selected:
                            selected_columns.append(col)
            
            columns_to_analyse = selected_columns
            
        else:
            # For larger datasets, use traditional multiselect but with better defaults
            default_selection = protected_cols if protected_cols else []
            
            columns_to_analyse = st.multiselect(
                "Choose features to analyze:",
                options=valid_columns,
                default=default_selection,
                help="💡 Tip: Start with demographic features like age, gender, location, etc."
            )
        
        if columns_to_analyse:
            st.success(f"✅ Selected {len(columns_to_analyse)} features: {', '.join(columns_to_analyse)}")
        else:
            st.info("👆 **Feature Selection Required**\n\n"
                    "Please select at least one feature from the options above to perform fairness analysis. "
                    "We recommend starting with demographic features like age, location, or other attributes "
                    "that might create different groups in your data.")
    
    else:  # "🔄 Analyze All Features"
        columns_to_analyse = valid_columns
        
        st.info(f"""
        📋 **Analyzing all {len(valid_columns)} available features.**
        
        This will analyze every feature in your dataset for potential bias.
        """)

    # === 4. SMART WARNINGS AND RECOMMENDATIONS ===
    if columns_to_analyse:
        # Check for potential issues
        warnings = []
        
        # Check for too many features
        if len(columns_to_analyse) > MAX_FEATURES_BEFORE_WARNING:
            warnings.append(f"⏱️ Analyzing {len(columns_to_analyse)} features may take longer")
            
            # Offer to limit
            limit_analysis = st.checkbox(
                f"Limit to first {FAST_ANALYSIS_FEATURE_LIMIT} features for faster analysis?", 
                value=False,
                key="limit_features"
            )
            if limit_analysis:
                columns_to_analyse = columns_to_analyse[:FAST_ANALYSIS_FEATURE_LIMIT]
                st.info(f"Limited to first {FAST_ANALYSIS_FEATURE_LIMIT} features: {', '.join(columns_to_analyse)}")
        
        # Check for high-cardinality features
        high_cardinality = []
        for col in columns_to_analyse:
            if col in valid_columns:
                unique_count = st.session_state.builder.X_test[col].nunique()
                if unique_count > MAX_UNIQUE_VALUES_FOR_ANALYSIS:
                    high_cardinality.append((col, unique_count))
        
        if high_cardinality:
            warnings.append("🔢 Some features have many unique values")
        
        # Check for small groups
        small_group_features = []
        for col in columns_to_analyse:
            if col in valid_columns:
                min_group_size = st.session_state.builder.X_test[col].value_counts().min()
                if min_group_size < MIN_SAMPLE_SIZE_WARNING:
                    small_group_features.append((col, min_group_size))
        
        if small_group_features:
            warnings.append("👥 Some features have very small groups")
            
        # Show summary
        if warnings:
            st.markdown("### ⚠️ Analysis Notes")
            for warning in warnings:
                st.caption(warning)
            if high_cardinality:
                with st.expander("⚠️ High-Cardinality Features", expanded=False):
                    st.warning("These features have many unique values, which may affect analysis quality:")
                    
                    # Add auto-binning option
                    use_auto_binning = st.checkbox(
                        "✨ **Use Automatic Grouping (Recommended)**",
                        value=True,
                        key="use_auto_binning",
                        help="Automatically group high-cardinality features into 5-7 meaningful categories for better analysis"
                    )
                    
                    if use_auto_binning:
                        n_bins = 6
                    #    n_bins = st.slider(
                    #        "Number of groups to create:",
                    #        min_value=3,
                    #        max_value=10,
                    #        value=6,
                    #        key="auto_bin_count",
                    #        help="Fewer groups = simpler analysis, more groups = more detail"
                    #    )
                        st.session_state.auto_binning_enabled = True
                        st.session_state.auto_binning_n_bins = n_bins
                    else:
                        st.session_state.auto_binning_enabled = False
                    
                    # Show specific recommendations for each feature
                    for col, count in high_cardinality:
                        st.markdown(f"**{col}**: {count} unique values")
                        
                        # Get binning suggestion
                        feature_data = st.session_state.builder.X_test[col]
                        suggestion = suggest_binning_strategy(feature_data, col)
                        
                        with st.expander(f"💡 Recommendation for {col}", expanded=False):
                            st.info(suggestion['reasoning'])
                            
                            if suggestion['preview']:
                                st.markdown("**Preview of suggested grouping:**")
                                if isinstance(suggestion['preview'], dict) and 'top_categories' in suggestion['preview']:
                                    # Categorical preview
                                    st.markdown("Top categories:")
                                    for cat, count in suggestion['preview']['top_categories'].items():
                                        st.caption(f"• {cat}: {count} samples")
                                    if suggestion['preview']['other_count'] > 0:
                                        st.caption(f"• Other: {suggestion['preview']['other_count']} samples")
                                else:
                                    # Numeric preview
                                    preview_df = pd.DataFrame([
                                        {'Group': k, 'Sample Count': v} 
                                        for k, v in list(suggestion['preview'].items())[:7]
                                    ])
                                    st.dataframe(preview_df, hide_index=True, use_container_width=True)
                    
                    if not use_auto_binning:
                        st.markdown("""
                        **Manual Recommendations:**
                        - Consider grouping values into ranges (e.g., age groups: 18-25, 26-35, etc.)
                        - Focus on features with fewer categories for more reliable results
                        - Remove features that are essentially unique identifiers
                        """)

            if small_group_features:
                with st.expander("👥 Small Group Sizes", expanded=False):
                    st.warning("These features have very small groups, which may give unreliable results:")
                    for col, min_size in small_group_features:
                        st.markdown(f"• **{col}**: smallest group has {min_size} samples")
                    
                    st.markdown("""
                    **Consider:**
                    - Collecting more data for underrepresented groups
                    - Combining small categories where appropriate
                    - Interpreting results for small groups with caution
                    """)
                    
    return columns_to_analyse

# === CORE ANALYSIS FUNCTIONS ===

def analyse_model_fairness(X_test, y_test, predictions, columns_to_analyse, problem_type):
    """
    Analyze model fairness across protected attributes and demographic groups.
    
    This function performs comprehensive fairness analysis by calculating fairness metrics
    for each selected feature/attribute. It supports binary classification, multiclass 
    classification, and regression problems with different fairness measures for each.
    
    Args:
        X_test (pd.DataFrame): Test dataset features containing the protected attributes
        y_test (pd.Series or np.array): True target values for test dataset  
        predictions (np.array): Model predictions on test dataset
        columns_to_analyse (list): List of column names to analyze for bias/fairness
        problem_type (str): Type of ML problem - 'binary_classification', 
                           'multiclass_classification', or 'regression'
    
    Returns:
        dict: Comprehensive fairness analysis results containing:
            - success (bool): Whether analysis completed successfully
            - message (str): Error message if analysis failed
            - fairness_score (float): Overall fairness score (min across all features)
            - column_scores (dict): Individual fairness scores by feature
            - bias_detected (bool): Whether bias was detected in any feature
            - bias_types (list): List of bias descriptions for detected issues
            - recommendations (list): Actionable recommendations (empty in this function)
            - metric_frames (dict): Fairlearn MetricFrames for detailed analysis
            - feature_summaries (dict): Summary statistics for each analyzed feature
    
    Fairness Metrics by Problem Type:
        Binary Classification:
            - Demographic Parity: Equal positive prediction rates across groups
            - Equalized Odds: Equal true/false positive rates across groups
            - Overall Score: Minimum of the two above (stricter fairness standard)
            
        Multiclass Classification:
            - Demographic Parity: Equal prediction distributions across all classes
            - Accuracy Consistency: Similar accuracy across all groups
            - Overall Score: Minimum of the two above
            
        Regression:
            - Error Consistency: Similar Mean Absolute Error across groups
            - Overall Score: Ratio of best to worst group performance
    
    Notes:
        - Fairness scores range from 0 (completely unfair) to 1 (perfectly fair)
        - Scores >= 0.8 are generally considered acceptable
        - Scores < 0.6 indicate severe bias requiring immediate attention
        - Small sample sizes may produce unreliable results
        - Progress indicators are displayed during analysis for user feedback
    
    Raises:
        Does not raise exceptions - returns error information in result dict instead
        
    Example:
        >>> fairness_results = analyse_model_fairness(
        ...     X_test=df_test[['age', 'gender']], 
        ...     y_test=df_test['approved'],
        ...     predictions=model.predict(X_test),
        ...     columns_to_analyse=['age', 'gender'],
        ...     problem_type='binary_classification'
        ... )
        >>> print(f"Overall fairness: {fairness_results['fairness_score']:.3f}")
    """
    
    # Enhanced input validation
    if X_test is None or y_test is None or predictions is None:
        return _create_empty_fairness_results("Missing required data")
    
    if len(X_test) != len(y_test) or len(X_test) != len(predictions):
        return _create_empty_fairness_results("Inconsistent data lengths")
    
    if len(X_test) < MIN_SAMPLE_SIZE_WARNING:
        st.warning(f"⚠️ **Small Sample Size Warning**\n\n"
                   f"Your test dataset has only **{len(X_test)} samples**, which may not provide "
                   f"statistically reliable fairness results.\n\n"
                   f"**Recommended minimum:** At least 100 samples total, with at least 10 samples "
                   f"per demographic group.\n\n"
                   f"**Impact:** Results may be unstable and conclusions about bias may not be trustworthy. "
                   f"Consider collecting more data or using a larger test set if possible.")
    
    fairness_results = {
        "success": True,
        "message": "",
        "fairness_score": None,
        "column_scores": {},
        "bias_detected": False,
        "bias_types": [],
        "recommendations": [],
        "metric_frames": {},
        "feature_summaries": {}
    }
    
    try:
        # Define metrics based on problem type - handle both binary and multiclass classification
        if problem_type in ["binary_classification", "multiclass_classification", "classification"]:
            performance_metrics = {
                'accuracy': accuracy_score,
                'precision': safe_precision_score,
                'recall': safe_recall_score,
                'f1': safe_f1_score,
                'auc': safe_auc_score,
                'count': count
            }
            
            # For multiclass, use different fairness metrics that work with multiclass
            if problem_type == "multiclass_classification":
                fairness_metrics = {
                    'selection_rate': selection_rate,  # This works for multiclass
                    'accuracy': accuracy_score,  # Simple fairness check - accuracy across groups
                    'precision': safe_precision_score,  # Macro-averaged precision
                    'recall': safe_recall_score  # Macro-averaged recall
                }
            else:
                # Binary classification fairness metrics
                fairness_metrics = {
                    'selection_rate': selection_rate,
                    'false_positive_rate': false_positive_rate,
                    'false_negative_rate': false_negative_rate,
                    'true_positive_rate': true_positive_rate
                }
        else:  # regression
            performance_metrics = {
                'r2': r2_score,
                'mse': mean_squared_error,
                'rmse': rmse_score,
                'mae': mean_absolute_error,
                'count': count
            }
            
            fairness_metrics = {
                'mean_prediction': lambda y_true, y_pred: np.mean(y_pred),
                'mae': lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
                'mse': lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
                'r2': lambda y_true, y_pred: r2_score(y_true, y_pred)
            }
        
        # Add progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Store binning information for later use
        binning_info = {}
        
        # Analyse each protected attribute
        for i, feature_col in enumerate(columns_to_analyse):
            try:
                # Update progress
                progress = (i + 1) / len(columns_to_analyse)
                status_text.text(f"Analyzing feature {i+1}/{len(columns_to_analyse)}: {feature_col}")
                progress_bar.progress(progress)
                
                # Check if auto-binning should be applied
                feature_data = X_test[feature_col]
                use_binning = (
                    hasattr(st.session_state, 'auto_binning_enabled') and 
                    st.session_state.auto_binning_enabled and
                    feature_data.nunique() > 15
                )
                
                if use_binning:
                    # Apply auto-binning
                    n_bins = getattr(st.session_state, 'auto_binning_n_bins', 6)
                    binned_feature, bin_info = auto_bin_high_cardinality_feature(
                        feature_data, 
                        feature_col, 
                        n_bins=n_bins
                    )
                    
                    if bin_info['was_binned']:
                        feature_data = binned_feature
                        binning_info[feature_col] = bin_info
                        status_text.text(
                            f"Analyzing feature {i+1}/{len(columns_to_analyse)}: {feature_col} "
                            f"(auto-grouped from {bin_info['n_original_groups']} to {bin_info['n_final_groups']} groups)"
                        )
                
                # Create MetricFrame for performance metrics
                perf_metric_frame = MetricFrame(
                    metrics=performance_metrics,
                    y_true=y_test,
                    y_pred=predictions,
                    sensitive_features=feature_data
                )
                
                # Create MetricFrame for fairness metrics (use potentially binned feature_data)
                fairness_metric_frame = MetricFrame(
                    metrics=fairness_metrics,
                    y_true=y_test,
                    y_pred=predictions,
                    sensitive_features=feature_data
                )
                
                # Store both MetricFrames
                fairness_results["metric_frames"][feature_col] = {
                    'performance': perf_metric_frame,
                    'fairness': fairness_metric_frame
                }
                
                # Calculate fairness scores - handle all classification types
                if problem_type in ["binary_classification", "multiclass_classification", "classification"]:
                    if problem_type == "multiclass_classification":
                        # For multiclass, use simpler fairness measures
                        try:
                            # Calculate demographic parity difference (works for multiclass)
                            dp_diff = demographic_parity_difference(
                                y_true=y_test,
                                y_pred=predictions,
                                sensitive_features=feature_data
                            )
                            dp_score = PERFECT_FAIRNESS_SCORE - abs(dp_diff)
                            
                            # For multiclass, use accuracy consistency across groups as fairness measure
                            accuracy_frame = fairness_metric_frame.by_group['accuracy']
                            accuracy_min = accuracy_frame.min()
                            accuracy_max = accuracy_frame.max()
                            accuracy_fairness = accuracy_min / accuracy_max if accuracy_max > 0 else PERFECT_FAIRNESS_SCORE
                            
                            # Use the worse of the two scores
                            fairness_score = min(dp_score, accuracy_fairness)
                            
                            # Store individual scores for display
                            individual_scores = {
                                'demographic_parity': dp_score,
                                'accuracy_consistency': accuracy_fairness,
                                'overall': fairness_score
                            }
                        except Exception as e:
                            # Log error and continue with None values for this feature
                            fairness_score = None
                            individual_scores = {
                                'demographic_parity': None,
                                'accuracy_consistency': None,
                                'overall': None
                            }
                            # Could add logging here if needed: st.session_state.logger.log_error(...)
                    else:
                        # Binary classification - use traditional fairness metrics
                        try:
                            # Calculate demographic parity difference
                            dp_diff = demographic_parity_difference(
                                y_true=y_test,
                                y_pred=predictions,
                                sensitive_features=feature_data
                            )
                            
                            # Calculate equalized odds difference
                            eo_diff = equalized_odds_difference(
                                y_true=y_test,
                                y_pred=predictions,
                                sensitive_features=feature_data
                            )
                            
                            # Convert differences to fairness scores
                            dp_score = PERFECT_FAIRNESS_SCORE - abs(dp_diff)
                            eo_score = PERFECT_FAIRNESS_SCORE - abs(eo_diff)
                            
                            # Use the worse of the two scores
                            fairness_score = min(dp_score, eo_score)
                            
                            # Store individual scores for display
                            individual_scores = {
                                'demographic_parity': dp_score,
                                'equalized_odds': eo_score,
                                'overall': fairness_score
                            }
                        except Exception as e:
                            # Log error and continue with None values for this feature  
                            fairness_score = None
                            individual_scores = {
                                'demographic_parity': None,
                                'equalized_odds': None,
                                'overall': None
                            }
                            # Could add logging here if needed: st.session_state.logger.log_error(...)
                    
                else:  # regression
                    try:
                        # Calculate ratio of group MAEs
                        group_maes = fairness_metric_frame.by_group['mae']
                        max_mae = group_maes.max()
                        min_mae = group_maes.min()
                        
                        if max_mae > 0:
                            mae_ratio = min_mae / max_mae
                        else:
                            mae_ratio = PERFECT_FAIRNESS_SCORE
                            
                        fairness_score = mae_ratio
                        
                        # For regression, we only have one main fairness measure
                        individual_scores = {
                            'error_consistency': fairness_score,
                            'overall': fairness_score
                        }
                    except Exception as e:
                        fairness_score = None
                        individual_scores = {
                            'error_consistency': None,
                            'overall': None
                        }
                
                fairness_results["column_scores"][feature_col] = fairness_score
                
                # Generate enhanced feature summary
                fairness_results["feature_summaries"][feature_col] = {
                    "score": fairness_score,
                    "individual_scores": individual_scores,
                    "groups": feature_data.nunique(),
                    "smallest_group": feature_data.value_counts().min(),
                    "largest_group": feature_data.value_counts().max(),
                    "is_numeric": pd.api.types.is_numeric_dtype(feature_data),
                    "was_binned": feature_col in binning_info,
                    "binning_info": binning_info.get(feature_col, {})
                }
                
                # Check for bias
                if fairness_score is not None and fairness_score < FAIRNESS_THRESHOLD:
                    fairness_results["bias_detected"] = True
                    fairness_results["bias_types"].append(
                        f"Potential bias detected in {feature_col} (score: {fairness_score:.3f})"
                    )
                
            except Exception as e:
                error_message = str(e)
                
                # Provide user-friendly error messages based on error type
                if "not supported between instances of" in error_message:
                    st.error(f"❌ **Cannot analyze feature '{feature_col}'**\n\n"
                             f"**Issue:** This feature contains mixed data types (text and numbers) that cannot be compared directly.\n\n"
                             f"**Common Causes:**\n"
                             f"- Feature was automatically grouped but resulted in inconsistent labels\n"
                             f"- Original data contains mixed text and numeric values\n"
                             f"- Feature engineering created incompatible value types\n\n"
                             f"**Solutions:**\n"
                             f"1. Try disabling automatic grouping for this feature (uncheck 'Use Automatic Grouping')\n"
                             f"2. Pre-process this feature in the Data Preprocessing stage to ensure consistent types\n"
                             f"3. Remove this feature from fairness analysis and analyze other features\n\n"
                             f"**Technical Details (for debugging):** {error_message}")
                elif "insufficient data" in error_message.lower() or "too few" in error_message.lower():
                    st.error(f"❌ **Insufficient data for feature '{feature_col}'**\n\n"
                             f"**Issue:** This feature doesn't have enough samples in each group for reliable analysis.\n\n"
                             f"**Recommendation:** Collect more data or combine small groups together.")
                else:
                    st.error(f"❌ **Failed to analyze feature '{feature_col}'**\n\n"
                             f"**Error Details:** {error_message}\n\n"
                             f"**Possible Causes:**\n"
                             f"- Insufficient data for this feature (too few samples)\n"
                             f"- All groups have identical values (no variance)\n"
                             f"- Data type incompatibility with fairness metrics\n\n"
                             f"**Recommendation:** Try selecting different features or check data quality for '{feature_col}'")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Calculate overall fairness score
        valid_scores = [score for score in fairness_results["column_scores"].values() if score is not None]
        if valid_scores:
            fairness_results["fairness_score"] = min(valid_scores)
        else:
            # If no valid scores, set to None and add explanation
            fairness_results["fairness_score"] = None
            fairness_results["message"] = "Could not calculate fairness scores for any selected features. This may be due to insufficient data variance, incompatible data types, or multiclass prediction format issues."
        
        return fairness_results
        
    except Exception as e:
        fairness_results["success"] = False
        fairness_results["message"] = (f"Fairness analysis failed: {str(e)}. "
                                       f"This may be due to: (1) Incompatible data types, "
                                       f"(2) Insufficient data variance, (3) All features having identical values, "
                                       f"or (4) Prediction format issues for multiclass models. "
                                       f"Please verify your data quality and feature selection.")
        return fairness_results

def _create_empty_fairness_results(message):
    """Create empty fairness results structure with error message."""
    return {
        "success": False,
        "message": message,
        "fairness_score": None,
        "column_scores": {},
        "bias_detected": False,
        "bias_types": [],
        "recommendations": [],
        "metric_frames": {},
        "feature_summaries": {}
    }

def generate_specific_recommendations(fairness_results, X_test, problem_type):
    """
    Generate specific, actionable recommendations for addressing detected bias.
    
    This function analyzes fairness results and generates targeted recommendations
    based on the type of bias detected, the problem type, and feature characteristics.
    Recommendations are prioritized by severity and include effort estimates.
    
    Args:
        fairness_results (dict): Results from analyse_model_fairness() containing
                               fairness scores and feature summaries
        X_test (pd.DataFrame): Test dataset features for analyzing group characteristics
        problem_type (str): ML problem type ('binary_classification', 
                           'multiclass_classification', or 'regression')
    
    Returns:
        list: List of recommendation dictionaries, each containing:
            - feature (str): Name of the biased feature
            - issue (str): Description of the fairness issue detected
            - action (str): Specific recommended action to address the issue
            - priority (str): Priority level ('Critical', 'High', 'Medium', 'Low')
            - effort (str): Estimated effort required ('Low', 'Medium', 'High')
    
    Recommendation Categories:
        Critical Priority:
            - Severe bias (score < 0.6) requiring immediate attention
            - Protected attribute violations with legal implications
            - Major demographic parity or equalized odds violations
            
        High Priority:
            - Moderate bias (score 0.6-0.8) that should be addressed
            - Small group sizes affecting statistical reliability
            - Protected attribute concerns with moderate impact
            
        Medium Priority:
            - High cardinality features needing data preprocessing
            - General fairness improvements for multiclass models
            - Feature engineering suggestions for better fairness
    
    Effort Estimates:
        Low: Data preprocessing, feature grouping, documentation
        Medium: Model retraining, fairness constraints, threshold tuning
        High: Data collection, new model architectures, major system changes
    
    Notes:
        - Only generates recommendations for features with scores < FAIRNESS_THRESHOLD
        - Takes into account problem-specific fairness challenges
        - Considers feature characteristics (cardinality, group sizes, data types)
        - Provides business context and actionable next steps
        
    Example:
        >>> recommendations = generate_specific_recommendations(
        ...     fairness_results={'column_scores': {'age': 0.5, 'gender': 0.9}},
        ...     X_test=test_data,
        ...     problem_type='binary_classification'
        ... )
        >>> for rec in recommendations:
        ...     print(f"{rec['priority']}: {rec['action']}")
    """
    
    recommendations = []
    
    for feature, score in fairness_results["column_scores"].items():
        if score is None or score >= FAIRNESS_THRESHOLD:
            continue
            
        # Analyze the feature characteristics
        feature_data = X_test[feature]
        n_groups = feature_data.nunique()
        group_sizes = feature_data.value_counts()
        summary = fairness_results.get("feature_summaries", {}).get(feature, {})
        individual_scores = summary.get("individual_scores", {})
        
        # Problem-type specific recommendations
        if problem_type == "multiclass_classification":
            # Multiclass-specific recommendations
            if score < SEVERE_BIAS_THRESHOLD:
                if 'demographic_parity' in individual_scores and individual_scores['demographic_parity'] and individual_scores['demographic_parity'] < SEVERE_BIAS_THRESHOLD:
                    recommendations.append({
                        "feature": feature,
                        "issue": "Severe demographic parity violation across classes",
                        "action": f"Review class prediction distributions for '{feature}' groups. Consider class-specific fairness constraints or rebalancing training data",
                        "priority": "Critical",
                        "effort": "High"
                    })
                
                if 'accuracy_consistency' in individual_scores and individual_scores['accuracy_consistency'] and individual_scores['accuracy_consistency'] < SEVERE_BIAS_THRESHOLD:
                    recommendations.append({
                        "feature": feature,
                        "issue": "Large accuracy gaps between groups",
                        "action": f"Investigate why some '{feature}' groups have much lower accuracy. Consider group-specific model tuning or feature engineering",
                        "priority": "Critical",
                        "effort": "High"
                    })
                    
            elif score < 0.8:
                recommendations.append({
                    "feature": feature,
                    "issue": "Moderate bias in multiclass predictions",
                    "action": f"Monitor '{feature}' groups for class-specific disparities. Consider ensemble methods that account for group membership",
                    "priority": "High", 
                    "effort": "Medium"
                })
                
            # Multiclass-specific guidance
            if n_groups > HIGH_CARDINALITY_WARNING_LIMIT:
                recommendations.append({
                    "feature": feature,
                    "issue": "Too many groups for multiclass fairness analysis",
                    "action": f"Group '{feature}' into 3-10 meaningful categories to better understand class-specific bias patterns",
                    "priority": "Medium",
                    "effort": "Low"
                })
                
        else:
            # Binary classification and regression recommendations
            if n_groups > 20:
                recommendations.append({
                    "feature": feature,
                    "issue": "Too many groups for effective analysis",
                    "action": f"Consider binning '{feature}' into fewer categories",
                    "priority": "Medium",
                    "effort": "Low"
                })
        
            if score < SEVERE_BIAS_THRESHOLD:
                recommendations.append({
                    "feature": feature,
                    "issue": "Severe fairness violations",
                    "action": f"Apply fairness constraints during model training",
                    "priority": "Critical",
                    "effort": "Medium"
                })
        
        # Common recommendations for all problem types
        if group_sizes.min() < MIN_SAMPLE_SIZE_WARNING:
            recommendations.append({
                "feature": feature,
                "issue": "Small group sizes affecting reliability",
                "action": f"Collect more data for underrepresented groups in '{feature}'",
                "priority": "High",
                "effort": "High"
            })
        
        # Protected attribute recommendations
        if feature.lower() in ['age', 'income', 'salary', 'gender', 'race', 'ethnicity']:
            if problem_type == "multiclass_classification":
                recommendations.append({
                    "feature": feature,
                        "issue": "Potential indirect discrimination across multiple classes",
                        "action": f"Apply fairness-aware preprocessing and consider class-specific bias mitigation for '{feature}'",
                        "priority": "High",
                    "effort": "Medium"
                })
            else:
                recommendations.append({
                    "feature": feature,
                    "issue": "Potential indirect discrimination",
                        "action": f"Consider fairness-aware preprocessing techniques for '{feature}'",
                    "priority": "High",
                    "effort": "Low"
                    })
    
    # Add general multiclass recommendations if applicable
    if problem_type == "multiclass_classification" and any(score < FAIRNESS_THRESHOLD for score in fairness_results["column_scores"].values() if score is not None):
        recommendations.append({
            "feature": "All features",
            "issue": "Multiclass fairness challenges detected",
            "action": "Consider using fairness-aware multiclass algorithms (e.g., fair multiclass classification with equalized odds)",
            "priority": "Medium",
            "effort": "High"
            })
    
    return recommendations

# === COMPREHENSIVE DASHBOARD ===

def create_comprehensive_fairness_dashboard(fairness_results, problem_type, columns_to_analyse):
    """Create a comprehensive single-page fairness dashboard."""
    
    fairness_score = fairness_results.get("fairness_score", PERFECT_FAIRNESS_SCORE)
    
    # === 0. FAIRNESS THRESHOLD CUSTOMIZATION ===
    st.markdown("## 📊 Fairness Analysis Dashboard")
    
    with st.expander("⚙️ Fairness Settings", expanded=False):
        st.markdown("""
        **Customize your fairness threshold** based on your industry standards and requirements.
        The threshold determines what scores are considered "fair" vs "biased".
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Preset profiles
            preset = st.radio(
                "Choose a preset:",
                ["Standard (0.80)", "Strict (0.85)", "Lenient (0.75)", "Custom"],
                horizontal=True,
                key="fairness_preset"
            )
            
            if preset == "Standard (0.80)":
                threshold_value = 0.80
            elif preset == "Strict (0.85)":
                threshold_value = 0.85
            elif preset == "Lenient (0.75)":
                threshold_value = 0.75
            else:  # Custom
                threshold_value = st.slider(
                    "Custom threshold:",
                    min_value=0.60,
                    max_value=0.95,
                    value=0.80,
                    step=0.05,
                    key="custom_threshold",
                    help="Lower = more lenient, Higher = stricter fairness requirements"
                )
        
        with col2:
            st.metric("Active Threshold", f"{threshold_value:.2f}")
            
            # Show impact of threshold choice
            if fairness_results.get("column_scores"):
                scores = [s for s in fairness_results["column_scores"].values() if s is not None]
                flagged = sum(1 for s in scores if s < threshold_value)
                st.caption(f"🚩 {flagged} of {len(scores)} features flagged")
        
        # Store threshold in session state for use throughout dashboard
        st.session_state.fairness_threshold = threshold_value
        
        # Educational note
        st.info("""
        💡 **Understanding Thresholds:**
        - **0.85 (Strict)**: Recommended for high-stakes applications (healthcare, legal, hiring)
        - **0.80 (Standard)**: Industry standard for most applications
        - **0.75 (Lenient)**: May be acceptable for low-stakes exploratory models
        """)
    
    # Use the threshold throughout the dashboard
    FAIRNESS_THRESHOLD = threshold_value
    
    # === 1. EXECUTIVE SUMMARY ===
    
    # Quick metrics row
    col1, col2, col3, col4 = st.columns(UI_COLUMNS_METRICS)
    
    with col1:
        if fairness_score is None:
            st.metric("Overall Fairness", "N/A", "❌ Analysis Failed")
        elif fairness_score >= EXCELLENT_FAIRNESS_THRESHOLD:
            st.metric("Overall Fairness", f"{fairness_score:.3f}", "🏆 Excellent")
        elif fairness_score >= FAIRNESS_THRESHOLD:
            st.metric("Overall Fairness", f"{fairness_score:.3f}", "✅ Good")
        else:
            st.metric("Overall Fairness", f"{fairness_score:.3f}", "⚠️ Needs Attention")
    
    with col2:
        st.metric("Features Analyzed", len(fairness_results.get("column_scores", {})))
    
    with col3:
        issues = len(fairness_results.get("bias_types", []))
        st.metric("Issues Found", issues)
    
    with col4:
        fair_count = sum(1 for score in fairness_results.get("column_scores", {}).values() 
                        if score and score >= FAIRNESS_THRESHOLD)
        st.metric("Fair Features", f"{fair_count}/{len(fairness_results.get('column_scores', {}))}")

    # === 4. ALL FEATURES SUMMARY ===
    st.markdown("### 📋 All Features Analysis")
    
    if fairness_results.get("column_scores"):
        # Count features by risk level
        critical_count = 0
        review_count = 0
        fair_count = 0
        
        features_data = []
        for feature, score in fairness_results["column_scores"].items():
            if score is None:
                continue
            
            summary = fairness_results.get("feature_summaries", {}).get(feature, {})
            
            # Determine priority and status
            if score >= FAIRNESS_THRESHOLD:
                priority_emoji = "🟢"
                status = "Fair"
                risk = "Low"
                fair_count += 1
            elif score >= SEVERE_BIAS_THRESHOLD:
                priority_emoji = "🟡"
                status = "Review Needed"
                risk = "Medium"
                review_count += 1
            else:
                priority_emoji = "🔴"
                status = "Critical"
                risk = "High"
                critical_count += 1
            
            # Check if feature was auto-binned
            binning_note = ""
            if summary.get('was_binned', False):
                bin_info = summary.get('binning_info', {})
                if bin_info:
                    binning_note = f" (grouped: {bin_info.get('n_original_groups', '?')}→{summary.get('groups', '?')})"
            
            features_data.append({
                "Priority": priority_emoji,
                "Feature": feature + binning_note,
                "Score": score,
                "Fairness Score": f"{score:.3f}",
                "Status": status,
                "Groups": summary.get('groups', 'N/A'),
                "Risk": risk
            })
        
        if features_data:
            # Show quick stats
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                if critical_count > 0:
                    st.error(f"🔴 **{critical_count}** Critical")
                else:
                    st.success(f"✅ **0** Critical")
            with stat_col2:
                if review_count > 0:
                    st.warning(f"🟡 **{review_count}** Review Needed")
                else:
                    st.info(f"✅ **0** Review Needed")
            with stat_col3:
                st.success(f"🟢 **{fair_count}** Fair")
            
            st.markdown("")

            # Create dataframe
            df = pd.DataFrame(features_data)

            # Sort by score (worst first)
            df = df.sort_values('Score', ascending=True)
            
            # Apply top-N limit
            df = df.head(critical_count)
            
            # Drop the numeric score column used for sorting
            display_df = df.drop('Score', axis=1)
            
            # Create styled dataframe
            def highlight_rows(row):
                if row['Risk'] == 'High':
                    return ['background-color: #ffe6e6'] * len(row)
                elif row['Risk'] == 'Medium':
                    return ['background-color: #fff4e6'] * len(row)
                else:
                    return ['background-color: #e6f7e6'] * len(row)
            
            styled_df = display_df.style.apply(highlight_rows, axis=1)
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Add fairness heatmap visualization
            st.markdown("")
            with st.expander("🗺️ Visual Fairness Heatmap", expanded=True):
                st.markdown("""
                This heatmap shows fairness scores across all features at a glance. 
                **Red** areas indicate bias concerns, **orange** areas need review, and **green** areas are fair.
                Hover over cells for details.
                """)
                
                heatmap_fig = create_fairness_heatmap(fairness_results, problem_type)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True, config={'responsive': True})
                else:
                    st.info("Heatmap not available - insufficient data.")

    # === 2. INTERPRETATION ===
    if fairness_score is None:
        st.error("""
        ❌ **Analysis could not be completed.** The fairness analysis failed, likely due to:
        
        **Possible causes:**
        - Insufficient data for reliable analysis
        - All groups have identical predictions (no variance)
        - Data format incompatibility with fairness metrics
        - Selected features may not be suitable for fairness analysis
        
        **What to try:** Select different features or check your data quality.
        """)
    elif fairness_score >= EXCELLENT_FAIRNESS_THRESHOLD:
        if problem_type == "multiclass_classification":
            st.success("""
            🏆 **Excellent!** Your multiclass model treats all groups very fairly across all classes. There's minimal risk of discrimination.
            
            **Real-world impact:** 
            - All groups have similar prediction distributions across all classes
            - All groups receive similar prediction accuracy
            - Very low risk of systematic bias in any class assignment
            """)
        else:
            st.success("""
            🏆 **Excellent!** Your model treats all groups very fairly. There's minimal risk of discrimination.
            
            **Real-world impact:** Your model's decisions are likely to be fair and equitable across all groups.
            """)
    elif fairness_score >= FAIRNESS_THRESHOLD:
        if problem_type == "multiclass_classification":
            st.info("""
            ✅ **Good fairness.** Your multiclass model meets industry standards with only minor differences between groups.
            
            **Real-world impact:** 
            - Low risk of discrimination across classes
            - Some minor variations in class predictions or accuracy exist but are within acceptable ranges
            - Consider monitoring specific classes that may show larger disparities
            """)
        else:
            st.info("""
            ✅ **Good fairness.** Your model meets industry standards with only minor differences between groups.
            
            **Real-world impact:** Low risk of discrimination. Some minor variations exist but are within acceptable ranges.
            """)
    else:
        if problem_type == "multiclass_classification":
            st.warning("""
            ⚠️ **Fairness concerns detected.** Your multiclass model shows concerning differences between groups.
            
            **Real-world impact:** 
            - Some groups may be unfairly disadvantaged in certain class predictions
            - Prediction accuracy may vary significantly between groups
            - Risk of systematic bias in class assignments
            - **Action needed:** Review class-specific disparities and consider model adjustments
            """)
        else:
            st.warning("""
            ⚠️ **Fairness concerns detected.** Your model shows concerning differences between groups.
            
            **Real-world impact:** Some groups may be unfairly disadvantaged by your model's decisions.
            """)

    # === 3. INTERACTIVE REAL-WORLD IMPACT VISUALIZATION ===
    st.markdown("### 🎯 Real-World Impact Analysis")
    
    if fairness_score is not None:
        # Add volume input for customization
        impact_col1, impact_col2 = st.columns([2, 1])
        
        with impact_col1:
            st.markdown("""
            Understand how bias in your model could affect real people in your application. 
            Adjust the monthly volume to match your use case.
            """)
        
        with impact_col2:
            monthly_volume = st.number_input(
                "Monthly Applications:",
                min_value=100,
                max_value=1000000,
                value=1000,
                step=100,
                key="impact_volume",
                help="Adjust this to match your expected application volume"
            )
        
        # Create and display interactive visualization
        impact_fig = create_interactive_impact_visualization(
            fairness_score, 
            fairness_results, 
            problem_type,
            monthly_volume=monthly_volume
        )
        
        if impact_fig:
            st.plotly_chart(impact_fig, use_container_width=True, config={'responsive': True})
            
            # Add brief explanation
            st.caption("""
            💡 **How to interpret this:** The visualization shows the estimated number of people potentially 
            affected by bias in your model. "Potentially Affected" includes anyone who might experience unfair treatment. 
            "Likely Discriminated" is a subset representing those most likely to experience actual harm. 
            The gauge shows your overall fairness score against the industry standard threshold (0.8).
            """)
        else:
            st.info("Impact visualization not available - fairness score could not be calculated.")
    else:
        st.markdown("**Impact analysis not available due to fairness calculation failure.**")

    st.markdown("---")

    # === 5. FEATURE SELECTOR FOR DETAILED ANALYSIS ===
    st.markdown("### 🔍 Performance Metric Analysis")
    selected_feature = st.selectbox(
        "Choose a feature to analyze in detail:",
        columns_to_analyse,
        key="feature_selector",
        help="Select which feature you want to see detailed fairness analysis for."
    )

    # === 6. SELECTED FEATURE DEEP DIVE ===
    st.markdown(f"### 🔍 Detailed Analysis: {selected_feature}")
    
    feature_score = fairness_results["column_scores"].get(selected_feature, PERFECT_FAIRNESS_SCORE)
    feature_info = fairness_results.get("feature_summaries", {}).get(selected_feature, {})
    individual_scores = feature_info.get("individual_scores", {})
    
    # Show overall score prominently
    detail_col1, detail_col2, detail_col3 = st.columns(UI_COLUMNS_STANDARD)
    
    with detail_col1:
        st.metric("Overall Fairness", f"{feature_score:.3f}")
    
    with detail_col2:
        st.metric("Groups", feature_info.get('groups', 'N/A'))
    
    with detail_col3:
        smallest = feature_info.get('smallest_group', 'N/A')
        st.metric("Smallest Group", smallest)
        if isinstance(smallest, int) and smallest < MIN_SAMPLE_SIZE_WARNING:
            st.caption("⚠️ Small sample size")

    # === 7. PERFORMANCE BY GROUP ===
    if selected_feature in fairness_results.get("metric_frames", {}):
        st.markdown("#### 📊 Model Performance by Group")
        
        # === METRIC EXPLAINER SECTION ===
        with st.expander("📚 Understanding Performance Metrics", expanded=False):

            st.markdown("### 🎯 What Each Metric Means")
            
            if problem_type in ["binary_classification", "multiclass_classification", "classification"]:
                st.markdown("""
                **Classification Metrics Explained:**
                
                🎯 **Accuracy** = (Correct Predictions) / (Total Predictions)
                - What it means: Overall percentage of correct predictions
                - Good for: Balanced datasets where all classes are equally important
                - Can be misleading when: Dataset is imbalanced (e.g., 95% negative, 5% positive cases)
                
                🎯 **Precision** = (True Positives) / (True Positives + False Positives) 
                - What it means: Of all positive predictions, how many were actually correct?
                - Good for: When false positives are costly (e.g., spam detection, fraud alerts)
                - High precision = Few false alarms
                
                🎯 **Recall (Sensitivity)** = (True Positives) / (True Positives + False Negatives)
                - What it means: Of all actual positives, how many did we correctly identify?
                - Good for: When missing positives is costly (e.g., disease diagnosis, fraud detection)
                - High recall = Few missed cases
                
                🎯 **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
                - What it means: Harmonic mean of precision and recall
                - Good for: Balancing precision and recall, especially with imbalanced data
                - High F1 = Good balance of precision and recall
                
                🎯 **AUC (Area Under Curve)** = Area under the ROC curve
                - What it means: Model's ability to distinguish between classes at all thresholds
                - Range: 0.5 (random) to 1.0 (perfect)
                - Good for: Overall model discrimination ability
                """)
                
                st.markdown("---")
                st.markdown("### 🤔 Why Metrics Can Seem Contradictory")
                
                st.warning("""
                **Common Scenario: High Accuracy, Low Recall/F1**
                
                **Example:** A loan approval model shows:
                - Group A: Accuracy = 85.9%, Recall = 29%, F1 = 43.5%
                - Group B: Accuracy = 87.2%, Recall = 78%, F1 = 82.1%
                
                **What's happening?** 
                
                🔍 **The Hidden Story:**
                - Group A might have very few people who actually qualify for loans (class imbalance)
                - If only 10% of Group A applicants should get loans, the model can achieve 85.9% accuracy by being very conservative
                - But this conservativeness means it only identifies 29% of qualified applicants (low recall)
                - The F1 score (43.5%) reveals this isn't actually good performance
                
                **Real-world impact:**
                - Group A: 71% of qualified applicants get wrongly rejected
                - Group B: Only 22% of qualified applicants get wrongly rejected
                - This suggests potential bias against Group A
                """)
                
                st.info("""
                **Key Insight:** High accuracy can hide bias!
                
                When groups have different base rates (% of positive cases), accuracy alone can be misleading:
                - A conservative model gets high accuracy on low-positive-rate groups
                - The same model might have lower accuracy but better recall on high-positive-rate groups
                - Always look at precision, recall, and F1 together for the full picture
                """)
                
                st.markdown("### 📊 How to Interpret Cross-Group Differences")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success("""
                    **✅ Good Fairness Indicators:**
                    - Similar F1 scores across groups
                    - Similar recall rates (important for opportunity)
                    - Precision differences ≤ 10%
                    - AUC scores within 5% of each other
                    """)
                
                with col2:
                    st.error("""
                    **⚠️ Potential Bias Indicators:**
                    - Large recall differences (>20%)
                    - F1 scores varying significantly
                    - One group has high accuracy but low recall
                    - AUC differences > 10%
                    """)
                
                st.markdown("""
                **🎯 Practical Guidelines:**
                
                1. **For High-Stakes Decisions (loans, hiring, medical):**
                   - Focus on recall and F1 score differences
                   - Accuracy alone is insufficient
                   - Ensure no group has systematically lower recall
                
                2. **For Screening Applications (fraud, spam):**
                   - Precision is often more important
                   - But still check for fairness in recall
                   - False positive rates should be similar across groups
                
                3. **Red Flags to Watch:**
                   - Group A: High accuracy, low recall → Model is too conservative for this group
                   - Group B: Lower accuracy, high recall → Model might be too liberal for this group
                   - Large F1 differences → Unequal treatment quality
                """)
                
            else:  # regression
                st.markdown("""
                **Regression Metrics Explained:**
                
                🎯 **R² (R-squared)** = 1 - (Sum of Squared Residuals) / (Total Sum of Squares)
                - What it means: Percentage of variance in target explained by the model
                - Range: 0% to 100% (higher is better)
                - Good for: Understanding overall model fit
                
                🎯 **MSE (Mean Squared Error)** = Average of (Actual - Predicted)²
                - What it means: Average squared difference between actual and predicted values
                - Units: Squared units of your target variable
                - Good for: Penalizing large errors more heavily
                
                🎯 **RMSE (Root Mean Squared Error)** = √(MSE)
                - What it means: Average error in the same units as your target
                - Units: Same as your target variable
                - Good for: Interpretable error magnitude
                
                🎯 **MAE (Mean Absolute Error)** = Average of |Actual - Predicted|
                - What it means: Average absolute difference between actual and predicted
                - Units: Same as your target variable  
                - Good for: Understanding typical error size (less sensitive to outliers)
                """)
                
                st.markdown("### 🤔 Why Regression Metrics Can Vary Across Groups")
                
                st.warning("""
                **Common Scenario: Different Error Patterns**
                
                **Example:** A salary prediction model shows:
                - Group A: R² = 0.85, RMSE = $8,000, MAE = $5,000
                - Group B: R² = 0.72, RMSE = $15,000, MAE = $12,000
                
                **What's happening?**
                - The model explains variance much better for Group A (85% vs 72%)
                - Group B has much larger prediction errors on average
                - This suggests the model learned patterns that work well for Group A but not Group B
                
                **Potential causes:**
                - Training data had more examples from Group A
                - Different salary structures or career patterns between groups
                - Model features work better for predicting Group A salaries
                """)
                
                st.info("""
                **Key Insight:** Similar R² doesn't guarantee fair errors!
                
                Even if R² scores are similar, check:
                - Are RMSE/MAE values similar? (error magnitude)
                - Are errors consistently over/under-predicting for any group?
                - Do some groups have more variable errors than others?
                """)
            
            st.markdown("---")
            st.markdown("### 🎯 Action Items Based on Metric Patterns")
            
            if problem_type == "classification":
                st.markdown("""
                **If you see high accuracy but low recall/F1 for a group:**
                1. Check class balance - is this group's positive rate much lower?
                2. Consider adjusting decision threshold for this group
                3. Collect more positive examples for this group
                4. Use fairness-aware training methods
                
                **If you see consistent metric differences across groups:**
                1. Investigate root causes (data quality, representation, feature relevance)
                2. Consider group-specific models or post-processing adjustments
                3. Document and monitor these differences
                4. Consult with domain experts about acceptable trade-offs
                """)
            else:
                st.markdown("""
                **If you see large RMSE/MAE differences:**
                1. Check if groups have different target distributions
                2. Consider group-specific feature engineering
                3. Investigate if model assumptions hold for all groups
                4. Consider ensemble methods or group-specific models
                
                **If you see systematic over/under-prediction:**
                1. Check for bias in training data representation
                2. Consider calibration techniques
                3. Investigate feature interactions that might favor certain groups
                4. Add group-aware regularization during training
                """)
        
        metric_frame = fairness_results["metric_frames"][selected_feature]
        if 'performance' in metric_frame:
            perf_data = metric_frame['performance'].by_group
            
            # Use enhanced group comparison visualizations
            # Get fairness threshold from session state if available
            fairness_threshold = getattr(st.session_state, 'fairness_threshold', FAIRNESS_THRESHOLD)
            
            # Generate enhanced visualizations
            enhanced_figs = create_enhanced_group_comparison(
                perf_data, 
                selected_feature, 
                problem_type,
                fairness_threshold=fairness_threshold
            )
            
            # Display the figures
            if enhanced_figs:
                for fig in enhanced_figs:
                    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
            else:
                st.info("Unable to generate visualizations for this feature.")
            
            # Performance table (for both classification and regression)
            display_perf = perf_data.copy()
            if 'count' in display_perf.columns:
                display_perf = display_perf.drop('count', axis=1)
            st.dataframe(display_perf.style.format("{:.3f}"), width='stretch')
            
            # Show group distribution (for both classification and regression)
            if 'count' in perf_data.columns:
                st.markdown("**Sample Distribution:**")
                total_samples = perf_data['count'].sum()
                dist_data = []
                for group, count in perf_data['count'].items():
                    percentage = (count / total_samples) * 100
                    dist_data.append({
                        "Group": str(group),
                        "Count": f"{count:,}",
                        "Percentage": f"{percentage:.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(dist_data), width='stretch', hide_index=True)

    st.markdown("---")

    # === 8. ACTIONABLE RECOMMENDATIONS ===
    st.markdown("### 💡 What You Should Do")
    
    recs = generate_specific_recommendations(fairness_results, st.session_state.builder.X_test, problem_type)
    
    if recs:
        # Group by priority and show clearly
        critical = [r for r in recs if r.get("priority") == "Critical"]
        high = [r for r in recs if r.get("priority") == "High"]
        medium = [r for r in recs if r.get("priority") == "Medium"]
        
        if critical:
            st.error("🚨 **Critical Actions Needed:**")
            for i, rec in enumerate(critical, 1):
                st.markdown(f"{i}. **{rec['action']}** (Feature: {rec['feature']})")
        
        if high:
            st.warning("⚠️ **High Priority Actions:**")
            for i, rec in enumerate(high, 1):
                st.markdown(f"{i}. **{rec['action']}** (Feature: {rec['feature']})")
        
        if medium:
            st.info("ℹ️ **Consider These Improvements:**")
            for i, rec in enumerate(medium, 1):
                st.markdown(f"{i}. **{rec['action']}** (Feature: {rec['feature']})")
    else:
        st.success("🎉 No specific actions needed - your model shows good fairness!")
        
        # Show best practices even for fair models
        st.markdown("""
        #### 📚 Best Practices for Maintaining Fairness
        
        - **Regular Monitoring:** Check fairness metrics monthly with new data
        - **Documentation:** Keep records of fairness assessments and decisions  
        - **Stakeholder Review:** Have domain experts validate your fairness conclusions
        - **Continuous Learning:** Stay updated on fairness research and regulations
        """)
    
    # === 9. EXPORT FAIRNESS REPORT ===
    st.markdown("---")
    st.markdown("### 📥 Export Fairness Report")
    st.markdown("Save your fairness analysis in various formats for documentation and sharing.")
    
    export_col1, export_col2 = st.columns(2)
    
    # Pre-generate reports
    json_report = export_fairness_report(fairness_results, problem_type, columns_to_analyse, format='json')
    html_report = export_fairness_report(fairness_results, problem_type, columns_to_analyse, format='html')
    md_report = export_fairness_report(fairness_results, problem_type, columns_to_analyse, format='markdown')

    with export_col1:
        report_format = st.selectbox("Select Report Format:", ["JSON", "HTML", "Markdown"], key="report_format",
                                     help="Choose the format for your report.")
    
    with export_col2:
        if report_format == "JSON":
            st.download_button("Download JSON Report", json_report, "fairness_report.json", "application/json")
        elif report_format == "HTML":
            st.download_button("Download HTML Report", html_report, "fairness_report.html", "text/html")
        elif report_format == "Markdown":
            st.download_button("Download Markdown Report", md_report, "fairness_report.md", "text/markdown")

# === MAIN RENDER FUNCTION ===

def render_fairness_analysis():
    """Main function to render the fairness analysis interface."""
    
    st.markdown("# 🎯 Model Fairness Analysis")
    
    # Get problem type to customize introduction
    if hasattr(st.session_state, 'problem_type'):
        problem_type = st.session_state.problem_type
    else:
        problem_type = st.session_state.builder.model.get("problem_type", "unknown")
    
    # Introduction with problem-type specific content
    if problem_type == "multiclass_classification":
        st.markdown("""
            ### 🤔 What is Model Fairness for Multiclass Classification?
            
            Model fairness ensures your model treats all groups fairly across **all classes** without discrimination. 
            For multiclass models, fairness is more complex because we need to ensure:
            
            - **Equal representation**: All groups have similar prediction distributions across all classes
            - **Consistent accuracy**: All groups receive predictions of similar quality
            - **No systematic bias**: No group is consistently favored or disadvantaged for any particular class
            
            **Example**: In a medical diagnosis model predicting 3 conditions (Healthy, Condition A, Condition B), 
            fairness means that patients from different demographic groups have:
            - Similar chances of being diagnosed with each condition (when appropriate)
            - Similar diagnostic accuracy across all conditions
            
            This analysis helps you identify potential bias and provides actionable recommendations.
        """)
    elif problem_type == "binary_classification":
        st.markdown("""
            ### 🤔 What is Model Fairness for Binary Classification?
            
            Model fairness ensures your model treats all groups fairly without discrimination. 
            For binary classification, this means ensuring:
            
            - **Equal opportunity**: Qualified individuals from all groups have similar chances of positive outcomes
            - **Equal treatment**: Similar false positive and false negative rates across groups
            - **Demographic parity**: Overall positive prediction rates are similar across groups
            
            This analysis helps you identify potential bias and provides actionable recommendations.
        """)
    else:
        st.markdown("""
            ### 🤔 What is Model Fairness?
            Model fairness ensures your model treats all groups fairly without discrimination. 
            This analysis helps you identify potential bias and provides actionable recommendations.
        """)
    
    # Get data and validate
    if not hasattr(st.session_state.builder, 'X_test') or st.session_state.builder.X_test is None:
        st.error("❌ **No Test Data Available for Fairness Analysis**\n\n"
                 "**What This Means:** The fairness analysis requires test data (X_test) to evaluate "
                 "how your model performs across different demographic groups.\n\n"
                 "**To Fix This Issue:**\n"
                 "1. Complete the data preprocessing step to generate train/test splits\n"
                 "2. Train a model in the Model Training section\n"
                 "3. Ensure your model has been evaluated on test data\n\n"
                 "**Next Steps:** Go back to the Data Preprocessing page to split your data, "
                 "then proceed through Model Selection and Training before returning here.")
        return
    
    # Process columns
    all_available_columns = st.session_state.builder.X_test.columns.tolist()
    protected_cols = st.session_state.builder.get_protected_attributes()
    
    valid_columns = []
    filtered_columns = []
    for col in all_available_columns:
        if not is_metric_column(col):
            valid_columns.append(col)
        else:
            filtered_columns.append(col)
    
    # === CLEAN FEATURE SELECTION INTERFACE ===
    columns_to_analyse = create_clean_feature_selection_interface(
        valid_columns, protected_cols, filtered_columns
    )
    
    if not columns_to_analyse:
        st.info("👆 Please select features to analyze above.")
        st.stop()
    
    # Add educational content about fairness metrics
    with st.expander("📚 Understanding Fairness Metrics for Your Model Type", expanded=True):
        if problem_type == "multiclass_classification":
            st.markdown("""
            ## Fairness Metrics for Multiclass Classification
            
            Since your model predicts multiple classes, we use specialized fairness measures:
            
            ### 🎯 **Demographic Parity**
            - **What it measures**: Whether all groups have similar prediction distributions across all classes
            - **Why it matters**: Ensures no group is systematically over- or under-represented in any class
            - **Example**: In a hiring model with outcomes (Reject, Interview, Hire), demographic parity ensures that the percentage of each outcome is similar across demographic groups
            
            ### 📊 **Accuracy Consistency** 
            - **What it measures**: Whether all groups receive predictions of similar quality
            - **Why it matters**: Ensures no group gets systematically worse predictions
            - **Calculation**: Ratio of minimum group accuracy to maximum group accuracy
            - **Example**: If Group A has 85% accuracy and Group B has 90% accuracy, consistency = 0.94
            
            ### 🎯 **Overall Fairness Score**
            - Takes the minimum of Demographic Parity and Accuracy Consistency
            - **Score ≥ 0.8**: Generally considered fair
            - **Score < 0.8**: May indicate bias requiring attention
            
            ### ⚠️ **Why we don't use traditional binary fairness metrics**:
            - **False Positive/Negative Rates**: Don't directly apply to multiclass (which "positive" class?)
            - **Equalized Odds**: Designed for binary outcomes
            - **ROC-AUC**: Requires probability scores, complex for multiclass
            """)
        elif problem_type == "binary_classification":
            st.markdown("""
            ## Fairness Metrics for Binary Classification
            
            For binary classification, we use well-established fairness measures:
            
            ### 🎯 **Demographic Parity**
            - **What it measures**: Whether all groups have similar positive prediction rates
            - **Why it matters**: Ensures equal representation in positive outcomes
            
            ### ⚖️ **Equalized Odds**
            - **What it measures**: Whether true positive and false positive rates are similar across groups
            - **Why it matters**: Ensures equal treatment quality across groups
            
            ### 🎯 **Overall Fairness Score**
            - Takes the minimum of Demographic Parity and Equalized Odds scores
            - **Score ≥ 0.8**: Generally considered fair
            - **Score < 0.8**: May indicate bias requiring attention
            """)
        else:
            st.markdown("""
            ## Fairness Metrics for Regression
            
            For regression models, fairness focuses on prediction quality:
            
            ### 📊 **Error Consistency**
            - **What it measures**: Whether prediction errors are similar across groups
            - **Why it matters**: Ensures no group receives systematically worse predictions
            """)
        
        st.markdown("""
        ---
        ### 💡 **General Interpretation Guidelines**
        
        - **Scores near 1.0**: Excellent fairness
        - **Scores 0.8-0.9**: Good fairness, acceptable for most applications
        - **Scores 0.6-0.8**: Moderate bias, should be investigated
        - **Scores below 0.6**: Significant bias, requires immediate attention
        
        Remember: Perfect fairness (score = 1.0) is rare in real-world data. The key is understanding 
        whether observed differences are acceptable for your specific use case and legal/ethical requirements.
        """)
    
    # Get problem type from session state if available
    if hasattr(st.session_state, 'problem_type'):
        problem_type = st.session_state.problem_type
    else:
        # Fallback to model's problem type
        problem_type = st.session_state.builder.model.get("problem_type", "unknown")
    
    # === RUN ANALYSIS ===
    st.markdown("---")
    
    with st.spinner("🔍 Analyzing fairness across selected features..."):
        # Get predictions in the right format
        model = st.session_state.builder.model["model"]
        X_test = st.session_state.builder.X_test
        predictions = model.predict(X_test)
        

        
        fairness_results = analyse_model_fairness(
            X_test, 
            st.session_state.builder.y_test, 
            predictions, 
            columns_to_analyse, 
            problem_type
        )
    
    if not fairness_results["success"]:
        st.error(f"❌ **Fairness Analysis Failed**\n\n"
                 f"**Error Details:** {fairness_results['message']}\n\n"
                 f"**What You Can Try:**\n"
                 f"1. Select different features with more data variation\n"
                 f"2. Check that your selected features have multiple distinct groups\n"
                 f"3. Ensure your model predictions are in the correct format\n"
                 f"4. Verify that test data contains the selected features\n\n"
                 f"**Need Help?** Use the Custom Selection method to choose features with "
                 f"fewer unique values and larger group sizes.")
        return
    
    # === COMPREHENSIVE DASHBOARD ===
    create_comprehensive_fairness_dashboard(
        fairness_results, problem_type, columns_to_analyse
    )
    
    # Log the analysis
    if fairness_results.get("column_scores"):
        score_data = []
        for feature, score in fairness_results["column_scores"].items():
            if score is not None:
                score_data.append({
                    "Feature": feature,
                    "Score": score,
                    "Status": "Fair" if score >= FAIRNESS_THRESHOLD else "Bias Risk"
                })
        
        if hasattr(st.session_state, 'logger'):
            st.session_state.logger.log_journey_point(
                stage="MODEL_EXPLANATION",
                decision_type="MODEL_EXPLANATION",
                description="Fairness Analysis Completed",
                details={"Fairness Results": score_data},
                parent_id=None
            ) 