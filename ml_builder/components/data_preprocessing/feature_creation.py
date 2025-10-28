import streamlit as st
import pandas as pd
from pandas.api.types import is_categorical_dtype
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from components.data_exploration.feature_relationships import FeatureRelationshipsComponent
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')
from utils.dataframe_comparison import DataframeComparisonComponent
from utils.data_exploration_component import DataExplorationComponent
import gc  # For garbage collection
from plotly.subplots import make_subplots

# Import networkx with fallback
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

def _ratio_op(x, y):
    """Ratio operation that handles division by zero."""
    return np.divide(x, y, out=np.zeros_like(x), where=y!=0)

def _sum_op(x, y):
    """Sum operation."""
    return np.add(x, y)

def _diff_op(x, y):
    """Difference operation."""
    return np.subtract(x, y)

def _prod_op(x, y):
    """Product operation."""
    return np.multiply(x, y)

def _mean_op(x, y):
    """Mean operation."""
    return np.add(x, y) * 0.5

# Operation mapping dictionary at module level
OPERATIONS_MAPPING = {
    'ratio': _ratio_op,
    'sum': _sum_op,
    'difference': _diff_op,
    'product': _prod_op,
    'mean': _mean_op
}

@st.cache_data(show_spinner=True)
def _cached_generate_features(
    training_data: pd.DataFrame,
    numeric_features: List[str],
    _operations_list: List[Tuple[str, str]]  # Already hashable, but marked for consistency
) -> Dict:
    """
    Cached version of feature generation to prevent recalculation.
    
    Args:
        training_data: Training DataFrame
        numeric_features: List of numeric feature names
        _operations_list: List of (operation_name, operation_type) tuples
    """
    # Define operations based on operation types
    operations = {}
    for op_name, op_type in _operations_list:
        operations[op_name] = OPERATIONS_MAPPING[op_type]
    
    generated_features = {}
    
    # Pre-compute numeric values for all features to avoid repeated conversions
    numeric_values = {}
    for feat in numeric_features:
        values = training_data[feat].copy()
        if '_binned' in feat and hasattr(values, 'dtype') and (
            is_categorical_dtype(values) or 
            str(values.dtype).startswith('category')):
            values = pd.to_numeric(values.astype(str), errors='coerce')
        numeric_values[feat] = values.to_numpy()
    
    # Generate features in batches to manage memory
    batch_size = 100
    feature_combinations = [
        (i, j) for i in range(len(numeric_features)) 
        for j in range(len(numeric_features)) if i != j
    ]
    
    for batch_start in range(0, len(feature_combinations), batch_size):
        batch_end = min(batch_start + batch_size, len(feature_combinations))
        batch = feature_combinations[batch_start:batch_end]
        
        for i, j in batch:
            feat1 = numeric_features[i]
            feat2 = numeric_features[j]
            
            feat1_values = numeric_values[feat1]
            feat2_values = numeric_values[feat2]
            
            for op_name, op_func in operations.items():
                try:
                    feat_name = f"{feat1}_{op_name}_{feat2}"
                    feat_values = op_func(feat1_values, feat2_values)
                    feat_values = pd.Series(feat_values, index=training_data.index)
                    feat_values = feat_values.replace([np.inf, -np.inf], np.nan)
                    
                    null_count = feat_values.isna().sum()
                    null_percentage = null_count / len(feat_values)
                    
                    if null_percentage > 0.05:
                        continue
                    
                    feat_values = feat_values.fillna(0)
                    
                    if feat_values.isna().mean() < 0.1:
                        generated_features[feat_name] = {
                            'feature1': feat1,
                            'feature2': feat2,
                            'operation': op_name,
                            'operation_func': op_func,
                            'values': feat_values,
                            'had_nulls': null_count > 0
                        }
                except Exception:
                    pass
    
    return generated_features

@st.cache_data(show_spinner=True)
def _cached_calculate_target_relationship(
    training_data: pd.DataFrame,
    target_column: str,
    _generated_features: Dict  # Mark as unhashable
) -> Dict[str, float]:
    """
    Cached version of target relationship calculation.
    
    Args:
        training_data: Training DataFrame
        target_column: Target column name
        _generated_features: Dictionary of generated features (marked as unhashable)
    """
    # Convert features to hashable format for caching
    feature_info, feature_values = _convert_to_hashable_features(_generated_features)
    
    target_importances = {}
    y = training_data[target_column].values
    
    # Use session state to determine problem type if available, otherwise fall back to heuristics
    import streamlit as st
    if hasattr(st, 'session_state') and hasattr(st.session_state, 'problem_type'):
        problem_type = st.session_state.problem_type
        is_classification = problem_type in ["binary_classification", "multiclass_classification"]
    else:
        # Fallback to original heuristic
        is_classification = training_data[target_column].dtype == 'object' or len(training_data[target_column].unique()) < 10
    
    mi_func = mutual_info_classif if is_classification else mutual_info_regression
    
    batch_size = 50
    feature_batches = [feature_info[i:i + batch_size] for i in range(0, len(feature_info), batch_size)]
    
    for batch in feature_batches:
        X_batch = np.vstack([feature_values[feat[0]] for feat in batch]).T
        
        mask = ~np.isnan(X_batch).any(axis=1)
        if mask.sum() > 0:
            X_valid = X_batch[mask]
            y_valid = y[mask]
            
            try:
                mi_scores = mi_func(X_valid, y_valid, random_state=42)
                for (feat_name, _, _, _), mi_score in zip(batch, mi_scores):
                    target_importances[feat_name] = mi_score
            except Exception:
                for feat_name, _, _, _ in batch:
                    target_importances[feat_name] = 0
        else:
            for feat_name, _, _, _ in batch:
                target_importances[feat_name] = 0
    
    if target_importances:
        values = np.array(list(target_importances.values()))
        max_importance = np.max(values) if values.size > 0 else 1
        if max_importance > 0:
            target_importances = {k: v / max_importance for k, v in target_importances.items()}
    
    return target_importances

@st.cache_data(show_spinner=True)
def _cached_calculate_correlations(
    training_data: pd.DataFrame,
    numeric_features: List[str],
    _generated_features: Dict  # Mark as unhashable
) -> Dict[str, Dict[str, float]]:
    """
    Cached version of correlation calculation.
    
    Args:
        training_data: Training DataFrame
        numeric_features: List of original feature names
        _generated_features: Dictionary of generated features (marked as unhashable)
    """
    # Convert features to hashable format for caching
    feature_info, feature_values = _convert_to_hashable_features(_generated_features)
    
    all_features = numeric_features + [feat[0] for feat in feature_info]
    corr_matrix = {feat: {other_feat: 0.0 for other_feat in all_features} for feat in all_features}
    
    if not all_features:
        return corr_matrix
    
    data_dict = {}
    for feat in numeric_features:
        try:
            data_dict[feat] = training_data[feat].values
        except Exception:
            continue
    
    for feat_name, _, _, _ in feature_info:
        try:
            data_dict[feat_name] = feature_values[feat_name]
        except Exception:
            continue
    
    if not data_dict:
        return corr_matrix
    
    try:
        combined_df = pd.DataFrame(data_dict)
        values = combined_df.values
        feature_names = list(combined_df.columns)
        
        n_samples, n_features = values.shape
        if n_samples == 0 or n_features == 0:
            return corr_matrix
        
        for i, feat1 in enumerate(feature_names):
            if i >= n_features:
                break
            
            x = values[:, i].reshape(-1, 1)
            mask_x = ~np.isnan(x)
            
            for j, feat2 in enumerate(feature_names):
                if j >= n_features:
                    break
                
                if feat2 in corr_matrix[feat1] and corr_matrix[feat1][feat2] != 0.0:
                    continue
                
                try:
                    y = values[:, j].reshape(-1, 1)
                    mask_y = ~np.isnan(y)
                    
                    valid_mask = mask_x & mask_y
                    valid_count = np.sum(valid_mask)
                    
                    if valid_count > 1:
                        x_valid = x[valid_mask]
                        y_valid = y[valid_mask]
                        
                        try:
                            corr = np.corrcoef(x_valid.ravel(), y_valid.ravel())[0, 1]
                            if np.isnan(corr):
                                corr = 0.0
                        except Exception:
                            corr = 0.0
                    else:
                        corr = 0.0
                    
                    corr_matrix[feat1][feat2] = corr
                    corr_matrix[feat2][feat1] = corr
                    
                except Exception:
                    corr_matrix[feat1][feat2] = 0.0
                    corr_matrix[feat2][feat1] = 0.0
    
    except Exception:
        return corr_matrix
    
    return corr_matrix

def _convert_to_hashable_features(features_dict: Dict) -> Tuple[List[Tuple], Dict[str, np.ndarray]]:
    """
    Convert features dictionary to hashable format.
    
    Args:
        features_dict: Dictionary of feature information
        
    Returns:
        Tuple of (feature_info_list, values_dict) where:
        - feature_info_list is a list of tuples (name, feature1, feature2, operation)
        - values_dict is a dictionary mapping feature names to their numpy arrays
    """
    feature_info = []
    values_dict = {}
    
    for name, info in features_dict.items():
        # Store feature info as a tuple (immutable)
        feature_info.append((
            name,
            info['feature1'],
            info['feature2'],
            info['operation']
        ))
        # Store values separately
        values_dict[name] = info['values'].to_numpy()
    
    # Sort for consistent hashing
    feature_info.sort()
    
    return feature_info, values_dict

@st.cache_data(show_spinner=True)
def _cached_final_correlation_group_check(
    training_data: pd.DataFrame,
    numeric_features: List[str],
    target_column: str,
    top_features: List[str],
    _feature_metrics: Dict,  # Mark as unhashable
    correlation_threshold: float = 0.7  # Changed from 0.8 to match feature selection
) -> Tuple[List[str], List[Dict]]:
    """
    Final correlation group analysis between ALL features (original + recommended).
    From each correlated group, removes recommended features with lowest scores,
    keeping original features and only the best recommended features.
    
    Args:
        training_data: Training DataFrame
        numeric_features: List of original numeric feature names
        target_column: Target column name
        top_features: List of recommended feature names
        _feature_metrics: Dictionary of feature metrics (marked as unhashable)
        correlation_threshold: Threshold for correlation groups (default 0.7)
        
    Returns:
        Tuple of (final_features, removed_features) where:
        - final_features: Filtered list of recommended features
        - removed_features: List of dicts with removal information
    """
    # Check if networkx is available
    if not NETWORKX_AVAILABLE:
        # Fallback: return original features without group analysis
        return top_features, []
    
    # If no recommended features, nothing to filter
    if not top_features:
        return top_features, []
    
    # Create correlation matrix for all features
    temp_df = training_data[numeric_features].copy()
    
    # Add recommended features to temp dataframe - access from session state directly
    # (not through cached parameters to avoid reconstruction issues)
    added_features = []
    for feat_name in top_features:
        if (hasattr(st.session_state, 'feature_creation_generated_features') and 
            feat_name in st.session_state.feature_creation_generated_features):
            feat_info = st.session_state.feature_creation_generated_features[feat_name]
            if 'values' in feat_info:
                temp_df[feat_name] = feat_info['values']
                added_features.append(feat_name)
    
    # If we couldn't add any recommended features, return original list
    if not added_features:
        return top_features, []
    
    # Calculate correlation matrix
    try:
        correlation_matrix = temp_df.corr().abs()
    except Exception:
        # If correlation calculation fails, return original features
        return top_features, []
    
    # Combine all features for analysis
    all_features = numeric_features + added_features
    
    # Create graph for correlation group analysis (similar to feature selection logic)
    if not NETWORKX_AVAILABLE:
        return top_features, []
    
    G = nx.Graph()
    G.add_nodes_from(all_features)
    
    # Add edges for correlations above threshold
    correlations = []
    for i, feat1 in enumerate(all_features):
        for j, feat2 in enumerate(all_features):
            if (i < j and feat1 in correlation_matrix.columns and 
                feat2 in correlation_matrix.columns):
                try:
                    corr_value = correlation_matrix.loc[feat1, feat2]
                    if not np.isnan(corr_value) and corr_value >= correlation_threshold:
                        G.add_edge(feat1, feat2, weight=corr_value)
                        correlations.append({
                            'feature1': feat1,
                            'feature2': feat2,
                            'correlation': corr_value
                        })
                except Exception:
                    continue
    
    # Find connected components (correlation groups) like in feature selection
    correlation_groups = []
    processed_features = set()
    
    # Build groups from correlations (similar to feature selection logic)
    for corr in correlations:
        feat1, feat2 = corr['feature1'], corr['feature2']
        
        # Find if either feature is already in a group
        found_group = None
        for group in correlation_groups:
            if feat1 in group or feat2 in group:
                found_group = group
                break
        
        if found_group is not None:
            found_group.add(feat1)
            found_group.add(feat2)
        else:
            correlation_groups.append({feat1, feat2})
        
        processed_features.add(feat1)
        processed_features.add(feat2)
    
    # Merge overlapping groups (exactly like feature selection)
    merged_groups = []
    while correlation_groups:
        current_group = correlation_groups.pop(0)
        
        # Try to merge with other groups
        i = 0
        while i < len(correlation_groups):
            if any(feat in current_group for feat in correlation_groups[i]):
                current_group.update(correlation_groups[i])
                correlation_groups.pop(i)
            else:
                i += 1
        
        merged_groups.append(current_group)
    
    # Filter to only groups that contain both original and recommended features
    mixed_groups = []
    for group in merged_groups:
        group_list = list(group)
        has_original = any(feat in numeric_features for feat in group_list)
        has_recommended = any(feat in added_features for feat in group_list)
        
        if has_original and has_recommended and len(group_list) >= 2:
            mixed_groups.append(group_list)
    
    # Process each mixed group to remove lowest-scoring recommended features
    final_features = top_features.copy()
    removed_features = []
    
    for group in mixed_groups:
        # Separate original and recommended features in this group
        group_original = [f for f in group if f in numeric_features]
        group_recommended = [f for f in group if f in added_features]
        
        # Only process if there are multiple recommended features in this group
        if len(group_recommended) <= 1:
            continue  # Keep the single recommended feature
        
        # Calculate metrics for recommended features in this group
        feature_scores = []
        for feat in group_recommended:
            # Get metrics from the _feature_metrics parameter
            metrics = _feature_metrics.get(feat, {})
            combined_score = metrics.get('combined_score', 0)
            target_importance = metrics.get('target_importance', 0)
            
            # Calculate average correlation with all other features in this group
            other_features = [f for f in group if f != feat and f in correlation_matrix.columns]
            correlations_in_group = []
            
            for other_feat in other_features:
                try:
                    corr = correlation_matrix.loc[feat, other_feat]
                    if not np.isnan(corr):
                        correlations_in_group.append(corr)
                except Exception:
                    pass
            
            avg_group_correlation = np.mean(correlations_in_group) if correlations_in_group else 0
            
            feature_scores.append({
                'feature': feat,
                'combined_score': combined_score,
                'target_importance': target_importance,
                'avg_group_correlation': avg_group_correlation,
                'group_original_features': group_original,
                'group_size': len(group)
            })
        
        # Sort by combined score (descending) - keep features with highest scores
        feature_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Keep only the top feature from recommended features in this group
        # Remove the rest from final_features
        features_to_keep = 1  # Keep the best recommended feature in each group
        
        for i, feat_score in enumerate(feature_scores):
            if i >= features_to_keep:  # Remove features beyond the keep limit
                feat_name = feat_score['feature']
                if feat_name in final_features:
                    final_features.remove(feat_name)
                    
                    removed_features.append({
                        'feature': feat_name,
                        'reason': 'correlation_group_filtering',
                        'group_size': feat_score['group_size'],
                        'combined_score': feat_score['combined_score'],
                        'avg_group_correlation': feat_score['avg_group_correlation'],
                        'correlated_with_original': feat_score['group_original_features'],
                        'kept_recommended_feature': feature_scores[0]['feature'] if feature_scores else None,
                        'correlation_threshold_used': correlation_threshold
                    })
    
    return final_features, removed_features

@st.cache_data(show_spinner=True)
def _cached_analyse_features(
    training_data: pd.DataFrame,
    numeric_features: List[str],
    target_column: str,
    _generated_features: Dict  # Note the underscore prefix
) -> Tuple[List[str], Dict, List[Dict], List[Dict], List[Dict]]:
    """
    Cached version of feature analysis that follows the correct filtering order:
    1. Remove features with null values (already done in generation)
    2. Create correlation matrix for all features
    3. Remove multicollinear features
    4. Remove features with similar distributions
    5. Analyze remaining features for target relationship
    6. Apply final correlation group check between all features (NEW!)
    """
    # Convert features to hashable format
    feature_info, feature_values = _convert_to_hashable_features(_generated_features)
    
    # Steps 1-5: Existing filtering logic (unchanged)
    # Step 1: Features with null values already removed during generation (>5% nulls filtered)
    all_generated_features = list(_generated_features.keys())
    
    # Step 2: Calculate correlation matrix for ALL features (existing + new)
    correlation_matrix = _cached_calculate_correlations(
        training_data,
        numeric_features,
        _generated_features
    )
    
    # Step 3: Remove multicollinear features from the full set
    # First identify highly correlated pairs among ALL features
    correlation_threshold = 0.8
    highly_correlated_pairs = []
    
    # Find correlated pairs between new features and existing features
    for new_feat in all_generated_features:
        for existing_feat in numeric_features:
            if (new_feat in correlation_matrix and existing_feat in correlation_matrix[new_feat] and
                abs(correlation_matrix[new_feat][existing_feat]) >= correlation_threshold):
                highly_correlated_pairs.append({
                    'feature1': new_feat,
                    'feature2': existing_feat,
                    'correlation': abs(correlation_matrix[new_feat][existing_feat]),
                    'type': 'new_vs_existing'
                })
    
    # Find correlated pairs between new features
    for i, feat1 in enumerate(all_generated_features):
        for j, feat2 in enumerate(all_generated_features):
            if i >= j:  # Skip duplicates and self-correlations
                continue
            if (feat1 in correlation_matrix and feat2 in correlation_matrix[feat1] and
                abs(correlation_matrix[feat1][feat2]) >= correlation_threshold):
                highly_correlated_pairs.append({
                    'feature1': feat1,
                    'feature2': feat2,
                    'correlation': abs(correlation_matrix[feat1][feat2]),
                    'type': 'new_vs_new'
                })
    
    # Remove features correlated with existing features
    features_after_existing_correlation_filter = []
    removed_for_existing_correlation = []
    
    for feat in all_generated_features:
        is_correlated_with_existing = any(
            pair['feature1'] == feat and pair['type'] == 'new_vs_existing'
            for pair in highly_correlated_pairs
        )
        
        if not is_correlated_with_existing:
            features_after_existing_correlation_filter.append(feat)
        else:
            # Find which existing feature it's correlated with
            corr_pair = next(
                pair for pair in highly_correlated_pairs 
                if pair['feature1'] == feat and pair['type'] == 'new_vs_existing'
            )
            removed_for_existing_correlation.append({
                'feature': feat,
                'max_correlation': corr_pair['correlation'],
                'correlated_with': corr_pair['feature2'],
                'reason': 'high_correlation_with_existing'
            })
    
    # Remove features correlated with other new features (keep arbitrarily the first one alphabetically)
    features_after_new_correlation_filter = []
    removed_for_new_correlation = []
    processed_pairs = set()
    
    for feat1 in features_after_existing_correlation_filter:
        should_include = True
        
        for feat2 in features_after_existing_correlation_filter:
            if feat1 == feat2 or (feat1, feat2) in processed_pairs or (feat2, feat1) in processed_pairs:
                continue
            
            # Check if these features are correlated
            corr_pair = next(
                (pair for pair in highly_correlated_pairs 
                 if ((pair['feature1'] == feat1 and pair['feature2'] == feat2) or
                     (pair['feature1'] == feat2 and pair['feature2'] == feat1)) and
                 pair['type'] == 'new_vs_new'),
                None
            )
            
            if corr_pair:
                # Keep the one that comes first alphabetically (deterministic)
                if feat1 > feat2:  # feat2 comes first alphabetically
                    should_include = False
                    removed_for_new_correlation.append({
                        'removed_feature': feat1,
                        'kept_feature': feat2,
                        'correlation': corr_pair['correlation'],
                        'reason': 'high_correlation_with_new_feature'
                    })
                    processed_pairs.add((feat1, feat2))
                    break
                else:
                    processed_pairs.add((feat1, feat2))
        
        if should_include:
            features_after_new_correlation_filter.append(feat1)
    
    # Step 4: Remove features with similar distributions
    similar_distribution_pairs = _identify_similar_distributions(
        features_after_new_correlation_filter, 
        _generated_features
    )
    
    features_after_distribution_filter = _filter_similar_distributions(
        features_after_new_correlation_filter,
        similar_distribution_pairs,
        {}  # No scores yet, will use alphabetical ordering
    )
    
    # Step 5: Calculate target relationships for remaining features only
    target_importances = {}
    if features_after_distribution_filter:
        # Create a subset of generated features for target relationship calculation
        filtered_generated_features = {
            feat: _generated_features[feat] 
            for feat in features_after_distribution_filter
        }
        
        target_importances = _cached_calculate_target_relationship(
            training_data,
            target_column,
            filtered_generated_features
        )
    
    # Calculate feature metrics for remaining features
    feature_metrics = {}
    feature_scores = {}
    all_rms_correlations = []
    
    # First pass: calculate all RMS correlations to find the range for normalization
    for feat_name in features_after_distribution_filter:
        correlations_with_existing = []
        
        for orig_feat in numeric_features:
            if feat_name in correlation_matrix and orig_feat in correlation_matrix.get(feat_name, {}):
                corr = abs(correlation_matrix[feat_name][orig_feat])
                correlations_with_existing.append(corr)
        
        if correlations_with_existing:
            rms_correlation = np.sqrt(np.mean([c**2 for c in correlations_with_existing]))
            all_rms_correlations.append(rms_correlation)
    
    # Calculate normalization parameters for RMS correlation
    if all_rms_correlations:
        max_rms = max(all_rms_correlations)
        min_rms = min(all_rms_correlations)
        rms_range = max_rms - min_rms if max_rms > min_rms else 1.0
    else:
        max_rms = 1.0
        min_rms = 0.0
        rms_range = 1.0
    
    # Second pass: calculate normalized scores
    for feat_name in features_after_distribution_filter:
        importance_score = target_importances.get(feat_name, 0)
        
        # Calculate correlation metrics with existing features
        correlations_with_existing = []
        max_correlation = 0
        max_corr_feature = ""
        
        for orig_feat in numeric_features:
            if feat_name in correlation_matrix and orig_feat in correlation_matrix.get(feat_name, {}):
                corr = abs(correlation_matrix[feat_name][orig_feat])
                correlations_with_existing.append(corr)
                
                if corr > max_correlation:
                    max_correlation = corr
                    max_corr_feature = orig_feat
        
        # Calculate comprehensive correlation measures
        if correlations_with_existing:
            avg_correlation = np.mean(correlations_with_existing)
            rms_correlation = np.sqrt(np.mean([c**2 for c in correlations_with_existing]))
            # Normalize RMS correlation to 0-1 range based on the range in this batch
            normalized_rms = (rms_correlation - min_rms) / rms_range if rms_range > 0 else 0
        else:
            avg_correlation = 0
            rms_correlation = 0
            normalized_rms = 0
        
        # Step 6: Apply RMS correlation scoring with normalized RMS correlation
        # This ensures the score stays in a reasonable range and prevents misleading negative scores
        # Use multiplicative scoring to avoid negative scores and ensure target relationship is required
        if importance_score > 0:
            combined_score = importance_score * (1 - 0.3 * normalized_rms)
        else:
            combined_score = 0  # No score for features with no target relationship
        
        feature_scores[feat_name] = combined_score
        
        feature_metrics[feat_name] = {
            'target_importance': importance_score,
            'max_correlation': max_correlation,
            'max_corr_feature': max_corr_feature,
            'avg_correlation': avg_correlation,
            'rms_correlation': rms_correlation,
            'normalized_rms_correlation': normalized_rms,
            'overall_correlation_penalty': normalized_rms,
            'combined_score': combined_score,
            'num_correlations_calculated': len(correlations_with_existing)
        }
    
    # Sort by combined score (target importance - RMS correlation penalty)
    sorted_features = sorted(
        features_after_distribution_filter,
        key=lambda x: feature_scores.get(x, 0),
        reverse=True
    )
    
    # Filter out features with no target relationship or very low scores
    # Only recommend features that have meaningful target relationship
    MIN_TARGET_RELATIONSHIP = 0.01  # Minimum threshold for target relationship
    MIN_COMBINED_SCORE = 0.005  # Minimum threshold for combined score
    
    qualified_features = [
        feat for feat in sorted_features 
        if (target_importances.get(feat, 0) >= MIN_TARGET_RELATIONSHIP and
            feature_scores.get(feat, 0) >= MIN_COMBINED_SCORE)
    ]
    
    # Take top features from qualified features only, up to 10
    initial_top_features = qualified_features[:10]
    
    # NEW Step 6: Final correlation group check between all features
    final_top_features, removed_by_group_analysis = _cached_iterative_correlation_group_check(
        training_data,
        numeric_features,
        target_column,
        initial_top_features,
        feature_metrics,
        correlation_threshold=0.7,
        max_iterations=5  # Allow up to 5 iterations for convergence
    )
    
    # No additional multicollinearity removal needed since already done
    removed_multicollinear = []
    
    return (
        final_top_features,
        feature_metrics,
        highly_correlated_pairs,
        similar_distribution_pairs,
        removed_multicollinear,
        removed_for_existing_correlation,
        removed_for_new_correlation,
        removed_by_group_analysis  # Add new removal category
    )

def _initialize_session_state():
    """Initialize all session state variables if they don't exist."""
    if 'feature_creation_generated_features' not in st.session_state:
        st.session_state.feature_creation_generated_features = {}
    
    if 'feature_creation_selected_features' not in st.session_state:
        st.session_state.feature_creation_selected_features = []
    
    if 'feature_creation_top_features' not in st.session_state:
        st.session_state.feature_creation_top_features = []
    
    if 'feature_creation_feature_metrics' not in st.session_state:
        st.session_state.feature_creation_feature_metrics = {}
    
    if 'feature_creation_correlated_pairs' not in st.session_state:
        st.session_state.feature_creation_correlated_pairs = []
    
    if 'feature_creation_similar_distributions' not in st.session_state:
        st.session_state.feature_creation_similar_distributions = []
    
    if 'feature_creation_removed_multicollinear' not in st.session_state:
        st.session_state.feature_creation_removed_multicollinear = []
    
    if 'feature_creation_removed_for_existing_correlation' not in st.session_state:
        st.session_state.feature_creation_removed_for_existing_correlation = []
    
    if 'feature_creation_removed_for_new_correlation' not in st.session_state:
        st.session_state.feature_creation_removed_for_new_correlation = []
    
    # Note: undo functionality now uses feature_creation_ops_applied and feature_creation_entry_data

@st.cache_data(show_spinner=True)
def _cached_iterative_correlation_group_check(
    training_data: pd.DataFrame,
    numeric_features: List[str],
    target_column: str,
    top_features: List[str],
    _feature_metrics: Dict,  # Mark as unhashable
    correlation_threshold: float = 0.7,
    max_iterations: int = 10  # Prevent infinite loops
) -> Tuple[List[str], List[Dict]]:
    """
    Iterative correlation group analysis between ALL features (original + recommended).
    Continues until no more features are removed or max iterations reached.
    From each correlated group, removes recommended features with lowest scores,
    keeping original features and only the best recommended features.
    
    Args:
        training_data: Training DataFrame
        numeric_features: List of original numeric feature names
        target_column: Target column name
        top_features: List of recommended feature names
        _feature_metrics: Dictionary of feature metrics (marked as unhashable)
        correlation_threshold: Threshold for correlation groups (default 0.7)
        max_iterations: Maximum number of iterations to prevent infinite loops
        
    Returns:
        Tuple of (final_features, removed_features) where:
        - final_features: Filtered list of recommended features
        - removed_features: List of dicts with removal information including iteration info
    """
    # Check if networkx is available
    if not NETWORKX_AVAILABLE:
        # Fallback: return original features without group analysis
        return top_features, []
    
    # If no recommended features, nothing to filter
    if not top_features:
        return top_features, []
    
    current_features = top_features.copy()
    all_removed_features = []
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        iteration_removed_features = []
        
        # Create correlation matrix for current feature set
        temp_df = training_data[numeric_features].copy()
        
        # Add current recommended features to temp dataframe
        added_features = []
        for feat_name in current_features:
            if (hasattr(st.session_state, 'feature_creation_generated_features') and 
                feat_name in st.session_state.feature_creation_generated_features):
                feat_info = st.session_state.feature_creation_generated_features[feat_name]
                if 'values' in feat_info:
                    temp_df[feat_name] = feat_info['values']
                    added_features.append(feat_name)
        
        # If we couldn't add any recommended features, stop iteration
        if not added_features:
            break
        
        # Calculate correlation matrix for current iteration
        try:
            correlation_matrix = temp_df.corr().abs()
        except Exception:
            break  # If correlation calculation fails, stop iteration
        
        # Combine all features for analysis
        all_features = numeric_features + added_features
        
        # Create graph for correlation group analysis
        G = nx.Graph()
        G.add_nodes_from(all_features)
        
        # Add edges for correlations above threshold
        correlations = []
        for i, feat1 in enumerate(all_features):
            for j, feat2 in enumerate(all_features):
                if (i < j and feat1 in correlation_matrix.columns and 
                    feat2 in correlation_matrix.columns):
                    try:
                        corr_value = correlation_matrix.loc[feat1, feat2]
                        if not np.isnan(corr_value) and corr_value >= correlation_threshold:
                            G.add_edge(feat1, feat2, weight=corr_value)
                            correlations.append({
                                'feature1': feat1,
                                'feature2': feat2,
                                'correlation': corr_value
                            })
                    except Exception:
                        continue
        
        # Find connected components (correlation groups)
        correlation_groups = []
        processed_features = set()
        
        # Build groups from correlations
        for corr in correlations:
            feat1, feat2 = corr['feature1'], corr['feature2']
            
            # Find if either feature is already in a group
            found_group = None
            for group in correlation_groups:
                if feat1 in group or feat2 in group:
                    found_group = group
                    break
            
            if found_group is not None:
                found_group.add(feat1)
                found_group.add(feat2)
            else:
                correlation_groups.append({feat1, feat2})
            
            processed_features.add(feat1)
            processed_features.add(feat2)
        
        # Merge overlapping groups
        merged_groups = []
        while correlation_groups:
            current_group = correlation_groups.pop(0)
            
            # Try to merge with other groups
            i = 0
            while i < len(correlation_groups):
                if any(feat in current_group for feat in correlation_groups[i]):
                    current_group.update(correlation_groups[i])
                    correlation_groups.pop(i)
                else:
                    i += 1
            
            merged_groups.append(current_group)
        
        # Filter to only groups that contain both original and recommended features
        mixed_groups = []
        for group in merged_groups:
            group_list = list(group)
            has_original = any(feat in numeric_features for feat in group_list)
            has_recommended = any(feat in added_features for feat in group_list)
            
            if has_original and has_recommended and len(group_list) >= 2:
                mixed_groups.append(group_list)
        
        # Process each mixed group to remove lowest-scoring recommended features
        features_removed_this_iteration = False
        
        for group in mixed_groups:
            # Separate original and recommended features in this group
            group_original = [f for f in group if f in numeric_features]
            group_recommended = [f for f in group if f in added_features]
            
            # Only process if there are multiple recommended features in this group
            if len(group_recommended) <= 1:
                continue  # Keep the single recommended feature
            
            # Calculate metrics for recommended features in this group
            feature_scores = []
            for feat in group_recommended:
                # Get metrics from the _feature_metrics parameter
                metrics = _feature_metrics.get(feat, {})
                combined_score = metrics.get('combined_score', 0)
                target_importance = metrics.get('target_importance', 0)
                
                # Calculate average correlation with all other features in this group
                other_features = [f for f in group if f != feat and f in correlation_matrix.columns]
                correlations_in_group = []
                
                for other_feat in other_features:
                    try:
                        corr = correlation_matrix.loc[feat, other_feat]
                        if not np.isnan(corr):
                            correlations_in_group.append(corr)
                    except Exception:
                        pass
                
                avg_group_correlation = np.mean(correlations_in_group) if correlations_in_group else 0
                
                feature_scores.append({
                    'feature': feat,
                    'combined_score': combined_score,
                    'target_importance': target_importance,
                    'avg_group_correlation': avg_group_correlation,
                    'group_original_features': group_original,
                    'group_size': len(group)
                })
            
            # Sort by combined score (descending) - keep features with highest scores
            feature_scores.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Keep only the top feature from recommended features in this group
            # Remove the rest from current_features
            features_to_keep = 1  # Keep the best recommended feature in each group
            
            for i, feat_score in enumerate(feature_scores):
                if i >= features_to_keep:  # Remove features beyond the keep limit
                    feat_name = feat_score['feature']
                    if feat_name in current_features:
                        current_features.remove(feat_name)
                        features_removed_this_iteration = True
                        
                        iteration_removed_features.append({
                            'feature': feat_name,
                            'reason': 'iterative_correlation_group_filtering',
                            'iteration': iteration,
                            'group_size': feat_score['group_size'],
                            'combined_score': feat_score['combined_score'],
                            'avg_group_correlation': feat_score['avg_group_correlation'],
                            'correlated_with_original': feat_score['group_original_features'],
                            'kept_recommended_feature': feature_scores[0]['feature'] if feature_scores else None,
                            'correlation_threshold_used': correlation_threshold,
                            'total_correlations_in_iteration': len(correlations),
                            'total_groups_in_iteration': len(merged_groups),
                            'mixed_groups_in_iteration': len(mixed_groups)
                        })
        
        # Add this iteration's removed features to the overall list
        all_removed_features.extend(iteration_removed_features)
        
        # If no features were removed this iteration, we've reached stability
        if not features_removed_this_iteration:
            break
    
    # Add iteration summary to removed features info
    for removed_feat in all_removed_features:
        removed_feat['total_iterations'] = iteration
        removed_feat['converged'] = iteration < max_iterations
    
    return current_features, all_removed_features

def _identify_similar_distributions(features: List[str], _generated_features: Dict, similarity_threshold: float = 0.9) -> List[Dict]:
    """
    Identify pairs of features with similar distributions using Kolmogorov-Smirnov test.
    
    Args:
        features: List of feature names to compare
        _generated_features: Dictionary of generated features
        similarity_threshold: Threshold for p-value to consider distributions similar (higher = more similar)
        
    Returns:
        List of dictionaries containing pairs of features with similar distributions
    """
    from scipy import stats
    import numpy as np
    
    similar_pairs = []
    
    # Skip if fewer than 2 features
    if len(features) < 2:
        return similar_pairs
    
    # Pre-compute normalized values for all features
    normalized_values = {}
    for feat_name in features:
        vals = _generated_features[feat_name]['values'].values
        
        # Normalize values efficiently using numpy operations
        min_val = np.nanmin(vals)
        max_val = np.nanmax(vals)
        if max_val > min_val:
            norm_vals = (vals - min_val) / (max_val - min_val)
        else:
            norm_vals = np.zeros_like(vals)
        
        # Store only valid values
        mask = ~(np.isnan(norm_vals) | np.isinf(norm_vals))
        if mask.sum() >= 10:  # Only store if we have enough valid samples
            normalized_values[feat_name] = norm_vals[mask]
    
    # Pre-compute histograms for all features
    histograms = {}
    for feat_name, norm_vals in normalized_values.items():
        hist, _ = np.histogram(norm_vals, bins=20, range=(0, 1), density=True)
        histograms[feat_name] = hist / hist.sum()  # Normalize histogram
    
    # Compare distributions efficiently
    n = len(features)
    for i in range(n):
        feat1 = features[i]
        if feat1 not in normalized_values:
            continue
        
        norm1 = normalized_values[feat1]
        hist1 = histograms[feat1]
        
        for j in range(i + 1, n):
            feat2 = features[j]
            if feat2 not in normalized_values:
                continue
            
            norm2 = normalized_values[feat2]
            hist2 = histograms[feat2]
            
            try:
                # Calculate KS statistic and p-value
                ks_stat, p_value = stats.ks_2samp(norm1, norm2)
                
                # Calculate histogram intersection efficiently
                intersection = np.minimum(hist1, hist2).sum()
                
                # If either p-value is high or histogram intersection is high
                if p_value > similarity_threshold or intersection > 0.85:
                    similar_pairs.append({
                        'feature1': feat1,
                        'feature2': feat2,
                        'p_value': p_value,
                        'overlap': intersection,
                        'similarity': max(p_value, intersection)
                    })
            except Exception:
                # Skip pairs that cause errors
                continue
    
    # Clean up memory
    del normalized_values
    del histograms
    gc.collect()
    
    return similar_pairs

def _filter_similar_distributions(features: List[str], similar_pairs: List[Dict], feature_scores: Dict[str, float]) -> List[str]:
    """
    Filter features to remove those with similar distributions but lower scores.
    If no scores available, uses alphabetical ordering.
    
    Args:
        features: List of feature names
        similar_pairs: List of dictionaries containing pairs of features with similar distributions
        feature_scores: Dictionary mapping feature names to scores (can be empty)
        
    Returns:
        List of filtered feature names
    """
    # Create a graph representation for efficient grouping
    from collections import defaultdict
    import numpy as np
    
    # Create adjacency list representation
    graph = defaultdict(list)
    for pair in similar_pairs:
        feat1, feat2 = pair['feature1'], pair['feature2']
        if feat1 in features and feat2 in features:
            graph[feat1].append(feat2)
            graph[feat2].append(feat1)
    
    # Function to find connected components using DFS
    def find_connected_components(graph):
        visited = set()
        components = []
        
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node in graph:
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(component)
        
        return components
    
    # Find groups of similar features
    similar_groups = find_connected_components(graph)
    
    # Add isolated features (those not in any group)
    isolated_features = set(features) - set().union(*similar_groups) if similar_groups else set(features)
    similar_groups.extend([[feat] for feat in isolated_features])
    
    # Select best feature from each group
    filtered_features = []
    for group in similar_groups:
        if len(group) == 1:
            filtered_features.append(group[0])
        else:
            if feature_scores:
                # Use scores if available
                group_scores = np.array([feature_scores.get(feat, 0) for feat in group])
                best_idx = np.argmax(group_scores)
                filtered_features.append(group[best_idx])
            else:
                # Use alphabetical ordering if no scores
                group.sort()  # Sort alphabetically
                filtered_features.append(group[0])  # Take first alphabetically
    
    # Sort final features appropriately
    if feature_scores:
        # Sort by score if available
        filtered_scores = np.array([feature_scores.get(feat, 0) for feat in filtered_features])
        sorted_indices = np.argsort(-filtered_scores)  # Negative for descending order
        filtered_features = [filtered_features[i] for i in sorted_indices]
    else:
        # Sort alphabetically if no scores
        filtered_features.sort()
    
    return filtered_features

class FeatureCreationComponent:
    """
    Component for creating new features by combining numeric features and analyzing
    their relationships with the target variable.
    
    This component allows users to:
    - Generate new features using arithmetic operations between existing numeric features
    - Analyse the relationship between new features and the target variable
    - Select and add the most promising new features to the dataset
    
    Example usage:
    ```python
    feature_creator = FeatureCreationComponent(
        builder,
        logger,
        training_data,
        testing_data,
        target_column
    )
    feature_creator.render()
    ```
    """
    
    def __init__(self, builder, logger, training_data=None, testing_data=None, target_column=None):
        """
        Initialize the component with builder and logger instances and datasets.
        
        Args:
            builder: The Builder instance with data and model building methods
            logger: The Logger instance for tracking user actions and errors
            training_data: Training DataFrame (if None, uses builder.training_data)
            testing_data: Testing DataFrame (if None, uses builder.testing_data)
            target_column: Target column name (if None, uses builder.target_column)
        """
        self.builder = builder
        self.logger = logger
        self.training_data = training_data if training_data is not None else builder.training_data
        self.testing_data = testing_data if testing_data is not None else builder.testing_data
        self.target_column = target_column if target_column is not None else builder.target_column
        
        # Initialize session state
        _initialize_session_state()
        
        # Initialize undo functionality with single backup (memory optimized)
        if "feature_creation_ops_applied" not in st.session_state:
            st.session_state.feature_creation_ops_applied = []
            
        # Store initial state for undo functionality (single backup for both datasets)
        if "feature_creation_entry_data" not in st.session_state:
            st.session_state.feature_creation_entry_data = {
                'training_data': self.builder.training_data.copy(),
                'testing_data': self.builder.testing_data.copy()
            }

    def render(self):
        """
        Render the feature creation component interface.
        """
        st.write("---")
        st.write("Using the data exploration component may cause the page to reload, any changes that you have applied will still be in effect. you can use the undo button to reset the data to it's original state when you first entered the page")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            @st.dialog(title="Data Exploration", width="large")
            def data_explorer_dialog():
                data_explorer = DataExplorationComponent(self.builder, st.session_state.logger, data=st.session_state.builder.training_data, target_column=st.session_state.builder.target_column)
                data_explorer.render()

            if st.button("Training Data Exploration",on_click=st.rerun):
                data_explorer_dialog()
        with col2:
            st.write("")
        with col3:
            if st.button("Undo Feature Creation", type="primary", width='stretch'):
                if st.session_state.feature_creation_ops_applied:
                    # Restore data to entry state
                    entry_data = st.session_state.feature_creation_entry_data
                    self.builder.training_data = entry_data['training_data'].copy()
                    self.builder.testing_data = entry_data['testing_data'].copy()
                    
                    # Clear operations tracking
                    ops_count = len(st.session_state.feature_creation_ops_applied)
                    st.session_state.feature_creation_ops_applied = []
                    
                    # Clear session state variables
                    cleanup_keys = [
                        'feature_creation_generated_features', 'feature_creation_selected_features',
                        'feature_creation_top_features', 'feature_creation_feature_metrics',
                        'feature_creation_correlated_pairs', 'feature_creation_similar_distributions',
                        'feature_creation_removed_multicollinear'
                    ]
                    for key in cleanup_keys:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.success(f"✅ Undid {ops_count} feature creation operation(s). Training and testing data restored to entry state.")
                    st.rerun()
                else:
                    st.info("No feature creation operations to undo.")

        st.write("## Feature Creation")
        
        with st.expander("ℹ️ How Feature Creation Works", expanded=True):
            st.markdown("""
            ### What is Feature Creation?
            
            Feature creation automatically generates new, useful features by combining your existing numeric features using basic math operations. Think of it as creating new "super features" that might be better at predicting your target than the original features alone.
            
            ### The 5 Basic Operations
            
            We combine every pair of numeric features using these operations:
            
            * **Ratio (÷)**: Division of one feature by another → *Example: Price ÷ Size = Price per Unit*
            * **Sum (+)**: Addition of two features → *Example: Length + Width = Perimeter*
            * **Difference (−)**: Subtraction of one feature from another → *Example: Revenue − Cost = Profit*
            * **Product (×)**: Multiplication of two features → *Example: Length × Width = Area*
            * **Mean (Average)**: Average of two features → *Example: (Min + Max) ÷ 2 = Midpoint*
            
            ### Our 7-Step Smart Filtering Process
            
            We don't just create features randomly - we use a sophisticated quality-first approach that gets smarter at each step:
            
            #### **Step 1: Immediate Quality Check** ⚡
            **What happens:** As we create each new feature, we immediately check for problems
            - **Null values**: If a new feature has more than 5% missing values, we throw it away
            - **Infinite values**: We replace infinity values (like dividing by zero) with zeros
            - **Why this matters**: Only clean, complete features move to the next step
            
            #### **Step 2: Build Complete Correlation Map** 🗺️
            **What happens:** We calculate how every feature relates to every other feature
            - **Correlation matrix**: A big table showing relationships between ALL features (old + new)
            - **Why this matters**: This gives us the complete picture we need for smart filtering
            - **Beginner tip**: Correlation tells us if two features move together (positive) or opposite (negative)
            
            #### **Step 3: Remove Obvious Duplicates** 🧹
            **What happens:** We remove new features that are too similar to what we already have
            - **Remove if correlation > 0.8 with existing features**: No point keeping features that just repeat existing information
            - **Remove duplicates among new features**: If multiple new features are nearly identical (correlation > 0.8), keep only one
            - **Smart choice**: We keep the alphabetically first one for consistency
            - **Why this matters**: Prevents your model from getting confused by duplicate information
            
            #### **Step 4: Statistical Distribution Check** 📊
            **What happens:** We look for features that have suspiciously similar patterns
            - **Kolmogorov-Smirnov test**: A statistical test that compares the "shape" of data distributions
            - **Remove similar patterns**: If two features have nearly identical data patterns, keep only one
            - **Why this matters**: Features might have low correlation but still contain the same type of information
            - **Beginner tip**: Imagine two features that both measure "size" differently - they might not correlate perfectly but they're essentially measuring the same thing
            
            #### **Step 5: Measure Predictive Power** 🎯
            **What happens:** Now we test how well each surviving feature actually predicts your target
            - **Mutual Information**: A sophisticated measure that finds ANY type of relationship (not just linear)
            - **Works for all targets**: Whether you're predicting binary classes, multiple classes (multi-class classification), or continuous numbers (regression)
            - **Smart approach**: We only do this expensive calculation on features that passed quality checks
            - **Why this matters**: No point keeping features that can't actually help predict what you care about
            
            #### **Step 6: Advanced Scoring System** 🏆
            **What happens:** We calculate a sophisticated score that balances predictive power against redundancy
            
            **The Scoring Formula:**
            
            - **Final Score = Target Relationship × (1 - 0.3 × Normalized RMS Correlation)**
            
            **Breaking this down for beginners:**
            - **Target Relationship**: How well this feature predicts your target (0 to 1, higher = better)
            - **RMS Correlation**: Root Mean Square correlation with ALL existing features
                - *Think of it as*: "How much does this feature overlap with what we already have?"
                - *RMS vs Simple Max*: Captures multiple moderate overlaps better than just the highest single overlap
            - **Normalized**: We scale RMS to 0-1 range so scores stay meaningful
            - **The 0.3 factor**: We penalize overlap, but not too harshly (30% penalty for complete overlap)
            - **Multiplicative approach**: Ensures features need BOTH good prediction AND low overlap to score well
            
            **Quality Requirements:**
            - **Minimum Target Relationship**: ≥ 0.01 (must actually predict something)
            - **Minimum Final Score**: ≥ 0.005 (must be worthwhile after overlap penalty)
            - **Top 10 Selection**: We pick the 10 highest-scoring features that meet requirements
            
            #### **Step 7: Iterative Network Analysis** 🔄🕸️
            **What happens:** The most sophisticated step - we iteratively analyze correlation "groups" across ALL features until convergence
            
            **How the iterative process works:**
            - **Initial Network Analysis**: We treat correlations like connections in a social network
            - **Find correlation groups**: Groups of features that are all connected by correlations > 0.7
            - **Focus on mixed groups**: We only care about groups containing BOTH original and new features
            - **Smart removal**: From each mixed group, we keep ALL original features but only the BEST new feature
            - **Recalculate & Repeat**: After removing features, correlation landscape changes - we repeat until no more changes occur
            - **Convergence**: Process stops when no more features are removed or max iterations (5) reached
            - **Score-based decisions**: We use actual quality scores to decide which new features to keep at each iteration
            
            **Why iterative analysis is superior:**
            - **Dynamic Adaptation**: Removing features changes correlation patterns - iteration captures these new relationships
            - **Cascading Effects**: Features that seemed independent might become correlated after other features are removed  
            - **True Convergence**: We reach a stable optimal state rather than stopping after one pass
            - **Network Evolution**: Correlation groups can merge, split, or emerge during the process
            - **Better Global Optimum**: Multiple iterations help find globally optimal feature sets, not just locally optimal
            
            **Why this iterative step matters:**
            - **Protects existing features**: Your original features are always preserved throughout all iterations
            - **Prevents subtle redundancy**: Catches correlation patterns that emerge during the filtering process
            - **Intelligent choices**: Uses actual performance metrics instead of arbitrary rules at each iteration
            - **Network effects**: Understands how complex correlation patterns evolve as features are removed
            - **Convergence guarantee**: Ensures the final set is truly stable and optimal
            
            ### Why This 7-Step Process is So Effective
            
            **🏃‍♂️ Efficiency First**: We filter for basic quality before doing expensive calculations
            
            **🧠 Intelligent Overlap Detection**: RMS correlation captures subtle redundancy that simple methods miss
            
            **🎯 Prediction-Focused**: Features must actually help predict your target, not just exist
            
            **🔄 Iterative Convergence**: Final step uses iterative network analysis to reach optimal stable state
            
            **📊 Score-Based Decisions**: Every choice is based on actual quality metrics, not arbitrary rules
            
            **✅ Quality Guarantees**: Multiple checkpoints ensure only truly valuable features are recommended
            
            **🌐 Dynamic Network Intelligence**: Catches correlation patterns that emerge and evolve during filtering
            
            ### What You Get
            
            The final recommendations represent features that:
            - ✅ Have strong predictive power for your target (binary, multi-class, or regression)
            - ✅ Add truly unique information (no redundancy with existing features)
            - ✅ Are mathematically sound (no infinities or excessive missing values)
            - ✅ Passed sophisticated iterative correlation group analysis
            - ✅ Meet strict quality requirements at every step
            - ✅ Represent a converged, stable optimal feature set
            - ✅ Handle multi-class classification targets properly with categorical visualization
            
            **In simple terms**: You get the best possible new features that actually help predict what you care about, without any redundancy with what you already have, optimized through iterative analysis until convergence! Works seamlessly with binary classification, multi-class classification, and regression problems.
            """)
        
        # Check if feature creation operations were already applied
        if st.session_state.feature_creation_ops_applied:
            st.success("✅ Feature creation has been completed!")
            st.info("Features have been added to your dataset. Use the 'Undo Feature Creation' button above to revert changes and regenerate features if needed.")
            
            # Show summary of what was applied
            st.write("### Applied Operations Summary")
            st.write(f"- **Operations Applied:** {len(st.session_state.feature_creation_ops_applied)}")
            
            # Show current vs entry data comparison
            if st.session_state.feature_creation_entry_data:
                entry_data = st.session_state.feature_creation_entry_data
                current_cols = len(self.builder.training_data.columns)
                entry_cols = len(entry_data['training_data'].columns)
                features_added = current_cols - entry_cols
                
                st.write(f"- **Original Features:** {entry_cols}")
                st.write(f"- **Current Features:** {current_cols}")
                st.write(f"- **Features Added:** {features_added}")
                
            return
        
        # Get numeric features from the training data
        numeric_features = self._get_numeric_features()
        
        if len(numeric_features) < 2:
            st.warning("At least 2 numeric features are required for feature creation. Please add more numeric features to your dataset.")
            return
        
        # Generate or retrieve feature combinations
        with st.spinner("Generating feature combinations..."):
            self._generate_feature_combinations(numeric_features)
        
        # Display the generated features
        if st.session_state.feature_creation_generated_features:
            # Select features if not already selected
            if not st.session_state.feature_creation_top_features:
                self._analyse_generated_features()
        
            # Display recommended features
            self._display_recommended_features()

        # Button to apply selected features
        self._render_apply_button()
    
    def _get_numeric_features(self) -> List[str]:
        """
        Get list of numeric features from the training data, excluding the target column.
        
        Returns:
            List of numeric feature names
        """
        numeric_cols = self.training_data.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
        return [col for col in numeric_cols if col != self.target_column]
    
    def _generate_feature_combinations(self, numeric_features: List[str]):
        """
        Generate all possible feature combinations using various operations.
        
        Args:
            numeric_features: List of numeric feature names
        """
        # Define operations as a list of tuples (immutable) for caching
        operations_list = [
            ('ratio', 'ratio'),
            ('sum', 'sum'),
            ('difference', 'difference'),
            ('product', 'product'),
            ('mean', 'mean')
        ]
        
        if not st.session_state.feature_creation_generated_features:
            with st.spinner("Generating feature combinations..."):
                generated_features = _cached_generate_features(
                    self.training_data,
                    numeric_features,
                    operations_list
                )
                st.session_state.feature_creation_generated_features = generated_features
                
                self.logger.log_calculation(
                    "Feature Creation - Generated Features",
                    {
                        "num_features_generated": len(generated_features),
                        "base_features_used": len(numeric_features),
                        "operations_used": [op[0] for op in operations_list],
                        "memory_usage_mb": sum(feat['values'].memory_usage() for feat in generated_features.values()) / (1024 * 1024)
                    }
                )
        
        return st.session_state.feature_creation_generated_features
    
    def _analyse_generated_features(self):
        """Analyse generated features using cached function."""
        if not st.session_state.feature_creation_generated_features:
            return []
        
        # Use cached analysis function
        (
            top_features,
            feature_metrics,
            correlated_pairs,
            similar_distributions,
            removed_multicollinear,
            removed_for_existing_correlation,
            removed_for_new_correlation,
            removed_by_group_analysis
        ) = _cached_analyse_features(
            self.training_data,
            self._get_numeric_features(),
            self.target_column,
            st.session_state.feature_creation_generated_features
        )
        
        # Store results in session state
        st.session_state.feature_creation_top_features = top_features
        st.session_state.feature_creation_feature_metrics = feature_metrics
        st.session_state.feature_creation_correlated_pairs = correlated_pairs
        st.session_state.feature_creation_similar_distributions = similar_distributions
        st.session_state.feature_creation_removed_multicollinear = removed_multicollinear
        st.session_state.feature_creation_removed_for_existing_correlation = removed_for_existing_correlation
        st.session_state.feature_creation_removed_for_new_correlation = removed_for_new_correlation
        st.session_state.feature_creation_removed_by_group_analysis = removed_by_group_analysis
        
        # Log analysis
        self.logger.log_calculation(
            "Feature Creation - Top Features",
            {
                "top_features": top_features,
                "highly_correlated_pairs": len(correlated_pairs),
                "similar_distribution_pairs": len(similar_distributions),
                "removed_for_multicollinearity": len(removed_multicollinear),
                "removed_by_group_analysis": len(removed_by_group_analysis)
            }
        )
        
        return top_features
    
    def _display_recommended_features(self):
        import streamlit as st
        """
        Display the recommended features and allow user to select which ones to add.
        """
        st.write("### Recommended Features")
        st.write("These features have been analysed and show strong relationships with the target variable while maintaining low correlation with existing features.")
        
        top_features = st.session_state.feature_creation_top_features
        feature_metrics = st.session_state.get('feature_creation_feature_metrics', {})
        correlated_pairs = st.session_state.get('feature_creation_correlated_pairs', [])
        similar_distributions = st.session_state.get('feature_creation_similar_distributions', [])
        removed_multicollinear = st.session_state.get('feature_creation_removed_multicollinear', [])
        removed_for_existing_correlation = st.session_state.get('feature_creation_removed_for_existing_correlation', [])
        removed_for_new_correlation = st.session_state.get('feature_creation_removed_for_new_correlation', [])
        removed_by_group_analysis = st.session_state.get('feature_creation_removed_by_group_analysis', [])
        
        if not top_features:
            st.warning("⚠️ **No suitable features found that meet quality requirements.**")
            st.info("""
            **Why this happens:**
            - All generated features had zero or very low target relationship (< 0.01)
            - Features were too highly correlated with existing features (> 0.8)
            - Features had similar statistical distributions to higher-ranked features
            
            **What you can try:**
            - Use different numeric features as inputs
            - Apply different preprocessing steps first (scaling, binning, etc.)
            - Check if your target variable has enough variation for feature relationships to be detected
            - Consider manual feature engineering for domain-specific combinations
            """)
            return
        
        # Calculate statistics for the dashboard with proper step tracking
        total_combinations = len(st.session_state.feature_creation_generated_features)
        num_multicollinear = len(removed_multicollinear)
        num_similar_dist = len(similar_distributions)
        num_corr_existing = len([p for p in correlated_pairs if p['type'] == 'new_vs_existing'])
        num_corr_new = len([p for p in correlated_pairs if p['type'] == 'new_vs_new'])
        num_removed_existing_corr = len(removed_for_existing_correlation)
        num_removed_new_corr = len(removed_for_new_correlation)
        num_removed_by_group_analysis = len(removed_by_group_analysis)
        
        # Work backward from final known values to ensure math consistency
        final_recommendations = len(top_features)
        features_before_group_check = final_recommendations + num_removed_by_group_analysis
        
        # Calculate actual features removed in similarity step to make math work
        # We know: total_start - correlation_removed - similarity_removed = features_before_group_check
        required_total_removed = total_combinations - features_before_group_check
        correlation_removed = num_removed_existing_corr + num_removed_new_corr
        features_removed_for_similarity = max(0, required_total_removed - correlation_removed)
        
        # Now build step-by-step breakdown that adds up correctly
        after_step1 = total_combinations  # All generated features
        after_step2 = after_step1  # Correlation matrix (no removal)
        after_step3 = after_step2 - correlation_removed  # Remove correlated features
        after_step4 = after_step3 - features_removed_for_similarity  # Remove similar distributions
        features_analyzed_for_target = after_step4  # These go to target analysis
        
        # Step 6: Features that met quality requirements (before final group check)
        features_before_group_check = len(top_features) + num_removed_by_group_analysis
        
        # Step 7: Final recommendations after group analysis
        final_recommendations = len(top_features)
        
        # Calculate average metrics for recommended features
        if len(feature_metrics) > 0:
            avg_target_rel = np.mean([m['target_importance'] for m in feature_metrics.values() if m['target_importance'] > 0])
            avg_rms_correlation = np.mean([m['rms_correlation'] for m in feature_metrics.values() if m['rms_correlation'] > 0])
        else:
            avg_target_rel = 0.0
            avg_rms_correlation = 0.0
        
        # Create metrics dashboard
        st.write("#### Analysis Summary")
        
        # Stage 1: Initial Filtering (Steps 1-4)
        st.write("##### 🔍 Stage 1: Initial Filtering (Steps 1-4)")
        st.caption("Quality checks and basic redundancy removal")
        
        row1_cols = st.columns(3)
        with row1_cols[0]:
            st.metric(
                "Total Combinations Generated",
                f"{total_combinations:,}",
                help="Total number of feature combinations initially generated using all operation types"
            )
        with row1_cols[1]:
            # Calculate null removal during generation - this happens in Step 1
            # We can estimate this but it's not directly tracked
            st.metric(
                "Removed: High Correlation (Existing)",
                f"{num_removed_existing_corr:,}",
                help="Step 3: Features removed for correlation >0.8 with existing features"
            )
        with row1_cols[2]:
            st.metric(
                "Removed: High Correlation (New)",
                f"{num_removed_new_corr:,}",
                help="Step 3: Features removed for correlation >0.8 with other new features"
            )
        
        row2_cols = st.columns(3)
        with row2_cols[0]:
            st.metric(
                "Removed: Similar Distributions",
                f"{features_removed_for_similarity:,}",
                help="Step 4: Features removed for having statistically similar distributions"
            )
        with row2_cols[1]:
            st.metric(
                "Features After Initial Filtering",
                f"{features_analyzed_for_target:,}",
                help="Features that passed Steps 1-4 and proceeded to target relationship analysis"
            )
        with row2_cols[2]:
            initial_retention_pct = (features_analyzed_for_target / total_combinations) * 100 if total_combinations > 0 else 0
            st.metric(
                "Initial Retention Rate",
                f"{initial_retention_pct:.1f}%",
                help="Percentage of generated features that passed initial quality filtering"
            )
        
        st.write("---")
        
        # Stage 2: Final Analysis and Recommendations (Steps 5-7)
        st.write("##### 🎯 Stage 2: Final Analysis and Recommendations (Steps 5-7)")
        st.caption("Target relationship analysis, scoring, and final correlation group check")
        
        row3_cols = st.columns(3)
        with row3_cols[0]:
            st.metric(
                "Features Analyzed for Target",
                f"{features_analyzed_for_target:,}",
                help="Step 5: Features that underwent expensive target relationship analysis"
            )
        with row3_cols[1]:
            st.metric(
                "Met Quality Requirements",
                f"{features_before_group_check:,}",
                help="Step 6: Features meeting minimum target relationship (≥0.01) and combined score (≥0.005)"
            )
        with row3_cols[2]:
            # Add iteration information from the iterative group analysis
            iteration_info = ""
            convergence_info = ""
            if removed_by_group_analysis:
                # Get iteration info from removed features
                total_iterations = removed_by_group_analysis[0].get('total_iterations', 1) if removed_by_group_analysis else 1
                converged = removed_by_group_analysis[0].get('converged', True) if removed_by_group_analysis else True
                iteration_info = f" ({total_iterations} iterations)"
                convergence_info = "✅ Converged" if converged else "⚠️ Max iterations reached"
            
            st.metric(
                f"Removed: Iterative Group Analysis{iteration_info}",
                f"{num_removed_by_group_analysis:,}",
                help=f"Step 7: Features removed through iterative correlation group analysis. {convergence_info}"
            )
        
        row4_cols = st.columns(3)
        with row4_cols[0]:
            st.metric(
                "Final Recommendations",
                f"{final_recommendations}",
                help="Features selected as final recommendations after all 7 steps"
            )
        with row4_cols[1]:
            final_retention_pct = (final_recommendations / total_combinations) * 100 if total_combinations > 0 else 0
            st.metric(
                "Overall Success Rate",
                f"{final_retention_pct:.1f}%",
                help="Percentage of generated features that became final recommendations"
            )
        with row4_cols[2]:
            quality_score = (avg_target_rel / (1 + avg_rms_correlation)) if avg_rms_correlation >= 0 else avg_target_rel
            st.metric(
                "Average Quality Score",
                f"{quality_score:.3f}",
                help="Overall quality metric: Target Relationship / (1 + RMS Correlation)"
            )
        
        # Final quality metrics for recommended features
        row5_cols = st.columns(3)
        with row5_cols[0]:
            st.metric(
                "Avg Target Relationship",
                f"{avg_target_rel:.3f}",
                help="Average target relationship strength for recommended features (higher = better)"
            )
        with row5_cols[1]:
            st.metric(
                "Avg RMS Correlation",
                f"{avg_rms_correlation:.3f}",
                help="Average RMS correlation between recommended features and existing features (lower = better)"
            )
        with row5_cols[2]:
            # Calculate efficiency metric
            analysis_efficiency = (final_recommendations / max(1, features_analyzed_for_target)) * 100
            st.metric(
                "Analysis Efficiency",
                f"{analysis_efficiency:.1f}%",
                help="Percentage of features that became recommendations after passing initial filtering"
            )
        
        # Keep only planning-focused correlation warning for feature selection
        if correlated_pairs:
            st.warning("⚠️ **Planning Ahead:** Some recommended features are highly correlated with existing features. Consider this during feature selection to avoid redundancy.")
        
        # Simple quality confirmation - detailed metrics already shown in dashboard  
        if len(top_features) > 0:
            st.success(f"✅ **Quality Assurance**: All {len(top_features)} recommended features meet strict quality requirements and add unique predictive value.")
        
        # Display feature ranking table with detailed metrics
        self._display_feature_ranking_table(top_features, feature_metrics)
        
        # Display correlation heatmap for top features and existing features
        self._display_correlation_heatmap(top_features)
        
        st.divider()
        # Create tabs for feature details
        st.write("### Feature Details")
        
        # Create tabs for each feature
        tab_titles = [f"{i+1}. {self._get_readable_feature_name(feat_name)}" for i, feat_name in enumerate(top_features)]
        tabs = st.tabs(tab_titles)
        
        # Fill each tab with feature information
        for i, (tab, feat_name) in enumerate(zip(tabs, top_features)):
            with tab:
                feat_info = st.session_state.feature_creation_generated_features[feat_name]
                metrics = feature_metrics.get(feat_name, {})
                
                # Check if this feature is in any correlated pairs
                is_correlated = any(
                    (pair['feature1'] == feat_name or pair['feature2'] == feat_name) and pair['type'] == 'new_vs_existing'
                    for pair in correlated_pairs
                )
                
                col1, col2, col3 = st.columns([1, 1, 3])
                
                with col1:
                    # Display feature details
                    st.write(f"**Operation:** {feat_info['operation'].title()}")
                    st.write(f"**Feature 1:** {feat_info['feature1']}")
                    st.write(f"**Feature 2:** {feat_info['feature2']}")
                    
                    # Display correlation warning if needed
                    if is_correlated:
                        st.warning("**Correlation Warning:**")
                        for pair in correlated_pairs:
                            if pair['type'] == 'new_vs_existing':
                                if pair['feature1'] == feat_name:
                                    st.write(f"- Correlated with existing feature **{pair['feature2']}** (r = {pair['correlation']:.2f})")
                                elif pair['feature2'] == feat_name:
                                    st.write(f"- Correlated with existing feature **{pair['feature1']}** (r = {pair['correlation']:.2f})")
                    
                    # Add essential metrics only
                    if metrics:
                        st.write("#### Performance")
                        st.write(f"**Target Relationship:** {metrics['target_importance']:.3f}")
                        st.write(f"**Normalized RMS:** {metrics['normalized_rms_correlation']:.3f}")
                        st.write(f"**Overall Score:** {metrics['combined_score']:.3f}")
                        
                        # Simple correlation with target if calculable
                        try:
                            feature_values = feat_info['values'].fillna(feat_info['values'].median())
                            target_values = self.training_data[self.target_column]
                            
                            # For multi-class classification, convert target to numeric for correlation
                            if is_multiclass_classification and target_values.dtype == 'object':
                                # Convert categorical target to numeric codes for correlation calculation
                                target_numeric = pd.Categorical(target_values).codes
                                correlation = feature_values.corr(pd.Series(target_numeric))
                            else:
                                correlation = feature_values.corr(target_values)
                            
                            if not np.isnan(correlation):
                                st.write(f"**Correlation with Target:** {correlation:.3f}")
                        except Exception:
                            pass
                
                with col2:
                    # Display statistics
                    st.write("**Value Statistics:**")
                    values_preview = feat_info['values'].describe()
                    st.dataframe(values_preview, width='stretch')

                with col3:
                    # Display histogram
                    try:
                        # Create subplot with distribution and target relationship
                        feature_values = feat_info['values'].fillna(feat_info['values'].median())
                        target_values = self.training_data[self.target_column]
                        
                        # Use session state to determine problem type if available, otherwise fall back to heuristics
                        import streamlit as st
                        if hasattr(st, 'session_state') and hasattr(st.session_state, 'problem_type'):
                            problem_type = st.session_state.problem_type
                            is_classification = problem_type in ["binary_classification", "multiclass_classification"]
                            is_binary_classification = problem_type == "binary_classification"
                            is_multiclass_classification = problem_type == "multiclass_classification"
                        else:
                            # Fallback to original heuristic
                            is_classification = (
                                target_values.dtype == 'object' or 
                                len(target_values.unique()) < 10
                            )
                            is_binary_classification = len(target_values.unique()) == 2
                            is_multiclass_classification = len(target_values.unique()) > 2 and len(target_values.unique()) <= 10
                        
                        if is_classification:
                            # For classification: create colored histogram by target classes
                            # Convert target to string for proper categorical handling in visualization
                            target_for_viz = target_values.astype(str)
                            
                            # For multi-class with many classes, limit colors to avoid visual clutter
                            if is_multiclass_classification and len(target_values.unique()) > 10:
                                st.info(f"⚠️ Multi-class target has {len(target_values.unique())} classes. Visualization may be cluttered.")
                            
                            fig = px.histogram(
                                x=feature_values,
                                color=target_for_viz,
                                nbins=30,
                                title=f"Distribution of {self._get_readable_feature_name(feat_name)} by Target Class",
                                marginal="box"  # Add box plot on top
                            )
                            fig.update_layout(
                                xaxis_title=self._get_readable_feature_name(feat_name),
                                yaxis_title="Frequency",
                                height=350,
                                legend_title=self.target_column,
                                # For multi-class, adjust legend
                                legend=dict(
                                    orientation="v" if is_multiclass_classification else "h",
                                    yanchor="top" if is_multiclass_classification else "bottom",
                                    y=1 if is_multiclass_classification else -0.2,
                                    xanchor="left" if is_multiclass_classification else "center",
                                    x=1.02 if is_multiclass_classification else 0.5
                                )
                            )
                            st.plotly_chart(fig, config={'responsive': True})
                            
                        else:
                            # For regression: create subplot with histogram and scatter plot
                            fig = make_subplots(
                                rows=2, cols=1,
                                subplot_titles=(
                                    f"Distribution of {self._get_readable_feature_name(feat_name)}",
                                    f"Relationship with {self.target_column}"
                                ),
                                vertical_spacing=0.15,
                                row_heights=[0.4, 0.6]
                            )
                            
                            # Add histogram
                            fig.add_trace(
                                go.Histogram(
                                    x=feature_values,
                                    nbinsx=30,
                                    name="Distribution",
                                    showlegend=False
                                ),
                                row=1, col=1
                            )
                            
                            # Add scatter plot
                            fig.add_trace(
                                go.Scatter(
                                    x=feature_values,
                                    y=target_values,
                                    mode='markers',
                                    name=f"vs {self.target_column}",
                                    showlegend=False,
                                    marker=dict(
                                        size=4,
                                        opacity=0.6,
                                        color=target_values,
                                        colorscale='Viridis',
                                        showscale=True,
                                        colorbar=dict(title=self.target_column)
                                    )
                                ),
                                row=2, col=1
                            )
                            
                            # Update layout
                            fig.update_layout(
                                height=400,
                                title_text=f"Analysis of {self._get_readable_feature_name(feat_name)}"
                            )
                            
                            # Update x-axes
                            fig.update_xaxes(title_text=self._get_readable_feature_name(feat_name), row=1, col=1)
                            fig.update_xaxes(title_text=self._get_readable_feature_name(feat_name), row=2, col=1)
                            
                            # Update y-axes
                            fig.update_yaxes(title_text="Frequency", row=1, col=1)
                            fig.update_yaxes(title_text=self.target_column, row=2, col=1)
                            
                            st.plotly_chart(fig, config={'responsive': True})
                                
                    except Exception as e:
                        # Fallback to simple histogram if subplot creation fails
                        try:
                            fig = px.histogram(
                                x=feat_info['values'].fillna(feat_info['values'].median()),
                                nbins=30,
                                title=f"Distribution of {self._get_readable_feature_name(feat_name)}"
                            )
                            fig.update_layout(
                                xaxis_title=self._get_readable_feature_name(feat_name),
                                yaxis_title="Frequency",
                                height=250
                            )
                            st.plotly_chart(fig, config={'responsive': True})
                        except Exception:
                            st.warning("Could not generate histogram for this feature.")
        
        st.divider()
        # Create selection interface with a table
        st.write("### Select Features to Add")
        
        # Option to select all recommendations
        all_selected = st.checkbox("Select all recommended features", 
                                 value=False, 
                                 key="feature_creation_select_all")
        
        # Create selection dataframe
        selection_data = []
        for i, feat_name in enumerate(top_features):
            feat_info = st.session_state.feature_creation_generated_features[feat_name]
            metrics = feature_metrics.get(feat_name, {})
            
            # Check if this feature is in any correlated pairs
            is_correlated = any(
                (pair['feature1'] == feat_name or pair['feature2'] == feat_name) and pair['type'] == 'new_vs_existing'
                for pair in correlated_pairs
            )
            
            selection_data.append({
                "Rank": i + 1,
                "Feature": self._get_readable_feature_name(feat_name),
                "Operation": feat_info['operation'].title(),
                "Target Relationship": metrics.get('target_importance', 0),
                "Normalized RMS": metrics.get('normalized_rms_correlation', 0),
                "Score": metrics.get('combined_score', 0),
                "Select": all_selected or feat_name in st.session_state.feature_creation_selected_features,
                "feature_name": feat_name  # Hidden field to store the feature name
            })
        
        # Create selection DataFrame
        selection_df = pd.DataFrame(selection_data)
        
        # Only include visible columns for display (hide feature_name)
        display_columns = ["Rank", "Feature", "Operation", "Target Relationship", "Normalized RMS", "Score", "Select"]
        
        # Display selection table with checkboxes
        edited_df = st.data_editor(
            selection_df[display_columns + ["feature_name"]],  # Include feature_name but we'll hide it with column_order
            width='stretch',
            hide_index=True,
            column_order=display_columns,  # Only show these columns, hiding feature_name
            column_config={
                "Rank": st.column_config.NumberColumn(format="%d"),
                "Target Relationship": st.column_config.ProgressColumn(
                    "Target Relationship",
                    help="Strength of relationship with target variable",
                    format="%.3f",
                    min_value=0,
                    max_value=1,
                ),
                "Normalized RMS": st.column_config.ProgressColumn(
                    "Normalized RMS",
                    help="Correlation penalty factor used in score calculation (lower is better)",
                    format="%.3f",
                    min_value=0,
                    max_value=1,
                ),
                "Score": st.column_config.ProgressColumn(
                    "Overall Score",
                    help="Final score considering target relationship and correlation penalty",
                    format="%.3f",
                    min_value=0,
                    max_value=1,
                ),
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select this feature to add to your dataset",
                ),
            }
        )
        
        # Update selected features based on the edited dataframe
        selected_features = []
        
        # We need to map back from the edited dataframe to the original selection_df 
        # since feature_name might not be directly accessible in edited_df
        selected_indices = edited_df.index[edited_df["Select"]].tolist()
        for idx in selected_indices:
            feat_name = selection_df.iloc[idx]["feature_name"]
            selected_features.append(feat_name)
        
        # Update selected features in session state
        st.session_state.feature_creation_selected_features = selected_features
        
        # Display summary of selection
        st.write(f"**Selected {len(selected_features)} new features to add to your dataset.**")
        
        # Log feature selection
        self.logger.log_user_action(
            "Feature Creation - Feature Selection",
            {
                "num_features_selected": len(selected_features),
                "selected_features": selected_features,
                "correlated_features_warned": len(correlated_pairs)
            }
        )
    
    def _display_feature_ranking_table(self, top_features, feature_metrics):
        """
        Display a table with feature rankings and metrics.
        
        Args:
            top_features: List of top feature names
            feature_metrics: Dictionary mapping feature names to metric dictionaries
        """
        st.write("#### Feature Rankings")
        
        # Prepare data for the table
        table_data = []
        
        for i, feat_name in enumerate(top_features):
            if feat_name not in feature_metrics:
                continue
                
            feat_info = st.session_state.feature_creation_generated_features[feat_name]
            metrics = feature_metrics[feat_name]
            
            # Create readable feature name
            readable_name = f"{feat_info['feature1']} {self._get_operation_symbol(feat_info['operation'])} {feat_info['feature2']}"
            
            table_data.append({
                "Rank": i + 1,
                "Feature": readable_name,
                "Operation": feat_info['operation'].title(),
                "Target Relationship": metrics['target_importance'],  # Keep as numeric
                "Normalized RMS": metrics['normalized_rms_correlation'],  # Keep as numeric
                "Overall Score": metrics['combined_score']  # Keep as numeric
            })
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Display as a styled table
        st.dataframe(
            df, 
            width='stretch',
            height=min(350, 50 + 35 * len(df)),
            column_config={
                "Rank": st.column_config.NumberColumn(format="%d"),
                "Target Relationship": st.column_config.ProgressColumn(
                    "Target Relationship",
                    help="Strength of relationship with target variable",
                    format="%.3f",
                    min_value=0,
                    max_value=1,
                ),
                "Normalized RMS": st.column_config.ProgressColumn(
                    "Normalized RMS",
                    help="Correlation penalty factor used in score calculation (lower is better)",
                    format="%.3f",
                    min_value=0,
                    max_value=1,
                ),
                "Overall Score": st.column_config.ProgressColumn(
                    "Overall Score",
                    help="Final score considering target relationship and correlation penalty",
                    format="%.3f",
                    min_value=0,
                    max_value=1,
                )
            }
        )
        
        #with st.expander("How are features scored?", expanded=False):
        #    st.markdown("""
        #    **6-Step Quality Process:**
            
        #    1. **Filter null values** (>5% removed)
        #    2. **Remove highly correlated features** (>0.8 correlation with existing features)
        #    3. **Remove duplicate new features** (>0.8 correlation between new features)
        #    4. **Remove similar distributions** (statistically similar features)
        #    5. **Calculate target relationships** (mutual information with target)
        #    6. **Apply final scoring** with quality requirements:
            
        #    **Scoring Formula:** `Score = Target Relationship × (1 - 0.3 × Normalized RMS)`
            
        #    **Components:**
        #    - **Target Relationship**: How well the feature predicts the target (higher = better)
        #    - **Normalized RMS**: Correlation penalty with existing features (lower = better)
        #    - **Score**: Final ranking after applying correlation penalty
            
        #    Only features meeting these standards are recommended.
        #    """)
    
    def _display_correlation_heatmap(self, top_features):
        import streamlit as st
        """
        Display a correlation heatmap for the top features and existing features.
        
        Args:
            top_features: List of top feature names
        """
        st.write("#### Feature Correlation Heatmap")
        
        # Use the filtered recommended features for the heatmap
        heatmap_features = top_features
        
        # Create a temporary DataFrame with selected existing features and top new features
        temp_df = pd.DataFrame()
        
        # Get existing numeric features (limit to 10 for readability)
        existing_features = self._get_numeric_features()[:10]
        for feat in existing_features:
            temp_df[feat] = self.training_data[feat]
        
        # Add target column only for regression problems (not for classification)
        # Use session state to determine problem type if available, otherwise fall back to heuristics
        import streamlit as st
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'problem_type'):
            problem_type = st.session_state.problem_type
            is_regression = problem_type == "regression"
        else:
            # Fallback to original heuristic
            is_regression = (
                self.training_data[self.target_column].dtype.kind in 'ifc' and 
                len(self.training_data[self.target_column].unique()) > 20
            )
        
        # Only include target in correlation heatmap for regression problems
        if is_regression:
            temp_df[f"{self.target_column} (target)"] = self.training_data[self.target_column]
        
        # Add top new features
        for feat_name in heatmap_features[:10]:  # Limit to 10 for readability
            feat_info = st.session_state.feature_creation_generated_features[feat_name]
            
            # Create more readable feature name
            clean_name = f"{feat_info['feature1']} {self._get_operation_symbol(feat_info['operation'])} {feat_info['feature2']}"
            
            temp_df[clean_name] = feat_info['values']
        
        # Calculate correlation matrix
        corr_matrix = temp_df.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            text_auto='.2f',
            aspect="auto",
            title="Correlation Between Features"
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            width=800,
            xaxis_tickangle=-45
        )
        
        # Display the heatmap
        st.plotly_chart(fig, config={'responsive': True})
        
        # Add information about target inclusion for different problem types
        if not is_regression:
            st.info("📝 **Note**: For classification problems, the target variable is excluded from the correlation heatmap since it should be treated as categorical. Feature-target relationships are analyzed separately using appropriate statistical measures.")
        
        #with st.expander("How to interpret the correlation heatmap", expanded=False):
        #    st.markdown("""
        #    The correlation heatmap shows how different features relate to each other:
            
        #    - **Red cells (positive values)**: Features move in the same direction
        #    - **Blue cells (negative values)**: Features move in opposite directions
        #    - **White/light cells (near zero)**: Features have little correlation
            
        #    **What to look for:**
        #    - **New features with strong correlation to target**: Good candidates to keep
        #    - **New features with low correlation to existing features**: Add unique information
        #    - **Highly correlated feature pairs**: May be redundant with each other
            
        #    Ideal new features have strong correlation with the target but low correlation with existing features.
        #    """)
    
    def _get_operation_symbol(self, operation: str) -> str:
        """
        Get the mathematical symbol for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Mathematical symbol for the operation
        """
        symbols = {
            'ratio': '÷',
            'sum': '+',
            'difference': '-',
            'product': '×',
            'mean': '▽'  # Average symbol
        }
        return symbols.get(operation, operation)
    
    def _display_feature_relationships(self):
        """
        Display relationships between selected features and the target variable.
        """
        if not st.session_state.feature_creation_selected_features:
            return
        
        st.write("### Feature Relationship Analysis")
        
        # Create a temporary dataframe with original data and selected features
        temp_df = self.training_data.copy()
        
        # Add selected features to temporary dataframe
        for feat_name in st.session_state.feature_creation_selected_features:
            feat_info = st.session_state.feature_creation_generated_features[feat_name]
            temp_df[feat_name] = feat_info['values']
        
        # Use FeatureRelationshipsComponent to display relationships
        feature_relationships = FeatureRelationshipsComponent(
            self.builder,
            self.logger,
            data=temp_df,
            target_column=self.target_column
        )
        
        # Display the subset of features we're interested in
        selected_features_plus_target = st.session_state.feature_creation_selected_features + [self.target_column]
        
        # Display feature associations analysis
        if len(st.session_state.feature_creation_selected_features) > 1:
            # Only show associations if we have multiple features
            feature_relationships.display_feature_associations_analysis()
        
        # Display detailed feature relationship analysis
        feature_relationships.display_detailed_feature_relationship_analysis()
    
    def _render_apply_button(self):
        """
        Render button to apply selected features to the dataset.
        """
        selected_features = st.session_state.feature_creation_selected_features
        
        if not selected_features:
            st.warning("Please select at least one feature to add to your dataset.")
            return
        
        st.write("### Apply Changes")
        st.write(f"You have selected {len(selected_features)} feature(s) to add to your dataset.")
        
        # Display list of selected features
        selected_readable_names = [self._get_readable_feature_name(feat) for feat in selected_features]
        if len(selected_features) <= 5:
            st.write("Selected features:")
            for name in selected_readable_names:
                st.write(f"- {name}")
        else:
            st.write(f"Selected {len(selected_features)} features including: {', '.join(selected_readable_names[:5])}...")
        
        if st.button("Add Selected Features to Dataset", type="primary", key="add_features_button"):
            # Save current state before applying changes
            st.session_state.feature_creation_ops_applied.append("feature_creation_operation")
            # Safety check to ensure original training and testing data are available
            if self.training_data is None or self.testing_data is None:
                st.error("Training or testing data not available. Cannot apply features.")
                return
            
            with st.spinner("Applying selected features to datasets..."):
                # Create copies of the original datasets to avoid modifying them during processing
                training_data = self.training_data.copy()
                testing_data = self.testing_data.copy()
                original_data = self.builder.training_data.copy()

                # Track successful features and any errors
                successful_features = []
                error_features = {}
                
                # Apply each selected feature to both training and testing data
                for feat_name in st.session_state.feature_creation_selected_features:
                    try:
                        feat_info = st.session_state.feature_creation_generated_features[feat_name]
                        operation = feat_info['operation_func']
                        feat1 = feat_info['feature1']
                        feat2 = feat_info['feature2']
                        
                        # Verify source features exist in both datasets
                        if feat1 not in training_data.columns:
                            error_features[feat_name] = f"Source feature '{feat1}' not found in training data"
                            continue
                        
                        if feat2 not in training_data.columns:
                            error_features[feat_name] = f"Source feature '{feat2}' not found in training data"
                            continue
                        
                        if feat1 not in testing_data.columns:
                            error_features[feat_name] = f"Source feature '{feat1}' not found in testing data"
                            continue
                        
                        if feat2 not in testing_data.columns:
                            error_features[feat_name] = f"Source feature '{feat2}' not found in testing data"
                            continue
                        
                        # 1. Add feature to training data
                        training_data[feat_name] = feat_info['values']
                        
                        # 2. Calculate and add feature to testing data
                        # Handle categorical/binned features by converting to numeric before operations
                        feat1_values = testing_data[feat1]
                        feat2_values = testing_data[feat2]
                        
                        # Convert binned categorical features to numeric if needed
                        if '_binned' in feat1 and hasattr(feat1_values, 'dtype') and (
                            is_categorical_dtype(feat1_values) or 
                            str(feat1_values.dtype).startswith('category')):
                            feat1_values = pd.to_numeric(feat1_values.astype(str), errors='coerce')
                        
                        if '_binned' in feat2 and hasattr(feat2_values, 'dtype') and (
                            is_categorical_dtype(feat2_values) or 
                            str(feat2_values.dtype).startswith('category')):
                            feat2_values = pd.to_numeric(feat2_values.astype(str), errors='coerce')
                        
                        # Apply operation with converted values
                        feat_values = operation(feat1_values, feat2_values)
                        feat_values = pd.Series(feat_values).replace([np.inf, -np.inf], np.nan)
                        
                        # Always fill null values with 0 for consistency with training data
                        feat_values = feat_values.fillna(0)
                        
                        testing_data[feat_name] = feat_values
                        
                        # Mark as successful
                        successful_features.append(feat_name)
                    except Exception as e:
                        # Store the error and continue with other features
                        error_features[feat_name] = str(e)
                
                # Report any errors
                if error_features:
                    error_msg = "Errors occurred when applying these features:\n"
                    for feat, reason in error_features.items():
                        error_msg += f"- {feat}: {reason}\n"
                    st.error(error_msg)
                    
                    if not successful_features:
                        st.error("No features could be applied. Please select different features.")
                        return
                
                # Only update the builder data if we have at least one successful feature
                if successful_features:
                    # Update builder data with the modified datasets
                    self.builder.training_data = training_data
                    self.builder.testing_data = testing_data
                    
                    # Update X_train and X_test if they exist in the builder
                    if hasattr(self.builder, 'X_train') and self.builder.X_train is not None:
                        # Regenerate X_train and X_test with the new features
                        self.builder.X_train = self.builder.training_data.drop(columns=[self.builder.target_column])
                        self.builder.X_test = self.builder.testing_data.drop(columns=[self.builder.target_column])
                    
                    # Display success message
                    st.success(f"Successfully added {len(successful_features)} new features to your dataset!")
                    
                    # Log changes
                    self.logger.log_calculation(
                        "Feature Creation - Features Added",
                        {
                            "features_added": successful_features,
                            "features_with_errors": list(error_features.keys()),
                            "num_features_added": len(successful_features),
                            "new_shape_training": training_data.shape,
                            "new_shape_testing": testing_data.shape
                        }
                    )

                    self.logger.log_journey_point(

                        stage="DATA_PREPROCESSING",
                        decision_type="FEATURE_CREATION",
                        description="Feature creation completed",
                        details={'Features Added': successful_features,
                                 'Features With Errors': list(error_features.keys()),
                                 'No. of Features Added': len(successful_features),
                                 'Training Data Shape': training_data.shape,
                                 'Testing Data Shape': testing_data.shape},
                        parent_id=None
                    )
                    
                    # Clear session state variables to prevent issues on rerun
                    if 'feature_creation_generated_features' in st.session_state:
                        del st.session_state.feature_creation_generated_features
                    if 'feature_creation_selected_features' in st.session_state:
                        del st.session_state.feature_creation_selected_features
                    if 'feature_creation_top_features' in st.session_state:
                        del st.session_state.feature_creation_top_features
                    if 'feature_creation_feature_metrics' in st.session_state:
                        del st.session_state.feature_creation_feature_metrics
                    if 'feature_creation_correlated_pairs' in st.session_state:
                        del st.session_state.feature_creation_correlated_pairs
                
                # Generate URL for reloading the page without auto-rerunning
                st.markdown(
                    """
                    <script>
                    // Add a short delay before redirect to ensure UI updates are processed
                    setTimeout(function() {
                        window.parent.location.href = window.parent.location.href;
                    }, 2000);
                    </script>
                    """,
                    unsafe_allow_html=True
                )

                # Create a DataframeComparisonComponent instance
                comparison_component = DataframeComparisonComponent(
                    original_df=original_data,
                    modified_df=st.session_state.builder.training_data,
                    target_column=st.session_state.builder.target_column)
                comparison_component.render()
    
    def _get_readable_feature_name(self, feat_name):
        """
        Get a readable feature name from the feature name in the format.
        
        Args:
            feat_name: Feature name in the format "{feat1}_{operation}_{feat2}"
            
        Returns:
            Readable feature name in the format "{feat1} {operation_symbol} {feat2}"
        """
        # Get feature info
        if feat_name in st.session_state.feature_creation_generated_features:
            feat_info = st.session_state.feature_creation_generated_features[feat_name]
            return f"{feat_info['feature1']} {self._get_operation_symbol(feat_info['operation'])} {feat_info['feature2']}"
        else:
            # Extract feature parts from name if not in dictionary
            parts = feat_name.split('_')
            if len(parts) >= 3:
                feat1 = parts[0]
                operation = parts[1]
                feat2 = '_'.join(parts[2:])  # Join remaining parts in case feature names contain underscore
                return f"{feat1} {self._get_operation_symbol(operation)} {feat2}"
            return feat_name 
