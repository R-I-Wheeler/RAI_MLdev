import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from dython.nominal import identify_nominal_columns, associations
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.preprocessing import LabelEncoder

def display_info_message(message: str, type: str = "info"):
    """Display a styled message in Streamlit."""
    if type == "info":
        st.info(message, icon="ℹ️")
    elif type == "warning":
        st.warning(message, icon="⚠️")
    elif type == "success":
        st.success(message, icon="✅")
    elif type == "error":
        st.error(message, icon="🚨")

class DataframeComparisonComponent:
    """Component for comparing two dataframes and analyzing changes in features and their relationships with the target."""
    
    def __init__(self, original_df: pd.DataFrame, modified_df: pd.DataFrame, target_column: str):
        """
        Initialize the component with two dataframes to compare and the target column.
        
        Args:
            original_df: The original dataframe
            modified_df: The modified dataframe
            target_column: Name of the target column
        """
        # Validate inputs
        if original_df is None or modified_df is None:
            raise ValueError("Both original and modified dataframes must be provided")
        if not isinstance(original_df, pd.DataFrame) or not isinstance(modified_df, pd.DataFrame):
            raise ValueError("Both inputs must be pandas DataFrames")
        if len(original_df) == 0 or len(modified_df) == 0:
            raise ValueError("Both dataframes must contain data")
            
        self.original_df = original_df
        self.modified_df = modified_df
        self.target_column = target_column
        self.target_type = self._determine_target_type()
        self._setup_explanations()
        
    def _determine_target_type(self) -> str:
        """Determine the target type using session state information."""
        try:
            if self.target_column not in self.original_df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in original dataframe")
            
            # Use session state information if available
            if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type:
                return st.session_state.problem_type
            elif hasattr(st.session_state, 'is_regression') and st.session_state.is_regression:
                return "regression"
            elif hasattr(st.session_state, 'is_binary') and st.session_state.is_binary:
                return "binary_classification"
            elif hasattr(st.session_state, 'is_multiclass') and st.session_state.is_multiclass:
                return "multiclass_classification"
            else:
                # Fallback to original logic if session state not available
                target_data = self.original_df[self.target_column]
                if pd.api.types.is_numeric_dtype(target_data):
                    unique_count = target_data.nunique()
                    if unique_count <= 2:
                        return "binary_classification"
                    elif unique_count <= 10:  # Reasonable threshold for multiclass
                        return "multiclass_classification"
                    else:
                        return "regression"
                else:
                    unique_count = target_data.nunique()
                    if unique_count <= 2:
                        return "binary_classification"
                    else:
                        return "multiclass_classification"
        except Exception as e:
            st.error(f"Error determining target type: {str(e)}")
            raise

    def _setup_explanations(self):
        """Setup explanation dictionaries for statistical tests and visualisations."""
        self.test_explanations = {
            "ks_test": {
                "title": "Kolmogorov-Smirnov Test",
                "description": """
                The Kolmogorov-Smirnov test compares two distributions to determine if they are significantly different.
                
                **Interpretation:**
                - **Statistic:** Ranges from 0 to 1. Higher values indicate more different distributions.
                - **P-value:** If less than 0.05, the distributions are significantly different.
                
                **When is it used?**
                - For comparing continuous numeric features
                - When we want to detect any type of distribution change
                """
            },
            "chi_square": {
                "title": "Chi-square Test",
                "description": """
                The Chi-square test compares categorical distributions to determine if they are independent.
                
                **Interpretation:**
                - **Statistic:** Higher values indicate stronger evidence of different distributions.
                - **P-value:** If less than 0.05, the distributions are significantly different.
                
                **When is it used?**
                - For categorical features
                - When comparing frequency distributions
                """
            },
            "point_biserial": {
                "title": "Point-Biserial Correlation",
                "description": """
                Point-biserial correlation measures the relationship between a numeric feature and a binary target.
                
                **Interpretation:**
                - **Correlation:** Ranges from -1 to 1
                    - -1: Perfect negative relationship
                    - 0: No relationship
                    - 1: Perfect positive relationship
                - **P-value:** If less than 0.05, the relationship is statistically significant
                
                **Strength Guidelines:**
                - 0.7 to 1.0: Strong relationship
                - 0.3 to 0.7: Moderate relationship
                - 0.0 to 0.3: Weak relationship
                """
            },
            "cramers_v": {
                "title": "Cramer's V",
                "description": """
                Cramer's V measures the association between categorical variables.
                
                **Interpretation:**
                - **Value:** Ranges from 0 to 1
                    - 0: No association
                    - 1: Perfect association
                
                **Strength Guidelines:**
                - > 0.5: Strong association
                - 0.3 to 0.5: Moderate association
                - < 0.3: Weak association
                
                **When is it used?**
                - For categorical features with binary targets
                - When comparing categorical relationships
                """
            },
            "pearson": {
                "title": "Pearson Correlation",
                "description": """
                Pearson correlation measures the linear relationship between two numeric variables.
                
                **Interpretation:**
                - **Correlation:** Ranges from -1 to 1
                    - -1: Perfect negative correlation
                    - 0: No correlation
                    - 1: Perfect positive correlation
                - **P-value:** If less than 0.05, the correlation is statistically significant
                
                **Strength Guidelines:**
                - 0.7 to 1.0: Strong correlation
                - 0.3 to 0.7: Moderate correlation
                - 0.0 to 0.3: Weak correlation
                """
            },
            "eta_squared": {
                "title": "Eta-squared (η²)",
                "description": """
                Eta-squared measures the proportion of variance explained by group differences.
                
                **For Regression:** Proportion of variance in numeric target explained by a categorical feature.
                **For Classification:** Proportion of variance in numeric feature explained by class differences.
                
                **Interpretation:**
                - **Value:** Ranges from 0 to 1
                    - 0: No effect
                    - 1: Perfect explanation
                
                **Effect Size Guidelines:**
                - > 0.26: Large effect
                - 0.13 to 0.26: Medium effect
                - 0.02 to 0.13: Small effect
                """
            }
        }
        
        self.visualization_explanations = {
            "correlation_heatmap": {
                "title": "Correlation Heatmap",
                "description": """
                A heatmap showing the strength and direction of relationships between features.
                
                **How to Read:**
                - Colors: Represent correlation strength
                    - Red: Positive correlation/relationship
                    - Blue: Negative correlation/relationship
                    - White/Light: No correlation/relationship
                - Numbers: Exact correlation values
                
                **Metrics Used and Their Interpretation:**
                
                1. **Pearson Correlation (Numeric ↔ Numeric)**
                   - Range: -1 to 1
                   - Interpretation:
                     * -1: Perfect negative linear relationship
                     * 0: No linear relationship
                     * 1: Perfect positive linear relationship
                   - Strength Guidelines:
                     * |0.7 - 1.0|: Strong relationship
                     * |0.4 - 0.7|: Moderate relationship
                     * |0.2 - 0.4|: Weak relationship
                     * |0.0 - 0.2|: Very weak/no relationship
                
                2. **Cramer's V (Categorical ↔ Categorical)**
                   - Range: 0 to 1
                   - Interpretation:
                     * 0: No association
                     * 1: Perfect association
                   - Strength Guidelines:
                     * 0.5+: Strong association
                     * 0.3 - 0.5: Moderate association
                     * 0.1 - 0.3: Weak association
                     * 0.0 - 0.1: Very weak/no association
                   - Note: Only shows strength, not direction
                
                3. **Correlation Ratio (η) (Numeric ↔ Categorical)**
                   - Range: 0 to 1
                   - Interpretation:
                     * 0: No relationship
                     * 1: Perfect relationship
                   - Strength Guidelines:
                     * 0.6+: Strong relationship
                     * 0.4 - 0.6: Moderate relationship
                     * 0.2 - 0.4: Weak relationship
                     * 0.0 - 0.2: Very weak/no relationship
                   - Note: Measures how well categories predict numeric values
                
                **Changes in Correlations:**
                - Positive values (red): Relationship strengthened
                - Negative values (blue): Relationship weakened
                - "Changed": Different metrics used due to data type changes
                
                **What to Look For:**
                - Strong correlations between features (potential redundancy)
                - Changes in relationship strengths after modifications
                - Features with strong relationships to the target
                - Clusters of related features
                """
            },
            "histogram": {
                "title": "Histogram",
                "description": """
                A histogram shows the distribution of numeric data by splitting it into bins.
                
                **How to Read:**
                - X-axis: Value ranges (bins)
                - Y-axis: Count or frequency
                - Height of bars: Number of observations in each bin
                
                **What to Look For:**
                - Shape of distribution (normal, skewed, bimodal)
                - Outliers or unusual patterns
                - Changes in distribution between original and modified data
                """
            },
            "bar_chart": {
                "title": "Bar Chart",
                "description": """
                A bar chart shows the frequency or count of categorical values.
                
                **How to Read:**
                - X-axis: Categories
                - Y-axis: Count or frequency
                - Height of bars: Number of observations in each category
                
                **What to Look For:**
                - Most common categories
                - Rare categories
                - Changes in category frequencies
                """
            }
        }

    def _show_test_explanation(self, test_type: str):
        """Display explanation for a statistical test."""
        if test_type in self.test_explanations:
            with st.expander(f"ℹ️ Understanding {self.test_explanations[test_type]['title']}"):
                st.markdown(self.test_explanations[test_type]['description'])
    
    def _show_visualization_explanation(self, viz_type: str):
        """Display explanation for a visualization type."""
        if viz_type in self.visualization_explanations:
            with st.expander(f"ℹ️ How to Read This {self.visualization_explanations[viz_type]['title']}"):
                st.markdown(self.visualization_explanations[viz_type]['description'])

    def _is_binary_feature(self, series: pd.Series) -> bool:
        """Check if a series represents binary data in any common format."""
        # Drop NA values for the check
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return False
            
        # Convert to strings and lowercase for comparison
        str_vals = clean_series.astype(str).str.lower()
        unique_vals = set(str_vals)
        
        # Common binary value pairs
        binary_pairs = [
            {'0', '1'},
            {'false', 'true'},
            {'f', 't'},
            {'no', 'yes'},
            {'n', 'y'}
        ]
        
        # Check if values match any binary pair
        return len(unique_vals) <= 2 and any(unique_vals <= pair for pair in binary_pairs)

    def _normalize_binary_values(self, series: pd.Series) -> pd.Series:
        """Convert binary values to 0/1 format for comparison."""
        # Drop NA values for the conversion
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return series
            
        # Convert to strings and lowercase for matching
        str_vals = clean_series.astype(str).str.lower()
        
        # Define mapping for common binary values
        true_values = {'1', 'true', 't', 'yes', 'y'}
        false_values = {'0', 'false', 'f', 'no', 'n'}
        
        # Create normalized series
        normalized = pd.Series(index=series.index, dtype=float)
        normalized[str_vals.isin(true_values)] = 1
        normalized[str_vals.isin(false_values)] = 0
        normalized[~series.index.isin(clean_series.index)] = np.nan
        
        return normalized

    def _analyse_feature_changes(self) -> Dict[str, Any]:
        """Analyse changes between the two dataframes."""
        changes = {
            "new_features": [],
            "removed_features": [],
            "dtype_changes": [],
            "distribution_changes": [],
            "all_distribution_changes": [],  # Track all changes regardless of significance
            "missing_value_changes": {},
            "row_count_change": len(self.modified_df) - len(self.original_df)
        }
        
        # Find new and removed features
        original_cols = set(self.original_df.columns)
        modified_cols = set(self.modified_df.columns)
        changes["new_features"] = list(modified_cols - original_cols)
        changes["removed_features"] = list(original_cols - modified_cols)
        
        # Analyse common features
        common_features = list(original_cols.intersection(modified_cols))
        for feature in common_features:
            # Check dtype changes
            orig_dtype = self.original_df[feature].dtype
            mod_dtype = self.modified_df[feature].dtype
            if orig_dtype != mod_dtype:
                changes["dtype_changes"].append({
                    "feature": feature,
                    "original_dtype": str(orig_dtype),
                    "modified_dtype": str(mod_dtype)
                })
            
            # Check missing values
            orig_missing = self.original_df[feature].isna().sum()
            mod_missing = self.modified_df[feature].isna().sum()
            if orig_missing != mod_missing:
                changes["missing_value_changes"][feature] = {
                    "original": orig_missing,
                    "modified": mod_missing,
                    "difference": mod_missing - orig_missing
                }
            
            # Check if either version is binary
            is_orig_binary = self._is_binary_feature(self.original_df[feature])
            is_mod_binary = self._is_binary_feature(self.modified_df[feature])
            
            # Special handling for binary features and categorical to binary conversions
            if is_orig_binary or is_mod_binary:
                # Get value counts for both versions
                orig_counts = self.original_df[feature].value_counts(normalize=True)
                mod_counts = self.modified_df[feature].value_counts(normalize=True)
                
                # For categorical to binary conversion, we need to map the categories
                if (pd.api.types.is_object_dtype(orig_dtype) or 
                    isinstance(orig_dtype, pd.CategoricalDtype)) and pd.api.types.is_numeric_dtype(mod_dtype):
                    # Get unique values from both
                    orig_unique = set(self.original_df[feature].dropna().unique())
                    mod_unique = set(self.modified_df[feature].dropna().unique())
                    
                    # If original has two categories and modified is 0/1
                    if len(orig_unique) == 2 and mod_unique.issubset({0, 1}):
                        # Compare proportions of the first category
                        orig_prop = orig_counts.iloc[0]
                        mod_prop = mod_counts.get(1, mod_counts.get(True, 0))  # Get proportion of 1/True
                        
                        # Calculate absolute difference in proportions
                        # Convert to float explicitly to avoid boolean subtraction issues
                        prop_diff = abs(float(orig_prop) - float(mod_prop))
                        
                        # Use chi-square test for significance
                        orig_freqs = self.original_df[feature].value_counts()
                        mod_freqs = self.modified_df[feature].value_counts()
                        
                        contingency = pd.DataFrame({
                            'original': [orig_freqs.iloc[0], orig_freqs.iloc[1]],
                            'modified': [mod_freqs.get(1, mod_freqs.get(True, 0)), 
                                       mod_freqs.get(0, mod_freqs.get(False, 0))]
                        })
                        
                        try:
                            chi2, p_value, _, _ = chi2_contingency(contingency)
                            # Track all changes where proportions differ
                            if prop_diff > 0:
                                changes["all_distribution_changes"].append({
                                    "feature": feature,
                                    "test": "Binary",
                                    "statistic": prop_diff,
                                    "p_value": p_value
                                })
                            # Only track significant changes for detailed analysis
                            if p_value < 0.05:
                                changes["distribution_changes"].append({
                                    "feature": feature,
                                    "test": "Binary",
                                    "statistic": prop_diff,
                                    "p_value": p_value
                                })
                        except ValueError:
                            pass
                else:
                    # Both are binary, compare proportions directly
                    # Normalize both to 0/1 format first
                    orig_norm = self._normalize_binary_values(self.original_df[feature])
                    mod_norm = self._normalize_binary_values(self.modified_df[feature])
                
                    # Get proportions of 1s
                    orig_prop = (orig_norm == 1).mean()
                    mod_prop = (mod_norm == 1).mean()
                    
                    # Calculate absolute difference in proportions
                    # Convert to float explicitly to avoid boolean subtraction issues
                    prop_diff = abs(float(orig_prop) - float(mod_prop))
                    
                    # Use chi-square test for significance
                    contingency = pd.DataFrame({
                        'original': [(orig_norm == 1).sum(), (orig_norm == 0).sum()],
                        'modified': [(mod_norm == 1).sum(), (mod_norm == 0).sum()]
                    })
                    
                    try:
                        chi2, p_value, _, _ = chi2_contingency(contingency)
                        # Track all changes where proportions differ
                        if prop_diff > 0:
                            changes["all_distribution_changes"].append({
                                "feature": feature,
                                "test": "Binary",
                                "statistic": prop_diff,
                                "p_value": p_value
                            })
                        # Only track significant changes for detailed analysis
                        if p_value < 0.05:
                            changes["distribution_changes"].append({
                                "feature": feature,
                                "test": "Binary",
                                    "statistic": prop_diff,
                                "p_value": p_value
                            })
                    except ValueError:
                        pass
            else:
                # Regular distribution change detection for non-binary features
                if pd.api.types.is_numeric_dtype(self.original_df[feature]) and \
                   pd.api.types.is_numeric_dtype(self.modified_df[feature]):
                    # For numeric features, use KS test
                    orig_data = self.original_df[feature].dropna()
                    mod_data = self.modified_df[feature].dropna()
                    if len(orig_data) > 0 and len(mod_data) > 0:
                        statistic, p_value = ks_2samp(orig_data, mod_data)
                        # Track all changes where statistic indicates a difference
                        if statistic > 0:
                            changes["all_distribution_changes"].append({
                                "feature": feature,
                                "test": "KS",
                                "statistic": statistic,
                                "p_value": p_value
                            })
                        # Only track significant changes for detailed analysis
                        if p_value < 0.05:
                            changes["distribution_changes"].append({
                                "feature": feature,
                                "test": "KS",
                                "statistic": statistic,
                                "p_value": p_value
                            })
                elif pd.api.types.is_categorical_dtype(self.original_df[feature]) or \
                     isinstance(self.original_df[feature].dtype, pd.CategoricalDtype) or \
                     self.original_df[feature].dtype == 'object':
                    # For categorical features, use Chi-square test
                    orig_counts = self.original_df[feature].value_counts()
                    mod_counts = self.modified_df[feature].value_counts()
                    all_categories = list(set(orig_counts.index) | set(mod_counts.index))
                    
                    # Create contingency table
                    contingency = pd.DataFrame({
                        'original': [orig_counts.get(cat, 0) for cat in all_categories],
                        'modified': [mod_counts.get(cat, 0) for cat in all_categories]
                    }, index=all_categories)
                    
                    try:
                        chi2, p_value, _, _ = chi2_contingency(contingency)
                        # Track all changes where chi2 indicates a difference
                        if chi2 > 0:
                            changes["all_distribution_changes"].append({
                                "feature": feature,
                                "test": "Chi-square",
                                "statistic": chi2,
                                "p_value": p_value
                            })
                        # Only track significant changes for detailed analysis
                        if p_value < 0.05:
                            changes["distribution_changes"].append({
                                "feature": feature,
                                "test": "Chi-square",
                                "statistic": chi2,
                                "p_value": p_value
                            })
                    except ValueError:
                        # Handle cases where chi-square test cannot be performed
                        pass
        
        return changes

    def _calculate_feature_target_relationship(
        self, 
        data: pd.DataFrame, 
        feature: str
    ) -> Dict[str, float]:
        """Calculate the relationship strength between a feature and the target."""
        if feature == self.target_column:
            return {}
            
        result = {}
        feature_data = data[feature]
        target_data = data[self.target_column]
        
        # Remove missing values
        valid_mask = ~(feature_data.isna() | target_data.isna())
        feature_data = feature_data[valid_mask]
        target_data = target_data[valid_mask]
        
        if len(feature_data) < 2:
            return {}
            
        if self.target_type in ["binary_classification", "multiclass_classification"]:
            if pd.api.types.is_numeric_dtype(feature_data):
                if self.target_type == "binary_classification":
                    # Point-biserial correlation for numeric features with binary target
                    try:
                        # Convert target to numeric if needed
                        if not pd.api.types.is_numeric_dtype(target_data):
                            le = LabelEncoder()
                            target_numeric = le.fit_transform(target_data)
                        else:
                            target_numeric = target_data
                        
                        correlation, p_value = stats.pointbiserialr(target_numeric, feature_data)
                        result.update({
                            "correlation": correlation,
                            "p_value": p_value,
                            "method": "point_biserial"
                        })
                    except Exception:
                        pass
                else:  # multiclass_classification
                    # ANOVA for numeric features with multiclass target
                    try:
                        groups = [group for name, group in feature_data.groupby(target_data)]
                        if len(groups) >= 2:
                            f_stat, p_value = stats.f_oneway(*groups)
                            # Calculate effect size (eta-squared)
                            n = len(feature_data)
                            groups_df = len(groups) - 1
                            eta_squared = (f_stat * groups_df) / (f_stat * groups_df + (n - groups_df - 1))
                            result.update({
                                "correlation": eta_squared,
                                "p_value": p_value,
                                "method": "eta_squared"
                            })
                    except Exception:
                        pass
            else:
                # Chi-square test for categorical features with classification targets
                try:
                    contingency = pd.crosstab(feature_data, target_data)
                    chi2, p_value, _, _ = chi2_contingency(contingency)
                    # Calculate Cramer's V
                    n = len(feature_data)
                    min_dim = min(contingency.shape) - 1
                    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                    result.update({
                        "correlation": cramers_v,
                        "p_value": p_value,
                        "method": "cramers_v"
                    })
                except Exception:
                    pass
        else:  # regression
            if pd.api.types.is_numeric_dtype(feature_data):
                # Pearson correlation for numeric features
                try:
                    correlation, p_value = stats.pearsonr(feature_data, target_data)
                    result.update({
                        "correlation": correlation,
                        "p_value": p_value,
                        "method": "pearson"
                    })
                except Exception:
                    pass
            else:
                # ANOVA for categorical features
                try:
                    groups = [group for name, group in target_data.groupby(feature_data)]
                    if len(groups) >= 2:
                        f_stat, p_value = stats.f_oneway(*groups)
                        # Calculate effect size (eta-squared)
                        n = len(feature_data)
                        groups_df = len(groups) - 1
                        eta_squared = (f_stat * groups_df) / (f_stat * groups_df + (n - groups_df - 1))
                        result.update({
                            "correlation": eta_squared,
                            "p_value": p_value,
                            "method": "eta_squared"
                        })
                except Exception:
                    pass
                    
        return result

    def _get_association_metric(self, df: pd.DataFrame, feature1: str, feature2: str) -> Tuple[float, str]:
        """
        Calculate the appropriate association metric between two features based on their data types.
        
        Args:
            df: DataFrame containing the features
            feature1: Name of first feature
            feature2: Name of second feature
            
        Returns:
            Tuple of (correlation value, metric name)
        """
        try:
            # Get data and remove missing values
            data1 = df[feature1]
            data2 = df[feature2]
            mask = ~(data1.isna() | data2.isna())
            data1 = data1[mask]
            data2 = data2[mask]
            
            if len(data1) < 2 or len(data2) < 2:
                return 0.0, "insufficient_data"
            
            # Determine data types
            is_numeric1 = pd.api.types.is_numeric_dtype(data1)
            is_numeric2 = pd.api.types.is_numeric_dtype(data2)
            
            if is_numeric1 and is_numeric2:
                # Both numeric - use Pearson correlation
                corr, _ = stats.pearsonr(data1, data2)
                return corr, "pearson"
            elif not is_numeric1 and not is_numeric2:
                # Both categorical - use Cramer's V
                contingency = pd.crosstab(data1, data2)
                chi2, _, _, _ = chi2_contingency(contingency)
                n = len(data1)
                min_dim = min(contingency.shape) - 1
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                return cramers_v, "cramers_v"
            else:
                # Mixed types - use correlation ratio (eta)
                if is_numeric1:
                    numeric_data = data1
                    categorical_data = data2
                else:
                    numeric_data = data2
                    categorical_data = data1
                
                # Calculate correlation ratio
                categories = categorical_data.unique()
                grand_mean = numeric_data.mean()
                n_total = len(numeric_data)
                
                ss_total = ((numeric_data - grand_mean) ** 2).sum()
                if ss_total == 0:
                    return 0.0, "correlation_ratio"
                
                ss_between = sum(
                    len(numeric_data[categorical_data == cat]) * 
                    (numeric_data[categorical_data == cat].mean() - grand_mean) ** 2 
                    for cat in categories
                )
                
                eta = np.sqrt(ss_between / ss_total)
                return eta, "correlation_ratio"
                
        except Exception:
            return 0.0, "error"

    def _get_metric_abbreviation(self, metric: str) -> str:
        """Get abbreviated form of metric name."""
        metric_map = {
            "pearson": "pr",
            "spearman": "ρ",
            "cramers_v": "cv",
            "correlation_ratio": "cr",
            "eta_squared": "η²",
            "error": "err",
            "insufficient_data": "n/a"
        }
        return metric_map.get(metric, metric)

    def _get_effect_size_interpretation(self, effect_size: float, metric_type: str) -> str:
        """Get interpretation of effect size based on metric type.
        
        Args:
            effect_size: The calculated effect size value
            metric_type: The type of metric ('eta_squared', 'cramers_v', 'correlation', etc.)
            
        Returns:
            str: Description of the effect size strength
        """
        if metric_type == "eta_squared":
            if effect_size >= 0.26:
                return "Strong"
            elif effect_size >= 0.13:
                return "Moderate"
            elif effect_size >= 0.02:
                return "Weak"
            else:
                return "Very Weak"
        elif metric_type == "cramers_v":
            if effect_size >= 0.5:
                return "Strong"
            elif effect_size >= 0.3:
                return "Moderate"
            elif effect_size >= 0.1:
                return "Weak"
            else:
                return "Very Weak"
        elif metric_type == "Binary":
            if effect_size >= 0.3:
                return "Strong"
            elif effect_size >= 0.2:
                return "Moderate"
            elif effect_size >= 0.1:
                return "Weak"
            else:
                return "Very Weak"
        elif metric_type == "KS":  # Add interpretation for KS test
            if effect_size >= 0.5:
                return "Strong"
            elif effect_size >= 0.3:
                return "Moderate"
            elif effect_size >= 0.1:
                return "Weak"
            else:
                return "Very Weak"
        elif metric_type in ["correlation", "pearson", "spearman", "point_biserial"]:
            effect_size = abs(effect_size)  # Use absolute value for correlations
            if effect_size >= 0.7:
                return "Strong"
            elif effect_size >= 0.4:
                return "Moderate"
            elif effect_size >= 0.2:
                return "Weak"
            else:
                return "Very Weak"
        else:
            return "Undetermined"

    def _format_p_value(self, p_value) -> str:
        """Format p-value with significance indicators."""
        if p_value is None:
            return "N/A"
        try:
            if p_value < 0.001:
                return f"<0.001 ***"
            elif p_value < 0.01:
                return f"{p_value:.3f} **"
            elif p_value < 0.05:
                return f"{p_value:.3f} *"
            else:
                return f"{p_value:.3f}"
        except:
            return "N/A"

    def _add_metric_legend(self, fig: go.Figure) -> go.Figure:
        """Add metric abbreviation legend to the figure."""
        legend_text = (
            "Metric Types:<br>"
            "pr: Pearson (Numeric↔Numeric)<br>"
            "cv: Cramer's V (Categorical↔Categorical)<br>"
            "cr: Correlation Ratio (Numeric↔Categorical)<br>"
            "—: Self-correlation<br>"
            "n/a: Insufficient Data"
        )
        
        # Add legend as annotation in top-right corner
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1.15,  # Position legend to the right of the heatmap
            y=1,
            text=legend_text,
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
        # Adjust layout to make room for legend
        fig.update_layout(
            margin=dict(r=150)  # Increase right margin
        )
        
        return fig

    def _analyse_associations(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure]:
        """Calculate and visualize feature associations."""
        # Create a copy of the data and treat classification targets as categorical
        data_copy = data.copy()
        if self.target_type in ["binary_classification", "multiclass_classification"]:
            if self.target_column in data_copy.columns:
                # Convert target to categorical for proper association analysis
                data_copy[self.target_column] = data_copy[self.target_column].astype('category')
        
        # Create empty correlation matrix
        features = data_copy.columns
        n_features = len(features)
        correlation_matrix = pd.DataFrame(
            np.zeros((n_features, n_features)),
            index=features,
            columns=features
        )
        
        # Fill correlation matrix with appropriate metrics
        metric_types = pd.DataFrame(
            np.empty((n_features, n_features), dtype=object),
            index=features,
            columns=features
        )
        
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i != j:
                    corr, metric = self._get_association_metric(data_copy, feat1, feat2)
                    correlation_matrix.iloc[i, j] = corr
                    metric_types.iloc[i, j] = metric
                else:
                    correlation_matrix.iloc[i, j] = 1.0
                    metric_types.iloc[i, j] = "self"
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            hovertemplate='%{y} → %{x}<br>' +
                         'Correlation: %{z:.3f}<br>' +
                         'Metric: %{customdata}<extra></extra>',
            customdata=metric_types.values
        ))
        
        # Add annotations
        annotations = []
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    metric_abbr = self._get_metric_abbreviation(metric_types.iloc[i, j])
                    annotations.append(
                        dict(
                            x=j,
                            y=i,
                            text=f"{correlation_matrix.iloc[i, j]:.2f}<br>{metric_abbr}",
                            showarrow=False,
                            font=dict(size=10)
                        )
                    )
        
        fig.update_layout(
            height=800,
            width=800,
            xaxis_tickangle=-45,
            annotations=annotations
        )
        
        # Add metric legend
        fig = self._add_metric_legend(fig)
        
        return correlation_matrix, fig, metric_types

    def _compare_associations(self, orig_corr: pd.DataFrame, mod_corr: pd.DataFrame, 
                            orig_metrics: pd.DataFrame, mod_metrics: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure]:
        """Compare association matrices accounting for metric changes."""
        # Get all features from both dataframes
        all_features = list(set(mod_corr.index) | set(orig_corr.index))
        
        if not all_features:
            return pd.DataFrame(), None
            
        # Initialize change matrix with all features
        changes = pd.DataFrame(
            np.zeros((len(all_features), len(all_features))),
            index=all_features,
            columns=all_features
        )
        
        # Initialize annotations matrix and metric transitions matrix
        annotations = pd.DataFrame(
            np.empty((len(all_features), len(all_features)), dtype=object),
            index=all_features,
            columns=all_features
        )
        
        metric_info = pd.DataFrame(
            np.empty((len(all_features), len(all_features)), dtype=object),
            index=all_features,
            columns=all_features
        )
        
        # Calculate changes and create annotations
        for i, feat1 in enumerate(all_features):
            for j, feat2 in enumerate(all_features):
                if i != j:  # Skip diagonal
                    # Check if features exist in both dataframes
                    in_orig = feat1 in orig_corr.index and feat2 in orig_corr.index
                    in_mod = feat1 in mod_corr.index and feat2 in mod_corr.index
                    
                    try:
                        if in_orig and in_mod:
                            # Both features exist in both dataframes
                            orig_val = float(abs(orig_corr.loc[feat1, feat2]))
                            mod_val = float(abs(mod_corr.loc[feat1, feat2]))
                            orig_metric = orig_metrics.loc[feat1, feat2]
                            mod_metric = mod_metrics.loc[feat1, feat2]
                            
                            # Calculate change regardless of metric type
                            change = mod_val - orig_val
                            changes.loc[feat1, feat2] = change
                            
                            if orig_metric == mod_metric and orig_metric != "error" and orig_metric != "insufficient_data":
                                # Same metric type - show just the change and metric
                                metric_abbr = self._get_metric_abbreviation(mod_metric)
                                annotations.loc[feat1, feat2] = f"{change:.2f}<br>{metric_abbr}"
                                metric_info.loc[feat1, feat2] = mod_metric
                            else:
                                # Different metrics - show change and metric transition
                                orig_abbr = self._get_metric_abbreviation(orig_metric)
                                mod_abbr = self._get_metric_abbreviation(mod_metric)
                                annotations.loc[feat1, feat2] = f"{change:.2f}<br>{orig_abbr}→{mod_abbr}"
                                metric_info.loc[feat1, feat2] = f"{orig_metric} → {mod_metric}"
                        elif in_mod:
                            # New feature pair - use absolute value from modified
                            mod_val = float(abs(mod_corr.loc[feat1, feat2]))
                            mod_metric = mod_metrics.loc[feat1, feat2]
                            if mod_metric != "error" and mod_metric != "insufficient_data":
                                changes.loc[feat1, feat2] = mod_val
                                metric_abbr = self._get_metric_abbreviation(mod_metric)
                                annotations.loc[feat1, feat2] = f"{mod_val:.2f}<br>{metric_abbr} (new)"
                                metric_info.loc[feat1, feat2] = f"{mod_metric} (new)"
                            else:
                                changes.loc[feat1, feat2] = 0
                                annotations.loc[feat1, feat2] = "n/a"
                                metric_info.loc[feat1, feat2] = "insufficient data"
                    except Exception as e:
                        print(f"Error processing {feat1}-{feat2}: {str(e)}")
                        changes.loc[feat1, feat2] = 0
                        annotations.loc[feat1, feat2] = "error"
                        metric_info.loc[feat1, feat2] = "error"
                else:
                    # Diagonal elements
                    changes.loc[feat1, feat2] = 0
                    annotations.loc[feat1, feat2] = "—"
                    metric_info.loc[feat1, feat2] = "self"
        
        # Create visualization
        fig = go.Figure(data=go.Heatmap(
            z=changes.values,
            x=changes.columns,
            y=changes.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            hovertemplate='%{y} → %{x}<br>' +
                         'Change: %{z:.3f}<br>' +
                         'Metric: %{customdata}<extra></extra>',
            customdata=metric_info.values
        ))
        
        # Add annotations
        fig_annotations = []
        for i, feat1 in enumerate(all_features):
            for j, feat2 in enumerate(all_features):
                if pd.notna(annotations.loc[feat1, feat2]):  # Only add annotation if not NaN
                    fig_annotations.append(
                        dict(
                            x=j,
                            y=i,
                            text=annotations.loc[feat1, feat2],
                            showarrow=False,
                            font=dict(size=12)
                        )
                    )
        
        fig.update_layout(
            title='Changes in Feature Associations',
            height=800,
            width=800,
            xaxis_tickangle=-45,
            annotations=fig_annotations
        )
        
        # Add metric legend
        fig = self._add_metric_legend(fig)
        
        return changes, fig

    def _create_feature_distribution_plot(
        self,
        data: pd.DataFrame,
        feature: str,
        title: str
    ) -> List[go.Figure]:
        """Create distribution plots that include target relationship visualization.
        
        Returns a list of figures for different aspects of the distribution.
        """
        figures = []
        
        # Check if the feature is binned/categorical but stored as numeric
        unique_values = data[feature].nunique()
        is_binned_numeric = pd.api.types.is_numeric_dtype(data[feature]) and unique_values <= 20
        
        if pd.api.types.is_numeric_dtype(data[feature]) and not is_binned_numeric:
            if self.target_type in ["binary_classification", "multiclass_classification"]:
                # Create separate histogram for each class
                target_classes = sorted(data[self.target_column].unique())
                
                # Histogram by class
                fig_hist = px.histogram(
                    data,
                    x=feature,
                    color=self.target_column,
                    title=f"Distribution by Class - {title}",
                    opacity=0.7,
                    nbins=30,
                    labels={feature: feature, self.target_column: "Target Class"},
                    barmode='overlay'
                )
                
                fig_hist.update_layout(
                    height=400,
                    showlegend=True,
                    hovermode='x unified',
                    xaxis_title=feature,
                    yaxis_title="Count",
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                # Add grid lines
                fig_hist.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig_hist.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                figures.append(fig_hist)
                
                # Box plot by class
                fig_box = px.box(
                    data,
                    x=self.target_column,
                    y=feature,
                    title=f"Box Plot by Class - {title}",
                    points="outliers",
                    labels={feature: feature, self.target_column: "Target Class"}
                )
                
                fig_box.update_layout(
                    height=400,
                    showlegend=False,
                    hovermode='closest',
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                # Add grid lines
                fig_box.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig_box.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                figures.append(fig_box)
                
            else:  # regression
                # Histogram of feature
                fig_hist = px.histogram(
                    data,
                    x=feature,
                    title=f"Feature Distribution - {title}",
                    nbins=30,
                    labels={feature: feature}
                )
                
                fig_hist.update_layout(
                    height=400,
                    showlegend=False,
                    hovermode='x unified',
                    xaxis_title=feature,
                    yaxis_title="Count",
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                # Add grid lines
                fig_hist.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig_hist.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                figures.append(fig_hist)
                
                # Scatter plot with target
                fig_scatter = px.scatter(
                    data.dropna(subset=[feature, self.target_column]),
                    x=feature,
                    y=self.target_column,
                    title=f"Relationship with Target - {title}",
                    labels={feature: feature, self.target_column: self.target_column},
                    trendline="ols"  # Add trend line
                )
                
                fig_scatter.update_layout(
                    height=400,
                    showlegend=False,
                    hovermode='closest',
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                # Add grid lines
                fig_scatter.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig_scatter.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                figures.append(fig_scatter)
                
        else:  # categorical feature or binned numeric
            if self.target_type in ["binary_classification", "multiclass_classification"]:
                # Bar chart of category counts
                value_counts = data[feature].value_counts().sort_index()
                
                # Bar chart for counts
                fig_counts = px.bar(
                    x=[str(x) for x in value_counts.index],
                    y=value_counts.values,
                    title=f"Category Counts - {title}",
                    labels={"x": feature, "y": "Count"}
                )
                
                fig_counts.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                # Add grid lines
                fig_counts.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig_counts.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                figures.append(fig_counts)
                
                # Stacked bar chart for target proportions
                proportions = pd.crosstab(
                    data[feature],
                    data[self.target_column],
                    normalize='index'
                ).sort_index()
                
                # Convert to long format for Plotly Express
                proportions_df = proportions.reset_index()
                proportions_df = pd.melt(
                    proportions_df,
                    id_vars=[feature],
                    var_name='Target Class',
                    value_name='Proportion'
                )
                
                fig_props = px.bar(
                    proportions_df,
                    x=feature,
                    y='Proportion',
                    color='Target Class',
                    title=f"Target Class Proportions - {title}",
                    labels={feature: feature, 'Proportion': 'Proportion'},
                    barmode='stack'
                )
                
                fig_props.update_layout(
                    height=400,
                    yaxis_tickformat='.0%',
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                # Add grid lines
                fig_props.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig_props.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                figures.append(fig_props)
                
            else:  # regression
                # Bar chart of category counts
                value_counts = data[feature].value_counts().sort_index()
                
                fig_counts = px.bar(
                    x=[str(x) for x in value_counts.index],
                    y=value_counts.values,
                    title=f"Category Counts - {title}",
                    labels={"x": feature, "y": "Count"}
                )
                
                fig_counts.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                # Add grid lines
                fig_counts.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig_counts.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                figures.append(fig_counts)
                
                # Box plot of target values by category
                fig_box = px.box(
                    data,
                    x=feature,
                    y=self.target_column,
                    title=f"Target Distribution by Category - {title}",
                    labels={feature: feature, self.target_column: self.target_column},
                    points="outliers"
                )
                
                fig_box.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                # Add grid lines
                fig_box.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig_box.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                figures.append(fig_box)
        
        return figures

    def _group_feature_changes(self) -> Dict[str, List[str]]:
        """Group features by type of change for better organization."""
        changes = self._analyse_feature_changes()
        
        # Initialize groups
        change_groups = {
            "new": [],
            "removed": [],
            "type_changed": [],
            "distribution_changed": [],
            "missing_values_changed": [],
            "unchanged": []
        }
        
        # Add features to appropriate groups
        change_groups["new"] = changes["new_features"]
        change_groups["removed"] = changes["removed_features"]
        
        # Get features with type changes (excluding removed features)
        type_changed = [
            change["feature"] for change in changes["dtype_changes"]
            if change["feature"] not in changes["removed_features"]
        ]
        change_groups["type_changed"] = type_changed
        
        # Get features with distribution changes (excluding removed features)
        dist_changed = [
            change["feature"] for change in changes["distribution_changes"]
            if change["feature"] not in changes["removed_features"]
        ]
        change_groups["distribution_changed"] = dist_changed
        
        # Get features with missing value changes (excluding removed features)
        missing_changed = [
            feature for feature in changes["missing_value_changes"].keys()
            if feature not in changes["removed_features"]
        ]
        change_groups["missing_values_changed"] = missing_changed
        
        # Find unchanged features (excluding removed and new features)
        all_changed = set(
            change_groups["new"] +
            change_groups["removed"] +
            change_groups["type_changed"] +
            change_groups["distribution_changed"] +
            change_groups["missing_values_changed"]
        )
        all_features = set(self.modified_df.columns)  # Only consider features in modified df
        change_groups["unchanged"] = list(all_features - all_changed)
        
        return change_groups

    def _get_relationship_info(self, correlation: float, method: str) -> Dict[str, str]:
        """
        Determine relationship type and strength classification based on correlation value and method.
        
        Args:
            correlation: The correlation value
            method: The method used to calculate the correlation
            
        Returns:
            Dictionary containing strength classification and relationship type
        """
        abs_corr = abs(correlation)
        
        # Initialize result dictionary
        result = {
            'strength': '',
            'relationship_type': '',
            'interpretation': ''
        }
        
        # Determine strength based on method and problem type
        if self.target_type == "regression":
            if method == "pearson":
                if abs_corr >= 0.7:
                    result['strength'] = "Strong"
                elif abs_corr >= 0.4:
                    result['strength'] = "Moderate"
                elif abs_corr >= 0.2:
                    result['strength'] = "Weak"
                else:
                    result['strength'] = "Very Weak"
                    
                # Determine relationship type for regression
                if abs_corr >= 0.95:
                    result['relationship_type'] = "Linear"
                elif abs_corr >= 0.7:
                    result['relationship_type'] = "Mostly Linear"
                else:
                    result['relationship_type'] = "Potentially Non-linear"
                    
        else:  # binary or multiclass classification
            if method == "eta_squared":
                if abs_corr >= 0.26:
                    result['strength'] = "Strong"
                elif abs_corr >= 0.13:
                    result['strength'] = "Moderate"
                elif abs_corr >= 0.02:
                    result['strength'] = "Weak"
                else:
                    result['strength'] = "Very Weak"
                if self.target_type == "multiclass_classification":
                    result['relationship_type'] = "Multi-Class Effect"
                else:
                    result['relationship_type'] = "Category Effect"
                
            elif method == "point_biserial":
                if abs_corr >= 0.7:
                    result['strength'] = "Strong"
                elif abs_corr >= 0.4:
                    result['strength'] = "Moderate"
                elif abs_corr >= 0.2:
                    result['strength'] = "Weak"
                else:
                    result['strength'] = "Very Weak"
                result['relationship_type'] = "Class Separation"
                
            elif method == "cramers_v":
                if abs_corr >= 0.5:
                    result['strength'] = "Strong"
                elif abs_corr >= 0.3:
                    result['strength'] = "Moderate"
                elif abs_corr >= 0.1:
                    result['strength'] = "Weak"
                else:
                    result['strength'] = "Very Weak"
                if self.target_type == "multiclass_classification":
                    result['relationship_type'] = "Multi-Class Association"
                else:
                    result['relationship_type'] = "Category Effect"
        
        # Add interpretation
        result['interpretation'] = f"{result['strength']} {result['relationship_type']}"
        
        return result

    def render(self):
        """Render the component in Streamlit."""
        try:
            st.markdown("""
                        ## Impact Analysis
                        """)
            
            # Analyse changes first
            changes = self._analyse_feature_changes()
            change_groups = self._group_feature_changes()
            
            # Determine which sections have meaningful changes
            has_feature_changes = (
                len(changes["new_features"]) > 0 or
                any(change["feature"] for change in changes["dtype_changes"] 
                    if any(dist_change["feature"] == change["feature"] 
                          for dist_change in changes["distribution_changes"]))
            )
            has_distribution_changes = len(changes["distribution_changes"]) > 0
            has_all_distribution_changes = len(changes["all_distribution_changes"]) > 0
            has_missing_value_changes = len(changes["missing_value_changes"]) > 0
            has_row_count_change = changes["row_count_change"] != 0
            has_removed_features = len(changes["removed_features"]) > 0
            
            # Check if removals are the only changes
            only_removals = (
                has_removed_features and
                not has_feature_changes and
                not has_distribution_changes and
                not has_missing_value_changes and
                not has_row_count_change
            )
            
            # Create dynamic list of tabs based on changes
            tabs = ["Overview"]
            
            # Only add Feature Changes tab if there are actual changes to display
            if has_feature_changes or has_distribution_changes:
                tabs.append("Feature Changes")
            
            # Show Target Relationships and Correlation Analysis unless only features were removed
            show_correlation = not only_removals
            if not only_removals:
                # Only add Target Relationships tab if there are features to analyse
                changed_features = list(set(
                    changes["new_features"] +
                    [c["feature"] for c in changes["dtype_changes"]] +
                    [c["feature"] for c in changes["distribution_changes"]]
                ))
                # Filter out target column and check if there are any valid relationships
                analyzable_features = [f for f in changed_features if f != self.target_column]
                if analyzable_features:
                    # Check if any features have valid relationships
                    has_relationships = False
                    for feature in analyzable_features:
                        if feature in self.modified_df.columns:
                            rel = self._calculate_feature_target_relationship(self.modified_df, feature)
                            if rel:  # If there's a valid relationship
                                has_relationships = True
                                break
                        if feature in self.original_df.columns:
                            rel = self._calculate_feature_target_relationship(self.original_df, feature)
                            if rel:  # If there's a valid relationship
                                has_relationships = True
                                break
                    
                    if has_relationships:
                        tabs.append("Target Relationships")
                
                tabs.append("Correlation Analysis")
            
            # Create tabs dynamically
            tab_dict = {
                name: tab for name, tab in zip(
                    tabs,
                    st.tabs(tabs)
                )
            }
            
            with tab_dict["Overview"]:
                st.write("### Summary of Changes")
                
                with st.expander("ℹ️ Understanding the Summary Metrics"):
                    st.markdown("""
                    **Overview Metrics Explanation:**
                    
                    1. **New Features:**
                       - Count of features present in the modified dataset but not in the original
                       - Important for tracking feature engineering and data enrichment
                    
                    2. **Modified Features:**
                       - Count of features that have changed in type or distribution
                       - Includes both data type changes and significant distribution changes
                    
                    3. **Distribution Changes:**
                       - Total Changes: All detected changes in feature distributions
                       - Significant Changes: Changes that meet statistical significance (p < 0.05)
                       - Only significant changes are shown in detailed analysis
                    
                    4. **Removed Features:**
                       - Features that were present in the original dataset but removed in the modified dataset
                       - Displays original data type and target relationship strength
                       - Warns about removal of features with strong target relationships (>0.3)
                    
                    5. **Row Count Change:**
                       - Difference in number of rows between modified and original datasets
                       - Positive: Rows added, Negative: Rows removed
                    """)
                
                # Only show metrics for changes that exist
                metrics = []
                if len(changes["new_features"]) > 0:
                    metrics.append(("New Features", len(changes["new_features"])))
                if has_removed_features:
                    metrics.append(("Removed Features", len(changes["removed_features"])))
                if len(changes["dtype_changes"]) > 0:
                    metrics.append(("Type Changes", len(changes["dtype_changes"])))
                if has_all_distribution_changes:
                    total_changes = len(changes["all_distribution_changes"])
                    significant_changes = len(changes["distribution_changes"])
                    metrics.append(("Distribution Changes", f"{significant_changes}/{total_changes} significant"))
                    
                    # Add information message about distribution changes
                    if total_changes > significant_changes:
                        st.info(
                            f"📊 {total_changes} total distribution changes detected, but only {significant_changes} "
                            f"meet the significance threshold (p < 0.05) and are shown in the detailed analysis.",
                            icon="ℹ️"
                        )
                if has_row_count_change:
                    metrics.append(("Row Count Change", changes["row_count_change"]))
                
                # Display metrics in columns dynamically
                if metrics:
                    cols = st.columns(min(len(metrics), 3))
                    for i, (label, value) in enumerate(metrics):
                        with cols[i % 3]:
                            st.metric(label, value)
                
                # Display list of new features if they exist
                if len(changes["new_features"]) > 0:
                    st.write("### New Features Analysis")
                    
                    # Create enhanced DataFrame with more information
                    new_features_data = []
                    for feature in changes["new_features"]:
                        # Get relationship info
                        relationship = self._calculate_feature_target_relationship(
                                self.modified_df,
                                feature
                        ) if feature != self.target_column else {}
                        
                        if relationship:
                            rel_info = self._get_relationship_info(
                                relationship.get('correlation', 0),
                                relationship.get('method', '')
                            )
                        
                        # Get basic statistics
                        if pd.api.types.is_numeric_dtype(self.modified_df[feature]):
                            stats = self.modified_df[feature].describe()
                            stats_summary = f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}"
                            unique_ratio = (self.modified_df[feature].nunique() / len(self.modified_df)) * 100
                            stats_summary += f", Unique: {unique_ratio:.1f}%"
                        else:
                            top_category = self.modified_df[feature].value_counts().nlargest(1)
                            category_pct = (top_category.values[0] / len(self.modified_df) * 100)
                            stats_summary = f"Categories: {self.modified_df[feature].nunique()}, "
                            stats_summary += f"Top: '{top_category.index[0]}' ({category_pct:.1f}%)"
                        
                        # Calculate missing values percentage
                        missing_pct = (self.modified_df[feature].isna().sum() / len(self.modified_df)) * 100
                        
                        # Format p-value with significance indicator
                        new_features_data.append({
                            "Feature": feature,
                            "Type": str(self.modified_df[feature].dtype),
                            "Basic Statistics": stats_summary,
                            "Missing Values": f"{missing_pct:.1f}%" if missing_pct > 0 else "None",
                            "Target Relationship": f"{abs(relationship.get('correlation', 0)):.3f}" if relationship else "N/A",
                            "Strength": rel_info.get('strength', 'N/A') if relationship else "N/A",
                            "Relationship Type": rel_info.get('relationship_type', 'N/A') if relationship else "N/A",
                            "Significance": self._format_p_value(relationship.get('p_value')) if relationship else "N/A"
                        })
                    
                    new_features_df = pd.DataFrame(new_features_data)
                    
                    # Style the DataFrame
                    def style_new_features(row):
                        styles = [''] * len(row)
                        # Highlight based on strength
                        if row['Strength'] != 'N/A':
                            if row['Strength'] == 'Strong':
                                styles = ['background-color: rgba(255,165,0,0.3)'] * len(row)
                            elif row['Strength'] == 'Moderate':
                                styles = ['background-color: rgba(255,255,0,0.2)'] * len(row)
                        # Highlight high missing values
                        if row['Missing Values'] != 'None':
                            try:
                                missing_val = float(row['Missing Values'].strip('%'))
                                if missing_val > 20:
                                    styles = ['background-color: rgba(255,0,0,0.1)'] * len(row)
                            except:
                                pass
                        # Highlight significant p-values
                        if row['Significance'] != 'N/A':
                            if '***' in row['Significance']:
                                styles[-1] = 'background-color: rgba(144,238,144,0.4)'  # Strong significance
                            elif '**' in row['Significance']:
                                styles[-1] = 'background-color: rgba(144,238,144,0.3)'  # Moderate significance
                            elif '*' in row['Significance']:
                                styles[-1] = 'background-color: rgba(144,238,144,0.2)'  # Weak significance
                        return styles
                    
                    # Add warning for strongly related features
                    strong_features = [
                        f for f in new_features_df['Feature']
                        if new_features_df[new_features_df['Feature'] == f]['Strength'].values[0] in ['Strong', 'Moderate']
                    ]
                    
                    if strong_features:
                        st.info(
                            "ℹ️ The following new features have strong or moderate relationships with the target: " +
                            ", ".join(f"'{f}'" for f in strong_features),
                            icon="ℹ️"
                        )
                    
                    # Display styled DataFrame
                    styled_df = new_features_df.style.apply(style_new_features, axis=1)
                    st.dataframe(styled_df)
                    
                    # Add explanation
                    with st.expander("ℹ️ Understanding New Features Analysis"):
                        st.markdown("""
                        **Interpreting New Features:**
                        
                        1. **Type:**
                           - Data type of the feature
                           - Determines which statistical methods are used
                        
                        2. **Basic Statistics:**
                           - For numeric features:
                             * Mean and standard deviation
                             * Percentage of unique values
                           - For categorical features:
                             * Number of unique categories
                             * Most frequent category and its percentage
                        
                        3. **Missing Values:**
                           - Percentage of missing values
                           - Highlighted in red if > 20%
                           - "None" if no missing values
                        
                        4. **Target Relationship:**
                           - Absolute value of relationship with target
                           - Values closer to 1 indicate stronger relationships
                        
                        5. **Strength:**
                           - Interpretation of relationship strength
                           - Categories:
                             * Strong (highlighted in orange)
                             * Moderate (highlighted in yellow)
                             * Weak
                             * Very Weak
                        
                        6. **Relationship Type:**
                           - Describes the nature of the relationship
                           - Types vary based on feature and target types
                           - Examples: Linear, Class Separation, Category Effect
                        
                        7. **Significance:**
                           - Statistical significance of the relationship
                           - Highlighted in green with stars:
                             * *** (p < 0.001): Very strong evidence
                             * ** (p < 0.01): Strong evidence
                             * * (p < 0.05): Moderate evidence
                           - No stars: Not statistically significant
                        """)
                
                # Display removed features if they exist
                if has_removed_features:
                    st.write("### Removed Features Analysis")
                    
                    # Create enhanced DataFrame with more information
                    removed_features_data = []
                    for feature in changes["removed_features"]:
                        # Get relationship info
                        relationship = self._calculate_feature_target_relationship(
                            self.original_df,
                            feature
                        ) if feature != self.target_column else {}
                        
                        if relationship:
                            rel_info = self._get_relationship_info(
                                relationship.get('correlation', 0),
                                relationship.get('method', '')
                            )
                        
                        # Get basic statistics from original data
                        if pd.api.types.is_numeric_dtype(self.original_df[feature]):
                            # Check if it's a binary feature
                            is_binary = self._is_binary_feature(self.original_df[feature])
                            if is_binary:
                                # For binary features, show proportion of each value
                                value_counts = self.original_df[feature].value_counts(normalize=True)
                                # Get the most frequent value
                                most_freq_val = value_counts.index[0]
                                most_freq_pct = value_counts.iloc[0] * 100
                                stats_summary = f"Binary feature - Most frequent value: {most_freq_val} ({most_freq_pct:.1f}%)"
                            else:
                                # For non-binary numeric features, show standard statistics
                                stats = self.original_df[feature].describe()
                                stats_summary = f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}"
                                unique_ratio = (self.original_df[feature].nunique() / len(self.original_df)) * 100
                                stats_summary += f", Unique: {unique_ratio:.1f}%"
                        else:
                            top_category = self.original_df[feature].value_counts().nlargest(1)
                            category_pct = (top_category.values[0] / len(self.original_df) * 100)
                            stats_summary = f"Categories: {self.original_df[feature].nunique()}, "
                            stats_summary += f"Top: '{top_category.index[0]}' ({category_pct:.1f}%)"
                        
                        # Calculate missing values percentage
                        missing_pct = (self.original_df[feature].isna().sum() / len(self.original_df)) * 100
                        
                        removed_features_data.append({
                            "Feature": feature,
                            "Original Type": str(self.original_df[feature].dtype),
                            "Basic Statistics": stats_summary,
                            "Missing Values": f"{missing_pct:.1f}%" if missing_pct > 0 else "None",
                            "Target Relationship": f"{abs(relationship.get('correlation', 0)):.3f}" if relationship else "N/A",
                            "Strength": rel_info.get('strength', 'N/A') if relationship else "N/A",
                            "Relationship Type": rel_info.get('relationship_type', 'N/A') if relationship else "N/A",
                            "Significance": self._format_p_value(relationship.get('p_value')) if relationship else "N/A"
                        })
                    
                    removed_features_df = pd.DataFrame(removed_features_data)
                    
                    # Style the DataFrame
                    def style_removed_features(row):
                        styles = [''] * len(row)
                        # Highlight based on strength
                        if row['Strength'] != 'N/A':
                            if row['Strength'] == 'Strong':
                                styles = ['background-color: rgba(255,0,0,0.2)'] * len(row)
                            elif row['Strength'] == 'Moderate':
                                styles = ['background-color: rgba(255,165,0,0.2)'] * len(row)
                        # Highlight significant p-values
                        if row['Significance'] != 'N/A':
                            if '***' in row['Significance']:
                                styles[-1] = 'background-color: rgba(144,238,144,0.4)'  # Strong significance
                            elif '**' in row['Significance']:
                                styles[-1] = 'background-color: rgba(144,238,144,0.3)'  # Moderate significance
                            elif '*' in row['Significance']:
                                styles[-1] = 'background-color: rgba(144,238,144,0.2)'  # Weak significance
                        return styles
                    
                    # Add warning for strongly related features
                    strong_features = [
                        f for f in removed_features_df['Feature']
                        if removed_features_df[removed_features_df['Feature'] == f]['Strength'].values[0] in ['Strong', 'Moderate']
                    ]
                    
                    if strong_features:
                        st.warning(
                            "⚠️ The following removed features had strong or moderate relationships with the target: " +
                            ", ".join(f"'{f}'" for f in strong_features),
                            icon="⚠️"
                        )
                    
                    # Display styled DataFrame
                    styled_df = removed_features_df.style.apply(style_removed_features, axis=1)
                    st.dataframe(styled_df)
                    
                    # Add explanation
                    with st.expander("ℹ️ Understanding Removed Features Analysis"):
                        st.markdown("""
                        **Interpreting Removed Features:**
                        
                        1. **Original Type:**
                           - Data type of the feature before removal
                           - Important for understanding what kind of information was lost
                        
                        2. **Basic Statistics:**
                           - For numeric features:
                             * Mean and standard deviation
                             * Percentage of unique values
                           - For categorical features:
                             * Number of unique categories
                             * Most frequent category and its percentage
                        
                        3. **Missing Values:**
                           - Percentage of missing values in original data
                           - May help understand why the feature was removed
                        
                        4. **Target Relationship:**
                           - Absolute value of relationship with target before removal
                           - Values closer to 1 indicate stronger relationships
                        
                        5. **Strength:**
                           - Interpretation of relationship strength
                           - Categories:
                             * Strong (highlighted in red) - Critical loss
                             * Moderate (highlighted in orange) - Significant loss
                             * Weak - Minor loss
                             * Very Weak - Minimal loss
                        
                        6. **Relationship Type:**
                           - Describes the nature of the lost relationship
                           - Important for understanding the impact of removal
                        
                        7. **Significance:**
                           - P-value of the relationship before removal
                           - Indicates reliability of the lost relationship
                        
                        **Note:** Pay special attention to removed features with strong and significant 
                        relationships to the target, as their removal might impact model performance.
                        """)
                
                # Add information about non-significant changes
                if has_all_distribution_changes and len(changes["all_distribution_changes"]) > len(changes["distribution_changes"]):
                    non_significant_changes = [
                        change for change in changes["all_distribution_changes"]
                        if change["feature"] not in [c["feature"] for c in changes["distribution_changes"]]
                    ]
                    if non_significant_changes:
                        st.write("### Minor Distribution Changes")
                        st.info(
                            "The following features showed distribution changes that didn't meet the significance threshold "
                            "(p < 0.05) but may still be worth noting. While these changes aren't statistically significant, "
                            "they might be practically relevant depending on your use case.",
                            icon="ℹ️"
                        )
                        
                        # Create enhanced DataFrame with more information
                        minor_changes_data = []
                        for change in non_significant_changes:
                            feature = change["feature"]
                            test_type = change["test"]
                            
                            # Calculate effect size and get interpretation
                            effect_size = None
                            effect_interpretation = ""
                            if test_type == "KS":
                                effect_size = change["statistic"]
                                if effect_size >= 0.5:
                                    effect_interpretation = "Large difference"
                                elif effect_size >= 0.3:
                                    effect_interpretation = "Moderate difference"
                                elif effect_size >= 0.1:
                                    effect_interpretation = "Small difference"
                                else:
                                    effect_interpretation = "Minimal difference"
                            elif test_type == "Chi-square":
                                # Calculate Cramer's V for categorical variables
                                n = len(self.original_df[feature].dropna())
                                df = min(
                                    len(self.original_df[feature].unique()) - 1,
                                    len(self.modified_df[feature].unique()) - 1
                                )
                                if df > 0:
                                    effect_size = np.sqrt(change["statistic"] / (n * df))
                                    if effect_size >= 0.5:
                                        effect_interpretation = "Large change"
                                    elif effect_size >= 0.3:
                                        effect_interpretation = "Moderate change"
                                    elif effect_size >= 0.1:
                                        effect_interpretation = "Small change"
                                    else:
                                        effect_interpretation = "Minimal change"
                            elif test_type == "Binary":
                                # For binary features, the statistic is already the effect size (proportion difference)
                                effect_size = change["statistic"]
                                if effect_size >= 0.3:
                                    effect_interpretation = "Large change"
                                elif effect_size >= 0.2:
                                    effect_interpretation = "Moderate change"
                                elif effect_size >= 0.1:
                                    effect_interpretation = "Small change"
                                else:
                                    effect_interpretation = "Minimal change"
                            
                            # Get basic stats comparison
                            if pd.api.types.is_numeric_dtype(self.original_df[feature]) and \
                               pd.api.types.is_numeric_dtype(self.modified_df[feature]):
                                orig_mean = self.original_df[feature].mean()
                                mod_mean = self.modified_df[feature].mean()
                                # Convert to float explicitly to avoid boolean subtraction issues
                                mean_change_pct = ((float(mod_mean) - float(orig_mean)) / float(orig_mean) * 100) if orig_mean != 0 else float('inf')
                                
                                orig_std = self.original_df[feature].std()
                                mod_std = self.modified_df[feature].std()
                                # Convert to float explicitly to avoid boolean subtraction issues
                                std_change_pct = ((float(mod_std) - float(orig_std)) / float(orig_std) * 100) if orig_std != 0 else float('inf')
                                
                                stats_summary = f"Mean Δ: {mean_change_pct:+.1f}%, Std Δ: {std_change_pct:+.1f}%"
                            else:
                                # For categorical features, show changes in most frequent categories
                                orig_top = self.original_df[feature].value_counts().nlargest(1)
                                mod_top = self.modified_df[feature].value_counts().nlargest(1)
                                
                                orig_pct = (orig_top.values[0] / len(self.original_df) * 100)
                                mod_pct = (mod_top.values[0] / len(self.modified_df) * 100)
                                
                                # Convert to float explicitly to avoid boolean subtraction issues
                                stats_summary = f"Top category frequency Δ: {float(mod_pct) - float(orig_pct):+.1f}%"
                            
                            minor_changes_data.append({
                            "Feature": feature,
                                "Test Type": test_type,
                                "Effect Size": f"{effect_size:.4f}" if effect_size is not None else "N/A",
                                "Effect Interpretation": effect_interpretation,
                                "P-value": f"{change['p_value']:.4f}",
                                "Distribution Changes": stats_summary
                            })
                        
                        minor_changes_df = pd.DataFrame(minor_changes_data)
                        
                        # Style the DataFrame
                        def style_minor_changes(row):
                            styles = [''] * len(row)
                            # Highlight based on effect size
                            if row['Effect Size'] != 'N/A':
                                try:
                                    effect = float(row['Effect Size'])
                                    if effect >= 0.3:  # Moderate or large effect
                                        styles = ['background-color: rgba(255,165,0,0.2)'] * len(row)
                                    elif effect >= 0.1:  # Small effect
                                        styles = ['background-color: rgba(255,255,0,0.1)'] * len(row)
                                except:
                                    pass
                            return styles
                        
                        # Display styled DataFrame
                        styled_df = minor_changes_df.style.apply(style_minor_changes, axis=1)
                        st.dataframe(styled_df)
                        
                        # Add explanation of the metrics
                        with st.expander("ℹ️ Understanding Minor Distribution Changes"):
                            st.markdown("""
                            **Interpreting Minor Distribution Changes:**
                            
                            1. **Test Type:**
                               - KS (Kolmogorov-Smirnov): Used for numeric features
                               - Chi-square: Used for categorical features
                               - Binary: Used for binary features (true/false, 0/1, yes/no)
                            
                            2. **Effect Size:**
                               - Measures the magnitude of the change regardless of statistical significance
                               - For KS test:
                                 * ≥ 0.5: Large difference
                                 * ≥ 0.3: Moderate difference
                                 * ≥ 0.1: Small difference
                               - For Chi-square (Cramer's V):
                                 * ≥ 0.5: Large change
                                 * ≥ 0.3: Moderate change
                                 * ≥ 0.1: Small change
                               - For Binary features:
                                 * ≥ 0.3: Large change
                                 * ≥ 0.2: Moderate change
                                 * ≥ 0.1: Small change
                            
                            3. **P-value:**
                               - All changes here have p ≥ 0.05 (not statistically significant)
                               - Lower values suggest stronger evidence of real changes
                            
                            4. **Distribution Changes:**
                               - For numeric features: Shows changes in mean and standard deviation
                               - For categorical features: Shows changes in most frequent category
                               - Percentage changes help assess practical significance
                            
                            **Note:** While these changes aren't statistically significant, they may still be 
                            practically important depending on your specific use case and domain knowledge.
                            """)
                
                # Display data type changes if they exist
                if changes["dtype_changes"]:
                    st.write("### Data Type Changes")
                    dtype_changes_df = pd.DataFrame([
                        {
                            "Feature": change["feature"],
                            "Original Type": change["original_dtype"],
                            "Modified Type": change["modified_dtype"]
                        }
                        for change in changes["dtype_changes"]
                    ])
                    st.dataframe(dtype_changes_df)
                
                # Display significant distribution changes
                if len(changes["distribution_changes"]) > 0:
                    st.write("### Significant Distribution Changes")
                    st.info(
                        "The following features showed statistically significant changes in their distributions "
                        "(p < 0.05). These changes may affect model performance and should be reviewed carefully."
                    )
                    
                    significant_changes_data = []
                    for change in changes["distribution_changes"]:
                        feature = change["feature"]
                        test_type = change["test"]
                        
                        # Calculate effect size and interpretation
                        effect_size = change.get("statistic", None)
                        effect_interpretation = self._get_effect_size_interpretation(effect_size, change["test"])
                        
                        # Get basic stats comparison
                        if pd.api.types.is_numeric_dtype(self.original_df[feature]) and \
                           pd.api.types.is_numeric_dtype(self.modified_df[feature]):
                            orig_mean = self.original_df[feature].mean()
                            mod_mean = self.modified_df[feature].mean()
                            # Convert to float explicitly to avoid boolean subtraction issues
                            mean_change_pct = ((float(mod_mean) - float(orig_mean)) / float(orig_mean) * 100) if orig_mean != 0 else float('inf')
                            
                            orig_std = self.original_df[feature].std()
                            mod_std = self.modified_df[feature].std()
                            # Convert to float explicitly to avoid boolean subtraction issues
                            std_change_pct = ((float(mod_std) - float(orig_std)) / float(orig_std) * 100) if orig_std != 0 else float('inf')
                            
                            stats_summary = f"Mean Δ: {mean_change_pct:+.1f}%, Std Δ: {std_change_pct:+.1f}%"
                        else:
                            # For categorical features, show changes in most frequent categories
                            orig_top = self.original_df[feature].value_counts().nlargest(1)
                            mod_top = self.modified_df[feature].value_counts().nlargest(1)
                            
                            orig_pct = (orig_top.values[0] / len(self.original_df) * 100)
                            mod_pct = (mod_top.values[0] / len(self.modified_df) * 100)
                            
                            # Convert to float explicitly to avoid boolean subtraction issues
                            stats_summary = f"Top category frequency Δ: {float(mod_pct) - float(orig_pct):+.1f}%"
                        
                        # Calculate target relationship change
                        orig_rel = self._calculate_feature_target_relationship(self.original_df, feature)
                        mod_rel = self._calculate_feature_target_relationship(self.modified_df, feature)
                        
                        target_rel_change = ""
                        if orig_rel and mod_rel and orig_rel.get('method') == mod_rel.get('method'):
                            orig_corr = abs(orig_rel.get('correlation', 0))
                            mod_corr = abs(mod_rel.get('correlation', 0))
                            # Convert to float explicitly to avoid boolean subtraction issues
                            rel_change = float(mod_corr) - float(orig_corr)
                            target_rel_change = f"{rel_change:+.3f} ({mod_rel.get('method', 'unknown')})"
                        else:
                            target_rel_change = "Method changed"
                        
                        significant_changes_data.append({
                            "Feature": feature,
                            "Test Type": test_type,
                            "Effect Size": f"{effect_size:.4f}" if effect_size is not None else "N/A",
                            "Effect Interpretation": effect_interpretation,
                            "P-value": self._format_p_value(change['p_value']),
                            "Distribution Changes": stats_summary,
                            "Target Relationship Δ": target_rel_change
                        })
                    
                    significant_changes_df = pd.DataFrame(significant_changes_data)
                    
                    # Style the DataFrame
                    def style_significant_changes(row):
                        styles = [''] * len(row)
                        # Highlight based on effect size
                        if row['Effect Size'] != 'N/A':
                            try:
                                effect = float(row['Effect Size'])
                                if effect >= 0.5:  # Large effect
                                    styles = ['background-color: rgba(255,165,0,0.3)'] * len(row)
                                elif effect >= 0.3:  # Moderate effect
                                    styles = ['background-color: rgba(255,255,0,0.2)'] * len(row)
                            except:
                                pass
                        return styles
                    
                    # Display styled DataFrame
                    styled_df = significant_changes_df.style.apply(style_significant_changes, axis=1)
                    st.dataframe(styled_df)
                    
                    # Add explanation of the metrics
                    with st.expander("ℹ️ Understanding Significant Distribution Changes"):
                        st.markdown("""
                        **Interpreting Significant Distribution Changes:**
                        
                        1. **Test Type:**
                           - KS (Kolmogorov-Smirnov): Used for numeric features
                           - Chi-square: Used for categorical features
                           - Binary: Used for binary features (true/false, 0/1, yes/no)
                        
                        2. **Effect Size:**
                           - Measures the magnitude of the change
                           - For KS test:
                             * ≥ 0.5: Large difference
                             * ≥ 0.3: Moderate difference
                             * ≥ 0.1: Small difference
                           - For Chi-square (Cramer's V):
                             * ≥ 0.5: Large change
                             * ≥ 0.3: Moderate change
                             * ≥ 0.1: Small change
                           - For Binary features:
                             * ≥ 0.3: Large change
                             * ≥ 0.2: Moderate change
                             * ≥ 0.1: Small change
                        
                        3. **P-value:**
                           - All changes here have p < 0.05 (statistically significant)
                           - Lower values indicate stronger evidence of real changes
                        
                        4. **Distribution Changes:**
                           - For numeric features: Shows changes in mean and standard deviation
                           - For categorical features: Shows changes in most frequent category
                           - Percentage changes help assess practical significance
                        
                        5. **Target Relationship Δ:**
                           - Shows how the feature's relationship with the target has changed
                           - Positive values: Relationship strengthened
                           - Negative values: Relationship weakened
                           - "Method changed": Different metrics used due to data type changes
                        
                        **Note:** These changes are statistically significant and likely to impact model performance.
                        Pay special attention to features with large effect sizes or substantial target relationship changes.
                        """)
                
                # Show missing values changes only if they exist
                if has_missing_value_changes:
                    st.write("### Missing Values Changes")
                    missing_df = pd.DataFrame.from_dict(
                        changes["missing_value_changes"],
                        orient='index'
                    )
                    st.dataframe(missing_df)
            
            if "Feature Changes" in tab_dict:
                with tab_dict["Feature Changes"]:
                    with st.expander("ℹ️ Understanding Feature Analysis"):
                        st.markdown("""
                        The Feature Changes tab provides detailed analysis of how features have changed between the original and modified datasets. Changes are organized into two main categories:

                        **1. New Features Analysis:**
                        - Shows features present only in the modified dataset
                        - Provides basic statistics (count, mean, std, min/max, quartiles)
                        - Displays the feature's distribution through visualisations:
                          * For numeric features: Histograms and box plots
                          * For categorical features: Bar charts showing value frequencies
                        - Shows relationship with the target variable:
                          * For classification: Distribution by class
                          * For regression: Scatter plots with trend lines
                        - Calculates and displays target relationship strength

                        **2. Distribution Changes Analysis:**
                        - Identifies statistically significant changes in feature distributions
                        - Provides comprehensive statistics comparison:
                          * Shows original and modified values for key statistics
                          * Calculates absolute and percentage differences
                          * Highlights significant changes (>10%) for easy identification
                          * Includes: count, mean, std, min/max, quartiles
                        - Shows detailed statistical test results:
                          * For numeric features (Kolmogorov-Smirnov test):
                            - Tests if distributions are significantly different
                            - Effect size shows magnitude of distribution change
                            - Includes interpretation of practical significance
                          * For categorical features (Chi-square test):
                            - Tests if category proportions have changed
                            - Cramer's V shows strength of changes
                            - Includes interpretation of practical significance
                          * Color-coded results for quick interpretation:
                            - Significant p-values highlighted in green
                            - Large effects highlighted in orange
                            - Moderate effects highlighted in yellow
                          * Expandable explanation of test interpretation
                        - Provides side-by-side visualization comparison:
                          * For numeric features:
                            - Histograms showing distribution shape changes
                            - Box plots showing changes in central tendency and spread
                            - For classification: Distribution by target class
                            - For regression: Scatter plots with target
                          * For categorical features:
                            - Bar charts showing frequency changes
                            - Target relationship visualisations
                            - Proportion plots for classification problems

                        **How to Use This Tab:**
                        1. Use the main tabs to switch between New Features and Distribution Changes
                        2. Within each section, use feature-specific tabs to analyse individual features
                        3. Review the statistics comparison table to identify significant changes
                        4. Compare visualisations side-by-side to understand changes
                        5. Check statistical test results to confirm significance of changes
                        6. Pay special attention to changes in target relationships
                        
                        **Key Metrics and Thresholds:**
                        - Statistical significance: p-value < 0.05
                        - Distribution changes: highlighted when > 10% change
                        - Target relationship strength:
                          * Strong: > 0.7 (correlation) or > 0.5 (Cramer's V)
                          * Moderate: 0.4-0.7 (correlation) or 0.3-0.5 (Cramer's V)
                          * Weak: < 0.4 (correlation) or < 0.3 (Cramer's V)
                        """)
                    
                    # Only show groups for new features and distribution changes
                    active_groups = [(key, len(features)) for key, features in change_groups.items() 
                                   if len(features) > 0 and key in ["new"]]
                    
                    # Add distribution changes separately to avoid duplicate displays
                    distribution_changed_features = [change["feature"] for change in changes["distribution_changes"]
                                                   if change["feature"] in self.modified_df.columns]
                    if distribution_changed_features:
                        active_groups.append(("distribution_changed", len(distribution_changed_features)))
                        change_groups["distribution_changed"] = distribution_changed_features
                    
                    if active_groups:
                        # Create tabs for each change type
                        change_tabs = st.tabs([f"{key.replace('_', ' ').title()} ({count})" 
                                             for key, count in active_groups])
                        
                        for tab, (group_key, _) in zip(change_tabs, active_groups):
                            with tab:
                                features = change_groups[group_key]
                                if features:
                                    # Create sub-tabs for each feature
                                    feature_tabs = st.tabs([f"📊 {feature}" for feature in features])
                                    for feature_tab, feature in zip(feature_tabs, features):
                                        with feature_tab:
                                            self._render_feature_analysis(feature, group_key, changes)
            
            if "Target Relationships" in tab_dict:
                with tab_dict["Target Relationships"]:
                    with st.expander("ℹ️ Understanding Target Relationships"):
                        st.markdown("""
                        **Target Relationship Analysis:**
                        
                        1. **Relationship Strength:**
                           - Shows how strongly each feature relates to the target
                           - Different metrics used based on feature and target types:
                             * Numeric features (Regression): Pearson correlation
                             * Numeric features (Binary): Point-biserial correlation
                             * Categorical features: Cramer's V or Eta-squared
                        
                        2. **Relationship Types:**
                           - Linear: Simple straight-line relationship
                           - Non-linear: Curved or complex pattern
                           - Category Effect: Categorical variable impact
                           - Class Separation: Feature's ability to distinguish classes
                        
                        3. **Strength Classifications:**
                           For Regression (correlation coefficients):
                           - Strong: |correlation| ≥ 0.7
                           - Moderate: |correlation| ≥ 0.4
                           - Weak: |correlation| ≥ 0.2
                           - Very Weak: |correlation| < 0.2
                           
                           For Binary Classification:
                           - Categorical (Cramer's V):
                             * Strong: ≥ 0.5
                             * Moderate: ≥ 0.3
                             * Weak: ≥ 0.1
                             * Very Weak: < 0.1
                           - Numeric (Eta-squared):
                             * Strong: ≥ 0.26
                             * Moderate: ≥ 0.13
                             * Weak: ≥ 0.02
                             * Very Weak: < 0.02
                           
                           For Multiclass Classification:
                           - Same thresholds as Binary Classification
                           - Uses Cramer's V for categorical features
                           - Uses Eta-squared for numeric features
                        """)
                    
                    # Only analyse features that have changes
                    changed_features = list(set(
                        changes["new_features"] +
                        [c["feature"] for c in changes["dtype_changes"]] +
                        [c["feature"] for c in changes["distribution_changes"]]
                    ))
                    
                    if changed_features:
                        relationships_orig = {}
                        relationships_mod = {}
                        
                        for feature in changed_features:
                            if feature != self.target_column:
                                if feature in self.original_df.columns:
                                    orig_rel = self._calculate_feature_target_relationship(
                                        self.original_df,
                                        feature
                                    )
                                    relationships_orig[feature] = orig_rel
                                if feature in self.modified_df.columns:
                                    mod_rel = self._calculate_feature_target_relationship(
                                        self.modified_df,
                                        feature
                                    )
                                    relationships_mod[feature] = mod_rel
                        
                        # Create comparison DataFrame for changed features only
                        comparison_data = []
                        for feature in changed_features:
                            if feature != self.target_column:
                                orig_rel = relationships_orig.get(feature, {})
                                mod_rel = relationships_mod.get(feature, {})
                                
                                if orig_rel or mod_rel:
                                    # Get relationship info for both original and modified
                                    orig_info = self._get_relationship_info(
                                        orig_rel.get('correlation', 0),
                                        orig_rel.get('method', '')
                                    ) if orig_rel else {'strength': 'N/A', 'relationship_type': 'N/A', 'interpretation': 'N/A'}
                                    
                                    mod_info = self._get_relationship_info(
                                        mod_rel.get('correlation', 0),
                                        mod_rel.get('method', '')
                                    ) if mod_rel else {'strength': 'N/A', 'relationship_type': 'N/A', 'interpretation': 'N/A'}
                                    
                                    comparison_data.append({
                                        'Feature': feature,
                                        'Original Metric': orig_rel.get('method', 'N/A'),
                                        'Original Strength': abs(orig_rel.get('correlation', 0)),
                                        'Original Classification': orig_info['interpretation'],
                                        'Modified Metric': mod_rel.get('method', 'N/A'),
                                        'Modified Strength': abs(mod_rel.get('correlation', 0)),
                                        'Modified Classification': mod_info['interpretation'],
                                        'Strength Change': abs(mod_rel.get('correlation', 0)) - abs(orig_rel.get('correlation', 0))
                                    })
                        
                        # Add new helper method for metric transition explanation
                        def _get_metric_transition_explanation(orig_metric: str, mod_metric: str) -> str:
                            """Get explanation for metric transition."""
                            if orig_metric == mod_metric:
                                return None
                                
                            transitions = {
                                ("point_biserial", "cramers_v"): 
                                    "Feature converted from numeric to categorical. Point-biserial (numeric) changed to Cramer's V (categorical).",
                                ("pearson", "cramers_v"): 
                                    "Feature converted from numeric to categorical. Pearson correlation (numeric) changed to Cramer's V (categorical).",
                                ("cramers_v", "point_biserial"): 
                                    "Feature converted from categorical to numeric. Cramer's V (categorical) changed to Point-biserial (numeric).",
                                ("cramers_v", "pearson"): 
                                    "Feature converted from categorical to numeric. Cramer's V (categorical) changed to Pearson correlation (numeric).",
                                ("eta_squared", "cramers_v"): 
                                    "Feature converted from numeric to categorical. Eta-squared (numeric) changed to Cramer's V (categorical).",
                                ("cramers_v", "eta_squared"): 
                                    "Feature converted from categorical to numeric. Cramer's V (categorical) changed to Eta-squared (numeric)."
                            }
                            
                            return transitions.get((orig_metric, mod_metric))

                        # Update the comparison DataFrame creation
                        if comparison_data:
                            comparison_df = pd.DataFrame(comparison_data)
                            comparison_df = comparison_df.sort_values('Strength Change', ascending=False)
                            
                            # Add metric transition explanations
                            metric_changes = []
                            for _, row in comparison_df.iterrows():
                                explanation = _get_metric_transition_explanation(
                                    row['Original Metric'],
                                    row['Modified Metric']
                                )
                                metric_changes.append(explanation if explanation else "")
                            
                            comparison_df['Metric Change'] = metric_changes
                            
                            # Categorize features by change type
                            dtype_changes = set(c["feature"] for c in changes["dtype_changes"])
                            distribution_changes = set(c["feature"] for c in changes["distribution_changes"])
                            new_features = set(changes["new_features"])
                            
                            # Create separate DataFrames for each category
                            dtype_changes_df = comparison_df[comparison_df['Feature'].isin(dtype_changes)]
                            dist_changes_df = comparison_df[
                                (comparison_df['Feature'].isin(distribution_changes)) & 
                                (~comparison_df['Feature'].isin(dtype_changes))
                            ]
                            new_features_df = comparison_df[
                                (comparison_df['Feature'].isin(new_features)) & 
                                (~comparison_df['Feature'].isin(dtype_changes)) & 
                                (~comparison_df['Feature'].isin(distribution_changes))
                            ]
                            
                            # Helper function for styling DataFrames
                            def style_relationship_df(df):
                                return df.style.format({
                                    'Original Strength': '{:.3f}',
                                    'Modified Strength': '{:.3f}',
                                    'Strength Change': '{:.3f}',
                                    'Original Classification': '{}',
                                    'Modified Classification': '{}',
                                    'Original Metric': '{}',
                                    'Modified Metric': '{}',
                                    'Metric Change': '{}'
                                }).apply(lambda x: [
                                    'background-color: rgba(255, 255, 0, 0.2)' 
                                    if x['Original Metric'] != x['Modified Metric'] 
                                    else '' for _ in x
                                ], axis=1)
                            
                            # Display tables for each category if they have data
                            if not dtype_changes_df.empty:
                                st.write("### Features with Data Type Changes")
                                st.info("""
                                These features have changed their data type, resulting in different metrics 
                                being used to measure their relationship with the target. The strength 
                                classifications may differ due to the different scales of the metrics.
                                """)
                                st.dataframe(style_relationship_df(dtype_changes_df), width=1200)
                            
                            if not dist_changes_df.empty:
                                st.write("### Features with Distribution Changes")
                                st.info("""
                                These features have maintained their data type but show significant changes 
                                in their distribution. The relationship strength changes indicate how these 
                                distribution changes have affected their predictive power.
                                """)
                                st.dataframe(style_relationship_df(dist_changes_df), width=1200)
                            
                            if not new_features_df.empty:
                                st.write("### Newly Added Features")
                                st.info("""
                                These are newly added features that weren't present in the original dataset. 
                                Their relationship metrics show their potential predictive value for the target.
                                """)
                                st.dataframe(style_relationship_df(new_features_df), width=1200)

                            # Add metric interpretation guide
                            with st.expander("📊 Metric Interpretation Guide"):
                                st.markdown("""
                                **Metric Types and Their Use Cases:**
                                
                                1. **Point-Biserial Correlation**
                                   - Used for: Numeric features with binary target
                                   - Range: -1 to 1
                                   - Interpretation:
                                     * Strong: |r| ≥ 0.7
                                     * Moderate: 0.4 ≤ |r| < 0.7
                                     * Weak: 0.2 ≤ |r| < 0.4
                                     * Very Weak: |r| < 0.2
                                
                                2. **Cramer's V**
                                   - Used for: Categorical features (binary/multiclass)
                                   - Range: 0 to 1
                                   - Interpretation:
                                     * Strong: V ≥ 0.5
                                     * Moderate: 0.3 ≤ V < 0.5
                                     * Weak: 0.1 ≤ V < 0.3
                                     * Very Weak: V < 0.1
                                
                                3. **Eta-Squared**
                                   - Used for: Numeric features with classification targets (binary/multiclass)
                                   - Range: 0 to 1
                                   - Interpretation:
                                     * Strong: η² ≥ 0.26
                                     * Moderate: 0.13 ≤ η² < 0.26
                                     * Weak: 0.02 ≤ η² < 0.13
                                     * Very Weak: η² < 0.02
                                
                                4. **Pearson Correlation**
                                   - Used for: Numeric features with numeric target
                                   - Range: -1 to 1
                                   - Interpretation:
                                     * Strong: |r| ≥ 0.7
                                     * Moderate: 0.4 ≤ |r| < 0.7
                                     * Weak: 0.2 ≤ |r| < 0.4
                                     * Very Weak: |r| < 0.2
                                
                                **Note:** When metrics change due to data type transformations, 
                                strength classifications may change even if the feature's actual 
                                predictive power remains similar. This is because different metrics 
                                have different scales and thresholds for interpretation.
                                """)
            
            # Only process correlation analysis if the tab exists and should be shown
            if show_correlation and "Correlation Analysis" in tab_dict:
                with tab_dict["Correlation Analysis"]:
                    with st.expander("ℹ️ Understanding Correlation Analysis"):
                        st.markdown("""
                        **Correlation Analysis Components:**
                        
                        1. **Feature Associations:**
                           - Shows relationships between all features
                           - Values range from -1 to 1 (or 0 to 1 for some metrics)
                           - Different metrics used based on data types:
                             * Numeric + Numeric: Pearson correlation
                             * Categorical + Categorical: Cramer's V
                             * Numeric + Categorical: Correlation ratio (eta)
                        
                        2. **Changes in Correlations Heatmap:**
                           - Shows how the strength of relationships between features has changed
                           - Values calculated as: |modified correlation| - |original correlation|
                           - Color interpretation:
                             * Red (positive values): Relationship became stronger
                             * Blue (negative values): Relationship became weaker
                             * White/Light: Little to no change in relationship strength
                           - Special cases:
                             * New features: Shows absolute correlation value from modified dataset with "(new)" label
                             * Changed metrics: Shows "changed" with the metric transition (e.g., "pr→cv")
                           
                           **Example interpretations:**
                           - Value of 0.3: Absolute correlation increased by 0.3 (stronger relationship)
                           - Value of -0.2: Absolute correlation decreased by 0.2 (weaker relationship)
                           - "0.75 (new)": New feature with absolute correlation of 0.75
                           - "changed pr→cv": Metric changed from Pearson to Cramer's V
                        
                        **What to Look For:**
                        - New feature interactions (marked with "new")
                        - Strengthened relationships (red cells)
                        - Weakened relationships (blue cells)
                        - Changes in metric types (marked as "changed")
                        - Clusters of related features
                        - Features that became more/less independent
                        
                        **Metric Abbreviations:**
                        - pr: Pearson correlation (Numeric ↔ Numeric)
                        - cv: Cramer's V (Categorical ↔ Categorical)
                        - cr: Correlation ratio (Numeric ↔ Categorical)
                        - —: Self-correlation (diagonal)
                        - n/a: Insufficient data
                        """)
                    
                    try:
                        # Calculate associations for both dataframes first
                        orig_corr, orig_fig, orig_metrics = self._analyse_associations(self.original_df)
                        mod_corr, mod_fig, mod_metrics = self._analyse_associations(self.modified_df)
                        
                        # Create nested tabs for correlation analysis
                        correlation_tabs = st.tabs(["Changes in Associations", "Current Associations"])
                        
                        with correlation_tabs[0]:
                            st.write("### Changes in Feature Associations")
                            # Calculate and display correlation changes
                            changes, changes_fig = self._compare_associations(orig_corr, mod_corr, orig_metrics, mod_metrics)
                            
                            if changes_fig is not None:
                                st.plotly_chart(changes_fig)
                            else:
                                display_info_message("No common features found for correlation comparison", "info")
                        
                        with correlation_tabs[1]:
                            #col1, col2 = st.columns(2)
                            
                            #with col1:
                            #st.write("### Original Dataframe Associations")
                            #st.plotly_chart(orig_fig, config={'responsive': True})
                            
                            #with col2:
                            st.write("### Current Dataframe Associations")
                            st.plotly_chart(mod_fig, config={'responsive': True})
                                
                    except Exception as e:
                        display_info_message(f"Error analyzing correlations: {str(e)}", "error")
                    
        except Exception as e:
            display_info_message(
                f"An error occurred while rendering the comparison analysis: {str(e)}\n\n"
                "Please check your input data and try again.",
                "error"
            )
            
    def _render_feature_analysis(self, feature: str, group: str, changes: Dict):
        """Helper method to render feature analysis."""
        try:
            if group == "new":
                with st.expander(f"📊 {feature} Analysis", expanded=True):
                    relationship = self._calculate_feature_target_relationship(
                        self.modified_df, 
                        feature
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Basic Statistics:**")
                        stats = self.modified_df[feature].describe()
                        st.dataframe(stats)
                        
                        if relationship:
                            st.write("### Target Relationship Analysis")
                            
                            # Get relationship info for interpretation
                            rel_info = self._get_relationship_info(
                                relationship.get('correlation', 0),
                                relationship.get('method', '')
                            )
                            
                            # Create formatted results
                            test_info = {
                                'Metric': [
                                    'Relationship Type',
                                    'Test/Metric Used',
                                    'Correlation/Association',
                                    'P-value',
                                    'Strength',
                                    'Interpretation'
                                ],
                                'Value': [
                                    rel_info['relationship_type'],
                                    relationship.get('method', 'N/A'),
                                    f"{abs(relationship.get('correlation', 0)):.4f}",
                                    self._format_p_value(relationship.get('p_value', 0)),
                                    rel_info['strength'],
                                    rel_info['interpretation']
                                ]
                            }
                            
                            # Create and style the DataFrame
                            test_df = pd.DataFrame(test_info)
                            
                            # Apply styling
                            def style_relationship_results(row):
                                styles = [''] * len(row)
                                if row['Metric'] == 'P-value':
                                    try:
                                        p_val = float(row['Value'].split('e')[0])
                                        if p_val < 0.05:
                                            styles = ['background-color: rgba(144,238,144,0.3)'] * len(row)
                                    except:
                                        pass
                                elif row['Metric'] == 'Correlation/Association':
                                    try:
                                        corr_val = float(row['Value'])
                                        if corr_val >= 0.5:
                                            styles = ['background-color: rgba(255,165,0,0.3)'] * len(row)
                                        elif corr_val >= 0.3:
                                            styles = ['background-color: rgba(255,255,0,0.2)'] * len(row)
                                    except:
                                        pass
                                return styles
                            
                            # Display styled test results
                            styled_df = test_df.style.apply(style_relationship_results, axis=1)
                            st.dataframe(styled_df)
                            
                            # Add explanation based on the metric type
                            st.write("ℹ️ **Understanding the Relationship Metrics**")
                            if relationship.get('method') == "pearson":
                                st.markdown("""
                                **Pearson Correlation Interpretation:**
                                - **What it measures:** Linear relationship between feature and target
                                - **Range:** -1 to 1 (absolute value shown)
                                - **Strength interpretation:**
                                  * ≥ 0.7: Strong linear relationship
                                  * ≥ 0.4: Moderate linear relationship
                                  * ≥ 0.2: Weak linear relationship
                                  * < 0.2: Very weak linear relationship
                                - **P-value:** Probability of seeing such correlation by chance
                                  * < 0.05: Statistically significant relationship
                                  * ≥ 0.05: Relationship may be due to chance
                                """)
                            elif relationship.get('method') == "point_biserial":
                                st.markdown("""
                                **Point-Biserial Correlation Interpretation:**
                                - **What it measures:** Relationship between numeric feature and binary target
                                - **Range:** -1 to 1 (absolute value shown)
                                - **Strength interpretation:**
                                  * ≥ 0.7: Strong class separation
                                  * ≥ 0.4: Moderate class separation
                                  * ≥ 0.2: Weak class separation
                                  * < 0.2: Very weak class separation
                                - **P-value:** Probability of seeing such separation by chance
                                  * < 0.05: Statistically significant separation
                                  * ≥ 0.05: Separation may be due to chance
                                """)
                            elif relationship.get('method') == "cramers_v":
                                st.markdown("""
                                **Cramer's V Interpretation:**
                                - **What it measures:** Association between categorical feature and target
                                - **Range:** 0 to 1
                                - **Strength interpretation:**
                                  * ≥ 0.5: Strong association
                                  * ≥ 0.3: Moderate association
                                  * ≥ 0.1: Weak association
                                  * < 0.1: Very weak association
                                - **P-value:** Probability of seeing such association by chance
                                  * < 0.05: Statistically significant association
                                  * ≥ 0.05: Association may be due to chance
                                """)
                            elif relationship.get('method') == "eta_squared":
                                st.markdown("""
                                **Eta-squared Interpretation:**
                                - **What it measures:** Proportion of variance in numeric feature explained by class differences
                                - **Range:** 0 to 1
                                - **Use cases:** 
                                  * Binary classification: How well classes separate on numeric feature
                                  * Multiclass classification: How well multiple classes separate on numeric feature
                                - **Strength interpretation:**
                                  * ≥ 0.26: Strong effect
                                  * ≥ 0.13: Moderate effect
                                  * ≥ 0.02: Weak effect
                                  * < 0.02: Very weak effect
                                - **P-value:** Probability of seeing such effect by chance
                                  * < 0.05: Statistically significant effect
                                  * ≥ 0.05: Effect may be due to chance
                                """)
                    
                    with col2:
                        figures = self._create_feature_distribution_plot(
                            self.modified_df,
                            feature,
                            f"Distribution of {feature}"
                        )
                        # Display each figure separately
                        for fig_idx, fig in enumerate(figures):
                            try:
                                plot_key = f"new_{feature}_dist_{fig_idx}"
                                st.plotly_chart(fig, config={'responsive': True}, key=plot_key)
                            except Exception as e:
                                st.error(f"Error displaying plot: {str(e)}")
            
            elif group == "distribution_changed":
                # Find the distribution change details
                dist_change = next(
                    (change for change in changes["distribution_changes"] 
                     if change["feature"] == feature),
                    None
                )
                if dist_change:
                    with st.expander(f"📊 {feature} Distribution Change", expanded=True):
                        # Statistical Test Results
                        st.write("### Statistical Test Results")
                        
                        # Create test results DataFrame
                        test_type = dist_change['test']
                        test_stat = dist_change['statistic']
                        p_value = dist_change['p_value']
                        
                        # Calculate effect size and get interpretation
                        effect_size = None
                        effect_interpretation = ""
                        if test_type == "KS":
                            # For KS test, the statistic itself is a measure of effect size
                            effect_size = test_stat
                            if effect_size >= 0.5:
                                effect_interpretation = "Large difference in distributions"
                            elif effect_size >= 0.3:
                                effect_interpretation = "Moderate difference in distributions"
                            elif effect_size >= 0.1:
                                effect_interpretation = "Small difference in distributions"
                            else:
                                effect_interpretation = "Minimal difference in distributions"
                        elif test_type == "Chi-square":
                            # Calculate Cramer's V for categorical variables
                            n = len(self.original_df[feature].dropna())
                            df = min(
                                len(self.original_df[feature].unique()) - 1,
                                len(self.modified_df[feature].unique()) - 1
                            )
                            if df > 0:  # Avoid division by zero
                                effect_size = np.sqrt(test_stat / (n * df))
                                if effect_size >= 0.5:
                                    effect_interpretation = "Large change in category distributions"
                                elif effect_size >= 0.3:
                                    effect_interpretation = "Moderate change in category distributions"
                                elif effect_size >= 0.1:
                                    effect_interpretation = "Small change in category distributions"
                                else:
                                    effect_interpretation = "Minimal change in category distributions"
                        
                        # Create formatted results
                        test_info = {
                            'Metric': [
                                'Test Type',
                                'Test Statistic',
                                'P-value',
                                'Effect Size',
                                'Significance',
                                'Interpretation'
                            ],
                            'Value': [
                                f"{test_type} Test",
                                f"{test_stat:.4f}",
                                self._format_p_value(p_value),
                                f"{effect_size:.4f}" if effect_size is not None else "N/A",
                                "Significant change detected" if p_value < 0.05 else "No significant change",
                                effect_interpretation
                            ]
                        }
                        
                        # Create and style the DataFrame
                        test_df = pd.DataFrame(test_info)
                        
                        # Apply styling
                        def style_test_results(row):
                            styles = [''] * len(row)
                            if row['Metric'] == 'P-value':
                                try:
                                    p_val = float(row['Value'].split('e')[0])
                                    if p_val < 0.05:
                                        styles = ['background-color: rgba(144,238,144,0.3)'] * len(row)
                                except:
                                    pass
                            elif row['Metric'] == 'Effect Size':
                                if row['Value'] != 'N/A':
                                    try:
                                        effect_val = float(row['Value'])
                                        if effect_val >= 0.5:
                                            styles = ['background-color: rgba(255,165,0,0.3)'] * len(row)
                                        elif effect_val >= 0.3:
                                            styles = ['background-color: rgba(255,255,0,0.2)'] * len(row)
                                    except:
                                        pass
                            return styles
                        
                        # Display styled test results
                        styled_df = test_df.style.apply(style_test_results, axis=1)
                        st.dataframe(styled_df)
                        
                        # Add explanation of the test using a different UI element
                        st.write("ℹ️ **Understanding the Test Results**")
                        if test_type == "KS":
                            st.markdown("""
                            **Kolmogorov-Smirnov (KS) Test Interpretation:**
                            - **What it tests:** Whether two samples come from the same distribution
                            - **Test statistic:** Maximum distance between the cumulative distributions (0 to 1)
                            - **Effect size interpretation:**
                              * ≥ 0.5: Large difference
                              * ≥ 0.3: Moderate difference
                              * ≥ 0.1: Small difference
                              * < 0.1: Minimal difference
                            - **P-value:** Probability of seeing such differences by chance
                              * < 0.05: Strong evidence of different distributions
                              * ≥ 0.05: Insufficient evidence of different distributions
                            """)
                        else:  # Chi-square
                            st.markdown("""
                            **Chi-square Test Interpretation:**
                            - **What it tests:** Whether category proportions have changed
                            - **Test statistic:** Sum of standardized squared differences
                            - **Effect size (Cramer's V) interpretation:**
                              * ≥ 0.5: Large change in proportions
                              * ≥ 0.3: Moderate change in proportions
                              * ≥ 0.1: Small change in proportions
                              * < 0.1: Minimal change in proportions
                            - **P-value:** Probability of seeing such changes by chance
                              * < 0.05: Strong evidence of changed proportions
                              * ≥ 0.05: Insufficient evidence of changed proportions
                            """)
                        
                        # Basic Statistics Comparison
                        st.write("### Basic Statistics Comparison")
                        
                        # Check if the feature is categorical
                        is_categorical = (not pd.api.types.is_numeric_dtype(self.original_df[feature]) or 
                                       not pd.api.types.is_numeric_dtype(self.modified_df[feature]))
                        
                        if is_categorical:
                            # Get value counts for both distributions
                            orig_counts = self.original_df[feature].value_counts()
                            mod_counts = self.modified_df[feature].value_counts()
                            
                            # Get all unique categories and convert to strings for consistent handling
                            all_categories = set(orig_counts.index) | set(mod_counts.index)
                            all_categories = [str(cat) for cat in all_categories]
                            
                            # Sort categories (handle special case for binary features)
                            if len(all_categories) <= 2:
                                # For binary features, try to sort in a sensible way (0/1, False/True, No/Yes, etc.)
                                binary_sort_order = {
                                    ('0', '1'): ['0', '1'],
                                    ('false', 'true'): ['false', 'true'],
                                    ('no', 'yes'): ['no', 'yes'],
                                    ('n', 'y'): ['n', 'y'],
                                    ('f', 't'): ['f', 't']
                                }
                                
                                # Convert to lowercase for comparison
                                cats_lower = set(cat.lower() for cat in all_categories)
                                for key_pair, ordered_vals in binary_sort_order.items():
                                    if cats_lower == set(key_pair):
                                        # Found a matching binary pattern, use its order
                                        # but preserve original case
                                        case_map = {cat.lower(): cat for cat in all_categories}
                                        all_categories = [case_map[val] for val in ordered_vals]
                                        break
                                else:
                                    # If no special pattern matches, use regular string sorting
                                    all_categories = sorted(all_categories)
                            else:
                                # For non-binary categorical, use regular string sorting
                                all_categories = sorted(all_categories)
                            
                            # Create comparison DataFrame
                            cat_comparison = pd.DataFrame(index=all_categories)
                            
                            # Add counts and convert to numeric
                            cat_comparison['Original Count'] = pd.to_numeric([
                                orig_counts.get(str(cat), orig_counts.get(cat, 0)) 
                                for cat in all_categories
                            ], errors='coerce')
                            
                            cat_comparison['Modified Count'] = pd.to_numeric([
                                mod_counts.get(str(cat), mod_counts.get(cat, 0)) 
                                for cat in all_categories
                            ], errors='coerce')
                            
                            # Calculate percentages
                            orig_total = cat_comparison['Original Count'].sum()
                            mod_total = cat_comparison['Modified Count'].sum()
                            
                            cat_comparison['Original %'] = (cat_comparison['Original Count'] / orig_total * 100 
                                                          if orig_total > 0 else 0)
                            cat_comparison['Modified %'] = (cat_comparison['Modified Count'] / mod_total * 100 
                                                          if mod_total > 0 else 0)
                            
                            # Calculate changes
                            # Convert to float explicitly to avoid boolean subtraction issues
                            cat_comparison['Count Change'] = cat_comparison['Modified Count'].astype(float) - cat_comparison['Original Count'].astype(float)
                            cat_comparison['% Point Change'] = cat_comparison['Modified %'].astype(float) - cat_comparison['Original %'].astype(float)
                            
                            # Format the categorical comparison table
                            formatted_cat = pd.DataFrame(index=cat_comparison.index)
                            
                            # Format counts and percentages with error handling
                            def safe_format_count(x):
                                try:
                                    return f"{int(x):,}" if pd.notnull(x) else "0"
                                except:
                                    return "0"
                            
                            def safe_format_percentage(x):
                                try:
                                    return f"{float(x):.2f}%" if pd.notnull(x) else "0.00%"
                                except:
                                    return "0.00%"
                            
                            def safe_format_change(x):
                                try:
                                    return f"{int(x):+,}" if pd.notnull(x) else "+0"
                                except:
                                    return "+0"
                            
                            def safe_format_pct_change(x):
                                try:
                                    return f"{float(x):+.2f}%" if pd.notnull(x) else "+0.00%"
                                except:
                                    return "+0.00%"
                            
                            formatted_cat['Original Count'] = cat_comparison['Original Count'].apply(safe_format_count)
                            formatted_cat['Modified Count'] = cat_comparison['Modified Count'].apply(safe_format_count)
                            formatted_cat['Original %'] = cat_comparison['Original %'].apply(safe_format_percentage)
                            formatted_cat['Modified %'] = cat_comparison['Modified %'].apply(safe_format_percentage)
                            formatted_cat['Count Change'] = cat_comparison['Count Change'].apply(safe_format_change)
                            formatted_cat['% Point Change'] = cat_comparison['% Point Change'].apply(safe_format_pct_change)
                            
                            # Create styling function for categorical data
                            def highlight_cat_changes(row):
                                styles = [''] * len(row)
                                try:
                                    pct_change = float(row['% Point Change'].strip('%'))
                                    if abs(pct_change) > 5:  # Highlight changes > 5 percentage points
                                        return ['background-color: rgba(255,165,0,0.1)'] * len(row)
                                except:
                                    pass
                                return styles
                            
                            # Display the categorical comparison
                            st.write("**Category Distribution Comparison:**")
                            styled_cat = formatted_cat.style.apply(highlight_cat_changes, axis=1)
                            st.dataframe(styled_cat)
                            
                        else:
                            # Get statistics for both distributions
                            orig_stats = self.original_df[feature].describe()
                            mod_stats = self.modified_df[feature].describe()
                            
                            # Create a comparison DataFrame
                            stats_comparison = pd.DataFrame({
                                'Original': orig_stats,
                                'Modified': mod_stats
                            })
                            
                            # Calculate absolute and percentage differences
                            stats_comparison['Absolute Difference'] = stats_comparison['Modified'].astype(float) - stats_comparison['Original'].astype(float)
                            stats_comparison['Percentage Change'] = (
                                (stats_comparison['Modified'].astype(float) - stats_comparison['Original'].astype(float)) / 
                                stats_comparison['Original'].astype(float) * 100
                            )
                            
                            # Format the statistics table
                            formatted_stats = stats_comparison.copy()
                            
                            # Format numeric columns
                            for col in ['Original', 'Modified', 'Absolute Difference']:
                                formatted_stats[col] = formatted_stats[col].apply(
                                    lambda x: f"{x:,.3f}" if abs(x) >= 0.001 else f"{x:.3e}"
                                )
                            
                            # Format percentage change
                            formatted_stats['Percentage Change'] = formatted_stats['Percentage Change'].apply(
                                lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A"
                            )
                            
                            # Create styling function for numeric data
                            def highlight_changes(row):
                                styles = [''] * len(row)
                                try:
                                    pct_change = row['Percentage Change']
                                    if isinstance(pct_change, str) and pct_change != 'N/A':
                                        pct_value = float(pct_change.strip('%'))
                                        if abs(pct_value) > 10:
                                            return ['background-color: rgba(255,165,0,0.1)'] * len(row)
                                except:
                                    pass
                                return styles
                            
                            # Display the formatted statistics with styling
                            styled_stats = formatted_stats.style.apply(highlight_changes, axis=1)
                            st.dataframe(styled_stats)
                        
                        # Distribution visualisations
                        st.write("### Distribution Comparison")
                        # Show original and modified distributions side by side
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("### Original Distribution")
                            figs_orig = self._create_feature_distribution_plot(
                                self.original_df,
                                feature,
                                f"Original Distribution of {feature}"
                            )
                            for fig_idx, fig in enumerate(figs_orig):
                                try:
                                    plot_key = f"dist_change_orig_{feature}_{fig_idx}"
                                    st.plotly_chart(fig, config={'responsive': True}, key=plot_key)
                                except Exception as e:
                                    st.error(f"Error displaying plot: {str(e)}")
                        
                        with col2:
                            st.write("### Modified Distribution")
                            figs_mod = self._create_feature_distribution_plot(
                                self.modified_df,
                                feature,
                                f"Modified Distribution of {feature}"
                            )
                            for fig_idx, fig in enumerate(figs_mod):
                                try:
                                    plot_key = f"dist_change_mod_{feature}_{fig_idx}"
                                    st.plotly_chart(fig, config={'responsive': True}, key=plot_key)
                                except Exception as e:
                                    st.error(f"Error displaying plot: {str(e)}")
                                    
        except Exception as e:
            display_info_message(f"Error analyzing feature {feature}: {str(e)}", "error") 