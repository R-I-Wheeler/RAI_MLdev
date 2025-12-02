"""
Calculation explanations content for ML Builder.

Contains method explanations and interpretation guides for various
machine learning calculations and metrics.
"""

from typing import Dict

CALCULATION_EXPLANATIONS = {
    "feature_importance": {
        "method": """
            **Feature Importance Analysis**
            - Shows which features have the strongest influence on predictions
            - Based on statistical analysis and model coefficients
            - Helps identify key predictors in your data
            - Guides feature selection decisions
        """,
        "interpretation": """
            **Understanding the Results:**

            **Importance Score:**
            - Higher score = more important feature
            - Relative scale (compare between features)
            - Consider both positive and negative impacts
            - Look for natural groupings of importance

            **What to Look For:**
            1. Features with very high scores
               - These are your key predictors
               - Focus on these for data quality
               - May need extra validation

            2. Features with very low scores
               - Might not be useful for prediction
               - Consider removing to simplify model
               - Check if they're correlated with other features

            3. Unexpected results
               - Challenge your assumptions
               - Look for data quality issues
               - Consider domain knowledge

            **Using the Results:**
            - Focus data cleaning on important features
            - Consider removing lowest importance features
            - Guide future data collection
            - Validate against domain expertise
        """
    },
    "statistical_tests": {
        "method": """
            **Statistical Tests Explained**

            **Chi-square Test (For Category vs Category)**
            - Helps understand if two categorical variables are related
            - Example: Is there a connection between color preference and gender?
            - Like asking if the patterns we see are real or just by chance

            **Independent T-test (For Yes/No vs Numbers)**
            - Compares averages between two groups
            - Example: Do people who exercise regularly weigh different from those who don't?
            - Helps decide if the difference between groups is real

            **ANOVA (For Multiple Groups vs Numbers)**
            - Like T-test but for more than two groups
            - Example: Do people from different cities have different income levels?
            - Helps spot differences across multiple groups

            **Pearson Correlation (For Number vs Number)**
            - Shows if two numbers move together
            - Ranges from -1 (opposite movement) to +1 (same movement)
            - Example: As height increases, does weight tend to increase too?
        """,
        "interpretation": """
            **Understanding the Results:**

            **Chi-square Value:**
            - Measures how different the actual patterns are from what we'd expect by chance
            - Like measuring the gap between what we see and what we'd expect if there was no relationship
            - Bigger number = bigger difference from random chance
            - Example: If we see Chi² = 0, the pattern is exactly what we'd expect by chance
            - Example: If we see Chi² = 10, there's a bigger gap from random chance

            **T Statistic (or Test Statistic):**
            - Think of it as the strength of the difference we found
            - Bigger number = stronger evidence of a real difference
            - Small number = difference might be just random chance

            **P Value (Probability Value):**
            - The chance that what we see is just random luck
            - Less than 0.05 (5%) = We're pretty confident it's a real pattern
            - More than 0.05 = Could just be random chance
            - Think of it like a weather forecast: 5% chance of being wrong

            **Effect Size:**
            - Shows how big the difference or relationship is
            - Like measuring the size of a wave, not just if there is a wave
            - Small effect: Tiny but maybe still important
            - Medium effect: Notable difference
            - Large effect: Big, obvious difference

            **Simple Steps to Interpret:**
            1. Look at P-value first:
               - Below 0.05? = "We found something!"
               - Above 0.05? = "Not enough evidence"

            2. If P-value shows something interesting:
               - Check effect size to see how important it might be
               - Small effect might still matter for important decisions
               - Large effect usually worth paying attention to

            3. Think about real-world meaning:
               - Statistical significance doesn't always mean practical importance
               - Consider what the difference means for your specific problem
        """
    },
    "classification_report": {
        "method": """
            **Classification Report**
            - Provides detailed performance metrics for each class
            - Shows precision, recall, and F1-score
            - Includes weighted averages across all classes
            - Helps identify class-specific performance issues
        """,
        "interpretation": """
            **Key Metrics Explained:**

            **Precision:**
            - How many of our positive predictions were correct
            - Higher is better (max 1.0)
            - Important when false positives are costly
            - Example: If precision is 0.9, 90% of our positive predictions were correct

            **Recall:**
            - How many actual positives did we catch
            - Higher is better (max 1.0)
            - Important when false negatives are costly
            - Example: If recall is 0.8, we caught 80% of all positive cases

            **F1-Score:**
            - Balance between precision and recall
            - Higher is better (max 1.0)
            - Good when you need both precision and recall
            - Helps compare overall performance

            **Macro Avg:**
            - Simple average across all classes
            - Treats all classes equally
            - Good for balanced datasets

            **Weighted Avg:**
            - Average weighted by class size
            - Accounts for class imbalance
            - More representative for imbalanced datasets
        """
    },
    "cross_validation": {
        "method": """
            **Cross-Validation**
            - Technique to assess model performance on different data splits
            - Divides data into multiple training and testing sets
            - Trains and tests model multiple times
            - Helps understand model stability and reliability
            - Reduces impact of random sampling
        """,
        "interpretation": """
            **Understanding the Results:**

            **Mean Score:**
            - Average performance across all folds
            - More reliable than single test score
            - Higher values generally better
            - Example: Mean accuracy of 0.85 means model typically gets 85% correct

            **Standard Deviation:**
            - Shows how much scores vary between folds
            - Lower is better (more stable model)
            - High variation might indicate:
              * Overfitting
              * Data quality issues
              * Insufficient data

            **Score Range:**
            - Difference between highest and lowest scores
            - Smaller range suggests more stable model
            - Large range might indicate:
              * Model sensitivity to data splits
              * Need for more data
              * Potential outliers

            **What to Look For:**
            1. High mean score with low standard deviation = Ideal
            2. High mean score with high variation = Potentially unstable
            3. Low mean score with low variation = Consistently poor
            4. Low mean score with high variation = Needs investigation
        """
    },
    "confusion_matrix": {
        "method": """
            **Confusion Matrix**
            - Shows how well the model predicts each class
            - Rows show actual classes
            - Columns show predicted classes
            - Numbers show how many predictions fall in each category
            - Perfect prediction would only have numbers on the diagonal
        """,
        "interpretation": """
            **How to Read the Results:**
            - Diagonal numbers (bottom-left to top-right) show correct predictions
            - Other numbers show different types of mistakes
            - Higher numbers on the diagonal is better
            - Look for patterns in the mistakes
            - Consider if some mistakes are more serious than others
        """
    },
    "regression_metrics": {
        "method": """
            **Regression Performance Metrics**
            - R² Score: How much of the variation is explained (0-100%)
            - MAE: Represents the average absolute difference between predicted and actual values. MAE is in the same units as your target variable
            - MSE: This is the average of squared differences between predicted and actual values and it penalizes larger errors more heavily than MAE. The units are squared (so if predicting prices in dollars, this would be dollars²)
            - RMSE: Square root of MSE bringing it back to the same units as the target variable
        """,
        "interpretation": """
            **How to Read the Results:**
            - R² closer to 100% is better
            - Lower MAE, MSE, and RMSE values are better
            - RMSE is most interpretable (same units as what you're predicting)
            - Compare these numbers to what's acceptable in your field
        """
    },
    "roc_curve": {
        "method": """
            **ROC Curve**
            - Shows model performance at different decision thresholds
            - Plots true positives vs false positives
            - Perfect model would go straight up, then right
            - Random guessing would follow the diagonal line
            - Area Under Curve (AUC) summarizes overall performance
        """,
        "interpretation": """
            **How to Read the Results:**
            - Curve closer to top-left corner is better
            - AUC of 1.0 is perfect
            - AUC of 0.5 is random guessing
            - Higher AUC means better model
            - Use to compare different models
        """
    },
    "learning_curve": {
        "method": """
            **Learning Curve**
            - Shows how model performance changes with more training data
            - Dark Blue line shows training performance
            - Light Blue line shows testing performance
            - Helps identify if more data would help
            - Shows if model is overfitting or underfitting
        """,
        "interpretation": """
            **How to Read the Results:**
            - Lines close together = good fit
            - Big gap between lines = overfitting
            - Both lines low = underfitting
            - Flat lines = more data might not help
            - Rising lines = more data might help
        """
    },
    "residuals": {
        "method": """
            **Residual Analysis**
            - Residuals are the differences between actual and predicted values
            - Shows where the model makes mistakes
            - Multiple plots show different aspects of the errors
            - Helps identify if the model is biased
            - Shows if errors are random or systematic
        """,
        "interpretation": """
            **How to Read the Results:**
            - Residuals vs Predicted: Look for random scatter
            - Distribution: Should look like a bell curve
            - Q-Q Plot: Points should follow the diagonal line
            - Scale-Location: Spread should be even
            - Patterns suggest the model might be missing something
        """
    },
    "shap_values": {
        "method": """
            **SHAP Values Explanation**
            - Shows how each feature affects individual predictions
            - Red means the feature value is high
            - Blue means the feature value is low
            - Length of bars shows how much impact the feature has
            - Position (left/right) shows if it increases/decreases the prediction
        """,
        "interpretation": """
            **How to Read the Results:**
            - Longer bars mean bigger impact
            - Red bars pushing right increase the prediction
            - Blue bars pushing left decrease the prediction
            - Look for consistent patterns
            - Focus on features with longest bars
        """
    },
    "feature_associations": {
        "method": """
            **Feature Associations Analysis**
            - Used to analyse relationships between features
            - Works with both numerical and categorical variables
            - Calculates appropriate correlation metrics based on data type:
              * Pearson's correlation for numerical vs numerical
              * Correlation ratio for numerical vs categorical
              * Cramer's V for categorical vs categorical
            - Provides a comprehensive view of feature relationships
        """,
        "interpretation": """
            **How to Read the Results:**

            **Correlation Values:**
            - Range from -1 to 1 for numerical relationships
            - Range from 0 to 1 for categorical relationships
            - Darker colors indicate stronger relationships

            **Interpretation Guidelines:**
            1. Numerical vs Numerical:
               - 1: Perfect positive correlation
               - -1: Perfect negative correlation
               - 0: No linear relationship

            2. Categorical vs Categorical (Cramer's V):
               - 0: No association
               - 1: Perfect association
               - >0.3: Strong association

            3. Numerical vs Categorical (Correlation Ratio):
               - 0: No association
               - 1: Perfect association
               - >0.4: Strong association

            **What to Look For:**
            - Strong associations between features (may indicate redundancy)
            - Unexpected relationships that merit investigation
            - Groups of related features
            - Potential feature selection insights
        """
    },
    "feature_distributions": {
        "method": """
            **Feature Distribution Analysis**
            - Shows how values are spread across each numerical feature
            - Includes histogram and box plot visualisations
            - Histogram shows frequency of values
            - Box plot shows quartiles and outliers
            - Helps identify patterns, skewness, and unusual values
        """,
        "interpretation": """
            **How to Read the Results:**

            **Histogram:**
            - Height shows how common each value is
            - Shape indicates distribution type:
              * Bell curve = Normal distribution
              * Skewed right = Long tail on right
              * Skewed left = Long tail on left
            - Multiple peaks suggest distinct groups

            **Box Plot:**
            - Box shows 25th to 75th percentiles
            - Line in box = Median (50th percentile)
            - Whiskers extend to most extreme non-outlier points
            - Individual points = Outliers

            **What to Look For:**
            - Unusual shapes or patterns
            - Presence of outliers
            - Skewness in the data
            - Gaps or clusters
            - Potential need for transformations
        """
    },
    "missing_values": {
        "method": """
            **Missing Values Analysis**
            - Visualizes patterns of missing data
            - Shows relationships between missing values
            - Identifies potential systematic missingness
            - Helps choose appropriate imputation strategies
            - Reveals potential data quality issues
        """,
        "interpretation": """
            **How to Read the Results:**

            **Missing Values Heatmap:**
            - Each row is an observation
            - Each column is a feature
            - Dark spots show missing values
            - Patterns may indicate:
              * Random missing values
              * Systematic missing values
              * Related missing values

            **Missing Values Correlation:**
            - Shows if missing values in one feature predict missing values in another
            - Red = Positive correlation
            - Blue = Negative correlation
            - Stronger colors = Stronger relationships

            **What to Look For:**
            1. Missing Completely at Random (MCAR):
               - Random scatter of missing values
               - No clear patterns

            2. Missing at Random (MAR):
               - Missing values related to other features
               - Clear patterns in correlation matrix

            3. Missing Not at Random (MNAR):
               - Systematic patterns
               - May require special handling

            **Implications:**
            - High correlation: Consider similar imputation strategies
            - Random missing: Simple imputation may work
            - Systematic missing: May need advanced techniques
            - Many missing: Consider dropping feature or using indicators
        """
    },
    "correlation": {
        "method": """
            **Correlation Analysis**
            - Measures linear relationships between numerical features
            - Values range from -1 to 1
            - Shows strength and direction of relationships
            - Helps identify redundant features
            - Useful for feature selection
        """,
        "interpretation": """
            **How to Read the Results:**

            **Correlation Values:**
            - 1: Perfect positive correlation
            - -1: Perfect negative correlation
            - 0: No linear relationship

            **Color Coding:**
            - Red: Positive correlation
            - Blue: Negative correlation
            - Darker colors: Stronger relationships
            - White/Light colors: Weak relationships

            **What to Look For:**
            1. Strong Correlations (|r| > 0.7):
               - May indicate redundant features
               - Consider removing one feature
               - Check for multicollinearity

            2. Moderate Correlations (0.3 < |r| < 0.7):
               - Features have some relationship
               - May be useful for prediction
               - Consider keeping both features

            3. Weak Correlations (|r| < 0.3):
               - Features mostly independent
               - May provide unique information
               - Consider importance for target
        """
    },
    "classification_metrics": {
        "method": """
            **Classification Performance Metrics**

            **Accuracy:**
            - Percentage of correct predictions (both positive and negative)
            - Range: 0 to 1 (higher is better)
            - Good for balanced datasets

            **Precision:**
            - Of all positive predictions, how many were actually positive
            - Range: 0 to 1 (higher is better)
            - Important when false positives are costly

            **Recall (Sensitivity):**
            - Of all actual positives, how many did we catch
            - Range: 0 to 1 (higher is better)
            - Important when false negatives are costly

            **F1 Score:**
            - Harmonic mean of precision and recall
            - Range: 0 to 1 (higher is better)
            - Balances precision and recall
        """,
        "interpretation": """
            **How to Interpret the Results:**

            **Accuracy:**
            - 0.90+ : Excellent performance
            - 0.80-0.90: Good performance
            - 0.60-0.80: Moderate performance
            - <0.60: Poor performance
            - Note: Consider class balance when interpreting

            **Precision:**
            - High: Few false positives
            - Low: Many false positives
            - Important for: Spam detection, medical diagnosis

            **Recall:**
            - High: Few false negatives
            - Low: Many false negatives
            - Important for: Fraud detection, disease screening

            **F1 Score:**
            - High: Good balance of precision and recall
            - Low: Poor performance in either precision or recall
            - Best for imbalanced datasets

            **Common Issues:**
            1. High accuracy but low F1: Check for class imbalance
            2. High precision, low recall: Model is too conservative
            3. High recall, low precision: Model is too aggressive
            4. All metrics low: Fundamental model issues
        """
    },
    "actual_vs_predicted": {
        "method": """
            **Actual vs Predicted Plot**
            - Shows how well predictions match actual values
            - Perfect predictions would fall on the diagonal line
            - Points above line: Model overestimates
            - Points below line: Model underestimates
            - Spread shows prediction uncertainty
        """,
        "interpretation": """
            **How to Read the Results:**
            - Points close to diagonal = Good predictions
            - Even scatter around line = Unbiased model
            - Points far from line = Large errors
            - Patterns in spread = Potential systematic errors
            - Look for areas where predictions are consistently off
        """
    },
    "model_selection_guide": {
        "method": """
            **Model Selection Guide**
            - Understanding the strengths and trade-offs of different machine learning models is crucial for making the right choice for your specific use case
            - Each model has different characteristics in terms of complexity, speed, interpretability, and performance
            - The choice depends on your data size, complexity, interpretability requirements, and computational constraints
        """,
        "interpretation": """
            **Choosing the Right Model:**

            **Consider these factors:**
            1. **Dataset size**: Larger datasets can support more complex models
            2. **Feature complexity**: Non-linear relationships may require more sophisticated models
            3. **Interpretability needs**: Some models are more explainable than others
            4. **Computational resources**: Training time and memory requirements vary significantly
            5. **Production requirements**: Prediction speed and model size constraints

            **General Guidelines:**
            - Start simple: Linear/Logistic regression for baseline
            - For complex patterns: Tree-based models or neural networks
            - For speed: Linear models or LightGBM
            - For interpretability: Linear models or decision trees
            - For best performance: Ensemble methods (XGBoost, LightGBM, Random Forest)
        """
    },
    "model_comparison_metrics": {
        "method": """
            **Quick Model Comparison**
            - Compares all available models using a small sample of your data with default parameters
            - Provides rough performance estimates to guide model selection
            - Uses maximum 1000 rows for fast computation
            - All models use default hyperparameters (not optimized)
        """,
        "interpretation": """
            **Important Considerations:**

            **Limitations of Quick Comparison:**
            - Uses only a small sample of your data
            - Default parameters may not be optimal for any model
            - Results may change significantly with full dataset and proper tuning
            - Rankings may be different in final evaluation

            **How to Use Results:**
            1. Look for models that consistently perform well
            2. Consider the balance between different metrics
            3. Don't rely solely on this comparison for final decisions
            4. Use as guidance for which models to focus on during tuning

            **Next Steps:**
            - Select model for hyperparameter tuning
            - Factor in interpretability and computational requirements
        """
    }
}