"""
Calculation explanations content for ML Builder.

Contains method explanations and interpretation guides for various
machine learning calculations and metrics.
"""

from typing import Dict

CALCULATION_EXPLANATIONS = {
    "threshold_tuning_curves": {
        "method": """
            **Threshold Tuning Curves**
            - Visualizes how different performance metrics change as you vary the decision threshold
            - **X-axis**: Decision Threshold (probability cutoff for "Positive" prediction)
            - **Y-axis**: Metric Score (0 to 1)
            - Helps you find the "sweet spot" that best meets your business goals
        """,
        "interpretation": """
            **How to Read the Results:**
            
            **Key Trade-offs:**
            - **As Threshold Increases (Moving Right):**
              * **Precision** usually goes UP (fewer false alarms)
              * **Recall** usually goes DOWN (miss more actual positives)
              * **Specificity** goes UP (better at identifying negatives)
            
            - **As Threshold Decreases (Moving Left):**
              * **Recall** usually goes UP (catch more positives)
              * **Precision** usually goes DOWN (more false alarms)
            
            **Finding the Optimal Point:**
            - **F1 Score Peak**: Best balance of Precision and Recall
            - **Crossing Points**: Where different metrics intersect (e.g., Precision = Recall)
            - **Plateaus**: Ranges where metrics are stable (robust choices)
            
            **Vertical Dashed Line:** 
            - Shows the mathematically optimal threshold based on your chosen criterion
        """
    },
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
    },
    "precision_recall_curve": {
        "method": """
            **Precision-Recall Curve**
            - Shows the trade-off between precision and recall for different decision thresholds.
            - **X-axis (Recall)**: How many actual positive cases the model finds
            - **Y-axis (Precision)**: How accurate the model is when it predicts positive
            - **AP Score**: Average Precision - summarizes the curve (higher is better)
            - **Red dashed line**: Random classifier baseline
        """,
        "interpretation": """
            **Key Points:**
            - **High precision, low recall**: Model is very accurate but misses many positive cases
            - **Low precision, high recall**: Model finds most positive cases but makes many false predictions
            - **Curve close to top-right**: Excellent performance
            - **Curve close to baseline**: Poor performance

            **When to use:**
            - Imbalanced datasets (better than ROC curve)
            - When false positives are costly
            - When you need to understand precision-recall trade-offs

            **What good looks like:**
            - Curve stays close to the top of the plot
            - AP score > 0.7 is generally good
            - Significantly above the red baseline
        """
    },
    "probability_distribution_binary": {
        "method": """
            **Probability Distribution**
            - Shows how confident the model is in its predictions for each class.
            - **Blue bars**: Distribution for actual negative class (0)
            - **Red bars**: Distribution for actual positive class (1)
            - **X-axis**: Prediction probability (0 = confident negative, 1 = confident positive)
            - **Black dashed line**: Default decision threshold (0.5)
        """,
        "interpretation": """
            **Key Points:**

            **What good separation looks like:**
            - Blue bars clustered near 0 (model confident about negatives)
            - Red bars clustered near 1 (model confident about positives)
            - Minimal overlap between the two distributions

            **What poor separation looks like:**
            - Both distributions clustered around 0.5
            - Heavy overlap between blue and red bars
            - Similar shapes for both classes

            **How to use this:**
            - Well-separated distributions → model has learned good decision boundaries
            - Overlapping distributions → model struggles to distinguish classes
            - Can help identify optimal probability thresholds
            - Shows model confidence levels
        """
    },
    "probability_distribution_multiclass": {
        "method": """
            **Confidence Distribution**
            - Shows how confident the model is when making predictions for each true class.
            - **Each subplot**: One true class (what the samples actually are)
            - **Green bars**: Correct predictions (model got it right)
            - **Red bars**: Incorrect predictions (model got it wrong)
            - **X-axis**: Model confidence level (0 = uncertain, 1 = very confident)
            - **Accuracy %**: Shown in top-right corner of each subplot
        """,
        "interpretation": """
            **Key Points:**

            **What good performance looks like:**
            - **Green bars on the right**: Correct predictions with high confidence (> 0.8)
            - **Red bars on the left**: Incorrect predictions with low confidence (< 0.6)
            - **High accuracy %**: Most predictions are correct for this class

            **What poor performance looks like:**
            - **Red bars on the right**: Incorrect but confident predictions (dangerous!)
            - **Green bars in middle**: Correct but uncertain predictions
            - **Low accuracy %**: Model struggles with this class

            **How to interpret each subplot:**
            - **True Class X**: All samples that are actually class X
            - **Green distribution**: When model correctly identifies class X samples
            - **Red distribution**: When model incorrectly predicts something else
            - **Ideal pattern**: Green bars clustered right (confident + correct), red bars clustered left (uncertain + wrong)

            **Actionable insights:**
            - **Classes with low accuracy**: Need more training data or feature engineering
            - **Confident but wrong (red on right)**: Review those specific samples for labeling errors
            - **Uncertain but right (green in middle)**: Consider confidence thresholds for decision-making
        """
    },
    "error_by_confidence": {
        "method": """
            **Error Rate by Confidence**
            - Shows how often the model makes mistakes based on how confident it is, with trend analysis and calibration metrics.
            - **Blue bars**: Error rate for each confidence level
            - **Red trend line**: Overall relationship pattern
            - **Calibration metrics**: How well probabilities match reality
        """,
        "interpretation": """
            **Key Points:**

            **What good calibration looks like:**
            - **Trend line slopes downward**: Higher confidence → Lower error rate
            - **Low Brier Score** (< 0.1): Accurate probability predictions
            - **Low ECE** (< 0.05): Predicted probabilities match actual outcomes

            **What poor calibration looks like:**
            - **Flat trend line**: Confidence doesn't correlate with accuracy
            - **High Brier Score** (> 0.2): Inaccurate probability predictions
            - **High ECE** (> 0.1): Model overconfident or underconfident

            **Calibration Metrics Explained:**
            - **Brier Score**: Overall accuracy of probability predictions (0 = perfect)
            - **Expected Calibration Error (ECE)**: How much predicted probabilities deviate from actual frequencies

            **Business Implications:**
            - **Well-calibrated**: Can trust probability thresholds for decisions
            - **Overconfident**: Lower thresholds for high-stakes decisions
            - **Underconfident**: Can be more aggressive with predictions
            - **Poor calibration**: Consider probability calibration techniques
        """
    },
    "error_by_feature": {
        "method": """
            **Error by Feature Ranges**
            - Shows if the model struggles more in certain regions of your feature space.
            - **X-axis**: Feature value ranges
            - **Y-axis**: Error rate in that range
            - **Bars**: How often the model is wrong in each range
        """,
        "interpretation": """
            **Key Points:**

            **What to look for:**
            - **Even error rates**: Model performs consistently across feature ranges
            - **High error spikes**: Model struggles in specific ranges
            - **Edge effects**: Higher errors at extreme values

            **Common patterns:**
            - **Data sparsity**: Higher errors where training data is sparse
            - **Class imbalance**: Higher errors for underrepresented regions
            - **Non-linear relationships**: Errors in regions where linear assumptions break down

            **How to improve:**
            - Collect more data in high-error regions
            - Consider feature transformations
            - Review outliers in problematic ranges
            - Adjust model complexity for those regions
        """
    },
    "error_by_prediction_range": {
        "method": """
            **Error by Prediction Range**
            - Shows how model accuracy varies across different prediction ranges.
            - **X-axis**: Prediction value ranges (grouped into bins)
            - **Y-axis**: Absolute error magnitude
            - **Box plots**: Distribution of errors within each range
        """,
        "interpretation": """
            **Key Points:**

            **What good performance looks like:**
            - Similar box heights across all ranges
            - Low median error (box center) in all ranges
            - Few outliers in any range

            **What poor performance looks like:**
            - Much higher errors in certain ranges
            - Many outliers in specific ranges
            - Systematic patterns (increasing/decreasing errors)

            **How to use this:**
            - Identify where your model struggles most
            - Detect if model performs worse for high/low values
            - Guide data collection efforts
            - Inform confidence in predictions by range

            **Common patterns:**
            - **Heteroscedasticity**: Errors increase with prediction values
            - **Range limitation**: Poor performance at extreme values
            - **Data sparsity**: High errors where training data is sparse
        """
    },
    "influential_points": {
        "method": """
            **Influential Points Analysis**
            - Identifies data points that have unusual impact on the model.
            - **X-axis**: Sample index (data point number)
            - **Y-axis**: Influence score (higher = more influential)
            - **Color**: Intensity shows influence level
            - **Red line**: High influence threshold
        """,
        "interpretation": """
            **Key Points:**

            **What high influence means:**
            - Points that strongly affect model fit
            - Combination of unusual feature values and large residuals
            - May be outliers or important edge cases

            **How to interpret:**
            - **Points above red line**: Potentially problematic
            - **Scattered low values**: Normal, healthy pattern
            - **Few high spikes**: May indicate data quality issues

            **What to do with influential points:**
            1. **Investigate**: Are these data entry errors?
            2. **Domain check**: Do they represent valid but rare cases?
            3. **Consider removal**: If they're errors or not representative
            4. **Collect more data**: In regions with high influence

            **Important notes:**
            - Not all influential points are bad
            - Some may represent important edge cases
            - Always investigate before removing data
        """
    },
    "residuals_analysis": {
        "method": """
            **Residuals** are the differences between actual and predicted values. They help us understand:
            - How well our model fits the data
            - Whether our model's assumptions are met
            - Where our model makes the biggest errors
            - If our model is biased in certain regions
        """,
        "interpretation": """
            **Good Model Characteristics:**
            1. Random scatter in Residuals vs Predicted
            2. Normal distribution of residuals
            3. Points following Q-Q line
            4. Constant spread in Scale-Location

            **Common Issues:**
            1. **Non-linearity**
                - Curved patterns in residual plots
                - Solution: Try non-linear transformations

            2. **Heteroscedasticity**
                - Funnel shapes in residual plots
                - Solution: Try variable transformation

            3. **Non-normal Errors**
                - Skewed distribution
                - Q-Q plot deviations
                - Solution: Check outliers, try transformations

            4. **Outliers**
                - Points far from others
                - Solution: Investigate and possibly remove

            **Next Steps if Issues Found:**
            - Review feature engineering
            - Consider data transformations
            - Check for outliers
            - Try different model types
        """
    },
    "influential_points_table": {
        "method": """
            **Understanding the Influential Points Table**
            - **Sample_Index**: Original position in your test dataset
            - **Actual_Value**: True target value for this sample
            - **Predicted_Value**: Model's prediction for this sample
            - **Residual**: Difference between actual and predicted (error)
            - **Influence_Score**: Calculated influence measure (higher = more influential)
            - **Feature Columns**: Values of the most important features for these samples
        """,
        "interpretation": """
            **Color Coding:**
            - 🔴 **Light Red**: Influence Score column (these are all high influence points)
            - 🟡 **Light Yellow**: Residuals that are particularly large (> 2 standard deviations)

            **What to Look For:**
            - **Unusual feature combinations**: Do these samples have strange feature values?
            - **Data entry errors**: Are there any obviously incorrect values?
            - **Edge cases**: Do these represent rare but valid scenarios?
            - **Patterns**: Are the influential points clustered in certain feature ranges?

            **Next Steps:**
            1. **Investigate each point**: Look at the feature values and check if they make sense
            2. **Domain validation**: Consult with subject matter experts about these cases
            3. **Data quality check**: Verify these aren't data collection or entry errors
            4. **Consider action**: Decide whether to keep, correct, or remove these points
        """
    },
    "prediction_intervals": {
        "method": """
            **Prediction Intervals**
            - Shows the uncertainty in model predictions using ensemble variation.
            - **Green shaded area**: 90% prediction interval
            - **Red dots**: Actual values
            - **Blue dots**: Model predictions
            - **X-axis**: Sample index (sorted by actual value)
        """,
        "interpretation": """
            **Key Points:**

            **What good intervals look like:**
            - Most actual values (red dots) fall within green area
            - Interval width reflects true uncertainty
            - Predictions (blue dots) close to actual values

            **What poor intervals look like:**
            - Many actual values outside green area
            - Intervals too narrow or too wide
            - Systematic bias in predictions

            **How to use this:**
            - **Narrow intervals**: High model confidence
            - **Wide intervals**: High uncertainty - be cautious
            - **Values outside intervals**: May need investigation
            - **Systematic patterns**: Suggest model improvements needed

            **Business applications:**
            - Risk assessment for predictions
            - Setting safety margins
            - Identifying when to seek additional information
            - Communicating uncertainty to stakeholders

            **Technical notes:**
            - Available for ensemble models (Random Forest, Extra Trees) and boosting models (XGBoost, LightGBM, Gradient Boosting)
            - Traditional ensembles: Based on variation across individual estimators
            - Boosting models: Generated using prediction variations or input perturbations
            - 90% interval means 90% of predictions should fall within bounds
        """
    },
    "residuals_vs_predicted": {
        "method": """
            **Residuals vs Predicted**
            - Shows the relationship between model predictions and errors.
        """,
        "interpretation": """
            **What to Look For:**
            - Points should scatter randomly around the horizontal line at y=0
            - No clear patterns or trends
            - Even spread above and below the line

            **Red Flags:**
            - Curved patterns suggest non-linear relationships
            - Funnel shapes indicate heteroscedasticity
            - Clusters suggest missing variables
            - Outliers far from y=0

            **Ideal Pattern:**
            Random scatter with:
            - Even spread vertically
            - No obvious patterns
            - Most points between -2 and 2 on y-axis
        """
    },
    "residuals_qq_plot": {
        "method": """
            **Normal Q-Q Plot**
            - Compares the distribution of errors to a perfect normal distribution.
        """,
        "interpretation": """
            **What to Look For:**
            - Points following the diagonal line
            - Minimal deviation from line
            - Even spread along the line

            **Red Flags:**
            - S-shaped curve
            - Points far from diagonal
            - Heavy tails (ends deviate)

            **Why It Matters:**
            - Tests normality assumption
            - Shows outlier impact
            - Identifies systematic deviations
        """
    },
    "residuals_distribution": {
        "method": """
            **Residual Distribution**
            - Shows the frequency of different error sizes.
        """,
        "interpretation": """
            **What to Look For:**
            - Bell-shaped (normal) distribution
            - Centered at zero
            - Symmetric spread

            **Red Flags:**
            - Skewness (leaning left or right)
            - Multiple peaks
            - Long tails
            - Center not at zero

            **Why It Matters:**
            - Shows if errors are normally distributed
            - Helps identify bias in predictions
            - Validates regression assumptions
        """
    },
    "residuals_scale_location": {
        "method": """
            **Scale-Location Plot**
            - Shows if the spread of errors changes with prediction value.
        """,
        "interpretation": """
            **What to Look For:**
            - Horizontal line with constant spread
            - Random scatter of points
            - No obvious patterns

            **Red Flags:**
            - Funnel shapes
            - Increasing/decreasing spread
            - Clusters or patterns

            **Why It Matters:**
            - Tests homoscedasticity
            - Shows if error variance is constant
            - Identifies prediction reliability
        """
    }
}