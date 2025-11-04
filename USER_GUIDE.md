# ML Builder — User Guide

A guided, 9‑stage workflow for building, training, evaluating, and explaining machine‑learning models with responsible‑AI built in.


## What ML Builder Does

ML Builder helps you:
- Load and validate a dataset (CSV)
- Explore and clean data
- Engineer and select features
- Choose and train models (with tuning options)
- Evaluate performance with visualizations
- Explain model behavior (SHAP) and assess fairness
- Export artifacts and a reproducible script

The app enforces a linear, stage‑by‑stage journey to ensure good practice and reproducibility.


## Who It’s For

- Practitioners and students who want a structured ML workflow
- Teams who need transparency, fairness checks, and reproducibility
- Anyone wanting an interactive, code‑optional way to train and explain models


## Installation and Launch

You can run ML Builder via the CLI after installing the package.

```bash
# Recommended: create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in editable (dev) mode from repo root
pip install -e .

# Launch the application (CLI entry point)
ml-builder
```

Alternate ways to run:

```bash
# Run via Python module
python -m ml_builder.cli

# Run Streamlit directly (from the ml_builder/ directory)
cd ml_builder
streamlit run app.py
```


## Sample Datasets

Two example files ship with the app for quick testing:
- Classification: `ml_builder/sample_data/Titanic_Dataset_simple.csv`
- Regression: `ml_builder/sample_data/miami-housing_xs.csv`


## How to Use the App: The 9‑Stage Workflow

Navigation lives in the sidebar. You must complete each stage in order; the next stage unlocks once the current one is completed. Your actions are tracked in a journey log for auditability.

### 1) Data Loading

**Goal:** Import your dataset, validate it, and select your target variable.

**What the application does:**
- **Multi-layer file validation**: Checks MIME type (via python-magic), file size (max 100MB), CSV structure, and column names for dangerous patterns
- **Sample datasets**: Provides two pre-loaded datasets (Titanic classification, Miami Housing regression) with detailed variable descriptions
- **Automatic missing target handling**: Detects and removes rows with missing target values, showing clear warnings
- **Intelligent problem type detection**: Automatically identifies binary classification, multiclass classification, or regression based on target characteristics
- **Smart categorical encoding**: For categorical targets, automatically applies label encoding (e.g., "Yes"/"No" → 0/1) with clear mapping displays

**Step-by-step process:**
1. Choose to load a sample dataset OR upload your own CSV file
2. The app validates file structure, size, and content safety
3. View the dataset overview: shape, data types, memory usage, missing values summary
4. Select your target variable from a dropdown
5. For categorical or ambiguous numeric targets, confirm whether to treat as classification or regression
6. Review the encoding mapping if categorical encoding was applied
7. Confirm and proceed to exploration

**Key features:**
- Detailed dataset statistics card showing rows, columns, numeric/categorical split
- Automatic detection of binary (2 classes), multiclass (3-10 classes), and regression targets
- Warning system for high percentages of missing target values (>50% triggers error)
- Small dataset warnings (<50 rows remaining after cleanup)
- Target value distribution preview and validation

**Tips:**
- Use sample datasets first to understand the workflow
- Ensure column names don't contain formula injection characters (=, +, -, @)
- For multiclass targets without 0, the app automatically applies label encoding
- All target transformations are logged and reversible via encoding mappings

### 2) Data Exploration

**Goal:** Thoroughly understand your dataset before making transformation decisions.

**What the application does:**
- **Automatic duplicate detection and removal**: Identifies both exact duplicates and partial duplicates (same features, different target)
- **Five comprehensive analysis sections** accessible via pills navigation:

**1. Target Feature Analysis**
- **Target distribution visualization**: Shows class balance (classification) or value spread (regression)
- **Feature-target relationships**: Automatic detection of linear, non-linear, or complex relationships
- **Statistical strength assessment**: Categorizes each relationship as Strong, Moderate, Weak, or Very Weak
- **Problem-specific metrics**:
  - Regression: Pearson correlation, Spearman correlation, Mutual Information
  - Binary classification: Point-Biserial correlation, ANOVA F-test, Chi-square, Cramér's V
  - Multiclass: ANOVA F-test, Chi-square, Cramér's V, Eta Squared
- Separate tabs for numerical and categorical feature analysis

**2. Feature Analysis**
- **Interactive feature selector**: Choose any feature for detailed examination
- **For numerical features**: Histograms, density plots, box plots, scatter plots vs. target, outlier statistics
- **For categorical features**: Bar charts, frequency tables, category distribution, relationship with target
- **Automated insights**: Identifies skewness, outliers, rare categories, encoding needs

**3. Correlation/Association Analysis**
- **Mixed-type correlation matrix**: Uses advanced methods (Pearson, Cramér's V, correlation ratios) for all data types
- **Interactive heatmap**: Color-coded with exact values on hover
- **Low information quality features**: Identifies features with:
  - Low target correlation
  - Low variance (mostly constant)
  - High missing values (>50%)
  - Weak overall relationships
- Provides keep/remove recommendations for each low-quality feature

**4. Correlation Groups Analysis**
- **Intelligent clustering**: Uses network analysis to group highly correlated features (>0.85 threshold)
- **Smart redundancy removal**: Advanced algorithm prioritizes features with:
  - Higher target correlation
  - Lower missing values
  - Better relationships with other features
- **Visual network graphs**: Shows feature clusters and connections
- **Actionable recommendations**: Data-driven suggestions on which features to keep/remove
- **Impact preview**: Shows potential feature reduction and benefits

**5. Feature Relationships**
- **Pairwise feature analysis**: Select any two features to examine their relationship
- **Optional grouping dimension**: Add a third feature to segment the analysis
- **Automatic visualization selection**: Chooses appropriate chart type (scatter, box, violin, heatmap) based on data types
- **Statistical testing**: Runs appropriate tests (t-test, ANOVA, chi-square, correlation) and displays results

**6. Data Quality Analysis**
- **Missing values heatmap**: Visual pattern detection
- **Column-by-column missing statistics**: Counts, percentages, patterns
- **Quality score calculation**: Overall dataset health assessment
- **Data integrity checks**: Identifies consistency issues

**Advanced Automated Preprocessing Option:**
At the end of Data Exploration, you can optionally run automated preprocessing that applies:
- Missing value imputation (mode for categorical, median for numerical)
- One-hot encoding for categorical features
- IQR-based outlier capping
This creates a ready-to-train dataset but limits fine-grained control.

**Tips:**
- Focus on Strong/Moderate relationship features—these are your most predictive variables
- Use correlation groups analysis to reduce redundancy before modeling
- Check for class imbalance in target distribution
- Investigate features with unexpected relationships to the target
- Consider removing low-quality features identified by the analysis

### 3) Data Preprocessing

**Goal:** Transform raw data into a clean, model-ready format through a 10-step guided pipeline.

**What the application does:**
This stage uses a sequential, step-by-step approach with visual progress tracking. Each step must be completed before moving to the next.

**Step 1: Feature Management**
- **Column removal interface**: Review all features and remove unwanted columns (IDs, redundant features)
- **Feature list with data types**: See which columns are numeric vs. categorical
- **Bulk selection**: Remove multiple columns at once
- All removals are logged and can be undone via "Undo All Changes" button

**Step 2: Zero Values Analysis**
- **Zero value detection**: Identifies columns with significant zero values
- **Context-aware flagging**: Determines if zeros are likely missing data or legitimate values
- **Feature-by-feature decisions**: Choose to:
  - Convert zeros to NaN (if they represent missing data)
  - Keep zeros (if they're valid measurements)
- **Impact preview**: Shows how many values will be affected

**Step 3: Train-Test Split**
- **Configurable split ratio**: Default 80/20, adjustable via slider
- **Stratification options**: For classification, ensures balanced class distribution
- **Random state control**: Set seed for reproducibility
- **Split summary statistics**: Shows resulting dataset sizes and class distributions
- **Visual confirmation**: Bar charts comparing train/test distributions

**Step 4: Missing Values Handling**
- **Comprehensive missing value report**: Shows count and percentage for each feature
- **Feature-by-feature strategy selection**:
  - **Drop column**: Remove features with >50% missing
  - **Drop rows**: Remove rows with missing values (use sparingly)
  - **Simple imputation**: Mean, median, mode
  - **Advanced imputation**: Forward fill, backward fill, interpolation, KNN imputation
- **Preview before applying**: See the effect of each strategy
- **Batch application**: Apply all selected strategies at once
- **Mandatory completion**: Must handle all missing values before proceeding

**Step 5: Feature Binning**
- **Numerical feature binning**: Convert continuous variables into categorical bins
- **Multiple binning strategies**:
  - **Equal-width**: Bins of equal range
  - **Equal-frequency (quantile)**: Bins with equal number of samples
  - **Custom bins**: Define your own bin edges
  - **Optimal binning**: Supervised binning optimized for target relationship (uses OptBinning library)
- **Interactive configuration**: Set number of bins, choose labels
- **Visual preview**: Histograms showing bin distributions
- **Multiple feature binning**: Apply binning to several features

**Step 6: Outlier Detection and Handling**
- **Automatic outlier detection**: Uses IQR method (1.5 × IQR rule)
- **Visual identification**: Box plots highlighting outliers
- **Handling strategies for each feature**:
  - **Keep**: Leave outliers unchanged
  - **Remove rows**: Delete outlier samples
  - **Cap**: Replace with threshold values (recommended)
  - **Transform**: Apply log or square root transformation
- **Statistics**: Shows number of outliers and their impact
- **Feature-by-feature control**: Different strategies for different features

**Step 7: Categorical Encoding**
- **Automatic categorical detection**: Identifies all categorical features
- **Rich encoding options**:
  - **One-Hot Encoding**: Creates binary columns (best for tree models)
  - **Label Encoding**: Ordinal integers (for ordinal features)
  - **Target Encoding**: Mean target value per category (powerful but risk of overfitting)
  - **Frequency Encoding**: Replace with category frequency
  - **Binary Encoding**: Efficient for high-cardinality features
  - **Ordinal Encoding**: Custom ordering for ordinal variables
- **Automatic drop_first for one-hot**: Prevents multicollinearity
- **Encoding mappings stored**: All transformations are reversible
- **Preview**: See sample transformations before applying
- **Mandatory completion**: All categorical features must be encoded

**Step 8: Feature Creation**
- **Mathematical operations**: Create new features from existing numerical features
- **Available operations**:
  - Addition, Subtraction, Multiplication, Division
  - Power, Square Root, Log, Exponential
  - Absolute Value, Reciprocal
- **Interactive builder**: Select two features, choose operation, name new feature
- **Validation**: Prevents division by zero, log of negative numbers
- **Multiple creations**: Add several engineered features
- **Optional step**: Can skip if no feature engineering needed

**Step 9: Data Types Optimization**
- **Memory optimization**: Converts data types to more efficient representations
- **Automatic downcasting**: 
  - int64 → int32 or int16 where possible
  - float64 → float32 where precision allows
  - object → category for low-cardinality strings
- **Memory savings report**: Shows MB saved
- **Safe conversions**: Preserves data integrity
- **One-click application**: "Optimize Data Types" button

**Step 10: Final Data Review**
- **Comprehensive dataset summary**: Final shape, data types, memory usage
- **Preprocessing history**: All transformations applied
- **Feature list with details**: Names, types, missing values, unique values
- **Comparison to original**: Shows changes from raw data
- **Ready-to-train confirmation**: Validates dataset is complete
- **Proceed to Feature Selection**: Final button to exit preprocessing

**Global Features:**
- **"Undo All Changes" button**: Resets dataset to original state from start of preprocessing
- **Progress indicator**: Visual bar showing current step in 10-step sequence
- **Scroll to Top button**: Quick navigation on long pages
- **Change tracking**: Shows dataset size changes at each step
- **Automatic logging**: All decisions recorded in journey tracker

**Note on Normalization:**
This app intentionally excludes normalization/standardization to maintain model explainability. Use binning and outlier handling instead to manage scale issues.

**Tips:**
- Always complete zero values analysis first—it affects missing value counts
- For categorical encoding, use one-hot for tree models, target encoding for boosted trees
- Cap outliers rather than removing them to preserve sample size
- Use optimal binning for features with non-linear target relationships
- Feature creation is optional—start simple and add complexity only if needed
- The "Undo All Changes" button is your safety net—use it if you want to restart

### 4) Feature Selection

**Goal:** Identify and select the most valuable features while removing redundancy and addressing fairness concerns.

**What the application does:**
This stage provides three main sections accessible via pills navigation:

**Section 1: Analysis Results**
- **Automatic feature importance analysis**: Runs on first visit, computing:
  - **Statistical importance scores**: Based on correlation, mutual information, ANOVA F-scores
  - **Correlation analysis**: Feature-to-feature and feature-to-target relationships
  - **Data quality metrics**: Missing values, variance, outliers per feature
  - **Protected attribute detection**: Flags gender, race, age-related features for fairness review
  
- **Comprehensive feature table** showing:
  - Importance score (0-1 scale)
  - Correlation with target
  - Missing value percentage
  - Data type and unique value count
  - Protected attribute flag (if applicable)
  - Quality warnings (high missing, low variance, etc.)

- **Visualization suite**:
  - Feature importance bar chart (sortable)
  - Correlation heatmap with hierarchical clustering
  - Missing values heatmap
  - Variance distribution plot

- **Bias and fairness alerts**:
  - Lists detected protected attributes
  - Identifies potential proxy variables (high correlation with protected attributes)
  - Provides fairness considerations for each protected feature

**Section 2: Feature Selection**
- **Interactive selection interface**: 
  - Table with checkboxes to exclude features
  - Bulk selection options (select all, deselect all, select by threshold)
  - Search/filter functionality

- **Smart recommendation system**:
  - Suggests features to remove based on:
    - Very low importance (<0.1)
    - High redundancy (>0.95 correlation with another feature)
    - Excessive missing values (>80%)
    - Near-zero variance
  - Explains reasoning for each recommendation

- **Automated feature selection options**:
  - **Correlation-based removal**: Removes one feature from highly correlated pairs (keeps the one with higher target correlation)
  - **Importance threshold filtering**: Removes features below a configurable importance score
  - **Missing value threshold**: Removes features above a missing percentage
  - **Variance threshold**: Removes near-constant features

- **Selection impact preview**:
  - Shows how many features will remain
  - Estimates information loss
  - Displays which redundancy groups will be affected
  - Compares before/after feature count

- **Manual override**: You can keep any recommended-for-removal feature if domain knowledge suggests it's important

**Section 3: Dataset Review**
- **Final feature list**: All features that will be used for modeling
- **Dataset summary statistics**: Shape, memory, completeness
- **Feature details table**: Name, type, missing %, unique values, importance
- **Duplicate detection report**: Shows any remaining duplicates
- **Readiness checklist**: Confirms:
  - No missing values in selected features
  - All features encoded properly
  - No perfect multicollinearity
  - Sufficient sample size

**Key features:**
- **Journey tracking**: All selection decisions logged with reasoning
- **Reversibility**: Can return to previous stages if needed
- **Protected attribute tracking**: Ensures fairness considerations are documented
- **Export selections**: Download feature list as CSV

**Proceed to Model Selection:**
Once satisfied with your feature set, click "Proceed to Model Selection" to move forward.

**Tips:**
- Start by reviewing automated recommendations, but apply domain expertise
- For highly correlated feature pairs, keep the one with higher target correlation
- Be cautious removing protected attributes—document your reasoning carefully
- If a low-importance feature has domain significance, keep it
- Aim for parsimony: simpler models with fewer features are often more robust
- Run correlation analysis after feature selection to verify redundancy is reduced

### 5) Model Selection

**Goal:** Choose the most appropriate machine learning algorithm for your problem and data characteristics.

**What the application does:**

**Problem Type Detection:**
- **Automatic problem type identification**: Binary classification, multiclass classification, or regression
- **XGBoost compatibility checking**: Verifies target classes start from 0 for multiclass problems
- **Clear problem type display**: Shows detected task with visual confirmation

**Model Explainer Section:**
- **Comprehensive model descriptions**: For each available algorithm:
  - How it works (intuitive explanation)
  - Strengths and weaknesses
  - Best use cases
  - Interpretability level
  - Computational requirements
  - Hyperparameter sensitivity

- **Available models by problem type**:
  - **Classification**: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, MLP (Neural Network), XGBoost, LightGBM, Hist Gradient Boosting, CatBoost
  - **Regression**: Linear Regression, Decision Tree, Random Forest, MLP, XGBoost, LightGBM, Hist Gradient Boosting, CatBoost

**Performance Metrics Explainer:**
- **Detailed metric descriptions** with:
  - What the metric measures
  - How to interpret it
  - When to use it
  - Ideal values and ranges
- **Classification metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression metrics**: RMSE, MAE, R², Adjusted R², MAPE

**Training Data Exploration Widget:**
- **Interactive data browser**: View training data characteristics
- **Summary statistics**: Feature distributions, correlations, target balance
- **On-demand access**: Dialog window that doesn't interrupt workflow

**Model Recommendation System:**
- **Data-driven recommendations**: Analyzes your dataset to suggest optimal algorithm
- **Reasoning display**: Explains why each model was recommended, e.g.:
  - "Large number of features (>50) → Random Forest handles high dimensionality well"
  - "Nonlinear relationships detected → Gradient boosting methods recommended"
  - "Small dataset (<500 samples) → Simple models to avoid overfitting"
  - "Many categorical features → CatBoost or tree-based models"
  - "Highly imbalanced classes → Models that support class weights"

**Quick Model Comparison:**
- **Automated benchmarking**: Trains multiple models on a sample (max 1000 rows) for speed
- **Comparison table showing**:
  - Model name
  - Training time
  - Key performance metrics (problem-specific)
  - Memory usage
  - Relative ranking
- **Best model highlighting**: Top performer highlighted in green
- **Balanced evaluation**: Uses cross-validation for reliability
- **XGBoost exclusion**: Automatically excludes XGBoost if compatibility issues detected

**Model Selection Interface:**
- **Dropdown selector**: Choose from all compatible models
- **Default to recommended**: Pre-selects the recommended model
- **Model information card**: Shows details for selected model
- **Hyperparameter preview**: Displays default parameters that will be used

**Automated Model Selection & Training (Optional):**
Optional workflow that:
1. Runs quick comparison on all models
2. Selects the best-performing model automatically
3. Trains with hyperparameter tuning (Optuna)
4. Evaluates and prepares for deployment
This is a fast path for users who want optimal results without manual configuration.

**Navigate to Training:**
Once you've selected a model, click "Proceed to Model Training" to configure and train it.

**Tips:**
- Trust the recommendation system—it considers many data characteristics
- For interpretability-critical applications, prefer Logistic Regression or Decision Trees
- For maximum performance, choose gradient boosting methods (XGBoost, LightGBM, CatBoost)
- Quick comparison is useful but not definitive—full training may yield different results
- If dataset is small (<1000 samples), avoid complex models (MLP, deep ensembles)
- Tree-based models handle missing values and mixed data types well
- Linear models are fast and interpretable but assume linear relationships

### 6) Model Training

**Goal:** Train your selected model with optimal hyperparameters, handling class imbalance and calibration if needed.

**What the application does:**

**Dataset Overview:**
- **Training/test split summary**: Sample counts, feature counts, test percentage
- **Class distribution**: For classification, shows balance across classes in both sets
- **Visual comparison**: Bar charts comparing train and test distributions

**Class Imbalance Detection & Handling (Classification Only):**
If class imbalance is detected (any class <20% of total):

- **Imbalance analysis preview**:
  - Class distribution visualization
  - Imbalance ratio calculation
  - Severity assessment
  - Potential impact on model performance

- **Handling strategies**:
  - **No action**: Train on imbalanced data (baseline)
  - **Class weights**: Automatically weight classes inversely to frequency
  - **SMOTE**: Synthetic Minority Over-sampling Technique (generates synthetic minority samples)
  - **Random undersampling**: Reduces majority class samples
  - **Random oversampling**: Duplicates minority class samples
  - **SMOTE + Undersampling**: Combined approach
  - **ADASYN**: Adaptive Synthetic Sampling (focuses on hard-to-learn examples)

- **Strategy comparison**:
  - Before/after class distributions
  - New training set size
  - Warnings about potential overfitting

- **Application**: One-click button to apply selected strategy
- **Skip option**: Can proceed without handling imbalance (not recommended)

**Training Mode Selection:**
Three options with increasing sophistication:

**1. Standard Training**
- **Simple fit**: Uses model default parameters or manually configured values
- **Cross-validation**: 5-fold CV for robust performance estimation
- **Fast**: Completes quickly, good for baseline
- **Metrics reported**: Mean CV score, standard deviation, train/test scores
- **No tuning**: Uses default hyperparameters

**2. Random Search Hyperparameter Tuning**
- **Search space**: Pre-defined ranges for each model's key hyperparameters
- **Configurable**:
  - Number of iterations (10-100)
  - CV folds (3-10)
  - Scoring metric
  - Random state for reproducibility
- **Sampling strategy**: Randomly samples from parameter distributions
- **Progress tracking**: Real-time updates showing:
  - Current iteration
  - Best score so far
  - Time elapsed
  - Estimated time remaining
- **Results**:
  - Best parameters found
  - CV scores distribution
  - Parameter importance analysis
  - Training vs. validation curves

**3. Optuna Bayesian Optimization**
- **Intelligent search**: Uses Bayesian optimization to find optimal hyperparameters efficiently
- **Configuration**:
  - Number of trials (20-200)
  - CV folds
  - Optimization metric
  - Early stopping (prune unpromising trials)
- **Advanced features**:
  - **Pruning**: Stops poor-performing trials early
  - **Parallel evaluation**: Can use multiple cores
  - **Visualization**: Optimization history, parameter importance, parallel coordinates
- **Rich results**:
  - Best trial details
  - Optimization history plot
  - Hyperparameter importance
  - Slice plots showing parameter effects
  - Parallel coordinate plot
  - Trials dataframe with all attempts
- **Most powerful**: Generally finds better hyperparameters than random search

**Model Calibration (Classification Only):**
Optional post-training calibration to improve probability estimates:

- **What it does**: Adjusts predicted probabilities to match true frequencies
- **Methods**:
  - **Isotonic regression**: Non-parametric, flexible (needs more data)
  - **Sigmoid/Platt scaling**: Parametric, works with smaller datasets
- **CV-based**: Uses cross-validation to avoid overfitting
- **Reliability diagrams**: Shows calibration before/after
- **Metrics**: ECE (Expected Calibration Error), Brier score
- **When to use**: If probability estimates matter (not just predictions)

**Threshold Optimization (Binary Classification Only):**
Optional threshold tuning for binary classifiers:

- **Default threshold**: 0.5 may not be optimal for imbalanced classes
- **Optimization criteria**:
  - **F1 Score**: Balances precision and recall
  - **Precision**: Minimize false positives
  - **Recall**: Minimize false negatives
  - **Balanced**: Optimizes for equal performance across classes
- **ROC curve analysis**: Visual display of threshold effect
- **Confusion matrix at different thresholds**: See trade-offs
- **Recommended threshold**: Highlighted based on criterion
- **Application**: One-click to apply optimal threshold

**Training Results Display:**
After training completes:

- **For Standard/Random Search**:
  - Best parameters
  - Cross-validation scores (mean, std, all folds)
  - Training time
  - Train vs. test performance
  - Parameter values table
  - Learning curves (if available)

- **For Optuna**:
  - All of the above, plus:
  - Optimization history (score over trials)
  - Parameter importance plot
  - Slice plots for each parameter
  - Parallel coordinate plot
  - Full trials database
  - Convergence analysis

**Model Selection (After Random Search/Optuna):**
If multiple parameter sets were explored:

- **Selection criteria**:
  - **Mean score**: Highest average CV score (default)
  - **Adjusted score**: Balances mean and consistency (mean - std)
- **Model comparison table**: Shows top candidates
- **Interactive selection**: Choose which model to proceed with
- **Rationale**: Explains why each model ranks where it does

**Session State Management:**
- **Automatic cleanup**: Frees memory from large trial histories
- **Cache optimization**: Keeps only essential data
- **Memory usage warnings**: Alerts if session is getting large

**Proceed to Evaluation:**
After training is complete and you're satisfied with results, click "Proceed to Model Evaluation" to assess performance in detail.

**Tips:**
- Always address class imbalance if detected—it significantly impacts performance
- Start with Standard Training to get a baseline, then tune if needed
- For small datasets (<1000 samples), use fewer CV folds (3) and fewer trials
- Optuna is more efficient than Random Search but takes longer per trial
- Calibration is important if you're using predicted probabilities (e.g., risk scores)
- Threshold optimization is crucial for imbalanced binary classification
- If Optuna finds similar scores across many trials, your model is stable
- Monitor training time—some models (MLP, XGBoost) can take much longer

### 7) Model Evaluation

**Goal:** Comprehensively assess model performance, understand its behavior, and identify areas for improvement.

**What the application does:**

**Dataset Overview Section:**
- **Side-by-side comparison**: Training vs. test dataset statistics
- **Sample counts**: Number of samples in each set
- **Feature counts**: Verification of feature consistency
- **Class distributions**: For classification, bar charts showing class balance in train and test sets
- **Test set percentage**: Proportion of data held out for validation

**Model Training Information Dashboard:**
- **5-metric summary cards** showing:
  1. **Model Type**: Algorithm used (e.g., Random Forest, XGBoost)
  2. **Optimization Method**: Standard, Random Search, or Optuna
  3. **Enhancements**: Calibration and/or threshold optimization status
  4. **Target Column**: Confirmed target variable
  5. **CV Score**: Cross-validation performance

**Performance Metrics Section:**

**For Classification:**
- **Core metrics with explanations**:
  - **Accuracy**: Overall correctness (with warnings if classes imbalanced)
  - **Precision**: How many positive predictions were correct
  - **Recall**: How many actual positives were found
  - **F1-Score**: Harmonic mean of precision and recall
  - **ROC-AUC**: Area under ROC curve (discrimination ability)
  - **Macro/Weighted averages**: For multiclass problems

- **Classification report table**:
  - Per-class precision, recall, F1
  - Support (sample count per class)
  - Color-coded performance (green = good, red = poor)

- **Confusion matrix**:
  - Interactive heatmap
  - Raw counts and percentages
  - Diagonal highlighting (correct predictions)
  - Misclassification patterns visible
  - Encoded and decoded versions (if target was encoded)

**For Regression:**
- **Core metrics with explanations**:
  - **RMSE**: Root Mean Squared Error (average prediction error in target units)
  - **MAE**: Mean Absolute Error (average absolute deviation)
  - **R²**: Coefficient of determination (variance explained, 0-1 scale)
  - **Adjusted R²**: R² adjusted for number of features
  - **MAPE**: Mean Absolute Percentage Error (% error)

- **Regression report**:
  - All metrics in a summary table
  - Comparison to baseline (mean predictor)
  - Percentage improvements

**Visualization Section:**

**For Classification:**
- **ROC Curve** (binary and multiclass):
  - True Positive Rate vs. False Positive Rate
  - AUC score for each class
  - Optimal threshold marked
  - Random classifier baseline

- **Precision-Recall Curve**:
  - Especially useful for imbalanced classes
  - Shows precision-recall trade-off
  - Average precision score

- **Calibration Curve** (if model was calibrated):
  - Predicted probabilities vs. actual frequencies
  - Perfect calibration line
  - Histogram of prediction distribution

- **Feature Importance Plot**:
  - Bar chart of top features
  - Relative importance scores
  - Sortable and filterable

**For Regression:**
- **Actual vs. Predicted Plot**:
  - Scatter plot with perfect prediction line
  - Shows systematic over/under-prediction
  - Color-coded by error magnitude

- **Residuals Plot**:
  - Residuals vs. predicted values
  - Identifies heteroscedasticity
  - Shows outliers

- **Residuals Distribution**:
  - Histogram of errors
  - Should be centered at 0
  - Identifies skewed errors

- **Q-Q Plot**:
  - Tests normality of residuals
  - Points should follow diagonal line

- **Error Distribution by Feature**:
  - How errors vary across feature ranges
  - Identifies where model struggles

**Sample Predictions Section:**
- **Interactive prediction browser**:
  - Shows random samples from test set
  - Displays actual vs. predicted values
  - Feature values for each sample
  - Prediction confidence/probability
  - Error magnitude

- **For classification**:
  - Shows predicted class and probability
  - All class probabilities
  - Correct/incorrect flag

- **For regression**:
  - Shows predicted value
  - Actual value
  - Absolute and percentage error

- **Filtering and sorting**:
  - View only errors
  - Sort by confidence or error
  - Search for specific samples

**Model Health & Improvements Section:**
- **Overfitting analysis**:
  - Compares train vs. test performance
  - Flags significant gaps
  - Suggests regularization if needed

- **Bias-Variance assessment**:
  - High bias: underfitting (model too simple)
  - High variance: overfitting (model too complex)
  - Recommendations for each case

- **Feature leakage detection**:
  - Identifies suspiciously perfect features
  - Warns if feature importance is too concentrated

- **Improvement recommendations**:
  - More data if sample size small
  - Feature engineering if linear relationships weak
  - Hyperparameter tuning if not yet done
  - Ensemble methods if single model insufficient
  - Calibration if probabilities uncalibrated
  - Threshold tuning if imbalanced classification

**Cross-Validation Results (if applicable):**
- **Fold-by-fold scores**: Shows consistency across folds
- **Mean and standard deviation**: Average performance and stability
- **Score distribution plot**: Box plot or violin plot of CV scores
- **High variance**: Indicates model instability or small dataset

**Proceed to Model Explanation:**
Once you're satisfied with model performance, click "Proceed to Model Explanation" to understand how your model makes decisions.

**Tips:**
- Don't rely solely on accuracy for imbalanced classification—check precision, recall, and F1
- For regression, RMSE is in target units—easier to interpret than R²
- Examine confusion matrix diagonal—should be much brighter than off-diagonal
- If train score >> test score, you have overfitting
- If both train and test scores are low, you have underfitting
- Sample predictions help sanity-check the model—do predictions make sense?
- For business applications, align metrics with cost structure (e.g., false negatives may be more costly)

### 8) Model Explanation

**Goal:** Understand how your model makes predictions, identify biases, and communicate model behavior to stakeholders.

**What the application does:**

This stage uses SHAP (SHapley Additive exPlanations) for model-agnostic explainability and provides four main sections:

**Navigation Guide (Expandable):**
- **Overview of explanation concepts**: What SHAP values are and how to interpret them
- **Section summaries**: Quick guide to what each section offers
- **Best practices**: When to use global vs. local explanations

**Test Data Summary:**
- **Quick access dialog**: View test dataset characteristics without leaving the page
- **On-demand statistics**: Shape, data types, distributions

**Encoded and Binned Feature Details:**
- **Transformation reference**: Shows all encodings and binning applied
- **Mapping tables**: Original values → encoded values
- **Bin definitions**: For binned features, shows bin edges and labels
- **Essential for interpretation**: Helps translate model inputs back to original values

**Section 1: Feature Analysis**

**Global Feature Importance:**
- **SHAP-based importance**: Mean absolute SHAP values across all predictions
- **Feature importance bar chart**: Ranked from most to least important
- **Comparison to traditional importance**: Shows difference from built-in feature importance
- **Downloadable**: Export feature importance as CSV

**SHAP Summary Plots:**
- **Beeswarm plot**: Each dot is a prediction
  - Position on x-axis: SHAP value (impact on prediction)
  - Color: Feature value (red = high, blue = low)
  - Shows how feature values affect predictions
  - Identifies positive vs. negative impacts

- **Violin plot**: Distribution of SHAP values per feature
  - Width: Density of SHAP values at that impact level
  - Reveals typical impact ranges

**Feature Dependence Plots:**
- **Interactive feature selector**: Choose any feature to analyze
- **Dependence plot**: SHAP value vs. feature value
  - Shows if relationship is linear, monotonic, or complex
  - Identifies thresholds or tipping points
  - Color by interaction feature (automatically selected or manual)
- **Partial Dependence Plot (PDP)**: Average effect of feature
- **ICE Plots**: Individual Conditional Expectation (one line per sample)
  - Shows heterogeneity in feature effects

**ALE Plots (Accumulated Local Effects):**
- **Unbiased feature effects**: Like PDP but accounts for feature correlations
- **More reliable** than PDP when features are correlated
- **Centered at zero**: Shows deviation from average prediction

**Section 2: Individual Predictions**

**Sample Selection:**
- **Random sample selector**: Pick any test sample
- **Filter options**: 
  - Correct vs. incorrect predictions
  - High vs. low confidence
  - Specific class (for classification)
- **Sample details**: Shows all feature values and actual/predicted outcome

**Waterfall Plot:**
- **For one prediction**: Shows how features contribute
- **Base value**: Average model output
- **Feature contributions**: Red (increases prediction), blue (decreases)
- **Final prediction**: After all contributions
- **Interactive**: Hover for exact values
- **Great for stakeholder communication**: "This prediction was Y because of features A (+0.3), B (-0.1)..."

**Force Plot:**
- **Visual push/pull**: Features pushing prediction higher (red) or lower (blue)
- **Magnitude**: Width of segment = strength of contribution
- **Output value**: Where the prediction lands

**Decision Plot:**
- **Shows decision path**: From base value to final prediction
- **Feature by feature**: Y-axis = cumulative SHAP, X-axis = feature value
- **Multiple samples**: Can overlay to compare decision paths

**Local Feature Importance:**
- **Top contributing features**: For this specific prediction
- **Ranked list with values**: Feature name, value, SHAP contribution

**Section 3: What-If Analysis**

**Interactive Scenario Builder:**
- **Create custom inputs**: Enter feature values manually
- **Dropdowns for categoricals**: Select from valid options (shows original, not encoded)
- **Sliders for numerical**: Min-max ranges with current value
- **Automatic validation**: Prevents invalid inputs

**Real-Time Prediction:**
- **Instant results**: Updates as you change inputs
- **Prediction value**: Shows outcome (class or regression value)
- **Confidence/probability**: For classification, shows probability distribution

**Feature Contribution Breakdown:**
- **SHAP waterfall**: Shows why this particular input gives this prediction
- **Sensitivity analysis**: Which features matter most for this scenario

**Scenario Comparison:**
- **Save scenarios**: Store multiple what-if configurations
- **Side-by-side comparison**: See how changing one feature affects prediction
- **Great for stakeholder questions**: "What if we increase feature X by 20%?"

**Section 4: Fairness Analysis**

**Protected Attribute Selection:**
- **Automatic detection**: Identifies potential protected attributes (gender, race, age)
- **Manual override**: Select any feature for fairness analysis
- **Multiple attributes**: Analyze several at once

**Group-Wise Performance Metrics:**
- **For classification**:
  - Accuracy, Precision, Recall, F1 per group
  - Confusion matrix per group
  - ROC curves per group

- **For regression**:
  - RMSE, MAE, R² per group
  - Residuals distribution per group
  - Mean error per group

**Fairness Metrics:**
- **Demographic Parity**: Are positive predictions equally distributed?
- **Equal Opportunity**: Do all groups have equal TPR (true positive rate)?
- **Equalized Odds**: Are TPR and FPR equal across groups?
- **Predictive Parity**: Is precision equal across groups?
- **Disparate Impact**: Ratio of positive rates (should be close to 1.0)

**Visual Fairness Analysis:**
- **Performance gap charts**: Bar charts comparing metrics across groups
- **Disparity heatmaps**: Color-coded fairness violations
- **Distribution plots**: Show prediction distributions per group

**Bias Identification:**
- **Flags significant disparities**: Highlights metrics that differ >10% across groups
- **Recommends actions**:
  - Re-weighting classes
  - Threshold optimization per group
  - Feature engineering to remove proxies
  - Collecting more data for underrepresented groups

**Fairness Report:**
- **Summary of findings**: Written narrative
- **Severity assessment**: Low, moderate, high bias
- **Mitigation strategies**: Actionable recommendations
- **Downloadable**: Export fairness report as PDF/text

**Section 5: Limitations & Recommendations (via Model Explanation)**

**Model Limitations:**
- **Scope limitations**: What the model can and cannot predict
- **Data limitations**: Training data biases, missing scenarios
- **Performance caveats**: Where the model struggles (e.g., rare classes, extreme values)
- **Uncertainty**: Confidence in different prediction ranges

**Improvement Recommendations:**
- **Data collection**: Which features or samples would help most
- **Feature engineering**: Suggested new features or transformations
- **Model architecture**: Whether a different algorithm might help
- **Hyperparameter tuning**: If further optimization is possible
- **Ensemble methods**: Combining multiple models

**Deployment Considerations:**
- **Monitoring recommendations**: Which metrics to track in production
- **Retraining triggers**: When to retrain (data drift, performance degradation)
- **Ethical guidelines**: How to use model responsibly

**Proceed to Summary:**
Once you've thoroughly understood your model, click "Proceed to Summary" to export artifacts and generate reproduction scripts.

**Tips:**
- Start with global feature importance to get the big picture
- Use individual explanations to verify model makes sense for specific cases
- What-if analysis is powerful for stakeholder buy-in
- Always run fairness analysis for models affecting people
- If a feature has high importance but low domain relevance, investigate for data leakage
- SHAP values are additive—contributions sum to final prediction
- Positive SHAP = pushes prediction higher; negative = pushes lower
- For proxies (e.g., ZIP code for race), consider fairness implications carefully

### 9) Summary and Export

**Goal:** Package all results, export artifacts, and ensure full reproducibility of your ML pipeline.

**What the application does:**

**Overall Progress Visualization:**
- **Stage completion progress bar**: Visual indicator showing 9/9 stages completed
- **Stage-by-stage status**: Checkmarks for completed stages, pending for incomplete
- **Journey overview**: High-level summary of decisions made

**Model Performance Summary:**
- **Metrics visualization**: Bar chart of key performance metrics
  - For classification: Accuracy, Precision, Recall, F1, AUC
  - For regression: RMSE, MAE, R², Adjusted R²
- **Training information recap**:
  - Model type
  - Optimization method
  - Training duration
  - Dataset sizes
  - Number of features

**Journey Log Viewer:**
- **Complete audit trail**: Every decision made throughout the 9 stages
- **Hierarchical tree view**: Shows parent-child relationships between decisions
- **Searchable and filterable**: Find specific actions or stages
- **Timestamps**: Tracks when each action occurred
- **Details on demand**: Expand any node to see full context
- **Export options**: Download as JSON or formatted text

**Comprehensive Export Options:**

**1. Trained Model**
- **Format**: Pickle file (.pkl)
- **Contents**: Trained model with all fitted parameters
- **Calibration included**: If model was calibrated
- **Threshold included**: If threshold was optimized
- **Ready for deployment**: Can be loaded with `joblib.load()` or `pickle.load()`
- **File naming**: `model_{model_type}_{timestamp}.pkl`

**2. Preprocessing Artifacts**
- **Encoders**: All categorical encoders (one-hot, label, target, etc.)
- **Scalers/Transformers**: If any normalization was applied (rare in this app)
- **Imputers**: Missing value imputation strategies
- **Binning definitions**: Bin edges and labels for binned features
- **Format**: Pickle file with dictionary of all transformers
- **Usage**: Apply same transformations to new data for predictions

**3. Training and Test Datasets**
- **X_train.csv**: Training features (post-preprocessing)
- **y_train.csv**: Training target
- **X_test.csv**: Test features
- **y_test.csv**: Test target
- **Ready to use**: Can be loaded directly for model retraining or validation

**4. Feature Lists**
- **selected_features.csv**: Final features used in model
- **Includes**: Feature names, data types, importance scores
- **Useful for**: Feature documentation, model cards, compliance

**5. Model Recreation Script**
- **Full Python script**: Complete code to recreate model from scratch
- **Includes**:
  - All imports (scikit-learn, xgboost, etc.)
  - Data loading (assumes downloaded datasets)
  - Exact train-test split (with random state)
  - Model initialization with exact hyperparameters
  - Training code
  - Calibration code (if applied)
  - Threshold optimization code (if applied)
  - Evaluation metrics calculation
  - Prediction examples

- **Hyperparameter precision**: Uses exact parameters from training
- **Reproducibility**: Setting random states ensures identical results
- **Ready to run**: Copy-paste and execute
- **Comments**: Explains each step

**6. Environment Configuration**
- **requirements.txt**: All Python packages and versions
- **Package versions**: Exact versions used (e.g., `scikit-learn==1.3.0`)
- **Usage**: `pip install -r requirements.txt` to recreate environment

**7. Model Card (Comprehensive Documentation)**
- **Model details**: Algorithm, hyperparameters, training method
- **Intended use**: Problem type, target variable, decision context
- **Performance metrics**: All evaluation metrics with explanations
- **Training data**: Description, size, date range, source
- **Ethical considerations**: Fairness analysis results, bias assessment
- **Limitations**: Known constraints and failure modes
- **Monitoring recommendations**: What to track in production
- **Maintenance**: Retraining schedule and triggers
- **Format**: Markdown or PDF

**8. Encoding and Transformation Mappings**
- **Detailed mappings**: Original → encoded values for all categorical features
- **Target encoding**: If target was categorical, shows class labels → integers
- **Binning details**: For binned features, shows ranges and labels
- **Usage**: Essential for interpreting predictions on original scale
- **Format**: JSON or CSV

**9. Journey Log Export**
- **Complete decision history**: All actions taken in all 9 stages
- **Structured format**: JSON with hierarchical relationships
- **Includes**:
  - Stage transitions
  - Feature selections
  - Preprocessing decisions
  - Model selection rationale
  - Training configuration
  - Evaluation insights
  - Explanation findings
- **Compliance-ready**: Audit trail for regulated industries

**10. SHAP Explanation Plots**
- **All explanation visualizations**: SHAP summary, dependence, waterfall plots
- **Format**: PNG or interactive HTML
- **Usage**: Include in reports, presentations, documentation

**11. Performance Visualizations**
- **All evaluation plots**: Confusion matrix, ROC curve, residuals, etc.
- **Format**: PNG or interactive HTML (Plotly)
- **High resolution**: Suitable for publication or reports

**Bulk Download:**
- **"Download All Artifacts" button**: Packages everything into a ZIP file
- **Organized structure**: Folders for models, data, scripts, docs, plots
- **README included**: Explains what each file is and how to use it

**Model Deployment Readiness Checklist:**
Automatically generated checklist confirming:
- ✓ Model trained and validated
- ✓ Performance meets requirements (user confirms)
- ✓ Fairness analysis completed
- ✓ Artifacts exported
- ✓ Reproduction script generated
- ✓ Documentation complete
- ✓ Environment requirements specified

**Next Steps Guidance:**
- **Deployment recommendations**: How to integrate model into production
- **Monitoring setup**: Which metrics to track, how often
- **Retraining triggers**: When to update the model (e.g., every 3 months, or when performance drops >5%)
- **Stakeholder communication**: How to present model to non-technical audiences

**Start New Project:**
- **Reset button**: Clears all session state and starts fresh workflow
- **Confirmation required**: Prevents accidental loss of work

**Tips:**
- Always download the reproduction script—it's your insurance policy
- Store the journey log for compliance and audit purposes
- The model card is essential for stakeholder communication
- Export the encoding mappings—you'll need them to interpret production predictions
- Keep requirements.txt with the model to ensure environment compatibility
- The full artifact ZIP is the complete package—store it in version control (Git LFS for large files)
- Reproduction scripts include random states, so results should be identical
- Include the SHAP plots in model documentation for explainability


## Responsible AI and Fairness

- Protected attributes are surfaced during feature selection and explanation.
- Fairness metrics are provided to compare performance across demographic groups.
- Use these tools to avoid introducing or amplifying bias and to document decisions.


## Privacy and Security

- Files are processed locally; no external transmission by default.
- Uploads are validated (MIME type and size limits) to improve safety.
- Only use data that you are authorized to process.


## Best Practices and Workflow Tips

### General Workflow Strategy
- **Start simple, iterate**: Begin with a baseline model (e.g., Logistic Regression, Decision Tree) before trying complex algorithms
- **Use sample datasets first**: Familiarize yourself with the workflow using the built-in Titanic or Miami Housing datasets
- **Don't skip stages**: The 9-stage sequence is designed to surface issues early; skipping leads to problems later
- **Document as you go**: The journey log captures decisions automatically, but add notes about your reasoning

### Data Loading and Exploration
- **Check target balance immediately**: Class imbalance affects everything downstream
- **Spend time in Data Exploration**: Understanding your data is more valuable than trying 10 different models
- **Use correlation analysis**: Identify redundant features before preprocessing to save time
- **Run the automated preprocessing option** if you want a quick baseline, then iterate with manual preprocessing for better control

### Preprocessing Strategy
- **Handle zeros before missing values**: Zero values might represent missing data
- **Be conservative with feature removal**: It's easier to remove features later in Feature Selection than to go back
- **Use optimal binning for non-linear relationships**: It often outperforms manual binning
- **Cap outliers rather than remove**: Preserves sample size while reducing extreme value impact
- **One-hot encode for tree models, target encode for boosting**: Different models benefit from different encodings

### Feature Selection
- **Trust the importance scores, but apply domain knowledge**: Low-importance features might have business value
- **Remove highly correlated pairs**: Keep the one with higher target correlation
- **Don't remove protected attributes without careful consideration**: Document your reasoning for fairness compliance
- **Use the correlation groups analysis**: It's more sophisticated than simple pairwise correlation

### Model Selection and Training
- **Trust the recommendation system**: It considers many factors you might miss
- **Start with Standard Training**: Get a baseline before spending time on hyperparameter tuning
- **Use Optuna for best results**: It's more efficient than Random Search
- **Always handle class imbalance**: Ignoring it leads to models that predict only the majority class
- **Calibrate if using probabilities**: Raw classifier probabilities are often poorly calibrated
- **Optimize thresholds for imbalanced binary classification**: The default 0.5 is rarely optimal

### Evaluation and Explanation
- **Don't rely on a single metric**: Use the full suite (precision, recall, F1 for classification; RMSE, MAE, R² for regression)
- **Examine sample predictions**: They reveal issues that aggregate metrics hide
- **Use SHAP for all stakeholder communication**: Waterfall plots are intuitive for non-technical audiences
- **Always run fairness analysis for human-impacting decisions**: Regulatory requirements are increasing
- **Generate what-if scenarios for key stakeholders**: Answers "what if" questions before they're asked

### Export and Reproducibility
- **Always download the reproduction script**: It's your documentation and insurance policy
- **Export the journey log**: Essential for compliance and troubleshooting
- **Store encoding mappings with the model**: You'll need them to interpret production predictions
- **Use version control for artifacts**: Git for scripts and logs, Git LFS for model files
- **Test the reproduction script**: Verify it runs and produces the same results before deploying


## Troubleshooting Common Issues

### Navigation and Stage Progression

**Problem**: Can't proceed to the next stage / "Proceed" button is disabled
- **Cause**: Current stage requirements not met
- **Solutions**:
  - Check for warning messages at the top of the page
  - For Preprocessing: Ensure all missing values handled and all categorical features encoded
  - For Feature Selection: Confirm feature analysis has run successfully
  - For Model Selection: Verify a model is selected
  - For Model Training: Complete training and select a model (if using hyperparameter tuning)
  - Look for red error messages indicating specific issues

**Problem**: Lost progress after closing browser
- **Cause**: Streamlit session state is browser-specific and temporary
- **Solutions**:
  - Streamlit sessions persist only while browser tab is open
  - Export artifacts frequently (especially after training)
  - Download the journey log to recover decision history
  - Use the reproduction script to quickly rebuild

### Data Loading Issues

**Problem**: File upload fails or validation error
- **Possible causes and solutions**:
  - **File too large (>100MB)**: Sample your data or split into chunks
  - **Invalid MIME type**: Ensure file is actually CSV (not Excel saved as CSV)
  - **Malformed CSV**: Check for inconsistent column counts, unescaped quotes
  - **Special characters in column names**: Avoid =, +, -, @, | in column names
  - **Too many columns (>1000)**: Consider dimensionality reduction before upload

**Problem**: Target variable shows as invalid
- **Possible causes and solutions**:
  - **All target values missing**: Check data quality; this target can't be used
  - **Too many unique values for classification (>10)**: Consider regression or reduce categories
  - **Numeric target with string values**: Clean data to ensure type consistency
  - **Mixed data types in target**: Standardize to one type (all numeric or all categorical)

### Data Exploration and Preprocessing Issues

**Problem**: Missing values analysis shows unexpected results
- **Possible causes and solutions**:
  - **Zeros showing as missing**: Use Zero Values Analysis step to flag these
  - **String "NA", "null", "None" not recognized**: These are valid strings, not missing; convert to NaN first
  - **Consistent missing patterns**: May indicate data collection issues

**Problem**: Categorical encoding fails
- **Possible causes and solutions**:
  - **Too many categories**: Use target encoding or frequency encoding instead of one-hot
  - **Unseen categories in test set**: This shouldn't happen after train-test split, but check split was done correctly
  - **Memory error with one-hot**: Too many features created; use different encoding

**Problem**: Train-test split shows imbalanced classes in one set
- **Cause**: Small dataset or rare classes
- **Solutions**:
  - Enable stratification (should be automatic for classification)
  - Adjust split ratio (e.g., 70/30 instead of 80/20) for very small datasets
  - Consider if you have enough data for reliable modeling (need at least 50+ samples per class)

### Model Training Issues

**Problem**: Model training fails with error message
- **Possible causes and solutions**:
  - **XGBoost multiclass error**: Target classes must start from 0; re-encode target
  - **Memory error**: Dataset too large; try sampling or simpler model
  - **NaN/Inf in features**: Missing values weren't handled properly; return to preprocessing
  - **All features same value**: Near-zero variance features; remove in Feature Selection
  - **Incompatible feature types**: Some models require all-numeric; check encoding

**Problem**: Training takes too long (>10 minutes for Optuna)
- **Solutions**:
  - Reduce number of trials (use 20-50 instead of 100+)
  - Reduce CV folds (use 3 instead of 5)
  - Enable Optuna pruning (should be default)
  - Use Random Search instead of Optuna
  - Sample your data (use 10,000 rows max for hyperparameter tuning)
  - Choose faster models (avoid MLP, use Hist Gradient Boosting instead)

**Problem**: Model achieves perfect or near-perfect training accuracy but poor test accuracy
- **Cause**: Overfitting or data leakage
- **Solutions**:
  - Check for leakage: features that directly reveal the target (e.g., "approved" column predicting approval)
  - Reduce model complexity: simpler model, fewer features, stronger regularization
  - Get more training data if possible
  - Use cross-validation to detect overfitting early
  - Check feature importance: if one feature dominates (>90%), investigate for leakage

**Problem**: Model performance is poor (accuracy <60% for balanced classification, R² <0.3 for regression)
- **Possible causes and solutions**:
  - **Insufficient features**: Return to Data Exploration, engineer new features
  - **Model too simple**: Try more complex models (Random Forest, XGBoost)
  - **Class imbalance not addressed**: Use SMOTE or class weights
  - **Poor feature quality**: Too many missing values, low variance features
  - **Inherently difficult problem**: Some targets are just hard to predict; consider if problem is solvable
  - **Need more data**: Small datasets (<500 samples) often perform poorly

**Problem**: Cross-validation scores have high variance (std > 0.1)
- **Cause**: Model instability or small dataset
- **Solutions**:
  - Increase regularization
  - Use more stable model (Random Forest instead of Decision Tree)
  - Get more data if possible
  - Check for outliers or anomalies causing fold-to-fold differences

### Model Explanation Issues

**Problem**: SHAP calculation takes too long or crashes
- **Possible causes and solutions**:
  - **Too many samples**: SHAP uses only 100 samples by default; if still slow, reduce further
  - **Too many features (>50)**: SHAP slows with high dimensionality; consider feature selection
  - **Complex model (MLP with many layers)**: Use KernelExplainer with fewer background samples
  - **Memory error**: Reduce number of samples used for SHAP calculation

**Problem**: SHAP values seem counterintuitive
- **Cause**: Feature interactions or encoding issues
- **Solutions**:
  - Check encoded feature mappings: are you interpreting the right way?
  - Examine dependence plots: might reveal non-linear or threshold effects
  - Look for feature interactions: use SHAP interaction plots
  - Verify model is actually performing well: poor models have meaningless explanations

### Performance and Session Issues

**Problem**: Application is slow or unresponsive
- **Possible causes and solutions**:
  - **Large dataset (>100k rows)**: Consider sampling for exploration/training
  - **Many features (>100)**: Reduce via feature selection
  - **Memory buildup**: Restart the session (refresh browser)
  - **Complex visualizations**: Plotly with many points can be slow
  - **Optuna with many trials**: Progress bar updates can lag; just wait

**Problem**: Session state seems corrupted (unexpected errors)
- **Cause**: Bug or rare edge case
- **Solutions**:
  - Use "Undo All Changes" button in Preprocessing
  - Export your current progress (journey log, datasets)
  - Refresh the browser and start from last completed stage
  - Clear browser cache if problems persist
  - Report issue with details (browser console errors helpful)

### Export and Deployment Issues

**Problem**: Downloaded model fails to load
- **Possible causes and solutions**:
  - **Version mismatch**: Ensure same package versions (use requirements.txt)
  - **Python version mismatch**: Use same major.minor version (e.g., 3.9.x)
  - **Missing dependencies**: Install all requirements
  - **Pickle security**: Use `joblib.load()` with trusted sources only

**Problem**: Reproduction script doesn't produce same results
- **Possible causes and solutions**:
  - **Different random state**: Verify script uses same random_state values
  - **Package version differences**: Check sklearn, xgboost versions match
  - **Data preprocessing order**: Ensure same sequence of operations
  - **Floating point differences**: Minor variations (<0.001) are normal across systems


## Frequently Asked Questions (FAQ)

### General Questions

**Q: What file formats are supported?**
A: CSV files only. The file must:
- Have a header row with column names
- Use standard CSV delimiters (comma, semicolon, or tab)
- Be under 100MB in size
- Have at least 2 columns
- Not have dangerous characters in column names (=, +, -, @)

**Q: Can I use Excel files?**
A: Not directly. Save your Excel file as CSV first (File → Save As → CSV format).

**Q: What types of problems does the app support?**
A: 
- **Binary classification**: 2-class problems (e.g., yes/no, approved/rejected)
- **Multiclass classification**: 3-10 class problems (e.g., low/medium/high)
- **Regression**: Continuous numeric targets (e.g., price, temperature)
- Currently does not support: time series, NLP, computer vision, clustering, anomaly detection

**Q: Do I need to know how to code?**
A: No coding required to use the application. It's entirely point-and-click. However:
- Exported reproduction scripts are Python code
- Understanding basic ML concepts helps interpret results
- The app provides explanations for all concepts

**Q: Is my data secure? Where is it sent?**
A: 
- All processing is local—your data never leaves your machine
- Files are processed in-memory only
- No data is transmitted to external servers
- Session data is browser-specific and temporary
- When you close the browser, all data is cleared

**Q: How long does a typical workflow take?**
A: Depends on dataset size and choices:
- **Small dataset (<1000 rows), Standard Training**: 10-15 minutes total
- **Medium dataset (1000-10k rows), Random Search**: 30-45 minutes
- **Large dataset (>10k rows), Optuna**: 1-2 hours
- Data Exploration and Preprocessing are usually quick (<10 minutes)
- Model Training is the longest stage (especially with hyperparameter tuning)

### Data and Preprocessing

**Q: How much data do I need?**
A: Minimum recommendations:
- **Binary/Multiclass classification**: 100+ samples per class (500+ total preferred)
- **Regression**: 200+ samples (1000+ preferred)
- **Complex models (MLP, XGBoost)**: 1000+ samples
- More data generally yields better, more stable models

**Q: What if I have missing values?**
A: The app provides comprehensive missing value handling:
- Detection and visualization in Data Exploration
- Multiple imputation strategies in Preprocessing (mean, median, mode, KNN, etc.)
- You must handle all missing values before training
- Features with >80% missing values should usually be dropped

**Q: Can I handle imbalanced classes?**
A: Yes, the app provides multiple strategies:
- Automatic detection during Model Training
- SMOTE (Synthetic Minority Over-sampling)
- Random over/undersampling
- Class weights (automatic rebalancing)
- ADASYN (Adaptive Synthetic Sampling)
- Threshold optimization for binary classification

**Q: What if my categorical feature has 50+ categories?**
A: High-cardinality categoricals require special handling:
- Avoid one-hot encoding (creates too many features)
- Use target encoding, frequency encoding, or binary encoding
- Consider grouping rare categories into "Other"
- CatBoost handles high-cardinality categoricals natively

### Model Training and Selection

**Q: Which model should I choose?**
A: The app recommends a model based on your data, but general guidelines:
- **Interpretability priority**: Logistic/Linear Regression, Decision Tree
- **Performance priority**: XGBoost, LightGBM, CatBoost, Random Forest
- **Small datasets**: Logistic Regression, Naive Bayes (avoid complex models)
- **Large datasets**: Gradient boosting methods
- **Many categorical features**: CatBoost, tree-based models
- **Tabular data (general)**: Gradient boosting usually best

**Q: What's the difference between Standard Training, Random Search, and Optuna?**
A:
- **Standard Training**: Uses default hyperparameters; fast baseline
- **Random Search**: Randomly tries different hyperparameter combinations; good balance of speed and performance
- **Optuna**: Bayesian optimization intelligently searches hyperparameter space; best performance but slowest
- Recommendation: Start with Standard, use Optuna for final model

**Q: Can I use my own hyperparameters?**
A: Partially. The app uses predefined search spaces for tuning. For full control:
- Export the reproduction script
- Modify hyperparameters in the script
- Run training externally

**Q: Why is XGBoost grayed out?**
A: XGBoost requires multiclass targets to have classes starting from 0 (i.e., 0, 1, 2, ...). If your classes are 1, 2, 3, you'll see a compatibility warning. Solutions:
- Let the app auto-encode your target (it should do this automatically)
- Or use a different model (LightGBM, CatBoost have no such restriction)

**Q: What if training takes too long?**
A: Several options to speed up:
- Use Standard Training instead of tuning
- Reduce hyperparameter search: fewer iterations, fewer CV folds
- Sample your data (use 10,000 rows for tuning)
- Choose faster models: Hist Gradient Boosting instead of MLP
- Enable Optuna pruning (default) to stop poor trials early

### Evaluation and Explanation

**Q: What's a good accuracy/R²/RMSE?**
A: It depends entirely on the problem:
- **Classification accuracy**: >80% is good for balanced classes, but meaningless for imbalanced; use F1 instead
- **R² (regression)**: >0.7 is good, >0.9 is excellent (but check for overfitting)
- **RMSE (regression)**: Compare to target variable's range; RMSE = 10% of range is reasonable
- Always compare to baseline (random predictions or mean prediction)
- Domain matters: 95% accuracy might be poor for some problems, excellent for others

**Q: What are SHAP values?**
A: SHAP (SHapley Additive exPlanations) values explain predictions:
- Show how much each feature contributed to a prediction
- Positive SHAP = pushes prediction higher; negative = pushes lower
- Derived from game theory (fair credit allocation)
- Model-agnostic (works with any model)
- Additive: all SHAP values sum to the final prediction
- More reliable than feature importance for understanding individual predictions

**Q: Should I use global or local explanations?**
A: Use both:
- **Global (feature importance, summary plots)**: Understand overall model behavior
- **Local (waterfall, force plots)**: Explain specific predictions to stakeholders
- Global for debugging and model improvement
- Local for regulatory compliance and stakeholder communication

**Q: What if my model shows bias in fairness analysis?**
A: The app flags disparities and suggests actions:
- **Re-weight classes**: Give more weight to underrepresented groups
- **Threshold optimization per group**: Different thresholds for different groups (if legally allowed)
- **Feature engineering**: Remove proxy variables
- **Collect more data**: For underrepresented groups
- **Different model**: Some algorithms are more inherently fair
- **Post-processing**: Adjust predictions to ensure fairness constraints
- Important: Always consult legal/ethical guidelines for your domain

### Export and Reproducibility

**Q: Can I export everything I've built?**
A: Yes, the Summary stage exports:
- Trained model (.pkl file)
- Training and test datasets (CSV)
- Preprocessing artifacts (encoders, imputers)
- Feature list
- Full reproduction script (Python code)
- Journey log (audit trail)
- Model card (documentation)
- All visualizations (plots)
- SHAP explanations
- Use "Download All Artifacts" for a complete ZIP

**Q: How do I use the trained model for new predictions?**
A: 
1. Load the model: `model = joblib.load('model.pkl')`
2. Load preprocessing artifacts: `encoders = joblib.load('preprocessing.pkl')`
3. Prepare new data with same preprocessing (use encoding mappings)
4. Predict: `predictions = model.predict(new_data)`
5. Or use the reproduction script as a template

**Q: Can I retrain the model later with new data?**
A: Yes:
- Use the reproduction script as a starting point
- Replace the data loading section with your new data
- Ensure new data has same features and preprocessing
- Run the script—it will retrain with exact same configuration
- Random states ensure reproducibility

**Q: What if I want to use the model in a different programming language (R, Java, etc.)?**
A: 
- Export options are Python-specific
- For other languages:
  - Export model as ONNX (not built-in; requires external conversion)
  - Or reimplement the model in target language using the model card and hyperparameters
  - Or create a Python API wrapper (e.g., Flask) and call from other languages

**Q: How do I share my model with non-technical stakeholders?**
A: Several options:
- **Model Card**: Comprehensive documentation in plain language
- **SHAP waterfall plots**: Visual explanations of predictions
- **Journey log**: Shows all decisions made (good for transparency)
- **What-if scenarios**: Let stakeholders explore predictions interactively (export scenarios)
- **Performance visualizations**: Confusion matrix, ROC curves (export as PNG)

### Advanced and Technical Questions

**Q: Does the app perform feature scaling/normalization?**
A: No, intentionally excluded to maintain interpretability. Most included models (tree-based) don't require scaling. If you need scaling:
- Export reproduction script
- Add StandardScaler or MinMaxScaler before training
- Or use the Data Types Optimization (handles some scaling implicitly)

**Q: Can I use ensemble methods (stacking, voting)?**
A: Not directly in the UI. To create ensembles:
- Train multiple models through the app (different algorithms)
- Export each trained model
- Combine externally using scikit-learn VotingClassifier/Regressor or StackingClassifier/Regressor

**Q: Can I do feature engineering beyond what's offered?**
A: Limited to provided operations (arithmetic, transformations). For advanced feature engineering:
- Do it before uploading (in Excel, Python, etc.)
- Or export reproduction script and add custom feature engineering

**Q: Does the app support GPUs for training?**
A: Some models can use GPU if available:
- XGBoost: Uses GPU if installed with GPU support
- LightGBM: Can use GPU
- CatBoost: Can use GPU
- Most others: CPU only
- App doesn't explicitly configure GPU usage; relies on library defaults

**Q: Can I use this app for production systems?**
A: The app is designed for model development and experimentation, not production deployment. For production:
- Export trained model and artifacts
- Set up proper ML pipeline (MLflow, Kubeflow, etc.)
- Add monitoring, logging, retraining pipelines
- Implement API for serving predictions (Flask, FastAPI)
- Consider containerization (Docker)
- The reproduction script is a good starting point for production code

**Q: How can I contribute or report issues?**
A: Contact the development team or file issues in the GitHub repository (if public).


## Getting Help

- Review on‑screen guidance for each stage.
- Use the journey log to trace what happened.
- File issues or ask questions in your team’s preferred channel.
