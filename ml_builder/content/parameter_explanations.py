"""
Parameter explanations content for ML Builder.

Contains detailed explanations for machine learning model parameters,
cross-validation concepts, and hyperparameter search strategies.
"""

from typing import Dict

# Model parameter explanations dictionary
PARAMETER_EXPLANATIONS = {
    "ridge_regression": """
        For Ridge Regression, we'll tune these key parameters:

        - **alpha**: Regularization strength (L2 penalty)
            - *What it does*: Controls how much to penalize large coefficients to prevent overfitting
            - *Range we'll test*: 0.1 to 100 for normal data, 0.01 to 10 for high-dimensional data (log scale)
            - *Effect*:
                - Smaller values: Less regularization, model fits training data more closely (may overfit)
                - Larger values: More regularization, coefficients shrink toward zero (prevents overfitting)
            - *Why it matters*: Balances between fitting the training data and generalizing to new data

        - **max_iter**: Maximum number of iterations for optimization
            - *What it does*: Controls how long the solver runs to find optimal coefficients
            - *Range we'll test*: 100 to 1000 iterations
            - *Effect*: More iterations allow better convergence but take longer

        **Note**: Ridge Regression is particularly effective when features are correlated (multicollinearity). Unlike Lasso,
        it shrinks coefficients but keeps all features in the model.
    """,
    "naive_bayes": """
        For Naive Bayes (GaussianNB), we'll tune this key parameter:

        - **var_smoothing**: Portion of the largest variance of all features added to variances for stability
            - *What it does*: Prevents zero probabilities and adds numerical stability
            - *Range we'll test*: 1e-11 to 1e-7 (log scale)
            - *Effect*:
                - Smaller values: Trusts the training data more, may overfit
                - Larger values: More smoothing, more conservative predictions
            - *Why it matters*: Balances between model flexibility and stability

        **Note**: Naive Bayes is extremely fast and has minimal tuning requirements. It assumes feature independence,
        which makes it simple but potentially less accurate than more complex models for correlated features.
    """,
    "decision_tree": """
        For Decision Tree, we'll tune these key parameters:

        - **max_depth**: Maximum depth of each tree
            - *What it does*: Controls how detailed each tree's decisions can be
            - *Range we'll test*: 3 to 20 levels deep
            - *Adaptive behavior*: Shallower trees for high-dimensional data to prevent overfitting

        - **min_samples_split**: Minimum samples needed to split a node
            - *What it does*: Controls when a tree should stop splitting
            - *Range we'll test*: 2 to 50 samples
            - *Adaptive behavior*: Higher values for small datasets to ensure reliable splits

        - **min_samples_leaf**: Minimum samples required at leaf nodes
            - *What it does*: Controls the minimum size of leaf nodes
            - *Range we'll test*: 1 to 20 samples
            - *Adaptive behavior*: Higher values for small datasets to prevent overfitting

        - **max_features**: Maximum number of features to consider for splitting
            - *What it does*: Controls the number of features to consider for splitting
            - *Options we'll test*: 'sqrt', 'log2', None
            - *Adaptive behavior*: More restricted feature selection for high-dimensional data

        - **min_impurity_decrease**: **NEW** - Minimum impurity decrease required for split
            - *What it does*: A node will be split if it induces a decrease in impurity greater than or equal to this value
            - *Range we'll test*: 0.0 to 0.1
            - *Effect*:
                - 0.0: No constraint (default behavior)
                - Higher values: Only allows splits that significantly improve the model
            - *Why it matters*: Helps prevent overfitting by avoiding insignificant splits
            - *Benefit*: Creates simpler, more generalizable trees

        - **ccp_alpha**: **NEW** - Cost-complexity pruning parameter
            - *What it does*: Complexity parameter used for Minimal Cost-Complexity Pruning
            - *Range we'll test*: 0.0 to 0.05
            - *Effect*:
                - 0.0: No pruning (full tree)
                - Higher values: More aggressive pruning, simpler trees
            - *Why it matters*: Post-pruning technique that removes parts of the tree that don't improve performance
            - *Benefit*: Often produces better generalization than pre-pruning alone

        - **splitter**: **NEW** - Strategy to choose the split at each node
            - *What it does*: Controls how splits are chosen
            - *Options we'll test*:
                - 'best': Choose the best split (default, most accurate)
                - 'random': Choose random split (faster, adds randomness)
            - *Why it matters*: Random splitting can help with overfitting and is faster
            - *Benefit*: Useful for large datasets or when combined with other trees

        - **criterion**: The function to measure the quality of a split
            - *What it does*: Determines how the model measures and minimises prediction errors
            - *For Classification*:
                - 'gini': Measures impurity, faster to compute
                - 'entropy': Measures information gain, slightly more balanced
            - *For Regression*:
                - 'squared_error': Standard loss function that minimises mean squared error
                - 'absolute_error': Minimises the absolute difference between predicted and actual values
                - 'friedman_mse': Uses mean squared error with Friedman's improvement score
    """,
    "random_forest": """
        For Random Forest, we'll tune these key parameters:

        - **n_estimators**: Number of trees in the forest
            - *What it does*: More trees = more robust predictions, but slower training
            - *Range we'll test*: 50 to 200 trees for small datasets, 50 to 300 for larger datasets
            - *Adaptive behavior*: Fewer trees for small datasets to prevent overfitting

        - **max_depth**: Maximum depth of each tree
            - *What it does*: Controls how detailed each tree's decisions can be
            - *Range we'll test*: 3 to 10 levels for high-dimensional data, 3 to 15 for normal data
            - *Adaptive behavior*: Shallower trees for high-dimensional data

        - **min_samples_split**: Minimum samples needed to split a node
            - *What it does*: Controls when a tree should stop splitting (prevents overfitting)
            - *Range we'll test*: 2 to 20 samples
            - *Adaptive behavior*: Higher values for small datasets

        - **min_samples_leaf**: Minimum samples required at leaf nodes
            - *What it does*: Controls the minimum size of leaf nodes (prevents overfitting)
            - *Range we'll test*: 1 to 10 samples
            - *Adaptive behavior*: Higher values for small datasets

        - **max_features**: Maximum number of features to consider for splitting
            - *What it does*: Controls feature randomness when building trees
            - *Options we'll test*: 'sqrt', 'log2' for high-dimensional data; 'sqrt', 'log2', None for normal data
            - *Adaptive behavior*: More restricted feature selection for high-dimensional data

        - **bootstrap**: Whether to use bootstrapping
            - *What it does*: Controls if samples are drawn with replacement
            - *Options we'll test*: True, False
            - *Adaptive behavior*: Helps prevent overfitting, especially in small datasets

        - **min_impurity_decrease**: **NEW** - Minimum impurity decrease required for split
            - *What it does*: A node will be split only if it induces a decrease in impurity greater than this value
            - *Range we'll test*: 0.0 to 0.1
            - *Effect*:
                - 0.0: No constraint (default behavior)
                - Higher values: Only allows significant splits
            - *Why it matters*: Helps prevent overfitting by avoiding weak splits
            - *Benefit*: Creates more generalized trees in the forest

        - **ccp_alpha**: **NEW** - Cost-complexity pruning parameter
            - *What it does*: Complexity parameter for pruning each tree in the forest
            - *Range we'll test*: 0.0 to 0.05
            - *Effect*:
                - 0.0: No pruning (full trees)
                - Higher values: More pruning, simpler trees
            - *Why it matters*: Applies post-pruning to reduce overfitting
            - *Benefit*: Can improve generalization, especially with deep trees

        - **max_samples**: **NEW** - Maximum samples to use for each tree (when bootstrap=True)
            - *What it does*: Controls the size of the bootstrap sample for each tree
            - *Range we'll test*: 0.5 to 1.0 (50% to 100% of training data)
            - *Effect*:
                - Lower values: More diversity between trees, potentially better generalization
                - Higher values: Each tree sees more data, potentially more accurate
            - *Why it matters*: Provides another way to control the bias-variance tradeoff
            - *Benefit*: Can speed up training while maintaining or improving accuracy

        - **criterion**: The function to measure the quality of a split
            - *What it does*: Determines how the model measures and minimises prediction errors
            - *For Classification*:
                - 'gini': Measures impurity (default, faster)
                - 'entropy': Measures information gain (slightly different split choices)
            - *For Regression*:
                - 'squared_error': Standard loss function that minimises mean squared error
                - 'absolute_error': Minimises the absolute difference (more robust to outliers)
                - 'friedman_mse': Uses mean squared error with Friedman's improvement score
    """,
    "gradient_boosting": """
        For Gradient Boosting, we'll tune these key parameters:

        - **n_estimators**: Number of boosting stages
            - *What it does*: More stages = more complex model, but risk of overfitting
            - *Range we'll test*: 50 to 200 stages for small datasets, 50 to 300 for larger datasets
            - *Adaptive behavior*: Fewer stages for small datasets

        - **learning_rate**: How much each tree contributes
            - *What it does*: Controls how quickly or cautiously the model learns
            - *Range we'll test*: 0.01 to 0.3
            - *Adaptive behavior*: Lower rates for small datasets

        - **max_depth**: Maximum depth of each tree
            - *What it does*: Controls complexity of each decision tree
            - *Range we'll test*: 3 to 6 for high-dimensional data, 3 to 10 for normal data
            - *Adaptive behavior*: Shallower trees for high-dimensional data

        - **subsample**: Fraction of samples to use for fitting trees
            - *What it does*: Controls the randomness in training (helps prevent overfitting)
            - *Range we'll test*: 0.6 to 1.0
            - *Adaptive behavior*: Lower values for small datasets

        - **min_samples_split**: Minimum samples needed to split a node
            - *What it does*: Controls when a tree should stop splitting
            - *Range we'll test*: 2 to 20 samples
            - *Adaptive behavior*: Higher values for small datasets

        - **min_samples_leaf**: Minimum samples required at leaf nodes
            - *What it does*: Controls the minimum size of leaf nodes
            - *Range we'll test*: 1 to 10 samples
            - *Adaptive behavior*: Higher values for small datasets

        - **If this is a regression model, we'll tune the following extra parameter:**
            - **loss**: Loss function to be optimised
                - *What it does*: Determines how the model measures and minimises prediction errors
                - *Options we'll test*:
                    - 'squared_error': Standard loss function that minimises mean squared error
                    - 'quantile': Better for skewed target distributions
                - *Adaptive behavior*: Automatically selects quantile loss for skewed targets
    """,
    "logistic_regression": """
        For Logistic Regression, we'll tune these key parameters:

        - **C**: Inverse of regularisation strength
            - *What it does*: Controls how much to penalise complex models
            - *Range we'll test*: 0.01 to 1.0 for high-dimensional data, 0.1 to 10.0 for normal data
            - *Adaptive behavior*: Stronger regularization (lower C) for high-dimensional data

        - **max_iter**: Maximum iterations to converge
            - *What it does*: How long to try finding the optimal solution
            - *Range we'll test*: 100 to 500
            - *Adaptive behavior*: Higher values for complex datasets
    """,
    "mlp": """
        For Multilayer Perceptron, we'll tune these key parameters:

        - **hidden_layer_sizes**: Architecture of the neural network
            - *What it does*: Defines the number and size of hidden layers
            - *Range we'll test*:
                - High-dimensional data:
                    - Single layer: (100,), (200,)
                    - Two layers: (100,50), (200,100)
                    - Three layers: (200,100,50)
                - Normal data:
                    - Single layer: (50,), (100,)
                    - Two layers: (50,25), (100,50)
                    - Three layers: (100,50,25)
            - *Adaptive behavior*: Larger and deeper architectures for high-dimensional data

        - **solver**: **NEW** - Optimization algorithm
            - *What it does*: Selects the algorithm used to optimize the weights
            - *Options we'll test*:
                - 'adam': Adaptive Moment Estimation (default, works well in most cases)
                - 'sgd': Stochastic Gradient Descent (better with momentum, can escape local minima)
            - *Why it matters*: Different optimizers can significantly affect convergence speed and final performance
            - *Benefit*: SGD with momentum can sometimes find better solutions than Adam

        - **learning_rate_init**: Initial learning rate
            - *What it does*: Controls how much to adjust weights in each step
            - *Range we'll test*:
                - Small datasets: 0.0001 to 0.01
                - Large datasets: 0.001 to 0.1
            - *Adaptive behavior*: Lower rates for small datasets

        - **alpha**: L2 regularisation parameter
            - *What it does*: Controls model complexity to prevent overfitting
            - *Range we'll test*:
                - Dense features: 0.0001 to 0.01
                - Sparse features: 0.001 to 0.1
            - *Adaptive behavior*: Higher values for sparse data

        - **batch_size**: Size of minibatches
            - *What it does*: Controls how many samples to use in each training step
            - *Range we'll test*:
                - Small datasets: [16, 32, 64, "auto"]
                - Large datasets: [32, 64, 128, "auto"]
            - *Adaptive behavior*: Smaller batches for small datasets

        - **activation**: Activation function
            - *What it does*: Introduces non-linearity into the network
            - *Options we'll test*: 'relu', 'tanh'

        - **learning_rate**: Learning rate schedule
            - *What it does*: Controls how learning rate changes during training
            - *Options we'll test*: 'constant', 'adaptive', 'invscaling'

        - **early_stopping**: Whether to use early stopping
            - *What it does*: Stops training when validation score stops improving
            - *Options we'll test*: True, False
            - *Adaptive behavior*: Helps prevent overfitting, especially in small datasets

        - **validation_fraction**: **NEW** - Validation set size for early stopping
            - *What it does*: Fraction of training data to set aside for early stopping validation
            - *Range we'll test*: 0.1 to 0.2 (10% to 20% of training data)
            - *Effect*: Only used when early_stopping=True
            - *Why it matters*: Ensures early stopping uses reliable validation performance
            - *Benefit*: Prevents overfitting and reduces unnecessary training time

        - **n_iter_no_change**: **NEW** - Patience for early stopping
            - *What it does*: Maximum number of epochs with no improvement before stopping
            - *Range we'll test*: 5 to 20 epochs
            - *Effect*:
                - Smaller values: Stops sooner, faster training
                - Larger values: More patient, may find better solutions
            - *Why it matters*: Balances between training time and finding optimal weights
            - *Benefit*: Can reduce training time by 20-35% while improving generalization

        - **momentum**: **NEW** - Momentum for SGD optimizer
            - *What it does*: Helps accelerate SGD in relevant direction and dampens oscillations
            - *Range we'll test*: 0.8 to 0.99
            - *Effect*: Only used when solver='sgd'
            - *Why it matters*: Momentum can significantly speed up convergence and help escape local minima
            - *Benefit*: Often essential for good performance when using SGD solver
    """,
    "linear_regression": """
        Linear Regression does not have any parameters to tune
    """,
    "xgboost": """
        For XGBoost, we'll tune these key parameters:

        - **n_estimators**: Number of boosting stages
            - *What it does*: More stages = more complex model, but risk of overfitting
            - *Range we'll test*: 100 to 300 for small datasets, 100 to 500 for larger datasets
            - *Adaptive behavior*: Fewer trees for small datasets

        - **max_depth**: Maximum depth of each tree
            - *What it does*: Controls how deep each tree can grow
            - *Range we'll test*: 3 to 6 for high-dimensional data, 3 to 10 for normal data
            - *Adaptive behavior*: Shallower trees for high-dimensional data

        - **learning_rate**: How much each tree contributes
            - *What it does*: Controls how quickly or cautiously the model learns
            - *Range we'll test*: 0.01 to 0.3
            - *Adaptive behavior*: Lower rates for small datasets

        - **subsample**: Fraction of samples used for tree building
            - *What it does*: Controls the randomness in training (helps prevent overfitting)
            - *Range we'll test*: 0.6 to 1.0
            - *Adaptive behavior*: Lower values for small datasets

        - **min_child_weight**: Minimum sum of instance weight needed in a child
            - *What it does*: Controls the minimum number of instances needed for a split
            - *Range we'll test*: 1 to 5 for high-dimensional data, 1 to 7 for normal data
            - *Adaptive behavior*: Higher values for high-dimensional data

        - **colsample_bytree**: Fraction of features used for tree construction
            - *What it does*: Controls feature randomness when building trees
            - *Range we'll test*: 0.6 to 1.0
            - *Adaptive behavior*: Lower values for high-dimensional data

        - **gamma**: Minimum loss reduction for split
            - *What it does*: Controls the minimum loss reduction needed for a split
            - *Range we'll test*: 1e-8 to 1.0 for high-dimensional data, 1e-8 to 0.5 for normal data
            - *Adaptive behavior*: Higher maximum value for high-dimensional data

        - **reg_alpha**: L1 regularisation term
            - *What it does*: Controls L1 regularisation (Lasso)
            - *Range we'll test*: 1e-8 to 1.0
            - *Adaptive behavior*: Higher values for high-dimensional data

        - **reg_lambda**: L2 regularisation term
            - *What it does*: Controls L2 regularisation (Ridge)
            - *Range we'll test*: 1e-8 to 1.0
            - *Adaptive behavior*: Higher values for high-dimensional data

        Note: The model automatically uses different evaluation metrics:
        - For classification: 'logloss' (logarithmic loss)
        - For regression: 'rmse' (root mean squared error)
    """,
    "lightgbm": """
        For LightGBM, we'll tune these key parameters:

        - **n_estimators**: Number of boosting stages
            - *What it does*: Controls the number of trees in the model
            - *Range we'll test*: 100 to 300 for small datasets, 100 to 500 for larger datasets
            - *Adaptive behavior*: Fewer trees for small datasets

        - **boosting_type**: Boosting algorithm strategy
            - *What it does*: Selects the boosting algorithm to use
            - *Options we'll test*:
                - 'gbdt': Traditional Gradient Boosting Decision Tree (default, most reliable)
                - 'dart': Dropout Additive Regression Trees (better generalization, slower)
                - 'goss': Gradient-based One-Side Sampling (faster for large datasets)
            - *Why it matters*: Different strategies can significantly impact both speed and accuracy

        - **num_leaves**: Maximum number of leaves in one tree
            - *What it does*: Controls the complexity of each tree
            - *Range we'll test*: 20 to 50 for high-dimensional data, 20 to 100 for normal data
            - *Adaptive behavior*: Fewer leaves for high-dimensional data

        - **learning_rate**: Step size shrinkage
            - *What it does*: Controls how much each tree contributes
            - *Range we'll test*: 0.01 to 0.3
            - *Adaptive behavior*: Lower rates for small datasets

        - **max_depth**: Maximum depth of each tree
            - *What it does*: Limits the maximum depth of each tree
            - *Range we'll test*: 3 to 10
            - *Adaptive behavior*: Shallower trees for high-dimensional data

        - **min_child_samples**: Minimum number of data points needed in a leaf
            - *What it does*: Controls the minimum number of samples required
            - *Range we'll test*: 10 to 50
            - *Adaptive behavior*: Higher values for small datasets

        - **subsample**: Fraction of samples used for training each tree
            - *What it does*: Controls the random sampling of data points
            - *Range we'll test*: 0.6 to 1.0
            - *Adaptive behavior*: Lower values for small datasets

        - **colsample_bytree**: Fraction of features used for training each tree
            - *What it does*: Controls the random sampling of features
            - *Range we'll test*: 0.6 to 1.0
            - *Adaptive behavior*: Lower values for high-dimensional data

        - **reg_alpha**: L1 regularisation term
            - *What it does*: Controls L1 regularisation (Lasso)
            - *Range we'll test*: 1e-8 to 1.0
            - *Adaptive behavior*: Higher values for high-dimensional data

        - **reg_lambda**: L2 regularisation term
            - *What it does*: Controls L2 regularisation (Ridge)
            - *Range we'll test*: 1e-8 to 1.0
            - *Adaptive behavior*: Higher values for high-dimensional data
    """,
    "hist_gradient_boosting": """
        For Histogram-based Gradient Boosting, we'll tune these key parameters:

        - **max_iter**: Number of boosting iterations
            - *What it does*: Controls the number of trees in the ensemble
            - *Range we'll test*: 100 to 300 for small datasets, 100 to 500 for larger datasets
            - *Adaptive behavior*: Fewer iterations for small datasets to prevent overfitting
            - *Note*: Higher maximum than before because early stopping will find the optimal point

        - **early_stopping**: **NEW** - Enable automatic early stopping
            - *What it does*: Automatically stops training when validation score stops improving
            - *Options we'll test*: True, 'auto'
            - *Effect*:
                - True: Always use early stopping with validation split
                - 'auto': Automatically determine whether to use early stopping
            - *Why it matters*: Prevents overfitting and reduces unnecessary training time
            - *Major benefit*: Can reduce training time by 25-40% while improving generalization

        - **validation_fraction**: **NEW** - Validation set size for early stopping
            - *What it does*: Fraction of training data to set aside for early stopping validation
            - *Range we'll test*: 0.1 to 0.2 (10% to 20% of training data)
            - *Effect*: Larger fractions provide better early stopping decisions but reduce training data
            - *Why it matters*: Ensures early stopping uses reliable validation performance

        - **n_iter_no_change**: **NEW** - Patience for early stopping
            - *What it does*: Number of iterations with no improvement before stopping
            - *Range we'll test*: 5 to 20 iterations
            - *Effect*:
                - Smaller values: Stops sooner, faster training
                - Larger values: More patient, may find better solutions
            - *Why it matters*: Balances between training time and finding optimal model

        - **tol**: **NEW** - Convergence tolerance
            - *What it does*: Minimum improvement required to continue training
            - *Range we'll test*: 1e-7 to 1e-3 (log scale)
            - *Effect*: Smaller values require more improvement to continue
            - *Why it matters*: Helps identify when model has converged

        - **max_depth**: Maximum depth of each tree
            - *What it does*: Controls how deep each tree can grow
            - *Range we'll test*: 3 to 10 for high-dimensional data, 3 to 15 for normal data
            - *Adaptive behavior*: Shallower trees for high-dimensional data

        - **learning_rate**: Learning rate shrinkage
            - *What it does*: Controls how much each tree contributes to the final prediction
            - *Range we'll test*: 0.01 to 0.3
            - *Adaptive behavior*: Lower rates for small datasets for better generalization

        - **l2_regularization**: L2 regularization parameter
            - *What it does*: Penalizes large leaf values to prevent overfitting
            - *Range we'll test*: 0.0 to 1.0
            - *Adaptive behavior*: Higher values for complex datasets

        - **max_bins**: Maximum number of bins for discretization
            - *What it does*: Controls the granularity of the histogram-based algorithm
            - *Range we'll test*: 128 to 255
            - *Why it matters*: More bins = more precision but slower training

        - **min_samples_leaf**: Minimum samples required in a leaf
            - *What it does*: Controls the minimum number of samples in terminal nodes
            - *Range we'll test*: 10 to 50
            - *Adaptive behavior*: Higher values for small datasets to prevent overfitting
    """,
    "catboost": """
        For CatBoost, we'll tune these key parameters:

        - **iterations**: Number of boosting iterations
            - *What it does*: Controls the number of trees in the model
            - *Range we'll test*: 100 to 300 for small datasets, 100 to 500 for larger datasets
            - *Adaptive behavior*: Fewer iterations for small datasets

        - **depth**: Depth of the trees
            - *What it does*: Controls the maximum depth of each tree
            - *Range we'll test*: 4 to 6 for high-dimensional data, 4 to 10 for normal data
            - *Adaptive behavior*: Shallower trees for high-dimensional data

        - **learning_rate**: Learning rate
            - *What it does*: Controls the step size at each boosting iteration
            - *Range we'll test*: 0.01 to 0.3
            - *Adaptive behavior*: Lower rates for small datasets

        - **l2_leaf_reg**: L2 regularization coefficient
            - *What it does*: Controls L2 regularization on leaf weights
            - *Range we'll test*: 1.0 to 10.0
            - *Adaptive behavior*: Higher values for high-dimensional data

        - **border_count**: Number of splits for numerical features
            - *What it does*: Controls the number of splits for continuous features
            - *Range we'll test*: 32 to 255
            - *Why it matters*: More splits = better precision but slower training

        - **bagging_temperature**: Bagging temperature
            - *What it does*: Controls the randomness in bagging (0 = no bagging, 1 = full bagging)
            - *Range we'll test*: 0.0 to 1.0
            - *Why it matters*: Higher values add more randomness to prevent overfitting

        - **random_strength**: Random strength for scoring splits
            - *What it does*: Amount of randomness to use for scoring splits
            - *Range we'll test*: 0.0 to 10.0
            - *Why it matters*: Higher values make the model more robust but less precise
    """
}

# Cross-validation explanation
CV_EXPLANATION = """
### What is Cross-Validation?
Cross-validation is like giving your model multiple practice tests before the final exam! Here's how it works:

1. **Split Your Data**: Your training data is divided into several equal parts (called "folds")
2. **Train and Test Multiple Times**: The model:
   - Trains on most of the folds
   - Tests itself on the remaining fold
   - Repeats this process, using different folds for testing each time

### Why is it Important?
- **Better Assessment**: Instead of testing your model just once, you test it multiple times
- **Reliability**: Helps ensure your model performs consistently across different parts of your data
- **Confidence**: Gives you more confidence that your model will work well on new, unseen data

### Choosing the Number of Folds
- **Lower (2-3)**: Faster training, but less reliable assessment
- **Medium (5)**: Good balance of reliability and training time ✨ (Recommended for most cases)
- **Higher (10)**: More reliable assessment, but takes longer to train
"""

# Cross-validation visual example
CV_VISUAL_EXAMPLE = """
```
Fold 1: [🔵Test][✨Train][✨Train][✨Train][✨Train]
Fold 2: [✨Train][🔵Test][✨Train][✨Train][✨Train]
Fold 3: [✨Train][✨Train][🔵Test][✨Train][✨Train]
Fold 4: [✨Train][✨Train][✨Train][🔵Test][✨Train]
Fold 5: [✨Train][✨Train][✨Train][✨Train][🔵Test]
```
"""

# Parameter search explanation
SEARCH_EXPLANATION = """
### Understanding Parameter Search

Parameter search is the process of finding the best combination of settings for your machine learning model. It's like tuning a radio to find the clearest station!

#### Two Main Approaches:

**1. Random Search** 🎲
- Randomly samples parameter combinations within the ranges you specify
- Much faster than grid search
- Often finds good results with fewer iterations
- Best when you have many parameters or want faster results

**2. Optuna (Advanced)** 🚀
- Uses smart algorithms to learn from previous trials
- Focuses search on promising areas
- Most efficient for complex parameter spaces
- Can automatically stop when no improvement is found

#### Why Parameter Tuning Matters:
- **Better Performance**: Well-tuned models often perform significantly better
- **Prevent Overfitting**: Proper regularization parameters help your model generalize
- **Optimize for Your Data**: Different datasets need different parameter settings
"""