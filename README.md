# ML Model Development Guide

A guided, web-based machine learning model development application built with Streamlit. This application provides a step-by-step approach to building, training, evaluating, and explaining machine learning models for both classification and regression problems, with a focus on responsible AI practices.

## Overview

ML Builder is an educational and professional tool designed to guide users through the complete machine learning development lifecycle. Each stage includes built-in guidance, interactive visualizations, and ethical considerations to ensure responsible model development.

## Key Features

- **9-Stage Guided Workflow**: Complete ML pipeline from data loading to model explanation
- **Dual Problem Support**: Handles both classification and regression tasks
- **Interactive Visualizations**: Rich data analysis and model performance visualizations powered by Plotly
- **Responsible AI Integration**: Built-in bias detection, fairness analysis, and ethical considerations
- **Model Comparison**: Quick performance comparison across multiple algorithms
- **Educational Content**: Comprehensive explanations and guidance at each stage
- **Export Capabilities**: Download models, datasets, and complete reproduction scripts

2. **Begin development** by clicking "Start ML Development" and either:
   - Upload your own CSV dataset
   - Use one of the provided sample datasets (Titanic or Miami Housing)

## Development Pipeline

### 1. Data Loading
- Secure CSV file upload with validation (100MB limit, MIME type checking)
- Sample datasets provided (Titanic classification, Miami Housing regression)
- Target variable selection with automatic problem type detection
- Basic data quality assessment

### 2. Data Exploration
- Multi-tab dataset analysis interface
- Statistical summaries and distributions
- Correlation analysis using Dython for mixed data types
- Missing value pattern analysis
- Interactive visualizations

### 3. Data Preprocessing
- Missing value imputation (mean, median, mode, advanced strategies)
- Categorical encoding (One-Hot, Label, Target encoding)
- Feature scaling (Standard, MinMax, Robust scalers)
- Data type optimization

### 4. Feature Selection
- Statistical feature importance analysis
- Correlation-based filtering
- Bias detection for protected attributes
- Interactive feature selection interface
- Responsible AI considerations

### 5. Model Selection
- Intelligent model recommendations based on data characteristics
- Quick model comparison with sample data
- Non-linearity detection using mutual information
- XGBoost compatibility checking for multiclass problems
- Educational model guide with performance characteristics

### 6. Model Training
- Multiple training approaches:
  - Standard training with default parameters
  - Random Search hyperparameter optimization
  - Optuna Bayesian optimization
- Cross-validation with detailed metrics
- Model enhancement (calibration, threshold optimization)

### 7. Model Evaluation
- Comprehensive performance metrics
- Interactive visualizations (ROC curves, confusion matrices, residual plots)
- Sample predictions display
- Model stability assessment
- Cross-validation results analysis

### 8. Model Explanation
- SHAP value analysis and visualizations
- Feature importance explanations
- Individual prediction explanations
- Bias and fairness assessment
- Model transparency reporting

### 9. Summary & Export
- Complete development journey overview
- Model recreation Python scripts
- Processed dataset downloads
- Development timeline and logging
- Environment setup instructions

## Supported Algorithms

### Classification Models
- Logistic Regression
- Decision Tree
- Random Forest
- Multilayer Perceptron (MLP)
- Gradient Boosting
- XGBoost (with multiclass compatibility checking)
- LightGBM

### Regression Models
- Linear Regression
- Decision Tree
- Random Forest
- Multilayer Perceptron (MLP)
- Gradient Boosting
- XGBoost
- LightGBM

## Sample Datasets

Two sample datasets are included:

1. **Titanic Dataset** (Classification)
   - Binary classification problem (survival prediction)
   - 8 features including passenger class, age, sex, fare
   - Educational dataset for learning classification concepts

2. **Miami Housing Dataset** (Regression)
   - Housing price prediction problem
   - 15+ features including location, size, distance metrics
   - Real-world regression problem with mixed feature types

## Installation and Launch

You can run ML Builder via the CLI after installing the package.

```bash
# Recommended: create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install 
pip install rai_mldev 

# Launch the application (CLI entry point)
ml-builder
```

## Technical Dependencies

### Core Framework
- streamlit>=1.24.0 (Web application framework)
- pandas>=1.5.0 (Data manipulation)
- numpy>=1.23.0 (Numerical computing)
- scikit-learn>=1.0.2 (Machine learning algorithms)

### Visualization
- plotly>=5.13.0,<7.0.0 (Interactive visualizations)
- matplotlib>=3.6.0 (Static plotting)
- seaborn>=0.12.0 (Statistical visualization)

### Advanced ML
- xgboost>=2.1.4 (Gradient boosting)
- lightgbm>=4.6.0 (Gradient boosting)
- optuna>=3.5.0 (Hyperparameter optimization)
- shap>=0.41.0 (Model explanation)

### Specialized Libraries
- category_encoders>=2.6.0 (Advanced encoding)
- dython>=0.7.3 (Mixed-type correlation)
- fairlearn>=0.7.0 (Fairness assessment)
- python-magic>=0.4.27 (File type detection)
- imblearn (Imbalanced learning)
- Boruta>=0.3 (Feature selection)

## Project Structure

```
ML_Builder_New/
├── app.py                     # Main application entry point
├── Builder.py                 # Core ML pipeline class
├── requirements.txt           # Project dependencies
├── pages/                     # Individual stage pages
│   ├── 1_Data_Loading.py
│   ├── 2_Data_Exploration.py
│   ├── 3_Data_Preprocessing.py
│   ├── 4_Feature_Selection.py
│   ├── 5_Model_Selection.py
│   ├── 6_Model_Training.py
│   ├── 7_Model_Evaluation.py
│   ├── 8_Model_Explanation.py
│   └── 9_Summary.py
├── components/                # Reusable components
├── utils/                     # Utility functions
├── content/                   # Educational content
├── sample_data/              # Sample datasets
└── docs/                     # Comprehensive documentation
```

## Security Features

- **File Upload Validation**: Multi-layer security including file size limits, MIME type detection, and content validation
- **Input Sanitization**: Protection against code injection in column names
- **Local Processing**: All data processing occurs locally without external transmission
- **Error Handling**: Robust error boundaries prevent application crashes

## Responsible AI Features

### Bias Detection
- Automatic identification of potentially protected attributes
- Statistical bias testing across demographic groups
- Feature correlation analysis for fairness assessment

### Model Interpretability
- SHAP value analysis for prediction explanations
- Feature importance rankings and visualizations
- Individual prediction breakdowns
- Model decision boundary analysis

### Ethical Guidelines
- Stage-specific ethical considerations and prompts
- Best practice recommendations
- Impact assessment guidance
- Fairness metric calculations

## Educational Components

- **Interactive Model Guide**: Detailed explanations of algorithm characteristics
- **Performance Metrics Education**: Clear explanations of evaluation metrics
- **Statistical Concept Explanations**: Educational content for statistical tests and measures
- **Contextual Help**: Stage-specific guidance and tips
- **Journey Tracking**: Complete audit trail of development decisions

## Usage Tips

- **Data Preparation**: Ensure CSV files have clear headers and proper formatting
- **Problem Type**: The application automatically detects classification vs regression problems
- **Model Selection**: Use the quick comparison feature to identify promising algorithms
- **Hyperparameter Tuning**: Start with Random Search before using Optuna for final optimization
- **Evaluation**: Review all metrics and visualizations before proceeding to explanation
- **Export**: Use the Summary page to download complete project artifacts

## Performance Considerations

- **File Size Limit**: 100MB maximum for uploaded CSV files
- **Sample-Based Analysis**: Large datasets are automatically sampled for quick model comparison
- **Memory Management**: Efficient data handling with automatic cleanup
- **Parallel Processing**: Multi-core support for compatible algorithms

## Troubleshooting

### Common Issues
- **Upload Failures**: Verify CSV format and file encoding (UTF-8 recommended)
- **Model Training Errors**: Check for missing values or incompatible data types
- **Memory Issues**: Reduce dataset size or use data sampling options
- **XGBoost Compatibility**: For multiclass problems, ensure class labels start from 0

### Getting Help
- Review comprehensive documentation in the `docs/` directory
- Check application logs and error messages
- Verify all dependencies are correctly installed
- Ensure Python version compatibility (3.8+)

## Contributing

Contributions are welcome! Areas for contribution include:
- New algorithm implementations
- Enhanced visualization components
- Additional evaluation metrics
- Documentation improvements
- Bug fixes and performance optimizations

## License

This software is licensed under a **Proprietary Evaluation License**.

### Permitted Uses

- **Educational Use**: Free for personal learning and classroom instruction at accredited educational institutions
- **Corporate Evaluation**: 30-day evaluation period for assessing the software for potential commercial licensing
- **Internal Modifications**: Allowed within the scope of permitted uses

### Restrictions

- **No Commercial Use**: Production deployments, revenue-generating activities, SaaS offerings, or integration into commercial products require a separate commercial license
- **No Redistribution**: The software and modified versions may not be redistributed, sublicensed, sold, or transferred to third parties
- **Output Restrictions**: All models, datasets, scripts, and other outputs generated by the software may only be used for non-commercial purposes

### Commercial Licensing

For commercial use, production deployments, or use beyond the 30-day evaluation period, please contact:

**Email**: richard.wheeler@priosym.com

### Third-Party Components

This software depends on third-party open-source components that remain subject to their respective licenses. The Proprietary Evaluation License governs only the ML Builder software itself.

See the [LICENSE](LICENSE) file for complete terms and conditions.

## Author

**Richard Wheeler**  
*ML Engineer & Responsible AI Researcher*

---

*Version 1.0.0 | A comprehensive tool for responsible machine learning model development*