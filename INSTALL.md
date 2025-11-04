# ML Builder Installation Guide

## License Notice

**IMPORTANT**: This software is licensed under a **Proprietary Evaluation License**.

- **Free for**: Personal learning and classroom instruction at accredited educational institutions
- **Evaluation**: Corporate entities may evaluate for 30 days
- **Commercial use requires**: Separate commercial license
- **Contact**: richard.wheeler@priosym.com for commercial licensing

See the [LICENSE](LICENSE) file for complete terms and conditions.

## Prerequisites

- **Python**: 3.8 or higher
- **pip**: Python package installer (comes with Python)
- **Operating System**: Windows 10+, macOS 10.14+, or Linux

## Quick Install

### Method 1: From PyPI (Recommended when published)
```bash
pip install ml-builder
ml-builder
```

### Method 2: From GitHub Repository
```bash
pip install git+https://github.com/R-I-Wheeler/ML_Builder_Beta.git
ml-builder
```

### Method 3: From Downloaded Package File
If you have the `.whl` file:
```bash
pip install ml_builder-1.0.0-py3-none-any.whl
ml-builder
```

## Installation in Virtual Environment (Recommended)

### Windows
```bash
# Create virtual environment
python -m venv ml-builder-env

# Activate virtual environment
ml-builder-env\Scripts\activate

# Install ML Builder
pip install ml-builder

# Run the application
ml-builder
```

### macOS/Linux
```bash
# Create virtual environment
python -m venv ml-builder-env

# Activate virtual environment
source ml-builder-env/bin/activate

# Install ML Builder
pip install ml-builder

# Run the application
ml-builder
```

## System Requirements

- **RAM**: Minimum 4GB (8GB+ recommended for large datasets)
- **Disk Space**: ~500MB for installation and dependencies
- **Internet**: Required for initial package download and some visualizations

## What Gets Installed

ML Builder installs with all necessary dependencies:

- **Core ML Libraries**: scikit-learn, pandas, numpy
- **Visualization**: plotly, matplotlib, seaborn
- **Advanced ML**: XGBoost, LightGBM, CatBoost
- **Model Explanation**: SHAP, LIME
- **Optimization**: Optuna for hyperparameter tuning
- **Responsible AI**: Fairlearn for bias detection
- **Web Framework**: Streamlit for the user interface

## Verifying Installation

After installation, verify everything works:

```bash
# Check if command is available
ml-builder --help

# Or run directly
python -c "import ml_builder; print(ml_builder.__version__)"
```

## Troubleshooting

### Command Not Found

If `ml-builder` command is not found after installation, try:

**Option 1**: Use full Python module path
```bash
python -m ml_builder.cli
```

**Option 2**: Add Python scripts to PATH

**Windows**: Add to PATH:
```
%USERPROFILE%\AppData\Local\Programs\Python\Python3X\Scripts
```

**macOS/Linux**: Add to PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Dependency Conflicts

If you encounter dependency conflicts:

1. **Use a fresh virtual environment:**
```bash
python -m venv fresh-env
source fresh-env/bin/activate  # Windows: fresh-env\Scripts\activate
pip install ml-builder
```

2. **Update pip:**
```bash
pip install --upgrade pip
pip install ml-builder
```

### Platform-Specific Issues

**Windows - Visual C++ Build Tools**
Some packages (like LightGBM) may require Visual C++ Build Tools:
- Download from Microsoft or use: `pip install lightgbm --prefer-binary`

**macOS - libmagic**
```bash
brew install libmagic
pip install ml-builder
```

**Linux - libmagic**
```bash
sudo apt-get update
sudo apt-get install libmagic1
pip install ml-builder
```

### Memory Issues

For large datasets or memory-constrained systems:
- Use smaller dataset samples during initial exploration
- Close other applications before running ML Builder
- Consider using cloud environments with more RAM

### Browser Not Opening

If Streamlit doesn't automatically open your browser:
1. Look for the URL in the terminal output (usually `http://localhost:8501`)
2. Manually open the URL in your web browser
3. Ensure no firewall is blocking local connections

## Usage

Once installed and running:

1. **Access the application** at `http://localhost:8501`
2. **Upload your CSV dataset** or use the provided sample datasets
3. **Follow the 9-stage guided workflow:**
   - Data Loading
   - Data Exploration  
   - Data Preprocessing
   - Feature Selection
   - Model Selection
   - Model Training
   - Model Evaluation
   - Model Explanation
   - Summary & Export

## Sample Data

The package includes two sample datasets for learning:
- **Titanic Dataset**: Binary classification problem
- **Miami Housing**: Regression problem

## Updating

To update to the latest version:
```bash
pip install --upgrade ml-builder
```

## Uninstalling

To remove ML Builder:
```bash
pip uninstall ml-builder
```

## Getting Help

- **Documentation**: See README.md for detailed feature information
- **Issues**: Report problems on the GitHub repository
- **Sample Datasets**: Use included Titanic and Miami Housing datasets for learning

## Development Installation

For developers who want to modify the code:

```bash
# Clone the repository
git clone https://github.com/richardwheeler/ML_Builder_Beta.git
cd ML_Builder_Beta

# Install in development mode
pip install -e .

# Run the application
ml-builder
```

This allows you to make changes to the code without reinstalling the package.

---

**Support**: For additional help, please refer to the comprehensive documentation or file an issue on the GitHub repository.