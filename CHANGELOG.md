# Changelog

All notable changes to the ML Builder project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [1.2.0] - 07-12-2025

### Changed

- Updated Fairness Analysis in Model Explanation
  - Fairness Heatmap: Introduces a new, interactive Plotly heatmap to provide an at-a-glance overview of fairness across all features and components, color-coded by severity (Green, Orange, Red) and sorted by score (worst first).
  - Interactive Impact Visualization: Replaces the text-based impact calculator with a visual chart (e.g., a horizontal bar chart) that allows users to input application volume and see the real-world impact (affected applications/populations) visually.
  - High-Cardinality Feature Solutions (Handling Complex Data)
     - Automatic Intelligent Binning: Implements a new function to automatically group features with >15 unique values (e.g., zip codes, continuous age) into 5−7 meaningful ranges (using quantile or domain-appropriate binning) before analysis.
  - Export Fairness Report: Adds a button to export the comprehensive report (including key visuals and recommendations) to PDF, HTML, and JSON for stakeholder communication and documentation.
  - Fairness Threshold Customization: Allows users to adjust the fairness threshold (default 0.8) via a slider and presets, which dynamically updates all visualizations and status labels.
- Removed the 'Load a Random Test Sample' option from the 'What-If Analysis'
- Fixed an issue where the page did not clear correctly when returning to the page to try a different model after using the 'Automated Model Selection and Training' function
- Updated all Feature Selection functions to add basic statistic information for features that have been removed in journey points for reference


## [1.1.0] - 05-11-2025

### Changed

- **License**: Changed from MIT License to Proprietary Evaluation License
  - **Permitted Uses**:
    - Educational use (personal learning and classroom instruction at accredited institutions)
    - 30-day corporate evaluation period
    - Internal modifications within permitted use scope
  - **Restrictions**:
    - No commercial use without separate license
    - No redistribution, sublicensing, or resale
    - Output artifacts restricted to non-commercial purposes
  - **Commercial Licensing**: Contact richard.wheeler@priosym.com

### Notes

- No functional changes to the application
- All features and capabilities remain unchanged
- This is a licensing-only update

## [1.0.0] - 1-11-2025

### Added

- Initial release of ML Builder
- 9-stage guided ML development workflow
- Support for classification and regression problems
- Interactive data exploration and visualization
- Comprehensive preprocessing tools
- Feature selection with bias detection
- Model selection and recommendation engine
- Multiple training approaches (standard, random search, Optuna)
- Model evaluation with extensive metrics
- SHAP-based model explanation
- Fairness and bias assessment
- Export capabilities for models and reproduction scripts
- Sample datasets (Titanic, Miami Housing)
