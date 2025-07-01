# Patient Readmission Risk Prediction

## Overview
This project aims to predict the risk of patient readmission within 30 days of hospital discharge using machine learning. The solution is designed to help hospitals proactively identify high-risk patients, reduce unnecessary readmissions, improve patient outcomes, and optimize resource allocation.

## Features
- Comprehensive data strategy leveraging EHRs, demographics, and social determinants
- Advanced preprocessing and feature engineering pipeline
- Gradient Boosting (XGBoost/LightGBM) model for accurate predictions
- Model evaluation with precision, recall, ROC AUC, and feature importance
- Deployment and integration plan for hospital systems
- HIPAA-compliant data handling and privacy safeguards
- Optimization techniques to prevent overfitting

## Data Sources
- `diabetic_data.csv`: Main dataset containing patient records
- `IDS_mapping.csv`: Mapping file for feature engineering

## Project Structure
- `Patient Readmission Risk Prediction.md`: Main report covering scope, data, modeling, deployment, and optimization
- `preprocessing_pipeline.ipynb`: Jupyter notebook for data processing, model training, evaluation, and optimization

## Setup Instructions
1. Clone the repository or download the project files.
2. Ensure you have Python 3.8+ installed.
3. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib
   ```
4. Place the data files (`diabetic_data.csv`, `IDS_mapping.csv`) in the project directory.
5. Open and run `preprocessing_pipeline.ipynb` in Jupyter Notebook or JupyterLab.

## Usage
- Review the main report in `Patient Readmission Risk Prediction.md` for methodology and results.
- Use the notebook to preprocess data, train models, evaluate performance, and experiment with optimization techniques.
- Follow the deployment section in the report for integration guidance.

## Contact
For questions or collaboration, please contact the project maintainer. 