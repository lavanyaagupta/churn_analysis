# Customer Churn Analysis 
A comprehensive machine learning project that predicts customer churn using advanced data science techniques. This end-to-end solution demonstrates the complete data science pipeline from exploratory data analysis to actionable business insights.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![Pandas](https://img.shields.io/badge/pandas-latest-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Project Overview

Customer churn is one of the most critical metrics for subscription-based businesses. This project provides a complete framework for analyzing customer behavior patterns and predicting which customers are likely to churn. The solution includes automated data generation, comprehensive analysis, and machine learning models that achieve high predictive accuracy.

## Key Features

- **Complete ML Pipeline**: End-to-end workflow from data ingestion to model deployment
- **Advanced EDA**: Comprehensive exploratory data analysis with interactive visualizations
- **Feature Engineering**: Automated creation of predictive features from raw data
- **Multiple ML Models**: Comparison of Logistic Regression and Random Forest algorithms
- **Model Evaluation**: ROC curves, confusion matrices, and cross-validation metrics
- **Business Insights**: Actionable recommendations based on data-driven findings
- **Professional Visualizations**: Publication-ready charts and graphs
- **Modular Design**: Clean, object-oriented code structure for easy extension

## Technical Implementation

### Machine Learning Models
- **Logistic Regression**: Linear approach with feature scaling
- **Random Forest**: Ensemble method with feature importance analysis
- **Cross-Validation**: 5-fold CV for robust model evaluation
- **Hyperparameter Optimization**: Grid search for optimal performance

### Data Science Techniques
- **Statistical Analysis**: Distribution analysis, correlation studies
- **Feature Engineering**: Creating predictive variables from existing data
- **Data Preprocessing**: Missing value imputation, categorical encoding
- **Model Selection**: AUC-based comparison with multiple metrics
- **Visualization**: Matplotlib and Seaborn for professional graphics

## Sample Results

The model achieves:
- **AUC Score**: 0.85+ on test data
- **Precision**: 80%+ for churn prediction
- **Feature Importance**: Identifies top 10 churn predictors
- **Business Impact**: Quantifies potential revenue retention

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/customer-churn-analysis.git
cd customer-churn-analysis

# Install required packages
pip install -r requirements.txt
```

### Usage

```python
# Run the complete analysis
python customer_churn_analysis.py

# Or use programmatically
from customer_churn_analysis import CustomerChurnAnalyzer

analyzer = CustomerChurnAnalyzer()
df, models, results = analyzer.run_full_analysis()
```

### With Your Own Data

```python
# Load your own CSV file
analyzer = CustomerChurnAnalyzer()
analyzer.run_full_analysis('your_data.csv')
```

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Project Structure

```
customer-churn-analysis/
│
├── customer_churn_analysis.py  # Main analysis script
├── README.md                   # This file
├── requirements.txt           # Python dependencies
├── data/                      # Data directory (optional)
│   └── sample_data.csv       # Sample dataset
├── outputs/                   # Generated visualizations
│   ├── eda_plots.png         # Exploratory data analysis
│   ├── model_comparison.png  # Model performance charts
│   └── feature_importance.png # Feature importance plot
└── notebooks/                 # Jupyter notebooks (optional)
    └── analysis.ipynb        # Interactive analysis
```

## Key Insights Generated

The analysis provides actionable insights such as:

- **Contract Type Impact**: Month-to-month customers show 3x higher churn rates
- **Payment Method Risk**: Electronic check users have elevated churn probability
- **Tenure Analysis**: Customers with >24 months tenure show significantly lower churn
- **Value Segmentation**: High-value customers require different retention strategies
- **Seasonal Patterns**: Identification of time-based churn trends

## Visualizations

The project generates comprehensive visualizations including:

- Customer demographic distributions
- Churn rate analysis by customer segments
- Feature correlation heatmaps
- ROC curves for model comparison
- Feature importance rankings
- Confusion matrices for model evaluation

## Learning Outcomes

This project demonstrates proficiency in:

- **Data Science Methodology**: Complete CRISP-DM workflow implementation
- **Statistical Analysis**: Hypothesis testing and statistical inference
- **Machine Learning**: Supervised learning with proper validation
- **Data Visualization**: Professional chart creation and storytelling
- **Business Analytics**: Translating data insights into business value
- **Python Programming**: Clean, efficient, and well-documented code
- **Software Engineering**: Modular design and best practices

## Future Enhancements

Potential extensions to this project:

- **Deep Learning Models**: Neural network implementation for complex patterns
- **Real-time Prediction**: API development for live churn scoring
- **A/B Testing Framework**: Experiment design for retention strategies
- **Advanced Feature Engineering**: Time-series and interaction features
- **Dashboard Creation**: Interactive web application with Streamlit/Dash
- **MLOps Integration**: Model versioning and automated retraining

## Contact

Lavanyaa Gupta - lavanyaagupta24@gmail.com
Project Link: [https://github.com/lavanyaagupta/customer-churn-analysis](https://github.com/lavanyaagupta/customer-churn-analysis)
