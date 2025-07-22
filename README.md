# churn_analysis
Complete machine learning project predicting customer churn with 85%+ accuracy.

## Project Overview
Customer churn is one of the most critical metrics for subscription-based businesses. This project provides a complete framework for analyzing customer behavior patterns and predicting which customers are likely to churn. The solution includes automated data generation, comprehensive analysis, and machine learning models that achieve high predictive accuracy.

## Installation
- Clone the repository
  
git clone https://github.com/yourusername/customer-churn-analysis.git
cd customer-churn-analysis

- Install required packages
  
pip install -r requirements.txt

- Usage
  
python customer_churn_analysis.py
from customer_churn_analysis import CustomerChurnAnalyzer
analyzer = CustomerChurnAnalyzer()
df, models, results = analyzer.run_full_analysis()

analyzer = CustomerChurnAnalyzer()
analyzer.run_full_analysis('your_data.csv')



