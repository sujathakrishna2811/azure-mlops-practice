# Telco Customer Churn Prediction

## Project Goal
Analyze customer behavior and predict customer churn using Machine Learning to help businesses improve customer retention strategies.

## Tools & Technologies Used

- Python
- Pandas
- NumPy
- Seaborn & Matplotlib
- Scikit-learn
- Azure Machine Learning
- MLflow
- GitHub Actions


## Key Business Questions

- Which customers are most likely to churn?
- What factors influence customer churn?
- How can churn be reduced?


## Exploratory Data Analysis (EDA) Insights

- Month-to-month contract customers have the highest churn risk.
- Customers with low tenure are more likely to leave.
- Higher monthly charges increase churn probability.
- Customers without TechSupport and OnlineSecurity churn more frequently.
- Fiber optic users show higher churn rates.
- Electronic check users churn more frequently.
- Long-term contracts significantly reduce churn.


## Machine Learning Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier


## Model Performance

- Random Forest Classifier achieved the best overall performance.
- Cross-validation confirmed model stability.
- ROC-AUC score was used for model comparison and evaluation.
- Hyperparameter tuning was performed using Azure ML Sweep Jobs.


## Final Model Selection

### Selected Model:
**Random Forest Classifier**

### Reasons for Selection:
- Balanced precision and recall
- Stable cross-validation performance
- Strong ROC-AUC performance
- Good interpretability and feature importance analysis

## MLOps Workflow

This project also demonstrates an end-to-end Azure ML MLOps workflow:

- Model training pipeline
- Hyperparameter tuning with Azure ML Sweep
- MLflow metric logging
- Best model registration
- Managed online endpoint deployment
- Inference scoring script with validation
- GitHub Actions workflow automation

## Business Recommendations

- Offer incentives for long-term contracts
- Reduce high monthly charges or provide flexible pricing plans
- Promote value-added services such as TechSupport and OnlineSecurity
- Improve customer experience for electronic check users
- Focus on early engagement strategies for new customers

## Repository Structure

azureml-telco-churn-mlops/
│
├── src/
├── scripts/
├── env/
├── sample/
├── .github/workflows/
└── README.md
