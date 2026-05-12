
# Titanic Survival Prediction - Azure ML MLOps Project

## Project Overview

This project builds, tunes, registers, and deploys a machine learning model to predict passenger survival on the Titanic dataset.

The goal of this project is to demonstrate an end-to-end Azure Machine Learning workflow using:

- Data preprocessing
- Feature engineering
- Random Forest classification
- Hyperparameter tuning using Azure ML sweep jobs
- MLflow metric logging
- Model registration
- Managed online endpoint deployment
- GitHub Actions CI/CD automation

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- MLflow
- Azure Machine Learning
- Azure ML SDK v2
- GitHub Actions

## Project Structure

├── src/
│   ├── train_tuning.py
│   └── score.py
│
├── scripts/
│   ├── model_tuning_submit_job_script.py
│   ├── register_best_model.py
│   └── model_deployment_script.py
│
├── .github/
│   └── workflows/
│       └── train-deploy.yml
│
├── requirements.txt
└── README.md

# Machine Learning Workflow:

# 1. Data Asset

- The Titanic dataset is stored as an Azure ML data asset: Titanic_Survival_Dataset
- The training job reads this registered data asset during execution.

# 2. Training and Tuning

The model training script performs:

- Missing value handling
- Feature engineering
- Train-test split
- Preprocessing using ColumnTransformer
- One-hot encoding for categorical variables
- Scaling for numeric variables
- Random Forest model training
- Accuracy and ROC-AUC evaluation
- MLflow metric logging
- Model artifact saving

The model is saved as: outputs/model_pipeline.pkl

# 3. Hyperparameter Sweep

- Azure ML sweep job tunes the following Random Forest parameters:
    - n_estimators
    - max_depth
    - min_samples_split
- The primary metric used for model selection is: accuracy

# 4. Model Registration

- After the sweep job completes, the best child run is selected automatically.
- The best model is registered in Azure ML as: titanic-survival-pipeline-model

# 5. Deployment

- The registered model is deployed to an Azure ML managed online endpoint: titanic-endpoint
- Deployment name: blue
- The deployment uses: src/score.py for inference.

# 6. GitHub Actions Workflow

- The workflow is defined in: .github/workflows/train-deploy.yml
- It runs on: Manual trigger using workflow_dispatch. Push to the main branch
- The workflow performs the following steps:

  - Checkout repository
  - Set up Python
  - Install dependencies
  - Log in to Azure
  - Submit Azure ML sweep job
  - Register the best model
  - Deploy the model to an online endpoint
  - Required GitHub Secrets

- The following secrets are configured in GitHub:
  - AZURE_CREDENTIALS
  - AZURE_SUBSCRIPTION_ID
  - AZURE_RESOURCE_GROUP
  - AZURE_ML_WORKSPACE

# 7. Sample Input for Endpoint
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}

# 8. Sample Output
{
  "model_version": "4",
  "deployment": "blue",
  "predictions": [0],
  "probabilities": [0.18],
  "latency_ms": 25
}

# 9. Features Used:

The model uses the following input features:
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

# 10. Additional features are created during preprocessing:

- Family_Size = SibSp + Parch
- Alone = 1 if Family_Size is 0 else 0
- Model Evaluation Metrics

# 11. The model logs the following metrics to MLflow:

- accuracy
- val_auc
- Notes

This project is intended as a learning and portfolio project to demonstrate practical MLOps skills using Azure Machine Learning and GitHub Actions.

It shows how to move from local model development to automated cloud-based training, model registration, and deployment.
