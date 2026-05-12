import argparse
import os
import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# -----------------------------
# STEP 1: Read input arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
parser.add_argument("--min_samples_split", type=int, default=2)
args = parser.parse_args()

# -----------------------------
# STEP 2: Load data
# -----------------------------
df = pd.read_csv(args.data)

# -----------------------------
# STEP 3: Data Cleaning
# -----------------------------
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# -----------------------------
# STEP 4: Split features / target
# -----------------------------

df['Family_Size'] = df['SibSp'] + df['Parch']
df['Alone'] =  df['Family_Size'].apply( lambda x:1 if x==0 else 0)

df.drop(['SibSp', 'Parch'], axis = 1, inplace = True)


X = df.drop("Survived", axis=1)
y = df["Survived"]

# -----------------------------
# STEP 5: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# STEP 6: Preprocessing + pipeline
# -----------------------------
num_cols = ['Age', 'Fare', 'Family_Size']
cat_cols = ['Pclass', 'Sex', 'Embarked', 'Alone']

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42,
        class_weight="balanced"
    ))
])

# -----------------------------
# STEP 7: Train Random Forest
# -----------------------------

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, preds)
val_auc = roc_auc_score(y_test, probs)

print(f"accuracy={accuracy}")
print(f"val_auc={val_auc}")
print(classification_report(y_test, preds))

# -----------------------------
# STEP 8: MLflow logging
# -----------------------------

# MLflow metric logging for Azure ML sweep tracking
mlflow.log_metric("accuracy", float(accuracy))
mlflow.log_metric("val_auc", float(val_auc))

# optional: log params too
mlflow.log_param("n_estimators", args.n_estimators)
mlflow.log_param("max_depth", args.max_depth)
mlflow.log_param("min_samples_split", args.min_samples_split)

# -----------------------------
# STEP 9: Save outputs
# -----------------------------
os.makedirs("outputs", exist_ok=True)

MODEL_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

model_bundle = {
    "model": pipeline,
    "features": MODEL_FEATURES,
    "metadata": {
        "project": "Titanic Survival",
        "model_type": "RandomForestClassifier",
        "target": "Survived",
        "numeric_features": ["Age", "Fare", "Family_Size"],
        "categorical_features": ["Pclass", "Sex", "Embarked", "Alone"],
        "accuracy": float(accuracy),
        "val_auc": float(val_auc),
    }
}

joblib.dump(model_bundle, "outputs/model_pipeline.pkl")

print("Saved model to outputs/model_pipeline.pkl")
