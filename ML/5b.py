import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the models
ada_classifier = AdaBoostClassifier(n_estimators=50, random_state=0)
gboost_classifier = GradientBoostingClassifier(n_estimators=50, random_state=0)
xgboost_classifier = xgb.XGBClassifier(n_estimators=50, random_state=0)

ada_classifier.fit(X_train, y_train)
gboost_classifier.fit(X_train, y_train)
xgboost_classifier.fit(X_train, y_train)

# Make predictions
ada_pred = ada_classifier.predict(X_test)
gboost_pred = gboost_classifier.predict(X_test)
xgboost_pred = xgboost_classifier.predict(X_test)

# Evaluate the models
metrics = {
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "F1 Score": f1_score,
    "ROC-AUC": roc_auc_score,
}

results = {"Model": [], "Metric": [], "Score": []}

models = {
    "AdaBoost": ada_pred,
    "Gradient Tree Boosting": gboost_pred,
    "XGBoost": xgboost_pred,
}

for model_name, predictions in models.items():
    for metric_name, metric_function in metrics.items():
        score = metric_function(y_test, predictions)
        results["Model"].append(model_name)
        results["Metric"].append(metric_name)
        results["Score"].append(score)

results_df = pd.DataFrame(results)
print(results_df)
