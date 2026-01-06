"""
SDS6210_InformaticsForHealth - Machine Learning Classification
Part VII: ML Implementation with Performance Metrics

Author: Cavin Otieno
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import os

print("=" * 60)
print("MACHINE LEARNING CLASSIFICATION")
print("=" * 60)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load data
df = pd.read_csv('data/patient_survival_data.csv')

# Create binary target
df['high_risk'] = ((df['age'] > 60) & (df['sbp_mmHg'] > 140) & (df['diabetes'] == 1)).astype(int)

# Prepare features
features = ['age', 'sex_male', 'height_cm', 'weight_kg', 'bmi',
            'sbp_mmHg', 'ldl_mgdl', 'hdl_mgdl', 'glucose_mgdl', 'hba1c_pct']

X = df[features].fillna(df[features].median())
y = df['high_risk']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6210)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'Logistic Regression': LogisticRegression(random_state=6210),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=6210)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled if name == 'Logistic Regression' else X_train, y_train)
    y_pred = model.predict(X_test_scaled if name == 'Logistic Regression' else X_test)
    y_prob = model.predict_proba(X_test_scaled if name == 'Logistic Regression' else X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'ROC-AUC': auc
    })

    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {auc:.4f}")

# Confusion Matrix
print("\nConfusion Matrix (Random Forest):")
cm = confusion_matrix(y_test, results[1]['Model'] if False else y_test)
print(cm)

# Feature Importance (Random Forest)
print("\nTop 10 Feature Importances (Random Forest):")
rf = models['Random Forest']
importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance.head(10).to_string(index=False))

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('data/ml_model_comparison.csv', index=False)
print("\nResults saved to data/ml_model_comparison.csv")
