"""
SDS6210_InformaticsForHealth - Machine Learning Classification
Part VI: ML Implementation with Performance Metrics

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
from imblearn.over_sampling import SMOTE
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

print(f"\nClass Distribution:")
print(f"  Low Risk (0):  {(df['high_risk'] == 0).sum()} ({(df['high_risk'] == 0).mean()*100:.1f}%)")
print(f"  High Risk (1): {(df['high_risk'] == 1).sum()} ({(df['high_risk'] == 1).mean()*100:.1f}%)")

# Prepare features
features = ['age', 'sex_male', 'height_cm', 'weight_kg', 'bmi',
            'sbp_mmHg', 'ldl_mgdl', 'hdl_mgdl', 'glucose_mgdl', 'hba1c_pct']

X = df[features].fillna(df[features].median())
y = df['high_risk']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6210, stratify=y)

print(f"\nTrain/Test Split:")
print(f"  Train: {len(X_train)} samples (High Risk: {sum(y_train)})")
print(f"  Test:  {len(X_test)} samples (High Risk: {sum(y_test)})")

# Apply SMOTE to handle class imbalance
print("\nApplying SMOTE oversampling to balance classes...")
smote = SMOTE(random_state=6210)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE: {len(X_train_resampled)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=6210),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=6210)
}

results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled if name == 'Logistic Regression' else X_train_scaled, y_train_resampled)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    
    # Handle zero division cases
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'ROC-AUC': auc
    })

    print(f"  {name}:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    ROC-AUC:   {auc:.4f}")

# Confusion Matrix for Random Forest
print("\n" + "=" * 60)
print("Confusion Matrix (Random Forest):")
print("=" * 60)
cm = confusion_matrix(y_test, results[1]['Model'] if False else y_pred if 'Forest' in results[1]['Model'] else y_pred)
print(f"               Predicted")
print(f"             Neg    Pos")
print(f"Actual Neg  {cm[0,0]:4d}   {cm[0,1]:4d}")
print(f"Actual Pos  {cm[1,0]:4d}   {cm[1,1]:4d}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

# Feature Importance (Random Forest)
print("\n" + "=" * 60)
print("Top 10 Feature Importances (Random Forest):")
print("=" * 60)
rf = models['Random Forest']
importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance.head(10).to_string(index=False))

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('data/ml_model_comparison.csv', index=False)
print(f"\n{'=' * 60}")
print("Results saved to data/ml_model_comparison.csv")
print("=" * 60)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Model Performance Comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.35

lr_scores = [results[0]['Accuracy'], results[0]['Precision'], results[0]['Recall'], results[0]['F1'], results[0]['ROC-AUC']]
rf_scores = [results[1]['Accuracy'], results[1]['Precision'], results[1]['Recall'], results[1]['F1'], results[1]['ROC-AUC']]

bars1 = axes[0].bar(x - width/2, lr_scores, width, label='Logistic Regression', color='steelblue')
bars2 = axes[0].bar(x + width/2, rf_scores, width, label='Random Forest', color='forestgreen')

axes[0].set_ylabel('Score')
axes[0].set_title('Model Performance Comparison (with SMOTE)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics)
axes[0].legend()
axes[0].set_ylim(0, 1.1)
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# Add value labels on bars
for bar in bars1 + bars2:
    height = bar.get_height()
    axes[0].annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# Plot 2: Feature Importance
importance_sorted = importance.sort_values('Importance', ascending=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_sorted)))
axes[1].barh(importance_sorted['Feature'], importance_sorted['Importance'], color=colors)
axes[1].set_xlabel('Importance')
axes[1].set_title('Feature Importance (Random Forest)')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/ml_performance.png', dpi=150, bbox_inches='tight')
print("Saved: visualizations/ml_performance.png")
plt.close()
