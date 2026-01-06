"""
SDS6210_InformaticsForHealth - Survival Analysis
Part II & IV: Kaplan-Meier Analysis and Cox Proportional Hazards Model

Author: Cavin Otieno
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter

print("=" * 60)
print("PYTHON SURVIVAL ANALYSIS")
print("=" * 60)

# Load data
df = pd.read_csv('data/patient_survival_data.csv')

# PART I: Theoretical Concepts
print("\nPART I: THEORETICAL CONCEPTS")
print("============================")
print("Censoring: Incomplete observation of survival time")
print("Types: Right, Left, Interval")
print("Kaplan-Meier: S(t) = Product(1 - d_i/n_i)")
print("Cox PH: h(t) = h0(t) * exp(beta*X)")

# PART II: Kaplan-Meier Analysis
print("\nPART II: KAPLAN-MEIER ANALYSIS")
print("=============================")

kmf = KaplanMeierFitter()
kmf.fit(df['time_ascvd_yrs'], df['ascvd_event'], label='Overall Survival')

print("\nSurvival Probabilities at Key Time Points:")
for t in [1, 2, 3, 5, 7, 10]:
    prob = kmf.survival_function_at_times(t).values[0]
    print(f"  S({t} years) = {prob:.4f}")

print(f"\nMedian Survival Time: {kmf.median_survival_time_:.2f} years")

# Visualizations
plt.figure(figsize=(10, 6))
kmf.plot_survival_function(ci_show=True)
plt.xlabel('Time (Years)')
plt.ylabel('Survival Probability')
plt.title('Kaplan-Meier Survival Curve')
plt.savefig('visualizations/km_curve_python.png', dpi=150)
plt.close()

# PART IV: Cox Proportional Hazards Model
print("\nPART IV: COX PROPORTIONAL HAZARDS MODEL")
print("=======================================")

cph = CoxPHFitter()
cph.fit(df[['time_ascvd_yrs', 'ascvd_event', 'age', 'sbp_mmHg', 'diabetes']],
        duration_col='time_ascvd_yrs', event_col='ascvd_event')

print("\nCox Model Summary:")
cph.print_summary()

print("\nHazard Ratios:")
print(np.exp(cph.summary['coef']))

# PART VI: Multiple Linear Regression
print("\nPART VI: MULTIPLE LINEAR REGRESSION")
print("===================================")

from sklearn.linear_model import LinearRegression

X = df[['age', 'height_cm', 'weight_kg']]
y = df['sbp_mmHg']

lm = LinearRegression()
lm.fit(X, y)

print(f"\nCoefficients:")
for name, coef in zip(['age', 'height_cm', 'weight_kg'], lm.coef_):
    print(f"  {name}: {coef:.3f}")
print(f"  Intercept: {lm.intercept_:.3f}")
print(f"  R-squared: {lm.score(X, y):.4f}")

print("\nAnalysis Complete!")
