"""
SDS6210_InformaticsForHealth - Data Generation
Dataset: Hugging Face DBbun/10M_CIRCULATIONAHA.120.052430_v1.0

Author: Cavin Otieno
"""

import numpy as np
import pandas as pd
import os

print("=" * 60)
print("Generating Cardiovascular Dataset")
print("=" * 60)

np.random.seed(6210)
os.makedirs('data', exist_ok=True)

n = 500

data = {
    'id': range(1, n + 1),
    'age': np.round(np.random.normal(55, 12, n)).astype(int),
    'sex_male': np.random.choice([0, 1], n, p=[0.48, 0.52]),
    'height_cm': np.round(np.random.normal(170, 10, n)).astype(int),
    'weight_kg': np.round(np.random.normal(80, 15, n)).astype(int),
    'bmi': np.round(np.random.normal(27.5, 5, n), 1),
    'sbp_mmHg': np.round(np.random.normal(130, 20, n)).astype(int),
    'dbp_mmHg': np.round(np.random.normal(80, 12, n)).astype(int),
    'hypertension': (np.random.normal(0, 1, n) > 0.5).astype(int),
    'ldl_mgdl': np.round(np.random.normal(120, 35, n)).astype(int),
    'hdl_mgdl': np.round(np.random.normal(50, 15, n)).astype(int),
    'glucose_mgdl': np.round(np.random.normal(100, 25, n)).astype(int),
    'hba1c_pct': np.round(np.random.normal(5.8, 1.2, n), 1),
    'diabetes': (np.random.normal(0, 1, n) > 0.3).astype(int),
    'time_ascvd_yrs': np.round(np.random.uniform(0.5, 15, n), 2),
    'ascvd_event': np.random.choice([0, 1], n, p=[0.8, 0.2])
}

df = pd.DataFrame(data)
df['age'] = df['age'].clip(30, 90)
df['sbp_mmHg'] = df['sbp_mmHg'].clip(90, 200)

df.to_csv('data/patient_survival_data.csv', index=False)

print(f"\nGenerated {n} patient records")
print(f"Saved to data/patient_survival_data.csv")
print(f"\nDataset Summary:")
print(f"  Age: {df['age'].mean():.1f} ± {df['age'].std():.1f} years")
print(f"  SBP: {df['sbp_mmHg'].mean():.1f} ± {df['sbp_mmHg'].std():.1f} mmHg")
print(f"  Diabetes: {df['diabetes'].sum()} ({df['diabetes'].mean()*100:.1f}%)")
print(f"  Events: {df['ascvd_event'].sum()} ({df['ascvd_event'].mean()*100:.1f}%)")
