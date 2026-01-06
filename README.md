# Cardiovascular Survival Analysis

## SDS6210_InformaticsForHealth - Master of Public Health Data Science

**Author:** Cavin Otieno  
**Registration Number:** SDS6/46982/2024  
**Dataset:** Hugging Face DBbun/10M_CIRCULATIONAHA.120.052430_v1.0

---

## Overview

This repository contains the complete implementation of survival analysis assignment for the Health Informatics course. The analysis uses cardiovascular disease time-to-event data to demonstrate key survival analysis techniques.

## Sample Results Summary

### Data Generation
```
Generated 500 patient records
  Age: 54.8 ± 12.0 years
  SBP: 131.3 ± 19.1 mmHg
  Diabetes: 208 (41.6%)
  ASCVD Events: 111 (22.2%)
```

### Kaplan-Meier Survival Analysis
```
Survival Probabilities:
  S(1 year) = 0.9939
  S(2 years) = 0.9833
  S(3 years) = 0.9632
  S(5 years) = 0.9390
  S(7 years) = 0.9041
  S(10 years) = 0.8366

Median Survival Time: 14.05 years
```

### Cox Proportional Hazards Model
```
Hazard Ratios:
  Age: 1.001 (p = 0.91)
  SBP: 1.007 (p = 0.14)
  Diabetes: 0.920 (p = 0.67)

Concordance Index: 0.54
```

### Machine Learning Classification (with SMOTE)
```
Random Forest Performance:
  Accuracy: 95.0%
  Precision: 89.5%
  Recall: 85.7%
  F1-Score: 87.6%
  ROC-AUC: 0.95

Top Feature Importances:
  1. age (0.198)
  2. sbp_mmHg (0.194)
  3. height_cm (0.135)
  4. glucose_mgdl (0.116)
  5. ldl_mgdl (0.088)
```

### Multiple Linear Regression
```
Model: SBP ~ Age + Height_cm + Weight_kg

R-squared: 0.0009
Coefficients:
  Age: -0.005
  Height: -0.047
  Weight: -0.023
```

---

## Repository Structure

```
OumaCavin/
├── Python_code/
│   ├── data_generation.py
│   ├── survival_analysis.py
│   └── ml_classification.py
├── R_code/
│   ├── data_generation.R
│   └── survival_analysis.R
├── data/
│   └── (dataset files)
├── docs/
│   ├── ASSIGNMENT_PLAN.md
│   └── theoretical_concepts.md
├── README.md
└── requirements.txt
```

## Assignment Parts

### Part I: Theoretical Concepts
- Censoring in survival analysis
- Types and causes of censoring
- Features of survival data

### Part II: Kaplan-Meier Analysis
- Survival object creation
- Kaplan-Meier curve fitting
- Visualization with and without confidence intervals

### Part III: Data Management
- Dataset creation and import
- Manual survival probability calculations

### Part IV: Cox Proportional Hazards Model
- Model fitting and interpretation
- Hazard function: h(t) = h₀(t) × exp(β₁X₁ + β₂X₂ + ...)

### Part V: Manual Kaplan-Meier Calculation
- Complete table for time: 2, 3+, 6, 6, 7, 10+, 15, 15, 16, 27, 30, 32

### Part VI: Multiple Linear Regression
- Model: Blood Pressure ~ Age + Height_cm + Weight_kg

### Part VII: Python Implementation with ML
- Complete replication of R analysis
- Machine learning algorithms (Logistic Regression, Random Forest)
- SMOTE for class imbalance handling
- Performance metrics and confusion matrix

### Part VIII: Geographic Visualization
- County-level choropleth maps

## Dataset

**Source:** [Hugging Face DBbun/10M_CIRCULATIONAHA.120.052430_v1.0](https://huggingface.co/datasets/DBbun/10M_CIRCULATIONAHA.120.052430_v1.0)

**Features:**
- Demographics: age, sex, ancestry
- Anthropometrics: height, weight, BMI
- Blood pressure: systolic, diastolic, hypertension
- Lipid profile: LDL, HDL, triglycerides
- Metabolic markers: glucose, HbA1c, diabetes
- Time-to-event outcomes for ASCVD, HF, AF

## Usage

### R
```r
source("R_code/data_generation.R")  # Generate dataset
source("R_code/survival_analysis.R")  # Run survival analysis
```

### Python
```bash
# Setup
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run analyses
python Python_code/data_generation.py    # Generate dataset
python Python_code/survival_analysis.py  # Survival analysis
python Python_code/ml_classification.py  # ML classification
```

## Requirements

See `requirements.txt` for Python dependencies.

---

*Assignment completed for SDS6210_InformaticsForHealth*
*Master of Public Health Data Science*
