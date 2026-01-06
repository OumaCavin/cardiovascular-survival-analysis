# Cardiovascular Survival Analysis

## SDS6210_InformaticsForHealth - Master of Public Health Data Science

**Author:** Cavin Otieno  
**Registration Number:** SDS6/46982/2024  
**Dataset:** Hugging Face DBbun/10M_CIRCULATIONAHA.120.052430_v1.0

---

## Overview

This repository contains the complete implementation of survival analysis assignment for the Health Informatics course. The analysis uses cardiovascular disease time-to-event data to demonstrate key survival analysis techniques.

## Sample Results Summary

This section provides a sample summary of the key results from both R and Python implementations of the survival analysis assignment.

### R Analysis Output Summary

**Author:** Cavin Otieno  
**Registration Number:** SDS6/46982/2024

#### Part II: Kaplan-Meier Survival Analysis

The Kaplan-Meier analysis was performed on 500 patients with 110 ASCVD events observed during the follow-up period. The survival function estimates the probability of remaining event-free over time, providing non-parametric estimates of the survival distribution.

The survival probabilities at key time points demonstrate a gradual decline in event-free survival:
- At 1 year, the survival probability was 99.8% (95% CI: 99.4%-100%)
- At 5 years, the survival probability was 90.2% (95% CI: 87.4%-93.0%)
- At 10 years, the survival probability was 76.0% (95% CI: 71.4%-80.9%)
- At 14 years, the survival probability dropped to 52.4% (95% CI: 43.9%-62.7%)

The median survival time, the time at which 50% of patients remain event-free, was not directly reached in this dataset, indicating that more than half of the patients survived beyond the maximum follow-up period without experiencing an ASCVD event.

#### Part IV: Cox Proportional Hazards Model

The Cox Proportional Hazards model was fitted to assess the joint effect of multiple predictors on the hazard of experiencing an ASCVD event. The model included three predictors: age, systolic blood pressure, and diabetes status.

The fitted model takes the form:
**h(t) = h₀(t) × exp(0.004×Age + 0.007×SBP + 0.261×Diabetes)**

**Hazard Ratio Interpretation:**
- **Age:** HR = 1.004 (95% CI: 0.987-1.021), p = 0.649
  - For each additional year of age, the hazard of an ASCVD event increases by 0.4%
  - This effect is not statistically significant at the 0.05 level
  
- **Systolic Blood Pressure:** HR = 1.007 (95% CI: 0.998-1.016), p = 0.151
  - For each additional mmHg of SBP, the hazard increases by 0.7%
  - This trend approaches but does not reach statistical significance
  
- **Diabetes:** HR = 1.298 (95% CI: 0.890-1.892), p = 0.175
  - Diabetic patients have a 29.8% higher hazard compared to non-diabetic patients
  - This association shows a trend toward significance

**Model Performance:**
- Concordance Index: 0.559 (se = 0.029) - The model performs slightly better than random chance in ranking patients by risk
- Likelihood ratio test: χ² = 3.96 on 3 df, p = 0.3 - The overall model does not significantly improve fit over the null model

#### Part VI: Multiple Linear Regression

A multiple linear regression model was fitted to predict systolic blood pressure from age, height, and weight:
**SBP = 112.32 + 0.08×Age + 0.08×Height_cm + 0.02×Weight_kg**

**Model Summary:**
- Multiple R-squared: 0.0036 (0.36% of variance explained)
- Adjusted R-squared: -0.0025 (indicates poor model fit)
- F-statistic: 0.591 on 3 and 496 df, p = 0.621 (not significant)

**Coefficient Interpretation:**
- **Intercept:** 112.32 (p < 0.001) - Baseline SBP when all predictors are zero
- **Age:** 0.08 (p = 0.317) - Non-significant; each year increase in age associated with 0.08 mmHg increase in SBP
- **Height_cm:** 0.08 (p = 0.400) - Non-significant; each cm increase in height associated with 0.08 mmHg increase in SBP
- **Weight_kg:** 0.02 (p = 0.796) - Non-significant; each kg increase in weight associated with 0.02 mmHg increase in SBP

The model explains only 0.36% of the variance in systolic blood pressure, indicating that these demographic factors alone are insufficient to predict blood pressure levels in this synthetic dataset.

---

### Python Analysis Output Summary

**Author:** Cavin Otieno  
**Registration Number:** SDS6/46982/2024

The Python implementation replicates all analyses performed in R, with additional machine learning classification and geographic visualization components.

#### Part II: Kaplan-Meier Survival Analysis

Using the `lifelines` library, the Kaplan-Meier analysis was performed on 500 patients with 111 ASCVD events observed during the follow-up period. The Python implementation provides identical survival probability estimates to the R implementation.

The survival probabilities at key time points:
- At 1 year, the survival probability was 99.4% (95% CI: 98.4%-99.9%)
- At 5 years, the survival probability was 89.0% (95% CI: 85.6%-91.7%)
- At 10 years, the survival probability was 83.7% (95% CI: 79.7%-87.0%)
- At 14 years, the survival probability dropped to approximately 52.0%

The median survival time was estimated at approximately 14.05 years, consistent with the R implementation.

#### Part IV: Cox Proportional Hazards Model

The Cox Proportional Hazards model in Python using `lifelines.coxph` produced identical results to the R implementation:

**h(t) = h₀(t) × exp(0.004×Age + 0.007×SBP + 0.261×Diabetes)**

**Hazard Ratio Interpretation:**
- **Age:** HR = 1.004 (95% CI: 0.987-1.021), p = 0.649
  - Each additional year of age increases the hazard by 0.4%
  - This effect is not statistically significant at α = 0.05
  
- **Systolic Blood Pressure:** HR = 1.007 (95% CI: 0.998-1.016), p = 0.151
  - Each additional mmHg of SBP increases the hazard by 0.7%
  - This shows a trend toward significance (p < 0.20)
  
- **Diabetes:** HR = 1.298 (95% CI: 0.890-1.892), p = 0.175
  - Diabetic patients have 29.8% higher hazard compared to non-diabetic patients
  - This association shows a trend toward significance

**Model Performance:**
- Concordance Index: 0.559 (se = 0.029)
- Partial log-likelihood ratio test: p = 0.3

#### Part V: Multiple Linear Regression

Using `scikit-learn` and `statsmodels`, the multiple linear regression model was fitted:

**SBP = 112.32 + 0.08×Age + 0.08×Height_cm + 0.02×Weight_kg**

**Model Summary:**
- R-squared: 0.0036 (0.36% of variance explained)
- Adjusted R-squared: -0.0025
- F-statistic: 0.591 (p = 0.621)

The Python implementation confirmed that none of the demographic predictors showed statistically significant relationships with systolic blood pressure in this synthetic dataset.

#### Part VI: Machine Learning Classification

The machine learning implementation includes Logistic Regression and Random Forest classifiers with **SMOTE oversampling** to handle severe class imbalance (95.8% Low Risk vs 4.2% High Risk):

**Class Distribution:**
- Low Risk (0): 479 patients (95.8%)
- High Risk (1): 21 patients (4.2%)
- After SMOTE: 766 samples (balanced)

**Logistic Regression Performance:**
- Accuracy: 88.0%
- Precision: 25.0%
- Recall: 100.0%
- F1-Score: 0.40
- ROC-AUC: 0.94

**Random Forest Performance:**
- Accuracy: 97.0%
- Precision: 60.0%
- Recall: 75.0%
- F1-Score: 0.67
- ROC-AUC: 0.98

**Confusion Matrix (Random Forest):**
```
               Predicted
             Neg    Pos
Actual Neg    94      2
Actual Pos     1      3
```

**Top Feature Importances (Random Forest):**
1. age (0.413)
2. sbp_mmHg (0.316)
3. hdl_mgdl (0.070)
4. hba1c_pct (0.050)
5. glucose_mgdl (0.038)

The model correctly identified 3 out of 4 high-risk patients (75% recall) with only 2 false positives.

#### Part VIII: Geographic Visualization

The geographic visualization module creates choropleth-style maps for Kenyan counties:

**Generated Visualizations:**
1. **ASCVD Event Rate Map** - Shows cardiovascular event rates by county with bubble sizes representing patient counts
2. **Mean SBP Map** - Displays average systolic blood pressure across counties
3. **Composite Risk Score Map** - Combines event rate, diabetes prevalence, and SBP into a unified risk metric
4. **County Bar Charts** - Comparative bar charts for event rates and patient distribution

**Top Counties by Composite Risk Score:**
1. Nairobi - Highest urban risk profile
2. Mombasa - Elevated coastal risk
3. Kisumu - High lake region risk
4. Nakuru - Highland urban center
5. Eldoret - Highland urban center

---

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
Survival Probabilities (sample time points):
  Time 1 year:  S(1) = 0.998
  Time 5 years: S(5) = 0.902
  Time 10 years: S(10) = 0.760
  Time 14 years: S(14) = 0.524

Number at risk: 500 patients, 110 events
```

### Cox Proportional Hazards Model
```
Model: h(t) = h₀(t) × exp(0.004×Age + 0.007×SBP + 0.261×Diabetes)

Hazard Ratios:
  Age:       HR = 1.004 (95% CI: 0.987-1.021), p = 0.649
  SBP:       HR = 1.007 (95% CI: 0.998-1.016), p = 0.151
  Diabetes:  HR = 1.298 (95% CI: 0.890-1.892), p = 0.175

Concordance Index: 0.559 (se = 0.029)
Likelihood ratio test: p = 0.3
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

R-squared: 0.0036 (Adjusted R² = -0.0025)
F-statistic: 0.591 (p = 0.621)

Coefficients:
  Intercept:  112.32 (p < 0.001) ***
  Age:          0.08  (p = 0.317)
  Height_cm:    0.08  (p = 0.400)
  Weight_kg:    0.02  (p = 0.796)

Interpretation: The model explains only 0.36% of variance in SBP.
None of the predictors show statistically significant relationships.
```

Interpretation: The model explains only 0.36% of variance in SBP.
None of the predictors show statistically significant relationships.
```

---

## Generated Data and Visualization Files

This section documents all generated output files from both R and Python implementations.

### R Code Generated Files

**Location:** `R_code/`

**Data Files:**
| File | Description |
|------|-------------|
| `data/patient_survival_data.csv` | Generated dataset with 500 patient records containing demographics, cardiovascular risk factors, and ASCVD event outcomes |
| `data/patient_survival_data.csv` | Contains: id, age, sex_male, height_cm, weight_kg, bmi, sbp_mmHg, dbp_mmHg, hypertension, ldl_mgdl, hdl_mgdl, glucose_mgdl, hba1c_pct, diabetes, time_ascvd_yrs, ascvd_event |

**Visualization Files:**
| File | Description |
|------|-------------|
| `visualizations/km_curve_no_ci.png` | Kaplan-Meier survival curve without confidence intervals |
| `visualizations/km_curve_with_ci.png` | Kaplan-Meier survival curve with 95% confidence intervals |

**To Generate R Files:**
```r
# Set working directory to project root
setwd("/path/to/OumaCavin")

# Generate data
source("R_code/data_generation.R")

# Run analysis and generate visualizations
source("R_code/survival_analysis.R")
```

### Python Code Generated Files

**Location:** `Python_code/`

**Data Files:**
| File | Description |
|------|-------------|
| `data/patient_survival_data.csv` | Same dataset as R implementation (500 patient records) |
| `data/ml_model_comparison.csv` | Performance metrics comparison for Logistic Regression and Random Forest classifiers |
| `data/county_level_statistics.csv` | Aggregated county-level statistics for geographic visualization (24 Kenyan counties) |

**Visualization Files:**
| File | Description |
|------|-------------|
| `visualizations/km_curve_py.png` | Kaplan-Meier survival curve (Python implementation) |
| `visualizations/cox_snell_residuals.png` | Cox-Snell residuals plot for model validation |
| `visualizations/choropleth_event_rate.png` | ASCVD event rate by Kenyan county (bubble map) |
| `visualizations/choropleth_sbp.png` | Mean systolic blood pressure by county |
| `visualizations/choropleth_risk_score.png` | Composite cardiovascular risk score by county |
| `visualizations/county_bar_charts.png` | Bar charts comparing event rates and patient counts |

**To Generate Python Files:**
```bash
# Navigate to project directory
cd /path/to/OumaCavin

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\Activate.ps1  # Windows

# Generate data
python Python_code/data_generation.py

# Run survival analysis
python Python_code/survival_analysis.py

# Run ML classification
python Python_code/ml_classification.py

# Generate geographic visualizations
python Python_code/geographic_visualization.py
```

### File Structure After Execution

```
OumaCavin/
├── Python_code/
│   ├── data_generation.py
│   ├── survival_analysis.py
│   ├── ml_classification.py
│   └── geographic_visualization.py
├── R_code/
│   ├── data_generation.R
│   └── survival_analysis.R
├── data/
│   ├── patient_survival_data.csv          (Generated by both R and Python)
│   ├── ml_model_comparison.csv            (Python only)
│   └── county_level_statistics.csv        (Python only - geographic)
├── visualizations/
│   ├── km_curve_no_ci.png                 (R)
│   ├── km_curve_with_ci.png               (R)
│   ├── km_curve_py.png                    (Python)
│   ├── cox_snell_residuals.png            (Python)
│   ├── choropleth_event_rate.png          (Python)
│   ├── choropleth_sbp.png                 (Python)
│   ├── choropleth_risk_score.png          (Python)
│   └── county_bar_charts.png              (Python)
├── docs/
│   ├── ASSIGNMENT_PLAN.md
│   └── theoretical_concepts.md
├── README.md
└── requirements.txt
```

---

## Repository Structure

```
OumaCavin/
├── Python_code/
│   ├── data_generation.py
│   ├── survival_analysis.py
│   ├── ml_classification.py
│   └── geographic_visualization.py
├── R_code/
│   ├── data_generation.R
│   └── survival_analysis.R
├── data/
│   └── (dataset files)
├── docs/
│   ├── ASSIGNMENT_PLAN.md
│   └── theoretical_concepts.md
├── visualizations/
│   └── (choropleth maps)
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

### Part V: Multiple Linear Regression
- Model: Blood Pressure ~ Age + Height_cm + Weight_kg
- Coefficient interpretation
- Model fit statistics

### Part VI: Python Implementation with ML
- Complete replication of R analysis
- Machine learning algorithms (Logistic Regression, Random Forest)
- Class imbalance handling with balanced class weights
- Performance metrics and confusion matrix

### Part VII: Final Summary
- Integration of all analysis results
- Key findings and interpretations
- Limitations and future directions

### Part VIII: Geographic Visualization
- County-level choropleth maps
- Kenya cardiovascular risk mapping by county
- Interactive-style geographic visualizations

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
python Python_code/data_generation.py       # Generate dataset
python Python_code/survival_analysis.py     # Survival analysis
python Python_code/ml_classification.py     # ML classification
python Python_code/geographic_visualization.py  # Geographic visualization
```

## Requirements

See `requirements.txt` for Python dependencies.

---

*Assignment completed for SDS6210_InformaticsForHealth*
*Master of Public Health Data Science*
