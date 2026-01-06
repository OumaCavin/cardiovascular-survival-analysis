# Cardiovascular Survival Analysis

## SDS6210_InformaticsForHealth - Master of Public Health Data Science

**Author:** Cavin Otieno  
**Dataset:** Hugging Face DBbun/10M_CIRCULATIONAHA.120.052430_v1.0

---

## Overview

This repository contains the complete implementation of survival analysis assignment for the Health Informatics course. The analysis uses cardiovascular disease time-to-event data to demonstrate key survival analysis techniques.

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
- Machine learning algorithms
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
source("R_code/survival_analysis.R")
```

### Python
```bash
python Python_code/survival_analysis.py
python Python_code/ml_classification.py
```

## Requirements

See `requirements.txt` for Python dependencies.

---

*Assignment completed for SDS6210_InformaticsForHealth*
