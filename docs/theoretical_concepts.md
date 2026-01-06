# Theoretical Concepts in Survival Analysis

## SDS6210_InformaticsForHealth - Master of Public Health Data Science

**Author: Cavin Otieno**

---

## 1. Censoring in Survival Analysis

### 1.1 Definition

Censoring occurs when information about the subject's survival time is incomplete. The subject leaves the study before the event of interest occurs, or the study ends before the event occurs for all subjects.

### 1.2 Causes of Censoring

- Subject withdraws from the study
- Subject is lost to follow-up
- Study ends before all events occur
- Subject dies from unrelated cause

### 1.3 Types of Censoring

**Right Censoring (Most Common)**: The event time is known to be greater than a certain value.

**Left Censoring**: The event time is known to be less than a certain value.

**Interval Censoring**: The event time is known to fall within a specific interval.

### 1.4 Features of Survival Data

- Time-to-event outcome (non-negative, often right-skewed)
- Censoring indicator (1=event occurred, 0=censored)
- Covariates/predictors
- Non-increasing survival function S(t)
- Hazard function h(t) represents instantaneous risk

---

## 2. Kaplan-Meier Estimator

### 2.1 Formula

S(t) = ∏(1 - dᵢ/nᵢ) for all tᵢ ≤ t

Where:
- tᵢ = ordered distinct event times
- dᵢ = number of events at time tᵢ
- nᵢ = number at risk at time tᵢ

### 2.2 Properties

- Non-parametric estimator
- Step function with jumps at event times
- Naturally handles right-censored data
- Confidence intervals via Greenwood's formula

---

## 3. Cox Proportional Hazards Model

### 3.1 Model Formulation

h(t) = h₀(t) × exp(β₁X₁ + β₂X₂ + ... + βₙXₙ)

### 3.2 Key Concepts

**Hazard Ratio (HR)**: exp(β) represents the hazard ratio comparing groups.

**Proportional Hazards Assumption**: The hazard ratio is constant over time.

---

## References

1. Cox, D.R. (1972). Regression Models and Life-Tables. JRSS B.
2. Kaplan, E.L., Meier, P. (1958). Nonparametric Estimation. JASA.

*Document prepared for SDS6210_InformaticsForHealth*
