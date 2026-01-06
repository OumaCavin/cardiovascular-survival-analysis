# Assignment Plan: SDS6210_InformaticsForHealth

## Master of Public Health Data Science - Survival Analysis Assignment

### Assignment Overview

This assignment covers comprehensive survival analysis techniques including theoretical concepts, manual calculations, and implementations in both R and Python.

---

## Part I: Theoretical Concepts

### 1.1 Censoring in Survival Analysis

**Definition**: Censoring occurs when information about the subject's survival time is incomplete.

**Causes of Censoring**:
- Subject withdraws from the study
- Subject is lost to follow-up
- Study ends before the event occurs
- Subject dies from unrelated cause

**Types of Censoring**:
1. **Right Censoring**: Event time is known to be greater than a certain value
2. **Left Censoring**: Event time is known to be less than a certain value
3. **Interval Censoring**: Event time is known to fall within an interval

---

## Part II: R Implementation - Kaplan-Meier Analysis

### Data Structure
- `Time`: Survival/censoring time
- `Event`: Indicator (1=event occurred, 0=censored)

### Required Packages
```r
install.packages("survival")
install.packages("survminer")
```

---

## Part III: Data Management

### Manual Calculation (Kaplan-Meier Table)

**Given Dataset**: Time: 2, 3+, 6, 6, 7, 10+, 15, 15, 16, 27, 30, 32

| Time | At Risk | Died | HR | Survival Probability |
|------|---------|------|-----|---------------------|
| 2    | 12      | 1    | 0.083 | 0.917 |
| 3+   | 11      | 0    | 0.000 | 0.917 |
| 6    | 10      | 2    | 0.200 | 0.733 |
| 7    | 7       | 1    | 0.143 | 0.629 |
| 10+  | 6       | 0    | 0.000 | 0.629 |
| 15   | 5       | 2    | 0.400 | 0.377 |
| 16   | 2       | 1    | 0.500 | 0.189 |
| 27   | 1       | 0    | 0.000 | 0.189 |
| 30   | 1       | 0    | 0.000 | 0.189 |
| 32   | 1       | 1    | 1.000 | 0.000 |

---

## Part IV: Cox Proportional Hazards Model

### Model Formula
h(t) = h₀(t) × exp(β₁X₁ + β₂X₂ + β₃X₃ + ...)

---

## Part V: Multiple Linear Regression

### Model Specification
Blood_Pressure ~ Age + Height_cm + Weight_kg

---

## Part VI: Python Implementation with ML

### ML Components
- Logistic Regression
- Random Forest
- Performance Metrics
- Confusion Matrix

---

## Dataset

**Source**: Hugging Face DBbun/10M_CIRCULATIONAHA.120.052430_v1.0

*Author: Cavin Otieno*
