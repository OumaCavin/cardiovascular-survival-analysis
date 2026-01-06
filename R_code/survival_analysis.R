################################################################################
# SDS6210_InformaticsForHealth - Survival Analysis Assignment
# Part II & IV: Kaplan-Meier Analysis and Cox Proportional Hazards Model
#
# Author: Cavin Otieno
################################################################################

# Load libraries
library(survival)
library(survminer)

# Theoretical concepts
cat("PART I: THEORETICAL CONCEPTS\n")
cat("============================\n")
cat("Censoring: Incomplete observation of survival time\n")
cat("Types: Right, Left, Interval\n")
cat("Kaplan-Meier: S(t) = Product(1 - d_i/n_i)\n")
cat("Cox PH: h(t) = h0(t) * exp(beta*X)\n\n")

# Load data
patient_data <- read.csv("data/patient_survival_data.csv")

# PART II: Kaplan-Meier Analysis
cat("PART II: KAPLAN-MEIER SURVIVAL ANALYSIS\n")
cat("======================================\n")

# Create survival object
surv_object <- Surv(time = patient_data$time_ascvd_yrs,
                    event = patient_data$ascvd_event)

# Fit Kaplan-Meier
km_fit <- survfit(surv_object ~ 1, data = patient_data)

# Print summary
cat("\nSurvival Summary:\n")
print(summary(km_fit))

# Create visualizations directory if it doesn't exist
if (!dir.exists("visualizations")) {
  dir.create("visualizations")
}

# Visualizations
# 1. Kaplan-Meier curve without confidence intervals
png("visualizations/km_curve_no_ci.png", width = 800, height = 600)
plot(km_fit, conf.int = FALSE, xlab = "Time (Years)",
     ylab = "Survival Probability",
     main = "Kaplan-Meier Survival Curve (No CI)")
dev.off()

# 2. Kaplan-Meier curve with 95% confidence intervals
png("visualizations/km_curve_with_ci.png", width = 800, height = 600)
plot(km_fit, conf.int = TRUE, xlab = "Time (Years)",
     ylab = "Survival Probability",
     main = "Kaplan-Meier Survival Curve (With 95% CI)")
dev.off()

# 3. Kaplan-Meier curve with marked time points (mark-up time visualization)
png("visualizations/km_curve_marked.png", width = 800, height = 600)
plot(km_fit, conf.int = TRUE, xlab = "Time (Years)",
     ylab = "Survival Probability",
     main = "Kaplan-Meier Survival Curve (Marked Time Points)",
     mark.time = TRUE,  # Mark the event times on the curve
     col.mark = "red",   # Color for the marks
     cex.mark = 0.8)     # Size of the marks

# Add vertical lines at key time points
abline(v = c(1, 5, 10), col = "gray", lty = 2, lwd = 1)
text(c(1, 5, 10), c(0.95, 0.85, 0.75), 
     labels = c("1 yr", "5 yr", "10 yr"), 
     pos = 3, cex = 0.8)
dev.off()

# PART IV: Cox Proportional Hazards Model
cat("\nPART IV: COX PROPORTIONAL HAZARDS MODEL\n")
cat("======================================\n")

# Fit Cox model
cox_model <- coxph(surv_object ~ age + sbp_mmHg + diabetes,
                   data = patient_data)

cat("\nCox Model Summary:\n")
print(summary(cox_model))

# Hazard ratios
cat("\nHazard Ratios:\n")
print(exp(coef(cox_model)))

# PART VI: Multiple Linear Regression
cat("\nPART VI: MULTIPLE LINEAR REGRESSION\n")
cat("==================================\n")

lm_model <- lm(sbp_mmHg ~ age + height_cm + weight_kg, data = patient_data)
cat("\nLinear Regression Summary:\n")
print(summary(lm_model))

cat("\nAnalysis Complete!\n")
