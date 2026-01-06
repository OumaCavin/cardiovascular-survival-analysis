################################################################################
# SDS6210_InformaticsForHealth - Data Generation
# Dataset: Hugging Face DBbun/10M_CIRCULATIONAHA.120.052430_v1.0
#
# Author: Cavin Otieno
################################################################################

# Install required packages
required_packages <- c("survival", "survminer", "ggplot2", "dplyr", "readr")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "http://cran.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# Set seed for reproducibility
set.seed(6210)

# Generate synthetic cardiovascular data based on Hugging Face structure
n <- 500

patient_data <- data.frame(
  id = 1:n,
  age = round(rnorm(n, mean = 55, sd = 12)),
  sex_male = sample(c(0, 1), n, replace = TRUE, prob = c(0.48, 0.52)),
  height_cm = round(rnorm(n, mean = 170, sd = 10)),
  weight_kg = round(rnorm(n, mean = 80, sd = 15)),
  bmi = round(rnorm(n, mean = 27.5, sd = 5), 1),
  sbp_mmHg = round(rnorm(n, mean = 130, sd = 20)),
  dbp_mmHg = round(rnorm(n, mean = 80, sd = 12)),
  hypertension = ifelse(rnorm(n, mean = 0, sd = 1) > 0.5, 1, 0),
  ldl_mgdl = round(rnorm(n, mean = 120, sd = 35)),
  hdl_mgdl = round(rnorm(n, mean = 50, sd = 15)),
  glucose_mgdl = round(rnorm(n, mean = 100, sd = 25)),
  hba1c_pct = round(rnorm(n, mean = 5.8, sd = 1.2), 1),
  diabetes = ifelse(rnorm(n, mean = 0, sd = 1) > 0.3, 1, 0),
  time_ascvd_yrs = round(runif(n, min = 0.5, max = 15), 2),
  ascvd_event = sample(c(0, 1), n, replace = TRUE, prob = c(0.8, 0.2))
)

# Ensure realistic ranges
patient_data$age <- pmax(patient_data$age, 30)
patient_data$age <- pmin(patient_data$age, 90)
patient_data$sbp_mmHg <- pmax(patient_data$sbp_mmHg, 90)
patient_data$sbp_mmHg <- pmin(patient_data$sbp_mmHg, 200)

# Save to CSV
write.csv(patient_data, "data/patient_survival_data.csv", row.names = FALSE)

cat("Data saved to data/patient_survival_data.csv\n")
cat(paste("Generated", n, "patient records\n"))
