# -------------------------------------------------------------------
# Model non-investigation rates and colorectal investigation findings
# by age group, sex, deprivation (IMD), and screening history
# 
# Motivation: identify *potential* two-way interactions.
# These can later be checked by visualisation or by looking at outcome
# proportions by cell, to understand whether they are also clinically
# and not just statistically significant.
# -------------------------------------------------------------------
#install.packages("mgcv", version="1.9-4")
#install.packages("data.table", version="1.17.8")
#install.packages('dplyr', version="1.1.4")
#library(mgcv)
library(data.table)
library(dplyr)


out_path <- "Z:/andres/nhsebcss/results/primary"


# Helper: fit a logistic model containing all six two-way interaction groups, then test
# each interaction group by dropping it from the full model and comparing the reduced
# model with the full model using a likelihood-ratio test (LRT). An interaction is kept
# if dropping it significantly worsens fit (p < alpha), and dropped otherwise. Every
# interaction is tested against the same full model (no forward/backward stepwise path).
# Args:
#  outcome : name of the binary outcome column in dataframe 'data'
#  data    : dataframe containing the outcome and predictor columns
#  alpha   : significance threshold (p-value) for the LRT test
# Returns:
#  a list with elements
#   `model`    (fitted model with the kept interactions),
#   `kept`     (character vector of retained interactions),
#   `kept_str` (character string of retained interactions)
# Created originally with Claude Code (and Opus 4.8), then verified/modified.
select_interactions <- function(outcome, data, alpha=0.05){
  base_terms <- "age_group_granular + subject_gender + prevalent_incident_status + imd_quintile"

  # Fit the main-effects model plus any extra terms
  fit_with <- function(extra){
    rhs <- base_terms
    if(length(extra) > 0){
      rhs <- paste(rhs, "+", paste(extra, collapse=" + "))
    }
    glm(as.formula(paste(outcome, "~ 1 +", rhs)), data=data, family=binomial())
  }

  # Interaction terms to test
  interactions <- c("age_group_granular:subject_gender",
                    "age_group_granular:imd_quintile",
                    "age_group_granular:prevalent_incident_status",
                    "imd_quintile:subject_gender",
                    "imd_quintile:prevalent_incident_status",
                    "subject_gender:prevalent_incident_status")

  # Base model is the full model with all six interaction groups. Each interaction is
  # then tested by dropping it from the full model (likelihood-ratio test).
  cat(sprintf("\n=== Selecting two-way interactions for '%s' ===\n", outcome))
  fit_full <- fit_with(interactions)
  kept <- character(0)
  for(term in interactions){
    fit_drop <- fit_with(setdiff(interactions, term))          # full model minus this interaction
    pval <- anova(fit_drop, fit_full, test="Chisq")[["Pr(>Chi)"]][2]
    keep <- !is.na(pval) && pval < alpha                       # keep if dropping worsens fit
    cat(sprintf("  %-48s p = %-12.4g %s\n", gsub(":", " * ", term), pval,
                if(keep) "KEPT" else "dropped"))
    if(keep){
      kept <- c(kept, term)
    }
  }

  fit_final <- fit_with(kept)

  kept_str <- paste(gsub(":", " * ", kept), collapse=", ")
  if(nchar(kept_str) == 0) kept_str <- "(none)"
  cat(sprintf("Interactions kept for '%s': %s\n", outcome, kept_str))

  return(list(model=fit_final, kept=kept, kept_str=kept_str))
}


# ---- Data prep ----

# Read data
data_path <- "C:/Users/andres.Tamm/Desktop/bcss_clean_data/episodes_clean.csv"
df <- data.table::fread(data_path)
nrow(df)

# Keep episodes with positive FIT
outcomes_without_positive_fit <- c('FIT negative', 
                                   'No FIT result')
mask <- df$outcome %in% outcomes_without_positive_fit
sum(mask)
df <- df[!mask,]
nrow(df)  # 351,359

for(i in 1:10){  # Free up memory after reducing the dataset
  gc()
}

# Drop episodes with missing IMD data
mask <- df$imd_quintile == ""
sum(mask)  ## 532
mean(mask) * 100  ## 0.15
df <- df[!mask,]
nrow(df)  # 350,827

s <- df %>% group_by(imd_quintile) %>% summarise(count=n())
min(s$count)  # min 64,043

# Drop ages <55 as those not present for all screening histories
table(df[df$prevalent_incident_status == 'Prevalent',]$subject_age_at_episode_start)
table(df[df$prevalent_incident_status == 'Incident',]$subject_age_at_episode_start)
mask <- df$subject_age_at_episode_start < 55
df <- df[!mask, ]
nrow(df)  # 346,070

# Add granular age groups
age_max = max(df$subject_age_at_episode_start) + 1
breaks = c(55, 60, 65, 70, 75, age_max)
labels = c("55-59", "60-64", "65-69", "70-74", "75+")
df$age_group_granular <- cut(df$subject_age_at_episode_start, breaks=breaks, labels=labels, right=FALSE)
table(df$age_group_granular)

# Categorical variables to factors
df$age_group_granular <- as.factor(df$age_group_granular)
df$age_group_granular <- relevel(df$age_group_granular, ref="55-59")

df$subject_gender <- as.factor(df$subject_gender)
df$subject_gender <- relevel(df$subject_gender, ref="Female")

df$prevalent_incident_status <- as.factor(df$prevalent_incident_status)
df$prevalent_incident_status <- relevel(df$prevalent_incident_status, ref="Prevalent")

df$imd_quintile <- as.factor(df$imd_quintile)
df$imd_quintile <- relevel(df$imd_quintile, ref="05 - Least deprived")

# Define some required outcome variables (indicator for advanced polyps is already in the df)
df$crc <- ifelse(df$outcome == "Colorectal cancer", 1, 0)
df$acp <- ifelse(df$outcome == "Advanced premalignant polyp", 1, 0)
df$no_investigation <- ifelse(df$outcome == 'FIT positive, no investigation', 1, 0)

# Use simple outcome cats
df$outcome <- df$outcome_simple


# ---- Model non-investigation rate ----
sum(df$no_investigation)  # 74,824

# Main effects model
fit0 <- glm(no_investigation ~ 1 + age_group_granular + subject_gender + prevalent_incident_status + imd_quintile,
            data=df, family=binomial())

# Check which two-way interactions can be dropped from a full model with main effects and all six interaction groups
noinv_interactions <- select_interactions("no_investigation", df)
noinv_model <- noinv_interactions$model
noinv_kept  <- noinv_interactions$kept_str


# ---- Model CRC rate ----

# Keep episodes with colorectal investigation
outcomes_without_investigation <- c('FIT negative', 
                                    'No FIT result', 
                                    'FIT positive, no investigation',
                                    'FIT positive, unknown outcome')
mask <- df$outcome %in% outcomes_without_investigation
sum(mask)
df <- df[!mask,]
nrow(df)     # 268,147
sum(df$crc)  # 23,410
sum(df$acp)  # 83,149
for(i in 1:10){
  gc()
}

s <- df %>% group_by(imd_quintile) %>% summarise(count=n())
min(s$count)  ## min 44,948

# Main effects model
fit0 <- glm(crc ~ 1 + age_group_granular + subject_gender + prevalent_incident_status + imd_quintile,
            data=df, family=binomial())
summary(fit0)

# Check which two-way interactions can be dropped from a full model with main effects and all six interaction groups
crc_interactions <- select_interactions("crc", df)
crc_model <- crc_interactions$model
crc_kept  <- crc_interactions$kept_str


# ---- Model ACP rate ----

# Main effects model
fit0 <- glm(acp ~ 1 + age_group_granular + subject_gender + prevalent_incident_status + imd_quintile,
            data=df, family=binomial())
summary(fit0)

# Check which two-way interactions can be dropped from a full model with main effects and all six interaction groups
acp_interactions <- select_interactions("acp", df)
acp_model <- acp_interactions$model
acp_kept  <- acp_interactions$kept_str


# ---- Save kept interactions per outcome ----
interactions_kept <- data.frame(
  outcome = c("no investigation", "crc", "acp"),
  interactions = c(noinv_interactions$kept_str,
                   crc_interactions$kept_str,
                   acp_interactions$kept_str),
  stringsAsFactors = FALSE
)
print(interactions_kept)
write.csv(interactions_kept, paste(out_path, '/glm_interactions.csv', sep=''),
          row.names=FALSE)

# Map a p-value to a significance label
signif_label <- function(p){
  ifelse(is.na(p),   "",
  ifelse(p < 0.001,  "<0.001",
  ifelse(p < 0.01,   "<0.01",
  ifelse(p < 0.05,   "<0.05",
                     "ns"))))
}

# Build a tidy coefficient table for one fitted glm
coef_table <- function(fit, outcome_label){
  sm <- summary(fit)$coefficients          # Estimate, Std. Error, z value, Pr(>|z|)
  p  <- sm[, "Pr(>|z|)"]
  data.frame(
    outcome   = outcome_label,
    term      = rownames(sm),
    coef      = sm[, "Estimate"],
    or        = exp(sm[, "Estimate"]),
    std_error = sm[, "Std. Error"],
    p_value   = p,
    signif    = signif_label(p),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

coefs_table <- rbind(
  coef_table(noinv_interactions$model, "no investigation"),
  coef_table(crc_interactions$model,   "crc"),
  coef_table(acp_interactions$model,   "acp")
)

print(coefs_table)
write.csv(coefs_table, paste(out_path, '/glm_coefficients.csv', sep=''),
          row.names=FALSE)
