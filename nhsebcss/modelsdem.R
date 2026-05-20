# Outcome rates by demographics - adjusted models
#install.packages("mgcv", version="1.9-4")
#install.packages("data.table", version="1.17.8")
#install.packages('dplyr', version="1.1.4")
library(mgcv)
library(data.table)
library(dplyr)


out_path <- "Z:/andres/nhsebcss/results/primary"


# Helper 
coef_df <- function(fit, model_name){
  s <- data.frame(summary(fit)$coefficients)
  colnames(s) <- c("estimate", "se", "z", "p_val")
  s$or <- exp(s$estimate)
  s <- s[, colnames(s) != 'se']
  s$model <- model_name
  s$term <- rownames(s)
  s$p_val_sig <- ''
  for(v in c(0.05, 0.01, 0.001)){
    mask <- s$p_val < v
    s[mask, 'p_val_sig'] <- paste('<', v)
  }
  
  ci <- exp(confint.default(fit))
  colnames(ci) <- c("or_low", "or_upp")
  
  s <- cbind(s, ci)
  s <- s[, c("model", "term", "or", "or_low", "or_upp", "z", "p_val", "p_val_sig")]
  rownames(s) <- NULL
  return(s)
}


# ---- Data prep ----

# Read data
data_path <- "C:/Users/andres.Tamm/Desktop/bcss_clean_data/episodes_clean.csv"
df <- data.table::fread(data_path)
nrow(df)

# Keep episodes with positive FIT
outcomes_without_positive_fit <- c('FIT negative', 
                                   'FIT inadequate participation')
mask <- df$outcome %in% outcomes_without_positive_fit
sum(mask)
df <- df[!mask,]
nrow(df)

for(i in 1:10){  # Free up memory after reducing the dataset
  gc()
}

# Drop episodes with missing IMD data
mask <- df$imd_quintile == ""
sum(mask)  ## 528
mean(mask) * 100  ## 0.15
df <- df[!mask,]
nrow(df)

s <- df %>% group_by(imd_quintile) %>% summarise(count=n())
s  # min 63,540

# Categorical variables to factors
df$age_group_screen2 <- as.factor(df$age_group_screen2)
df$age_group_screen2 <- relevel(df$age_group_screen2, ref="50-59")

df$subject_gender <- as.factor(df$subject_gender)
df$subject_gender <- relevel(df$subject_gender, ref="Female")

df$prevalent_incident_status <- as.factor(df$prevalent_incident_status)
df$prevalent_incident_status <- relevel(df$prevalent_incident_status, ref="Prevalent")

df$imd_quintile <- as.factor(df$imd_quintile)
df$imd_quintile <- relevel(df$imd_quintile, ref="05 - Least deprived")

# Define some required outcome variables (indicator for advanced polyps is already in the df)
df$crc <- ifelse(df$outcome == "Colorectal cancer", 1, 0)
df$no_investigation <- ifelse(df$outcome == 'FIT positive, no investigation', 1, 0)


# ---- Model non-investigation rate ----

# Basic model
fit0 <- glm(no_investigation ~ 1 + age_group_screen2 + subject_gender + prevalent_incident_status + imd_quintile,
            data=df, family=binomial())
summary(fit0)

# Model with interactions
fit1 <- glm(no_investigation ~ 1 + age_group_screen2 + subject_gender + prevalent_incident_status + imd_quintile +
              age_group_screen2 * imd_quintile + prevalent_incident_status * imd_quintile,
            data=df, family=binomial())
summary(fit1)
anova(fit0, fit1, test = "Chisq")

# Model with interactions
fit2 <- glm(no_investigation ~ 1 + age_group_screen2 + subject_gender + prevalent_incident_status + imd_quintile +
               subject_gender * imd_quintile,
            data=df, family=binomial())
summary(fit2)
anova(fit0, fit2, test = "Chisq")

betas <- coef(fit2)
bf <- betas['imd_quintile01 - Most deprived']
bm <- bf + betas['subject_genderMale:imd_quintile01 - Most deprived']
exp(bf)
exp(bm)

r <- coef_df(fit2, 'noninvestigation')

## get CI for OR for males too 
df$subject_gender_relevel <- relevel(df$subject_gender, ref="Male")
fit3 <- glm(no_investigation ~ 1 + age_group_screen2 + subject_gender_relevel + prevalent_incident_status + imd_quintile +
              subject_gender_relevel * imd_quintile,
            data=df, family=binomial())

r2 <- coef_df(fit3, 'noninvestigation')
r2 <- r2[r2$term == 'imd_quintile01 - Most deprived',]
r2[, 'term'] <- 'imd_quintile01 - Most deprived_Male'
r <- rbind(r, r2)


# Dbl check whether effect remains in GAM (yes)
fit4 <- mgcv::gam(no_investigation ~ 1 + s(subject_age_at_episode_start) + subject_gender + prevalent_incident_status + 
                    imd_quintile + subject_gender * imd_quintile, data=df, family=binomial())
summary(fit4)


s <- summary(fit4)$p.coeff
s['imd_quintile01 - Most deprived_Male'] <- s['imd_quintile01 - Most deprived'] + s['subject_genderMale:imd_quintile01 - Most deprived']
s <- exp(s)
s['subject_genderMale']
1 / s['subject_genderMale']
s['prevalent_incident_statusIncident']
s['imd_quintile01 - Most deprived']
s['imd_quintile01 - Most deprived_Male']


## save coef
write.csv(r, paste(out_path, '/glm_noinvestigation.csv', sep=''),
          row.names=FALSE)


# Explore predicted probabilities of no investigation from the model
pred_data <- df[,c("age_group_screen2", "prevalent_incident_status", "subject_gender", "imd_quintile")]
pred_data <- pred_data[!duplicated(pred_data),]
pred_data <- pred_data %>% arrange(prevalent_incident_status, age_group_screen2, imd_quintile)
gc()

pred <- predict(fit2, newdata=pred_data, type='response', se.fit=TRUE)
pred_data$y_prob <- pred$fit * 100
pred_data$y_prob_se <- pred$se.fit * 100
pred_data$y_prob_low <- pred_data$y_prob - 1.96 * pred_data$y_prob_se
pred_data$y_prob_upp <- pred_data$y_prob + 1.96 * pred_data$y_prob_se
write.csv(pred_data, paste(out_path, '/glm_noinvestigation_pred.csv', sep=''),
          row.names=FALSE)


# ---- Model CRC rate ----

# Keep episodes with whole colon investigation
outcomes_without_investigation <- c('FIT negative', 
                                    'FIT inadequate participation', 
                                    'FIT positive, no investigation')
mask <- df$outcome %in% outcomes_without_investigation
sum(mask)
df <- df[!mask,]
nrow(df)

for(i in 1:10){
  gc()
}

s <- df %>% group_by(imd_quintile) %>% summarise(count=n())
s   ## min 45692


# CRC
fit0 <- glm(crc ~ 1 + age_group_screen2 + subject_gender + prevalent_incident_status + imd_quintile,
            data=df, family=binomial())
summary(fit0)

fit1 <- glm(crc ~ 1 + age_group_screen2 + subject_gender + prevalent_incident_status + imd_quintile +
              age_group_screen2 * imd_quintile + prevalent_incident_status * imd_quintile + 
              subject_gender * imd_quintile,
            data=df, family=binomial())
summary(fit1)
anova(fit0, fit1, test = "Chisq")

r <- coef_df(fit0, 'crc_main')
write.csv(r, paste(out_path, '/glm_crc.csv', sep=''),
          row.names=FALSE)


# Dbl check whether effect remains in GAM (yes)
fit2 <- mgcv::gam(crc ~ 1 + s(subject_age_at_episode_start) + subject_gender + prevalent_incident_status + 
                    imd_quintile, data=df, family=binomial())
summary(fit2)
exp(coef(fit2))


s <- summary(fit2)$p.coeff
s <- exp(s)
s['subject_genderMale']
s['prevalent_incident_statusIncident']
s['imd_quintile01 - Most deprived']



# Explore predicted probabilities of CRC from the model
pred_data <- df[,c("age_group_screen2", "prevalent_incident_status", "subject_gender", "imd_quintile")]
pred_data <- pred_data[!duplicated(pred_data),]
pred_data <- pred_data %>% arrange(prevalent_incident_status, age_group_screen2, imd_quintile)
gc()

pred <- predict(fit0, newdata=pred_data, type='response', se.fit=TRUE)
pred_data$y_prob <- pred$fit * 100
pred_data$y_prob_se <- pred$se.fit * 100
pred_data$y_prob_low <- pred_data$y_prob - 1.96 * pred_data$y_prob_se
pred_data$y_prob_upp <- pred_data$y_prob + 1.96 * pred_data$y_prob_se
write.csv(pred_data, paste(out_path, '/glm_crc_pred.csv', sep=''),
          row.names=FALSE)


# ---- Model ACP rate ----
fit0 <- glm(advanced_polyp ~ 1 + age_group_screen2 + subject_gender + prevalent_incident_status + imd_quintile,
            data=df, family=binomial())
summary(fit0)

fit1 <- glm(advanced_polyp ~ 1 + age_group_screen2 + subject_gender + prevalent_incident_status + imd_quintile +
              age_group_screen2 * imd_quintile + prevalent_incident_status * imd_quintile + subject_gender * imd_quintile,
            data=df, family=binomial())
summary(fit1)

anova(fit0, fit1, test = "Chisq")

r <- coef_df(fit0, 'acp_main')
write.csv(r, paste(out_path, '/glm_acp.csv', sep=''),
          row.names=FALSE)


# Dbl check whether effect remains in GAM (yes)
fit2 <- mgcv::gam(advanced_polyp ~ 1 + s(subject_age_at_episode_start) + subject_gender + prevalent_incident_status + 
                    imd_quintile, data=df, family=binomial())
summary(fit2)
exp(coef(fit2))


s <- summary(fit2)$p.coeff
s <- exp(s)
s['subject_genderMale']
s['prevalent_incident_statusIncident']
s['imd_quintile01 - Most deprived']



# Explore predicted probabilities of ACP from the model
pred <- predict(fit0, newdata=pred_data, type='response', se.fit=TRUE)
pred_data$y_prob <- pred$fit * 100
pred_data$y_prob_se <- pred$se.fit * 100
pred_data$y_prob_low <- pred_data$y_prob - 1.96 * pred_data$y_prob_se
pred_data$y_prob_upp <- pred_data$y_prob + 1.96 * pred_data$y_prob_se
write.csv(pred_data, paste(out_path, '/glm_acp_pred.csv', sep=''),
          row.names=FALSE)
