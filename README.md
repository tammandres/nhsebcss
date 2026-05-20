# Analysis of NHSE Bowel Screening data

This repository accompanies the publication "Faecal immunochemical test for haemoglobin positivity rates and outcomes across 18 million episodes in the English Colorectal Cancer Screening Programme between 2019 and 2024".

Andres Tamm

21 May 2026


# Installation
```
conda create -n bcss python=3.9
conda activate bcss
pip install -e .
```

# Usage

Scripts were run in the following order:

```sh
cd ./nhsebcss/
python dataprep.py       # Prepare the dataset and apply inclusion criteria
python inclusioncrit.py  # Summarise excluded episodes
python descriptives.py   # Descriptive statistics table and overall summary of screening outcomes
python outcomesdem.py    # Screening outcomes by age, sex and deprivation
python fitpositivity.py  # Summaries of FIT-positive rates
python episodestime.py   # Characterise number of screening episodes over time
Rscript modelsdem.R      # Model cancer and non-investigation rates by age, sex, deprivation, screening history
python airlock.py        # Copy outputs for export from the trusted research environment
```

The following outputs are created in the `./results/primary` directory:
```
chisq_outcomes-investigated.*csv : chisq tests wrt reference level for investigated outcomes by demographics
chisq_outcomes-pos.*csv          : chisq tests wrt reference level for FIT positive outcomes by demographics
dataprep_log.txt                 : log file (terminal output) when running dataprep.py
fig3_num-episodes.*(png|svg)     : plots of num episodes over year-quarters
fig4_outcomes-by-demo.*(png|svg) : panel plots of outcome rates by demographics
fig5_positivity.*(png|svg)          : plots of overall FIT-positivity rates
fig6_positivity-by-demo.*(png|svg)  : plots of FIT-positivity rates by demographics
glm_.*_pred.csv                     : predicted probabilities from logistic models predicting ACP, CRC, non-investigation rates
glm_(acp|crc|noinvestigation).csv   : odds ratios from logistic models predicting ACP, CRC, non-investigation rates
no-fit-result-by-demo.*                     : plots for the counts/percentages of no FIT result by demographics
num-episodes-by-demographics.*(csv|png|svg) : data and plots for total number of episodes by demographics
num-episodes-by-quarter.*csv                : counts/percentages for total number of episodes by year-quarter
outcomes-all-by-demographics.*csv           : counts/percentages for all outcomes by demographics
outcomes-investigated-by-demographics.*csv  : counts/percentages for outcomes of whole colon investigation by demographics
outcomes-pos-by-demographics.*csv           : counts/percentages for outcomes of FIT-positive episodes by demographics
positivity-by-demographics-hist.csv         : data for FIT positivity by demographics, stratified by screening history
positivity-by-demographics.csv       : data for FIT positivity by demographics
removed_episodes_simple.csv          : num episodes removed due to each reason (broader categories)
removed_episodes.csv                 : num episodes removed due to each reason (detailed)
suppl_fig5.*(png|svg)    : plots for non-investigation rates by demographics
suppl_fig6.*(png|svg)    : plots FIT positivity rates by demographics, stratified by screening history
table1.*csv  : descriptive statistics for participants and episodes
table2.*csv  : counts/percentages for screening outcomes overall
table3.*csv  : counts/percentages of FIT-positive episodes by threshold
```

Note about the scripts:

* `airlock.py`  : copy subset of outputs for export from the trusted research environment

* `dataprep.py` : prepare dataset for analysis (extracting relevant kit result per episode, defining screening outcomes, applying inclusion criteria, etc)

* `descriptives.py`  : summarise screening outcomes and compute descriptive statistics for the dataset

* `episodestime.py`  : summarise number of episodes over time

* `fitpositivity.py` : analyse FIT positivity rates by threshold, time and demographics 

* `inclusioncrit.py` : summarise included and excluded episodes in a simpler way by using broader categories

* `lateresponder.py` : check missing late-responder episodes

* `modelsdem.R`      : run multivariable logistic regression models for obtaining adjusted odds ratios for age, sex, deprivation, and screening history.

* `outcomesdem.py`   : analyse screening outcomes by age, sex, deprivation (and screening history)

* `utils.py`         : some helper functions, esp. for computing percentages with confidence intervals
