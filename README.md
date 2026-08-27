# Analysis of NHSE Bowel Screening data

This repository accompanies the publication `Faecal immunochemical test for haemoglobin positivity rates and outcomes across 18 million episodes in the English Colorectal Cancer Screening Programme between 2019 and 2024`.

Andres Tamm

26 August 2026


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
python fitpositivity.py  # Summarise FIT-positive rates overall and by age, sex, deprivation
python episodestime.py   # Characterise number of screening episodes over time
Rscript modelsdem.R      # Model cancer and non-investigation rates by age, sex, deprivation, screening history
python airlock.py        # Copy outputs for export from the trusted research environment
```

# Outputs
The following key outputs are created in the `./results/primary` directory. The file names "fig3...", "fig4..." etc do not always match the figure numbers in the manuscript.
```
chisq_outcomes-by-demographics.csv          : chisq tests wrt reference level for investigation outcomes and non-investigation
dataprep_log.txt                            : log file (terminal output) when running dataprep.py
fig3_num-episodes-by-quarter.(png|svg)      : plots of num episodes over year-quarters
fig4_outcomes-by-demographics_age-broad.(png|svg)               : panel plots of investigation findings by age, sex, imd; broad age grp
fig4_outcomes-by-demographics_age-granular.(png|svg)            : panel plots of investigation findings by age, sex, imd; granular age grp fig5_no-investigation-by-demographics_age-broad.(png|svg)    : panel plots of non-investigation by age, sex, imd; broad age grp
fig5_no-investigation-by-demographics_age-granular.(png|svg)    : panel plots of non-investigation by age, sex, imd; granular age grp
fig6_positivity.(png|svg)                                       : plot of overall FIT-positivity at different FIT thr
fig7_positivity-by-demographics_age-broad.(png|svg)             : plot of FIT-positivity by age, sex, imd; broad age grp
fig7_positivity-by-demographics_age-granular.(png|svg)          : plot of FIT-positivity by age, sex, imd; broad age grp
fig7b_positivity-by-demographics-flipped_age-broad.(png|svg)    : plot of FIT-positivity by age, sex, imd; broad age grp, flipped axes
fig7b_positivity-by-demographics-flipped_age-granular.(png|svg) : plot of FIT-positivity by age, sex, imd; granular age grp, flipped axes
num-episodes-by-quarter.csv                             : num episodes by the quarter of each year
outcomes-investigated-by-demographics-history-sex.csv   : counts/percentages of investigation findings by age, imd | screening hist, sex
outcomes-investigated-by-demographics-history.csv       : counts/percentages of investigation findings by age, sex, imd | screening hist
outcomes-investigated-by-demographics.csv               : counts/percentages of investigation findings by age, sex, imd
outcomes-polyp-nonexclusive-by-demographics-history.csv : counts/percentages of total ACP/NACP by age, sex, imd | screening history
outcomes-polyp-nonexclusive-by-demographics.csv         : counts/percentages of total ACP/NACP by age, sex, imd
outcomes-pos-by-demographics-history-sex.csv     : counts/percentages of non-investigation by age, imd | screening hist, sex
outcomes-pos-by-demographics-history.csv         : counts/percentages of non-investigation by age, imd, sex | screening hist
outcomes-pos-by-demographics.csv                 : counts/percentages of non-investigation by age, imd, sex
positivity-by-demographics_age-broad.csv         : counts/percentages of FIT positivity by demographics, broad age grp
positivity-by-demographics_age-granular.csv      : counts/percentages of FIT positivity by demographics, granular age grp
positivity-by-demographics-hist_age-broad.csv    : counts/percentages of FIT positivity by demographics and screening history, broad age grp
positivity-by-demographics-hist_age-granular.csv : counts/percentages of FIT positivity by demographics and screening history, granular age 
removed_episodes_simple.csv                      : num episodes removed due to each reason (broad categories)
removed_episodes.csv                             : num episodes removed due to each reason (detailed categories)
suppl_fig5_outcomes-investigated-by-demographics-history.*(png|svg)         : plots of investigation findings by age, sex, imd; broad and granular age grp
suppl_fig6_no-investigation-investigated-by-demographics-history.*(png|svg) : plots of non-investigation by age, sex, imd; broad and granular age grp
suppl_fig7_polyp-nonexclusive-by-demographics.*(png|svg)                    : plots of total ACP/NACP by age, sex, imd; broad and granular age grp
table1_participant-episode_statistics.csv   : descriptive stats for participants and episodes
table2_outcomes-summarycsv                  : counts/percentages for screening outcomes
table3.*csv                                 : counts/percentages of FIT-positive episodes by threshold
```
