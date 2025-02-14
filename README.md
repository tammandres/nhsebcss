# Initial analysis of the NHSE Bowel Screening data

Andres Tamm

12 February 2025


# Installation
```
conda create -n bcss python=3.9
conda activate bcss
pip install -r requirements.txt
```

# Usage

For data quality checks, run `dataquality.py`.

To reproduce results of the analysis, first run `dataprep.py` to clean and prepare the dataset,
and then `analysis.py` to produce tables and figures. 
If the code is run on a different computer, it is necessary to change the file paths 
in lines 7-9 in `dataprep.py` and in lines 14-15 in `analysis.py`.

```
python dataprep.py
python analysis.py
```

The following outputs are created in the `./results` directory:
```
./dq                       : data quality checks
./outcomes_by_demographics : tables and figures showing screening outcomes by age, deprivation, and sex
./outcomes_over_time       : tables and figures showing screening outcomes over time
./positivity               : tables and figures showing FIT positivity, overall, over time, and by demographics
cohort.csv                 : episode level descriptive statistics
cohort_patient.csv         : person level descriptive statistics
outcomes_investigated.csv  : count and percentage for outcomes that were associated with a whole colon investigation
outcomes.csv               : count and percentage for all screening outcomes
removed_episodes.csv       : summary of excluded episodes
```

Note: `tmp.py` contains unused code snippets that I did not want to delete.