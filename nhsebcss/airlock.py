from pathlib import Path
import shutil
import os
import pandas as pd


airlock_path = Path('Z:/andres/nhsebcss/airlock/20260511_nhsebcsp')
airlock_path.mkdir(exist_ok=True, parents=True)

results_path = Path(r'Z:\andres\nhsebcss\results\primary')


# Summary
files = os.listdir(results_path)

files_exclude = ['removed_episodes.csv', 
                 'glm_acp_pred.csv', 
                 'glm_crc_pred.csv', 
                 'glm_noinvestigation_pred.csv',
                 'dataprep_log.txt'
                 ]
assert all(f in files for f in files_exclude)
files = [f for f in files if f not in files_exclude]


# Dbl check csv have no counts < 10
for f in files:
    if '.csv' in f:
        df = pd.read_csv(results_path / f)

        mask = df.columns.str.lower().str.contains('num|number|count')
        mask = mask & (~df.columns.str.lower().str.contains('perc|percent'))
        cols_use = df.columns[mask]

        print('\n===Dataframe:', f)
        print('--All cols:', df.columns.tolist())
        print('--Checked cols:', cols_use.tolist())

        for c in cols_use:
            test = df[c].min() >= 10
            assert test

# Copy files
for f in files:
    shutil.copyfile(results_path / f, airlock_path / f)

