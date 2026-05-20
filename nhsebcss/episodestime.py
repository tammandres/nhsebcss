"""Analyse outcome rates over time. Assumes that dataprep.py has been run"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from nhsebcss.utils import grouped_count


# Paths
data_path_clean = Path("C:/Users/andres.Tamm/Desktop/bcss_clean_data")

out_path = Path("Z:/andres/nhsebcss/results/primary")
out_path.mkdir(exist_ok=True, parents=True)

out_path_sub = out_path
out_path_sub.mkdir(exist_ok=True, parents=True)

# Method for proportion confidence intervals
ci_method = 'wilson'



# ---- Read data ----
#region 
print('\nREADING DATA...')

# Read cleaned data
df = pd.read_csv(data_path_clean / 'episodes_clean.csv',
                 usecols=['anon_subject_epis_id', 'anon_screening_subject_id', 'episode_subtype',
                          'episode_start_date', 'episode_end_date', 'test_kit_logged_date', 
                          'test_kit_logged_year', 'test_kit_logged_quarter', 'test_kit_result_date',
                          'subject_gender', 'prevalent_incident_status', 'age_group_screen2', 'imd_quintile',
                          'outcome', 'advanced_polyp', 'nonadvanced_polyp', 'premalignant_polyp',
                          'analyser_reading_used'])
n_epi = df.anon_subject_epis_id.nunique()
assert df.shape[0] == n_epi

# Dates to datetime (note that date format is different as dates were already reformatted in clean data)
date_cols = ['episode_start_date', 'episode_end_date', 'test_kit_logged_date', 'test_kit_result_date']
assert all([c in df.columns for c in date_cols])
for c in date_cols:
    print(c)
    df[c] = pd.to_datetime(df[c], format='%Y-%m-%d', exact=True)

# Create a variable that denotes each year-quarter
df['year_quarter'] = df.test_kit_logged_year.astype(str) + '-Q' + df.test_kit_logged_quarter.astype(str)

# Get outcome categories that were associated with an investigation, and those that were not
outcomes = df.outcome.unique().tolist()
outcomes_without_investigation = ['FIT negative', 'No FIT result', 
                                  'FIT positive, no investigation', 
                                  'FIT positive, unknown outcome']
outcomes_with_investigation = [c for c in outcomes if c not in outcomes_without_investigation]
print(outcomes_without_investigation, outcomes_with_investigation)
assert all(c in outcomes for c in outcomes_without_investigation)

# Get outcomes implying a positive FIT test
outcomes_fit_pos = outcomes_with_investigation + ['FIT positive, no investigation', 
                                                  'FIT positive, unknown outcome']
assert all(c in outcomes for c in outcomes_fit_pos)

# Fill missing IMD values? Not atm - see comment in outcomesdem.py
test = df.imd_quintile.isna()
df['imd_mis'] = test
print('IMD missing: {} ({:.2f}%)'.format(test.sum(), test.mean()*100))
df.groupby('imd_mis')['outcome'].value_counts(normalize=False)
#df.imd_quintile = df.imd_quintile.fillna('NA')
df.groupby('imd_quintile', dropna=False).size()

# Check for missing values in other group cols
group_cols = ['subject_gender', 'prevalent_incident_status', 'age_group_screen2']
assert not any(df[c].isna().any() for c in group_cols)

# Add rows for nonhierarchical outcomes, and update outcome lists
# This makes it easier to compute rates of all outcomes using the grouped_count()
assert df.anon_subject_epis_id.nunique() == df.shape[0]

# Subset episodes with adequate FIT participation
df_fit = df.loc[df.outcome != 'No FIT result']
assert not df_fit.analyser_reading_used.isna().any()

#endregion


# ---- Plot number of episodes over time ----
#region
print('\nCOMPUTING EPISODE COUNTS OVER TIME...')

# For plotting outcomes over time,
# drop second quarter of 2019 and 2024, as these are not complete quarters
print(df.test_kit_logged_date.min(), df.test_kit_logged_date.max())
mask = (df.test_kit_logged_year.isin([2019, 2024])) & (df.test_kit_logged_quarter == 2)
dfsub = df.loc[~mask]

# Get count and percentage of all outcomes over time (by year-quarter)
# Percentages are computed with the total number of episodes in the denominator
count_over_time = grouped_count(dfsub, ['year_quarter'], 'outcome', ci_method=ci_method)

# Map year-quarter to x-axis coordinates
tmap = count_over_time[['year_quarter']].drop_duplicates().sort_values(by='year_quarter')
tmap['x'] = np.arange(tmap.shape[0])
tmap['xticklabel'] = tmap.year_quarter.str[5:] + '\n' + tmap.year_quarter.str[:4]
tmap['year'] = tmap.year_quarter.str[:4].astype(int)
tmap['quarter'] = tmap.year_quarter.str[6:].astype(int)
mask = tmap.year_quarter.str.lower().str.contains('q1')
tmap.loc[~mask, ['xticklabel']] = None

# Add x-axis coordinates to count data
count_over_time = count_over_time.merge(tmap, how='left')

# This is a simple graph showing the number of episodes by year-quarter
count_all = count_over_time[['year_quarter', 'ntot', 'x', 'xticklabel']].drop_duplicates()
assert (count_all['ntot'] >= 10).all()

fig, ax = plt.subplots(1, 1, figsize=(5, 4), tight_layout=True)
ax.plot(count_all['x'], count_all['ntot'])
ax.scatter(count_all['x'], count_all['ntot'], s=8)
ax.set_xticks(count_all['x'])
ax.set_xticklabels(count_all['xticklabel'], fontsize=8)
yticks = np.arange(200000, 1400000, 200000)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=10)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.grid(which='major', alpha=0.5)
ax.set_xlabel('Test kit logged date (year-quarter)', fontsize=10)
ax.set_ylabel('Number of episodes', fontsize=10)
plt.savefig(out_path_sub / 'fig3_num-episodes-by-quarter.png', dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / 'fig3_num-episodes-by-quarter.svg', bbox_inches='tight')
plt.close()

count_all.drop(labels=['x', 'xticklabel'], axis=1).to_csv(out_path_sub / 'num-episodes-by-quarter.csv', index=False)
#endregion

