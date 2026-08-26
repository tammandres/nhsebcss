"""Descriptive statistics and overall summary of screening outcomes
Assumes dataprep.py has been run
"""
import pandas as pd
from pathlib import Path
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from nhsebcss.utils import summarise_cat


# Paths
data_path_clean = Path("C:/Users/andres.Tamm/Desktop/bcss_clean_data")
out_path = Path("Z:/andres/nhsebcss/results/primary")
out_path.mkdir(exist_ok=True, parents=True)


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
                          'analyser_reading_used', 'imd_decile', 'cancer_at_endoscopy', 'num_kit',
                          'num_reading', 'num_premalignant_polyp', 'premalignant_polyp', 'kit_result',
                          'subject_age_at_episode_start'])
n_epi = df.anon_subject_epis_id.nunique()
assert df.shape[0] == n_epi

# Dates to datetime (note that date format is different as dates were already reformatted in clean data)
date_cols = ['episode_start_date', 'episode_end_date', 'test_kit_logged_date', 'test_kit_result_date']
assert all([c in df.columns for c in date_cols])
for c in date_cols:
    print(c)
    df[c] = pd.to_datetime(df[c], format='%Y-%m-%d', exact=True)

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

# Create a variable that denotes each year-quarter
df['year_quarter'] = df.test_kit_logged_year.astype(str) + '-Q' + df.test_kit_logged_quarter.astype(str)

# Update the late-responder episode counts as this was not done when late-responder outcomes were fixed
# in dataprep.py
df['days_to_return'] = (df.test_kit_logged_date - df.episode_start_date).dt.days
mask = (df.episode_subtype != 'Late Responder') & (df.days_to_return >= 181)
df.loc[mask, 'episode_subtype'].value_counts()
df.loc[mask, 'episode_subtype'] = 'Late Responder'
df.episode_subtype.value_counts()

#endregion


# ---- Table 1: Descriptive statistics for participants and episodes ----
#region
print('\nGENERATING DESCRIPTIVE STATISTICS TABLE...')

def summarise_cat(df, col, name, digits=1, sort=True):
    """Summarise categorical data"""
    s = df[col]

    # Designate missing values with 'NULL'
    if 'NULL' in s:
        raise ValueError("NULL is among values")
    else:
        s = s.fillna('NULL')

    counts = s.value_counts(sort=sort)
    perc = counts / df.shape[0] * 100
    perc = perc.round(digits)

    out = pd.concat(objs=[counts, perc], axis=1)
    out = out.reset_index()
    out.columns = ['Characteristic', 'Count', 'Percent']

    if 'NULL' in out.Characteristic.tolist():
        out0 = out.loc[out.Characteristic != 'NULL']
        out1 = out.loc[out.Characteristic == 'NULL']
        out = pd.concat(objs=[out0, out1], axis=0)

    header = pd.DataFrame({'Characteristic': [name], 'Count': [''], 'Percent': ['']})
    out = pd.concat(objs=[header, out], axis=0)
    out.Characteristic = out.Characteristic.replace({'NULL': 'Not known'})

    return out


# .... Get participant level statistics first
desc = pd.DataFrame()

row = pd.DataFrame({'Characteristic': ['A. Participant statistics'], 'Count': [''], 'Percent': ['']})
desc = pd.concat(objs=[desc, row], axis=0)

## Num participants
num_person = df.anon_screening_subject_id.nunique()
row = pd.DataFrame({'Characteristic': ['Number of participants'], 'Count': [num_person], 'Percent': [100.]})
desc = pd.concat(objs=[desc, row], axis=0)

## Select first episode per participant for computing stats
dfsub = df[['anon_screening_subject_id', 'subject_gender', 'age_group_screen2', 'episode_start_date',
            'imd_quintile', 'imd_decile']]
idxmin = dfsub.groupby('anon_screening_subject_id').episode_start_date.idxmin()
dfsub = dfsub.loc[idxmin]
assert dfsub.shape[0] == dfsub.anon_screening_subject_id.nunique()

## Summarise sex, age group, deprivation
cat_cols = {'subject_gender': 'Gender', 
            'age_group_screen2': 'Age group (first episode)',
            'imd_quintile': 'IMD quintile (first episode)'
            }
for c, name in cat_cols.items():
    print(c)
    row = summarise_cat(dfsub, c, name)
    desc = pd.concat(objs=[desc, row], axis=0)

## Add num episodes per person
epi_count = df.groupby('anon_screening_subject_id').size()
epi_count = epi_count.rename('num_epi').reset_index()
row = summarise_cat(epi_count, 'num_epi', 'Number of episodes per participant')
desc = pd.concat(objs=[desc, row], axis=0)


# .... Get episode level statistics
row = pd.DataFrame({'Characteristic': ['B. Episode statistics*'], 'Count': [''], 'Percent': ['']})
desc = pd.concat(objs=[desc, row], axis=0)

# Add number of episodes
num_epi = df.anon_subject_epis_id.nunique()
row = pd.DataFrame({'Characteristic': ['Number of episodes'], 'Count': [num_epi], 'Percent': ['']})
desc = pd.concat(objs=[desc, row], axis=0)

# Summarise categorical columns 
cat_cols = {'subject_gender': 'Gender', 
            'age_group_screen2': 'Age group',
            'imd_quintile': 'IMD quintile',
            'prevalent_incident_status': 'Prevalent incident status', 
            'episode_subtype': 'Episode subtype'
            }

for c, name in cat_cols.items():
    print(c)
    row = summarise_cat(df, c, name)
    if name == 'Episode subtype':
        row = row.set_index('Characteristic')
        row = row.loc[['Routine', 'Over Age', 'Late Responder', 'Opt-in']]
        row = row.reset_index()
    desc = pd.concat(objs=[desc, row], axis=0)

# Rename prev incident status
desc.Characteristic = desc.Characteristic.replace({'Incident': 'Incident (previous responders)',
                                                   'Prevalent': 'Prevalent (first-time responders)'
                                                   })

# Save
assert desc.Count.replace({'': 1000}).astype(float).min() > 10
desc.to_csv(out_path / 'table1_participant-episode-statistics.csv', index=False)

#endregion


# ---- Table 2: Summary of screening outcomes ----
#region
print('\nSUMMARISING OUTCOMES OVERALL...')

# Get num episodes with each outcome
#  dataframe df has a single row per episode -> value counts gives num episodes with outcome
out = df.outcome.value_counts().reset_index()

# Add a row for total num episodes
n_total = df.shape[0]
row = pd.DataFrame([['Total number of episodes', n_total]], columns=['outcome', 'count'])
out = pd.concat(objs=[row, out], axis=0).reset_index(drop=True)

# Add a row for FIT positive episodes
n_fit_pos = out.loc[out.outcome.isin(outcomes_fit_pos), 'count'].sum()
row = pd.DataFrame([['FIT positive', n_fit_pos]], columns=['outcome', 'count'])
out = pd.concat(objs=[out, row], axis=0).reset_index(drop=True)

# Add a row for investigated episodes
n_investigated = out.loc[out.outcome.isin(outcomes_with_investigation), 'count'].sum()
row = pd.DataFrame([['Episodes with investigation', n_investigated]], columns=['outcome', 'count'])
out = pd.concat(objs=[out, row], axis=0).reset_index(drop=True)

# Add a row for "Other findings"
outcomes_other = ['Polyp(s) without endoscopy record', 'No premalignant polyps']
n_other = out.loc[out.outcome.isin(outcomes_other), 'count'].sum()
row = pd.DataFrame([['Other findings', n_other]], columns=['outcome', 'count'])
out = pd.concat(objs=[out, row], axis=0).reset_index(drop=True)

# Get percentage with respect to total num episodes, FIT positive episodes and investigated episodes
out['ntot'] = n_total
out['perc'] = out['count'] / out.ntot * 100
n_fit_pos = out.loc[out.outcome.isin(outcomes_fit_pos), 'count'].sum()
n_investigated = out.loc[out.outcome.isin(outcomes_with_investigation), 'count'].sum()
out['perc_fit_pos'] = out['count'] / n_fit_pos * 100
out['perc_investigated'] = out['count'] / n_investigated * 100

outcome_mask = outcomes_with_investigation + ['Other findings']
out.loc[~out.outcome.isin(outcome_mask), 'perc_investigated'] = np.nan
assert np.abs((out.loc[out.outcome.isin(outcomes_with_investigation), 'perc_investigated'].sum() - 100)) < 1e-5

outcome_mask = outcomes_fit_pos + ['Other findings']
out.loc[~out.outcome.isin(outcome_mask), 'perc_fit_pos'] = np.nan
assert np.abs((out.loc[out.outcome.isin(outcomes_fit_pos), 'perc_fit_pos'].sum() - 100)) < 1e-5

# Reorder rows, so that episodes without investigation come last
row_order = ['Total number of episodes', 
                'FIT negative', 
                'FIT positive', 
                'Episodes with investigation', 
                'Colorectal cancer', 
                'Advanced premalignant polyp', 
                'Non-advanced premalignant polyp',
                'Other findings',
                'Polyp(s) without endoscopy record', 
                'No premalignant polyps',
                'No abnormality',
                'No result', 
                'FIT positive, no investigation',
                'FIT positive, unknown outcome',
                'No FIT result'
                ]
out = out.set_index('outcome').loc[row_order].reset_index()
out = out.drop(labels='ntot', axis=1)

# Add total perc
out.loc[out.outcome == 'Episodes with investigation', 'perc_investigated'] = 100.0
out.loc[out.outcome == 'FIT positive', 'perc_fit_pos'] = 100.0
out.loc[out.outcome == 'Episodes with investigation', 'perc_fit_pos'] = \
    out.loc[out.outcome == 'Episodes with investigation', 'count'] / n_fit_pos * 100
 
out.columns = ['Outcome', 
               'Num episodes', 
               'Percent episodes', 
               'Percent of FIT positive episodes', 
               'Percent of investigated episodes']

# Check
out = out.set_index('Outcome')
assert out.loc[outcomes_with_investigation, 'Num episodes'].sum() == out.loc['Episodes with investigation', 'Num episodes']
assert np.abs(out.loc[outcomes_with_investigation, 'Percent of investigated episodes'].sum() - 100) < 0.01
assert out.loc[outcomes_fit_pos, 'Num episodes'].sum() == out.loc['FIT positive', 'Num episodes']
assert np.abs(out.loc[outcomes_fit_pos, 'Percent of FIT positive episodes'].sum() - 100) < 0.01

out['Percent episodes'] = out['Percent episodes'].round(2)
out['Percent of FIT positive episodes'] = out['Percent of FIT positive episodes'].round(1)
out['Percent of investigated episodes'] = out['Percent of investigated episodes'].round(1)
out = out.reset_index()
out

out.to_csv(out_path / 'table2_outcomes-summary.csv', index=False)
#endregion


## Dbl check: when no FIT result, is it spoilt kit? 99.7% times yes.
dfsub = df.loc[df.outcome == 'No FIT result']
dfsub.kit_result.value_counts(normalize=True) * 100


## Dbl check num episodes with age <60 and <55
dfsub = df.loc[df.subject_age_at_episode_start < 60]
dfsub.subject_age_at_episode_start.value_counts(sort=False)

dfsub = df.loc[df.subject_age_at_episode_start < 55]
dfsub.subject_age_at_episode_start.value_counts(sort=False)