"""Data quality checks"""
import pandas as pd
import numpy as np
import os
from pathlib import Path


data_path = Path("C:/Users/p0Po/Desktop/BCSS-16540-LIVE-1")
out_path = Path("Z:/andres/nhsebcss/results/dq")
out_path.mkdir(exist_ok=True, parents=True)


# ---- 1. Basic summary of each table ----
#region
# Including number of rows, number of rows with missing value, number of unique values
# and top 30 unique values with counts.
def summarise_data(df: pd.DataFrame, table_name: str = None, nmax: int = 30):
                   #pat_ignore: str = '_id'):
    """Summarise dataframe df
    At the moment, treating all columns as categorical
    """
    columns = pd.DataFrame()
    values = pd.DataFrame()

    for c in df.columns:
        print("Processing column", c)

        s = df[c]

        if 'NULL' in s:
            raise ValueError("NULL is among values")
        else:
            n_null = s.isna().sum()
            n_notnull = (~s.isna()).sum()
            n_unique = s.dropna().nunique()
            p_null = n_null / s.shape[0] * 100
            
        # Basic statistics about the current column
        row = {'table': [table_name],
               'column': [c], 
               'n_null': [n_null],
               'percent_null': [p_null],
               'n_notnull': [n_notnull], 
               'n_unique': [n_unique]}
        row = pd.DataFrame(row)
        columns = pd.concat(objs=[columns, row], axis=0)

        # Value counts for the current column
        #if not re.search(pat_ignore, c, flags=re.I):
        s = s.fillna('NULL')
        v = s.value_counts()
        v = v[:nmax]
        v = v.reset_index()
        v.columns = ['value', 'count']
        v['column'] = c
        v['table'] = table_name
        v = v[['table', 'column', 'value', 'count']] 
        values = pd.concat(objs=[values, v], axis=0)
    
    return columns, values


# Summarise all tables
run_summary = False
if run_summary:
    files = os.listdir(data_path)
    print(files)

    columns = pd.DataFrame()
    values = pd.DataFrame()

    for f in files:
        print(f)
        df = pd.read_csv(data_path / f)
        c, v = summarise_data(df, table_name=f)
        columns = pd.concat(objs=[columns, c], axis=0)
        values = pd.concat(objs=[values, v], axis=0)


    columns = columns.reset_index(drop=True)
    values = values.reset_index(drop=True)

    columns.to_csv(out_path / 'dq_columns.csv', index=False)
    values.to_csv(out_path / 'dq_values.csv', index=False)

# Note that percentage of rows with missing values is generally very small!
columns = pd.read_csv(out_path / 'dq_columns.csv')
columns.loc[columns.n_null > 0].sort_values(by=(['table', 'percent_null']))

#endregion


# ---- 2. Check identifiers ----
#region
stat = {}

# Read data
df = pd.read_csv(data_path / 'BCSS-16540-T1-LIVE-1.csv')
df_polyp = pd.read_csv(data_path / 'BCSS-16540-T2-LIVE-1.csv')
df_cancer = pd.read_csv(data_path / 'BCSS-16540-T3-LIVE-1.csv')
df_stalk = pd.read_csv(data_path / 'BCSS-16540-T4-LIVE-1.csv')
print(df.shape)
print(df.columns)

# Do any subjects share subject episode ID? No, as expected
u = df.groupby('ANON_SUBJECT_EPIS_ID').ANON_SCREENING_SUBJECT_ID.nunique()
test = (u == 1).all().item()
assert test
stat['All episode identifiers are unique to a patient'] = [test]

# Do any episodes share test kit ID? No, as expected
u = df.groupby('ANON_TEST_KIT_ID').ANON_SUBJECT_EPIS_ID.nunique()
test = (u == 1).all().item()
assert test
stat['All test kit identifiers are unique to an episode'] = [test]

# Do any episodes share polyp ID? No, as expected
u = df_polyp.groupby('ANON_POLYP_ID').ANON_SUBJECT_EPIS_ID.nunique()
test = (u == 1).all().item()
assert test
stat['All polyp identifiers are unique to an episode'] = [test]

# Do any episodes share cancer ID? No, as expected
u = df_cancer.groupby('ANON_CANCER_ID').ANON_SUBJECT_EPIS_ID.nunique()
test = (u == 1).all().item()
assert test
stat['All cancer identifiers are unique to an episode'] = [test]

# Save
stat = pd.DataFrame.from_dict(stat, orient='index')
stat = stat.reset_index()
stat.columns = ['Characteristic', 'Test']
stat
stat.to_csv(out_path / 'dq_identifiers.csv', index=False)

#endregion


# ---- 3. Check dates in main screening table ----
#region
stat = {}

# Read data
#df = pd.read_csv(data_path / 'BCSS-16540-T1-LIVE-1.csv')
#print(df.shape)
#print(df.columns)

# Number of episodes
n_epi = df.ANON_SUBJECT_EPIS_ID.nunique()
print(n_epi, df.shape[0])

# Dates to datetime
date_cols = ['EPISODE_START_DATE', 'EPISODE_END_DATE', 'TEST_KIT_LOGGED_DATE', 'TEST_KIT_RESULT_DATE']
assert all([c in df.columns for c in date_cols])
for c in date_cols:
    print(c)
    df[c] = pd.to_datetime(df[c], format='%m/%Y', exact=True)

# Explore dates
#  TEST_KIT_LOGGED_DATE is as expected.
#  TEST_KIT_RESULT_DATE is as expected.
#  Episode start and end dates, a few are not as expected.
for c in date_cols:
    print(c, '\n', df[c].min(), df[c].max())

for c in date_cols:
    print('\n----', c)
    print(df[c].describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]))

# Look at episode start dates more specifically
df['EPISODE_START_YEAR'] = df.EPISODE_START_DATE.dt.year
year_counts = df.groupby('EPISODE_START_YEAR').ANON_SUBJECT_EPIS_ID.nunique()
year_counts

n_before_2018 = df.loc[df.EPISODE_START_YEAR < 2018].ANON_SUBJECT_EPIS_ID.nunique()
n_2018 = df.loc[df.EPISODE_START_YEAR == 2018].ANON_SUBJECT_EPIS_ID.nunique()

stat['Episodes with start date before 2018'] = [n_before_2018, n_before_2018 / n_epi * 100]
stat['Episodes with start date in 2018'] = [n_2018, n_2018 / n_epi * 100]

# Explore episodes with start date before 2018
dfsub = df.loc[df.EPISODE_START_DATE < '2018-01-01']
dfsub.shape
dfsub.EPISODE_END_DATE.describe()
(dfsub.EPISODE_END_DATE >= '2019-06-01').sum()
dfsub[['EPISODE_END_DATE', 'TEST_KIT_LOGGED_DATE']].sort_values(by=['EPISODE_END_DATE'])
dfsub.shape[0] == dfsub.ANON_SUBJECT_EPIS_ID.nunique()
dfsub.EPISODE_SUBTYPE.value_counts()

# Episodes with end date before 2018: already detected when looking at unusual start dates.
df.loc[df.EPISODE_END_DATE < '2018-01-01'].ANON_SUBJECT_EPIS_ID.isin(dfsub.ANON_SUBJECT_EPIS_ID)

# Episodes with end date before 2019
df['EPISODE_END_YEAR'] = df.EPISODE_END_DATE.dt.year
year_counts = df.groupby('EPISODE_END_YEAR').ANON_SUBJECT_EPIS_ID.nunique()
year_counts

n_before_2019 = df.loc[df.EPISODE_END_YEAR < 2019].ANON_SUBJECT_EPIS_ID.nunique()
stat['Episodes with end date before 2019'] = [n_before_2019, n_before_2019 / n_epi * 100]

df.loc[df.EPISODE_END_YEAR < 2019].transpose()

# Episodes where end date is before test kit logged date
mask = (df.EPISODE_END_DATE < df.TEST_KIT_LOGGED_DATE) & (~df.EPISODE_END_DATE.isna())
dfsub = df.loc[mask]
print(dfsub.ANON_SUBJECT_EPIS_ID.nunique())
print(dfsub.ANON_SUBJECT_EPIS_ID.nunique() / n_epi * 100)
print(dfsub.EPISODE_SUBTYPE.value_counts())
dfsub.shape
dfsub.EPISODE_START_DATE.describe()

stat['Episodes with end date before test kit logged date'] = [dfsub.ANON_SUBJECT_EPIS_ID.nunique(), 
                                                              dfsub.ANON_SUBJECT_EPIS_ID.nunique() / n_epi * 100]

# End date and test kit logged date equal: most episodes
# It just means they are in the same month as information about day is not available
mask = (df.EPISODE_END_DATE == df.TEST_KIT_LOGGED_DATE) & (~df.EPISODE_END_DATE.isna())
dfsub = df.loc[mask]
print(dfsub.ANON_SUBJECT_EPIS_ID.nunique())
print(dfsub.ANON_SUBJECT_EPIS_ID.nunique() / n_epi * 100)

stat['Episodes with end date in the same month as test kit logged date'] = [dfsub.ANON_SUBJECT_EPIS_ID.nunique(), 
                                                                            dfsub.ANON_SUBJECT_EPIS_ID.nunique() / n_epi * 100]

mask = (df.EPISODE_END_DATE > df.TEST_KIT_LOGGED_DATE) & (~df.EPISODE_END_DATE.isna())
dfsub = df.loc[mask]
print(dfsub.ANON_SUBJECT_EPIS_ID.nunique())
print(dfsub.ANON_SUBJECT_EPIS_ID.nunique() / n_epi * 100)

stat['Episodes with end date in one of the months after test kit logged date'] = [dfsub.ANON_SUBJECT_EPIS_ID.nunique(), 
                                                                                  dfsub.ANON_SUBJECT_EPIS_ID.nunique() / n_epi * 100]

# Compare test kit logged date and test kit result date
#  Equal vast majority of times: equal means they occurred in the same month but not necessarily same day
mask = (df.TEST_KIT_LOGGED_DATE == df.TEST_KIT_RESULT_DATE)
dfsub = df.loc[mask]
print(dfsub.ANON_SUBJECT_EPIS_ID.nunique())
print(dfsub.ANON_SUBJECT_EPIS_ID.nunique() / n_epi * 100)

stat['Episodes with test kit logged date and test kit result date in same month'] = [dfsub.ANON_SUBJECT_EPIS_ID.nunique(), 
                                                                                     dfsub.ANON_SUBJECT_EPIS_ID.nunique() / n_epi * 100]

# Episodes with kit result date less than logged date
mask = (df.TEST_KIT_RESULT_DATE < df.TEST_KIT_LOGGED_DATE)
dfsub = df.loc[mask]
print(mask.sum())
print(dfsub.loc[mask].ANON_SUBJECT_EPIS_ID.item() in df.loc[~mask].ANON_SUBJECT_EPIS_ID)
print(dfsub.loc[mask, date_cols])

stat['Episodes with test kit result date before test kit logged date'] = [dfsub.ANON_SUBJECT_EPIS_ID.nunique(), 
                                                                          dfsub.ANON_SUBJECT_EPIS_ID.nunique() / n_epi * 100]

stat = pd.DataFrame.from_dict(stat, orient='index')
stat = stat.reset_index()
stat.columns = ['Characteristic', 'Count', 'Percent']
stat
stat.to_csv(out_path / 'dq_dates.csv', index=False)

#endregion


# ---- 4. Are cancers in the main table present in polyp and cancer tables? ----
#region
stat = {}

# Read polyp and cancer tables
df_polyp = pd.read_csv(data_path / 'BCSS-16540-T2-LIVE-1.csv')
df_cancer = pd.read_csv(data_path / 'BCSS-16540-T3-LIVE-1.csv')

# Trim polyp and cancer tables to episodes present in the secreening table
# (They were originally created with outer join)
print(df_polyp.shape, df_cancer.shape)
df_polyp = df_polyp.loc[df_polyp.ANON_SUBJECT_EPIS_ID.isin(df.ANON_SUBJECT_EPIS_ID)]
df_cancer = df_cancer.loc[df_cancer.ANON_SUBJECT_EPIS_ID.isin(df.ANON_SUBJECT_EPIS_ID)]
print(df_polyp.shape, df_cancer.shape)

# Screening episodes where outcome was detected cancer
crc_episodes_screening = df.loc[df.EPISODE_RESULT == 'Cancer Detected', 'ANON_SUBJECT_EPIS_ID'].drop_duplicates()
print(len(crc_episodes_screening))  #  23,665 episodes with cancer

# Try matching these to polyp and cancer tables
crc_episodes_polyp = df_polyp.loc[df_polyp.CARCINOMA == 'Yes'].ANON_SUBJECT_EPIS_ID.drop_duplicates()
crc_episodes_cancer = df_cancer.ANON_SUBJECT_EPIS_ID.drop_duplicates()
crc_episodes_polyp_cancer = pd.concat([crc_episodes_polyp, crc_episodes_cancer]).drop_duplicates()

n_cancer = len(crc_episodes_screening)
not_matched = crc_episodes_screening.copy()
print(len(not_matched))  # Initially 23665 episodes
stat['Number of episodes with cancer'] = [n_cancer, 100.]

not_matched = not_matched[~not_matched.isin(crc_episodes_cancer)]
print(len(not_matched))  # 6,138 left
n_bowelwall = n_cancer - len(not_matched)
stat['Number of episodes with bowel wall (non-polyp) cancer detected at endoscopy'] = \
    [n_bowelwall, n_bowelwall / n_cancer * 100]

not_matched = not_matched[~not_matched.isin(crc_episodes_polyp)]
print(len(not_matched))  # 2,020 left
n_polyp_cancer = n_cancer - n_bowelwall - len(not_matched)
n_nonendoscopy = len(not_matched)
stat['Number of episodes with cancerous polyps (but no bowel wall cancer) detected at endoscopy'] = \
    [n_polyp_cancer, n_polyp_cancer / n_cancer * 100]
stat['Number of episodes with cancer not detected at endoscopy'] = \
    [n_nonendoscopy, n_nonendoscopy / n_cancer * 100]

stat = pd.DataFrame.from_dict(stat, orient='index')
stat = stat.reset_index()
stat.columns = ['Characteristic', 'Count', 'Percent']
stat
stat.to_csv(out_path / 'dq_cancer.csv', index=False)

# From Claire: it makes sense that some cancers are detected at CT so are not present in endoscopy tables (T2 and T3)!

# Explore the unmatched ones ...
uncertain_episodes_polyp = df_polyp.loc[df_polyp.CARCINOMA == 'Uncertain'].ANON_SUBJECT_EPIS_ID.drop_duplicates()
not_matched2 = not_matched[~not_matched.isin(uncertain_episodes_polyp)]
print(len(not_matched2))  # 1,294 left
len(not_matched2) - len(not_matched) # 726 were matched to uncertain

(~not_matched2.isin(df_polyp.ANON_SUBJECT_EPIS_ID)).sum()  # 420 are NOT in polyp table
(not_matched2.isin(df_polyp.ANON_SUBJECT_EPIS_ID)).sum()  # 874 are in polyp table without cancer

# Have a look whether these 2020 unmatched episodes are somewhat different
#  Nothing unusual about test kit logged date
#  Most kit results were abnormal as expected (if they have cancer detected they should have been investigated further)
df_non = df.loc[df.ANON_SUBJECT_EPIS_ID.isin(not_matched)]
df_non.shape
df_non.EPISODE_RESULT.value_counts()
df_non.TEST_KIT_LOGGED_DATE.describe()
df_non.KIT_RESULT.value_counts()

#endregion


# ---- 5. Check and understand kit results ----
#region
stat = {}

# Subset of data with kit results
r = df[['ANON_SCREENING_SUBJECT_ID', 'ANON_SUBJECT_EPIS_ID', 
        'ANON_TEST_KIT_ID', 'TEST_KIT_LOGGED_DATE', 'TEST_KIT_RESULT_DATE',
        'KIT_RESULT', 'EPISODE_RESULT', 
        'ANALYSER_READING', 'ANALYSER_READING_USED',
        'ANALYSER_ERROR_CODE']].drop_duplicates()
assert r.ANON_TEST_KIT_ID.nunique() == r.shape[0] # Each row is a test kit result
stat['All test kits'] = [r.shape[0], 100]

# When both analyser reading columns are empty (in 1.96% of rows)
#  then corresponding KIT_RESULT is SPOILT and ANALYSER_ERROR_CODE is 1, as expected.
mask = r.ANALYSER_READING.isna() & r.ANALYSER_READING_USED.isna() 
rsub = r.loc[mask]
rsub.KIT_RESULT.value_counts()
rsub.ANALYSER_ERROR_CODE.fillna('NULL').value_counts()
stat['No reading'] = [rsub.shape[0], rsub.shape[0] / r.shape[0] * 100]

# When analyser reading used, and analyser reading are equal ...
#  - then KIT_RESULT is NORMAL or ABNORMAL in 97.37% of all rows, which makes sense.
#  - there are 2,398 rows with KIT_RESULT = 'INVALIDATED', and few with 'SPOILT'
#  -- Where the result is 'SPOILT' there is no ANALYSER_ERROR_CODE and EPISODE_RESULT is normal,
#     so this could be data entry mistake.
#  -- Where the result is 'INVALIDATED', there is no analyser error code and episode result is predominantly 
#     "Definitive normal FOBt outcome", and only in 155 cases it is inadequate participation.
mask = r.ANALYSER_READING == r.ANALYSER_READING_USED
mask.mean() * 100  # 97.38% of rows
rsub = r.loc[mask]
rsub.isna().sum()
stat['Reading and reading used equal'] = [rsub.shape[0], rsub.shape[0] / r.shape[0] * 100]

c = rsub.KIT_RESULT.value_counts()
normal_or_abnormal = c.loc[['NORMAL', 'ABNORMAL']].sum().item()
stat['Reading and reading used equal: kit result norm or abnorm'] = [normal_or_abnormal, normal_or_abnormal / rsub.shape[0] * 100]

invalidated = c.loc[['INVALIDATED']].sum().item()
stat['Reading and reading used equal: kit result invalidated'] = [invalidated, invalidated / rsub.shape[0] * 100]

spoilt = c.loc[['SPOILT']].sum().item()
stat['Reading and reading used equal: kit result spoilt'] = [spoilt, spoilt / rsub.shape[0] * 100]

rsub.loc[rsub.KIT_RESULT == 'SPOILT'].transpose()

rsubsub = rsub.loc[rsub.KIT_RESULT == 'INVALIDATED']
rsubsub.ANALYSER_ERROR_CODE.value_counts()
rsubsub.EPISODE_RESULT.value_counts()
rsubsub.shape[0] / r.shape[0] * 100

# When analyser reading used and analyser reading are not equal, then analyser reading is ALWAYS missing
#  KIT_RESULT is abnormal in all cases (which does not necessarily have to be, but it means there was a kit result)
#  and analyser error code is 4 or 5 (both mean above dilution range)
#  and the reading is always either 201 or 50001 (which is probably assigned to be above the maximum possible value)
mask = r.ANALYSER_READING.isna() & r.ANALYSER_READING_USED.isna()
rsub = r.loc[~mask]
mask2 = rsub.ANALYSER_READING != rsub.ANALYSER_READING_USED
rsub = rsub.loc[mask2]
rsub.isna().sum() == rsub.shape[0]
rsub.isna().mean()

stat['Reading and reading used not equal'] = [rsub.shape[0], rsub.shape[0] / r.shape[0] * 100]

rsub.KIT_RESULT.value_counts()
rsub.ANALYSER_ERROR_CODE.value_counts()
rsub.ANALYSER_READING_USED.unique()
nmis = rsub.ANALYSER_READING.isna().sum().item()
ndil = rsub.loc[rsub.ANALYSER_ERROR_CODE.isin([4, 5])].shape[0]

stat['Reading and reading used not equal: reading is empty'] = [nmis, nmis / rsub.shape[0] * 100]
stat['Reading and reading used not equal: error code indicates above dilution range'] = [ndil, ndil / rsub.shape[0] * 100]

# Multiple analyser readings used per episode
rsub = r.loc[~r.ANALYSER_READING_USED.isna()]
rsub = rsub[['ANON_SUBJECT_EPIS_ID', 'ANALYSER_READING_USED']].drop_duplicates()
rsub = rsub.loc[rsub.ANON_SUBJECT_EPIS_ID.duplicated(keep = False)]  #  Episodes with multiple analyser readings
print(rsub.shape)

n_mult = rsub.ANON_SUBJECT_EPIS_ID.nunique()

stat['All episodes'] = [n_epi, 100]
stat['Episodes with multiple values for analyser reading used'] = [n_mult, n_mult / n_epi * 100]

count = rsub.groupby('ANON_SUBJECT_EPIS_ID').size().value_counts()
n2 = count.loc[2].item()
stat['Episodes with multiple values for analyser reading used: 2 readings'] = [n2, n2 / n_mult * 100]

rank = rsub.groupby('ANON_SUBJECT_EPIS_ID').ANALYSER_READING_USED.rank()
rank = rank.rename('rank')
rank.unique()
rsub = rsub.merge(rank, left_index=True, right_index=True)
p = rsub.pivot(index='ANON_SUBJECT_EPIS_ID', columns='rank', values='ANALYSER_READING_USED')

p.max(axis=1)
p

# When there is no reading at all for an episode, is the episode result inadequate participation? 
#   About 95% of times inadequate participation, about 5% times 'normal', and other outcomes in a few number of cases
# When result is inadequate participation, how many times is there no reading? About 44% of times.
#   In the remaining 56% of times, the kit result is predominantly normal, for more than 100,000 episodes.
#   Conclusion: inadequate participation does not always mean no kit result?
r['no_reading'] = r.ANALYSER_READING_USED.isna()
mask = r.groupby('ANON_SUBJECT_EPIS_ID').no_reading.all()
mask.sum()

stat['Episodes without any reading'] = [mask.sum().item(), mask.mean().item() * 100]

s = r[['ANON_SUBJECT_EPIS_ID', 'EPISODE_RESULT']].drop_duplicates() # Table with one episode per row
assert s.shape[0] == n_epi
epi_without_reading = mask[mask].index
s2 = s.loc[s.ANON_SUBJECT_EPIS_ID.isin(epi_without_reading)]
s2.EPISODE_RESULT.value_counts()

test = s2.EPISODE_RESULT == 'FOBt inadequate participation'
stat['Episodes without any reading: FIT inadequate participation'] = [test.sum().item(), test.mean().item() * 100]

test = s2.EPISODE_RESULT == 'Definitive normal FOBt outcome'
stat['Episodes without any reading: definitive normal FOBt'] = [test.sum().item(), test.mean().item() * 100]

epi_inad = s.loc[s.EPISODE_RESULT == 'FOBt inadequate participation'].ANON_SUBJECT_EPIS_ID
len(epi_inad)
test = epi_inad.isin(epi_without_reading)
stat['Episodes with inadequate participation'] = [len(epi_inad), len(epi_inad) / n_epi * 100]
stat['Episodes with inadequate participation: no reading'] = [test.sum().item(), test.mean().item() * 100]

rsub = r.loc[r.ANON_SUBJECT_EPIS_ID.isin(epi_inad)]
rsub.KIT_RESULT.value_counts()
rsub.groupby('no_reading').KIT_RESULT.value_counts()
rsub.TEST_KIT_LOGGED_DATE.describe()

normal_result = r.loc[r.KIT_RESULT == 'NORMAL'].ANON_SUBJECT_EPIS_ID.drop_duplicates()
test = epi_inad.isin(normal_result)
stat['Episodes with inadequate participation: normal kit result'] = [test.sum().item(), test.mean().item() * 100]


# Save
stat = pd.DataFrame.from_dict(stat, orient='index')
stat = stat.reset_index()
stat.columns = ['Characteristic', 'Count', 'Percent']
stat
stat.to_csv(out_path / 'dq_reading.csv', index=False)

#endregion


# ---- 6. Can new endoscopy outcome categories be recreated from polyp and cancer tables? ----
#region

stat = {}

# Get data with one row per episode
dfsub = df[['ANON_SCREENING_SUBJECT_ID', 'ANON_SUBJECT_EPIS_ID', 
            'EPISODE_START_DATE', 'EPISODE_END_DATE', 'PREVALENT_INCIDENT_STATUS', 
            'EPISODE_STATUS', 'EPISODE_SUBTYPE', 'SUBJECT_AGE_AT_EPISODE_START',
            'SUBJECT_GENDER', 'IMD_QUINTILE', 'IMD_DECILE', 'IMD_SCORE', 'IMD_RANK',
            'EPISODE_RESULT']]
dfsub = dfsub.drop_duplicates()
assert dfsub.shape[0] == n_epi

# Get endoscopy findings
df_polyp = pd.read_csv(data_path / 'BCSS-16540-T2-LIVE-1.csv')
df_cancer_endo = pd.read_csv(data_path / 'BCSS-16540-T3-LIVE-1.csv')

# Subset endoscopy findings to episodes present in the screening episodes table
print(df_polyp.shape, df_cancer_endo.shape)
df_polyp = df_polyp.loc[df_polyp.ANON_SUBJECT_EPIS_ID.isin(df.ANON_SUBJECT_EPIS_ID)]
df_cancer_endo = df_cancer_endo.loc[df_cancer_endo.ANON_SUBJECT_EPIS_ID.isin(df.ANON_SUBJECT_EPIS_ID)]
print(df_polyp.shape, df_cancer_endo.shape)

# Find cancers detected at endoscopy
cancer_endo1 = df_polyp.loc[df_polyp.CARCINOMA == 'Yes'].ANON_SUBJECT_EPIS_ID.drop_duplicates()
cancer_endo2 = df_cancer_endo.ANON_SUBJECT_EPIS_ID.drop_duplicates()
cancer_endo = pd.concat(objs=[cancer_endo1, cancer_endo2], axis=0).drop_duplicates()
print(len(cancer_endo))

cancer_all = dfsub.loc[dfsub.EPISODE_RESULT == 'Cancer Detected'].ANON_SUBJECT_EPIS_ID.drop_duplicates()
print(len(cancer_all))

n_cancer_all = len(cancer_all)
n_cancer_endo = len(cancer_endo)
stat['All cancers detected'] = [n_cancer_all, 100.]
stat['Cancers present in endoscopy tables'] = [n_cancer_endo, n_cancer_endo / n_cancer_all * 100]

# Total number of polyps detected
n_polyp = df_polyp.ANON_POLYP_ID.nunique()
stat['All polyps'] = [n_polyp, 100.]
assert df_polyp.shape[0] == n_polyp

# Is anatomical location of polyp always available? Yes.
test = ~df_polyp.POLYP_LOCATION.isna()
stat['Polyps where polyp location is recorded'] = [test.sum().item(), test.mean().item() * 100]

# Are there any polyps in anus? As generally we are focussed on colorectal cancer.
test = df_polyp.POLYP_LOCATION == 'Anus'
stat['Polyps in anus'] = [test.sum().item(), test.mean().item() * 100]

df_polyp.CARCINOMA.unique()
s = df_polyp.loc[(df_polyp.POLYP_LOCATION == 'Anus') & (df_polyp.CARCINOMA == 'Yes')]
stat['Cancerous anal polyps'] = [s.shape[0], s.shape[0] / n_polyp * 100]

# Can serrated polyps uniquely be found from polyp type? Yes.
df_polyp.groupby('POLYP_TYPE').SUB_TYPE.value_counts()

# Is polyp type present for nearly all polyps? Predominantly.
df_polyp.POLYP_TYPE.fillna('NULL').value_counts()
test = ~df_polyp.POLYP_TYPE.isna()
stat['Polyps where polyp type is recorded'] = [test.sum().item(), test.mean().item() * 100]
stat['Polyps where polyp type is not recorded'] = [(~test).sum().item(), (~test).mean().item() * 100]

# Is polyp sub type present for nearly all polyps? Predominantly.
df_polyp.SUB_TYPE.fillna('NULL').value_counts()
test = ~df_polyp.SUB_TYPE.isna()
stat['Polyps where polyp sub type is recorded'] = [test.sum().item(), test.mean().item() * 100]
stat['Polyps where polyp sub type is not recorded'] = [(~test).sum().item(), (~test).mean().item() * 100]

# Get premalignant polyps, according to Rutter et al 2020, https://doi.org/10.1136/gutjnl-2019-319858
#  Adenomatous polyp, 
#  or serrated polyp (excludes 1-5mm rectal hyperplastic polyps)
print(df_polyp.POLYP_TYPE.unique())
mask = df_polyp.POLYP_TYPE.fillna('').str.lower().str.contains('adenom', regex=True)
adenoma = df_polyp.loc[mask]
print(adenoma.POLYP_TYPE.unique())

mask = df_polyp.POLYP_TYPE.fillna('').str.lower().str.contains('serrated', regex=True)
serrated = df_polyp.loc[mask]
print(serrated.POLYP_TYPE.unique())

df_polyp.POLYP_LOCATION.unique()
df_polyp.SUB_TYPE.value_counts()
mask = (serrated.POLYP_LOCATION.str.lower() == 'rectum') & (serrated.POLYP_HISTOLOGY_SIZE <= 5) & (serrated.SUB_TYPE == 'Hyperplastic polyp')
print(serrated.loc[mask, 'POLYP_LOCATION'].unique())
print(serrated.loc[mask, 'POLYP_HISTOLOGY_SIZE'].max())
print(serrated.loc[mask, 'SUB_TYPE'].value_counts())
serrated_premalignant = serrated.loc[~mask]
print(serrated.ANON_SUBJECT_EPIS_ID.nunique(), serrated_premalignant.ANON_SUBJECT_EPIS_ID.nunique())

polyp_pm = pd.concat(objs=[adenoma, serrated_premalignant], axis=0)
print(polyp_pm.ANON_SUBJECT_EPIS_ID.nunique())
print(polyp_pm.POLYP_TYPE.value_counts())

df_polyp['PREMALIGNANT'] = 0
df_polyp.loc[df_polyp.ANON_POLYP_ID.isin(polyp_pm.ANON_POLYP_ID), 'PREMALIGNANT'] = 1

df_polyp.loc[df_polyp.PREMALIGNANT == 1, 'CARCINOMA'].value_counts()

# Get advanced premalignant polyps (Rutter et al, 2020)
#  serrated polyp >= 10 mm
#  adenoma >= 10 mm
#  serrated polyp with dysplasia
#  adenoma with high-grade dysplasia
a1 = serrated.loc[serrated.POLYP_HISTOLOGY_SIZE >= 10]
print(a1.POLYP_TYPE.unique(), a1.POLYP_HISTOLOGY_SIZE.min())

a2 = adenoma.loc[adenoma.POLYP_HISTOLOGY_SIZE >= 10]
print(a2.POLYP_TYPE.unique(), a2.POLYP_HISTOLOGY_SIZE.min())

print(df_polyp.DYSPLASIA.unique())
a3 = serrated.loc[serrated.DYSPLASIA.str.lower().isin(['high grade dysplasia', 'low grade dysplasia'])]
print(a3.DYSPLASIA.unique())

a4 = adenoma.loc[adenoma.DYSPLASIA.str.lower().isin(['high grade dysplasia'])]
print(a4.DYSPLASIA.unique())

polyp_advanced = pd.concat(objs=[a1, a2, a3, a4], axis=0)
print(polyp_advanced.POLYP_TYPE.unique(), polyp_advanced.DYSPLASIA.unique())

df_polyp['ADVANCED'] = 0
df_polyp.loc[df_polyp.ANON_POLYP_ID.isin(polyp_advanced.ANON_POLYP_ID), 'ADVANCED'] = 1

# Get high risk findings
#  >= 5 premalignant polyps
#  >= 2 premalignant polyps where at least 1 is advanced
num_polyp = polyp_pm.groupby('ANON_SUBJECT_EPIS_ID').ANON_POLYP_ID.nunique().rename('NUM_PREMALIGNANT_POLYP')
num_polyp = num_polyp.reset_index()

high_risk1 = num_polyp[num_polyp.NUM_PREMALIGNANT_POLYP >= 5].ANON_SUBJECT_EPIS_ID.drop_duplicates()
high_risk2 = num_polyp[num_polyp.NUM_PREMALIGNANT_POLYP >= 2].ANON_SUBJECT_EPIS_ID.drop_duplicates()
high_risk2 = high_risk2[high_risk2.isin(polyp_advanced.ANON_SUBJECT_EPIS_ID)]
high_risk = pd.concat(objs=[high_risk1, high_risk2], axis=0).drop_duplicates()

high_risk_no_cancer = high_risk[~high_risk.isin(cancer_endo)]
len(high_risk_no_cancer)
len(high_risk)

# Indicator for cancer detected at endoscopy
dfsub['CANCER_DETECTED'] = (dfsub.EPISODE_RESULT == 'Cancer Detected').astype(int)
dfsub['CANCER_AT_ENDOSCOPY'] = 0
dfsub.loc[dfsub.ANON_SUBJECT_EPIS_ID.isin(cancer_endo), 'CANCER_AT_ENDOSCOPY'] = 1

# Number of premalignant polyps and indicator for premalignant findings
shape_before = dfsub.shape[0]
dfsub = dfsub.merge(num_polyp, how='left', on='ANON_SUBJECT_EPIS_ID')
assert shape_before == dfsub.shape[0]
dfsub.NUM_PREMALIGNANT_POLYP = dfsub.NUM_PREMALIGNANT_POLYP.fillna(0)
dfsub.NUM_PREMALIGNANT_POLYP.value_counts()

dfsub['PREMALIGNANT_POLYP'] = (dfsub.NUM_PREMALIGNANT_POLYP > 0).astype(int)
dfsub['PREMALIGNANT_POLYP_NO_CANCER'] = dfsub.PREMALIGNANT_POLYP.copy()  ## Excluding non endoscopic cancers from indicator
dfsub.loc[dfsub.EPISODE_RESULT == 'Cancer Detected', 'PREMALIGNANT_POLYP_NO_CANCER'] = 0
dfsub.loc[dfsub.PREMALIGNANT_POLYP_NO_CANCER == 1].EPISODE_RESULT.value_counts()
dfsub.loc[dfsub.PREMALIGNANT_POLYP == 1].EPISODE_RESULT.value_counts()

# Indicator for high risk findings
# These also include LNPCP as I don't have data about stalk.
mask = dfsub.ANON_SUBJECT_EPIS_ID.isin(high_risk) & ~dfsub.ANON_SUBJECT_EPIS_ID.isin(cancer_endo)
dfsub['HIGH_RISK_FINDINGS'] = 0
dfsub.loc[mask, 'HIGH_RISK_FINDINGS'] = 1

dfsub['HIGH_RISK_FINDINGS_NO_CANCER'] = dfsub.HIGH_RISK_FINDINGS.copy()  ## Excluding non endoscopic cancers from indicator
dfsub.loc[dfsub.EPISODE_RESULT == 'Cancer Detected', 'HIGH_RISK_FINDINGS_NO_CANCER'] = 0
dfsub.loc[dfsub.HIGH_RISK_FINDINGS_NO_CANCER == 1].EPISODE_RESULT.value_counts()

dfsub.loc[dfsub.HIGH_RISK_FINDINGS == 1].EPISODE_RESULT.value_counts()

assert dfsub.shape[0] == n_epi
dfsub.columns

# Check consistency between high-risk findings according to old def and new def 
s = dfsub.loc[dfsub.EPISODE_START_DATE >= '2023-01-01']
assert s.ANON_SUBJECT_EPIS_ID.nunique() == s.shape[0]
h_old = s.loc[s.EPISODE_RESULT == 'High-risk findings'].ANON_SUBJECT_EPIS_ID
h_and_lnpcp_old = s.loc[s.EPISODE_RESULT.isin(['High-risk findings', 'LNPCP'])].ANON_SUBJECT_EPIS_ID
h_lnpcp_old = s.loc[s.EPISODE_RESULT.isin(['LNPCP'])].ANON_SUBJECT_EPIS_ID

## Old in new
test = h_old.isin(high_risk_no_cancer)
stat['Since 2023, episodes with result as high-findings that are in new high-risk findings'] = [test.sum().item(), test.mean().item()*100]
test = h_and_lnpcp_old.isin(high_risk_no_cancer)
stat['Since 2023, episodes with result as high-risk findings or LNPCP that are in new high-risk findings'] = [test.sum().item(), test.mean().item()*100]
test = h_lnpcp_old.isin(high_risk_no_cancer)
stat['Since 2023, episodes with result as LNPCP that are in new high-risk findings'] = [test.sum().item(), test.mean().item()*100]

## New in old
hnew = s.loc[s.ANON_SUBJECT_EPIS_ID.isin(high_risk_no_cancer)].ANON_SUBJECT_EPIS_ID
test = hnew.isin(h_old)
stat['Since 2023, episodes with newly defined high-risk findings that have episode result as high-risk findings'] = [test.sum().item(), test.mean().item()*100]
test = hnew.isin(h_and_lnpcp_old)
stat['Since 2023, episodes with newly defined high-risk findings that have episode result as high-risk findings or LNPCP'] = [test.sum().item(), test.mean().item()*100]

# Create new outcome variable 
dfsub['OUTCOME'] = 'Other'

repl = {'Definitive normal FOBt outcome': 'FIT negative',
        'Definitive abnormal FOBt outcome': 'FIT positive, no investigation',
        'Normal (No Abnormalities Found)': 'No abnormality at whole colon investigation',
        'No Result': 'No result at whole colon investigation',
        'Cancer Detected': 'Colorectal cancer',
        'FOBt inadequate participation': 'FIT inadequate participation'}
u = dfsub.EPISODE_RESULT.unique()
for key, value in repl.items():
    print(key)
    assert key in u
    dfsub.loc[dfsub.EPISODE_RESULT == key, 'OUTCOME'] = value

dfsub.OUTCOME = dfsub.OUTCOME.replace(repl)

dfsub.loc[dfsub.EPISODE_RESULT == 'Abnormal', 'OUTCOME'] = 'Other abnormality'  # First assign abnormal
dfsub.loc[dfsub.ANON_SUBJECT_EPIS_ID.isin(polyp_advanced.ANON_SUBJECT_EPIS_ID), 'OUTCOME'] = 'Advanced polyp' 
dfsub.loc[dfsub.PREMALIGNANT_POLYP == 1, 'OUTCOME'] = 'Premalignant polyp(s)'  # Then assign premalignant (over-writes some abnormal)
dfsub.loc[dfsub.HIGH_RISK_FINDINGS == 1, 'OUTCOME'] = 'High-risk findings or LNPCP'  # Then assign high-risk (over-writes some premalignant)
dfsub.loc[dfsub.EPISODE_RESULT == 'Cancer Detected', 'OUTCOME'] = 'Colorectal cancer'  # Then assign cancer
dfsub.OUTCOME.value_counts()

# Double check outcomes over time
dfsub['episode_start_year'] = dfsub.EPISODE_START_DATE.dt.year
dfsub['episode_start_month'] = dfsub.EPISODE_START_DATE.dt.month
dfsub['episode_start_quarter'] = np.nan
dfsub['episode_start_month'].sort_values().unique()
dfsub.loc[dfsub.episode_start_month.isin([1, 2, 3]), 'episode_start_quarter'] = 1
dfsub.loc[dfsub.episode_start_month.isin([4, 5, 6]), 'episode_start_quarter'] = 2
dfsub.loc[dfsub.episode_start_month.isin([7, 8, 9]), 'episode_start_quarter'] = 3
dfsub.loc[dfsub.episode_start_month.isin([10, 11, 12]), 'episode_start_quarter'] = 4
dfsub['episode_start_quarter'].value_counts()

t = dfsub.loc[dfsub.episode_start_year >= 2019]
out = t.groupby(['episode_start_year', 'episode_start_quarter']).OUTCOME.value_counts()
out = out.reset_index().pivot(index=['episode_start_year', 'episode_start_quarter'], columns=['OUTCOME'], values='count')
out = out.transpose()

out_perc = (out / out.sum(axis=0) * 100)
out_perc.sum(axis=0)
out_perc = out_perc.round(3)
out_perc
out_perc = out_perc.iloc[:, 1:]

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16, 8))
for i in out_perc.index:
    print(i)
    if i == 'FIT negative':
        continue
    y = out_perc.loc[i]
    xlab = [str(int(j[0])) + '-' + str(int(j[1])) for j in y.index]
    x = range(len(y))
    ax.plot(x, y, label=i)
ax.set_xticks(x)
ax.set_xticklabels(xlab)
ax.legend(frameon=False)
#plt.show()


stat = pd.DataFrame.from_dict(stat, orient='index')
stat = stat.reset_index()
stat.columns = ['Characteristic', 'Count', 'Percent']
stat

stat.to_csv(out_path / 'dq_high-risk-findings.csv', index=False)

#endregion


# ---- 7. Combine data quality summaries ---- 
#region
files = {'dq_dates.csv': 'Dates', 
         'dq_cancer.csv': 'Cancer outcomes', 
         'dq_reading.csv': 'Test kit results',
         'dq_high-risk-findings.csv': 'High-risk findings'
         }
dq = pd.DataFrame()
for f, cat in files.items():
    t = pd.read_csv(out_path / f)
    #t.columns = ['Characteristic', 'Count', 'Percent']
    t['Category'] = cat
    dq = pd.concat(objs=[dq, t], axis=0)
dq = dq[['Category', 'Characteristic', 'Count', 'Percent']]
dq.to_csv(out_path / 'dq_dates-cancer-kit.csv', index=False)
#endregion

