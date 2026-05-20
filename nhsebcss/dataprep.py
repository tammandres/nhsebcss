"""Prepare data for analysis
Note: 
- in VSCODE, to replace capital letter variable names with lowercase,
  use ((?:[A-Z]+_){1,}[A-Z]+) for find with Aa and .* turned on
  and \L$0 for replace
- this code was run on a 4 CPU machine with 16 GB of RAM 
  at one point it used at least 80% of available memory.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path


# Paths
data_path = Path("C:/Users/andres.Tamm/Desktop/BCSS-16540-LIVE-1")  # Where raw data is
data_path_clean = Path("C:/Users/andres.Tamm/Desktop/bcss_clean_data")  # Where to save cleaned data
out_path = Path("Z:/andres/nhsebcss/results/primary")  # Where to save summary stats

if not data_path_clean.exists():
    data_path_clean.mkdir(exist_ok=True)
if not out_path.exists():
    out_path.mkdir(exist_ok=True)

# Redirect output
sys.stdout = open(out_path / 'dataprep_log.txt', 'w')


# ---- 1. Read data, reformat dates, transform to one row per episode,
# including most recent kit result per episode ----
#region
print('\n====READING DATA...')

# Read data
df = pd.read_csv(data_path / 'BCSS-16540-T1-LIVE-1.csv')
df.columns = df.columns.str.lower()
print(df.shape)
print(df.columns)

# Check missingness
nmis = df.isna().sum()
print(nmis.sort_values())
print(nmis[nmis > 0])

# Dates to datetime
date_cols = ['episode_start_date', 'episode_end_date', 'test_kit_logged_date', 'test_kit_result_date']
assert all([c in df.columns for c in date_cols])
for c in date_cols:
    print(c)
    df[c] = pd.to_datetime(df[c], format='%m/%Y', exact=True)

# Get data with one row per episode (episode data, without kit data here)
n_epi = df.anon_subject_epis_id.nunique()
dfsub = df[['anon_screening_subject_id', 'anon_subject_epis_id', 
            'episode_start_date', 'episode_end_date', 'prevalent_incident_status', 
            'episode_status', 'episode_subtype', 'subject_age_at_episode_start',
            'subject_gender', 'imd_quintile', 'imd_decile', 'imd_score', 'imd_rank',
            'episode_result']]
dfsub = dfsub.drop_duplicates()
assert dfsub.shape[0] == n_epi


# .... [START] Get one fit result per episode ....
# Note that of > 18 million episodes, 736 have two or three readings per episode (three < 10)
# These episodes could be excluded from analysis. 
# However, to maximize data use, the most recent reading can be chosen

## Get test kit data and drop missing analyser readings
fit = df[['anon_subject_epis_id', 'test_kit_logged_date', 'test_kit_result_date', 
          'kit_result', 'analyser_reading_used', 'analyser_error_code']]
fit = fit.dropna(subset=['analyser_reading_used'])

## Then drop invalidated and spoilt readings
print(fit.kit_result.unique())
mask = fit.kit_result.str.lower().isin(['invalidated', 'spoilt'])
print('Num spoilt/invalidated readings:', mask.sum())
assert mask.sum() > 0
assert fit.loc[mask, 'kit_result'].nunique() == 2
print(fit.loc[mask, 'analyser_reading_used'].iloc[0:10])
fit.loc[mask, 'analyser_reading_used'] = np.nan  # Drop invalidated readings
fit = fit.dropna(subset=['analyser_reading_used'])
n_epi_valid_fit = fit.anon_subject_epis_id.nunique()

## Drop duplicate rows
fit = fit.drop_duplicates()

## How many episodes have more than 1 reading? 736 have 2 or 3 readings; < 10 have three (rare)
nfit = fit.groupby('anon_subject_epis_id').size()
print(nfit.value_counts())
dfsub['mult_reading'] = 0
dfsub.loc[dfsub.anon_subject_epis_id.isin(nfit[nfit > 1].index), 'mult_reading'] = 1
print(dfsub.mult_reading.sum())

## How many episodes have multiple qualitatively different readings (i.e. abnormal and normal)? 120
fit_mult = fit.loc[fit.anon_subject_epis_id.isin(nfit[nfit > 1].index)]
fit_mult.kit_result.unique()
n_unique_results = fit_mult.groupby('anon_subject_epis_id').kit_result.nunique()
n_unique_results.value_counts()

## How many episodes have multiple qualitatively different readings on same kit logged date? 15
## Given that only valid kit results are retained, only two different qualitative results are possible (abnormal, normal)
fit_mult_sub = fit_mult.loc[fit_mult.anon_subject_epis_id.isin(n_unique_results[n_unique_results == 2].index)]
n_date_sub = fit_mult_sub.groupby('anon_subject_epis_id').test_kit_logged_date.nunique()
s = fit_mult_sub.loc[fit_mult_sub.anon_subject_epis_id.isin(n_date_sub[n_date_sub == 1].index)]
print(s.anon_subject_epis_id.nunique())

## How many episodes have kit results (not necessarily qualitatively different) on same logged date? 93
## Create an indicator for these in the main episode table
n_date = fit_mult.groupby('anon_subject_epis_id').test_kit_logged_date.nunique()
fit_mult_same_date = fit_mult.loc[fit_mult.anon_subject_epis_id.isin(n_date[n_date == 1].index)]
print(fit_mult_same_date.anon_subject_epis_id.nunique())
dfsub['mult_reading_same_date'] = 0
dfsub.loc[dfsub.anon_subject_epis_id.isin(fit_mult_same_date.anon_subject_epis_id), 'mult_reading_same_date'] = 1
print(dfsub.mult_reading_same_date.sum())

## Get the most recent reading per episode.
## If two readings have the same date, then 
##   (A) if the readings are qualitatively different, choose the one that corresponds to the episode result
##   (B) if readings are not qualitatively different, choose the one with higher reading 
##       (.idxmax below selects the first row if two values the same: 
##         "Return index of first occurrence of maximum over requested axis")
## Only 96 episodes have more than 1 reading with the same result date, 
## and 14 have qualitatively different readings on the same date
tmp = fit.loc[fit.anon_subject_epis_id.isin(nfit[nfit > 1].index) & fit[['anon_subject_epis_id', 'test_kit_result_date']].duplicated(keep=False)]
print(tmp.anon_subject_epis_id.nunique())
tmp = tmp.loc[~tmp[['anon_subject_epis_id', 'kit_result']].duplicated(keep=False)]
print(tmp.anon_subject_epis_id.nunique())

#### For multiple qual dif readings, choose reading based on epi result
fit_mult_sub = fit_mult.loc[fit_mult.anon_subject_epis_id.isin(n_unique_results[n_unique_results == 2].index)]
n_date_sub = fit_mult_sub.groupby('anon_subject_epis_id').test_kit_logged_date.nunique()
s = fit_mult_sub.loc[fit_mult_sub.anon_subject_epis_id.isin(n_date_sub[n_date_sub == 1].index)]
print(s.anon_subject_epis_id.nunique())

print(s.shape)
s = s.merge(df[['anon_subject_epis_id', 'episode_result']].drop_duplicates(), how='left')
print(s.shape)
s[['anon_subject_epis_id', 'kit_result', 'episode_result']]

fit_pos = ['Abnormal', 'Normal (No Abnormalities Found)', 'Cancer Detected',
           'Definitive abnormal FOBt outcome', 'High-risk findings']
fit_neg = ['Definitive normal FOBt outcome']
assert all(outcome in df.episode_result.unique() for outcome in fit_pos + fit_neg)

#### The lines below work, because episodes have only one result,
#### so s_pos only contains readings belonging to FIT positive episodes and selects the abnormal reading
#### and s_neg only contains readings belonging to FIT negative episodes and selects the normal reading
s_pos = s.loc[s.episode_result.isin(fit_pos) & (s.kit_result == 'ABNORMAL')]
s_neg = s.loc[s.episode_result.isin(fit_neg) & (s.kit_result == 'NORMAL')]
fit_mult_qual_dif = pd.concat(objs=[s_pos, s_neg], axis=0).drop(labels=['episode_result'], axis=1)
assert s.anon_subject_epis_id.isin(fit_mult_qual_dif.anon_subject_epis_id).all()
assert fit_mult_qual_dif.anon_subject_epis_id.nunique() == fit_mult_qual_dif.shape[0]

#### For remaining readings, choose the most recent (if dif dates), or max reading (if same date)
fit.shape[0]
fit = fit.loc[~fit.anon_subject_epis_id.isin(fit_mult_qual_dif.anon_subject_epis_id)]
fit.shape[0]

fit = fit.sort_values(by=['analyser_reading_used'], ascending=False)  # max analyser reading is the first row
idxmax = fit.groupby('anon_subject_epis_id').test_kit_result_date.idxmax()
fit = fit.loc[idxmax]
fit = pd.concat(objs=[fit, fit_mult_qual_dif], axis=0)
fit = fit.sort_values(by=['test_kit_logged_date'])

assert fit.shape[0] == fit.anon_subject_epis_id.nunique()
fit.isna().sum()

## Add num readings
nreading = nfit.rename('num_reading').reset_index()
fit = fit.merge(nreading, how='left')
assert fit.shape[0] == fit.anon_subject_epis_id.nunique()
assert fit.shape[0] == n_epi_valid_fit

# .... [END] Get the most recent fit result per episode ....


# Get most recent kit result for episodes without valid analyser reading
kit = df[['anon_subject_epis_id', 'test_kit_logged_date', 'test_kit_result_date', 'kit_result', 'analyser_reading_used', 'analyser_error_code']]
kit = kit.loc[~kit.anon_subject_epis_id.isin(fit.anon_subject_epis_id)]
kit = kit.dropna(subset=['kit_result']).drop_duplicates()
assert kit.shape[0] != kit.anon_subject_epis_id.nunique()
print(kit.shape[0], kit.anon_subject_epis_id.nunique())
idmax = kit.groupby('anon_subject_epis_id').test_kit_logged_date.idxmax()
kit = kit.loc[idmax]
kit.analyser_reading_used = np.nan  # Set to nan as some readings that are not nan are spoilt or invalidated
assert kit.shape[0] == kit.anon_subject_epis_id.nunique()
assert kit.analyser_reading_used.isna().all()

# Get a dataframe with single kit per episode
fit_and_kit = pd.concat(objs=[fit, kit], axis=0)
del fit, kit

# Add num kits
nkit = df.groupby('anon_subject_epis_id').anon_test_kit_id.nunique().rename('num_kit').reset_index()
fit_and_kit = fit_and_kit.merge(nkit, how='left')
assert fit_and_kit.shape[0] == fit_and_kit.anon_subject_epis_id.nunique()

# Add maximum FIT result, and kit results without FIT result, to the data
dfsub = dfsub.merge(fit_and_kit, how='left', on='anon_subject_epis_id')
assert dfsub.shape[0] == n_epi
dfsub.isna().sum()


# Get year for each date
dfsub['episode_start_year'] = dfsub.episode_start_date.dt.year
dfsub['episode_end_year'] = dfsub.episode_end_date.dt.year
dfsub['test_kit_logged_year'] = dfsub.test_kit_logged_date.dt.year
dfsub['test_kit_result_year'] = dfsub.test_kit_result_date.dt.year
assert dfsub.shape[0] == n_epi


# Get quarter for each date
dfsub['episode_start_quarter'] = dfsub.episode_start_date.dt.quarter
dfsub['episode_end_quarter'] = dfsub.episode_end_date.dt.quarter
dfsub['test_kit_logged_quarter'] = dfsub.test_kit_logged_date.dt.quarter
dfsub['test_kit_result_quarter'] = dfsub.test_kit_result_date.dt.quarter
assert dfsub['episode_start_quarter'].nunique() == 4


# Create age groups
age = dfsub.subject_age_at_episode_start
print(age.isna().sum())
age_max = age.max() + 1
age_group = pd.cut(age, bins=[50, 60, 70, 80, 90, age_max], right=False)
age_group.value_counts(sort=False)
dfsub['age_group'] = age_group.astype(str)
dfsub.loc[age < 50, 'age_group'] = '<50'
dfsub.age_group = dfsub.age_group.replace({'[50, 60)': '50-59',
                                           '[60, 70)': '60-69',
                                           '[70, 80)': '70-79',
                                           '[80, 90)': '80-89'})
dfsub.age_group = dfsub.age_group.replace({'[90, ' + str(age_max) + ')': '90+'})
print(dfsub.age_group.unique())
assert dfsub.age_group.isna().sum() == 0
del age_group

age_group_screen = pd.cut(age, bins=[50, 54, 60, 75, age_max], right=False)
age_group_screen.value_counts(sort=False)
dfsub['age_group_screen'] = age_group_screen.astype(str)
dfsub.loc[age < 50, 'age_group_screen'] = '<50'
dfsub.age_group_screen = dfsub.age_group_screen.replace({'[50, 54)': '50-53',
                                                         '[54, 60)': '54-59',
                                                         '[60, 75)': '60-74'})
dfsub.age_group_screen = dfsub.age_group_screen.replace({'[75, ' + str(age_max) + ')': '75+'})
print(dfsub.age_group_screen.unique())
assert dfsub.age_group_screen.isna().sum() == 0
del age_group_screen

dfsub['age_group_screen2'] = dfsub.age_group_screen.copy()
dfsub['age_group_screen2'] = dfsub['age_group_screen2'].replace({'54-59': '50-59', 
                                                                 '50-53': '50-59'})

# Age groups similar to Moss et al 2016
age_group_def_moss = {'59-64': np.arange(59, 64+1),
                      '65-69': np.arange(65, 69+1),
                      '70-75': np.arange(70, 75+1)}
age_group_moss = np.empty(age.shape, dtype=np.object_)
for label, values in age_group_def_moss.items():
    age_group_moss[age.isin(values)] = label
age_group_moss[age_group_moss == None] = 'other'
dfsub['age_group_moss'] = age_group_moss
dfsub.age_group_moss.value_counts()
values_other = dfsub.loc[dfsub.age_group_moss=='other', 'subject_age_at_episode_start']
assert all((values_other < 59) | (values_other >75))
assert dfsub.age_group_moss.isna().sum() == 0
del age_group_moss


# Reformat imd_decile so it matches imd_quintile
mask = dfsub.imd_decile.isna()
dfsub['imd_decile'] = dfsub.imd_decile.astype(str)
dfsub.imd_decile = dfsub.imd_decile.str.replace('.0', '', regex=False).str.zfill(2)
dfsub.imd_decile = dfsub.imd_decile.replace({'01': '01 - Most deprived', '10': '10 - Least deprived'})
dfsub.loc[mask, 'imd_decile'] = np.nan
dfsub.imd_decile.unique()

for c in dfsub.columns:
    print(c)

#endregion


# ---- 2. Identify premalignant and advanced polyps ----
#region
print('\n====CLASSIFYING POLYPS...')

# Get endoscopy findings
df_polyp = pd.read_csv(data_path / 'BCSS-16540-T2-LIVE-1.csv')
df_polyp.columns = df_polyp.columns.str.lower()

df_cancer_endo = pd.read_csv(data_path / 'BCSS-16540-T3-LIVE-1.csv')
df_cancer_endo.columns = df_cancer_endo.columns.str.lower()

# Subset endoscopy findings to episodes present in the screening episodes table
# (It was an outer join during data extraction)
print(df_polyp.shape, df_cancer_endo.shape)
df_polyp = df_polyp.loc[df_polyp.anon_subject_epis_id.isin(df.anon_subject_epis_id)]
df_cancer_endo = df_cancer_endo.loc[df_cancer_endo.anon_subject_epis_id.isin(df.anon_subject_epis_id)]
print(df_polyp.shape, df_cancer_endo.shape)

# Find cancers detected at endoscopy, add indicator to episodes table
cancer_endo1 = df_polyp.loc[df_polyp.carcinoma == 'Yes'].anon_subject_epis_id.drop_duplicates()
cancer_endo2 = df_cancer_endo.anon_subject_epis_id.drop_duplicates()
cancer_endo = pd.concat(objs=[cancer_endo1, cancer_endo2], axis=0).drop_duplicates()
print(len(cancer_endo))
cancer_all = dfsub.loc[dfsub.episode_result == 'Cancer Detected'].anon_subject_epis_id.drop_duplicates()
print(len(cancer_all))

dfsub['cancer_at_endoscopy'] = 0
dfsub.loc[dfsub.anon_subject_epis_id.isin(cancer_endo), 'cancer_at_endoscopy'] = 1

# Get premalignant polyps (Rutter et al 2020, https://doi.org/10.1136/gutjnl-2019-319858)
#  Adenomatous polyp, 
#  or serrated polyp excluding 1-5mm rectal hyperplastic polyps
# Note: sub type of serrated polyps makes sense with respect to Rutter: hyperplastic, sessile, traditional, mixed
print(df_polyp.polyp_type.unique())
mask = df_polyp.polyp_type.fillna('').str.lower().str.contains('adenom', regex=True)
adenoma = df_polyp.loc[mask]
print(adenoma.polyp_type.unique())
print(adenoma.sub_type.unique())
assert adenoma.polyp_type.str.lower().str.contains('adenoma').all()

mask = df_polyp.polyp_type.fillna('').str.lower().str.contains('serrated', regex=True)
serrated = df_polyp.loc[mask]
print(serrated.polyp_type.unique())
print(serrated.sub_type.unique())
assert serrated.polyp_type.str.lower().str.contains('serrated').all()

df_polyp.polyp_location.unique()
mask = (serrated.polyp_location.str.lower() == 'rectum') & (serrated.polyp_histology_size <= 5) & (serrated.sub_type == 'Hyperplastic polyp')
print(serrated.loc[mask, 'polyp_location'].unique())
print(serrated.loc[mask, 'polyp_histology_size'].max())
print(serrated.loc[mask, 'sub_type'].value_counts())
serrated_premalignant = serrated.loc[~mask]
print(serrated.anon_subject_epis_id.nunique(), serrated_premalignant.anon_subject_epis_id.nunique())

polyp_pm = pd.concat(objs=[adenoma, serrated_premalignant], axis=0)
polyp_pm = polyp_pm.drop_duplicates()
print(polyp_pm.anon_subject_epis_id.nunique())
print(polyp_pm.polyp_type.value_counts())

df_polyp['premalignant'] = 0
df_polyp.loc[df_polyp.anon_polyp_id.isin(polyp_pm.anon_polyp_id), 'premalignant'] = 1

dfsub['premalignant_polyp'] = 0
dfsub.loc[dfsub.anon_subject_epis_id.isin(polyp_pm.anon_subject_epis_id), 'premalignant_polyp'] = 1

# Get advanced premalignant polyps (Rutter et al, 2020)
#  Adenoma with any of 
#   1. size >= 10 mm
#   2. high-grade dysplasia
#   3. tubulovillous or villous morphology
#  serrated polyp with any of
#   4. size >= 10 mm
#   5. any dysplasia (including traditional serrated adenomas)
a1 = adenoma.loc[adenoma.polyp_histology_size >= 10]
print(a1.polyp_type.unique(), a1.polyp_histology_size.min())
assert a1.polyp_histology_size.min() >= 10

a2 = adenoma.loc[adenoma.dysplasia.str.lower().isin(['high grade dysplasia'])]
print(a2.dysplasia.unique())
assert a2.dysplasia.str.lower().str.contains('high grade').all()

a3 = adenoma.loc[adenoma.sub_type.str.lower().str.contains('villous')]
print(a3.sub_type.value_counts())
a3.anon_polyp_id.isin(a2.anon_polyp_id).mean() 
a3.dysplasia.value_counts(dropna=False)

a4 = serrated.loc[serrated.polyp_histology_size >= 10]
print(a4.polyp_type.unique(), a4.polyp_histology_size.min())
assert a4.polyp_histology_size.min() >= 10

print(df_polyp.dysplasia.unique())
a5 = serrated.loc[serrated.dysplasia.str.lower().fillna('').str.contains('high grade|low grade', regex=True)]
print(a5.dysplasia.unique())
assert a5.dysplasia.str.lower().str.contains('low|high').all()

a6 = serrated.loc[serrated.sub_type.str.lower().isin(['traditional serrated adenoma'])]
print(a6.sub_type.unique())
assert a6.sub_type.str.lower().str.contains('traditional serrated').all()
a6.anon_polyp_id.isin(a5.anon_polyp_id).mean()  # About 96% of the tsa are among serrated polyps with dysplasia, the remaining are not as dysplasia is not reproted
a6.loc[~a6.anon_polyp_id.isin(a5.anon_polyp_id)].dysplasia.value_counts()

polyp_advanced = pd.concat(objs=[a1, a2, a3, a4, a5, a6], axis=0)
polyp_advanced = polyp_advanced.drop_duplicates()
print(polyp_advanced.polyp_type.unique(), polyp_advanced.dysplasia.unique(), polyp_advanced.sub_type.unique())

df_polyp['advanced'] = 0
df_polyp.loc[df_polyp.anon_polyp_id.isin(polyp_advanced.anon_polyp_id), 'advanced'] = 1

dfsub['advanced_polyp'] = 0
dfsub.loc[dfsub.anon_subject_epis_id.isin(polyp_advanced.anon_subject_epis_id), 'advanced_polyp'] = 1

# Get non-advanced premalignant polyps (to be able to compute the number of episodes with at least one non-advanced polyp)
polyp_nonadvanced = polyp_pm.loc[~polyp_pm.anon_polyp_id.isin(polyp_advanced.anon_polyp_id)]
dfsub['nonadvanced_polyp'] = 0
dfsub.loc[dfsub.anon_subject_epis_id.isin(polyp_nonadvanced.anon_subject_epis_id), 'nonadvanced_polyp'] = 1
print(dfsub.advanced_polyp.sum(), dfsub.nonadvanced_polyp.sum(), dfsub.premalignant_polyp.sum())
mask = (dfsub.nonadvanced_polyp == 1) & (dfsub.advanced_polyp == 1)
mask.sum()  # >0 as expected

# Get adenomas and premalignant polyps >= 20mm
adenoma_20 = adenoma.loc[adenoma.polyp_histology_size >= 20]
polyp_pm_20 = polyp_pm.loc[polyp_pm.polyp_histology_size >= 20]

dfsub['adenoma_20'] = 0
dfsub.loc[dfsub.anon_subject_epis_id.isin(adenoma_20.anon_subject_epis_id), 'adenoma_20'] = 1

dfsub['premalignant_20'] = 0
dfsub.loc[dfsub.anon_subject_epis_id.isin(polyp_pm_20.anon_subject_epis_id), 'premalignant_20'] = 1

# Get episodes with high risk findings
#  >= 5 premalignant polyps
#  >= 2 premalignant polyps where at least 1 is advanced
#  a single adenoma greater than 20mm [added by James East]
num_polyp = polyp_pm.groupby('anon_subject_epis_id').anon_polyp_id.nunique().rename('num_premalignant_polyp')
num_polyp = num_polyp.reset_index()

high_risk1 = num_polyp[num_polyp.num_premalignant_polyp >= 5].anon_subject_epis_id.drop_duplicates()
high_risk2 = num_polyp[num_polyp.num_premalignant_polyp >= 2].anon_subject_epis_id.drop_duplicates()
high_risk2 = high_risk2[high_risk2.isin(polyp_advanced.anon_subject_epis_id)]
high_risk3 = adenoma_20.anon_subject_epis_id.drop_duplicates()
high_risk = pd.concat(objs=[high_risk1, high_risk2, high_risk3], axis=0).drop_duplicates()
print(high_risk.isin(polyp_pm.anon_subject_epis_id).mean())
print(polyp_pm.anon_subject_epis_id.isin(high_risk).mean())

dfsub['high_risk_findings'] = 0
dfsub.loc[dfsub.anon_subject_epis_id.isin(high_risk), 'high_risk_findings'] = 1

# Add number of premalignant polyps as well
shape_before = dfsub.shape[0]
dfsub = dfsub.merge(num_polyp, how='left', on='anon_subject_epis_id')
assert shape_before == dfsub.shape[0]
dfsub.num_premalignant_polyp = dfsub.num_premalignant_polyp.fillna(0)
dfsub.num_premalignant_polyp.value_counts()

assert dfsub.shape[0] == n_epi
dfsub.columns

# In addition, create the advanced adenoma category based on Moss et al definition for comparison
# "Subjects with high-risk adenomas (≥5 adenomas or ≥3 adenomas at least one of which was ≥1 cm) 
#  or intermediate-risk adenomas (3–4 small adenomas or at least one ≥1 cm) 
#  were defined as having advanced adenomas"
mask = df_polyp.polyp_type.str.lower().str.contains('adenoma') | df_polyp.sub_type.str.lower().str.contains('adenoma')
adenomas_moss = df_polyp.loc[mask]
adenomas_moss.polyp_type.value_counts()
adenomas_moss.sub_type.value_counts()
num_adenoma_moss = adenomas_moss.groupby('anon_subject_epis_id').size().rename('n_adenoma')

large_adenoma_moss = adenomas_moss.loc[adenomas_moss.polyp_histology_size >= 10]
small_adenoma_moss = adenomas_moss.loc[adenomas_moss.polyp_histology_size < 10]
num_small_adenoma_moss = small_adenoma_moss.groupby('anon_subject_epis_id').size().rename('n_adenoma')

episodes_5_small_adenomas = num_small_adenoma_moss[num_small_adenoma_moss >= 5].index.to_series()
episodes_3to4_small_adenomas = num_small_adenoma_moss[num_small_adenoma_moss.isin([3,4])].index.to_series()
episodes_large_adenoma_moss = large_adenoma_moss.anon_subject_epis_id.drop_duplicates()
episodes_advanced_adenoma_moss = pd.concat([episodes_5_small_adenomas,
                                            episodes_3to4_small_adenomas,
                                            episodes_large_adenoma_moss]).drop_duplicates()
print(len(episodes_advanced_adenoma_moss), adenomas_moss.anon_subject_epis_id.nunique())
dfsub['advanced_adenoma_moss'] = 0
dfsub.loc[dfsub.anon_subject_epis_id.isin(episodes_advanced_adenoma_moss), 'advanced_adenoma_moss'] = 1

print(dfsub.advanced_polyp.sum(), dfsub.advanced_adenoma_moss.sum())
print(dfsub[['advanced_polyp', 'advanced_adenoma_moss']].value_counts())

#endregion


# ---- 3. Categorise episodes based on polyp findings ----
#region
print('\n====CATEGORISING OUTCOMES...')

# ........................................................
# .... 3.1. Classification of outcomes based on consulting with East and Rutter
# ........................................................
#region

# Main categories:
# · CRC
# · Advanced premalignant polyp
# · Non-advanced premalignant polyp
# · No premalignant polyps (NB this would include e.g. inflammatory polyps)
# 
# Advanced premalignant polyp defined as any of:
# · Adenoma with any of
#    - Size ≥10mm
#    - high-grade dysplasia
#    - tubulovillous or villous morphology
# · Serrated with any of
#    - Size ≥10mm
#    - Any dysplasia (so this would include all traditional serrated adenomas)

## First assign a value 'Other', which will be overwritten 
## and will serve as a placeholder for values not yet assigned
dfsub['outcome'] = 'Other'

## Then add some existing outcome categories, but rename them for simplicity
repl = {'Normal (No Abnormalities Found)': 'No abnormality',
        'No Result': 'No result',
        'Cancer Detected': 'Colorectal cancer',
        'Definitive normal FOBt outcome': 'FIT negative',
        'Definitive abnormal FOBt outcome': 'FIT positive, no investigation',
        'FOBt inadequate participation': 'No FIT result'}
u = dfsub.episode_result.unique()
for key, value in repl.items():
    print(key)
    assert key in u
    dfsub.loc[dfsub.episode_result == key, 'outcome'] = value

## Any advanced polyp(s) present in the remaining?
mask = (dfsub.outcome == 'Other') & (dfsub.advanced_polyp == 1)
dfsub.loc[mask, 'outcome'] = 'Advanced premalignant polyp'  

## Any premalignant polyp(s) present in the remaining?
mask = (dfsub.outcome == 'Other') & (dfsub.premalignant_polyp == 1)
dfsub.loc[mask, 'outcome'] = 'Non-advanced premalignant polyp'

## Polyps without endoscopy record (862) -> in that case, there is no endoscopy record
all_outcomes = df.episode_result.unique()
polyp_outcomes = ['Low-risk Adenoma', 'Intermediate-risk Adenoma', 'High-risk findings', 'High-risk Adenoma', 'LNPCP']
assert all(p in all_outcomes for p in polyp_outcomes)
mask = (dfsub.outcome == 'Other') & (dfsub.episode_result.isin(polyp_outcomes)) & (~dfsub.anon_subject_epis_id.isin(df_polyp.anon_subject_epis_id))
mask.sum()
dfsub.loc[mask, 'episode_result'].value_counts()
dfsub.loc[mask, 'outcome'] = 'Polyp(s) without endoscopy record'


## ---- [START] Check remaining epi where episode_result is a polyp finding and endoscopy data does not match it
## 146 epi remain with polyp results but endoscopy data does not support classification
## classify these too as polyps without endoscopy record
assert dfsub.outcome.isna().sum() == 0
dfsub.loc[dfsub.outcome == 'Other', 'episode_result'].value_counts()
dfsub.loc[dfsub.outcome == 'Other', 'outcome'].value_counts()

### None of the remaining episodes with the official result as high-risk findings meet the high-risk findings category based on the recorded polyps
tmp = dfsub.loc[(dfsub.outcome == 'Other') & (dfsub.episode_result == 'High-risk findings')]
print(tmp.shape)
df_polyp.loc[df_polyp.anon_subject_epis_id.isin(tmp.anon_subject_epis_id)]

### None of the remaining episodes with the official result as low-risk adenoma meet the premalignant polyp def based on the recorded polyps
### Most are not adenomas or serrated. And ones that are, are diminuitive rectal hyperplastic ones.
tmp = dfsub.loc[(dfsub.outcome == 'Other') & (dfsub.episode_result == 'Low-risk Adenoma')]
print(tmp.shape)
s = df_polyp.loc[df_polyp.anon_subject_epis_id.isin(tmp.anon_subject_epis_id), ['polyp_type', 'polyp_location', 'sub_type', 'polyp_histology_size']]
s.polyp_type.value_counts()
s.loc[s.polyp_type == 'Serrated lesion']

### None of the remaining episodes with the official result as LNPCP meet the LNPCP criteria based on recorded polyps (i.e. only normal mucosa or inflammatory polyps < 10 mm)
tmp = dfsub.loc[(dfsub.outcome == 'Other') & (dfsub.episode_result == 'LNPCP')]
print(tmp.shape)
s = df_polyp.loc[df_polyp.anon_subject_epis_id.isin(tmp.anon_subject_epis_id)]
s.polyp_type.value_counts()

mask = (dfsub.outcome == 'Other') & (dfsub.episode_result.isin(polyp_outcomes))
mask.sum() # 146
dfsub.loc[mask, 'outcome'] = 'Polyp(s) without endoscopy record'
## ---- [END]


## Assign the remaining uncategorised episodes into the "No premalignant polyps" (other abnormality) category
## The official result for these episodes is predominantly Abnormal, and in 180 cases NaN (open episodes)
assert dfsub.outcome.isna().sum() == 0
dfsub.loc[dfsub.outcome == 'Other', 'episode_result'].value_counts(dropna=False)  
dfsub.loc[(dfsub.outcome == 'Other') & dfsub.episode_result.isna()].episode_status.value_counts()

dfsub.loc[dfsub.outcome == 'Other', 'outcome'].value_counts()
dfsub.loc[dfsub.outcome == 'Other', 'episode_result'].value_counts()

dfsub.loc[dfsub.outcome == 'Other', 'outcome'] = 'No premalignant polyps'


## ---- [START] fix outcomes for missing late-responder episodes
## The issue: when test kit is returned late, a new late-responder episode will be created that will store the episode result
## but our data extract always contains one episode per test kit, which is not necessarily the latest
## So for some participants who returned their kit late, only the first Routine episode without episode result was retained
## was retained in the data extract
dfsub['days_to_return'] = (dfsub.test_kit_logged_date - dfsub.episode_start_date).dt.days
dfsub.days_to_return.describe()

# Episodes with inadequate participation and > 181 day return
# these should have a matching late-responder episode, but that is not present
mask_late = ((dfsub.episode_result == "FOBt inadequate participation") & 
             (dfsub.days_to_return >= 181) #&
             #(dfsub.episode_subtype.isin(['Routine'])) 
             )
print('Inadequate FIT, kit returned >= 181 after start, not marked as late responder:', mask_late.sum())
mask_neg = mask_late & (dfsub.analyser_reading_used < 120)
print('Inadequate FIT, kit returned >= 181 after start, not marked as late responder, FIT negative:', mask_neg.sum())
mask_pos = mask_late & (dfsub.analyser_reading_used >= 120)
print('Inadequate FIT, kit returned >= 181 after start, not marked as late responder, FIT positive:', mask_pos.sum())

dfsub.loc[mask_late, 'episode_subtype'].value_counts()  # Mostly routine (> 150k), Over Age or Opt-In 733
dfsub.loc[mask_neg, 'outcome'] = 'FIT negative' # Assign outcome to be FIT negative
dfsub.loc[mask_pos, 'outcome'] = 'FIT positive, unknown outcome'  # Assign outcome to be unknown

# Check how many episodes are left that have FIT inadequate participation but kit result
mask = ((dfsub.outcome == "No FIT result") &
         dfsub.kit_result.isin(['NORMAL', 'ABNORMAL'])
        )
print(mask.sum())  # 152 remain that have inadequate participation but reading
dfsub.loc[mask, 'episode_subtype'].value_counts()
dfsub.loc[mask, 'days_to_return'].describe()
dfsub.loc[mask, 'num_reading'].value_counts()
dfsub.loc[mask, 'episode_status'].value_counts()

# Check - those marked as late returns but inadequate participation
#  predominantly routine (> 150k), but some other episode subtypes too
dfsub.loc[mask_late, 'episode_subtype'].value_counts()
dfsub.loc[mask_neg, 'episode_subtype'].value_counts()
dfsub.loc[mask_pos, 'episode_subtype'].value_counts()

## ---- [END]


## When episode is open or pending, assign result to NaN (these will be removed anyway)
mask = dfsub.episode_status.isin(['Open', 'Pending'])
mask.sum()
dfsub.loc[mask, 'outcome'] = np.nan


## Check outcome counts
## The category of polyps without endoscopy record is 0.36% of all episodes with investigation - very small effect on other polyp %
dfsub.outcome.value_counts()
dfsub.outcome.value_counts(normalize=True) * 100

outcomes = dfsub.outcome.unique()
outcomes_without_investigation = ['FIT negative', 'No FIT result', 'FIT positive, no investigation']
outcomes_with_investigation = [c for c in outcomes if c not in outcomes_without_investigation]
dfsub.loc[dfsub.outcome.isin(outcomes_with_investigation), 'outcome'].value_counts(normalize=True) * 100


## Create a simplified outcome categorisation as well
dfsub['outcome_simple'] = dfsub.outcome.copy()
dfsub['outcome_simple'] = dfsub['outcome_simple'].replace({'Polyp(s) without endoscopy record': 'Other findings',
                                                           'No premalignant polyps': 'Other findings'
                                                           })
dfsub.outcome_simple.value_counts()


#endregion

# ........................................................
# .... 3.2. Another outcome definition that aims to be more directly comparable to Moss et al 2016
# ....      in how advanced adenomas are defined
# ........................................................
#region

## First assign a value 'Other', which will be overwritten 
## and will serve as a placeholder for values not yet assigned
dfsub['outcome_moss'] = 'Other'

## Then add some existing outcome categories, but rename them for simplicity
repl = {'Normal (No Abnormalities Found)': 'No abnormality',
        'No Result': 'No result',
        'Cancer Detected': 'Colorectal cancer',
        'Definitive normal FOBt outcome': 'FIT negative',
        'Definitive abnormal FOBt outcome': 'FIT positive, no investigation',
        'FOBt inadequate participation': 'No FIT result'}
u = dfsub.episode_result.unique()
for key, value in repl.items():
    print(key)
    assert key in u
    dfsub.loc[dfsub.episode_result == key, 'outcome_moss'] = value

## Any advanced polyp(s)?
mask = (dfsub.outcome_moss == 'Other') & (dfsub.advanced_adenoma_moss == 1)
dfsub.loc[mask, 'outcome_moss'] = 'Advanced adenoma'  
#endregion

#endregion


# ---- 4. Apply inclusion criteria and save cleaned data ----
#region
print('\n====APPLYING INCLUSION CRITERIA AND SAVING THE DATA...')

# Helper function
def remove_episodes(df, epi_rm, n_epi, n_crc, n_pat):
    """Remove episodes in epi_rm from df"""

    # Remove episodes
    print('... num epi before removal', df.shape)
    dfsub = df.loc[~df.anon_subject_epis_id.isin(epi_rm)]  
    print('... num epi after removal', df.shape)

    # Num and percent episodes retrained wrt total num epi (n_epi)
    stats = {} 
    stats['n_epi'] = len(epi_rm)
    stats['p_epi'] = stats['n_epi'] / n_epi * 100

    # Num and percent cancers retained wrt total num crc (n_crc)
    stats['n_crc'] = dfsub.loc[dfsub.episode_result == 'Cancer Detected'].shape[0]
    stats['p_crc'] = stats['n_crc'] / n_crc * 100 

    # Num and percent patients retained
    stats['n_pat'] = df.anon_screening_subject_id.nunique() - dfsub.anon_screening_subject_id.nunique()
    stats['p_pat'] = stats['n_pat'] / n_pat * 100
    return dfsub, stats


cols_view = ['anon_subject_epis_id', 'analyser_reading_used', 'analyser_error_code', 
             'episode_result', 'episode_start_date', 'episode_end_date', 
             'test_kit_logged_date', 'test_kit_result_date']

stat = {}
n_epi = dfsub.shape[0]
n_crc = dfsub.loc[dfsub.episode_result == 'Cancer Detected'].shape[0]
n_pat = dfsub.anon_screening_subject_id.nunique()
stat['Episodes'] = {'n_epi': n_epi,  # total num episodes
                    'p_epi': 100.,
                    'n_crc': n_crc,  # total num cancers
                    'p_crc': 100.,
                    'n_pat': n_pat,  # total num patients
                    'p_pat': 100.
                    }

# episodes with end date before kit logged date? --> keep.
# There are >150k, most have FOBt inadequate participatuon, only < 10 have normal FIT, all ret > 150 days, 15 ret < 181 days
# most of these are probably episoes that would have been marked as late-responders but the late-responder epi is missing from our data
print("Check epi w end date < test kit logged date")
tmp = dfsub.loc[dfsub.episode_end_date < dfsub.test_kit_logged_date]  
tmp.episode_result.value_counts()
tmp.outcome.value_counts()

# episodes with kit returned more than 5 years after episode start? --> keep.
# Only < 10 epi match that criterion.
# Let's keep as we are not doing statistics on return times 
# and presumably the kit logged date and result date are correct
print(dfsub.days_to_return.max(), dfsub.days_to_return.max() / 365.25)
years_to_return = [0, 0.5, 1, 2, 3, 4, 5]
for y in years_to_return:
    d = y * 365.25
    test = dfsub.days_to_return <= d
    print("{} ({:.2f}%) kits returned within {} years ({} days) from episode start; {} after".\
          format(test.sum(), test.mean()*100, y, d, dfsub.shape[0] - test.sum())) 
#epi_rm = dfsub.loc[dfsub.days_to_return > 5*365.25].anon_subject_epis_id
#tmp = df.loc[df.anon_subject_epis_id.isin(epi_rm)]
#print(tmp[cols_view])
#dfsub, s = remove_episodes(dfsub, epi_rm, n_epi, n_crc, n_pat)
#stat['Episodes with kit returned more than 5 years from episode start'] = s

# age < 50 --> remove as outside eligibility criteria
epi_rm = dfsub.loc[dfsub.subject_age_at_episode_start < 50].anon_subject_epis_id
tmp = df.loc[df.anon_subject_epis_id.isin(epi_rm)]
print(tmp[cols_view])
dfsub, s = remove_episodes(dfsub, epi_rm, n_epi, n_crc, n_pat)
stat['Episodes with age < 50'] = s

# open or pending --> remove as we focus on complete episodes and likelihood of bias small due to small num excluded
epi_rm = dfsub.loc[dfsub.episode_status.isin(['Open', 'Pending'])].anon_subject_epis_id
tmp = df.loc[df.anon_subject_epis_id.isin(epi_rm)]
print(tmp[cols_view])
print((tmp.test_kit_logged_date >= '2023-01-01').mean() *100)  # 92%
print((tmp.test_kit_logged_date >= '2023-01-01').sum()) # 4469
dfsub, s = remove_episodes(dfsub, epi_rm, n_epi, n_crc, n_pat)
stat['Episodes open or pending'] = s
print(dfsub.episode_status.unique())

# episode result implies analyser reading but there is no reading (spoilt) --> rm as likely newer kit not incl in our data
# (in 99.8% cases, spoilt kit logged in 2024)
dfsub_with_result = dfsub.loc[~dfsub.episode_result.isin(['FOBt inadequate participation'])]
dfsub_with_result_no_reading = dfsub_with_result.loc[dfsub_with_result.analyser_reading_used.isna()]
epi_rm = dfsub_with_result_no_reading.anon_subject_epis_id
dfsub, s = remove_episodes(dfsub, epi_rm, n_epi, n_crc, n_pat)
stat['Episode has result but no kit reading'] = s

mask = dfsub_with_result_no_reading.test_kit_logged_year >= 2024
print(mask.mean() * 100)  # 99.8%
#epi_rm0 = dfsub_with_result_no_reading.loc[mask].anon_subject_epis_id
#epi_rm1 = epi_rm[~epi_rm.isin(epi_rm0)]

# episodes with multiple analyser readings on the same date --> rm as true kit reading not known
mask = (dfsub.mult_reading_same_date == 1) & (~dfsub.anon_subject_epis_id.isin(fit_mult_qual_dif.anon_subject_epis_id))
epi_rm = dfsub.loc[mask].anon_subject_epis_id
dfsub, s = remove_episodes(dfsub, epi_rm, n_epi, n_crc, n_pat)
stat['Episodes with multiple analyser readings on the same date'] = s

# episodes marked as FIT negative but have a positive reading --> rm as true kit reading not known
# Note that date is in 2024 -> upper datacut limits new data
epi_rm = dfsub.loc[(dfsub.outcome == 'FIT negative') & (dfsub.analyser_reading_used >= 120)].anon_subject_epis_id

tmp = dfsub.loc[dfsub.anon_subject_epis_id.isin(epi_rm)]
tmp['kit_result'].value_counts() # kit result is abnormal
tmp['episode_result'].value_counts() # episode result is normal
tmp.test_kit_logged_date.describe(percentiles=[0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99])
tmp.num_reading.value_counts()

dfsub, s = remove_episodes(dfsub, epi_rm, n_epi, n_crc, n_pat)
stat['Episodes where the result is FIT negative and reading >= 120'] = s

# Episodes where the result implies a positive kit but reading is negative --> rm as true kit reading not known (<10)
out = ['FIT negative', 'No FIT result']
assert all([o in dfsub.outcome.tolist() for o in out])
rm = dfsub.loc[~dfsub.outcome.isin(out) & (dfsub.analyser_reading_used < 120)]
epi_rm = rm.anon_subject_epis_id

print(len(epi_rm))
tmp = dfsub.loc[dfsub.anon_subject_epis_id.isin(epi_rm)]
tmp['analyser_reading_used'].describe()  # Reading is below 50
tmp['kit_result'].value_counts() #  KIT RESULT is normal
tmp['episode_result'].value_counts()  #  But episode result is for example abnormal and other categories
tmp['num_reading'].value_counts()
tmp.num_reading.value_counts()
tmp.test_kit_logged_date.describe()
r = df.loc[df.anon_subject_epis_id.isin(epi_rm), ['anon_subject_epis_id', 'anon_test_kit_id', 'analyser_reading_used',
                                                  'test_kit_logged_date', 'test_kit_result_date', 'kit_result', 'analyser_error_code',
                                                  'episode_result']].drop_duplicates()
print(r.transpose())

dfsub, s = remove_episodes(dfsub, epi_rm, n_epi, n_crc, n_pat)
stat['Episodes where result implies a positive FIT but reading < 120'] = s


# Reformat statistics and save
stat = pd.DataFrame.from_dict(stat, orient='index')
stat = stat.reset_index()
stat.columns = ['Characteristic', 'Number of episodes', 'Percent episodes', 'Number of cancers', 'Percent cancers',
                'Number of patients', 'Percent patients']

n_removed = stat.iloc[1:]['Number of episodes'].sum().item()
ncrc_removed = stat.iloc[1:]['Number of cancers'].sum().item()
npat_removed = n_pat - dfsub.anon_screening_subject_id.nunique()   # Patients can overlap between removed episodes. Hence calc like this.
p_removed = n_removed / n_epi * 100
pcrc_removed = ncrc_removed / n_crc * 100
ppat_removed = npat_removed / n_pat * 100
#npat_removed, ppat_removed = np.nan, np.nan 
row = pd.DataFrame([['All removed episodes', n_removed, p_removed, ncrc_removed, pcrc_removed, npat_removed, ppat_removed]], 
                    columns=stat.columns)
stat = pd.concat(objs=[stat, row], axis=0).reset_index(drop=True)

stat.to_csv(out_path / 'removed_episodes.csv', index=False)


# Save prepared data
dfsub.to_csv(data_path_clean / 'episodes_clean.csv', index=False)


# Save additional data for test kit readings
cols = ['anon_subject_epis_id', 'anon_screening_subject_id', 
        'anon_test_kit_id', 'test_kit_logged_date', 'test_kit_result_date', 'analyser_reading', 'analyser_reading_used',
        'analyser_error_code', 'kit_result']
reading = df.loc[df.anon_subject_epis_id.isin(dfsub.anon_subject_epis_id), cols]
reading = reading.drop_duplicates(subset=['anon_subject_epis_id', 'anon_test_kit_id'])
reading.to_csv(data_path_clean / 'reading_included.csv', index=False)

assert reading.shape[0] == reading.anon_test_kit_id.nunique()

# Print outcomes in included data
print('OUTCOMES IN INCLUDED DATA')
print(dfsub.outcome.value_counts())
#endregion


# ---- 5. A few additional checks -----
#region
print('\nFEW EXTRA CHECKS...')

# Sanity check: the same episodes have cancer in new and old outcome variable?
dfsub.loc[dfsub.outcome == 'Colorectal cancer', 'episode_result'].value_counts()
dfsub.loc[dfsub.episode_result == 'Cancer Detected', 'outcome'].value_counts()
assert all((dfsub.outcome.fillna('NULL') == 'Colorectal cancer') == (dfsub.episode_result == 'Cancer Detected'))

# Sanity check: when analyser reading is present, kit result is normal or abnormal
test = dfsub.loc[~dfsub.analyser_reading_used.isna()].kit_result.isin(['NORMAL', 'ABNORMAL'])
assert test.all()

# Sanity check: when reading >= 120, kit_result is always abnormal
test = dfsub.loc[dfsub.analyser_reading_used >= 120, 'kit_result'] == 'ABNORMAL'
assert test.all()

# Sanity check: when reading is < 120, kit_result is normal
test = dfsub.loc[(dfsub.analyser_reading_used < 120) & (~dfsub.analyser_reading_used.isna()), 'kit_result'] == 'NORMAL'
assert test.all()

# Estimate again how many LNPCPs may be in high risk findings in included data.
#  Since 2023, 7.0% of episodes with high-risk findings have pre-existing episode result category as LNPCP
#  Over all time, 22.2% have at least one premalignant polyp of size 20mm
#  Therefore, the new category is called "High-risk findings or LNPCP"
h = dfsub.loc[(dfsub.outcome == 'High-risk findings or LNPCP')]
h2023 = h.loc[(h.test_kit_logged_year >= 2023)]
test = (h2023.episode_result == 'LNPCP')
test.mean() * 100
test.sum()

test = h.anon_subject_epis_id.isin(polyp_pm_20.anon_subject_epis_id)
test.mean() * 100

# Estimate again how many LNPCPs may be in the premalignant polyps category in included data.
#  Since 2023, 0.41% of episodes with the premalignant polyp(s) category have pre-existing episode result as LNPCP
#  Over all time, 0.12% have at least one premalignant polyp of size 20mm
# It therefore seems OK to call this "Pre-malignant polyp(s)" and not specifically mention LNPCP
h = dfsub.loc[(dfsub.outcome == 'Premalignant polyp(s)')]
h2023 = h.loc[(h.test_kit_logged_year >= 2023)]
test = (h2023.episode_result == 'LNPCP')
test.mean() * 100
test.sum()

test = h.anon_subject_epis_id.isin(polyp_pm_20.anon_subject_epis_id)
test.mean() * 100

# Since 2023, the pre-existing LNPCP category subdivides 
# as 90.9% in high-risk and 7.0% in premalignant polyps, and a few other categories
t = dfsub.loc[(dfsub.episode_result == 'LNPCP') & (dfsub.test_kit_logged_year >= 2023)]
t.outcome.value_counts(normalize=True) * 100
t.outcome.value_counts(normalize=False)

#endregion
