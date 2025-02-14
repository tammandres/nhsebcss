"""Prepare data for analysis"""
import numpy as np
import pandas as pd
from pathlib import Path


# Paths
data_path = Path("C:/Users/p0Po/Desktop/BCSS-16540-LIVE-1")  # Where raw data is
data_path_clean = Path("C:/Users/p0Po/Desktop/bcss_clean_data")  # Where to save cleaned data
out_path = Path("Z:/andres/nhsebcss/results")  # Where to save summary stats

if not data_path_clean.exists():
    data_path_clean.mkdir(exist_ok=True)
if not out_path.exists():
    out_path.mkdir(exist_ok=True)


# ---- 1. Read data, reformat dates, transform to one row per episode including max FIT result per episode ----
#region

# Read data
df = pd.read_csv(data_path / 'BCSS-16540-T1-LIVE-1.csv')
print(df.shape)
print(df.columns)

# Check missingness
nmis = df.isna().sum()
print(nmis.sort_values())
print(nmis[nmis > 0])

# Dates to datetime
date_cols = ['EPISODE_START_DATE', 'EPISODE_END_DATE', 'TEST_KIT_LOGGED_DATE', 'TEST_KIT_RESULT_DATE']
assert all([c in df.columns for c in date_cols])
for c in date_cols:
    print(c)
    df[c] = pd.to_datetime(df[c], format='%m/%Y', exact=True)

# Get data with one row per episode
n_epi = df.ANON_SUBJECT_EPIS_ID.nunique()
dfsub = df[['ANON_SCREENING_SUBJECT_ID', 'ANON_SUBJECT_EPIS_ID', 
            'EPISODE_START_DATE', 'EPISODE_END_DATE', 'PREVALENT_INCIDENT_STATUS', 
            'EPISODE_STATUS', 'EPISODE_SUBTYPE', 'SUBJECT_AGE_AT_EPISODE_START',
            'SUBJECT_GENDER', 'IMD_QUINTILE', 'IMD_DECILE', 'IMD_SCORE', 'IMD_RANK',
            'EPISODE_RESULT']]
dfsub = dfsub.drop_duplicates()
assert dfsub.shape[0] == n_epi

# Get maximum FIT result per episode
fit = df[['ANON_SUBJECT_EPIS_ID', 'TEST_KIT_LOGGED_DATE', 'TEST_KIT_RESULT_DATE', 'KIT_RESULT', 'ANALYSER_READING_USED', 'ANALYSER_ERROR_CODE']]
fit = fit.dropna(subset=['ANALYSER_READING_USED'])
fit = fit.drop_duplicates()
idmax = fit.groupby('ANON_SUBJECT_EPIS_ID').ANALYSER_READING_USED.idxmax()
fit = fit.loc[idmax]
assert fit.shape[0] == fit.ANON_SUBJECT_EPIS_ID.nunique()
fit.isna().sum()

# Get most recent kit result for episodes without analyser reading
kit = df[['ANON_SUBJECT_EPIS_ID', 'TEST_KIT_LOGGED_DATE', 'TEST_KIT_RESULT_DATE', 'KIT_RESULT', 'ANALYSER_READING_USED', 'ANALYSER_ERROR_CODE']]
kit = kit.loc[~kit.ANON_SUBJECT_EPIS_ID.isin(fit.ANON_SUBJECT_EPIS_ID)]
kit = kit.dropna(subset=['KIT_RESULT']).drop_duplicates()
idmax = kit.groupby('ANON_SUBJECT_EPIS_ID').TEST_KIT_LOGGED_DATE.idxmax()
kit = kit.loc[idmax]
assert kit.shape[0] == kit.ANON_SUBJECT_EPIS_ID.nunique()
assert kit.ANALYSER_READING_USED.isna().all()

# Add maximum FIT result, and kit results without FIT result, to the data
fit_and_kit = pd.concat(objs=[fit, kit], axis=0)
dfsub = dfsub.merge(fit_and_kit, how='left', on='ANON_SUBJECT_EPIS_ID')
assert dfsub.shape[0] == n_epi
dfsub.isna().sum()

# Get year for each date
dfsub['EPISODE_START_YEAR'] = dfsub.EPISODE_START_DATE.dt.year
dfsub['EPISODE_END_YEAR'] = dfsub.EPISODE_END_DATE.dt.year
dfsub['TEST_KIT_LOGGED_YEAR'] = dfsub.TEST_KIT_LOGGED_DATE.dt.year
dfsub['TEST_KIT_RESULT_YEAR'] = dfsub.TEST_KIT_RESULT_DATE.dt.year
assert dfsub.shape[0] == n_epi

# Get quarter for each date
dfsub['EPISODE_START_QUARTER'] = dfsub.EPISODE_START_DATE.dt.quarter
dfsub['EPISODE_END_QUARTER'] = dfsub.EPISODE_END_DATE.dt.quarter
dfsub['TEST_KIT_LOGGED_QUARTER'] = dfsub.TEST_KIT_LOGGED_DATE.dt.quarter
dfsub['TEST_KIT_RESULT_QUARTER'] = dfsub.TEST_KIT_RESULT_DATE.dt.quarter
assert dfsub['EPISODE_START_QUARTER'].nunique() == 4

# Create age groups
age = dfsub.SUBJECT_AGE_AT_EPISODE_START
print(age.isna().sum())
age_max = age.max() + 1
age_group = pd.cut(age, bins=[50, 60, 70, 80, 90, age_max], right=False)
age_group.value_counts(sort=False)

dfsub['AGE_GROUP'] = age_group.astype(str)
dfsub.loc[age < 50, 'AGE_GROUP'] = '<50'
dfsub.AGE_GROUP = dfsub.AGE_GROUP.replace({'[50, 60)': '50-59',
                                           '[60, 70)': '60-69',
                                           '[70, 80)': '70-79',
                                           '[80, 90)': '80-89'})
dfsub.AGE_GROUP = dfsub.AGE_GROUP.replace({'[90, ' + str(age_max) + ')': '90+'})
print(dfsub.AGE_GROUP.unique())
print(dfsub.AGE_GROUP.isna().sum())

# Reformat IMD_DECILE so it matches IMD_QUINTILE
mask = dfsub.IMD_DECILE.isna()
dfsub['IMD_DECILE'] = dfsub.IMD_DECILE.astype(str)
dfsub.IMD_DECILE = dfsub.IMD_DECILE.str.replace('.0', '').str.zfill(2)
dfsub.IMD_DECILE = dfsub.IMD_DECILE.replace({'01': '01 - Most deprived', '10': '10 - Least deprived'})
dfsub.loc[mask, 'IMD_DECILE'] = np.nan
dfsub.IMD_DECILE.unique()

#endregion


# ---- 2. For each episode, get number of premalignant polyps and identify high-risk findings ----
#region

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

# Get premalignant polyps (Rutter et al 2020, https://doi.org/10.1136/gutjnl-2019-319858)
#  Adenomatous polyp, 
#  or serrated polyp excluding 1-5mm rectal hyperplastic polyps
# Note: sub type of serrated polyps makes sense with respect to Rutter: hyperplastic, sessile, traditional, mixed
print(df_polyp.POLYP_TYPE.unique())
mask = df_polyp.POLYP_TYPE.fillna('').str.lower().str.contains('adenom', regex=True)
adenoma = df_polyp.loc[mask]
print(adenoma.POLYP_TYPE.unique())

mask = df_polyp.POLYP_TYPE.fillna('').str.lower().str.contains('serrated', regex=True)
serrated = df_polyp.loc[mask]
print(serrated.POLYP_TYPE.unique())

df_polyp.POLYP_LOCATION.unique()
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

df_polyp['PREMALIGNANT_20'] = 0
polyp_pm_20 = polyp_pm.loc[polyp_pm.POLYP_HISTOLOGY_SIZE >= 20]
df_polyp.loc[df_polyp.ANON_POLYP_ID.isin(polyp_pm_20.ANON_POLYP_ID), 'PREMALIGNANT_20'] = 1
df_polyp.loc[df_polyp.PREMALIGNANT_20 == 1, 'CARCINOMA'].value_counts()

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

# Indicator for cancer detected at endoscopy
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

dfsub['PREMALIGNANT_POLYP_20'] = 0
dfsub.loc[dfsub.ANON_SUBJECT_EPIS_ID.isin(polyp_pm_20.ANON_SUBJECT_EPIS_ID), 'PREMALIGNANT_POLYP_20'] = 1
dfsub['PREMALIGNANT_POLYP_20_NO_CANCER'] = dfsub.PREMALIGNANT_POLYP_20.copy()  ## Excluding non endoscopic cancers from indicator
dfsub.loc[dfsub.EPISODE_RESULT == 'Cancer Detected', 'PREMALIGNANT_POLYP_20_NO_CANCER'] = 0

# Indicator for high risk findings
# Should these be called HIGH_RISK_FINDINGS_OR_LNPCP as I don't have data about stalk?
mask = dfsub.ANON_SUBJECT_EPIS_ID.isin(high_risk)
dfsub['HIGH_RISK_FINDINGS'] = 0
dfsub.loc[mask, 'HIGH_RISK_FINDINGS'] = 1

dfsub['HIGH_RISK_FINDINGS_NO_CANCER'] = dfsub.HIGH_RISK_FINDINGS.copy()  ## Excluding cancers from indicator
dfsub.loc[dfsub.EPISODE_RESULT == 'Cancer Detected', 'HIGH_RISK_FINDINGS_NO_CANCER'] = 0
dfsub.loc[dfsub.HIGH_RISK_FINDINGS_NO_CANCER == 1].EPISODE_RESULT.value_counts()

dfsub.loc[dfsub.HIGH_RISK_FINDINGS == 1].EPISODE_RESULT.value_counts()
dfsub.loc[dfsub.HIGH_RISK_FINDINGS_NO_CANCER == 1].EPISODE_RESULT.value_counts()

assert dfsub.shape[0] == n_epi
dfsub.columns

# Create an outcome variable that is consistently defined over time

## First assign value 'Other'
dfsub['OUTCOME'] = 'Other'

## Then add some existing outcome categories, but rename them for simplicity
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

## Then assign abnormal category
dfsub.loc[dfsub.EPISODE_RESULT == 'Abnormal', 'OUTCOME'] = 'Other abnormality at whole colon investigation'  

## Then assign premalignant polyps (over-writes some abnormal)
dfsub.loc[dfsub.PREMALIGNANT_POLYP == 1, 'OUTCOME'] = 'Premalignant polyp(s)'  
#dfsub.loc[dfsub.PREMALIGNANT_POLYP == 1, 'OUTCOME'] = 'Premalignant polyp(s) < 20mm'  
#dfsub.loc[dfsub.PREMALIGNANT_POLYP_20 == 1, 'OUTCOME'] = 'Premalignant polyp ≥ 20mm'  
#dfsub.loc[dfsub.ANON_SUBJECT_EPIS_ID.isin(polyp_advanced.ANON_SUBJECT_EPIS_ID), 'OUTCOME'] = 'Advanced polyp' 

## Then assign high-risk (over-writes some premalignant)
dfsub.loc[dfsub.HIGH_RISK_FINDINGS == 1, 'OUTCOME'] = 'High-risk findings or LNPCP' 
#dfsub.loc[dfsub.HIGH_RISK_FINDINGS == 1, 'OUTCOME'] = 'High-risk findings < 20mm' 
#dfsub.loc[(dfsub.HIGH_RISK_FINDINGS == 1) & (dfsub.PREMALIGNANT_POLYP_20 == 1), 'OUTCOME'] = 'High-risk findings ≥ 20mm'  

## Then assign colorectal cancer (may over-write some high-risk findings and premalignant findings)
dfsub.loc[dfsub.EPISODE_RESULT == 'Cancer Detected', 'OUTCOME'] = 'Colorectal cancer'
dfsub.loc[(dfsub.OUTCOME == 'Other') & (dfsub.ANON_SUBJECT_EPIS_ID.isin(df_polyp.ANON_SUBJECT_EPIS_ID)), 'OUTCOME'] = 'Other polyp'

## Check outcome counts
##  How many in the high-risk findings and premalignant polyp(s) category are potentially LNPCP?
##  If subdividing these categories based on 20mm polyp size (as in commented in code above), then ...
##    3021 episodes w premalignant polyps have >= 20 mm, and 90724 have < 20 mm: at most 3.2% of premalignant polyps are LNPCP
##      3021 / (3021 + 90724) * 100 = 3.2
##    and 16906 high-risk findings have polyp >= 20 mm, 55443 have < 20mm: at most 23.4% of high-risk findings are LNPCP
##      16906 / (16906 + 55443) * 100 = 23.4
dfsub.OUTCOME.value_counts()
#assert (dfsub.loc[dfsub.OUTCOME == 'Premalignant polyp ≥ 20mm', 'NUM_PREMALIGNANT_POLYP'] == 1).all()
#assert (dfsub.loc[dfsub.OUTCOME == 'Premalignant polyp(s) < 20mm', 'NUM_PREMALIGNANT_POLYP'] < 5).all()
#assert (dfsub.loc[dfsub.OUTCOME == 'Advanced polyp', 'NUM_PREMALIGNANT_POLYP'] == 1).all()

# Explore the "Other" category: EPISODE_RESULT is adenoma, high-risk findings or LNPCP (but there is no record in polyp table)
# It is predominantly low-risk or intermediate-risk adenoma.
dfsub.loc[dfsub.OUTCOME == 'Other', 'EPISODE_RESULT'].value_counts()

# Rename the "Other" category
dfsub.loc[dfsub.OUTCOME == 'Other', 'OUTCOME'] = 'Polyps without endoscopy record'
dfsub.OUTCOME.value_counts()

# In few cases, when outcome marked as FIT negative, reading is positive
a = dfsub.loc[dfsub.OUTCOME == 'FIT negative', 'ANALYSER_READING_USED']
test = a >= 120
test.sum()
test.mean()
a[test]

# In few cases, when outcome is not marked as FIT positive and participation is adequate, FIT is not positive
a = dfsub.loc[~dfsub.OUTCOME.isin(['FIT negative', 'FIT inadequate participation'])]
a = a.loc[a.ANALYSER_READING_USED < 120]
a.OUTCOME.value_counts()
a.EPISODE_RESULT.value_counts()
a = a.ANALYSER_READING_USED
test = a < 120
test.sum()
test.mean()
a[test]

#endregion


# ---- 3. Apply inclusion criteria and save cleaned data ----
#region

# Remove episodes where 
#  end date before 2019, 
#  start date before 2018, 
#  age < 50,
#  open or pending,
#  there is result but no analyser reading
def remove_episodes(df, df_crc, epi_rm):
    n_epi_rm = len(epi_rm)  # Num episodes removed
    n_crc_rm = df_crc.loc[df_crc.ANON_SUBJECT_EPIS_ID.isin(epi_rm)].shape[0]  # Num cancers removed
    p_epi_rm = n_epi_rm / n_epi * 100  # Percent episodes removed
    p_crc_rm = n_crc_rm / n_crc * 100  # Percent cancers removed
    print(df.shape)
    dfsub = df.loc[~df.ANON_SUBJECT_EPIS_ID.isin(epi_rm)]  # Remove episodes
    print(dfsub.shape)
    return dfsub, n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm

stat = {}
n_epi = dfsub.shape[0]
df_crc = dfsub.loc[dfsub.EPISODE_RESULT == 'Cancer Detected']
n_crc = df_crc.shape[0]
stat['Episodes'] = [n_epi, 100, n_crc, 100]

epi_rm = dfsub.loc[(dfsub.EPISODE_END_YEAR < 2019) | (dfsub.EPISODE_START_YEAR < 2018)].ANON_SUBJECT_EPIS_ID
dfsub, n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm = remove_episodes(dfsub, df_crc, epi_rm)
stat['Episodes with start date before 2018 or end date before 2019'] = [n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm]

#epi_rm = dfsub.loc[dfsub.EPISODE_START_YEAR < 2018].ANON_SUBJECT_EPIS_ID
#dfsub, n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm = remove_episodes(dfsub, df_crc, epi_rm)
#stat['Episodes with start date before 2018'] = [n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm]

epi_rm = dfsub.loc[dfsub.SUBJECT_AGE_AT_EPISODE_START < 50].ANON_SUBJECT_EPIS_ID
dfsub, n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm = remove_episodes(dfsub, df_crc, epi_rm)
stat['Episodes with age < 50'] = [n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm]

epi_rm = dfsub.loc[dfsub.EPISODE_STATUS.isin(['Open', 'Pending'])].ANON_SUBJECT_EPIS_ID
dfsub, n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm = remove_episodes(dfsub, df_crc, epi_rm)
stat['Episodes open or pending'] = [n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm]
print(dfsub.EPISODE_STATUS.unique())

dfsub_with_result = dfsub.loc[~dfsub.EPISODE_RESULT.isin(['FOBt inadequate participation'])]
epi_rm = dfsub_with_result.loc[dfsub_with_result.ANALYSER_READING_USED.isna()].ANON_SUBJECT_EPIS_ID

## At least 98% of episodes with adequate FIT participation (meaning: participation NOT inadrequate) 
## but no analyser reading have normal FOBt outcome as episode result
## But all have kit result as SPOILT
dfsub_with_result.loc[dfsub_with_result.ANON_SUBJECT_EPIS_ID.isin(epi_rm)].EPISODE_RESULT.value_counts(normalize=True)
dfsub_with_result.loc[dfsub_with_result.ANON_SUBJECT_EPIS_ID.isin(epi_rm)].EPISODE_RESULT.value_counts(normalize=False)
dfsub_with_result.loc[dfsub_with_result.ANON_SUBJECT_EPIS_ID.isin(epi_rm)].KIT_RESULT.fillna('NULL').value_counts(normalize=True)
df.loc[df.ANON_SUBJECT_EPIS_ID.isin(epi_rm)].KIT_RESULT.fillna('NULL').value_counts(normalize=True)

dfsub, n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm = remove_episodes(dfsub, df_crc, epi_rm)
stat['Episodes with adequate FIT participation but no analyser reading'] = [n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm]

# Episodes marked as FIT negative but positive reading; 
# or that are not marked as FIT negative but have a negative reading.
epi_rm = dfsub.loc[(dfsub.OUTCOME == 'FIT negative') & (dfsub.ANALYSER_READING_USED >= 120)].ANON_SUBJECT_EPIS_ID
dfsub.loc[dfsub.ANON_SUBJECT_EPIS_ID.isin(epi_rm), 'ANALYSER_READING_USED'].describe()  #  Reading is >120
dfsub.loc[dfsub.ANON_SUBJECT_EPIS_ID.isin(epi_rm), 'KIT_RESULT'].value_counts() # KIT RESULT is abnormal always
dfsub.loc[dfsub.ANON_SUBJECT_EPIS_ID.isin(epi_rm), 'EPISODE_RESULT'].value_counts()  # But episode result is normal

out = ['FIT negative', 'FIT inadequate participation']
assert all([o in dfsub.OUTCOME.tolist() for o in out])
rm = dfsub.loc[~dfsub.OUTCOME.isin(out) & (dfsub.ANALYSER_READING_USED < 120)]
epi_rm2 = rm.ANON_SUBJECT_EPIS_ID
dfsub.loc[dfsub.ANON_SUBJECT_EPIS_ID.isin(epi_rm2), 'ANALYSER_READING_USED'].describe()  # Reading is below 50
dfsub.loc[dfsub.ANON_SUBJECT_EPIS_ID.isin(epi_rm2), 'KIT_RESULT'].value_counts()  #  KIT RESULT is normal
dfsub.loc[dfsub.ANON_SUBJECT_EPIS_ID.isin(epi_rm2), 'EPISODE_RESULT'].value_counts()  #  But episode result is for example abnormal and other categories

epi_rm = pd.concat([epi_rm, epi_rm2], axis=0).drop_duplicates()

dfsub, n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm = remove_episodes(dfsub, df_crc, epi_rm)
stat['Episodes with result as FIT negative but reading >= 120, or result not negative and reading < 120'] = [n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm]

# Remove episodes where outcome is adenoma, high-risk findings or LNPCP but no endoscopy result
#epi_rm = dfsub.loc[(dfsub.OUTCOME == 'Polyps without endoscopy record')].ANON_SUBJECT_EPIS_ID
#dfsub, n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm = remove_episodes(dfsub, df_crc, epi_rm)
#stat['Episodes reported as adenoma, high-risk findings or LNPCP, but there is no endoscopy result'] = [n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm]

#epi_rm = dfsub.loc[dfsub.EPISODE_RESULT.isin(['FOBt inadequate participation'])].ANON_SUBJECT_EPIS_ID
#dfsub, n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm = remove_episodes(dfsub, df_crc, epi_rm)
#stat['Episodes with inadequate participation'] = [n_epi_rm, p_epi_rm, n_crc_rm, p_crc_rm]

# Reformat statistics and save
stat = pd.DataFrame.from_dict(stat, orient='index')
stat = stat.reset_index()
stat.columns = ['Characteristic', 'Number of episodes', 'Percent episodes', 'Number of cancers', 'Percent cancers']

n_removed = stat.iloc[1:]['Number of episodes'].sum().item()
ncrc_removed = stat.iloc[1:]['Number of cancers'].sum().item()
p_removed = n_removed / n_epi * 100
pcrc_removed = ncrc_removed / n_crc * 100
row = pd.DataFrame([['All removed episodes', n_removed, p_removed, ncrc_removed, pcrc_removed]], 
                    columns=stat.columns)
stat = pd.concat(objs=[stat, row], axis=0).reset_index(drop=True)

stat.to_csv(out_path / 'removed_episodes.csv', index=False)


# Save prepared data
dfsub.to_csv(data_path_clean / 'episodes_clean.csv', index=False)


# Sanity check: the same episodes have cancer in new and old outcome variable
dfsub.loc[dfsub.OUTCOME == 'Colorectal cancer', 'EPISODE_RESULT'].value_counts()
dfsub.loc[dfsub.EPISODE_RESULT == 'Cancer Detected', 'OUTCOME'].value_counts()
assert all((dfsub.OUTCOME.fillna('NULL') == 'Colorectal cancer') == (dfsub.EPISODE_RESULT == 'Cancer Detected'))

# Sanity check: when analyser reading is present, predominantly kit result is normal or abnormal (>99.98% of times)
dfsub.loc[~dfsub.ANALYSER_READING_USED.isna()].KIT_RESULT.value_counts(normalize=True) * 100

# Sanity check: when reading >= 120, KIT_RESULT is always abnormal
test = dfsub.loc[dfsub.ANALYSER_READING_USED >= 120, 'KIT_RESULT'] == 'ABNORMAL'
assert test.all()

# Sanity check: when reading is < 120, KIT_RESULT is predominantly normal, 
# but in about 1100 cases INVALIDATED, and SPOILT value is also present
dfsub.loc[(dfsub.ANALYSER_READING_USED < 120) & (~dfsub.ANALYSER_READING_USED.isna()), 'KIT_RESULT'].value_counts()

# When reading is present and kit result is INVALIDATED,
# episode result is usually marked as normal (87%) - so perhaps it is OK to  keep these.
dfsub.loc[(~dfsub.ANALYSER_READING_USED.isna()) & (dfsub.KIT_RESULT == 'INVALIDATED'), 'EPISODE_RESULT'].value_counts()
dfsub.loc[(~dfsub.ANALYSER_READING_USED.isna()) & (dfsub.KIT_RESULT == 'INVALIDATED'), 'EPISODE_RESULT'].value_counts(normalize=True)

# Estimate again how many LNPCPs may be in high risk findings in included data.
#  Since 2023, 6.7% of episodes with high-risk findings have pre-existing episode result category as LNPCP
#  Over all time, 20.7% have at least one premalignant polyp of size 20mm
#  Therefore, the new category is called "High-risk findings or LNPCP"
h = dfsub.loc[(dfsub.OUTCOME == 'High-risk findings or LNPCP')]
h2023 = h.loc[(h.TEST_KIT_LOGGED_YEAR >= 2023)]
test = (h2023.EPISODE_RESULT == 'LNPCP')
test.mean() * 100
test.sum()

test = h.ANON_SUBJECT_EPIS_ID.isin(polyp_pm_20.ANON_SUBJECT_EPIS_ID)
test.mean() * 100

# Estimate again how many LNPCPs may be in the premalignant polyps category in included data.
#  Since 2023, 1.1% of episodes with the premalignant polyp(s) category have pre-existing episode result as LNPCP
#  Over all time, 2.9% have at least one premalignant polyp of size 20mm
# It therefore seems OK to call this "Pre-malignant polyp(s)" and not specifically mention LNPCP
h = dfsub.loc[(dfsub.OUTCOME == 'Premalignant polyp(s)')]
h2023 = h.loc[(h.TEST_KIT_LOGGED_YEAR >= 2023)]
test = (h2023.EPISODE_RESULT == 'LNPCP')
test.mean() * 100
test.sum()

test = h.ANON_SUBJECT_EPIS_ID.isin(polyp_pm_20.ANON_SUBJECT_EPIS_ID)
test.mean() * 100

# Although note that since 2023, the pre-existing LNPCP category subdivides 
# as 78.8% in high-risk and 19.1% in premalignant polyps, 
# and smaller number under polyps without endoscopy record and other polyp
t = dfsub.loc[(dfsub.EPISODE_RESULT == 'LNPCP') & (dfsub.TEST_KIT_LOGGED_YEAR >= 2023)]
t.OUTCOME.value_counts(normalize=True) * 100
t.OUTCOME.value_counts(normalize=False)

#endregion
