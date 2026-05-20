"""Individuals who returned a kit late in their Routine episode 
have the original episode marked as "FOBt inadequate participation",
and a new late-responder episode is created for the same kit ID that stores the investigation outcome.
About half of these late-responder episodes are not in the original data extract
and this script checks this."""
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Data
data_path = Path("C:/Users/andres.Tamm/Desktop/BCSS-16540-LIVE-1")
df = pd.read_csv(data_path / 'BCSS-16540-T1-LIVE-1.csv')
df.columns = df.columns.str.lower()

# Is there one kit ID per episode? Yes. So both episodes in each late-responder pair are not present
u = df.groupby('anon_test_kit_id').anon_subject_epis_id.nunique()
test = (u == 1).all().item()
assert test

# Compute time to kit return
df['start'] = pd.to_datetime(df['episode_start_date'], format='%m/%Y', exact=True)
df['logged'] = pd.to_datetime(df['test_kit_logged_date'], format='%m/%Y', exact=True)
df['delta'] = (df.logged - df.start).dt.days


# Routine episodes with inadequate participation, valid analyser reading, and >= 181 day return
# these should have a matching late-responder episode, but that is not present
mask = ((df.episode_result == "FOBt inadequate participation") &
        (df.kit_result.isin(['NORMAL', 'ABNORMAL'])) &
        (df.episode_subtype == 'Routine')
        )
print(mask.sum(), mask.mean())
dfsub = df.loc[mask]
print(dfsub.shape)

dfsub.delta.min()
mask = dfsub.delta >= 181
print(mask.sum(), mask.mean())
dfsub = dfsub.loc[mask]
print(dfsub.shape)
print(dfsub.anon_subject_epis_id.nunique())

# A subset of these with FIT positive result
dfsubsub = dfsub.loc[dfsub.analyser_reading_used >= 120]
dfsubsub.shape # 3077

# For these, can we find polyp and cancer findings? The polyp and cancer tables ONLY have episode IDs
df_polyp = pd.read_csv(data_path / 'BCSS-16540-T2-LIVE-1.csv')
df_polyp.columns = df_polyp.columns.str.lower()

df_cancer_endo = pd.read_csv(data_path / 'BCSS-16540-T3-LIVE-1.csv')
df_cancer_endo.columns = df_cancer_endo.columns.str.lower()

df_cancer_endo.columns
df_polyp.columns

# Would polyp and cancer tables still have findings under the original Routine episode ID? Nope
test = dfsubsub.anon_subject_epis_id.isin(df_polyp.anon_subject_epis_id).sum()
print(test)
test = dfsubsub.anon_subject_epis_id.isin(df_cancer_endo.anon_subject_epis_id).sum()
print(test)

# Conversely, episodes known to be late responders
dflate = df.loc[df.episode_subtype == 'Late Responder']
print(dflate.shape[0])
print(dflate.anon_subject_epis_id.nunique())
dflate[['anon_subject_epis_id', 'episode_result']].drop_duplicates().episode_result.value_counts()

dflatesub = dflate.loc[dflate.episode_result == 'FOBt inadequate participation']
dflatesub.anon_subject_epis_id.nunique()
dflatesub.shape[0]

# All epi with inadequate participation as the result
df_inadequate = df.loc[df.episode_result == "FOBt inadequate participation"]

mask = df_inadequate.delta >= 181
mask.sum()
df_late = df_inadequate.loc[mask]
df_late.anon_subject_epis_id.nunique()

df_nolate = df_inadequate.loc[~mask]
df_nolate.anon_subject_epis_id.nunique()

df_nolate.delta.describe(percentiles=[0.5, 0.9, 0.99])
df_nolate.kit_result.isin(['NORMAL', 'ABNORMAL']).mean()
df_late.kit_result.isin(['NORMAL', 'ABNORMAL']).mean()

df_late[['anon_subject_epis_id', 'episode_subtype']].drop_duplicates().episode_subtype.value_counts()
df_nolate[['anon_subject_epis_id', 'episode_subtype']].drop_duplicates().episode_subtype.value_counts()

# Compare df_late result and no result
mask = df_late.kit_result.isin(['NORMAL', 'ABNORMAL'])
mask.sum()
df_late_result = df_late.loc[mask]
df_late_noresult = df_late.loc[~df_late.anon_subject_epis_id.isin(df_late_result.anon_subject_epis_id)] 


df_late_result.anon_subject_epis_id.nunique()
df_late_noresult.anon_subject_epis_id.nunique()

df_late_noresult.columns
df_late_noresult.analyser_error_code.value_counts()
df_late_noresult.episode_subtype.value_counts()


# // Some anon IDs to check for Claire - start//
dfsub = df.loc[df.episode_result == "FOBt inadequate participation"]
dfsub.shape
dfsub = dfsub.loc[dfsub.episode_subtype=='Routine']
dfsub.shape
dfsub = dfsub.loc[dfsub.analyser_reading_used > 120]
print(dfsub.analyser_error_code.value_counts())
dfsub

df.kit_result.value_counts()
dfsub.kit_result.value_counts()

dfsub.iloc[0]
dfsub.iloc[1]

dfsub.analyser_reading_used

dfsub.logged.describe()

dfsub = dfsub.loc[(dfsub.analyser_reading_used > 120) & (dfsub.analyser_reading_used < 250)]
print(dfsub.analyser_error_code.value_counts())
dfsub
s = dfsub[['anon_subject_epis_id', 'anon_test_kit_id']]
s.iloc[3]
dfsub.iloc[3]
# // Some anon IDs to check for Claire - end //
