"""Outcomes by demographics
Assumes dataprep.py has been run"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from statsmodels.stats.proportion import proportion_confint
from nhsebcss.utils import grouped_count
import statsmodels.api as stat
import re


# Paths
data_path_clean = Path("C:/Users/andres.Tamm/Desktop/bcss_clean_data")
out_path = Path("Z:/andres/nhsebcss/results/primary")
out_path.mkdir(exist_ok=True, parents=True)


# Method for proportion confidence intervals
ci_method = 'wilson'


# ---- Read and prepare data ----
#region 
print('\nREADING DATA...')

# Read cleaned data
df = pd.read_csv(data_path_clean / 'episodes_clean.csv',
                 usecols=['anon_subject_epis_id', 'anon_screening_subject_id', 'episode_subtype',
                          'episode_start_date', 'episode_end_date', 'test_kit_logged_date', 'test_kit_result_date',
                          'subject_gender', 'prevalent_incident_status', 'age_group_screen2', 'imd_quintile',
                          'outcome', 'advanced_polyp', 'nonadvanced_polyp', 'premalignant_polyp',
                          'analyser_reading_used', 'outcome_simple', 'episode_result',
                          'subject_age_at_episode_start'])
n_epi = df.anon_subject_epis_id.nunique()
assert df.shape[0] == n_epi

# Dates to datetime (note that date format is different as dates were already reformatted in clean data)
date_cols = ['episode_start_date', 'episode_end_date', 'test_kit_logged_date', 'test_kit_result_date']
assert all([c in df.columns for c in date_cols])
for c in date_cols:
    print(c)
    df[c] = pd.to_datetime(df[c], format='%Y-%m-%d', exact=True)

# Include missing IMD values? Not in plots - small count. Most missing values occur in FIT negative (25,433 of 26,142)
# Also, all IMD categories have at least 2,526,734 observations, which is much larger than the 26,142 missing values in total
test = df.imd_quintile.isna()
df['imd_mis'] = test
print('IMD missing: {} ({:.2f}%)'.format(test.sum(), test.mean()*100))
df.groupby('imd_mis')['outcome'].value_counts(normalize=False)
#df.imd_quintile = df.imd_quintile.fillna('NA')
df.groupby('imd_quintile', dropna=False).size()
#df.imd_quintile = df.imd_quintile.fillna('NA')

# Check for missing values in other group cols
group_cols = ['subject_gender', 'prevalent_incident_status', 'age_group_screen2']
assert not any(df[c].isna().any() for c in group_cols)

# Use the slightly simpler outcome classification
df.outcome = df.outcome_simple

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

# Add granular age group
def format_bin(s, age_max):
    low, high = map(int, re.findall(r'\d+', s))
    if high >= age_max:
        return f"{low}+"
    return f"{low}-{high - 1}"

age       = df.subject_age_at_episode_start
age_max   = age.max() + 1
age_group = pd.cut(age, bins=[50, 55, 60, 65, 70, 75, age_max], right=False)
age_group.value_counts(sort=False)
df['age_group_granular'] = age_group.astype(str)
unique_bins = df['age_group_granular'].astype(str).unique()
mapping = {b: format_bin(b, age_max) for b in unique_bins}
df['age_group_granular'] = df['age_group_granular'].astype(str).replace(mapping)

# Create two data subsets: episodes with investigation, and FIT positive episodes
dfsub = df.loc[df.outcome.isin(outcomes_with_investigation)]
dfsub_pos = df.loc[df.outcome.isin(outcomes_fit_pos)]

# Check prevalent/incident status by age group
dfsub.groupby('age_group_granular').prevalent_incident_status.value_counts(normalize=True, sort=False) * 100
dfsub.groupby('age_group_screen2').prevalent_incident_status.value_counts(normalize=True, sort=False) * 100
dfsub.groupby('age_group_granular').subject_gender.value_counts(normalize=True, sort=False) * 100
dfsub.groupby('age_group_screen2').subject_gender.value_counts(normalize=True, sort=False) * 100

# Compute number and % of each outcome
#  for colorectal investigation outcomes: % of investigated episodes
#  for no investigation after positive FIT: % of FIT-positive episodes
compute_counts_and_proportions = True
if compute_counts_and_proportions:

    # Variables to group outcomes by
    cols = ['age_group_screen2', 'age_group_granular', 'imd_quintile', 'subject_gender']
    cols_expanded = cols + ['prevalent_incident_status']

    # Compute: outcome ~ age, outcome ~ imd, outcome ~ sex
    res = pd.DataFrame()
    for i, group_col in enumerate(cols):
        print(group_col)
        g = grouped_count(dfsub, [group_col], outcome_col='outcome', ci_method=ci_method)
        g['column'] = group_col
        g = g.rename(columns={group_col: 'value'})
        res = pd.concat(objs=[res, g], axis=0)
    res = res[['column', 'value'] + [c for c in res.columns if c not in ['column', 'value']]]
    assert res['count'].min() >= 10
    res.to_csv(out_path / 'outcomes-investigated-by-demographics.csv', index=False)

    # Compute: outcome ~ age, outcome ~ imd, outcome ~ sex | history
    res_hist = pd.DataFrame()
    for i, group_col in enumerate(cols):
        print(group_col)
        g = grouped_count(dfsub, ['prevalent_incident_status', group_col],
                          outcome_col='outcome', ci_method=ci_method)
        g['column'] = group_col
        g = g.rename(columns={group_col: 'value'})
        res_hist = pd.concat(objs=[res_hist, g], axis=0)
    res_hist = res_hist[['column', 'value'] + [c for c in res_hist.columns if c not in ['column', 'value']]]
    assert res_hist['count'].min() >= 10
    res_hist.to_csv(out_path / 'outcomes-investigated-by-demographics-history.csv', index=False)

    # Compute: outcome ~ age, outcome ~ imd | history, sex
    res_hist_sex = pd.DataFrame()
    for group_col in ['age_group_screen2', 'age_group_granular', 'imd_quintile']:
        print(group_col)
        g = grouped_count(dfsub, ['subject_gender', 'prevalent_incident_status', group_col],
                          outcome_col='outcome', ci_method=ci_method)
        g['column'] = group_col
        g = g.rename(columns={group_col: 'value'})
        res_hist_sex = pd.concat(objs=[res_hist_sex, g], axis=0)
    res_hist_sex = res_hist_sex[['column', 'value'] + [c for c in res_hist_sex.columns if c not in ['column', 'value']]]
    res_hist_sex = res_hist_sex.loc[res_hist_sex['count'] >= 10]
    assert res_hist_sex['count'].min() >= 10
    res_hist_sex.to_csv(out_path / 'outcomes-investigated-by-demographics-history-sex.csv', index=False)

    # Compute non-investigation (ni) rates: ni ~ age, ni ~ imd, ni ~ sex
    res_pos = pd.DataFrame()
    for i, group_col in enumerate(cols):
        print(group_col)
        g = grouped_count(dfsub_pos, [group_col], outcome_col='outcome', ci_method=ci_method)
        g['column'] = group_col
        g = g.rename(columns={group_col: 'value'})
        res_pos = pd.concat(objs=[res_pos, g], axis=0)
    res_pos = res_pos[['column', 'value'] + [c for c in res_pos.columns if c not in ['column', 'value']]]
    res_pos = res_pos.loc[res_pos.outcome == 'FIT positive, no investigation']
    assert res_pos['count'].min() >= 10
    res_pos.to_csv(out_path / 'outcomes-pos-by-demographics.csv', index=False)

    # Compute: ni ~ age, ni ~ imd, ni ~ sex | history
    res_pos_hist = pd.DataFrame()
    for i, group_col in enumerate(cols):
        print(group_col)
        g = grouped_count(dfsub_pos, ['prevalent_incident_status', group_col],
                          outcome_col='outcome', ci_method=ci_method)
        g['column'] = group_col
        g = g.rename(columns={group_col: 'value'})
        res_pos_hist = pd.concat(objs=[res_pos_hist, g], axis=0)
    res_pos_hist = res_pos_hist[['column', 'value'] + [c for c in res_pos_hist.columns if c not in ['column', 'value']]]
    res_pos_hist = res_pos_hist.loc[res_pos_hist.outcome == 'FIT positive, no investigation']
    res_pos_hist = res_pos_hist.loc[res_pos_hist['count'] > 10]
    res_pos_hist.to_csv(out_path / 'outcomes-pos-by-demographics-history.csv', index=False)

    # Compute: ni ~ age, ni ~ imd | history, sex
    res_pos_hist_sex = pd.DataFrame()
    for group_col in ['age_group_granular', 'imd_quintile']:
        print(group_col)
        g = grouped_count(dfsub_pos, ['subject_gender', 'prevalent_incident_status', group_col],
                          outcome_col='outcome', ci_method=ci_method)
        g = g.loc[g.outcome == 'FIT positive, no investigation']
        g['column'] = group_col
        g = g.rename(columns={group_col: 'value'})
        res_pos_hist_sex = pd.concat(objs=[res_pos_hist_sex, g], axis=0)
    res_pos_hist_sex = res_pos_hist_sex[['column', 'value'] + 
                                        [c for c in res_pos_hist_sex.columns if c not in ['column', 'value']]]
    res_pos_hist_sex = res_pos_hist_sex.loc[res_pos_hist_sex['count'] >= 10]
    res_pos_hist_sex.to_csv(out_path / 'outcomes-pos-by-demographics-history-sex.csv', index=False)

    # Compute acp and nacp, defined non-exclusively (not as worst outcomes): acp, nacp ~ age, imd, sex
    polyp_indicators = {'advanced_polyp': 'Advanced colorectal polyp (non-exclusive)',
                        'nonadvanced_polyp': 'Non-advanced colorectal polyp (non-exclusive)'}
    res_polyp = pd.DataFrame()
    for i, group_col in enumerate(cols):
        print(group_col)
        for ind_col, ind_label in polyp_indicators.items():
            g = grouped_count(dfsub, [group_col], outcome_col=ind_col, ci_method=ci_method)
            g = g.loc[g[ind_col] == 1].drop(columns=[ind_col])  # keep the "has this polyp" rows
            g['outcome'] = ind_label
            g['column'] = group_col
            g = g.rename(columns={group_col: 'value'})
            res_polyp = pd.concat(objs=[res_polyp, g], axis=0)
    res_polyp = res_polyp[['column', 'value', 'outcome'] +
                          [c for c in res_polyp.columns if c not in ['column', 'value', 'outcome']]]
    assert res_polyp['count'].min() >= 10
    res_polyp.to_csv(out_path / 'outcomes-polyp-nonexclusive-by-demographics.csv', index=False)

    # acp, nacp ~ age, imd, sex | history
    res_polyp_hist = pd.DataFrame()
    for i, group_col in enumerate(cols):
        print(group_col)
        for ind_col, ind_label in polyp_indicators.items():
            g = grouped_count(dfsub, ['prevalent_incident_status', group_col],
                              outcome_col=ind_col, ci_method=ci_method)
            g = g.loc[g[ind_col] == 1].drop(columns=[ind_col])  # keep the "has this polyp" rows
            g['outcome'] = ind_label
            g['column'] = group_col
            g = g.rename(columns={group_col: 'value'})
            res_polyp_hist = pd.concat(objs=[res_polyp_hist, g], axis=0)
    res_polyp_hist = res_polyp_hist[['column', 'value', 'outcome'] +
                                    [c for c in res_polyp_hist.columns if c not in ['column', 'value', 'outcome']]]
    assert res_polyp_hist['count'].min() >= 10
    res_polyp_hist.to_csv(out_path / 'outcomes-polyp-nonexclusive-by-demographics-history.csv', index=False)

    # Compute total num episodes by demographics
    res_num = pd.DataFrame()
    for i, group_col in enumerate(cols_expanded):
        print(group_col)
        n_group = df.groupby(group_col).anon_subject_epis_id.nunique()
        p_group = n_group / n_group.sum() * 100
        g = pd.concat(objs=[n_group, p_group], axis=1)
        g.columns = ['count', 'perc']
        g['ntot'] = n_group.sum()
        for j, row in g.iterrows():
            ci_low, ci_high = proportion_confint(count=row['count'], nobs=row['ntot'], method=ci_method)
            g.loc[j, 'perc_low'] = ci_low * 100
            g.loc[j, 'perc_high'] = ci_high * 100
        g = g.reset_index()
        g['column'] = group_col
        g = g.rename(columns={group_col: 'value'})
        res_num = pd.concat(objs=[res_num, g], axis=0)
    res_num = res_num[['column', 'value'] + [c for c in res_num.columns if c not in ['column', 'value']]]
    assert res_num['count'].min() >= 10
    res_num.to_csv(out_path / 'num-episodes-by-demographics.csv', index=False)

    # Compute % of no FIT result by demographics
    res_all = pd.DataFrame()
    for i, group_col in enumerate(cols):
        print(group_col)
        g = grouped_count(df, [group_col], outcome_col='outcome', ci_method=ci_method)
        g['column'] = group_col
        g = g.rename(columns={group_col: 'value'})
        res_all = pd.concat(objs=[res_all, g], axis=0)
    res_all = res_all[['column', 'value'] + [c for c in res_all.columns if c not in ['column', 'value']]]
    res_all = res_all.loc[res_all.outcome == 'No FIT result']
    assert res_all['count'].min() >= 10
    res_all.to_csv(out_path / 'outcomes-all-by-demographics.csv', index=False)
else:
    res              = pd.read_csv(out_path / 'outcomes-investigated-by-demographics.csv')
    res_hist         = pd.read_csv(out_path / 'outcomes-investigated-by-demographics-history.csv')
    res_hist_sex     = pd.read_csv(out_path / 'outcomes-investigated-by-demographics-history-sex.csv')
    res_pos          = pd.read_csv(out_path / 'outcomes-pos-by-demographics.csv')
    res_pos_hist     = pd.read_csv(out_path / 'outcomes-pos-by-demographics-history.csv')
    res_pos_hist_sex = pd.read_csv(out_path / 'outcomes-pos-by-demographics-history-sex.csv')
    res_all          = pd.read_csv(out_path / 'outcomes-all-by-demographics.csv')
    res_num          = pd.read_csv(out_path / 'num-episodes-by-demographics.csv')
    res_polyp        = pd.read_csv(out_path / 'outcomes-polyp-nonexclusive-by-demographics.csv')
    res_polyp_hist   = pd.read_csv(out_path / 'outcomes-polyp-nonexclusive-by-demographics-history.csv')
    
    for data in [res, res_hist, res_hist_sex, res_pos, res_pos_hist, res_pos_hist_sex,
                 res_all, res_num, res_polyp, res_polyp_hist]:
        mincount = data['count'].min()
        print(mincount)
        assert mincount >= 10

# Outcome labels for plotting
outcome_labels = {
    'Colorectal cancer': 'Colorectal\ncancer', 
    'Advanced premalignant polyp': 'Advanced\ncolorectal polyp', 
    'Non-advanced premalignant polyp': 'Non-advanced\ncolorectal polyp', 
    'Other findings': 'Other findings',
    'No abnormality': 'No abnormality',
    'FIT positive, no investigation': 'FIT ≥120 µg/g, no investigation'
    }

#endregion


# ---- Outcomes by age, deprivation, sex (one variable at a time) ----
# [Updated by Claude Code to use multiple age group definitions; manually verified]
# Note: not computing outcomes by prev/inc status atm, as this is confounded by age.
#region
print('\nSUMMARISING OUTCOMES BY DEMOGRAPHICS ONE VARIABLE AT A TIME...')

# Two age-group definitions: the broad groups used by default (age_group_screen2)
# and the granular 5-year groups (age_group_granular). Each plot that uses age group
# is produced once per definition, distinguished by the filename suffix.
age_group_cols = {'age_group_screen2': '_age-broad',
                  'age_group_granular': '_age-granular'}


# .... Grouped line plot for outcomes with investigation
#region
outcome_plot = ['Colorectal cancer',
                'Advanced premalignant polyp',
                'Non-advanced premalignant polyp',
                'Other findings',
                'No abnormality'
                ]
pointsize = 18

cols_ystep = {'age_group_screen2': 10000,
              'age_group_granular': 5000,
              'imd_quintile': 2000,
              'subject_gender': 10000}

for age_group_col, age_suffix in age_group_cols.items():
    col_and_label = {age_group_col: 'Age group',
                     'imd_quintile': 'IMD quintile',
                     'subject_gender': 'Sex'}

    figsize = (12, 7)
    fig, ax = plt.subplots(2, 3, figsize=figsize, sharey=False, tight_layout=True)
    for i, (group_col, title) in enumerate(col_and_label.items()):
        print(group_col)
        g = res.loc[res.column == group_col].copy()
        g = g.rename(columns={'value': group_col})
        ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
        ticks['x'] = np.arange(ticks.shape[0])
        ticks['xticklabel'] = ticks[group_col]
        if group_col == 'age_group_granular':  # show only the lower bound (keep the open-ended top group)
            ticks['xticklabel'] = [lab if lab.endswith('+') else re.sub('-', '-\n', lab) # lab.split('-')[0]
                                   for lab in ticks['xticklabel']]
        if group_col == 'imd_quintile':
            ticks.xticklabel = ticks.xticklabel.replace({
                '01 - Most deprived': '01\nMost deprived',
                '05 - Least deprived': '05\nLeast deprived'
            })
        g = g.merge(ticks, how='left')
        for j, outcome in enumerate(outcome_plot):
            color = 'C' + str(j)
            s = g.loc[g.outcome == outcome]
            ax[0, i].plot(s['x'], s['count'], label=outcome_labels[outcome], color=color)
            ax[0, i].scatter(s['x'], s['count'], s=pointsize, color=color)
            ax[1, i].plot(s['x'], s['perc'], label=outcome_labels[outcome], color=color)
            ax[1, i].scatter(s['x'], s['perc'], s=pointsize, color=color)
            ax[1, i].fill_between(s['x'], s.perc_low, s.perc_high, facecolor=color, alpha=0.25)

        if not title.startswith('IMD'):
            title_panel = title.lower()
        else:
            title_panel = title
        xmax = ticks['x'].max()
        ax[0, i].set(title='Number of outcomes by ' + title_panel)
        ax[0, i].grid(which='major', alpha=0.5)
        ax[0, i].set_xticks(ticks['x'])
        ax[0, i].set_xticklabels(ticks['xticklabel'])
        ax[0, i].set_xlim(0 - 0.15 * xmax, xmax + xmax * 0.15)
        ax[0, i].set_ylabel('Number of episodes')
        ax[0, i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax[0, i].set_xlabel(title)
        ax[1, i].set(title='Percent of outcomes by ' + title_panel)
        ax[1, i].grid(which='major', alpha=0.5)
        ax[1, i].set_xticks(ticks['x'])
        ax[1, i].set_xticklabels(ticks['xticklabel'])

        ax[1, i].set_xlim(0 - 0.15 * xmax, xmax + xmax * 0.15)
        ax[1, i].set_ylabel('Percent of episodes')
        ax[1, i].set_xlabel(title)
        ystep = cols_ystep[group_col]
        yticks0 = np.arange(0, g['count'].max() + ystep, ystep)
        yticks1 = np.arange(0, g.perc.max() + 4, 4)
        y0_max = yticks0.max()
        y1_max = yticks1.max()
        ax[0, i].set_yticks(yticks0)
        ax[0, i].set_ylim(0 - y0_max * 0.1, y0_max + y0_max * 0.1)
        ax[1, i].set_yticks(yticks1)
        ax[1, i].set_ylim(0 - y1_max * 0.1, y1_max + y1_max * 0.1)
    ax[0, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')
    ax[1, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')

    ax = ax.flatten()
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    for i in range(len(ax)):
        ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

    out_name = 'fig4_outcomes-by-demographics' + age_suffix + '.png'
    out_name_svg = out_name[:-4] + '.svg'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
    plt.savefig(out_path / out_name_svg, dpi=300, bbox_inches='tight')
    plt.close()
#endregion

# .... Grouped line plot for episodes with no kit result
#region
outcome_plot = ['No FIT result']

pointsize = 18

cols_ystep = {'age_group_screen2': 40000,
              'age_group_granular': 20000,
              'imd_quintile': 10000,
              'subject_gender': 20000}

for age_group_col, age_suffix in age_group_cols.items():
    col_and_label = {age_group_col: 'Age group',
                     'imd_quintile': 'IMD quintile',
                     'subject_gender': 'Sex'}

    figsize = (12, 7)
    fig, ax = plt.subplots(2, 3, figsize=figsize, sharey=False, tight_layout=True,
                           gridspec_kw={'wspace': 0.5, 'hspace': 0.6})
    for i, (group_col, title) in enumerate(col_and_label.items()):
        print(group_col)
        g = res_all.loc[res_all.column == group_col].copy()
        g = g.loc[g.outcome.isin(outcome_plot)]
        g = g.rename(columns={'value': group_col})
        ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
        ticks['x'] = np.arange(ticks.shape[0])
        ticks['xticklabel'] = ticks[group_col]
        if group_col == 'age_group_granular':  # show only the lower bound (keep the open-ended top group)
            ticks['xticklabel'] = [lab if lab.endswith('+') else re.sub('-', '-\n', lab) # lab.split('-')[0]
                                   for lab in ticks['xticklabel']]

        if group_col == 'imd_quintile':
            ticks.xticklabel = ticks.xticklabel.replace({'01 - Most deprived': '01\nMost deprived',
                                                         '05 - Least deprived': '05\nLeast deprived'})
        g = g.merge(ticks, how='left')

        for j, outcome in enumerate(outcome_plot):
            color = 'C' + str(j)
            s = g.loc[g.outcome == outcome]
            ax[0, i].plot(s['x'], s['count'], label='No kit result', color=color)
            ax[0, i].scatter(s['x'], s['count'], s=pointsize, color=color)
            ax[1, i].plot(s['x'], s['perc'], label='No kit result', color=color)
            ax[1, i].scatter(s['x'], s['perc'], s=pointsize, color=color)
            ax[1, i].fill_between(s['x'], s.perc_low, s.perc_high, facecolor=color, alpha=0.25)

        if not title.startswith('IMD'):
            title_panel = title.lower()
        else:
            title_panel = title

        xmax = ticks['x'].max()

        ax[0, i].set(title = 'Number of episodes by\n' + title_panel)
        ax[0, i].grid(which='major', alpha=0.5)
        ax[0, i].set_xticks(ticks['x'])
        ax[0, i].set_xticklabels(ticks['xticklabel'])
        ax[0, i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
        ax[0, i].set_ylabel('Number of episodes')
        ax[0, i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax[0, i].set_xlabel(title)

        ax[1, i].set(title = 'Percent of episodes by\n' + title_panel)
        ax[1, i].grid(which='major', alpha=0.5)
        ax[1, i].set_xticks(ticks['x'])
        ax[1, i].set_xticklabels(ticks['xticklabel'])
        ax[1, i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
        ax[1, i].set_ylabel('Percent of episodes\n(of all episodes)')
        ax[1, i].set_xlabel(title)

        ystep = cols_ystep[group_col]
        yticks0 = np.arange(0, g['count'].max() + ystep, ystep)
        yticks1 = np.arange(0, g.perc.max() + 0.5, 0.5)
        y0_max = yticks0.max()
        y1_max = yticks1.max()

        ax[0, i].set_yticks(yticks0)
        ax[0, i].set_ylim(0 - y0_max * 0.1, y0_max + y0_max * 0.1)
        ax[1, i].set_yticks(yticks1)
        ax[1, i].set_ylim(0 - y1_max * 0.1, y1_max + y1_max * 0.1)

    ax[0, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')
    ax[1, 2].legend(frameon=False, bbox_to_anchor=(1.7, 1), title='Outcome', alignment='left')

    ax = ax.flatten()
    letters = ['A', 'B', 'C', 'D', 'E', 'F']  #, 'G', 'H']
    for i in range(len(ax)):
        ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

    out_name = 'no-fit-result-by-demographics' + age_suffix + '.png'
    out_name_svg = out_name[:-4] + '.svg'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
    plt.savefig(out_path / out_name_svg, dpi=300, bbox_inches='tight')
    plt.close()
#endregion

# .... Grouped line plot for episodes with no investigation
#region
outcome_plot = ['FIT positive, no investigation']

pointsize = 18

cols_ystep = {'age_group_screen2': 10000,
              'age_group_granular': 5000,
              'imd_quintile': 5000,
              'subject_gender': 10000}

for age_group_col, age_suffix in age_group_cols.items():
    col_and_label = {age_group_col: 'Age group',
                     'imd_quintile': 'IMD quintile',
                     'subject_gender': 'Sex'}

    figsize = (11.5, 7)
    fig, ax = plt.subplots(2, 3, figsize=figsize, sharey=False, tight_layout=True,
                            gridspec_kw={'wspace': 0.5, 'hspace': 0.6})
    for i, (group_col, title) in enumerate(col_and_label.items()):
        print(group_col)

        g = res_pos.loc[res_pos.column == group_col].copy()
        g = g.loc[g.outcome.isin(outcome_plot)]
        g = g.rename(columns={'value': group_col})

        ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
        ticks['x'] = np.arange(ticks.shape[0])
        ticks['xticklabel'] = ticks[group_col]
        if group_col == 'age_group_granular':  # show only the lower bound (keep the open-ended top group)
            ticks['xticklabel'] = [lab if lab.endswith('+') else re.sub('-', '-\n', lab) # lab.split('-')[0]
                                   for lab in ticks['xticklabel']]

        if group_col == 'imd_quintile':
            ticks.xticklabel = ticks.xticklabel.replace({'01 - Most deprived': '01\nMost deprived',
                                                         '05 - Least deprived': '05\nLeast deprived'})
        g = g.merge(ticks, how='left')

        for j, outcome in enumerate(outcome_plot):
            color = 'C' + str(j)
            s = g.loc[g.outcome == outcome]

            ax[0, i].plot(s['x'], s['count'], label='FIT ≥120 µg/g,\nno investigation', color=color)
            ax[0, i].scatter(s['x'], s['count'], s=pointsize, color=color)
            ax[1, i].plot(s['x'], s['perc'], label='FIT ≥120 µg/g,\nno investigation', color=color)
            ax[1, i].scatter(s['x'], s['perc'], s=pointsize, color=color)
            ax[1, i].fill_between(s['x'], s.perc_low, s.perc_high, facecolor=color, alpha=0.25)

        if not title.startswith('IMD'):
            title_panel = title.lower()
        else:
            title_panel = title

        xmax = ticks['x'].max()

        ax[0, i].set(title = 'Number of episodes by\n' + title_panel)
        ax[0, i].grid(which='major', alpha=0.5)
        ax[0, i].set_xticks(ticks['x'])
        ax[0, i].set_xticklabels(ticks['xticklabel'])
        ax[0, i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
        ax[0, i].set_ylabel('Number of episodes')
        ax[0, i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax[0, i].set_xlabel(title)

        ax[1, i].set(title = 'Percent of episodes by\n' + title_panel)
        ax[1, i].grid(which='major', alpha=0.5)
        ax[1, i].set_xticks(ticks['x'])
        ax[1, i].set_xticklabels(ticks['xticklabel'])
        ax[1, i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
        ax[1, i].set_ylabel('Percent of episodes')
        ax[1, i].set_xlabel(title)

        ystep = cols_ystep[group_col]
        yticks0 = np.arange(0, g['count'].max() + ystep, ystep)
        yticks1 = np.arange(0, g.perc.max() + 5, 5)
        y0_max = yticks0.max()
        y1_max = yticks1.max()

        ax[0, i].set_yticks(yticks0)
        ax[0, i].set_ylim(0 - y0_max * 0.1, y0_max + y0_max * 0.1)
        ax[1, i].set_yticks(yticks1)
        ax[1, i].set_ylim(0 - y1_max * 0.1, y1_max + y1_max * 0.1)

    ax[0, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')
    ax[1, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')

    ax = ax.flatten()
    letters = ['A', 'B', 'C', 'D', 'E', 'F'] #, 'G', 'H']
    for i in range(len(ax)):
        ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

    out_name = 'fig5_no-investigation-by-demographics' + age_suffix + '.png'
    out_name_svg = out_name[:-4] + '.svg'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
    plt.savefig(out_path / out_name_svg, dpi=300, bbox_inches='tight')
    plt.close()
#endregion

# .... Grouped line plot for total number of episodes
#region
pointsize = 18

cols_ystep = {'age_group_screen2' : [3000000, 10],
              'age_group_granular': [1000000, 10],
              'imd_quintile': [1000000, 5],
              'subject_gender': [2000000, 10],
              'prevalent_incident_status': [2000000, 10]}

for age_group_col, age_suffix in age_group_cols.items():
    col_and_label_full = {age_group_col: 'Age group',
                          'imd_quintile': 'IMD quintile',
                          'subject_gender': 'Sex',
                          'prevalent_incident_status': 'Screening history'}

    figsize = (14, 7)
    fig, ax = plt.subplots(2, 4, figsize=figsize, sharey=False, tight_layout=True)
    for i, (group_col, title) in enumerate(col_and_label_full.items()):
        print(group_col)

        g = res_num.loc[res_num.column == group_col].copy()
        g = g.rename(columns={'value': group_col})
        ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
        ticks['x'] = np.arange(ticks.shape[0])
        ticks['xticklabel'] = ticks[group_col]
        if group_col == 'age_group_granular':  # show only the lower bound (keep the open-ended top group)
            ticks['xticklabel'] = [lab if lab.endswith('+') else re.sub('-', '-\n', lab) # lab.split('-')[0]
                                   for lab in ticks['xticklabel']]
        if group_col == 'imd_quintile':
            ticks.xticklabel = ticks.xticklabel.replace({
                '01 - Most deprived': '01\nMost deprived',
                '05 - Least deprived': '05\nLeast deprived'
            })
        g = g.merge(ticks, how='left')

        ax[0, i].plot(g['x'], g['count'], label='FIT ≥120 µg/g,\nno investigation', color='C0')
        ax[0, i].scatter(g['x'], g['count'], s=pointsize, color='C0')
        ax[1, i].plot(g['x'], g['perc'], label='FIT ≥120 µg/g,\nno investigation', color='C0')
        ax[1, i].scatter(g['x'], g['perc'], s=pointsize, color='C0')
        ax[1, i].fill_between(g['x'], g.perc_low, g.perc_high, facecolor='C0', alpha=0.25)

        if not title.startswith('IMD'):
            title_panel = title.lower()
        else:
            title_panel = title

        xmax = ticks['x'].max()

        ax[0, i].set(title = 'Number of episodes by\n' + title_panel)
        ax[0, i].grid(which='major', alpha=0.5)
        ax[0, i].set_xticks(ticks['x'])
        ax[0, i].set_xticklabels(ticks['xticklabel'])
        ax[0, i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
        ax[0, i].set_ylabel('Number of episodes')
        ax[0, i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax[0, i].set_xlabel(title)

        ax[1, i].set(title = 'Percent of episodes by\n' + title_panel)
        ax[1, i].grid(which='major', alpha=0.5)
        ax[1, i].set_xticks(ticks['x'])
        ax[1, i].set_xticklabels(ticks['xticklabel'])
        ax[1, i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
        ax[1, i].set_ylabel('Percent of episodes\n(of all episodes)')
        ax[1, i].set_xlabel(title)

        ystep0 = cols_ystep[group_col][0]
        yticks0 = np.arange(0, g['count'].max() + ystep0, ystep0)
        y0_max = yticks0.max()
        ax[0, i].set_yticks(yticks0)
        ax[0, i].set_ylim(0 - y0_max * 0.1, y0_max + y0_max * 0.1)

        ystep1 = cols_ystep[group_col][1]
        yticks1 = np.arange(0, g.perc.max() + ystep1, ystep1)
        y1_max = yticks1.max()
        ax[1, i].set_yticks(yticks1)
        ax[1, i].set_ylim(0 - y1_max * 0.1, y1_max + y1_max * 0.1)

        # ax[1, i].set_yticks(np.arange(0, 100, 10))
        # ax[1, i].set_ylim(0 - 100 * 0.1, 100 + 100 * 0.1)

    ax = ax.flatten()
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for i in range(len(ax)):
        ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

    out_name = 'num-episodes-by-demographics' + age_suffix + '.png'
    out_name_svg = out_name[:-4] + '.svg'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
    plt.savefig(out_path / out_name_svg, dpi=300, bbox_inches='tight')
    plt.close()
#endregion

# .... Chi squared tests
#region
ref_level = {'age_group_screen2': '50-59',
             'age_group_granular': '50-54',
             'imd_quintile': '05 - Least deprived',
             'subject_gender': 'Female'}

out = pd.DataFrame()
for r, name in zip([res, res_pos, res_all], ['investigated', 'pos', 'all']):
    data = r[['column', 'value', 'outcome', 'ntot', 'count', 'perc']].copy()
    data = data.sort_values(by=['column', 'value', 'outcome'])
    assert data['count'].min() >= 10

    df_ref = data.loc[data.value.isin(ref_level.values())].drop(labels=['value'], axis=1)
    df_ref = df_ref.rename(columns={'perc': 'perc_ref','count': 'count_ref', 'ntot': 'ntot_ref'})
    df_noref = data.loc[~data.value.isin(ref_level.values())]
    df_noref = df_noref.merge(df_ref, how='left')
    df_noref['ratio'] = (df_noref.perc / df_noref.perc_ref)
    df_noref['ratio2'] = (df_noref.perc_ref / df_noref.perc)
    df_noref['subs'] = (df_noref.ratio >= 1.1) | (df_noref.ratio2 >= 1.1)

    test = df_noref.apply(lambda x: stat.stats.proportions_chisquare(count=x[['count_ref', 'count']].tolist(),
                                                                     nobs=x[['ntot_ref', 'ntot']].tolist()),
                                                                     axis=1)
    df_noref['xstat'] = test.apply(lambda x: x[0])
    df_noref['pval'] = test.apply(lambda x: x[1])
    df_noref['psig'] = ''
    df_noref.loc[df_noref.pval < 0.05, 'psig'] = '< 0.05'
    df_noref.loc[df_noref.pval < 0.01, 'psig'] = '< 0.01'
    df_noref.loc[df_noref.pval < 0.001, 'psig'] = '< 0.001'

    df_noref = df_noref.round(1)
    df_noref['data_subset'] = name
    out = pd.concat(objs=[out, df_noref], axis=0)

out.to_csv(out_path / ('chisq_outcomes-by-demographics.csv'), index=False)
#endregion

#endregion


# ---- Outcomes by demographics, stratified by screening history ----
# [Updated by Claude Code to use multiple age group definitions; manually verified]
# These two sections mirror the "Grouped line plot for outcomes with investigation"
# and "Grouped line plot for episodes with no investigation" sections above, but the
# rows now correspond to screening history rather than count vs percent: the top row
# shows first-time responders (prevalent episodes) and the bottom row previous
# responders (incident episodes), and only percentages are plotted.
#region
print('\nSUMMARISING OUTCOMES BY DEMOGRAPHICS AND SCREENING HISTORY...')

# Screening history -> plot row wording (prevalent on top, incident on the bottom)
hist_and_label = {'Prevalent': 'First-time responders',
                  'Incident': 'Previous responders'}


# .... Grouped line plot for outcomes with investigation, by screening history
#region
outcome_plot_hist = ['Colorectal cancer',
                     'Advanced premalignant polyp',
                     'Non-advanced premalignant polyp',
                     'Other findings',
                     'No abnormality']
pointsize = 18

for age_group_col, age_suffix in age_group_cols.items():
    col_and_label_demo = {age_group_col: 'Age group',
                          'imd_quintile': 'IMD quintile',
                          'subject_gender': 'Sex'}

    figsize = (12, 8)
    fig, ax = plt.subplots(2, 3, figsize=figsize, sharey=False, tight_layout=True)
    for i, (group_col, title) in enumerate(col_and_label_demo.items()):
        print(group_col)
        g = res_hist.loc[res_hist.column == group_col].copy()
        g = g.rename(columns={'value': group_col})
        ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
        ticks['x'] = np.arange(ticks.shape[0])
        ticks['xticklabel'] = ticks[group_col]
        if group_col == 'age_group_granular':  # show only the lower bound (keep the open-ended top group)
            ticks['xticklabel'] = [lab if lab.endswith('+') else re.sub('-', '-\n', lab) # lab.split('-')[0]
                                   for lab in ticks['xticklabel']]
        if group_col == 'imd_quintile':
            ticks.xticklabel = ticks.xticklabel.replace({
                '01 - Most deprived': '01\nMost deprived',
                '05 - Least deprived': '05\nLeast deprived'
            })
        g = g.merge(ticks, how='left')

        for k, (hist, hist_label) in enumerate(hist_and_label.items()):
            gh = g.loc[g.prevalent_incident_status == hist]
            for j, outcome in enumerate(outcome_plot_hist):
                color = 'C' + str(j)
                s = gh.loc[gh.outcome == outcome]
                ax[k, i].plot(s['x'], s['perc'], label=outcome_labels[outcome], color=color)
                ax[k, i].scatter(s['x'], s['perc'], s=pointsize, color=color)
                ax[k, i].fill_between(s['x'], s.perc_low, s.perc_high, facecolor=color, alpha=0.25)

        if not title.startswith('IMD'):
            title_panel = title.lower()
        else:
            title_panel = title
        xmax = ticks['x'].max()
        yticks1 = np.arange(0, g.perc.max() + 4, 4)
        y1_max = yticks1.max()

        for k, (hist, hist_label) in enumerate(hist_and_label.items()):
            ax[k, i].set(title=hist_label + ':\nOutcomes by ' + title_panel)
            ax[k, i].grid(which='major', alpha=0.5)
            ax[k, i].set_xticks(ticks['x'])
            ax[k, i].set_xticklabels(ticks['xticklabel'])
            ax[k, i].set_xlim(0 - 0.15 * xmax, xmax + xmax * 0.15)
            ax[k, i].set_ylabel('Percent of episodes')
            ax[k, i].set_xlabel(title)
            ax[k, i].set_yticks(yticks1)
            ax[k, i].set_ylim(0 - y1_max * 0.1, y1_max + y1_max * 0.1)

    ax[0, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')
    ax[1, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')

    ax = ax.flatten()
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    for i in range(len(ax)):
        ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

    out_name = 'suppl_fig5_outcomes-investigated-by-demographics-history' + age_suffix + '.png'
    out_name_svg = out_name[:-4] + '.svg'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
    plt.savefig(out_path / out_name_svg, dpi=300, bbox_inches='tight')
    plt.close()
#endregion


# .... Grouped line plot for episodes with no investigation, by screening history
#region
outcome_plot_noinv = ['FIT positive, no investigation']
pointsize = 18

for age_group_col, age_suffix in age_group_cols.items():
    col_and_label_demo = {age_group_col: 'Age group',
                          'imd_quintile': 'IMD quintile',
                          'subject_gender': 'Sex'}

    figsize = (12, 8)
    fig, ax = plt.subplots(2, 3, figsize=figsize, sharey=False, tight_layout=True,
                           gridspec_kw={'wspace': 0.5, 'hspace': 0.6})
    for i, (group_col, title) in enumerate(col_and_label_demo.items()):
        print(group_col)
        g = res_pos_hist.loc[res_pos_hist.column == group_col].copy()
        g = g.loc[g.outcome.isin(outcome_plot_noinv)]
        g = g.rename(columns={'value': group_col})
        ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
        ticks['x'] = np.arange(ticks.shape[0])
        ticks['xticklabel'] = ticks[group_col]
        if group_col == 'age_group_granular':  # show only the lower bound (keep the open-ended top group)
            ticks['xticklabel'] = [lab if lab.endswith('+') else re.sub('-', '-\n', lab) # lab.split('-')[0]
                                   for lab in ticks['xticklabel']]
        if group_col == 'imd_quintile':
            ticks.xticklabel = ticks.xticklabel.replace({'01 - Most deprived': '01\nMost deprived',
                                                         '05 - Least deprived': '05\nLeast deprived'})
        g = g.merge(ticks, how='left')

        for k, (hist, hist_label) in enumerate(hist_and_label.items()):
            gh = g.loc[g.prevalent_incident_status == hist]
            for j, outcome in enumerate(outcome_plot_noinv):
                color = 'C' + str(j)
                s = gh.loc[gh.outcome == outcome]
                ax[k, i].plot(s['x'], s['perc'], label='FIT ≥120 µg/g,\nno investigation', color=color)
                ax[k, i].scatter(s['x'], s['perc'], s=pointsize, color=color)
                ax[k, i].fill_between(s['x'], s.perc_low, s.perc_high, facecolor=color, alpha=0.25)

        if not title.startswith('IMD'):
            title_panel = title.lower()
        else:
            title_panel = title
        xmax = ticks['x'].max()
        yticks1 = np.arange(0, g.perc.max() + 5, 5)
        y1_max = yticks1.max()

        for k, (hist, hist_label) in enumerate(hist_and_label.items()):
            ax[k, i].set(title=hist_label + ':\nEpisodes by ' + title_panel)
            ax[k, i].grid(which='major', alpha=0.5)
            ax[k, i].set_xticks(ticks['x'])
            ax[k, i].set_xticklabels(ticks['xticklabel'])
            ax[k, i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
            ax[k, i].set_ylabel('Percent of episodes')
            ax[k, i].set_xlabel(title)
            ax[k, i].set_yticks(yticks1)
            ax[k, i].set_ylim(0 - y1_max * 0.1, y1_max + y1_max * 0.1)

    ax[0, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')
    ax[1, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')

    ax = ax.flatten()
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    for i in range(len(ax)):
        ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

    out_name = 'suppl_fig6_no-investigation-by-demographics-history' + age_suffix + '.png'
    out_name_svg = out_name[:-4] + '.svg'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
    plt.savefig(out_path / out_name_svg, dpi=300, bbox_inches='tight')
    plt.close()
#endregion

#endregion


# ---- Advanced / non-advanced polyps (non-exclusive indicators) by demographics ----
# [Created by Claude Code by reusing previous plotting code; manually verified]
# Mirrors the "outcomes-investigated-by-demographics-history" plot: the top row shows
# first-time responders (prevalent episodes) and the bottom row previous responders
# (incident episodes), and only percentages are shown. Plots the non-exclusive polyp
# indicators (advanced_polyp, nonadvanced_polyp) rather than the mutually-exclusive
# 'worst outcome' categories. A separate figure is produced per age-group definition.
#region
print('\nSUMMARISING NON-EXCLUSIVE POLYP INDICATORS BY DEMOGRAPHICS AND SCREENING HISTORY...')

outcome_plot_polyp = ['Advanced colorectal polyp (non-exclusive)',
                      'Non-advanced colorectal polyp (non-exclusive)']
polyp_labels = {'Advanced colorectal polyp (non-exclusive)': 'Advanced colorectal\npolyp (non-exclusive)',
                'Non-advanced colorectal polyp (non-exclusive)': 'Non-advanced colorectal\npolyp (non-exclusive)'}
polyp_colors = {'Advanced colorectal polyp (non-exclusive)': 'C1',
                'Non-advanced colorectal polyp (non-exclusive)': 'C2'}


pointsize = 18

for age_group_col, age_suffix in age_group_cols.items():
    col_and_label_demo = {age_group_col: 'Age group',
                          'imd_quintile': 'IMD quintile',
                          'subject_gender': 'Sex'}

    figsize = (12, 7)
    fig, ax = plt.subplots(2, 3, figsize=figsize, sharey=False, tight_layout=True)
    for i, (group_col, title) in enumerate(col_and_label_demo.items()):
        print(group_col)
        g = res_polyp_hist.loc[res_polyp_hist.column == group_col].copy()
        g = g.rename(columns={'value': group_col})
        ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
        ticks['x'] = np.arange(ticks.shape[0])
        ticks['xticklabel'] = ticks[group_col]
        if group_col == 'age_group_granular':  # show only the lower bound (keep the open-ended top group)
            ticks['xticklabel'] = [lab if lab.endswith('+') else re.sub('-', '-\n', lab) # lab.split('-')[0]
                                   for lab in ticks['xticklabel']]
        if group_col == 'imd_quintile':
            ticks.xticklabel = ticks.xticklabel.replace({
                '01 - Most deprived': '01\nMost deprived',
                '05 - Least deprived': '05\nLeast deprived'
            })
        g = g.merge(ticks, how='left')

        for k, (hist, hist_label) in enumerate(hist_and_label.items()):
            gh = g.loc[g.prevalent_incident_status == hist]
            for j, outcome in enumerate(outcome_plot_polyp):
                color = polyp_colors[outcome]
                s = gh.loc[gh.outcome == outcome]
                ax[k, i].plot(s['x'], s['perc'], label=polyp_labels[outcome], color=color)
                ax[k, i].scatter(s['x'], s['perc'], s=pointsize, color=color)
                ax[k, i].fill_between(s['x'], s.perc_low, s.perc_high, facecolor=color, alpha=0.25)

        if not title.startswith('IMD'):
            title_panel = title.lower()
        else:
            title_panel = title
        xmax = ticks['x'].max()
        yticks1 = np.arange(0, g.perc.max() + 5, 5)
        y1_max = yticks1.max()

        for k, (hist, hist_label) in enumerate(hist_and_label.items()):
            ax[k, i].set(title=hist_label + ':\nEpisodes with polyp by ' + title_panel)
            ax[k, i].grid(which='major', alpha=0.5)
            ax[k, i].set_xticks(ticks['x'])
            ax[k, i].set_xticklabels(ticks['xticklabel'])
            ax[k, i].set_xlim(0 - 0.15 * xmax, xmax + xmax * 0.15)
            ax[k, i].set_ylabel('Percent of episodes\n(of investigated episodes)')
            ax[k, i].set_xlabel(title)
            ax[k, i].set_yticks(yticks1)
            ax[k, i].set_ylim(0 - y1_max * 0.1, y1_max + y1_max * 0.1)

    ax[0, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')
    ax[1, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')

    ax = ax.flatten()
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    for i in range(len(ax)):
        ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

    out_name = 'suppl_fig7_polyp-nonexclusive-by-demographics-history' + age_suffix + '.png'
    out_name_svg = out_name[:-4] + '.svg'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
    plt.savefig(out_path / out_name_svg, dpi=300, bbox_inches='tight')
    plt.close()
#endregion


# .... Grouped line plot for non-exclusive polyps, not stratified by screening history
# Mirrors the "outcomes with investigation" plot: number of episodes in the top row and
# percent in the bottom row. Completes the set (unstratified + history-stratified versions).
#region
cols_ystep = {'age_group_screen2': 20000,
              'age_group_granular': 10000,
              'imd_quintile': 5000,
              'subject_gender': 20000}
pointsize = 18

for age_group_col, age_suffix in age_group_cols.items():
    col_and_label = {age_group_col: 'Age group',
                     'imd_quintile': 'IMD quintile',
                     'subject_gender': 'Sex'}

    figsize = (12.75, 7)
    fig, ax = plt.subplots(2, 3, figsize=figsize, sharey=False, tight_layout=True)
    for i, (group_col, title) in enumerate(col_and_label.items()):
        print(group_col)
        g = res_polyp.loc[res_polyp.column == group_col].copy()
        g = g.rename(columns={'value': group_col})
        ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
        ticks['x'] = np.arange(ticks.shape[0])
        ticks['xticklabel'] = ticks[group_col]
        if group_col == 'age_group_granular':  # show only the lower bound (keep the open-ended top group)
            ticks['xticklabel'] = [lab if lab.endswith('+') else re.sub('-', '-\n', lab) #lab.split('-')[0]
                                   for lab in ticks['xticklabel']]
        if group_col == 'imd_quintile':
            ticks.xticklabel = ticks.xticklabel.replace({
                '01 - Most deprived': '01\nMost deprived',
                '05 - Least deprived': '05\nLeast deprived'
            })
        g = g.merge(ticks, how='left')
        for j, outcome in enumerate(outcome_plot_polyp):
            color = polyp_colors[outcome]
            s = g.loc[g.outcome == outcome]
            ax[0, i].plot(s['x'], s['count'], label=polyp_labels[outcome], color=color)
            ax[0, i].scatter(s['x'], s['count'], s=pointsize, color=color)
            ax[1, i].plot(s['x'], s['perc'], label=polyp_labels[outcome], color=color)
            ax[1, i].scatter(s['x'], s['perc'], s=pointsize, color=color)
            ax[1, i].fill_between(s['x'], s.perc_low, s.perc_high, facecolor=color, alpha=0.25)

        if not title.startswith('IMD'):
            title_panel = title.lower()
        else:
            title_panel = title
        xmax = ticks['x'].max()
        ax[0, i].set(title='Number of episodes by\n' + title_panel)
        ax[0, i].grid(which='major', alpha=0.5)
        ax[0, i].set_xticks(ticks['x'])
        ax[0, i].set_xticklabels(ticks['xticklabel'])
        ax[0, i].set_xlim(0 - 0.15 * xmax, xmax + xmax * 0.15)
        ax[0, i].set_ylabel('Number of episodes')
        ax[0, i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax[0, i].set_xlabel(title)
        ax[1, i].set(title='Percent of episodes by\n' + title_panel)
        ax[1, i].grid(which='major', alpha=0.5)
        ax[1, i].set_xticks(ticks['x'])
        ax[1, i].set_xticklabels(ticks['xticklabel'])
        ax[1, i].set_xlim(0 - 0.15 * xmax, xmax + xmax * 0.15)
        ax[1, i].set_ylabel('Percent of episodes\n(of investigated episodes)')
        ax[1, i].set_xlabel(title)
        ystep = cols_ystep[group_col]
        yticks0 = np.arange(0, g['count'].max() + ystep, ystep)
        yticks1 = np.arange(0, g.perc.max() + 5, 5)
        y0_max = yticks0.max()
        y1_max = yticks1.max()
        ax[0, i].set_yticks(yticks0)
        ax[0, i].set_ylim(0 - y0_max * 0.1, y0_max + y0_max * 0.1)
        ax[1, i].set_yticks(yticks1)
        ax[1, i].set_ylim(0 - y1_max * 0.1, y1_max + y1_max * 0.1)
    ax[0, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')
    ax[1, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')

    ax = ax.flatten()
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    for i in range(len(ax)):
        ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

    out_name = 'suppl_fig7_polyp-nonexclusive-by-demographics' + age_suffix + '.png'
    out_name_svg = out_name[:-4] + '.svg'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
    plt.savefig(out_path / out_name_svg, dpi=300, bbox_inches='tight')
    plt.close()
#endregion


# ---- No investigation by age and deprivation, given sex and screening history ----
# [Created by Claude Code by reusing previous plotting code; manually verified]
# [Created as a sanity check; not included in manuscript]
# The demographic analyses above are essentially main effects (one variable at a time).
# This figure instead conditions on sex and screening history, to assess how the
# non-investigation rate of FIT positive episodes varies with age and with deprivation
# within each sex x screening-history group:
#   Subplot A: no investigation ~ age (granular) | sex, screening history
#   Subplot B: no investigation ~ deprivation    | sex, screening history
# Each line is one sex x screening-history combination.
#region
print('\nSUMMARISING NON-INVESTIGATION BY AGE AND DEPRIVATION | SEX, SCREENING HISTORY...')

noinv_cols = {'age_group_granular': 'Age group',
              'imd_quintile': 'IMD quintile'}

# Sex x screening-history combinations (one line each)
combos = [(sex, hist) for sex in ['Female', 'Male'] for hist in ['Prevalent', 'Incident']]
pointsize = 18

fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=False, tight_layout=True)
for i, (group_col, title) in enumerate(noinv_cols.items()):
    print(group_col)
    g = res_pos_hist_sex.loc[res_pos_hist_sex.column == group_col].copy()
    g = g.rename(columns={'value': group_col})

    ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
    ticks['x'] = np.arange(ticks.shape[0])
    ticks['xticklabel'] = ticks[group_col]
    #if group_col == 'age_group_granular':  # show only the lower bound (keep the open-ended top group)
    #    ticks['xticklabel'] = [lab if lab.endswith('+') else re.sub('-', '-\n', lab) # lab.split('-')[0]
    #                           for lab in ticks['xticklabel']]
    if group_col == 'imd_quintile':
        ticks.xticklabel = ticks.xticklabel.replace({'01 - Most deprived': '01\nMost deprived',
                                                     '05 - Least deprived': '05\nLeast deprived'})
    g = g.merge(ticks, how='left')

    for j, (sex, hist) in enumerate(combos):
        color = 'C' + str(j)
        s = g.loc[(g.subject_gender == sex) & (g.prevalent_incident_status == hist)].sort_values(by='x')
        label = sex + ', ' + hist_and_label[hist].lower()
        ax[i].plot(s['x'], s['perc'], label=label, color=color)
        ax[i].scatter(s['x'], s['perc'], s=pointsize, color=color)
        ax[i].fill_between(s['x'], s.perc_low, s.perc_high, facecolor=color, alpha=0.2)

    if not title.startswith('IMD'):
        title_panel = title.lower()
    else:
        title_panel = title
    xmax = ticks['x'].max()
    ax[i].set(title='Non-investigation by ' + title_panel)
    ax[i].grid(which='major', alpha=0.5)
    ax[i].set_xticks(ticks['x'])
    ax[i].set_xticklabels(ticks['xticklabel'])
    ax[i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
    ax[i].set_xlabel(title)
    ax[i].set_ylabel('Non-investigation rate (%)\n(of FIT positive episodes)')
    ymax = g.perc_high.max()
    ax[i].set_ylim(0, ymax + ymax * 0.1)

ax[0].legend(frameon=False, title='Sex, screening history')
ax[1].legend(frameon=False, title='Sex, screening history')

letters = ['A', 'B']
for i in range(len(ax)):
    ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

out_name = 'suppl_fig8_no-investigation-by-age-imd-given-sex-history.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()
#endregion


# ---- Investigated outcomes by age and deprivation, given sex and screening history ----
# [Created by Claude Code by reusing previous plotting code; manually verified]
# [Created as a sanity check; not included in manuscript]
# Mirrors the "no investigation ~ age/imd | sex, screening history" figure above, but
# instead plots the investigated-outcome percentages for colorectal cancer, advanced
# colorectal polyps and non-advanced colorectal polyps. A single figure has six
# subplots: columns are the three outcomes (colorectal cancer, advanced polyp,
# non-advanced polyp) and rows are the two demographic variables (top: granular age,
# bottom: deprivation). Each line is one sex x screening-history combination:
#   Column 1: colorectal cancer          ~ age/imd | sex, screening history
#   Column 2: advanced colorectal polyp  ~ age/imd | sex, screening history
#   Column 3: non-adv. colorectal polyp  ~ age/imd | sex, screening history
#region
print('\nSUMMARISING OUTCOMES BY AGE AND DEPRIVATION | SEX, SCREENING HISTORY...')

# Outcome column value -> column title (uses the requested display names)
outcome_hist_sex = {'Colorectal cancer': 'Colorectal cancer',
                    'Advanced premalignant polyp': 'Advanced colorectal polyp',
                    'Non-advanced premalignant polyp': 'Non-advanced colorectal polyp'}

# Demographic variable per row (top: age, bottom: deprivation)
hist_sex_cols = {'age_group_granular': 'Age group',
                 'imd_quintile': 'IMD quintile'}

# Sex x screening-history combinations (one line each)
combos = [(sex, hist) for sex in ['Female', 'Male'] for hist in ['Prevalent', 'Incident']]
pointsize = 18

fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharey=False, tight_layout=True)
for r, (group_col, title) in enumerate(hist_sex_cols.items()):
    print(group_col)
    g_all = res_hist_sex.loc[res_hist_sex.column == group_col].copy()
    g_all = g_all.rename(columns={'value': group_col})

    ticks = g_all[[group_col]].drop_duplicates().sort_values(by=group_col)
    ticks['x'] = np.arange(ticks.shape[0])
    ticks['xticklabel'] = ticks[group_col]
    if group_col == 'imd_quintile':
        ticks.xticklabel = ticks.xticklabel.replace({'01 - Most deprived': '01\nMost deprived',
                                                     '05 - Least deprived': '05\nLeast deprived'})
    g_all = g_all.merge(ticks, how='left')

    for m, (outcome, outcome_title) in enumerate(outcome_hist_sex.items()):
        go = g_all.loc[g_all.outcome == outcome]
        for j, (sex, hist) in enumerate(combos):
            color = 'C' + str(j)
            s = go.loc[(go.subject_gender == sex) &
                       (go.prevalent_incident_status == hist)].sort_values(by='x')
            label = sex + ', ' + hist_and_label[hist].lower()
            ax[r, m].plot(s['x'], s['perc'], label=label, color=color)
            ax[r, m].scatter(s['x'], s['perc'], s=pointsize, color=color)
            ax[r, m].fill_between(s['x'], s.perc_low, s.perc_high, facecolor=color, alpha=0.2)

        if not title.startswith('IMD'):
            title_panel = title.lower()
        else:
            title_panel = title
        xmax = ticks['x'].max()
        ax[r, m].set(title=outcome_title + ' by ' + title_panel)
        ax[r, m].grid(which='major', alpha=0.5)
        ax[r, m].set_xticks(ticks['x'])
        ax[r, m].set_xticklabels(ticks['xticklabel'])
        ax[r, m].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
        ax[r, m].set_xlabel(title)
        ax[r, m].set_ylabel('Percent of episodes\n(of investigated episodes)')
        ymax = go.perc_high.max()
        ax[r, m].set_ylim(0, ymax + ymax * 0.1)
        ax[r, m].legend(frameon=False)

ax = ax.flatten()
letters = ['A', 'B', 'C', 'D', 'E', 'F']
for i in range(len(ax)):
    ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

out_name = 'suppl_fig9_outcomes-by-age-imd-given-sex-history.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()
#endregion

