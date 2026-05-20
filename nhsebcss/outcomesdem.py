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


# Paths
data_path_clean = Path("C:/Users/andres.Tamm/Desktop/bcss_clean_data")
out_path = Path("Z:/andres/nhsebcss/results/primary")
out_path.mkdir(exist_ok=True, parents=True)


# Method for proportion confidence intervals
ci_method = 'wilson'


# ---- Read data ----
#region 
print('\nREADING DATA...')

# Read cleaned data
df = pd.read_csv(data_path_clean / 'episodes_clean.csv',
                 usecols=['anon_subject_epis_id', 'anon_screening_subject_id', 'episode_subtype',
                          'episode_start_date', 'episode_end_date', 'test_kit_logged_date', 'test_kit_result_date',
                          'subject_gender', 'prevalent_incident_status', 'age_group_screen2', 'imd_quintile',
                          'outcome', 'advanced_polyp', 'nonadvanced_polyp', 'premalignant_polyp',
                          'analyser_reading_used', 'outcome_simple'])
n_epi = df.anon_subject_epis_id.nunique()
assert df.shape[0] == n_epi

# Dates to datetime (note that date format is different as dates were already reformatted in clean data)
date_cols = ['episode_start_date', 'episode_end_date', 'test_kit_logged_date', 'test_kit_result_date']
assert all([c in df.columns for c in date_cols])
for c in date_cols:
    print(c)
    df[c] = pd.to_datetime(df[c], format='%Y-%m-%d', exact=True)

# Include missing IMD values? Not atm - small count. Most missing values occur in FIT negative (25,433 of 26,142)
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

#endregion


# ---- Outcomes by age, deprivation, sex (one variable at a time) ----
# Note: not computing outcomes by prev/inc status atm, as this is confounded by age.
#region
print('\nSUMMARISING OUTCOMES BY DEMOGRAPHICS ONE VARIABLE AT A TIME...')

# Create two subsets: episodes with investigation, and FIT positive episodes
dfsub = df.loc[df.outcome.isin(outcomes_with_investigation)]
dfsub_pos = df.loc[df.outcome.isin(outcomes_fit_pos)]

# Grouping variables and axis labels
col_and_label = {'age_group_screen2': 'Age group',
                 'imd_quintile': 'IMD quintile',
                 'subject_gender': 'Sex'
                 }
col_and_label_full = {'age_group_screen2': 'Age group',
                      'imd_quintile': 'IMD quintile',
                      'subject_gender': 'Sex',
                      'prevalent_incident_status': 'Screening history'
                     }

# Compute num and % wrt investigated outcomes by demographics
res = pd.DataFrame()
for i, (group_col, title) in enumerate(col_and_label.items()):
    print(group_col)
    g = grouped_count(dfsub, [group_col], outcome_col='outcome', ci_method=ci_method)
    g['column'] = group_col
    g = g.rename(columns={group_col: 'value'})
    res = pd.concat(objs=[res, g], axis=0)
res = res[['column', 'value'] + [c for c in res.columns if c not in ['column', 'value']]]
assert res['count'].min() >= 10
res.to_csv(out_path / 'outcomes-investigated-by-demographics.csv', index=False)

# Compute num and % wrt all outcomes by demographics
res_all = pd.DataFrame()
for i, (group_col, title) in enumerate(col_and_label.items()):
    print(group_col)
    g = grouped_count(df, [group_col], outcome_col='outcome', ci_method=ci_method)
    g['column'] = group_col
    g = g.rename(columns={group_col: 'value'})
    res_all = pd.concat(objs=[res_all, g], axis=0)
res_all = res_all[['column', 'value'] + [c for c in res_all.columns if c not in ['column', 'value']]]
assert res_all['count'].min() >= 10
res_all.to_csv(out_path / 'outcomes-all-by-demographics.csv', index=False)

# Compute num and % wrt FIT positive outcomes by demographics
res_pos = pd.DataFrame()
for i, (group_col, title) in enumerate(col_and_label.items()):
    print(group_col)
    g = grouped_count(dfsub_pos, [group_col], outcome_col='outcome', ci_method=ci_method)
    g['column'] = group_col
    g = g.rename(columns={group_col: 'value'})
    res_pos = pd.concat(objs=[res_pos, g], axis=0)
res_pos = res_pos[['column', 'value'] + [c for c in res_pos.columns if c not in ['column', 'value']]]
assert res_pos['count'].min() >= 10
res_pos.to_csv(out_path / 'outcomes-pos-by-demographics.csv', index=False)

# Compute total num episodes by demographics
res_num = pd.DataFrame()
for i, (group_col, title) in enumerate(col_and_label_full.items()):
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
res_num.to_csv(out_path / 'num-episodes-by-demographics.csv', index=False)

# Outcome labels for plotting
outcome_labels = {
    'Colorectal cancer': 'Colorectal\ncancer', 
    'Advanced premalignant polyp': 'Advanced\npremalignant polyp', 
    'Non-advanced premalignant polyp': 'Non-advanced\npremalignant polyp', 
    'Other findings': 'Other findings',
    'No abnormality': 'No abnormality',
    'FIT positive, no investigation': 'FIT positive, no investigation'
    }


# .... Grouped line plot for outcomes with investigation
#region
outcome_plot = ['Colorectal cancer', 
                'Advanced premalignant polyp', 
                'Non-advanced premalignant polyp',
                'Other findings',
                'No abnormality'
                ]

cols_ystep = {'age_group_screen2': 10000,
              'imd_quintile': 2000,
              'subject_gender': 10000
              }
pointsize = 18

fig, ax = plt.subplots(2, 3, figsize=(12, 7), sharey=False, tight_layout=True)
for i, (group_col, title) in enumerate(col_and_label.items()):
    print(group_col)
    g = res.loc[res.column == group_col].copy()
    g = g.rename(columns={'value': group_col})
    ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
    ticks['x'] = np.arange(ticks.shape[0])
    ticks['xticklabel'] = ticks[group_col]
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

out_name = 'fig4_outcomes-by-demographics.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()
#endregion

# .... Grouped line plot for episodes with no kit result
#region
cols_ystep = {'age_group_screen2': 40000, 
              'imd_quintile': 10000, 
              'subject_gender': 20000,
              }

outcome_plot = ['No FIT result']

pointsize = 18


fig, ax = plt.subplots(2, 3, figsize=(12, 7), sharey=False, tight_layout=True,
                       gridspec_kw={'wspace': 0.5, 'hspace': 0.6})
for i, (group_col, title) in enumerate(col_and_label.items()):
    print(group_col)
    g = res_all.loc[res_all.column == group_col].copy()
    g = g.loc[g.outcome.isin(outcome_plot)]
    g = g.rename(columns={'value': group_col})
    ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
    ticks['x'] = np.arange(ticks.shape[0])
    ticks['xticklabel'] = ticks[group_col]

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

out_name = 'no-fit-result-by-demographics.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()
#endregion

# .... Grouped line plot for episodes with no investigation
#region
cols_ystep = {'age_group_screen2': 10000, 
              'imd_quintile': 5000, 
              'subject_gender': 10000
              }

outcome_plot = ['FIT positive, no investigation']

pointsize = 18

fig, ax = plt.subplots(2, 3, figsize=(12, 7), sharey=False, tight_layout=True,
                        gridspec_kw={'wspace': 0.5, 'hspace': 0.6})
for i, (group_col, title) in enumerate(col_and_label.items()):
    print(group_col)

    g = res_pos.loc[res_pos.column == group_col].copy()
    g = g.loc[g.outcome.isin(outcome_plot)]
    g = g.rename(columns={'value': group_col})

    ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
    ticks['x'] = np.arange(ticks.shape[0])
    ticks['xticklabel'] = ticks[group_col]

    if group_col == 'imd_quintile':
        ticks.xticklabel = ticks.xticklabel.replace({'01 - Most deprived': '01\nMost deprived',
                                                     '05 - Least deprived': '05\nLeast deprived'})
    g = g.merge(ticks, how='left')

    for j, outcome in enumerate(outcome_plot):
        color = 'C' + str(j)
        s = g.loc[g.outcome == outcome]

        ax[0, i].plot(s['x'], s['count'], label='FIT above thr,\nno investigation', color=color)
        ax[0, i].scatter(s['x'], s['count'], s=pointsize, color=color)
        ax[1, i].plot(s['x'], s['perc'], label='FIT above thr,\nno investigation', color=color)
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
    ax[1, i].set_ylabel('Percent of episodes\n(of episodes with FIT above thr)')
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

out_name = 'suppl_fig5_no-investigation-by-demographics.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()
#endregion

# .... Grouped line plot for total number of episodes
#region
cols_ystep = {'age_group_screen2': [3000000, 10],
              'imd_quintile': [1000000, 5],
              'subject_gender': [2000000, 10],
              'prevalent_incident_status': [2000000, 10]}
pointsize = 18
col_and_label['prevalent_incident_status'] = 'Screening history'

fig, ax = plt.subplots(2, 4, figsize=(14, 7), sharey=False, tight_layout=True)
for i, (group_col, title) in enumerate(col_and_label_full.items()):
    print(group_col)

    g = res_num.loc[res_num.column == group_col].copy()
    g = g.rename(columns={'value': group_col})
    ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
    ticks['x'] = np.arange(ticks.shape[0])
    ticks['xticklabel'] = ticks[group_col]
    if group_col == 'imd_quintile':
        ticks.xticklabel = ticks.xticklabel.replace({
            '01 - Most deprived': '01\nMost deprived',
            '05 - Least deprived': '05\nLeast deprived'
        })
    g = g.merge(ticks, how='left')

    ax[0, i].plot(g['x'], g['count'], label='FIT above thr,\nno investigation', color='C0')
    ax[0, i].scatter(g['x'], g['count'], s=pointsize, color='C0')
    ax[1, i].plot(g['x'], g['perc'], label='FIT above thr,\nno investigation', color='C0')
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

out_name = 'num-episodes-by-demographics.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()
#endregion

# .... Chi squared tests
#region
ref_level = {'age_group_screen2': '50-59',
             'imd_quintile': '05 - Least deprived',
             'subject_gender': 'Female'}

for r, name in zip([res, res_pos, res_all], ['investigated', 'pos', 'all']):
    dfsub = r[['column', 'value', 'outcome', 'ntot', 'count', 'perc']].copy()
    dfsub = dfsub.sort_values(by=['column', 'value', 'outcome'])

    df_ref = dfsub.loc[dfsub.value.isin(ref_level.values())].drop(labels=['value'], axis=1)
    df_ref = df_ref.rename(columns={'perc': 'perc_ref','count': 'count_ref', 'ntot': 'ntot_ref'})
    df_noref = dfsub.loc[~dfsub.value.isin(ref_level.values())]
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
    df_noref.to_csv(out_path / ('chisq_outcomes-'+name+'-by-demographics.csv'), index=False)
#endregion

#endregion

