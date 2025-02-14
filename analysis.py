"""Main analysis
Assumes that dataprep.py has been run.
"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint
from utils import grouped_count, hide_low_counts, summarise_cat


# Paths
data_path_clean = Path("C:/Users/p0Po/Desktop/bcss_clean_data") 
out_path = Path("Z:/andres/nhsebcss/results")

# Method for proportion confidence intervals
ci_method = 'wilson'

# Read cleaned data
df = pd.read_csv(data_path_clean / 'episodes_clean.csv')
n_epi = df.ANON_SUBJECT_EPIS_ID.nunique()
assert df.shape[0] == n_epi

# Dates to datetime (note that date format is different as dates were already reformatted in clean data)
date_cols = ['EPISODE_START_DATE', 'EPISODE_END_DATE', 'TEST_KIT_LOGGED_DATE', 'TEST_KIT_RESULT_DATE']
assert all([c in df.columns for c in date_cols])
for c in date_cols:
    print(c)
    df[c] = pd.to_datetime(df[c], format='%Y-%m-%d', exact=True)

# Get outcome categories that were associated with an investigation, and those that were not
outcomes = df.OUTCOME.unique().tolist()
outcomes_without_investigation = ['FIT negative', 'FIT inadequate participation', 'FIT positive, no investigation']
outcomes_with_investigation = [c for c in outcomes if c not in outcomes_without_investigation]
print(outcomes_without_investigation, outcomes_with_investigation)
assert all(c in outcomes for c in outcomes_without_investigation)

# Outcome categories associated with incomplete participation
outcomes_incomplete = ['FIT inadequate participation', 'FIT positive, no investigation']
assert all(c in outcomes for c in outcomes_incomplete)


# ---- 1. Screening outcomes overall ----
#region

# Get episode count for each outcome
digits = 3
out = df.OUTCOME.value_counts().reset_index()

# Add total number of episodes
# and number of episodes that were followed by investigation and that were not
n_investigated = out.loc[out.OUTCOME.isin(outcomes_with_investigation), 'count'].sum()
n_not_investigated = out.loc[out.OUTCOME.isin(outcomes_without_investigation), 'count'].sum()
n_total = out['count'].sum()

row = pd.DataFrame([['Episodes with investigation', n_investigated]], columns=['OUTCOME', 'count'])
out = pd.concat(objs=[out, row], axis=0).reset_index(drop=True)

row = pd.DataFrame([['Episodes without investigation', n_not_investigated]], columns=['OUTCOME', 'count'])
out = pd.concat(objs=[out, row], axis=0).reset_index(drop=True)

row = pd.DataFrame([['Total number of episodes', n_total]], columns=['OUTCOME', 'count'])
out = pd.concat(objs=[out, row], axis=0).reset_index(drop=True)

# Get percentage and CI
out['ntot'] = n_total
out['perc'] = out['count'] / out.ntot * 100
out.perc = out.perc.round(digits)
for c in ['perc_low', 'perc_high']:
    out[c] = np.nan
for i, row in out.iterrows():
    ci_low, ci_high = proportion_confint(count=row['count'], nobs=row['ntot'], method=ci_method)
    out.loc[i, 'perc_low'] = np.round(ci_low * 100, digits)
    out.loc[i, 'perc_high'] = np.round(ci_high * 100, digits)

# Do the same for outcomes with investigation
out2 = out.loc[out.OUTCOME.isin(outcomes_with_investigation)].copy()
out2['ntot'] = out2['count'].sum()
out2['perc'] = out2['count'] / out2.ntot * 100
out2.perc = out2.perc.round(digits)
for i, row in out2.iterrows():
    ci_low, ci_high = proportion_confint(count=row['count'], nobs=row['ntot'], method=ci_method)
    out2.loc[i, 'perc_low'] = np.round(ci_low * 100, digits)
    out2.loc[i, 'perc_high'] = np.round(ci_high * 100, digits)

# Add reformatted percentage 
out['perc_reformat'] = out.perc.astype(str) + ' (' + out.perc_low.astype(str) + ', ' + out.perc_high.astype(str) + ')'
out2['perc_reformat'] = out2.perc.astype(str) + ' (' + out2.perc_low.astype(str) + ', ' + out2.perc_high.astype(str) + ')'

# Reorder rows, so that episodes without investigation come last
row_order = ['Total number of episodes', 'Episodes with investigation'] + \
    [o for o in out.OUTCOME if o in outcomes_with_investigation] + \
    ['Episodes without investigation'] + \
    [o for o in out.OUTCOME if o in outcomes_without_investigation]   
out = out.set_index('OUTCOME').loc[row_order].reset_index()

# Reformat and save
out = out.drop(labels='ntot', axis=1)
colnames = ['Outcome', 'Num episodes', 'Percent episodes',
            'Percent episodes (ci low)', 'Percent episodes (ci upp)', 'Percent episodes (ci low, ci upp)']
out.columns = colnames
out.to_csv(out_path / 'outcomes.csv', index=False)

out2 = out2.drop(labels='ntot', axis=1)
colnames = ['Outcome', 'Num episodes', 'Percent episodes',
            'Percent episodes (ci low)', 'Percent episodes (ci upp)', 'Percent episodes (ci low, ci upp)']
out2.columns = colnames
out2.to_csv(out_path / 'outcomes_investigated.csv', index=False)

#b = vcount2.reset_index()
#sns.barplot(data=b, x='count', y='OUTCOME', color='C0', orient='h', hue_order=outcomes_with_investigation)
#endregion


# ---- 2. Screening outcomes over time ----
#region
out_path_sub = out_path / 'outcomes_over_time'
out_path_sub.mkdir(exist_ok=True, parents=True)

# For plotting outcomes over time,
# drop second quarter of 2019 and 2024, as these are not complete quarters
print(df.TEST_KIT_LOGGED_DATE.min(), df.TEST_KIT_LOGGED_DATE.max())
mask = (df.TEST_KIT_LOGGED_YEAR.isin([2019, 2024])) & (df.TEST_KIT_LOGGED_QUARTER == 2)
dfsub = df.loc[~mask]

# Create grouping variable
dfsub['YEAR_QUARTER'] = dfsub.TEST_KIT_LOGGED_YEAR.astype(str) + '-' + dfsub.TEST_KIT_LOGGED_QUARTER.astype(str)

# Get count and percentage for all outcomes over time
count_over_time = grouped_count(dfsub, 'YEAR_QUARTER', 'OUTCOME', ci_method=ci_method)

# Get count and percentage for outcomes with investigation over time
dfsub_investigated = dfsub.loc[dfsub.OUTCOME.isin(outcomes_with_investigation)]
count_over_time_investigated = grouped_count(dfsub_investigated, 'YEAR_QUARTER', 'OUTCOME', ci_method=ci_method)

# Map to x-axis coordinates
tmap = count_over_time[['YEAR_QUARTER']].drop_duplicates().sort_values(by='YEAR_QUARTER')
tmap['x'] = np.arange(tmap.shape[0])
tmap['xticklabel'] = tmap.YEAR_QUARTER
tmap['year'] = tmap.YEAR_QUARTER.str[:4].astype(int)
tmap['quarter'] = tmap.YEAR_QUARTER.str[5:].astype(int)
print(tmap)
count_over_time = count_over_time.merge(tmap, how='left')
count_over_time_investigated = count_over_time_investigated.merge(tmap, how='left')


# Plot outcomes that were followed by investigation
outcome_plot = ['Colorectal cancer', 'High-risk findings or LNPCP', 'Premalignant polyp(s)',
                'Other abnormality at whole colon investigation', 'No abnormality at whole colon investigation', 
                'No result at whole colon investigation']

fig, ax = plt.subplots(2, 1, figsize=(16, 8), tight_layout=True)
for i, outcome in enumerate(outcome_plot):
    s = count_over_time_investigated.loc[count_over_time_investigated.OUTCOME == outcome]
    color = 'C' + str(i)
    ax[0].plot(s['x'], s['count'], label=outcome, color=color)
    ax[0].scatter(s['x'], s['count'], s=8, color=color)
    ax[1].plot(s['x'], s['perc'], label=outcome, color=color)
    ax[1].scatter(s['x'], s['perc'], s=8, color=color)
    ax[1].fill_between(s['x'], s.perc_low, s.perc_high, facecolor=color, alpha=0.25)

ax[0].set_xticks(tmap['x'])
ax[0].set_xticklabels(tmap['xticklabel'])
ax[0].legend(frameon=False, bbox_to_anchor=(1.01, 1), fontsize=10, title='Outcome', alignment='left')
ax[0].grid(which='major', alpha=0.5)
ax[0].set_xlabel('Test kit logged date (year-quarter)', fontsize=10)
ax[0].set_ylabel('Number of episodes', fontsize=10)
ax[0].set_title('A. Number of episodes with each outcome')

ax[1].set_xticks(tmap['x'])
ax[1].set_xticklabels(tmap['xticklabel'])
ax[1].legend(frameon=False, bbox_to_anchor=(1.01, 1), fontsize=10, title='Outcome', alignment='left')
ax[1].grid(which='major', alpha=0.5)
ax[1].set_xlabel('Test kit logged date (year-quarter)', fontsize=10)
ax[1].set_ylabel('Percent of episodes \n(of episodes with investigation)', fontsize=10)
ax[1].set_title('B. Percent of episodes with each outcome')

plt.savefig(out_path_sub / 'outcomes_by_quarter.png', dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / 'outcomes_by_quarter.svg', bbox_inches='tight')
#plt.show()
plt.close()

# Plot number of episodes where participation was incomplete
outcome_plot = ['FIT inadequate participation', 'FIT positive, no investigation']

fig, ax = plt.subplots(2, 1, figsize=(12, 8), tight_layout=True)
for i, outcome in enumerate(outcome_plot):
    s = count_over_time.loc[count_over_time.OUTCOME == outcome]
    color = 'C' + str(i)
    ax[0].plot(s['x'], s['count'], label=outcome, color=color)
    ax[0].scatter(s['x'], s['count'], s=8, color=color)
    ax[1].plot(s['x'], s['perc'], label=outcome, color=color)
    ax[1].scatter(s['x'], s['perc'], s=8, color=color)
    ax[1].fill_between(s['x'], s.perc_low, s.perc_high, facecolor=color, alpha=0.25)

ax[0].set_xticks(tmap['x'])
ax[0].set_xticklabels(tmap['xticklabel'])
ax[0].legend(frameon=False, fontsize=10)
ax[0].grid(which='major', alpha=0.5)
ax[0].set_xlabel('Test kit logged date (year-quarter)', fontsize=10)
ax[0].set_ylabel('Number of episodes', fontsize=10)
ax[0].set_title('A. Number of episodes with incomplete participation')

ax[1].set_xticks(tmap['x'])
ax[1].set_xticklabels(tmap['xticklabel'])
ax[1].legend(frameon=False, fontsize=10)
ax[1].grid(which='major', alpha=0.5)
ax[1].set_xlabel('Test kit logged date (year-quarter)', fontsize=10)
ax[1].set_ylabel('Percent of episodes (of all episodes)', fontsize=10)
ax[1].set_title('B. Percent of episodes with incomplete participation')

plt.savefig(out_path_sub / 'incomplete_participation_by_quarter.png', dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / 'incomplete_participation_by_quarter.svg', bbox_inches='tight')
plt.close()

# Plot total number of episodes over time
count_all = count_over_time[['YEAR_QUARTER', 'ntot', 'x', 'xticklabel']].drop_duplicates()

fig, ax = plt.subplots(1, 1, figsize=(12, 4), tight_layout=True)

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

plt.savefig(out_path_sub / 'num_episodes_by_quarter.png', dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / 'num_episodes_by_quarter.svg', bbox_inches='tight')
#plt.show()
plt.close()


# Save as tables 
t1 = count_over_time.drop(labels=['x', 'xticklabel'], axis=1)
t1 = hide_low_counts(t1)
t1 = t1[['year', 'quarter', 'OUTCOME', 'count', 'perc', 'perc_low', 'perc_high', 'perc_reformat']]
t1.columns = ['Year', 'Quarter', 'Outcome', 'Num episodes', 'Percent episodes', 
              'Percent episodes (ci low)', 'Percent episodes (ci high)', 'Percent episodes (ci low, ci high)']
row_order = ['Colorectal cancer', 'High-risk findings or LNPCP', 'Premalignant polyp(s)',
             'Other abnormality at whole colon investigation', 'No abnormality at whole colon investigation', 
             'No result at whole colon investigation', 'Polyps without endoscopy record', 'Other polyp',
             'FIT negative', 'FIT inadequate participation', 'FIT positive, no investigation']
assert all(o in outcomes for o in row_order)
t1 = t1.set_index('Outcome').loc[row_order].reset_index()

t2 = count_over_time_investigated.drop(labels=['x', 'xticklabel'], axis=1)
t2 = hide_low_counts(t2)
t2 = t2[['year', 'quarter', 'OUTCOME', 'count', 'perc', 'perc_low', 'perc_high', 'perc_reformat']]
t2.columns = ['Year', 'Quarter', 'Outcome', 'Num episodes', 'Percent episodes', 
              'Percent episodes (ci low)', 'Percent episodes (ci high)', 'Percent episodes (ci low, ci high)']
row_order = ['Colorectal cancer', 'High-risk findings or LNPCP', 'Premalignant polyp(s)',
             'Other abnormality at whole colon investigation', 'No abnormality at whole colon investigation', 
             'No result at whole colon investigation', 'Polyps without endoscopy record', 'Other polyp']
assert all(o in outcomes_with_investigation for o in row_order)
t2 = t2.set_index('Outcome').loc[row_order].reset_index()

t1.to_csv(out_path_sub / 'outcomes_by_quarter.csv', index=False)
t2.to_csv(out_path_sub / 'outcomes_investigated_by_quarter.csv', index=False)

#endregion


# ---- 3. Screening outcomes by age, deprivation, sex ----
#region

# Save path
out_path_sub = out_path / 'outcomes_by_demographics'
out_path_sub.mkdir(exist_ok=True, parents=True)

# Axis labels for columns
col_and_label = {'AGE_GROUP': 'Age group',
                 #'IMD_DECILE': 'IMD decile',
                 'IMD_QUINTILE': 'IMD quintile',
                 'SUBJECT_GENDER': 'Gender'}

# Subset data to outcomes with investigation
dfsub = df.loc[df.OUTCOME.isin(outcomes_with_investigation)]


# Grouped line plot for outcomes with investigation
cols_ystep = {'AGE_GROUP': 10000, 
              'IMD_QUINTILE': 5000, 
              'SUBJECT_GENDER': 10000}

outcome_plot = ['Colorectal cancer', 'High-risk findings or LNPCP', 'Premalignant polyp(s)',
                 'Other abnormality at whole colon investigation', 'No abnormality at whole colon investigation', 
                 'No result at whole colon investigation']

pointsize = 18

fig, ax = plt.subplots(2, 3, figsize=(16, 7), sharey=False, tight_layout=True)
for i, (group_col, title) in enumerate(col_and_label.items()):
    print(group_col)

    g = grouped_count(dfsub, group_col, ci_method=ci_method)
    g = g.loc[g.ntot >= 100]

    ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
    ticks['x'] = np.arange(ticks.shape[0])
    ticks['xticklabel'] = ticks[group_col]

    if group_col == 'IMD_QUINTILE':
        ticks.xticklabel = ticks.xticklabel.replace({'01 - Most deprived': '01\nMost deprived',
                                                     '05 - Least deprived': '05\nLeast deprived'})
    g = g.merge(ticks, how='left')

    for j, outcome in enumerate(outcome_plot):
        color = 'C' + str(j)
        s = g.loc[g.OUTCOME == outcome]

        ax[0, i].plot(s['x'], s['count'], label=outcome, color=color)
        ax[0, i].scatter(s['x'], s['count'], s=pointsize, color=color)
        ax[1, i].plot(s['x'], s['perc'], label=outcome, color=color)
        ax[1, i].scatter(s['x'], s['perc'], s=pointsize, color=color)
        ax[1, i].fill_between(s['x'], s.perc_low, s.perc_high, facecolor=color, alpha=0.25)

    if not title.startswith('IMD'):
        title_panel = title.lower()
    else:
        title_panel = title

    xmax = ticks['x'].max()

    ax[0, i].set(title = 'Number of outcomes by ' + title_panel)
    ax[0, i].grid(which='major', alpha=0.5)
    ax[0, i].set_xticks(ticks['x'])
    ax[0, i].set_xticklabels(ticks['xticklabel'])
    ax[0, i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
    ax[0, i].set_ylabel('Number of episodes')
    ax[0, i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax[0, i].set_xlabel(title)

    ax[1, i].set(title = 'Percent of outcomes by ' + title_panel)
    ax[1, i].grid(which='major', alpha=0.5)
    ax[1, i].set_xticks(ticks['x'])
    ax[1, i].set_xticklabels(ticks['xticklabel'])
    ax[1, i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
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

out_name = 'outcomes_lineplot_by_demographics.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path_sub / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()


# Grouped line plot for outcomes with incomplete participation
cols_ystep = {'AGE_GROUP': 20000, 
              'IMD_QUINTILE': 10000, 
              'SUBJECT_GENDER': 20000}

outcome_plot = outcomes_incomplete

pointsize = 18

fig, ax = plt.subplots(2, 3, figsize=(16, 7), sharey=False, tight_layout=True)
for i, (group_col, title) in enumerate(col_and_label.items()):
    print(group_col)

    g = grouped_count(df, group_col, ci_method=ci_method)
    g = g.loc[g.OUTCOME.isin(outcome_plot)]
    g = g.loc[g.ntot >= 100]

    ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
    ticks['x'] = np.arange(ticks.shape[0])
    ticks['xticklabel'] = ticks[group_col]

    if group_col == 'IMD_QUINTILE':
        ticks.xticklabel = ticks.xticklabel.replace({'01 - Most deprived': '01\nMost deprived',
                                                     '05 - Least deprived': '05\nLeast deprived'})
    g = g.merge(ticks, how='left')

    for j, outcome in enumerate(outcome_plot):
        color = 'C' + str(j)
        s = g.loc[g.OUTCOME == outcome]

        ax[0, i].plot(s['x'], s['count'], label=outcome, color=color)
        ax[0, i].scatter(s['x'], s['count'], s=pointsize, color=color)
        ax[1, i].plot(s['x'], s['perc'], label=outcome, color=color)
        ax[1, i].scatter(s['x'], s['perc'], s=pointsize, color=color)
        ax[1, i].fill_between(s['x'], s.perc_low, s.perc_high, facecolor=color, alpha=0.25)

    if not title.startswith('IMD'):
        title_panel = title.lower()
    else:
        title_panel = title

    xmax = ticks['x'].max()

    ax[0, i].set(title = 'Number of episodes by ' + title_panel)
    ax[0, i].grid(which='major', alpha=0.5)
    ax[0, i].set_xticks(ticks['x'])
    ax[0, i].set_xticklabels(ticks['xticklabel'])
    ax[0, i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
    ax[0, i].set_ylabel('Number of episodes')
    ax[0, i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax[0, i].set_xlabel(title)

    ax[1, i].set(title = 'Percent of episodes by ' + title_panel)
    ax[1, i].grid(which='major', alpha=0.5)
    ax[1, i].set_xticks(ticks['x'])
    ax[1, i].set_xticklabels(ticks['xticklabel'])
    ax[1, i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
    ax[1, i].set_ylabel('Percent of episodes')
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
ax[1, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')

ax = ax.flatten()
letters = ['A', 'B', 'C', 'D', 'E', 'F']
for i in range(len(ax)):
    ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

out_name = 'incomplete_participation_lineplot_by_demographics.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path_sub / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()


# Grouped line plot for total number of episodes
cols_ystep = {'AGE_GROUP': 2000000, 
              'IMD_QUINTILE': 1000000, 
              'SUBJECT_GENDER': 2000000}

outcome_plot = outcomes_incomplete

pointsize = 18

fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=False, tight_layout=True)
for i, (group_col, title) in enumerate(col_and_label.items()):
    print(group_col)

    g = grouped_count(df, group_col, ci_method=ci_method)
    g = g[[group_col, 'ntot']].drop_duplicates()

    ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
    ticks['x'] = np.arange(ticks.shape[0])
    ticks['xticklabel'] = ticks[group_col]

    if group_col == 'IMD_QUINTILE':
        ticks.xticklabel = ticks.xticklabel.replace({'01 - Most deprived': '01\nMost deprived',
                                                     '05 - Least deprived': '05\nLeast deprived'})
    g = g.merge(ticks, how='left')

    ax[i].plot(g['x'], g['ntot'], label=outcome, color='C0')
    ax[i].scatter(g['x'], g['ntot'], s=pointsize, color='C0')

    if not title.startswith('IMD'):
        title_panel = title.lower()
    else:
        title_panel = title

    xmax = ticks['x'].max()

    ax[i].set(title = 'Number of episodes by ' + title_panel)
    ax[i].grid(which='major', alpha=0.5)
    ax[i].set_xticks(ticks['x'])
    ax[i].set_xticklabels(ticks['xticklabel'])
    ax[i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
    ax[i].set_ylabel('Number of episodes')
    ax[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax[i].set_xlabel(title)

    ystep = cols_ystep[group_col]
    yticks0 = np.arange(0, g['ntot'].max() + ystep, ystep)
    y0_max = yticks0.max()

    ax[i].set_yticks(yticks0)
    ax[i].set_ylim(0 - y0_max * 0.1, y0_max + y0_max * 0.1)

ax = ax.flatten()
letters = ['A', 'B', 'C']
for i in range(len(ax)):
    ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

out_name = 'num_episodes_lineplot_by_demographics.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path_sub / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()


# Save tables to disk
col_and_label_all = {'AGE_GROUP': 'Age group',
                 'IMD_DECILE': 'IMD decile',
                 'IMD_QUINTILE': 'IMD quintile',
                 'SUBJECT_GENDER': 'Gender'}
for i, (group_col, group_name) in enumerate(col_and_label_all.items()):
    print(group_col)
    
    g1 = grouped_count(df, group_col, ci_method=ci_method, include_zero_count=False)
    g2 = grouped_count(dfsub, group_col, ci_method=ci_method, include_zero_count=False)

    g1 = hide_low_counts(g1)
    g2 = hide_low_counts(g2)

    g1.columns = [group_name, 'Outcome', 'Total num episodes', 'Num episodes',
                  'Percent episodes', 'Percent episodes (ci low)', 'Percent episodes (ci upp)', 'Percent episodes (ci low, ci upp)']

    g1.columns = [group_name, 'Outcome', 'Total num episodes', 'Num episodes', 'Percent episodes',
                  'Percent episodes (ci low)', 'Percent episodes (ci upp)', 'Percent episodes (ci low, ci upp)']

    out_name1 = 'outcomes_by_' + group_col.lower() + '.csv'
    out_name2 = 'outcomes_investigated_by_' + group_col.lower() + '.csv'

    g1.to_csv(out_path_sub / out_name1, index=False)
    g2.to_csv(out_path_sub / out_name2, index=False)


# Grouped bar plots
barplot = True
if barplot:

    # Outcomes with investigation bar plot
    cols_ystep = {'AGE_GROUP': 10000, 
                  'IMD_QUINTILE': 5000, 
                  'SUBJECT_GENDER': 10000}

    outcome_plot = ['Colorectal cancer', 'High-risk findings or LNPCP', 'Premalignant polyp(s)',
                    'Other abnormality at whole colon investigation', 'No abnormality at whole colon investigation', 
                    'No result at whole colon investigation']
    bar_width = 0.1

    fig, ax = plt.subplots(2, 3, figsize=(16, 7), sharey=False, tight_layout=True)
    for i, (group_col, title) in enumerate(col_and_label.items()):
        print(group_col)

        g = grouped_count(dfsub, group_col, ci_method=ci_method)
        g = g.loc[g.ntot >= 100]
        g['err_low'] = g.perc - g.perc_low
        g['err_high'] = g.perc_high - g.perc

        ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
        ticks['x'] = np.arange(ticks.shape[0])
        ticks['xticklabel'] = ticks[group_col]

        if group_col == 'IMD_QUINTILE':
            ticks.xticklabel = ticks.xticklabel.replace({'01 - Most deprived': '01\nMost deprived',
                                                         '05 - Least deprived': '05\nLeast deprived'})
        g = g.merge(ticks, how='left')

        for j, outcome in enumerate(outcome_plot):
            color = 'C' + str(j)
            gsub = g.loc[g.OUTCOME == outcome]

            yerr = gsub[['err_low', 'err_high']].transpose().to_numpy()

            x = gsub['x'] + bar_width * j
            ax[0, i].bar(x, gsub['count'], width=bar_width, label=outcome, color=color)
            ax[1, i].bar(x, gsub['perc'], width=bar_width, label=outcome, color=color)
            ax[1, i].errorbar(x=x, y=gsub['perc'], yerr=yerr, fmt='none', ecolor='black')

        if not title.startswith('IMD'):
            title_panel = title.lower()
        else:
            title_panel = title

        xmax = ticks['x'].max()
        ncat = len(outcome_plot)
        xmid = (ncat - 1)* bar_width / 2

        ax[0, i].set(title = 'Number of outcomes by ' + title_panel)
        ax[0, i].grid(which='major', alpha=0.5)
        ax[0, i].set_xticks(ticks['x'] + xmid)
        ax[0, i].set_xticklabels(ticks['xticklabel'])
        #ax[0, i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
        ax[0, i].set_ylabel('Number of episodes')
        ax[0, i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax[0, i].set_xlabel(title)

        ax[1, i].set(title = 'Percent of outcomes by ' + title_panel)
        ax[1, i].grid(which='major', alpha=0.5)
        ax[1, i].set_xticks(ticks['x'] + xmid)
        ax[1, i].set_xticklabels(ticks['xticklabel'])
        #ax[1, i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
        ax[1, i].set_ylabel('Percent of episodes')
        ax[1, i].set_xlabel(title)
        
        ystep = cols_ystep[group_col]
        yticks0 = np.arange(0, g['count'].max() + ystep, ystep)
        yticks1 = np.arange(0, g.perc.max() + 4, 4)
        y0_max = yticks0.max()
        y1_max = yticks1.max()

        ax[0, i].set_yticks(yticks0)
        ax[0, i].set_ylim(0, y0_max + y0_max * 0.1)
        ax[1, i].set_yticks(yticks1)
        ax[1, i].set_ylim(0, y1_max + y1_max * 0.1)

    ax[0, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')
    ax[1, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')

    ax = ax.flatten()
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    for i in range(len(ax)):
        ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

    out_name = 'outcomes_barplot_by_demographics.png'
    out_name_svg = out_name[:-4] + '.svg'
    plt.savefig(out_path_sub / out_name, dpi=300, bbox_inches='tight')
    plt.savefig(out_path_sub / out_name_svg, dpi=300, bbox_inches='tight')
    plt.close()


    # Incomplete participation bar plot
    cols_ystep = {'AGE_GROUP': 20000, 
                  'IMD_QUINTILE': 10000, 
                  'SUBJECT_GENDER': 20000}

    outcome_plot = ['FIT inadequate participation', 'FIT positive, no investigation']
    bar_width = 0.25

    fig, ax = plt.subplots(2, 3, figsize=(16, 7), sharey=False, tight_layout=True)
    for i, (group_col, title) in enumerate(col_and_label.items()):
        print(group_col)

        g = grouped_count(df, group_col, ci_method=ci_method)
        g = g.loc[g.ntot >= 100]
        g = g.loc[g.OUTCOME.isin(outcome_plot)]
        g['err_low'] = g.perc - g.perc_low
        g['err_high'] = g.perc_high - g.perc

        ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
        ticks['x'] = np.arange(ticks.shape[0])
        ticks['xticklabel'] = ticks[group_col]

        if group_col == 'IMD_QUINTILE':
            ticks.xticklabel = ticks.xticklabel.replace({'01 - Most deprived': '01\nMost deprived',
                                                         '05 - Least deprived': '05\nLeast deprived'})
        g = g.merge(ticks, how='left')

        for j, outcome in enumerate(outcome_plot):
            color = 'C' + str(j)
            gsub = g.loc[g.OUTCOME == outcome]

            yerr = gsub[['err_low', 'err_high']].transpose().to_numpy()

            x = gsub['x'] + bar_width * j
            ax[0, i].bar(x, gsub['count'], width=bar_width, label=outcome, color=color)
            ax[1, i].bar(x, gsub['perc'], width=bar_width, label=outcome, color=color)
            ax[1, i].errorbar(x=x, y=gsub['perc'], yerr=yerr, fmt='none', ecolor='black')

        if not title.startswith('IMD'):
            title_panel = title.lower()
        else:
            title_panel = title

        xmax = ticks['x'].max()
        ncat = len(outcome_plot)
        xmid = (ncat - 1)* bar_width / 2

        ax[0, i].set(title = 'Number of outcomes by ' + title_panel)
        ax[0, i].grid(which='major', alpha=0.5)
        ax[0, i].set_xticks(ticks['x'] + xmid)
        ax[0, i].set_xticklabels(ticks['xticklabel'])
        #ax[0, i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
        ax[0, i].set_ylabel('Number of episodes')
        ax[0, i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax[0, i].set_xlabel(title)

        ax[1, i].set(title = 'Percent of outcomes by ' + title_panel)
        ax[1, i].grid(which='major', alpha=0.5)
        ax[1, i].set_xticks(ticks['x'] + xmid)
        ax[1, i].set_xticklabels(ticks['xticklabel'])
        #ax[1, i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
        ax[1, i].set_ylabel('Percent of episodes')
        ax[1, i].set_xlabel(title)
        
        ystep = cols_ystep[group_col]
        yticks0 = np.arange(0, g['count'].max() + ystep, ystep)
        yticks1 = np.arange(0, g.perc.max() + 0.5, 0.5)
        y0_max = yticks0.max()
        y1_max = yticks1.max()

        ax[0, i].set_yticks(yticks0)
        ax[0, i].set_ylim(0, y0_max + y0_max * 0.2)
        ax[1, i].set_yticks(yticks1)
        ax[1, i].set_ylim(0, y1_max + y1_max * 0.2)

    ax[0, 2].legend(frameon=False, bbox_to_anchor=(1.05, 1), title='Outcome', alignment='left')
    ax[1, 2].legend(frameon=False, bbox_to_anchor=(1.90, 1), title='Outcome', alignment='left')

    ax = ax.flatten()
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    for i in range(len(ax)):
        ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

    out_name = 'incomplete participation_barplot_by_demographics.png'
    out_name_svg = out_name[:-4] + '.svg'
    plt.savefig(out_path_sub / out_name, dpi=300, bbox_inches='tight')
    plt.savefig(out_path_sub / out_name_svg, dpi=300, bbox_inches='tight')
    plt.close()


    # Grouped bar plot for total number of episodes
    cols_ystep = {'AGE_GROUP': 2000000, 
                  'IMD_QUINTILE': 1000000, 
                  'SUBJECT_GENDER': 2000000}

    outcome_plot = outcomes_incomplete
    bar_width = 0.5

    fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=False, tight_layout=True)
    for i, (group_col, title) in enumerate(col_and_label.items()):
        print(group_col)

        g = grouped_count(df, group_col, ci_method=ci_method)
        g = g[[group_col, 'ntot']].drop_duplicates()

        ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
        ticks['x'] = np.arange(ticks.shape[0])
        ticks['xticklabel'] = ticks[group_col]

        if group_col == 'IMD_QUINTILE':
            ticks.xticklabel = ticks.xticklabel.replace({'01 - Most deprived': '01\nMost deprived',
                                                        '05 - Least deprived': '05\nLeast deprived'})
        g = g.merge(ticks, how='left')

        ax[i].bar(g['x'], g['ntot'], width=bar_width, label=outcome, color='C0')

        if not title.startswith('IMD'):
            title_panel = title.lower()
        else:
            title_panel = title

        xmax = ticks['x'].max()

        ax[i].set(title = 'Number of episodes by ' + title_panel)
        ax[i].grid(which='major', alpha=0.5)
        ax[i].set_xticks(ticks['x'])
        ax[i].set_xticklabels(ticks['xticklabel'])
        #ax[i].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
        ax[i].set_ylabel('Number of episodes')
        ax[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax[i].set_xlabel(title)

        ystep = cols_ystep[group_col]
        yticks0 = np.arange(0, g['ntot'].max() + ystep, ystep)
        y0_max = yticks0.max()

        ax[i].set_yticks(yticks0)
        ax[i].set_ylim(0, y0_max + y0_max * 0.1)

    ax = ax.flatten()
    letters = ['A', 'B', 'C']
    for i in range(len(ax)):
        ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

    out_name = 'num_episodes_barplot_by_demographics.png'
    out_name_svg = out_name[:-4] + '.svg'
    plt.savefig(out_path_sub / out_name, dpi=300, bbox_inches='tight')
    plt.savefig(out_path_sub / out_name_svg, dpi=300, bbox_inches='tight')
    plt.close()

#endregion


# ---- 4. FIT positivity overall ----
#region

# Helper functions
def positivity(df, thr=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
               digits=3):
    """Compute FIT positivity at prespecified thresholds"""
    pos = pd.DataFrame()

    for t in thr:
        mask = df.ANALYSER_READING_USED >= t
        npos = mask.sum().item()
        ntot = df.shape[0]
        ppos = npos / ntot * 100
        ci_low, ci_high = proportion_confint(count=npos, nobs=ntot, method='wilson')
        row = {'fit_thr': [t],
               'num_episodes': [ntot],
               'num_positives': [npos], 
               'percent_positives': [ppos],
               'perc_low': [ci_low * 100],
               'perc_high': [ci_high * 100]}
        row = pd.DataFrame(row)
        pos = pd.concat(objs=[row, pos], axis=0)
    pos = pos.reset_index(drop=True)
    for c in ['percent_positives', 'perc_low', 'perc_high']:
        pos[c] = pos[c].round(digits)
    return pos


def plot_positivity(pos, ax, label=None, color='C0', plot_percent=True, pointsize=24):
    """Plot FIT positivity at prespecified thresholds"""
    thr = pos.fit_thr
    
    if plot_percent:
        y = pos.percent_positives
        ylabel = 'Percent of positives'
    else:
        y = pos.num_positives
        ylabel = 'Number of positives'

    ax.plot(thr, y, label=label, color=color)
    ax.scatter(thr, y, color=color, s=pointsize)
    if plot_percent:
        ax.fill_between(thr, pos.perc_low, pos.perc_high, alpha=0.25, facecolor=color)
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    ax.set(xlabel='FIT threshold (µg/g)', ylabel=ylabel)
    ax.set_xticks(np.arange(thr.min(), thr.max() + 10, 10))
    ax.set_ylim(0 - y.max() * 0.05, y.max() + y.max() * 0.05)
    #ax.set_yticks(np.arange(0, ymax))
    ax.grid(which='major', alpha=0.5)

    return ax

# Path
out_path_sub = out_path / 'positivity'
out_path_sub.mkdir(exist_ok=True, parents=True)

# Get episodes with adequate FIT participation
df_fit = df.loc[df.OUTCOME != 'FIT inadequate participation']
assert not df_fit.ANALYSER_READING_USED.isna().any()

# Compute positivity
pos = positivity(df_fit)

# Plot positivity
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_positivity(pos, ax[0], plot_percent=False)
plot_positivity(pos, ax[1], plot_percent=True)
ax[0].set_title('A. Number of positives')
ax[1].set_title('B. Percent of positives')
ax[0].set_yticks(np.arange(0, pos.num_positives.max() + 200000, 200000))
ax[1].set_yticks(np.arange(0, pos.percent_positives.max() + 1, 1))

out_name = 'positivity.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path_sub / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()

# Save table
pos.columns = ['FIT threshold (µg/g)', 'Num episodes', 'Num positives', 'Percent positive',
               'Percent positive (ci low)', 'Percent positive (ci upp)']
pos.to_csv(out_path_sub / 'positivity.csv', index=False)

#endregion


# ---- 5. FIT positivity over time ----
#region

# Drop the second quarter of 2019 and 2024 as these are not complete quarters
mask = (df_fit.TEST_KIT_LOGGED_YEAR.isin([2019, 2024])) & (df_fit.TEST_KIT_LOGGED_QUARTER == 2)
df_fit_time = df_fit.loc[~mask]

# Compute FIT positivity at various thresholds within each year-quarter
df_fit_time['YEAR_QUARTER'] = df_fit_time.TEST_KIT_LOGGED_YEAR.astype(str) + '-' + df_fit_time.TEST_KIT_LOGGED_QUARTER.astype(str)
year_quarter = df_fit_time.YEAR_QUARTER.drop_duplicates().sort_values()

fit_thr = [10, 20, 40, 60, 80, 100, 120]
pos = pd.DataFrame()
for i, value in enumerate(year_quarter):
    print(value)
    s = df_fit_time.loc[df_fit_time.YEAR_QUARTER == value]
    p = positivity(s, thr=fit_thr)
    p['YEAR_QUARTER'] = value
    pos = pd.concat(objs=[pos, p], axis=0)
pos = pos[['YEAR_QUARTER'] + [c for c in pos.columns if c not in ['YEAR_QUARTER']]]
pos = pos.reset_index(drop=True)

# Map year-quarter to x-axis coordinates
ticks = df_fit_time[['YEAR_QUARTER']].drop_duplicates().sort_values(by='YEAR_QUARTER')
ticks['x'] = np.arange(ticks.shape[0])
ticks['xticklabel'] = ticks.YEAR_QUARTER
ticks['year'] = ticks.YEAR_QUARTER.str[:4].astype(int)
ticks['quarter'] = ticks.YEAR_QUARTER.str[5:].astype(int)
ticks = ticks.reset_index(drop=True)
print(ticks)
pos = pos.merge(ticks, how='left', on='YEAR_QUARTER')

# Plot outcomes that were followed by investigation
fig, ax = plt.subplots(2, 1, figsize=(14, 8), tight_layout=True)
for i, t in enumerate(fit_thr):
    s = pos.loc[pos.fit_thr == t]
    label = str(t) + ' µg/g'
    color = 'C' + str(i)
    ax[0].plot(s['x'], s['num_positives'], label=label, color=color)
    ax[0].scatter(s['x'], s['num_positives'], s=8, color=color)
    ax[1].plot(s['x'], s['percent_positives'], label=label, color=color)
    ax[1].scatter(s['x'], s['percent_positives'], s=8, color=color)
    ax[1].fill_between(s['x'], s.perc_low, s.perc_high, facecolor=color, alpha=0.25)

ax[0].set_xticks(ticks['x'])
ax[0].set_xticklabels(ticks['xticklabel'])
ax[0].legend(frameon=False, bbox_to_anchor=(1.01, 1), fontsize=10, title='FIT threshold', alignment='left')
ax[0].grid(which='major', alpha=0.5)
ax[0].set_xlabel('Test kit logged date (year-quarter)', fontsize=10)
ax[0].set_ylabel('Number of episodes', fontsize=10)
ax[0].set_title('A. Number of episodes with a positive test')
ax[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax[1].set_xticks(ticks['x'])
ax[1].set_xticklabels(ticks['xticklabel'])
ax[1].legend(frameon=False, bbox_to_anchor=(1.01, 1), fontsize=10, title='FIT threshold', alignment='left')
ax[1].grid(which='major', alpha=0.5)
ax[1].set_xlabel('Test kit logged date (year-quarter)', fontsize=10)
ax[1].set_ylabel('Percent of episodes', fontsize=10)
ax[1].set_title('B. Percent of episodes with a positive test')

plt.savefig(out_path_sub / 'positivity_by_quarter.png', dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / 'positivity_by_quarter.svg', bbox_inches='tight')
#plt.show()
plt.close()

# Save table
p = pos.drop(labels=['x', 'xticklabel', 'YEAR_QUARTER'], axis=1)
p = p[['year', 'quarter'] + [c for c in p.columns if c not in ['year', 'quarter']]]
p['perc_reformat'] = p.percent_positives.astype(str) + ' (' + p.perc_low.astype(str) + ', ' + p.perc_high.astype(str) + ')'
p.columns = ['Test kit logged year', 'Test kit logged quarter', 
             'FIT threshold (µg/g)', 'Num episodes', 'Num positives', 'Percent positive',
             'Percent positive (ci low)', 'Percent positive (ci upp)', 'Percent positive (ci low, ci upp)']
out_name = 'positivity_by_quarter.csv'
p.to_csv(out_path_sub / out_name, index=False)

del df_fit_time
#endregion


# ---- 6. FIT positivity by age, deprivation, sex ----
#region

# Compute positivity rate by age, deprivation and sex
cols = ['AGE_GROUP', 'IMD_QUINTILE', 'IMD_DECILE', 'SUBJECT_GENDER']
pos = pd.DataFrame()
for c in cols:
    print(c)
    unique_values = df_fit[c].dropna().drop_duplicates().sort_values()
    for v in unique_values:
        print(v)
        s = df_fit.loc[df_fit[c] == v]
        p = positivity(s, thr=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        p['col'] = c
        p['value'] = v
        pos = pd.concat(objs=[pos, p], axis=0)
pos = pos[['col', 'value'] + [c for c in pos.columns if c not in ['col', 'value']]]

pd.set_option('display.min_rows', 200, 'display.max_rows', 200)
pos

# Plot
cols_plot = {'AGE_GROUP': 'Age group', 
             'IMD_QUINTILE': 'IMD quintile', 
             'SUBJECT_GENDER': 'Gender'}

cols_ystep = {'AGE_GROUP': 200000, 
              'IMD_QUINTILE': 100000, 
              'SUBJECT_GENDER': 200000}

fig, ax = plt.subplots(2, 3, figsize=(16, 9), sharey=False, tight_layout=True)

for i, (col, title) in enumerate(cols_plot.items()):
    print(col)

    pos_sub = pos.loc[pos.col == col]

    values = pos_sub.value.drop_duplicates()
    for j, v in enumerate(values):
        color = 'C' + str(j)
        pos_value = pos_sub.loc[pos_sub.value == v]
        plot_positivity(pos_value, ax[0, i], label=v, plot_percent=False, color=color, pointsize=18)  
        plot_positivity(pos_value, ax[1, i], label=v, plot_percent=True, color=color, pointsize=18)  

    if not title.startswith('IMD'):
        title_panel = title.lower()
    else:
        title_panel = title
    ax[0, i].legend(frameon=False, title=title, alignment='left')
    ax[0, i].set(title = 'Number of positives by ' + title_panel)
    ax[0, i].grid(which='major', alpha=0.5)

    ax[1, i].legend(frameon=False, title=title, alignment='left')
    ax[1, i].set(title = 'Percent of positives by ' + title_panel)
    ax[1, i].grid(which='major', alpha=0.5)

    pmax = pos_sub.percent_positives.max()
    nmax = pos_sub.num_positives.max()
    
    ystep = cols_ystep[col]
    ax[0, i].set_yticks(np.arange(0, pos_sub.num_positives.max() + ystep, ystep))
    ax[0, i].set_ylim(0 - nmax * 0.1, nmax + nmax * 0.1)
    ax[1, i].set_yticks(np.arange(0, pos_sub.percent_positives.max() + 2, 2))
    ax[1, i].set_ylim(0 - pmax * 0.1, pmax + pmax * 0.1)

ax = ax.flatten()
letters = ['A', 'B', 'C', 'D', 'E', 'F']
for i in range(len(ax)):
    ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

out_name = 'positivity_by_demographics.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path_sub / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()

# Save as tables
cols_table = {'AGE_GROUP': 'Age group', 
              'IMD_QUINTILE': 'IMD quintile', 
              'IMD_DECILE': 'IMD decile',
              'SUBJECT_GENDER': 'Gender'}
for col, title in cols_table.items():
    print(col)
    p = pos.loc[pos.col == col].copy()
    p['perc_reformat'] = p.percent_positives.astype(str) + ' (' + p.perc_low.astype(str) + ', ' + p.perc_high.astype(str) + ')'
    p = p.drop(labels=['col'], axis=1)
    p.columns = [title, 'FIT threshold (µg/g)', 'Num episodes', 'Num positives', 'Percent positive',
                 'Percent positive (ci low)', 'Percent positive (ci upp)', 'Percent positive (ci low, ci upp)']
    out_name = 'positivity_by_' + col.lower() + '.csv'
    p.to_csv(out_path_sub / out_name, index=False)

#endregion


# ---- 7. What FIT threshold would give equivalent number of referrals for males and females? ----
#region

# First, look again at the percentage of positive tests for males and females 
fit_thr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
pos = pd.DataFrame()
unique_values = df_fit.SUBJECT_GENDER.dropna().drop_duplicates().sort_values()
for v in unique_values:
    print(v)
    s = df_fit.loc[df_fit.SUBJECT_GENDER == v]
    p = positivity(s, thr=fit_thr)
    p['gender'] = v
    pos = pd.concat(objs=[pos, p], axis=0)

pos = pos[['fit_thr', 'gender'] + [c for c in pos.columns if c not in ['fit_thr', 'gender']]]
pos = pos.sort_values(by=['fit_thr', 'gender'])

# Note: females have a lower FIT positivity rate at all thresholds
# This implies that to achieve equal rate of positivity while not reducing referrals,
# one could decrease FIT threshold for females.
m = pos.loc[pos.gender == 'Male']
f = pos.loc[pos.gender == 'Female']
assert all(m.fit_thr == f.fit_thr)
delta = m.percent_positives - f.percent_positives
assert all(delta > 0)

# FIT test results are given with 0.2 ug/g precision
u = df_fit.ANALYSER_READING_USED.drop_duplicates().sort_values()
for v in u:
    print(v)
decimal = u % 1
decimal = decimal.sort_values().round(1).unique()
for d in decimal:
    print(d)

# Find a lower threshold for females that yields equivalent percentage of positivity, using a 0.2 ug/g increment
# There's probably a much more efficient algorithm - but as this code runs fast, it is easy to just increment the threshold
df_female = df_fit.loc[df_fit.SUBJECT_GENDER == 'Female']
df_male = df_fit.loc[df_fit.SUBJECT_GENDER == 'Male']

ntot_male = df_male.shape[0]
ntot_female = df_female.shape[0]

increment = np.array(0.2)
pos_new = pd.DataFrame()
for t in fit_thr:
    print('Threshold', t)
    npos_male = (df_male.ANALYSER_READING_USED >= t).sum().item()
    npos_female = (df_female.ANALYSER_READING_USED >= t).sum().item()
    ppos_male = npos_male / ntot_male * 100
    ppos_female = npos_female / ntot_female * 100

    t_new = t
    #while num_pos_male > num_pos_female:
    while ppos_male > ppos_female:
        print('...lowered to', t_new)
        if t_new > increment:
            t_new = np.round(t_new - increment, 1)
        else:
            print('The next increment will decrease thr below zero')
            break
        npos_female = (df_female.ANALYSER_READING_USED >= t_new).sum().item()
        ppos_female = npos_female / ntot_female * 100

    ci_low, ci_high = proportion_confint(count=npos_female, nobs=ntot_female, method='wilson')
    ci_low, ci_high = ci_low * 100, ci_high * 100
    p = pd.DataFrame([[t, t_new, 'Female', ntot_female, npos_female, ppos_female, ci_low, ci_high]])
    p.columns=['fit_thr', 'fit_thr_new', 'gender', 'num_episodes', 'num_positives', 'percent_positives', 'perc_low', 'perc_high']
    p[['percent_positives', 'perc_low', 'perc_high']] = p[['percent_positives', 'perc_low', 'perc_high']].round(3)
    pos_new = pd.concat(objs=[pos_new, p], axis=0)
pos_new = pos_new.reset_index(drop=True)

# Add positivity rates at new thresholds to the existing threshold table
pos2 = pos.copy()
pos2 = pd.concat([pos_new, pos2], axis=0)
pos2.loc[pos2.fit_thr_new.isna(), 'fit_thr_new'] = pos2.loc[pos2.fit_thr_new.isna(), 'fit_thr']
pos2 = pos2.sort_values(by=['fit_thr', 'gender', 'fit_thr_new'], ascending=[True, False, True])

# Create a simpler version of the threshold table
pos_male = pos.loc[pos.gender == 'Male', ['fit_thr',  'num_positives', 'percent_positives']].reset_index(drop=True)
pos_female = pos_new[['fit_thr_new', 'num_positives', 'percent_positives']].reset_index(drop=True)
pos_female.columns = ['fit_thr_new', 'num_positives_new', 'percent_positives_new']
assert (pos_male.fit_thr.to_numpy() == pos_new.fit_thr.to_numpy()).all()
pos_compare = pd.concat(objs=[pos_male, pos_female], axis=1)

# Plot equivalent thresholds  for males and females
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(pos_compare.fit_thr, pos_compare.fit_thr_new)
ax.scatter(pos_compare.fit_thr, pos_compare.fit_thr_new)
ax.set(xlabel='FIT threshold male (µg/g)', ylabel='FIT threshold female (µg/g)')
ax.grid(which='major', alpha=0.6)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])

x = pos_compare.fit_thr
y = pos_compare.fit_thr_new
value = pos_compare.percent_positives.round(1)
value_new = pos_compare.percent_positives_new.round(1)
delta = value - value_new
for i in range(len(x)): 
    plt.annotate(value[i], (x[i] - 4, y[i] + 4), color='C0') 

plt.savefig(out_path_sub / 'positivity_gender_equivalent.png', dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / 'positivity_gender_equivalent.svg', bbox_inches='tight')
plt.close()

# Reformat the tables and save
pos_compare.columns = ['FIT threshold male (µg/g)', 'Num positives male', 'Percent positives male',
                       'FIT threshold female (µg/g)', 'Num positives female', 'Percent positives female']
pos_compare = pos_compare.iloc[:, [0, 3, 1, 4, 2, 5]]
print(pos_compare)
pos_compare.to_csv(out_path_sub / 'positivity_gender_equivalent_simple.csv', index=False)

pos2['perc_reformat'] = pos2.percent_positives.astype(str) + ' (' + pos2.perc_low.astype(str) + ', ' + pos2.perc_high.astype(str) + ')'
pos2.columns = ['FIT threshold (µg/g)', 'FIT threshold new (µg/g)', 'Gender', 'Num episodes', 'Num positives', 'Percent positive',
                'Percent positive (ci low)', 'Percent positive (ci upp)', 'Percent positive (ci low, ci upp)']
pos2 = pos2.reset_index(drop=True)
pos2.to_csv(out_path_sub / 'positivity_gender_equivalent.csv', index=False)
print(pos2)

#for i in range(100):
#    plt.close()
#endregion


# ---- 8. f-HB concentration for cancers and precancerous lesions ----
#region
outcomes_include = ['Colorectal cancer', 'High-risk findings or LNPCP', 'Premalignant polyp(s)']
assert all(o in outcomes for o in outcomes_include)

# Include episodes with most relevant cancer or polyp outcomes. As expected, min reading is 120
dfsub = df_fit.loc[df_fit.OUTCOME.isin(outcomes_include)]
print(dfsub.ANALYSER_READING_USED.min())

# About 33% of these episodes have analyser reading above dilution range
# Most readings are marked as 201 (about 59000), the remaining as 500001 (about 1200)
# Apparently, they seem to be marked as 1 unit above what is the maximum possible value
# There's also about 34% of episodes where the reading is above 201 but below 500001.
print(dfsub.ANALYSER_ERROR_CODE.unique())
mask = dfsub.ANALYSER_ERROR_CODE.isin([4, 5])
print(mask.mean())
dfsub.loc[mask].ANALYSER_READING_USED.value_counts()
(dfsub.ANALYSER_READING_USED > 201).mean()
((dfsub.ANALYSER_READING_USED > 201) & (dfsub.ANALYSER_READING_USED < 50001)).mean()
dfsub.groupby('ANALYSER_ERROR_CODE').ANALYSER_READING_USED.value_counts()

# How to include episodes above dilution range? Option one: categorise >=200 as 200+
fit = dfsub.ANALYSER_READING_USED
fit_max = fit.max() + 1
bins = np.array([120, 140, 160, 180, 200, fit_max])
fit_group = pd.cut(fit, bins=bins, right=False)
fit_group.value_counts(sort=False)

fit_group = fit_group.astype(str)
fit_group = fit_group.replace({'[120.0, 140.0)': '120-139',
                               '[140.0, 160.0)': '140-159',
                               '[160.0, 180.0)': '160-179',
                               '[180.0, 200.0)': '180-199',
                               '[200.0, 50002.0)': '200+'})
fit_group.value_counts(sort=False)
fit_group.value_counts(sort=False, normalize=True)
dfsub['fit_group'] = fit_group

# Compute num and percent of each fit category by outcome 
g = grouped_count(dfsub, group_col='OUTCOME', outcome_col='fit_group')
g = g.sort_values(by=['OUTCOME', 'fit_group'], ascending=[True, True])
g['err_low'] = g.perc - g.perc_low
g['err_high'] = g.perc_high - g.perc


# Create bar graph 
outcome_plot = fit_group.sort_values().unique()
bar_width = 0.15
group_col = 'OUTCOME'

fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=False, tight_layout=True)

ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
ticks['x'] = np.arange(ticks.shape[0])
ticks['xticklabel'] = ticks[group_col]
ticks.loc[ticks.OUTCOME == 'High-risk findings or LNPCP', 'xticklabel'] =  'High-risk findings\nor LNPCP'
g = g.merge(ticks, how='left')

for j, outcome in enumerate(outcome_plot):
    color = 'C' + str(j)
    gsub = g.loc[g.fit_group == outcome]

    yerr = gsub[['err_low', 'err_high']].transpose().to_numpy()

    x = gsub['x'] + bar_width * j
    ax[0].bar(x, gsub['count'], width=bar_width, label=outcome, color=color)
    ax[1].bar(x, gsub['perc'], width=bar_width, label=outcome, color=color)
    ax[1].errorbar(x=x, y=gsub['perc'], yerr=yerr, fmt='none', ecolor='black')

xmax = ticks['x'].max()
ncat = len(outcome_plot)
xmid = (ncat - 1)* bar_width / 2

ax[0].set(title = 'Number of outcomes by FIT value category')
ax[0].grid(which='major', alpha=0.5)
ax[0].set_xticks(ticks['x'] + xmid)
ax[0].set_xticklabels(ticks['xticklabel'])
#ax[0].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
ax[0].set_ylabel('Number of episodes')
ax[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax[0].set_xlabel('Outcome')

ax[1].set(title = 'Percent of outcomes by FIT value category')
ax[1].grid(which='major', alpha=0.5)
ax[1].set_xticks(ticks['x'] + xmid)
ax[1].set_xticklabels(ticks['xticklabel'])
#ax[1].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
ax[1].set_ylabel('Percent of episodes')
ax[1].set_xlabel('Outcome')

ystep = 5000
yticks0 = np.arange(0, g['count'].max() + ystep, ystep)
yticks1 = np.arange(0, g.perc.max() + 10, 10)
y0_max = yticks0.max()
y1_max = yticks1.max()

ax[0].set_yticks(yticks0)
ax[0].set_ylim(0, y0_max + y0_max * 0.1)
ax[1].set_yticks(yticks1)
ax[1].set_ylim(0, y1_max + y1_max * 0.1)

ax[1].legend(frameon=False, bbox_to_anchor=(1.3, 1), title='FIT value (µg/g)', alignment='left')

ax = ax.flatten()
letters = ['A', 'B']
for i in range(len(ax)):
    ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

out_name = 'fit_value_barplot_by_outcomes.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path_sub / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()


# Create line graph 
outcome_plot = fit_group.sort_values().unique()
pointsize = 24
group_col = 'OUTCOME'

fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=False, tight_layout=True)

ticks = g[[group_col]].drop_duplicates().sort_values(by=group_col)
ticks['x'] = np.arange(ticks.shape[0])
ticks['xticklabel'] = ticks[group_col]
ticks.loc[ticks.OUTCOME == 'High-risk findings or LNPCP', 'xticklabel'] =  'High-risk findings\nor LNPCP'
g = g.merge(ticks, how='left')

for j, outcome in enumerate(outcome_plot):
    color = 'C' + str(j)
    s = g.loc[g.fit_group == outcome]
    ax[0].plot(s['x'], s['count'], label=outcome, color=color)
    ax[0].scatter(s['x'], s['count'], s=pointsize, color=color)
    ax[1].plot(s['x'], s['perc'], label=outcome, color=color)
    ax[1].scatter(s['x'], s['perc'], s=pointsize, color=color)
    ax[1].fill_between(s['x'], s.perc_low, s.perc_high, facecolor=color, alpha=0.25)

xmax = ticks['x'].max()

ax[0].set(title = 'Number of outcomes by FIT value category')
ax[0].grid(which='major', alpha=0.5)
ax[0].set_xticks(ticks['x'])
ax[0].set_xticklabels(ticks['xticklabel'])
ax[0].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
ax[0].set_ylabel('Number of episodes')
ax[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax[0].set_xlabel('Outcome')

ax[1].set(title = 'Percent of outcomes by FIT value category')
ax[1].grid(which='major', alpha=0.5)
ax[1].set_xticks(ticks['x'])
ax[1].set_xticklabels(ticks['xticklabel'])
ax[1].set_xlim(0 - 0.1 * xmax, xmax + xmax * 0.1)
ax[1].set_ylabel('Percent of episodes')
ax[1].set_xlabel('Outcome')

ystep = 5000
yticks0 = np.arange(0, g['count'].max() + ystep, ystep)
yticks1 = np.arange(0, g.perc.max() + 10, 10)
y0_max = yticks0.max()
y1_max = yticks1.max()

ax[0].set_yticks(yticks0)
ax[0].set_ylim(0 - y0_max * 0.1, y0_max + y0_max * 0.1)
ax[1].set_yticks(yticks1)
ax[1].set_ylim(0 - y1_max * 0.1, y1_max + y1_max * 0.1)

ax[1].legend(frameon=False, bbox_to_anchor=(1.3, 1), title='FIT value (µg/g)', alignment='left')

ax = ax.flatten()
letters = ['A', 'B']
for i in range(len(ax)):
    ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

out_name = 'fit_value_lineplot_by_outcomes.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path_sub / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()


# Save table to disk
g = g[['OUTCOME', 'fit_group', 'ntot', 'count', 'perc', 'perc_low', 'perc_high', 'perc_reformat']]
g.columns = ['Outcome', 'FIT value (µg/g)', 'Total num episodes', 'Num episodes', 'Percent episodes',
             'Percent episodes (ci low)', 'Percent episodes (ci upp)', 'Percent episodes (ci low, ci upp)']
g.to_csv(out_path_sub / 'fit_value_by_outcomes.csv', index=False)


# Compute percentiles (BUT -- about 30% have above dilution range with value 200 which does not enable assessing percentiles well)
# Do not save this atm
percentiles = False
if percentiles:
    q = dfsub.groupby('OUTCOME').ANALYSER_READING_USED.quantile([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    q = q.reset_index()
    q.columns = ['outcome', 'percentile', 'fit_val']

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, o in enumerate(q.outcome.unique()):
        qsub = q.loc[q.outcome == o]
        x = qsub.percentile
        y = np.log10(qsub.fit_val + 1)
        ax.scatter(qsub.percentile, y, color='C' + str(i))
        ax.plot(qsub.percentile, y, color='C' + str(i))

    ax.set_xticks(np.arange(0, 1.1, 0.1))

    yticklabels = np.array([10, 120, 1000, 10000, 50000])
    yticks = np.log10(yticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # About 30% of all FIT values for these outcomes are exactly 201 (one value corresponding to above dilution range error)
    fig, ax = plt.subplots(figsize=(8, 6))

    for o in outcomes_include:
        print(o)
        s = dfsub.loc[dfsub.OUTCOME == o]
        s['fit_log10'] = np.log10(s.ANALYSER_READING_USED)
        sns.kdeplot(data=s, x='fit_log10', clip=(np.log10(120+1), np.log10(50000+1)), ax=ax)

    xticklabels = np.array([0, 10, 20, 40, 60, 80, 100, 120, 500, 5000, 50000])
    xticklabels = np.array([120, 1000, 10000, 50000])
    xticks = np.log10(xticklabels + 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.grid(which='major', alpha=0.5)

#endregion


# ---- 9. Overall descriptive statistics for the dataset ----
#region

# Categorise FIT values and reformat
fit = df.ANALYSER_READING_USED
fit_max = fit.max() + 1
bins = np.array([0, 10, 80, 120, 140, 160, 180, 200, fit_max])
fit_group = pd.cut(fit, bins=bins, right=False)
print(fit_group.unique())

fit_group = fit_group.astype(str)
fit_group_names = {'[0.0, 10.0)': '0-9',
                    '[10.0, 80.0)': '10-79',
                    '[80.0, 120.0)': '80-119',
                    '[120.0, 140.0)': '120-140',
                    '[140.0, 160.0)': '140-159',
                    '[160.0, 180.0)': '160-189',
                    '[180.0, 200.0)': '180-199',
                    '[200.0, 50002.0)': '200+'}
fit_group = fit_group.replace(fit_group_names)
fit_group.value_counts(sort=False)

df['fit_group'] = fit_group
df.loc[df.ANALYSER_READING_USED.isna(), 'fit_group'] = np.nan
print(df.fit_group.isna().sum())

# Variable that shows FIT positivity
df['fit_positive'] = df.ANALYSER_READING_USED >= 120
df.loc[df.ANALYSER_READING_USED.isna(), 'fit_positive'] = np.nan

# Result container
desc = pd.DataFrame()

# Add number of episodes and persons
num_epi = df.ANON_SUBJECT_EPIS_ID.nunique()
row = pd.DataFrame({'Characteristic': ['Number of episodes'], 'Category': [''], 'Value': [num_epi]})
desc = pd.concat(objs=[desc, row], axis=0)

num_person = df.ANON_SCREENING_SUBJECT_ID.nunique()
row = pd.DataFrame({'Characteristic': ['Number of participants'], 'Category': [''], 'Value': [num_person]})
desc = pd.concat(objs=[desc, row], axis=0)

# Add number of episodes per person
epi_count = df.groupby('ANON_SCREENING_SUBJECT_ID').size()
epi_count = epi_count.rename('num_epi').reset_index()
row = summarise_cat(epi_count, 'num_epi', 'Number of episodes per participant')
row = row.sort_values(by='Category')
desc = pd.concat(objs=[desc, row], axis=0)

# Summarise categorical columns 
cat_cols = {'SUBJECT_GENDER': 'Gender (per episode)', 
            'AGE_GROUP': 'Age group (per episode)',
            'PREVALENT_INCIDENT_STATUS': 'Prevalent incident status', 
            'EPISODE_SUBTYPE': 'Episode subtype',
            'IMD_QUINTILE': 'IMD quintile (per episode)', 
            'IMD_DECILE': 'IMD decile (per episode)',
            'OUTCOME': 'Episode result',
            'fit_group': 'FIT value (µg/g)',
            'fit_positive': 'FIT positive'
            }

outcome_order = outcomes_with_investigation + outcomes_without_investigation 

for c, name in cat_cols.items():
    print(c)
    row = summarise_cat(df, c, name)
    row.Category = row.Category.astype(str)
    if c == 'fit_group':
        order = list(fit_group_names.values()) + ['NULL']
        new_row = pd.DataFrame()
        for o in order:
            r = row.loc[row.Category == o]
            new_row = pd.concat(objs=[new_row, r], axis=0)
        row = new_row
    elif c == 'OUTCOME':
        print(0)
        #    order = [o for o in row.Category if o in outcomes_with_investigation] + \
        #            [o for o in row.Category if o in outcomes_without_investigation]
        #    new_row = pd.DataFrame()
        #    for o in order:
        #        r = row.loc[row.Category == o]
        #        new_row = pd.concat(objs=[new_row, r], axis=0)
        #    row = new_row
    else:
        row = row.sort_values(by='Category')
    if 'NULL' in row.Category.tolist():
        row0 = row.loc[row.Category != 'NULL']
        row1 = row.loc[row.Category == 'NULL']
        row = pd.concat(objs=[row0, row1], axis=0)
    desc = pd.concat(objs=[desc, row], axis=0)

desc = desc.reset_index(drop=True)

# Save
assert desc.Value.min() > 10
desc.to_csv(out_path / 'cohort.csv', index=False)


# Get additional patient level (not episode level) descriptive statistics
dfsub = df[['ANON_SCREENING_SUBJECT_ID', 'SUBJECT_GENDER', 'AGE_GROUP', 'EPISODE_START_DATE',
            'IMD_QUINTILE', 'IMD_DECILE']]
idxmin = dfsub.groupby('ANON_SCREENING_SUBJECT_ID').EPISODE_START_DATE.idxmin()
dfsub = dfsub.loc[idxmin]
assert dfsub.shape[0] == dfsub.ANON_SCREENING_SUBJECT_ID.nunique()

## Result container
desc = pd.DataFrame()

## Add num participants
num_person = df.ANON_SCREENING_SUBJECT_ID.nunique()
row = pd.DataFrame({'Characteristic': ['Number of participants'], 
                    'Category': [''], 
                    'Value': [num_person], 
                    'Percent': [100.0], 
                    'Value (percent)': [str(num_person) + ' (100.0)']})
desc = pd.concat(objs=[desc, row], axis=0)

## Add num episodes per person
epi_count = df.groupby('ANON_SCREENING_SUBJECT_ID').size()
epi_count = epi_count.rename('num_epi').reset_index()
row = summarise_cat(epi_count, 'num_epi', 'Number of episodes per participant')
row = row.sort_values(by='Category')
desc = pd.concat(objs=[desc, row], axis=0)

## Add descriptives by sex, age, deprivation
cat_cols = {'SUBJECT_GENDER': 'Gender', 
            'AGE_GROUP': 'Age group (first episode)',
            'IMD_QUINTILE': 'IMD quintile (first episode)', 
            'IMD_DECILE': 'IMD decile (first episode)'
            }

for c, name in cat_cols.items():
    print(c)
    row = summarise_cat(df, c, name)
    row.Category = row.Category.astype(str)
    row = row.sort_values(by='Category')
    if 'NULL' in row.Category.tolist():
        row0 = row.loc[row.Category != 'NULL']
        row1 = row.loc[row.Category == 'NULL']
        row = pd.concat(objs=[row0, row1], axis=0)
    desc = pd.concat(objs=[desc, row], axis=0)

assert desc.Value.min() > 10
desc.to_csv(out_path / 'cohort_patient.csv', index=False)

#endregion
