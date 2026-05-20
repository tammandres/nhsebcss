"""Analyse FIT positivity. Assumes that dataprep.py has been run
To-do: could fit outcome ~ fit_val model using mgcv, then plot in python"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from statsmodels.stats.proportion import proportion_confint
import statsmodels.api as stat


# Paths
data_path_clean = Path("C:/Users/andres.Tamm/Desktop/bcss_clean_data")

out_path = Path("Z:/andres/nhsebcss/results/primary")
out_path.mkdir(exist_ok=True, parents=True)

out_path_sub = out_path
out_path_sub.mkdir(exist_ok=True, parents=True)

# Method for proportion confidence intervals
ci_method = 'wilson'

# Helper functions
def positivity(df, thr=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
               digits=3, one_row_per_episode=True):
    """Compute FIT positivity at prespecified thresholds"""
    pos = pd.DataFrame()

    if not one_row_per_episode:
        df_use = df[['anon_subject_epis_id', 'analyser_reading_used']]
        df_use = df_use.drop_duplicates(subset=['anon_subject_epis_id'])
    else:
        df_use = df

    for t in thr:
        mask = df_use.analyser_reading_used >= t
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

# Include missing IMD values? Not atm - small count. Most missing values occur in FIT negative (25,433 of 26,142)
# Also, all IMD categories have at least 2,526,734 observations, which is much larger than the 26,142 missing values in total
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
add_nonh_oucomes_as_rows = False
if add_nonh_oucomes_as_rows:
    rows_to_add = pd.DataFrame()
    for col in ['advanced_polyp', 'nonadvanced_polyp']:
        print(col)
        tmp = df.loc[df[col] == 1].copy()
        if col == 'advanced_polyp':
            result = 'Advanced premalignant polyp'
        else:
            result = 'Non-advanced premalignant polyp'
        tmp.outcome = result + ' (nonh)'
        rows_to_add = pd.concat(objs=[rows_to_add, tmp], axis=0)
    df = pd.concat(objs=[df, rows_to_add], axis=0).drop_duplicates()
    outcomes_nonh = rows_to_add.outcome.unique().tolist()

    outcomes += outcomes_nonh
    outcomes_fit_pos += outcomes_nonh
    outcomes_with_investigation += outcomes_nonh
else:
    assert df.anon_subject_epis_id.nunique() == df.shape[0]

# Subset episodes with adequate FIT participation
df_fit = df.loc[df.outcome != 'No FIT result']
assert not df_fit.analyser_reading_used.isna().any()

#endregion


# ---- FIT positivity overall ----
#region
print('\nSUMMARISING FIT POSITIVITY OVERALL...')

# Compute positivity
pos = positivity(df_fit)
assert (pos.num_positives >= 10).all()

# Plot positivity
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_positivity(pos, ax[0], plot_percent=False)
plot_positivity(pos, ax[1], plot_percent=True)
ax[0].set_title('A. Number of positives')
ax[1].set_title('B. Percent of positives')
ax[0].set_yticks(np.arange(0, pos.num_positives.max() + 200000, 200000))
ax[1].set_yticks(np.arange(0, pos.percent_positives.max() + 1, 1))

out_name = 'fig5_positivity.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path_sub / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()

# Save table
num_pos_120 = pos.loc[pos.fit_thr == 120, 'num_positives'].item()
pos['perc_increase'] = (pos.num_positives - num_pos_120) / num_pos_120 * 100
cols = ['percent_positives', 'perc_low', 'perc_high', 'perc_increase']
pos[cols] = pos[cols].round(2)
pos.columns = ['FIT threshold (µg/g)', 'Num episodes', 'Num positives', 'Percent positive',
               'Percent positive (ci low)', 'Percent positive (ci upp)',
               'Percent increase in the number of FIT positive episodes compared to the 120 µg/g threshold']

pos.to_csv(out_path_sub / 'table3_positivity.csv', index=False)

#endregion


# ---- FIT positivity by age, deprivation, sex ----
#region
print('\nSUMMARISING FIT POSITIVITY BY DEMOGRAPHICS...')

# Compute positivity rate by age, deprivation and sex
cols = ['age_group_screen2', 'imd_quintile', 'subject_gender']
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
assert (pos.num_positives >= 10).all()

pd.set_option('display.min_rows', 200, 'display.max_rows', 200)
pos

# Plot
cols_plot = {'age_group_screen2': 'Age group', 
             'imd_quintile': 'IMD quintile', 
             'subject_gender': 'Sex'}

cols_ystep = {'age_group_screen2': 200000, 
              'imd_quintile': 100000, 
              'subject_gender': 200000}

fig, ax = plt.subplots(2, 3, figsize=(14, 8.5), sharey=False, tight_layout=True)

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
    ax[0, i].set(title = 'Number of episodes by ' + title_panel)
    ax[0, i].grid(which='major', alpha=0.5)

    ax[1, i].legend(frameon=False, title=title, alignment='left')
    ax[1, i].set(title = 'Percent of episodes by ' + title_panel)
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

out_name = 'fig6_positivity-by-demographics.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path_sub / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()

# Save as table
p = pos
assert not any(p.num_positives < 10)
p['perc_reformat'] = p.percent_positives.astype(str) + ' (' + p.perc_low.astype(str) + ', ' + p.perc_high.astype(str) + ')'
p.to_csv(out_path_sub / 'positivity-by-demographics.csv', index=False)
#endregion


# ---- FIT positivity by age, deprivation, sex - stratified by screening history ----
#region

# Compute positivity rate by age, deprivation and sex
cols = ['age_group_screen2', 'imd_quintile', 'subject_gender']
pos = pd.DataFrame()
for prevalent_incident_status in ['Prevalent', 'Incident']:
    cols = ['age_group_screen2', 'imd_quintile', 'subject_gender']
    df_fit_sub = df_fit.loc[df_fit.prevalent_incident_status == prevalent_incident_status]
    for c in cols:
        print(c)
        unique_values = df_fit_sub[c].dropna().drop_duplicates().sort_values()
        for v in unique_values:
            print(v)
            s = df_fit_sub.loc[df_fit_sub[c] == v]
            p = positivity(s, thr=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
            p['prevalent_incident_status'] = prevalent_incident_status
            p['col'] = c
            p['value'] = v
            pos = pd.concat(objs=[pos, p], axis=0)
pos = pos[['col', 'value'] + [c for c in pos.columns if c not in ['col', 'value']]]
assert (pos.num_positives >= 10).all()

pd.set_option('display.min_rows', 200, 'display.max_rows', 200)
pos


# Plot
cols_plot = {'age_group_screen2': 'Age group', 
             'imd_quintile': 'IMD quintile', 
             'subject_gender': 'Sex'}

cols_ystep = {'age_group_screen2': 200000, 
              'imd_quintile': 100000, 
              'subject_gender': 200000}

fig, ax = plt.subplots(2, 3, figsize=(12.8, 9), sharey=False, tight_layout=True)

for i, (col, title) in enumerate(cols_plot.items()):
    print(col)

    pos_sub = pos.loc[pos.col == col]

    for k, prevalent_incident_status in enumerate(['Prevalent', 'Incident']):
        pos_subsub = pos_sub.loc[pos_sub.prevalent_incident_status == prevalent_incident_status]
        values = pos_subsub.value.drop_duplicates()
        for j, v in enumerate(values):
            color = 'C' + str(j)
            pos_value = pos_subsub.loc[pos_subsub.value == v]
            plot_positivity(pos_value, ax[k, i], label=v, plot_percent=True, color=color, pointsize=18)  

    if not title.startswith('IMD'):
        title_panel = title.lower()
    else:
        title_panel = title
    ax[0, i].legend(frameon=False, title=title, alignment='left')
    ax[0, i].set(title = 'First-time responders:\nf-Hb above thr by ' + title_panel)
    ax[0, i].grid(which='major', alpha=0.5)

    ax[1, i].legend(frameon=False, title=title, alignment='left')
    ax[1, i].set(title = 'Previous responders:\nf-Hb above thr by ' + title_panel)
    ax[1, i].grid(which='major', alpha=0.5)

    pmax = pos_sub.percent_positives.max()
    nmax = pos_sub.num_positives.max()
    
    ystep = cols_ystep[col]
    for k in [0, 1]:
        ax[k, i].set_yticks(np.arange(0, pos_sub.percent_positives.max() + 2, 2))
        ax[k, i].set_ylim(0 - pmax * 0.1, pmax + pmax * 0.1)

ax = ax.flatten()
letters = ['A', 'B', 'C', 'D', 'E', 'F']
for i in range(len(ax)):
    ax[i].set_title(letters[i] + '. ' + ax[i].get_title())

out_name = 'suppl_fig6_positivity_by_demographics-hist.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path_sub / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path_sub / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()

p = pos
assert not any(p.num_positives < 10)
p['perc_reformat'] = p.percent_positives.astype(str) + ' (' + p.perc_low.astype(str) + ', ' + p.perc_high.astype(str) + ')'
p.to_csv(out_path_sub / 'positivity-by-demographics-hist.csv', index=False)

#endregion
