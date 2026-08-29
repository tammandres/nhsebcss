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



# ---- Outcomes by age, deprivation, sex (one variable at a time) ----
# Note: not computing outcomes by prev/inc status atm, as this is confounded by age.
#region
print('\nSUMMARISING OUTCOMES BY DEMOGRAPHICS ONE VARIABLE AT A TIME...')

res = pd.read_csv(r'/Users/andres/Library/CloudStorage/OneDrive-Nexus365/PostDoc2024/BowelScreeningData/airlock/20260511_nhsebcsp/outcomes-investigated-by-demographics.csv')
res_pos = pd.read_csv(r'/Users/andres/Library/CloudStorage/OneDrive-Nexus365/PostDoc2024/BowelScreeningData/airlock/20260511_nhsebcsp/outcomes-pos-by-demographics.csv')
out_path = Path('/Users/andres/Library/CloudStorage/OneDrive-Nexus365/PostDoc2024/BowelScreeningData/second_draft/figs')      

                  
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

# Outcome labels for plotting
outcome_labels = {
    'Colorectal cancer': 'Colorectal\ncancer', 
    'Advanced premalignant polyp': 'Advanced\ncolorectal polyp', 
    'Non-advanced premalignant polyp': 'Non-advanced\ncolorectal polyp', 
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

out_name = 'fig4_outcomes-by-demographics_labelfix.png'
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

out_name = 'suppl_fig5_no-investigation-by-demographics_labelfix.png'
out_name_svg = out_name[:-4] + '.svg'
plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
plt.savefig(out_path / out_name_svg, dpi=300, bbox_inches='tight')
plt.close()
#endregion