"""Some helper functions for the analysis"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
import statsmodels.api as stat


def grouped_count(df: pd.DataFrame, group_cols: list = ['age_group'], outcome_col: str = 'outcome', 
                  confint: bool = True, digits: int = 3, ci_method: str = 'wilson', 
                  include_zero_count: bool = True,
                  total_col: str = 'anon_subject_epis_id'):
    """Compute the number and percentage of outcomes (outcome_col) within each category
    of the grouping variable (group_col)"""
    
    # Within each category of the grouping variable, count the number of different outcomes
    g = df.groupby(group_cols)[outcome_col].value_counts(sort=False).reset_index()

    # If an outcome does not exist within a category of the grouping variable, add zero count
    if include_zero_count and g.groupby(group_cols).size().nunique() > 1:  
        # the g.groupby(group_col).size().nunique() > 1 is True when all outcome categories (e.g. CRC, advanced adenoma, ...)
        # are not present for all categories of the grouping variable (meaning that some outcome values did not occur at all for some categories)
        g = g.reset_index().pivot(index=group_cols, columns=[outcome_col], values='count')
        g = g.fillna(0)
        g = g.reset_index().melt(id_vars=group_cols, value_vars = g.columns, var_name=outcome_col, value_name='count')

    # Compute total number of episodes within each level of the grouping variables
    # and then the percentage of each outcome.
    #  Total number of episodes is computed by counting the number of unique episodes within each level
    #  because the input dataframe may have multiple rows per episode (e.g. two different outcomes in same episode)
    ntot = df.groupby(group_cols)[total_col].nunique().rename('ntot')
    #ntot = g.groupby(group_cols)['count'].sum().rename('ntot')
    g = g.merge(ntot.reset_index(), how='left')
    g['perc'] = g['count'] / g.ntot * 100  
    g.perc = g.perc.round(digits)
    g = g[group_cols + [outcome_col, 'ntot', 'count', 'perc']]

    # Add confidence intervals for percentages
    if confint:
        for c in ['perc_low', 'perc_high']:
            g[c] = np.nan
        for i, row in g.iterrows():
            ci_low, ci_high = proportion_confint(count=row['count'], nobs=row['ntot'], method=ci_method)
            g.loc[i, 'perc_low'] = np.round(ci_low * 100, digits)
            g.loc[i, 'perc_high'] = np.round(ci_high * 100, digits)
        g['perc_reformat'] = g.perc.astype(str) + ' (' + g.perc_low.astype(str) + ', ' + g.perc_high.astype(str) + ')'
            
    g = g.sort_values(by=group_cols + [outcome_col])

    return g


def hide_low_counts(g: pd.DataFrame):
    """Hide counts less than 10.
    Assumes that g is the output of grouped_count()"""
    test = g['count'] < 10
    if test.any():
        print('Hiding count <10')
        cols = ['count', 'perc', 'perc_low', 'perc_high']
        for c in cols:
            g[c] = g[c].astype('object')
        g.loc[test, 'count'] = '<10'
        perc_new = np.round(11 / g.loc[test, 'ntot'] * 100, 4)
        g.loc[test, 'perc'] = '<' + perc_new.astype(str)
        g.loc[test, ['perc_low', 'perc_high', 'perc_reformat']] = ''
        print(g.loc[test])
    return g


def summarise_cat(df, col, name, digits=2, sort=True):
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
    out['reformat'] = counts.astype(str) + ' (' + perc.astype(str) + ')'
    out = out.reset_index()
    out.columns = ['Category', 'Value', 'Percent', 'Value (percent)']
    out['Characteristic'] = name
    out = out[['Characteristic', 'Category', 'Value', 'Percent', 'Value (percent)']]
    return out


def summarise_con(df, col, name, digits=2, compact=False):
    """Summarise continuous data"""
    s = df[col]

    # Compute statistics
    stat = s.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])

    # Reformat
    stat = stat.loc[~stat.index.isin(['count'])]
    stat = stat.round(digits)

    if compact:
        mean_std = stat['mean'].astype(str) + ' (' + stat['std'].astype(str) + ')'
        med_iqr = stat['50%'].astype(str) + ' (' + stat['25%'].astype(str) + ', ' + stat['75%'].astype(str) + ')'
        min_and_max = stat['min'].astype(str) + ', ' + stat['max'].astype(str) 
        p1_and_p99 = stat['1%'].astype(str) + ', ' + stat['99%'].astype(str) 

        out = {'Category': ['Mean (std)', 'Median (25th, 75th)', '1st and 99th percentiles', 'Min and max'],
            'Value': [mean_std, med_iqr, p1_and_p99, min_and_max]}
        out = pd.DataFrame(out)
    else:
        out = stat.reset_index()
        out.columns = ['Category', 'Value']
        out.Category = out.Category.replace({'mean': 'Mean', 'std': 'Std', 'min': 'Min', 'max': 'Max', '50%': 'Median'})
    
    out['Characteristic'] = name
    out = out[['Characteristic', 'Category', 'Value']]
        
    # Add missing value count, if any
    if s.isna().any():
        nmis = s.isna().sum()
        pmis = nmis / df.shape[0]
        mis = nmis.astype(str) + ' (' + pmis.round(digits).astype(str) + ')'
        row = pd.DataFrame({'Characteristic': [name], 'Category': ['NULL'], 'Value': [mis]})
        out = pd.concat(objs=[out, row], axis=0)

    return out


def xtest(data: pd.DataFrame, group_col: str, ref_category: str,
          outcome_col: str = 'outcome',
          total_col: str = 'ntot', count_col: str = 'count'):
    
    categories = data[group_col].unique()
    nonref_categories = [c for c in categories if c != ref_category]

    res = pd.DataFrame()
    for i, outcome in enumerate(data[outcome_col].unique()):
        print(outcome)

        for cat in nonref_categories:
            print('Comparing', ref_category, 'vs', cat)
            dsub = data.loc[(data[outcome_col] == outcome) & (data[group_col].isin([ref_category, cat]))]

            dsub = dsub.set_index(group_col)
            assert dsub.shape[0] == 2

            n_total = dsub[total_col]
            n_outcome = dsub[count_col]
            p = n_outcome / n_total * 100
            xstat, pval, (obs, exp) = stat.stats.proportions_chisquare(count = n_outcome, nobs = n_total)

            row = {'group_col': group_col,
                   'outcome': outcome, 
                   'ref': ref_category,
                   'cat': cat,
                   'pref': p[ref_category].item(), 
                   'pcat': p[cat].item(), 
                   'xstat': xstat.item(), 'pval': pval.item()}
            row = pd.DataFrame(row, index=[i])
            res = pd.concat(objs=[row, res], axis=0)
            print("   perc_0 {:.1f}, perc_1 {:.1f}, X {:.1f}, pval {}".format(p.iloc[0], p.iloc[1], xstat, pval))

    res[['pref', 'pcat', 'xstat']] = res[['pref', 'pcat', 'xstat']].round(1)

    res['pval_cat'] = res.pval
    values = [0.05, 0.01, 0.001]
    for v in values:
        res.loc[res.pval < v, 'pval_cat'] = '< ' + str(v)
    return res