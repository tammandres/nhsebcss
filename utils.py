"""Some helper functions for the analysis"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint


def grouped_count(df: pd.DataFrame, group_col: str = 'AGE_GROUP', outcome_col: str = 'OUTCOME', 
                  confint: bool = True, digits: int = 3, ci_method: str = 'wilson', include_zero_count: bool = True):
    """Compute the number and percentage of outcomes (outcome_col) within each category
    of the grouping variable (group_col)"""

    # Within each category of the grouping variable, count the number of different outcomes
    g = df.groupby(group_col)[outcome_col].value_counts(sort=False).reset_index()

    # If an outcome does not exist within a category of the grouping variable, add zero count
    if include_zero_count and g.groupby(group_col).size().nunique() > 1:
        g = g.reset_index().pivot(index=[group_col], columns=[outcome_col], values='count')
        g = g.fillna(0)
        g = g.reset_index().melt(id_vars=[group_col], value_vars = g.columns, var_name=outcome_col, value_name='count')

    # Compute total number of episodes and percentage within each category of the grouping variable
    ntot = g.groupby([group_col])['count'].sum().rename('ntot')
    g = g.merge(ntot.reset_index(), how='left')
    g['perc'] = g['count'] / g.ntot * 100  
    g.perc = g.perc.round(digits)
    g = g[[group_col, outcome_col, 'ntot', 'count', 'perc']]

    # Add confidence interval for percentage
    if confint:
        for c in ['perc_low', 'perc_high']:
            g[c] = np.nan
        for i, row in g.iterrows():
            ci_low, ci_high = proportion_confint(count=row['count'], nobs=row['ntot'], method=ci_method)
            g.loc[i, 'perc_low'] = np.round(ci_low * 100, digits)
            g.loc[i, 'perc_high'] = np.round(ci_high * 100, digits)
        g['perc_reformat'] = g.perc.astype(str) + ' (' + g.perc_low.astype(str) + ', ' + g.perc_high.astype(str) + ')'
            
    g = g.sort_values(by=[group_col, outcome_col])

    return g


def hide_low_counts(g):
    test = g['count'] < 11
    if test.any():
        print('Hiding counts <11')
        cols = ['count', 'perc', 'perc_low', 'perc_high']
        for c in cols:
            g[c] = g[c].astype('object')
        print(g.loc[test])
        g.loc[test, 'count'] = '<11'
        perc_new = np.round(11 / g.loc[test, 'ntot'] * 100, 4)
        g.loc[test, 'perc'] = '<' + perc_new.astype(str)
        g.loc[test, ['perc_low', 'perc_high', 'perc_reformat']] = ''
        print(g.loc[test])
    return g


def proportion_confint_copy(count: int, nobs: int, alpha: float = 0.05, 
                            method: str = 'wilson', alternative: str = 'two-sided'):
    """Proportion confidence interval, copied from the statsmodels package codebase

    Taken from https://www.statsmodels.org/dev/_modules/statsmodels/stats/proportion.html#proportion_confint
    (statsmodels version 0.15.0)

    Some confidence interval options are removed in this code to simplify it.
    And instead of statsmodels array_like(), I use np.array on count and nobs.

    I am currently not using this function in analysis, but I copied it originally
    because I was not able to install statsmodels (that was fixed later)
    """

    count_a = np.array(count) 
    nobs_a = np.array(nobs)

    q_ = count_a / nobs_a

    if alternative == 'two-sided':
        alpha = alpha / 2.0

    if method == "normal":
        std_ = np.sqrt(q_ * (1 - q_) / nobs_a)
        dist = stats.norm.isf(alpha) * std_
        ci_low = q_ - dist
        ci_upp = q_ + dist
    elif method == "agresti_coull":
        crit = stats.norm.isf(alpha)
        nobs_c = nobs_a + crit**2
        q_c = (count_a + crit**2 / 2.0) / nobs_c
        std_c = np.sqrt(q_c * (1.0 - q_c) / nobs_c)
        dist = crit * std_c
        ci_low = q_c - dist
        ci_upp = q_c + dist
    elif method == "wilson":
        crit = stats.norm.isf(alpha)
        crit2 = crit**2
        denom = 1 + crit2 / nobs_a
        center = (q_ + crit2 / (2 * nobs_a)) / denom
        dist = crit * np.sqrt(
            q_ * (1.0 - q_) / nobs_a + crit2 / (4.0 * nobs_a**2)
        )
        dist /= denom
        ci_low = center - dist
        ci_upp = center + dist
    else:
        raise NotImplementedError(f"method {method} is not available")
    if method in ["normal", "agresti_coull"]:
        ci_low = np.clip(ci_low, 0, 1)
        ci_upp = np.clip(ci_upp, 0, 1)

    return ci_low, ci_upp
    

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
