"""Summarise data exclusion"""
import numpy as np
import pandas as pd
from pathlib import Path


out_path = Path("Z:/andres/nhsebcss/results/primary")


df = pd.read_csv(out_path / 'removed_episodes.csv')
df = df.set_index('Characteristic')

cat = ['Episodes', 
       'Episodes with age < 50',
       'Episodes open or pending',
       'Episode has result but no kit reading',
       'Episodes with multiple analyser readings on the same date',
       'Episodes where the result is FIT negative and reading >= 120',
       'Episodes where result implies a positive FIT but reading < 120',
       'All removed episodes']

newcat = {'Incomplete data': 
            ['Episode has result but no kit reading',
             'Episodes open or pending',
             'Episodes where the result is FIT negative and reading >= 120',
             'Episodes where result implies a positive FIT but reading < 120'
             ],
          'Age < 50': 
            ['Episodes with age < 50'],
          'Multiple analyser readings on the same date' : 
            ['Episodes with multiple analyser readings on the same date']
}


dfnew = df.copy()
for key, val in newcat.items():

    dfsub = dfnew.loc[dfnew.index.isin(val)]
    s = dfsub.sum(axis=0)
    s = pd.DataFrame([s], index=[key])

    dfnew = dfnew.loc[~dfnew.index.isin(val)]
    dfnew = pd.concat(objs=[dfnew, s], axis=0)

dfnew = dfnew.reset_index()
dfnew = dfnew.rename(columns={'index': 'Characteristic'})
dfnew.to_csv(out_path / 'removed_episodes_simple.csv', index=False)