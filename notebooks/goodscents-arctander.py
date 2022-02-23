# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# #### Basically, I just started from the mergedOdorants file that Joel sent me, and converted SMILES strings to CIDs

import os
import pandas as pd
import pyrfume
from pyrfume import odorants

file_path = os.path.join(pyrfume.DATA, 'mergedOdorants.csv')
df = pd.read_csv(file_path, index_col=0)

# Get CIDs from PubChem
smiles_cids = odorants.get_cids(df['NAME'], kind='smiles')

# Merge back into this list
df = pd.Series(smiles_cids, name='PubChemID').to_frame().join(df.set_index('NAME'))

# Save back to a file of just CIDs
for lib, name in [('goodscent', 'goodscents'), ('arc', 'arctander')]:
    file_path = os.path.join(pyrfume.DATA, '%s_cids.txt' % name)
    cids = sorted(set(df[df['lib']==lib]['PubChemID']) - {0})
    pd.Series(cids, name='CID').to_csv(file_path, header=True, index=False)

file_path = os.path.join(pyrfume.DATA, 'mergedOdorants_with_cids.csv')
df.to_csv(file_path)
