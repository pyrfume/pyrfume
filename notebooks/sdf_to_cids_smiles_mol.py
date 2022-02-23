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

# +
import os

import pandas as pd
from rdkit import Chem

import pyrfume
from pyrfume import odorants
from rickpy import ProgressBar
# -

# ## Create a dictionary of mol files

file_path = os.path.join(pyrfume.DATA, 'all_cids.sdf')
f = Chem.SDMolSupplier(file_path)
result = {}
for mol in f:
    x = mol.GetProp('_Name')
    cid, smiles = x.split(':')
    cid = int(cid)
    smiles = smiles.strip()
    result[cid] = {'smiles': smiles, 'mol': mol}


