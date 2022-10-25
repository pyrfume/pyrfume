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

# ### SMILES strings for the first N PubChem IDs
# #### 2020/02/07: N = 100,000

# %load_ext autoreload
# %autoreload 2

import pandas as pd

import pyrfume
from pyrfume.odorants import from_cids

if "results" not in locals():
    results = {}
n = int(1e5)
by = int(1e4)  # In case there are errors, we will only have to go back this far
for first in range(1, n + 1, by):
    if first not in results:
        last = first + by
        x = from_cids(range(first, last))
        results[first] = x

results.keys()

df = pd.concat([pd.DataFrame(results[x]).set_index("CID") for x in results])

pyrfume.save_data(df, "odorants/cids-smiles-pubchem-100000.csv")
