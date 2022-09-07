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

import os

import pandas as pd

import pyrfume
from pyrfume import odorants, sigma_ff

# Load raw Sigma FF data
descriptors, data = sigma_ff.get_data()

# Turn CAS into CIDs
cas_list = list(data.keys())
results = odorants.get_cids(cas_list, kind="name", verbose=False)

# Format Sigma FF data into Dataframe with CIDs
# Odorants without CIDs will have a CID of 0
sigma = pd.DataFrame(index=cas_list, columns=["CID"] + descriptors, data=0)
sigma.index.name = "CAS"
for cas, desc in data.items():
    sigma.loc[cas, "CID"] = results[cas]
    sigma.loc[cas, desc] = 1
sigma.head()

# Create a new file with CIDs and store here in `cids` dictionary
file_path = os.path.join(pyrfume.DATA, "sigma", "sigma.csv")
sigma.to_csv(file_path)
