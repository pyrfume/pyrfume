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

import pandas as pd
import pyrfume
from pyrfume import odorants

# Load the table of CIDs by sources
file_path = pyrfume.DATA_DIR / "odorants" / "all_cids.csv"
all_cids = pd.read_csv(file_path).set_index("CID")

# +
# Get info from PubChem
info = odorants.from_cids(all_cids.index)

# Turn the info into a dataframe
info = pd.DataFrame.from_records(info).set_index("CID")
info.head()
# -

# Join the CIDs and sources with the PubCheminfo
df = info.join(all_cids)
df = df.rename(columns={"MolecularWeight": "MW", "IsomericSMILES": "SMILES", "name": "Name"})
df = df[["Name", "MW", "SMILES", "IUPACName"] + list(df.columns[4:])]
df.head()

file_path = pyrfume.DATA_DIR / "odorants" / "all_cids_properties.csv"
df.to_csv(file_path)
