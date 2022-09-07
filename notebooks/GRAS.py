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
from pyrfume import odorants

file_path = os.path.join(pyrfume.DATA, "GRAS.smi")
gras_data_raw = pd.read_csv(file_path, header=None, names=["SMILES", "CAS"], sep="\t")

results = odorants.get_cids(gras_data_raw["SMILES"], kind="SMILES", verbose=False)

gras_data = pd.Series(results, name="CID").to_frame().join(gras_data_raw.set_index("SMILES"))
gras_data.head()

file_path = os.path.join(pyrfume.DATA, "gras.csv")
gras_data.to_csv(file_path)
