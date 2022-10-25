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

original = pyrfume.load_data("physicochemical/AllDragon-20190730-mayhew.csv")
original.head()

new = pyrfume.load_data("physicochemical/ExtraEight.txt", delimiter="\t")
new = new.set_index("NAME").sort_index()
new.index.name = "PubChemID"
new.index

infos = odorants.from_cids(new.index)
for info in infos:
    new.loc[info["CID"], "SMILES"] = info["IsomericSMILES"]
new = new[["SMILES"] + [x for x in list(original) if x != "SMILES"]]
new.head()

assert list(original) == list(new)

df = pd.concat([original, new])
df = df.groupby(level=0).first()  # Drop duplicate PubChem IDs
df.shape

pyrfume.save_data(df, "physicochemical/AllDragon.csv")
