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
from rdkit import Chem

import pyrfume
from pyrfume import odorants

file_path = os.path.join(pyrfume.DATA, "westeros", "molecules.csv")
leffingwell_data_raw = pd.read_csv(file_path, sep="\t")

results = odorants.get_cids(leffingwell_data_raw["smiles"], kind="SMILES", verbose=False)

leffingwell_data = (
    pd.Series(results, name="CID").to_frame().join(leffingwell_data_raw.set_index("smiles"))
)
leffingwell_data.head()

for smiles in leffingwell_data[leffingwell_data["CID"] == 0].index:
    name = leffingwell_data.loc[smiles, "chemical_name"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Bad smiles: %s" % smiles)
    else:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    cid = odorants.get_cid(smiles, kind="smiles", verbose=True)
    if cid:
        print(name, cid)
    else:
        print(name, smiles)

leffingwell_data = (
    pd.Series(results, name="CID").to_frame().join(leffingwell_data_raw.set_index("smiles"))
)
leffingwell_data[leffingwell_data["CID"] == 0]

x = leffingwell_data.reset_index().set_index("chemical_name")
# x.loc['calcium alginate', 0]
x[x["CID"] == 0].head()

file_path = os.path.join(pyrfume.DATA, "westeros", "westeros.csv")
leffingwell_data.to_csv(file_path)
