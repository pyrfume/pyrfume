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

# # Add CIDS to parsed_threshold_data_in_air.csv

import pandas as pd
from rickpy import ProgressBar

import pyrfume
from pyrfume.odorants import get_cid, get_cids
from rdkit.Chem import MolFromSmiles, MolToSmiles

df = pyrfume.load_data("thresholds/parsed_threshold_data_in_air.csv")
df = df.set_index("canonical SMILES")

smiles_cids = get_cids(df.index, kind="SMILES")

df = df.join(pd.Series(smiles_cids, name="CID"))
df.head()

df["SMILES"] = df.index
p = ProgressBar(len(smiles_cids))
for i, (old, cid) in enumerate(smiles_cids.items()):
    p.animate(i, status=old)
    if cid == 0:
        mol = MolFromSmiles(old)
        if mol is None:
            new = ""
        else:
            new = MolToSmiles(mol, isomericSmiles=True)
            if old != new:
                cid = get_cid(new, kind="SMILES")
        df.loc[old, ["SMILES", "CID"]] = [new, cid]
p.animate(i + 1, status="Done")

df[df["SMILES"] == ""]

ozone_smiles = ozone_cid = get_cid("[O-][O+]=O", kind="SMILES")
df.loc["O=[O]=O", ["SMILES", "CID"]] = [ozone_smiles, ozone_cid]

df = df.set_index("CID").drop(["ez_smiles"], axis=1)

df = df.rename(columns={"author": "year", "year": "author"})

df.head()

pyrfume.save_data(df, "thresholds/parsed_threshold_data_in_air_fixed.csv")
