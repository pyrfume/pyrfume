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

from rdkit.Chem.rdinchi import InchiToInchiKey
from tqdm.auto import tqdm

import pyrfume
from pyrfume.odorants import get_cid

df = pyrfume.load_data("arctander_1960/Arctander Master.xlsx")

df["InChiKey"] = df["InChiKey"].apply(
    lambda x: InchiToInchiKey(x) if "InChI=" in str(x) and str(x) != "nan" else x
)

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    # if index < 215:
    #    continue
    cid = 0
    for j, col in enumerate(["InChiKey", "SMILES", "CAS", "ChemicalName"]):
        if not str(row[col]) == "nan":
            cid = get_cid(row[col], kind=(col if j < 2 else "name"))
            if cid:
                break
    df.loc[index, "new_CID"] = cid

df[df["new_CID"].isnull()]

df.join(df[[]])

pyrfume.save_data(df, "arctander_1960/arctander.csv")

df.dropna(subset=["ChemicalName"]).shape

x = dict(df.dropna(subset=["ChemicalName"]).set_index("ChemicalName")["Description"])

dict(df.set_index("CID")["Description"])
