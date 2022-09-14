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
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from rickpy import ProgressBar

import pyrfume

# -

file_path = os.path.join(pyrfume.DATA, "all_cids_properties.csv")
df = pd.read_csv(file_path).set_index("CID")

# ## Make 3D optimized versions of the molecules

# +
# Make basic mol objects
mols = {cid: Chem.MolFromSmiles(smi) for cid, smi in df["IsomericSMILES"].items()}

# Then optimize them
s = SaltRemover()
p = ProgressBar(len(df))
for i, (cid, mol) in enumerate(mols.items()):
    p.animate(i, status=cid)
    try:
        mol.SetProp("_Name", "%d: %s" % (cid, df.loc[cid, "IsomericSMILES"]))
        mol = s.StripMol(mol, dontRemoveEverything=True)
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)  # Is this deterministic?
    except Exception as e:
        p.log("Exception for %d: %s" % (cid, e))
        mols[cid] = None
    else:
        mols[cid] = mol

# Remove CIDs without a successful optimization
mols = {cid: mol for cid, mol in mols.items() if mol}
# -

print("%d mol files successfully optimized from %d CIDs" % (len(mols), len(df)))

# ## Write to an SDF file

file_path = os.path.join(pyrfume.DATA, "all_cids.sdf")
f = Chem.SDWriter(file_path)
for cid, mol in mols.items():
    f.write(mol)
f.close()

# Write the last molecule to a mol file
mol_block = Chem.MolToMolBlock(mol)
file_path = os.path.join(pyrfume.DATA, "random.mol")
with open(file_path, "w+") as f:
    print(mol_block, file=f)
