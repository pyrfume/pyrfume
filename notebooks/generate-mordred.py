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
from mordred import Calculator
from mordred import descriptors as all_descriptors
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover
from fancyimpute import KNN

smiles = pd.read_csv("data/snitz-odorant-info.csv").set_index("CID")["IsomericSMILES"]

calc = Calculator(all_descriptors)
print("Convering SMILES string to Mol format...")
mols = {cid: Chem.MolFromSmiles(smi) for cid, smi in smiles.items()}
print("Computing 3D coordinates...")
s = SaltRemover.SaltRemover()
for i, (cid, mol) in enumerate(mols.items()):
    if i > 0 and i % 100 == 0:
        print("Finished %d" % i)
    try:
        mol.SetProp("_Name", "%d: %s" % (cid, smiles[cid]))
        mol = s.StripMol(mol, dontRemoveEverything=True)
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)  # Is this deterministic?
    except Exception as e:
        print("Exception for %d" % cid)
        mols[cid] = None
    else:
        mols[cid] = mol
mols = {cid: mol for cid, mol in mols.items() if mol}

len(set(smiles.index))

results = calc.pandas(mols.values())
results = results.set_index(pd.Index(mols.keys(), name="CID"))
results.head()

results.shape

def fix(x):
    try:
        x = float(x)
    except Exception:
        x = None
    return x

results = results.applymap(fix)

frac_bad = results.isnull().mean()
good = frac_bad[frac_bad < 0.3].index
results = results.loc[:, good]

knn = KNN(k=5)
results[:] = knn.fit_transform(results.values)

results.to_csv("data/snitz-mordred.csv")

results.shape

