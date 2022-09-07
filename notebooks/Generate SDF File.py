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
from rdkit.Chem import AllChem, SaltRemover, rdmolfiles

smiles = pd.read_csv("data/cids-names-smiles.csv").set_index("CID")["IsomericSMILES"]

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

writer = rdmolfiles.SDWriter("data/cids-smiles.sdf")
for smile, mol in mols.items():
    writer.write(mol)
writer.close()

suppl = rdmolfiles.SDMolSupplier("data/cids-smiles.sdf")
for mol in suppl:
    print(mol.GetNumAtoms())
