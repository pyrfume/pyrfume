import os
import pathlib
import numpy as np
import pandas as pd
from rickpy import ProgressBar
import warnings
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

try:
    from mordred import Calculator, descriptors as all_descriptors
    from rdkit import Chem
    from rdkit.Chem.Descriptors import MolWt
    from rdkit.Chem import inchi, AllChem, SaltRemover
    from rdkit import DataStructs
    from rdkit.ML.Descriptors import MoleculeDescriptors
except ImportError:
    warnings.warn("Parts of mordred and/or rdkit could not be imported; try installing rdkit via conda",
                  UserWarning)

from .base import DATA_DIR

FEATURES_DIR = DATA_DIR / 'physicochemical'
DRAGON_STEM = 'AllDragon%s.csv'


def load_dragon(suffix=''):
    """Loads dragon features.
    Use a suffix to specify a precomputed cleaning of this data"""
    file_name = DRAGON_STEM % suffix
    path = FEATURES_DIR / file_name
    dragon = pd.read_csv(path).set_index('PubChemID')
    return dragon


def clean_features(raw_features, max_nan_frac=0.3):
    n_molecules = raw_features.shape[0]
    n_allowed_nans = n_molecules*max_nan_frac
    # Remove features with too many NaNs
    good = raw_features.columns[raw_features.isnull().sum() < n_allowed_nans]
    cleaned_features = raw_features[good]
    cols = [c for c in list(cleaned_features) if c != 'SMILES']
    cleaned_features = cleaned_features[cols]  # Drop SMILES column
    return cleaned_features


def scale_features(cleaned_features, scaler):
    if scaler == 'standardize':
        # Scale to mean 0, variance 1
        s = StandardScaler()
    elif scaler == 'normalize':
        # Scale to length 1
        s = Normalizer()
    elif scaler == 'minmax':
        # Scale to min 0, max 1
        s = MinMaxScaler()
    else:
        raise Exception(("scaler must be one of 'standardize',"
                         " 'normalize', or 'minmax'"))
    scaled_data = s.fit_transform(cleaned_features.astype('float'))
    scaled_features = pd.DataFrame(scaled_data,
                                   index=cleaned_features.index,
                                   columns=cleaned_features.columns)
    return scaled_features


def impute_features(scaled_features):
    # Impute missing values
    from fancyimpute import KNN
    knn = KNN(k=5)
    imputed_values = knn.fit_transform(scaled_features.values)
    imputed_features = pd.DataFrame(imputed_values,
                                    index=scaled_features.index,
                                    columns=scaled_features.columns)
    return imputed_features


def save_dragon(dragon, suffix):
    file_name = DRAGON_STEM % suffix
    dest = FEATURES_DIR / file_name
    dragon.to_csv(dest)


def cid_names():
    """TODO: Fix this to use the larger file"""
    path = FEATURES_DIR / 'cids-names-smiles.csv'
    names = pd.read_csv(path).set_index('CID')['name']
    return names


def smiles_to_mordred(smiles, features=None):
    # create descriptor calculator with all descriptors
    calc = Calculator(all_descriptors)
    print("Convering SMILES string to Mol format...")
    mols_raw = [Chem.MolFromSmiles(smi) for smi in smiles]
    print("Computing 3D coordinates...")
    s = SaltRemover.SaltRemover()
    mols = {}
    n = len(mols_raw)
    p = ProgressBar(n)
    for i, mol in enumerate(mols_raw):
        p.animate(i, status="Embedding %s" % smiles[i])
        try:
            mol = s.StripMol(mol, dontRemoveEverything=True)
            mol = Chem.AddHs(mol)
            AllChem.Compute2DCoords(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol) # Is this deterministic?  
        except Exception as e:
            print('Exception for %s' % smiles[i])
        else:
            mols[smiles[i]] = mol
    p.animate(n, status="Finished embedding all molecules")
    print("\nComputing Mordred features...")
    df = calc.pandas(mols.values())
    if features is not None:
        df = df[features] # Retain only the specified features
    mordred = pd.DataFrame(df.values, index=mols.keys(), columns=df.columns)
    #mordred = mordred.astype(float)
    print("There are %d molecules and %d features" % mordred.shape)
    return mordred

def make_graphs(smiles):
    eden_graph_generator = obabel_to_eden(smiles, file_format='smi') # Convert from SMILES to EdEN format
    graphs = [graph for graph in eden_graph_generator] # Compute graphs for each molecule
    vectorizer = Vectorizer(min_r=0,min_d=0,r=1,d=2)
    sparse = vectorizer.transform(graphs) # Compute the NSPDK features and store in a sparse array
    return sparse
  
def smiles_to_nspdk(smiles,features=None):
    nspdk_sparse = make_graphs(smiles) # Compute the NSPDK features and store in a sparse array
    n_molecules,n_features = nspdk_sparse.shape
    print("There are %d molecules and %d potential features per molecule" % (n_molecules,n_features))
    # Extract the indices of NSPDK features where at least one molecules is non-zero
    if features is None:
        original_indices = sorted(list(set(nspdk_sparse.nonzero()[1])))
        n_used_features = len(original_indices)
        print('Only %d of these features are used (%.1f features per molecule; %.1f molecules per feature)' % \
              (n_used_features,nspdk_sparse.size/n_molecules, nspdk_sparse.size/n_used_features))
        # Create a dense array from those non-zero features
        nspdk_dense = nspdk_sparse[:, original_indices].todense()
        indices = original_indices
    else:
        n_used_features = len(features)
        nspdk_sparse = nspdk_sparse[:,features]
        print('Only %d of these features will be used (%.1f features per molecule; %.1f molecules per feature)' % \
              (n_used_features, nspdk_sparse.size/n_molecules, nspdk_sparse.size/n_used_features))
        nspdk_dense = nspdk_sparse.todense() # Include only the desired features
        indices = features
    # Create a Pandas DataFrame
    nspdk = pd.DataFrame(nspdk_dense,index=smiles,columns=indices)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    return nspdk

def smiles_to_nspdk_sim(smiles, ref_smiles,features=None):
    sparse = make_graphs(smiles)
    ref_sparse = make_graphs(ref_smiles)
    sim = sparse.dot(ref_sparse.T).todense()
    sim = pd.DataFrame(sim, index=smiles, columns=ref_smiles)
    return sim

def smiles_to_dragon(smiles, suffix='', features=None):
    dragon = pyrfume.load_data('physicochemical/AllDragon%s.csv' % suffix)
    if features is None:
        features = list(dragon)
    dragon = dragon.loc[smiles, dragon]
    return dragon

def smiles_to_morgan(smiles, radius=5, features=None):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    fps = [AllChem.GetMorganFingerprint(mol,radius) for mol in mols]
    fp_ids = []
    for fp in fps:
        fp_ids += list(fp.GetNonzeroElements().keys())
    fp_ids = list(set(fp_ids))
    morgan = np.empty((len(fps), len(fp_ids)))
    for i,fp in enumerate(fps):
        for j,fp_id in enumerate(fp_ids):
            morgan[i, j] = fp[fp_id]
    morgan = pd.DataFrame(morgan, index=smiles, columns=fp_ids)
    if features is not None:
        morgan = morgan[features] # Retain only the specified features
    return morgan

def smiles_to_morgan_sim(smiles, ref_smiles, radius=5, features=None):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    fps = [AllChem.GetMorganFingerprint(mol,radius) for mol in mols]
    morgan = np.empty((len(smiles),len(ref_smiles)))
    for i,ref_smile in enumerate(ref_smiles):
        ref = Chem.MolFromSmiles(ref_smile)
        fp_ref = AllChem.GetMorganFingerprint(ref,radius)
        morgan[:,i] = np.array([DataStructs.DiceSimilarity(fp,fp_ref) for fp in fps])
    morgan = pd.DataFrame(morgan,index=smiles,columns=ref_smiles)
    print("%d similarity features for %d molecules" % morgan.shape[::-1])
    return morgan

def mol_file_to_smiles(mol_file_path):
    # Takes a path to a .mol file and returns a SMILES string
    x = Chem.MolFromMolFile(mol_file_path)
    result = Chem.MolToSmiles(a,isomericSmiles=True)
    return result