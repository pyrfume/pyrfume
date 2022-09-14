import functools
import warnings
from typing import Callable, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
from eden.graph import Vectorizer
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from tqdm.auto import tqdm, trange

from pyrfume import load_data, odorants, save_data
from pyrfume.mol2networx import smiles_to_eden

try:
    from mordred import Calculator
    from mordred import descriptors as all_descriptors
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        "Parts of mordred and/or rdkit could not be imported; try installing rdkit via conda",
        UserWarning,
    )

FEATURES_DIR = "physicochemical"
DRAGON_STEM = "AllDragon%s.csv"


def load_dragon(suffix=""):
    """Loads dragon features.
    Use a suffix to specify a precomputed cleaning of this data"""
    path = "%s/%s" % (FEATURES_DIR, DRAGON_STEM % suffix)
    dragon = load_data(path)
    return dragon


def clean_features(raw_features, max_nan_frac=0.3):
    n_molecules = raw_features.shape[0]
    n_allowed_nans = n_molecules * max_nan_frac
    # Remove features with too many NaNs
    good = raw_features.columns[raw_features.isnull().sum() < n_allowed_nans]
    cleaned_features = raw_features[good]
    cols = [c for c in list(cleaned_features) if c != "SMILES"]
    cleaned_features = cleaned_features[cols]  # Drop SMILES column
    return cleaned_features


def scale_features(cleaned_features, scaler):
    if scaler == "standardize":
        # Scale to mean 0, variance 1
        s = StandardScaler()
    elif scaler == "normalize":
        # Scale to length 1
        s = Normalizer()
    elif scaler == "minmax":
        # Scale to min 0, max 1
        s = MinMaxScaler()
    else:
        raise Exception(("scaler must be one of 'standardize'," " 'normalize', or 'minmax'"))
    scaled_data = s.fit_transform(cleaned_features.astype("float"))
    scaled_features = pd.DataFrame(
        scaled_data, index=cleaned_features.index, columns=cleaned_features.columns
    )
    return scaled_features


def impute_features(scaled_features):
    # Impute missing values
    from fancyimpute import KNN

    knn = KNN(k=5)
    imputed_values = knn.fit_transform(scaled_features.values)
    imputed_features = pd.DataFrame(
        imputed_values, index=scaled_features.index, columns=scaled_features.columns
    )
    return imputed_features


def save_dragon(dragon, suffix):
    path = "%s/%s" % (FEATURES_DIR, DRAGON_STEM % suffix)
    save_data(dragon, path)


def cid_names():
    """TODO: Fix this to use the larger file"""
    path = FEATURES_DIR / "cids-names-smiles.csv"
    names = load_data(path).set_index("CID")["name"]
    return names


def mol_to_mordred(mols, features=None):
    calc = Calculator(all_descriptors)
    print("\nComputing Mordred features...")
    df = calc.pandas(mols.values())
    df = df.fill_missing()  # Use NaN instead of Missing object
    if features is not None:
        df = df[features]  # Retain only the specified features
    mordred = pd.DataFrame(df.values, index=mols.keys(), columns=df.columns)
    print("There are %d molecules and %d features" % mordred.shape)
    return mordred


def smiles_to_mordred(smiles, features=None, max_attempts=10):
    mols = odorants.smiles_to_mol(smiles, max_attempts=max_attempts)
    mordred = mol_to_mordred(mols, features=features)
    return mordred


def make_graphs(smiles):
    eden_graph_generator = [
        smiles_to_eden(smi) for smi in smiles
    ]  # Convert from SMILES to EdEN format
    graphs = [graph for graph in eden_graph_generator]  # Compute graphs for each molecule
    vectorizer = Vectorizer(min_r=0, min_d=0, r=1, d=2)
    sparse = vectorizer.transform(graphs)  # Compute the NSPDK features and store in a sparse array
    return sparse


def smiles_to_nspdk(smiles, features=None):
    nspdk_sparse = make_graphs(smiles)  # Compute the NSPDK features and store in a sparse array
    n_molecules, n_features = nspdk_sparse.shape
    print(
        "There are %d molecules and %d potential features per molecule" % (n_molecules, n_features)
    )
    # Extract the indices of NSPDK features where at least one molecules is non-zero
    if features is None:
        original_indices = sorted(list(set(nspdk_sparse.nonzero()[1])))
        n_used_features = len(original_indices)
        print(
            "Only %d of these features are used "
            "(%.1f features per molecule; %.1f molecules per feature)"
            % (
                n_used_features,
                nspdk_sparse.size / n_molecules,
                nspdk_sparse.size / n_used_features,
            )
        )
        # Create a dense array from those non-zero features
        nspdk_dense = nspdk_sparse[:, original_indices].todense()
        indices = original_indices
    else:
        n_used_features = len(features)
        nspdk_sparse = nspdk_sparse[:, features]
        print(
            "Only %d of these features will be used "
            "(%.1f features per molecule; %.1f molecules per feature)"
            % (
                n_used_features,
                nspdk_sparse.size / n_molecules,
                nspdk_sparse.size / n_used_features,
            )
        )
        nspdk_dense = nspdk_sparse.todense()  # Include only the desired features
        indices = features
    # Create a Pandas DataFrame
    nspdk = pd.DataFrame(nspdk_dense, index=smiles, columns=indices)
    return nspdk


def smiles_to_nspdk_sim(smiles, ref_smiles, features=None):
    sparse = make_graphs(smiles)
    ref_sparse = make_graphs(ref_smiles)
    sim = sparse.dot(ref_sparse.T).todense()
    sim = pd.DataFrame(sim, index=smiles, columns=ref_smiles)
    return sim


def smiles_to_dragon(smiles, suffix="", features=None):
    dragon = load_data("physicochemical/AllDragon%s.csv" % suffix)
    if features is None:
        features = list(dragon)
    dragon = dragon.loc[smiles, dragon]
    return dragon


def smiles_to_morgan(smiles, radius=5, features=None):
    mols = odorants.smiles_to_mol(smiles)
    morgan = mol_to_morgan(mols, radius=radius, features=features)
    return morgan


def mol_to_morgan(mols, radius=5, features=None):
    fps = []
    for index, mol in tqdm(mols.items(), desc="Computing Morgan Fingerprints"):
        fp = AllChem.GetMorganFingerprint(mol, radius)
        fp = pd.Series(fp.GetNonzeroElements(), name=index)
        fps.append(fp)

    batch_size = 100
    if len(mols) < batch_size:
        morgan = pd.DataFrame().join(fps, how="outer")
    else:
        morgans = []
        for start in trange(0, len(mols), batch_size, desc="Joining DataFrames"):
            morgan = pd.DataFrame().join(fps[start : start + batch_size], how="outer")
            morgans.append(morgan)
        morgan = pd.DataFrame().join(morgans, how="outer")
    morgan = morgan.fillna(0).T
    if features is not None:
        morgan = morgan[features]  # Retain only the specified features
    return morgan


def smiles_to_morgan_sim(smiles, ref_smiles, radius=5, features=None):
    mols = odorants.smiles_to_mol(smiles)
    ref_mols = odorants.smiles_to_mol(ref_smiles)
    morgan_sim = mol_to_morgan_sim(mols, ref_mols, radius=radius, features=features)
    return morgan_sim


def mol_to_morgan_sim(mols, ref_mols, radius=5, features=None):
    all_mols = pd.concat([pd.Series(mols), pd.Series(ref_mols)]).drop_duplicates()
    fps = all_mols.apply(lambda x: AllChem.GetMorganFingerprint(x, radius))
    fps = fps[~fps.index.duplicated()]
    morgan_sim = pd.DataFrame(index=mols.keys(), columns=ref_mols.keys())

    def dice(col):
        return np.array([DataStructs.DiceSimilarity(fps[col.name], fps[key]) for key in col.index])

    for ref in tqdm(ref_mols):
        morgan_sim[ref] = dice(morgan_sim[ref])
    return morgan_sim

    # print("%d similarity features for %d molecules" % morgan.shape[::-1])
    # return morgan


# rdkit's DataStructs.ExplicitBitVect is more efficient for rdkit-internal use.
get_morgan_fp: Callable[[Chem.Mol], DataStructs.ExplicitBitVect] = functools.partial(
    Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect, radius=2, nBits=2048
)


def tanimoto_sim(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """Compute Tanimoto similarity for just two molecules."""
    return DataStructs.FingerprintSimilarity(
        get_morgan_fp(mol1), get_morgan_fp(mol2), metric=DataStructs.TanimotoSimilarity
    )


def _bulk_similarity(
    mols1: Iterable[Chem.Mol], mols2: Optional[Iterable[Chem.Mol]] = None
) -> Iterator[np.ndarray]:
    if mols2 is None:
        mols2 = mols1
    mol1_fps = map(get_morgan_fp, mols1)
    mol2_fps = tuple(map(get_morgan_fp, mols2))
    for fp in mol1_fps:
        yield DataStructs.BulkTanimotoSimilarity(fp, mol2_fps)


def get_maximum_tanimoto_similarity(
    molecules: Iterable[Chem.Mol], reference_set: Optional[Iterable[Chem.Mol]] = None
) -> np.ndarray:
    """Compute maximal Tanimoto similarity to `reference_set` for all given molecules."""
    computing_self_similarity = reference_set is None

    result = []
    for i, similarities in enumerate(_bulk_similarity(molecules, reference_set)):
        if computing_self_similarity:
            similarities[i] = 0
        result.append(max(similarities))
    return np.array(result)


def mol_file_to_smiles(mol_file_path):
    # Takes a path to a .mol file and returns a SMILES string
    x = Chem.MolFromMolFile(mol_file_path)
    result = Chem.MolToSmiles(x, isomericSmiles=True)
    return result
