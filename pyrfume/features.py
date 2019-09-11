import os
import pathlib
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from fancyimpute import KNN

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
