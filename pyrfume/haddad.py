"""Module for interacting with Haddad et al, 2008"""

import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

import pyrfume
from . import features
from .base import DEFAULT_DATA_PATH

HADDAD_DIR = DEFAULT_DATA_PATH / "haddad_2008"


def get_haddad_dragon(use_original=True, regenerate=False):
    path = HADDAD_DIR / "haddad_dragon.csv"
    if not path.is_file() or regenerate:
        haddad_feature_names = get_haddad_weights().index
        # Use standard scaling as in the Snitz paper
        standard_scaled_dragon = features.load_dragon(suffix="-cleaned-standardized-imputed")
        df = standard_scaled_dragon[haddad_feature_names]
        pyrfume.save_data(df, path)
    else:
        df = pyrfume.load_data(path)  # .set_index('PubChemID')
    return df


def get_haddad_weights():
    """Get a pandas Series of weights for Dragon features in Haddad"""
    # haddad_info_path = HADDAD_DIR / 'haddad-optimized_v6-symbols.xlsx'
    # df_haddad_list = pd.read_excel(haddad_info_path)
    # haddad_list = df_haddad_list['Dragon 6.0 symbol']
    # haddad_weights = df_haddad_list['Weight']
    # haddad_weights.index = haddad_list
    path = HADDAD_DIR / "haddad-weights-from-paper-dragon-6.csv"
    haddad_weights = pyrfume.load_data(path, index_col=0)["Weight"]
    return haddad_weights


def get_haddad_features(dragon, weights=None):
    """Extract weighted features from the dataframe `dragon`"""
    if not weights:
        haddad_weights = get_haddad_weights()

    # Scale dragon dataframe (may already be scaled)
    ss = StandardScaler()
    scaled = ss.fit_transform(dragon[haddad_weights.index])
    haddad_features = pd.DataFrame(scaled, index=dragon.index, columns=haddad_weights.index)

    # Multiply by weights
    haddad_features = haddad_features.mul(haddad_weights, axis=1)

    return haddad_features


def get_haddad_distances(dragon, weights=None):
    haddad_features = get_haddad_features(dragon, weights=weights)
    # Make Haddad distance matrix
    x = pdist(haddad_features.values, "euclidean")
    haddad_distances = pd.DataFrame(
        squareform(x), index=haddad_features.index, columns=haddad_features.index
    )

    return haddad_distances
