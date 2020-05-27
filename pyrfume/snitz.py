import pandas as pd
from scipy.spatial.distance import pdist, squareform

import pyrfume

from . import features
from .base import DEFAULT_DATA_PATH

SNITZ_DIR = DEFAULT_DATA_PATH / "snitz_2013"


def get_snitz_dragon(use_original=True, regenerate=False):
    path = SNITZ_DIR / "snitz_dragon.csv"
    if not path.is_file() or regenerate:
        snitz_feature_names = get_snitz_weights().index
        # Use minmax scaling as in the Snitz paper
        minmax_scaled_dragon = features.load_dragon(suffix="-cleaned-minmaxed-imputed")
        df = minmax_scaled_dragon[snitz_feature_names]
        pyrfume.save_data(df, path)
    else:
        df = pyrfume.load_data(path)  # .set_index('PubChemID')
    return df


def get_snitz_weights(use_original=True):
    """Return a pandas Series of weights for Dragon features in Snitz"""
    if use_original:  # Use the ones from the Snitz paper, with no weights
        file_name = "snitz-descriptors-from-paper-dragon-6.csv"
        path = SNITZ_DIR / file_name
        snitz_weights = pyrfume.load_data(path, index_col=0)["Weight"]
    else:
        # Use the ones that I derived, with weights computed by optimization
        # using Snitz-space projections of each molecule's original unit vector
        file_name = "snitz_dragon_weights.csv"
        path = SNITZ_DIR / file_name
        snitz_weights = -pyrfume.load_data(path, index_col=0)["Weight"]
    return snitz_weights


def get_snitz_features(dragon, snitz_weights=None, use_original=True):
    if snitz_weights is None:
        snitz_weights = get_snitz_weights(use_original=use_original)
    snitz_features = dragon[snitz_weights.index] * snitz_weights
    # CIDs where the Snitz vector has no length
    null_vectors = list(snitz_features.index[snitz_features.sum(axis=1) == 0])
    print("Number of zero-length Snitz vectors is %d " % len(null_vectors))
    # Fill these with median features (these molecules will become useless)
    for nv in null_vectors:
        snitz_features.loc[nv, :] = snitz_features.median()
    return snitz_features


def get_snitz_distances(dragon, snitz_features=None, snitz_weights=None, use_original=True):
    # Compute Snitz distances.
    # Cosine distance is approximately the same as angle distance
    # for small-ish angles
    if snitz_features is None:
        snitz_features = get_snitz_features(
            dragon, snitz_weights=snitz_weights, use_original=use_original
        )
    x = pdist(snitz_features.values, "cosine")
    snitz_distances = pd.DataFrame(
        squareform(x), index=snitz_features.index, columns=snitz_features.index
    )
    return snitz_distances
