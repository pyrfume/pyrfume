"""Module for interacting with unpublished data from Westeros"""

import os
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

import pyrfume
from . import features
from .base import DATA_DIR


def get_x_embeddings():
    emb_raw = pyrfume.load_data('westeros/embeddings.csv')
    emb = emb_raw[[x for x in list(emb_raw) if 'emb_' in x]]
    emb = emb.dropna()
    return emb

def get_x_distances(embeddings):
    emb = get_x_embeddings()
    # Compute distance matrix.
    # This will give basically the same low-D embedding as using raw high-dimensional embeddings does
    dist = pdist(emb, metric='euclidean')
    dist = squareform(dist)
    distances = pd.DataFrame(dist, index=emb.index, columns=emb.index)
    return distances
