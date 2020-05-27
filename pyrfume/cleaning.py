import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

from missingpy import KNNImputer

from .base import DATA_DIR


def clean_dragon(save=False):
    source = os.path.join(DATA_DIR, "cids-smiles-dragon.txt")
    df = pd.read_csv(source).set_index("CID")
    df = df.iloc[:, 1:]  # Drop SMILES column

    # Scale to mean 0, variance 1
    ss = StandardScaler()
    good = df.columns[df.isnull().sum() < 500]
    df = df[good]
    scaled = ss.fit_transform(df.astype("float"))
    df = pd.DataFrame(scaled, index=df.index, columns=df.columns)

    # Impute missing values
    knn = KNNImputer(k=5)
    imputed = knn.fit_transform(df.values)
    df = pd.DataFrame(imputed, index=df.index, columns=df.columns)

    # Optionally save to disk
    if save:
        dest = os.path.join(DATA_DIR, "cids-smiles-dragon-scaled-imputed.txt")
        df.to_csv(dest)

    return df
