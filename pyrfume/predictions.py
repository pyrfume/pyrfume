import os
import pandas as pd

from .base import DATA_DIR


def get_predicted_intensities():
    file_name = 'cids-names-smiles-mordredpredintensities.csv'
    path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(path).set_index('CID')
    return df['Intensity']
