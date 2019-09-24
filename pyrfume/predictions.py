import os
import pandas as pd

import pyrfume

def get_predicted_intensities():
    file_name = 'cids-names-smiles-mordredpredintensities.csv'
    path = pyrfume.DATA_DIR / 'physicochemical' / file_name
    df = pd.read_csv(path).set_index('CID')
    return df['Intensity']
