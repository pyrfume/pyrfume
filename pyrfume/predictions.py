import os
import pandas as pd

import pyrfume

def get_predicted_intensities():
    """Return the DREAM model predicted intensities using Mordred (not Dragon) features"""
    path = 'physicochemical/cids-names-smiles-mordredpredintensities.csv'
    df = pyrfume.load_data(path)
    return df['Intensity']

def get_predicted_odorless():
    """Return a pandas Series that is True for molecules predicted to have an odor
    and False for those predicted to be odorless"""
    path = 'odorants/predicted_odorless.csv'
    df = pyrfume.load_data(path)
    return df['PredictedOdorless']