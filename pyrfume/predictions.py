import pandas as pd

import pyrfume
from pyrfume.features import smiles_to_mordred, smiles_to_morgan_sim
from pyrfume.odorants import all_smiles


def get_predicted_intensities():
    """Return the DREAM model predicted intensities using Mordred (not Dragon) features"""
    path = "physicochemical/cids-names-smiles-mordredpredintensities.csv"
    df = pyrfume.load_data(path)
    return df["Intensity"]


def get_predicted_odorless():
    """Return a pandas Series that is True for molecules predicted to have an odor
    and False for those predicted to be odorless"""
    path = "odorants/predicted_odorless.csv"
    df = pyrfume.load_data(path)
    return df["PredictedOdorless"]


def load_dream_model():
    """Load the open source DREAM model created in `open-source-dream.ipynb`"""
    path = "keller_2017/open-source-dream.pkl"
    model, use_features, descriptors, imputer = pyrfume.load_data(path)
    # model: The model to use for prediction
    # features: The features used for each column
    # descriptors: The descriptors it will predict for each output
    # imputer: The missing data imputer that should be used on any Morgan features
    #          before joining with other features and passing it to the model
    return model, use_features, descriptors, imputer


def smiles_to_features(smiles, use_features, imputer):
    """Turn SMILES into model input"""
    # Compute Mordred features for the new molecules
    mordred = smiles_to_mordred(smiles)
    # Impute any missing values based on the missing data imputer
    print(mordred.shape)
    mordred[:] = imputer.transform(mordred)
    # Compute Morgan features for the new molecules
    sim_smiles = list(set(all_smiles()))
    morgan_sim = smiles_to_morgan_sim(smiles, sim_smiles)
    # Combine Mordred (after imputation) and Morgan features into one dataframe
    features = mordred.join(morgan_sim)
    # Add the `Dilution` column
    features["Dilution"] = -3
    # Restrict to only those features used in the model, and in the same order
    features = features[use_features]
    # Make sure the list of features used for prediction is identical to that used for training
    # i.e. make sure the line above did what was intended
    assert list(features) == use_features
    return features


def predict(model, features, descriptors):
    # Predict perceptual descriptors for the new molecules based on their physicochemical features
    predictions = model.predict(features).round(2)
    # Make a dataframe for these predictions
    predictions = pd.DataFrame(index=features.index, columns=descriptors, data=predictions)
    return predictions
