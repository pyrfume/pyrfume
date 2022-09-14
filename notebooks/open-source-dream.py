# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Creation of an open source model for DREAM descriptor prediction

# ### Preliminaries and Imports

# %load_ext autoreload
# %autoreload 2

import pickle

import numpy as np
import opc_python.utils.loading as dream_loading
import pandas as pd
from missingpy import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.multioutput import MultiOutputRegressor

import pyrfume
from pyrfume.features import smiles_to_mordred, smiles_to_morgan_sim
from pyrfume.odorants import from_cids
from pyrfume.predictions import load_dream_model, predict, smiles_to_features

from pyrfume.odorants import all_smiles, from_cids

# ### Load the perceptual data from Keller and Vosshall, 2016

kv2016_perceptual_data = dream_loading.load_raw_bmc_data()
kv2016_perceptual_data = dream_loading.format_bmc_data(
    kv2016_perceptual_data,
    only_dream_subjects=False,  # Whether to only keep DREAM subjects
    only_dream_descriptors=True,  # Whether to only keep DREAM descriptors
    only_dream_molecules=True,
)  # Whether to only keep DREAM molecules)

# Get the list of PubChem IDs from this data
kv_cids = list(kv2016_perceptual_data.index.get_level_values("CID").unique())

# Get information from PubChem about these molecules
info = from_cids(kv_cids)
# Make a Pandas series relating PubChem IDs to SMILES strings
smiles = pd.Series(index=kv_cids, data=[x["IsomericSMILES"] for x in info])
smiles.head()

# ### Compute physicochemical features for the DREAM data (and some other molecules)

# Get a list of all SMILES strings from the Pyrfume library
ref_smiles = list(set(all_smiles()))

mordred_features_ref = smiles_to_mordred(ref_smiles)

# A KNN imputer instance
imputer_knn = KNNImputer(n_neighbors=10, col_max_missing=1)
X = mordred_features_ref.astype(float)
imputer_knn.fit(X)

# Compute Mordred features from these SMILES strings
mordred_features = smiles_to_mordred(smiles.values)

# The computed Mordred features as floats (so errors become NaNs)
X = mordred_features.astype(float)
# Whether a column (one feature, many molecules) has at least 50% non-NaN values
is_good_col = X.isnull().mean() < 0.5
# The list of such "good" columns (i.e. well-behaved features)
good_cols = is_good_col[is_good_col].index
# Impute the missing (NaN) values
X[:] = imputer_knn.fit_transform(X)
# Restrict Mordred features to those from the good columns, even after imputation
X = X[good_cols]
# Put back into a dataframe
mordred_features_knn = pd.DataFrame(index=mordred_features.index, columns=good_cols, data=X)

# Compute Morgan fingerprint similarities from these SMILES strings
morgan_sim_features = smiles_to_morgan_sim(smiles.values, ref_smiles)

len(ref_smiles), len(set(ref_smiles))

len(list(morgan_sim_features)), len(set(morgan_sim_features))

# Combine Mordred (after imputation) and Morgan features into one dataframe
all_features = mordred_features_knn.join(morgan_sim_features, lsuffix="mordred_", rsuffix="morgan_")
assert len(all_features.index) == len(all_features.index.unique())
assert list(all_features.index) == list(smiles.values)
all_features.index = smiles.index
all_features.index.name = "PubChem CID"
all_features.head()

len(list(all_features)), len(set(all_features))

# ### Organize perceptual data

# Compute the descriptor mean across subjects
data_mean = kv2016_perceptual_data.mean(axis=1)
# Compute the subject-averaged descriptor mean across replicates
data_mean = (
    data_mean.unstack("Descriptor").reset_index().groupby(["CID", "Dilution"]).mean().iloc[:, 1:]
)
# Fix the index for joining
data_mean.index = data_mean.index.rename(["PubChem CID", "Dilution"])
# Show the dataframe
data_mean.head()

# ### Join the features and the descriptors and split again for prediction

# Create a joined data frame with perceptual descriptors and physicochemical features
df = data_mean.join(all_features, how="inner")
# Add a column for dilution (used in prediction)
df["Dilution"] = df.index.get_level_values("Dilution")
# Make a list of all the columns that will be used in prediction
predictor_columns = [col for col in list(df) if col not in list(data_mean)]
# Make a list of all the columns that must be predicted
data_columns = list(data_mean)
# Create the feature matrix and the target matrix
X = df[predictor_columns]
Y = df[data_columns]

# Each feature name is only used once
assert pd.Series(predictor_columns).value_counts().max() == 1


# ### Verify that this model gets reasonable out-of-sample performance

# +
# A function to compute the correlation between the predicted and observed ratings
# (for a given descriptor columns)
def get_r(Y, Y_pred, col=0):
    pred = Y_pred[:, col]
    obs = Y.iloc[:, col]
    return np.corrcoef(pred, obs)[0, 1]


# A series of scorers, one for each descriptor
scorers = {desc: make_scorer(get_r, col=i) for i, desc in enumerate(Y.columns)}
# The number of splits to use in cross-validation
n_splits = 5
# The number of descriptors in the perceptual data
n_descriptors = Y.shape[1]
# A vanilla Random Forest model with only 10 trees (performance will increase with more trees)
rfr = RandomForestRegressor(n_estimators=10, random_state=0)
# A multioutput regressor used to fit one model per descriptor, in parallel
mor = MultiOutputRegressor(rfr, n_jobs=n_descriptors)

# Check the cross-validation performance of this kind of model
cv_scores = cross_validate(mor, X, Y, scoring=scorers, cv=n_splits)
# -

# An empty dataframe to hold the cross-validation summary
rs = pd.DataFrame(index=list(Y))
# Compute the mean and standard deviation across cross-validation splits
rs["Mean"] = [cv_scores["test_%s" % desc].mean() for desc in list(Y)]
rs["StDev"] = [cv_scores["test_%s" % desc].std() for desc in list(Y)]
# Show the results
rs

# ### Fit the final model and save it

# A random forest regressor with more trees
rfr = RandomForestRegressor(n_estimators=250, random_state=0)
# Wrap in a class that will fit one model per descriptor
mor = MultiOutputRegressor(rfr, n_jobs=n_descriptors)
# Fit the model
# %time mor.fit(X, Y);

len(list(X)), len(set(list(X)))

# Save the fitted model
path = pyrfume.DATA_DIR / "keller_2017" / "open-source-dream.pkl"
with open(path, "wb") as f:
    pickle.dump([mor, list(X), list(Y), imputer_knn], f)

# ### Demonstration: using the fitted model (can be run independently if the above has been run at some point)

novel_cids = [14896, 228583]  # Beta-pinene and 2-Furylacetone
novel_info = from_cids(novel_cids)
novel_smiles = [x["IsomericSMILES"] for x in novel_info]
model_, use_features_, descriptors_, imputer_ = load_dream_model()
features_ = smiles_to_features(novel_smiles, use_features_, imputer_)
predict(model_, features_, descriptors_)
