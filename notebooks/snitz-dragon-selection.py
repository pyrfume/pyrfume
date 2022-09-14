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

import pickle

import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pandas as pd
from fancyimpute import KNN
from sklearn.linear_model import Lasso
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.preprocessing import MinMaxScaler, Normalizer

# ### Load Snitz Dataset #1

df1 = pd.read_csv(
    "data/snitz/experiment1_comparisons.csv", header=0, index_col=0, names=["A", "B", "Similarity"]
)
df1_cids = pd.read_csv("data/snitz/experiment1_cids.csv", index_col=0)
df1_cids = df1_cids.applymap(
    lambda x: x.replace("[", "").replace("]", "").strip().replace(" ", ",")
)
df1_cids
df1.loc[:, ["A", "B"]] = df1.loc[:, ["A", "B"]].applymap(lambda x: df1_cids.loc[x]["Mixture Cids"])
df1.head()

df1.shape[0], len(set(df1[["A", "B"]].values.ravel()))

df1.hist("Similarity")

# ### Load Snitz Dataset #2

df2 = pd.read_csv(
    "data/snitz/experiment2_comparisons.csv", header=0, index_col=0, names=["A", "B", "Similarity"]
)
df2_cids = pd.read_csv("data/snitz/experiment2_cids.csv", index_col=0)
df2_cids = df2_cids.applymap(
    lambda x: x.replace("[", "").replace("]", "").strip().replace(" ", ",")
)
df2_cids
df2.loc[:, ["A", "B"]] = df2.loc[:, ["A", "B"]].applymap(lambda x: df2_cids.loc[x]["Mixture Cids"])
df2.head()

df2.shape[0], len(set(df2[["A", "B"]].values.ravel()))

df2.hist("Similarity")

# ### Load Snitz Dataset #3

df3 = pd.read_csv(
    "data/snitz/experiment3_comparisons.csv", header=0, index_col=0, names=["A", "B", "Similarity"]
)
df3.head()

df3.shape[0], len(set(df3[["A", "B"]].values.ravel()))

df3.hist("Similarity")

# ### Get all Snitz CIDs

snitz_cids = []
for x in df1_cids["Mixture Cids"]:
    snitz_cids += x.split(",")
for x in df2_cids["Mixture Cids"]:
    snitz_cids += x.split(",")
for x in df3[["A", "B"]].values.ravel():
    snitz_cids += [x]
snitz_cids = np.array(snitz_cids).astype(int)
snitz_cids = set(snitz_cids)
print("There are %d distinct CIDs across all of the Snitz datasets" % len(snitz_cids))

# ### Load the Dragon data and scale each features to 0-1.

# +
df_dragon = pd.read_csv("data/cids-smiles-dragon.txt").set_index("CID")
df_dragon = df_dragon.iloc[:, 1:]  # Remove SMILES column

# Normalize every feature to [0, 1]
mms = MinMaxScaler()
df_dragon[:] = mms.fit_transform(df_dragon)

with open("data/dragon-minmaxscaler.pickle", "wb") as f:
    pickle.dump(mms, f)
# -

# ### Cleanup and Impute

# No dragon info yet for these CIDs
no_dragon = snitz_cids.difference(df_dragon.index)
no_dragon

# +
# Remove these from the Snitz data
df_snitz_dragon = df_dragon.loc[snitz_cids.difference(no_dragon)]

for nd in no_dragon:
    df_snitz_dragon.loc[nd, :] = 0

# +
# Remove bad features (too many NaNs) and impute remaining NaNs
frac_bad = df_snitz_dragon.isnull().mean()
good = frac_bad[frac_bad < 0.3].index
df_snitz_dragon = df_snitz_dragon.loc[:, good]

knn = KNN(k=5)
df_snitz_dragon[:] = knn.fit_transform(df_snitz_dragon.values)

# +
# from olfactometer.odorants import from_cids
# pubchem_data = from_cids([int(x) for x in snitz_cids])
# pd.DataFrame.from_dict(pubchem_data).set_index('CID').to_csv('data/snitz-odorant-info.csv')

# +
# df_snitz_mordred = pd.read_csv('data/snitz-mordred.csv').set_index('CID')
# df_snitz_mordred[:] = mms.fit_transform(df_snitz_mordred.values)
# df_snitz_mordred.head()
# -

df_snitz_features = df_snitz_dragon

# Normalize every molecule to have unit norm (to be unit vector in feature space)
nmr = Normalizer()
df_snitz_features[:] = nmr.fit_transform(df_snitz_features)


def get_unit_distance(row):
    """Convert feature vectors to unit vectors, summing across odorants if needed
    and then getting the vector difference, which will be related to the cosine of
    of the angle between them"""
    a, b, similarity = row
    if isinstance(a, str):
        a = [int(x) for x in a.split(",")]
        b = [int(x) for x in b.split(",")]
    A = df_snitz_features.loc[a, :].values
    B = df_snitz_features.loc[b, :].values
    if A.ndim > 1:
        A = A.sum(axis=0)
        B = B.sum(axis=0)
        A /= np.linalg.norm(A)
        B /= np.linalg.norm(B)
    return pd.Series(np.abs(A - B), index=df_snitz_features.columns, name=row.name)


df_distance = pd.concat([df1, df2, df3]).reset_index(drop=True)
features = list(df_snitz_features.columns)
unit_distances = df_distance.apply(get_unit_distance, axis=1)
df_distance = df_distance.join(df_distance.apply(get_unit_distance, axis=1))
df_distance.loc[:, "Similarity"] /= 100
df_distance.head()

# %matplotlib inline

model = Lasso(alpha=1e-4, max_iter=1e5)
X = df_distance[features]
y = df_distance["Similarity"]
model.fit(X, y)

plt.plot(1 + np.arange(len(model.coef_)), sorted(np.abs(model.coef_))[::-1])
plt.xscale("log")

# +


def r_score(model, X, y_true):
    y_pred = model.predict(X)
    # print(y_true.shape, y_pred.shape)
    return np.corrcoef(y_true, y_pred)[0, 1]


alphas = np.logspace(-5, -2, 9)
n_splits = 25
cv = ShuffleSplit(n_splits=n_splits, test_size=0.2)
training = np.zeros((len(alphas), n_splits))
testing = np.zeros((len(alphas), n_splits))

for i, alpha in enumerate(alphas):
    print(alpha)
    model = Lasso(alpha=alpha, max_iter=1e5)
    fff = cross_validate(model, X, y, cv=cv, return_train_score=True, scoring=r_score)
    training[i, :] = fff["train_score"]
    testing[i, :] = fff["test_score"]
# -

plt.errorbar(alphas, training.mean(axis=1), yerr=training.std(axis=1), label="Train")
plt.errorbar(alphas, testing.mean(axis=1), yerr=testing.std(axis=1), label="Test")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("R")
plt.legend()

model = Lasso(alpha=1e-4, max_iter=1e5)
model.fit(X, y)

snitz_space_weights = pd.Series(model.coef_, index=features, name="Weight")
snitz_space_weights = snitz_space_weights[np.abs(snitz_space_weights) > 1e-5]
snitz_space_weights

snitz_space_weights.to_csv("data/snitz_dragon_weights.csv", header=True)
