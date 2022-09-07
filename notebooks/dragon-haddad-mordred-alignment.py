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

import numpy as np
import pandas as pd

# Fix the file that Emily sent me
"""
df = pd.read_csv('data/CIDS_dragon.txt', delimiter='\t')
df['CID'] = df['NAME'].apply(lambda x:x.split(': ')[0])
df['SMILES'] = df['NAME'].apply(lambda x:x.split(': ')[1])
df = df.set_index('CID')
df = df[['SMILES'] + list(df)[2:-1]] # Move SMILES to front and drop NAME and No.
df.index = df.index.astype(int)
df = df.sort_index()
df.head()
df.to_csv('data/cids-smiles-dragon.txt')
"""

dfs = {}
dfs["dragon"] = pd.read_csv("data/cids-smiles-dragon.txt").set_index("CID")
dfs["mordred"] = pd.read_csv("data/mordred-features.csv").set_index("CID")

dfs["dragon"].head()

dfs["mordred"].head()

# Only one molecule has mordred features but no dragon features
dfs["mordred"].index.difference(dfs["dragon"].index)

dfs["mordred"] = dfs["mordred"].loc[dfs["dragon"].index].iloc[:, 3:]
dfs["mordred"].head()

dfs["dragon"] = dfs["dragon"].iloc[:, 1:]
dfs["dragon"].head()

from sklearn.preprocessing import StandardScaler

ss = {}
for kind in ["dragon", "mordred"]:
    ss[kind] = StandardScaler()
    good = dfs[kind].columns[dfs[kind].isnull().sum() < 500]
    df = dfs[kind][good]
    scaled = ss[kind].fit_transform(df.astype("float"))
    dfs[kind + "_good"] = pd.DataFrame(scaled, index=df.index, columns=df.columns)

from fancyimpute import KNN

knns = {}
for kind in ["dragon", "mordred"]:
    knns[kind] = KNN(k=5)
    df = dfs[kind + "_good"]
    imputed = knns[kind].fit_transform(df.values)
    dfs[kind + "_imputed"] = pd.DataFrame(imputed, index=df.index, columns=df.columns)

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(dfs["dragon_imputed"].values, dfs["mordred_imputed"].values[:, :])

predicted = lasso.predict(dfs["dragon_imputed"])
observed = dfs["mordred_imputed"]
rs = np.zeros(observed.shape[1])
for i, col in enumerate(observed):
    rs[i] = np.corrcoef(observed[col], predicted[:, i])[0, 1]

# %matplotlib inline
import matplotlib.pyplot as plt

plt.plot(sorted(sorted(rs)))

plt.plot(np.linspace(0, 1, len(lasso.coef_.ravel())), sorted(np.abs(lasso.coef_.ravel()))[::-1])
plt.xscale("log")
plt.xlabel("Quantile rank (Top X% of coefficients)")
plt.ylabel("Absolute value of coefficient")
