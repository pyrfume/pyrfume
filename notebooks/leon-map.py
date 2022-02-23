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

# %matplotlib inline
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ## Load and plot Decanal

plt.figure(figsize=(4.5,6))
ax = plt.gca()
df = pd.read_csv('data/MamloukCSV3/decanal_SG3.csv',header=1).astype(float).replace(-100.0,np.nan)
df.columns = range(df.shape[1])
sns.heatmap(df,ax=ax,cmap='RdBu_r',vmin=-5,vmax=5,cbar_kws={'label':'Z-score'},xticklabels='',yticklabels='')

print("There are %d total maps" % len(os.listdir('data/MamloukCSV3')))

# ## Load and plot the first 25 maps

fig,axes = plt.subplots(5,5,figsize=(12,16))
path = 'data/MamloukCSV3'
for i,file in enumerate(os.listdir(path)[:25]):
    ax = axes.flat[i]
    file = os.path.join(path,file)
    try:
        df = pd.read_csv(file,header=1)
        molecule = df.columns[0]
        df = df.astype(float).replace(-100.0,np.nan)
    except ValueError:
        df = pd.read_csv(file,header=2).astype(float).replace(-100.0,np.nan)
        molecule = '%s\n(%s)' % (molecule,df.columns[0].strip())
    df.columns = range(df.shape[1])
    sns.heatmap(df,ax=ax,cmap='RdBu_r',vmin=-5,vmax=5,cbar=None,
                cbar_kws={'label':'Z-score'},xticklabels='',yticklabels='')
    ax.set_title(molecule)
plt.tight_layout()
