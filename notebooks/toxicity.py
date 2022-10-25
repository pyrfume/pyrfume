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
import matplotlib.pyplot as plt
import pandas as pd
from rickpy import ProgressBar

import pyrfume
from pyrfume.pubchem import get_ghs_classification, parse_ghs_classification_for_odor

path = "odorants/all_cids_properties.csv"
details = pyrfume.load_data(path, usecols=range(5))
details.head()

# ### Cramer Toxicity Class Predictions

tox = pyrfume.load_data("odorants/toxTree.csv")
cramer = details.join(tox, on="SMILES")["Cramer Class"]
cramer = cramer.apply(lambda x: len(x.split(" ")[-1][:-1]))
cramer.head()

pyrfume.save_data(cramer.to_frame(), "odorants/cramer.csv")

embedded_coords = {
    key: pyrfume.load_data("odorants/%s_umap.pkl" % key) for key in ("snitz", "haddad")
}


# +
def plot_tox(space, ax):
    coords = embedded_coords[space].join(cramer)
    color_dict = {1: "gray", 2: "green", 3: "red"}
    colors = [color_dict[n] for n in coords["Cramer Class"]]
    ax.scatter(*coords[["X", "Y"]].values.T, color=colors, s=0.5, alpha=0.5)


fig, ax = plt.subplots(1, 2, figsize=(20, 10))
for i, space in enumerate(("snitz", "haddad")):
    plot_tox(space, ax[i])
    ax[i].set_title(space)
plt.suptitle("Cramer Class Toxicity: Red=III, Green=II, Gray=I")
plt.savefig("tox.png", dpi=400)
# -

# ### Empirical Toxicity Information

# Download all toxicity information using PubChem
tox_dict = {}
n_odorants = details.shape[0]
p = ProgressBar(n_odorants)
for i, CID in enumerate(details.index):
    p.animate(i)
    ghs_info = get_ghs_classification(CID)
    strings = parse_ghs_classification_for_odor(
        ghs_info,
        codes=["H330", "H331", "H334", "H340", "H350", "H350i", "H351", "H36", "H37"],
        only_percent=True,
        code_only=True,
    )
    if strings:
        tox_dict[CID] = strings

# Reformat into a dataframe
df_tox = pd.DataFrame(columns=["CID", "Code", "%"])
index = 0
for cid, info in tox_dict.items():
    for x in info:
        code, other = x.split(" ")
        num = float(other[1:-2])
        # print(cid, code, num)
        df_tox.loc[index] = [cid, code, num]
        index += 1
df_tox

empirical_tox = details.copy()
for key, value in df_tox.groupby(["CID", "Code"]).mean()["%"].items():
    cid, code = key
    empirical_tox.loc[cid, code] = value
empirical_tox = empirical_tox.fillna(0)
empirical_tox.head()

pyrfume.save_data(empirical_tox, "odorants/hcode_tox.csv")
