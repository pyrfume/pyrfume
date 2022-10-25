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

# %load_ext autoreload
# %autoreload 2
import pickle

import pandas as pd
from rickpy import get_sheet

import pyrfume

cids = {}
DATA = pyrfume.DATA_DIR

# ## From Sigma Fragrance and Flavor Catalog (2014)

# Create a new file with CIDs and store here in `cids` dictionary
file_path = DATA / "sigma_2014" / "sigma.csv"
df = pd.read_csv(file_path)
cids["sigma-2014"] = set(df["CID"]) - {0}

# ## Dravnieks Atlas of Odor Character

# Create a new file with CIDs and store here in `cids` dictionary
file_path = DATA / "dravnieks_1985" / "dravnieks.csv"
df = pd.read_csv(file_path)
cids["dravnieks-1985"] = set(df["CID"]) - {0}

# ## Abraham et al, 2013

file_path = DATA / "abraham_2011" / "abraham-2011-with-CIDs.csv"
df = pd.read_csv(file_path)
cids["abraham-2013"] = set(df["CID"]) - {0}

# ## Bushdid et al, 2014

# Create a new file with CIDs and store here in `cids` dictionary
file_path = DATA / "bushdid_2014" / "bushdid.csv"
df = pd.read_csv(file_path)
cids["bushdid-2014"] = set(df["CID"]) - {0}

# ## Chae et al, 2019

# Create a new file with CIDs and store here in `cids` dictionary
file_path = DATA / "chae_2019" / "odorants.csv"
df = pd.read_csv(file_path)
cids["chae-2019"] = set(df["CID"]) - {0}

# ## Prestwick

file_path = DATA / "prestwick" / "prestwick.csv"
df = pd.read_csv(file_path)
cids["prestwick"] = set(df["CID"]) - {0}

# ## GRAS

file_path = DATA / "GRAS" / "gras.csv"
df = pd.read_csv(file_path)
cids["gras"] = set(df["CID"]) - {0}

# ## Sobel Lab (Weiss 2012, Snitz 2013)

file_path = DATA / "snitz_2013" / "snitz.csv"
df = pd.read_csv(file_path)
cids["sobel-2013"] = set(df["CID"]) - {0}

# ## Leffingwell

file_path = DATA / "westeros" / "westeros.csv"
df = pd.read_csv(file_path)
cids["leffingwell"] = set(df["CID"]) - {0}

# ## Davison

file_path = DATA / "davison_2007" / "davison-katz.csv"
df = pd.read_csv(file_path, index_col=0)
cids["davison-2007"] = set(df["CID"]) - {0}

# ## FDB

file_path = DATA / "fragrancedb" / "FragranceDB_CIDs.txt"
df = pd.read_csv(file_path)
cids["fragrance-db"] = set(df["CID"]) - {0}

# ## Mainland

file_path = DATA / "cabinets" / "Mainland Odor Cabinet with CIDs.csv"
df = pd.read_csv(file_path)
cids["mainland-cabinet"] = set(df["CID"]) - {0}

file_path = DATA / "mainland_intensity" / "mainland-intensity-odorant-info.csv"
df = pd.read_csv(file_path)
cids["mainland-intensity"] = set(df["CID"]) - {0}

file_path = DATA / "mainland_2015" / "Odors.tsv"
df = pd.read_csv(file_path, sep="\t")
cids["mainland-receptors"] = set(df["CID"].dropna().astype(int)) - {0}

# ## Enantiomers

file_path = DATA / "shadmany" / "enantiomers.csv"
df = pd.read_csv(file_path)
cids["enantiomers"] = set(df["CID"]) - {0}

# ## Haddad (just the clusters)

file_path = DATA / "haddad_2008" / "haddad-clusters.csv"
df = pd.read_csv(file_path)
cids["haddad-2008"] = set(df["CID"]) - {0}

# ## U19 PIs


gerkin_sheet = "1PlU4zHyRXtcI7Y-O6xYtlIyKoKk8hX1I9zfx8KFELdc"
u19_sheet = "1B2sEj9pCk2_zS1X1Cg2ulAB4E_BWPboJBSvH4Gwc8fs"
dfs = {}
dfs["gerkin-cabinet"] = get_sheet(gerkin_sheet, "gerkin-compounds").set_index("CID")
dfs["smith-cabinet"] = get_sheet(gerkin_sheet, "smith-compounds").set_index("CID")
dfs["rinberg-glomeruli"] = get_sheet(u19_sheet, "rinberg").set_index("CID")
dfs["fleischmann-cabinet"] = get_sheet(u19_sheet, "fleischmann").set_index("CID")
dfs["datta-cabinet"] = get_sheet(u19_sheet, "datta").set_index("CID")
dfs["bozza-cabinet"] = get_sheet(u19_sheet, "bozza").set_index("CID")

for name, df in dfs.items():
    cids[name] = set(df.index) - {0}

# ## Goodscents

file_path = DATA / "goodscents" / "goodscents_cids.txt"
df = pd.read_csv(file_path, index_col=False)
cids["goodscents"] = set(df["CID"]) - {0}

# ## Arctander

file_path = DATA / "arctander_1960" / "arctander_cids.txt"
df = pd.read_csv(file_path, index_col=False)
cids["arctander-1960"] = set(df["CID"]) - {0}

# ## Flavornet

file_path = DATA / "flavornet" / "flavornet.csv"
df = pd.read_csv(file_path)
cids["flavornet"] = set(df["CID"]) - {0}

# ## Scott et al, 2014

file_path = DATA / "scott_2014" / "data.csv"
df = pd.read_csv(file_path)
cids["scott-2014"] = set(df["CID"]) - {0}

# ## Superscent

file_path = DATA / "superscent" / "superscent_cids.txt"
df = pd.read_csv(file_path)
cids["superscent"] = set(df["CID"]) - {0}

# ## SenseLab

file_path = DATA / "senselab" / "senselab.csv"
df = pd.read_csv(file_path)
cids["senselab"] = set(df["CID"]) - {0}

file_path = DATA / "wakayama_2019" / "wakayama-intensity_with-CIDs.txt"
df = pd.read_csv(file_path, sep="\t")
cids["wakayama-2019"] = set(df["CID"]) - {0}

# ## Save

file_path = DATA / "odorants" / "cids.pkl"
with open(file_path, "wb") as f:
    pickle.dump(cids, f)

# ## Load

# +
# with open(file_path, 'rb') as f:
#    cids2 = pickle.load(f)
# -

# ## Merge

all_cids = set()
for key in cids:
    all_cids |= cids[key]
all_cids = pd.DataFrame(index=sorted(list(all_cids)), columns=sorted(list(cids))).fillna(0)
all_cids.index.name = "CID"
for key in cids:
    all_cids.loc[list(cids[key]), key] = 1
file_path = DATA / "odorants" / "all_cids.csv"
all_cids.to_csv(file_path)

all_cids.shape
