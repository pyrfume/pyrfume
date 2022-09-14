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

# ### Clean the extracted FIG pdf data

import re

import pyrfume
from pyrfume.cabinets import get_mainland
from pyrfume.odorants import get_cids

# Load the data extracted by Tabula using the "Stream" method
df = pyrfume.load_data("IFRA_FIG/ifra-fragrance-ingredient-glossary---oct-2019.csv")

df = df.reset_index()
df.head()

# These are the indices of overflow cells, which only contain the last few characters of the previous cell's molecule name
overflow_indices = df.index[df["CAS number"].isnull()]
overflow_indices

# Merge those last few characters into the previous cell's molecule name
for i in overflow_indices:
    df.loc[i - 1, "Principal name"] = "%s%s" % (
        df.loc[i - 1, "Principal name"],
        df.loc[i, "Principal name"],
    )

# Delete those overflow rows
df = df.loc[~df.index.isin(overflow_indices)]

# Fix problematic CAS numbers
for index, cas in df["CAS number"].items():
    if not re.match(r"[0-9]+-[0-9]+-[0-9]+", cas):
        print("Fixing %s" % cas)
        cas = cas.replace("(", "").replace(")", "")
        assert re.match(r"[0-9]+-[0-9]+-[0-9]+", cas)
        df.loc[index, "CAS number"] = cas

# + jupyter={"outputs_hidden": true}
# Get CIDs for these CAS numbers
# Many of these CAS numbers are for substances, not compounds, and so have SIDs instead (not yet supported)
cas_cids_dict = get_cids(df["CAS number"])
# -

# Add CIDs to the dataframe
for cas, cid in cas_cids_dict.items():
    df.loc[df["CAS number"] == cas, "CID"] = cid
# Convert CIDs to integers
df.loc[:, "CID"] = df.loc[:, "CID"].astype(int)
df.head()

# Use CID as the index and sort
df = df.set_index("CID").sort_index()
df.head()

pyrfume.save_data(df, "IFRA_FIG/ifra_fig.csv")

pyrfume.load_data("IFRA_FIG/ifra_fig.csv")


df_mainland = get_mainland()
len(set(df_mainland["CAS"]).intersection(df["CAS number"]))

len(df_mainland.index.intersection(df.index))

df[df.index.isin(df_mainland.index)]  #

x = df_mainland.join(df, how="inner")[
    ["CAS", "CAS number", "Primary descriptor", "Descriptor 2", "Descriptor 2"]
]

for cid in x.index:
    if cid > 0:
        y = x.loc[cid, "CAS"] != x.loc[cid, "CAS number"]
        if isinstance(y, bool):
            if y:
                print(cid)
        elif all(y):
            print(cid)

x.columns = ["Mainland CAS", "FIG CAS", "Primary descriptor", "Descriptor 2", "Descriptor 3"]

x.to_csv("FIG-in-Mainland.csv")
