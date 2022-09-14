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

import re
from pathlib import Path

import pandas as pd

import pyrfume
from pyrfume.odorants import get_cids

re.sub("l-(?![0-9])", "l ", "amyl-2-acetate")

p = Path(pyrfume.DATA)
old_names = []
new_names = []
for file in (p / "leon" / "3D_mol_files").glob("*.mol"):
    name = file.name.replace(".mol", "")
    old_names.append(name)
    # Replace underscore with hyphen
    name = name.replace("_", "-")
    # Remove extraneous hyphens
    name = re.sub(r"(?<![0-9\(])-(?![0-9])", " ", name)
    # Add back hyphens after prefixes
    for x in [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "tert",
        "L",
        "D",
        "m",
        "o",
        "p",
        "cis",
        "trans",
        "sec",
    ]:
        name = name.replace("%s " % x, "%s-" % x)
    # Move isomeric identifiers to the front of the name
    for x in ["(-)", "(+)"]:
        if x in name:
            name = "%s-%s" % (x, name.replace(x, ""))
    new_names.append(name)
    # print(name)

cids = get_cids(new_names)

df = pd.Series(cids, name="CID").to_frame()
df["Old Name"] = old_names
df.index.name = "Name"
df = df.reset_index()
df.head()

df[df["CID"] == 0]

df.loc[67, "CID"] = 19309
df.loc[76, "CID"] = 11160
df.loc[79, "CID"] = 8092
df.loc[155, "CID"] = 28500
df.loc[170, "CID"] = 251531

df = df.set_index("CID").sort_index()
df.head()

df.to_csv(p / "leon" / "leon_cids.csv", sep="\t")
