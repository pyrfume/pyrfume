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

import json
import pandas as pd
import pyrfume
import re

all_statements = pyrfume.load_data("pubchem_scrape_100000.pkl")

import json

with open("pubchem_100000.json", "w") as f:
    json.dump(all_statements, f)

df = pd.DataFrame(index=sorted(all_statements), columns=["Odor", "Odorless", "Statements"])
df.index.name = "CID"
for cid in sorted(all_statements):
    statements = all_statements[cid]
    odor = False
    odorless = False
    for statement in statements:
        statement = statement.lower()
        if re.findall("no odor", statement):
            odorless = True
        elif re.findall("no odour", statement):
            odorless = True
        elif re.findall("no smell", statement):
            odorless = True
        elif re.findall("no fragrance", statement):
            odorless = True
        elif re.findall("odorless", statement):
            odorless = True
        elif re.findall("odourless", statement):
            odorless = True
        elif re.findall("odoratus", statement):
            pass
        elif re.findall("sense of smell", statement):
            odor = True
        elif re.findall("odor", statement):
            odor = True
        elif re.findall("odour", statement):
            odor = True
        elif re.findall("smell", statement):
            odor = True
        elif re.findall("fragrance", statement):
            odor = True
        elif re.findall("aroma ", statement):
            odor = True
        else:
            pass
    if odor and odorless:
        pass  # print(statements)
    df.loc[cid, :] = [odor, odorless, statements]

df.head()

pyrfume.save_data(df, "pubchem_scrape_100000.csv")
