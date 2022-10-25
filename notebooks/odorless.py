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

# +
import pandas as pd

import pyrfume

# Read file sent by Emily Mayhew on Sept. 23, 2019
df = pd.read_csv("u19predictions.csv")
# -

df["CID"] = df["SMILEstring"].apply(lambda x: x.split(": ")[0])
df["SMILES"] = df["SMILEstring"].apply(lambda x: x.split(": ")[1])
df["PredictedOdorless"] = df["Prediction"] == "Odorless"
predicted_odorless = df.set_index("CID")["PredictedOdorless"]

pyrfume.save_data(predicted_odorless.to_frame(), "odorants/predicted_odorless.csv")
