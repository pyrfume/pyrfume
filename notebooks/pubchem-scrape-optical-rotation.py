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

import pyrfume
from pyrfume import pubchem

results = pubchem.get_results("Optical+Rotation")

path = "pubchem_optical_rotation/physics.pkl"
pyrfume.save_data(results, path)
