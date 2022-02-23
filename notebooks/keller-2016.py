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
from pyrfume import keller

raw = keller.load_raw_bmc_data(only_dream_subjects=False,  # Whether to only keep DREAM subjects
                               only_dream_descriptors=False,  # Whether to only keep DREAM descriptors
                               only_dream_molecules=False)  # Whether to only keep DREAM molecules)
raw.head()

cooked = keller.format_bmc_data(raw)
cooked.head()

cooked.index = cooked.index.reorder_levels([1, 0, 2, 3]) # Put CID first

cooked = cooked.sort_index(level=0) # Sort by CID ascending

pyrfume.save_data(cooked, 'keller_2016/data.csv')
