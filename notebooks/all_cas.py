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
import numpy as np
import pandas as pd
import pyrfume
from pyrfume.odorants import cids_to_cas

cids = pyrfume.load_data('odorants/all_cids.csv').index

cas = cids_to_cas(cids)

print("Out of %d molecules, %d have CAS numbers" % (len(cids), len([x for x in cas.values() if x])))

counts = pd.Series([len(x) for x in cas.values()]).value_counts()
counts.index.name = 'Number of unique CAS values'
counts.name = 'Number of molecules'
counts.to_frame()

to_save = pd.Series(cas)
to_save.index.name = 'CID'
to_save.name = 'CAS'
to_save.head()

pyrfume.save_data(to_save.to_frame(), 'odorants/cid_to_cas.csv')
