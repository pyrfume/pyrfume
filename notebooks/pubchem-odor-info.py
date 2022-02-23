# -*- coding: utf-8 -*-
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
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rickpy import get_sheet, ProgressBar
from olfactometer import odorants as o
sns.set(font_scale=1.5)

o.get_compound_odor(7945, raw=True);

o.get_compound_odor(8093)

o.get_compound_odor(7945)

# +
# Two Google sheets of odor cabinet data
smith_cabinet = get_sheet('1PlU4zHyRXtcI7Y-O6xYtlIyKoKk8hX1I9zfx8KFELdc', 'smith-compounds').set_index('CID')
gerkin_cabinet = get_sheet('1PlU4zHyRXtcI7Y-O6xYtlIyKoKk8hX1I9zfx8KFELdc', 'gerkin-compounds').set_index('CID')

# Concatenate
x = pd.concat([gerkin_cabinet[['ChemicalName']], smith_cabinet[['ChemicalName']]])

# Drop duplicate CIDs
x = x.groupby(level=0).first()

x.head()
# -

# Add odorant data from PubChem
p = ProgressBar(x.shape[0])
for i, CID in enumerate(x.index):
    p.animate(i)
    odor_info = o.get_compound_odor(CID)
    x.loc[CID, 'n_odor_mentions'] = int(len(odor_info))
    x.loc[CID, 'odor'] = ';'.join(odor_info)

# +
kinds = {}

# Which CIDs contain no odor info
kinds['nothing'] = x[x['odor']==''].index

# Which CIDs are claimed to be odorless
kinds['odorless'] = x[x['odor'].str.lower().str.contains('odorless')].index

# Which CIDs have some other kind of odor information
kinds['odorous'] = x[~x.index.isin(kinds['nothing']) &
                     ~x.index.isin(kinds['odorless'])].index

for kind in kinds:
    print("There are %d compounds in '%s'" % (len(kinds[kind]), kind))
# -

# Get basic molecular properties (MW and octanol/water partition coefficient)
records = o.from_cids(x.index, property_list=['MolecularWeight', 'XlogP'])

# Add these records to the dataframe
# Skip XLogP when it isn't found
for record in records:
    x.loc[record['CID'], 'MW'] = record['MolecularWeight']
    try:
        x.loc[record['CID'], 'XLogP'] = record['XLogP']
    except KeyError:
        print(record['name'])

plt.figure()
ax = plt.gca()
for color, kind in {'r':'nothing', 'g':'odorless', 'b':'odorous'}.items():
    x.loc[kinds[kind]].plot.scatter(x='MW', y='XLogP', alpha=0.5, c=color, ax=ax, label=kind)
plt.xscale('log')
plt.xlim(10,1000)
plt.ylim(-10,10)

# Add odorant boiling point data from PubChem
p = ProgressBar(x.shape[0])
for i, cid in enumerate(x.index):
    p.animate(i)
    z = o.get_compound_summary(cid, 'Boiling Point')
    if z and len(z):
        x.loc[cid, 'BP'] = str(o._parse_other_info(z)[0])

# Parse this into boiling points in Celsius
for cid, bp_raw in x['BP'].iteritems():
    if isinstance(bp_raw,str):#math.isnan(bp_raw):
        num = re.findall("[-+]?[0-9]*\.?[0-9]+", bp_raw)[0]
        units = None
        if any([s in bp_raw for s in ['° F','°F','deg F','DEG F']]):
            units = 'F' 
        if any([s in bp_raw for s in ['° C','°C','deg C','DEG C']]):
            units = 'C'
        #print(cid, num, units)
        if units == 'C':
            bp_C = float(num)
        elif units == 'F':
            bp_C = (5.0/9) * (float(num) - 40)
        if units:
            x.loc[cid, 'BP_C'] = bp_C

plt.figure()
ax = plt.gca()
for color, kind in {'r':'nothing', 'g':'odorless', 'b':'odorous'}.items():
    x.loc[kinds[kind]].plot.scatter(x='BP_C', y='XLogP', alpha=0.5, c=color, ax=ax, label=kind)
plt.xscale('log')
plt.xlim(10,1000)
plt.ylim(-10,10)
