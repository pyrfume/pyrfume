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
# Upgrade to latest version of Pyrfume
# #!pip install --user -U git+https://github.com/pyrfume/pyrfume
    
# And be sure to have pyrfume/data/cabinets/mainland.csv in place
# -

from itertools import combinations
import numpy as np
import pandas as pd
from pyrfume.cabinets import get_mainland
from pyrfume.optimization import OdorantSetOptimizer, get_coverage, get_entropy, get_spacing
from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity

# Get contents of Mainland cabinet (any dataframe could do)
cabinet = get_mainland()
# Get rid of any molecules without a valid SMILES stting
cabinet = cabinet.dropna(subset=['SMILES'])
# rdkit can't make mol from this one so drop that, too
cabinet = cabinet.drop(24766)
cids = list(cabinet.index)
cabinet.head()

# Compute all fingerprints
fps = cabinet['SMILES'].apply(Chem.MolFromSmiles).apply(Chem.RDKFingerprint)
# And then Tanimot distances
tanimoto = pd.DataFrame(index=cids, columns=cids)
tanimoto[:] = [[FingerprintSimilarity(fp1, fp2) for fp1 in fps] for fp2 in fps]
tanimoto.head()


# This could be any function whose first argument is integer indices (not CIDs!) into a dataframe
# of info about odorants (one odorant per row)
def mean_dist(indices, sim):
    """Return the summed Tanimoto distance of all pairs"""
    pairs = combinations(indices, 2)
    return np.mean([sim.iloc[x[0], x[1]] for x in pairs])


# Prettier printing
np.set_printoptions(precision=2, suppress=True)
# Some weight you can make up, each tuple is one weight under consideration.  
# Tuple item 1 is a cabinet dataframe column or a custom name
# Tuple item 2 is an operation of a function name and args to that function
# Tuple item 3 is the actual weight (make positive to maximize, negative to minimize)
weights = [('$/mol', 'mean', -5),
           ('MW', 'sum', 3),
           ('Tanimoto', (mean_dist, tanimoto), 1)]
# How many items you want in your set, e.g. how many odorants should the result contain?
n_desired = 25
# Create the optimizer.
# n_gen is the number of generations of optimization
# rescale_weights standardize all weights to be # of s.d. better/worse than random sets of items
optimizer = OdorantSetOptimizer(cabinet, n_desired, weights, n_gen=50, standardize_weights=True)
# Run the optimizer
pop, stats, hof, logbook = optimizer.run()

optimizer.plot_score_history()

# hof is the Hall of Fame, and contains the best 100 sets in descending order of fitness
# Get the very best one
best = hof[0]
# Show the first few items of this list, which is a subset of the original cabinet
cabinet.iloc[list(best)].head()


