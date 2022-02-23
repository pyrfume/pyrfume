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
import pandas as pd
import pyrfume
from pyrfume import features, snitz, haddad

# Load minmaxed, imputed Dragon features (cached from previous work) for all Pyrfume odorants
# (Alternatively, load raw Dragon features and apply `features.clean_dragon`.)
# Here we use minimax scaling instead of standard scaling because that is what the Snitz paper used.
minmax_scaled_dragon = features.load_dragon(suffix='-cleaned-minmaxed-imputed')
# Use the subset of features identified in Haddad and compute a cosine angle distance between each pair of odorants
distances['snitz'] = snitz.get_snitz_distances(minmax_scaled_dragon)
# Show the first 5 rows
distances['snitz'].head()

# Load scaled, imputed Dragon features (cached from previous work) for all Pyrfume odorants; 
# (Alternatively, load raw Dragon features and apply `features.clean_dragon`.)  
standard_scaled_dragon = features.load_dragon(suffix='-cleaned-standardized-imputed')
# Use the subset of features identified in Haddad and compute a Euclidean distance between each pair of odorants
distances['haddad'] = haddad.get_haddad_distances(standard_scaled_dragon)
# Show the first 5 rows
distances['haddad'].head()

# +
nondiagonal = distances['haddad'].values[np.triu_indices(distances['haddad'].shape[0], 1)]
density, bins, _ = plt.hist(nondiagonal, bins=np.linspace(0, 25, 100), density=True, cumulative=True)
shift = (bins[1]-bins[0])/2
haddad_density = pd.DataFrame(density, columns=['Cumulative Probability'], index=bins[:-1]+shift)
pyrfume.save_data(haddad_density, 'haddad_2008/haddad_cumulative_probability.csv')

nondiagonal = distances['snitz'].values[np.triu_indices(distances['snitz'].shape[0], 1)]
density, bins, _ = plt.hist(nondiagonal, bins=np.linspace(0, 0.5, 100), density=True, cumulative=True)
shift = (bins[1]-bins[0])/2
snitz_density = pd.DataFrame(density, columns=['Cumulative Probability'], index=bins[:-1]+shift)
pyrfume.save_data(snitz_density, 'snitz_2013/snitz_cumulative_probability.csv')
