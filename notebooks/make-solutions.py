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
# #%load_ext line_profiler

from pyrfume.odorants import Solution, Compound, ChemicalOrder, \
                                  Vendor, Molecule
import quantities as pq

# +
# Instantiate two molecules by CID number
beta_pinene = Molecule(14896, fill=True)  # Fruity smell
d_limonene = Molecule(440917, fill=True)  # Lemon smell (check)
l_limonene = Molecule(439250, fill=True)  # Orange smell (check)
light_mineral_oil = Molecule(347911206, fill=True)  # An odorless solvent

# Vapor pressures at 25 degrees Celsius (obtained from PubChem)
beta_pinene.vapor_pressure = 10 * pq.mmHg # Made up, fill with real values
d_limonene.vapor_pressure = 10 * pq.mmHg # Made up, fill with real values
l_limonene.vapor_pressure = 10 * pq.mmHg # Made up, fill with real values
light_mineral_oil.vapor_pressure = 0 * pq.mmHg  # Actually .0001 * pq.hPa

# Densities at 20 degrees Celsius (obtained from PubChem)
beta_pinene.density = 1.0 * pq.g/pq.cc # Made up, fill with real values
d_limonene.density = 1.0 * pq.g/pq.cc # Made up, fill with real values
l_limonene.density = 1.0 * pq.g/pq.cc # Made up, fill with real values
light_mineral_oil.density = 0.85 * pq.g/pq.cc # Made up, fill with real values

# Mineral oil does not have a proper molecular weight since
# it is itself a mixture, but we have to put something reasonable
# in order compute mole fraction of the solute
light_mineral_oil.molecular_weight = 500 * pq.g / pq.mol

# +
# Create a vendor
sigma_aldrich = Vendor('Sigma Aldrich', 'http://www.sigma.com')

# Specify two chemicals ordered from this vendor,
# which are nominally the above molecules
solvent_order = ChemicalOrder(light_mineral_oil, sigma_aldrich, '')
#chemical_orders = [ChemicalOrder(hexanal, sigma_aldrich, ''),
#                   ChemicalOrder(isoamyl_acetate, sigma_aldrich, '')]
chemical_orders = [ChemicalOrder(beta_pinene, sigma_aldrich, ''),
                   ChemicalOrder(d_limonene, sigma_aldrich, ''),
                   ChemicalOrder(l_limonene, sigma_aldrich, '')]

# These are now actual chemical on the shelf, with potential stock numbers
# dates arrived, dates opened.
solvent = Compound(solvent_order, is_solvent=True)
compounds = [Compound(chemical_order) for chemical_order in chemical_orders]
n_odorants = len(compounds)

# Two odorants stocks that we produced by diluting the compounds above
# TODO: An optimizer to determine what these solutions should be
solutions = []
solutions.append(Solution({compounds[0]: 1 * pq.mL,
                           solvent: 24 * pq.mL}))
solutions.append(Solution({compounds[1]: 1 * pq.mL,
                           solvent: 24 * pq.mL}))
solutions.append(Solution({compounds[2]: 1 * pq.mL,
                           solvent: 24 * pq.mL}))
solutions.append(Solution({compounds[0]: 0.01 * pq.mL,
                           solvent: 24 * pq.mL}))
solutions.append(Solution({compounds[1]: 0.01 * pq.mL,
                           solvent: 24.9 * pq.mL}))
solutions.append(Solution({compounds[2]: 0.01 * pq.mL,
                           solvent: 24.9 * pq.mL}))
n_solutions = len(solutions)
