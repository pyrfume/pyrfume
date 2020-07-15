from pyrfume.odorants import Solution, Compound, ChemicalOrder, \
                                  Vendor, Molecule
import quantities as pq
import unittest

class MakeSolutionTestCase(unittest.TestCase):
    def test_make_solution(self):

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

        light_mineral_oil.molecular_weight = 500 * pq.g / pq.mol

        sigma_aldrich = Vendor('Sigma Aldrich', 'http://www.sigma.com')

        solvent_order = ChemicalOrder(light_mineral_oil, sigma_aldrich, '')

        chemical_orders = [ChemicalOrder(beta_pinene, sigma_aldrich, ''),
                        ChemicalOrder(d_limonene, sigma_aldrich, ''),
                        ChemicalOrder(l_limonene, sigma_aldrich, '')]

        solvent = Compound(solvent_order, is_solvent=True)
        compounds = [Compound(chemical_order) for chemical_order in chemical_orders]
        n_odorants = len(compounds)

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
