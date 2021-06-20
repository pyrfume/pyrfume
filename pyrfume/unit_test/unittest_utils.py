from pyrfume.odorants import Compound, ChemicalOrder, Vendor, Molecule
from pyrfume.experiments import TriangleTest
from pyrfume.objects import Component, Mixture
from datetime import datetime
import quantities as pq


def get_water(substance_type: str = "mixtures"):
    vendor = Vendor("Test_Vendor", "")

    cid_H2O = 962
    cas_H2O = "7732-18-5"
    molecule_H2O = Molecule(cid_H2O, "Water", True)
    molecule_H2O.density = 	0.902 * pq.g / pq.mL
    molecule_H2O.molecular_weight = 	88.106 * pq.g / pq.mol
    molecule_H2O.cas = cas_H2O
    chemical_order_molecule_H2O = ChemicalOrder(molecule_H2O, vendor, "part 0", 0.5, None)
    compound_H2O = Compound(chemical_order_molecule_H2O, "TEST", datetime.now, datetime.now, True)
    component_H2O = Component(cid_H2O, "H2O", cas_H2O, 0.5, compound_H2O)
    mixture_H2O = Mixture(2, [component_H2O])
    descriptors = {
        cas_H2O : ["H2O unique descriptor", "common descriptor"],
        "dravnieks" : ["H2O dravnieks descriptor", "common dravnieks descriptor"],
        "sigma_ff" : ["H2O sigma_ff descriptor", "common sigma_ff descriptor"]
    }
    component_H2O.set_descriptors('unittest source', descriptors)

    if substance_type == "mixtures":
        return mixture_H2O
    elif substance_type == "components":
        return component_H2O
    elif substance_type == "compounds":
        return compound_H2O
    elif substance_type == "molecules":
        return molecule_H2O
    elif substance_type == "chemical_order":
        return chemical_order_molecule_H2O


def get_substances(substance_type: str = "mixtures"):
    vendor = Vendor("Test_Vendor", "")

    cid_C4H8O2 = 8857
    cas_C4H8O2 = "141-78-6"
    molecule_C4H8O2 = Molecule(cid_C4H8O2, "Ethyl acetate", True)
    molecule_C4H8O2.density = 	0.902 * pq.g / pq.mL
    molecule_C4H8O2.molecular_weight = 	88.106 * pq.g / pq.mol
    molecule_C4H8O2.vapor_pressure = 12.425 * pq.kPa
    molecule_C4H8O2.cas = cas_C4H8O2
    chemical_order_molecule_C4H8O2 = ChemicalOrder(molecule_C4H8O2, vendor, "part 0", 0.5, None)
    compound_C4H8O2 = Compound(chemical_order_molecule_C4H8O2, "TEST", datetime.now, datetime.now, False)
    component_C4H8O2 = Component(cid_C4H8O2, "C4H8O2", cas_C4H8O2, 0.5, compound_C4H8O2)
    mixture_C4H8O2 = Mixture(2, [component_C4H8O2])
    descriptors = {
        cas_C4H8O2 : ["C4H8O2 unique descriptor", "common descriptor"],
        "dravnieks" : ["C4H8O2 dravnieks descriptor", "common dravnieks descriptor"],
        "sigma_ff" : ["C4H8O2 sigma_ff descriptor", "common sigma_ff descriptor"]
    }
    component_C4H8O2.set_descriptors('unittest source', descriptors)

    cid_C2H6O = 702
    cas_C2H6O = "64-17-5"
    molecule_C2H6O = Molecule(cid_C2H6O, "Ethanol", True)
    molecule_C2H6O.density = 0.7893 * pq.g / pq.mL
    molecule_C2H6O.molecular_weight = 46.069 * pq.g / pq.mol
    molecule_C2H6O.vapor_pressure = 7.906 * pq.kPa
    molecule_C2H6O.cas = cas_C2H6O
    chemical_order_C2H6O = ChemicalOrder(molecule_C2H6O, vendor, "part 1", 0.5, None)
    compound_C2H6O = Compound(chemical_order_C2H6O, "TEST", datetime.now, datetime.now, False)
    component_C2H6O = Component(cid_C2H6O, "C2H6O", cas_C2H6O, 0.5, compound_C2H6O)
    mixture_C2H6O = Mixture(2, [component_C2H6O])
    descriptors = {
        cas_C2H6O : ["C2H6O unique descriptor", "common descriptor"],
        "dravnieks" : ["C2H6O dravnieks descriptor", "common dravnieks descriptor"],
        "sigma_ff" : ["C2H6O sigma_ff descriptor", "common sigma_ff descriptor"]
    }
    component_C2H6O.set_descriptors('unittest source', descriptors)

    cid_C4H8S = 1127
    cas_C4H8S = "110-01-0"
    molecule_C4H8S = Molecule(cid_C4H8S, "Tetrahydrothiophene", True)
    molecule_C4H8S.density = 0.997 * pq.g / pq.mL
    molecule_C4H8S.molecular_weight = 88.17 * pq.g / pq.mol
    molecule_C4H8S.vapor_pressure = 2.4 * pq.kPa
    chemical_order_C4H8S = ChemicalOrder(molecule_C4H8S, vendor, "part 2", 0.5, None)
    compound_C4H8S = Compound(chemical_order_C4H8S, "TEST", datetime.now, datetime.now, False)
    component_C4H8S = Component(cid_C4H8S, "C4H8S", cas_C4H8S, 0.5, compound_C4H8S)
    mixture_C4H8S = Mixture(2, [component_C4H8S])

    descriptors = {
        cas_C4H8S : ["C4H8S unique descriptor", "common descriptor"],
        "dravnieks" : ["C4H8S dravnieks descriptor", "common dravnieks descriptor"],
        "sigma_ff" : ["C4H8S sigma_ff descriptor", "common sigma_ff descriptor"]
    }
    component_C4H8S.set_descriptors('unittest source', descriptors)

    if substance_type == "mixtures":
        return (mixture_C4H8O2, mixture_C4H8S, mixture_C2H6O)
    elif substance_type == "components":
        return (component_C4H8O2, component_C4H8S, component_C2H6O)
    elif substance_type == "compounds":
        return (compound_C4H8O2, compound_C4H8S, compound_C2H6O)
    elif substance_type == "molecules":
        return (molecule_C4H8O2, molecule_C4H8S, molecule_C2H6O)
    elif substance_type == "chemical_order":
        return (chemical_order_molecule_C4H8O2, chemical_order_C4H8S, chemical_order_C2H6O)
