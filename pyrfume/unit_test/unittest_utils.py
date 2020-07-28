from pyrfume.odorants import Compound, ChemicalOrder, Vendor, Molecule
from pyrfume import TriangleTest, Component, Mixture
from datetime import datetime

def get_substances(substance_type: str = "mixtures"):
    vendor = Vendor("Test_Vendor", "")

    cid_C4H8O2 = 8857
    cas_C4H8O2 = "141-78-6"
    molecule_C4H8O2 = Molecule(cid_C4H8O2, "Ethyl acetate", True)
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

    cid_HCl = 313
    cas_HCl = "7647-01-0"
    molecule_HCl = Molecule(cid_HCl, "Hydrochloric acid", True)
    chemical_order_HCl = ChemicalOrder(molecule_HCl, vendor, "part 2", 0.5, None)
    compound_HCl = Compound(chemical_order_HCl, "TEST", datetime.now, datetime.now, False)
    component_HCl = Component(cid_HCl, "HCl", cas_HCl, 0.5, compound_HCl)
    mixture_HCl = Mixture(2, [component_HCl])

    descriptors = {
        cas_HCl : ["HCl unique descriptor", "common descriptor"],
        "dravnieks" : ["HCl dravnieks descriptor", "common dravnieks descriptor"],
        "sigma_ff" : ["HCl sigma_ff descriptor", "common sigma_ff descriptor"]
    }
    component_HCl.set_descriptors('unittest source', descriptors)

    if substance_type == "mixtures":
        return (mixture_C4H8O2, mixture_HCl, mixture_C2H6O)
    elif substance_type == "components":
        return (component_C4H8O2, component_HCl, component_C2H6O)
    elif substance_type == "compounds":
        return (compound_C4H8O2, compound_HCl, compound_C2H6O)
    elif substance_type == "molecules":
        return (molecule_C4H8O2, molecule_HCl, molecule_C2H6O)
    elif substance_type == "chemical_order":
        return (chemical_order_molecule_C4H8O2, chemical_order_HCl, chemical_order_C2H6O)
