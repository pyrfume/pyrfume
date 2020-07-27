from pyrfume.odorants import Compound, ChemicalOrder, Vendor, Molecule
from pyrfume import TriangleTest, Component, Mixture
from datetime import datetime

def get_substances(substance_type: str = "mixtures"):
    vendor = Vendor("Test_Vendor", "")

    cid_CaCl2 = 24854
    cas_CaCl2 = "10043-52-4"
    molecule_CaCl2 = Molecule(cid_CaCl2, "Calcium chloride", True)
    chemical_order_molecule_CaCl2 = ChemicalOrder(molecule_CaCl2, vendor, "part 0", 0.5, None)
    compound_CaCl2 = Compound(chemical_order_molecule_CaCl2, "TEST", datetime.now, datetime.now, False)
    component_CaCl2 = Component(cid_CaCl2, "CaCl2", cas_CaCl2, 0.5, compound_CaCl2)
    mixture_CaCl2 = Mixture(2, [component_CaCl2])
    descriptors = {
        cas_CaCl2 : ["CaCl2 unique descriptor", "common descriptor"],
        "dravnieks" : ["CaCl2 dravnieks descriptor", "common dravnieks descriptor"],
        "sigma_ff" : ["CaCl2 sigma_ff descriptor", "common sigma_ff descriptor"]
    }
    component_CaCl2.set_descriptors('unittest source', descriptors)

    cid_NaCl = 5234
    cas_NaCl = "7647-14-5"
    molecule_NaCl = Molecule(cid_NaCl, "Sodium chloride", True)
    chemical_order_NaCl = ChemicalOrder(molecule_NaCl, vendor, "part 1", 0.5, None)
    compound_NaCl = Compound(chemical_order_NaCl, "TEST", datetime.now, datetime.now, False)
    component_NaCl = Component(cid_NaCl, "NaCl", cas_NaCl, 0.5, compound_NaCl)
    mixture_NaCl = Mixture(2, [component_NaCl])
    descriptors = {
        cas_NaCl : ["NaCl unique descriptor", "common descriptor"],
        "dravnieks" : ["NaCl dravnieks descriptor", "common dravnieks descriptor"],
        "sigma_ff" : ["NaCl sigma_ff descriptor", "common sigma_ff descriptor"]
    }
    component_NaCl.set_descriptors('unittest source', descriptors)

    cid_HCl = 313
    cas_HCl = "7647-01-0"
    molecule_HCl = Molecule(cid_HCl, "Hydrochloric acid", True)
    chemical_order_HCl = ChemicalOrder(molecule_HCl, vendor, "part 1", 0.5, None)
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
        return (mixture_CaCl2, mixture_HCl, mixture_NaCl)
    elif substance_type == "components":
        return (component_CaCl2, component_HCl, component_NaCl)
    elif substance_type == "compounds":
        return (compound_CaCl2, compound_HCl, compound_NaCl)
    elif substance_type == "molecules":
        return (molecule_CaCl2, molecule_HCl, molecule_NaCl)
    elif substance_type == "chemical_order":
        return (chemical_order_molecule_CaCl2, chemical_order_HCl, chemical_order_NaCl)
