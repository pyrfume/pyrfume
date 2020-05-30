import csv
import os
import pathlib

import pyrfume
from pyrfume import Component, Mixture, Result, TriangleTest

# Parameters for Bushdid et al, 2014.
DILUTION = {"1/4": 0.25, "1/2": 0.5, "not diluted": 1.0}
CORRECT = {"right": True, "wrong": False}
OVERLAPS = {10: [9, 6, 3, 0], 20: [19, 15, 10, 5, 0], 30: [29, 20, 10, 0]}
N_TESTS = 20
N_SUBJECTS = 26
C = 128
BUSHDID_PATH = pathlib.Path("bushdid_2014")


def load_data():
    path = BUSHDID_PATH / "Bushdid-tableS1.csv"
    df = pyrfume.load_data(path, encoding="latin1")
    df = df.iloc[:, :4]
    df.columns = ["Name", "CAS", "Dilution", "Solvent"]
    df = df.set_index("CAS")
    df = df[df.index.notnull()]
    df.head()
    return df


# Functions for loading data from Bushdid et al, 2014.
def load_odorants_tests_results(all_components):
    """
    Given all odor components, loads the odorants, tests, and test results
    from Supplemental Table 2 of Bushdid et al.
    """
    odorants = {}
    tests = {}
    results = []
    path = pyrfume.DATA_DIR / BUSHDID_PATH / "Bushdid-tableS2.csv"
    f = open(path, "r", encoding="latin1")
    reader = csv.reader(f)
    next(reader)
    row_num = 0
    for row in reader:
        uid, n, r, percent, dilution, correct = row[:6]
        component_names = [x for x in row[6:36] if len(x)]
        # The next line is required to account for inconsistent naming of one
        # of the components across the two supplemental tables.
        component_names = [
            x.replace("4-Methyl-3-penten-2-one", "4-methylpent-3-en-2-one") for x in component_names
        ]
        outcomes = row[36:62]
        if uid.isdigit():
            uid = int(uid)
            dilution = DILUTION[dilution]
            odorant_key = hash(tuple(component_names))
            if odorant_key not in odorants:
                components = [
                    component for component in all_components if component.name in component_names
                ]
                if len(components) not in [1, 10, 20, 30]:
                    # If an odorant has a number of components which is not
                    # either 1, 10, 20, or 30.
                    print(
                        uid, [x for x in component_names if x not in [y.name for y in components]]
                    )
                odorant = Mixture(C, components)
                odorant.name = odorant_key
            elif row_num % 3 == 0:
                # If any component is repeated across all the tests.
                print("Repeat of this odorant: %d" % odorant_key)
            odorants[odorant_key] = odorant
            if uid not in tests:
                tests[uid] = TriangleTest(uid, [], dilution, correct)
            test = tests[uid]
            test.add_odorant(odorant)
            if correct == "right":
                test.correct = tests[uid].odorants.index(odorant)
        if len(outcomes[0]):
            for i, outcome in enumerate(outcomes):
                result = Result(test, i + 1, CORRECT[outcome])
                results.append(result)
        row_num += 1
    return odorants, tests, results


def load_components():
    """
    Loads all odorant components from Supplemental Table 1 of Bushdid et al.
    """

    components = []
    path = os.path.join(BUSHDID_PATH, "Bushdid-tableS1.csv")
    f = open(path, "r", encoding="latin1")
    reader = csv.reader(f)
    next(reader)
    component_id = 0
    for row in reader:
        name, cas, percent, solvent = row[:4]
        if len(name):
            component = Component(component_id, name, cas, percent, solvent)
            components.append(component)
            component_id += 1
        else:
            break
    return components


def get_results():
    components = load_components()
    odorants, tests, results = load_odorants_tests_results(components)
    return results
