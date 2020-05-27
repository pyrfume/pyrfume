"""
dravieks.txt produced by copying and pasting from the supplemental material of
"In search of the structure of human olfactory space", e.g. koulakov.pdf.
"""

import os

import pyrfume

DRAV_PATH = os.path.join(pyrfume.DATA_DIR, "dravnieks")


def get_data():
    cas_path = os.path.join(DRAV_PATH, "dravnieks_cas.txt")
    with open(cas_path, "r") as f:
        text = f.read()
    lines = text.split("\n")[1:]
    # A list of CAS numbers.
    cas = [line.split(" ")[1] for line in lines]

    descriptor_path = os.path.join(DRAV_PATH, "dravnieks_descriptors.txt")
    with open(descriptor_path, "r") as f:
        text = f.read()
    lines = text.split("\n")[1:]
    # A list of descriptors.
    descriptors = [" ".join(line.split(" ")[1:]) for line in lines]

    data_path = os.path.join(DRAV_PATH, "dravnieks_data.txt")
    with open(data_path, "r") as f:
        text = f.read()
    lines = text.split("\n")[:-1]
    data = {}
    for i, line in enumerate(lines):
        values = line.split("\t")
        data[cas[i]] = {descriptors[j]: float(value) for j, value in enumerate(values)}

    return cas, descriptors, data


if __name__ == "__main__":
    cas, descriptors, data = get_data()
