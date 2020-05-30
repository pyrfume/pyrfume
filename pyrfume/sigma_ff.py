"""
sigma_ff_catalog.txt produced using pdf2txt.py found in PDFminer
on sigma_ff_catalog.pdf.
"""

import platform

import pyrfume

CATALOG_PATH = pyrfume.DEFAULT_DATA_PATH / "sigma" / "sigma_ff_catalog.txt"


def get_data():
    with open(CATALOG_PATH, "r") as f:
        text = f.read()
        lines = text.split("\n")

    data = {}
    organoleptic = 0
    for line_num, line in enumerate(lines):
        if len(line):
            if not organoleptic and line[0] == "[":
                key = line.split("]")[0][1:]
                if platform.python_version() > "3.0":
                    key = key.replace("\u2011", "-")
                else:
                    key = key.decode("utf-8").replace("\u2011", "-").encode("ascii")
                organoleptic = 1
            if organoleptic and "Organoleptic" in line:
                try:
                    value = line.split(":")[1][1:]
                    if value[-1] in ["-", ","]:
                        if value[-1] == "-":
                            value = value[:-1]
                        else:
                            value = value + " "
                        value += lines[line_num + 1]
                    value = [i.strip() for i in value.split(";") if len(i.strip())]
                    data[key] = value
                    organoleptic = 0
                except Exception:
                    pass

    print("%d compounds described." % len(data))

    descriptors = []
    for x in data.values():
        descriptors += x
    descriptors = list(set(descriptors))  # Remove duplicates.
    print("%d descriptors used." % len(descriptors))
    return descriptors, data


if __name__ == "__main__":
    descriptors, data = get_data()
