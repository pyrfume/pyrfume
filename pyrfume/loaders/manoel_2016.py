import pandas as pd

# Load data about the odorants (names and PubChem IDs)
def get_translater():
    rosetta = pd.read_excel("SmilesInfo2.xlsx")
    threeletter_to_pubchem = dict(rosetta.iloc[:, -1:-3:-1].values)
    threeletter_to_pubchem["MEN"] = threeletter_to_pubchem["+MEN"]
    return threeletter_to_pubchem
    #pubchem_to_threeletter = {value: key for key, value in threeletter_to_pubchem.items()}
    #pubchem_to_threeletter[1201521] = "+FCH"
    #del pubchem_to_threeletter[16213045]

def get_raw():
    # Load raw mouse data (individual mouse level)
    raw = pd.read_csv(
    "raw behavioral scores mouse 73 odorants.csv", index_col=0, header=1
    ).dropna()
    raw.index.name = "odor"
    raw = raw.sort_index()
    return raw

def main():
    translater = get_translater()
    raw = get_raw()
    