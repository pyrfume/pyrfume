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

import re
import requests
from urllib.request import quote
import pandas as pd
import bs4

df = pd.read_excel("data/vcd_database.xlsx")

cas_list = set(list(df.iloc[:, 6].replace(0, None).dropna())[1:])
print(len(cas_list))

cas_cid_smiles = pd.DataFrame(columns=["CID", "SMILES"])
for cas in list(cas_list):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/isomericSMILES/JSON"
        % cas
    )
    r = requests.get(url)
    if r.status_code == 200:
        xs = r.json()["PropertyTable"]["Properties"]
        found_chiral_smiles = False
        for x in xs:
            if any([char in x["IsomericSMILES"] for char in ["@", "@@", "/", "\\"]]):  # Is chiral
                cas_cid_smiles.loc[cas] = [x["CID"], x["IsomericSMILES"]]
                found_chiral_smiles = True
                break
        if not found_chiral_smiles:
            print("No chiral SMILES found for CAS %s" % cas)
    else:
        print("No result for CAS %s" % cas)
cas_cid_smiles["Type"] = "Original"
cas_cid_smiles["Enantiomer"] = ""
print(cas_cid_smiles.shape[0])

original_smiles_list = list(cas_cid_smiles["SMILES"])
original_cas_list = list(cas_cid_smiles.index)
for i, smiles in enumerate(original_smiles_list):
    smiles_m = (
        smiles.replace("@@", "a")
        .replace("@", "b")
        .replace("/", "c")
        .replace("\\", "d")
        .replace("a", "@")
        .replace("b", "@@")
        .replace("c", "\\")
        .replace("d", "/")
    )
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/%s/synonyms/JSON" % quote(
        smiles_m
    )
    r = requests.get(url)
    if r.status_code == 200:
        x = r.json()["InformationList"]["Information"][0]
        CID = x["CID"]
        found_cas = False
        for synonym in x["Synonym"]:
            if re.match("[0-9]+-[0-9]+-[0-9]+", synonym) and " " not in synonym:
                found_cas = True
                cas = synonym
                original_cas = cas_cid_smiles.index[i]
                if cas == original_cas:
                    print("Rejected identical CAS %s" % cas)
                    continue
                if cas in original_cas_list:
                    print("Already have data on CAS %s" % cas)
                    continue
                cas_cid_smiles.loc[cas] = [x["CID"], smiles_m, "Mirrored", original_cas]
                cas_cid_smiles.loc[original_cas, "Enantiomer"] = cas
                break
        if not found_cas:
            print("No CAS found for mirror SMILES %s" % smiles_m)
    else:
        print("No result for mirror SMILES string %s" % smiles_m)
print(cas_cid_smiles.shape[0])

cas_cid_smiles["Type"].value_counts()

# This should print nothing if all the SMILES strings have stereometic information
for i, s in enumerate(cas_cid_smiles["SMILES"]):
    if not any([x in s for x in ["@", "@@", "/", "\\"]]):
        print(
            "This SMILES string has no stereomeric information: CAS=%s; CID=%d; SMILES=%s"
            % (cas_cid_smiles["Type"][i], cas_cid_smiles.index[i], cas_cid_smiles["CID"][i], s)
        )

pairs = cas_cid_smiles[cas_cid_smiles["Enantiomer"] != ""]
print(
    "There are %d pairs of enantiomers for which we have VCD spectra for at least one member of the pair."
    % (len(pairs) / 2)
)

pairs


def cas2soup(cas):
    import requests

    base_url = "http://www.thegoodscentscompany.com"
    url = "%s/search3.php?qName=%s&submit.x=0&submit.y=0" % (base_url, cas)
    response = requests.get(url)
    index = response.text.find("data/")
    soup = None
    if index > 0:
        path = response.text[index : (index + 25)].split("'")[0]
        url = "%s/%s" % (base_url, path)
        response = requests.get(url)
        soup = bs4.BeautifulSoup(response.text)
    return soup


def soup2odor(soup):
    chem_tables = soup.find_all("table", {"class": "cheminfo"})
    odor = None
    for t in chem_tables:
        if "Odor" in t.text:
            odor = [x.text for x in t.find_all("td")]
            break
    return odor


cas_odors = {}
for cas in cas_cid_smiles.index:
    print(cas)
    soup = cas2soup(cas)
    if soup:
        odor = soup2odor(soup)
        if odor:
            print("Found odor info for %s" % cas)
            cas_odors[cas] = odor

print(
    "We have GoodScents odors information for %d molecules in the enantiomer set" % len(cas_odors)
)

x = cas_cid_smiles.copy()
x["Odor"] = 0
for cas in cas_odors:
    x.loc[cas, "Odor"] = 1
x2 = x[(x["Odor"] == 1) & (x["Enantiomer"] != "")]
x3 = x2[x2["Enantiomer"].isin(list(x2.index))]
print("We have odor info and spectra for %d enantiomers (%d pairs)" % (len(x3), len(x3) / 2))

x3.shape

x3["Strength"] = ""
for cas, odor in cas_odors.items():
    for o in odor:
        if "Odor Strength" in o:
            strength = o.split(":")[1].split(",")[0].strip()
            if cas in x3.index:
                x3.loc[cas, "Strength"] = strength

x3
