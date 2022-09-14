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

import json
import matplotlib.pyplot as plt
import numpy as np
import requests
# %matplotlib inline
from tqdm.auto import tqdm

def update_results(records, results):
    for annotation in records["Annotations"]["Annotation"]:
        try:
            cids = annotation["LinkedRecords"]["CID"]
        except Exception:
            pass
        else:
            strings = []
            for x in annotation["Data"]:
                for y in x["Value"]["StringWithMarkup"]:
                    strings.append(y["String"])
            for cid in cids:
                if cid in results:
                    results[cid] += strings
                elif strings:
                    results[cid] = strings


def get_results(heading):
    page = 1
    results = {}
    with tqdm(total=100) as pbar:
        while True:
            url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/annotations/heading/"
                f"JSON?heading_type=Compound&heading={heading}&page={page}"
            )
            response = requests.get(url)
            records = response.json()
            update_results(records, results)
            totalPages = records["Annotations"]["TotalPages"]
            if page == 1:
                pbar.reset(total=totalPages)
            pbar.set_description("%d CIDs described" % len(results))
            pbar.update()
            page += 1
            if page > totalPages:
                break
    return results


pd_results = get_results("Vapor+Pressure")


# +
def make_hist(results):
    plt.hist(np.log10(list(results.keys())), bins=np.arange(10))
    xticks = np.arange(10)
    plt.xticks(xticks, ["$10^%d$" % x for x in xticks])
    plt.xlabel("PubChem ID")
    plt.ylabel("Entry Count")


make_hist(pd_results)
# -

pd_cids = set(key for key in pd_results)
len(pd_cids)

all_statements = {}
cids = sorted(set(pd_cids))
for cid in cids:
    ps = pd_results.get(cid, [])
    all_statements[cid] = ps
len(all_statements)


with open("pubchem-vapor-pressure.json", "w") as f:
    json.dump(all_statements, f)
