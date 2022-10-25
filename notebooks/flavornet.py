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

import os
from urllib.request import urlopen

# +
import bs4
import pandas as pd

import pyrfume
from pyrfume import odorants

# -

url = "http://www.flavornet.org/cas.html"
f = urlopen(url)
html = f.read()
soup = bs4.BeautifulSoup(html)

cas_list = []
rows = soup.find("table").find_all("tr")
for row in rows[1:]:
    cas = row.find("td").text
    cas_list.append(cas)

cids = odorants.get_cids(cas_list, kind="name")

df = pd.Series(cids, name="CID").to_frame()
df.head()

file_path = os.path.join(pyrfume.DATA, "flavornet.csv")
df.to_csv(file_path)
