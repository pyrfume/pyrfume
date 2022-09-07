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

# +
import bs4
import os
import pandas as pd
from urllib.request import urlopen

import pyrfume
from pyrfume import odorants

# -

# Scrape compounds names and links from SenseLab
info = []
for page in range(1, 8):
    url = "https://senselab.med.yale.edu/OdorDB/Browse?db=5&cl=1&page=%d" % page
    f = urlopen(url)
    html = f.read()
    soup = bs4.BeautifulSoup(html)
    table = soup.find("table")
    for span in table.find_all("span"):
        name = span.text.strip()
        link = span.find("a").get("href")
        info.append((name, link))

# Make into a dataframe
df = pd.DataFrame.from_records(info, columns=["name", "url"])
df.head()

# Get CIDS by searching the names
cids = odorants.get_cids(df["name"], kind="name")

# Add these CIDs to the dataframe
df = df.set_index("name").join(pd.Series(cids, name="CID"))

# Get CAS strings for compounds with no CID was found based on the name
for name, url_suffix in df[df["CID"] == 0]["url"].items():
    url = "https://senselab.med.yale.edu/OdorDB/%s" % url_suffix
    f = urlopen(url)
    html = f.read()
    soup = bs4.BeautifulSoup(html)
    table = soup.find("table")
    cas_row = table.find_all("tr")[5]
    cas_text = cas_row.find_all("span")[-1].text
    cas = cas_text.replace("\r\n", "").strip()
    df.loc[name, "CAS"] = cas

# +
# Add CIDs obtained from searching the CAS string
for name, cas in df[df["CAS"].notnull()]["CAS"].items():
    if cas:
        cid = odorants.get_cid(cas, kind="name")
        df.loc[name, "CID"] = cid

# Fill remaining missing CIDs with 0
df.loc[:, "CID"] = df["CID"].fillna(0)
# -

# Manual fills
df.loc["2,4,5-TRIMETHYLTHIAZOLINE", "CID"] = 263626
df.loc["METHYLSALICYLATE", "CID"] = 4133
df.loc["PHENYLETHYL ALCOHOL (PEA)", "CID"] = 6054
df.loc["Perillaalcohol", "CID"] = 10819
df.loc["Perillaaldehyde", "CID"] = 16441
# df[df['CID']==0]

file_path = os.path.join(pyrfume.DATA, "senselab.csv")
df.to_csv(file_path)
