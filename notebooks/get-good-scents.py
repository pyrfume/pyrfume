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

import bs4
import requests


# +
def name2soup(name):
    base_url = "http://www.thegoodscentscompany.com"
    url = "%s/search3.php" % base_url
    data = {"qName": name, "submit.x": 0, "submit.y": 0}
    response = requests.post(url, data)
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


# + tags=[]
name = "acetophenone"
soup = name2soup(name)
odors = soup2odor(soup)
# -

for odor in odors:
    if "Odor Strength" in odor:
        strength = odor.split(":")[1].split(",")[0].strip()
        print(strength)
