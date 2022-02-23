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
# -

url = 'http://bioinf-applied.charite.de/superscent/index.php?site=browse_scents'
f = urlopen(url)
html = f.read()
soup = bs4.BeautifulSoup(html)

groups = [x.text for x in soup.find_all('table')[1].find_all('td')]
groups

cids = set()
for group in groups:
    print(group)
    url = 'http://bioinf-applied.charite.de/superscent/index.php?site=browse_scents&char=%s' % group
    f = urlopen(url)
    html = f.read()
    soup = bs4.BeautifulSoup(html)
    results_table = soup.find_all('table')[2]
    links = results_table.find_all('a')
    print('%d links' % len(links))
    for link in links:
        compound_url = link.get('href')
        cid = int(compound_url.split('=')[-1])
        cids.add(cid)

len(cids)

file_path = os.path.join(pyrfume.DATA, 'superscent_cids.txt')
pd.Series(sorted(list(cids)), name='CID').to_csv(file_path, index=False, header=True)


