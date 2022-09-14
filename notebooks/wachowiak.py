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

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
from holoviews.selection import link_selections

n = 25
data = {
    "Name": ["Glom %d" % i for i in range(1, n + 1)],
    "x": np.random.randn(n).round(2),
    "y": np.random.randn(n).round(2),
    "dF/F": np.random.randint(10, 100, size=n),
    "Odorant": np.random.randint(10, 100, size=n),
    "-Log(C)": np.random.uniform(3, 12, size=n).round(2),
}
df = pd.DataFrame(data)  # .set_index('Name')
# df = pd.DataFrame(np.random.randint(10, 100, size=(25, 3)), columns=['x', 'y', 'z'])
scatter = df.hvplot.scatter(x="x", y="y", size="dF/F").opts(height=300, width=300)
bars = df.hvplot.bar(x="Name", y="-Log(C)").opts(width=300, xrotation=70)
link = link_selections.instance()
plots = link(scatter + bars)


@param.depends(link.param.selection_expr)
def selection_table(_):
    return hv.Table(hv.Dataset(df).select(link.selection_expr)).opts(width=600, height=200)


app = pn.Column(plots, selection_table, height=600)
app
# -

app.show(port=8951, websocket_origin=["spike.asu.edu:8952"])
