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
# Needed for all table stuff
import pandas as pd
# Needed for all plotting
import plotly.graph_objs as go
# Needed for offline stuff in the notebook
from plotly.offline import init_notebook_mode, iplot
import plotly
from plotly.plotly import plot

init_notebook_mode()

# Needed to make figures on the web
plotly.tools.set_credentials_file(username="rgerkin", api_key="DdQKMmvH8CMKvgNpBccy")
plotly.tools.set_config_file(world_readable=True, sharing="public")

# Load the data
df = pd.read_csv("data/Mainland Odor Cabinet with CIDs.csv").groupby("CID").first()

# The scatter plots themselves
data = [
    go.Scatter(
        x=df[df["SourcedFrom"] == i]["Price"],
        y=df[df["SourcedFrom"] == i]["MW"],
        text=df[df["SourcedFrom"] == i]["ChemicalName"],
        mode="markers",
        opacity=0.7,
        marker={"size": 15, "line": {"width": 0.5, "color": "white"}},
        name=i,
    )
    for i in df["SourcedFrom"].unique()
]

# The axes, etc.
layout = go.Layout(
    xaxis={"type": "log", "title": "Price"},
    yaxis={"type": "log", "title": "Molecular Weight"},
    margin={"l": 40, "b": 40, "t": 10, "r": 10},
    legend={"x": 0, "y": 1},
    hovermode="closest",
)
# -

# Make an interactive version in the notebook
fig = go.Figure(data, layout)
iplot(fig)

# Make a version in the web
fig = go.Figure(data, layout)
plot(fig, auto_open=True)
