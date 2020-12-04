import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from ipywidgets import Image, Layout, VBox

import pyrfume

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
except ImportError:
    warnings.warn(
        "Parts of rdkit could not be imported; try installing rdkit via conda", UserWarning
    )


def mpl_embedding(
    xy, colors=None, alpha=0.25, figsize=(6, 6), s=0.5, cmap="hsv", title=None, ax=None
):
    if not ax:
        plt.figure(figsize=figsize)
        plt.margins(0)
        ax = plt.gca()

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=(colors if colors is not None else "k"),  # set colors of markers
        cmap=cmap,  # set color map of markers
        alpha=alpha,  # set alpha of markers
        marker="o",  # use smallest available marker (square)
        s=s,  # set marker size. single pixel is 0.5 on retina,
        # 1.0 otherwise
        lw=0,  # don't use edges
        edgecolor="",
    )  # don't use edges

    # remove all axes and whitespace / borders
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_facecolor("white")
    if title:
        ax.set_title(title)


def plotly_embedding(embedding, features=None, show_features=None, colors=None, colorscale='rainbow'):
    """
    params:
        embedding: A dataframe wrapped around e.g. a fitted TSNE object, with an index of CIDs
        features: A dataframe of features, e.g. names, SMILES strings, or physicochemical features,
            with an index of CIDs
    """

    if features is None:
        features = pyrfume.load_data("odorants/all-cids-properties.csv", usecols=range(5))
        # Only retain those rows corresponding to odorants in the embedding
    features = features.loc[embedding.index]
    show_features = show_features or list(features)
    def format_features(col):
        return "%s: %s" % (index_name, x.values.split('<br>'))
    try:
        index_name = features.index.name or 'Index'
        names = (
            features.loc[:, show_features]
            .reset_index()
            .astype("str")
            .apply(format_features, axis=1)
        )
    except Exception:
        names = features.index
    assert embedding.shape[0] == features.shape[0]

    # The scatter plot
    scatter = go.Scatter(
        x=embedding.iloc[:, 0],
        y=embedding.iloc[:, 1],
        text=names,
        mode="markers",
        hoverinfo="text",
        opacity=0.5,
        marker={
            "size": 5,
            "line": {"width": 0.5, "color": "white"},
            "color": colors if colors is not None else "black",
            "colorscale": colorscale,
        },
    )
    
    # The axes, etc.
    layout = go.Layout(
        xaxis={"type": "linear", "title": "", "showline": False, "showticklabels": False},
        yaxis={"type": "linear", "title": "", "showline": False, "showticklabels": False},
        margin={"l": 40, "b": 40, "t": 10, "r": 10},
        legend={"x": 0, "y": 1},
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        width=500,
        height=500,
    )

    fig = go.FigureWidget(data=[scatter], layout=layout)
    fig.layout.hovermode = 'closest'

    # The 2D drawing of the molecule
    image_widget = Image(
        value=smiles_to_image("CCCCO"), layout=Layout(height="300px", width="300px")
    )

    def hover_fn(trace, points, state):
        ind = points.point_inds[0]
        smiles = features["SMILES"].iloc[ind]
        image_widget.value = smiles_to_image(smiles)

    scatter = fig.data[0]
    scatter.on_hover(hover_fn)
    canvas = VBox([fig, image_widget])
    return canvas
