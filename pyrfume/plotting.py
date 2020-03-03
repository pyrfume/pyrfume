import io
import os

from ipywidgets import HBox, VBox, Image, Layout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import iplot
import pyrfume
import warnings
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
except ImportError:
    warnings.warn("Parts of rdkit could not be imported; try installing rdkit via conda",
                  UserWarning)


def mpl_embedding(xy, colors=None, alpha=0.25, figsize=(6, 6),
                  s=0.5, cmap='hsv', title=None, ax=None):
    if not ax:
        plt.figure(figsize=figsize)
        plt.margins(0)
        ax = plt.gca()

    ax.scatter(xy[:, 0], xy[:, 1],
               c=(colors if colors is not None else 'k'),  # set colors of markers
               cmap=cmap,  # set color map of markers
               alpha=alpha,  # set alpha of markers
               marker='o',  # use smallest available marker (square)
               s=s,  # set marker size. single pixel is 0.5 on retina,
                     # 1.0 otherwise
               lw=0,  # don't use edges
               edgecolor='')  # don't use edges

    # remove all axes and whitespace / borders
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_facecolor('white')
    if title:
        ax.set_title(title)
        
        
def smiles_to_image(smiles):
    a = io.BytesIO()
    m = Chem.MolFromSmiles(smiles)
    z = Draw.MolToImage(m)
    z.save(a, format='PNG')
    return a.getvalue()


def plotly_embedding(embedding, features=None, colors=None):
    """
    params:
        embedding: A dataframe wrapped around e.g. a fitted TSNE object, with an index of CIDs
        features: A dataframe of features, e.g. names, SMILES strings, or physicochemical features, with an index of CIDs
    """
    
    if features is None:
        file_path = pyrfume.DATA_DIR / 'odorants' / 'all_cids_properties.csv'
        features = pd.read_csv(file_path, usecols=range(5), index_col=0)
    # Only retain those rows corresponding to odorants in the embedding
    features = features.loc[embedding.index]
    try:
        names = features.loc[:, ['Name', 'IUPACName']].reset_index().astype('str').apply(lambda x:'CID: %s<br>%s<br>%s' % (x[0], x[1], x[2]), axis=1)
    except:
        names = features.index
    assert embedding.shape[0] == features.shape[0]
    
    # The scatter plot
    scatter = go.Scattergl(
        x=embedding.iloc[:, 0],
        y=embedding.iloc[:, 1],
        text=names,
        mode='markers',
        hoverinfo='text',
        opacity=0.5,
        marker={
            'size': 5,
            'line': {'width': 0.5, 'color': 'white'},
            'color': colors if colors is not None else 'black',
            'colorscale': 'rainbow'
            },
        )
    
    # The axes, etc.
    layout = go.Layout(
        xaxis={'type': 'linear', 'title': '', 'showline':False, 'showticklabels': False},
        yaxis={'type': 'linear', 'title': '', 'showline':False, 'showticklabels': False},
        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
        legend={'x': 0, 'y': 1},
        hovermode='closest',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=500,
        height=500,
        )
    
    figure = go.Figure(data=[scatter], layout=layout)
    fig = go.FigureWidget(figure)

    # The 2D drawing of the molecule
    image_widget = Image(
        value=smiles_to_image('CCCC'),
        layout=Layout(height='300px', width='300px')
        )
    
    def hover_fn(trace, points, state):
        ind = points.point_inds[0]
        import numpy as np
        #smiles = 'C' * np.random.randint(10)
        smiles = features['SMILES'].iloc[ind]
        image_widget.value = smiles_to_image(smiles)
        
    scatter = fig.data[0]
    scatter.on_hover(hover_fn)
    canvas = VBox([fig,
                   image_widget])
    return canvas