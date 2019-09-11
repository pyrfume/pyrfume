import dash
import sys
print(sys.path)
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import flask

import os
from os.path import join
import pandas as pd
import numpy as np
import pickle
import pyrfume
from pyrfume.odorants import smiles_to_image


##### Initialize app #####
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']
app = flask.Flask(__name__)
dapp = dash.Dash(__name__,
                server = app,
                url_base_pathname = '/',
                external_stylesheets=external_stylesheets)

##### Load data #####
file_path = pyrfume.DATA_DIR / 'odorants' / 'all_cids_properties.csv'
details = pd.read_csv(file_path, usecols=range(5), index_col=0)
file_path = pyrfume.DATA_DIR / 'odorants' / 'all_cids.csv'
sources = pd.read_csv(file_path, index_col=0)

with open(pyrfume.DATA_DIR / 'odorants' / 'tsne_snitz.pkl', 'rb') as f:
    tsne_snitz_embedding = pickle.load(f)

with open(pyrfume.DATA_DIR / 'odorants' / 'tsne_snitz_clusters.pkl', 'rb') as f:
    tsne_snitz_clusters = pickle.load(f)


###### App layout ######
dapp.layout = html.Div(className='container', children=[
    html.Div(className='row', children=[
        dcc.Graph(
            id='basic-interactions',
            figure={
                'data': [
                    go.Scattergl(**{
                        'x': tsne_snitz_embedding[:, 0],
                        'y': tsne_snitz_embedding[:, 1],
                        'mode': 'markers',
                        'marker': {'size': 5,
                                   'color': tsne_snitz_clusters,
                                   'opacity': 0.5,
                                   'colorscale': 'rainbow'}
                    })],

                'layout': {
                    'clickmode': 'event+select',
                    'xaxis': {'type': 'linear', 
                              'title': '', 
                              'zeroline': False,
                              'showgrid': False,
                              'showline': False, 
                              'showticklabels': False},
                    'yaxis': {'type': 'linear',
                              'title': '',
                              'zeroline': False,
                              'showgrid': False,
                              'showline': False,
                              'showticklabels': False},
                    'margin': {'l': 40, 'b': 40, 't': 10, 'r': 10},
                    'legend': {'x': 0, 'y': 1},
                    'hovermode': 'closest',
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                    'width': 750,
                    'height': 750,
                    },
                }
            ),]),

    html.Div(className='row', children=[
        html.Div(className='col', children=[
            html.Table([
                html.Tr([html.Td(['CID:']), 
                         html.Td([html.A(id='cid', 
                                         href='#')])]),
                html.Tr([html.Td(['MW:']), 
                         html.Td(id='mw')]),
                html.Tr([html.Td(['Name:']), 
                         html.Td(id='name')]),
                html.Tr([html.Td(['SMILES:']), 
                         html.Td(id='smiles')]),
                html.Tr([html.Td(['IUPAC:']), 
                         html.Td(id='iupac')]),
                html.Tr([html.Td(['Sources:']), 
                         html.Td(id='sources')]),
                ], style={'vertical-align': 'middle',
                          'font-size': '160%'})]),
        html.Div(className='col', children=[
            html.Img(id='molecule-2d', src='')
            ,]),
        ])
    ])

# multiple callback error suppressed [Div has no .keys() atrribute]
#dapp.config['suppress_callback_exceptions']=True


##### App callbacks #####
@dapp.callback(
    [Output('cid', 'children'),
     Output('cid', 'href'),
     Output('mw', 'children'),
     Output('name', 'children'),
     Output('smiles', 'children'),
     Output('iupac', 'children'),
     Output('sources', 'children')],
    [Input('basic-interactions', 'hoverData')])
def _display_hover_text(hoverData):
    return display_hover_text(hoverData)
    
@dapp.callback(
    Output('molecule-2d', 'src'),
    [Input('basic-interactions', 'hoverData')])
def _display_hover_image(hoverData):
    return display_hover_image(hoverData)

def display_hover_text(hoverData):
    columns = ['CID', 'MW', 'Name', 'SMILES', 'IUPACName']
    try:
        index = hoverData['points'][0]['pointIndex']
    except TypeError:
        return [''] * (len(columns)+2) 
    info = details.reset_index().loc[index, columns]
    cid, mw, name, smiles, iupacname = list(info)
    cid_url = 'https://pubchem.ncbi.nlm.nih.gov/compound/%d' % cid
    source = sources.reset_index().loc[index]
    source = ', '.join(source[source==1].index)
    return cid, cid_url, mw, name, smiles, iupacname, source
    
def display_hover_image(hoverData):
    try:
        index = hoverData['points'][0]['pointIndex']
    except TypeError:
        return ''
    smiles = details.iloc[index]['SMILES']
    image = smiles_to_image(smiles, png=True, b64=True, crop=False)
    src = 'data:image/png;base64, %s' % image
    return src


##### Run app #####
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
