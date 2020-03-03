import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash_table import DataTable
import flask
import numpy as np
import pyrfume
from pyrfume.odorants import smiles_to_image, from_cids
from pyrfume.predictions import load_dream_model, smiles_to_features, predict


##### Initialize app #####

bootstrap = 'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css'
external_stylesheets = [bootstrap]
app = flask.Flask(__name__)
dapp = dash.Dash(__name__,
                 server=app,
                 url_base_pathname='/',
                 external_stylesheets=external_stylesheets)

##### Load model #####

model, use_features, descriptors, imputer = load_dream_model()


def make_odorant_selector():
    """Make an Input widget to select an odorant by CID"""
    return dcc.Input(
        id='smiles',
        placeholder='Input a SMILES string and press enter...',
        type='text',
        value='CCCCO',
        debounce=True,
        )


def make_descriptors_table(data=None):
    """Make a data table for the given space"""
    if data is None:
        data = [{desc: None for desc in descriptors}]
    return DataTable(id='model_output',
                     columns=[{"name": i, "id": i} for i in descriptors],
                     data=data,
                     style_header={'transform': 'translate(0px, 0px) rotate(-0.0deg)',
                                   'max-width': '20px',
                                   'backgroundColor': 'transparent'})


def make_elements(data=None):
    """Return a list of elements to be used in a space layout"""
    return [
        html.Div(html.Hr()),
        html.Div('Model Output',
                 id='output_title',
                 style={'font-size': '200%'}),
        make_descriptors_table(data),
        ]


def get_model_output(smiles):
    if not isinstance(smiles, list):
        smiles = [smiles]
    features = smiles_to_features(smiles, use_features, imputer)
    predictions = predict(model, features, descriptors).reset_index()
    print(predictions.shape, predictions.mean(), len(descriptors))
    return predictions

###### App layout ######

dapp.layout = html.Div(className='container-fluid', children=[
    html.Div(className='row', children=[
        html.Div(className='col',
                 children=[make_odorant_selector()])
        ]),

    html.Div(className='row', children=[
        html.Div(className='col', children=[
            html.Table([
                html.Tr([html.Td(['Name:']),
                         html.Td(id='name')]),
                ], style={'vertical-align': 'middle',
                          'font-size': '160%'}),
            html.Img(id='molecule-2d', src=''), ])
        ]),

    html.Div(className='row', children=[
        html.Div(className='col',
                 children=[make_descriptors_table()])
        ]),
    ]
)

##### App callbacks #####

outputs = []
outputs.append(Output('name', 'children'))
outputs.append(Output('molecule-2d', 'src'))
outputs.append(Output('model_output', 'data'))


@dapp.callback(
    outputs,
    [Input('smiles', 'value')],
    )
def select_cid(smiles):
    result = ['']*3
    result[0] = 'Something'
    result[1] = get_image_src(smiles)
    result[2] = get_model_output(smiles).to_dict('records')
    return result


def get_image_src(smiles):
    if not smiles:
        return ''
    image = smiles_to_image(smiles, png=True, b64=True, crop=True, size=250)
    src = 'data:image/png;base64, %s' % image
    return src


##### Run app #####

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)
