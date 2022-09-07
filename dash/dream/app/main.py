import pathlib

import dash_core_components as dcc
import dash_html_components as html
import flask
import numpy as np
from dash.dependencies import Input, Output
from dash_table import DataTable
from flask import send_file

import dash
import pyrfume
from pyrfume.odorants import cids_to_smiles, from_cids, smiles_to_image
from pyrfume.predictions import load_dream_model, predict, smiles_to_features

OUTPUT_DIR = pathlib.Path(__file__).parent / "data"

##### Initialize app #####

bootstrap = "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
external_stylesheets = [bootstrap]
app = flask.Flask(__name__)
dapp = dash.Dash(
    __name__, server=app, url_base_pathname="/", external_stylesheets=external_stylesheets
)

##### Load model #####

model, use_features, descriptors, imputer = load_dream_model()


def make_odorant_selector():
    """Make an Input widget to select an odorant by CID"""
    return dcc.Input(
        id="smiles",
        placeholder="Input a SMILES string and press enter...",
        type="text",
        value="CCCCO",
        debounce=True,
    )


def make_descriptors_table(data=None):
    """Make a data table for the given space"""
    if data is None:
        data = [{i: None for i in descriptors}]
    return DataTable(
        id="model_output",
        columns=[{"name": "SMILES", "id": "smiles"}] + [{"name": i, "id": i} for i in descriptors],
        data=data,
        style_header={
            "transform": "translate(0px, 0px) rotate(-0.0deg)",
            "max-width": "20px",
            "backgroundColor": "transparent",
        },
    )


def make_elements(data=None):
    """Return a list of elements to be used in a space layout"""
    return [
        html.Div(html.Hr()),
        html.Div("Model Output", id="output_title", style={"font-size": "200%"}),
        make_descriptors_table(data),
    ]


def get_model_output(smiles):
    if not isinstance(smiles, list):
        smiles = [smiles]
    cids = [int(s) for s in smiles if s.isnumeric()]
    if cids:
        x = cids_to_smiles(cids)
        print(cids, x)
        smiles = [x[int(s)] if s.isnumeric() else s for s in smiles]
    features = smiles_to_features(smiles, use_features, imputer)
    predictions = predict(model, features, descriptors).reset_index()
    predictions["smiles"] = smiles
    print(predictions.shape, predictions.mean(), len(descriptors))
    import os

    print(os.getcwd())
    print(OUTPUT_DIR / "dream.csv")
    predictions.set_index("smiles").to_csv(OUTPUT_DIR / "dream.csv")
    return predictions


###### App layout ######

dapp.layout = html.Div(
    className="container-fluid",
    children=[
        html.Div(
            className="row",
            children=[html.Div(className="col", children=[make_odorant_selector()])],
        ),
        # html.Div(className='row', children=[
        #    html.Div(className='col', children=[
        #        html.Table([
        #            html.Tr([html.Td(['Name:']),
        #                     html.Td(id='name')]),
        #            ], style={'vertical-align': 'middle',
        #                      'font-size': '160%'}),
        #        html.Img(id='molecule-2d', src=''), ])
        #    ]),
        html.Div(
            className="row",
            children=[html.Div(className="col", children=[make_descriptors_table()])],
        ),
        html.Div(
            className="row",
            children=[html.Div(className="col", children=[html.A("Download", href="download")])],
        ),
    ],
)


@dapp.server.route("/download")
def download_csv():
    return send_file(
        OUTPUT_DIR / "dream.csv",
        mimetype="text/csv",
        attachment_filename="dream.csv",
        as_attachment=True,
    )


##### App callbacks #####

outputs = []
# outputs.append(Output('name', 'children'))
# outputs.append(Output('molecule-2d', 'src'))
outputs.append(Output("model_output", "data"))


@dapp.callback(
    outputs,
    [Input("smiles", "value")],
)
def select_cid(smiles):
    smiles = smiles.split(",")
    result = [""] * 1
    # result[0] = 'Something'
    # result[1] = get_image_src(smiles[0])
    result[0] = get_model_output(smiles).to_dict("records")
    return result


def get_image_src(smiles):
    if not smiles:
        return ""
    image = smiles_to_image(smiles, png=True, b64=True, crop=True, size=250)
    src = "data:image/png;base64, %s" % image
    return src


##### Run app #####

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8000)
