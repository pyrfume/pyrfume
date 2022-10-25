import pickle
from collections import OrderedDict

import dash_core_components as dcc
import dash_html_components as html
import flask
import plotly.graph_objs as go
from dash.dependencies import Input, Output

import dash
import pyrfume
from pyrfume.odorants import all_odorants, all_sources, smiles_to_image

### Initialize app ###
external_stylesheets = ["https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"]
app = flask.Flask(__name__)
dapp = dash.Dash(
    __name__, server=app, url_base_pathname="/", external_stylesheets=external_stylesheets
)

### Load data ###
details = all_odorants()
sources = all_sources()

spaces = OrderedDict({"snitz": "Snitz Map", "haddad": "Haddad Map", "westeros": "Westeros Map"})
algorithm = "umap"
embeddings = {}
for space in spaces:
    key = "%s_%s" % (space, algorithm)
    with open(pyrfume.DATA_DIR / "odorants" / ("%s.pkl" % key), "rb") as f:
        embeddings[space] = pickle.load(f)
        # Remove molecules for which there are no details
        good_cids = details.index.intersection(embeddings[space].index)
        embeddings[space] = embeddings[space].loc[good_cids, :]

# Assert that all embeddings have the same number of molecules
assert len(set([embeddings[space].shape[0] for space in spaces])) == 1


def make_figure(embedding, title):
    return {
        "data": [
            go.Scattergl(
                **{
                    "x": embedding.loc[:, "X"].values,
                    "y": embedding.loc[:, "Y"].values,
                    "mode": "markers",
                    "marker": {
                        "size": 5,
                        "color": embedding.loc[:, "cluster"].values,
                        "opacity": 0.5,
                        "cmax": 9,
                        "cmin": 0,
                        "colorscale": "Rainbow",
                    },
                }
            ),
        ],
        "layout": {
            "title": title,
            "clickmode": "event",
            "xaxis": {
                "type": "linear",
                "title": "",
                "zeroline": False,
                "showgrid": False,
                "showline": False,
                "showticklabels": False,
            },
            "yaxis": {
                "type": "linear",
                "title": "",
                "zeroline": False,
                "showgrid": False,
                "showline": False,
                "showticklabels": False,
            },
            "margin": {"l": 40, "b": 40, "t": 50, "r": 10},
            "legend": {"x": 0, "y": 1},
            "hovermode": "closest",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "width": 700,
            "height": 500,
        },
    }


### App layout ###
dapp.layout = html.Div(
    className="container-fluid",
    children=[
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="col",
                    children=[
                        dcc.Graph(
                            id=space,
                            clear_on_unhover=True,
                            figure=make_figure(embeddings[space], title),
                        )
                    ],
                )
                for space, title in spaces.items()
            ],
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="col",
                    children=[
                        html.Table(
                            [
                                html.Tr([html.Td(["CID:"]), html.Td([html.A(id="cid", href="#")])]),
                                html.Tr([html.Td(["MW:"]), html.Td(id="mw")]),
                                html.Tr([html.Td(["Name:"]), html.Td(id="name")]),
                                html.Tr([html.Td(["SMILES:"]), html.Td(id="smiles")]),
                                html.Tr([html.Td(["IUPAC:"]), html.Td(id="iupac")]),
                                html.Tr([html.Td(["Sources:"]), html.Td(id="sources")]),
                            ],
                            style={"vertical-align": "middle", "font-size": "160%"},
                        )
                    ],
                ),
                html.Div(
                    className="col",
                    children=[
                        html.Img(id="molecule-2d", src=""),
                    ],
                ),
            ],
        ),
        html.Div(id="hidden-div", style={"display": "none"}),
    ],
)

# multiple callback error suppressed [Div has no .keys() atrribute]
# dapp.config['suppress_callback_exceptions']=True


def get_index(*hoverData):
    index = None
    for hoverDatum in hoverData:
        if hoverDatum:
            index = hoverDatum["points"][0]["pointIndex"]
    return index


### App callbacks ###
@dapp.callback(
    [
        Output("cid", "children"),
        Output("cid", "href"),
        Output("mw", "children"),
        Output("name", "children"),
        Output("smiles", "children"),
        Output("iupac", "children"),
        Output("sources", "children"),
        Output("molecule-2d", "src"),
    ],
    [Input(space, "hoverData") for space in spaces],
)
def _display_hover(*hoverData):
    index = get_index(*hoverData)
    if index is not None:
        print(index)
    return display_hover(index)


def display_hover(index):
    columns = ["CID", "MW", "Name", "SMILES", "IUPACName"]
    if index is None:
        return [""] * 8
    info = details.reset_index().iloc[index][columns]
    cid, mw, name, smiles, iupacname = info.values
    cid_url = "https://pubchem.ncbi.nlm.nih.gov/compound/%d" % cid
    source = sources.reset_index().loc[index]
    source = ", ".join(source[source == 1].index)
    hover_image_src = display_hover_image(index)
    return [cid, cid_url, mw, name, smiles, iupacname, source, hover_image_src]


def display_hover_image(index):
    if index is None:
        return ""
    smiles = details.iloc[index]["SMILES"]
    image = smiles_to_image(smiles, png=True, b64=True, crop=True, size=500)
    src = "data:image/png;base64, %s" % image
    return src


### Run app ###
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=80)
