from collections import OrderedDict

import dash_core_components as dcc
import dash_html_components as html
import flask
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from dash_table import DataTable

import dash
import pyrfume
from pyrfume import haddad, snitz
from pyrfume.odorants import smiles_to_image

### Initialize app ###

bootstrap = "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
external_stylesheets = [bootstrap]
app = flask.Flask(__name__)
dapp = dash.Dash(
    __name__, server=app, url_base_pathname="/", external_stylesheets=external_stylesheets
)

### Load data ###

# Pyrfume-data-relative file path
file_path = "odorants/all_cids_properties.csv"
# First 5 columns of all_cids file (name, MW, SMILES, etc.)
details = pyrfume.load_data(file_path, usecols=range(5), index_col=0)
# Dragon descriptor files with only the Snitz or Haddad features
dragon = {}
dragon["snitz"] = snitz.get_snitz_dragon().round(3)
dragon["haddad"] = haddad.get_haddad_dragon().round(3)
w = snitz.get_snitz_weights()
dragon["snitz"].loc["Weight", w.index] = w
w = haddad.get_haddad_weights()
dragon["haddad"].loc["Weight", w.index] = w
cumul = {}
cumul["snitz"] = pyrfume.load_data("snitz_2013/snitz_cumulative_probability.csv")
cumul["haddad"] = pyrfume.load_data("haddad_2008/haddad_cumulative_probability.csv")
# Spaces to show
spaces = OrderedDict({"snitz": "Snitz Map", "haddad": "Haddad Map"})


def make_odorant_selector(name):
    """Make an Input widget to select an odorant by CID"""
    return dcc.Input(
        id="cid_%s" % name,
        placeholder="Enter a PubChem ID number...",
        type="number",
        value=None,
    )


def make_space_table(space):
    """Make a data table for the given space"""
    columns = dragon["%s" % space].reset_index().columns
    return DataTable(
        id="%s_table" % space,
        columns=[{"name": i, "id": "%s" % i} for i in columns],
        data=get_table_data(space, []),
        style_header={
            "transform": "translate(0px, 0px) rotate(-0.0deg)",
            "max-width": "20px",
            "backgroundColor": "transparent",
        },
    )


def make_space_elements(space):
    """Return a list of elements to be used in a space layout"""
    return [
        html.Div(html.Hr()),
        html.Div("%s" % space.title(), id="%s_title" % space, style={"font-size": "200%"}),
        dcc.Graph(id="histogram_%s" % space, figure=show_histogram(space, 0)),
        make_space_table(space),
    ]


def get_table_data(space, cids):
    return dragon[space].loc[cids + ["Weight"]].reset_index().to_dict("rows")


def get_distance(space, cids):
    a, b = cids
    if space == "snitz":
        return snitz.get_snitz_distances(dragon[space]).loc[a, b]
    if space == "haddad":
        return haddad.get_haddad_distances(dragon[space]).loc[a, b]


def show_histogram(space, distance):
    index = cumul[space].index.get_loc(distance, "nearest")
    print(index)
    cumul_prob = cumul[space].iloc[index]["Cumulative Probability"]
    return {
        "data": [
            go.Bar(
                **{
                    "x": cumul[space].index,
                    "y": cumul[space]["Cumulative Probability"],
                    "name": "All pairs",
                }
            ),
            go.Scatter(
                **{"x": [distance], "y": [cumul_prob], "mode": "markers", "name": "This pair"}
            ),
        ],
        "layout": {
            "title": "",
            "width": 300,
            "height": 100,
            "margin": {"t": 10, "b": 20},
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
                    children=[html.Div("Molecule %s" % x)],
                    style={"font-size": "200%"},
                )
                for x in ("A", "B")
            ],
        ),
        html.Div(
            className="row",
            children=[
                html.Div(className="col", children=[make_odorant_selector(x)]) for x in ("A", "B")
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
                                html.Tr([html.Td(["Name:"]), html.Td(id="name_%s" % x)]),
                                html.Tr([html.Td(["SMILES:"]), html.Td(id="smiles_%s" % x)]),
                            ],
                            style={"vertical-align": "middle", "font-size": "160%"},
                        ),
                        html.Img(id="molecule-2d_%s" % x, src=""),
                    ],
                )
                for x in ("A", "B")
            ],
        ),
    ]
    + make_space_elements("snitz")
    + make_space_elements("haddad"),
)

### App callbacks ###

outputs = []
for x in ("A", "B"):
    outputs.append(Output("name_%s" % x, "children"))
    outputs.append(Output("smiles_%s" % x, "children"))
    outputs.append(Output("molecule-2d_%s" % x, "src"))
for space in spaces:
    outputs.append(Output("%s_table" % space, "data"))
    outputs.append(Output("%s_title" % space, "children"))
    outputs.append(Output("histogram_%s" % space, "figure"))


@dapp.callback(
    outputs,
    [Input("cid_%s" % x, "value") for x in ("A", "B")],
)
def select_cid(*values):
    cids = OrderedDict()
    for i, x in enumerate(["A", "B"]):
        cids[x] = values[i]
    result = [""] * 12
    show_cids = []
    for i, (x, cid) in enumerate(cids.items()):
        if cid is not None:
            try:
                name, mw, smiles, iupac = details.loc[cid]
            except KeyError:
                pass
            else:
                show_cids.append(cid)
                result[3 * i] = name
                result[3 * i + 1] = smiles
                result[3 * i + 2] = get_image_src(cid)
    for i, space in enumerate(spaces):
        result[6 + 3 * i] = get_table_data(space, show_cids)
        if len(show_cids) == 2:
            distance = get_distance(space, show_cids)
            result[7 + 3 * i] = "%s: %.3g" % (space.title(), distance)
            result[8 + 3 * i] = show_histogram(space, distance)
        else:
            result[7 + 3 * i] = "%s" % space.title()
            result[8 + 3 * i] = show_histogram(space, 0)
    return result


def get_image_src(cid):
    if cid is None:
        return ""
    smiles = details.loc[cid, "SMILES"]
    image = smiles_to_image(smiles, png=True, b64=True, crop=True, size=250)
    src = "data:image/png;base64, %s" % image
    return src


### Run app ###

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=80)
