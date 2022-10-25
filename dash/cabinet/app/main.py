# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html
import flask
import pandas as pd
import plotly.graph_objs as go

import dash
import pyrfume

### Initialize app ###
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = flask.Flask(__name__)
dapp = dash.Dash(
    __name__, server=app, url_base_pathname="/", external_stylesheets=external_stylesheets
)


### Load data ###
file_path = pyrfume.DATA_DIR / "cabinets" / "Mainland Odor Cabinet with CIDs.csv"
df = pd.read_csv(file_path).reset_index().groupby("CID").first()


### App layout ###
dapp.layout = html.Div(
    [
        dcc.Graph(
            id="odorants",
            figure={
                "data": [
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
                ],
                "layout": go.Layout(
                    xaxis={"type": "log", "title": "Price"},
                    yaxis={"type": "log", "title": "Molecular Weight"},
                    margin={"l": 40, "b": 40, "t": 10, "r": 10},
                    legend={"x": 0, "y": 1},
                    hovermode="closest",
                ),
            },
        )
    ]
)


### Run app ###
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=80)
