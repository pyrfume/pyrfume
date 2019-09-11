# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('data/Mainland Odor Cabinet with CIDs.csv').groupby('CID').first()
print(list(df))
app.layout = html.Div([
    dcc.Graph(
        id='odorants',
        figure={
            'data': [
                go.Scatter(
                    x=df[df['SourcedFrom'] == i]['Price'],
                    y=df[df['SourcedFrom'] == i]['MW'],
                    text=df[df['SourcedFrom'] == i]['ChemicalName'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=i
                ) for i in df['SourcedFrom'].unique()
            ],
            'layout': go.Layout(
                xaxis={'type': 'log', 'title': 'Price'},
                yaxis={'type': 'log', 'title': 'Molecular Weight'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)